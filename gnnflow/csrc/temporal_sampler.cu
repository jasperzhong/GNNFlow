#include <thrust/execution_policy.h>
#include <thrust/remove.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <numeric>
#include <rmm/device_vector.hpp>
#include <rmm/exec_policy.hpp>
#include <vector>

#include "common.h"
#include "sampling_kernels.h"
#include "temporal_sampler.h"
#include "utils.h"

namespace gnnflow {

struct is_invalid_edge {
  __host__ __device__ bool operator()(
      thrust::tuple<NIDType, EIDType, TimestampType, TimestampType> const&
          edge) {
    return thrust::get<0>(edge) == kInvalidNID;
  }
};

TemporalSampler::TemporalSampler(const DynamicGraph& graph,
                                 const std::vector<uint32_t>& fanouts,
                                 SamplingPolicy sampling_policy,
                                 uint32_t num_snapshots,
                                 float snapshot_time_window, bool prop_time,
                                 uint64_t seed)
    : graph_(graph),
      fanouts_(fanouts),
      sampling_policy_(sampling_policy),
      num_snapshots_(num_snapshots),
      snapshot_time_window_(snapshot_time_window),
      prop_time_(prop_time),
      num_layers_(fanouts.size()),
      seed_(seed),
      cpu_buffer_(nullptr),
      gpu_input_buffer_(nullptr),
      gpu_output_buffer_(nullptr),
      rand_states_(nullptr),
      maximum_sampled_nodes_(0) {
  if (num_snapshots_ == 1 && std::fabs(snapshot_time_window_) > 0.0f) {
    LOG(WARNING) << "Snapshot time window must be 0 when num_snapshots = 1. "
                    "Ignore the snapshot time window.";
  }
  shared_memory_size_ = GetSharedMemoryMaxSize();
  stream_holders_.reset(new StreamHolder[num_snapshots_]);
  device_ = graph_.device();
}

void TemporalSampler::InitBuffer(std::size_t num_root_nodes,
                                 std::size_t maximum_sampled_nodes) {
  cpu_buffer_.reset(
      new PinMemoryBuffer(maximum_sampled_nodes * kPerNodeOutputBufferSize));
  gpu_input_buffer_.reset(
      new GPUBuffer(num_root_nodes * kPerNodeInputBufferSize));
  gpu_output_buffer_.reset(
      new GPUBuffer(maximum_sampled_nodes * kPerNodeOutputBufferSize));

  if (sampling_policy_ == SamplingPolicy::kSamplingPolicyUniform) {
    rand_states_.reset(new CuRandStateHolder(num_root_nodes, seed_));
  }
}

TemporalSampler::InputBufferTuple TemporalSampler::GetInputBufferTuple(
    const Buffer& buffer, std::size_t num_root_nodes) const {
  return std::make_tuple(reinterpret_cast<NIDType*>(static_cast<char*>(buffer)),
                         reinterpret_cast<TimestampType*>(
                             buffer + num_root_nodes * sizeof(NIDType)));
}

TemporalSampler::OutputBufferTuple TemporalSampler::GetOutputBufferTuple(
    const Buffer& buffer, std::size_t num_root_nodes,
    std::size_t maximum_sampled_nodes) const {
  std::size_t offset1 = maximum_sampled_nodes * sizeof(NIDType);
  std::size_t offset2 = offset1 + maximum_sampled_nodes * sizeof(EIDType);
  std::size_t offset3 = offset2 + maximum_sampled_nodes * sizeof(TimestampType);
  std::size_t offset4 = offset3 + maximum_sampled_nodes * sizeof(TimestampType);
  return std::make_tuple(reinterpret_cast<NIDType*>(static_cast<char*>(buffer)),
                         reinterpret_cast<EIDType*>(buffer + offset1),
                         reinterpret_cast<TimestampType*>(buffer + offset2),
                         reinterpret_cast<TimestampType*>(buffer + offset3),
                         reinterpret_cast<uint32_t*>(buffer + offset4));
}

SamplingResult TemporalSampler::SampleLayer(
    const std::vector<NIDType>& dst_nodes,
    const std::vector<TimestampType>& dst_timestamps, uint32_t layer,
    uint32_t snapshot) {
  // NB: it seems to be necessary to set the device again.
  CUDA_CALL(cudaSetDevice(device_));

  std::size_t num_root_nodes = dst_nodes.size();
  std::size_t maximum_sampled_nodes = fanouts_[layer] * num_root_nodes;

  if (maximum_sampled_nodes > maximum_sampled_nodes_) {
    maximum_sampled_nodes_ = maximum_sampled_nodes;
    InitBuffer(num_root_nodes, maximum_sampled_nodes_);
  }

  // copy input to pin memory buffer
  auto input_buffer_tuple = GetInputBufferTuple(*cpu_buffer_, num_root_nodes);
  std::copy(dst_nodes.begin(), dst_nodes.end(),
            std::get<0>(input_buffer_tuple));
  std::copy(dst_timestamps.begin(), dst_timestamps.end(),
            std::get<1>(input_buffer_tuple));

  // copy input to GPU buffer
  CUDA_CALL(cudaMemcpyAsync(
      *gpu_input_buffer_, *cpu_buffer_,
      num_root_nodes * (sizeof(NIDType) + sizeof(TimestampType)),
      cudaMemcpyHostToDevice, stream_holders_[snapshot]));

  auto d_input_buffer_tuple =
      GetInputBufferTuple(*gpu_input_buffer_, num_root_nodes);
  NIDType* d_root_nodes = std::get<0>(d_input_buffer_tuple);
  TimestampType* d_root_timestamps = std::get<1>(d_input_buffer_tuple);

  // device output
  NIDType* d_src_nodes = nullptr;
  EIDType* d_eids = nullptr;
  TimestampType* d_timestamps = nullptr;
  TimestampType* d_delta_timestamps = nullptr;
  uint32_t* d_num_sampled = nullptr;

  std::tie(d_src_nodes, d_eids, d_timestamps, d_delta_timestamps,
           d_num_sampled) =
      GetOutputBufferTuple(*gpu_output_buffer_, num_root_nodes,
                           maximum_sampled_nodes);

  uint32_t num_threads_per_block = 256;
  uint32_t num_blocks =
      (num_root_nodes + num_threads_per_block - 1) / num_threads_per_block;

  if (sampling_policy_ == SamplingPolicy::kSamplingPolicyRecent) {
    SampleLayerRecentKernel<<<num_blocks, num_threads_per_block, 0,
                              stream_holders_[snapshot]>>>(
        graph_.get_device_node_table(), graph_.num_nodes(), prop_time_,
        d_root_nodes, d_root_timestamps, snapshot, num_snapshots_,
        snapshot_time_window_, num_root_nodes, fanouts_[layer], d_src_nodes,
        d_eids, d_timestamps, d_delta_timestamps, d_num_sampled);
  } else if (sampling_policy_ == SamplingPolicy::kSamplingPolicyUniform) {
    int offset_per_thread =
        shared_memory_size_ / sizeof(SamplingRange) / num_threads_per_block;

    LOG(DEBUG) << "Max shared memory size: " << shared_memory_size_ << " bytes"
               << ", offset per thread: " << offset_per_thread;

    SampleLayerUniformKernel<<<num_blocks, num_threads_per_block,
                               offset_per_thread * num_threads_per_block *
                                   sizeof(SamplingRange),
                               stream_holders_[snapshot]>>>(
        graph_.get_device_node_table(), graph_.num_nodes(), prop_time_,
        *rand_states_, seed_, offset_per_thread, d_root_nodes,
        d_root_timestamps, snapshot, num_snapshots_, snapshot_time_window_,
        num_root_nodes, fanouts_[layer], d_src_nodes, d_eids, d_timestamps,
        d_delta_timestamps, d_num_sampled);
  }

  // combine
  auto new_end = thrust::remove_if(
      thrust::cuda::par.on(stream_holders_[snapshot]),
      thrust::make_zip_iterator(thrust::make_tuple(
          d_src_nodes, d_eids, d_timestamps, d_delta_timestamps)),
      thrust::make_zip_iterator(thrust::make_tuple(
          d_src_nodes + maximum_sampled_nodes, d_eids + maximum_sampled_nodes,
          d_timestamps + maximum_sampled_nodes,
          d_delta_timestamps + maximum_sampled_nodes)),
      is_invalid_edge());

  uint32_t num_sampled_nodes = thrust::distance(
      thrust::make_zip_iterator(thrust::make_tuple(
          d_src_nodes, d_eids, d_timestamps, d_delta_timestamps)),
      new_end);

  LOG(DEBUG) << "Number of sampled nodes: " << num_sampled_nodes;

  NIDType* src_nodes = nullptr;
  EIDType* eids = nullptr;
  TimestampType* timestamps = nullptr;
  TimestampType* delta_timestamps = nullptr;
  uint32_t* num_sampled = nullptr;

  std::tie(src_nodes, eids, timestamps, delta_timestamps, num_sampled) =
      GetOutputBufferTuple(*cpu_buffer_, num_root_nodes, maximum_sampled_nodes);

  CUDA_CALL(cudaMemcpyAsync(src_nodes, d_src_nodes,
                            sizeof(NIDType) * num_sampled_nodes,
                            cudaMemcpyDeviceToHost, stream_holders_[snapshot]));
  CUDA_CALL(cudaMemcpyAsync(eids, d_eids, sizeof(EIDType) * num_sampled_nodes,
                            cudaMemcpyDeviceToHost, stream_holders_[snapshot]));
  CUDA_CALL(cudaMemcpyAsync(timestamps, d_timestamps,
                            sizeof(TimestampType) * num_sampled_nodes,
                            cudaMemcpyDeviceToHost, stream_holders_[snapshot]));
  CUDA_CALL(cudaMemcpyAsync(delta_timestamps, d_delta_timestamps,
                            sizeof(TimestampType) * num_sampled_nodes,
                            cudaMemcpyDeviceToHost, stream_holders_[snapshot]));
  CUDA_CALL(cudaMemcpyAsync(num_sampled, d_num_sampled,
                            sizeof(uint32_t) * num_root_nodes,
                            cudaMemcpyDeviceToHost, stream_holders_[snapshot]));

  SamplingResult sampling_result;
  // first copy dst nodes
  std::copy(dst_nodes.begin(), dst_nodes.end(),
            std::back_inserter(sampling_result.all_nodes));
  std::copy(dst_timestamps.begin(), dst_timestamps.end(),
            std::back_inserter(sampling_result.all_timestamps));

  sampling_result.col.resize(num_sampled_nodes);
  std::iota(sampling_result.col.begin(), sampling_result.col.end(),
            num_root_nodes);

  sampling_result.num_dst_nodes = num_root_nodes;
  sampling_result.num_src_nodes = num_root_nodes + num_sampled_nodes;

  sampling_result.all_nodes.reserve(sampling_result.num_src_nodes);
  sampling_result.all_timestamps.reserve(sampling_result.num_src_nodes);
  sampling_result.delta_timestamps.reserve(num_sampled_nodes);
  sampling_result.eids.reserve(num_sampled_nodes);

  // synchronize memcpy
  CUDA_CALL(cudaStreamSynchronize(stream_holders_[snapshot]));

  std::copy(src_nodes, src_nodes + num_sampled_nodes,
            std::back_inserter(sampling_result.all_nodes));
  std::copy(timestamps, timestamps + num_sampled_nodes,
            std::back_inserter(sampling_result.all_timestamps));
  std::copy(delta_timestamps, delta_timestamps + num_sampled_nodes,
            std::back_inserter(sampling_result.delta_timestamps));
  std::copy(eids, eids + num_sampled_nodes,
            std::back_inserter(sampling_result.eids));

  sampling_result.row.resize(num_sampled_nodes);
  uint32_t cumsum = 0;
  for (uint32_t i = 0; i < num_root_nodes; i++) {
    std::fill_n(sampling_result.row.begin() + cumsum, num_sampled[i], i);
    cumsum += num_sampled[i];
  }

  CHECK_EQ(cumsum, num_sampled_nodes);

  return sampling_result;
}

std::vector<std::vector<SamplingResult>> TemporalSampler::Sample(
    const std::vector<NIDType>& dst_nodes,
    const std::vector<TimestampType>& dst_timestamps) {
  CHECK_EQ(dst_nodes.size(), dst_timestamps.size());
  std::vector<std::vector<SamplingResult>> results;

  for (int layer = 0; layer < num_layers_; ++layer) {
    std::vector<SamplingResult> layer_results;
    if (layer == 0) {
      for (int snapshot = 0; snapshot < num_snapshots_; ++snapshot) {
        layer_results.push_back(
            SampleLayer(dst_nodes, dst_timestamps, layer, snapshot));
      }
      results.push_back(layer_results);
    } else {
      for (int snapshot = 0; snapshot < num_snapshots_; ++snapshot) {
        auto& prev_sample_result = results.back()[snapshot];
        auto& all_nodes = prev_sample_result.all_nodes;
        auto& all_timestamps = prev_sample_result.all_timestamps;
        layer_results.push_back(
            SampleLayer(all_nodes, all_timestamps, layer, snapshot));
      }
      results.push_back(layer_results);
    }
  }
  return results;
}

}  // namespace gnnflow
