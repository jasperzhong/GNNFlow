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

void TemporalSampler::FreeBuffer() {
  if (cpu_buffer_ != nullptr) {
    CUDA_CALL(cudaFreeHost(cpu_buffer_));
  }

  if (gpu_input_buffer_ != nullptr) {
    CUDA_CALL(cudaFree(gpu_input_buffer_));
  }

  if (gpu_output_buffer_ != nullptr) {
    CUDA_CALL(cudaFree(gpu_output_buffer_));
  }

  if (rand_states_ != nullptr) {
    CUDA_CALL(cudaFree(rand_states_));
  }
}

TemporalSampler::~TemporalSampler() { FreeBuffer(); }

void TemporalSampler::InitBuffer(std::size_t maximum_sampled_nodes) {
  CUDA_CALL(
      cudaMallocHost(&cpu_buffer_, kPerNodeBufferSize * maximum_sampled_nodes));

  CUDA_CALL(cudaMalloc(&gpu_input_buffer_,
                       kPerNodeBufferSize * maximum_sampled_nodes));
  CUDA_CALL(cudaMalloc(&gpu_output_buffer_,
                       kPerNodeBufferSize * maximum_sampled_nodes));

  LOG(DEBUG) << "Allocated CPU & GPU buffer: "
             << maximum_sampled_nodes * kPerNodeBufferSize << " bytes";

  if (sampling_policy_ == SamplingPolicy::kSamplingPolicyUniform) {
    CUDA_CALL(cudaMalloc((void**)&rand_states_,
                         maximum_sampled_nodes * sizeof(curandState)));
    uint32_t num_threads_per_block = 256;
    uint32_t num_blocks = (maximum_sampled_nodes + num_threads_per_block - 1) /
                          num_threads_per_block;

    InitCuRandStates<<<num_blocks, num_threads_per_block>>>(rand_states_,
                                                            seed_);
  }
}

SamplingResult TemporalSampler::SampleLayer(
    const std::vector<NIDType>& dst_nodes,
    const std::vector<TimestampType>& dst_timestamps, uint32_t layer,
    uint32_t snapshot) {
  // NB: it seems to be necessary to set the device again.
  CUDA_CALL(cudaSetDevice(device_));

  // update buffer. +1 means adding itself
  std::size_t maximum_sampled_nodes = (fanouts_[layer] + 1) * dst_nodes.size();
  if (maximum_sampled_nodes > maximum_sampled_nodes_) {
    FreeBuffer();
    maximum_sampled_nodes_ = maximum_sampled_nodes;
    InitBuffer(maximum_sampled_nodes);
  }

  // prepare input
  std::vector<std::size_t> num_root_nodes_list(num_snapshots_);
  std::size_t num_root_nodes = dst_nodes.size();
  char* root_nodes_dst = cpu_buffer_;
  char* root_timestamps_dst = cpu_buffer_ + num_root_nodes * sizeof(NIDType);
  // copy dst_nodes and dst_timestamps to cpu_buffer_
  Copy(root_nodes_dst, dst_nodes.data(), dst_nodes.size() * sizeof(NIDType));
  Copy(root_timestamps_dst, dst_timestamps.data(),
       dst_timestamps.size() * sizeof(TimestampType));

  CUDA_CALL(cudaMemcpyAsync(
      gpu_input_buffer_, cpu_buffer_,
      num_root_nodes * (sizeof(NIDType) + sizeof(TimestampType)),
      cudaMemcpyHostToDevice, stream_holders_[snapshot]));

  // launch kernel for current snapshot
  std::size_t max_sampled_nodes = num_root_nodes * fanouts_[layer];
  NIDType* d_root_nodes = reinterpret_cast<NIDType*>(gpu_input_buffer_);
  TimestampType* d_root_timestamps = reinterpret_cast<TimestampType*>(
      gpu_input_buffer_ + num_root_nodes * sizeof(NIDType));

  // device output
  std::size_t offset1 = max_sampled_nodes * sizeof(NIDType);
  std::size_t offset2 = offset1 + max_sampled_nodes * sizeof(EIDType);
  std::size_t offset3 = offset2 + max_sampled_nodes * sizeof(TimestampType);
  std::size_t offset4 = offset3 + max_sampled_nodes * sizeof(TimestampType);

  std::size_t total_output_size = offset4 + num_root_nodes * sizeof(uint32_t);

  LOG(DEBUG) << "Total output size: " << total_output_size;

  NIDType* d_src_nodes = reinterpret_cast<NIDType*>(gpu_output_buffer_);
  EIDType* d_eids = reinterpret_cast<EIDType*>(gpu_output_buffer_ + offset1);
  TimestampType* d_timestamps =
      reinterpret_cast<TimestampType*>(gpu_output_buffer_ + offset2);
  TimestampType* d_delta_timestamps =
      reinterpret_cast<TimestampType*>(gpu_output_buffer_ + offset3);
  uint32_t* d_num_sampled =
      reinterpret_cast<uint32_t*>(gpu_output_buffer_ + offset4);

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
        rand_states_, seed_, offset_per_thread, d_root_nodes, d_root_timestamps,
        snapshot, num_snapshots_, snapshot_time_window_, num_root_nodes,
        fanouts_[layer], d_src_nodes, d_eids, d_timestamps, d_delta_timestamps,
        d_num_sampled);
  }

  // combine
  auto new_end = thrust::remove_if(
      // rmm::exec_policy(stream_holders_[snapshot]),
      thrust::cuda::par.on(stream_holders_[snapshot]),
      thrust::make_zip_iterator(thrust::make_tuple(
          d_src_nodes, d_eids, d_timestamps, d_delta_timestamps)),
      thrust::make_zip_iterator(thrust::make_tuple(
          d_src_nodes + max_sampled_nodes, d_eids + max_sampled_nodes,
          d_timestamps + max_sampled_nodes,
          d_delta_timestamps + max_sampled_nodes)),
      is_invalid_edge());

  uint32_t num_sampled_nodes = thrust::distance(
      thrust::make_zip_iterator(thrust::make_tuple(
          d_src_nodes, d_eids, d_timestamps, d_delta_timestamps)),
      new_end);

  LOG(DEBUG) << "Number of sampled nodes: " << num_sampled_nodes;

  NIDType* src_nodes = reinterpret_cast<NIDType*>(cpu_buffer_);
  EIDType* eids = reinterpret_cast<EIDType*>(cpu_buffer_ + offset1);
  TimestampType* timestamps =
      reinterpret_cast<TimestampType*>(cpu_buffer_ + offset2);
  TimestampType* delta_timestamps =
      reinterpret_cast<TimestampType*>(cpu_buffer_ + offset3);
  uint32_t* num_sampled = reinterpret_cast<uint32_t*>(cpu_buffer_ + offset4);

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
