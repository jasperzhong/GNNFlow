#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <rmm/device_vector.hpp>

#include "common.h"
#include "sampling_kernels.h"
#include "temporal_sampler.h"
#include "utils.h"

namespace dgnn {

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
      cpu_buffer_(nullptr) {
  if (num_snapshots_ == 1 && std::fabs(snapshot_time_window_) > 0.0f) {
    LOG(WARNING) << "Snapshot time window must be 0 when num_snapshots = 1. "
                    "Ignore the snapshot time window.";
  }
}

TemporalSampler::~TemporalSampler() {
  if (cpu_buffer_ != nullptr) {
    cudaFreeHost(cpu_buffer_);
  }
}

void TemporalSampler::InitBuffer(std::size_t num_root_nodes) {
  CHECK_EQ(cpu_buffer_, nullptr);

  std::size_t maximum_sampled_nodes = num_root_nodes * num_snapshots_;
  for (int i = 0; i < num_layers_; i++) {
    // including itself
    maximum_sampled_nodes += maximum_sampled_nodes * fanouts_[i];
  }
  LOG(DEBUG) << "Maximum sampled nodes: " << maximum_sampled_nodes;

  constexpr std::size_t per_node_size =
      sizeof(NIDType) + sizeof(TimestampType) + sizeof(TimestampType) +
      sizeof(EIDType) + sizeof(uint32_t);

  CUDA_CALL(
      cudaMallocHost(&cpu_buffer_, per_node_size * maximum_sampled_nodes));
  LOG(DEBUG) << "Allocated CPU buffer: "
             << maximum_sampled_nodes * per_node_size << " bytes";

  if (sampling_policy_ == SamplingPolicy::kSamplingPolicyUniform) {
    CUDA_CALL(cudaMalloc((void**)&rand_states_,
                         maximum_sampled_nodes * sizeof(curandState)));
    uint32_t num_threads_per_block = 256;
    uint32_t num_blocks =
        (num_root_nodes + num_threads_per_block - 1) / num_threads_per_block;

    InitCuRandStates<<<num_blocks, num_threads_per_block>>>(rand_states_,
                                                            seed_);
  }

  CHECK_NE(cpu_buffer_, nullptr);
}

std::vector<SamplingResult> TemporalSampler::RootInputToSamplingResult(
    const std::vector<NIDType>& dst_nodes,
    const std::vector<TimestampType>& dst_timestamps) {
  std::vector<SamplingResult> sampling_results(num_snapshots_);
  for (int snapshot = 0; snapshot < num_snapshots_; ++snapshot) {
    auto& sampling_result = sampling_results[snapshot];
    sampling_result.all_nodes.insert(sampling_result.all_nodes.end(),
                                     dst_nodes.begin(), dst_nodes.end());
    sampling_result.all_timestamps.insert(sampling_result.all_timestamps.end(),
                                          dst_timestamps.begin(),
                                          dst_timestamps.end());
  }
  return sampling_results;
}

std::vector<SamplingResult> TemporalSampler::SampleLayer(
    uint32_t layer, const std::vector<SamplingResult>& prev_sampling_results) {
  CHECK_EQ(prev_sampling_results.size(), num_snapshots_);
  // host input
  std::vector<uint32_t> cumsum_num_nodes;
  uint32_t cumsum = 0;
  for (int snapshot = 0; snapshot < num_snapshots_; ++snapshot) {
    cumsum_num_nodes.push_back(cumsum);
    cumsum += prev_sampling_results.at(snapshot).all_nodes.size();
  }
  uint32_t num_root_nodes = cumsum;

  std::size_t total_input_size =
      num_root_nodes * (sizeof(NIDType) + sizeof(TimestampType)) +
      cumsum_num_nodes.size() * sizeof(uint32_t);

  LOG(DEBUG) << "Total input size: " << total_input_size;

  // copy all_nodes and all_timestamps to cpu_buffer_
  for (int snapshot = 0; snapshot < num_snapshots_; ++snapshot) {
    auto& sampling_result = prev_sampling_results.at(snapshot);
    auto& all_nodes = sampling_result.all_nodes;
    auto& all_timestamps = sampling_result.all_timestamps;

    std::size_t offset = cumsum_num_nodes[snapshot];
    char* root_nodes_dst = cpu_buffer_ + offset * sizeof(NIDType);
    char* root_timestamps_dst = cpu_buffer_ + num_root_nodes * sizeof(NIDType) +
                                offset * sizeof(TimestampType);

    Copy(root_nodes_dst, all_nodes.data(), all_nodes.size() * sizeof(NIDType));
    Copy(root_timestamps_dst, all_timestamps.data(),
         all_timestamps.size() * sizeof(TimestampType));
  }
  memcpy(
      cpu_buffer_ + num_root_nodes * (sizeof(NIDType) + sizeof(TimestampType)),
      cumsum_num_nodes.data(), cumsum_num_nodes.size() * sizeof(uint32_t));

  // device input
  auto mr = rmm::mr::get_current_device_resource();
  char* d_input = reinterpret_cast<char*>(mr->allocate(total_input_size));
  CUDA_CALL(cudaMemcpy(d_input, cpu_buffer_, total_input_size,
                       cudaMemcpyHostToDevice));

  NIDType* d_root_nodes = reinterpret_cast<NIDType*>(d_input);
  TimestampType* d_root_timestamps = reinterpret_cast<TimestampType*>(
      d_input + num_root_nodes * sizeof(NIDType));
  uint32_t* d_cumsum_num_nodes = reinterpret_cast<uint32_t*>(
      d_input + num_root_nodes * (sizeof(NIDType) + sizeof(TimestampType)));

  // device output
  std::size_t offset1 = num_root_nodes * fanouts_[layer] * sizeof(NIDType);
  std::size_t offset2 =
      offset1 + num_root_nodes * fanouts_[layer] * sizeof(TimestampType);
  std::size_t offset3 =
      offset2 + num_root_nodes * fanouts_[layer] * sizeof(TimestampType);
  std::size_t offset4 =
      offset3 + num_root_nodes * fanouts_[layer] * sizeof(EIDType);
  std::size_t total_output_size = offset4 + num_root_nodes * sizeof(uint32_t);

  LOG(DEBUG) << "Total output size: " << total_output_size;

  char* d_output = reinterpret_cast<char*>(mr->allocate(total_output_size));

  NIDType* d_src_nodes = reinterpret_cast<NIDType*>(d_output);
  TimestampType* d_timestamps =
      reinterpret_cast<TimestampType*>(d_output + offset1);
  TimestampType* d_delta_timestamps =
      reinterpret_cast<TimestampType*>(d_output + offset2);
  EIDType* d_eids = reinterpret_cast<EIDType*>(d_output + offset3);
  uint32_t* d_num_sampled = reinterpret_cast<uint32_t*>(d_output + offset4);

  uint32_t num_threads_per_block = 256;
  uint32_t num_blocks =
      (num_root_nodes + num_threads_per_block - 1) / num_threads_per_block;

  if (sampling_policy_ == SamplingPolicy::kSamplingPolicyRecent) {
    SampleLayerRecentKernel<<<num_blocks, num_threads_per_block>>>(
        graph_.get_device_node_table(), graph_.num_nodes(), prop_time_,
        d_root_nodes, d_root_timestamps, d_cumsum_num_nodes, num_snapshots_,
        snapshot_time_window_, num_root_nodes, fanouts_[layer], d_src_nodes,
        d_timestamps, d_delta_timestamps, d_eids, d_num_sampled);
  } else if (sampling_policy_ == SamplingPolicy::kSamplingPolicyUniform) {
    auto max_shared_memory_size = GetSharedMemoryMaxSize();
    int offset_per_thread =
        max_shared_memory_size / sizeof(SamplingRange) / num_threads_per_block;

    LOG(DEBUG) << "Max shared memory size: " << max_shared_memory_size
               << " bytes"
               << ", offset per thread: " << offset_per_thread;

    // launch sampling kernel
    SampleLayerUniformKernel<<<num_blocks, num_threads_per_block,
                               offset_per_thread * num_threads_per_block *
                                   sizeof(SamplingRange)>>>(
        graph_.get_device_node_table(), graph_.num_nodes(), prop_time_,
        rand_states_, seed_, offset_per_thread, d_root_nodes, d_root_timestamps,
        d_cumsum_num_nodes, num_snapshots_, snapshot_time_window_,
        num_root_nodes, fanouts_[layer], d_src_nodes, d_timestamps,
        d_delta_timestamps, d_eids, d_num_sampled);
  }

  // copy output to host
  CUDA_CALL(cudaMemcpy(cpu_buffer_, d_output, total_output_size,
                       cudaMemcpyDeviceToHost));

  // host output
  NIDType* src_nodes = reinterpret_cast<NIDType*>(cpu_buffer_);
  TimestampType* timestamps =
      reinterpret_cast<TimestampType*>(cpu_buffer_ + offset1);
  TimestampType* delta_timestamps =
      reinterpret_cast<TimestampType*>(cpu_buffer_ + offset2);
  EIDType* eids = reinterpret_cast<EIDType*>(cpu_buffer_ + offset3);
  uint32_t* num_sampled = reinterpret_cast<uint32_t*>(cpu_buffer_ + offset4);

  // convert to SamplingResult
  std::vector<SamplingResult> sampling_results(num_snapshots_);
  for (int snapshot = 0; snapshot < num_snapshots_; ++snapshot) {
    auto& sampling_result = sampling_results[snapshot];

    uint32_t num_nodes_this_snapshot =
        prev_sampling_results.at(snapshot).all_nodes.size();
    uint32_t snapshot_offset = cumsum_num_nodes[snapshot];

    uint32_t cumsum_num_sampled[num_nodes_this_snapshot];
    cumsum_num_sampled[0] = 0;
    for (uint32_t i = 1; i < num_nodes_this_snapshot; ++i) {
      cumsum_num_sampled[i] = cumsum_num_sampled[i - 1] + num_sampled[i - 1];
    }
    uint32_t num_sampled_total =
        cumsum_num_sampled[num_nodes_this_snapshot - 1] +
        num_sampled[num_nodes_this_snapshot - 1];

    sampling_result.col.resize(num_sampled_total);
    std::iota(sampling_result.col.begin(), sampling_result.col.end(),
              num_nodes_this_snapshot);

    sampling_result.num_dst_nodes = num_nodes_this_snapshot;
    sampling_result.num_src_nodes = num_nodes_this_snapshot + num_sampled_total;

    sampling_result.all_nodes.reserve(sampling_result.num_src_nodes);
    sampling_result.all_timestamps.reserve(sampling_result.num_src_nodes);
    sampling_result.row.reserve(num_sampled_total);
    sampling_result.delta_timestamps.reserve(num_sampled_total);
    sampling_result.eids.reserve(num_sampled_total);

    // copy dst nodes
    std::copy(prev_sampling_results.at(snapshot).all_nodes.begin(),
              prev_sampling_results.at(snapshot).all_nodes.end(),
              std::back_inserter(sampling_result.all_nodes));
    std::copy(prev_sampling_results.at(snapshot).all_timestamps.begin(),
              prev_sampling_results.at(snapshot).all_timestamps.end(),
              std::back_inserter(sampling_result.all_timestamps));

    for (uint32_t i = 0; i < num_nodes_this_snapshot; i++) {
      std::vector<NIDType> row(num_sampled[snapshot_offset + i]);
      std::fill(row.begin(), row.end(), i);
      std::copy(row.begin(), row.end(),
                std::back_inserter(sampling_result.row));

      std::copy(
          src_nodes + (snapshot_offset + i) * fanouts_[layer],
          src_nodes + num_sampled[i] + (snapshot_offset + i) * fanouts_[layer],
          std::back_inserter(sampling_result.all_nodes));

      std::copy(
          timestamps + (snapshot_offset + i) * fanouts_[layer],
          timestamps + num_sampled[i] + (snapshot_offset + i) * fanouts_[layer],
          std::back_inserter(sampling_result.all_timestamps));

      std::copy(delta_timestamps + (snapshot_offset + i) * fanouts_[layer],
                delta_timestamps + num_sampled[i] +
                    (snapshot_offset + i) * fanouts_[layer],
                std::back_inserter(sampling_result.delta_timestamps));

      std::copy(eids + (snapshot_offset + i) * fanouts_[layer],
                eids + num_sampled[i] + (snapshot_offset + i) * fanouts_[layer],
                std::back_inserter(sampling_result.eids));
    }
  }

  mr->deallocate(d_input, total_input_size);
  mr->deallocate(d_output, total_output_size);

  return sampling_results;
}

std::vector<std::vector<SamplingResult>> TemporalSampler::Sample(
    const std::vector<NIDType>& dst_nodes,
    const std::vector<TimestampType>& timestamps) {
  CHECK_EQ(dst_nodes.size(), timestamps.size());
  std::vector<std::vector<SamplingResult>> results;

  if (cpu_buffer_ == nullptr) {
    InitBuffer(dst_nodes.size());
  }

  for (int layer = 0; layer < num_layers_; ++layer) {
    if (layer == 0) {
      results.push_back(
          SampleLayer(layer, RootInputToSamplingResult(dst_nodes, timestamps)));
    } else {
      results.push_back(SampleLayer(layer, results.back()));
    }
  }
  return results;
}
}  // namespace dgnn
