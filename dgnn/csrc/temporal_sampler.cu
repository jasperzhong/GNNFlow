#include <curand_kernel.h>

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
      seed_(seed) {
  if (num_snapshots_ == 1 && std::fabs(snapshot_time_window_) > 0.0f) {
    LOG(WARNING) << "Snapshot time window must be 0 when num_snapshots = 1. "
                    "Ignore the snapshot time window.";
  }
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
  std::vector<NIDType> root_nodes;
  std::vector<TimestampType> root_timestamps;
  std::vector<TimestampType> time_offsets;
  std::vector<uint32_t> num_nodes_per_snapshot;

  // from old to new
  for (int snapshot = 0; snapshot < num_snapshots_; ++snapshot) {
    root_nodes.insert(root_nodes.end(),
                      prev_sampling_results.at(snapshot).all_nodes.begin(),
                      prev_sampling_results.at(snapshot).all_nodes.end());

    root_timestamps.insert(
        root_timestamps.end(),
        prev_sampling_results.at(snapshot).all_timestamps.begin(),
        prev_sampling_results.at(snapshot).all_timestamps.end());

    uint32_t num_nodes = prev_sampling_results.at(snapshot).all_nodes.size();
    num_nodes_per_snapshot.push_back(num_nodes);

    float time_offset = snapshot_time_window_ * (num_snapshots_ - snapshot - 1);
    time_offsets.insert(time_offsets.end(), num_nodes, time_offset);
  }

  // device input
  rmm::device_vector<NIDType> d_root_nodes(root_nodes.begin(),
                                           root_nodes.end());
  rmm::device_vector<TimestampType> d_root_timestamps(root_timestamps.begin(),
                                                      root_timestamps.end());
  rmm::device_vector<TimestampType> d_time_offsets(time_offsets.begin(),
                                                   time_offsets.end());

  // device output
  uint32_t num_root_nodes = root_nodes.size();
  rmm::device_vector<NIDType> d_src_nodes(num_root_nodes * fanouts_[layer]);
  rmm::device_vector<TimestampType> d_timestamps(num_root_nodes *
                                                 fanouts_[layer]);
  rmm::device_vector<TimestampType> d_delta_timestamps(num_root_nodes *
                                                       fanouts_[layer]);
  rmm::device_vector<EIDType> d_eids(num_root_nodes * fanouts_[layer]);
  rmm::device_vector<uint32_t> d_num_sampled(num_root_nodes);

  uint32_t num_threads_per_block = 256;
  uint32_t num_blocks =
      (num_root_nodes + num_threads_per_block - 1) / num_threads_per_block;

  if (sampling_policy_ == SamplingPolicy::kSamplingPolicyRecent) {
    SampleLayerRecentKernel<<<num_blocks, num_threads_per_block>>>(
        graph_.get_device_node_table(), graph_.num_nodes(), prop_time_,
        thrust::raw_pointer_cast(d_root_nodes.data()),
        thrust::raw_pointer_cast(d_root_timestamps.data()),
        thrust::raw_pointer_cast(d_time_offsets.data()), snapshot_time_window_,
        num_root_nodes, fanouts_[layer],
        thrust::raw_pointer_cast(d_src_nodes.data()),
        thrust::raw_pointer_cast(d_timestamps.data()),
        thrust::raw_pointer_cast(d_delta_timestamps.data()),
        thrust::raw_pointer_cast(d_eids.data()),
        thrust::raw_pointer_cast(d_num_sampled.data()));
  } else if (sampling_policy_ == SamplingPolicy::kSamplingPolicyUniform) {
    rmm::device_vector<curandState_t> d_rand_states(num_threads_per_block *
                                                    num_blocks);
    auto rand_states = thrust::raw_pointer_cast(d_rand_states.data());

    auto max_shared_memory_size = GetSharedMemoryMaxSize();
    int offset_per_thread =
        max_shared_memory_size / sizeof(SamplingRange) / num_threads_per_block;

    // launch sampling kernel
    SampleLayerUniformKernel<<<num_blocks, num_threads_per_block,
                               offset_per_thread * num_threads_per_block *
                                   sizeof(SamplingRange)>>>(
        graph_.get_device_node_table(), graph_.num_nodes(), prop_time_,
        rand_states, seed_, offset_per_thread,
        thrust::raw_pointer_cast(d_root_nodes.data()),
        thrust::raw_pointer_cast(d_root_timestamps.data()),
        thrust::raw_pointer_cast(d_time_offsets.data()), snapshot_time_window_,
        num_root_nodes, fanouts_[layer],
        thrust::raw_pointer_cast(d_src_nodes.data()),
        thrust::raw_pointer_cast(d_timestamps.data()),
        thrust::raw_pointer_cast(d_delta_timestamps.data()),
        thrust::raw_pointer_cast(d_eids.data()),
        thrust::raw_pointer_cast(d_num_sampled.data()));
  }

  // host output
  std::vector<NIDType> src_nodes(num_root_nodes * fanouts_[layer]);
  std::vector<TimestampType> timestamps(num_root_nodes * fanouts_[layer]);
  std::vector<TimestampType> delta_timestamps(num_root_nodes * fanouts_[layer]);
  std::vector<EIDType> eids(num_root_nodes * fanouts_[layer]);
  std::vector<uint32_t> num_sampled(num_root_nodes);

  // copy output from device
  thrust::copy(d_src_nodes.begin(), d_src_nodes.end(), src_nodes.begin());
  thrust::copy(d_timestamps.begin(), d_timestamps.end(), timestamps.begin());
  thrust::copy(d_delta_timestamps.begin(), d_delta_timestamps.end(),
               delta_timestamps.begin());
  thrust::copy(d_eids.begin(), d_eids.end(), eids.begin());
  thrust::copy(d_num_sampled.begin(), d_num_sampled.end(), num_sampled.begin());

  // convert to SamplingResult
  std::vector<SamplingResult> sampling_results(num_snapshots_);
  uint32_t snapshot_offset = 0;
  for (int snapshot = 0; snapshot < num_snapshots_; ++snapshot) {
    auto& sampling_result = sampling_results[snapshot];

    // copy dst nodes
    std::copy(prev_sampling_results.at(snapshot).all_nodes.begin(),
              prev_sampling_results.at(snapshot).all_nodes.end(),
              std::back_inserter(sampling_result.all_nodes));
    std::copy(prev_sampling_results.at(snapshot).all_timestamps.begin(),
              prev_sampling_results.at(snapshot).all_timestamps.end(),
              std::back_inserter(sampling_result.all_timestamps));

    uint32_t num_nodes_this_snapshot = num_nodes_per_snapshot[snapshot];

    uint32_t num_sampled_total = 0;
    for (uint32_t i = 0; i < num_nodes_this_snapshot; i++) {
      std::vector<NIDType> row(num_sampled[snapshot_offset + i]);
      std::fill(row.begin(), row.end(), i);
      std::copy(row.begin(), row.end(),
                std::back_inserter(sampling_result.row));

      std::copy(src_nodes.begin() + (snapshot_offset + i) * fanouts_[layer],
                src_nodes.begin() + num_sampled[i] +
                    (snapshot_offset + i) * fanouts_[layer],
                std::back_inserter(sampling_result.all_nodes));

      std::copy(timestamps.begin() + (snapshot_offset + i) * fanouts_[layer],
                timestamps.begin() + num_sampled[i] +
                    (snapshot_offset + i) * fanouts_[layer],
                std::back_inserter(sampling_result.all_timestamps));

      std::copy(
          delta_timestamps.begin() + (snapshot_offset + i) * fanouts_[layer],
          delta_timestamps.begin() + num_sampled[i] +
              (snapshot_offset + i) * fanouts_[layer],
          std::back_inserter(sampling_result.delta_timestamps));

      std::copy(eids.begin() + (snapshot_offset + i) * fanouts_[layer],
                eids.begin() + num_sampled[i] +
                    (snapshot_offset + i) * fanouts_[layer],
                std::back_inserter(sampling_result.eids));

      num_sampled_total += num_sampled[i];
    }
    sampling_result.col.resize(num_sampled_total);
    std::iota(sampling_result.col.begin(), sampling_result.col.end(),
              num_nodes_this_snapshot);

    sampling_result.num_dst_nodes = num_nodes_this_snapshot;
    sampling_result.num_src_nodes = num_nodes_this_snapshot + num_sampled_total;

    snapshot_offset += num_nodes_this_snapshot;
  }

  return sampling_results;
}

std::vector<std::vector<SamplingResult>> TemporalSampler::Sample(
    const std::vector<NIDType>& dst_nodes,
    const std::vector<TimestampType>& timestamps) {
  CHECK_EQ(dst_nodes.size(), timestamps.size());
  std::vector<std::vector<SamplingResult>> results;

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
