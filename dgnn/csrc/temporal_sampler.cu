#include <curand_kernel.h>
#include <thrust/device_vector.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>

#include "common.h"
#include "sampling_kernels.h"
#include "temporal_sampler.h"

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

std::vector<SamplingResult> TemporalSampler::SampleLayerFromRoot(
    const std::vector<NIDType>& dst_nodes,
    const std::vector<TimestampType>& timestamps) {
  std::vector<NIDType> root_nodes;
  std::vector<TimestampType> start_timestamps;
  std::vector<TimestampType> end_timestamps;

  for (int snapshot = 0; snapshot < num_snapshots_; snapshot++) {
    float time_offset = snapshot_time_window_ * snapshot;
    for (uint32_t i = 0; i < dst_nodes.size(); i++) {
      root_nodes.push_back(dst_nodes[i]);
      float end_timestamp = timestamps[i] - time_offset;
      float start_timestamp = (snapshot_time_window_ > 0)
                                  ? end_timestamp - snapshot_time_window_
                                  : 0;
      start_timestamps.push_back(start_timestamp);
      end_timestamps.push_back(end_timestamp);
    }
  }

  thrust::device_vector<NIDType> d_root_nodes(root_nodes.begin(),
                                              root_nodes.end());
  thrust::device_vector<TimestampType> d_start_timestamps(
      start_timestamps.begin(), start_timestamps.end());
  thrust::device_vector<TimestampType> d_end_timestamps(end_timestamps.begin(),
                                                        end_timestamps.end());

  uint32_t num_dst_nodes = root_nodes.size();
  thrust::device_vector<NIDType> d_src_nodes(num_dst_nodes * fanouts_[0]);
  thrust::device_vector<TimestampType> d_timestamps(num_dst_nodes *
                                                    fanouts_[0]);
  thrust::device_vector<TimestampType> d_delta_timestamps(num_dst_nodes *
                                                          fanouts_[0]);
  thrust::device_vector<EIDType> d_eids(num_dst_nodes * fanouts_[0]);
  thrust::device_vector<uint32_t> d_num_sampled(num_dst_nodes);

  uint32_t num_threads_per_block = 1024;
  uint32_t num_blocks =
      (num_dst_nodes + num_threads_per_block - 1) / num_threads_per_block;

  curandState_t* rand_states = nullptr;
  if (sampling_policy_ == SamplingPolicy::kSamplingPolicyUniform) {
    thrust::device_vector<curandState_t> d_rand_states(num_threads_per_block *
                                                       num_blocks);
    rand_states = thrust::raw_pointer_cast(d_rand_states.data());
  }

  SampleLayerFromRootKernel<<<num_blocks, num_threads_per_block>>>(
      graph_.get_device_node_table(), graph_.num_nodes(), sampling_policy_,
      rand_states, seed_, thrust::raw_pointer_cast(d_root_nodes.data()),
      thrust::raw_pointer_cast(d_start_timestamps.data()),
      thrust::raw_pointer_cast(d_end_timestamps.data()), num_dst_nodes,
      fanouts_[0], thrust::raw_pointer_cast(d_src_nodes.data()),
      thrust::raw_pointer_cast(d_timestamps.data()),
      thrust::raw_pointer_cast(d_delta_timestamps.data()),
      thrust::raw_pointer_cast(d_eids.data()),
      thrust::raw_pointer_cast(d_num_sampled.data()));

  std::vector<SamplingResult> sampling_results(num_snapshots_);
  uint32_t num_dst_nodes_per_snapshot = dst_nodes.size();
  for (int snapshot = 0; snapshot < num_snapshots_; snapshot++) {
    uint32_t offset = snapshot * num_dst_nodes_per_snapshot;

    std::vector<NIDType> src_nodes(num_dst_nodes_per_snapshot * fanouts_[0]);
    std::vector<TimestampType> timestamps(num_dst_nodes_per_snapshot *
                                          fanouts_[0]);
    std::vector<TimestampType> delta_timestamps(num_dst_nodes_per_snapshot *
                                                fanouts_[0]);
    std::vector<EIDType> eids(num_dst_nodes_per_snapshot * fanouts_[0]);

    // copy result from GPU
    thrust::copy(
        d_src_nodes.begin() + offset,
        d_src_nodes.begin() + offset + num_dst_nodes_per_snapshot * fanouts_[0],
        src_nodes.begin());

    thrust::copy(d_timestamps.begin() + offset,
                 d_timestamps.begin() + offset +
                     num_dst_nodes_per_snapshot * fanouts_[0],
                 timestamps.begin());

    thrust::copy(d_delta_timestamps.begin() + offset,
                 d_delta_timestamps.begin() + offset +
                     num_dst_nodes_per_snapshot * fanouts_[0],
                 delta_timestamps.begin());

    thrust::copy(
        d_eids.begin() + offset,
        d_eids.begin() + offset + num_dst_nodes_per_snapshot * fanouts_[0],
        eids.begin());

    std::vector<uint32_t> num_sampled(num_dst_nodes_per_snapshot);
    thrust::copy(d_num_sampled.begin() + offset,
                 d_num_sampled.begin() + offset + num_dst_nodes_per_snapshot,
                 num_sampled.begin());

    // copy result to sampling result
    auto& sampling_result = sampling_results[snapshot];

    // copy dst nodes
    std::copy(root_nodes.begin() + offset,
              root_nodes.begin() + offset + num_dst_nodes_per_snapshot,
              std::back_inserter(sampling_result.all_nodes));

    std::copy(end_timestamps.begin() + offset,
              end_timestamps.begin() + offset + num_dst_nodes_per_snapshot,
              std::back_inserter(sampling_result.all_timestamps));

    // copy src nodes
    uint32_t num_sampled_total = 0;
    for (uint32_t i = 0; i < num_dst_nodes_per_snapshot; i++) {
      std::vector<NIDType> row(num_sampled[i]);
      std::fill(row.begin(), row.end(), i);
      std::copy(row.begin(), row.end(),
                std::back_inserter(sampling_result.row));

      std::copy(src_nodes.begin() + i * fanouts_[0],
                src_nodes.begin() + num_sampled[i] + i * fanouts_[0],
                std::back_inserter(sampling_result.all_nodes));

      std::copy(timestamps.begin() + i * fanouts_[0],
                timestamps.begin() + num_sampled[i] + i * fanouts_[0],
                std::back_inserter(sampling_result.all_timestamps));

      std::copy(delta_timestamps.begin() + i * fanouts_[0],
                delta_timestamps.begin() + num_sampled[i] + i * fanouts_[0],
                std::back_inserter(sampling_result.delta_timestamps));

      std::copy(eids.begin() + i * fanouts_[0],
                eids.begin() + num_sampled[i] + i * fanouts_[0],
                std::back_inserter(sampling_result.eids));

      num_sampled_total += num_sampled[i];
    }
    sampling_result.col.resize(num_sampled_total);
    std::iota(sampling_result.col.begin(), sampling_result.col.end(),
              num_dst_nodes_per_snapshot);

    sampling_result.num_dst_nodes = num_dst_nodes_per_snapshot;
    sampling_result.num_src_nodes =
        num_dst_nodes_per_snapshot + num_sampled_total;
  }

  return sampling_results;
}

std::vector<std::vector<SamplingResult>> TemporalSampler::Sample(
    const std::vector<NIDType>& dst_nodes,
    const std::vector<TimestampType>& timestamps) {
  CHECK_EQ(dst_nodes.size(), timestamps.size());
  std::vector<std::vector<SamplingResult>> results;

  return results;
}
}  // namespace dgnn
