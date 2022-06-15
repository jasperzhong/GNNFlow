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
    const std::vector<TimestampType>& dst_timestamps) {
  std::vector<NIDType> root_nodes;
  std::vector<TimestampType> root_timestamps;
  std::vector<TimestampType> offsets(num_snapshots_ * dst_nodes.size());

  for (int snapshot = num_snapshots_ - 1; snapshot >= 0; snapshot--) {
    float time_offset = snapshot_time_window_ * (num_snapshots_ - snapshot - 1);
    root_nodes.insert(root_nodes.end(), dst_nodes.begin(), dst_nodes.end());
    root_timestamps.insert(root_timestamps.end(), dst_timestamps.begin(),
                           dst_timestamps.end());
    std::fill(
        offsets.begin() + (num_snapshots_ - snapshot - 1) * dst_nodes.size(),
        offsets.begin() + (num_snapshots_ - snapshot) * dst_nodes.size(),
        time_offset);
  }

  thrust::device_vector<NIDType> d_root_nodes(root_nodes.begin(),
                                              root_nodes.end());
  thrust::device_vector<TimestampType> d_root_timestamps(
      root_timestamps.begin(), root_timestamps.end());
  thrust::device_vector<TimestampType> d_offsets(offsets.begin(),
                                                 offsets.end());

  uint32_t num_root_nodes = root_nodes.size();
  thrust::device_vector<NIDType> d_src_nodes(num_root_nodes * fanouts_[0]);
  thrust::device_vector<TimestampType> d_timestamps(num_root_nodes *
                                                    fanouts_[0]);
  thrust::device_vector<TimestampType> d_delta_timestamps(num_root_nodes *
                                                          fanouts_[0]);
  thrust::device_vector<EIDType> d_eids(num_root_nodes * fanouts_[0]);
  thrust::device_vector<uint32_t> d_num_sampled(num_root_nodes);

  uint32_t num_threads_per_block = 1024;
  uint32_t num_blocks =
      (num_root_nodes + num_threads_per_block - 1) / num_threads_per_block;

  curandState_t* rand_states = nullptr;
  if (sampling_policy_ == SamplingPolicy::kSamplingPolicyUniform) {
    thrust::device_vector<curandState_t> d_rand_states(num_threads_per_block *
                                                       num_blocks);
    rand_states = thrust::raw_pointer_cast(d_rand_states.data());
  }

  SampleLayerFromRootKernel<<<num_blocks, num_threads_per_block>>>(
      graph_.get_device_node_table(), graph_.num_nodes(), sampling_policy_,
      prop_time_, rand_states, seed_,
      thrust::raw_pointer_cast(d_root_nodes.data()),
      thrust::raw_pointer_cast(d_root_timestamps.data()),
      thrust::raw_pointer_cast(d_offsets.data()), snapshot_time_window_,
      num_root_nodes, fanouts_[0], thrust::raw_pointer_cast(d_src_nodes.data()),
      thrust::raw_pointer_cast(d_timestamps.data()),
      thrust::raw_pointer_cast(d_delta_timestamps.data()),
      thrust::raw_pointer_cast(d_eids.data()),
      thrust::raw_pointer_cast(d_num_sampled.data()));

  std::vector<SamplingResult> sampling_results(num_snapshots_);
  uint32_t num_dst_nodes_per_snapshot = dst_nodes.size();
  for (int snapshot = num_snapshots_ - 1; snapshot >= 0; snapshot--) {
    uint32_t offset =
        (num_snapshots_ - snapshot - 1) * num_dst_nodes_per_snapshot;

    std::vector<NIDType> src_nodes(num_dst_nodes_per_snapshot * fanouts_[0]);
    std::vector<TimestampType> timestamps(num_dst_nodes_per_snapshot *
                                          fanouts_[0]);
    std::vector<TimestampType> delta_timestamps(num_dst_nodes_per_snapshot *
                                                fanouts_[0]);
    std::vector<EIDType> eids(num_dst_nodes_per_snapshot * fanouts_[0]);

    // copy result from GPU
    thrust::copy(d_src_nodes.begin() + offset * fanouts_[0],
                 d_src_nodes.begin() +
                     (offset + num_dst_nodes_per_snapshot) * fanouts_[0],
                 src_nodes.begin());

    thrust::copy(d_timestamps.begin() + offset * fanouts_[0],
                 d_timestamps.begin() +
                     (offset + num_dst_nodes_per_snapshot) * fanouts_[0],
                 timestamps.begin());

    thrust::copy(d_delta_timestamps.begin() + offset * fanouts_[0],
                 d_delta_timestamps.begin() +
                     (offset + num_dst_nodes_per_snapshot) * fanouts_[0],
                 delta_timestamps.begin());

    thrust::copy(
        d_eids.begin() + offset * fanouts_[0],
        d_eids.begin() + (offset + num_dst_nodes_per_snapshot) * fanouts_[0],
        eids.begin());

    std::vector<uint32_t> num_sampled(num_dst_nodes_per_snapshot);
    thrust::copy(d_num_sampled.begin() + offset,
                 d_num_sampled.begin() + offset + num_dst_nodes_per_snapshot,
                 num_sampled.begin());

    // copy result to sampling result
    auto& sampling_result = sampling_results[snapshot];

    // copy dst nodes
    std::copy(dst_nodes.begin(), dst_nodes.end(),
              std::back_inserter(sampling_result.all_nodes));

    std::copy(dst_timestamps.begin(), dst_timestamps.end(),
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
