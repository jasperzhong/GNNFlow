#include <algorithm>

#include "common.h"
#include "temporal_sampler.h"

namespace dgnn {
TemporalSampler::TemporalSampler(const DynamicGraph& graph,
                                 const std::vector<uint32_t>& fanouts,
                                 SamplingPolicy sampling_policy,
                                 uint32_t num_snapshots,
                                 float snapshot_time_window)
    : graph_(graph),
      fanouts_(fanouts),
      sampling_policy_(sampling_policy),
      num_snapshots_(num_snapshots),
      snapshot_time_window_(snapshot_time_window),
      num_layers_(fanouts.size()) {
  if (num_snapshots_ == 1 && snapshot_time_window_ == 0.0f) {
    LOG(WARNING) << "Snapshot time window must be 0 when num_snapshots = 1. "
                    "Ignore the snapshot time window.";
  }
}

__host__ __device__ void SampleLayerHelper(DoublyLinkedList* list,
                                           SamplingResult* sampling_result,
                                           uint32_t fanout,
                                           float start_timestamp,
                                           float end_timestamp,
                                           SamplingPolicy sampling_policy) {
  auto cur = list->head;
  while (cur != nullptr) {
    if (end_timestamp < cur->timestamps[0]) {
      // search in the next block
      cur = cur->next;
      continue;
    }

    if (start_timestamp > cur->timestamps[cur->size - 1]) {
      // no need to search in the next block
      break;
    }

    if (start_timestamp >= cur->timestamps[0] &&
        end_timestamp <= cur->timestamps[cur->size - 1]) {
      // all edges are in the current block
    } else if (start_timestamp < cur->timestamps[0] &&
               end_timestamp <= cur->timestamps[cur->size - 1]) {
      // only the edges before end_timestamp are in the current block
    } else if (start_timestamp >= cur->timestamps[0] &&
               end_timestamp > cur->timestamps[cur->size - 1]) {
      // only the edges after start_timestamp are in the current block
    } else {
      // the whole block is in the range
    }
  }
}

std::vector<SamplingResult> TemporalSampler::SampleLayerFromRoot(
    const std::vector<NIDType>& dst_nodes,
    const std::vector<TimestampType>& timestamps, bool prop_time,
    bool reverse) {
  for (uint32_t snapshot = num_snapshots_ - 1; snapshot >= 0; snapshot--) {
    float time_offset = snapshot_time_window_ * (num_snapshots_ - snapshot - 1);
  }
}

std::vector<std::vector<SamplingResult>> TemporalSampler::Sample(
    const std::vector<NIDType>& dst_nodes,
    const std::vector<TimestampType>& timestamps, bool prop_time,
    bool reverse) {
  CHECK_EQ(dst_nodes.size(), timestamps.size());
  std::vector<std::vector<SamplingResult>> results;

  return results;
}
}  // namespace dgnn
