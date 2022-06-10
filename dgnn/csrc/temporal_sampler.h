#ifndef DGNN_TEMPORAL_SAMPLER_H_
#define DGNN_TEMPORAL_SAMPLER_H_

#include "common.h"
#include "dynamic_graph.h"

namespace dgnn {

class TemporalSampler {
 public:
  TemporalSampler(const DynamicGraph& graph,
                  const std::vector<uint32_t>& fanouts,
                  SamplingPolicy sample_policy, uint32_t num_snapshots = 1,
                  float snapshot_time_window = 0.0f);
  ~TemporalSampler() = default;

  std::vector<std::vector<SamplingResult>> Sample(
      const std::vector<NIDType>& dst_nodes,
      const std::vector<TimestampType>& timestamps, bool prop_time = false,
      bool reverse = false);

  std::vector<SamplingResult> SampleLayerFromRoot(
      const std::vector<NIDType>& dst_nodes,
      const std::vector<TimestampType>& timestamps, bool prop_time = false,
      bool reverse = false);

  std::vector<SamplingResult> SampleLayerFromPreviousLayer(
      const std::vector<NIDType>& dst_nodes,
      const std::vector<TimestampType>& timestamps, bool prop_time = false,
      bool reverse = false);

 private:
  const DynamicGraph& graph_;
  const std::vector<uint32_t>& fanouts_;
  SamplingPolicy sampling_policy_;
  uint32_t num_snapshots_;
  float snapshot_time_window_;
  uint32_t num_layers_;
};

}  // namespace dgnn

#endif  // DGNN_TEMPORAL_SAMPLER_H_
