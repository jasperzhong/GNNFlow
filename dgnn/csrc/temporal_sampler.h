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
                  float snapshot_time_window = 0.0f, bool prop_time = false,
                  uint64_t seed = 1234);
  ~TemporalSampler() = default;

  std::vector<std::vector<SamplingResult>> Sample(
      const std::vector<NIDType>& dst_nodes,
      const std::vector<TimestampType>& dst_timestamps);

 private:
  std::vector<SamplingResult> SampleLayer(
      uint32_t layer, const std::vector<SamplingResult>& prev_sampling_results);

  std::vector<SamplingResult> RootInputToSamplingResult(
      const std::vector<NIDType>& dst_nodes,
      const std::vector<TimestampType>& dst_timestamps);

 private:
  const DynamicGraph& graph_;
  std::vector<uint32_t> fanouts_;
  SamplingPolicy sampling_policy_;
  uint32_t num_snapshots_;
  float snapshot_time_window_;
  bool prop_time_;
  uint32_t num_layers_;
  uint64_t seed_;
};

}  // namespace dgnn

#endif  // DGNN_TEMPORAL_SAMPLER_H_
