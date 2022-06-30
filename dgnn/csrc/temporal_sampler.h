#ifndef DGNN_TEMPORAL_SAMPLER_H_
#define DGNN_TEMPORAL_SAMPLER_H_

#include <cuda_runtime_api.h>
#include <curand_kernel.h>

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
  ~TemporalSampler();

  std::vector<std::vector<SamplingResult>> Sample(
      const std::vector<NIDType>& dst_nodes,
      const std::vector<TimestampType>& dst_timestamps);

 private:
  std::vector<SamplingResult> SampleLayer(
      uint32_t layer, const std::vector<SamplingResult>& prev_sampling_results);

  std::vector<SamplingResult> RootInputToSamplingResult(
      const std::vector<NIDType>& dst_nodes,
      const std::vector<TimestampType>& dst_timestamps);

  void InitBuffer(std::size_t num_root_nodes);

  void FreeBuffer();

 private:
  const DynamicGraph& graph_;
  std::vector<uint32_t> fanouts_;
  SamplingPolicy sampling_policy_;
  uint32_t num_snapshots_;
  float snapshot_time_window_;
  bool prop_time_;
  uint32_t num_layers_;
  uint64_t seed_;
  std::size_t shared_memory_size_;

  cudaStream_t* streams_;
  char** cpu_buffer_;
  char** gpu_input_buffer_;
  char** gpu_output_buffer_;
  curandState_t** rand_states_;
  std::size_t init_num_root_nodes_;
};

}  // namespace dgnn

#endif  // DGNN_TEMPORAL_SAMPLER_H_
