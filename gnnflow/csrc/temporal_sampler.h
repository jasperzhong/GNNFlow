#ifndef GNNFLOW_TEMPORAL_SAMPLER_H_
#define GNNFLOW_TEMPORAL_SAMPLER_H_

#include <cuda_runtime_api.h>
#include <curand_kernel.h>

#include <cstddef>
#include <cstdint>

#include "common.h"
#include "dynamic_graph.h"

namespace gnnflow {

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

  // NB: this function should handle input with dynamic length (i.e.,
  // `dst_nodes` and `dst_timestamps` can have dynamic lengths every time). Make
  // sure to re-allocate the buffer if the input size is larger than the current
  // buffer size.
  SamplingResult SampleLayer(const std::vector<NIDType>& dst_nodes,
                             const std::vector<TimestampType>& dst_timestamps,
                             uint32_t layer, uint32_t snapshot);

 private:
  void InitBuffer(std::size_t maximum_sampled_nodes);

  void FreeBuffer();

 private:
  constexpr static std::size_t per_node_size =
      sizeof(NIDType) + sizeof(TimestampType) + sizeof(EIDType) +
      sizeof(uint32_t);

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
  char* cpu_buffer_;
  char* gpu_input_buffer_;
  char* gpu_output_buffer_;
  curandState_t* rand_states_;

  std::size_t maximum_sampled_nodes_;

  int device_;
};

}  // namespace gnnflow

#endif  // GNNFLOW_TEMPORAL_SAMPLER_H_
