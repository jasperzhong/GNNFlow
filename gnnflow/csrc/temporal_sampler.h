#ifndef GNNFLOW_TEMPORAL_SAMPLER_H_
#define GNNFLOW_TEMPORAL_SAMPLER_H_

#include <cuda_runtime_api.h>
#include <curand_kernel.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <tuple>

#include "common.h"
#include "dynamic_graph.h"
#include "resource_holder.h"

namespace gnnflow {

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

  // NB: this function should handle input with dynamic length (i.e.,
  // `dst_nodes` and `dst_timestamps` can have dynamic lengths every time). Make
  // sure to re-allocate the buffer if the input size is larger than the current
  // buffer size.
  SamplingResult SampleLayer(const std::vector<NIDType>& dst_nodes,
                             const std::vector<TimestampType>& dst_timestamps,
                             uint32_t layer, uint32_t snapshot);

 private:
  constexpr static std::size_t kPerNodeInputBufferSize =
      sizeof(NIDType) + sizeof(TimestampType);

  constexpr static std::size_t kPerNodeOutputBufferSize =
      sizeof(NIDType) + sizeof(TimestampType) + sizeof(EIDType) +
      sizeof(TimestampType) + sizeof(uint32_t);

  typedef std::tuple<NIDType*, TimestampType*> InputBufferTuple;
  InputBufferTuple GetInputBufferTuple(const Buffer& buffer,
                                       std::size_t num_root_nodes) const;

  typedef std::tuple<NIDType*, EIDType*, TimestampType*, TimestampType*,
                     uint32_t*>
      OutputBufferTuple;
  OutputBufferTuple GetOutputBufferTuple(
      const Buffer& buffer, std::size_t num_root_nodes,
      std::size_t maximum_sampled_nodes) const;

  void InitBufferIfNeeded(std::size_t num_root_nodes,
                          std::size_t maximum_sampled_nodes);

 private:
  const DynamicGraph& graph_;  // sampling does not modify the graph
  std::vector<uint32_t> fanouts_;
  SamplingPolicy sampling_policy_;
  uint32_t num_snapshots_;
  float snapshot_time_window_;
  bool prop_time_;
  uint32_t num_layers_;
  uint64_t seed_;
  std::size_t shared_memory_size_;
  int device_;

  std::unique_ptr<StreamHolder[]> stream_holders_;
  std::unique_ptr<PinMemoryBuffer> cpu_buffer_;
  std::unique_ptr<GPUBuffer> gpu_input_buffer_;
  std::unique_ptr<GPUBuffer> gpu_output_buffer_;
  std::unique_ptr<CuRandStateHolder> rand_states_;

  std::size_t maximum_num_root_nodes_;
  std::size_t maximum_sampled_nodes_;
};

}  // namespace gnnflow

#endif  // GNNFLOW_TEMPORAL_SAMPLER_H_
