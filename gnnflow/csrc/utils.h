#ifndef GNNFLOW_UTILS_H_
#define GNNFLOW_UTILS_H_

#include <curand_kernel.h>

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <vector>

#include "common.h"

namespace gnnflow {

template <typename T>
std::vector<std::size_t> stable_sort_indices(const std::vector<T>& v) {
  // initialize original index locations
  std::vector<std::size_t> idx(v.size());
  std::iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  std::stable_sort(
      idx.begin(), idx.end(),
      [&v](std::size_t i1, std::size_t i2) { return v[i1] < v[i2]; });

  return idx;
}

template <typename T>
std::vector<T> sort_vector(const std::vector<T>& v,
                           const std::vector<std::size_t>& idx) {
  std::vector<T> sorted_v;
  sorted_v.reserve(v.size());
  for (auto i : idx) {
    sorted_v.emplace_back(v[i]);
  }
  return sorted_v;
}

/**
 * @brief Copy a temporal block on the GPU to another block.
 *
 * The destination block should have a size greater than or equal to the
 * source block. It assumes that the source block is on the GPU. But the
 * destination block can be on the CPU or on the GPU.
 *
 * @param dst The destination temporal block.
 * @param src The source temporal block.
 * @param device The device id.
 * @param stream The CUDA stream.
 */
void CopyTemporalBlock(TemporalBlock* src, TemporalBlock* dst, int device,
                       cudaStream_t stream = nullptr);

/**
 * @brief Copy edges on the CPU to the block on the GPU.
 *
 * The destination block should have a size greater than or equal to the
 * incoming edges.
 *
 * @param block The destination temporal block.
 * @param dst_nodes The destination nodes.
 * @param timestamps The timestamps of the incoming edges.
 * @param eids The ids of the incoming edges.
 * @param start_idx The start index of the incoming edges.
 * @param num_edges The number of incoming edges.
 * @param device The device id.
 * @param stream The CUDA stream.
 */
void CopyEdgesToBlock(TemporalBlock* block,
                      const std::vector<NIDType>& dst_nodes,
                      const std::vector<TimestampType>& timestamps,
                      const std::vector<EIDType>& eids, std::size_t start_idx,
                      std::size_t num_edges, int device,
                      cudaStream_t stream = nullptr);

std::size_t GetSharedMemoryMaxSize();

void Copy(void* dst, const void* src, std::size_t size);

__global__ void InitCuRandStates(curandState_t* state, std::size_t num_elements,
                                 uint64_t seed);

__host__ __device__ void LowerBound(TimestampType* timestamps, int num_edges,
                                    TimestampType timestamp, int* idx);

__host__ __device__ void QuickSort(uint32_t* indices, int lo, int hi);
}  // namespace gnnflow

#endif  // GNNFLOW_UTILS_H_
