#include <thrust/copy.h>
#include <thrust/device_ptr.h>

#include "common.h"
#include "logging.h"
#include "utils.h"

namespace gnnflow {
void CopyTemporalBlock(TemporalBlock* src, TemporalBlock* dst, int device,
                       cudaStream_t stream) {
  CHECK_NOTNULL(src);
  CHECK_NOTNULL(dst);
  CHECK_GE(dst->capacity, src->capacity);

  if (device == 0) {
    CUDA_CALL(cudaMemcpyAsync(dst->dst_nodes, src->dst_nodes,
                              src->size * sizeof(NIDType), cudaMemcpyDefault,
                              stream));

    CUDA_CALL(cudaMemcpyAsync(dst->timestamps, src->timestamps,
                              src->size * sizeof(TimestampType),
                              cudaMemcpyDefault, stream));

    CUDA_CALL(cudaMemcpyAsync(dst->eids, src->eids, src->size * sizeof(EIDType),
                              cudaMemcpyDefault, stream));
  }
  dst->size = src->size;
  dst->start_timestamp = src->start_timestamp;
  dst->end_timestamp = src->end_timestamp;
  dst->next = src->next;
}

void CopyEdgesToBlock(TemporalBlock* block,
                      const std::vector<NIDType>& dst_nodes,
                      const std::vector<TimestampType>& timestamps,
                      const std::vector<EIDType>& eids, std::size_t start_idx,
                      std::size_t num_edges, int device, cudaStream_t stream) {
  CHECK_NOTNULL(block);
  CHECK_EQ(dst_nodes.size(), timestamps.size());
  CHECK_EQ(eids.size(), timestamps.size());
  CHECK_LE(block->size + num_edges, block->capacity);
  // NB: we assume that the incoming edges are newer than the existing ones.
  CHECK_LE(block->end_timestamp, timestamps[start_idx + num_edges - 1]);

  if (device == 0) {
    CUDA_CALL(cudaMemcpyAsync(
        block->dst_nodes + block->size, &dst_nodes[start_idx],
        sizeof(NIDType) * num_edges, cudaMemcpyDefault, stream));

    CUDA_CALL(cudaMemcpyAsync(
        block->timestamps + block->size, &timestamps[start_idx],
        sizeof(TimestampType) * num_edges, cudaMemcpyDefault, stream));

    CUDA_CALL(cudaMemcpyAsync(block->eids + block->size, &eids[start_idx],
                              sizeof(EIDType) * num_edges, cudaMemcpyDefault,
                              stream));
  }
  block->size += num_edges;

  block->start_timestamp =
      std::min(block->start_timestamp, timestamps[start_idx]);
  block->end_timestamp = timestamps[start_idx + num_edges - 1];
}  // namespace gnnflow

std::size_t GetSharedMemoryMaxSize() {
  std::size_t max_size = 0;
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  max_size = prop.sharedMemPerBlock;
  return max_size;
}

__global__ void InitCuRandStates(curandState_t* states, uint64_t seed) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  curand_init(seed, tid, 0, &states[tid]);
}

__host__ __device__ void LowerBound(TimestampType* timestamps, int num_edges,
                                    TimestampType timestamp, int* idx) {
  int left = 0;
  int right = num_edges;
  while (left < right) {
    int mid = (left + right) / 2;
    if (timestamps[mid] < timestamp) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }
  *idx = left;
}

template <typename T>
__host__ __device__ void inline swap(T& a, T& b) {
  T c(a);
  a = b;
  b = c;
}

__host__ __device__ void QuickSort(uint32_t* indices, int lo, int hi) {
  if (lo >= hi || lo < 0 || hi < 0) return;

  uint32_t pivot = indices[hi];
  int i = lo - 1;
  for (int j = lo; j < hi; ++j) {
    if (indices[j] < pivot) {
      swap(indices[++i], indices[j]);
    }
  }
  swap(indices[++i], indices[hi]);

  QuickSort(indices, lo, i - 1);
  QuickSort(indices, i + 1, hi);
}

}  // namespace gnnflow
