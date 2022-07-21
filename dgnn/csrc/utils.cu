#include <thrust/copy.h>
#include <thrust/device_ptr.h>

#include "common.h"
#include "logging.h"
#include "utils.h"

namespace dgnn {
void CopyTemporalBlock(TemporalBlock* src, TemporalBlock* dst,
                       cudaStream_t stream) {
  CHECK_NOTNULL(src);
  CHECK_NOTNULL(dst);
  CHECK_GE(dst->capacity, src->capacity);

  // assume that the src block is on the GPU
  CUDA_CALL(cudaMemcpyAsync(dst->dst_nodes, src->dst_nodes,
                            src->size * sizeof(NIDType), cudaMemcpyDeviceToHost,
                            stream));

  CUDA_CALL(cudaMemcpyAsync(dst->timestamps, src->timestamps,
                            src->size * sizeof(TimestampType),
                            cudaMemcpyDeviceToHost, stream));

  CUDA_CALL(cudaMemcpyAsync(dst->eids, src->eids, src->size * sizeof(EIDType),
                            cudaMemcpyDeviceToHost, stream));

  dst->size = src->size;
  dst->prev = src->prev;
  dst->next = src->next;
}

void CopyEdgesToBlock(TemporalBlock* block,
                      const std::vector<NIDType>& dst_nodes,
                      const std::vector<TimestampType>& timestamps,
                      const std::vector<EIDType>& eids, std::size_t start_idx,
                      std::size_t num_edges, cudaStream_t stream) {
  CHECK_NOTNULL(block);
  CHECK_EQ(dst_nodes.size(), timestamps.size());
  CHECK_EQ(eids.size(), timestamps.size());
  CHECK_LE(block->size + num_edges, block->capacity);
  // assume that the block is on the GPU

  CUDA_CALL(cudaMemcpyAsync(block->dst_nodes + block->size,
                            &dst_nodes[start_idx], sizeof(NIDType) * num_edges,
                            cudaMemcpyHostToDevice, stream));

  CUDA_CALL(cudaMemcpyAsync(
      block->timestamps + block->size, &timestamps[start_idx],
      sizeof(TimestampType) * num_edges, cudaMemcpyHostToDevice, stream));

  CUDA_CALL(cudaMemcpyAsync(block->eids + block->size, &eids[start_idx],
                            sizeof(EIDType) * num_edges, cudaMemcpyHostToDevice,
                            stream));

  block->size += num_edges;
}

std::size_t GetSharedMemoryMaxSize() {
  std::size_t max_size = 0;
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  max_size = prop.sharedMemPerBlock;
  return max_size;
}

void Copy(void* dst, const void* src, std::size_t size) {
  auto in = (float*)src;
  auto out = (float*)dst;
#pragma omp parallel for simd num_threads(4)
  for (size_t i = 0; i < size / 4; ++i) {
    out[i] = in[i];
  }

  if (size % 4) {
    std::memcpy(out + size / 4, in + size / 4, size % 4);
  }
}

__global__ void InitCuRandStates(curandState_t* states, uint64_t seed) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  curand_init(seed, tid, 0, &states[tid]);
}

}  // namespace dgnn
