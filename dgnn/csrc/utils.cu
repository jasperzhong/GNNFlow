#include <thrust/copy.h>
#include <thrust/device_ptr.h>

#include "logging.h"
#include "utils.h"

namespace dgnn {
void CopyTemporalBlock(TemporalBlock* src, TemporalBlock* dst) {
  CHECK_NOTNULL(src);
  CHECK_NOTNULL(dst);
  CHECK_GE(dst->capacity, src->capacity);

  // assume that the src block is on the GPU
  thrust::copy(thrust::device_ptr<NIDType>(src->dst_nodes),
               thrust::device_ptr<NIDType>(src->dst_nodes) + src->size,
               dst->dst_nodes);
  thrust::copy(thrust::device_ptr<TimestampType>(src->timestamps),
               thrust::device_ptr<TimestampType>(src->timestamps) + src->size,
               dst->timestamps);
  thrust::copy(thrust::device_ptr<EIDType>(src->eids),
               thrust::device_ptr<EIDType>(src->eids + src->size), dst->eids);

  dst->size = src->size;
  dst->prev = src->prev;
  dst->next = src->next;
}

void CopyEdgesToBlock(TemporalBlock* block,
                      const std::vector<NIDType>& dst_nodes,
                      const std::vector<TimestampType>& timestamps,
                      const std::vector<EIDType>& eids, std::size_t start_idx,
                      std::size_t num_edges) {
  CHECK_NOTNULL(block);
  CHECK_EQ(dst_nodes.size(), timestamps.size());
  CHECK_EQ(eids.size(), timestamps.size());
  CHECK_LE(block->size + num_edges, block->capacity);
  // assume that the block is on the GPU

  thrust::copy(dst_nodes.begin() + start_idx,
               dst_nodes.begin() + start_idx + num_edges,
               thrust::device_ptr<NIDType>(block->dst_nodes) + block->size);

  thrust::copy(
      timestamps.begin() + start_idx,
      timestamps.begin() + start_idx + num_edges,
      thrust::device_ptr<TimestampType>(block->timestamps) + block->size);

  thrust::copy(eids.begin() + start_idx, eids.begin() + start_idx + num_edges,
               thrust::device_ptr<EIDType>(block->eids) + block->size);

  block->size += num_edges;
}

std::size_t GetSharedMemoryMaxSize() {
  std::size_t max_size = 0;
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  max_size = prop.sharedMemPerBlock;
  return max_size;
}
}  // namespace dgnn
