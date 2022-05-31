#ifndef DGNN_TEMPORAL_BLOCK_ALLOCATOR_H_
#define DGNN_TEMPORAL_BLOCK_ALLOCATOR_H_

#include <map>
#include <memory>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <stack>
#include <unordered_map>

#include "temporal_block.h"

namespace dgnn {

class TemporalBlockAllocator {
 public:
  TemporalBlockAllocator(std::size_t max_gpu_mem_pool_size,
                         std::size_t alignment);
  ~TemporalBlockAllocator();

  std::shared_ptr<TemporalBlock> AllocateTemporalBlock(std::size_t size);

  void DeallocateTemporalBlock(std::shared_ptr<TemporalBlock> block);

  std::shared_ptr<TemporalBlock> ReallocateTemporalBlock(
      std::shared_ptr<TemporalBlock> block, std::size_t size);

  void CopyTemporalBlock(std::shared_ptr<TemporalBlock> dst,
                         std::shared_ptr<TemporalBlock> src);

  void Print() const;  // for debug
 private:
  std::size_t AlignUp(std::size_t size);

  void AllocateInternal(std::shared_ptr<TemporalBlock> block,
                        std::size_t size) noexcept(false);

  void DeallocateInternal(std::shared_ptr<TemporalBlock> block);

  std::size_t alignment_;
  std::stack<rmm::mr::device_memory_resource*> gpu_resources_;

  // sequence number (how old the block is) -> block raw pointer
  std::map<std::size_t, std::shared_ptr<TemporalBlock>> blocks_on_device_;
  std::map<std::size_t, std::shared_ptr<TemporalBlock>> blocks_on_host_;

  std::unordered_map<std::shared_ptr<TemporalBlock>, std::size_t>
      block_to_sequence_number_;

  // a monotonically increasing sequence number
  std::size_t block_sequence_number_;
};

}  // namespace dgnn

#endif  // DGNN_TEMPORAL_BLOCK_ALLOCATOR_H_
