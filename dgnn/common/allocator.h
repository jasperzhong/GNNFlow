#ifndef CACHING_ALLOCATOR_H_
#define CACHING_ALLOCATOR_H_

#include <cstddef>
#include <map>
#include "common.h"

namespace dgnn {

class CachingAllocator {
 public:
  CachingAllocator(std::size_t gpu_mem_reserve, std::size_t block_size);
  ~CachingAllocator();

  // whether to align to the power of two of block size
  TemporalBlock Allocate(std::size_t size, bool align = false);

  void Deallocate(TemporalBlock);

  TemporalBlock Reallocate(TemporalBlock, std::size_t size, bool align = false);

  std::size_t memory_usage() const { return gpu_mem_used_; }

 private:
  std::size_t gpu_mem_reserve_;
  std::size_t block_size_;
  std::size_t gpu_mem_used_;

  typedef std::multimap<std::size_t, TemporalBlock*> FreeBlockType;
  typedef std::map<TemporalBlock*, std::size_t> AllocatedBlockType;

  FreeBlockType free_blocks_;
  AllocatedBlockType allocated_blocks_;

  // Sequence number is the logical time that is assigned to each
  // temporal block. It increases by 1 every time a temporal block is
  // allocated. The smaller the value, the older the block.
  unsigned int sequence_number_;
};

};  // namespace dgnn

#endif  // CACHING_ALLOCATOR_H_
