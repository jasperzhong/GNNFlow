#include "temporal_block_allocator.h"

#include <thrust/device_delete.h>
#include <thrust/device_new.h>

#include <rmm/mr/device/per_device_resource.hpp>

namespace dgnn {

TemporalBlockAllocator::TemporalBlockAllocator(std::size_t alignment)
    : alignment_(alignment),
      memory_manager_ptr_(rmm::mr::get_current_device_resource()),
      sequence_number_(0) {}

thrust::device_ptr<TemporalBlock> TemporalBlockAllocator::Allocate(
    std::size_t size) {
  std::size_t capacity = AlignUp(size);
  TemporalBlock block_on_host(capacity);

  try {
    // allocate device memory for the block
    AllocateDeviceMemoryForBlock(block_on_host);
  } catch (const rmm::bad_alloc &) {
    // allocation on device fails
    // release allocated memory first
    DeallocateDeviceMemoryForBlock(block_on_host);

    // swap old blocks to the host
    auto swap_size = GetTemporalBlockSpaceSize(block_on_host);
    SwapOldBlocksToHost(swap_size);

    // try to allocate again
    AllocateDeviceMemoryForBlock(block_on_host);
  }

  // move the block to the device
  auto block_on_device = thrust::device_new<TemporalBlock>();
  *block_on_device = block_on_host;

  // add the block to the map
  on_device_blocks_[block_on_device.get()] = sequence_number_;
  sequence_number_++;

  return block_on_device;
}

void TemporalBlockAllocator::Deallocate(TemporalBlock *block) {
  if (on_device_blocks_.find(block) != on_device_blocks_.end()) {
    // deallocate device memory
    DeallocateDeviceMemoryForBlock(*block);

    // remove the block from the map
    on_device_blocks_.erase(block);

    // delete the block
    thrust::device_delete(thrust::device_ptr<TemporalBlock>(block));
  } else if (on_host_blocks_.find(block) != on_host_blocks_.end()) {
    // deallocate host memory
    DeallocateHostMemoryForBlock(*block);

    // remove the block from the map
    on_host_blocks_.erase(block);

    // delete the block
    delete block;
  } else {
    throw std::runtime_error(
        "TemporalBlockAllocator::Deallocate: block not "
        "found");
  }
}

thrust::device_ptr<TemporalBlock> TemporalBlockAllocator::Reallocate(
    thrust::device_ptr<TemporalBlock> block, std::size_t size) {
  // allocate a new block
  auto new_block = Allocate(size);

  // TODO: copy data from the old block to the new block

  // deallocate the old block
  Deallocate(block.get());

  return new_block;
}

std::size_t TemporalBlockAllocator::AlignUp(std::size_t size) {
  if (size < alignment_) {
    return alignment_;
  }
  // round up to the next power of two
  return 1 << (64 - __builtin_clzl(size - 1));
}

void CopyBlock(thrust::device_ptr<TemporalBlock> &dst,
               thrust::device_ptr<TemporalBlock> &src);
}  // namespace dgnn
