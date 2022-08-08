#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pinned_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include "logging.h"
#include "temporal_block_allocator.h"
#include "utils.h"

namespace dgnn {

TemporalBlockAllocator::TemporalBlockAllocator(
    std::size_t initial_pool_size, std::size_t maximum_pool_size,
    std::size_t minium_block_size, MemoryResourceType mem_resource_type)
    : minium_block_size_(minium_block_size) {
  // create a memory pool
  switch (mem_resource_type) {
    case MemoryResourceType::kMemoryResourceTypeCUDA: {
      auto mem_res = new rmm::mr::cuda_memory_resource();
      mem_resources_.push(mem_res);
      auto pool_res =
          new rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>(
              mem_res, initial_pool_size, maximum_pool_size);
      mem_resources_.push(pool_res);
      break;
    }
    case MemoryResourceType::kMemoryResourceTypeUnified: {
      auto mem_res = new rmm::mr::managed_memory_resource();
      mem_resources_.push(mem_res);
      auto pool_res =
          new rmm::mr::pool_memory_resource<rmm::mr::managed_memory_resource>(
              mem_res, initial_pool_size, maximum_pool_size);
      mem_resources_.push(pool_res);
      break;
    }
    case MemoryResourceType::kMemoryResourceTypePinned: {
      auto mem_res = new rmm::mr::pinned_memory_resource();
      mem_resources_.push(mem_res);
      auto pool_res =
          new rmm::mr::pool_memory_resource<rmm::mr::pinned_memory_resource>(
              mem_res, initial_pool_size, maximum_pool_size);
      mem_resources_.push(pool_res);
      break;
    }
  }
}

TemporalBlockAllocator::~TemporalBlockAllocator() {
  for (auto &block : blocks_) {
    DeallocateInternal(block);
    delete block;
  }

  // release the memory pool
  while (!mem_resources_.empty()) {
    delete mem_resources_.top();
    mem_resources_.pop();
  }

  blocks_.clear();
}

std::size_t TemporalBlockAllocator::AlignUp(std::size_t size) {
  if (size < minium_block_size_) {
    return minium_block_size_;
  }
  // round up to the next power of two
  return 1 << (64 - __builtin_clzl(size - 1));
}

TemporalBlock *TemporalBlockAllocator::Allocate(std::size_t size) {
  auto block = new TemporalBlock();

  try {
    AllocateInternal(block, size);
  } catch (rmm::bad_alloc &) {
    // failed to allocate memory
    DeallocateInternal(block);

    LOG(FATAL) << "Failed to allocate memory for temporal block of size "
               << size;
  }

  {
    std::lock_guard<std::mutex> lock(mutex_);
    blocks_.push_back(block);
  }
  return block;
}

void TemporalBlockAllocator::Deallocate(TemporalBlock *block) {
  CHECK_NOTNULL(block);
  DeallocateInternal(block);

  {
    std::lock_guard<std::mutex> lock(mutex_);
    blocks_.erase(std::remove(blocks_.begin(), blocks_.end(), block),
                  blocks_.end());
  }
}

void TemporalBlockAllocator::Reallocate(TemporalBlock *block, std::size_t size,
                                        cudaStream_t stream) {
  CHECK_NOTNULL(block);

  TemporalBlock tmp;
  AllocateInternal(&tmp, size);
  CopyTemporalBlock(block, &tmp, stream);
  DeallocateInternal(block);

  *block = tmp;
}

void TemporalBlockAllocator::AllocateInternal(
    TemporalBlock *block, std::size_t size) noexcept(false) {
  std::size_t capacity = AlignUp(size);

  block->size = 0;  // empty block
  block->capacity = capacity;
  block->start_timestamp = 0;
  block->end_timestamp = 0;
  block->next = nullptr;

  // allocate memory for the block
  // NB: rmm is thread-safe
  auto mr = mem_resources_.top();
  block->dst_nodes =
      static_cast<NIDType *>(mr->allocate(capacity * sizeof(NIDType)));
  block->timestamps = static_cast<TimestampType *>(
      mr->allocate(capacity * sizeof(TimestampType)));
  block->eids =
      static_cast<EIDType *>(mr->allocate(capacity * sizeof(EIDType)));
}

void TemporalBlockAllocator::DeallocateInternal(TemporalBlock *block) {
  auto mr = mem_resources_.top();
  if (block->dst_nodes != nullptr) {
    mr->deallocate(block->dst_nodes, block->capacity * sizeof(NIDType));
    block->dst_nodes = nullptr;
  }
  if (block->timestamps != nullptr) {
    mr->deallocate(block->timestamps, block->capacity * sizeof(TimestampType));
    block->timestamps = nullptr;
  }
  if (block->eids != nullptr) {
    mr->deallocate(block->eids, block->capacity * sizeof(EIDType));
    block->eids = nullptr;
  }
}
}  // namespace dgnn
