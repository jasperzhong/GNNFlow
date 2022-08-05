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
    std::size_t max_gpu_mem_pool_size, std::size_t min_block_size)
    : min_block_size_(min_block_size) {
  // create a memory pool
  auto mem_res = new rmm::mr::managed_memory_resource();
  mem_resources_.push(mem_res);
  auto pool_res =
      new rmm::mr::pool_memory_resource<rmm::mr::managed_memory_resource>(
          mem_res, max_gpu_mem_pool_size, max_gpu_mem_pool_size);
  mem_resources_.push(pool_res);
  rmm::mr::set_current_device_resource(pool_res);
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
  if (size < min_block_size_) {
    return min_block_size_;
  }
  // round up to the next power of two
  return 1 << (64 - __builtin_clzl(size - 1));
}

TemporalBlock *TemporalBlockAllocator::Allocate(
    std::size_t size, cudaStream_t stream) noexcept(false) {
  auto block = new TemporalBlock();

  try {
    AllocateInternal(block, size, stream);
  } catch (rmm::bad_alloc &) {
    // failed to allocate memory
    DeallocateInternal(block, stream);

    LOG(WARNING) << "Failed to allocate memory for temporal block of size "
                 << size;
    throw;
  }

  {
    std::lock_guard<std::mutex> lock(mutex_);
    blocks_.push_back(block);
  }
  return block;
}

void TemporalBlockAllocator::Deallocate(TemporalBlock *block,
                                        cudaStream_t stream) {
  CHECK_NOTNULL(block);
  DeallocateInternal(block, stream);

  {
    std::lock_guard<std::mutex> lock(mutex_);
    blocks_.erase(std::remove(blocks_.begin(), blocks_.end(), block),
                  blocks_.end());
  }
}

void TemporalBlockAllocator::AllocateInternal(
    TemporalBlock *block, std::size_t size,
    cudaStream_t stream) noexcept(false) {
  std::size_t capacity = AlignUp(size);

  block->size = 0;  // empty block
  block->capacity = capacity;
  block->prev = nullptr;
  block->next = nullptr;

  // allocate GPU memory for the block
  // NB: rmm is thread-safe
  auto mr = rmm::mr::get_current_device_resource();
  block->dst_nodes = static_cast<NIDType *>(
      mr->allocate(capacity * sizeof(NIDType), rmm::cuda_stream_view(stream)));
  block->timestamps = static_cast<TimestampType *>(mr->allocate(
      capacity * sizeof(TimestampType), rmm::cuda_stream_view(stream)));
  block->eids = static_cast<EIDType *>(
      mr->allocate(capacity * sizeof(EIDType), rmm::cuda_stream_view(stream)));
}

void TemporalBlockAllocator::DeallocateInternal(TemporalBlock *block,
                                                cudaStream_t stream) {
  auto mr = rmm::mr::get_current_device_resource();
  if (block->dst_nodes != nullptr) {
    mr->deallocate(block->dst_nodes, block->capacity * sizeof(NIDType),
                   rmm::cuda_stream_view(stream));
    block->dst_nodes = nullptr;
  }
  if (block->timestamps != nullptr) {
    mr->deallocate(block->timestamps, block->capacity * sizeof(TimestampType),
                   rmm::cuda_stream_view(stream));
    block->timestamps = nullptr;
  }
  if (block->eids != nullptr) {
    mr->deallocate(block->eids, block->capacity * sizeof(EIDType),
                   rmm::cuda_stream_view(stream));
    block->eids = nullptr;
  }
}
}  // namespace dgnn
