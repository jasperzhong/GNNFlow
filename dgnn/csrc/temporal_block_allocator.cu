#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include "logging.h"
#include "temporal_block_allocator.h"
#include "utils.h"

namespace dgnn {

TemporalBlockAllocator::TemporalBlockAllocator(
    std::size_t max_gpu_mem_pool_size, std::size_t min_block_size)
    : min_block_size_(min_block_size), block_id_counter_(0) {
  // create a memory pool
  auto mem_res = new rmm::mr::cuda_memory_resource();
  gpu_resources_.push(mem_res);
  auto pool_res =
      new rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>(
          mem_res, max_gpu_mem_pool_size, max_gpu_mem_pool_size);
  gpu_resources_.push(pool_res);
  rmm::mr::set_current_device_resource(pool_res);
}

TemporalBlockAllocator::~TemporalBlockAllocator() {
  for (auto &block : blocks_on_device_) {
    DeallocateInternal(block.second, NULL);
    delete block.second;
  }

  // release the memory pool
  while (!gpu_resources_.empty()) {
    delete gpu_resources_.top();
    gpu_resources_.pop();
  }

  for (auto &block : blocks_on_host_) {
    if (block.second->dst_nodes != nullptr) {
      delete[] block.second->dst_nodes;
    }

    if (block.second->timestamps != nullptr) {
      delete[] block.second->timestamps;
    }

    if (block.second->eids != nullptr) {
      delete[] block.second->eids;
    }

    delete block.second;
  }

  blocks_on_device_.clear();
  blocks_on_host_.clear();
  block_to_id_.clear();
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
    blocks_on_device_[block_id_counter_] = block;
    block_to_id_[block] = block_id_counter_;
    block_id_counter_++;
  }
  return block;
}

void TemporalBlockAllocator::Deallocate(TemporalBlock *block,
                                        cudaStream_t stream = NULL) {
  CHECK_NOTNULL(block);
  DeallocateInternal(block, stream);

  {
    std::lock_guard<std::mutex> lock(mutex_);
    blocks_on_device_.erase(block_to_id_[block]);
    block_to_id_.erase(block);
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
                                                cudaStream_t stream = NULL) {
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

TemporalBlock *TemporalBlockAllocator::SwapBlockToHost(
    TemporalBlock *block, cudaStream_t stream = NULL) {
  if (block_to_id_.find(block) == block_to_id_.end()) {
    LOG(WARNING) << "Block not found in block_to_id_";
    return nullptr;
  }

  auto block_id = block_to_id_[block];
  if (block->size == 0) {
    LOG(WARNING) << "Block " << block_id << " is empty";
    return block;
  }

  if (blocks_on_device_.find(block_id) == blocks_on_device_.end() &&
      blocks_on_host_.find(block_id) != blocks_on_host_.end()) {
    LOG(WARNING) << "Block " << block_id << " is already on host";
    return block;
  }

  // allocate CPU memory for the block
  auto block_on_host = new TemporalBlock();
  block_on_host->capacity = block->capacity;
  block_on_host->dst_nodes = new NIDType[block_on_host->capacity];
  block_on_host->timestamps = new TimestampType[block_on_host->capacity];
  block_on_host->eids = new EIDType[block_on_host->capacity];

  CopyTemporalBlock(block, block_on_host);

  // release GPU memory
  Deallocate(block, stream);

  {
    std::lock_guard<std::mutex> lock(mutex_);
    blocks_on_host_[block_id] = block_on_host;
    block_to_id_[block_on_host] = block_id;
  }

  return block_on_host;
}

std::size_t TemporalBlockAllocator::num_blocks_on_device() const {
  return blocks_on_device_.size();
}

std::size_t TemporalBlockAllocator::num_blocks_on_host() const {
  return blocks_on_host_.size();
}

std::size_t TemporalBlockAllocator::used_space_on_device() const {
  std::size_t used_space = 0;
  for (auto &block : blocks_on_device_) {
    used_space += block.second->capacity * kBlockSpaceSize;
  }
  return used_space;
}

std::size_t TemporalBlockAllocator::used_space_on_host() const {
  std::size_t used_space = 0;
  for (auto &block : blocks_on_host_) {
    used_space += block.second->capacity * kBlockSpaceSize;
  }
  return used_space;
}

}  // namespace dgnn
