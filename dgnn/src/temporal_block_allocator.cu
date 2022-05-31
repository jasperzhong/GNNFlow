#include <thrust/copy.h>
#include <thrust/device_ptr.h>

#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include "logging.h"
#include "temporal_block_allocator.h"

namespace dgnn {
TemporalBlockAllocator::TemporalBlockAllocator(
    std::size_t max_gpu_mem_pool_size, std::size_t alignment)
    : alignment_(alignment) {
  // Create a pool memory resource
  auto mem_res = new rmm::mr::cuda_memory_resource();
  gpu_resources_.push(mem_res);
  auto pool_res =
      new rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>(
          mem_res, max_gpu_mem_pool_size, max_gpu_mem_pool_size);
  gpu_resources_.push(pool_res);
  rmm::mr::set_current_device_resource(pool_res);
}

TemporalBlockAllocator::~TemporalBlockAllocator() {
  while (!gpu_resources_.empty()) {
    delete gpu_resources_.top();
    gpu_resources_.pop();
  }
}

std::size_t TemporalBlockAllocator::AlignUp(std::size_t size) {
  if (size < alignment_) {
    return alignment_;
  }
  // round up to the next power of two
  return 1 << (64 - __builtin_clzl(size - 1));
}

std::shared_ptr<TemporalBlock> TemporalBlockAllocator::AllocateTemporalBlock(
    std::size_t size) {
  auto block = std::make_shared<TemporalBlock>();

  try {
    AllocateInternal(block, size);
  } catch (rmm::bad_alloc &) {
    // failed to allocate memory
    DeallocateInternal(block);

    //    TODO:  swap old blocks to the host
    //    std::size_t requested_size_to_swap =
    //    GetBlockMemorySize(AlignUp(size));
    //    SwapOldBlocksToHost(requested_size_to_swap);

    // try again
    AllocateInternal(block, size);
  }

  blocks_on_device_[block_sequence_number_] = block;
  block_to_sequence_number_[block] = block_sequence_number_;
  block_sequence_number_++;

  return block;
}

void TemporalBlockAllocator::DeallocateInternal(
    std::shared_ptr<TemporalBlock> block) {
  auto mr = rmm::mr::get_current_device_resource();
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
  block->size = 0;
  block->capacity = 0;
  block->next = nullptr;
}

void TemporalBlockAllocator::DeallocateTemporalBlock(
    std::shared_ptr<TemporalBlock> block) {
  CHECK_NOTNULL(block);
  DeallocateInternal(block);
  blocks_on_device_.erase(block_to_sequence_number_[block]);
  block_to_sequence_number_.erase(block);
}

std::shared_ptr<TemporalBlock> TemporalBlockAllocator::ReallocateTemporalBlock(
    std::shared_ptr<TemporalBlock> block, std::size_t size) {
  CHECK_NOTNULL(block);
  auto new_block = AllocateTemporalBlock(size);
  CHECK_GE(new_block->capacity, block->capacity);

  CopyTemporalBlock(new_block, block);

  DeallocateTemporalBlock(block);

  return new_block;
}

void TemporalBlockAllocator::AllocateInternal(
    std::shared_ptr<TemporalBlock> block, std::size_t size) noexcept(false) {
  std::size_t capacity = AlignUp(size);

  block->size = 0;  // empty block
  block->capacity = capacity;
  block->next = nullptr;

  auto mr = rmm::mr::get_current_device_resource();

  block->dst_nodes =
      static_cast<NIDType *>(mr->allocate(capacity * sizeof(NIDType)));
  block->timestamps = static_cast<TimestampType *>(
      mr->allocate(capacity * sizeof(TimestampType)));
  block->eids =
      static_cast<EIDType *>(mr->allocate(capacity * sizeof(EIDType)));
}

void TemporalBlockAllocator::CopyTemporalBlock(
    std::shared_ptr<TemporalBlock> dst_block,
    std::shared_ptr<TemporalBlock> src_block) {
  thrust::copy(src_block->dst_nodes, src_block->dst_nodes + src_block->size,
               dst_block->dst_nodes);
  thrust::copy(src_block->eids, src_block->eids + src_block->size,
               dst_block->eids);
  thrust::copy(src_block->timestamps, src_block->timestamps + src_block->size,
               dst_block->timestamps);

  dst_block->size = src_block->size;
  dst_block->next = src_block->next;
}

void TemporalBlockAllocator::Print() const {
  LOG(DEBUG) << "TemporalBlockAllocator:";
  LOG(DEBUG) << "  blocks_on_device_: " << blocks_on_device_.size();
  LOG(DEBUG) << "  blocks_on_host_: " << blocks_on_host_.size();
  LOG(DEBUG) << "  block_to_sequence_number_: "
             << block_to_sequence_number_.size();
  LOG(DEBUG) << "  block_sequence_number_: " << block_sequence_number_;

  LOG(DEBUG) << "  blocks_on_device_: ";
  for (auto it = blocks_on_device_.begin(); it != blocks_on_device_.end();
       ++it) {
    LOG(DEBUG) << "  block: " << it->first << " " << it->second->size << " "
               << it->second->capacity;
  }

  LOG(DEBUG) << "  blocks_on_host_: ";
  for (auto it = blocks_on_host_.begin(); it != blocks_on_host_.end(); ++it) {
    LOG(DEBUG) << "  block: " << it->first << " " << it->second->size << " "
               << it->second->capacity;
  }

  LOG(DEBUG) << "  block_to_sequence_number_: ";
  for (auto it = block_to_sequence_number_.begin();
       it != block_to_sequence_number_.end(); ++it) {
    LOG(DEBUG) << "  block: " << it->first << " " << it->second;
  }
}

}  // namespace dgnn
