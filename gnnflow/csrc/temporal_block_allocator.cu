#include <fstream>
#include <limits>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pinned_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/mr/device/shared_memory_resource.hpp>

#include "logging.h"
#include "temporal_block_allocator.h"
#include "utils.h"

namespace gnnflow {

TemporalBlockAllocator::TemporalBlockAllocator(
    std::size_t initial_pool_size, std::size_t maximum_pool_size,
    std::size_t minium_block_size, MemoryResourceType mem_resource_type,
    int device)
    : mem_resource_type_(mem_resource_type),
      minium_block_size_(minium_block_size),
      device_(device),
      allocated_(0) {
  LOG(DEBUG) << "set device to " << device;
  CUDA_CALL(cudaSetDevice(device));
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
    case MemoryResourceType::kMemoryResourceTypeShared: {
      // NB: device ID is equal to the local rank
      auto mem_res = new rmm::mr::shared_memory_resource(device);
      mem_resources_.push(mem_res);
      auto pool_res =
          new rmm::mr::pool_memory_resource<rmm::mr::shared_memory_resource>(
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
  return size;
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
    blocks_.erase(std::remove(blocks_.begin(), blocks_.end(), block));
  }

  delete block;
}

void TemporalBlockAllocator::Reallocate(TemporalBlock *block, std::size_t size,
                                        cudaStream_t stream) {
  CHECK_NOTNULL(block);

  TemporalBlock tmp;
  AllocateInternal(&tmp, size);
  CopyTemporalBlock(block, &tmp, device_, stream);
  DeallocateInternal(block);

  *block = tmp;
}

void TemporalBlockAllocator::AllocateInternal(
    TemporalBlock *block, std::size_t size) noexcept(false) {
  std::size_t capacity = AlignUp(size);

  block->size = 0;  // empty block
  block->capacity = capacity;
  block->start_timestamp = std::numeric_limits<TimestampType>::max();
  block->end_timestamp = 0;
  block->prev = nullptr;
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

  allocated_ +=
      capacity * (sizeof(NIDType) + sizeof(TimestampType) + sizeof(EIDType));
}

void TemporalBlockAllocator::DeallocateInternal(TemporalBlock *block) {
  auto mr = mem_resources_.top();
  if (block->dst_nodes != nullptr) {
    mr->deallocate(block->dst_nodes, block->capacity * sizeof(NIDType));
    block->dst_nodes = nullptr;
    allocated_ -= block->capacity * sizeof(NIDType);
  }
  if (block->timestamps != nullptr) {
    mr->deallocate(block->timestamps, block->capacity * sizeof(TimestampType));
    block->timestamps = nullptr;
    allocated_ -= block->capacity * sizeof(TimestampType);
  }
  if (block->eids != nullptr) {
    mr->deallocate(block->eids, block->capacity * sizeof(EIDType));
    block->eids = nullptr;
    allocated_ -= block->capacity * sizeof(EIDType);
  }

  block->size = 0;
  block->capacity = 0;
  // NB: we don't reset the timestamps and prev/next
}

void TemporalBlockAllocator::SaveToFile(TemporalBlock *block,
                                        NIDType src_node) {
  if (mem_resource_type_ != MemoryResourceType::kMemoryResourceTypePinned &&
      mem_resource_type_ != MemoryResourceType::kMemoryResourceTypeShared) {
    LOG(FATAL) << "Only pinned and shared memory resources are supported";
  }

  // NB: only first rank saves the temporal block
  if (device_ == 0) {
    std::string file_name = "temporal_block_" + std::to_string(src_node) + "-" +
                            std::to_string(num_saved_blocks_[src_node]) +
                            ".bin";
    std::ofstream file(file_name, std::ios::out | std::ios::binary);
    file.write(reinterpret_cast<char *>(&block->size), sizeof(block->size));
    file.write(reinterpret_cast<char *>(&block->capacity),
               sizeof(block->capacity));
    file.write(reinterpret_cast<char *>(&block->start_timestamp),
               sizeof(block->start_timestamp));
    file.write(reinterpret_cast<char *>(&block->end_timestamp),
               sizeof(block->end_timestamp));
    file.write(reinterpret_cast<char *>(block->dst_nodes),
               sizeof(NIDType) * block->size);
    file.write(reinterpret_cast<char *>(block->timestamps),
               sizeof(TimestampType) * block->size);
    file.write(reinterpret_cast<char *>(block->eids),
               sizeof(EIDType) * block->size);
    file.write(reinterpret_cast<char *>(&block->prev), sizeof(block->prev));
    file.write(reinterpret_cast<char *>(&block->next), sizeof(block->next));
    file.close();

    saved_blocks_[block] = file_name;
    num_saved_blocks_[src_node]++;

    LOG(INFO) << "Temporal block saved to " << file_name;
  }

  // NB: all ranks need to deallocate the temporal block (but only first rank
  // release the memory)
  DeallocateInternal(block);
}

void TemporalBlockAllocator::ReadFromFile(TemporalBlock *block,
                                          NIDType src_node) {
  if (mem_resource_type_ != MemoryResourceType::kMemoryResourceTypePinned) {
    LOG(FATAL) << "Only pinned memory resources are supported";
    // NB: shared memory resources are not supported because we need to know the
    // offset of the temporal block in the shared memory
  }
  CHECK_EQ(device_, 0) << "Only first rank can read temporal blocks from file";

  std::string file_name = saved_blocks_[block];
  std::ifstream file(file_name, std::ios::in | std::ios::binary);
  file.read(reinterpret_cast<char *>(&block->size), sizeof(block->size));
  file.read(reinterpret_cast<char *>(&block->capacity),
            sizeof(block->capacity));
  AllocateInternal(block, block->capacity);
  file.read(reinterpret_cast<char *>(&block->start_timestamp),
            sizeof(block->start_timestamp));
  file.read(reinterpret_cast<char *>(&block->end_timestamp),
            sizeof(block->end_timestamp));
  file.read(reinterpret_cast<char *>(block->dst_nodes),
            sizeof(NIDType) * block->size);
  file.read(reinterpret_cast<char *>(block->timestamps),
            sizeof(TimestampType) * block->size);
  file.read(reinterpret_cast<char *>(block->eids),
            sizeof(EIDType) * block->size);
  file.read(reinterpret_cast<char *>(&block->prev), sizeof(block->prev));
  file.read(reinterpret_cast<char *>(&block->next), sizeof(block->next));
  file.close();

  saved_blocks_.erase(block);
  num_saved_blocks_[src_node]--;

  LOG(INFO) << "Temporal block read from " << file_name;
}
}  // namespace gnnflow
