#ifndef GNNFLOW_TEMPORAL_BLOCK_ALLOCATOR_H_
#define GNNFLOW_TEMPORAL_BLOCK_ALLOCATOR_H_

#include <mutex>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <stack>
#include <vector>

#include "common.h"

namespace gnnflow {
/**
 * @brief This class implements a thread-safe memory resource that allocates
 * temporal blocks.
 *
 * The allocated blocks are in the host memory. But the edges are stored in the
 * device memory or managed memory or pinned memory, depending on the memory
 * resource type.
 */
class TemporalBlockAllocator {
 public:
  /**
   * @brief Constructor.
   *
   * It creates a memory pool.
   *
   * @param initial_pool_size The initial size of the memory pool.
   * @param maxmium_pool_size The maximum size of the memory pool
   * @param minimum_block_size The minimum size of the temporal block.
   * @param MemoryResourceType The type of memory resource.
   * @param device The device id.
   */
  TemporalBlockAllocator(std::size_t initial_pool_size,
                         std::size_t maximum_pool_size,
                         std::size_t minimum_block_size,
                         MemoryResourceType mem_resource_type, int device);

  /**
   * @brief Destructor.
   *
   * It frees all the temporal blocks.
   */
  ~TemporalBlockAllocator();

  /**
   * @brief Allocate a temporal block.
   *
   * NB: the block itself is in the host memory.
   *
   * @param size The size of the temporal block.
   *
   * @return A host pointer to the temporal block.
   */
  TemporalBlock* Allocate(std::size_t size);

  /**
   * @brief Deallocate a temporal block.
   *
   * @param block The temporal block to deallocate. It must be in the host
   * memory.
   */
  void Deallocate(TemporalBlock* block);

  /**
   * @brief Reallocate a temporal block.
   *
   * NB: We only change the content of the temporal block. The host pointer to
   * the temporal block is not changed.
   *
   * @param block The temporal block to reallocate. It must be in the host
   * memory.
   * @param size The new size of the temporal block.
   * @param stream The stream to use. If nullptr, the default stream is used.
   */
  void Reallocate(TemporalBlock* block, std::size_t size,
                  cudaStream_t stream = nullptr);

  void SaveToFile(TemporalBlock* block, NIDType src_node);
  void ReadFromFile(TemporalBlock* block, NIDType src_node);

  std::size_t get_total_memory_usage() const { return allocated_; }

 private:
  std::size_t AlignUp(std::size_t size);

  void AllocateInternal(TemporalBlock* block, std::size_t size) noexcept(false);

  void DeallocateInternal(TemporalBlock* block);

  std::vector<TemporalBlock*> blocks_;

  std::size_t minium_block_size_;
  MemoryResourceType mem_resource_type_;
  std::stack<rmm::mr::device_memory_resource*> mem_resources_;

  std::unordered_map<TemporalBlock*, std::string> saved_blocks_;
  std::unordered_map<NIDType, std::size_t> num_saved_blocks_;

  std::mutex mutex_;

  const int device_;

  std::size_t allocated_;
};

}  // namespace gnnflow

#endif  // GNNFLOW_TEMPORAL_BLOCK_ALLOCATOR_H_
