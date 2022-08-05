#ifndef DGNN_TEMPORAL_BLOCK_ALLOCATOR_H_
#define DGNN_TEMPORAL_BLOCK_ALLOCATOR_H_

#include <mutex>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <vector>
#include <stack>

#include "common.h"

namespace dgnn {
/**
 * @brief This class implements a thread-safe memory resource that allocates
 * temporal blocks.
 *
 * The allocator allocates temporal blocks using UVA.
 */
class TemporalBlockAllocator {
 public:
  /**
   * @brief Constructor.
   *
   * It creates a memory pool.
   *
   * @param max_mem_pool_size The maximum size of the memory pool.
   * @param min_block_size The minimum size of the temporal block.
   */
  TemporalBlockAllocator(std::size_t max_gpu_mem_pool_size,
                         std::size_t min_block_size);

  /**
   * @brief Destructor.
   *
   * It frees all the temporal blocks.
   */
  ~TemporalBlockAllocator();

  /**
   * @brief Allocates a temporal block.
   *
   * It may fail if the memory pool is full.
   *
   * @param size The size of the temporal block.
   *
   * @return A pointer to the temporal block.
   *
   * @throw rmm::bad_alloc If the allocation fails.
   */
  TemporalBlock* Allocate(std::size_t size,
                          cudaStream_t stream = nullptr) noexcept(false);

  /**
   * @brief Deallocates a temporal block.
   *
   * @param block The temporal block to deallocate.
   */
  void Deallocate(TemporalBlock* block, cudaStream_t stream = nullptr);

    /**
   * @brief Align up a size to the min_block_size.
   *
   * If the size is less than the min_block_size, it returns the min_block_size.
   * If not, it rounds up the size to the next power of two.
   *
   * @param size The size to align up.
   *
   * @return The aligned size.
   */
  std::size_t AlignUp(std::size_t size);

 private:
  void AllocateInternal(TemporalBlock* block, std::size_t size,
                        cudaStream_t stream = nullptr) noexcept(false);

  void DeallocateInternal(TemporalBlock* block, cudaStream_t stream = nullptr);

  std::vector<TemporalBlock*> blocks_;

  std::size_t min_block_size_;
  std::stack<rmm::mr::device_memory_resource*> mem_resources_;

  std::mutex mutex_;
};

}  // namespace dgnn

#endif  // DGNN_TEMPORAL_BLOCK_ALLOCATOR_H_
