#ifndef DGNN_TEMPORAL_BLOCK_ALLOCATOR_H_
#define DGNN_TEMPORAL_BLOCK_ALLOCATOR_H_

#include <map>
#include <memory>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <stack>
#include <unordered_map>

#include "common.h"

namespace dgnn {
/**
 * @brief This class implements a memory resource that allocates temporal
 * blocks for a GPU.
 *
 * The allocator allocates temporal blocks on the GPU. Each block is attached
 * to a specified identifier. The allocator keeps track of the blocks on the
 * GPU and the blocks on the CPU.
 */
class TemporalBlockAllocator {
 public:
  /**
   * @brief Constructor.
   *
   * It creates a memory pool on the GPU.
   *
   * @param max_gpu_mem_pool_size The maximum size of the GPU memory pool.
   * @param alignment The alignment of the temporal blocks.
   */
  TemporalBlockAllocator(std::size_t max_gpu_mem_pool_size,
                         std::size_t alignment);

  /**
   * @brief Destructor.
   *
   * It frees all the temporal blocks.
   */
  ~TemporalBlockAllocator();

  /**
   * @brief Allocates a temporal block on the GPU.
   *
   * It may fail if the GPU memory pool is full.
   *
   * @param size The size of the temporal block.
   *
   * @return A pointer to the temporal block.
   *
   * @throw rmm::bad_alloc If the allocation fails.
   */
  std::shared_ptr<TemporalBlock> Allocate(std::size_t size) noexcept(false);

  /**
   * @brief Deallocates a temporal block on the GPU.
   *
   * @param block The temporal block to deallocate.
   */
  void Deallocate(std::shared_ptr<TemporalBlock> block);

  /**
   * @brief Reallocate a temporal block on the GPU.
   *
   * @param block The temporal block to reallocate.
   * @param size The new size of the temporal block.
   *
   * @return A pointer to the temporal block.
   */
  std::shared_ptr<TemporalBlock> Reallocate(
      std::shared_ptr<TemporalBlock> block, std::size_t size);

  /**
   * @brief Copy a temporal block on the GPU to another block.
   *
   * The destination block should have a size greater than or equal to the
   * source block. It assumes that the source block is on the GPU. But the
   * destination block can be on the CPU or on the GPU.
   *
   * @param dst The destination temporal block.
   * @param src The source temporal block.
   */
  void Copy(std::shared_ptr<TemporalBlock> src,
            std::shared_ptr<TemporalBlock> dst);

  /**
   * @brief Copy a temporal block from GPU to CPU.
   *
   * @param block The temporal block to copy.
   *
   * @return A pointer to the temporal block on the CPU.
   */
  std::shared_ptr<TemporalBlock> SwapBlockToHost(
      std::shared_ptr<TemporalBlock> block);

  /**
   * @brief Align up a size to the alignment.
   *
   * If the size is less than the alignment, it returns the alignment. If not,
   * it rounds up the size to the next power of two.
   *
   * @param size The size to align up.
   *
   * @return The aligned size.
   */
  std::size_t AlignUp(std::size_t size);

  std::size_t num_blocks_on_device() const;
  std::size_t num_blocks_on_host() const;

  std::size_t used_space_on_device() const;
  std::size_t used_space_on_host() const;

 private:
  void AllocateInternal(std::shared_ptr<TemporalBlock> block,
                        std::size_t size) noexcept(false);

  void DeallocateInternal(std::shared_ptr<TemporalBlock> block);

  std::size_t alignment_;
  std::stack<rmm::mr::device_memory_resource*> gpu_resources_;

  // id -> block raw pointer
  std::map<uint64_t, std::shared_ptr<TemporalBlock>> blocks_on_device_;
  std::map<uint64_t, std::shared_ptr<TemporalBlock>> blocks_on_host_;

  std::unordered_map<std::shared_ptr<TemporalBlock>, uint64_t> block_to_id_;

  // a monotonically increasing sequence number
  uint64_t block_id_counter_;
};

}  // namespace dgnn

#endif  // DGNN_TEMPORAL_BLOCK_ALLOCATOR_H_
