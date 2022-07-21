#ifndef DGNN_TEMPORAL_BLOCK_ALLOCATOR_H_
#define DGNN_TEMPORAL_BLOCK_ALLOCATOR_H_

#include <map>
#include <mutex>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <stack>
#include <unordered_map>

#include "common.h"

namespace dgnn {
/**
 * @brief This class implements a thread-safe memory resource that allocates
 * temporal blocks for a GPU.
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
  TemporalBlock* Allocate(std::size_t size,
                          cudaStream_t stream = nullptr) noexcept(false);

  /**
   * @brief Deallocates a temporal block on the GPU.
   *
   * @param block The temporal block to deallocate.
   */
  void Deallocate(TemporalBlock* block, cudaStream_t stream = nullptr);

  /**
   * @brief Copy a temporal block from GPU to CPU.
   *
   * @param block The temporal block to copy.
   *
   * @return A pointer to the temporal block on the CPU.
   */
  TemporalBlock* SwapBlockToHost(TemporalBlock* block,
                                 cudaStream_t stream = nullptr);

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

  TemporalBlock* GetTheOldestBlockOnDevice() const;

  std::size_t num_blocks_on_device() const;
  std::size_t num_blocks_on_host() const;

  std::size_t used_space_on_device() const;
  std::size_t used_space_on_host() const;

 private:
  void AllocateInternal(TemporalBlock* block, std::size_t size,
                        cudaStream_t stream = nullptr) noexcept(false);

  void DeallocateInternal(TemporalBlock* block, cudaStream_t stream = nullptr);

  std::size_t min_block_size_;
  std::stack<rmm::mr::device_memory_resource*> gpu_resources_;

  // sequence number -> block raw pointer
  std::map<uint64_t, TemporalBlock*> blocks_on_device_;
  std::map<uint64_t, TemporalBlock*> blocks_on_host_;

  std::unordered_map<TemporalBlock*, uint64_t> block_to_seq_;

  // a monotonically increasing sequence number
  uint64_t block_sequence_number_;

  std::mutex mutex_;
};

}  // namespace dgnn

#endif  // DGNN_TEMPORAL_BLOCK_ALLOCATOR_H_
