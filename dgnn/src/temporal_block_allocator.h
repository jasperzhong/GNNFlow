#ifndef DGNN_COMMON_TEMPORAL_BLOCK_ALLOCATOR_H_
#define DGNN_COMMON_TEMPORAL_BLOCK_ALLOCATOR_H_

#include <thrust/device_ptr.h>

#include <cstddef>
#include <map>
#include <memory>
#include <type_traits>

#include "temporal_block.h"

namespace dgnn {

/**
 * @brief Allocate device memory for temporal blocks.
 *
 *  The allocator allocates temporal blocks on the GPU. When the GPU memory
 *  usage reaches a threshold, the allocator swaps old and unused temporal
 *  blocks on the GPU to the host. The allocator keeps track of the temporal
 *  blocks that are currently in use.
 *
 *  Note that each GPU device has its own allocator.
 */
class TemporalBlockAllocator {
 public:
  // minimum block size
  static constexpr std::size_t kDefaultAlignment = 16;

  /**
   * @brief Constructor.
   *
   * @param alignment The minimum alignment of the block. Default is 16.
   */
  explicit TemporalBlockAllocator(std::size_t alignment = kDefaultAlignment);

  ~TemporalBlockAllocator();

  /**
   * @brief Allocate a TemporalBlock on device.
   *
   * If the memory is not enough, swap old blocks to the host.
   *
   * @param size The size of the block.
   *
   * @return The device pointer to the allocated block.
   */
  thrust::device_ptr<TemporalBlock> Allocate(std::size_t size);

  /**
   * @brief Free a TemporalBlock.
   *
   * @param block The pointer to the block to free.
   */
  void Deallocate(TemporalBlock *block);

  /**
   * @brief Reallocate a TemporalBlock on device.
   *
   * @param block The pointer to the block to reallocate.
   * @param size The new size of the block.
   */
  thrust::device_ptr<TemporalBlock> Reallocate(
      thrust::device_ptr<TemporalBlock> block, std::size_t size);

 private:
  std::size_t AlignUp(std::size_t size);

  void SwapOldBlocksToHost(std::size_t swap_size);

  std::size_t alignment_;
  // block raw pointer (device/host) -> sequence number (how old the block is)
  std::map<TemporalBlock *, std::size_t> on_device_blocks_;
  std::map<TemporalBlock *, std::size_t> on_host_blocks_;

  // a monotonically increasing sequence number
  std::size_t sequence_number_;
};

};  // namespace dgnn

#endif  // DGNN_COMMON_TEMPORAL_BLOCK_ALLOCATOR_H_
