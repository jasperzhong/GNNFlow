#ifndef DGNN_COMMON_TEMPORAL_BLOCK_H_
#define DGNN_COMMON_TEMPORAL_BLOCK_H_

#include <cstddef>
#include <vector>

#include "logging.h"

namespace dgnn {

enum class Device { CPU, GPU };
using NIDType = uint64_t;
using EIDType = uint64_t;
using TimestampType = float;

/** @brief This class is used to store the temporal blocks in the graph.
 *  The blocks are stored in a linked list. The first block is the newest block.
 *  Each block stores the neighbor vertices, timestamps of the edges and IDs of
 *  edges. The neighbor vertices and corresponding edge ids are sorted by
 *  timestamps. The block has a maximum capacity and can only store a certain
 *  number of edges. The block can be moved to a different device.
 */
class TemporalBlock {
 public:
  /** @brief Constructor.
   *
   *  It does not allocate memory.
   *
   *  @param capacity The maximum number of edges that can be stored in the
   *  @param device The device where the block is stored.
   */
  TemporalBlock(std::size_t capacity, Device device);

  // It does not release memory. It is the caller's responsibility to release.
  ~TemporalBlock() = default;

  TemporalBlock(const TemporalBlock&) = delete;

  /**
   * @brief Assignment operator.
   *
   * Note that the `next_` pointer is not copied.
   *
   * @param other The block to be copied. This can be on the same device or on
   * another device.
   *
   * @return The block after the assignment.
   */
  TemporalBlock& operator=(const TemporalBlock&);

  /**
   * @brief Allocate memory for the block.
   *
   * @throw std::bad_alloc or rmm::bad_alloc if the memory cannot be allocated.
   */
  void AllocateMemory() noexcept(false);

  /**
   * @brief Deallocate memory for the block.
   */
  void DeallocateMemory();

  /**
   * @brief Add edges to the block. Assume that the edges are sorted by
   * timestamp.
   *
   * @param neighbor_vertices The target vertices of the edges.
   * @param neighbor_edges The edge ids of the edges.
   * @param timestamps The timestamps of the edges.
   */
  void AddEdges(const std::vector<NIDType>& neighbor_vertices,
                const std::vector<EIDType>& neighbor_edges,
                const std::vector<TimestampType>& timestamps);

  /**
   * @brief Move the block to a different device.
   *
   * @param device The device to move the block to.
   */
  void MoveTo(Device device);

  /**
   * @brief Check if the block is empty. @return True if the block is empty.
   */
  bool IsEmpty() const;

  /**
   * @brief Check if the block is full. @return True if the block is full.
   */
  bool IsFull() const;

  /**
   * @brief Get the number of edges in the block. @return The number of edges in
   * the block.
   */
  std::size_t get_size() const;

  /**
   * @brief Get the space used by the block.
   * @return The space used by the block in bytes.
   */
  std::size_t get_mem_space() const;

  TemporalBlock* get_next_block() const;

 private:
  // NB: the pointers below can be host or device pointers.
  // It depends on the `device_`.
  NIDType* neighbor_vertices_;
  EIDType* neighbor_edges_;
  TimestampType* timestamps_;

  std::size_t size_;
  std::size_t capacity_;
  Device device_;

  // The next block in the linked list.
  TemporalBlock* next_;
};

}  // namespace dgnn
#endif  // DGNN_COMMON_TEMPORAL_BLOCK_H_
