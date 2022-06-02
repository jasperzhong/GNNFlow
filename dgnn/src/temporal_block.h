#ifndef DGNN_COMMON_H_
#define DGNN_COMMON_H_

#include <thrust/device_ptr.h>
#include <thrust/pair.h>

#include <cstddef>

#include "logging.h"
namespace dgnn {

using NIDType = uint64_t;
using EIDType = uint64_t;
using TimestampType = float;

static constexpr std::size_t kBlockSpaceSize =
    (sizeof(NIDType) + sizeof(EIDType) + sizeof(TimestampType));

/**
 * @brief This POD is used to store the temporal blocks in the graph.
 *
 * The blocks are stored in a doubly linked list. The first block is the newest
 * block. Each block stores the neighbor nodes, timestamps of the edges and IDs
 * of edges. The neighbor nodes and corresponding edge ids are sorted by
 * timestamps. The block has a maximum capacity and can only store a certain
 * number of edges. The block can be moved to a different device.
 */
struct TemporalBlock {
  NIDType* dst_nodes;
  TimestampType* timestamps;
  EIDType* eids;

  std::size_t size;
  std::size_t capacity;

  TemporalBlock* prev;
  TemporalBlock* next;
};

/**
 * @brief This class is doubly linked list of temporal blocks.
 */
struct DoublyLinkedList {
  TemporalBlock head;
  TemporalBlock tail;
  std::size_t size;

  DoublyLinkedList() : size(0) {
    head.prev = nullptr;
    head.next = &tail;
    tail.prev = &head;
    tail.next = nullptr;
  }
};

__host__ __device__ void InsertBlockToDoublyLinkedList(
    DoublyLinkedList* node_table, NIDType node_id, TemporalBlock* block);

__global__ void InsertBlockToDoublyLinkedListKernel(
    DoublyLinkedList* node_table, NIDType node_id, TemporalBlock* block);

}  // namespace dgnn
#endif  // DGNN_COMMON_H_
