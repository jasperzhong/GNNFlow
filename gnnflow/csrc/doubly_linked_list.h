#ifndef GNNFLOW_DOUBLY_LINKED_LIST_H_
#define GNNFLOW_DOUBLY_LINKED_LIST_H_

#include <thrust/device_ptr.h>
#include <thrust/pair.h>

#include "common.h"
#include "logging.h"

namespace gnnflow {

/**
 * @brief This class is doubly linked list of temporal blocks.
 */
struct DoublyLinkedList {
  TemporalBlock* tail;

  __device__ DoublyLinkedList() : tail(nullptr) {}
};

struct HostDoublyLinkedList {
  TemporalBlock* head;
  TemporalBlock* tail;
  std::size_t num_edges;
  std::size_t num_insertions;
  std::size_t size;

  HostDoublyLinkedList()
      : head(nullptr),
        tail(nullptr),
        num_edges(0),
        num_insertions(0),
        size(0) {}
};

__device__ void InsertBlockToDoublyLinkedList(DoublyLinkedList* node_table,
                                              NIDType node_id,
                                              TemporalBlock* block);

__device__ void RemoveBlockFromDoublyLinkedList(DoublyLinkedList* node_table,
                                                NIDType node_id,
                                                TemporalBlock* block);

void InsertBlockToDoublyLinkedList(HostDoublyLinkedList* node_table,
                                   NIDType node_id, TemporalBlock* block);

void RemoveBlockFromDoublyLinkedList(HostDoublyLinkedList* node_table,
                                     NIDType node_id, TemporalBlock* block);

__global__ void InsertBlockToDoublyLinkedListKernel(
    DoublyLinkedList* node_table, NIDType node_id, TemporalBlock* block);

__global__ void RemoveBlockFromDoublyLinkedListKernel(
    DoublyLinkedList* node_table, NIDType node_id, TemporalBlock* next_block);

}  // namespace gnnflow
#endif  // GNNFLOW_DOUBLY_LINKED_LIST_H_
