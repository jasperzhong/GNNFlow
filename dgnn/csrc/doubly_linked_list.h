#ifndef DGNN_DOUBLY_LINKED_LIST_H_
#define DGNN_DOUBLY_LINKED_LIST_H_

#include <thrust/device_ptr.h>
#include <thrust/pair.h>

#include "common.h"
#include "logging.h"

namespace dgnn {

/**
 * @brief This class is doubly linked list of temporal blocks.
 */
struct DoublyLinkedList {
  TemporalBlock* head;
  TemporalBlock* tail;

  __host__ __device__ DoublyLinkedList() : head(nullptr), tail(nullptr) {}
};

__host__ __device__ void InsertBlockToDoublyLinkedList(
    DoublyLinkedList* node_table, NIDType node_id, TemporalBlock* block);

__global__ void InsertBlockToDoublyLinkedListKernel(
    DoublyLinkedList* node_table, NIDType node_id, TemporalBlock* block);

__host__ __device__ void ReplaceBlockInDoublyLinkedList(
    DoublyLinkedList* node_table, NIDType node_id, TemporalBlock* block);

__global__ void ReplaceBlockInDoublyLinkedListKernel(
    DoublyLinkedList* node_table, NIDType node_id, TemporalBlock* block);

__host__ __device__ void DeleteTailFromDoublyLinkedList(
    DoublyLinkedList* node_table, NIDType node_id);

__global__ void DeleteTailFromDoublyLinkedListKernel(
    DoublyLinkedList* node_table, NIDType node_id);

}  // namespace dgnn
#endif  // DGNN_DOUBLY_LINKED_LIST_H_
