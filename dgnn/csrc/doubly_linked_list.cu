#include <thrust/device_delete.h>

#include "doubly_linked_list.h"

namespace dgnn {

__host__ __device__ void InsertBlockToDoublyLinkedList(
    DoublyLinkedList* node_table, NIDType node_id, TemporalBlock* block) {
  auto& list = node_table[node_id];
  if (list.head == nullptr) {
    list.head = block;
    list.tail = block;
    block->prev = nullptr;
    block->next = nullptr;
  } else {
    block->prev = nullptr;
    block->next = list.head;
    list.head->prev = block;
    list.head = block;
  }
  list.size++;
}

__global__ void InsertBlockToDoublyLinkedListKernel(
    DoublyLinkedList* node_table, NIDType node_id, TemporalBlock* block) {
  InsertBlockToDoublyLinkedList(node_table, node_id, block);
}

__host__ __device__ void ReplaceBlockInDoublyLinkedList(
    DoublyLinkedList* node_table, NIDType node_id, TemporalBlock* block) {
  auto& list = node_table[node_id];
  block->prev = nullptr;
  block->next = list.head->next;
  if (list.head->next != nullptr) {
    list.head->next->prev = block;
  }
  list.head->next = nullptr;
  list.head->prev = nullptr;
  list.head = block;
}

__global__ void ReplaceBlockInDoublyLinkedListKernel(
    DoublyLinkedList* node_table, NIDType node_id, TemporalBlock* block) {
  ReplaceBlockInDoublyLinkedList(node_table, node_id, block);
}

__host__ __device__ void DeleteTailFromDoublyLinkedList(
    DoublyLinkedList* node_table, NIDType node_id) {
  auto& list = node_table[node_id];

  if (list.tail->prev != nullptr) {
    list.tail->prev->next = nullptr;
    list.tail = list.tail->prev;
  } else {
    list.head = nullptr;
    list.tail = nullptr;
  }
  list.size--;
}

__global__ void DeleteTailFromDoublyLinkedListKernel(
    DoublyLinkedList* node_table, NIDType node_id) {
  DeleteTailFromDoublyLinkedList(node_table, node_id);
}

}  // namespace dgnn
