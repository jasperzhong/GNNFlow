#include <thrust/device_delete.h>

#include "temporal_block.h"

namespace dgnn {

__host__ __device__ void InsertBlockToDoublyLinkedList(
    DoublyLinkedList* node_table, NIDType node_id, TemporalBlock* block) {
  auto& list = node_table[node_id];
  auto head_next = list.head.next;
  list.head.next = block;
  block->prev = &list.head;
  block->next = head_next;
  head_next->prev = block;
  list.size++;
}

__global__ void InsertBlockToDoublyLinkedListKernel(
    DoublyLinkedList* node_table, NIDType node_id, TemporalBlock* block) {
  InsertBlockToDoublyLinkedList(node_table, node_id, block);
}

__host__ __device__ void ReplaceBlockInDoublyLinkedList(
    DoublyLinkedList* node_table, NIDType node_id, TemporalBlock* block) {
  auto& list = node_table[node_id];
  auto to_delete = list.head.next;
  list.head.next = block;
  block->prev = &list.head;
  block->next = to_delete->next;
  to_delete->next->prev = block;
}

__global__ void ReplaceBlockInDoublyLinkedListKernel(
    DoublyLinkedList* node_table, NIDType node_id, TemporalBlock* block) {
  ReplaceBlockInDoublyLinkedList(node_table, node_id, block);
}

}  // namespace dgnn
