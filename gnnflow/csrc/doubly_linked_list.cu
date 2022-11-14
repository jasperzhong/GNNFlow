#include "doubly_linked_list.h"

namespace gnnflow {

__device__ void InsertBlockToDoublyLinkedList(DoublyLinkedList* node_table,
                                              NIDType node_id,
                                              TemporalBlock* block) {
  auto& list = node_table[node_id];
  if (list.tail == nullptr) {
    list.tail = block;
    block->prev = nullptr;
    block->next = nullptr;
  } else {
    // append to the tail
    list.tail->next = block;
    block->prev = list.tail;
    block->next = nullptr;
    list.tail = block;
  }
}

void InsertBlockToDoublyLinkedList(HostDoublyLinkedList* node_table,
                                   NIDType node_id, TemporalBlock* block) {
  auto& list = node_table[node_id];
  if (list.tail == nullptr) {
    list.tail = block;
    block->prev = nullptr;
    block->next = nullptr;
  } else {
    // append to the tail
    list.tail->next = block;
    block->prev = list.tail;
    block->next = nullptr;
    list.tail = block;
  }
  list.size++;
}

__global__ void InsertBlockToDoublyLinkedListKernel(
    DoublyLinkedList* node_table, NIDType node_id, TemporalBlock* block) {
  InsertBlockToDoublyLinkedList(node_table, node_id, block);
}

}  // namespace gnnflow
