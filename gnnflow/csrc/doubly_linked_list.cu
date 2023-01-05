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

__device__ void RemoveBlockFromDoublyLinkedList(DoublyLinkedList* node_table,
                                                NIDType node_id,
                                                TemporalBlock* block) {
  auto& list = node_table[node_id];
  if (block->prev == nullptr && block->next == nullptr) {
    // only one block
    list.tail = nullptr;
  } else if (block->prev == nullptr) {
    // block is the head
    block->next->prev = nullptr;
  } else if (block->next == nullptr) {
    // block is the tail
    list.tail = block->prev;
    block->prev->next = nullptr;
  } else {
    // block is in the middle
    block->prev->next = block->next;
    block->next->prev = block->prev;
  }
}

void InsertBlockToDoublyLinkedList(HostDoublyLinkedList* node_table,
                                   NIDType node_id, TemporalBlock* block) {
  auto& list = node_table[node_id];
  if (list.tail == nullptr) {
    list.tail = block;
    list.head = block;
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

void RemoveBlockFromDoublyLinkedList(HostDoublyLinkedList* node_table,
                                     NIDType node_id, TemporalBlock* block) {
  auto& list = node_table[node_id];
  if (block->prev == nullptr && block->next == nullptr) {
    // only one block
    list.head = list.tail = nullptr;
  } else if (block->prev == nullptr) {
    // block is the head
    list.head = block->next;
    block->next->prev = nullptr;
  } else if (block->next == nullptr) {
    // block is the tail
    list.tail = block->prev;
    block->prev->next = nullptr;
  } else {
    // block is in the middle
    block->prev->next = block->next;
    block->next->prev = block->prev;
  }
  list.size--;
}

__global__ void InsertBlockToDoublyLinkedListKernel(
    DoublyLinkedList* node_table, NIDType node_id, TemporalBlock* block) {
  InsertBlockToDoublyLinkedList(node_table, node_id, block);
}

__global__ void RemoveBlockFromDoublyLinkedListKernel(
    DoublyLinkedList* node_table, NIDType node_id, TemporalBlock* next_block) {
  RemoveBlockFromDoublyLinkedList(node_table, node_id, next_block);
}
}  // namespace gnnflow
