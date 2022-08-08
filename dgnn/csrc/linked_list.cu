#include <thrust/device_delete.h>

#include "linked_list.h"

namespace dgnn {

__host__ __device__ void InsertBlockToLinkedList(LinkedList* node_table,
                                                 NIDType node_id,
                                                 TemporalBlock* block) {
  auto& list = node_table[node_id];
  if (list.head == nullptr) {
    list.head = block;
    list.tail = block;
    block->next = nullptr;
  } else {
    list.tail->next = block;
    block->next = nullptr;
    list.tail = block;
  }
  list.size++;
}

__global__ void InsertBlockToLinkedListKernel(LinkedList* node_table,
                                              NIDType node_id,
                                              TemporalBlock* block) {
  InsertBlockToLinkedList(node_table, node_id, block);
}

}  // namespace dgnn
