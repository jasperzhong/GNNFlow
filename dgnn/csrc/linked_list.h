#ifndef DGNN_LINKED_LIST_H_
#define DGNN_LINKED_LIST_H_

#include <thrust/device_ptr.h>
#include <thrust/pair.h>

#include "common.h"
#include "logging.h"

namespace dgnn {

/**
 * @brief This class is linked list of temporal blocks.
 */
struct LinkedList {
  TemporalBlock* head;
  TemporalBlock* tail;
  uint32_t size;

  __host__ __device__ LinkedList() : head(nullptr), tail(nullptr), size(0) {}
};

__host__ __device__ void InsertBlockToLinkedList(LinkedList* node_table,
                                                 NIDType node_id,
                                                 TemporalBlock* block);

__global__ void InsertBlockToLinkedListKernel(LinkedList* node_table,
                                              NIDType node_id,
                                              TemporalBlock* block);
}  // namespace dgnn
#endif  // DGNN_DOUBLY_LINKED_LIST_H_
