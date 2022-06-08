#include "temporal_sampler.h"

namespace dgnn {

__global__ void sample_layer_from_root(
    NIDType* dst_nodes, TimestampType* timestamps, uint32_t num_sample,
    uint32_t fanout, uint32_t num_snapshot, DoublyLinkedList* node_table,
    uint64_t num_nodes, NIDType* ret_src_nodes, TimestampType* ret_timestamps,
    EIDType* ret_eids, TimestampType* ret_delta_ts, uint32_t* ret_num_nodes) {
  // Get the index of the current thread.
  uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= num_sample) {
    return;
  }

  // Get the node ID and timestamp of the current thread.
  NIDType nid = dst_nodes[tid];
  TimestampType timestamp = timestamps[tid];

  // Get the doubly linked list of the current thread.
  auto& list = node_table[nid];
  auto head = list.head;
  auto tail = list.tail;

  auto cur = head.next;
  while (cur != &tail) {
    auto start_timestamp = cur->timestamps[0];
    auto end_timestamp = cur->timestamps[cur->size - 1];

    if (timestamp < start_timestamp) {
      // this block does not contain any edges before the timestamp
      continue;
    } else if (timestamp > end_timestamp) {
      // this block contains all edges before the timestamp

    } else {

    }

    cur = cur->next;
  }
}
}  // namespace dgnn
