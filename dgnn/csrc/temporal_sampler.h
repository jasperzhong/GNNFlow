#ifndef DGNN_TEMPORAL_SAMPLER_H_
#define DGNN_TEMPORAL_SAMPLER_H_

#include "common.h"
#include "doubly_linked_list.h"

namespace dgnn {

__global__ void sample_layer_from_root(
    NIDType* dst_nodes, TimestampType* timestamps, uint32_t fanout,
    DoublyLinkedList* node_table, uint64_t num_nodes, NIDType* ret_src_nodes,
    TimestampType* ret_timestamps, EIDType* ret_eids,
    TimestampType* ret_delta_ts);

}  // namespace dgnn

#endif  // DGNN_TEMPORAL_SAMPLER_H_
