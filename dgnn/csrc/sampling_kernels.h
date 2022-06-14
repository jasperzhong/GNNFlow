#ifndef DGNN_SAMPLING_KERNELS_H_
#define DGNN_SAMPLING_KERNELS_H_

#include "doubly_linked_list.h"

namespace dgnn {

__global__ void SampleLayerFromRootKernel(
    const DoublyLinkedList* node_table, std::size_t num_nodes,
    SamplingPolicy sampling_policy, curandState_t* rand_states, uint64_t seed,
    NIDType* root_nodes, TimestampType* start_timestamps,
    TimestampType* end_timestamps, std::size_t num_dst_nodes, uint32_t fanout,
    NIDType* src_nodes, TimestampType* timestamps,
    TimestampType* delta_timestamps, EIDType* eids, uint32_t* num_sampled);

}  // namespace dgnn

#endif  // DGNN_SAMPLING_KERNELS_H_
