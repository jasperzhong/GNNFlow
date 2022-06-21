#ifndef DGNN_SAMPLING_KERNELS_H_
#define DGNN_SAMPLING_KERNELS_H_

#include <curand_kernel.h>

#include "doubly_linked_list.h"

namespace dgnn {

__global__ void SampleLayerRecentKernel(
    const DoublyLinkedList* node_table, std::size_t num_nodes, bool prop_time,
    const NIDType* root_nodes, const TimestampType* root_timestamps,
    const TimestampType* time_offsets, TimestampType snapshot_time_window,
    uint32_t num_root_nodes, uint32_t fanout, NIDType* src_nodes,
    TimestampType* timestamps, TimestampType* delta_timestamps, EIDType* eids,
    uint32_t* num_sampled);

__global__ void SampleLayerUniformKernel(
    const DoublyLinkedList* node_table, std::size_t num_nodes, bool prop_time,
    curandState_t* rand_states, uint64_t seed, const NIDType* root_nodes,
    const TimestampType* root_timestamps, const TimestampType* time_offsets,
    TimestampType snapshot_time_window, uint32_t num_root_nodes,
    uint32_t fanout, NIDType* src_nodes, TimestampType* timestamps,
    TimestampType* delta_timestamps, EIDType* eids, uint32_t* num_sampled);

}  // namespace dgnn

#endif  // DGNN_SAMPLING_KERNELS_H_
