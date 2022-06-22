#ifndef DGNN_SAMPLING_KERNELS_H_
#define DGNN_SAMPLING_KERNELS_H_

#include <curand_kernel.h>

#include "doubly_linked_list.h"

namespace dgnn {

__global__ void SampleLayerRecentKernel(
    const DoublyLinkedList* node_table, std::size_t num_nodes, bool prop_time,
    const NIDType* root_nodes, const TimestampType* root_timestamps,
    const uint32_t* cumsum_num_nodes, uint32_t num_snapshots,
    TimestampType snapshot_time_window, uint32_t num_root_nodes,
    uint32_t fanout, NIDType* src_nodes, TimestampType* timestamps,
    TimestampType* delta_timestamps, EIDType* eids, uint32_t* num_sampled);

__global__ void SampleLayerUniformKernel(
    const DoublyLinkedList* node_table, std::size_t num_nodes, bool prop_time,
    curandState_t* rand_states, uint64_t seed, uint32_t offset_per_thread,
    const NIDType* root_nodes, const TimestampType* root_timestamps,
    const uint32_t* cumsum_num_nodes, uint32_t num_snapshots,
    TimestampType snapshot_time_window, uint32_t num_root_nodes,
    uint32_t fanout, NIDType* src_nodes, TimestampType* timestamps,
    TimestampType* delta_timestamps, EIDType* eids, uint32_t* num_sampled);

}  // namespace dgnn

#endif  // DGNN_SAMPLING_KERNELS_H_
