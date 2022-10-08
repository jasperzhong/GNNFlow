#ifndef GNNFLOW_SAMPLING_KERNELS_H_
#define GNNFLOW_SAMPLING_KERNELS_H_

#include <curand_kernel.h>

#include "doubly_linked_list.h"

namespace gnnflow {

__global__ void SampleLayerRecentKernel(
    const DoublyLinkedList* node_table, std::size_t num_nodes, bool prop_time,
    const NIDType* root_nodes, const TimestampType* root_timestamps,
    uint32_t snapshot_idx, uint32_t num_snapshots,
    TimestampType snapshot_time_window, uint32_t num_root_nodes,
    uint32_t fanout, NIDType* src_nodes, EIDType* eid,
    TimestampType* timestamps, TimestampType* delta_timestamps,
    uint32_t* num_sampled);

__global__ void SampleLayerUniformKernel(
    const DoublyLinkedList* node_table, std::size_t num_nodes, bool prop_time,
    curandState_t* rand_states, uint64_t seed, uint32_t offset_per_thread,
    const NIDType* root_nodes, const TimestampType* root_timestamps,
    uint32_t snapshot_idx, uint32_t num_snapshots,
    TimestampType snapshot_time_window, uint32_t num_root_nodes,
    uint32_t fanout, NIDType* src_nodes, EIDType* eids,
    TimestampType* timestamps, TimestampType* delta_timestamps,
    uint32_t* num_sampled);

}  // namespace gnnflow

#endif  // GNNFLOW_SAMPLING_KERNELS_H_
