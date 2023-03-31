#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

#include "common.h"
#include "sampling_kernels.h"
#include "utils.h"

namespace gnnflow {

__global__ void SampleLayerRecentKernel(
    const DoublyLinkedList* node_table, std::size_t num_nodes, bool prop_time,
    uint32_t offset_per_thread, const NIDType* root_nodes,
    const TimestampType* root_timestamps, uint32_t snapshot_idx,
    uint32_t num_snapshots, TimestampType snapshot_time_window,
    uint32_t maximum_sampled_nodes, uint32_t fanout, NIDType* src_nodes,
    EIDType* eids, TimestampType* timestamps, TimestampType* delta_timestamps,
    uint32_t* num_sampled) {
  uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= maximum_sampled_nodes) {
    return;
  }
  uint32_t nid_index = tid / fanout;
  uint32_t sample_index = tid % fanout;

  NIDType nid = root_nodes[nid_index];
  TimestampType root_timestamp = root_timestamps[nid_index];
  TimestampType start_timestamp, end_timestamp;
  if (num_snapshots == 1) {
    if (abs(snapshot_time_window) < 1e-6) {
      start_timestamp = 0;
    } else {
      start_timestamp = root_timestamp - snapshot_time_window;
    }
    end_timestamp = root_timestamp;
  } else {
    end_timestamp = root_timestamp -
                    (num_snapshots - snapshot_idx - 1) * snapshot_time_window;
    start_timestamp = end_timestamp - snapshot_time_window;
  }

  // NB: the tail block is the newest block
  auto curr = node_table[nid].tail;
  uint32_t offset = nid_index * fanout;
  int start_idx, end_idx;
  int index = sample_index;
  while (curr != nullptr) {
    auto curr_block = *curr;
    if (curr_block.capacity == 0) {
      // the block is empty
      curr = curr_block.prev;
      continue;
    }

    if (end_timestamp < curr_block.start_timestamp) {
      // search in the previous block
      curr = curr_block.prev;
      continue;
    }

    if (start_timestamp > curr_block.end_timestamp) {
      // no need to search in the previous block
      break;
    }    

    // search in the current block
    if (start_timestamp >= curr_block.start_timestamp &&
        end_timestamp <= curr_block.end_timestamp) {
      // all edges in the current block
      LowerBound(curr_block.timestamps, curr_block.size, start_timestamp, &start_idx);
      LowerBound(curr_block.timestamps, curr_block.size, end_timestamp, &end_idx);
    } else if (start_timestamp < curr_block.start_timestamp &&
               end_timestamp <= curr_block.end_timestamp) {
      // only the edges before end_timestamp are in the current block
      start_idx = 0;
      LowerBound(curr_block.timestamps, curr_block.size, end_timestamp, &end_idx);
    } else if (start_timestamp > curr_block.start_timestamp &&
               end_timestamp > curr_block.end_timestamp) {
      // only the edges after start_timestamp are in the current block
      LowerBound(curr_block.timestamps, curr_block.size, start_timestamp, &start_idx);
      end_idx = curr_block.size;
    } else {
      // the whole block is in the range
      start_idx = 0;
      end_idx = curr_block.size;
    }

    int32_t i = end_idx - 1 - index;
    if (i < start_idx) {
      index -= end_idx - start_idx;
      curr = curr_block.prev;
      continue;
    } else {
      // copy the edges to the output
      src_nodes[offset + sample_index] = curr_block.dst_nodes[i];
      eids[offset + sample_index] = curr_block.eids[i];
      auto timestamp = curr_block.timestamps[i];
      timestamps[offset + sample_index] =
          prop_time ? root_timestamp : timestamp;
      delta_timestamps[offset + sample_index] = root_timestamp - timestamp;
      atomicAdd(&num_sampled[nid_index], 1u);
      return;
    }
  }

  src_nodes[offset + sample_index] = kInvalidNID;
}

__global__ void SampleLayerUniformKernel(
    const DoublyLinkedList* node_table, std::size_t num_nodes, bool prop_time,
    curandState_t* rand_states, uint64_t seed, uint32_t offset_per_thread,
    const NIDType* root_nodes, const TimestampType* root_timestamps,
    uint32_t snapshot_idx, uint32_t num_snapshots,
    TimestampType snapshot_time_window, uint32_t maximum_sampled_nodes,
    uint32_t fanout, NIDType* src_nodes, EIDType* eids,
    TimestampType* timestamps, TimestampType* delta_timestamps,
    uint32_t* num_sampled) {
  uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= maximum_sampled_nodes) {
    return;
  }
  uint32_t nid_index = tid / fanout;
  uint32_t sample_index = tid % fanout;

  extern __shared__ SamplingRange ranges[];

  NIDType nid = root_nodes[nid_index];
  TimestampType root_timestamp = root_timestamps[nid_index];
  TimestampType start_timestamp, end_timestamp;
  if (num_snapshots == 1) {
    if (abs(snapshot_time_window) < 1e-6) {
      start_timestamp = 0;
    } else {
      start_timestamp = root_timestamp - snapshot_time_window;
    }
    end_timestamp = root_timestamp;
  } else {
    end_timestamp = root_timestamp -
                    (num_snapshots - snapshot_idx - 1) * snapshot_time_window;
    start_timestamp = end_timestamp - snapshot_time_window;
  }

  auto& list = node_table[nid];
  uint32_t num_candidates = 0;

  // NB: the tail block is the newest block
  auto curr = list.tail;
  int start_idx, end_idx;
  int curr_idx = 0;
  const int offset_by_thread = offset_per_thread * threadIdx.x;
  while (curr != nullptr) {
    auto curr_block = *curr;
    if (curr_block.capacity == 0) {
      // the block is empty
      curr = curr_block.prev;
      continue;
    }
    if (end_timestamp < curr_block.start_timestamp) {
      // search in the prev block
      curr = curr_block.prev;
      curr_idx += 1;
      continue;
    }

    if (start_timestamp > curr_block.end_timestamp) {
      // no need to search in the prev block
      break;
    }

    // search in the current block
    if (start_timestamp >= curr_block.start_timestamp &&
        end_timestamp <= curr_block.end_timestamp) {
      // all edges in the current block
      LowerBound(curr_block.timestamps, curr_block.size, start_timestamp, &start_idx);
      LowerBound(curr_block.timestamps, curr_block.size, end_timestamp, &end_idx);
    } else if (start_timestamp < curr_block.start_timestamp &&
               end_timestamp <= curr_block.end_timestamp) {
      // only the edges before end_timestamp are in the current block
      start_idx = 0;
      LowerBound(curr_block.timestamps, curr_block.size, end_timestamp, &end_idx);
    } else if (start_timestamp > curr_block.start_timestamp &&
               end_timestamp > curr_block.end_timestamp) {
      // only the edges after start_timestamp are in the current block
      LowerBound(curr_block.timestamps, curr_block.size, start_timestamp, &start_idx);
      end_idx = curr_block.size;
    } else {
      // the whole block is in the range
      start_idx = 0;
      end_idx = curr_block.size;
    }

    if (curr_idx < offset_per_thread) {
      ranges[offset_by_thread + curr_idx].start_idx = start_idx;
      ranges[offset_by_thread + curr_idx].end_idx = end_idx;
    }

    num_candidates += end_idx - start_idx;
    curr = curr_block.prev;
    curr_idx += 1;
  }

  uint32_t index = curand(rand_states + tid) % num_candidates;
  uint32_t offset = nid_index * fanout;

  curr = list.tail;
  curr_idx = 0;
  while (curr != nullptr) {
    auto curr_block = *curr;
    if (curr_block.capacity == 0) {
      // the block is empty
      curr = curr_block.prev;
      continue;
    }
    if (end_timestamp < curr_block.start_timestamp) {
      // search in the prev block
      curr = curr_block.prev;
      curr_idx += 1;
      continue;
    }

    if (start_timestamp > curr_block.end_timestamp) {
      // no need to search in the prev block
      break;
    }

    if (curr_idx < offset_per_thread) {
      start_idx = ranges[offset_by_thread + curr_idx].start_idx;
      end_idx = ranges[offset_by_thread + curr_idx].end_idx;
    } else {
      // search in the current block
      if (start_timestamp >= curr_block.start_timestamp &&
          end_timestamp <= curr_block.end_timestamp) {
        // all edges in the current block
        LowerBound(curr_block.timestamps, curr_block.size, start_timestamp, &start_idx);
        LowerBound(curr_block.timestamps, curr_block.size, end_timestamp, &end_idx);
      } else if (start_timestamp < curr_block.start_timestamp &&
                 end_timestamp <= curr_block.end_timestamp) {
        // only the edges before end_timestamp are in the current block
        start_idx = 0;
        LowerBound(curr_block.timestamps, curr_block.size, end_timestamp, &end_idx);
      } else if (start_timestamp > curr_block.start_timestamp &&
                 end_timestamp > curr_block.end_timestamp) {
        // only the edges after start_timestamp are in the current block
        LowerBound(curr_block.timestamps, curr_block.size, start_timestamp, &start_idx);
        end_idx = curr_block.size;
      } else {
        // the whole block is in the range
        start_idx = 0;
        end_idx = curr_block.size;
      }
    }

    int32_t i = end_idx - 1 - index;
    if (i < start_idx) {
      index -= end_idx - start_idx;
      curr = curr_block.prev;
      curr_idx += 1;
      continue;
    } else {
      // copy the edges to the output
      src_nodes[offset + sample_index] = curr_block.dst_nodes[i];
      eids[offset + sample_index] = curr_block.eids[i];
      auto timestamp = curr_block.timestamps[i];
      timestamps[offset + sample_index] =
          prop_time ? root_timestamp : timestamp;
      delta_timestamps[offset + sample_index] = root_timestamp - timestamp;
      atomicAdd(&num_sampled[nid_index], 1u);
      return;
    }
  }

  src_nodes[offset + sample_index] = kInvalidNID;
}

}  // namespace gnnflow
