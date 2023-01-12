#include <math.h>
#include <stdio.h>

#include "common.h"
#include "sampling_kernels.h"
#include "utils.h"

namespace gnnflow {

__global__ void SampleLayerRecentKernel(
    const DoublyLinkedList* node_table, std::size_t num_nodes, bool prop_time,
    const NIDType* root_nodes, const TimestampType* root_timestamps,
    uint32_t snapshot_idx, uint32_t num_snapshots,
    TimestampType snapshot_time_window, uint32_t num_root_nodes,
    uint32_t fanout, NIDType* src_nodes, EIDType* eids,
    TimestampType* timestamps, TimestampType* delta_timestamps,
    uint32_t* num_sampled) {
  uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= num_root_nodes) {
    return;
  }

  NIDType nid = root_nodes[tid];
  TimestampType root_timestamp = root_timestamps[tid];
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
  uint32_t offset = tid * fanout;
  int start_idx, end_idx;
  uint32_t sampled = 0;
  while (curr != nullptr && curr->capacity > 0 && sampled < fanout) {
    if (end_timestamp < curr->start_timestamp) {
      // search in the previous block
      curr = curr->prev;
      continue;
    }

    if (start_timestamp > curr->end_timestamp) {
      // no need to search in the previous block
      break;
    }

    // search in the current block
    if (start_timestamp >= curr->start_timestamp &&
        end_timestamp <= curr->end_timestamp) {
      // all edges in the current block
      LowerBound(curr->timestamps, curr->size, start_timestamp, &start_idx);
      LowerBound(curr->timestamps, curr->size, end_timestamp, &end_idx);
    } else if (start_timestamp < curr->start_timestamp &&
               end_timestamp <= curr->end_timestamp) {
      // only the edges before end_timestamp are in the current block
      start_idx = 0;
      LowerBound(curr->timestamps, curr->size, end_timestamp, &end_idx);
    } else if (start_timestamp > curr->start_timestamp &&
               end_timestamp > curr->end_timestamp) {
      // only the edges after start_timestamp are in the current block
      LowerBound(curr->timestamps, curr->size, start_timestamp, &start_idx);
      end_idx = curr->size;
    } else {
      // the whole block is in the range
      start_idx = 0;
      end_idx = curr->size;
    }

    // copy the edges to the output
    for (int i = end_idx - 1; sampled < fanout && i >= start_idx; --i) {
      src_nodes[offset + sampled] = curr->dst_nodes[i];
      eids[offset + sampled] = curr->eids[i];
      timestamps[offset + sampled] =
          prop_time ? root_timestamp : curr->timestamps[i];
      delta_timestamps[offset + sampled] = root_timestamp - curr->timestamps[i];
      ++sampled;
    }

    curr = curr->prev;
  }

  num_sampled[tid] = sampled;

  while (sampled < fanout) {
    src_nodes[offset + sampled] = kInvalidNID;
    ++sampled;
  }
}

__global__ void SampleLayerUniformKernel(
    const DoublyLinkedList* node_table, std::size_t num_nodes, bool prop_time,
    curandState_t* rand_states, uint64_t seed, uint32_t offset_per_thread,
    const NIDType* root_nodes, const TimestampType* root_timestamps,
    uint32_t snapshot_idx, uint32_t num_snapshots,
    TimestampType snapshot_time_window, uint32_t num_root_nodes,
    uint32_t fanout, NIDType* src_nodes, EIDType* eids,
    TimestampType* timestamps, TimestampType* delta_timestamps,
    uint32_t* num_sampled) {
  uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= num_root_nodes) {
    return;
  }

  extern __shared__ SamplingRange ranges[];

  NIDType nid = root_nodes[tid];
  TimestampType root_timestamp = root_timestamps[tid];
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
  while (curr != nullptr && curr->capacity > 0) {
    if (end_timestamp < curr->start_timestamp) {
      // search in the prev block
      curr = curr->prev;
      curr_idx += 1;
      continue;
    }

    if (start_timestamp > curr->end_timestamp) {
      // no need to search in the prev block
      break;
    }

    // search in the current block
    if (start_timestamp >= curr->start_timestamp &&
        end_timestamp <= curr->end_timestamp) {
      // all edges in the current block
      LowerBound(curr->timestamps, curr->size, start_timestamp, &start_idx);
      LowerBound(curr->timestamps, curr->size, end_timestamp, &end_idx);
    } else if (start_timestamp < curr->start_timestamp &&
               end_timestamp <= curr->end_timestamp) {
      // only the edges before end_timestamp are in the current block
      start_idx = 0;
      LowerBound(curr->timestamps, curr->size, end_timestamp, &end_idx);
    } else if (start_timestamp > curr->start_timestamp &&
               end_timestamp > curr->end_timestamp) {
      // only the edges after start_timestamp are in the current block
      LowerBound(curr->timestamps, curr->size, start_timestamp, &start_idx);
      end_idx = curr->size;
    } else {
      // the whole block is in the range
      start_idx = 0;
      end_idx = curr->size;
    }

    if (curr_idx < offset_per_thread) {
      ranges[offset_by_thread + curr_idx].start_idx = start_idx;
      ranges[offset_by_thread + curr_idx].end_idx = end_idx;
    }

    num_candidates += end_idx - start_idx;
    curr = curr->prev;
    curr_idx += 1;
  }

  uint32_t indices[kMaxFanout];
  uint32_t to_sample = min(fanout, num_candidates);
  for (uint32_t i = 0; i < to_sample; i++) {
    indices[i] = curand(rand_states + tid) % num_candidates;
  }
  QuickSort(indices, 0, to_sample - 1);

  uint32_t sampled = 0;
  uint32_t offset = tid * fanout;

  curr = list.tail;
  curr_idx = 0;
  uint32_t cumsum = 0;
  while (curr != nullptr && curr->capacity > 0) {
    if (end_timestamp < curr->start_timestamp) {
      // search in the prev block
      curr = curr->prev;
      curr_idx += 1;
      continue;
    }

    if (start_timestamp > curr->end_timestamp) {
      // no need to search in the prev block
      break;
    }

    if (curr_idx < offset_per_thread) {
      start_idx = ranges[offset_by_thread + curr_idx].start_idx;
      end_idx = ranges[offset_by_thread + curr_idx].end_idx;
    } else {
      // search in the current block
      if (start_timestamp >= curr->start_timestamp &&
          end_timestamp <= curr->end_timestamp) {
        // all edges in the current block
        LowerBound(curr->timestamps, curr->size, start_timestamp, &start_idx);
        LowerBound(curr->timestamps, curr->size, end_timestamp, &end_idx);
      } else if (start_timestamp < curr->start_timestamp &&
                 end_timestamp <= curr->end_timestamp) {
        // only the edges before end_timestamp are in the current block
        start_idx = 0;
        LowerBound(curr->timestamps, curr->size, end_timestamp, &end_idx);
      } else if (start_timestamp > curr->start_timestamp &&
                 end_timestamp > curr->end_timestamp) {
        // only the edges after start_timestamp are in the current block
        LowerBound(curr->timestamps, curr->size, start_timestamp, &start_idx);
        end_idx = curr->size;
      } else {
        // the whole block is in the range
        start_idx = 0;
        end_idx = curr->size;
      }
    }

    auto idx = indices[sampled] - cumsum;
    while (sampled < to_sample && idx < end_idx - start_idx) {
      // start from end_idx (newer edges)
      src_nodes[offset + sampled] = curr->dst_nodes[end_idx - idx - 1];
      eids[offset + sampled] = curr->eids[end_idx - idx - 1];
      timestamps[offset + sampled] =
          prop_time ? root_timestamp : curr->timestamps[end_idx - idx - 1];
      delta_timestamps[offset + sampled] =
          root_timestamp - curr->timestamps[end_idx - idx - 1];
      idx = indices[++sampled] - cumsum;
    }

    if (sampled >= to_sample) {
      break;
    }

    cumsum += end_idx - start_idx;
    curr = curr->prev;
    curr_idx += 1;
  }

  num_sampled[tid] = sampled;

  while (sampled < fanout) {
    src_nodes[offset + sampled] = kInvalidNID;
    ++sampled;
  }
}

}  // namespace gnnflow
