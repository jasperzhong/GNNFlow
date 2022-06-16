#include <math.h>

#include "sampling_kernels.h"

namespace dgnn {

__host__ __device__ void LowerBound(TimestampType* timestamps,
                                    std::size_t num_edges,
                                    TimestampType timestamp, std::size_t* idx) {
  std::size_t left = 0;
  std::size_t right = num_edges;
  while (left < right) {
    std::size_t mid = (left + right) / 2;
    if (timestamps[mid] < timestamp) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }
  *idx = left;
}

__host__ __device__ void UpperBound(TimestampType* timestamps,
                                    std::size_t num_edges,
                                    TimestampType timestamp, std::size_t* idx) {
  std::size_t left = 0;
  std::size_t right = num_edges;
  while (left < right) {
    std::size_t mid = (left + right) / 2;
    if (timestamps[mid] <= timestamp) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }
  *idx = left;
}
template <typename T>
__device__ void inline swap(T a, T b) {
  T c(a);
  a = b;
  b = c;
}

__device__ void QuickSort(uint32_t* indices, int lo, int hi) {
  if (lo >= hi || lo < 0) return;

  uint32_t pivot = indices[hi];
  int i = lo - 1;
  for (int j = lo; j < hi; ++j) {
    if (indices[j] < pivot) {
      swap(indices[++i], indices[j]);
    }
  }
  swap(indices[++i], indices[hi]);

  QuickSort(indices, lo, i - 1);
  QuickSort(indices, i + 1, hi);
}

struct SamplingRangeInBlock {
  // [start_idx, end_idx)
  TemporalBlock* block;
  std::size_t start_idx;
  std::size_t end_idx;

  __host__ __device__ SamplingRangeInBlock()
      : block(nullptr), start_idx(0), end_idx(0) {}
};

__global__ void SampleLayerKernel(
    const DoublyLinkedList* node_table, std::size_t num_nodes,
    SamplingPolicy sampling_policy, bool prop_time, curandState_t* rand_states,
    uint64_t seed, NIDType* root_nodes, TimestampType* root_timestamps,
    TimestampType* time_offsets, TimestampType snapshot_time_window,
    std::size_t num_root_nodes, uint32_t fanout, NIDType* src_nodes,
    TimestampType* timestamps, TimestampType* delta_timestamps, EIDType* eids,
    uint32_t* num_sampled) {
  uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= num_root_nodes) {
    return;
  }

  NIDType nid = root_nodes[tid];
  TimestampType root_timestamp = root_timestamps[tid];
  TimestampType end_timestamp = root_timestamp - time_offsets[tid];
  TimestampType start_timestamp = fabs(snapshot_time_window) > 1e-6
                                      ? end_timestamp - snapshot_time_window
                                      : 0;

  auto& list = node_table[nid];
  uint32_t num_candidates = 0;

  SamplingRangeInBlock* sampling_range = new SamplingRangeInBlock[list.size];
  auto curr = list.head;
  uint32_t curr_idx = 0;
  while (curr != nullptr) {
    if (end_timestamp < curr->timestamps[0]) {
      // search in the next block
      curr = curr->next;
      curr_idx++;
      continue;
    }

    if (start_timestamp > curr->timestamps[curr->size - 1]) {
      // no need to search in the next block
      break;
    }

    // search in the current block
    if (start_timestamp >= curr->timestamps[0] &&
        end_timestamp <= curr->timestamps[curr->size - 1]) {
      // all edges in the current block
      LowerBound(curr->timestamps, curr->size, start_timestamp,
                 &sampling_range[curr_idx].start_idx);
      UpperBound(curr->timestamps, curr->size, end_timestamp,
                 &sampling_range[curr_idx].end_idx);
      sampling_range[curr_idx].block = curr;

      num_candidates +=
          sampling_range[curr_idx].end_idx - sampling_range[curr_idx].start_idx;
      break;
    } else if (start_timestamp < curr->timestamps[0] &&
               end_timestamp >= curr->timestamps[curr->size - 1]) {
      // only the edges before end_timestamp are in the current block
      sampling_range[curr_idx].start_idx = 0;
      UpperBound(curr->timestamps, curr->size, end_timestamp,
                 &sampling_range[curr_idx].end_idx);
      sampling_range[curr_idx].block = curr;

      num_candidates +=
          sampling_range[curr_idx].end_idx - sampling_range[curr_idx].start_idx;
      curr = curr->next;
      curr_idx++;
      continue;
    } else if (start_timestamp >= curr->timestamps[0] &&
               end_timestamp > curr->timestamps[curr->size - 1]) {
      // only the edges after start_timestamp are in the current block
      LowerBound(curr->timestamps, curr->size, start_timestamp,
                 &sampling_range[curr_idx].start_idx);
      sampling_range[curr_idx].end_idx = curr->size;
      sampling_range[curr_idx].block = curr;

      num_candidates +=
          sampling_range[curr_idx].end_idx - sampling_range[curr_idx].start_idx;
      break;
    } else {
      // the whole block is in the range
      sampling_range[curr_idx].start_idx = 0;
      sampling_range[curr_idx].end_idx = curr->size;
      sampling_range[curr_idx].block = curr;

      num_candidates += curr->size;
      curr = curr->next;
      curr_idx++;
      continue;
    }
  }

  uint32_t* indices = new uint32_t[fanout];
  if (sampling_policy == SamplingPolicy::kSamplingPolicyRecent) {
    for (uint32_t i = 0; i < fanout; i++) {
      indices[i] = i;
    }
  } else if (sampling_policy == SamplingPolicy::kSamplingPolicyUniform) {
    curand_init(seed, tid, 0, &rand_states[tid]);
    for (uint32_t i = 0; i < fanout; i++) {
      indices[i] = curand(rand_states + tid) % num_candidates;
    }
    QuickSort(indices, 0, fanout - 1);
  }

  uint32_t cumsum = 0;
  uint32_t j = 0;
  uint32_t offset = tid * fanout;
  for (uint32_t i = 0; i < list.size; i++) {
    if (sampling_range[i].block == nullptr) {
      continue;
    }
    auto idx = indices[j] - cumsum;
    auto start_idx = sampling_range[i].start_idx;
    auto end_idx = sampling_range[i].end_idx;

    while (j < fanout && idx < end_idx - start_idx) {
      // start from end_idx (newer edges)
      src_nodes[offset + j] =
          sampling_range[i].block->dst_nodes[end_idx - idx - 1];
      timestamps[offset + j] =
          prop_time ? root_timestamp
                    : sampling_range[i].block->timestamps[end_idx - idx - 1];
      delta_timestamps[offset + j] =
          root_timestamp -
          sampling_range[i].block->timestamps[end_idx - idx - 1];
      eids[offset + j] = sampling_range[i].block->eids[end_idx - idx - 1];
      idx = indices[++j] - cumsum;
    }

    if (j >= fanout) {
      break;
    }
    cumsum += end_idx - start_idx;
  }

  num_sampled[tid] = j;

  delete[] sampling_range;
  delete[] indices;
}

}  // namespace dgnn
