#include <thrust/execution_policy.h>
#include <thrust/remove.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <rmm/device_vector.hpp>
#include <rmm/exec_policy.hpp>

#include "common.h"
#include "sampling_kernels.h"
#include "temporal_sampler.h"
#include "utils.h"

namespace dgnn {

struct is_invalid_edge {
  __host__ __device__ bool operator()(
      thrust::tuple<NIDType, EIDType, TimestampType, TimestampType> const&
          edge) {
    return thrust::get<0>(edge) == kInvalidNID;
  }
};

void LowerBound(TimestampType* timestamps, int num_edges,
                TimestampType timestamp, int* idx) {
  int left = 0;
  int right = num_edges;
  while (left < right) {
    int mid = (left + right) / 2;
    if (timestamps[mid] < timestamp) {
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
  if (lo >= hi || lo < 0 || hi < 0) return;

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

TemporalSampler::TemporalSampler(const DynamicGraph& graph,
                                 const std::vector<uint32_t>& fanouts,
                                 SamplingPolicy sampling_policy,
                                 uint32_t num_snapshots,
                                 float snapshot_time_window, bool prop_time,
                                 uint64_t seed)
    : graph_(graph),
      fanouts_(fanouts),
      sampling_policy_(sampling_policy),
      num_snapshots_(num_snapshots),
      snapshot_time_window_(snapshot_time_window),
      prop_time_(prop_time),
      num_layers_(fanouts.size()),
      seed_(seed),
      streams_(nullptr),
      cpu_buffer_(nullptr),
      gpu_input_buffer_(nullptr),
      gpu_output_buffer_(nullptr),
      rand_states_(nullptr),
      init_num_root_nodes_(0) {
  if (num_snapshots_ == 1 && std::fabs(snapshot_time_window_) > 0.0f) {
    LOG(WARNING) << "Snapshot time window must be 0 when num_snapshots = 1. "
                    "Ignore the snapshot time window.";
  }
  shared_memory_size_ = GetSharedMemoryMaxSize();

  streams_ = new cudaStream_t[num_snapshots_];
  for (uint32_t i = 0; i < num_snapshots_; ++i) {
    CUDA_CALL(
        cudaStreamCreateWithPriority(&streams_[i], cudaStreamNonBlocking, -1));
  }

  cpu_buffer_ = new char*[num_snapshots_];
  gpu_input_buffer_ = new char*[num_snapshots_];
  gpu_output_buffer_ = new char*[num_snapshots_];
  rand_states_ = new curandState_t*[num_snapshots_];
}

void TemporalSampler::FreeBuffer() {
  if (init_num_root_nodes_ == 0) return;

  for (uint32_t i = 0; i < num_snapshots_; ++i) {
    if (cpu_buffer_[i] != nullptr) {
      cudaFreeHost(cpu_buffer_[i]);
    }

    if (gpu_input_buffer_[i] != nullptr) {
      cudaFree(gpu_input_buffer_[i]);
    }

    if (gpu_output_buffer_[i] != nullptr) {
      cudaFree(gpu_output_buffer_[i]);
    }

    if (rand_states_[i] != nullptr) {
      cudaFree(rand_states_[i]);
    }
  }
}

TemporalSampler::~TemporalSampler() {
  FreeBuffer();
  delete[] cpu_buffer_;
  delete[] gpu_input_buffer_;
  delete[] gpu_output_buffer_;
  delete[] rand_states_;

  for (uint32_t i = 0; i < num_snapshots_; ++i) {
    cudaStreamDestroy(streams_[i]);
  }
  delete[] streams_;
}

void TemporalSampler::InitBuffer(std::size_t num_root_nodes) {
  std::size_t maximum_sampled_nodes = num_root_nodes;
  for (int i = 0; i < num_layers_; i++) {
    // including itself
    maximum_sampled_nodes += maximum_sampled_nodes * fanouts_[i];
  }
  LOG(DEBUG) << "Maximum sampled nodes: " << maximum_sampled_nodes;

  constexpr std::size_t per_node_size =
      sizeof(NIDType) + sizeof(TimestampType) + sizeof(TimestampType) +
      sizeof(EIDType) + sizeof(uint32_t);

  for (uint32_t i = 0; i < num_snapshots_; ++i) {
    // Double the CPU buffer to contain the data from GPU and CPU simultaneously.
    CUDA_CALL(
        cudaMallocHost(&cpu_buffer_[i], per_node_size * maximum_sampled_nodes * 2));

    CUDA_CALL(cudaMalloc(&gpu_input_buffer_[i],
                         per_node_size * maximum_sampled_nodes));
    CUDA_CALL(cudaMalloc(&gpu_output_buffer_[i],
                         per_node_size * maximum_sampled_nodes));

    LOG(DEBUG) << "Allocated CPU & GPU buffer: "
               << maximum_sampled_nodes * per_node_size << " bytes";

    if (sampling_policy_ == SamplingPolicy::kSamplingPolicyUniform) {
      CUDA_CALL(cudaMalloc((void**)&rand_states_[i],
                           maximum_sampled_nodes * sizeof(curandState)));
      uint32_t num_threads_per_block = 256;
      uint32_t num_blocks =
          (maximum_sampled_nodes + num_threads_per_block - 1) /
          num_threads_per_block;

      InitCuRandStates<<<num_blocks, num_threads_per_block>>>(rand_states_[i],
                                                              seed_);
    }
  }
}

std::vector<SamplingResult> TemporalSampler::RootInputToSamplingResult(
    const std::vector<NIDType>& dst_nodes,
    const std::vector<TimestampType>& dst_timestamps) {
  std::vector<SamplingResult> sampling_results(num_snapshots_);
  for (int snapshot = 0; snapshot < num_snapshots_; ++snapshot) {
    auto& sampling_result = sampling_results[snapshot];
    sampling_result.all_nodes.insert(sampling_result.all_nodes.end(),
                                     dst_nodes.begin(), dst_nodes.end());
    sampling_result.all_timestamps.insert(sampling_result.all_timestamps.end(),
                                          dst_timestamps.begin(),
                                          dst_timestamps.end());
  }
  return sampling_results;
}

std::vector<SamplingResult> TemporalSampler::SampleLayer(
    uint32_t layer, const std::vector<SamplingResult>& prev_sampling_results) {
  CHECK_EQ(prev_sampling_results.size(), num_snapshots_);

  // prepare input
  std::vector<std::size_t> num_root_nodes_list(num_snapshots_);
  for (uint32_t snapshot = 0; snapshot < num_snapshots_; ++snapshot) {
    auto& sampling_result = prev_sampling_results.at(snapshot);
    auto& all_nodes = sampling_result.all_nodes;
    auto& all_timestamps = sampling_result.all_timestamps;
    std::size_t num_root_nodes = all_nodes.size();
    num_root_nodes_list[snapshot] = num_root_nodes;

    char* root_nodes_dst = cpu_buffer_[snapshot];
    char* root_timestamps_dst =
        cpu_buffer_[snapshot] + num_root_nodes * sizeof(NIDType);

    // copy all_nodes and all_timestamps to cpu_buffer_
    Copy(root_nodes_dst, all_nodes.data(), all_nodes.size() * sizeof(NIDType));
    Copy(root_timestamps_dst, all_timestamps.data(),
         all_timestamps.size() * sizeof(TimestampType));

    CUDA_CALL(cudaMemcpyAsync(
        gpu_input_buffer_[snapshot], cpu_buffer_[snapshot],
        num_root_nodes * (sizeof(NIDType) + sizeof(TimestampType)),
        cudaMemcpyHostToDevice, streams_[snapshot]));
  }

  // launch kernel for each snapshot
  std::vector<std::size_t> total_output_size_list(num_snapshots_);
  for (uint32_t snapshot = 0; snapshot < num_snapshots_; ++snapshot) {
    std::size_t num_root_nodes = num_root_nodes_list[snapshot];
    std::size_t max_sampled_nodes = num_root_nodes * fanouts_[layer];

    NIDType* d_root_nodes =
        reinterpret_cast<NIDType*>(gpu_input_buffer_[snapshot]);
    TimestampType* d_root_timestamps = reinterpret_cast<TimestampType*>(
        gpu_input_buffer_[snapshot] + num_root_nodes * sizeof(NIDType));

    // device output
    std::size_t offset1 = max_sampled_nodes * sizeof(NIDType);
    std::size_t offset2 = offset1 + max_sampled_nodes * sizeof(EIDType);
    std::size_t offset3 = offset2 + max_sampled_nodes * sizeof(TimestampType);
    std::size_t offset4 = offset3 + max_sampled_nodes * sizeof(TimestampType);

    std::size_t total_output_size = offset4 + num_root_nodes * sizeof(uint32_t);
    total_output_size_list[snapshot] = total_output_size;

    LOG(DEBUG) << "Total output size: " << total_output_size;

    NIDType* d_src_nodes =
        reinterpret_cast<NIDType*>(gpu_output_buffer_[snapshot]);
    EIDType* d_eids =
        reinterpret_cast<EIDType*>(gpu_output_buffer_[snapshot] + offset1);
    TimestampType* d_timestamps = reinterpret_cast<TimestampType*>(
        gpu_output_buffer_[snapshot] + offset2);
    TimestampType* d_delta_timestamps = reinterpret_cast<TimestampType*>(
        gpu_output_buffer_[snapshot] + offset3);
    uint32_t* d_num_sampled =
        reinterpret_cast<uint32_t*>(gpu_output_buffer_[snapshot] + offset4);

    uint32_t num_threads_per_block = 256;
    uint32_t num_blocks =
        (num_root_nodes + num_threads_per_block - 1) / num_threads_per_block;

    if (sampling_policy_ == SamplingPolicy::kSamplingPolicyRecent) {
      SampleLayerRecentKernel<<<num_blocks, num_threads_per_block, 0,
                                streams_[snapshot]>>>(
          graph_.get_device_node_table(), graph_.num_nodes(), prop_time_,
          d_root_nodes, d_root_timestamps, snapshot, num_snapshots_,
          snapshot_time_window_, num_root_nodes, fanouts_[layer], d_src_nodes,
          d_eids, d_timestamps, d_delta_timestamps, d_num_sampled);
    } else if (sampling_policy_ == SamplingPolicy::kSamplingPolicyUniform) {
      int offset_per_thread =
          shared_memory_size_ / sizeof(SamplingRange) / num_threads_per_block;

      LOG(DEBUG) << "Max shared memory size: " << shared_memory_size_
                 << " bytes"
                 << ", offset per thread: " << offset_per_thread;

      SampleLayerUniformKernel<<<num_blocks, num_threads_per_block,
                                 offset_per_thread * num_threads_per_block *
                                     sizeof(SamplingRange),
                                 streams_[snapshot]>>>(
          graph_.get_device_node_table(), graph_.num_nodes(), prop_time_,
          rand_states_[snapshot], seed_, offset_per_thread, d_root_nodes,
          d_root_timestamps, snapshot, num_snapshots_, snapshot_time_window_,
          num_root_nodes, fanouts_[layer], d_src_nodes, d_eids, d_timestamps,
          d_delta_timestamps, d_num_sampled);
    }
  }

  // combine
  std::vector<uint32_t> num_sampled_nodes_list(num_snapshots_);
  for (uint32_t snapshot = 0; snapshot < num_snapshots_; ++snapshot) {
    std::size_t num_root_nodes = num_root_nodes_list[snapshot];

    std::size_t max_sampled_nodes = num_root_nodes * fanouts_[layer];
    std::size_t offset1 = max_sampled_nodes * sizeof(NIDType);
    std::size_t offset2 = offset1 + max_sampled_nodes * sizeof(EIDType);
    std::size_t offset3 = offset2 + max_sampled_nodes * sizeof(TimestampType);
    std::size_t offset4 = offset3 + max_sampled_nodes * sizeof(TimestampType);

    auto d_src_nodes = reinterpret_cast<NIDType*>(gpu_output_buffer_[snapshot]);
    auto d_eids =
        reinterpret_cast<EIDType*>(gpu_output_buffer_[snapshot] + offset1);
    auto d_timestamps = reinterpret_cast<TimestampType*>(
        gpu_output_buffer_[snapshot] + offset2);
    auto d_delta_timestamps = reinterpret_cast<TimestampType*>(
        gpu_output_buffer_[snapshot] + offset3);
    uint32_t* d_num_sampled =
        reinterpret_cast<uint32_t*>(gpu_output_buffer_[snapshot] + offset4);

    auto new_end = thrust::remove_if(
        rmm::exec_policy(streams_[snapshot]),
        thrust::make_zip_iterator(thrust::make_tuple(
            d_src_nodes, d_eids, d_timestamps, d_delta_timestamps)),
        thrust::make_zip_iterator(thrust::make_tuple(
            d_src_nodes + max_sampled_nodes, d_eids + max_sampled_nodes,
            d_timestamps + max_sampled_nodes,
            d_delta_timestamps + max_sampled_nodes)),
        is_invalid_edge());

    uint32_t num_sampled_nodes = thrust::distance(
        thrust::make_zip_iterator(thrust::make_tuple(
            d_src_nodes, d_eids, d_timestamps, d_delta_timestamps)),
        new_end);

    LOG(DEBUG) << "Number of sampled nodes: " << num_sampled_nodes;
    num_sampled_nodes_list[snapshot] = num_sampled_nodes;

    NIDType* src_nodes = reinterpret_cast<NIDType*>(cpu_buffer_[snapshot]);
    EIDType* eids = reinterpret_cast<EIDType*>(cpu_buffer_[snapshot] + offset1);
    TimestampType* timestamps =
        reinterpret_cast<TimestampType*>(cpu_buffer_[snapshot] + offset2);
    TimestampType* delta_timestamps =
        reinterpret_cast<TimestampType*>(cpu_buffer_[snapshot] + offset3);
    uint32_t* num_sampled =
        reinterpret_cast<uint32_t*>(cpu_buffer_[snapshot] + offset4);

    CUDA_CALL(cudaMemcpyAsync(src_nodes, d_src_nodes,
                              sizeof(NIDType) * num_sampled_nodes,
                              cudaMemcpyDeviceToHost, streams_[snapshot]));
    CUDA_CALL(cudaMemcpyAsync(eids, d_eids, sizeof(EIDType) * num_sampled_nodes,
                              cudaMemcpyDeviceToHost, streams_[snapshot]));
    CUDA_CALL(cudaMemcpyAsync(timestamps, d_timestamps,
                              sizeof(TimestampType) * num_sampled_nodes,
                              cudaMemcpyDeviceToHost, streams_[snapshot]));
    CUDA_CALL(cudaMemcpyAsync(delta_timestamps, d_delta_timestamps,
                              sizeof(TimestampType) * num_sampled_nodes,
                              cudaMemcpyDeviceToHost, streams_[snapshot]));
    CUDA_CALL(cudaMemcpyAsync(num_sampled, d_num_sampled,
                              sizeof(uint32_t) * num_root_nodes,
                              cudaMemcpyDeviceToHost, streams_[snapshot]));
  }

  // combine
  std::vector<SamplingResult> sampling_results(num_snapshots_);
  for (uint32_t snapshot = 0; snapshot < num_snapshots_; ++snapshot) {
    auto& prev_sampling_result = prev_sampling_results.at(snapshot);
    std::size_t num_root_nodes = num_root_nodes_list[snapshot];
    std::size_t max_sampled_nodes = num_root_nodes * fanouts_[layer];

    std::size_t offset1 = max_sampled_nodes * sizeof(NIDType);
    std::size_t offset2 = offset1 + max_sampled_nodes * sizeof(EIDType);
    std::size_t offset3 = offset2 + max_sampled_nodes * sizeof(TimestampType);
    std::size_t offset4 = offset3 + max_sampled_nodes * sizeof(TimestampType);

    // host output
    NIDType* src_nodes = reinterpret_cast<NIDType*>(cpu_buffer_[snapshot]);
    EIDType* eids = reinterpret_cast<EIDType*>(cpu_buffer_[snapshot] + offset1);
    TimestampType* timestamps =
        reinterpret_cast<TimestampType*>(cpu_buffer_[snapshot] + offset2);
    TimestampType* delta_timestamps =
        reinterpret_cast<TimestampType*>(cpu_buffer_[snapshot] + offset3);
    uint32_t* num_sampled =
        reinterpret_cast<uint32_t*>(cpu_buffer_[snapshot] + offset4);

    auto& sampling_result = sampling_results[snapshot];

    // copy dst nodes
    std::copy(prev_sampling_results.at(snapshot).all_nodes.begin(),
              prev_sampling_results.at(snapshot).all_nodes.end(),
              std::back_inserter(sampling_result.all_nodes));
    std::copy(prev_sampling_results.at(snapshot).all_timestamps.begin(),
              prev_sampling_results.at(snapshot).all_timestamps.end(),
              std::back_inserter(sampling_result.all_timestamps));

    uint32_t num_sampled_nodes = num_sampled_nodes_list[snapshot];

    sampling_result.col.resize(num_sampled_nodes);
    std::iota(sampling_result.col.begin(), sampling_result.col.end(),
              num_root_nodes);

    sampling_result.num_dst_nodes = num_root_nodes;
    sampling_result.num_src_nodes = num_root_nodes + num_sampled_nodes;

    sampling_result.all_nodes.reserve(sampling_result.num_src_nodes);
    sampling_result.all_timestamps.reserve(sampling_result.num_src_nodes);
    sampling_result.delta_timestamps.reserve(num_sampled_nodes);
    sampling_result.eids.reserve(num_sampled_nodes);

    // synchronize memcpy
    CUDA_CALL(cudaStreamSynchronize(streams_[snapshot]));

    std::copy(src_nodes, src_nodes + num_sampled_nodes,
              std::back_inserter(sampling_result.all_nodes));
    std::copy(timestamps, timestamps + num_sampled_nodes,
              std::back_inserter(sampling_result.all_timestamps));
    std::copy(delta_timestamps, delta_timestamps + num_sampled_nodes,
              std::back_inserter(sampling_result.delta_timestamps));
    std::copy(eids, eids + num_sampled_nodes,
              std::back_inserter(sampling_result.eids));

    sampling_result.row.resize(num_sampled_nodes);
    uint32_t cumsum = 0;
    for (uint32_t i = 0; i < num_root_nodes; i++) {
      std::fill_n(sampling_result.row.begin() + cumsum, num_sampled[i], i);
      cumsum += num_sampled[i];
    }

    CHECK_EQ(cumsum, num_sampled_nodes);
  }

  return sampling_results;
}

std::vector<std::vector<SamplingResult>> TemporalSampler::Sample(
    const std::vector<NIDType>& dst_nodes,
    const std::vector<TimestampType>& timestamps) {
  CHECK_EQ(dst_nodes.size(), timestamps.size());
  std::vector<std::vector<SamplingResult>> results;

  if (dst_nodes.size() > init_num_root_nodes_) {
    FreeBuffer();
    init_num_root_nodes_ = dst_nodes.size();
    InitBuffer(init_num_root_nodes_);
  }

  for (int layer = 0; layer < num_layers_; ++layer) {
    if (layer == 0) {
      results.push_back(
          SampleLayer(layer, RootInputToSamplingResult(dst_nodes, timestamps)));
    } else {
      results.push_back(SampleLayer(layer, results.back()));
    }
  }
  return results;
}

// Recent Snapshot
void RecentLayerUniform(
    const DoublyLinkedList* node_table, std::size_t num_nodes, bool prop_time,
    curandState_t* rand_states, uint64_t seed, uint32_t offset_per_thread,
    const NIDType* root_nodes, const TimestampType* root_timestamps,
    uint32_t snapshot_idx, uint32_t num_snapshots,
    TimestampType snapshot_time_window, uint32_t num_root_nodes,
    uint32_t fanout,
    NIDType* src_nodes, EIDType* eids, // Return Value
    TimestampType* timestamps, TimestampType* delta_timestamps, // Return Value
    uint32_t* num_sampled // Return Value
    ) {

  // tid means the id of the node (probably the thread in the future)
  uint32_t tid = 0;


  // tot_sampled means the max index of the whole sampled result
  // (e.g. the max index of <src_nodes>, <eids> etc.)
  uint32_t tot_sampled = 0;

  for(; tid < num_root_nodes; tid++) {
    NIDType nid = root_nodes[tid];
    TimestampType root_timestamp = root_timestamps[tid];
    TimestampType start_timestamp, end_timestamp;

    if (num_snapshots == 1) {
      start_timestamp = 0;
      end_timestamp = root_timestamp;
    } else {
      end_timestamp = root_timestamp -
                      (num_snapshots - snapshot_idx - 1) * snapshot_time_window;
      start_timestamp = end_timestamp - snapshot_time_window;
    }

    auto curr = node_table[nid].head;
    int start_idx, end_idx;
    uint32_t sampled = 0;
    while(curr != nullptr && sampled < fanout) {
      if (end_timestamp < curr->timestamps[0]) {
        // search in the next block
        curr = curr->next;
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
        LowerBound(curr->timestamps, curr->size, start_timestamp, &start_idx);
        LowerBound(curr->timestamps, curr->size, end_timestamp, &end_idx);
      } else if (start_timestamp < curr->timestamps[0] &&
                 end_timestamp <= curr->timestamps[curr->size - 1]) {
        // only the edges before end_timestamp are in the current block
        start_idx = 0;
        LowerBound(curr->timestamps, curr->size, end_timestamp, &end_idx);
      } else if (start_timestamp > curr->timestamps[0] &&
                 end_timestamp > curr->timestamps[curr->size - 1]) {
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
        src_nodes[tot_sampled] = curr->dst_nodes[i];
        eids[tot_sampled] = curr->eids[i];
        timestamps[tot_sampled] =
            prop_time ? root_timestamp : curr->timestamps[i];
        delta_timestamps[tot_sampled] = root_timestamp - curr->timestamps[i];
        ++sampled;
        ++tot_sampled;
      }

      curr = curr->next;
    }

    num_sampled[tid] = sampled;

  }
}

// Single Snapshot
void SampleLayerUniform(
    const DoublyLinkedList* node_table, std::size_t num_nodes, bool prop_time,
    curandState_t* rand_states, uint64_t seed, uint32_t offset_per_thread,
    const NIDType* root_nodes, const TimestampType* root_timestamps,
    uint32_t snapshot_idx, uint32_t num_snapshots,
    TimestampType snapshot_time_window, uint32_t num_root_nodes,
    uint32_t fanout,
    NIDType* src_nodes, EIDType* eids, // Return Value
    TimestampType* timestamps, TimestampType* delta_timestamps, // Return Value
    uint32_t* num_sampled // Return Value
    ) {

  // tid means the id of the node (probably the thread in the future)
  uint32_t tid = 0;

  // tot_sampled means the max index of the whole sampled result
  // (e.g. the max index of <src_nodes>, <eids> etc.)
  uint32_t tot_sampled = 0;

  for(; tid < num_root_nodes; tid++) {
    NIDType nid = root_nodes[tid];
    TimestampType root_timestamp = root_timestamps[tid];
    TimestampType start_timestamp, end_timestamp;

    if(num_snapshots == 1) {
      start_timestamp = 0;
      end_timestamp = root_timestamp;
    } else {
      end_timestamp = root_timestamp -
                      (num_snapshots - snapshot_idx - 1) * snapshot_time_window;
      start_timestamp = end_timestamp - snapshot_time_window;
    }

    auto& list = node_table[nid];
    uint32_t num_candidates = 0;

    auto curr = list.head;
    int start_idx, end_idx;
    int curr_idx = 0;

    // memory each block's candidate info
    std::vector<SamplingRange*> ranges;

    while(curr != nullptr) {
      // ascending order internal, descending order external
      if(end_timestamp < curr->timestamps[0]) {
        // search in the next block
        curr = curr->next;
        curr_idx += 1;
        continue ;
      }

      if(start_timestamp > curr->timestamps[curr->size - 1]) {
        // no need to search in the next block
        break;
      }

      //search in the current block
      if(start_timestamp >= curr->timestamps[0] &&
          end_timestamp <= curr->timestamps[curr->size - 1]) {
        // all edges in the current block
        LowerBound(curr->timestamps, curr->size, start_timestamp, &start_idx);
        LowerBound(curr->timestamps, curr->size, end_timestamp, &end_idx);
      } else if (start_timestamp < curr->timestamps[0] &&
                 end_timestamp <= curr->timestamps[curr->size - 1]) {
        // only the edges before end_timestamp are in the current block
        start_idx = 0;
        LowerBound(curr->timestamps, curr->size, end_timestamp, &end_idx);
      } else if (start_timestamp > curr->timestamps[0] &&
                 end_timestamp > curr->timestamps[curr->size - 1]) {
        // only the edges after start_timestamp are in the current block
        LowerBound(curr->timestamps, curr->size, start_timestamp, &start_idx);
        end_idx = curr->size
      } else {
        // the whole block is in the range
        start_idx = 0;
        end_idx = curr->size;
      }

      // buffer the sampling range (the index is identical to <curr_idx>)
      auto sampling_range = new SamplingRange();
      sampling_range->start_idx = start_idx;
      sampling_range->end_idx = end_idx;
      ranges.push_back(sampling_range);

      // update
      num_candidates = num_candidates + (end_idx - start_idx);
      curr = curr->next;
      curr_idx = curr_idx + 1;
    }

    // indices[] contains the randomly picked positions with respect to the <num_candidates>.
    uint32_t indices[MAX_FANOUT];
    uint32_t to_sample = min(fanout, num_candidates);

    // random engine
    std::default_random_engine e;
    e.seed(time(0));

    for(uint32_t i = 0; i < to_sample; i++) {
      indices[i] = e() % num_candidates;
    }

    QuickSort(indices, 0, to_sample - 1);

    uint32_t sampled = 0;
    curr = list.head;
    curr_idx = 0;
    uint32_t cumsum = 0;
    while (curr != nullptr) {
      if (end_timestamp < curr->timestamps[0]) {
        // search in the next block
        curr = curr->next;
        curr_idx += 1;
        continue;
      }

      if (start_timestamp > curr->timestamps[curr->size - 1]) {
        // no need to search in the next block
        break;
      }

      // directly get the data from the buffer
      start_idx = ranges[curr_idx].start_idx;
      end_idx = ranges[curr_idx].end_idx;

      auto idx = indices[sampled] - cumsum;

      // use tot_sampled as index;
      while (sampled < to_sample && idx < end_idx - start_idx) {
        src_nodes[tot_sampled] = curr->dst_nodes[end_idx - idx - 1];
        eids[tot_sampled] = curr->eids[end_idx - idx - 1];
        timestamps[tot_sampled] =
            prop_time ? root_timestamp : curr->timestamps[end_idx - idx - 1];
        delta_timestamps[tot_sampled] =
            root_timestamp - curr->timestamps[end_idx - idx - 1];
        idx = indices[++sampled] - cumsum;

        // the differece between tot_sampled and sampled is:
        // sampled is a locally version while the tot_sampled is the global version
        tot_sampled ++;
      }

      if(sampled >= to_sample) {
        break;
      }

      cumsum += end_idx - start_idx;
      curr = curr->next;
      curr_idx += 1;
    }

    // update num_sampled
    num_sampled[tid] = sampled;
  }
}

}  // namespace dgnn
