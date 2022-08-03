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


// Fisher–Yates shuffle
template<class TIter>
void random_shuffle_unique(TIter begin, TIter end, size_t m) {
  size_t left = std::distance(begin, end);
  while (m--) {
    std::srand(std::time(0));
    TIter r = begin;
    std::advance(r, rand() % left);
    std::swap(*begin, *r);
    ++begin;
    --left;
  }
}

// generate a sorted randomized array with O(n)
std::vector<uint32_t> randomized_shuffle(uint32_t origin_size, uint32_t need_size) {

  std::vector<uint32_t> randomized_array;

  // cover the empty case;
  if(origin_size == 0) {
    return randomized_array;
  }

  for(uint32_t i = 0; i < origin_size; i++) {
    randomized_array.push_back(i);
  }

  // all shuffle
  random_shuffle_unique(randomized_array.begin(), randomized_array.end(), origin_size);

  // sort and truncate
  std::sort(randomized_array.begin(), randomized_array.begin() + need_size);
  randomized_array.resize(need_size);

  return randomized_array;
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

  // NID + EID + TS + Delta_TS + NUM_SAMPLED_NODES + NUM_CANDIDATES
  constexpr std::size_t per_node_size =
      sizeof(NIDType) + sizeof(EIDType) + sizeof(TimestampType) + sizeof(TimestampType)
      + sizeof(uint32_t) + sizeof(uint32_t);

  for (uint32_t i = 0; i < num_snapshots_; ++i) {
    // Double the CPU buffer to contain the data from GPU and CPU simultaneously.
    // TODO: check size
    CUDA_CALL(
        cudaMallocHost(&cpu_buffer_[i], per_node_size * maximum_sampled_nodes * 2));

    CUDA_CALL(cudaMalloc(&gpu_input_buffer_[i],
                         per_node_size * maximum_sampled_nodes));
    CUDA_CALL(cudaMalloc(&gpu_output_buffer_[i],
                         per_node_size * maximum_sampled_nodes));

    LOG(DEBUG) << "Allocated CPU buffer: "
               << maximum_sampled_nodes * per_node_size * 2 << " bytes. "
               << "Allocated GPU buffer: "
               << maximum_sampled_nodes * per_node_size << " bytes. ";

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

  // cpu sampler_node_list
  std::vector<uint32_t> cpu_num_sampled_nodes_list(num_snapshots_);
  std::vector<uint32_t> cpu_num_candidates_list(num_snapshots_);

  for (uint32_t snapshot = 0; snapshot < num_snapshots_; ++snapshot) {
    std::size_t num_root_nodes = num_root_nodes_list[snapshot];
    std::size_t max_sampled_nodes = num_root_nodes * fanouts_[layer];

    // device input
    NIDType* d_root_nodes =
        reinterpret_cast<NIDType*>(gpu_input_buffer_[snapshot]);
    TimestampType* d_root_timestamps = reinterpret_cast<TimestampType*>(
        gpu_input_buffer_[snapshot] + num_root_nodes * sizeof(NIDType));

    // device output
    std::size_t offset1 = max_sampled_nodes * sizeof(NIDType);
    std::size_t offset2 = offset1 + max_sampled_nodes * sizeof(EIDType);
    std::size_t offset3 = offset2 + max_sampled_nodes * sizeof(TimestampType);
    std::size_t offset4 = offset3 + max_sampled_nodes * sizeof(TimestampType);
    std::size_t offset5 = offset4 + max_sampled_nodes * sizeof(uint32_t); // num_candidates

    // cpu sampler input (hs = host sampler)
    NIDType* hs_root_nodes =
        reinterpret_cast<NIDType*>(cpu_buffer_[snapshot]);
    TimestampType* hs_root_timestamps = reinterpret_cast<TimestampType*>(
        cpu_buffer_[snapshot] + num_root_nodes * sizeof(NIDType));

    // cpu sampler output offset
    std::size_t offset6 = offset5 + max_sampled_nodes * sizeof(uint32_t); // NID
    std::size_t offset7 = offset6 + max_sampled_nodes * sizeof(NIDType);  // EID
    std::size_t offset8 = offset7 + max_sampled_nodes * sizeof(EIDType);  // TS
    std::size_t offset9 = offset8 + max_sampled_nodes * sizeof(TimestampType); // D_TS
    std::size_t offset10 = offset9 + max_sampled_nodes * sizeof(TimestampType); // NUM_SAMPLED
    std::size_t offset11 = offset10 + max_sampled_nodes * sizeof(uint32_t); // NUM_TIMESTAMP


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
    uint32_t* d_num_candidates =
        reinterpret_cast<uint32_t*>(gpu_output_buffer_[snapshot] + offset5);

    uint32_t num_threads_per_block = 256;
    uint32_t num_blocks =
        (num_root_nodes + num_threads_per_block - 1) / num_threads_per_block;

    if (sampling_policy_ == SamplingPolicy::kSamplingPolicyRecent) {
      SampleLayerRecentKernel<<<num_blocks, num_threads_per_block, 0,
                                streams_[snapshot]>>>(
          graph_.get_device_node_table(), graph_.num_nodes(), prop_time_,
          d_root_nodes, d_root_timestamps, snapshot, num_snapshots_,
          snapshot_time_window_, num_root_nodes, fanouts_[layer],
          d_src_nodes, d_eids, d_timestamps, d_delta_timestamps, d_num_sampled);

      // Sampling Recent Policy needs to complement will be conducted after analyzing the kernel result
      // This is an issue; cpu_buffer will be overwritten after the kernel sampling result ??

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
          num_root_nodes, fanouts_[layer],
          d_src_nodes, d_eids, d_timestamps, d_delta_timestamps, d_num_sampled, d_num_candidates);


      NIDType* hs_src_nodes =
          reinterpret_cast<NIDType*>(cpu_buffer_[snapshot] + offset6);
      EIDType* hs_eids =
          reinterpret_cast<EIDType*>(cpu_buffer_[snapshot] + offset7);
      TimestampType* hs_timestamps = reinterpret_cast<TimestampType*>(
          cpu_buffer_[snapshot] + offset8);
      TimestampType* hs_delta_timestamps = reinterpret_cast<TimestampType*>(
          cpu_buffer_[snapshot] + offset9);
      uint32_t* hs_num_sampled =
          reinterpret_cast<uint32_t*>(cpu_buffer_[snapshot] + offset10);
      uint32_t* hs_num_candidates =
          reinterpret_cast<uint32_t*>(cpu_buffer_[snapshot] + offset11);

      // CPU Uniformly Sampling
      SampleLayerUniform(graph_.get_host_node_table(), graph_.num_nodes(), prop_time_,
                         rand_states_[snapshot], seed_, offset_per_thread, hs_root_nodes,
                         hs_root_timestamps, snapshot, num_snapshots_, snapshot_time_window_,
                         num_root_nodes, fanouts_[layer],
                         hs_src_nodes, hs_eids, hs_timestamps, hs_delta_timestamps,
                         hs_num_sampled, hs_num_candidates);

      // calculate the cpu sampler's num_sampled_nodes and num_candidates
      uint32_t cpu_snapshot_num_sampled_nodes = 0;
      uint32_t cpu_snapshot_num_candidates = 0;
      for(int i = 0; i < num_root_nodes; ++i) {
        cpu_snapshot_num_sampled_nodes = cpu_snapshot_num_sampled_nodes + hs_num_sampled[i];
        cpu_snapshot_num_candidates = cpu_snapshot_num_candidates + hs_num_candidates[i];
      }

      cpu_num_sampled_nodes_list[snapshot] = cpu_snapshot_num_sampled_nodes;
      cpu_num_candidates_list[snapshot] = cpu_snapshot_num_candidates;
    }
  }

  // combine (copy the value from GPU to CPU buffer)
  std::vector<uint32_t> num_sampled_nodes_list(num_snapshots_);
  for (uint32_t snapshot = 0; snapshot < num_snapshots_; ++snapshot) {
    std::size_t num_root_nodes = num_root_nodes_list[snapshot];

    std::size_t max_sampled_nodes = num_root_nodes * fanouts_[layer];
    std::size_t offset1 = max_sampled_nodes * sizeof(NIDType);
    std::size_t offset2 = offset1 + max_sampled_nodes * sizeof(EIDType);
    std::size_t offset3 = offset2 + max_sampled_nodes * sizeof(TimestampType);
    std::size_t offset4 = offset3 + max_sampled_nodes * sizeof(TimestampType);
    std::size_t offset5 = offset4 + max_sampled_nodes * sizeof(uint32_t);

    auto d_src_nodes = reinterpret_cast<NIDType*>(gpu_output_buffer_[snapshot]);
    auto d_eids =
        reinterpret_cast<EIDType*>(gpu_output_buffer_[snapshot] + offset1);
    auto d_timestamps = reinterpret_cast<TimestampType*>(
        gpu_output_buffer_[snapshot] + offset2);
    auto d_delta_timestamps = reinterpret_cast<TimestampType*>(
        gpu_output_buffer_[snapshot] + offset3);
    uint32_t* d_num_sampled =
        reinterpret_cast<uint32_t*>(gpu_output_buffer_[snapshot] + offset4);
    uint32_t* d_num_candidates =
        reinterpret_cast<uint32_t*>(gpu_output_buffer_[snapshot] + offset5);

    auto new_end = thrust::remove_if(
        thrust::cuda::par.on(streams_[snapshot]),
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
    uint32_t* num_candidates =
        reinterpret_cast<uint32_t*>(cpu_buffer_[snapshot] + offset5);

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
    CUDA_CALL(cudaMemcpyAsync(num_candidates, d_num_candidates,
                              sizeof(uint32_t) * num_root_nodes,
                              cudaMemcpyDeviceToHost, streams_[snapshot]));
  }


  // reconstruct cpu_buffer
  for(uint32_t snapshot = 0; snapshot < num_snapshots_; ++snapshot) {

    // combine CPU sampler result
    std::size_t num_root_nodes_snapshot = num_root_nodes_list[snapshot];

    // get GPU CPU MAX sampled candidates in order
    std::size_t gpu_num_sampled_nodes = num_sampled_nodes_list[snapshot];
    std::size_t cpu_num_sampled_nodes = cpu_num_sampled_nodes_list[snapshot];
    std::size_t max_sampled_nodes = num_root_nodes_snapshot * fanouts_[layer];

    std::size_t cpu_num_candidates = cpu_num_candidates_list[snapshot];

    std::size_t per_node_size = sizeof(NIDType) + sizeof(EIDType) +
                                sizeof(TimestampType) + sizeof(TimestampType) +
                                sizeof(uint32_t) + sizeof(uint32_t);

    char* cpu_sampler_buffer = cpu_buffer_[snapshot]
                               + max_sampled_nodes * per_node_size;

    // NEW: Synchronize Memcpy
    CUDA_CALL(cudaStreamSynchronize(streams_[snapshot]));

    // Merge
    MergeHostDeviceResultByPolicy(
        cpu_buffer_[snapshot], cpu_sampler_buffer,
        gpu_num_sampled_nodes, cpu_num_sampled_nodes, max_sampled_nodes,
        cpu_num_candidates,
        num_root_nodes_snapshot, sampling_policy_);
  }


  // cpu_buffer_ -> vector<> sampling_results
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
    // TODO: prevent previous layer from re-sampling
    std::copy(prev_sampling_results.at(snapshot).all_nodes.begin(),
              prev_sampling_results.at(snapshot).all_nodes.end(),
              std::back_inserter(sampling_result.all_nodes));
    std::copy(prev_sampling_results.at(snapshot).all_timestamps.begin(),
              prev_sampling_results.at(snapshot).all_timestamps.end(),
              std::back_inserter(sampling_result.all_timestamps));

    // TODO: change it to new size
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

    // CHANGED synchronize memcpy
//    CUDA_CALL(cudaStreamSynchronize(streams_[snapshot]));

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

//    CHECK_EQ(cumsum, num_sampled_nodes);
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

// Merge by different policy (snapshot-wise)
void TemporalSampler:: MergeHostDeviceResultByPolicy(
    char* gpu_sampler_buffer_on_cpu, char* cpu_sampler_buffer,
    std::size_t gpu_num_sampled, std::size_t cpu_num_sampled, std::size_t max_num_sampled,
    std::size_t cpu_num_candidates, std::size_t num_root_nodes, SamplingPolicy policy) {

  // Offset Configuration
  std::size_t offset1 = max_num_sampled * sizeof(NIDType);
  std::size_t offset2 = offset1 + max_num_sampled * sizeof(EIDType);
  std::size_t offset3 = offset2 + max_num_sampled * sizeof(TimestampType);
  std::size_t offset4 = offset3 + max_num_sampled * sizeof(TimestampType);
  std::size_t offset5 = offset4 + max_num_sampled * sizeof(uint32_t);

  // GPU sampler position pointer cast (on CPU)
  NIDType* d_src_nodes = reinterpret_cast<NIDType*>(gpu_sampler_buffer_on_cpu);
  EIDType* d_eids = reinterpret_cast<EIDType*>(gpu_sampler_buffer_on_cpu + offset1);
  TimestampType* d_timestamps =
      reinterpret_cast<TimestampType*>(gpu_sampler_buffer_on_cpu + offset2);
  TimestampType* d_delta_timestamps =
      reinterpret_cast<TimestampType*>(gpu_sampler_buffer_on_cpu + offset3);
  uint32_t* d_num_sampled =
      reinterpret_cast<uint32_t*>(gpu_sampler_buffer_on_cpu + offset4);
  uint32_t* d_num_candidates =
      reinterpret_cast<uint32_t*>(gpu_sampler_buffer_on_cpu + offset5);

  // CPU sampler position pointer cast
  NIDType* h_src_nodes = reinterpret_cast<NIDType*>(cpu_sampler_buffer);
  EIDType* h_eids = reinterpret_cast<EIDType*>(cpu_sampler_buffer + offset1);
  TimestampType* h_timestamps =
      reinterpret_cast<TimestampType*>(cpu_sampler_buffer + offset2);
  TimestampType* h_delta_timestamps =
      reinterpret_cast<TimestampType*>(cpu_sampler_buffer + offset3);
  uint32_t* h_num_sampled =
      reinterpret_cast<uint32_t*>(cpu_sampler_buffer + offset4);
  uint32_t* h_num_candidates =
      reinterpret_cast<uint32_t*>(cpu_sampler_buffer + offset5);


  if (policy == SamplingPolicy::kSamplingPolicyRecent) {

    // If no cpu sampled or enough gpu sampled, return;
    if(cpu_num_sampled == 0 || gpu_num_sampled == max_num_sampled) {
      return ;
    }
    // fill gpu sampled
    // TODO: change with memcpy
    uint32_t cpu_sampler_offset = 0;
    uint32_t gpu_sampler_offset = gpu_num_sampled;
    while(cpu_sampler_offset < cpu_num_sampled
           && gpu_sampler_offset < max_num_sampled) {
      d_src_nodes[gpu_sampler_offset] = h_src_nodes[cpu_sampler_offset];
      d_eids[gpu_sampler_offset] = h_eids[cpu_sampler_offset];
      d_timestamps[gpu_sampler_offset] = h_timestamps[cpu_sampler_offset];
      d_delta_timestamps[gpu_sampler_offset] = h_delta_timestamps[cpu_sampler_offset];

      // num_sampled and num_candidates
      d_num_sampled[gpu_sampler_offset] = h_num_sampled[cpu_sampler_offset];
      d_num_candidates[gpu_sampler_offset] = h_num_candidates[cpu_sampler_offset];

      gpu_sampler_offset = gpu_sampler_offset + 1;
      cpu_sampler_offset = cpu_sampler_offset + 1;
    }

  } else if(policy == SamplingPolicy::kSamplingPolicyUniform) {
    // uniform (3 cases)

    // 1. Calculate gpu_num_candidates (this is why gpu_num_candidates is not in the parameter.)
    std::size_t gpu_num_candidates = 0;
    for(uint32_t i = 0; i < num_root_nodes; ++i) {
      gpu_num_candidates = gpu_num_candidates + d_num_candidates[i];
    }

    // 2. Case Judgement
    if(max_num_sampled > gpu_num_candidates && max_num_sampled > cpu_num_candidates) {

      if(gpu_num_sampled + cpu_num_sampled <= max_num_sampled) {
        // concatenate
        LOG(INFO) << "Use Concatenate Policy. "
                  << " GPU NUM SAMPLED: " << gpu_num_sampled
                  << " CPU NUM SAMPLED: " << cpu_num_sampled
                  << " MAX NUM SAMPLED: " << max_num_sampled;
        uint32_t cpu_sampler_offset = 0;
        uint32_t gpu_sampler_offset = gpu_num_sampled;
        while(cpu_sampler_offset < cpu_num_sampled
               && gpu_sampler_offset < max_num_sampled) {
          d_src_nodes[gpu_sampler_offset] = h_src_nodes[cpu_sampler_offset];
          d_eids[gpu_sampler_offset] = h_eids[cpu_sampler_offset];
          d_timestamps[gpu_sampler_offset] = h_timestamps[cpu_sampler_offset];
          d_delta_timestamps[gpu_sampler_offset] = h_delta_timestamps[cpu_sampler_offset];
          d_num_sampled[gpu_sampler_offset] = h_num_sampled[cpu_sampler_offset];
          d_num_candidates[gpu_sampler_offset] = h_num_candidates[cpu_sampler_offset];

          gpu_sampler_offset = gpu_sampler_offset + 1;
          cpu_sampler_offset = cpu_sampler_offset + 1;
        }
      } else {
        LOG(INFO) << "Use PURLY RANDOMIZED policy. "
                  << " GPU NUM SAMPLED: " << gpu_num_sampled
                  << " CPU NUM SAMPLED: " << cpu_num_sampled
                  << " MAX NUM SAMPLED: " << max_num_sampled;
        // purly randomized
        std::vector<uint32_t> randomized_array = randomized_shuffle(cpu_num_sampled + gpu_num_sampled, max_num_sampled);
        for(uint32_t current_offset = 0; current_offset < max_num_sampled; ++current_offset) {

          // get virtual index
          uint32_t theoretical_randomized_offset = randomized_array[current_offset];

          if(theoretical_randomized_offset < gpu_num_sampled) {
            // on GPU buffer
            uint32_t real_randomized_offset = theoretical_randomized_offset;

            // overwritten
            d_src_nodes[current_offset] = d_src_nodes[real_randomized_offset];
            d_eids[current_offset] = d_eids[real_randomized_offset];
            d_timestamps[current_offset] = d_timestamps[real_randomized_offset];
            d_delta_timestamps[current_offset] = d_delta_timestamps[real_randomized_offset];
            d_num_sampled[current_offset] = d_num_sampled[real_randomized_offset];
            d_num_candidates[current_offset] = d_num_candidates[real_randomized_offset];

          } else {
            // on CPU buffer
            uint32_t real_randomized_offset = theoretical_randomized_offset - gpu_num_sampled;

            // overwritten
            d_src_nodes[current_offset] = h_src_nodes[real_randomized_offset];
            d_eids[current_offset] = h_eids[real_randomized_offset];
            d_timestamps[current_offset] = h_timestamps[real_randomized_offset];
            d_delta_timestamps[current_offset] = h_delta_timestamps[real_randomized_offset];
            d_num_sampled[current_offset] = h_num_sampled[real_randomized_offset];
            d_num_candidates[current_offset] = h_num_candidates[real_randomized_offset];
          }
        }
      }

    } else {
      LOG(INFO) << "Use RANDOMIZED RATIO policy. "
                << " GPU NUM SAMPLED: " << gpu_num_sampled
                << " CPU NUM SAMPLED: " << cpu_num_sampled
                << " MAX NUM SAMPLED: " << max_num_sampled;

      // randomized with ratio (gpu_sampler * alpha + cpu_sampler * (1 - alpha))
      // alpha = gpu_num_candidates / (gpu_num_candidates + cpu_num_candidates)
      double alpha = (double) gpu_num_candidates / (double) (gpu_num_candidates + cpu_num_candidates);

      uint32_t gpu_actual_size = (uint32_t) std::round(( ( (double) max_num_sampled ) * alpha ));
      uint32_t cpu_actual_size = (uint32_t) std::round((( (double) max_num_sampled ) * (1.00 - alpha )));

      uint32_t actual_size = std::min(max_num_sampled, (std::size_t) (gpu_actual_size + cpu_actual_size));

      std::vector<uint32_t> gpu_randomized_array = randomized_shuffle(gpu_num_sampled, gpu_actual_size);
      std::vector<uint32_t> cpu_randomized_array = randomized_shuffle(cpu_num_sampled, cpu_actual_size);

      uint32_t current_offset = 0;

      for(auto gpu_iter = gpu_randomized_array.begin(); gpu_iter != gpu_randomized_array.end(); gpu_iter++) {

        if(current_offset >= actual_size) {
          break;
        }

        uint32_t gpu_idx = *gpu_iter;

        // overwritten
        d_src_nodes[current_offset] = d_src_nodes[gpu_idx];
        d_eids[current_offset] = d_eids[gpu_idx];
        d_timestamps[current_offset] = d_timestamps[gpu_idx];
        d_delta_timestamps[current_offset] = d_delta_timestamps[gpu_idx];
        d_num_sampled[current_offset] = d_num_sampled[gpu_idx];
        d_num_candidates[current_offset] = d_num_candidates[gpu_idx];

        current_offset = current_offset + 1;
      }

      for(auto cpu_iter = cpu_randomized_array.begin(); cpu_iter != cpu_randomized_array.end(); cpu_iter++) {

        if(current_offset >= actual_size) {
          break;
        }

        uint32_t cpu_idx = *cpu_iter;

        // overwritten
//        std::memcpy(d_src_nodes + current_offset, h_src_nodes + cpu_idx, sizeof(NIDType));
//        std::memcpy(d_eids + current_offset, h_eids + cpu_idx, sizeof(EIDType));
//        std::memcpy(d_timestamps + current_offset, h_timestamps + cpu_idx, sizeof(TimestampType));
//        std::memcpy(d_delta_timestamps + current_offset, h_delta_timestamps + cpu_idx, sizeof(TimestampType));
//        std::memcpy(d_num_sampled + current_offset, h_num_sampled + cpu_idx, sizeof(uint32_t));
//        std::memcpy(d_num_candidates + current_offset, h_num_candidates + cpu_idx, sizeof(uint32_t));

        d_src_nodes[current_offset] = h_src_nodes[cpu_idx];
        d_eids[current_offset] = h_eids[cpu_idx];
        d_timestamps[current_offset] = h_timestamps[cpu_idx];
        d_delta_timestamps[current_offset] = h_delta_timestamps[cpu_idx];
//        d_num_sampled[current_offset] = h_num_sampled[cpu_idx];
        d_num_candidates[current_offset] = h_num_candidates[cpu_idx];

        current_offset = current_offset + 1;
      }
      LOG(INFO)<<"current_offset: "<< current_offset;
    }
  }

  return ;
}

// Recent Snapshot
void TemporalSampler::SampleLayerRecent(
    const DoublyLinkedList* node_table, std::size_t num_nodes, bool prop_time,
    const NIDType* root_nodes, const TimestampType* root_timestamps,
    uint32_t snapshot_idx, uint32_t num_snapshots,
    TimestampType snapshot_time_window, uint32_t num_root_nodes,
    uint32_t fanout,
    NIDType* src_nodes, EIDType* eid, // Return Value
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
        eid[tot_sampled] = curr->eids[i];
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
void TemporalSampler::SampleLayerUniform(
    const DoublyLinkedList* node_table, std::size_t num_nodes, bool prop_time,
    curandState_t* rand_states, uint64_t seed, uint32_t offset_per_thread,
    const NIDType* root_nodes, const TimestampType* root_timestamps,
    uint32_t snapshot_idx, uint32_t num_snapshots,
    TimestampType snapshot_time_window, uint32_t num_root_nodes,
    uint32_t fanout,
    NIDType* src_nodes, EIDType* eids, // Return Value
    TimestampType* timestamps, TimestampType* delta_timestamps, // Return Value
    uint32_t* num_sampled_arr, // Return Value
    uint32_t* num_candidates_arr // Return Value
    ) {

  // TODO: multithread optimization
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
    std::vector<SamplingRange> ranges;

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
        end_idx = curr->size;
      } else {
        // the whole block is in the range
        start_idx = 0;
        end_idx = curr->size;
      }

      // buffer the sampling range (the index is identical to <curr_idx>)
      SamplingRange sampling_range;
      sampling_range.start_idx = start_idx;
      sampling_range.end_idx = end_idx;
      ranges.push_back(sampling_range);

      // update
      num_candidates = num_candidates + (end_idx - start_idx);
      curr = curr->next;
      curr_idx = curr_idx + 1;
    }

    // record the total of num_candidates
    num_candidates_arr[tid] = num_candidates;

    // indices[] contains the randomly picked positions with respect to the <num_candidates>.
    uint32_t indices[kMaxFanout];
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

    num_sampled_arr[tid] = sampled;

  }
}

}  // namespace dgnn
