#include <thrust/execution_policy.h>
#include <thrust/remove.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
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
      initialized_(false) {
  if (num_snapshots_ == 1 && std::fabs(snapshot_time_window_) > 0.0f) {
    LOG(WARNING) << "Snapshot time window must be 0 when num_snapshots = 1. "
                    "Ignore the snapshot time window.";
  }
  shared_memory_size_ = GetSharedMemoryMaxSize();

  streams_ = new cudaStream_t[num_snapshots_];
  for (uint32_t i = 0; i < num_snapshots_; ++i) {
    CUDA_CALL(cudaStreamCreate(&streams_[i]));
  }

  cpu_buffer_ = new char*[num_snapshots_];
  gpu_input_buffer_ = new char*[num_snapshots_];
  gpu_output_buffer_ = new char*[num_snapshots_];
  rand_states_ = new curandState_t*[num_snapshots_];
}

TemporalSampler::~TemporalSampler() {
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
    CUDA_CALL(
        cudaMallocHost(&cpu_buffer_[i], per_node_size * maximum_sampled_nodes));

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

  initialized_ = true;
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

    //    thrust::copy(thrust::cuda::par.on(streams_[snapshot]),
    //                 thrust::make_zip_iterator(thrust::make_tuple(
    //                     d_src_nodes, d_eids, d_timestamps,
    //                     d_delta_timestamps)),
    //                 new_end,
    //                 thrust::make_zip_iterator(thrust::make_tuple(
    //                     src_nodes, eids, timestamps, delta_timestamps)));
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
  }

  return sampling_results;
}

std::vector<std::vector<SamplingResult>> TemporalSampler::Sample(
    const std::vector<NIDType>& dst_nodes,
    const std::vector<TimestampType>& timestamps) {
  CHECK_EQ(dst_nodes.size(), timestamps.size());
  std::vector<std::vector<SamplingResult>> results;

  if (initialized_ == false) InitBuffer(dst_nodes.size());

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
}  // namespace dgnn
