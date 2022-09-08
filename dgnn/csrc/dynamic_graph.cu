#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_delete.h>
#include <thrust/device_new.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <rmm/detail/error.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/fixed_size_memory_resource.hpp>
#include <rmm/mr/device/logging_resource_adaptor.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <type_traits>

#include "common.h"
#include "dynamic_graph.h"
#include "logging.h"
#include "utils.h"

namespace dgnn {

DynamicGraph::DynamicGraph(std::size_t initial_pool_size,
                           std::size_t maximum_pool_size,
                           MemoryResourceType mem_resource_type,
                           std::size_t minium_block_size,
                           std::size_t blocks_to_preallocate,
                           InsertionPolicy insertion_policy, int device)
    : allocator_(initial_pool_size, maximum_pool_size, minium_block_size,
                 mem_resource_type, device),
      insertion_policy_(insertion_policy),
      num_nodes_(0),
      num_edges_(0),
      device_(device) {
  for (int i = 0; i < kNumStreams; i++) {
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    streams_.push_back(stream);
  }

  auto mem_res = new rmm::mr::cuda_memory_resource();
  mem_resources_for_metadata_.push(mem_res);
  auto pool_res =
      new rmm::mr::fixed_size_memory_resource<rmm::mr::cuda_memory_resource>(
          mem_res, sizeof(TemporalBlock), blocks_to_preallocate);
  mem_resources_for_metadata_.push(pool_res);

#ifdef DGNN_DEBUG
  auto logging_res = new rmm::mr::logging_resource_adaptor<
      std::remove_reference_t<decltype(*pool_res)>>(pool_res, "rmm_log.txt",
                                                    true);
  mem_resources_for_metadata_.push(logging_res);
#endif

  rmm::mr::set_current_device_resource(mem_resources_for_metadata_.top());
}

DynamicGraph::~DynamicGraph() {
  for (auto& stream : streams_) {
    cudaStreamDestroy(stream);
  }

  // release the memory of node table
  d_node_table_.clear();
  d_node_table_.shrink_to_fit();

  // release the memory of blocks
  auto mr = rmm::mr::get_current_device_resource();
  for (auto iter = std::begin(h2d_mapping_); iter != std::end(h2d_mapping_);
       ++iter) {
    auto& block = iter->second;
    mr->deallocate(block, sizeof(TemporalBlock));
  }

  // release the memory pool
  while (!mem_resources_for_metadata_.empty()) {
    delete mem_resources_for_metadata_.top();
    mem_resources_for_metadata_.pop();
  }
}

void DynamicGraph::AddEdges(std::vector<NIDType>& src_nodes,
                            std::vector<NIDType>& dst_nodes,
                            std::vector<TimestampType>& timestamps,
                            bool add_reverse_edges) {
  CHECK_GT(src_nodes.size(), 0);
  CHECK_EQ(src_nodes.size(), dst_nodes.size());
  CHECK_EQ(src_nodes.size(), timestamps.size());

  std::vector<EIDType> eids(src_nodes.size());
  std::iota(eids.begin(), eids.end(), num_edges_);
  // NB: num_edges_ does not count the reverse edges
  num_edges_ += eids.size();

  // for undirected graphs, we need to add the reverse edges
  if (add_reverse_edges) {
    src_nodes.insert(src_nodes.end(), dst_nodes.begin(), dst_nodes.end());
    dst_nodes.insert(dst_nodes.end(), src_nodes.begin(),
                     src_nodes.begin() + dst_nodes.size());
    timestamps.insert(timestamps.end(), timestamps.begin(), timestamps.end());
    eids.insert(eids.end(), eids.begin(), eids.end());
  }

  // add nodes
  NIDType max_node =
      std::max(*std::max_element(src_nodes.begin(), src_nodes.end()),
               *std::max_element(dst_nodes.begin(), dst_nodes.end()));
  AddNodes(max_node);

  std::unordered_map<NIDType, std::vector<NIDType>> src_to_dst_map;
  std::unordered_map<NIDType, std::vector<TimestampType>> src_to_ts_map;
  std::unordered_map<NIDType, std::vector<EIDType>> src_to_eid_map;

  for (std::size_t i = 0; i < src_nodes.size(); ++i) {
    src_to_dst_map[src_nodes[i]].push_back(dst_nodes[i]);
    src_to_ts_map[src_nodes[i]].push_back(timestamps[i]);
    src_to_eid_map[src_nodes[i]].push_back(eids[i]);
  }

  int i = 0;
  for (auto iter = std::begin(src_to_dst_map); iter != std::end(src_to_dst_map);
       iter++) {
    NIDType src_node = iter->first;
    auto& dst_nodes = iter->second;
    auto& timestamps = src_to_ts_map[src_node];
    auto& eids = src_to_eid_map[src_node];

    // sort the edges by timestamp
    auto idx = stable_sort_indices(timestamps);

    dst_nodes = sort_vector(dst_nodes, idx);
    timestamps = sort_vector(timestamps, idx);
    eids = sort_vector(eids, idx);

    AddEdgesForOneNode(src_node, dst_nodes, timestamps, eids,
                       streams_[i % kNumStreams]);
    i++;
  }

  for (auto& stream : streams_) {
    CUDA_CALL(cudaStreamSynchronize(stream));
  }
}

void DynamicGraph::AddNodes(NIDType max_node) {
  if (max_node < num_nodes_) {
    return;
  }
  num_nodes_ = max_node + 1;
  d_node_table_.resize(num_nodes_);
  h_copy_of_d_node_table_.resize(num_nodes_);
}

std::size_t DynamicGraph::num_nodes() const { return num_nodes_; }

std::size_t DynamicGraph::num_edges() const { return num_edges_; }

void DynamicGraph::InsertBlock(NIDType node_id, TemporalBlock* block,
                               cudaStream_t stream) {
  CHECK_NOTNULL(block);
  // host
  InsertBlockToDoublyLinkedList(h_copy_of_d_node_table_.data(), node_id, block);

  // allocate a block on the device
  TemporalBlock* d_block = nullptr;
  try {
    auto mr = rmm::mr::get_current_device_resource();
    d_block = static_cast<TemporalBlock*>(mr->allocate(sizeof(TemporalBlock)));
  } catch (rmm::bad_alloc&) {
    LOG(FATAL) << "Failed to allocate memory for temporal block";
  }

  // insert the block into the linked list
  InsertBlockToDoublyLinkedListKernel<<<1, 1, 0, stream>>>(
      thrust::raw_pointer_cast(d_node_table_.data()), node_id, d_block);

  // update the mapping
  h2d_mapping_[block] = d_block;
}

void DynamicGraph::SyncBlock(TemporalBlock* block, cudaStream_t stream) {
  // copy the metadata from the host to the device
  CUDA_CALL(cudaMemcpyAsync(h2d_mapping_[block], block, 48,
                            cudaMemcpyHostToDevice, stream));
}

void DynamicGraph::AddEdgesForOneNode(
    NIDType src_node, const std::vector<NIDType>& dst_nodes,
    const std::vector<TimestampType>& timestamps,
    const std::vector<EIDType>& eids, cudaStream_t stream) {
  std::size_t num_edges = dst_nodes.size();

  // NB: reference is necessary here since the value is updated in
  // `InsertBlock`
  auto& h_tail_block = h_copy_of_d_node_table_[src_node].tail;

  std::size_t start_idx = 0;
  if (h_tail_block == nullptr) {
    // case 1: empty list
    auto block = allocator_.Allocate(num_edges);
    InsertBlock(src_node, block, stream);
  } else if (h_tail_block->size + num_edges > h_tail_block->capacity) {
    // case 2: not enough space in the current block
    if (insertion_policy_ == InsertionPolicy::kInsertionPolicyInsert) {
      // make the current block full
      std::size_t num_edges_to_current_block =
          h_tail_block->capacity - h_tail_block->size;
      if (num_edges_to_current_block > 0) {
        CopyEdgesToBlock(h_tail_block, dst_nodes, timestamps, eids, 0,
                         num_edges_to_current_block, stream);
        // NB: sync is necessary here since the value is updated in
        // `InsertBlock` right after the copy
        SyncBlock(h_tail_block, stream);

        start_idx = num_edges_to_current_block;
        num_edges -= num_edges_to_current_block;
      }

      // allocate and insert a new block
      auto new_block = allocator_.Allocate(num_edges);
      InsertBlock(src_node, new_block, stream);
    } else {
      // reallocate the block
      // NB: the pointer to the block is not changed, only the content of
      // the block (e.g., pointers, capcacity) is updated
      allocator_.Reallocate(h_tail_block, h_tail_block->size + num_edges,
                            stream);
    }
  }

  // copy data to block
  CopyEdgesToBlock(h_tail_block, dst_nodes, timestamps, eids, start_idx,
                   num_edges, stream);
  SyncBlock(h_tail_block, stream);
}

std::size_t DynamicGraph::out_degree(NIDType node) const {
  size_t out_degree = 0;
  {
    auto& list = h_copy_of_d_node_table_[node];
    auto block = list.head;
    while (block != nullptr) {
      out_degree += block->size;
      block = block->next;
    }
  }

  return out_degree;
}

DynamicGraph::NodeNeighborTuple DynamicGraph::get_temporal_neighbors(
    NIDType node) const {
  NodeNeighborTuple result;
  {
    // NB: reference is necessary
    auto& list = h_copy_of_d_node_table_[node];
    auto block = list.head;
    while (block != nullptr) {
      std::vector<NIDType> dst_nodes(block->size);
      std::vector<TimestampType> timestamps(block->size);
      std::vector<EIDType> eids(block->size);

      thrust::copy(thrust::device_ptr<NIDType>(block->dst_nodes),
                   thrust::device_ptr<NIDType>(block->dst_nodes) + block->size,
                   dst_nodes.begin());

      thrust::copy(
          thrust::device_ptr<TimestampType>(block->timestamps),
          thrust::device_ptr<TimestampType>(block->timestamps) + block->size,
          timestamps.begin());

      thrust::copy(thrust::device_ptr<EIDType>(block->eids),
                   thrust::device_ptr<EIDType>(block->eids) + block->size,
                   eids.begin());

      std::get<0>(result).insert(std::end(std::get<0>(result)),
                                 std::begin(dst_nodes), std::end(dst_nodes));
      std::get<1>(result).insert(std::end(std::get<1>(result)),
                                 std::begin(timestamps), std::end(timestamps));
      std::get<2>(result).insert(std::end(std::get<2>(result)),
                                 std::begin(eids), std::end(eids));

      block = block->next;
    }
  }
  // reverse the order of the result
  std::reverse(std::get<0>(result).begin(), std::get<0>(result).end());
  std::reverse(std::get<1>(result).begin(), std::get<1>(result).end());
  std::reverse(std::get<2>(result).begin(), std::get<2>(result).end());

  return result;
}

const DoublyLinkedList* DynamicGraph::get_device_node_table() const {
  return thrust::raw_pointer_cast(d_node_table_.data());
}

}  // namespace dgnn
