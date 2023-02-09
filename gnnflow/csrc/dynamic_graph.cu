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
#include <vector>

#include "common.h"
#include "dynamic_graph.h"
#include "logging.h"
#include "utils.h"

namespace gnnflow {

DynamicGraph::DynamicGraph(std::size_t initial_pool_size,
                           std::size_t maximum_pool_size,
                           MemoryResourceType mem_resource_type,
                           std::size_t minium_block_size,
                           std::size_t blocks_to_preallocate,
                           InsertionPolicy insertion_policy, int device,
                           bool adaptive_block_size)
    : allocator_(initial_pool_size, maximum_pool_size, minium_block_size,
                 mem_resource_type, device),
      insertion_policy_(insertion_policy),
      max_node_id_(0),
      device_(device),
      adaptive_block_size_(adaptive_block_size) {
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

void DynamicGraph::AddEdges(const std::vector<NIDType>& src_nodes,
                            const std::vector<NIDType>& dst_nodes,
                            const std::vector<TimestampType>& timestamps,
                            const std::vector<EIDType>& eids) {
  CHECK_GT(src_nodes.size(), 0);
  CHECK_EQ(src_nodes.size(), dst_nodes.size());
  CHECK_EQ(src_nodes.size(), timestamps.size());
  CHECK_EQ(src_nodes.size(), eids.size());

  // NB: it seems to be necessary to set the device again.
  CUDA_CALL(cudaSetDevice(device_));

  src_nodes_.insert(src_nodes.begin(), src_nodes.end());
  nodes_.insert(src_nodes.begin(), src_nodes.end());
  nodes_.insert(dst_nodes.begin(), dst_nodes.end());

  for (auto eid : eids) {
    edges_[eid]++;
  }

  nodes_.insert(src_nodes.begin(), src_nodes.end());

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
  if (max_node < max_node_id_) {
    return;
  }
  max_node_id_ = max_node;
  d_node_table_.resize(max_node_id_ + 1);
  h_copy_of_d_node_table_.resize(max_node_id_ + 1);
}

std::size_t DynamicGraph::num_nodes() const { return nodes_.size(); }
std::size_t DynamicGraph::num_src_nodes() const { return src_nodes_.size(); }
std::size_t DynamicGraph::num_edges() const { return edges_.size(); }

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

void DynamicGraph::RemoveBlock(NIDType node_id, TemporalBlock* block,
                               cudaStream_t stream) {
  CHECK_NOTNULL(block);
  // host
  RemoveBlockFromDoublyLinkedList(h_copy_of_d_node_table_.data(), node_id,
                                  block);

  // device
  auto d_block = h2d_mapping_[block];
  RemoveBlockFromDoublyLinkedListKernel<<<1, 1, 0, stream>>>(
      thrust::raw_pointer_cast(d_node_table_.data()), node_id, d_block);

  // release the memory
  auto mr = rmm::mr::get_current_device_resource();
  mr->deallocate(d_block, sizeof(TemporalBlock));

  // update the mapping
  h2d_mapping_.erase(block);
}

void DynamicGraph::SyncBlock(TemporalBlock* block, cudaStream_t stream) {
  // copy the metadata from the host to the device
  CUDA_CALL(cudaMemcpyAsync(h2d_mapping_[block], block, 48,
                            cudaMemcpyHostToDevice, stream));
}

inline std::size_t get_next_power_of_two(std::size_t n) {
  return 1 << (64 - __builtin_clzl(n - 1));
}

void DynamicGraph::AddEdgesForOneNode(
    NIDType src_node, const std::vector<NIDType>& dst_nodes,
    const std::vector<TimestampType>& timestamps,
    const std::vector<EIDType>& eids, cudaStream_t stream) {
  std::size_t num_edges = dst_nodes.size();

  // NB: reference is necessary here since the value is updated in
  // `InsertBlock`
  auto& h_list = h_copy_of_d_node_table_[src_node];
  auto& h_tail_block = h_list.tail;
  TemporalBlock* h_block = nullptr;
  bool is_new_block = false;

  std::size_t start_idx = 0;
  if (h_tail_block == nullptr) {
    // case 1: empty list
    h_block = allocator_.Allocate(num_edges);
    is_new_block = true;
  } else if (h_tail_block->size + num_edges > h_tail_block->capacity) {
    // case 2: not enough space in the current block
    if (insertion_policy_ == InsertionPolicy::kInsertionPolicyInsert) {
      // make the current block full
      std::size_t num_edges_to_current_block =
          h_tail_block->capacity - h_tail_block->size;
      if (num_edges_to_current_block > 0) {
        CopyEdgesToBlock(h_tail_block, dst_nodes, timestamps, eids, 0,
                         num_edges_to_current_block, device_, stream);
        // NB: sync is necessary here since the value is updated in
        // `InsertBlock` right after the copy
        SyncBlock(h_tail_block, stream);

        start_idx = num_edges_to_current_block;
        num_edges -= num_edges_to_current_block;
      }

      // allocate and insert a new block
      // calculate avg edge per insertion
      std::size_t avg_edges_per_insertion;
      if (h_list.num_insertions == 0) {
        avg_edges_per_insertion = num_edges;
      } else {
        avg_edges_per_insertion = h_list.num_edges / h_list.num_insertions;
      }

      std::size_t new_block_size;
      if (adaptive_block_size_) {
        new_block_size = std::max(num_edges, avg_edges_per_insertion);
        // round up to the nearest power of 2
        new_block_size = get_next_power_of_two(new_block_size);
      } else {
        new_block_size = num_edges;
      }

      h_block = allocator_.Allocate(new_block_size);
      is_new_block = true;
    } else {
      // reallocate the block
      // NB: the pointer to the block is not changed, only the content of
      // the block (e.g., pointers, capcacity) is updated
      allocator_.Reallocate(h_tail_block, h_tail_block->size + num_edges,
                            stream);
    }
  }

  if (!is_new_block) {
    // case 3: there is enough space in the current block
    h_block = h_tail_block;
  }

  // copy data to block
  CopyEdgesToBlock(h_block, dst_nodes, timestamps, eids, start_idx, num_edges,
                   device_, stream);

  if (is_new_block) {
    InsertBlock(src_node, h_block, stream);
  }
  SyncBlock(h_block, stream);

  // update the number of edges
  h_list.num_edges += dst_nodes.size();
  h_list.num_insertions++;
}

std::vector<std::size_t> DynamicGraph::out_degree(
    const std::vector<NIDType>& nodes) const {
  std::vector<size_t> out_degrees;
  for (auto& node : nodes) {
    auto h_list = h_copy_of_d_node_table_[node];
    out_degrees.push_back(h_list.num_edges);
  }
  return out_degrees;
}

DynamicGraph::NodeNeighborTuple DynamicGraph::get_temporal_neighbors(
    NIDType node) const {
  NodeNeighborTuple result;
  {
    // NB: reference is necessary
    auto& list = h_copy_of_d_node_table_[node];
    auto block = list.tail;
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
                                 std::rbegin(dst_nodes), std::rend(dst_nodes));
      std::get<1>(result).insert(std::end(std::get<1>(result)),
                                 std::rbegin(timestamps),
                                 std::rend(timestamps));
      std::get<2>(result).insert(std::end(std::get<2>(result)),
                                 std::rbegin(eids), std::rend(eids));

      block = block->prev;
    }
  }

  return result;
}

const DoublyLinkedList* DynamicGraph::get_device_node_table() const {
  return thrust::raw_pointer_cast(d_node_table_.data());
}

std::vector<NIDType> DynamicGraph::nodes() const {
  return {nodes_.begin(), nodes_.end()};
}
std::vector<NIDType> DynamicGraph::src_nodes() const {
  return {src_nodes_.begin(), src_nodes_.end()};
}
std::vector<EIDType> DynamicGraph::edges() const {
  std::vector<EIDType> keys;
  for (auto kv : edges_) {
    keys.push_back(kv.first);
  }
  return keys;
}

NIDType DynamicGraph::max_node_id() const { return max_node_id_; }

float DynamicGraph::avg_linked_list_length() const {
  float sum = 0;
  for (auto& node : nodes_) {
    auto& list = h_copy_of_d_node_table_[node];
    sum += list.size;
  }
  return sum / nodes_.size();
}

float DynamicGraph::graph_mem_usage() const {
  return allocator_.get_total_memory_usage();
}

float DynamicGraph::graph_metadata_mem_usage() {
  float sum = 0;
  // num blocks
  sum += sizeof(TemporalBlock) * h2d_mapping_.size();
  // node table
  d_node_table_.shrink_to_fit();
  sum += sizeof(DoublyLinkedList) * d_node_table_.capacity();
  return sum;
}

std::size_t DynamicGraph::OffloadOldBlocks(TimestampType timestamp,
                                           bool to_file) {
  std::size_t num_blocks = 0;
  for (auto& node : nodes_) {
    auto& list = h_copy_of_d_node_table_[node];
    auto cur = list.head;  // the oldest block for the node
    while (cur != nullptr) {
      auto next = cur->next;
      if (cur->end_timestamp < timestamp) {
        // remove from `edges_`
        for (auto i = 0; i < cur->size; i++) {
          auto eid = cur->eids[i];
          if (--edges_[eid] == 0) {
            edges_.erase(eid);
          }
        }

        RemoveBlock(node, cur);
        if (to_file) {
          allocator_.SaveToFile(cur, node);
        } else {
          allocator_.Deallocate(cur);  // `delete block`
        }
        num_blocks++;
      }
      cur = next;
    }
  }
  return num_blocks;
}
}  // namespace gnnflow
