#include <thrust/copy.h>
#include <thrust/device_delete.h>
#include <thrust/device_new.h>

#include <algorithm>
#include <rmm/detail/error.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include "dynamic_graph.h"
#include "logging.h"
#include "utils.h"

namespace dgnn {

DynamicGraph::DynamicGraph(std::size_t max_gpu_mem_pool_size,
                           std::size_t alignment,
                           InsertionPolicy insertion_policy)
    : allocator_(max_gpu_mem_pool_size, alignment),
      insertion_policy_(insertion_policy),
      num_nodes_(0),
      num_edges_(0) {}

void DynamicGraph::AddEdges(std::vector<NIDType>& src_nodes,
                            std::vector<NIDType>& dst_nodes,
                            std::vector<TimestampType>& timestamps,
                            bool add_reverse_edges) {
  CHECK_GT(src_nodes.size(), 0);
  CHECK_EQ(src_nodes.size(), dst_nodes.size());
  CHECK_EQ(src_nodes.size(), timestamps.size());

  std::vector<EIDType> eids(src_nodes.size());
  std::iota(eids.begin(), eids.end(), num_edges_);

  // for undirected graphs, we need to add the reverse edges
  if (add_reverse_edges) {
    src_nodes.insert(src_nodes.end(), dst_nodes.begin(), dst_nodes.end());
    dst_nodes.insert(dst_nodes.end(), src_nodes.begin(), src_nodes.end());
    timestamps.insert(timestamps.end(), timestamps.begin(), timestamps.end());
    eids.insert(eids.end(), eids.begin(), eids.end());
  }
  num_edges_ += eids.size();

  // add nodes
  NIDType max_node =
      std::max(*std::max_element(src_nodes.begin(), src_nodes.end()),
               *std::max_element(dst_nodes.begin(), dst_nodes.end()));
  AddNodes(max_node);

  std::map<NIDType, std::vector<NIDType>> src_to_dst_map;
  std::map<NIDType, std::vector<TimestampType>> src_to_ts_map;
  std::map<NIDType, std::vector<EIDType>> src_to_eid_map;
  for (std::size_t i = 0; i < src_nodes.size(); ++i) {
    src_to_dst_map[src_nodes[i]].push_back(dst_nodes[i]);
    src_to_ts_map[src_nodes[i]].push_back(timestamps[i]);
    src_to_eid_map[src_nodes[i]].push_back(eids[i]);
  }

  for (auto& src_to_dst : src_to_dst_map) {
    NIDType src_node = src_to_dst.first;
    auto& dst_nodes = src_to_dst.second;
    auto& timestamps = src_to_ts_map[src_node];
    auto& eids = src_to_eid_map[src_node];

    // sort the edges by timestamp
    auto idx = stable_sort_indices(timestamps);

    dst_nodes = sort_vector(dst_nodes, idx);
    timestamps = sort_vector(timestamps, idx);
    eids = sort_vector(eids, idx);

    AddEdgesForOneNode(src_node, dst_nodes, timestamps, eids);
  }
}

void DynamicGraph::AddNodes(NIDType max_node) {
  if (max_node < num_nodes_) {
    return;
  }
  num_nodes_ = max_node + 1;
  d_node_table_.resize(num_nodes_);
  h_node_table_.resize(num_nodes_);
  h_copy_of_d_node_table_.resize(num_nodes_);
}

std::size_t DynamicGraph::num_nodes() const { return num_nodes_; }

std::size_t DynamicGraph::num_edges() const { return num_edges_; }

TemporalBlock* DynamicGraph::AllocateBlock(std::size_t num_edges) {
  TemporalBlock* block;
  try {
    block = allocator_.Allocate(num_edges);
  } catch (rmm::bad_alloc) {
    // if we can't allocate the block, we need to free some memory
    std::size_t min_swap_size = allocator_.AlignUp(num_edges);
    auto swapped_size = SwapOldBlocksToCPU(min_swap_size);
    LOG(INFO) << "Swapped " << swapped_size << " bytes to CPU";

    // try again
    block = allocator_.Allocate(num_edges);
  }

  return block;
}

TemporalBlock* DynamicGraph::ReallocateBlock(TemporalBlock* block,
                                             std::size_t num_edges) {
  CHECK_NOTNULL(block);
  auto new_block = AllocateBlock(num_edges);

  CopyTemporalBlock(block, new_block);

  // release the old block
  allocator_.Deallocate(block);

  return new_block;
}

void DynamicGraph::InsertBlock(NIDType node_id, TemporalBlock* block) {
  CHECK_NOTNULL(block);
  // host
  InsertBlockToDoublyLinkedList(h_copy_of_d_node_table_.data(), node_id, block);

  // device
  thrust::device_ptr<TemporalBlock> d_block =
      thrust::device_new<TemporalBlock>(1);
  *d_block = *block;

  InsertBlockToDoublyLinkedListKernel<<<1, 1>>>(
      thrust::raw_pointer_cast(d_node_table_.data()), node_id, d_block.get());
  CUDA_CALL(cudaDeviceSynchronize());

  // mapping
  h2d_mapping_[block] = d_block;
}

void DynamicGraph::DeleteTailBlock(NIDType node_id) {
  // host
  auto tail = h_copy_of_d_node_table_[node_id].tail.prev;
  CHECK_NE(tail, &h_copy_of_d_node_table_[node_id].head);

  DeleteTailFromDoublyLinkedList(h_copy_of_d_node_table_.data(), node_id);

  // device
  DeleteTailFromDoublyLinkedListKernel<<<1, 1>>>(
      thrust::raw_pointer_cast(d_node_table_.data()), node_id);
  CUDA_CALL(cudaDeviceSynchronize());

  // delete
  thrust::device_delete(h2d_mapping_[tail]);
  h2d_mapping_.erase(tail);
  delete tail;
}

void DynamicGraph::ReplaceBlock(NIDType node_id, TemporalBlock* block) {
  CHECK_NOTNULL(block);
  // host
  auto old_block = h_copy_of_d_node_table_[node_id].head.next;
  ReplaceBlockInDoublyLinkedList(h_copy_of_d_node_table_.data(), node_id,
                                 block);

  // device
  thrust::device_ptr<TemporalBlock> d_block =
      thrust::device_new<TemporalBlock>(1);
  *d_block = *block;

  ReplaceBlockInDoublyLinkedListKernel<<<1, 1>>>(
      thrust::raw_pointer_cast(d_node_table_.data()), node_id, d_block.get());
  CUDA_CALL(cudaDeviceSynchronize());

  // delete
  thrust::device_delete(h2d_mapping_[old_block]);
  h2d_mapping_.erase(old_block);
  delete old_block;
}

void DynamicGraph::SyncBlock(TemporalBlock* block) {
  CHECK_NE(h2d_mapping_.find(block), h2d_mapping_.end());
  *h2d_mapping_[block] = *block;
}

void DynamicGraph::AddEdgesForOneNode(
    NIDType src_node, const std::vector<NIDType>& dst_nodes,
    const std::vector<TimestampType>& timestamps,
    const std::vector<EIDType>& eids) {
  std::size_t num_edges = dst_nodes.size();

  auto& head = h_copy_of_d_node_table_[src_node].head;
  auto& tail = h_copy_of_d_node_table_[src_node].tail;

  std::size_t start_idx = 0;
  if (head.next == &tail) {
    // empty list
    auto block = AllocateBlock(num_edges);
    InsertBlock(src_node, block);
  } else if (head.next->size + num_edges > head.next->capacity) {
    // not enough space in the current block
    if (insertion_policy_ == InsertionPolicy::kInsertionPolicyInsert) {
      // copy some to existing block
      std::size_t num_edges_to_existing_block =
          head.next->capacity - head.next->size;
      CopyEdgesToBlock(head.next, dst_nodes, timestamps, eids, 0,
                       num_edges_to_existing_block);
      SyncBlock(head.next);
      start_idx = num_edges_to_existing_block;

      num_edges -= num_edges_to_existing_block;
      // insert new block
      auto new_block = AllocateBlock(num_edges);
      InsertBlock(src_node, new_block);
    } else {
      // reallocate block
      auto new_block = ReallocateBlock(head.next, num_edges);
      ReplaceBlock(src_node, new_block);
    }
  }

  auto block = head.next;

  // copy data to block
  CopyEdgesToBlock(block, dst_nodes, timestamps, eids, start_idx, num_edges);
  SyncBlock(block);
}

std::size_t DynamicGraph::SwapOldBlocksToCPU(std::size_t min_swap_size) {
  std::size_t swapped_size = 0;

  // iterate over the list of blocks
  while (swapped_size < min_swap_size) {
    for (std::size_t src_node = 0; src_node < num_nodes_; ++src_node) {
      auto& list = h_copy_of_d_node_table_[src_node];
      auto block = list.tail.prev;
      if (block != &list.head && block->size > 0) {
        // block is not empty
        // copy to CPU
        auto block_on_host = allocator_.SwapBlockToHost(block);
        InsertBlockToDoublyLinkedList(h_node_table_.data(), src_node,
                                      block_on_host);

        DeleteTailBlock(src_node);
        swapped_size += block->capacity;
      }

      if (swapped_size >= min_swap_size) {
        break;
      }
    }
  }

  return swapped_size;
}

std::size_t DynamicGraph::out_degree(NIDType node) const {
  size_t out_degree = 0;
  {
    auto& list = h_copy_of_d_node_table_[node];
    auto block = list.head.next;
    while (block != &list.tail) {
      out_degree += block->size;
      block = block->next;
    }
  }

  {
    auto& list = h_node_table_[node];
    auto block = list.head.next;
    while (block != &list.tail) {
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
    auto block = list.head.next;
    while (block != &list.tail) {
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

      block = block->next;
    }
  }

  {
    // NB: reference is necessary
    auto& list = h_node_table_[node];
    auto block = list.head.next;
    while (block != &list.tail) {
      std::reverse_copy(block->dst_nodes, block->dst_nodes + block->size,
                        std::back_inserter(std::get<0>(result)));
      std::reverse_copy(block->timestamps, block->timestamps + block->size,
                        std::back_inserter(std::get<1>(result)));
      std::reverse_copy(block->eids, block->eids + block->size,
                        std::back_inserter(std::get<2>(result)));
      block = block->next;
    }
  }

  return result;
}

}  // namespace dgnn
