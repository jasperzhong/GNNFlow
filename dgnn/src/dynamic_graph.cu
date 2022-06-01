#include <thrust/copy.h>
#include <thrust/device_delete.h>
#include <thrust/device_new.h>

#include <algorithm>
#include <cstdio>
#include <numeric>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include "dynamic_graph.h"
#include "logging.h"
#include "temporal_block.h"
#include "utils.h"

namespace dgnn {

DynamicGraph::DynamicGraph(std::size_t max_gpu_mem_pool_size,
                           std::size_t alignment,
                           InsertionPolicy insertion_policy)
    : insertion_policy_(insertion_policy),
      allocator_(max_gpu_mem_pool_size, alignment),
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
  num_edges_ += eids.size();

  // for undirected graphs, we need to add the reverse edges
  if (add_reverse_edges) {
    src_nodes.insert(src_nodes.end(), dst_nodes.begin(), dst_nodes.end());
    dst_nodes.insert(dst_nodes.end(), src_nodes.begin(), src_nodes.end());
    timestamps.insert(timestamps.end(), timestamps.begin(), timestamps.end());
    eids.insert(eids.end(), eids.begin(), eids.end());
  }

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
  node_table_on_device_.resize(num_nodes_);
  node_table_on_device_host_copy_.resize(num_nodes_);
  node_table_on_host_.resize(num_nodes_);
}

std::size_t DynamicGraph::num_nodes() const { return num_nodes_; }

std::size_t DynamicGraph::num_edges() const { return num_edges_; }

void DynamicGraph::AddEdgesForOneNode(
    NIDType src_node, const std::vector<NIDType>& dst_nodes,
    const std::vector<TimestampType>& timestamps,
    const std::vector<EIDType>& eids) {
  std::size_t num_edges = dst_nodes.size();
  auto block = node_table_on_device_host_copy_[src_node];
  bool new_block_created = true;
  if (block == nullptr) {
    // allocate memory for the block
    auto block = allocator_.Allocate(num_edges);
    node_table_on_device_host_copy_[src_node] = block;
  } else if (block->size + num_edges > block->capacity) {
    if (insertion_policy_ == InsertionPolicy::kInsertionPolicyInsert) {
      // create a new block and insert it to the head of the list
      auto new_block = allocator_.Allocate(num_edges);
      // get the block pointer on device
      new_block->next = block.get();
      node_table_on_device_host_copy_[src_node] = new_block;
    } else if (insertion_policy_ == InsertionPolicy::kInsertionPolicyReplace) {
      // reallocate a new block to replace the old one
      auto new_block =
          allocator_.Reallocate(block, block->size + num_edges);
      node_table_on_device_host_copy_[src_node] = new_block;

      // delete the old block on device
      auto old_block = node_table_on_device_host_copy_map_[block];
      thrust::device_delete(old_block);
      node_table_on_device_host_copy_map_.erase(block);
    }
  } else {
    new_block_created = false;
  }

  block = node_table_on_device_host_copy_[src_node];

  // copy the edges to the device
  thrust::copy(dst_nodes.begin(), dst_nodes.end(),
               thrust::device_ptr<NIDType>(block->dst_nodes) + block->size);
  thrust::copy(
      timestamps.begin(), timestamps.end(),
      thrust::device_ptr<TimestampType>(block->timestamps) + block->size);
  thrust::copy(eids.begin(), eids.end(),
               thrust::device_ptr<EIDType>(block->eids) + block->size);

  block->size += num_edges;

  // update the node table on device
  if (new_block_created) {
    auto block_on_device = thrust::device_new<TemporalBlock>(1);
    TemporalBlock block_on_host = *block;
    if (block->next != nullptr) {
      std::shared_ptr<TemporalBlock> next_block(block->next);
      CHECK_NE(node_table_on_device_host_copy_map_.find(next_block),
               node_table_on_device_host_copy_map_.end());
      auto next_block_on_device =
          node_table_on_device_host_copy_map_[next_block];
      block_on_host.next = next_block_on_device.get();
    }
    *block_on_device = block_on_host;
    node_table_on_device_host_copy_map_[block] = block_on_device;
    node_table_on_device_[src_node] = block_on_device;
  } else {
    *node_table_on_device_host_copy_map_[block] = *block;
  }
}

const std::vector<std::shared_ptr<TemporalBlock>>&
DynamicGraph::node_table_on_device_host_copy() const {
    return node_table_on_device_host_copy_;
}

// void DynamicGraph::SwapOldBlocksToHost(std::size_t requested_size_to_swap) {
//   // TODO: how to set the previous block.next to nullptr ???
//   // It seems that we need doubly linked list ???
// }
//
// void DynamicGraph::SwapBlockToHost(std::shared_ptr<TemporalBlock> block) {
//   auto block_on_host = std::make_shared<TemporalBlock>();
//   block_on_host->size = block->size;
//   block_on_host->capacity = block->capacity;
//
//   block_on_host->dst_nodes = new NIDType[block_on_host->capacity];
//   block_on_host->timestamps = new TimestampType[block_on_host->capacity];
//   block_on_host->eids = new EIDType[block_on_host->capacity];
//
//   thrust::copy(block->dst_nodes, block->dst_nodes + block->size,
//                block_on_host->dst_nodes);
//   thrust::copy(block->timestamps, block->timestamps + block->size,
//                block_on_host->timestamps);
//   thrust::copy(block->eids, block->eids + block->size, block_on_host->eids);
// auto sequence_number = block_to_sequence_number_[block];
//   blocks_on_host_[sequence_number] = block_on_host;
//   DeallocateTemporalBlock(block);
//   block_to_sequence_number_[block_on_host] = sequence_number;
// }
//

}  // namespace dgnn
