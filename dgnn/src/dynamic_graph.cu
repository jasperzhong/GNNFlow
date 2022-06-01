#include <thrust/copy.h>
#include <thrust/device_delete.h>
#include <thrust/device_new.h>

#include <algorithm>
#include <cstdio>
#include <numeric>
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
  d_node_table_.resize(num_nodes_);
  h_node_table_.resize(num_nodes_);
  h_copy_of_d_node_table_.resize(num_nodes_);
}

std::size_t DynamicGraph::num_nodes() const { return num_nodes_; }

std::size_t DynamicGraph::num_edges() const { return num_edges_; }

std::shared_ptr<TemporalBlock> DynamicGraph::AllocateBlock(
    std::size_t num_edges) {
  std::shared_ptr<TemporalBlock> block;
  try {
    block = allocator_.Allocate(num_edges);
  } catch (rmm::bad_alloc) {
    // if we can't allocate the block, we need to free some memory
    std::size_t min_swap_size = allocator_.AlignUp(num_edges) * kBlockSpaceSize;
    auto swapped_size = SwapOldBlocksToCPU(min_swap_size);
    LOG(INFO) << "Swapped " << swapped_size << " bytes to CPU";

    // try again
    block = allocator_.Allocate(num_edges);
  }

  return block;
}

std::shared_ptr<TemporalBlock> DynamicGraph::ReallocateBlock(
    std::shared_ptr<TemporalBlock> block, std::size_t num_edges) {
  CHECK_NOTNULL(block);
  auto new_block = AllocateBlock(num_edges);

  allocator_.Copy(block, new_block);

  // release the old block
  allocator_.Deallocate(block);

  return new_block;
}

void DynamicGraph::InitilizeDoublyLinkedList(NIDType node_id) {
  // host
  auto h_head = std::make_shared<TemporalBlock>();
  auto h_tail = std::make_shared<TemporalBlock>();

  h_head->next = h_tail.get();
  h_tail->prev = h_head.get();

  h_head->prev = nullptr;
  h_tail->next = nullptr;

  h_node_table_[node_id] = HostDoublyLinkedList(h_head, h_tail);

  // device
  thrust::device_ptr<TemporalBlock> d_head =
      thrust::device_new<TemporalBlock>(1);
  thrust::device_ptr<TemporalBlock> d_tail =
      thrust::device_new<TemporalBlock>(1);

  auto tmp_head = *h_head;
  auto tmp_tail = *h_tail;

  tmp_head.next = d_tail.get();
  tmp_tail.prev = d_head.get();

  tmp_head.prev = nullptr;
  tmp_tail.next = nullptr;

  *d_head = tmp_head;
  *d_tail = tmp_tail;

  d_node_table_[node_id] = DeviceDoublyLinkedList(d_head, d_tail);

  // mapping
  h2d_mapping_[h_head] = d_head;
  h2d_mapping_[h_tail] = d_tail;
}

void DynamicGraph::InsertBlockToDoublyLinkedList(
    NIDType node_id, std::shared_ptr<TemporalBlock> block) {
  CHECK_NOTNULL(block);
  // host
  auto h_head = h_node_table_[node_id].first;

  auto h_head_next = std::shared_ptr<TemporalBlock>(h_head->next);
  h_head->next = block.get();
  block->prev = h_head.get();
  block->next = h_head_next.get();
  h_head_next->prev = block.get();

  // device
  thrust::device_ptr<TemporalBlock> d_block =
      thrust::device_new<TemporalBlock>(1);

  auto d_head = h2d_mapping_[h_head];
  auto d_head_next = h2d_mapping_[h_head_next];

  auto tmp_block = *block;
  auto tmp_head = *h_head;
  auto tmp_head_next = *h_head_next;

  tmp_head.next = d_block.get();
  tmp_block.prev = d_head.get();
  tmp_block.next = d_head_next.get();
  tmp_head_next.prev = d_block.get();
  // TODO
}

void DynamicGraph::ReplaceBlockInDoublyLinkedList(
    NIDType node_id, std::shared_ptr<TemporalBlock> block) {}

void DynamicGraph::AddEdgesForOneNode(
    NIDType src_node, const std::vector<NIDType>& dst_nodes,
    const std::vector<TimestampType>& timestamps,
    const std::vector<EIDType>& eids) {
  std::size_t num_edges = dst_nodes.size();

  auto block = h_copy_of_d_node_table_[src_node];
  bool new_block_created = true;
  if (block == nullptr) {
    // allocate memory for the block
    auto block = allocator_.Allocate(num_edges);
    h_copy_of_d_node_table_[src_node] = block;
  } else if (block->size + num_edges > block->capacity) {
    if (insertion_policy_ == InsertionPolicy::kInsertionPolicyInsert) {
      // create a new block and insert it to the head of the list
      auto new_block = allocator_.Allocate(num_edges);
      // get the block pointer on device
      new_block->next = block.get();
      h_copy_of_d_node_table_[src_node] = new_block;
    } else if (insertion_policy_ == InsertionPolicy::kInsertionPolicyReplace) {
      // reallocate a new block to replace the old one
      auto new_block = allocator_.Reallocate(block, block->size + num_edges);
      h_copy_of_d_node_table_[src_node] = new_block;

      // delete the old block on device
      auto old_block = h2d_mapping_[block];
      thrust::device_delete(old_block);
      h2d_mapping_.erase(block);
    }
  } else {
    new_block_created = false;
  }

  block = h_copy_of_d_node_table_[src_node];

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
      CHECK_NE(h2d_mapping_.find(next_block), h2d_mapping_.end());
      auto next_block_on_device = h2d_mapping_[next_block];
      block_on_host.next = next_block_on_device.get();
    }
    *block_on_device = block_on_host;
    h2d_mapping_[block] = block_on_device;
    d_node_table_[src_node] = block_on_device;
  } else {
    *h2d_mapping_[block] = *block;
  }
}

std::size_t DynamicGraph::SwapOldBlocksToCPU(std::size_t min_swap_size) {
  // TODO
  return 0;
}
}  // namespace dgnn
