#include "dynamic_graph.h"

#include <thrust/copy.h>
#include <thrust/device_delete.h>
#include <thrust/device_new.h>

#include <algorithm>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include "logging.h"
namespace dgnn {

std::size_t GetBlockMemorySize(std::size_t capacity) {
  return capacity * (sizeof(NIDType) + sizeof(EIDType) + sizeof(TimestampType));
}

DynamicGraph::DynamicGraph(std::size_t max_gpu_mem_pool_size,
                           std::size_t alignment,
                           InsertionPolicy insertion_policy)
    : max_gpu_mem_pool_size_(max_gpu_mem_pool_size),
      alignment_(alignment),
      insertion_policy_(insertion_policy),
      block_sequence_number_(0),
      num_nodes_(0),
      num_edges_(0) {
  // create the memory pool for the GPU
  rmm::mr::cuda_memory_resource mem_res;
  rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> pool_res(
      &mem_res, max_gpu_mem_pool_size_, max_gpu_mem_pool_size_);
  rmm::mr::set_current_device_resource(&pool_res);
}

void DynamicGraph::AddEdges(const std::vector<NIDType>& src_nodes,
                            const std::vector<NIDType>& dst_nodes,
                            const std::vector<TimestampType>& timestamps) {
  CHECK_EQ(src_nodes.size(), dst_nodes.size());
  CHECK_EQ(src_nodes.size(), timestamps.size());

  std::map<NIDType, std::vector<Edge>> src_to_edges;

  for (std::size_t i = 0; i < src_nodes.size(); ++i) {
    src_to_edges[src_nodes[i]].emplace_back(dst_nodes[i], timestamps[i]);
  }

  // add nodes
  NIDType max_node =
      std::max(*std::max_element(src_nodes.begin(), src_nodes.end()),
               *std::max_element(dst_nodes.begin(), dst_nodes.end()));
  AddNodes(max_node);

  for (auto& src_to_edges_pair : src_to_edges) {
    NIDType src_node = src_to_edges_pair.first;
    auto& edges = src_to_edges_pair.second;

    // sort the edges by timestamp
    std::sort(edges.begin(), edges.end());

    // add the edges to the graph
    AddEdgesForOneNode(src_node, edges);
  }
}

void DynamicGraph::AddNodes(NIDType max_node) {
  if (max_node < num_nodes_) {
    return;
  }
  num_nodes_ = max_node + 1;
  node_table_on_device_.resize(num_nodes_);
  node_table_on_host_.resize(num_nodes_);
}

std::size_t DynamicGraph::num_nodes() const { return num_nodes_; }

std::size_t DynamicGraph::num_edges() const { return num_edges_; }

void DynamicGraph::AddEdgesForOneNode(NIDType src_node,
                                      const std::vector<Edge>& edges) {
    // TODO 
}

TemporalBlock DynamicGraph::AllocateTemporalBlock(std::size_t size) {
  TemporalBlock block;

  try {
    block = AllocateInternal(size);
  } catch (rmm::bad_alloc&) {
    // failed to allocate memory
    // swap old blocks to the host
    std::size_t requested_size_to_swap = GetBlockMemorySize(block.capacity);
    SwapOldBlocksToHost(requested_size_to_swap);

    // try again
    block = AllocateInternal(size);
  }

  return block;
}

void DynamicGraph::DeallocateTemporalBlock(
    thrust::device_ptr<TemporalBlock> block) {
  auto mr = rmm::mr::get_current_device_resource();
  TemporalBlock block_on_host = *block;
  mr->deallocate(block_on_host.neighbor_nodes,
                 block_on_host.capacity * sizeof(NIDType));
  mr->deallocate(block_on_host.neighbor_edges,
                 block_on_host.capacity * sizeof(EIDType));
  mr->deallocate(block_on_host.timestamps,
                 block_on_host.capacity * sizeof(TimestampType));
}

TemporalBlock DynamicGraph::ReallocateTemporalBlock(
    thrust::device_ptr<TemporalBlock> block, std::size_t size) {
  auto new_block = AllocateTemporalBlock(size);

  // CopyTemporalBlock(new_block, block);

  DeallocateTemporalBlock(block);

  return new_block;
}

TemporalBlock DynamicGraph::AllocateInternal(std::size_t size) noexcept(false) {
  std::size_t capacity = AlignUp(size);

  TemporalBlock block_on_host;
  block_on_host.size = 0;  // empty block
  block_on_host.capacity = capacity;

  auto mr = rmm::mr::get_current_device_resource();
  block_on_host.neighbor_nodes =
      static_cast<NIDType*>(mr->allocate(capacity * sizeof(NIDType)));
  block_on_host.neighbor_edges =
      static_cast<EIDType*>(mr->allocate(capacity * sizeof(EIDType)));
  block_on_host.timestamps = static_cast<TimestampType*>(
      mr->allocate(capacity * sizeof(TimestampType)));

  return block_on_host;
}

void DynamicGraph::CopyTemporalBlock(thrust::device_ptr<TemporalBlock> dst,
                                     thrust::device_ptr<TemporalBlock> src) {
  // copy metadata to host
  TemporalBlock dst_block = *dst;
  TemporalBlock src_block = *src;

  thrust::copy(src_block.neighbor_nodes,
               src_block.neighbor_nodes + src_block.size,
               dst_block.neighbor_nodes);
  thrust::copy(src_block.neighbor_edges,
               src_block.neighbor_edges + src_block.size,
               dst_block.neighbor_edges);
  thrust::copy(src_block.timestamps, src_block.timestamps + src_block.size,
               dst_block.timestamps);

  dst_block.size = src_block.size;
  dst_block.next = src_block.next;

  // copy metadata back to device
  *dst = dst_block;
}

void DynamicGraph::SwapOldBlocksToHost(std::size_t requested_size_to_swap) {
  for (auto& kv : blocks_on_device_) {
  }
}

TemporalBlock* DynamicGraph::SwapBlockToHost(
    thrust::device_ptr<TemporalBlock> block) {
  TemporalBlock block_on_host = *block;
  TemporalBlock* new_block = new TemporalBlock();
  new_block->size = block_on_host.size;
  new_block->capacity = block_on_host.capacity;

  new_block->neighbor_nodes =
      new NIDType[block_on_host.capacity * sizeof(NIDType)];
  new_block->neighbor_edges =
      new EIDType[block_on_host.capacity * sizeof(EIDType)];
  new_block->timestamps =
      new TimestampType[block_on_host.capacity * sizeof(TimestampType)];

  thrust::copy(block_on_host.neighbor_nodes,
               block_on_host.neighbor_nodes + block_on_host.size,
               new_block->neighbor_nodes);
  thrust::copy(block_on_host.neighbor_edges,
               block_on_host.neighbor_edges + block_on_host.size,
               new_block->neighbor_edges);
}

std::size_t DynamicGraph::AlignUp(std::size_t size) {
  if (size < alignment_) {
    return alignment_;
  }
  // round up to the next power of two
  return 1 << (64 - __builtin_clzl(size - 1));
}
}  // namespace dgnn
