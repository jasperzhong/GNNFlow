#ifndef DGNN_DYNAMIC_GRAPH_H_
#define DGNN_DYNAMIC_GRAPH_H_

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <map>
#include <unordered_map>
#include <utility>
#include <vector>

#include "temporal_block.h"

namespace dgnn {

class DynamicGraph {
 public:
  enum class InsertionPolicy {
    kInsertionPolicyInsert,
    kInsertionPolicyReplace
  };

  static constexpr std::size_t kDefaultAlignment = 16;
  static constexpr InsertionPolicy kDefaultInsertionPolicy =
      InsertionPolicy::kInsertionPolicyInsert;

  DynamicGraph(std::size_t max_gpu_mem_pool_size,
               std::size_t alignment = kDefaultAlignment,
               InsertionPolicy insertion_policy = kDefaultInsertionPolicy);

  ~DynamicGraph();

  void AddEdges(const std::vector<NIDType>& src_nodes,
                const std::vector<NIDType>& dst_nodes,
                const std::vector<TimestampType>& timestamps);

  void AddNodes(NIDType max_node);

  std::size_t num_nodes() const;

  std::size_t num_edges() const;

 private:
  struct Edge {
    NIDType dst_node;
    TimestampType timestamp;

    Edge(NIDType dst_node, TimestampType timestamp)
        : dst_node(dst_node), timestamp(timestamp) {}

    bool operator<(const Edge& other) const {
      return timestamp < other.timestamp;
    }
  };

  void AddEdgesForOneNode(NIDType src_node, const std::vector<Edge>& edges);

  std::size_t AlignUp(std::size_t size);

  TemporalBlock AllocateTemporalBlock(std::size_t size);

  void DeallocateTemporalBlock(thrust::device_ptr<TemporalBlock> block);

  TemporalBlock ReallocateTemporalBlock(thrust::device_ptr<TemporalBlock> block,
                                        std::size_t size);

  TemporalBlock AllocateInternal(std::size_t size) noexcept(false);

  void CopyTemporalBlock(thrust::device_ptr<TemporalBlock> dst,
                         thrust::device_ptr<TemporalBlock> src);

  void SwapOldBlocksToHost(std::size_t requested_size_to_swap);

  TemporalBlock* SwapBlockToHost(thrust::device_ptr<TemporalBlock> block);

 private:
  std::size_t max_gpu_mem_pool_size_;
  std::size_t alignment_;
  InsertionPolicy insertion_policy_;

  // sequence number (how old the block is) -> block raw pointer
  std::map<std::size_t, TemporalBlock*> blocks_on_device_;
  std::map<std::size_t, TemporalBlock*> blocks_on_host_;

  // block raw pointer -> sequence number (how old the block is)
  std::unordered_map<TemporalBlock*, std::size_t> all_blocks_;

  // block raw pointer -> (size, capacity)
  std::unordered_map<TemporalBlock*, std::pair<std::size_t, std::size_t>>
      blocks_info_;

  // a monotonically increasing sequence number
  std::size_t block_sequence_number_;

  std::size_t num_nodes_;
  std::size_t num_edges_;

  thrust::device_vector<thrust::device_ptr<TemporalBlock>>
      node_table_on_device_;
  std::vector<TemporalBlock> node_table_on_host_;
};

}  // namespace dgnn

#endif  // DGNN_DYNAMIC_GRAPH_H_
