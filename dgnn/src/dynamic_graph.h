#ifndef DGNN_DYNAMIC_GRAPH_H_
#define DGNN_DYNAMIC_GRAPH_H_

#include <thrust/device_vector.h>

#include <map>
#include <memory>
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

  void AddEdges(std::vector<NIDType>& src_nodes,
                std::vector<NIDType>& dst_nodes,
                std::vector<TimestampType>& timestamps,
                bool add_reverse_edges = true);

  void AddNodes(NIDType max_node);

  std::size_t num_nodes() const;

  std::size_t num_edges() const;

 private:
  void AddEdgesForOneNode(NIDType src_node,
                          const std::vector<NIDType>& dst_nodes,
                          const std::vector<TimestampType>& timestamps,
                          const std::vector<EIDType>& eids);

  std::size_t AlignUp(std::size_t size);

  std::shared_ptr<TemporalBlock> AllocateTemporalBlock(std::size_t size);

  void DeallocateTemporalBlock(std::shared_ptr<TemporalBlock> block);

  std::shared_ptr<TemporalBlock> ReallocateTemporalBlock(
      std::shared_ptr<TemporalBlock> block, std::size_t size);

  void AllocateInternal(std::shared_ptr<TemporalBlock> block,
                        std::size_t size) noexcept(false);

  void DeallocateInternal(std::shared_ptr<TemporalBlock> block);

  void CopyTemporalBlock(std::shared_ptr<TemporalBlock> dst,
                         std::shared_ptr<TemporalBlock> src);

  void SwapOldBlocksToHost(std::size_t requested_size_to_swap);

  void SwapBlockToHost(std::shared_ptr<TemporalBlock> block);

 private:
  std::size_t max_gpu_mem_pool_size_;
  std::size_t alignment_;
  InsertionPolicy insertion_policy_;

  // sequence number (how old the block is) -> block raw pointer
  std::map<std::size_t, std::shared_ptr<TemporalBlock>> blocks_on_device_;
  std::map<std::size_t, std::shared_ptr<TemporalBlock>> blocks_on_host_;

  std::unordered_map<std::shared_ptr<TemporalBlock>, std::size_t>
      block_to_sequence_number_;

  // a monotonically increasing sequence number
  std::size_t block_sequence_number_;

  std::size_t num_nodes_;
  std::size_t num_edges_;

  thrust::device_vector<thrust::device_ptr<TemporalBlock>>
      node_table_on_device_;

  std::vector<std::shared_ptr<TemporalBlock>> node_table_on_device_host_copy_;

  std::map<std::shared_ptr<TemporalBlock>, thrust::device_ptr<TemporalBlock>>
      node_table_on_device_host_copy_map_;

  std::vector<std::shared_ptr<TemporalBlock>> node_table_on_host_;
};

}  // namespace dgnn

#endif  // DGNN_DYNAMIC_GRAPH_H_
