#ifndef DGNN_DYNAMIC_GRAPH_H_
#define DGNN_DYNAMIC_GRAPH_H_

#include <thrust/device_vector.h>

#include <map>
#include <memory>
#include <utility>
#include <vector>

#include "temporal_block.h"
#include "temporal_block_allocator.h"

namespace dgnn {

class DynamicGraph {
 public:
  enum class InsertionPolicy {
    kInsertionPolicyInsert,
    kInsertionPolicyReplace
  };

  static constexpr InsertionPolicy kDefaultInsertionPolicy =
      InsertionPolicy::kInsertionPolicyInsert;

  static constexpr std::size_t kDefaultAlignment = 16;

  DynamicGraph(std::size_t max_gpu_mem_pool_size,
               std::size_t alignment = kDefaultAlignment,
               InsertionPolicy insertion_policy = kDefaultInsertionPolicy);
  ~DynamicGraph() = default;

  void AddEdges(std::vector<NIDType>& src_nodes,
                std::vector<NIDType>& dst_nodes,
                std::vector<TimestampType>& timestamps,
                bool add_reverse_edges = true);

  void AddNodes(NIDType max_node);

  std::size_t num_nodes() const;

  std::size_t num_edges() const;

  const std::vector<std::shared_ptr<TemporalBlock>>&
  node_table_on_device_host_copy() const;

 private:
  void AddEdgesForOneNode(NIDType src_node,
                          const std::vector<NIDType>& dst_nodes,
                          const std::vector<TimestampType>& timestamps,
                          const std::vector<EIDType>& eids);

  // void SwapOldBlocksToHost(std::size_t requested_size_to_swap);

  // void SwapBlockToHost(std::shared_ptr<TemporalBlock> block);

 private:
  InsertionPolicy insertion_policy_;

  TemporalBlockAllocator allocator_;

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
