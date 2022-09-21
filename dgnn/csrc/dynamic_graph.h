#ifndef DGNN_DYNAMIC_GRAPH_H_
#define DGNN_DYNAMIC_GRAPH_H_

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#include <memory>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common.h"
#include "doubly_linked_list.h"
#include "temporal_block_allocator.h"

namespace dgnn {
typedef thrust::device_vector<DoublyLinkedList> DeviceNodeTable;
typedef std::vector<DoublyLinkedList> HostNodeTable;
/**
 * @brief A dynamic graph is a graph that can be modified at runtime.
 *
 * The dynamic graph is implemented as block adjacency list. It has a node
 * table where each entry is a linked list of temporal blocks.
 */
class DynamicGraph {
 public:
  /**
   * @brief Constructor.
   *
   * It initialize a temporal block allocator with a memory pool for storing
   * edges. The type of the memory resource is determined by the
   * `mem_resource_type` parameter. It also creates a device memory pool for
   * metadata (i.e., blocks).
   *
   * @param initial_pool_size The initial size of the memory pool.
   * @param maxmium_pool_size The maximum size of the memory pool.
   * @param mem_resource_type The type of memory resource for the memory pool.
   * @param minimum_block_size The minimum size of the temporal block.
   * @param blocks_to_preallocate The number of blocks to preallocate.
   * @param insertion_policy The insertion policy for the linked list.
   * @param device The device id.
   */

  DynamicGraph(std::size_t initial_pool_size, std::size_t maximum_pool_size,
               MemoryResourceType mem_resource_type,
               std::size_t minium_block_size, std::size_t blocks_to_preallocate,
               InsertionPolicy insertion_policy, int device);
  ~DynamicGraph();

  /**
   * @brief Add edges to the graph.
   *
   * Note that we do not assume that the incoming edges are sorted by
   * timestamps. The function will sort them.
   *
   * @params src_nodes The source nodes of the edges.
   * @params dst_nodes The destination nodes of the edges.
   * @params timestamps The timestamps of the edges.
   * @params eids The edge ids of the edges.
   *
   */
  void AddEdges(const std::vector<NIDType>& src_nodes,
                const std::vector<NIDType>& dst_nodes,
                const std::vector<TimestampType>& timestamps,
                const std::vector<EIDType>& eids);

  /**
   * @brief Add nodes to the graph.
   *
   * @params max_node The maximum node id.
   */
  void AddNodes(NIDType max_node);

  std::size_t num_nodes() const;

  std::size_t num_edges() const;

  std::size_t out_degree(NIDType node) const;

  // NB: it is inefficient to call this function every time for each node. Debug
  // only.
  typedef std::tuple<std::vector<NIDType>, std::vector<TimestampType>,
                     std::vector<EIDType>>
      NodeNeighborTuple;
  NodeNeighborTuple get_temporal_neighbors(NIDType node) const;

  const DoublyLinkedList* get_device_node_table() const;

 private:
  void AddEdgesForOneNode(NIDType src_node,
                          const std::vector<NIDType>& dst_nodes,
                          const std::vector<TimestampType>& timestamps,
                          const std::vector<EIDType>& eids,
                          cudaStream_t stream = nullptr);

  void InsertBlock(NIDType node_id, TemporalBlock* block,
                   cudaStream_t stream = nullptr);

  void SyncBlock(TemporalBlock* block, cudaStream_t stream = nullptr);

 private:
  TemporalBlockAllocator allocator_;

  // The device node table. Blocks are allocated in the device memory pool.
  DeviceNodeTable d_node_table_;

  // The copy of the device node table in the host.
  HostNodeTable h_copy_of_d_node_table_;

  // the host pointer to the block -> the device pointer to the block
  std::unordered_map<TemporalBlock*, TemporalBlock*> h2d_mapping_;

  InsertionPolicy insertion_policy_;

  std::vector<cudaStream_t> streams_;

  std::size_t num_nodes_;  // the maximum node id + 1
  std::size_t num_edges_;

  std::stack<rmm::mr::device_memory_resource*> mem_resources_for_metadata_;

  int device_;
};

}  // namespace dgnn

#endif  // DGNN_DYNAMIC_GRAPH_H_
