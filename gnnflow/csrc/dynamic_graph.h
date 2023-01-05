#ifndef GNNFLOW_DYNAMIC_GRAPH_H_
#define GNNFLOW_DYNAMIC_GRAPH_H_

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#include <memory>
#include <set>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common.h"
#include "doubly_linked_list.h"
#include "temporal_block_allocator.h"

namespace gnnflow {
typedef thrust::device_vector<DoublyLinkedList> DeviceNodeTable;
typedef std::vector<HostDoublyLinkedList> HostNodeTable;
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
   * @param adaptive_block_size Whether to use adaptive block size.
   */

  DynamicGraph(std::size_t initial_pool_size, std::size_t maximum_pool_size,
               MemoryResourceType mem_resource_type,
               std::size_t minium_block_size, std::size_t blocks_to_preallocate,
               InsertionPolicy insertion_policy, int device,
               bool adaptive_block_size);
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

  /**
   * @brief Offload all blocks that are older than the given timestamp.
   *
   * @params src_node The source node of the blocks.
   * @params timestamp The timestamp of the blocks.
   *
   * @return The number of blocks offloaded.
   */
  std::size_t OffloadOldBlocks(TimestampType timestamp, bool to_file = false);

  std::size_t num_nodes() const;
  std::size_t num_edges() const;
  std::size_t num_src_nodes() const;

  std::vector<NIDType> nodes() const;
  std::vector<NIDType> src_nodes() const;
  std::vector<EIDType> edges() const;

  NIDType max_node_id() const;

  std::vector<std::size_t> out_degree(const std::vector<NIDType>& nodes) const;

  // NB: it is inefficient to call this function every time for each node. Debug
  // only.
  typedef std::tuple<std::vector<NIDType>, std::vector<TimestampType>,
                     std::vector<EIDType>>
      NodeNeighborTuple;
  NodeNeighborTuple get_temporal_neighbors(NIDType node) const;

  const DoublyLinkedList* get_device_node_table() const;

  int device() const { return device_; }

  float avg_linked_list_length() const;

  // NB: does not include metadata. only the edge data.
  float graph_mem_usage() const;

  float graph_metadata_mem_usage();

 private:
  void AddEdgesForOneNode(NIDType src_node,
                          const std::vector<NIDType>& dst_nodes,
                          const std::vector<TimestampType>& timestamps,
                          const std::vector<EIDType>& eids,
                          cudaStream_t stream = nullptr);

  void InsertBlock(NIDType node_id, TemporalBlock* block,
                   cudaStream_t stream = nullptr);

  void RemoveBlock(NIDType node_id, TemporalBlock* block,
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

  std::size_t max_node_id_;

  std::set<NIDType> nodes_;
  std::set<NIDType> src_nodes_;
  std::unordered_map<EIDType, std::size_t> edges_;

  std::stack<rmm::mr::device_memory_resource*> mem_resources_for_metadata_;

  const int device_;
  bool adaptive_block_size_;
};

}  // namespace gnnflow

#endif  // GNNFLOW_DYNAMIC_GRAPH_H_
