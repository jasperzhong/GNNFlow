#ifndef DGNN_DYNAMIC_GRAPH_H_
#define DGNN_DYNAMIC_GRAPH_H_

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#include <memory>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "doubly_linked_list.h"
#include "temporal_block_allocator.h"

namespace dgnn {
typedef thrust::device_vector<DoublyLinkedList> DeviceNodeTable;
typedef std::vector<DoublyLinkedList> HostNodeTable;
/**
 * @brief A dynamic graph is a graph that can be modified at runtime.
 *
 * The dynamic graph is implemented as block adjacency list. It has a vertex
 * table where each entry is a doubly linked list of temporal blocks.
 */
class DynamicGraph {
 public:
  DynamicGraph(std::size_t max_gpu_mem_pool_size = kDefaultMaxGpuMemPoolSize,
               std::size_t min_block_size = kDefaultMinBlockSize,
               InsertionPolicy insertion_policy = kDefaultInsertionPolicy);
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
   * @params add_reverse_edges If true, add the reverse edges (undirected
   * graph).
   */
  void AddEdges(std::vector<NIDType>& src_nodes,
                std::vector<NIDType>& dst_nodes,
                std::vector<TimestampType>& timestamps,
                bool add_reverse_edges = true);

  /**
   * @brief Add nodes to the graph.
   *
   * @params max_node The maximum node id.
   */
  void AddNodes(NIDType max_node);

  std::size_t num_nodes() const;

  std::size_t num_edges() const;

  std::size_t out_degree(NIDType node) const;

  // it is inefficient to call this function every time for each node. Debug
  // only.
  typedef std::tuple<std::vector<NIDType>, std::vector<TimestampType>,
                     std::vector<EIDType>>
      NodeNeighborTuple;
  NodeNeighborTuple get_temporal_neighbors(NIDType node) const;

  const DoublyLinkedList* get_device_node_table() const;
  const DoublyLinkedList* get_host_node_table() const;

 private:
  void AddEdgesForOneNode(NIDType src_node,
                          const std::vector<NIDType>& dst_nodes,
                          const std::vector<TimestampType>& timestamps,
                          const std::vector<EIDType>& eids,
                          cudaStream_t stream = nullptr);

  std::size_t SwapOldBlocksToCPU(std::size_t min_swap_size,
                                 cudaStream_t stream = nullptr);

  TemporalBlock* AllocateBlock(std::size_t num_edges,
                               cudaStream_t stream = nullptr);

  TemporalBlock* ReallocateBlock(TemporalBlock* block, std::size_t num_edges,
                                 cudaStream_t stream = nullptr);

  void InsertBlock(NIDType node_id, TemporalBlock* block,
                   cudaStream_t stream = nullptr);

  void DeleteTailBlock(NIDType node_id, cudaStream_t stream = nullptr);

  void ReplaceBlock(NIDType node_id, TemporalBlock* block,
                    cudaStream_t stream = nullptr);

  void SyncBlock(TemporalBlock* block, cudaStream_t stream = nullptr);

 private:
  // The device node table. Blocks are allocated in the GPU memory pool.
  // Pointers in each block also point to GPU buffers.
  DeviceNodeTable d_node_table_;

  // The host node table. Blocks are allocated in the CPU.
  // Pointers in each block also point to CPU buffers.
  HostNodeTable h_node_table_;

  // Copy of the d_node_table_. Blocks are allocated in the CPU memory.
  // But the pointers in the table are pointing to the same memory as the
  // d_node_table_.
  HostNodeTable h_copy_of_d_node_table_;

  // mapping from the copied block on the CPU to the original block on the GPU
  std::unordered_map<TemporalBlock*, TemporalBlock*> h2d_mapping_;

  TemporalBlockAllocator allocator_;
  InsertionPolicy insertion_policy_;

  std::vector<cudaStream_t> streams_;

  std::size_t num_nodes_;  // the maximum node id + 1
  std::size_t num_edges_;

  // block in the CPU memory but points to the GPU buffers -> src node
  std::unordered_map<TemporalBlock*, NIDType> block_to_node_id_;
};

}  // namespace dgnn

#endif  // DGNN_DYNAMIC_GRAPH_H_
