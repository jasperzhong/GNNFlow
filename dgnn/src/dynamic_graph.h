#ifndef DGNN_DYNAMIC_GRAPH_H_
#define DGNN_DYNAMIC_GRAPH_H_

#include <thrust/device_vector.h>

#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "temporal_block_allocator.h"

namespace dgnn {
/**
 * @brief InsertionPolicy is used to decide how to insert a new temporal block
 * into the linked list.
 *
 * kInsertionPolicyInsert: insert the new block at the head of the list.
 * kInsertionPolicyReplace: replace the head block with a larger block.
 */
enum class InsertionPolicy { kInsertionPolicyInsert, kInsertionPolicyReplace };

/**
 * @brief A dynamic graph is a graph that can be modified at runtime.
 *
 * The dynamic graph is implemented as block adjacency list. It has a vertex
 * table where each entry is a doubly linked list of temporal blocks.
 */
class DynamicGraph {
 public:
  static constexpr std::size_t kDefaultMaxGpuMemPoolSize = 1 << 30;  // 1 GiB
  static constexpr InsertionPolicy kDefaultInsertionPolicy =
      InsertionPolicy::kInsertionPolicyInsert;
  static constexpr std::size_t kDefaultAlignment = 16;

  typedef thrust::device_vector<DoublyLinkedList> DeviceNodeTable;
  typedef std::vector<DoublyLinkedList> HostNodeTable;

  DynamicGraph(std::size_t max_gpu_mem_pool_size = kDefaultMaxGpuMemPoolSize,
               std::size_t alignment = kDefaultAlignment,
               InsertionPolicy insertion_policy = kDefaultInsertionPolicy);
  ~DynamicGraph() = default;

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

  const HostNodeTable& h_copy_of_d_node_table() const;

 private:
  void AddEdgesForOneNode(NIDType src_node,
                          const std::vector<NIDType>& dst_nodes,
                          const std::vector<TimestampType>& timestamps,
                          const std::vector<EIDType>& eids);

  std::size_t SwapOldBlocksToCPU(std::size_t min_swap_size);

  TemporalBlock* AllocateBlock(std::size_t num_edges);

  TemporalBlock* ReallocateBlock(
      TemporalBlock* block, std::size_t num_edges);

  void InsertBlock(NIDType node_id, TemporalBlock* block);

  void ReplaceBlock(NIDType node_id, TemporalBlock* block);

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

  std::size_t num_nodes_;  // the maximum node id + 1
  std::size_t num_edges_;
};

}  // namespace dgnn

#endif  // DGNN_DYNAMIC_GRAPH_H_
