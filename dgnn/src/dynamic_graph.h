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
 * @brief A dynamic graph is a graph that can be modified at runtime.
 *
 * The dynamic graph is implemented as block adjacency list. It has a vertex
 * table where each entry is a doubly linked list of temporal blocks.
 */
class DynamicGraph {
 public:
  static constexpr std::size_t kDefaultMaxGpuMemPoolSize = 1 << 30;  // 1 GiB
  static constexpr InsertionPolicy kDefaultInsertionPolicy =
      InsertionPolicy::kInsertionPolicyReplace;
  static constexpr std::size_t kDefaultAlignment = 16;

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

 private:
  void AddEdgesForOneNode(NIDType src_node,
                          const std::vector<NIDType>& dst_nodes,
                          const std::vector<TimestampType>& timestamps,
                          const std::vector<EIDType>& eids);

  std::size_t SwapOldBlocksToCPU(std::size_t min_swap_size);

  std::shared_ptr<TemporalBlock> AllocateBlock(std::size_t num_edges);

  std::shared_ptr<TemporalBlock> ReallocateBlock(
      std::shared_ptr<TemporalBlock> block, std::size_t num_edges);

  void InitilizeDoublyLinkedList(NIDType node_id);

  void InsertBlockToDoublyLinkedList(NIDType node_id,
                                     std::shared_ptr<TemporalBlock> block);
  void ReplaceBlockInDoublyLinkedList(NIDType node_id,
                                      std::shared_ptr<TemporalBlock> block);

 private:
  TemporalBlockAllocator allocator_;
  InsertionPolicy insertion_policy_;

  std::size_t num_nodes_;  // the maximum node id + 1
  std::size_t num_edges_;

  // doubly linked list. pair<head, tail>
  typedef thrust::pair<thrust::device_ptr<TemporalBlock>,
                       thrust::device_ptr<TemporalBlock>>
      DeviceDoublyLinkedList;
  typedef thrust::device_vector<DeviceDoublyLinkedList> DeviceNodeTable;

  typedef std::pair<std::shared_ptr<TemporalBlock>,
                    std::shared_ptr<TemporalBlock>>
      HostDoublyLinkedList;
  typedef std::vector<HostDoublyLinkedList> HostNodeTable;

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
  std::unordered_map<std::shared_ptr<TemporalBlock>,
                     thrust::device_ptr<TemporalBlock>>
      h2d_mapping_;
};

}  // namespace dgnn

#endif  // DGNN_DYNAMIC_GRAPH_H_
