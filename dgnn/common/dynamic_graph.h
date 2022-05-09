#ifndef DGNN_DYNAMIC_GRAPH_H_
#define DGNN_DYNAMIC_GRAPH_H_

#include <thrust/device_vector.h>
#include <vector>
#include "common.h"

namespace dgnn {

enum InsertionPolicy {
  kInsertionPolicyUndefined,
  kInsertionPolicyReplace,
  kInsertionPolicyInsert
};

class DynamicGraph {
 public:
  DynamicGraph(std::size_t gpu_mem_reserve, std::size_t block_size,
               InsertionPolicy insertion_policy = kInsertionPolicyReplace);

  ~DynamicGraph();

  void AddEdges(const std::vector<NIDType>& source_vertices,
                const std::vector<NIDType>& target_vertices,
                const std::vector<TimeType>& edge_times);

  void AddVertices(NIDType max_vertex);

  std::size_t num_vertices() const;

  std::size_t num_edges() const;

  std::size_t out_degree(NIDType vertex) const;

 private:
  thrust::device_vector<TemporalBlock> vertex_table_;
  size_t gpu_mem_reserve_;
  size_t block_size_;
  InsertionPolicy insertion_policy_;
};

};  // namespace dgnn

#endif  // DGNN_DYNAMIC_GRAPH_H_
