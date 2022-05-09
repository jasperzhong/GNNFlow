#include "dynamic_graph.h"

namespace dgnn {

DynamicGraph::DynamicGraph(std::size_t gpu_mem_reserve, std::size_t block_size,
                           InsertionPolicy insertion_policy)
    : gpu_mem_reserve_(gpu_mem_reserve),
      block_size_(block_size),
      insertion_policy_(insertion_policy) {
  // Initialize the GPU memory pool.
}
};  // namespace dgnn
