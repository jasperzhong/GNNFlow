#ifndef DGNN_COMMON_H_
#define DGNN_COMMON_H_

#include <cstddef>
namespace dgnn {

using NIDType = unsigned long;
using EIDType = unsigned long;
using TimeType = float;

struct TemporalBlock {
  NIDType* neighbor_vertices;
  EIDType* neighbor_edges;
  TimeType* edge_times;
  std::size_t size;
  std::size_t capacity;
  TemporalBlock* next;
};

};  // namespace dgnn

#endif  // DGNN_COMMON_H_
