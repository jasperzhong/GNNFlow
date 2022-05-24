#ifndef DGNN_COMMON_H_
#define DGNN_COMMON_H_

#include <cstddef>
#include <sstream>
#include <string>
namespace dgnn {
using NIDType = uint64_t;
using EIDType = uint64_t;
using TimeType = float;

struct TemporalBlock {
  NIDType* neighbor_vertices;
  EIDType* neighbor_edges;
  TimeType* edge_times;
  std::size_t size;
  std::size_t capacity;
  TemporalBlock* next;

  TemporalBlock(std::size_t capacity)
      : neighbor_vertices(nullptr),
        neighbor_edges(nullptr),
        edge_times(nullptr),
        size(0),
        capacity(capacity),
        next(nullptr) {}
};

};  // namespace dgnn

#endif  // DGNN_COMMON_H_
