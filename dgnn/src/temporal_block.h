#ifndef DGNN_TEMPORAL_BLOCK_H_
#define DGNN_TEMPORAL_BLOCK_H_

#include <cstddef>

#include "logging.h"
namespace dgnn {

using NIDType = uint64_t;
using EIDType = uint64_t;
using TimestampType = float;

/**
 * @brief This POD is used to store the temporal blocks in the graph.
 *
 * The blocks are stored in a doubly linked list. The first block is the newest
 * block. Each block stores the neighbor nodes, timestamps of the edges and IDs
 * of edges. The neighbor nodes and corresponding edge ids are sorted by
 * timestamps. The block has a maximum capacity and can only store a certain
 * number of edges. The block can be moved to a different device.
 */
struct TemporalBlock {
  NIDType* dst_nodes;
  TimestampType* timestamps;
  EIDType* eids;

  std::size_t size;
  std::size_t capacity;

  TemporalBlock* prev;
  TemporalBlock* next;
};

static constexpr std::size_t kBlockSpaceSize =
    (sizeof(NIDType) + sizeof(EIDType) + sizeof(TimestampType));

}  // namespace dgnn
#endif  // DGNN_TEMPORAL_BLOCK_H_
