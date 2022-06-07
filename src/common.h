#ifndef DGNN_COMMON_H_
#define DGNN_COMMON_H_

#include <cstddef>

namespace dgnn {

using NIDType = uint64_t;
using EIDType = uint64_t;
using TimestampType = float;

static constexpr std::size_t kBlockSpaceSize =
    (sizeof(NIDType) + sizeof(EIDType) + sizeof(TimestampType));

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

/**
 * @brief InsertionPolicy is used to decide how to insert a new temporal block
 * into the linked list.
 *
 * kInsertionPolicyInsert: insert the new block at the head of the list.
 * kInsertionPolicyReplace: replace the head block with a larger block.
 */
enum class InsertionPolicy { kInsertionPolicyInsert, kInsertionPolicyReplace };

static constexpr std::size_t kDefaultMaxGpuMemPoolSize = 1 << 30;  // 1 GiB
static constexpr InsertionPolicy kDefaultInsertionPolicy =
    InsertionPolicy::kInsertionPolicyInsert;
static constexpr std::size_t kDefaultAlignment = 16;

};  // namespace dgnn

#endif  // DGNN_COMMON_H_
