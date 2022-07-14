#ifndef DGNN_COMMON_H_
#define DGNN_COMMON_H_

#include <cstddef>
#include <vector>

namespace dgnn {

// NIDType is the type of node ID.
// TimestampType is the type of timestamp.
// EIDType is the type of edge ID.
// NB: PyTorch does not support converting uint64_t's numpy ndarray to int64_t.
using NIDType = int64_t;
using TimestampType = float;
using EIDType = int64_t;

constexpr int kMaxFanout = 32;

constexpr NIDType kInvalidNID = -1;

constexpr int kNumStreams = 8;

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

/** @brief This POD is used to store the sampling result. */
struct SamplingResult {
  std::vector<NIDType> row;
  std::vector<NIDType> col;
  std::vector<NIDType> all_nodes;
  std::vector<TimestampType> all_timestamps;
  std::vector<TimestampType> delta_timestamps;
  std::vector<EIDType> eids;
  std::size_t num_src_nodes;
  std::size_t num_dst_nodes;
};

struct SamplingRange {
  int start_idx;
  int end_idx;
};

/**
 * @brief InsertionPolicy is used to decide how to insert a new temporal block
 * into the linked list.
 *
 * kInsertionPolicyInsert: insert the new block at the head of the list.
 * kInsertionPolicyReplace: replace the head block with a larger block.
 */
enum class InsertionPolicy { kInsertionPolicyInsert, kInsertionPolicyReplace };

/**
 * @brief SamplePolicy is used to decide how to sample the dynamic graph.
 *
 * kSamplePolicyRecent: sample the most recent edges.
 * kSamplePolicyUniform: sample past edges uniformly.
 */
enum class SamplingPolicy { kSamplingPolicyRecent, kSamplingPolicyUniform };

static constexpr std::size_t kDefaultMaxGpuMemPoolSize = 1 << 30;  // 1 GiB
static constexpr InsertionPolicy kDefaultInsertionPolicy =
    InsertionPolicy::kInsertionPolicyInsert;

// 256 is the alignent size of rmm.
// 64 * sizeof(float) = 256 Bytes
static constexpr std::size_t kDefaultMinBlockSize = 64;

};  // namespace dgnn

#endif  // DGNN_COMMON_H_
