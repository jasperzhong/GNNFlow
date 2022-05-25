#ifndef DGNN_COMMON_TEMPORAL_BLOCK_H_
#define DGNN_COMMON_TEMPORAL_BLOCK_H_

#include <thrust/copy.h>

#include <cstddef>
#include <rmm/mr/device/per_device_resource.hpp>
#include <vector>

namespace dgnn {
/** @brief This class is used to store the temporal blocks in the graph.
 *  The blocks are stored in a linked list. The first block is the newest block.
 *  Each block stores the neighbor vertices, timestamps of the edges and IDs of
 *  edges. The neighbor vertices and corresponding edge ids are sorted by
 *  timestamps. The block has a maximum capacity and can only store a certain
 *  number of edges. The block can be moved to a different device.
 *
 * @tparam NIDType The type of the vertex ID.
 * @tparam EIDType The type of the edge ID.
 * @tparam TimestampType The type of the timestamp.
 */
template <typename NIDType, typename EIDType, typename TimestampType>
class TemporalBlock {
 public:
  enum class Device { CPU, GPU };
  explicit TemporalBlock(std::size_t capacity, Device device);

  ~TemporalBlock();

  TemporalBlock(const TemporalBlock&) = delete;
  TemporalBlock<NIDType, EIDType, TimestampType>& operator=(
      const TemporalBlock&);

  /**
   * @brief Add new edges to the block.
   *
   * @param neighbor_vertices The target vertices of the edges.
   * @param timestamps The timestamps of the edges.
   */
  void AddEdges(const std::vector<NIDType>& neighbor_vertices,
                const std::vector<TimestampType>& timestamps);

  /**
   * @brief Move the block to a different device.
   *
   * @param device The device to move the block to.
   */
  void MoveTo(Device device);

  /**
   * @brief Get the number of edges in the block. @return The number of edges in
   * the block.
   */
  std::size_t size() const;

  /**
   * @brief Check if the block is full. @return True if the block is full.
   */
  bool is_full() const;

  /**
   * @brief Check if the block is empty. @return True if the block is empty.
   */
  bool is_empty() const;

  /**
   * @brief Get the space used by the block.
   * @return The space used by the block in bytes.
   */
  std::size_t get_mem_space() const;

  TemporalBlock<NIDType, EIDType, TimestampType>* get_next_block() const;

 private:
  void AllocateDeviceMemory() throw(rmm::bad_alloc);
  void DeallocateDeviceMemory();

  void AllocateHostMemory();
  void DeallocateHostMemory();

  NIDType* neighbor_vertices_;
  EIDType* neighbor_edges_;
  TimestampType* timestamps_;
  std::size_t size_;
  std::size_t capacity_;
  Device device_;
  TemporalBlock<NIDType, EIDType, TimestampType>* next_;
};

template <typename NIDType, typename EIDType, typename TimestampType>
TemporalBlock<NIDType, EIDType, TimestampType>::TemporalBlock(
    std::size_t capacity, Device device)
    : neighbor_vertices_(nullptr),
      timestamps_(nullptr),
      neighbor_edges_(nullptr),
      size_(0),
      capacity_(capacity),
      device_(device),
      next_(nullptr) {}

template <typename NIDType, typename EIDType, typename TimestampType>
TemporalBlock<NIDType, EIDType, TimestampType>::~TemporalBlock() {
  if (device_ == Device::GPU) {
    DeallocateDeviceMemory();
  } else {
    DeallocateHostMemory();
  }
}

template <typename NIDType, typename EIDType, typename TimestampType>
TemporalBlock<NIDType, EIDType, TimestampType>&
TemporalBlock<NIDType, EIDType, TimestampType>::operator=(
    const TemporalBlock& other) {
  if (device_ == Device::GPU) {
    DeallocateDeviceMemory();
  } else {
    DeallocateHostMemory();
  }
  size_ = other.size_;
  capacity_ = other.capacity_;
  device_ = other.device_;
  // NB: next_ pointer is not copied.
  next_ = nullptr;
  if (device_ == Device::GPU) {
    AllocateDeviceMemory();
    thrust::copy(other.neighbor_vertices_, other.neighbor_vertices_ + size_,
                 neighbor_vertices_);
    thrust::copy(other.neighbor_edges_, other.neighbor_edges_ + size_,
                 neighbor_edges_);
    thrust::copy(other.timestamps_, other.timestamps_ + size_, timestamps_);
  } else {
    AllocateHostMemory();
    std::copy(other.neighbor_vertices_, other.neighbor_vertices_ + size_,
              neighbor_vertices_);
    std::copy(other.neighbor_edges_, other.neighbor_edges_ + size_,
              neighbor_edges_);
    std::copy(other.timestamps_, other.timestamps_ + size_, timestamps_);
  }
  return *this;
}

template <typename NIDType, typename EIDType, typename TimestampType>
void TemporalBlock<NIDType, EIDType, TimestampType>::
    AllocateDeviceMemory() throw(rmm::bad_alloc) {
  auto mr = rmm::mr::get_current_device_resource();
  neighbor_vertices_ =
      static_cast<NIDType*>(mr->allocate(capacity_ * sizeof(NIDType)));

  neighbor_edges_ =
      static_cast<EIDType*>(mr->allocate(capacity_ * sizeof(EIDType)));

  timestamps_ = static_cast<TimestampType*>(
      mr->allocate(capacity_ * sizeof(TimestampType)));
}

template <typename NIDType, typename EIDType, typename TimestampType>
void TemporalBlock<NIDType, EIDType, TimestampType>::DeallocateDeviceMemory() {
  auto mr = rmm::mr::get_current_device_resource();
  if (neighbor_vertices_ != nullptr) {
    mr->deallocate(neighbor_vertices_, capacity_ * sizeof(NIDType));
    neighbor_vertices_ = nullptr;
  }

  if (neighbor_edges_ != nullptr) {
    mr->deallocate(neighbor_edges_, capacity_ * sizeof(EIDType));
    neighbor_edges_ = nullptr;
  }

  if (timestamps_ != nullptr) {
    mr->deallocate(timestamps_, capacity_ * sizeof(TimestampType));
    timestamps_ = nullptr;
  }
}

template <typename NIDType, typename EIDType, typename TimestampType>
void TemporalBlock<NIDType, EIDType, TimestampType>::AllocateHostMemory() {
  neighbor_vertices_ = new NIDType[capacity_];
  neighbor_edges_ = new EIDType[capacity_];
  timestamps_ = new TimestampType[capacity_];
}

template <typename NIDType, typename EIDType, typename TimestampType>
void TemporalBlock<NIDType, EIDType, TimestampType>::DeallocateHostMemory() {
  if (neighbor_vertices_ != nullptr) {
    delete[] neighbor_vertices_;
    neighbor_vertices_ = nullptr;
  }

  if (neighbor_edges_ != nullptr) {
    delete[] neighbor_edges_;
    neighbor_edges_ = nullptr;
  }

  if (timestamps_ != nullptr) {
    delete[] timestamps_;
    timestamps_ = nullptr;
  }
}

template <typename NIDType, typename EIDType, typename TimestampType>
std::size_t TemporalBlock<NIDType, EIDType, TimestampType>::get_mem_space()
    const {
  return capacity_ * sizeof(NIDType) + capacity_ * sizeof(EIDType) +
         capacity_ * sizeof(TimestampType);
}
}  // namespace dgnn
#endif  // DGNN_COMMON_TEMPORAL_BLOCK_H_
