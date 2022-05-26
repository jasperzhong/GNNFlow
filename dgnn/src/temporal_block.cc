#include "temporal_block.h"

#include <thrust/copy.h>

#include <rmm/mr/device/per_device_resource.hpp>

namespace dgnn {

TemporalBlock::TemporalBlock(std::size_t capacity, Device device)
    : neighbor_vertices_(nullptr),
      neighbor_edges_(nullptr),
      timestamps_(nullptr),
      size_(0),
      capacity_(capacity),
      device_(device),
      next_(nullptr) {}

TemporalBlock& TemporalBlock::operator=(const TemporalBlock& other) {
  DeallocateMemory();
  size_ = other.size_;
  capacity_ = other.capacity_;
  device_ = other.device_;
  // NB: next_ pointer is not copied.
  next_ = nullptr;

  AllocateMemory();
  if (device_ == Device::GPU) {
    thrust::copy(other.neighbor_vertices_, other.neighbor_vertices_ + size_,
                 neighbor_vertices_);
    thrust::copy(other.neighbor_edges_, other.neighbor_edges_ + size_,
                 neighbor_edges_);
    thrust::copy(other.timestamps_, other.timestamps_ + size_, timestamps_);
  } else {
    std::copy(other.neighbor_vertices_, other.neighbor_vertices_ + size_,
              neighbor_vertices_);
    std::copy(other.neighbor_edges_, other.neighbor_edges_ + size_,
              neighbor_edges_);
    std::copy(other.timestamps_, other.timestamps_ + size_, timestamps_);
  }
  return *this;
}

void TemporalBlock::AddEdges(const std::vector<NIDType>& neighbor_vertices,
                             const std::vector<EIDType>& neighbor_edges,
                             const std::vector<TimestampType>& timestamps) {
  CHECK(neighbor_vertices.size() == neighbor_edges.size() == timestamps.size());
  CHECK_GT(size_ + neighbor_vertices.size(), capacity_);

  // defer memory allocation until the first edge is added
  if (neighbor_vertices_ == nullptr || neighbor_edges_ == nullptr ||
      timestamps_ == nullptr) {
    AllocateMemory();
  }

  if (device_ == Device::GPU) {
    thrust::copy(neighbor_vertices.begin(), neighbor_vertices.end(),
                 neighbor_vertices_ + size_);
    thrust::copy(neighbor_edges.begin(), neighbor_edges.end(),
                 neighbor_edges_ + size_);
    thrust::copy(timestamps.begin(), timestamps.end(), timestamps_ + size_);
  } else {
    std::copy(neighbor_vertices.begin(), neighbor_vertices.end(),
              neighbor_vertices_ + size_);
    std::copy(neighbor_edges.begin(), neighbor_edges.end(),
              neighbor_edges_ + size_);
    std::copy(timestamps.begin(), timestamps.end(), timestamps_ + size_);
  }
  size_ += neighbor_vertices.size();
}

void TemporalBlock::MoveTo(Device device) {
  if (device_ == device) {
    return;
  }

  if (device == Device::CPU) {
    // 1. copy to a temporary buffer on host
    TemporalBlock temp(capacity_, Device::CPU);
    temp.AllocateMemory();
    temp.size_ = size_;
    thrust::copy(neighbor_vertices_, neighbor_vertices_ + size_,
                 temp.neighbor_vertices_);
    thrust::copy(neighbor_edges_, neighbor_edges_ + size_,
                 temp.neighbor_edges_);
    thrust::copy(timestamps_, timestamps_ + size_, temp.timestamps_);

    // 2. deallocate device memory
    DeallocateMemory();

    // 3. pointers point to host
    neighbor_vertices_ = temp.neighbor_vertices_;
    neighbor_edges_ = temp.neighbor_edges_;
    timestamps_ = temp.timestamps_;
    device_ = Device::CPU;
  } else {
    // 1. copy to a temporary buffer on device
    TemporalBlock temp(capacity_, Device::GPU);
    temp.AllocateMemory();
    temp.size_ = size_;
    thrust::copy(neighbor_vertices_, neighbor_vertices_ + size_,
                 temp.neighbor_vertices_);
    thrust::copy(neighbor_edges_, neighbor_edges_ + size_,
                 temp.neighbor_edges_);
    thrust::copy(timestamps_, timestamps_ + size_, temp.timestamps_);

    // 2. deallocate host memory
    DeallocateMemory();

    // 3. pointers point to device
    neighbor_vertices_ = temp.neighbor_vertices_;
    neighbor_edges_ = temp.neighbor_edges_;
    timestamps_ = temp.timestamps_;
    device_ = Device::GPU;
  }
}

bool TemporalBlock::IsEmpty() const { return size_ == 0; }

bool TemporalBlock::IsFull() const { return size_ == capacity_; }

std::size_t TemporalBlock::get_size() const { return size_; }

TemporalBlock* TemporalBlock::get_next_block() const { return next_; }

void TemporalBlock::AllocateMemory() noexcept(false) {
  if (device_ == Device::GPU) {
    auto mr = rmm::mr::get_current_device_resource();
    // NB: may throw rmm::bad_alloc if allocation fails
    neighbor_vertices_ =
        static_cast<NIDType*>(mr->allocate(capacity_ * sizeof(NIDType)));
    neighbor_edges_ =
        static_cast<EIDType*>(mr->allocate(capacity_ * sizeof(EIDType)));
    timestamps_ = static_cast<TimestampType*>(
        mr->allocate(capacity_ * sizeof(TimestampType)));
  } else {
    neighbor_vertices_ = new NIDType[capacity_];
    neighbor_edges_ = new EIDType[capacity_];
    timestamps_ = new TimestampType[capacity_];
  }
}

void TemporalBlock::DeallocateMemory() {
  if (device_ == Device::GPU) {
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
  } else {
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
}

std::size_t TemporalBlock::get_mem_space() const {
  return capacity_ * sizeof(NIDType) + capacity_ * sizeof(EIDType) +
         capacity_ * sizeof(TimestampType);
}
}  // namespace dgnn
