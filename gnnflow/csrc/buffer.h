#ifndef GNNFLOW_BUFFER_H_
#define GNNFLOW_BUFFER_H_

#include <cstddef>

namespace gnnflow {

// Buffer is a wrapper of a pointer to a memory buffer.
class Buffer {
 public:
  Buffer() = default;
  Buffer(std::size_t size) : ptr_(nullptr), size_(size) {}
  virtual ~Buffer() = default;

  virtual void Allocate(std::size_t size) = 0;
  virtual void Deallocate() = 0;

  // conversion operators
  operator char*() const { return ptr_; }
  char* operator+(std::size_t offset) const { return ptr_ + offset; }

 protected:
  char* ptr_;
  std::size_t size_;
};

}  // namespace gnnflow

#endif  // GNNFLOW_BUFFER_H_
