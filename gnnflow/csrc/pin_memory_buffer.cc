#include "pin_memory_buffer.h"

#include <cuda_runtime_api.h>

#include "logging.h"

namespace gnnflow {

PinMemoryBuffer::PinMemoryBuffer(std::size_t size) : Buffer(size) {
  Allocate(size);
}

PinMemoryBuffer::~PinMemoryBuffer() { Deallocate(); }

void PinMemoryBuffer::Allocate(std::size_t size) {
  if (ptr_ != nullptr) {
    Deallocate();
  }
  CUDA_CALL(cudaMallocHost((void**)&ptr_, size));
}

void PinMemoryBuffer::Deallocate() {
  if (ptr_ != nullptr) {
    CUDA_CALL(cudaFreeHost(ptr_));
    ptr_ = nullptr;
  }
}
}  // namespace gnnflow
