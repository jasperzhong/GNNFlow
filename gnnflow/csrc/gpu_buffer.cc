#include "gpu_buffer.h"

#include <cuda_runtime_api.h>

#include "logging.h"

namespace gnnflow {

GPUBuffer::GPUBuffer(std::size_t size) : Buffer(size) { Allocate(size); }

GPUBuffer::~GPUBuffer() { Deallocate(); }

void GPUBuffer::Allocate(std::size_t size) {
  if (ptr_ != nullptr) {
    Deallocate();
  }
  CUDA_CALL(cudaMalloc((void**)&ptr_, size));
}

void GPUBuffer::Deallocate() {
  if (ptr_ != nullptr) {
    CUDA_CALL(cudaFree(ptr_));
    ptr_ = nullptr;
  }
}
}  // namespace gnnflow
