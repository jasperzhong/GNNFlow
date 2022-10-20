#ifndef GNNFLOW_RESOURCE_HOLDER_H
#define GNNFLOW_RESOURCE_HOLDER_H

#include <cuda_runtime_api.h>
#include <curand_kernel.h>

#include "logging.h"
#include "utils.h"

namespace gnnflow {

template <typename T>
class ResourceHolder {
 public:
  ResourceHolder() = default;
  ~ResourceHolder() = default;

  // convert to T
  operator T() const& { return resource_; }

 protected:
  T resource_;
};

template <>
class ResourceHolder<cudaStream_t> {
 public:
  ResourceHolder() {
    CUDA_CALL(cudaStreamCreate(&resource_));
    CUDA_CALL(
        cudaStreamCreateWithPriority(&resource_, cudaStreamNonBlocking, -1))
  }

  ~ResourceHolder() { cudaStreamDestroy(resource_); }

  // convert to cudaStream_t
  operator cudaStream_t() const& { return resource_; }

 protected:
  cudaStream_t resource_;
};
typedef ResourceHolder<cudaStream_t> StreamHolder;

template <>
class ResourceHolder<char*> {
 public:
  ResourceHolder() = default;
  ResourceHolder(std::size_t size) : resource_(nullptr), size_(size) {}
  virtual ~ResourceHolder() = default;

  operator char*() const& { return resource_; }
  char* operator+(std::size_t offset) const { return resource_ + offset; }

  std::size_t size() const { return size_; }

 protected:
  char* resource_;
  std::size_t size_;
};
typedef ResourceHolder<char*> Buffer;

class GPUResourceHolder : public ResourceHolder<char*> {
 public:
  GPUResourceHolder() = default;
  GPUResourceHolder(std::size_t size) : ResourceHolder<char*>(size) {
    CUDA_CALL(cudaMalloc(&resource_, size));
  }
  ~GPUResourceHolder() { cudaFree(resource_); }
};
typedef GPUResourceHolder GPUBuffer;

class PinMemoryResourceHolder : public ResourceHolder<char*> {
 public:
  PinMemoryResourceHolder() = default;
  PinMemoryResourceHolder(std::size_t size) : ResourceHolder<char*>(size) {
    CUDA_CALL(cudaMallocHost(&resource_, size));
  }
  ~PinMemoryResourceHolder() { cudaFreeHost(resource_); }
};
typedef PinMemoryResourceHolder PinMemoryBuffer;

class CuRandStateResourceHolder : public ResourceHolder<curandState_t*> {
 public:
  CuRandStateResourceHolder() = default;
  CuRandStateResourceHolder(std::size_t num_elements, uint64_t seed) {
    CUDA_CALL(
        cudaMalloc((void**)&resource_, num_elements * sizeof(curandState_t)));
    uint32_t num_threads_per_block = 256;
    uint32_t num_blocks =
        (num_elements + num_threads_per_block - 1) / num_threads_per_block;

    InitCuRandStates<<<num_blocks, num_threads_per_block>>>(resource_,
                                                            num_elements, seed);
  }

  ~CuRandStateResourceHolder() { cudaFree(resource_); }

  operator curandState_t*() const& { return resource_; }
};
typedef CuRandStateResourceHolder CuRandStateHolder;

}  // namespace gnnflow

#endif  // GNNFLOW_RESOURCE_HOLDER_H
