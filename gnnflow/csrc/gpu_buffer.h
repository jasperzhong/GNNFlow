#ifndef GNNFLOW_GPU_BUFFER_H_
#define GNNFLOW_GPU_BUFFER_H_

#include "buffer.h"

namespace gnnflow {

class GPUBuffer : public Buffer {
 public:
  GPUBuffer() = default;
  GPUBuffer(std::size_t size);
  ~GPUBuffer();

  void Allocate(std::size_t size) override;
  void Deallocate() override;
};

}  // namespace gnnflow

#endif  // GNNFLOW_GPU_BUFFER_H_
