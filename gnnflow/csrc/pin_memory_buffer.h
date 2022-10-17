#ifndef GNNFLOW_PIN_MEMORY_BUFFER_H_
#define GNNFLOW_PIN_MEMORY_BUFFER_H_

#include "buffer.h"

namespace gnnflow {

class PinMemoryBuffer : public Buffer {
 public:
  PinMemoryBuffer() = default;
  PinMemoryBuffer(std::size_t size);
  ~PinMemoryBuffer();

  void Allocate(std::size_t size) override;
  void Deallocate() override;
};

}  // namespace gnnflow

#endif  // GNNFLOW_PIN_MEMORY_BUFFER_H_
