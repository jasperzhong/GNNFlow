#include "stream_holder.h"

#include "logging.h"

namespace gnnflow {
StreamHolder::StreamHolder() {
  CUDA_CALL(cudaStreamCreate(&stream_));
  CUDA_CALL(cudaStreamCreateWithPriority(&stream_, cudaStreamNonBlocking, -1))
}
StreamHolder::~StreamHolder() { cudaStreamDestroy(stream_); }

StreamHolder::operator cudaStream_t() const & { return stream_; }

}  // namespace gnnflow
