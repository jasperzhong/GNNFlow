#ifndef GNNFLOW_STREAM_HOLDER_H
#define GNNFLOW_STREAM_HOLDER_H

#include <cuda_runtime_api.h>

namespace gnnflow {

class StreamHolder {
 public:
  StreamHolder();

  ~StreamHolder();

  // convert to cudaStream_t
  operator cudaStream_t() const &;

 private:
  cudaStream_t stream_;
};

}  // namespace gnnflow

#endif  // GNNFLOW_STREAM_HOLDER_H
