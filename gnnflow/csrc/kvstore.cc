#include "kvstore.h"

namespace gnnflow {
void KVStore::set(const std::vector<Key>& keys, const at::Tensor& values) {
  const std::size_t size = keys.size();
  for (size_t i = 0; i < size; ++i) {
    store_[keys[i]] = values[i];
  }
}

at::Tensor KVStore::get(const std::vector<Key>& keys) const {
  const auto size = keys.size();
  std::vector<at::Tensor> values(size);
  // #pragma omp parallel for num_threads(num_threads_) schedule(static)
  for (size_t i = 0; i < size; ++i) {
    values[i] = store_.at(keys[i]);
  }
  return at::stack(values);
}
}  // namespace gnnflow
