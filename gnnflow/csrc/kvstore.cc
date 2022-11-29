#include "kvstore.h"

#include "utils.h"

namespace gnnflow {
void KVStore::set(const std::vector<Key>& keys, const at::Tensor& values) {
  const std::size_t size = keys.size();
  for (size_t i = 0; i < size; ++i) {
    store_[keys[i]] = values[i];
  }
}

at::Tensor KVStore::get(const std::vector<Key>& keys) {
  const auto size = keys.size();
  // sort the keys
  auto indices = stable_sort_indices(keys);
  auto sorted_keys = sort_vector(keys, indices);

  std::vector<at::Tensor> values(size);
  // #pragma omp parallel for num_threads(num_threads_) schedule(static)
  for (size_t i = 0; i < size; ++i) {
    values[i] = store_[sorted_keys[i]];
  }

  // restore values in the original order
  for (size_t i = 0; i < size; ++i) {
    values[indices[i]] = values[i];
  }

  return at::stack(values);
}
}  // namespace gnnflow
