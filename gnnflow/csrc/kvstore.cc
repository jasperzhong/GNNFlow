#include "kvstore.h"

#include "utils.h"

namespace gnnflow {
void KVStore::set(const std::vector<Key>& keys, const at::Tensor& values) {
  std::lock_guard<std::mutex> lock(mutex_);
  const std::size_t size = keys.size();
  for (size_t i = 0; i < size; ++i) {
    store_[keys[i]] = values[i];
  }
}

at::Tensor KVStore::get(const std::vector<Key>& keys) {
  auto size = keys.size();
  // sort the keys
  auto indices = stable_sort_indices(keys);
  auto sorted_keys = sort_vector(keys, indices);

  std::vector<at::Tensor> values(size);
  for (size_t i = 0; i < size; ++i) {
    values[indices[i]] = store_[sorted_keys[i]];
  }

  // cat then reshape
  auto tensor = at::cat(values).reshape({static_cast<int64_t>(size), -1});

  return tensor;
}
}  // namespace gnnflow
