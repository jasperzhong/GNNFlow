#include "kvstore.h"

#include "utils.h"

namespace gnnflow {
void KVStore::set(const std::vector<Key>& keys, const at::Tensor& values) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto num_keys = keys.size();
  for (std::size_t i = 0; i < num_keys; ++i) {
    store_[keys[i]] = values[i];
  }
}

std::vector<at::Tensor> KVStore::get(const std::vector<Key>& keys) {
  auto num_keys = keys.size();
  std::vector<at::Tensor> values(num_keys);
  for (size_t i = 0; i < num_keys; ++i) {
    values[i] = store_[keys[i]];
  }
  return values;
}

void KVStore::fill_zeros() {
  for (auto it = store_.begin(); it != store_.end(); ++it) {
    it->second.fill_(0);
  }
}
}  // namespace gnnflow
