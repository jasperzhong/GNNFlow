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
  auto& first = store_[keys[0]];
  at::Tensor values =
      at::empty({static_cast<int64_t>(size), first.size(0)}, first.options());
  for (size_t i = 0; i < size; ++i) {
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
