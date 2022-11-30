#include "kvstore.h"

#include "utils.h"

namespace gnnflow {
void KVStore::set(py::list keys, const at::Tensor& values) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto num_keys = static_cast<std::size_t>(py::len(keys));
  for (std::size_t i = 0; i < num_keys; ++i) {
    auto key = py::cast<Key>(keys[i]);
    store_[key] = values[i];
  }
}

py::list KVStore::get(py::list keys) {
  auto num_keys = static_cast<std::size_t>(py::len(keys));
  std::vector<at::Tensor> values(num_keys);
#pragma omp parallel for num_threads(4)
  for (size_t i = 0; i < num_keys; ++i) {
    values[i] = store_[py::cast<Key>(keys[i])];
  }

  return py::cast(values);
}

void KVStore::fill_zeros() {
  for (auto it = store_.begin(); it != store_.end(); ++it) {
    it->second.fill_(0);
  }
}
}  // namespace gnnflow
