#include "kvstore.h"

#include <chrono>
#include <iostream>

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
  // auto indices = stable_sort_indices(keys);
  // auto sorted_keys = sort_vector(keys, indices);

  auto start = std::chrono::system_clock::now();
  std::vector<at::Tensor> values(size);
  for (size_t i = 0; i < size; ++i) {
    values[i] = store_[keys[i]];
    // values[indices[i]] = store_[sorted_keys[i]];
  }
  lookup_time_ += std::chrono::duration_cast<std::chrono::microseconds>(
                      std::chrono::system_clock::now() - start)
                      .count();

  start = std::chrono::system_clock::now();
  auto tensor = at::stack(values);
  stack_time_ += std::chrono::duration_cast<std::chrono::microseconds>(
                     std::chrono::system_clock::now() - start)
                     .count();

  std::cout << "lookup time: " << lookup_time_ << " us"
            << "\tstack time: " << stack_time_ << " us" << std::endl;
  return tensor;
}
}  // namespace gnnflow