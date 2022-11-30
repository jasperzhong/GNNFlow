#ifndef GNNFLOW_KVSTORE_H
#define GNNFLOW_KVSTORE_H

#include <torch/extension.h>

#include <mutex>
#include <vector>

#include "absl/container/flat_hash_map.h"

namespace gnnflow {

class KVStore {
 public:
  using Key = unsigned int;
  KVStore() = default;
  ~KVStore() = default;

  void set(const std::vector<Key>& keys, const at::Tensor& values);

  std::vector<at::Tensor> get(const std::vector<Key>& keys);

  void fill_zeros();

  std::size_t memory_usage() const {
    // only count the memory usage of the map
    std::size_t total = (sizeof(Key) + sizeof(at::Tensor)) * store_.size();
    return total;
  }

 private:
  absl::flat_hash_map<Key, at::Tensor> store_;
  std::mutex mutex_;
};

}  // namespace gnnflow
#endif  // GNNFLOW_KVSTORE_H
