#ifndef DGNN_UTILS_H_
#define DGNN_UTILS_H_

#include <algorithm>
#include <numeric>
#include <vector>

namespace dgnn {

template <typename T>
std::vector<std::size_t> stable_sort_indices(const std::vector<T>& v) {
  // initialize original index locations
  std::vector<std::size_t> idx(v.size());
  std::iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  std::stable_sort(
      idx.begin(), idx.end(),
      [&v](std::size_t i1, std::size_t i2) { return v[i1] < v[i2]; });

  return idx;
}

template <typename T>
std::vector<T> sort_vector(const std::vector<T>& v,
                           const std::vector<std::size_t>& idx) {
  std::vector<T> sorted_v;
  sorted_v.reserve(v.size());
  for (auto i : idx) {
    sorted_v.emplace_back(v[i]);
  }
  return sorted_v;
}
}  // namespace dgnn

#endif  // DGNN_UTILS_H_
