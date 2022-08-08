#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <functional>

#include "common.h"
#include "dynamic_graph.h"
#include "temporal_sampler.h"

namespace py = pybind11;

using namespace dgnn;

template <typename T>
inline py::array vec2npy(const std::vector<T> &vec) {
  // need to let python garbage collector handle C++ vector memory
  // see https://github.com/pybind/pybind11/issues/1042
  auto v = new std::vector<T>(vec);
  auto capsule = py::capsule(
      v, [](void *v) { delete reinterpret_cast<std::vector<T> *>(v); });
  return py::array(v->size(), v->data(), capsule);
}

PYBIND11_MODULE(libdgnn, m) {
  py::enum_<InsertionPolicy>(m, "InsertionPolicy")
      .value("INSERT", InsertionPolicy::kInsertionPolicyInsert)
      .value("REPLACE", InsertionPolicy::kInsertionPolicyReplace);

  py::enum_<SamplingPolicy>(m, "SamplingPolicy")
      .value("RECENT", SamplingPolicy::kSamplingPolicyRecent)
      .value("UNIFORM", SamplingPolicy::kSamplingPolicyUniform);

  py::enum_<MemoryResourceType>(m, "MemoryResourceType")
      .value("CUDA", MemoryResourceType::kMemoryResourceTypeCUDA)
      .value("UNIFIED", MemoryResourceType::kMemoryResourceTypeUnified)
      .value("PINNED", MemoryResourceType::kMemoryResourceTypePinned);

  py::class_<DynamicGraph>(m, "_DynamicGraph")
      .def(py::init<std::size_t, std::size_t, MemoryResourceType, std::size_t,
                    std::size_t, std::size_t, InsertionPolicy>(),
           py::arg("initial_pool_size"), py::arg("maximum_pool_size"),
           py::arg("mem_resource_type"),
           py::arg("initial_pool_size_for_metadata"),
           py::arg("maximum_pool_size_for_metadata"),
           py::arg("minium_block_size"), py::arg("insertion_policy"))
      .def("add_edges", &DynamicGraph::AddEdges, py::arg("source_vertices"),
           py::arg("target_vertices"), py::arg("timestamps"),
           py::arg("add_reverse") = true)
      .def("num_vertices", &DynamicGraph::num_nodes)
      .def("num_edges", &DynamicGraph::num_edges)
      .def("out_degree", &DynamicGraph::out_degree)
      .def("get_temporal_neighbors",
           [](const DynamicGraph &dgraph, NIDType node) {
             auto neighbors = dgraph.get_temporal_neighbors(node);
             return py::make_tuple(vec2npy(std::get<0>(neighbors)),
                                   vec2npy(std::get<1>(neighbors)),
                                   vec2npy(std::get<2>(neighbors)));
           });

  py::class_<SamplingResult>(m, "SamplingResult")
      .def("row",
           [](const SamplingResult &result) { return vec2npy(result.row); })
      .def("col",
           [](const SamplingResult &result) { return vec2npy(result.col); })
      .def("all_nodes",
           [](const SamplingResult &result) {
             return vec2npy(result.all_nodes);
           })
      .def("all_timestamps",
           [](const SamplingResult &result) {
             return vec2npy(result.all_timestamps);
           })
      .def("delta_timestamps",
           [](const SamplingResult &result) {
             return vec2npy(result.delta_timestamps);
           })
      .def("eids",
           [](const SamplingResult &result) { return vec2npy(result.eids); })
      .def("num_src_nodes",
           [](const SamplingResult &result) { return result.num_src_nodes; })
      .def("num_dst_nodes",
           [](const SamplingResult &result) { return result.num_dst_nodes; });

  py::class_<TemporalSampler>(m, "_TemporalSampler")
      .def(py::init<const DynamicGraph &, const std::vector<uint32_t> &,
                    SamplingPolicy, uint32_t, float, bool, uint64_t>(),
           py::arg("dgraph"), py::arg("fanouts"), py::arg("sampling_policy"),
           py::arg("num_snapshots"), py::arg("snapshot_time_window"),
           py::arg("prop_time"), py::arg("seed"))
      .def("sample", &TemporalSampler::Sample);
}
