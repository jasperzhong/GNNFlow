#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "dynamic_graph.h"

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

PYBIND11_MODULE(dgnn, m) {
  py::enum_<InsertionPolicy>(m, "InsertionPolicy")
      .value("insert", InsertionPolicy::kInsertionPolicyInsert)
      .value("replace", InsertionPolicy::kInsertionPolicyReplace);

  py::class_<DynamicGraph>(m, "DynamicGraph")
      .def(
          py::init<std::size_t, std::size_t, InsertionPolicy>(),
          py::arg("max_gpu_pool_size") = kDefaultMaxGpuMemPoolSize,
          py::arg("min_block_size") = kDefaultMinBlockSize,
          py::arg("insertion_policy") = InsertionPolicy::kInsertionPolicyInsert)
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
}
