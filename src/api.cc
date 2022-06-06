#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "dynamic_graph.h"

namespace py = pybind11;
using namespace dgnn;

PYBIND11_MODULE(dgnn, m) {
  py::enum_<InsertionPolicy>(m, "InsertionPolicy")
    .value("insert", InsertionPolicy::kInsertionPolicyInsert)
    .value("replace", InsertionPolicy::kInsertionPolicyReplace);

  py::class_<DynamicGraph>(m, "DynamicGraph")
      .def(py::init<std::size_t, std::size_t, InsertionPolicy>())
      .def("add_edges", &DynamicGraph::AddEdges);
}


