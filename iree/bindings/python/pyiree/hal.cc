// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "iree/bindings/python/pyiree/hal.h"

namespace iree {
namespace python {

void SetupHalBindings(pybind11::module m) {
  // Enums.
  py::enum_<iree_hal_memory_type_t>(m, "MemoryType")
      .value("NONE", IREE_HAL_MEMORY_TYPE_NONE)
      .value("TRANSIENT", IREE_HAL_MEMORY_TYPE_TRANSIENT)
      .value("HOST_VISIBLE", IREE_HAL_MEMORY_TYPE_HOST_VISIBLE)
      .value("HOST_COHERENT", IREE_HAL_MEMORY_TYPE_HOST_COHERENT)
      .value("HOST_CACHED", IREE_HAL_MEMORY_TYPE_HOST_CACHED)
      .value("HOST_LOCAL", IREE_HAL_MEMORY_TYPE_HOST_LOCAL)
      .value("DEVICE_VISIBLE", IREE_HAL_MEMORY_TYPE_HOST_VISIBLE)
      .value("DEVICE_LOCAL", IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL)
      .export_values();
  py::enum_<iree_hal_buffer_usage_t>(m, "BufferUsage")
      .value("NONE", IREE_HAL_BUFFER_USAGE_NONE)
      .value("CONSTANT", IREE_HAL_BUFFER_USAGE_CONSTANT)
      .value("TRANSFER", IREE_HAL_BUFFER_USAGE_TRANSFER)
      .value("MAPPING", IREE_HAL_BUFFER_USAGE_MAPPING)
      .value("DISPATCH", IREE_HAL_BUFFER_USAGE_DISPATCH)
      .value("ALL", IREE_HAL_BUFFER_USAGE_ALL)
      .export_values();
  py::enum_<iree_hal_memory_access_t>(m, "MemoryAccess")
      .value("NONE", IREE_HAL_MEMORY_ACCESS_NONE)
      .value("READ", IREE_HAL_MEMORY_ACCESS_READ)
      .value("WRITE", IREE_HAL_MEMORY_ACCESS_WRITE)
      .value("DISCARD", IREE_HAL_MEMORY_ACCESS_DISCARD)
      .value("DISCARD_WRITE", IREE_HAL_MEMORY_ACCESS_DISCARD_WRITE)
      .value("ALL", IREE_HAL_MEMORY_ACCESS_ALL)
      .export_values();

  py::class_<HalShape>(m, "Shape").def(py::init(&HalShape::FromIntVector));
  py::class_<HalBufferView>(m, "BufferView");
  py::class_<HalBuffer>(m, "Buffer")
      .def_static("allocate_heap", &HalBuffer::AllocateHeapBuffer,
                  py::arg("memory_type"), py::arg("usage"),
                  py::arg("allocation_size"))
      .def("fill_zero", &HalBuffer::FillZero, py::arg("byte_offset"),
           py::arg("byte_length"))
      .def("create_view", &HalBuffer::CreateView, py::arg("shape"),
           py::arg("element_size"));
}

}  // namespace python
}  // namespace iree
