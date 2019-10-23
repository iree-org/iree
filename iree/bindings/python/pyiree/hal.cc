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

#include "iree/hal/api.h"

namespace iree {
namespace python {

namespace {

class HalMappedMemory {
 public:
  HalMappedMemory(iree_hal_mapped_memory_t mapped_memory,
                  iree_hal_buffer_view_t* bv)
      : mapped_memory_(mapped_memory), bv_(bv) {
    iree_hal_buffer_view_retain(bv_);
  }
  ~HalMappedMemory() {
    if (bv_) {
      iree_hal_buffer_t* buffer = iree_hal_buffer_view_buffer(bv_);
      CHECK_EQ(iree_hal_buffer_unmap(buffer, &mapped_memory_), IREE_STATUS_OK);
      iree_hal_buffer_view_release(bv_);
    }
  }
  HalMappedMemory(HalMappedMemory&& other)
      : mapped_memory_(other.mapped_memory_), bv_(other.bv_) {
    other.bv_ = nullptr;
  }

  static HalMappedMemory Create(HalBufferView& bv) {
    iree_hal_buffer_t* buffer = iree_hal_buffer_view_buffer(bv.raw_ptr());
    iree_device_size_t byte_length = iree_hal_buffer_byte_length(buffer);
    iree_hal_mapped_memory_t mapped_memory;
    CheckApiStatus(iree_hal_buffer_map(buffer, IREE_HAL_MEMORY_ACCESS_READ,
                                       0 /* element_offset */, byte_length,
                                       &mapped_memory),
                   "Could not map memory");
    return HalMappedMemory(mapped_memory, bv.raw_ptr());
  }

  py::buffer_info ToBufferInfo() {
    iree_hal_buffer_t* buffer = iree_hal_buffer_view_buffer(bv_);
    iree_shape_t shape;
    CheckApiStatus(iree_hal_buffer_view_shape(bv_, &shape),
                   "Error getting buffer view shape");
    int8_t element_size = iree_hal_buffer_view_element_size(bv_);
    iree_device_size_t byte_length = iree_hal_buffer_byte_length(buffer);
    absl::InlinedVector<ssize_t, IREE_SHAPE_MAX_RANK> dims;
    dims.resize(shape.rank);
    for (int i = 0; i < shape.rank; ++i) {
      dims[i] = shape.dims[i];
    }
    absl::InlinedVector<ssize_t, IREE_SHAPE_MAX_RANK> strides;
    strides.resize(shape.rank);
    for (int i = 1; i < shape.rank; ++i) {
      strides[i - 1] = shape.dims[i] * element_size;
    }
    if (!strides.empty()) {
      strides.back() = 1 * element_size;
    }

    // TODO(laurenzo): We need to figure out how to propagate dtype in the
    // buffer view.
    return py::buffer_info(
        mapped_memory_.contents.data, element_size,
        py::format_descriptor<float>::format(),  // TODO(laurenzo): DTYPE!
        shape.rank, dims, strides);
  }

 private:
  iree_hal_mapped_memory_t mapped_memory_;
  iree_hal_buffer_view_t* bv_;
};

}  // namespace

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
  py::class_<HalBufferView>(m, "BufferView")
      .def("map", HalMappedMemory::Create);
  py::class_<HalMappedMemory>(m, "MappedMemory", py::buffer_protocol())
      .def_buffer(&HalMappedMemory::ToBufferInfo);
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
