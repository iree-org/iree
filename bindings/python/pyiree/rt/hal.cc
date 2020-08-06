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

#include "bindings/python/pyiree/rt/hal.h"

#include "absl/container/inlined_vector.h"
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
      IREE_CHECK_OK(iree_hal_buffer_unmap(buffer, &mapped_memory_));
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
    absl::InlinedVector<int32_t, 6> shape(iree_hal_buffer_view_shape_rank(bv_));
    CheckApiStatus(
        iree_hal_buffer_view_shape(bv_, shape.size(), shape.data(), nullptr),
        "Error getting buffer view shape");
    iree_hal_element_type_t element_type =
        iree_hal_buffer_view_element_type(bv_);
    int32_t element_size = iree_hal_element_byte_count(element_type);
    absl::InlinedVector<py::ssize_t, 6> dims(shape.size());
    for (int i = 0; i < shape.size(); ++i) {
      dims[i] = shape[i];
    }
    absl::InlinedVector<py::ssize_t, 8> strides(shape.size());
    if (!strides.empty()) {
      strides[shape.size() - 1] = element_size;
      for (int i = shape.size() - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
      }
    }

    // TODO(laurenzo): We need to figure out how to propagate dtype in the
    // buffer view.
    return py::buffer_info(
        mapped_memory_.contents.data, element_size,
        py::format_descriptor<float>::format(),  // TODO(laurenzo): DTYPE!
        shape.size(), dims, strides);
  }

 private:
  iree_hal_mapped_memory_t mapped_memory_;
  iree_hal_buffer_view_t* bv_;
};

}  // namespace

//------------------------------------------------------------------------------
// HalDriver
//------------------------------------------------------------------------------

std::vector<std::string> HalDriver::Query() {
  iree_string_view_t* driver_names;
  iree_host_size_t driver_count;
  CheckApiStatus(iree_hal_driver_registry_query_available_drivers(
                     iree_allocator_system(), &driver_names, &driver_count),
                 "Error querying drivers");

  std::vector<std::string> drivers;
  drivers.resize(driver_count);
  for (iree_host_size_t i = 0; i < driver_count; ++i) {
    drivers[i] = std::string(driver_names[i].data, driver_names[i].size);
  }
  free(driver_names);
  return drivers;
}

HalDriver HalDriver::Create(const std::string& driver_name) {
  iree_hal_driver_t* driver;
  CheckApiStatus(iree_hal_driver_registry_create_driver(
                     {driver_name.data(), driver_name.size()},
                     iree_allocator_system(), &driver),
                 "Error creating driver");
  return HalDriver::CreateRetained(driver);
}

HalDevice HalDriver::CreateDefaultDevice() {
  iree_hal_device_t* device;
  CheckApiStatus(iree_hal_driver_create_default_device(
                     raw_ptr(), iree_allocator_system(), &device),
                 "Error creating default device");
  return HalDevice::CreateRetained(device);
}

void SetupHalBindings(pybind11::module m) {
  // Enums.
  py::enum_<enum iree_hal_memory_type_e>(m, "MemoryType")
      .value("NONE", IREE_HAL_MEMORY_TYPE_NONE)
      .value("TRANSIENT", IREE_HAL_MEMORY_TYPE_TRANSIENT)
      .value("HOST_VISIBLE", IREE_HAL_MEMORY_TYPE_HOST_VISIBLE)
      .value("HOST_COHERENT", IREE_HAL_MEMORY_TYPE_HOST_COHERENT)
      .value("HOST_CACHED", IREE_HAL_MEMORY_TYPE_HOST_CACHED)
      .value("HOST_LOCAL", IREE_HAL_MEMORY_TYPE_HOST_LOCAL)
      .value("DEVICE_VISIBLE", IREE_HAL_MEMORY_TYPE_HOST_VISIBLE)
      .value("DEVICE_LOCAL", IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL)
      .export_values();
  py::enum_<enum iree_hal_buffer_usage_e>(m, "BufferUsage")
      .value("NONE", IREE_HAL_BUFFER_USAGE_NONE)
      .value("CONSTANT", IREE_HAL_BUFFER_USAGE_CONSTANT)
      .value("TRANSFER", IREE_HAL_BUFFER_USAGE_TRANSFER)
      .value("MAPPING", IREE_HAL_BUFFER_USAGE_MAPPING)
      .value("DISPATCH", IREE_HAL_BUFFER_USAGE_DISPATCH)
      .value("ALL", IREE_HAL_BUFFER_USAGE_ALL)
      .export_values();
  py::enum_<enum iree_hal_memory_access_e>(m, "MemoryAccess")
      .value("NONE", IREE_HAL_MEMORY_ACCESS_NONE)
      .value("READ", IREE_HAL_MEMORY_ACCESS_READ)
      .value("WRITE", IREE_HAL_MEMORY_ACCESS_WRITE)
      .value("DISCARD", IREE_HAL_MEMORY_ACCESS_DISCARD)
      .value("DISCARD_WRITE", IREE_HAL_MEMORY_ACCESS_DISCARD_WRITE)
      .value("ALL", IREE_HAL_MEMORY_ACCESS_ALL)
      .export_values();

  py::class_<HalDevice>(m, "HalDevice");
  py::class_<HalDriver>(m, "HalDriver")
      .def_static("query", &HalDriver::Query)
      .def_static("create", &HalDriver::Create, py::arg("driver_name"))
      .def("create_default_device", &HalDriver::CreateDefaultDevice);

  py::class_<HalShape>(m, "Shape").def(py::init(&HalShape::FromIntVector));
  py::class_<HalBufferView>(m, "BufferView")
      .def("map", HalMappedMemory::Create);
  py::class_<HalMappedMemory>(m, "MappedMemory", py::buffer_protocol())
      .def_buffer(&HalMappedMemory::ToBufferInfo);
  py::class_<HalBuffer>(m, "HalBuffer")
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
