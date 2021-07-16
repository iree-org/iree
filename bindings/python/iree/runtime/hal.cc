// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "bindings/python/iree/runtime/hal.h"

#include "iree/hal/api.h"

namespace iree {
namespace python {

//------------------------------------------------------------------------------
// HalDriver
//------------------------------------------------------------------------------

std::vector<std::string> HalDriver::Query() {
  iree_hal_driver_info_t* driver_infos = NULL;
  iree_host_size_t driver_info_count = 0;
  CheckApiStatus(
      iree_hal_driver_registry_enumerate(iree_hal_driver_registry_default(),
                                         iree_allocator_system(), &driver_infos,
                                         &driver_info_count),
      "Error enumerating HAL drivers");
  std::vector<std::string> driver_names(driver_info_count);
  for (iree_host_size_t i = 0; i < driver_info_count; ++i) {
    driver_names[i] = std::string(driver_infos[i].driver_name.data,
                                  driver_infos[i].driver_name.size);
  }
  iree_allocator_free(iree_allocator_system(), driver_infos);
  return driver_names;
}

HalDriver HalDriver::Create(const std::string& driver_name) {
  iree_hal_driver_t* driver;
  CheckApiStatus(iree_hal_driver_registry_try_create_by_name(
                     iree_hal_driver_registry_default(),
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
  py::enum_<enum iree_hal_memory_type_bits_t>(m, "MemoryType")
      .value("NONE", IREE_HAL_MEMORY_TYPE_NONE)
      .value("TRANSIENT", IREE_HAL_MEMORY_TYPE_TRANSIENT)
      .value("HOST_VISIBLE", IREE_HAL_MEMORY_TYPE_HOST_VISIBLE)
      .value("HOST_COHERENT", IREE_HAL_MEMORY_TYPE_HOST_COHERENT)
      .value("HOST_CACHED", IREE_HAL_MEMORY_TYPE_HOST_CACHED)
      .value("HOST_LOCAL", IREE_HAL_MEMORY_TYPE_HOST_LOCAL)
      .value("DEVICE_VISIBLE", IREE_HAL_MEMORY_TYPE_HOST_VISIBLE)
      .value("DEVICE_LOCAL", IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL)
      .export_values();
  py::enum_<enum iree_hal_buffer_usage_bits_t>(m, "BufferUsage")
      .value("NONE", IREE_HAL_BUFFER_USAGE_NONE)
      .value("CONSTANT", IREE_HAL_BUFFER_USAGE_CONSTANT)
      .value("TRANSFER", IREE_HAL_BUFFER_USAGE_TRANSFER)
      .value("MAPPING", IREE_HAL_BUFFER_USAGE_MAPPING)
      .value("DISPATCH", IREE_HAL_BUFFER_USAGE_DISPATCH)
      .value("ALL", IREE_HAL_BUFFER_USAGE_ALL)
      .export_values();
  py::enum_<enum iree_hal_memory_access_bits_t>(m, "MemoryAccess")
      .value("NONE", IREE_HAL_MEMORY_ACCESS_NONE)
      .value("READ", IREE_HAL_MEMORY_ACCESS_READ)
      .value("WRITE", IREE_HAL_MEMORY_ACCESS_WRITE)
      .value("DISCARD", IREE_HAL_MEMORY_ACCESS_DISCARD)
      .value("DISCARD_WRITE", IREE_HAL_MEMORY_ACCESS_DISCARD_WRITE)
      .value("ALL", IREE_HAL_MEMORY_ACCESS_ALL)
      .export_values();
  py::enum_<enum iree_hal_element_types_t>(m, "HalElementType")
      .value("NONE", IREE_HAL_ELEMENT_TYPE_NONE)
      .value("OPAQUE_8", IREE_HAL_ELEMENT_TYPE_OPAQUE_8)
      .value("OPAQUE_16", IREE_HAL_ELEMENT_TYPE_OPAQUE_16)
      .value("OPAQUE_32", IREE_HAL_ELEMENT_TYPE_OPAQUE_32)
      .value("OPAQUE_64", IREE_HAL_ELEMENT_TYPE_OPAQUE_64)
      .value("SINT_8", IREE_HAL_ELEMENT_TYPE_SINT_8)
      .value("SINT_16", IREE_HAL_ELEMENT_TYPE_SINT_16)
      .value("SINT_32", IREE_HAL_ELEMENT_TYPE_SINT_32)
      .value("SINT_64", IREE_HAL_ELEMENT_TYPE_SINT_64)
      .value("UINT_8", IREE_HAL_ELEMENT_TYPE_UINT_8)
      .value("UINT_16", IREE_HAL_ELEMENT_TYPE_UINT_16)
      .value("UINT_32", IREE_HAL_ELEMENT_TYPE_UINT_32)
      .value("UINT_64", IREE_HAL_ELEMENT_TYPE_UINT_64)
      .value("FLOAT_16", IREE_HAL_ELEMENT_TYPE_FLOAT_16)
      .value("FLOAT_32", IREE_HAL_ELEMENT_TYPE_FLOAT_32)
      .value("FLOAT_64", IREE_HAL_ELEMENT_TYPE_FLOAT_64)
      .value("BOOL_8",
             static_cast<iree_hal_element_types_t>(IREE_HAL_ELEMENT_TYPE_VALUE(
                 IREE_HAL_NUMERICAL_TYPE_INTEGER_SIGNED, 1)))
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
      .def("fill_zero", &HalBuffer::FillZero, py::arg("byte_offset"),
           py::arg("byte_length"))
      .def("create_view", &HalBuffer::CreateView, py::arg("shape"),
           py::arg("element_size"));
}

}  // namespace python
}  // namespace iree
