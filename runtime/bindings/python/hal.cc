// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "./hal.h"

#include "iree/base/tracing.h"
#include "iree/hal/api.h"
#include "pybind11/numpy.h"

namespace iree {
namespace python {

namespace {

// RAII wrapper for a Py_buffer which calls PyBuffer_Release when it goes
// out of scope.
class PyBufferReleaser {
 public:
  PyBufferReleaser(Py_buffer& b) : b_(b) {}
  ~PyBufferReleaser() { PyBuffer_Release(&b_); }

 private:
  Py_buffer& b_;
};

static std::string ToHexString(const uint8_t* data, size_t length) {
  static constexpr char kHexChars[] = {'0', '1', '2', '3', '4', '5', '6', '7',
                                       '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'};
  std::string s(length * 2, ' ');
  for (size_t i = 0; i < length; ++i) {
    s[2 * i + 0] = kHexChars[(data[i] & 0xF0) >> 4];
    s[2 * i + 1] = kHexChars[(data[i] & 0x0F) >> 0];
  }
  return s;
}
static std::string ToHexString(uint32_t value) {
  return ToHexString((const uint8_t*)&value, sizeof(value));
}

}  // namespace

//------------------------------------------------------------------------------
// HalAllocator
//------------------------------------------------------------------------------

py::dict HalAllocator::QueryStatistics() {
  py::dict items;
  iree_hal_allocator_statistics_t stats;
  iree_hal_allocator_query_statistics(raw_ptr(), &stats);
#if IREE_STATISTICS_ENABLE
  items["host_bytes_peak"] = stats.host_bytes_peak;
  items["host_bytes_allocated"] = stats.host_bytes_allocated;
  items["host_bytes_freed"] = stats.host_bytes_freed;
  items["device_bytes_peak"] = stats.device_bytes_peak;
  items["device_bytes_allocated"] = stats.device_bytes_allocated;
  items["device_bytes_freed"] = stats.device_bytes_freed;
#endif
  return items;
}

py::str HalAllocator::FormattedStatistics() {
  // Perform all allocating string manipulation without early exit.
  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  iree_hal_allocator_statistics_t stats;
  iree_hal_allocator_query_statistics(raw_ptr(), &stats);
  auto status = iree_hal_allocator_statistics_format(&stats, &builder);
  iree_string_view_t view = iree_string_builder_view(&builder);
  py::str result = py::str(view.data, view.size);
  iree_string_builder_deinitialize(&builder);

  // Check/raise after all memory alloc/dealloc.
  CheckApiStatus(status, "unable to format statistics");
  return result;
}

py::object HalAllocator::AllocateBufferCopy(
    int memory_type, int allowed_usage, py::object buffer,
    std::optional<iree_hal_element_types_t> element_type) {
  IREE_TRACE_SCOPE0("HalAllocator::AllocateBufferCopy");
  // Request a view of the buffer (use the raw python C API to avoid
  // some allocation and copying at the pybind level).
  Py_buffer py_view;
  // Note that only C-Contiguous ND-arrays are presently supported, so
  // only request that via PyBUF_ND. Long term, we should consult an
  // "oracle" in the runtime to determine the precise required format
  // and set flags accordingly (and fallback/copy on failure).
  int flags = PyBUF_FORMAT | PyBUF_ND;

  // Acquire the backing buffer and setup RAII release.
  if (PyObject_GetBuffer(buffer.ptr(), &py_view, flags) != 0) {
    // The GetBuffer call is required to set an appropriate error.
    throw py::error_already_set();
  }
  PyBufferReleaser py_view_releaser(py_view);

  iree_hal_buffer_params_t params = {0};
  // TODO: Should not require host visible :(
  params.type = memory_type | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
  params.usage = allowed_usage;

  iree_hal_buffer_t* hal_buffer = nullptr;
  iree_status_t status = iree_ok_status();
  {
    py::gil_scoped_release release;
    status = iree_hal_allocator_allocate_buffer(
        raw_ptr(), params, py_view.len,
        iree_make_const_byte_span(py_view.buf, py_view.len), &hal_buffer);
  }
  CheckApiStatus(status, "Failed to allocate device visible buffer");

  if (!element_type) {
    return py::cast(HalBuffer::StealFromRawPtr(hal_buffer),
                    py::return_value_policy::move);
  }

  // Create the buffer_view. (note that numpy shape is ssize_t, so we need to
  // copy).
  iree_hal_encoding_type_t encoding_type =
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR;
  std::vector<iree_hal_dim_t> dims(py_view.ndim);
  std::copy(py_view.shape, py_view.shape + py_view.ndim, dims.begin());
  iree_hal_buffer_view_t* hal_buffer_view;
  CheckApiStatus(
      iree_hal_buffer_view_create(
          hal_buffer, dims.data(), dims.size(), *element_type, encoding_type,
          iree_hal_allocator_host_allocator(raw_ptr()), &hal_buffer_view),
      "Error allocating buffer_view");
  iree_hal_buffer_release(hal_buffer);

  return py::cast(HalBufferView::StealFromRawPtr(hal_buffer_view),
                  py::return_value_policy::move);
}

//------------------------------------------------------------------------------
// HalBuffer
//------------------------------------------------------------------------------

namespace {

void AppendHalBufferRepr(iree_hal_buffer_t* buffer, std::string& repr) {
  repr.append(std::to_string(iree_hal_buffer_byte_length(buffer)));
  repr.append(" bytes (at offset ");
  repr.append(std::to_string(iree_hal_buffer_byte_offset(buffer)));
  repr.append(" into ");
  repr.append(std::to_string(iree_hal_buffer_allocation_size(buffer)));
  repr.append("), memory_type=");

  // Memory type.
  iree_bitfield_string_temp_t tmp;
  iree_string_view_t sv;
  sv = iree_hal_memory_type_format(iree_hal_buffer_memory_type(buffer), &tmp);
  repr.append(sv.data, sv.size);

  // Allowed access.
  repr.append(", allowed_access=");
  sv = iree_hal_memory_access_format(iree_hal_buffer_allowed_access(buffer),
                                     &tmp);
  repr.append(sv.data, sv.size);

  // Allowed usage.
  repr.append(", allowed_usage=");
  sv =
      iree_hal_buffer_usage_format(iree_hal_buffer_allowed_usage(buffer), &tmp);
  repr.append(sv.data, sv.size);
}

}  // namespace

py::str HalBuffer::Repr() {
  std::string repr("<HalBuffer ");
  AppendHalBufferRepr(raw_ptr(), repr);
  repr.append(">");
  return py::str(repr);
}

//------------------------------------------------------------------------------
// HalBufferView
//------------------------------------------------------------------------------

py::str HalBufferView::Repr() {
  std::string repr("<HalBufferView (");

  // Shape.
  iree_host_size_t rank = iree_hal_buffer_view_shape_rank(raw_ptr());
  for (iree_host_size_t i = 0; i < rank; ++i) {
    if (i > 0) {
      repr.append(", ");
    }
    repr.append(std::to_string(iree_hal_buffer_view_shape_dim(raw_ptr(), i)));
  }
  repr.append(")");

  // Element type.
  repr.append(", element_type=0x");
  auto element_type = iree_hal_buffer_view_element_type(raw_ptr());
  repr.append(ToHexString(static_cast<uint32_t>(element_type)));

  repr.append(", ");
  AppendHalBufferRepr(iree_hal_buffer_view_buffer(raw_ptr()), repr);
  repr.append(">");
  return py::str(repr);
}

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
  CheckApiStatus(iree_hal_driver_registry_try_create(
                     iree_hal_driver_registry_default(),
                     {driver_name.data(), driver_name.size()},
                     iree_allocator_system(), &driver),
                 "Error creating driver");
  return HalDriver::StealFromRawPtr(driver);
}

HalDevice HalDriver::CreateDefaultDevice() {
  iree_hal_device_t* device;
  CheckApiStatus(iree_hal_driver_create_default_device(
                     raw_ptr(), iree_allocator_system(), &device),
                 "Error creating default device");
  return HalDevice::StealFromRawPtr(device);
}

//------------------------------------------------------------------------------
// Enum helpers
//------------------------------------------------------------------------------

namespace {

py::object MapElementTypeToDType(iree_hal_element_type_t element_type) {
  // See: https://docs.python.org/3/c-api/arg.html#numbers
  // TODO: Handle dtypes that do not map to a code (i.e. fp16).
  const char* dtype_code;
  switch (element_type) {
    case IREE_HAL_ELEMENT_TYPE_INT_8:
    case IREE_HAL_ELEMENT_TYPE_SINT_8:
      dtype_code = "b";
      break;
    case IREE_HAL_ELEMENT_TYPE_UINT_8:
      dtype_code = "B";
      break;
    case IREE_HAL_ELEMENT_TYPE_INT_16:
    case IREE_HAL_ELEMENT_TYPE_SINT_16:
      dtype_code = "h";
      break;
    case IREE_HAL_ELEMENT_TYPE_UINT_16:
      dtype_code = "H";
      break;
    case IREE_HAL_ELEMENT_TYPE_INT_32:
    case IREE_HAL_ELEMENT_TYPE_SINT_32:
      dtype_code = "i";
      break;
    case IREE_HAL_ELEMENT_TYPE_UINT_32:
      dtype_code = "I";
      break;
    case IREE_HAL_ELEMENT_TYPE_INT_64:
    case IREE_HAL_ELEMENT_TYPE_SINT_64:
      dtype_code = "l";
      break;
    case IREE_HAL_ELEMENT_TYPE_UINT_64:
      dtype_code = "L";
      break;
    case IREE_HAL_ELEMENT_TYPE_FLOAT_32:
      dtype_code = "f";
      break;
    case IREE_HAL_ELEMENT_TYPE_FLOAT_64:
      dtype_code = "d";
      break;
    case IREE_HAL_ELEMENT_TYPE_VALUE(IREE_HAL_NUMERICAL_TYPE_INTEGER, 1):
      dtype_code = "?";
      break;
    default:
      throw RaiseValueError("Unsupported VM Buffer -> numpy dtype mapping");
  }
  return py::dtype(dtype_code);
}

}  // namespace

//------------------------------------------------------------------------------
// Bindings
//------------------------------------------------------------------------------

void SetupHalBindings(pybind11::module m) {
  // Enums.
  py::enum_<enum iree_hal_memory_type_bits_t>(m, "MemoryType")
      .value("NONE", IREE_HAL_MEMORY_TYPE_NONE)
      .value("TRANSIENT", IREE_HAL_MEMORY_TYPE_TRANSIENT)
      .value("HOST_VISIBLE", IREE_HAL_MEMORY_TYPE_HOST_VISIBLE)
      .value("HOST_COHERENT", IREE_HAL_MEMORY_TYPE_HOST_COHERENT)
      .value("HOST_CACHED", IREE_HAL_MEMORY_TYPE_HOST_CACHED)
      .value("HOST_LOCAL", IREE_HAL_MEMORY_TYPE_HOST_LOCAL)
      .value("DEVICE_VISIBLE", IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE)
      .value("DEVICE_LOCAL", IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL)
      .export_values()
      .def("__or__",
           [](enum iree_hal_memory_type_bits_t self,
              enum iree_hal_memory_type_bits_t other) { return self | other; })
      .def("__and__",
           [](enum iree_hal_memory_type_bits_t self,
              enum iree_hal_memory_type_bits_t other) { return self & other; });

  py::enum_<enum iree_hal_buffer_compatibility_bits_t>(m, "BufferCompatibility")
      .value("NONE", IREE_HAL_BUFFER_COMPATIBILITY_NONE)
      .value("ALLOCATABLE", IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE)
      .value("IMPORTABLE", IREE_HAL_BUFFER_COMPATIBILITY_IMPORTABLE)
      .value("EXPORTABLE", IREE_HAL_BUFFER_COMPATIBILITY_EXPORTABLE)
      .value("QUEUE_TRANSFER", IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_TRANSFER)
      .value("QUEUE_DISPATCH", IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_DISPATCH)
      .export_values()
      .def("__or__",
           [](enum iree_hal_buffer_compatibility_bits_t self,
              enum iree_hal_buffer_compatibility_bits_t other) {
             return self | other;
           })
      .def("__and__", [](enum iree_hal_buffer_compatibility_bits_t self,
                         enum iree_hal_buffer_compatibility_bits_t other) {
        return self & other;
      });

  py::enum_<enum iree_hal_buffer_usage_bits_t>(m, "BufferUsage")
      .value("NONE", IREE_HAL_BUFFER_USAGE_NONE)
      .value("CONSTANT", IREE_HAL_BUFFER_USAGE_CONSTANT)
      .value("TRANSFER", IREE_HAL_BUFFER_USAGE_TRANSFER)
      .value("MAPPING", IREE_HAL_BUFFER_USAGE_MAPPING)
      .value("DISPATCH", IREE_HAL_BUFFER_USAGE_DISPATCH)
      .export_values()
      .def("__or__",
           [](enum iree_hal_buffer_usage_bits_t self,
              enum iree_hal_buffer_usage_bits_t other) {
             return (enum iree_hal_buffer_usage_bits_t)(self | other);
           })
      .def("__and__", [](enum iree_hal_buffer_usage_bits_t self,
                         enum iree_hal_buffer_usage_bits_t other) {
        return (enum iree_hal_buffer_usage_bits_t)(self & other);
      });

  py::enum_<enum iree_hal_memory_access_bits_t>(m, "MemoryAccess")
      .value("NONE", IREE_HAL_MEMORY_ACCESS_NONE)
      .value("READ", IREE_HAL_MEMORY_ACCESS_READ)
      .value("WRITE", IREE_HAL_MEMORY_ACCESS_WRITE)
      .value("DISCARD", IREE_HAL_MEMORY_ACCESS_DISCARD)
      .value("DISCARD_WRITE", IREE_HAL_MEMORY_ACCESS_DISCARD_WRITE)
      .value("ALL", IREE_HAL_MEMORY_ACCESS_ALL)
      .export_values()
      .def(
          "__or__",
          [](enum iree_hal_memory_access_bits_t self,
             enum iree_hal_memory_access_bits_t other) { return self | other; })
      .def("__and__", [](enum iree_hal_memory_access_bits_t self,
                         enum iree_hal_memory_access_bits_t other) {
        return self & other;
      });

  py::enum_<enum iree_hal_element_types_t>(m, "HalElementType")
      .value("NONE", IREE_HAL_ELEMENT_TYPE_NONE)
      .value("OPAQUE_8", IREE_HAL_ELEMENT_TYPE_OPAQUE_8)
      .value("OPAQUE_16", IREE_HAL_ELEMENT_TYPE_OPAQUE_16)
      .value("OPAQUE_32", IREE_HAL_ELEMENT_TYPE_OPAQUE_32)
      .value("OPAQUE_64", IREE_HAL_ELEMENT_TYPE_OPAQUE_64)
      .value("INT_4", IREE_HAL_ELEMENT_TYPE_INT_4)
      .value("INT_8", IREE_HAL_ELEMENT_TYPE_INT_8)
      .value("INT_16", IREE_HAL_ELEMENT_TYPE_INT_16)
      .value("INT_32", IREE_HAL_ELEMENT_TYPE_INT_32)
      .value("INT_64", IREE_HAL_ELEMENT_TYPE_INT_64)
      .value("SINT_4", IREE_HAL_ELEMENT_TYPE_SINT_4)
      .value("SINT_8", IREE_HAL_ELEMENT_TYPE_SINT_8)
      .value("SINT_16", IREE_HAL_ELEMENT_TYPE_SINT_16)
      .value("SINT_32", IREE_HAL_ELEMENT_TYPE_SINT_32)
      .value("SINT_64", IREE_HAL_ELEMENT_TYPE_SINT_64)
      .value("UINT_4", IREE_HAL_ELEMENT_TYPE_UINT_4)
      .value("UINT_8", IREE_HAL_ELEMENT_TYPE_UINT_8)
      .value("UINT_16", IREE_HAL_ELEMENT_TYPE_UINT_16)
      .value("UINT_32", IREE_HAL_ELEMENT_TYPE_UINT_32)
      .value("UINT_64", IREE_HAL_ELEMENT_TYPE_UINT_64)
      .value("FLOAT_16", IREE_HAL_ELEMENT_TYPE_FLOAT_16)
      .value("FLOAT_32", IREE_HAL_ELEMENT_TYPE_FLOAT_32)
      .value("FLOAT_64", IREE_HAL_ELEMENT_TYPE_FLOAT_64)
      .value("BFLOAT_16", IREE_HAL_ELEMENT_TYPE_BFLOAT_16)
      .value("BOOL_8",
             static_cast<iree_hal_element_types_t>(IREE_HAL_ELEMENT_TYPE_VALUE(
                 IREE_HAL_NUMERICAL_TYPE_INTEGER, 1)))
      .export_values()
      .def_static("map_to_dtype", &MapElementTypeToDType);

  py::class_<HalDevice>(m, "HalDevice")
      .def_property_readonly(
          "allocator",
          [](HalDevice& self) {
            return HalAllocator::BorrowFromRawPtr(self.allocator());
          },
          py::keep_alive<0, 1>());

  py::class_<HalDriver>(m, "HalDriver")
      .def_static("query", &HalDriver::Query)
      .def_static("create", &HalDriver::Create, py::arg("driver_name"))
      .def("create_default_device", &HalDriver::CreateDefaultDevice,
           py::keep_alive<0, 1>());

  py::class_<HalAllocator>(m, "HalAllocator")
      .def("trim",
           [](HalAllocator& self) {
             CheckApiStatus(iree_hal_allocator_trim(self.raw_ptr()),
                            "Error trim()'ing HAL allocator");
           })
      .def_property_readonly(
          "has_statistics",
          [](HalAllocator& self) -> bool { return IREE_STATISTICS_ENABLE; })
      .def_property_readonly("statistics", &HalAllocator::QueryStatistics)
      .def_property_readonly("formatted_statistics",
                             &HalAllocator::FormattedStatistics)
      .def(
          "query_compatibility",
          [](HalAllocator& self, int memory_type, int allowed_usage,
             int intended_usage, iree_device_size_t allocation_size) -> int {
            iree_hal_buffer_params_t params = {0};
            params.type = memory_type;
            params.usage = allowed_usage & intended_usage;
            return iree_hal_allocator_query_compatibility(
                self.raw_ptr(), params, allocation_size);
          },
          py::arg("memory_type"), py::arg("allowed_usage"),
          py::arg("intended_usage"), py::arg("allocation_size"))
      .def(
          "allocate_buffer",
          [](HalAllocator& self, int memory_type, int allowed_usage,
             iree_device_size_t allocation_size) {
            iree_hal_buffer_params_t params = {0};
            params.type = memory_type;
            params.usage = allowed_usage;
            iree_hal_buffer_t* buffer = nullptr;
            iree_const_byte_span_t empty_initial_data{nullptr, 0};
            CheckApiStatus(iree_hal_allocator_allocate_buffer(
                               self.raw_ptr(), params, allocation_size,
                               empty_initial_data, &buffer),
                           "could not allocate buffer");
            return HalBuffer::StealFromRawPtr(buffer);
          },
          py::arg("memory_type"), py::arg("allowed_usage"),
          py::arg("allocation_size"), py::keep_alive<0, 1>(),
          "Allocates a new buffer with requested characteristics (does not "
          "initialize with specific data).")
      .def("allocate_buffer_copy", &HalAllocator::AllocateBufferCopy,
           py::arg("memory_type"), py::arg("allowed_usage"), py::arg("buffer"),
           py::arg("element_type") = py::none(), py::keep_alive<0, 1>(),
           "Allocates a new buffer and initializes it from a Python buffer "
           "object. If an element type is specified, wraps in a BufferView "
           "matching the characteristics of the Python buffer. The format is "
           "requested as ND/C-Contiguous, which may incur copies if not "
           "already in that format.");

  py::class_<HalBuffer>(m, "HalBuffer")
      .def("fill_zero", &HalBuffer::FillZero, py::arg("byte_offset"),
           py::arg("byte_length"))
      .def("create_view", &HalBuffer::CreateView, py::arg("shape"),
           py::arg("element_size"), py::keep_alive<0, 1>())
      .def("__repr__", &HalBuffer::Repr);

  py::class_<HalBufferView>(m, "HalBufferView")
      .def("map", HalMappedMemory::Create, py::keep_alive<0, 1>())
      .def_property_readonly(
          "shape",
          [](HalBufferView& self) {
            iree_host_size_t rank =
                iree_hal_buffer_view_shape_rank(self.raw_ptr());
            auto* dims = iree_hal_buffer_view_shape_dims(self.raw_ptr());
            py::list result;
            for (iree_host_size_t i = 0; i < rank; ++i) {
              result.append(dims[i]);
            }
            return result;
          })
      .def_property_readonly(
          "element_type",
          [](HalBufferView& self) {
            return iree_hal_buffer_view_element_type(self.raw_ptr());
          })
      .def("__repr__", &HalBufferView::Repr);

  py::class_<HalMappedMemory>(m, "MappedMemory", py::buffer_protocol())
      .def_buffer(&HalMappedMemory::ToBufferInfo)
      .def("asarray",
           [](HalMappedMemory& self, std::vector<iree_host_size_t> shape,
              py::object dtype) {
             py::object py_mapped_memory = py::cast(self);
             return py::array(std::move(dtype), shape,
                              self.mapped_memory().contents.data,
                              std::move(py_mapped_memory) /* base */);
           });

  py::class_<HalShape>(m, "Shape").def(py::init(&HalShape::FromIntVector));
}

}  // namespace python
}  // namespace iree
