// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "./hal.h"

#include "./vm.h"
#include "iree/base/internal/path.h"
#include "iree/base/tracing.h"
#include "iree/hal/api.h"
#include "iree/hal/utils/allocators.h"
#include "iree/modules/hal/module.h"
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
          hal_buffer, dims.size(), dims.data(), *element_type, encoding_type,
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
// HalDevice
//------------------------------------------------------------------------------

void HalDevice::BeginProfiling(const py::kwargs& kwargs) {
  iree_hal_device_profiling_options_t options;
  memset(&options, 0, sizeof(options));

  options.mode = IREE_HAL_DEVICE_PROFILING_MODE_QUEUE_OPERATIONS;
  if (kwargs.contains("mode")) {
    auto mode_str = kwargs["mode"].cast<std::string>();
    if (mode_str == "queue") {
      options.mode = IREE_HAL_DEVICE_PROFILING_MODE_QUEUE_OPERATIONS;
    } else if (mode_str == "dispatch") {
      options.mode = IREE_HAL_DEVICE_PROFILING_MODE_DISPATCH_COUNTERS;
    } else if (mode_str == "executable") {
      options.mode = IREE_HAL_DEVICE_PROFILING_MODE_EXECUTABLE_COUNTERS;
    } else {
      throw RaiseValueError("unrecognized profiling mode");
    }
  }

  std::string file_path = kwargs.contains("file_path")
                              ? kwargs["file_path"].cast<std::string>()
                              : "";
  options.file_path = !file_path.empty() ? file_path.c_str() : NULL;

  CheckApiStatus(iree_hal_device_profiling_begin(raw_ptr(), &options),
                 "starting device profiling");
}

void HalDevice::EndProfiling() {
  CheckApiStatus(iree_hal_device_profiling_end(raw_ptr()),
                 "ending device profiling");
}

//------------------------------------------------------------------------------
// HalDriver
//------------------------------------------------------------------------------

std::vector<std::string> HalDriver::Query() {
  iree_host_size_t driver_info_count = 0;
  iree_hal_driver_info_t* driver_infos = NULL;
  CheckApiStatus(
      iree_hal_driver_registry_enumerate(iree_hal_driver_registry_default(),
                                         iree_allocator_system(),
                                         &driver_info_count, &driver_infos),
      "Error enumerating HAL drivers");
  std::vector<std::string> driver_names(driver_info_count);
  for (iree_host_size_t i = 0; i < driver_info_count; ++i) {
    driver_names[i] = std::string(driver_infos[i].driver_name.data,
                                  driver_infos[i].driver_name.size);
  }
  iree_allocator_free(iree_allocator_system(), driver_infos);
  return driver_names;
}

py::object HalDriver::Create(const std::string& device_uri,
                             py::dict& driver_cache) {
  iree_string_view_t driver_name, device_path, params_str;
  iree_string_view_t device_uri_sv{device_uri.data(), device_uri.size()};
  iree_uri_split(device_uri_sv, &driver_name, &device_path, &params_str);

  // Check cache.
  py::str cache_key(driver_name.data, driver_name.size);
  py::object cached = driver_cache.attr("get")(cache_key);
  if (!cached.is_none()) {
    return cached;
  }

  // Create.
  iree_hal_driver_t* driver;
  CheckApiStatus(iree_hal_driver_registry_try_create(
                     iree_hal_driver_registry_default(), driver_name,
                     iree_allocator_system(), &driver),
                 "Error creating driver");

  // Cache.
  py::object driver_obj = py::cast(HalDriver::StealFromRawPtr(driver));
  driver_cache[cache_key] = driver_obj;
  return driver_obj;
}

py::list HalDriver::QueryAvailableDevices() {
  iree_hal_device_info_t* device_infos;
  iree_host_size_t count;
  CheckApiStatus(iree_hal_driver_query_available_devices(
                     raw_ptr(), iree_allocator_system(), &count, &device_infos),
                 "Error querying devices");
  py::list results;
  for (iree_host_size_t i = 0; i < count; ++i) {
    py::dict device_data;
    device_data["device_id"] = py::cast(device_infos[i].device_id);
    device_data["path"] =
        py::str(device_infos[i].path.data, device_infos[i].path.size);
    device_data["name"] =
        py::str(device_infos[i].name.data, device_infos[i].name.size);
    results.append(device_data);
  }

  iree_allocator_free(iree_allocator_system(), device_infos);
  return results;
}

// Configures |device| based on flags before returning it to the user.
static iree_status_t ConfigureDevice(iree_hal_device_t* device,
                                     const py::kwargs& kwargs) {
  // Optionally wrap the base device allocator with caching/pooling.
  // Doing this here satisfies the requirement that no buffers have been
  // allocated yet - if we returned the device without doing this the caller
  // can more easily break the rules.
  if (kwargs.contains("allocators")) {
    // NOTE: we need to pass string views that point to the std::string storage.
    // We do that in two passes because as we grow spec_storage it may
    // reallocate itself and invalidate the pointers - only after we're done
    // can we capture them in views.
    auto spec_list = py::cast<py::list>(kwargs["allocators"]);
    std::vector<std::string> spec_storage;
    spec_storage.reserve(spec_list.size());
    for (auto item : spec_list) {
      auto spec = py::cast<std::string>(item);
      spec_storage.push_back(std::move(spec));
    }
    std::vector<iree_string_view_t> spec_views;
    spec_views.reserve(spec_list.size());
    for (const auto& spec : spec_storage) {
      spec_views.push_back(iree_make_string_view(spec.data(), spec.size()));
    }
    IREE_RETURN_IF_ERROR(iree_hal_configure_allocator_from_specs(
        spec_views.size(), spec_views.data(), device));
  }
  return iree_ok_status();
}

HalDevice HalDriver::CreateDefaultDevice(const py::kwargs& kwargs) {
  iree_hal_device_t* device;
  CheckApiStatus(iree_hal_driver_create_default_device(
                     raw_ptr(), iree_allocator_system(), &device),
                 "Error creating default device");
  CheckApiStatus(ConfigureDevice(device, kwargs),
                 "Error configuring the device");
  return HalDevice::StealFromRawPtr(device);
}

HalDevice HalDriver::CreateDevice(iree_hal_device_id_t device_id,
                                  const py::kwargs& kwargs) {
  // Since the device ids are supposed to be opaque, we need to verify
  // them by querying available devices.
  py::list available_devices = QueryAvailableDevices();
  bool found = false;
  py::object compare_device_id = py::cast(device_id);
  for (auto record : available_devices) {
    // Each record is a dict:
    // {"device_id": obj, "path": str, "name": str}.
    auto record_dict = py::cast<py::dict>(record);
    py::object found_device_id = record_dict["device_id"];
    if (found_device_id.is(compare_device_id)) {
      found = true;
      break;
    }
  }

  if (!found) {
    std::string msg;
    msg.append("Device id ");
    msg.append(std::to_string(device_id));
    msg.append(" not found. Available devices: ");
    msg.append(py::repr(available_devices));
    throw std::invalid_argument(std::move(msg));
  }

  std::vector<iree_string_pair_t> params;
  iree_hal_device_t* device;
  CheckApiStatus(iree_hal_driver_create_device_by_id(
                     raw_ptr(), device_id, params.size(), &params.front(),
                     iree_allocator_system(), &device),
                 "Error creating default device");
  CheckApiStatus(ConfigureDevice(device, kwargs),
                 "Error configuring the device");
  return HalDevice::StealFromRawPtr(device);
}

HalDevice HalDriver::CreateDeviceByURI(std::string& device_uri,
                                       const py::kwargs& kwargs) {
  iree_hal_device_t* device;
  iree_string_view_t device_uri_sv{device_uri.data(), device_uri.size()};
  CheckApiStatus(
      iree_hal_driver_create_device_by_uri(raw_ptr(), device_uri_sv,
                                           iree_allocator_system(), &device),
      "Error creating device");
  CheckApiStatus(ConfigureDevice(device, kwargs),
                 "Error configuring the device");
  return HalDevice::StealFromRawPtr(device);
}

//------------------------------------------------------------------------------
// Enum helpers
//------------------------------------------------------------------------------

namespace {

py::object MapElementTypeToDType(iree_hal_element_type_t element_type) {
  // See:
  //   * https://numpy.org/doc/stable/reference/arrays.dtypes.html
  //   * https://docs.python.org/3/c-api/arg.html#numbers
  //
  // Single letter codes can be ambiguous across platforms, so prefer explicit
  // bit depth values, ("Type strings: Any string in numpy.sctypeDict.keys()").
  // See https://github.com/pybind/pybind11/issues/1908
  const char* dtype_string;
  switch (element_type) {
    case IREE_HAL_ELEMENT_TYPE_BOOL_8:
      dtype_string = "?";
      break;
    case IREE_HAL_ELEMENT_TYPE_INT_8:
    case IREE_HAL_ELEMENT_TYPE_SINT_8:
      dtype_string = "int8";
      break;
    case IREE_HAL_ELEMENT_TYPE_UINT_8:
      dtype_string = "uint8";
      break;
    case IREE_HAL_ELEMENT_TYPE_INT_16:
    case IREE_HAL_ELEMENT_TYPE_SINT_16:
      dtype_string = "int16";
      break;
    case IREE_HAL_ELEMENT_TYPE_UINT_16:
      dtype_string = "uint16";
      break;
    case IREE_HAL_ELEMENT_TYPE_INT_32:
    case IREE_HAL_ELEMENT_TYPE_SINT_32:
      dtype_string = "int32";
      break;
    case IREE_HAL_ELEMENT_TYPE_UINT_32:
      dtype_string = "uint32";
      break;
    case IREE_HAL_ELEMENT_TYPE_INT_64:
    case IREE_HAL_ELEMENT_TYPE_SINT_64:
      dtype_string = "int64";
      break;
    case IREE_HAL_ELEMENT_TYPE_UINT_64:
      dtype_string = "uint64";
      break;
    case IREE_HAL_ELEMENT_TYPE_FLOAT_16:
      dtype_string = "float16";
      break;
    case IREE_HAL_ELEMENT_TYPE_FLOAT_32:
      dtype_string = "float32";
      break;
    case IREE_HAL_ELEMENT_TYPE_FLOAT_64:
      dtype_string = "float64";
      break;
    default:
      throw RaiseValueError("Unsupported VM Buffer -> numpy dtype mapping");
  }
  return py::dtype(dtype_string);
}

}  // namespace

//------------------------------------------------------------------------------
// HAL module
//------------------------------------------------------------------------------

VmModule CreateHalModule(VmInstance* instance, HalDevice* device) {
  iree_vm_module_t* module = NULL;
  CheckApiStatus(iree_hal_module_create(instance->raw_ptr(), device->raw_ptr(),
                                        IREE_HAL_MODULE_FLAG_NONE,
                                        iree_allocator_system(), &module),
                 "Error creating hal module");
  return VmModule::StealFromRawPtr(module);
}

//------------------------------------------------------------------------------
// Bindings
//------------------------------------------------------------------------------

void SetupHalBindings(pybind11::module m) {
  py::dict driver_cache;

  // Built-in module creation.
  m.def("create_hal_module", &CreateHalModule);

  // Enums.
  py::enum_<enum iree_hal_memory_type_bits_t>(m, "MemoryType")
      .value("NONE", IREE_HAL_MEMORY_TYPE_NONE)
      .value("OPTIMAL", IREE_HAL_MEMORY_TYPE_OPTIMAL)
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
      .value("TRANSFER_SOURCE", IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE)
      .value("TRANSFER_TARGET", IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET)
      .value("TRANSFER", IREE_HAL_BUFFER_USAGE_TRANSFER)
      .value("DISPATCH_INDIRECT_PARAMS",
             IREE_HAL_BUFFER_USAGE_DISPATCH_INDIRECT_PARAMS)
      .value("DISPATCH_UNIFORM_READ",
             IREE_HAL_BUFFER_USAGE_DISPATCH_UNIFORM_READ)
      .value("DISPATCH_STORAGE_READ",
             IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE_READ)
      .value("DISPATCH_STORAGE_WRITE",
             IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE_WRITE)
      .value("DISPATCH_STORAGE", IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE)
      .value("DISPATCH_IMAGE_READ", IREE_HAL_BUFFER_USAGE_DISPATCH_IMAGE_READ)
      .value("DISPATCH_IMAGE_WRITE", IREE_HAL_BUFFER_USAGE_DISPATCH_IMAGE_WRITE)
      .value("DISPATCH_IMAGE", IREE_HAL_BUFFER_USAGE_DISPATCH_IMAGE)
      .value("SHARING_EXPORT", IREE_HAL_BUFFER_USAGE_SHARING_EXPORT)
      .value("SHARING_REPLICATE", IREE_HAL_BUFFER_USAGE_SHARING_REPLICATE)
      .value("SHARING_CONCURRENT", IREE_HAL_BUFFER_USAGE_SHARING_CONCURRENT)
      .value("SHARING_IMMUTABLE", IREE_HAL_BUFFER_USAGE_SHARING_IMMUTABLE)
      .value("MAPPING_SCOPED", IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED)
      .value("MAPPING_PERSISTENT", IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT)
      .value("MAPPING_OPTIONAL", IREE_HAL_BUFFER_USAGE_MAPPING_OPTIONAL)
      .value("MAPPING_ACCESS_RANDOM",
             IREE_HAL_BUFFER_USAGE_MAPPING_ACCESS_RANDOM)
      .value("MAPPING_ACCESS_SEQUENTIAL_WRITE",
             IREE_HAL_BUFFER_USAGE_MAPPING_ACCESS_SEQUENTIAL_WRITE)
      .value("MAPPING", IREE_HAL_BUFFER_USAGE_MAPPING)
      .value("DEFAULT", IREE_HAL_BUFFER_USAGE_DEFAULT)
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
      .value("BOOL_8", IREE_HAL_ELEMENT_TYPE_BOOL_8)
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
      .value("COMPLEX_64", IREE_HAL_ELEMENT_TYPE_COMPLEX_FLOAT_64)
      .value("COMPLEX_128", IREE_HAL_ELEMENT_TYPE_COMPLEX_FLOAT_128)
      .export_values()
      .def_static("map_to_dtype", &MapElementTypeToDType);

  py::class_<HalDevice>(m, "HalDevice")
      .def_property_readonly(
          "allocator",
          [](HalDevice& self) {
            return HalAllocator::BorrowFromRawPtr(self.allocator());
          },
          py::keep_alive<0, 1>())
      .def("begin_profiling", &HalDevice::BeginProfiling)
      .def("end_profiling", &HalDevice::EndProfiling);

  py::class_<HalDriver>(m, "HalDriver")
      .def_static("query", &HalDriver::Query)
      .def("create_default_device", &HalDriver::CreateDefaultDevice,
           py::keep_alive<0, 1>())
      .def("create_device", &HalDriver::CreateDevice, py::keep_alive<0, 1>())
      .def("create_device_by_uri", &HalDriver::CreateDeviceByURI,
           py::keep_alive<0, 1>())
      .def(
          "create_device",
          [](HalDriver& self, py::dict device_info,
             const py::kwargs& kwargs) -> HalDevice {
            // Alias of create_device that takes a dict as returned from
            // query_available_devices for convenience.
            auto device_id =
                py::cast<iree_hal_device_id_t>(device_info["device_id"]);
            return self.CreateDevice(device_id, kwargs);
          },
          py::keep_alive<0, 1>())
      .def("query_available_devices", &HalDriver::QueryAvailableDevices);

  m.def(
      "get_cached_hal_driver",
      [driver_cache](std::string device_uri) {
        return HalDriver::Create(device_uri,
                                 const_cast<py::dict&>(driver_cache));
      },
      py::arg("device_uri"));

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
          "query_buffer_compatibility",
          [](HalAllocator& self, int memory_type, int allowed_usage,
             int intended_usage, iree_device_size_t allocation_size) -> int {
            iree_hal_buffer_params_t params = {0};
            params.type = memory_type;
            params.usage = allowed_usage & intended_usage;
            return iree_hal_allocator_query_buffer_compatibility(
                self.raw_ptr(), params, allocation_size,
                /*out_params=*/nullptr, /*out_allocation_size=*/0);
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

  auto hal_buffer_view = py::class_<HalBufferView>(m, "HalBufferView");
  VmRef::BindRefProtocol(hal_buffer_view, iree_hal_buffer_view_type_id,
                         iree_hal_buffer_view_retain_ref,
                         iree_hal_buffer_view_deref, iree_hal_buffer_view_isa);
  hal_buffer_view.def("map", HalMappedMemory::Create, py::keep_alive<0, 1>())
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
