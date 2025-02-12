// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "./hal.h"

#include <nanobind/intrusive/ref.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <iterator>
#include <optional>

#include "./local_dlpack.h"
#include "./numpy_interop.h"
#include "./vm.h"
#include "iree/base/internal/path.h"
#include "iree/base/status.h"
#include "iree/hal/api.h"
#include "iree/hal/utils/allocators.h"
#include "iree/modules/hal/module.h"
#include "iree/tooling/device_util.h"

namespace iree {
namespace python {

namespace {

static const char kHalDeviceQueueAlloca[] =
    R"(Reserves and returns a device-local queue-ordered transient buffer.

Args:
  allocation_size: The size in bytes of the allocation.
  wait_semaphores: `List[Tuple[HalSemaphore, int]]` of semaphore values or
    a HalFence. The allocation will be made once these semaphores are
    satisfied.
  signal_semaphores: Semaphores/Fence to signal.

Returns:
  HalBuffer.
)";

static const char kHalDeviceQueueDealloca[] =
    R"(Deallocates a queue-ordered transient buffer.

Args:
  wait_semaphores: `List[Tuple[HalSemaphore, int]]` of semaphore values or
    a HalFence. The allocation will be made once these semaphores are
    satisfied.
  signal_semaphores: Semaphores/Fence to signal.

Returns:
  HalBuffer.
)";

static const char kHalDeviceQueueExecute[] =
    R"(Executes a sequence of command buffers.

Args:
  command_buffers: Sequence of command buffers to enqueue.
  wait_semaphores: `List[Tuple[HalSemaphore, int]]` of semaphore values or
    a HalFence. The allocation will be made once these semaphores are
    satisfied.
  signal_semaphores: Semaphores/Fence to signal.
)";

static const char kHalDeviceQueueCopy[] =
    R"(Copy data from a source buffer to destination buffer.

Args:
  source_buffer: `HalBuffer` that holds src data.
  target_buffer: `HalBuffer` that will receive data.
  wait_semaphores: `List[Tuple[HalSemaphore, int]]` of semaphore values or
    a HalFence. The allocation will be made once these semaphores are
    satisfied.
  signal_semaphores: Semaphores/Fence to signal.
)";

static const char kHalWait[] =
    R"(Waits until the semaphore or fence is signalled or errored.

Three wait cases are supported:
  * timeout: Relative nanoseconds to wait.
  * deadine: Absolute nanoseconds to wait.
  * Neither: Waits for infinite time.

Returns whether the wait succeeded (True) or timed out (False). If the fence was
asynchronously failed, an exception is raised.
)";

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

iree_timeout_t NormalizeTimeout(std::optional<iree_duration_t> timeout,
                                std::optional<iree_time_t> deadline) {
  if (!timeout && !deadline) {
    return iree_infinite_timeout();
  } else if (timeout && deadline) {
    throw std::invalid_argument("timeout and deadline cannot both be set");
  } else if (timeout) {
    return iree_make_timeout_ns(*timeout);
  } else {
    return iree_timeout_t{IREE_TIMEOUT_ABSOLUTE, *deadline};
  }
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
    int memory_type, int allowed_usage, HalDevice& device, py::object buffer,
    std::optional<uint64_t> raw_element_type) {
  IREE_TRACE_SCOPE_NAMED("HalAllocator::AllocateBufferCopy");
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
    throw py::python_error();
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
    status = iree_hal_allocator_allocate_buffer(raw_ptr(), params, py_view.len,
                                                &hal_buffer);
    if (iree_status_is_ok(status)) {
      status = iree_hal_device_transfer_h2d(
          device.raw_ptr(), py_view.buf, hal_buffer, 0, py_view.len,
          IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());
    }
  }
  CheckApiStatus(status, "Failed to allocate device visible buffer");

  if (!raw_element_type) {
    return py::cast(HalBuffer::StealFromRawPtr(hal_buffer),
                    py::rv_policy::move);
  }

  // Create the buffer_view. (note that numpy shape is ssize_t, so we need to
  // copy).
  iree_hal_element_types_t element_type =
      (iree_hal_element_types_t)*raw_element_type;
  iree_hal_encoding_type_t encoding_type =
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR;
  std::vector<iree_hal_dim_t> dims(py_view.ndim);
  std::copy(py_view.shape, py_view.shape + py_view.ndim, dims.begin());
  iree_hal_buffer_view_t* hal_buffer_view;
  CheckApiStatus(
      iree_hal_buffer_view_create(
          hal_buffer, dims.size(), dims.data(), element_type, encoding_type,
          iree_hal_allocator_host_allocator(raw_ptr()), &hal_buffer_view),
      "Error allocating buffer_view");
  iree_hal_buffer_release(hal_buffer);

  return py::cast(HalBufferView::StealFromRawPtr(hal_buffer_view),
                  py::rv_policy::move);
}

HalBuffer HalAllocator::AllocateHostStagingBufferCopy(HalDevice& device,
                                                      py::handle buffer) {
  IREE_TRACE_SCOPE_NAMED("HalAllocator::AllocateHostStagingBufferCopy");
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
    throw py::python_error();
  }
  PyBufferReleaser py_view_releaser(py_view);

  iree_hal_buffer_params_t params = {0};
  std::memset(&params, 0, sizeof(params));
  params.type = IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE;
  params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER;

  iree_hal_buffer_t* hal_buffer = nullptr;
  iree_status_t status = iree_ok_status();
  {
    py::gil_scoped_release release;
    status = iree_hal_allocator_allocate_buffer(raw_ptr(), params, py_view.len,
                                                &hal_buffer);
    if (iree_status_is_ok(status)) {
      status = iree_hal_device_transfer_h2d(
          device.raw_ptr(), py_view.buf, hal_buffer, 0, py_view.len,
          IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());
    }
  }
  CheckApiStatus(status, "Failed to allocate device visible buffer");

  return HalBuffer::StealFromRawPtr(hal_buffer);
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
  return py::str(py::cast(repr));
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
  return py::str(py::cast(repr));
}

//------------------------------------------------------------------------------
// HalDevice
//------------------------------------------------------------------------------

void HalDevice::BeginProfiling(std::optional<std::string> mode,
                               std::optional<std::string> file_path) {
  iree_hal_device_profiling_options_t options;
  memset(&options, 0, sizeof(options));

  options.mode = IREE_HAL_DEVICE_PROFILING_MODE_QUEUE_OPERATIONS;
  if (mode) {
    if (*mode == "queue") {
      options.mode = IREE_HAL_DEVICE_PROFILING_MODE_QUEUE_OPERATIONS;
    } else if (*mode == "dispatch") {
      options.mode = IREE_HAL_DEVICE_PROFILING_MODE_DISPATCH_COUNTERS;
    } else if (*mode == "executable") {
      options.mode = IREE_HAL_DEVICE_PROFILING_MODE_EXECUTABLE_COUNTERS;
    } else {
      throw RaiseValueError("unrecognized profiling mode");
    }
  }

  options.file_path = file_path ? file_path->c_str() : nullptr;
  CheckApiStatus(iree_hal_device_profiling_begin(raw_ptr(), &options),
                 "starting device profiling");
}

void HalDevice::FlushProfiling() {
  CheckApiStatus(iree_hal_device_profiling_flush(raw_ptr()),
                 "flushing device profiling");
}

void HalDevice::EndProfiling() {
  CheckApiStatus(iree_hal_device_profiling_end(raw_ptr()),
                 "ending device profiling");
}

HalSemaphore HalDevice::CreateSemaphore(uint64_t initial_value) {
  iree_hal_semaphore_t* out_sem;
  CheckApiStatus(
      iree_hal_semaphore_create(raw_ptr(), initial_value,
                                IREE_HAL_SEMAPHORE_FLAG_NONE, &out_sem),
      "creating semaphore");
  return HalSemaphore::StealFromRawPtr(out_sem);
}

HalBuffer HalDevice::QueueAlloca(uint64_t allocation_size,
                                 py::handle wait_semaphores,
                                 py::handle signal_semaphores) {
  iree_hal_buffer_params_t params;
  memset(&params, 0, sizeof(params));
  // TODO: Accept explicit params in API.
  params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;

  iree_hal_semaphore_list_t wait_list;
  iree_hal_semaphore_list_t signal_list;

  // Wait list.
  if (py::isinstance<HalFence>(wait_semaphores)) {
    wait_list = iree_hal_fence_semaphore_list(
        py::cast<HalFence*>(wait_semaphores)->raw_ptr());
  } else {
    size_t wait_count = py::len(wait_semaphores);
    wait_list = {
        wait_count,
        /*semaphores=*/
        static_cast<iree_hal_semaphore_t**>(
            alloca(sizeof(iree_hal_semaphore_t*) * wait_count)),
        /*payload_values=*/
        static_cast<uint64_t*>(alloca(sizeof(uint64_t) * wait_count)),
    };
    for (size_t i = 0; i < wait_count; ++i) {
      py::tuple pair = wait_semaphores[i];
      wait_list.semaphores[i] = py::cast<HalSemaphore*>(pair[0])->raw_ptr();
      wait_list.payload_values[i] = py::cast<uint64_t>(pair[1]);
    }
  }

  // Signal list.
  if (py::isinstance<HalFence>(signal_semaphores)) {
    signal_list = iree_hal_fence_semaphore_list(
        py::cast<HalFence*>(signal_semaphores)->raw_ptr());
  } else {
    size_t signal_count = py::len(signal_semaphores);
    signal_list = {
        signal_count,
        /*semaphores=*/
        static_cast<iree_hal_semaphore_t**>(
            alloca(sizeof(iree_hal_semaphore_t*) * signal_count)),
        /*payload_values=*/
        static_cast<uint64_t*>(alloca(sizeof(uint64_t) * signal_count)),
    };
    for (size_t i = 0; i < signal_count; ++i) {
      py::tuple pair = signal_semaphores[i];
      signal_list.semaphores[i] = py::cast<HalSemaphore*>(pair[0])->raw_ptr();
      signal_list.payload_values[i] = py::cast<uint64_t>(pair[1]);
    }
  }

  iree_hal_buffer_t* out_buffer;
  // TODO: Accept params for queue affinity and pool.
  CheckApiStatus(iree_hal_device_queue_alloca(
                     raw_ptr(), IREE_HAL_QUEUE_AFFINITY_ANY, wait_list,
                     signal_list, IREE_HAL_ALLOCATOR_POOL_DEFAULT, params,
                     allocation_size, &out_buffer),
                 "allocating memory on queue");
  return HalBuffer::StealFromRawPtr(out_buffer);
}

void HalDevice::QueueDealloca(HalBuffer& buffer, py::handle wait_semaphores,
                              py::handle signal_semaphores) {
  iree_hal_semaphore_list_t wait_list;
  iree_hal_semaphore_list_t signal_list;

  // Wait list.
  if (py::isinstance<HalFence>(wait_semaphores)) {
    wait_list = iree_hal_fence_semaphore_list(
        py::cast<HalFence*>(wait_semaphores)->raw_ptr());
  } else {
    size_t wait_count = py::len(wait_semaphores);
    wait_list = {
        wait_count,
        /*semaphores=*/
        static_cast<iree_hal_semaphore_t**>(
            alloca(sizeof(iree_hal_semaphore_t*) * wait_count)),
        /*payload_values=*/
        static_cast<uint64_t*>(alloca(sizeof(uint64_t) * wait_count)),
    };
    for (size_t i = 0; i < wait_count; ++i) {
      py::tuple pair = wait_semaphores[i];
      wait_list.semaphores[i] = py::cast<HalSemaphore*>(pair[0])->raw_ptr();
      wait_list.payload_values[i] = py::cast<uint64_t>(pair[1]);
    }
  }

  // Signal list.
  if (py::isinstance<HalFence>(signal_semaphores)) {
    signal_list = iree_hal_fence_semaphore_list(
        py::cast<HalFence*>(signal_semaphores)->raw_ptr());
  } else {
    size_t signal_count = py::len(signal_semaphores);
    signal_list = {
        signal_count,
        /*semaphores=*/
        static_cast<iree_hal_semaphore_t**>(
            alloca(sizeof(iree_hal_semaphore_t*) * signal_count)),
        /*payload_values=*/
        static_cast<uint64_t*>(alloca(sizeof(uint64_t) * signal_count)),
    };
    for (size_t i = 0; i < signal_count; ++i) {
      py::tuple pair = signal_semaphores[i];
      signal_list.semaphores[i] = py::cast<HalSemaphore*>(pair[0])->raw_ptr();
      signal_list.payload_values[i] = py::cast<uint64_t>(pair[1]);
    }
  }

  CheckApiStatus(
      iree_hal_device_queue_dealloca(raw_ptr(), IREE_HAL_QUEUE_AFFINITY_ANY,
                                     wait_list, signal_list, buffer.raw_ptr()),
      "deallocating memory on queue");
}

void HalDevice::QueueExecute(py::handle command_buffer,
                             py::handle wait_semaphores,
                             py::handle signal_semaphores) {
  iree_hal_semaphore_list_t wait_list;
  iree_hal_semaphore_list_t signal_list;

  // Wait list.
  if (py::isinstance<HalFence>(wait_semaphores)) {
    wait_list = iree_hal_fence_semaphore_list(
        py::cast<HalFence*>(wait_semaphores)->raw_ptr());
  } else {
    size_t wait_count = py::len(wait_semaphores);
    wait_list = {
        wait_count,
        /*semaphores=*/
        static_cast<iree_hal_semaphore_t**>(
            alloca(sizeof(iree_hal_semaphore_t*) * wait_count)),
        /*payload_values=*/
        static_cast<uint64_t*>(alloca(sizeof(uint64_t) * wait_count)),
    };
    for (size_t i = 0; i < wait_count; ++i) {
      py::tuple pair = wait_semaphores[i];
      wait_list.semaphores[i] = py::cast<HalSemaphore*>(pair[0])->raw_ptr();
      wait_list.payload_values[i] = py::cast<uint64_t>(pair[1]);
    }
  }

  // Signal list.
  if (py::isinstance<HalFence>(signal_semaphores)) {
    signal_list = iree_hal_fence_semaphore_list(
        py::cast<HalFence*>(signal_semaphores)->raw_ptr());
  } else {
    size_t signal_count = py::len(signal_semaphores);
    signal_list = {
        signal_count,
        /*semaphores=*/
        static_cast<iree_hal_semaphore_t**>(
            alloca(sizeof(iree_hal_semaphore_t*) * signal_count)),
        /*payload_values=*/
        static_cast<uint64_t*>(alloca(sizeof(uint64_t) * signal_count)),
    };
    for (size_t i = 0; i < signal_count; ++i) {
      py::tuple pair = signal_semaphores[i];
      signal_list.semaphores[i] = py::cast<HalSemaphore*>(pair[0])->raw_ptr();
      signal_list.payload_values[i] = py::cast<uint64_t>(pair[1]);
    }
  }

  // Unpack command buffers.
  iree_hal_command_buffer_t* cb =
      !command_buffer.is_none()
          ? py::cast<HalCommandBuffer*>(command_buffer)->raw_ptr()
          : NULL;

  CheckApiStatus(iree_hal_device_queue_execute(
                     raw_ptr(), IREE_HAL_QUEUE_AFFINITY_ANY, wait_list,
                     signal_list, cb, iree_hal_buffer_binding_table_empty()),
                 "executing command buffers");
}

void HalDevice::QueueCopy(HalBuffer& source_buffer, HalBuffer& target_buffer,
                          py::handle wait_semaphores,
                          py::handle signal_semaphores) {
  iree_hal_semaphore_list_t wait_list;
  iree_hal_semaphore_list_t signal_list;

  // Wait list.
  if (py::isinstance<HalFence>(wait_semaphores)) {
    wait_list = iree_hal_fence_semaphore_list(
        py::cast<HalFence*>(wait_semaphores)->raw_ptr());
  } else {
    size_t wait_count = py::len(wait_semaphores);
    wait_list = {
        wait_count,
        /*semaphores=*/
        static_cast<iree_hal_semaphore_t**>(
            alloca(sizeof(iree_hal_semaphore_t*) * wait_count)),
        /*payload_values=*/
        static_cast<uint64_t*>(alloca(sizeof(uint64_t) * wait_count)),
    };
    for (size_t i = 0; i < wait_count; ++i) {
      py::tuple pair = wait_semaphores[i];
      wait_list.semaphores[i] = py::cast<HalSemaphore*>(pair[0])->raw_ptr();
      wait_list.payload_values[i] = py::cast<uint64_t>(pair[1]);
    }
  }

  // Signal list.
  if (py::isinstance<HalFence>(signal_semaphores)) {
    signal_list = iree_hal_fence_semaphore_list(
        py::cast<HalFence*>(signal_semaphores)->raw_ptr());
  } else {
    size_t signal_count = py::len(signal_semaphores);
    signal_list = {
        signal_count,
        /*semaphores=*/
        static_cast<iree_hal_semaphore_t**>(
            alloca(sizeof(iree_hal_semaphore_t*) * signal_count)),
        /*payload_values=*/
        static_cast<uint64_t*>(alloca(sizeof(uint64_t) * signal_count)),
    };
    for (size_t i = 0; i < signal_count; ++i) {
      py::tuple pair = signal_semaphores[i];
      signal_list.semaphores[i] = py::cast<HalSemaphore*>(pair[0])->raw_ptr();
      signal_list.payload_values[i] = py::cast<uint64_t>(pair[1]);
    }
  }

  // TODO: Accept params for src_offset and target_offset. Just check that
  // the source will fit in the target buffer for now.
  iree_device_size_t source_length =
      iree_hal_buffer_byte_length(source_buffer.raw_ptr());
  if (source_length > iree_hal_buffer_byte_length(target_buffer.raw_ptr())) {
    throw std::invalid_argument(
        "Source and buffer length must be less than the target buffer length "
        "and it does not. Please check allocations");
  }
  CheckApiStatus(
      iree_hal_device_queue_copy(
          raw_ptr(), IREE_HAL_QUEUE_AFFINITY_ANY, wait_list, signal_list,
          source_buffer.raw_ptr(), 0, target_buffer.raw_ptr(), 0, source_length,
          IREE_HAL_COPY_FLAG_NONE),
      "Copying buffer on queue");
}

py::object HalDevice::CreateDLPackCapsule(HalBufferView& buffer_view,
                                          int device_type_code, int device_id) {
  const size_t kStaticDimLimit = 6;
  struct ExtDLManagedTensor : public DLManagedTensor {
    ~ExtDLManagedTensor() {
      if (retained_buffer) {
        iree_hal_buffer_release(retained_buffer);
      }
      if (dl_tensor.ndim > kStaticDimLimit) {
        delete[] dim_storage.dynamic_shape;
      }
    }
    iree_hal_buffer_t* retained_buffer = nullptr;
    union {
      int64_t static_shape[kStaticDimLimit];
      int64_t* dynamic_shape;
    } dim_storage;
  };
  auto tensor = std::make_unique<ExtDLManagedTensor>();
  memset(static_cast<DLManagedTensor*>(tensor.get()), 0,
         sizeof(DLManagedTensor));
  auto capsule_destructor = +[](PyObject* capsule) {
    const char* actual_name = PyCapsule_GetName(capsule);
    if (!actual_name || strcmp(actual_name, "dltensor") != 0) {
      // Caller consumed the capsule. Do nothing.
      return;
    }

    // Capsule was dropped on the floor before consumed. Release resources.
    void* capsule_ptr = PyCapsule_GetPointer(capsule, "dltensor");
    if (!capsule_ptr) {
      return;
    }
    DLManagedTensor* tensor_ptr = static_cast<DLManagedTensor*>(capsule_ptr);
    tensor_ptr->deleter(tensor_ptr);
  };
  auto deleter = +[](struct DLManagedTensor* self) {
    auto* ext_self = static_cast<ExtDLManagedTensor*>(self);
    delete ext_self;
  };

  // Populate the DLManagedTensor.
  tensor->deleter = deleter;
  auto& dl_tensor = tensor->dl_tensor;
  dl_tensor.device.device_type = static_cast<DLDeviceType>(device_type_code);
  dl_tensor.device.device_id = device_id;

  // Convert metadata.
  iree_hal_element_type_t et =
      iree_hal_buffer_view_element_type(buffer_view.raw_ptr());
  dl_tensor.dtype.bits = iree_hal_element_bit_count(et);
  dl_tensor.dtype.lanes = 1;
  switch (iree_hal_element_numerical_type(et)) {
    case IREE_HAL_NUMERICAL_TYPE_FLOAT_IEEE:
      dl_tensor.dtype.code = kDLFloat;
      break;
    case IREE_HAL_NUMERICAL_TYPE_INTEGER_UNSIGNED:
      dl_tensor.dtype.code = kDLUInt;
      break;
    case IREE_HAL_NUMERICAL_TYPE_INTEGER:
    case IREE_HAL_NUMERICAL_TYPE_INTEGER_SIGNED:
      dl_tensor.dtype.code = kDLInt;
      break;
    case IREE_HAL_NUMERICAL_TYPE_FLOAT_BRAIN:
      dl_tensor.dtype.code = kDLBfloat;
      break;
    case IREE_HAL_NUMERICAL_TYPE_FLOAT_COMPLEX:
      dl_tensor.dtype.code = kDLComplex;
      break;
    case IREE_HAL_NUMERICAL_TYPE_BOOLEAN:
      dl_tensor.dtype.code = kDLBool;
      break;
    default:
      throw std::invalid_argument(
          "dlpack unsupported buffer view element type");
  }

  // Shape.
  // Leave strides nullptr to signify dense row-major.
  auto rank = iree_hal_buffer_view_shape_rank(buffer_view.raw_ptr());
  auto* bv_dims = iree_hal_buffer_view_shape_dims(buffer_view.raw_ptr());
  if (rank > kStaticDimLimit) {
    dl_tensor.shape = new int64_t[rank];
    tensor->dim_storage.dynamic_shape = dl_tensor.shape;
  } else {
    dl_tensor.shape = tensor->dim_storage.static_shape;
  }
  for (size_t i = 0; i < rank; ++i) {
    dl_tensor.shape[i] = bv_dims[i];
  }
  dl_tensor.ndim = rank;

  // Export buffer view.
  iree_hal_buffer_t* buffer =
      iree_hal_buffer_view_buffer(buffer_view.raw_ptr());
  auto offset = iree_hal_buffer_byte_offset(buffer);
  buffer = iree_hal_buffer_allocated_buffer(buffer);
  iree_hal_allocator_t* alloc = iree_hal_device_allocator(raw_ptr());
  iree_hal_external_buffer_t external_buffer;
  CheckApiStatus(
      iree_hal_allocator_export_buffer(
          alloc, buffer, IREE_HAL_EXTERNAL_BUFFER_TYPE_DEVICE_ALLOCATION,
          IREE_HAL_EXTERNAL_BUFFER_FLAG_NONE, &external_buffer),
      "Cannot export device buffer");
  static_assert(sizeof(dl_tensor.data) >=
                sizeof(external_buffer.handle.device_allocation.ptr));
  dl_tensor.data =
      reinterpret_cast<void*>(external_buffer.handle.device_allocation.ptr);
  dl_tensor.byte_offset = offset;

  // Create and return capsule.
  PyObject* capsule = PyCapsule_New(static_cast<DLManagedTensor*>(tensor.get()),
                                    "dltensor", capsule_destructor);
  if (!capsule) {
    throw py::python_error();
  }

  // Retain the backing buffer view bound to the capsule lifetime.
  tensor->retained_buffer = buffer;
  iree_hal_buffer_retain(buffer);
  tensor.release();
  return py::steal<py::object>(capsule);
}

HalBufferView HalDevice::FromDLPackCapsule(py::object input_capsule) {
  struct State {
    ~State() {
      if (managed_tensor && managed_tensor->deleter) {
        managed_tensor->deleter(managed_tensor);
      }
    }
    py::object capsule;
    void* raw = nullptr;
    DLManagedTensor* managed_tensor = nullptr;
  } state;
  state.capsule = std::move(input_capsule);
  state.raw = PyCapsule_GetPointer(state.capsule.ptr(), "dltensor");
  if (!state.raw) {
    throw py::python_error();
  }
  state.managed_tensor = static_cast<DLManagedTensor*>(state.raw);
  // Takes ownership.
  if (PyCapsule_SetName(state.capsule.ptr(), "used_dltensor")) {
    throw py::python_error();
  }

  DLTensor* dlt = &state.managed_tensor->dl_tensor;

  // Some validation on what we accept.
  if (dlt->dtype.lanes != 1) {
    throw std::invalid_argument("Unsupported dtype lanes != 1");
  }

  iree_hal_element_type_t et;
  switch (dlt->dtype.code) {
    case kDLInt:
      et = IREE_HAL_ELEMENT_TYPE_VALUE(IREE_HAL_NUMERICAL_TYPE_INTEGER_SIGNED,
                                       dlt->dtype.bits);
      break;
    case kDLUInt:
      et = IREE_HAL_ELEMENT_TYPE_VALUE(IREE_HAL_NUMERICAL_TYPE_INTEGER_UNSIGNED,
                                       dlt->dtype.bits);
      break;
    case kDLFloat:
      et = IREE_HAL_ELEMENT_TYPE_VALUE(IREE_HAL_NUMERICAL_TYPE_FLOAT_IEEE,
                                       dlt->dtype.bits);
      break;
    case kDLBfloat:
      et = IREE_HAL_ELEMENT_TYPE_VALUE(IREE_HAL_NUMERICAL_TYPE_FLOAT_BRAIN,
                                       dlt->dtype.bits);
      break;
    case kDLComplex:
      et = IREE_HAL_ELEMENT_TYPE_VALUE(IREE_HAL_NUMERICAL_TYPE_FLOAT_COMPLEX,
                                       dlt->dtype.bits);
      break;
    case kDLBool:
      et = IREE_HAL_ELEMENT_TYPE_VALUE(IREE_HAL_NUMERICAL_TYPE_BOOLEAN,
                                       dlt->dtype.bits);
      break;
    default:
      throw std::invalid_argument("Unsupported dlpack dtype code");
  }

  // Verify dense row major strides (for now a requirement).
  if (dlt->strides && dlt->ndim > 0) {
    int64_t stride = 1;
    for (int32_t i = dlt->ndim - 1; i >= 0; --i) {
      auto dim = dlt->shape[i];
      // The stride value for 1 or 0 dims is undefined and dlpack can normalize
      // it, so we skip validation for these.
      // See:
      // https://github.com/pytorch/pytorch/issues/99803#issuecomment-1521214463
      if (dim == 1 || dim == 0) continue;
      if (dlt->strides[i] != stride) {
        throw std::invalid_argument("Unsupported strided tensor");
      }
      stride *= dim;
    }
  }

  // Verify no byte offset. We could technically allow this, but there are all
  // kinds of bugs and caveats listed, and would like to see how it is used.
  if (dlt->byte_offset != 0) {
    throw std::invalid_argument("NYI: dlpack byte_offset != 0");
  }

  // Compute size.
  auto* dims = static_cast<iree_hal_dim_t*>(
      iree_alloca(sizeof(iree_hal_dim_t) * dlt->ndim));
  iree_device_size_t byte_size = iree_hal_element_bit_count(et);
  if (dlt->ndim > 0) {
    for (int32_t i = 0; i < dlt->ndim; ++i) {
      byte_size *= dlt->shape[i];
      dims[i] = dlt->shape[i];
    }
  }
  if ((byte_size % 8) != 0) {
    throw std::invalid_argument(
        "dlpack tensor does not have a byte aligned size");
  }
  byte_size /= 8;

  iree_hal_buffer_t* imported_buffer;
  iree_hal_allocator_t* allocator = iree_hal_device_allocator(raw_ptr());
  iree_hal_buffer_params_t params;
  memset(&params, 0, sizeof(params));
  params.usage = IREE_HAL_BUFFER_USAGE_DEFAULT;
  params.access = IREE_HAL_MEMORY_ACCESS_ANY;
  params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  iree_hal_external_buffer_t external_buffer;
  memset(&external_buffer, 0, sizeof(external_buffer));
  external_buffer.type = IREE_HAL_EXTERNAL_BUFFER_TYPE_DEVICE_ALLOCATION;
  external_buffer.size = byte_size;
  external_buffer.handle.device_allocation.ptr =
      reinterpret_cast<uint64_t>(dlt->data);
  iree_hal_buffer_release_callback_t release_callback = {
      +[](void* user_data, struct iree_hal_buffer_t* buffer) {
        auto managed_tensor = static_cast<DLManagedTensor*>(user_data);
        if (managed_tensor->deleter) {
          managed_tensor->deleter(managed_tensor);
        }
      },
      state.raw,
  };
  CheckApiStatus(
      iree_hal_allocator_import_buffer(allocator, params, &external_buffer,
                                       release_callback, &imported_buffer),
      "Could not import external device buffer");
  state.managed_tensor = nullptr;  // Ownership transferred.

  // Create Buffer View.
  iree_hal_buffer_view_t* buffer_view;
  iree_status_t status =
      iree_hal_buffer_view_create(imported_buffer, dlt->ndim, dims, et,
                                  IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
                                  iree_allocator_system(), &buffer_view);

  if (!iree_status_is_ok(status)) {
    iree_hal_buffer_release(imported_buffer);
    CheckApiStatus(status, "Failed to create buffer view");
  }

  return HalBufferView::StealFromRawPtr(buffer_view);
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

HalDriver::DeviceUri::DeviceUri(const std::string& device_uri) {
  iree_string_view_t device_uri_sv{
      device_uri.data(), static_cast<iree_host_size_t>(device_uri.size())};
  iree_uri_split(device_uri_sv, &driver_name, &device_path, &params_str);
}

py::object HalDriver::Create(const DeviceUri& device_uri) {
  iree_hal_driver_t* driver;
  CheckApiStatus(iree_hal_driver_registry_try_create(
                     iree_hal_driver_registry_default(), device_uri.driver_name,
                     iree_allocator_system(), &driver),
                 "Error creating driver");

  py::object driver_obj = py::cast(HalDriver::StealFromRawPtr(driver));
  return driver_obj;
}

py::object HalDriver::Create(const std::string& device_uri) {
  DeviceUri parsed_uri(device_uri);
  return HalDriver::Create(parsed_uri);
}

py::object HalDriver::Create(const std::string& device_uri,
                             py::dict& driver_cache) {
  // Look up the driver by driver name in the cache, and return it if found.
  DeviceUri parsed_uri(device_uri);
  py::str cache_key(parsed_uri.driver_name.data, parsed_uri.driver_name.size);
  py::object cached = driver_cache.attr("get")(cache_key);
  if (!cached.is_none()) {
    return cached;
  }

  // Create a new driver and put it in the cache.
  py::object driver_obj = HalDriver::Create(parsed_uri);
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
                                     std::optional<py::list> allocators) {
  // Optionally wrap the base device allocator with caching/pooling.
  // Doing this here satisfies the requirement that no buffers have been
  // allocated yet - if we returned the device without doing this the caller
  // can more easily break the rules.
  if (allocators) {
    // NOTE: we need to pass string views that point to the std::string storage.
    // We do that in two passes because as we grow spec_storage it may
    // reallocate itself and invalidate the pointers - only after we're done
    // can we capture them in views.
    auto& spec_list = *allocators;
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

  IREE_RETURN_IF_ERROR(iree_hal_device_set_default_channel_provider(device));

  return iree_ok_status();
}

HalDevice HalDriver::CreateDefaultDevice(std::optional<py::list> allocators) {
  iree_hal_device_t* device;
  CheckApiStatus(iree_hal_driver_create_default_device(
                     raw_ptr(), iree_allocator_system(), &device),
                 "Error creating default device");
  CheckApiStatus(ConfigureDevice(device, allocators),
                 "Error configuring the device");
  return HalDevice::StealFromRawPtr(device);
}

HalDevice HalDriver::CreateDevice(iree_hal_device_id_t device_id,
                                  std::optional<py::list> allocators) {
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
    if (found_device_id.equal(compare_device_id)) {
      found = true;
      break;
    }
  }

  if (!found) {
    std::string msg;
    msg.append("Device id ");
    msg.append(std::to_string(device_id));
    msg.append(" not found. Available devices: ");
    msg.append(py::cast<std::string>(py::repr(available_devices)));
    throw std::invalid_argument(std::move(msg));
  }

  std::vector<iree_string_pair_t> params;
  iree_hal_device_t* device;
  CheckApiStatus(iree_hal_driver_create_device_by_id(
                     raw_ptr(), device_id, params.size(),
                     (params.empty() ? nullptr : &params.front()),
                     iree_allocator_system(), &device),
                 "Error creating default device");
  CheckApiStatus(ConfigureDevice(device, allocators),
                 "Error configuring the device");
  return HalDevice::StealFromRawPtr(device);
}

HalDevice HalDriver::CreateDeviceByURI(std::string& device_uri,
                                       std::optional<py::list> allocators) {
  iree_hal_device_t* device;
  iree_string_view_t device_uri_sv{
      device_uri.data(), static_cast<iree_host_size_t>(device_uri.size())};
  CheckApiStatus(
      iree_hal_driver_create_device_by_uri(raw_ptr(), device_uri_sv,
                                           iree_allocator_system(), &device),
      "Error creating device");
  CheckApiStatus(ConfigureDevice(device, allocators),
                 "Error configuring the device");
  return HalDevice::StealFromRawPtr(device);
}

//------------------------------------------------------------------------------
// HAL module
//------------------------------------------------------------------------------

VmModule CreateHalModule(
    VmInstance* instance, std::optional<HalDevice*> device,
    std::optional<py::list> devices,
    std::optional<py::ref<HalModuleDebugSink>> debug_sink) {
  if (device && devices) {
    PyErr_SetString(
        PyExc_ValueError,
        "\"device\" and \"devices\" are mutually exclusive arguments.");
  }
  std::vector<iree_hal_device_t*> devices_vector;
  iree_hal_device_t* device_ptr;
  iree_hal_device_t** devices_ptr;
  iree_host_size_t device_count;
  iree_vm_module_t* module = NULL;
  if (device) {
    device_ptr = device.value()->raw_ptr();
    devices_ptr = &device_ptr;
    device_count = 1;
  } else {
    // Set device related arguments in the case of multiple devices.
    devices_vector.reserve(devices->size());
    for (auto devicesIt = devices->begin(); devicesIt != devices->end();
         ++devicesIt) {
      devices_vector.push_back(py::cast<HalDevice*>(*devicesIt)->raw_ptr());
    }
    devices_ptr = devices_vector.data();
    device_count = devices_vector.size();
  }

  iree_hal_module_debug_sink_t iree_hal_module_debug_sink =
      iree_hal_module_debug_sink_stdio(stderr);
  if (debug_sink) {
    iree_hal_module_debug_sink = (*debug_sink)->AsIreeHalModuleDebugSink();
  }

  CheckApiStatus(iree_hal_module_create(instance->raw_ptr(), device_count,
                                        devices_ptr, IREE_HAL_MODULE_FLAG_NONE,
                                        iree_hal_module_debug_sink,
                                        iree_allocator_system(), &module),
                 "Error creating hal module");
  VmModule vm_module = VmModule::StealFromRawPtr(module);
  if (debug_sink) {
    // Retain a reference. We want the callback to be valid after
    // the user has dropped its reference to the HAL module Python object and
    // not burden the user with lifetime management.
    // The counter will be decremented once the IREE runtime does not use the
    // debug sink anymore.
    (*debug_sink)->inc_ref();
  }
  return vm_module;
}

HalModuleDebugSink::HalModuleDebugSink(
    HalModuleBufferViewTraceCallback buffer_view_trace_callback)
    : buffer_view_trace_callback_(buffer_view_trace_callback) {}

iree_hal_module_debug_sink_t HalModuleDebugSink::AsIreeHalModuleDebugSink()
    const {
  iree_hal_module_debug_sink_t res;
  memset(&res, 0, sizeof(res));
  res.buffer_view_trace.fn = HalModuleDebugSink::IreeHalModuleBufferViewTrace;
  res.buffer_view_trace.user_data = const_cast<HalModuleDebugSink*>(this);
  res.destroy.fn = HalModuleDebugSink::DestroyCallback;
  res.destroy.user_data = const_cast<HalModuleDebugSink*>(this);
  return res;
}

HalModuleBufferViewTraceCallback&
HalModuleDebugSink::GetHalModuleBufferViewTraceCallback() {
  return this->buffer_view_trace_callback_;
}

static std::vector<HalBufferView> CreateHalBufferViewVector(
    iree_host_size_t buffer_view_count, iree_hal_buffer_view_t** buffer_views) {
  std::vector<HalBufferView> res;
  res.reserve(buffer_view_count);
  std::transform(buffer_views, buffer_views + buffer_view_count,
                 std::back_inserter(res),
                 [](iree_hal_buffer_view_t* buffer_view) {
                   return HalBufferView::BorrowFromRawPtr(buffer_view);
                 });
  return res;
}

iree_status_t HalModuleDebugSink::DestroyCallback(void* user_data) {
  HalModuleDebugSink* debug_sink =
      reinterpret_cast<HalModuleDebugSink*>(user_data);
  debug_sink->dec_ref();
  return iree_ok_status();
}

iree_status_t HalModuleDebugSink::IreeHalModuleBufferViewTrace(
    void* user_data, iree_string_view_t key, iree_host_size_t buffer_view_count,
    iree_hal_buffer_view_t** buffer_views, iree_allocator_t host_allocator) {
  auto debug_sink = reinterpret_cast<HalModuleDebugSink*>(user_data);
  std::vector<HalBufferView> buffer_views_vec =
      CreateHalBufferViewVector(buffer_view_count, buffer_views);
  try {
    debug_sink->buffer_view_trace_callback_(std::string(key.data, key.size),
                                            buffer_views_vec);
  } catch (const py::python_error& e) {
    return iree_make_status(IREE_STATUS_UNKNOWN, "%s", e.what());
  }

  return iree_ok_status();
}

static int HalModuleDebugSinkTpTraverse(PyObject* self, visitproc visit,
                                        void* arg) {
  // Inform Python's garbage collector about the references we hold.

  // Retrieve a pointer to the C++ instance associated with 'self'
  // (never fails)
  HalModuleDebugSink* debug_sink = py::inst_ptr<HalModuleDebugSink>(self);

  // Although we are not tracking cycles involving the HAL module or VM context
  // we still want to properly destroy the callback and let the GC know what
  // references we hold. If debug_sink->GetHalModuleBufferViewTraceCallback()
  // has an associated CPython object, return it. If not, value.ptr() will equal
  // NULL, which is also fine.
  py::handle buffer_view_trace_callback =
      py::find(debug_sink->GetHalModuleBufferViewTraceCallback());

  // Inform the Python GC about the instance.
  Py_VISIT(buffer_view_trace_callback.ptr());

  return 0;
}

int HalModuleDebugSinkTpClear(PyObject* self) {
  // Retrieve a pointer to the C++ instance associated with 'self'
  // (never fails)
  HalModuleDebugSink* debug_sink = py::inst_ptr<HalModuleDebugSink>(self);
  debug_sink->GetHalModuleBufferViewTraceCallback() = nullptr;

  return 0;
}

//------------------------------------------------------------------------------
// Bindings
//------------------------------------------------------------------------------

void SetupHalBindings(nanobind::module_ m) {
  py::dict driver_cache;

  // Built-in module creation.
  m.def("create_hal_module", &CreateHalModule, py::arg("instance"),
        py::arg("device") = py::none(), py::arg("devices") = py::none(),
        py::arg("debug_sink") = py::none());

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
      .def("__or__", [](uint64_t self, uint64_t other) { return self | other; })
      .def("__and__",
           [](uint64_t self, uint64_t other) { return self & other; })
      .def("__int__", [](enum iree_hal_memory_type_bits_t self) {
        return (uint64_t)self;
      });

  py::enum_<enum iree_hal_buffer_compatibility_bits_t>(m, "BufferCompatibility")
      .value("NONE", IREE_HAL_BUFFER_COMPATIBILITY_NONE)
      .value("ALLOCATABLE", IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE)
      .value("IMPORTABLE", IREE_HAL_BUFFER_COMPATIBILITY_IMPORTABLE)
      .value("EXPORTABLE", IREE_HAL_BUFFER_COMPATIBILITY_EXPORTABLE)
      .value("QUEUE_TRANSFER", IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_TRANSFER)
      .value("QUEUE_DISPATCH", IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_DISPATCH)
      .export_values()
      .def("__or__", [](uint64_t self, uint64_t other) { return self | other; })
      .def("__and__",
           [](uint64_t self, uint64_t other) { return self & other; })
      .def("__int__", [](enum iree_hal_buffer_compatibility_bits_t self) {
        return (uint64_t)self;
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
      .def("__or__", [](enum iree_hal_buffer_usage_bits_t self,
                        uint64_t other) { return self | other; })
      .def("__and__", [](enum iree_hal_buffer_usage_bits_t self,
                         uint64_t other) { return self & other; })
      .def("__int__", [](enum iree_hal_buffer_usage_bits_t self) {
        return (uint64_t)self;
      });

  py::enum_<enum iree_hal_memory_access_bits_t>(m, "MemoryAccess")
      .value("NONE", IREE_HAL_MEMORY_ACCESS_NONE)
      .value("READ", IREE_HAL_MEMORY_ACCESS_READ)
      .value("WRITE", IREE_HAL_MEMORY_ACCESS_WRITE)
      .value("DISCARD", IREE_HAL_MEMORY_ACCESS_DISCARD)
      .value("DISCARD_WRITE", IREE_HAL_MEMORY_ACCESS_DISCARD_WRITE)
      .value("ALL", IREE_HAL_MEMORY_ACCESS_ALL)
      .export_values()
      .def("__or__", [](uint64_t self, uint64_t other) { return self | other; })
      .def("__and__",
           [](uint64_t self, uint64_t other) { return self & other; })
      .def("__int__", [](enum iree_hal_memory_access_bits_t self) {
        return (uint64_t)self;
      });

  // Use compatibility type to enable def_static.
  // See: https://github.com/wjakob/nanobind/issues/597
  auto hal_element_type = nanobind1_compat_enum_<enum iree_hal_element_types_t>(
      m, "HalElementType");
  hal_element_type
      .def_static("map_to_dtype",
                  [](iree_hal_element_type_t element_type) {
                    int typenum = numpy::ConvertHalElementTypeToNumPyTypeNum(
                        element_type);
                    return numpy::DescrNewFromType(typenum);
                  })
      .def_static("is_byte_aligned",
                  [](iree_hal_element_type_t element_type) {
                    return iree_hal_element_is_byte_aligned(element_type);
                  })
      .def_static("dense_byte_count", [](iree_hal_element_type_t element_type) {
        return iree_hal_element_dense_byte_count(element_type);
      });
  hal_element_type.value("NONE", IREE_HAL_ELEMENT_TYPE_NONE)
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
      .value("FLOAT_8_E4M3", IREE_HAL_ELEMENT_TYPE_FLOAT_8_E4M3)
      .value("FLOAT_8_E4M3_FNUZ", IREE_HAL_ELEMENT_TYPE_FLOAT_8_E4M3_FNUZ)
      .value("FLOAT_8_E5M2", IREE_HAL_ELEMENT_TYPE_FLOAT_8_E5M2)
      .value("FLOAT_8_E5M2_FNUZ", IREE_HAL_ELEMENT_TYPE_FLOAT_8_E5M2_FNUZ)
      .export_values()
      .def("__int__",
           [](enum iree_hal_element_types_t self) { return (uint64_t)self; });

  py::class_<HalDevice>(m, "HalDevice")
      .def_prop_ro(
          "allocator",
          [](HalDevice& self) {
            return HalAllocator::BorrowFromRawPtr(self.allocator());
          },
          py::keep_alive<0, 1>())
      .def("begin_profiling", &HalDevice::BeginProfiling,
           py::arg("mode") = py::none(), py::arg("file_path") = py::none())
      .def("flush_profiling", &HalDevice::FlushProfiling)
      .def("end_profiling", &HalDevice::EndProfiling)
      .def("create_semaphore", &HalDevice::CreateSemaphore,
           py::arg("initial_value"))
      .def("queue_alloca", &HalDevice::QueueAlloca, py::arg("allocation_size"),
           py::arg("wait_semaphores"), py::arg("signal_semaphores"),
           kHalDeviceQueueAlloca)
      .def("queue_dealloca", &HalDevice::QueueDealloca, py::arg("buffer"),
           py::arg("wait_semaphores"), py::arg("signal_semaphores"),
           kHalDeviceQueueDealloca)
      .def("queue_execute", &HalDevice::QueueExecute,
           py::arg("command_buffers"), py::arg("wait_semaphores"),
           py::arg("signal_semaphores"), kHalDeviceQueueExecute)
      .def("queue_copy", &HalDevice::QueueCopy, py::arg("source_buffer"),
           py::arg("target_buffer"), py::arg("wait_semaphores"),
           py::arg("signal_semaphores"), kHalDeviceQueueCopy)
      .def("create_dlpack_capsule", &HalDevice::CreateDLPackCapsule,
           py::arg("buffer_view"), py::arg("device_type_code"),
           py::arg("device_id"))
      .def("from_dlpack_capsule", &HalDevice::FromDLPackCapsule)
      .def("__repr__", [](HalDevice& self) {
        auto id_sv = iree_hal_device_id(self.raw_ptr());
        return std::string(id_sv.data, id_sv.size);
      });

  py::class_<HalDriver>(m, "HalDriver")
      .def_static("query", &HalDriver::Query)

      // All 'create_device' functions take optional kwargs that should be kept
      // in sync.
      .def("create_default_device", &HalDriver::CreateDefaultDevice,
           py::keep_alive<0, 1>(), py::arg("allocators") = py::none())
      .def("create_device", &HalDriver::CreateDevice, py::keep_alive<0, 1>(),
           py::arg("device_id"), py::arg("allocators") = py::none())
      .def("create_device_by_uri", &HalDriver::CreateDeviceByURI,
           py::keep_alive<0, 1>(), py::arg("device_uri"),
           py::arg("allocators") = py::none())
      .def(
          "create_device",
          [](HalDriver& self, py::dict device_info,
             std::optional<py::list> allocators) -> HalDevice {
            // Alias of create_device that takes a dict as returned from
            // query_available_devices for convenience.
            auto device_id =
                py::cast<iree_hal_device_id_t>(device_info["device_id"]);
            return self.CreateDevice(device_id, allocators);
          },
          py::keep_alive<0, 1>(), py::arg("device_info"),
          py::arg("allocators") = py::none())
      .def("query_available_devices", &HalDriver::QueryAvailableDevices)
      .def("dump_device_info",
           [](HalDriver& self, iree_hal_device_id_t device_id) {
             iree_string_builder_t builder;
             iree_string_builder_initialize(iree_allocator_system(), &builder);
             CheckApiStatus(iree_hal_driver_dump_device_info(
                                self.raw_ptr(), device_id, &builder),
                            "Querying device info");
             iree_string_view_t view = iree_string_builder_view(&builder);
             py::str result(view.data, view.size);
             iree_string_builder_deinitialize(&builder);
             return result;
           });

  m.def(
      "get_cached_hal_driver",
      [driver_cache](std::string device_uri) {
        return HalDriver::Create(device_uri,
                                 const_cast<py::dict&>(driver_cache));
      },
      py::arg("device_uri"));

  m.def(
      "create_hal_driver",
      [](std::string device_uri) { return HalDriver::Create(device_uri); },
      py::arg("device_uri"));

  m.def("clear_hal_driver_cache",
        [driver_cache]() { const_cast<py::dict&>(driver_cache).clear(); });

  py::class_<HalAllocator>(m, "HalAllocator")
      .def("trim",
           [](HalAllocator& self) {
             CheckApiStatus(iree_hal_allocator_trim(self.raw_ptr()),
                            "Error trim()'ing HAL allocator");
           })
      .def_prop_ro(
          "has_statistics",
          [](HalAllocator& self) -> bool { return IREE_STATISTICS_ENABLE; })
      .def_prop_ro("statistics", &HalAllocator::QueryStatistics)
      .def_prop_ro("formatted_statistics", &HalAllocator::FormattedStatistics)
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
            CheckApiStatus(
                iree_hal_allocator_allocate_buffer(self.raw_ptr(), params,
                                                   allocation_size, &buffer),
                "could not allocate buffer");
            return HalBuffer::StealFromRawPtr(buffer);
          },
          py::arg("memory_type"), py::arg("allowed_usage"),
          py::arg("allocation_size"), py::keep_alive<0, 1>(),
          "Allocates a new buffer with requested characteristics (does not "
          "initialize with specific data).")
      .def("allocate_buffer_copy", &HalAllocator::AllocateBufferCopy,
           py::arg("memory_type"), py::arg("allowed_usage"), py::arg("device"),
           py::arg("buffer"), py::arg("element_type") = py::none(),
           py::keep_alive<0, 1>(),
           "Allocates a new buffer and initializes it from a Python buffer "
           "object. If an element type is specified, wraps in a BufferView "
           "matching the characteristics of the Python buffer. The format is "
           "requested as ND/C-Contiguous, which may incur copies if not "
           "already in that format.")
      .def("allocate_host_staging_buffer_copy",
           &HalAllocator::AllocateHostStagingBufferCopy, py::arg("device"),
           py::arg("initial_contents"), py::keep_alive<0, 1>(),
           "Allocates a new buffer and initializes it from a Python buffer "
           "object. The buffer is configured as optimal for use on the device "
           "as a transfer buffer. For buffers of unknown providence, this is a "
           "last resort method for making them compatible for transfer to "
           "arbitrary devices.");

  auto hal_buffer = py::class_<HalBuffer>(m, "HalBuffer");
  VmRef::BindRefProtocol(hal_buffer, iree_hal_buffer_type,
                         iree_hal_buffer_retain_ref, iree_hal_buffer_deref,
                         iree_hal_buffer_isa);
  hal_buffer
      .def("fill_zero", &HalBuffer::FillZero, py::arg("byte_offset"),
           py::arg("byte_length"))
      .def("byte_length", &HalBuffer::byte_length)
      .def("memory_type", &HalBuffer::memory_type)
      .def("allowed_usage", &HalBuffer::allowed_usage)
      .def("create_view", &HalBuffer::CreateView, py::arg("shape"),
           py::arg("element_size"), py::keep_alive<0, 1>())
      .def("map", HalMappedMemory::CreateFromBuffer, py::keep_alive<0, 1>())
      .def("__repr__", &HalBuffer::Repr);

  auto hal_buffer_view = py::class_<HalBufferView>(m, "HalBufferView");
  VmRef::BindRefProtocol(hal_buffer_view, iree_hal_buffer_view_type,
                         iree_hal_buffer_view_retain_ref,
                         iree_hal_buffer_view_deref, iree_hal_buffer_view_isa);
  hal_buffer_view.def(
      "__init__",
      [](HalBufferView* new_self, HalBuffer& buffer, py::handle shape,
         iree_hal_element_type_t element_type) {
        size_t rank = py::len(shape);
        iree_hal_dim_t* dims =
            static_cast<iree_hal_dim_t*>(alloca(sizeof(iree_hal_dim_t) * rank));
        for (size_t i = 0; i < rank; ++i) {
          dims[i] = py::cast<iree_hal_dim_t>(shape[i]);
        }
        iree_hal_buffer_view_t* out_bv;
        CheckApiStatus(iree_hal_buffer_view_create(
                           buffer.raw_ptr(), rank, dims, element_type,
                           IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
                           iree_allocator_system(), &out_bv),
                       "creating buffer view");
        new (new_self) HalBufferView();
        *new_self = HalBufferView::StealFromRawPtr(out_bv);
      },
      py::arg("buffer"), py::arg("shape"), py::arg("element_type"));
  hal_buffer_view
      .def("map", HalMappedMemory::CreateFromBufferView, py::keep_alive<0, 1>())
      .def("get_buffer", HalBuffer::CreateFromBufferView,
           py::keep_alive<0, 1>())
      .def_prop_ro("shape",
                   [](HalBufferView& self) {
                     iree_host_size_t rank =
                         iree_hal_buffer_view_shape_rank(self.raw_ptr());
                     auto* dims =
                         iree_hal_buffer_view_shape_dims(self.raw_ptr());
                     py::list result;
                     for (iree_host_size_t i = 0; i < rank; ++i) {
                       result.append(dims[i]);
                     }
                     return result;
                   })
      .def_prop_ro("element_type",
                   [](HalBufferView& self) {
                     return iree_hal_buffer_view_element_type(self.raw_ptr());
                   })
      .def_prop_ro("byte_length",
                   [](HalBufferView& self) {
                     return iree_hal_buffer_view_byte_length(self.raw_ptr());
                   })
      .def("__repr__", &HalBufferView::Repr);

  py::class_<HalSemaphore>(m, "HalSemaphore")
      .def(
          "fail",
          [](HalSemaphore& self, std::string& message) {
            // TODO: Take some category enum and use that is available.
            iree_status_t status =
                iree_make_status(IREE_STATUS_UNKNOWN, "%s", message.c_str());
            iree_hal_semaphore_fail(self.raw_ptr(), status);
          },
          py::arg("message"))
      .def("query",
           [](HalSemaphore& self) {
             uint64_t out_value;
             CheckApiStatus(
                 iree_hal_semaphore_query(self.raw_ptr(), &out_value),
                 "querying semaphore");
             return out_value;
           })
      .def("signal",
           [](HalSemaphore& self, uint64_t new_value) {
             CheckApiStatus(
                 iree_hal_semaphore_signal(self.raw_ptr(), new_value),
                 "signaling semaphore");
           })
      .def(
          "wait",
          [](HalSemaphore& self, uint64_t payload,
             std::optional<iree_duration_t> timeout,
             std::optional<iree_time_t> deadline) -> bool {
            iree_timeout_t t = NormalizeTimeout(timeout, deadline);
            iree_status_t status;
            uint64_t unused_value;
            {
              py::gil_scoped_release release;
              status = iree_hal_semaphore_wait(self.raw_ptr(), payload, t);
            }
            if (iree_status_is_deadline_exceeded(status)) {
              // Time out.
              return false;
            } else if (iree_status_is_aborted(status)) {
              // Synchronous failure.
              iree_status_ignore(status);
              status = iree_hal_semaphore_query(self.raw_ptr(), &unused_value);
              if (iree_status_is_ok(status)) {
                status = iree_make_status(
                    IREE_STATUS_FAILED_PRECONDITION,
                    "expected synchronous status failure missing");
              }
              CheckApiStatus(status, "synchronous semaphore failure");
            } else {
              // General failure check.
              CheckApiStatus(status, "waiting for semaphore");
            }

            // Asynchronous failure.
            status = iree_hal_semaphore_query(self.raw_ptr(), &unused_value);
            if (iree_status_is_deferred(status)) {
              return false;
            }
            CheckApiStatus(status, "asynchronous semaphore failure");
            return true;
          },
          py::arg("payload"), py::arg("timeout") = py::none(),
          py::arg("deadline") = py::none(), kHalWait);

  auto hal_fence = py::class_<HalFence>(m, "HalFence");
  VmRef::BindRefProtocol(hal_fence, iree_hal_fence_type,
                         iree_hal_fence_retain_ref, iree_hal_fence_deref,
                         iree_hal_fence_isa);
  hal_fence
      .def(
          "__init__",
          [](HalFence* new_fence, iree_host_size_t capacity) {
            iree_hal_fence_t* out_fence;
            CheckApiStatus(iree_hal_fence_create(
                               capacity, iree_allocator_system(), &out_fence),
                           "creating fence");
            new (new_fence) HalFence();
            (*new_fence) = HalFence::StealFromRawPtr(out_fence);
          },
          py::arg("capacity"))
      .def_static(
          "create_at",
          [](HalSemaphore& sem, uint64_t value) {
            iree_hal_fence_t* out_fence;
            CheckApiStatus(
                iree_hal_fence_create_at(sem.raw_ptr(), value,
                                         iree_allocator_system(), &out_fence),
                "creating fence");
            return HalFence::StealFromRawPtr(out_fence);
          },
          py::arg("sem"), py::arg("value"))
      .def_static(
          "join",
          [](py::sequence fences) {
            size_t count = py::len(fences);
            iree_hal_fence_t** fence_ptrs = static_cast<iree_hal_fence_t**>(
                alloca(sizeof(iree_hal_fence_t*) * count));
            for (size_t i = 0; i < count; ++i) {
              fence_ptrs[i] = py::cast<HalFence*>(fences[i])->raw_ptr();
            }
            iree_hal_fence_t* out_fence;
            CheckApiStatus(
                iree_hal_fence_join(count, fence_ptrs, iree_allocator_system(),
                                    &out_fence),
                "joining fences");
            return HalFence::StealFromRawPtr(out_fence);
          },
          py::arg("fences"))
      .def_prop_ro("timepoint_count",
                   [](HalFence& self) {
                     return iree_hal_fence_timepoint_count(self.raw_ptr());
                   })
      .def(
          "insert",
          [](HalFence& self, HalSemaphore& sem, uint64_t value) {
            CheckApiStatus(
                iree_hal_fence_insert(self.raw_ptr(), sem.raw_ptr(), value),
                "insertint into fence");
          },
          py::arg("sem"), py::arg("value"))
      .def(
          "extend",
          [](HalFence& self, HalFence& from_fence) {
            CheckApiStatus(
                iree_hal_fence_extend(self.raw_ptr(), from_fence.raw_ptr()),
                "extending fence");
          },
          py::arg("from_fence"))
      .def(
          "fail",
          [](HalFence& self, std::string& message) {
            // TODO: Take some category enum and use that is available.
            iree_status_t status =
                iree_make_status(IREE_STATUS_UNKNOWN, "%s", message.c_str());
            iree_hal_fence_fail(self.raw_ptr(), status);
          },
          py::arg("message"))
      .def("signal",
           [](HalFence& self) {
             CheckApiStatus(iree_hal_fence_signal(self.raw_ptr()),
                            "signalling fence");
           })
      .def(
          "wait",
          [](HalFence& self, std::optional<iree_duration_t> timeout,
             std::optional<iree_time_t> deadline) -> bool {
            iree_timeout_t t = NormalizeTimeout(timeout, deadline);
            iree_status_t status;
            {
              py::gil_scoped_release release;
              status = iree_hal_fence_wait(self.raw_ptr(), t);
            }
            if (iree_status_is_deadline_exceeded(status)) {
              // Time out.
              return false;
            } else if (iree_status_is_aborted(status)) {
              // Synchronous failure.
              iree_status_ignore(status);
              status = iree_hal_fence_query(self.raw_ptr());
              if (iree_status_is_ok(status)) {
                status = iree_make_status(
                    IREE_STATUS_FAILED_PRECONDITION,
                    "expected synchronous status failure missing");
              }
              CheckApiStatus(status, "synchronous fence failure");
            } else {
              // General failure check.
              CheckApiStatus(status, "waiting for fence");
            }

            // Asynchronous failure.
            status = iree_hal_fence_query(self.raw_ptr());
            if (iree_status_is_deferred(status)) {
              return false;
            }
            CheckApiStatus(status, "asynchronous fence failure");
            return true;
          },
          py::arg("timeout") = py::none(), py::arg("deadline") = py::none(),
          kHalWait);

  py::class_<HalMappedMemory>(m, "MappedMemory")
      .def(
          "asarray",
          [](HalMappedMemory* self, py::handle shape, py::object dtype_descr) {
            py::object py_mapped_memory = py::cast(self);
            size_t rank = py::len(shape);
            intptr_t* dims =
                static_cast<intptr_t*>(alloca(sizeof(intptr_t) * rank));
            for (size_t i = 0; i < rank; ++i) {
              dims[i] = py::cast<intptr_t>(shape[i]);
            }
            int typenum = numpy::TypenumFromDescr(dtype_descr);
            return numpy::SimpleNewFromData(rank, dims, typenum,
                                            self->mapped_memory().contents.data,
                                            py_mapped_memory);
          },
          py::arg("shape"), py::arg("numpy_dtype_descr"));

  py::class_<HalShape>(m, "Shape")
      .def("__init__", [](HalShape* self, std::vector<iree_hal_dim_t> indices) {
        new (self) HalShape(indices);
      });

  py::class_<HalCommandBuffer>(m, "HalCommandBuffer")
      .def(
          "__init__",
          [](HalCommandBuffer* new_self, HalDevice& device,
             iree_host_size_t binding_capacity, bool begin) {
            iree_hal_command_buffer_t* out_cb;
            CheckApiStatus(iree_hal_command_buffer_create(
                               device.raw_ptr(),
                               /*mode=*/IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
                               /*categories=*/IREE_HAL_COMMAND_CATEGORY_ANY,
                               /*queue_affinity=*/IREE_HAL_QUEUE_AFFINITY_ANY,
                               binding_capacity, &out_cb),
                           "creating command buffer");
            HalCommandBuffer cb = HalCommandBuffer::StealFromRawPtr(out_cb);
            if (begin) {
              CheckApiStatus(iree_hal_command_buffer_begin(cb.raw_ptr()),
                             "command buffer begin");
            }
            new (new_self) HalCommandBuffer();
            *new_self = std::move(cb);
          },
          py::arg("device"), py::arg("binding_capacity") = 0,
          py::arg("begin") = true)
      .def("begin",
           [](HalCommandBuffer& self) {
             CheckApiStatus(iree_hal_command_buffer_begin(self.raw_ptr()),
                            "command buffer begin");
           })
      .def("end",
           [](HalCommandBuffer& self) {
             CheckApiStatus(iree_hal_command_buffer_end(self.raw_ptr()),
                            "command buffer end");
           })
      .def(
          "copy",
          [](HalCommandBuffer& self, HalBuffer& source_buffer,
             HalBuffer& target_buffer, iree_device_size_t source_offset,
             iree_device_size_t target_offset,
             std::optional<iree_device_size_t> length, bool end) {
            iree_device_size_t resolved_length;
            if (length) {
              resolved_length = *length;
            } else {
              resolved_length =
                  iree_hal_buffer_byte_length(source_buffer.raw_ptr());
              if (resolved_length !=
                  iree_hal_buffer_byte_length(target_buffer.raw_ptr())) {
                throw std::invalid_argument(
                    "If length is not provided, source and target bufer length "
                    "must match and it does not. Provide explicit length=");
              }
            }
            CheckApiStatus(
                iree_hal_command_buffer_copy_buffer(
                    self.raw_ptr(),
                    iree_hal_make_buffer_ref(source_buffer.raw_ptr(),
                                             source_offset, resolved_length),
                    iree_hal_make_buffer_ref(target_buffer.raw_ptr(),
                                             target_offset, resolved_length),
                    IREE_HAL_COPY_FLAG_NONE),
                "copy command");
            if (end) {
              CheckApiStatus(iree_hal_command_buffer_end(self.raw_ptr()),
                             "command buffer end");
            }
          },
          py::arg("source_buffer"), py::arg("target_buffer"),
          py::arg("source_offset") = 0, py::arg("target_offset") = 0,
          py::arg("length") = py::none(), py::arg("end") = false,
          "Copies a range from a source to target buffer. If the length is "
          "not specified, then it is taken from the source/target buffer, "
          "which must match.")
      .def(
          "fill",
          [](HalCommandBuffer& self, HalBuffer& target_buffer,
             py::handle pattern, iree_device_size_t target_offset,
             std::optional<iree_device_size_t> length, bool end) {
            Py_buffer pattern_view;
            int flags = PyBUF_FORMAT | PyBUF_ND;
            if (PyObject_GetBuffer(pattern.ptr(), &pattern_view, flags) != 0) {
              // The GetBuffer call is required to set an appropriate error.
              throw py::python_error();
            }
            PyBufferReleaser py_pattern_releaser(pattern_view);

            iree_device_size_t resolved_length;
            if (length) {
              resolved_length = *length;
            } else {
              resolved_length =
                  iree_hal_buffer_byte_length(target_buffer.raw_ptr());
            }
            CheckApiStatus(
                iree_hal_command_buffer_fill_buffer(
                    self.raw_ptr(),
                    iree_hal_make_buffer_ref(target_buffer.raw_ptr(),
                                             target_offset, resolved_length),
                    pattern_view.buf, pattern_view.len,
                    IREE_HAL_FILL_FLAG_NONE),
                "command buffer fill");
            if (end) {
              CheckApiStatus(iree_hal_command_buffer_end(self.raw_ptr()),
                             "command buffer end");
            }
          },
          py::arg("target_buffer"), py::arg("pattern"),
          py::arg("target_offset") = 0, py::arg("length") = py::none(),
          py::arg("end") = false);

  PyType_Slot debug_sink_slots[] = {
      {Py_tp_traverse, (void*)HalModuleDebugSinkTpTraverse},
      {Py_tp_clear, (void*)HalModuleDebugSinkTpClear},
      {0, nullptr}};
  py::class_<HalModuleDebugSink>(
      m, "HalModuleDebugSink", py::type_slots(debug_sink_slots),
      py::intrusive_ptr<HalModuleDebugSink>(
          [](HalModuleDebugSink* debug_sink, PyObject* po) noexcept {
            debug_sink->set_self_py(po);
          }))
      .def(
          "__init__",
          [](HalModuleDebugSink* self,
             HalModuleBufferViewTraceCallback buffer_view_trace_callback) {
            new (self) HalModuleDebugSink(buffer_view_trace_callback);
          },
          py::arg("buffer_view_trace_callback"))
      .def_prop_ro("buffer_view_trace_callback", [](HalModuleDebugSink& self) {
        return self.GetHalModuleBufferViewTraceCallback();
      });
}

}  // namespace python
}  // namespace iree
