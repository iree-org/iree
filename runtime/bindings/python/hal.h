// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BINDINGS_PYTHON_IREE_RT_HAL_H_
#define IREE_BINDINGS_PYTHON_IREE_RT_HAL_H_

#include <vector>

#include "./binding.h"
#include "./status_utils.h"
#include "./vm.h"
#include "iree/hal/api.h"

namespace iree {
namespace python {

//------------------------------------------------------------------------------
// Retain/release bindings
// Note that all HAL types have keep alive relationships in addition to this
// (using the py:keep_alive<>() facility). These relationships form a chain
// such that any live Python leaf (like a buffer or buffer_view) must keep
// alive the allocator, device and driver that created it.
//
// The hierarchy is:
//   HalDriver
//   HalDevice
//   HalAllocator
//   HalBuffer
//   HalBufferView
//
// Any Python API which produces one of the above must be annotated with
// py::keep_alive<0, 1>() in order to establish the relationship with the
// parent.
//
// Any Python API which consumes one of these objects such that its lifetime
// may extend outside of the current invocation must arrange to retain/release
// all backing devices that may need to survive.
//------------------------------------------------------------------------------

template <>
struct ApiPtrAdapter<iree_hal_driver_t> {
  static void Retain(iree_hal_driver_t* d) { iree_hal_driver_retain(d); }
  static void Release(iree_hal_driver_t* d) { iree_hal_driver_release(d); }
};

template <>
struct ApiPtrAdapter<iree_hal_device_t> {
  static void Retain(iree_hal_device_t* d) { iree_hal_device_retain(d); }
  static void Release(iree_hal_device_t* d) { iree_hal_device_release(d); }
};

template <>
struct ApiPtrAdapter<iree_hal_allocator_t> {
  static void Retain(iree_hal_allocator_t* d) { iree_hal_allocator_retain(d); }
  static void Release(iree_hal_allocator_t* d) {
    iree_hal_allocator_release(d);
  }
};

template <>
struct ApiPtrAdapter<iree_hal_buffer_t> {
  static void Retain(iree_hal_buffer_t* b) { iree_hal_buffer_retain(b); }
  static void Release(iree_hal_buffer_t* b) { iree_hal_buffer_release(b); }
};

template <>
struct ApiPtrAdapter<iree_hal_buffer_view_t> {
  static void Retain(iree_hal_buffer_view_t* bv) {
    iree_hal_buffer_view_retain(bv);
  }
  static void Release(iree_hal_buffer_view_t* bv) {
    iree_hal_buffer_view_release(bv);
  }
};

template <>
struct ApiPtrAdapter<iree_hal_semaphore_t> {
  static void Retain(iree_hal_semaphore_t* sem) {
    iree_hal_semaphore_retain(sem);
  }
  static void Release(iree_hal_semaphore_t* sem) {
    iree_hal_semaphore_release(sem);
  }
};

template <>
struct ApiPtrAdapter<iree_hal_fence_t> {
  static void Retain(iree_hal_fence_t* fence) { iree_hal_fence_retain(fence); }
  static void Release(iree_hal_fence_t* fence) {
    iree_hal_fence_release(fence);
  }
};

template <>
struct ApiPtrAdapter<iree_hal_command_buffer_t> {
  static void Retain(iree_hal_command_buffer_t* cb) {
    iree_hal_command_buffer_retain(cb);
  }
  static void Release(iree_hal_command_buffer_t* cb) {
    iree_hal_command_buffer_release(cb);
  }
};

//------------------------------------------------------------------------------
// ApiRefCounted types
//------------------------------------------------------------------------------

class HalBuffer;
class HalSemaphore;

class HalDevice : public ApiRefCounted<HalDevice, iree_hal_device_t> {
 public:
  iree_hal_allocator_t* allocator() {
    return iree_hal_device_allocator(raw_ptr());
  }

  void BeginProfiling(std::optional<std::string> mode,
                      std::optional<std::string> file_path);
  void FlushProfiling();
  void EndProfiling();
  HalSemaphore CreateSemaphore(uint64_t initial_value);
  HalBuffer QueueAlloca(uint64_t allocation_size, py::handle wait_semaphores,
                        py::handle signal_semaphores);
  void QueueDealloca(HalBuffer& buffer, py::handle wait_semaphores,
                     py::handle signal_semaphores);
  void QueueExecute(py::handle command_buffers, py::handle wait_semaphores,
                    py::handle signal_semaphores);
  void QueueCopy(HalBuffer& src_buffer, HalBuffer& dst_buffer,
                 py::handle wait_semaphores, py::handle signal_semaphores);
};

class HalDriver : public ApiRefCounted<HalDriver, iree_hal_driver_t> {
 public:
  static std::vector<std::string> Query();
  static py::object Create(const std::string& device_uri,
                           py::dict& driver_cache);

  py::list QueryAvailableDevices();
  HalDevice CreateDefaultDevice(std::optional<py::list> allocators);
  HalDevice CreateDevice(iree_hal_device_id_t device_id,
                         std::optional<py::list> allocators);
  HalDevice CreateDeviceByURI(std::string& device_uri,
                              std::optional<py::list> allocators);
};

class HalAllocator : public ApiRefCounted<HalAllocator, iree_hal_allocator_t> {
 public:
  py::dict QueryStatistics();
  py::str FormattedStatistics();

  py::object AllocateBufferCopy(
      int memory_type, int allowed_usage, HalDevice& device, py::object buffer,
      std::optional<iree_hal_element_types_t> element_type);
  HalBuffer AllocateHostStagingBufferCopy(HalDevice& device, py::handle buffer);
};

struct HalShape {
 public:
  HalShape(std::vector<iree_hal_dim_t>& indices) {
    s = {indices.begin(), indices.end()};
  }

  std::vector<iree_hal_dim_t> s;
};

class HalBufferView
    : public ApiRefCounted<HalBufferView, iree_hal_buffer_view_t> {
 public:
  py::str Repr();
};

class HalBuffer : public ApiRefCounted<HalBuffer, iree_hal_buffer_t> {
 public:
  iree_device_size_t byte_length() const {
    return iree_hal_buffer_byte_length(raw_ptr());
  }

  int memory_type() const { return iree_hal_buffer_memory_type(raw_ptr()); }

  int allowed_usage() const { return iree_hal_buffer_allowed_usage(raw_ptr()); }

  void FillZero(iree_device_size_t byte_offset,
                iree_device_size_t byte_length) {
    CheckApiStatus(
        iree_hal_buffer_map_zero(raw_ptr(), byte_offset, byte_length),
        "Error zero filling buffer");
  }

  // TODO(laurenzo): make this take element_type instead.
  HalBufferView CreateView(HalShape& shape, size_t element_size) {
    iree_hal_buffer_view_t* bv;
    iree_hal_element_type_t element_type = iree_hal_make_element_type(
        IREE_HAL_ELEMENT_TYPE_NONE, element_size * 8);
    iree_hal_encoding_type_t encoding_type =
        IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR;
    CheckApiStatus(iree_hal_buffer_view_create(
                       raw_ptr(), shape.s.size(), shape.s.data(), element_type,
                       encoding_type, iree_allocator_system(), &bv),
                   "Error creating buffer view");
    return HalBufferView::StealFromRawPtr(bv);
  }

  static HalBuffer CreateFromBufferView(HalBufferView& bv) {
    return HalBuffer::BorrowFromRawPtr(
        iree_hal_buffer_view_buffer(bv.raw_ptr()));
  }

  py::str Repr();
};

class HalSemaphore : public ApiRefCounted<HalSemaphore, iree_hal_semaphore_t> {
 public:
};

class HalFence : public ApiRefCounted<HalFence, iree_hal_fence_t> {
 public:
};

// Wrapper around an iree_hal_buffer_mapping_t and iree_hal_buffer_t
// which retains the latter and unmaps/releases on deallocation.
class HalMappedMemory {
 public:
  HalMappedMemory(iree_hal_buffer_mapping_t mapped_memory,
                  iree_hal_buffer_t* buffer)
      : mapped_memory_(mapped_memory), buffer_(buffer) {
    iree_hal_buffer_retain(buffer_);
  }
  ~HalMappedMemory() {
    if (buffer_) {
      iree_hal_buffer_unmap_range(&mapped_memory_);
      iree_hal_buffer_release(buffer_);
    }
  }
  HalMappedMemory(HalMappedMemory&& other)
      : mapped_memory_(other.mapped_memory_), buffer_(other.buffer_) {
    other.buffer_ = nullptr;
  }

  static HalMappedMemory Create(iree_hal_buffer_t* buffer) {
    iree_device_size_t byte_length = iree_hal_buffer_byte_length(buffer);
    iree_hal_buffer_mapping_t mapped_memory = {{0}};
    CheckApiStatus(
        iree_hal_buffer_map_range(buffer, IREE_HAL_MAPPING_MODE_SCOPED,
                                  IREE_HAL_MEMORY_ACCESS_READ, 0, byte_length,
                                  &mapped_memory),
        "Could not map memory");
    return HalMappedMemory(mapped_memory, buffer);
  }
  static HalMappedMemory CreateFromBuffer(HalBuffer& b) {
    return Create(b.raw_ptr());
  }
  static HalMappedMemory CreateFromBufferView(HalBufferView& bv) {
    return Create(iree_hal_buffer_view_buffer(bv.raw_ptr()));
  }

  iree_hal_buffer_mapping_t& mapped_memory() { return mapped_memory_; }

 private:
  iree_hal_buffer_mapping_t mapped_memory_ = {{0}};
  iree_hal_buffer_t* buffer_ = nullptr;
};

class HalCommandBuffer
    : public ApiRefCounted<HalCommandBuffer, iree_hal_command_buffer_t> {};

void SetupHalBindings(nanobind::module_ m);

}  // namespace python
}  // namespace iree

#endif  // IREE_BINDINGS_PYTHON_IREE_RT_HAL_H_
