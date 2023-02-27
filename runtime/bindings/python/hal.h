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

//------------------------------------------------------------------------------
// ApiRefCounted types
//------------------------------------------------------------------------------

class HalDevice : public ApiRefCounted<HalDevice, iree_hal_device_t> {
 public:
  iree_hal_allocator_t* allocator() {
    return iree_hal_device_allocator(raw_ptr());
  }

  void BeginProfiling(const py::kwargs& kwargs);
  void EndProfiling();
};

class HalDriver : public ApiRefCounted<HalDriver, iree_hal_driver_t> {
 public:
  static std::vector<std::string> Query();
  static py::object Create(const std::string& device_uri,
                           py::dict& driver_cache);

  py::list QueryAvailableDevices();
  HalDevice CreateDefaultDevice(const py::kwargs& kwargs);
  HalDevice CreateDevice(iree_hal_device_id_t device_id,
                         const py::kwargs& kwargs);
  HalDevice CreateDeviceByURI(std::string& device_uri,
                              const py::kwargs& kwargs);
};

class HalAllocator : public ApiRefCounted<HalAllocator, iree_hal_allocator_t> {
 public:
  py::dict QueryStatistics();
  py::str FormattedStatistics();

  py::object AllocateBufferCopy(
      int memory_type, int allowed_usage, py::object buffer,
      std::optional<iree_hal_element_types_t> element_type);
};

struct HalShape {
 public:
  static HalShape FromIntVector(std::vector<iree_hal_dim_t> indices) {
    HalShape s;
    s.s = {indices.begin(), indices.end()};
    return s;
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

  py::str Repr();
};

// Wrapper around an iree_hal_buffer_mapping_t and iree_hal_buffer_view_t
// which retains the latter and unmaps/releases on deallocation.
class HalMappedMemory {
 public:
  HalMappedMemory(iree_hal_buffer_mapping_t mapped_memory,
                  iree_hal_buffer_view_t* bv)
      : mapped_memory_(mapped_memory), bv_(bv) {
    iree_hal_buffer_view_retain(bv_);
  }
  ~HalMappedMemory() {
    if (bv_) {
      iree_hal_buffer_unmap_range(&mapped_memory_);
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
    iree_hal_buffer_mapping_t mapped_memory = {{0}};
    CheckApiStatus(
        iree_hal_buffer_map_range(buffer, IREE_HAL_MAPPING_MODE_SCOPED,
                                  IREE_HAL_MEMORY_ACCESS_READ, 0, byte_length,
                                  &mapped_memory),
        "Could not map memory");
    return HalMappedMemory(mapped_memory, bv.raw_ptr());
  }

  py::buffer_info ToBufferInfo() {
    std::vector<iree_hal_dim_t> shape(iree_hal_buffer_view_shape_rank(bv_));
    CheckApiStatus(
        iree_hal_buffer_view_shape(bv_, shape.size(), shape.data(), nullptr),
        "Error getting buffer view shape");
    iree_hal_element_type_t element_type =
        iree_hal_buffer_view_element_type(bv_);
    int32_t element_size = iree_hal_element_dense_byte_count(element_type);
    std::vector<py::ssize_t> dims(shape.size());
    for (int i = 0; i < shape.size(); ++i) {
      dims[i] = shape[i];
    }
    std::vector<py::ssize_t> strides(shape.size());
    if (!strides.empty()) {
      strides[shape.size() - 1] = element_size;
      for (int i = shape.size() - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
      }
    }

    return py::buffer_info(mapped_memory_.contents.data, element_size,
                           py::format_descriptor<float>::format(), shape.size(),
                           dims, strides);
  }

  iree_hal_buffer_mapping_t& mapped_memory() { return mapped_memory_; }

 private:
  iree_hal_buffer_mapping_t mapped_memory_ = {{0}};
  iree_hal_buffer_view_t* bv_ = nullptr;
};

void SetupHalBindings(pybind11::module m);

}  // namespace python
}  // namespace iree

#endif  // IREE_BINDINGS_PYTHON_IREE_RT_HAL_H_
