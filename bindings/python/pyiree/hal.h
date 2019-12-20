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

#ifndef IREE_BINDINGS_PYTHON_PYIREE_HAL_H_
#define IREE_BINDINGS_PYTHON_PYIREE_HAL_H_

#include "bindings/python/pyiree/binding.h"
#include "bindings/python/pyiree/status_utils.h"
#include "iree/hal/api.h"

namespace iree {
namespace python {

//------------------------------------------------------------------------------
// Retain/release bindings
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
};

class HalDriver : public ApiRefCounted<HalDriver, iree_hal_driver_t> {
 public:
  static std::vector<std::string> Query();
  static HalDriver Create(const std::string& driver_name);

  HalDevice CreateDefaultDevice();
};

struct HalShape {
 public:
  static HalShape FromIntVector(std::vector<int32_t> indices) {
    if (indices.size() > IREE_SHAPE_MAX_RANK) {
      throw RaiseValueError("Shape exceeded maximum rank");
    }
    HalShape s;
    s.s.rank = indices.size();
    for (size_t i = 0, e = indices.size(); i < e; ++i) {
      s.s.dims[i] = indices[i];
    }
    return s;
  }

  iree_shape_t s;
};

class HalBufferView
    : public ApiRefCounted<HalBufferView, iree_hal_buffer_view_t> {
 public:
};

class HalBuffer : public ApiRefCounted<HalBuffer, iree_hal_buffer_t> {
 public:
  static HalBuffer AllocateHeapBuffer(int32_t memory_type, int32_t usage,
                                      iree_host_size_t allocation_size) {
    iree_hal_buffer_t* buffer = nullptr;
    CheckApiStatus(
        iree_hal_heap_buffer_allocate(
            static_cast<iree_hal_memory_type_t>(memory_type),
            static_cast<iree_hal_buffer_usage_t>(usage), allocation_size,
            IREE_ALLOCATOR_SYSTEM, IREE_ALLOCATOR_SYSTEM, &buffer),
        "Error allocating heap buffer");
    return HalBuffer::CreateRetained(buffer);
  }

  iree_device_size_t byte_length() const {
    return iree_hal_buffer_byte_length(raw_ptr());
  }

  void FillZero(iree_device_size_t byte_offset,
                iree_device_size_t byte_length) {
    CheckApiStatus(iree_hal_buffer_zero(raw_ptr(), byte_offset, byte_length),
                   "Error zero filling buffer");
  }

  HalBufferView CreateView(HalShape& shape, size_t element_size) {
    iree_hal_buffer_view_t* bv;
    CheckApiStatus(iree_hal_buffer_view_create(raw_ptr(), shape.s, element_size,
                                               IREE_ALLOCATOR_SYSTEM, &bv),
                   "Error creating buffer view");
    return HalBufferView::CreateRetained(bv);
  }
};

void SetupHalBindings(pybind11::module m);

}  // namespace python
}  // namespace iree

#endif  // IREE_BINDINGS_PYTHON_PYIREE_HAL_H_
