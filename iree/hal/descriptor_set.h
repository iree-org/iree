// Copyright 2020 Google LLC
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

#ifndef IREE_HAL_DESCRIPTOR_SET_H_
#define IREE_HAL_DESCRIPTOR_SET_H_

#include <stdbool.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/buffer.h"
#include "iree/hal/descriptor_set_layout.h"
#include "iree/hal/resource.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_device_s iree_hal_device_t;

//===----------------------------------------------------------------------===//
// Types and Enums
//===----------------------------------------------------------------------===//

// Specifies a descriptor set binding.
// The range specified by [offset, length) will be made available to executables
// on the given binding. If the descriptor type is dynamic then the range will
// be [offset + dynamic_offset, length).
//
// The IREE HAL buffer type may internally be offset; such offset is applied
// here as if it were the base address of the buffer. Note that the offset will
// be applied at the time the binding is recording into the command buffer.
//
// Maps to VkDescriptorSetBinding.
typedef struct {
  // The binding number of this entry and corresponds to a resource of the
  // same binding number in the executable interface.
  uint32_t binding;
  // Buffer bound to the binding number.
  // May be NULL if the binding is not used by the executable.
  iree_hal_buffer_t* buffer;
  // Offset, in bytes, into the buffer that the binding starts at.
  // If the descriptor type is dynamic this will be added to the dynamic
  // offset provided during binding.
  iree_device_size_t offset;
  // Length, in bytes, of the buffer that is available to the executable.
  // This can be IREE_WHOLE_BUFFER, however note that if the entire buffer
  // contents are larger than supported by the device (~128MiB, usually) this
  // will fail. If the descriptor type is dynamic this will be used for all
  // ranges regardless of offset.
  iree_device_size_t length;
} iree_hal_descriptor_set_binding_t;

//===----------------------------------------------------------------------===//
// iree_hal_descriptor_set_t
//===----------------------------------------------------------------------===//

// Opaque handle to a descriptor set object.
// A "descriptor" is effectively a bound memory range and each dispatch can use
// one or more "descriptor sets" to access their I/O memory. Each descriptor set
// conforms to a template "descriptor set layout".
//
// Maps to VkDescriptorSet:
// https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VkDescriptorSet.html
typedef struct iree_hal_descriptor_set_s iree_hal_descriptor_set_t;

// Creates a descriptor set of the given layout and bindings.
// Descriptor sets are immutable and retain their bindings.
IREE_API_EXPORT iree_status_t iree_hal_descriptor_set_create(
    iree_hal_device_t* device, iree_hal_descriptor_set_layout_t* set_layout,
    iree_host_size_t binding_count,
    const iree_hal_descriptor_set_binding_t* bindings,
    iree_hal_descriptor_set_t** out_descriptor_set);

// Retains the given |set| for the caller.
IREE_API_EXPORT void iree_hal_descriptor_set_retain(
    iree_hal_descriptor_set_t* descriptor_set);

// Releases the given |set| from the caller.
IREE_API_EXPORT void iree_hal_descriptor_set_release(
    iree_hal_descriptor_set_t* descriptor_set);

//===----------------------------------------------------------------------===//
// iree_hal_descriptor_set_t implementation details
//===----------------------------------------------------------------------===//

typedef struct {
  // << HAL C porting in progress >>
  IREE_API_UNSTABLE

  void(IREE_API_PTR* destroy)(iree_hal_descriptor_set_t* descriptor_set);
} iree_hal_descriptor_set_vtable_t;

IREE_API_EXPORT void iree_hal_descriptor_set_destroy(
    iree_hal_descriptor_set_t* descriptor_set);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DESCRIPTOR_SET_H_
