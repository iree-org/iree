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

#ifndef IREE_VM_BUILTIN_TYPES_H_
#define IREE_VM_BUILTIN_TYPES_H_

#include "iree/base/api.h"
#include "iree/vm/ref.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Describes where a byte buffer originates from, what guarantees can be made
// about its lifetime and ownership, and how it may be accessed.
// Note that buffers may always be read.
enum iree_vm_buffer_access_e {
  // The guest is allowed to write to the buffer.
  IREE_VM_BUFFER_ACCESS_WRITE = 1u << 0,

  // Buffer references memory in the module space (rodata or rwdata) that is
  // guaranteed to be live for the lifetime of the module.
  IREE_VM_BUFFER_ACCESS_ORIGIN_MODULE = 1u << 1,
  // Buffer references memory created by the guest module code. It has a
  // lifetime less than that of the module but is always tracked with proper
  // references (a handle existing to the memory implies it is valid).
  IREE_VM_BUFFER_ACCESS_ORIGIN_GUEST = 1u << 2,
};
typedef uint32_t iree_vm_buffer_access_t;

// The built-in buffer type.
// This simply points at a span of memory. The memory could be owned (in which
// case a destroy function must be provided) or unowned (NULL destroy function).
// The access flags indicate what access is allowed from the VM.
typedef struct {
  iree_vm_ref_object_t ref_object;
  iree_vm_buffer_access_t access;
  iree_byte_span_t data;
  iree_vm_ref_destroy_t destroy;
} iree_vm_buffer_t;

// Returns the a string view referencing the given |value| buffer.
static inline iree_string_view_t iree_vm_buffer_as_string(
    const iree_vm_buffer_t* value) {
  return value ? iree_make_string_view((const char*)value->data.data,
                                       value->data.data_length)
               : iree_string_view_empty();
}

// Registers the builtin VM types. This must be called on startup. Safe to call
// multiple times.
IREE_API_EXPORT iree_status_t iree_vm_register_builtin_types();

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

IREE_VM_DECLARE_TYPE_ADAPTERS(iree_vm_buffer, iree_vm_buffer_t);

#endif  // IREE_VM_BUILTIN_TYPES_H_
