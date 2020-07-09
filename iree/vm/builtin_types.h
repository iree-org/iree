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

// The built-in constant buffer type.
// This simply points at a span of memory. The memory could be owned (in which
// case a destroy function must be provided) or unowned (NULL destroy function).
typedef struct {
  iree_vm_ref_object_t ref_object;
  iree_const_byte_span_t data;
  iree_vm_ref_destroy_t destroy;
} iree_vm_ro_byte_buffer_t;

// The built-in mutable buffer type.
// This simply points at a span of memory. The memory could be owned (in which
// case a destroy function must be provided) or unowned (NULL destroy function).
typedef struct {
  iree_vm_ref_object_t ref_object;
  iree_byte_span_t data;
  iree_vm_ref_destroy_t destroy;
} iree_vm_rw_byte_buffer_t;

// Registers the builtin VM types. This must be called on startup. Safe to call
// multiple times.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_vm_register_builtin_types();

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

IREE_VM_DECLARE_TYPE_ADAPTERS(iree_vm_ro_byte_buffer, iree_vm_ro_byte_buffer_t);
IREE_VM_DECLARE_TYPE_ADAPTERS(iree_vm_rw_byte_buffer, iree_vm_rw_byte_buffer_t);

#endif  // IREE_VM_BUILTIN_TYPES_H_
