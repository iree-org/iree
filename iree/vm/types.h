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

#ifndef IREE_VM_TYPES_H_
#define IREE_VM_TYPES_H_

#include <stdbool.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/vm/ref.h"

#ifdef __cplusplus
namespace iree {
namespace vm {
template <typename T>
struct ref_type_descriptor {
  static const iree_vm_ref_type_descriptor_t* get();
};
}  // namespace vm
}  // namespace iree
#define IREE_VM_DECLARE_CC_TYPE_LOOKUP(name, T)         \
  namespace iree {                                      \
  namespace vm {                                        \
  template <>                                           \
  struct ref_type_descriptor<T> {                       \
    static const iree_vm_ref_type_descriptor_t* get() { \
      return name##_get_descriptor();                   \
    }                                                   \
  };                                                    \
  }                                                     \
  }

#define IREE_VM_REGISTER_CC_TYPE(type, name, descriptor)  \
  descriptor.type_name = iree_make_cstring_view(name);    \
  descriptor.offsetof_counter = type::offsetof_counter(); \
  descriptor.destroy = type::DirectDestroy;               \
  IREE_RETURN_IF_ERROR(iree_vm_ref_register_type(&descriptor));
#else
#define IREE_VM_DECLARE_CC_TYPE_LOOKUP(name, T)
#define IREE_VM_REGISTER_CC_TYPE(type, name, descriptor)
#endif  // __cplusplus

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// TODO(benvanik): make these macros standard/document them.
#define IREE_VM_DECLARE_TYPE_ADAPTERS(name, T)                             \
  IREE_API_EXPORT iree_vm_ref_t IREE_API_CALL name##_retain_ref(T* value); \
  IREE_API_EXPORT iree_vm_ref_t IREE_API_CALL name##_move_ref(T* value);   \
  IREE_API_EXPORT T* IREE_API_CALL name##_deref(iree_vm_ref_t* ref);       \
  IREE_API_EXPORT const iree_vm_ref_type_descriptor_t* IREE_API_CALL       \
      name##_get_descriptor();                                             \
  inline bool name##_isa(iree_vm_ref_t* ref) {                             \
    return name##_get_descriptor()->type == ref->type;                     \
  }                                                                        \
  IREE_API_EXPORT iree_vm_ref_type_t IREE_API_CALL name##_type_id();       \
  IREE_VM_DECLARE_CC_TYPE_LOOKUP(name, T)

// TODO(benvanik): make these macros standard/document them.
#define IREE_VM_DEFINE_TYPE_ADAPTERS(name, T)                                 \
  IREE_API_EXPORT iree_vm_ref_t IREE_API_CALL name##_retain_ref(T* value) {   \
    iree_vm_ref_t ref = {0};                                                  \
    iree_vm_ref_wrap_retain(value, name##_descriptor.type, &ref);             \
    return ref;                                                               \
  }                                                                           \
  IREE_API_EXPORT iree_vm_ref_t IREE_API_CALL name##_move_ref(T* value) {     \
    iree_vm_ref_t ref = {0};                                                  \
    iree_vm_ref_wrap_assign(value, name##_descriptor.type, &ref);             \
    return ref;                                                               \
  }                                                                           \
  IREE_API_EXPORT T* IREE_API_CALL name##_deref(iree_vm_ref_t* ref) {         \
    if (!iree_status_is_ok(iree_vm_ref_check(ref, name##_descriptor.type))) { \
      return NULL;                                                            \
    }                                                                         \
    return (T*)ref->ptr;                                                      \
  }                                                                           \
  IREE_API_EXPORT const iree_vm_ref_type_descriptor_t* IREE_API_CALL          \
      name##_get_descriptor() {                                               \
    return &name##_descriptor;                                                \
  }                                                                           \
  IREE_API_EXPORT iree_vm_ref_type_t IREE_API_CALL name##_type_id() {         \
    return name##_descriptor.type;                                            \
  }

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

IREE_VM_DECLARE_TYPE_ADAPTERS(iree_vm_ro_byte_buffer, iree_vm_ro_byte_buffer_t);
IREE_VM_DECLARE_TYPE_ADAPTERS(iree_vm_rw_byte_buffer, iree_vm_rw_byte_buffer_t);

// Registers the builtin VM types. This must be called on startup. Safe to call
// multiple times.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_vm_register_builtin_types();

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_VM_TYPES_H_
