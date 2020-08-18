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

#ifndef IREE_VM_REF_H_
#define IREE_VM_REF_H_

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/base/atomics.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Defines the type of the reference-counted pointer.
// This is used to verify that operations dealing with the variant ref struct
// are correct at runtime. We don't allow control over the ref types from the
// VM ops and as such we can use the type specified as a safe way to avoid
// reinterpreting memory incorrectly.
typedef enum {
  IREE_VM_REF_TYPE_NULL = 0,

  // NOTE: these type values are assigned dynamically right now. Treat them as
  // opaque and unstable across process invocations.

  // Maximum type ID value. Type IDs are limited to 24-bits.
  IREE_VM_REF_TYPE_MAX_VALUE = 0x00FFFFFEu,

  // Wildcard type that indicates that a value may be a ref type but of an
  // unspecified internal type.
  IREE_VM_REF_TYPE_ANY = 0x00FFFFFFu,
} iree_vm_ref_type_t;

// Base for iree_vm_ref_t object targets.
//
// Usage (C):
//  typedef struct {
//    iree_vm_ref_object_t ref_object;
//    int my_fields;
//  } my_type_t;
//  void my_type_destroy(void* ptr) {
//    free(ptr);
//  }
//  static iree_vm_ref_type_descriptor_t my_type_descriptor;
//  my_type_descriptor.type_name = iree_string_view_t{"my_type", 7};
//  my_type_descriptor.destroy = my_type_destroy;
//  my_type_descriptor.offsetof_counter = offsetof(my_type_t,
//                                                 ref_object.counter);
//  iree_vm_ref_register_defined_type(&my_type_descriptor);
//
// Usage (C++):
//  Prefer using RefObject as a base type.
typedef struct {
  iree_atomic_intptr_t counter;
} iree_vm_ref_object_t;

// A pointer reference to a reference-counted object.
// The counter is stored within the target object itself ala intrusive_ptr.
//
// NOTE: we try to keep this small so that we aren't wasting stack space or
// copying around too much when we pass it to functions by value. This also
// helps make the CPU caches happier as we need no indirections to check the
// type and adjusting the counter happens without needing to query a descriptor.
// Ideally the iree_vm_ref_t is in-cache on the stack and the target ptr is
// either in cache from a previous use or will be used again after manipulating
// its ref count.
typedef struct {
  // Pointer to the object. Type is resolved based on the |type| field.
  // Will be NULL if the reference points to nothing.
  void* ptr;
  // Offset from ptr, in bytes, to the start of an atomic_intptr_t representing
  // the current reference count. We store this here to avoid the need for an
  // indirection in the (extremely common) case of just reference count inc/dec.
  uint32_t offsetof_counter : 8;
  // Registered type of the object pointed to by ptr.
  iree_vm_ref_type_t type : 24;
} iree_vm_ref_t;
static_assert(
    sizeof(iree_vm_ref_t) <= sizeof(void*) * 2,
    "iree_vm_ref_t dominates stack space usage and should be kept tiny");

typedef void(IREE_API_PTR* iree_vm_ref_destroy_t)(void* ptr);

#define IREE_VM_REF_DESTROY_FREE free
#define IREE_VM_REF_DESTROY_CC_DELETE +[](void* ptr) { delete ptr; }

// Describes a type for the VM.
typedef struct {
  // Function called when references of this type reach 0 and should be
  // destroyed.
  iree_vm_ref_destroy_t destroy;
  // Offset from ptr, in bytes, to the start of an atomic_intptr_t representing
  // the current reference count.
  uint32_t offsetof_counter : 8;
  // The type ID assigned to this type from the iree_vm_ref_type_t table (or an
  // external user source).
  iree_vm_ref_type_t type : 24;
  // Unretained type name that can be used for debugging.
  iree_string_view_t type_name;
} iree_vm_ref_type_descriptor_t;

// Directly retains the object with base |ptr| with the given |type_descriptor|.
//
// Note that this avoids any kind of type checking; for untrusted inputs use
// the iree_vm_ref_t-based methods.
IREE_API_EXPORT void IREE_API_CALL iree_vm_ref_object_retain(
    void* ptr, const iree_vm_ref_type_descriptor_t* type_descriptor);

// Directly release the object with base |ptr| with the given |type_descriptor|,
// possibly destroying it if it is the last reference. Assume that |ptr| is
// invalid after this function returns.
//
// Note that this avoids any kind of type checking; for untrusted inputs use
// the iree_vm_ref_t-based methods.
IREE_API_EXPORT void IREE_API_CALL iree_vm_ref_object_release(
    void* ptr, const iree_vm_ref_type_descriptor_t* type_descriptor);

// Registers a user-defined type with the IREE C ref system.
// The provided destroy function will be used to destroy objects when their
// reference count goes to 0. NULL can be used to no-op the destruction if the
// type is not owned by the VM.
//
// TODO(benvanik): keep names alive for user types?
// NOTE: the name is not retained and must be kept live by the caller. Ideally
// it is stored in static read-only memory in the binary.
//
// WARNING: this function is not thread-safe and should only be used at startup
// to register the types. Do not call this while any refs may be alive.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_vm_ref_register_type(iree_vm_ref_type_descriptor_t* descriptor);

// Returns the type name for the given type, if found.
IREE_API_EXPORT iree_string_view_t IREE_API_CALL
iree_vm_ref_type_name(iree_vm_ref_type_t type);

// Returns the registered type descriptor for the given type, if found.
IREE_API_EXPORT const iree_vm_ref_type_descriptor_t* IREE_API_CALL
iree_vm_ref_lookup_registered_type(iree_string_view_t full_name);

// Wraps a raw pointer in a iree_vm_ref_t reference and assigns it to |out_ref|.
// |out_ref| will be released if it already contains a reference. The target
// object will not be retained and must come in with a count >= 1.
//
// Usage (C):
//  my_type_t* my_type = (my_type_t*)malloc(sizeof(my_type_t));
//  my_type.ref_object.counter = IREE_ATOMIC_VAR_INIT(1);
//  iree_vm_ref_t my_ref;
//  iree_vm_ref_wrap_assign(my_type, IREE_VM_REF_TYPE_MY_TYPE, &my_ref);
//  iree_vm_ref_release(&my_ref);
//
// Usage (C++):
//  iree_vm_ref_t my_ref;
//  iree_vm_ref_wrap_assign(new MyType(), IREE_VM_REF_TYPE_MY_TYPE, &my_ref);
//  iree_vm_ref_release(&my_ref);
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_vm_ref_wrap_assign(
    void* ptr, iree_vm_ref_type_t type, iree_vm_ref_t* out_ref);

// Wraps a raw pointer in a iree_vm_ref_t reference and retains it in |out_ref|.
// |out_ref| will be released if it already contains a reference.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_vm_ref_wrap_retain(
    void* ptr, iree_vm_ref_type_t type, iree_vm_ref_t* out_ref);

// Checks that the given reference-counted pointer |ref| is of |type|.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_vm_ref_check(iree_vm_ref_t* ref, iree_vm_ref_type_t type);

#define IREE_VM_DEREF_OR_RETURN(value_type, value, ref, type) \
  IREE_RETURN_IF_ERROR(iree_vm_ref_check(ref, type));         \
  value_type* value = (value_type*)(ref)->ptr;

// Retains the reference-counted pointer |ref|.
// |out_ref| will be released if it already contains a reference.
IREE_API_EXPORT void IREE_API_CALL iree_vm_ref_retain(iree_vm_ref_t* ref,
                                                      iree_vm_ref_t* out_ref);

// Retains the reference-counted pointer |ref| and checks that it is of |type|.
// |out_ref| will be released if it already contains a reference.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_vm_ref_retain_checked(
    iree_vm_ref_t* ref, iree_vm_ref_type_t type, iree_vm_ref_t* out_ref);

// Retains or moves |ref| to |out_ref|.
// |out_ref| will be released if it already contains a reference.
IREE_API_EXPORT void IREE_API_CALL iree_vm_ref_retain_or_move(
    int is_move, iree_vm_ref_t* ref, iree_vm_ref_t* out_ref);

// Retains or moves |ref| to |out_ref| and checks that |ref| is of |type|.
// |out_ref| will be released if it already contains a reference.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_vm_ref_retain_or_move_checked(
    int is_move, iree_vm_ref_t* ref, iree_vm_ref_type_t type,
    iree_vm_ref_t* out_ref);

// Releases the reference-counted pointer |ref|, possibly freeing it.
IREE_API_EXPORT void IREE_API_CALL iree_vm_ref_release(iree_vm_ref_t* ref);

// Assigns the reference-counted pointer |ref| without incrementing the count.
// |out_ref| will be released if it already contains a reference.
IREE_API_EXPORT void IREE_API_CALL iree_vm_ref_assign(iree_vm_ref_t* ref,
                                                      iree_vm_ref_t* out_ref);

// Moves one reference to another without changing the reference count.
// Equivalent to an std::move of a ref_ptr.
// |out_ref| will be released if it already contains a reference.
IREE_API_EXPORT void IREE_API_CALL iree_vm_ref_move(iree_vm_ref_t* ref,
                                                    iree_vm_ref_t* out_ref);

// Returns true if the given |ref| is NULL.
IREE_API_EXPORT bool IREE_API_CALL iree_vm_ref_is_null(iree_vm_ref_t* ref);

// Returns true if the two references point at the same value (or are both
// null).
IREE_API_EXPORT bool IREE_API_CALL iree_vm_ref_equal(iree_vm_ref_t* lhs,
                                                     iree_vm_ref_t* rhs);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

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
#define IREE_VM_DEFINE_TYPE_ADAPTERS(name, T)                               \
  IREE_API_EXPORT iree_vm_ref_t IREE_API_CALL name##_retain_ref(T* value) { \
    iree_vm_ref_t ref = {0};                                                \
    iree_vm_ref_wrap_retain(value, name##_descriptor.type, &ref);           \
    return ref;                                                             \
  }                                                                         \
  IREE_API_EXPORT iree_vm_ref_t IREE_API_CALL name##_move_ref(T* value) {   \
    iree_vm_ref_t ref = {0};                                                \
    iree_vm_ref_wrap_assign(value, name##_descriptor.type, &ref);           \
    return ref;                                                             \
  }                                                                         \
  IREE_API_EXPORT T* IREE_API_CALL name##_deref(iree_vm_ref_t* ref) {       \
    iree_status_t status = iree_vm_ref_check(ref, name##_descriptor.type);  \
    if (!iree_status_is_ok(status)) {                                       \
      iree_status_ignore(status);                                           \
      return NULL;                                                          \
    }                                                                       \
    return (T*)ref->ptr;                                                    \
  }                                                                         \
  IREE_API_EXPORT const iree_vm_ref_type_descriptor_t* IREE_API_CALL        \
      name##_get_descriptor() {                                             \
    return &name##_descriptor;                                              \
  }                                                                         \
  IREE_API_EXPORT iree_vm_ref_type_t IREE_API_CALL name##_type_id() {       \
    return name##_descriptor.type;                                          \
  }

#endif  // IREE_VM_REF_H_
