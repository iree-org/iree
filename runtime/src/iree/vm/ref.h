// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_REF_H_
#define IREE_VM_REF_H_

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_vm_instance_t iree_vm_instance_t;

// Enable additional expensive checks to track down ref counting issues.
// #define IREE_VM_REF_PARANOID

// Number of least significant bits available in an iree_vm_ref_type_t value.
// These can be used for any purpose.
#define IREE_VM_REF_TYPE_TAG_BITS 3
#define IREE_VM_REF_TYPE_TAG_BIT_MASK 0b111

// Number of bits available in an iree_vm_ref_type_t value to hold the pointer.
#define IREE_VM_REF_TYPE_PTR_BITS \
  (sizeof(uintptr_t) * 8 - IREE_VM_REF_TYPE_TAG_BITS)
#define IREE_VM_REF_TYPE_PTR_BIT_MASK (~IREE_VM_REF_TYPE_TAG_BIT_MASK)

// (1 << IREE_VM_REF_TYPE_TAG_BITS) hardcoded for MSVC which cannot have
// expressions in its alignas in older versions.
#define IREE_VM_REF_TYPE_DESCRIPTOR_ALIGNMENT 8

// Alignment required for the reference counter.
// Used to pack the offset bits to fit in IREE_VM_REF_TYPE_TAG_BITS.
#define IREE_VM_REF_COUNTER_ALIGNMENT sizeof(iree_atomic_ref_count_t)

// Defines the type of the reference-counted pointer.
// This is used to verify that operations dealing with the variant ref struct
// are correct at runtime. We don't allow control over the ref types from the
// VM ops and as such we can use the type specified as a safe way to avoid
// reinterpreting memory incorrectly.
enum iree_vm_ref_type_bits_t {
  IREE_VM_REF_TYPE_NULL = 0,

  // NOTE: these type values are assigned dynamically right now. Treat them as
  // opaque and unstable across process invocations.

  // Wildcard type that indicates that a value may be a ref type but of an
  // unspecified internal type. Note that we mask off the bottom bits to allow
  // for tagging (all ref type values )
  IREE_VM_REF_TYPE_ANY = UINTPTR_MAX ^ IREE_VM_REF_TYPE_TAG_BIT_MASK,
};

typedef void(IREE_API_PTR* iree_vm_ref_destroy_t)(void* ptr);

// Describes a type for the VM.
typedef iree_alignas(IREE_VM_REF_TYPE_DESCRIPTOR_ALIGNMENT) struct
    iree_vm_ref_type_descriptor_t {
  // Function called when references of this type reach 0 and should be
  // destroyed.
  iree_vm_ref_destroy_t destroy;
  // Unretained type name that can be used for debugging.
  iree_string_view_t type_name;
  // Offset from ptr in units of IREE_VM_REF_COUNTER_ALIGNMENT to the start of
  // an iree_atomic_ref_count_t representing the current reference count.
  uintptr_t offsetof_counter : IREE_VM_REF_TYPE_TAG_BITS;
  uintptr_t reserved : IREE_VM_REF_TYPE_PTR_BITS;
} iree_vm_ref_type_descriptor_t;

// Type-erased reference counted type descriptor.
typedef uintptr_t iree_vm_ref_type_t;

// Makes an opaque reference to a type descriptor.
static inline iree_vm_ref_type_t iree_vm_make_ref_type(
    const iree_vm_ref_type_descriptor_t* descriptor) {
  return (iree_vm_ref_type_t)descriptor |
         (iree_vm_ref_type_t)descriptor->offsetof_counter;
}

// Returns the type name for the given type, if found.
IREE_API_EXPORT iree_string_view_t
iree_vm_ref_type_name(iree_vm_ref_type_t type);

// Returns the type descriptor backing the given |type|.
// The descriptor is an implementation detail.
static inline const iree_vm_ref_type_descriptor_t* iree_vm_ref_type_descriptor(
    iree_vm_ref_type_t type) {
  return (const iree_vm_ref_type_descriptor_t*)(type &
                                                IREE_VM_REF_TYPE_PTR_BIT_MASK);
}

// Base for iree_vm_ref_t object targets.
//
// Usage (C):
//  typedef struct my_type_t {
//    iree_vm_ref_object_t ref_object;
//    int my_fields;
//  } my_type_t;
//  void my_type_destroy(void* ptr) {
//    free(ptr);
//  }
//  static iree_vm_ref_type_descriptor_t my_type_descriptor;
//  my_type_descriptor.type_name = iree_string_view_t{"my_type", 7};
//  my_type_descriptor.destroy = my_type_destroy;
//  my_type_descriptor.offsetof_counter =
//      offsetof(my_type_t, ref_object.counter) / IREE_VM_REF_COUNTER_ALIGNMENT;
//  iree_vm_ref_register_defined_type(&my_type_descriptor);
//
// Usage (C++):
//  Prefer using iree::vm::RefObject as a base type.
typedef struct iree_vm_ref_object_t {
  iree_atomic_ref_count_t counter;
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
typedef struct iree_vm_ref_t {
  // Pointer to the object. Type is resolved based on the |type| field.
  // Will be NULL if the reference points to nothing.
  void* ptr;
  // Registered type of the object pointed to by ptr.
  iree_vm_ref_type_t type;
} iree_vm_ref_t;
static_assert(
    sizeof(iree_vm_ref_t) <= sizeof(void*) * 2,
    "iree_vm_ref_t dominates stack space usage and should be kept tiny");

// Directly retains the object with base |ptr| with the given |type|.
//
// Note that this avoids any kind of type checking; for untrusted inputs use
// the iree_vm_ref_t-based methods.
IREE_API_EXPORT void iree_vm_ref_object_retain(void* ptr,
                                               iree_vm_ref_type_t type);

// Directly release the object with base |ptr| with the given |type|,
// possibly destroying it if it is the last reference. Assume that |ptr| is
// invalid after this function returns.
//
// Note that this avoids any kind of type checking; for untrusted inputs use
// the iree_vm_ref_t-based methods.
IREE_API_EXPORT void iree_vm_ref_object_release(void* ptr,
                                                iree_vm_ref_type_t type);

// Returns a NULL ref wrapper.
static inline iree_vm_ref_t iree_vm_ref_null(void) {
  iree_vm_ref_t ref = {0};
  return ref;
}

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
IREE_API_EXPORT iree_status_t iree_vm_ref_wrap_assign(void* ptr,
                                                      iree_vm_ref_type_t type,
                                                      iree_vm_ref_t* out_ref);

// Wraps a raw pointer in a iree_vm_ref_t reference and retains it in |out_ref|.
// |out_ref| will be released if it already contains a reference.
IREE_API_EXPORT iree_status_t iree_vm_ref_wrap_retain(void* ptr,
                                                      iree_vm_ref_type_t type,
                                                      iree_vm_ref_t* out_ref);

// Checks that the given reference-counted pointer |ref| is of |type|.
static inline iree_status_t iree_vm_ref_check(const iree_vm_ref_t ref,
                                              iree_vm_ref_type_t type) {
  return IREE_LIKELY(ref.type == type)
             ? iree_ok_status()
             : iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                ref.type == IREE_VM_REF_TYPE_NULL
                                    ? "ref is null"
                                    : "ref type mismatch");
}

// Retains the reference-counted pointer |ref|.
IREE_API_EXPORT void iree_vm_ref_retain_inplace(iree_vm_ref_t* ref);

// Retains the reference-counted pointer |ref|.
// |out_ref| will be released if it already contains a reference.
IREE_API_EXPORT void iree_vm_ref_retain(iree_vm_ref_t* ref,
                                        iree_vm_ref_t* out_ref);

// Retains the reference-counted pointer |ref| and checks that it is of |type|.
// |out_ref| will be released if it already contains a reference.
IREE_API_EXPORT iree_status_t iree_vm_ref_retain_checked(
    iree_vm_ref_t* ref, iree_vm_ref_type_t type, iree_vm_ref_t* out_ref);

// Retains or moves |ref| to |out_ref|.
// |out_ref| will be released if it already contains a reference.
IREE_API_EXPORT void iree_vm_ref_retain_or_move(int is_move, iree_vm_ref_t* ref,
                                                iree_vm_ref_t* out_ref);

// Retains or moves |ref| to |out_ref| and checks that |ref| is of |type|.
// |out_ref| will be released if it already contains a reference.
IREE_API_EXPORT iree_status_t iree_vm_ref_retain_or_move_checked(
    int is_move, iree_vm_ref_t* ref, iree_vm_ref_type_t type,
    iree_vm_ref_t* out_ref);

// Releases the reference-counted pointer |ref|, possibly freeing it.
IREE_API_EXPORT void iree_vm_ref_release(iree_vm_ref_t* ref);

// Assigns the reference-counted pointer |ref| without incrementing the count.
// |out_ref| will be released if it already contains a reference.
IREE_API_EXPORT void iree_vm_ref_assign(iree_vm_ref_t* ref,
                                        iree_vm_ref_t* out_ref);

// Moves one reference to another without changing the reference count.
// |out_ref| will be released if it already contains a reference.
IREE_API_EXPORT void iree_vm_ref_move(iree_vm_ref_t* ref,
                                      iree_vm_ref_t* out_ref);

// Returns true if the given |ref| is NULL.
IREE_API_EXPORT bool iree_vm_ref_is_null(const iree_vm_ref_t* ref);

// Returns true if the two references point at the same value (or are both
// null).
IREE_API_EXPORT bool iree_vm_ref_equal(const iree_vm_ref_t* lhs,
                                       const iree_vm_ref_t* rhs);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Type adapter utilities for interfacing with the VM
//===----------------------------------------------------------------------===//

#if defined(IREE_VM_REF_PARANOID)
#define IREE_VM_REF_ASSERT(expr) IREE_ASSERT(expr)
#else
#define IREE_VM_REF_ASSERT(expr)
#endif  // IREE_VM_REF_PARANOID

// TODO(benvanik): make these macros standard/document them.
#define IREE_VM_DECLARE_TYPE_ADAPTERS(name, T)                              \
  IREE_API_EXPORT_VARIABLE iree_vm_ref_type_t name##_registration;          \
  static inline iree_vm_ref_type_t name##_type(void) {                      \
    IREE_VM_REF_ASSERT(name##_registration);                                \
    return name##_registration;                                             \
  }                                                                         \
  static inline bool name##_isa(const iree_vm_ref_t ref) {                  \
    IREE_VM_REF_ASSERT(name##_registration);                                \
    return name##_registration == ref.type;                                 \
  }                                                                         \
  IREE_API_EXPORT iree_vm_ref_t name##_retain_ref(T* value);                \
  IREE_API_EXPORT iree_vm_ref_t name##_move_ref(T* value);                  \
  IREE_ATTRIBUTE_UNUSED static inline T* name##_deref(                      \
      const iree_vm_ref_t ref) {                                            \
    IREE_VM_REF_ASSERT(name##_registration);                                \
    return IREE_LIKELY(name##_isa(ref)) ? (T*)ref.ptr : NULL;               \
  }                                                                         \
  IREE_API_EXPORT iree_status_t name##_check_deref(const iree_vm_ref_t ref, \
                                                   T** out_ptr);            \
  IREE_API_EXPORT iree_status_t name##_check_deref_or_null(                 \
      const iree_vm_ref_t ref, T** out_ptr);                                \
  IREE_VM_DECLARE_CC_TYPE_LOOKUP(name, T)

// TODO(benvanik): make these macros standard/document them.
#define IREE_VM_DEFINE_TYPE_ADAPTERS(name, T)                               \
  iree_vm_ref_type_t name##_registration = 0;                               \
  IREE_API_EXPORT iree_vm_ref_t name##_retain_ref(T* value) {               \
    IREE_VM_REF_ASSERT(name##_registration);                                \
    iree_vm_ref_t ref = {0};                                                \
    iree_vm_ref_wrap_retain(value, name##_type(), &ref);                    \
    return ref;                                                             \
  }                                                                         \
  IREE_API_EXPORT iree_vm_ref_t name##_move_ref(T* value) {                 \
    IREE_VM_REF_ASSERT(name##_registration);                                \
    iree_vm_ref_t ref = {0};                                                \
    iree_vm_ref_wrap_assign(value, name##_type(), &ref);                    \
    return ref;                                                             \
  }                                                                         \
  IREE_API_EXPORT iree_status_t name##_check_deref(const iree_vm_ref_t ref, \
                                                   T** out_ptr) {           \
    IREE_VM_REF_ASSERT(name##_registration);                                \
    IREE_RETURN_IF_ERROR(iree_vm_ref_check(ref, name##_type()));            \
    *out_ptr = (T*)ref.ptr;                                                 \
    return iree_ok_status();                                                \
  }                                                                         \
  IREE_API_EXPORT iree_status_t name##_check_deref_or_null(                 \
      const iree_vm_ref_t ref, T** out_ptr) {                               \
    IREE_VM_REF_ASSERT(name##_registration);                                \
    if (ref.type != IREE_VM_REF_TYPE_NULL) {                                \
      IREE_RETURN_IF_ERROR(iree_vm_ref_check(ref, name##_type()));          \
      *out_ptr = (T*)ref.ptr;                                               \
    } else {                                                                \
      *out_ptr = NULL;                                                      \
    }                                                                       \
    return iree_ok_status();                                                \
  }

// Optional C++ iree::vm::ref<T> wrapper.
#ifdef __cplusplus
#include "iree/vm/ref_cc.h"
#else
#define IREE_VM_DECLARE_CC_TYPE_LOOKUP(name, T)
#define IREE_VM_DECLARE_CC_TYPE_ADAPTERS(name, T)
#endif  // __cplusplus

#endif  // IREE_VM_REF_H_
