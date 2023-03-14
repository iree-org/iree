// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_RESOURCE_H_
#define IREE_HAL_RESOURCE_H_

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Abstract resource type whose lifetime is managed by reference counting.
// Used mostly just to get a virtual dtor and vtable, though we could add nicer
// logging by allowing resources to capture debug names, stack traces of
// creation, etc.
//
// All resource types must have the iree_hal_resource_t at offset 0. This allows
// the HAL code to cast any type pointer to a resource to gain access to the
// ref count and vtable at predictable locations. Note that this allows for the
// resource to be at >0 of the allocation but the pointers used with the HAL
// (iree_hal_event_t*, etc) must point to the iree_hal_resource_t.
typedef struct iree_hal_resource_t {
  // Reference count used to manage resource lifetime. The vtable->destroy
  // method will be called when the reference count falls to zero.
  iree_atomic_ref_count_t ref_count;

  // Opaque vtable for the resource object.
  // Must start with iree_hal_resource_vtable_t at offset 0.
  //
  // NOTE: this field may be hidden in the future. Only use this for
  // IREE_HAL_VTABLE_DISPATCH and not equality/direct dereferencing.
  const void* vtable;

  // TODO(benvanik): debug string/logging utilities.
} iree_hal_resource_t;

// Base vtable for all resources.
// This provides the base functions required to generically manipulate resources
// of various types.
//
// This must be aliased at offset 0 of all typed vtables:
//   typedef struct iree_hal_foo_vtable_t {
//     void(IREE_API_PTR* destroy)(...);
//     void(IREE_API_PTR* foo_method)(...);
//   } iree_hal_foo_vtable_t;
typedef struct iree_hal_resource_vtable_t {
  // Destroys the resource upon the final reference being released.
  // The resource pointer must be assumed invalid upon return from the function
  // (even if in some implementations its returned to a pool and still live).
  void(IREE_API_PTR* destroy)(iree_hal_resource_t* resource);
} iree_hal_resource_vtable_t;

// Verifies that the vtable has the right resource sub-vtable.
#define IREE_HAL_ASSERT_VTABLE_LAYOUT(vtable_type)   \
  static_assert(offsetof(vtable_type, destroy) == 0, \
                "iree_hal_resource_vtable_t must be at offset 0");

// Initializes the base resource type.
static inline void iree_hal_resource_initialize(
    const void* vtable, iree_hal_resource_t* out_resource) {
  iree_atomic_ref_count_init(&out_resource->ref_count);
  out_resource->vtable = vtable;
}

// Retains a resource for the caller.
static inline void iree_hal_resource_retain(const void* any_resource) {
  iree_hal_resource_t* resource = (iree_hal_resource_t*)any_resource;
  if (IREE_LIKELY(resource)) {
    iree_atomic_ref_count_inc(&resource->ref_count);
  }
}

// Releases a resource and destroys it if there are no more references.
// This routes through the vtable and can disable optimizations; always prefer
// to use the type-specific release functions (such as iree_hal_buffer_release)
// to allow for more optimizations and better compile-time type safety.
static inline void iree_hal_resource_release(const void* any_resource) {
  iree_hal_resource_t* resource = (iree_hal_resource_t*)any_resource;
  if (IREE_LIKELY(resource) &&
      iree_atomic_ref_count_dec(&resource->ref_count) == 1) {
    ((iree_hal_resource_vtable_t*)resource->vtable)->destroy(resource);
  }
}

// Returns true if the |resource| has the given |vtable| type.
// This is *not* a way to ensure that an instance is of a specific type but
// instead that it has a compatible vtable. This is because LTO may very rarely
// dedupe identical vtables and cause the pointer comparison to succeed even if
// the spellings of the types differs.
static inline bool iree_hal_resource_is(const void* resource,
                                        const void* vtable) {
  return resource ? ((const iree_hal_resource_t*)resource)->vtable == vtable
                  : false;
}

// Asserts (**DEBUG ONLY**) that the |resource| has the given |vtable| type.
// This is only useful to check for programmer error and may have false
// positives - do not rely on it for handling untrusted user input.
#define IREE_HAL_ASSERT_TYPE(resource, vtable)             \
  IREE_ASSERT_TRUE(iree_hal_resource_is(resource, vtable), \
                   "type does not match expected " #vtable)

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_RESOURCE_H_
