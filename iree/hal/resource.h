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

#ifndef IREE_HAL_RESOURCE_H_
#define IREE_HAL_RESOURCE_H_

#include <stdbool.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"
#include "iree/base/internal/debugging.h"

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
typedef struct iree_hal_resource_s {
  // Reference count used to manage resource lifetime. The vtable->destroy
  // method will be called when the reference count falls to zero.
  iree_atomic_ref_count_t ref_count;

  // Opaque vtable for the resource object.
  //
  // NOTE: this field may be hidden in the future. Only use this for
  // IREE_HAL_VTABLE_DISPATCH and not equality/direct dereferencing.
  const void* vtable;

  // TODO(benvanik): debug string/logging utilities.
} iree_hal_resource_t;

static inline void iree_hal_resource_initialize(
    const void* vtable, iree_hal_resource_t* out_resource) {
  iree_atomic_ref_count_init(&out_resource->ref_count);
  out_resource->vtable = vtable;
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
