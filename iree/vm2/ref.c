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

#include "iree/vm2/ref.h"

#include <assert.h>
#include <stdatomic.h>
#include <string.h>

#if ATOMIC_POINTER_LOCK_FREE != 2
#error "Your compiler does not ensure lock-free atomic ops"
#endif  // ATOMIC_POINTER_LOCK_FREE

#define IREE_GET_REF_COUNTER_PTR(ref) \
  ((volatile atomic_intptr_t*)(((uintptr_t)ref->ptr) + ref->offsetof_counter))

// A table of type descriptors registered at startup.
// These provide quick dereferencing of destruction functions and type names for
// debugging.
static iree_vm_ref_type_descriptor_t iree_vm_ref_type_builtin_descriptors
    [IREE_VM_REF_TYPE_FIRST_USER_DEFINED_TYPE] = {{0}};

// Returns the type descriptor (or NULL) for the given type ID.
static const iree_vm_ref_type_descriptor_t* iree_vm_ref_get_type_descriptor(
    iree_vm_ref_type_t type) {
  if (type < IREE_VM_REF_TYPE_FIRST_USER_DEFINED_TYPE) {
    return iree_vm_ref_type_builtin_descriptors[type].type == type
               ? &iree_vm_ref_type_builtin_descriptors[type]
               : NULL;
  } else {
    // TODO(benvanik): user-defined types.
    return NULL;
  }
}

IREE_API_EXPORT void IREE_API_CALL
iree_vm_ref_register_builtin_type(iree_vm_ref_type_descriptor_t descriptor) {
  assert(descriptor.type < IREE_VM_REF_TYPE_FIRST_USER_DEFINED_TYPE);
  if (descriptor.type >= IREE_VM_REF_TYPE_FIRST_USER_DEFINED_TYPE) {
    return;
  }
  iree_vm_ref_type_builtin_descriptors[descriptor.type] = descriptor;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_vm_ref_wrap(void* ptr, iree_vm_ref_type_t type, iree_vm_ref_t* out_ref) {
  const iree_vm_ref_type_descriptor_t* type_descriptor =
      iree_vm_ref_get_type_descriptor(type);
  if (!type_descriptor) {
    // Type not registered.
    return IREE_STATUS_INVALID_ARGUMENT;
  }

  if (out_ref->ptr != NULL) {
    // Release existing value.
    iree_vm_ref_release(out_ref);
  }

  // NOTE: we do not manipulate the counter here as we assume it starts at 1
  // or it's already coming in with some references.
  out_ref->ptr = ptr;
  out_ref->offsetof_counter = type_descriptor->offsetof_counter;
  out_ref->type = type;

  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_vm_ref_check(iree_vm_ref_t* ref, iree_vm_ref_type_t type) {
  return ref->type == type ? IREE_STATUS_OK : IREE_STATUS_INVALID_ARGUMENT;
}

IREE_API_EXPORT void IREE_API_CALL iree_vm_ref_retain(iree_vm_ref_t* ref,
                                                      iree_vm_ref_t* out_ref) {
  if (ref != out_ref && ref->ptr != out_ref->ptr) {
    // Output ref contains a value that should be released first.
    // Note that we check for it being the same as the new value so we don't
    // do extra work unless we have to.
    iree_vm_ref_release(out_ref);
  }

  // Assign ref to out_ref and increment the counter.
  memcpy(out_ref, ref, sizeof(*out_ref));
  if (out_ref->ptr) {
    volatile atomic_intptr_t* counter = IREE_GET_REF_COUNTER_PTR(out_ref);
    atomic_fetch_add(counter, 1);
  }
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_vm_ref_retain_checked(
    iree_vm_ref_t* ref, iree_vm_ref_type_t type, iree_vm_ref_t* out_ref) {
  if (ref->type != IREE_VM_REF_TYPE_NULL && ref->type != type) {
    // Make no changes on failure.
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  iree_vm_ref_retain(ref, out_ref);
  return IREE_STATUS_OK;
}

IREE_API_EXPORT void IREE_API_CALL iree_vm_ref_retain_or_move(
    int is_move, iree_vm_ref_t* ref, iree_vm_ref_t* out_ref) {
  if (ref != out_ref && ref->ptr != out_ref->ptr) {
    // Output ref contains a value that should be released first.
    // Note that we check for it being the same as the new value so we don't
    // do extra work unless we have to.
    iree_vm_ref_release(out_ref);
  }

  // Assign ref to out_ref and increment the counter if not moving.
  memcpy(out_ref, ref, sizeof(*out_ref));
  if (out_ref->ptr && !is_move) {
    // Retain by incrementing counter and preserving the source ref.
    volatile atomic_intptr_t* counter = IREE_GET_REF_COUNTER_PTR(out_ref);
    atomic_fetch_add(counter, 1);
  } else if (ref != out_ref) {
    // Move by not changing counter and clearing the source ref.
    memset(ref, 0, sizeof(*ref));
  }
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_vm_ref_retain_or_move_checked(
    int is_move, iree_vm_ref_t* ref, iree_vm_ref_type_t type,
    iree_vm_ref_t* out_ref) {
  if (ref->type != IREE_VM_REF_TYPE_NULL && ref->type != type) {
    // Make no changes on failure.
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  iree_vm_ref_retain_or_move(is_move, ref, out_ref);
  return IREE_STATUS_OK;
}

IREE_API_EXPORT void IREE_API_CALL iree_vm_ref_release(iree_vm_ref_t* ref) {
  if (ref->ptr != NULL) {
    volatile atomic_intptr_t* counter = IREE_GET_REF_COUNTER_PTR(ref);
    if (atomic_fetch_sub(counter, 1) == 1) {
      const iree_vm_ref_type_descriptor_t* type_descriptor =
          iree_vm_ref_get_type_descriptor(ref->type);
      if (type_descriptor->destroy) {
        // NOTE: this makes us not re-entrant, but I think that's OK.
        type_descriptor->destroy(ref->ptr);
      }
    }
  }

  // Reset ref to point at nothing.
  memset(ref, 0, sizeof(*ref));
}

IREE_API_EXPORT void IREE_API_CALL iree_vm_ref_assign(iree_vm_ref_t* ref,
                                                      iree_vm_ref_t* out_ref) {
  if (ref == out_ref) {
    // Source == target; ignore.
    return;
  } else if (out_ref->ptr != NULL) {
    // Release existing value.
    iree_vm_ref_release(out_ref);
  }

  // Assign ref to out_ref (without incrementing counter).
  memcpy(out_ref, ref, sizeof(*out_ref));
}

IREE_API_EXPORT void IREE_API_CALL iree_vm_ref_move(iree_vm_ref_t* ref,
                                                    iree_vm_ref_t* out_ref) {
  if (ref == out_ref) {
    // Source == target; ignore.
    return;
  } else if (out_ref->ptr != NULL) {
    // Release existing value.
    iree_vm_ref_release(out_ref);
  }

  // Assign ref to out_ref (without incrementing counter).
  memcpy(out_ref, ref, sizeof(*out_ref));

  // Reset input ref so it points at nothing.
  memset(ref, 0, sizeof(*ref));
}

IREE_API_EXPORT int IREE_API_CALL iree_vm_ref_equal(iree_vm_ref_t* lhs,
                                                    iree_vm_ref_t* rhs) {
  return memcmp(lhs, rhs, sizeof(*lhs)) == 0;
}
