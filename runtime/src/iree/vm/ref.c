// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/ref.h"

#include <string.h>

#include "iree/base/internal/atomics.h"

// TODO(benvanik): dynamic, if we care - otherwise keep small.
// After a dozen or so types the linear scan will likely start to spill the
// DCACHE and need to be reworked. I suspect at the time we have >=64 types
// we'll want to rewrite all of this anyway (using externalized type ID storage
// or something more complex).
#define IREE_VM_MAX_TYPE_ID 64

static inline volatile iree_atomic_ref_count_t* iree_vm_get_raw_counter_ptr(
    void* ptr, const iree_vm_ref_type_descriptor_t* type_descriptor) {
  return (volatile iree_atomic_ref_count_t*)ptr +
         type_descriptor->offsetof_counter;
}

static inline volatile iree_atomic_ref_count_t* iree_vm_get_ref_counter_ptr(
    iree_vm_ref_t* ref) {
  return (volatile iree_atomic_ref_count_t*)ref->ptr + ref->offsetof_counter;
}

IREE_API_EXPORT void iree_vm_ref_object_retain(
    void* ptr, const iree_vm_ref_type_descriptor_t* type_descriptor) {
  if (!ptr) return;
  volatile iree_atomic_ref_count_t* counter =
      iree_vm_get_raw_counter_ptr(ptr, type_descriptor);
  iree_atomic_ref_count_inc(counter);
}

IREE_API_EXPORT void iree_vm_ref_object_release(
    void* ptr, const iree_vm_ref_type_descriptor_t* type_descriptor) {
  if (!ptr) return;
  volatile iree_atomic_ref_count_t* counter =
      iree_vm_get_raw_counter_ptr(ptr, type_descriptor);
  if (iree_atomic_ref_count_dec(counter) == 1) {
    if (type_descriptor->destroy) {
      // NOTE: this makes us not re-entrant, but I think that's OK.
      type_descriptor->destroy(ptr);
    }
  }
}

IREE_API_EXPORT iree_string_view_t
iree_vm_ref_type_name(iree_vm_ref_type_t type) {
  return ((const iree_vm_ref_type_descriptor_t*)type)->type_name;
}

// A table of type descriptors registered at startup.
// These provide quick dereferencing of destruction functions and type names for
// debugging. Note that this just points to registered descriptors (or NULL) for
// each type ID in the type range and does not own the descriptors.
//
// Note that [0] is always the NULL type and has a NULL descriptor. We don't
// allow types to be registered there.
static const iree_vm_ref_type_descriptor_t*
    iree_vm_ref_type_descriptors[IREE_VM_MAX_TYPE_ID] = {0};

IREE_API_EXPORT iree_status_t iree_vm_instance_register_type(
    iree_vm_instance_t* instance, iree_vm_ref_type_descriptor_t* descriptor) {
  // HACK: until properly registering we do this scan each time.
  // Callers shouldn't be registering types in tight loops anyway.
  for (int i = 1; i <= IREE_VM_MAX_TYPE_ID; ++i) {
    if (iree_vm_ref_type_descriptors[i] == descriptor) {
      // Already registered.
      return iree_ok_status();
    }
    if (!iree_vm_ref_type_descriptors[i]) {
      // Store in free slot.
      iree_vm_ref_type_descriptors[i] = descriptor;
      return iree_ok_status();
    }
  }
  // Too many user-defined types registered; need to increase
  // IREE_VM_MAX_TYPE_ID.
  return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                          "too many user-defined types registered; new type "
                          "would exceed maximum of %d",
                          IREE_VM_MAX_TYPE_ID);
}

IREE_API_EXPORT void iree_vm_instance_unregister_type(
    iree_vm_instance_t* instance, iree_vm_ref_type_descriptor_t* descriptor) {
  for (int i = 1; i <= IREE_VM_MAX_TYPE_ID; ++i) {
    if (iree_vm_ref_type_descriptors[i] == descriptor) {
      iree_vm_ref_type_descriptors[i] = NULL;
      return;
    }
  }
}

IREE_API_EXPORT const iree_vm_ref_type_descriptor_t*
iree_vm_instance_lookup_type(iree_vm_instance_t* instance,
                             iree_string_view_t full_name) {
  for (int i = 1; i <= IREE_VM_MAX_TYPE_ID; ++i) {
    if (iree_vm_ref_type_descriptors[i] &&
        iree_string_view_equal(iree_vm_ref_type_descriptors[i]->type_name,
                               full_name)) {
      return iree_vm_ref_type_descriptors[i];
    }
  }
  return NULL;
}

// Useful debugging tool:
#if 0
static void iree_vm_ref_trace(const char* msg, iree_vm_ref_t* ref) {
  volatile iree_atomic_ref_count_t* counter = iree_vm_get_ref_counter_ptr(ref);
  iree_string_view_t name = iree_vm_ref_type_name(ref->type);
  fprintf(stderr, "%s %.*s 0x%p %d\n", msg, (int)name.size, name.data, ref->ptr,
          counter->__val);
}
#else
#define iree_vm_ref_trace(...)
#endif  // 0

IREE_API_EXPORT iree_status_t iree_vm_ref_wrap_assign(void* ptr,
                                                      iree_vm_ref_type_t type,
                                                      iree_vm_ref_t* out_ref) {
  const iree_vm_ref_type_descriptor_t* type_descriptor =
      (const iree_vm_ref_type_descriptor_t*)type;
  if (!type_descriptor) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "type not registered");
  }

  if (out_ref->ptr != NULL && out_ref->ptr != ptr) {
    // Release existing value.
    iree_vm_ref_release(out_ref);
  }

  // NOTE: we do not manipulate the counter here as we assume it starts at 1
  // or it's already coming in with some references.
  out_ref->ptr = ptr;
  out_ref->offsetof_counter = type_descriptor->offsetof_counter;
  out_ref->type = type;

  iree_vm_ref_trace("WRAP ASSIGN", out_ref);
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_vm_ref_wrap_retain(void* ptr,
                                                      iree_vm_ref_type_t type,
                                                      iree_vm_ref_t* out_ref) {
  IREE_RETURN_IF_ERROR(iree_vm_ref_wrap_assign(ptr, type, out_ref));
  if (out_ref->ptr) {
    volatile iree_atomic_ref_count_t* counter =
        iree_vm_get_ref_counter_ptr(out_ref);
    iree_atomic_ref_count_inc(counter);
    iree_vm_ref_trace("WRAP RETAIN", out_ref);
  }
  return iree_ok_status();
}

IREE_API_EXPORT void iree_vm_ref_retain_inplace(iree_vm_ref_t* ref) {
  if (ref->ptr) {
    volatile iree_atomic_ref_count_t* counter =
        iree_vm_get_ref_counter_ptr(ref);
    iree_atomic_ref_count_inc(counter);
    iree_vm_ref_trace("RETAIN", ref);
  }
}

IREE_API_EXPORT void iree_vm_ref_retain(iree_vm_ref_t* ref,
                                        iree_vm_ref_t* out_ref) {
  // NOTE: ref and out_ref may alias or be nested so we retain before we
  // potentially release.
  iree_vm_ref_t temp_ref = *ref;
  if (ref->ptr) {
    volatile iree_atomic_ref_count_t* counter =
        iree_vm_get_ref_counter_ptr(ref);
    iree_atomic_ref_count_inc(counter);
    iree_vm_ref_trace("RETAIN", ref);
  }
  if (out_ref->ptr) {
    // Output ref contains a value that should be released first.
    // Note that we check above for it being the same as the new value so we
    // don't do extra work unless we have to.
    iree_vm_ref_release(out_ref);
  }
  *out_ref = temp_ref;
}

IREE_API_EXPORT iree_status_t iree_vm_ref_retain_checked(
    iree_vm_ref_t* ref, iree_vm_ref_type_t type, iree_vm_ref_t* out_ref) {
  if (ref->type != IREE_VM_REF_TYPE_NULL && ref->type != type &&
      type != IREE_VM_REF_TYPE_ANY) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source ref type mismatch");
  }
  iree_vm_ref_retain(ref, out_ref);
  return iree_ok_status();
}

IREE_API_EXPORT void iree_vm_ref_retain_or_move(int is_move, iree_vm_ref_t* ref,
                                                iree_vm_ref_t* out_ref) {
  if (is_move) {
    iree_vm_ref_move(ref, out_ref);
  } else {
    iree_vm_ref_retain(ref, out_ref);
  }
}

IREE_API_EXPORT iree_status_t iree_vm_ref_retain_or_move_checked(
    int is_move, iree_vm_ref_t* ref, iree_vm_ref_type_t type,
    iree_vm_ref_t* out_ref) {
  if (ref->type != IREE_VM_REF_TYPE_NULL && ref->type != type &&
      type != IREE_VM_REF_TYPE_ANY) {
    // Make no changes on failure.
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source ref type mismatch");
  }
  iree_vm_ref_retain_or_move(is_move, ref, out_ref);
  return iree_ok_status();
}

IREE_API_EXPORT void iree_vm_ref_release(iree_vm_ref_t* ref) {
  if (ref->type == IREE_VM_REF_TYPE_NULL || ref->ptr == NULL) return;

  iree_vm_ref_trace("RELEASE", ref);
  volatile iree_atomic_ref_count_t* counter = iree_vm_get_ref_counter_ptr(ref);
  if (iree_atomic_ref_count_dec(counter) == 1) {
    const iree_vm_ref_type_descriptor_t* type_descriptor =
        (const iree_vm_ref_type_descriptor_t*)ref->type;
    if (type_descriptor->destroy) {
      // NOTE: this makes us not re-entrant, but I think that's OK.
      iree_vm_ref_trace("DESTROY", ref);
      type_descriptor->destroy(ref->ptr);
    }
  }

  // Reset ref to point at nothing.
  memset(ref, 0, sizeof(*ref));
}

IREE_API_EXPORT void iree_vm_ref_assign(iree_vm_ref_t* ref,
                                        iree_vm_ref_t* out_ref) {
  // NOTE: ref and out_ref may alias.
  iree_vm_ref_t temp_ref = *ref;
  if (ref == out_ref) {
    // Source == target; ignore entirely.
    return;
  } else if (out_ref->ptr != NULL) {
    // Release existing value.
    iree_vm_ref_release(out_ref);
  }

  // Assign ref to out_ref (without incrementing counter).
  *out_ref = temp_ref;
}

IREE_API_EXPORT void iree_vm_ref_move(iree_vm_ref_t* ref,
                                      iree_vm_ref_t* out_ref) {
  // NOTE: ref and out_ref may alias.
  if (ref == out_ref) {
    // Source == target; ignore entirely.
    return;
  }

  // Reset input ref so it points at nothing.
  iree_vm_ref_t temp_ref = *ref;
  memset(ref, 0, sizeof(*ref));

  if (out_ref->ptr != NULL) {
    // Release existing value.
    iree_vm_ref_release(out_ref);
  }

  // Assign ref to out_ref (without incrementing counter).
  *out_ref = temp_ref;
}

IREE_API_EXPORT bool iree_vm_ref_is_null(const iree_vm_ref_t* ref) {
  return ref->type == IREE_VM_REF_TYPE_NULL;
}

IREE_API_EXPORT bool iree_vm_ref_equal(const iree_vm_ref_t* lhs,
                                       const iree_vm_ref_t* rhs) {
  return lhs == rhs || memcmp(lhs, rhs, sizeof(*lhs)) == 0;
}
