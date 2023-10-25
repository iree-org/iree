// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/ref.h"

#include <string.h>

#include "iree/base/internal/atomics.h"

// Useful debugging tool:
#if 0
static inline volatile iree_atomic_ref_count_t* iree_vm_get_raw_counter_ptr(
    void* ptr, iree_vm_ref_type_t type);

static inline volatile iree_atomic_ref_count_t* iree_vm_get_ref_counter_ptr(
    iree_vm_ref_t* ref);

static void iree_vm_ref_trace(const char* msg, iree_vm_ref_t* ref) {
  volatile iree_atomic_ref_count_t* counter = iree_vm_get_ref_counter_ptr(ref);
  iree_string_view_t name = iree_vm_ref_type_name(ref->type);
  fprintf(stderr, "%s %.*s 0x%p %d\n", msg, (int)name.size, name.data, ref->ptr,
          iree_atomic_ref_count_load(counter));
}
static void iree_vm_ref_ptr_trace(const char* msg, void* ptr,
                                  iree_vm_ref_type_t type) {
  volatile iree_atomic_ref_count_t* counter =
      iree_vm_get_raw_counter_ptr(ptr, type);
  iree_string_view_t name = iree_vm_ref_type_name(type);
  fprintf(stderr, "%s %.*s 0x%p %d\n", msg, (int)name.size, name.data, ptr,
          iree_atomic_ref_count_load(counter));
}
#else
#define iree_vm_ref_trace(...)
#define iree_vm_ref_ptr_trace(...)
#endif  // 0

IREE_API_EXPORT iree_string_view_t
iree_vm_ref_type_name(iree_vm_ref_type_t type) {
  IREE_VM_REF_ASSERT(type);
  return iree_vm_ref_type_descriptor(type)->type_name;
}

static inline volatile iree_atomic_ref_count_t* iree_vm_get_raw_counter_ptr(
    void* ptr, iree_vm_ref_type_t type) {
  IREE_VM_REF_ASSERT(ptr);
  IREE_VM_REF_ASSERT(type_descriptor);
  return (volatile iree_atomic_ref_count_t*)ptr +
         (type & IREE_VM_REF_TYPE_TAG_BIT_MASK);
}

static inline volatile iree_atomic_ref_count_t* iree_vm_get_ref_counter_ptr(
    iree_vm_ref_t* ref) {
  IREE_VM_REF_ASSERT(ref);
  IREE_VM_REF_ASSERT(ref->ptr);
  return (volatile iree_atomic_ref_count_t*)ref->ptr +
         (ref->type & IREE_VM_REF_TYPE_TAG_BIT_MASK);
}

IREE_API_EXPORT void iree_vm_ref_object_retain(void* ptr,
                                               iree_vm_ref_type_t type) {
  if (!ptr) return;
  IREE_VM_REF_ASSERT(type);
  volatile iree_atomic_ref_count_t* counter =
      iree_vm_get_raw_counter_ptr(ptr, type);
  iree_atomic_ref_count_inc(counter);
  iree_vm_ref_ptr_trace("RETAIN", ptr, type);
}

IREE_API_EXPORT void iree_vm_ref_object_release(void* ptr,
                                                iree_vm_ref_type_t type) {
  if (!ptr) return;
  IREE_VM_REF_ASSERT(type);
  iree_vm_ref_ptr_trace("RELEASE", ptr, type);
  volatile iree_atomic_ref_count_t* counter =
      iree_vm_get_raw_counter_ptr(ptr, type);
  if (iree_atomic_ref_count_dec(counter) == 1) {
    const iree_vm_ref_type_descriptor_t* descriptor =
        iree_vm_ref_type_descriptor(type);
    if (descriptor->destroy) {
      // NOTE: this makes us not re-entrant, but I think that's OK.
      iree_vm_ref_ptr_trace("DESTROY", ptr, type);
      descriptor->destroy(ptr);
    }
  }
}

IREE_API_EXPORT iree_status_t iree_vm_ref_wrap_assign(void* ptr,
                                                      iree_vm_ref_type_t type,
                                                      iree_vm_ref_t* out_ref) {
  IREE_VM_REF_ASSERT(ptr);
  IREE_VM_REF_ASSERT(type);
  IREE_VM_REF_ASSERT(out_ref);
  IREE_VM_REF_ASSERT(iree_vm_ref_type_descriptor(type));

  if (out_ref->ptr != NULL && out_ref->ptr != ptr) {
    // Release existing value.
    iree_vm_ref_release(out_ref);
  }

  // NOTE: we do not manipulate the counter here as we assume it starts at 1
  // or it's already coming in with some references.
  out_ref->ptr = ptr;
  out_ref->type = type;

  iree_vm_ref_trace("WRAP ASSIGN", out_ref);
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_vm_ref_wrap_retain(void* ptr,
                                                      iree_vm_ref_type_t type,
                                                      iree_vm_ref_t* out_ref) {
  IREE_VM_REF_ASSERT(ptr);
  IREE_VM_REF_ASSERT(type);
  IREE_VM_REF_ASSERT(out_ref);
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
  IREE_VM_REF_ASSERT(ref);
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
  IREE_VM_REF_ASSERT(ref);
  IREE_VM_REF_ASSERT(out_ref);
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
  IREE_VM_REF_ASSERT(ref);
  IREE_VM_REF_ASSERT(type);
  IREE_VM_REF_ASSERT(out_ref);
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
  IREE_VM_REF_ASSERT(ref);
  IREE_VM_REF_ASSERT(out_ref);
  if (is_move) {
    iree_vm_ref_move(ref, out_ref);
  } else {
    iree_vm_ref_retain(ref, out_ref);
  }
}

IREE_API_EXPORT iree_status_t iree_vm_ref_retain_or_move_checked(
    int is_move, iree_vm_ref_t* ref, iree_vm_ref_type_t type,
    iree_vm_ref_t* out_ref) {
  IREE_VM_REF_ASSERT(ref);
  IREE_VM_REF_ASSERT(type);
  IREE_VM_REF_ASSERT(out_ref);
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
  IREE_VM_REF_ASSERT(ref);
  if (ref->type == IREE_VM_REF_TYPE_NULL || ref->ptr == NULL) return;

  iree_vm_ref_trace("RELEASE", ref);
  volatile iree_atomic_ref_count_t* counter = iree_vm_get_ref_counter_ptr(ref);
  if (iree_atomic_ref_count_dec(counter) == 1) {
    const iree_vm_ref_type_descriptor_t* descriptor =
        iree_vm_ref_type_descriptor(ref->type);
    if (descriptor->destroy) {
      // NOTE: this makes us not re-entrant, but I think that's OK.
      iree_vm_ref_trace("DESTROY", ref);
      descriptor->destroy(ref->ptr);
    }
  }

  // Reset ref to point at nothing.
  memset(ref, 0, sizeof(*ref));
}

IREE_API_EXPORT void iree_vm_ref_assign(iree_vm_ref_t* ref,
                                        iree_vm_ref_t* out_ref) {
  IREE_VM_REF_ASSERT(ref);
  IREE_VM_REF_ASSERT(out_ref);

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
  IREE_VM_REF_ASSERT(ref);
  IREE_VM_REF_ASSERT(out_ref);

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
  IREE_VM_REF_ASSERT(ref);
  return ref->type == IREE_VM_REF_TYPE_NULL;
}

IREE_API_EXPORT bool iree_vm_ref_equal(const iree_vm_ref_t* lhs,
                                       const iree_vm_ref_t* rhs) {
  return lhs == rhs || memcmp(lhs, rhs, sizeof(*lhs)) == 0;
}
