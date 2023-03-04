// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdlib.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/tracing.h"

//===----------------------------------------------------------------------===//
// iree_allocator_t (std::allocator-like interface)
//===----------------------------------------------------------------------===//

static iree_status_t iree_allocator_issue_alloc(
    iree_allocator_t allocator, iree_allocator_command_t command,
    iree_host_size_t byte_length, void** inout_ptr) {
  if (IREE_UNLIKELY(!allocator.ctl)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "allocator has no control routine");
  }
  iree_allocator_alloc_params_t params = {
      .byte_length = byte_length,
  };
  return allocator.ctl(allocator.self, command, &params, inout_ptr);
}

IREE_API_EXPORT iree_status_t iree_allocator_malloc(
    iree_allocator_t allocator, iree_host_size_t byte_length, void** out_ptr) {
  return iree_allocator_issue_alloc(allocator, IREE_ALLOCATOR_COMMAND_CALLOC,
                                    byte_length, out_ptr);
}

IREE_API_EXPORT iree_status_t iree_allocator_malloc_uninitialized(
    iree_allocator_t allocator, iree_host_size_t byte_length, void** out_ptr) {
  return iree_allocator_issue_alloc(allocator, IREE_ALLOCATOR_COMMAND_MALLOC,
                                    byte_length, out_ptr);
}

IREE_API_EXPORT iree_status_t
iree_allocator_realloc(iree_allocator_t allocator, iree_host_size_t byte_length,
                       void** inout_ptr) {
  return iree_allocator_issue_alloc(allocator, IREE_ALLOCATOR_COMMAND_REALLOC,
                                    byte_length, inout_ptr);
}

IREE_API_EXPORT iree_status_t
iree_allocator_clone(iree_allocator_t allocator,
                     iree_const_byte_span_t source_bytes, void** out_ptr) {
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_uninitialized(
      allocator, source_bytes.data_length, out_ptr));
  memcpy(*out_ptr, source_bytes.data, source_bytes.data_length);
  return iree_ok_status();
}

IREE_API_EXPORT void iree_allocator_free(iree_allocator_t allocator,
                                         void* ptr) {
  if (ptr && allocator.ctl) {
    iree_status_ignore(allocator.ctl(
        allocator.self, IREE_ALLOCATOR_COMMAND_FREE, /*params=*/NULL, &ptr));
  }
}

static iree_status_t iree_allocator_system_alloc(
    iree_allocator_command_t command,
    const iree_allocator_alloc_params_t* params, void** inout_ptr) {
  IREE_ASSERT_ARGUMENT(params);
  IREE_ASSERT_ARGUMENT(inout_ptr);
  iree_host_size_t byte_length = params->byte_length;
  if (IREE_UNLIKELY(byte_length == 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "allocations must be >0 bytes");
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  void* existing_ptr = *inout_ptr;
  void* new_ptr = NULL;
  if (existing_ptr && command == IREE_ALLOCATOR_COMMAND_REALLOC) {
    new_ptr = realloc(existing_ptr, byte_length);
  } else {
    existing_ptr = NULL;
    if (command == IREE_ALLOCATOR_COMMAND_CALLOC) {
      new_ptr = calloc(1, byte_length);
    } else {
      new_ptr = malloc(byte_length);
    }
  }
  if (!new_ptr) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "system allocator failed the request");
  }

  if (existing_ptr) {
    IREE_TRACE_FREE(existing_ptr);
  }
  IREE_TRACE_ALLOC(new_ptr, byte_length);

  *inout_ptr = new_ptr;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_allocator_system_free(void** inout_ptr) {
  IREE_ASSERT_ARGUMENT(inout_ptr);
  IREE_TRACE_ZONE_BEGIN(z0);
  void* ptr = *inout_ptr;
  if (IREE_LIKELY(ptr != NULL)) {
    IREE_TRACE_FREE(ptr);
    free(ptr);
    *inout_ptr = NULL;
  }
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t
iree_allocator_system_ctl(void* self, iree_allocator_command_t command,
                          const void* params, void** inout_ptr) {
  switch (command) {
    case IREE_ALLOCATOR_COMMAND_MALLOC:
    case IREE_ALLOCATOR_COMMAND_CALLOC:
    case IREE_ALLOCATOR_COMMAND_REALLOC:
      return iree_allocator_system_alloc(
          command, (const iree_allocator_alloc_params_t*)params, inout_ptr);
    case IREE_ALLOCATOR_COMMAND_FREE:
      return iree_allocator_system_free(inout_ptr);
    default:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "unsupported system allocator command");
  }
}

//===----------------------------------------------------------------------===//
// Aligned allocations via iree_allocator_t
//===----------------------------------------------------------------------===//

// Returns a pointer into |unaligned_ptr| where |offset| matches |alignment|.
static inline void* iree_aligned_ptr(void* unaligned_ptr,
                                     iree_host_size_t alignment,
                                     iree_host_size_t offset) {
  return (void*)((((uintptr_t)unaligned_ptr + (alignment + sizeof(void*)) +
                   offset) &
                  ~(uintptr_t)(alignment - 1)) -
                 offset);
}

// Returns the base unaligned pointer for |aligned_ptr|.
static inline void* iree_aligned_ptr_get_base(void* aligned_ptr) {
  void** ptr_ref =
      (void**)((uintptr_t)aligned_ptr & ~(uintptr_t)(sizeof(void*) - 1));
  return ptr_ref[-1];
}

// Sets the base unaligned pointer in |aligned_ptr|.
static inline void iree_aligned_ptr_set_base(void* aligned_ptr,
                                             void* base_ptr) {
  void** ptr_ref =
      (void**)((uintptr_t)aligned_ptr & ~(uintptr_t)(sizeof(void*) - 1));
  ptr_ref[-1] = base_ptr;
}

IREE_API_EXPORT iree_status_t iree_allocator_malloc_aligned(
    iree_allocator_t allocator, iree_host_size_t byte_length,
    iree_host_size_t min_alignment, iree_host_size_t offset, void** out_ptr) {
  IREE_ASSERT_ARGUMENT(out_ptr);
  if (IREE_UNLIKELY(byte_length == 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "allocations must be >0 bytes");
  }
  const iree_host_size_t alignment = iree_max(min_alignment, iree_max_align_t);
  if (IREE_UNLIKELY(!iree_host_size_is_power_of_two(alignment))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "alignments must be powers of two (got %" PRIhsz ")", min_alignment);
  }

  // [base ptr] [padding...] [aligned data] [padding...]
  const iree_host_size_t total_length =
      sizeof(uintptr_t) + byte_length + alignment;
  void* unaligned_ptr = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, total_length, (void**)&unaligned_ptr));
  void* aligned_ptr = iree_aligned_ptr(unaligned_ptr, alignment, offset);

  iree_aligned_ptr_set_base(aligned_ptr, unaligned_ptr);
  *out_ptr = aligned_ptr;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_allocator_realloc_aligned(
    iree_allocator_t allocator, iree_host_size_t byte_length,
    iree_host_size_t min_alignment, iree_host_size_t offset, void** inout_ptr) {
  IREE_ASSERT_ARGUMENT(inout_ptr);
  if (!*inout_ptr) {
    return iree_allocator_malloc_aligned(allocator, byte_length, min_alignment,
                                         offset, inout_ptr);
  }
  if (IREE_UNLIKELY(byte_length == 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "allocations must be >0 bytes");
  }
  const iree_host_size_t alignment = iree_min(min_alignment, iree_max_align_t);
  if (IREE_UNLIKELY(!iree_host_size_is_power_of_two(alignment))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "alignments must be powers of two (got %" PRIhsz ")", min_alignment);
  }
  void* aligned_ptr = *inout_ptr;
  void* unaligned_ptr = iree_aligned_ptr_get_base(aligned_ptr);
  if (IREE_UNLIKELY(aligned_ptr !=
                    iree_aligned_ptr(unaligned_ptr, alignment, offset))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "reallocation must have the same alignment as the "
                            "original allocation (got %" PRIhsz ")",
                            min_alignment);
  }

  // Since the reallocated memory block may have a different unaligned base to
  // aligned offset we may need to move the data. Capture the original offset
  // into the unaligned base where the valid data resides.
  uintptr_t old_offset = (uintptr_t)aligned_ptr - (uintptr_t)unaligned_ptr;

  // [base ptr] [padding...] [aligned data] [padding...]
  const iree_host_size_t total_length =
      sizeof(uintptr_t) + byte_length + alignment;
  IREE_RETURN_IF_ERROR(
      iree_allocator_realloc(allocator, total_length, (void**)&unaligned_ptr));
  aligned_ptr = iree_aligned_ptr(unaligned_ptr, alignment, offset);

  const uint8_t* old_data = (uint8_t*)unaligned_ptr + old_offset;
  uint8_t* new_data = (uint8_t*)aligned_ptr;
  if (old_data != new_data) {
    // Alignment at offset changed; copy data to the new aligned offset.
    // NOTE: this is copying up to the *new* byte length, as we don't store the
    // old length and don't know how much to copy. Since we've already
    // reallocated we know this will always be in-bounds, but it's inefficient.
    // NOTE: memmove instead of memcpy as the regions may overlap.
    memmove(new_data, old_data, byte_length);
  }

  iree_aligned_ptr_set_base(aligned_ptr, unaligned_ptr);
  *inout_ptr = aligned_ptr;
  return iree_ok_status();
}

IREE_API_EXPORT void iree_allocator_free_aligned(iree_allocator_t allocator,
                                                 void* ptr) {
  if (ptr) {
    void* unaligned_ptr = iree_aligned_ptr_get_base(ptr);
    iree_allocator_free(allocator, unaligned_ptr);
  }
}
