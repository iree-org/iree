// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdlib.h>

#include "iree/base/api.h"

//===----------------------------------------------------------------------===//
// libc allocator implementation
//===----------------------------------------------------------------------===//

static iree_status_t iree_allocator_libc_alloc(
    iree_allocator_command_t command,
    const iree_allocator_alloc_params_t* params, void** inout_ptr) {
  IREE_ASSERT_ARGUMENT(params);
  IREE_ASSERT_ARGUMENT(inout_ptr);
  iree_host_size_t byte_length = params->byte_length;
  if (IREE_UNLIKELY(byte_length == 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "allocations must be >0 bytes");
  }

  void* existing_ptr = *inout_ptr;

  IREE_TRACE(iree_zone_id_t z0 = 0);
  IREE_TRACE({
    if (existing_ptr && command == IREE_ALLOCATOR_COMMAND_REALLOC) {
      IREE_TRACE_ZONE_BEGIN_NAMED(z0_named, "iree_allocator_libc_realloc");
      z0 = z0_named;
    } else if (command == IREE_ALLOCATOR_COMMAND_CALLOC) {
      IREE_TRACE_ZONE_BEGIN_NAMED(z0_named, "iree_allocator_libc_calloc");
      z0 = z0_named;
    } else {
      IREE_TRACE_ZONE_BEGIN_NAMED(z0_named, "iree_allocator_libc_malloc");
      z0 = z0_named;
    }
  });

  void* existing_ptr_value = NULL;
  void* new_ptr = NULL;
  if (existing_ptr && command == IREE_ALLOCATOR_COMMAND_REALLOC) {
    existing_ptr_value = iree_tracing_obscure_ptr(existing_ptr);
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
                            "libc allocator failed the request");
  }

  if (existing_ptr_value) {
    IREE_TRACE_FREE(existing_ptr_value);
  }
  IREE_TRACE_ALLOC(new_ptr, byte_length);

  *inout_ptr = new_ptr;
  IREE_TRACE(IREE_TRACE_ZONE_END(z0));
  return iree_ok_status();
}

static iree_status_t iree_allocator_libc_free(void** inout_ptr) {
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
iree_allocator_libc_ctl(void* self, iree_allocator_command_t command,
                        const void* params, void** inout_ptr) {
  switch (command) {
    case IREE_ALLOCATOR_COMMAND_MALLOC:
    case IREE_ALLOCATOR_COMMAND_CALLOC:
    case IREE_ALLOCATOR_COMMAND_REALLOC:
      return iree_allocator_libc_alloc(
          command, (const iree_allocator_alloc_params_t*)params, inout_ptr);
    case IREE_ALLOCATOR_COMMAND_FREE:
      return iree_allocator_libc_free(inout_ptr);
    default:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "unsupported libc allocator command");
  }
}
