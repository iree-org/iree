// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdlib.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/tracing.h"

IREE_API_EXPORT iree_status_t iree_allocator_malloc(
    iree_allocator_t allocator, iree_host_size_t byte_length, void** out_ptr) {
  if (IREE_UNLIKELY(!allocator.ctl)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "allocator has no control routine");
  }
  iree_allocator_alloc_params_t params = {
      .byte_length = byte_length,
  };
  return allocator.ctl(allocator.self, IREE_ALLOCATOR_COMMAND_CALLOC, &params,
                       out_ptr);
}

IREE_API_EXPORT iree_status_t iree_allocator_malloc_uninitialized(
    iree_allocator_t allocator, iree_host_size_t byte_length, void** out_ptr) {
  if (IREE_UNLIKELY(!allocator.ctl)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "allocator has no control routine");
  }
  iree_allocator_alloc_params_t params = {
      .byte_length = byte_length,
  };
  return allocator.ctl(allocator.self, IREE_ALLOCATOR_COMMAND_MALLOC, &params,
                       out_ptr);
}

IREE_API_EXPORT iree_status_t
iree_allocator_realloc(iree_allocator_t allocator, iree_host_size_t byte_length,
                       void** inout_ptr) {
  if (IREE_UNLIKELY(!allocator.ctl)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "allocator has no control routine");
  }
  iree_allocator_alloc_params_t params = {
      .byte_length = byte_length,
  };
  return allocator.ctl(allocator.self, IREE_ALLOCATOR_COMMAND_REALLOC, &params,
                       inout_ptr);
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
  if (byte_length == 0) {
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
