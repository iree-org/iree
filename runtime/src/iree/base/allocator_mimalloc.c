// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/config.h"

#if IREE_ALLOCATOR_ENABLE_MIALLOC

#include "iree/base/allocator.h"
#include "iree/base/assert.h"
#include "iree/base/attributes.h"
#include "iree/base/status.h"
#include "iree/base/tracing.h"

// Include the entire mimalloc impl statically.
#include "mimalloc/src/static.c"

static iree_status_t iree_allocator_mimalloc_malloc(
    const iree_allocator_alloc_params_t* params, void** inout_ptr) {
  IREE_ASSERT_ARGUMENT(params);
  IREE_ASSERT_ARGUMENT(inout_ptr);
  iree_host_size_t byte_length = params->byte_length;
  if (IREE_UNLIKELY(byte_length == 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "allocations must be >0 bytes");
  }

  IREE_TRACE_ZONE_BEGIN(z0);
  void* new_ptr = mi_malloc(byte_length);
  iree_status_t status = iree_ok_status();
  if (new_ptr) {
    *inout_ptr = new_ptr;
  } else {
    status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "system allocator failed the request");
  }
  IREE_TRACE_ALLOC(new_ptr, byte_length);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_allocator_mimalloc_realloc(
    const iree_allocator_alloc_params_t* params, void** inout_ptr) {
  IREE_ASSERT_ARGUMENT(params);
  // Note that this will only be called by the control function if *inout_ptr
  // is not NULL.
  iree_host_size_t byte_length = params->byte_length;
  if (IREE_UNLIKELY(byte_length == 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "allocations must be >0 bytes");
  }

  IREE_TRACE_ZONE_BEGIN(z0);
  void* existing_ptr = *inout_ptr;
  void* existing_ptr_value = iree_tracing_obscure_ptr(existing_ptr);
  (void)existing_ptr_value;
  void* new_ptr = mi_realloc(existing_ptr, byte_length);
  iree_status_t status = iree_ok_status();
  if (new_ptr) {
    *inout_ptr = new_ptr;
  } else {
    status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "system allocator failed the request");
  }
  IREE_TRACE_FREE(existing_ptr_value);
  IREE_TRACE_ALLOC(new_ptr, byte_length);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_allocator_mimalloc_calloc(
    const iree_allocator_alloc_params_t* params, void** inout_ptr) {
  IREE_ASSERT_ARGUMENT(params);
  iree_host_size_t byte_length = params->byte_length;
  if (IREE_UNLIKELY(byte_length == 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "allocations must be >0 bytes");
  }

  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, byte_length);
  void* new_ptr = mi_calloc(1, byte_length);
  iree_status_t status = iree_ok_status();
  if (new_ptr) {
    *inout_ptr = new_ptr;
  } else {
    status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "system allocator failed the request");
  }
  IREE_TRACE_ALLOC(new_ptr, byte_length);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_allocator_mimalloc_free(void** inout_ptr) {
  IREE_TRACE_ZONE_BEGIN(z0);
  void* ptr = *inout_ptr;
  if (IREE_LIKELY(ptr != NULL)) {
    IREE_TRACE_FREE(ptr);
    mi_free(ptr);
    *inout_ptr = NULL;
  }
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t
iree_allocator_mimalloc_ctl(void* self, iree_allocator_command_t command,
                            const void* params, void** inout_ptr) {
  IREE_ASSERT_ARGUMENT(inout_ptr);
  switch (command) {
    case IREE_ALLOCATOR_COMMAND_MALLOC:
      return iree_allocator_mimalloc_malloc(params, inout_ptr);
    case IREE_ALLOCATOR_COMMAND_CALLOC:
      return iree_allocator_mimalloc_calloc(params, inout_ptr);
    case IREE_ALLOCATOR_COMMAND_REALLOC:
      if (!*inout_ptr) {
        return iree_allocator_mimalloc_malloc(params, inout_ptr);
      } else {
        return iree_allocator_mimalloc_realloc(params, inout_ptr);
      }
    case IREE_ALLOCATOR_COMMAND_FREE:
      return iree_allocator_mimalloc_free(inout_ptr);
    default:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "unsupported system allocator command");
  }
}

#endif  // IREE_ALLOCATOR_ENABLE_MIALLOC
