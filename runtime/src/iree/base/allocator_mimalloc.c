// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/api.h"

//===----------------------------------------------------------------------===//
// mimalloc dependency
//===----------------------------------------------------------------------===//

// Optionally include the entire library statically. The define should point to
// `"mimalloc/src/static.c"` in the source distribution. If not defined we
// assume mimalloc is being linked as a dynamic library or statically as part of
// the hosting framework/application.
#if defined(IREE_ALLOCATOR_MIMALLOC_STATIC_SRC)

// Since we're including mimalloc inline we will potentially get warnings that
// we enable that they do not. Ideally these would be fixed upstream as they are
// easy to fix and help readability (most are implicit type casts).
#if defined(IREE_COMPILER_MSVC)
#pragma warning(push)
#pragma warning(disable : 4024 4047 4133)
#endif  // IREE_COMPILER_MSVC

#include IREE_ALLOCATOR_MIMALLOC_STATIC_SRC

#if defined(IREE_COMPILER_MSVC)
#pragma warning(pop)
#endif  // IREE_COMPILER_MSVC

#else
#include <mimalloc.h>
#endif  // IREE_ALLOCATOR_MIMALLOC_STATIC_SRC

//===----------------------------------------------------------------------===//
// mimalloc allocator implementation
//===----------------------------------------------------------------------===//
// NOTE: currently we only use the default heap. Since the heap is just a void*
// we could treat our allocator `self` param as the heap instead to allow the
// user to specify a custom heap (via `iree_allocator_system_self`).

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
  void* new_ptr = mi_heap_malloc(mi_prim_get_default_heap(), byte_length);
  iree_status_t status = iree_ok_status();
  if (new_ptr) {
    *inout_ptr = new_ptr;
  } else {
    status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "mimalloc allocator failed the request");
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
  void* new_ptr =
      mi_heap_realloc(mi_prim_get_default_heap(), existing_ptr, byte_length);
  iree_status_t status = iree_ok_status();
  if (new_ptr) {
    *inout_ptr = new_ptr;
  } else {
    status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "mimalloc allocator failed the request");
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
  void* new_ptr = mi_heap_calloc(mi_prim_get_default_heap(), 1, byte_length);
  iree_status_t status = iree_ok_status();
  if (new_ptr) {
    *inout_ptr = new_ptr;
  } else {
    status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "mimalloc allocator failed the request");
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
    mi_heap_free(mi_prim_get_default_heap(), ptr);
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
                              "unsupported mimalloc allocator command");
  }
}
