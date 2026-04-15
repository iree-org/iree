// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/memory/tracing.h"

#include <string.h>

#include "iree/base/internal/debugging.h"

static iree_status_t iree_hal_memory_trace_initialize_impl(
    iree_string_view_t trace_name, const char* default_memory_id, bool enabled,
    iree_allocator_t host_allocator, iree_hal_memory_trace_t* out_trace) {
  IREE_ASSERT_ARGUMENT(out_trace);

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_ALLOCATION_TRACKING
  out_trace->memory_id = NULL;
  if (!enabled) return iree_ok_status();
  IREE_ASSERT_ARGUMENT(default_memory_id);
  out_trace->memory_id = default_memory_id;
  if (iree_string_view_is_empty(trace_name)) return iree_ok_status();
  if (IREE_UNLIKELY(trace_name.size == IREE_HOST_SIZE_MAX)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "trace memory identifier is too large");
  }

  char* trace_name_storage = NULL;
  IREE_LEAK_CHECK_DISABLE_PUSH();
  iree_status_t status = iree_allocator_malloc(
      host_allocator, trace_name.size + 1, (void**)&trace_name_storage);
  IREE_LEAK_CHECK_DISABLE_POP();
  if (iree_status_is_ok(status)) {
    memcpy(trace_name_storage, trace_name.data, trace_name.size);
    trace_name_storage[trace_name.size] = '\0';
    out_trace->memory_id = trace_name_storage;
  }
  return status;
#else
  (void)trace_name;
  (void)default_memory_id;
  (void)enabled;
  (void)host_allocator;
  out_trace->reserved = 0;
  return iree_ok_status();
#endif  // IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_ALLOCATION_TRACKING
}

IREE_API_EXPORT iree_status_t iree_hal_memory_trace_initialize(
    iree_string_view_t trace_name, const char* default_memory_id,
    iree_allocator_t host_allocator, iree_hal_memory_trace_t* out_trace) {
  return iree_hal_memory_trace_initialize_impl(trace_name, default_memory_id,
                                               /*enabled=*/true, host_allocator,
                                               out_trace);
}

IREE_API_EXPORT iree_status_t iree_hal_memory_trace_initialize_pool(
    iree_string_view_t trace_name, const char* default_memory_id,
    iree_allocator_t host_allocator, iree_hal_memory_trace_t* out_trace) {
  return iree_hal_memory_trace_initialize_impl(
      trace_name, default_memory_id,
      /*enabled=*/IREE_HAL_MEMORY_TRACE_ENABLE_POOL_STREAMS != 0,
      host_allocator, out_trace);
}
