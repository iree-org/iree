// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_MEMORY_TRACING_H_
#define IREE_HAL_MEMORY_TRACING_H_

#include "iree/base/api.h"

// Enables named-memory streams for every HAL pool reservation and backing-pool
// allocation. When disabled, only slab-provider backing allocations are traced.
// This keeps Tracy's memory UI focused on the highest-signal allocation events
// while preserving a one-line switch for detailed pool archaeology.
#ifndef IREE_HAL_MEMORY_TRACE_ENABLE_POOL_STREAMS
#define IREE_HAL_MEMORY_TRACE_ENABLE_POOL_STREAMS 0
#endif  // IREE_HAL_MEMORY_TRACE_ENABLE_POOL_STREAMS

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Persistent named-memory trace identity.
//
// Tracy requires the same stable name pointer for every alloc/free pair in a
// named memory stream. Static strings satisfy that naturally; dynamic per-pool
// names are copied into intentionally leaked process-lifetime storage in
// allocation-tracing builds.
typedef struct iree_hal_memory_trace_t {
#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_ALLOCATION_TRACKING
  // Stable name pointer passed to IREE_TRACE_ALLOC_NAMED/FREE_NAMED.
  const char* memory_id;
#else
  // Placeholder that keeps the type non-empty when tracing is compiled out.
  uint8_t reserved;
#endif  // IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_ALLOCATION_TRACKING
} iree_hal_memory_trace_t;

IREE_API_EXPORT iree_status_t iree_hal_memory_trace_initialize(
    iree_string_view_t trace_name, const char* default_memory_id,
    iree_allocator_t host_allocator, iree_hal_memory_trace_t* out_trace);

// Initializes a trace identity for a pool-owned stream. These streams are
// disabled unless IREE_HAL_MEMORY_TRACE_ENABLE_POOL_STREAMS is non-zero.
IREE_API_EXPORT iree_status_t iree_hal_memory_trace_initialize_pool(
    iree_string_view_t trace_name, const char* default_memory_id,
    iree_allocator_t host_allocator, iree_hal_memory_trace_t* out_trace);

static inline void iree_hal_memory_trace_deinitialize(
    iree_hal_memory_trace_t* trace) {
  (void)trace;
}

static inline void iree_hal_memory_trace_alloc(
    const iree_hal_memory_trace_t* trace, void* ptr,
    iree_device_size_t byte_length) {
#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_ALLOCATION_TRACKING
  if (ptr && trace->memory_id) {
    IREE_TRACE_ALLOC_NAMED(trace->memory_id, ptr,
                           (iree_host_size_t)byte_length);
  }
#else
  (void)trace;
  (void)ptr;
  (void)byte_length;
#endif  // IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_ALLOCATION_TRACKING
}

static inline void iree_hal_memory_trace_free(
    const iree_hal_memory_trace_t* trace, void* ptr) {
#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_ALLOCATION_TRACKING
  if (ptr && trace->memory_id) IREE_TRACE_FREE_NAMED(trace->memory_id, ptr);
#else
  (void)trace;
  (void)ptr;
#endif  // IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_ALLOCATION_TRACKING
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_MEMORY_TRACING_H_
