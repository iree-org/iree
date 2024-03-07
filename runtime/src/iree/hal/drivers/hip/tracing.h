// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_HIP_TRACING_H_
#define IREE_EXPERIMENTAL_HIP_TRACING_H_

#include "experimental/hip/dynamic_symbols.h"
#include "experimental/hip/hip_headers.h"
#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/base/tracing.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Per-stream HIP tracing context.
// No-op if IREE tracing is not enabled.
//
// Use the IREE_HIP_TRACE_* macros to trace a contiguous set of stream
// operations. Unlike the normal tracy macros there are no zone IDs and instead
// each stream gets an ID allocated once and passed to all tracing macros.
//
// Usage:
//   IREE_HIP_TRACE_ZONE_BEGIN(queue->tracing_context, stream);
//   hipModuleLaunchKernel(..., stream);
//   IREE_HIP_TRACE_ZONE_END(queue->tracing_context, stream);
//   ...
//   iree_hal_hip_tracing_context_collect(queue->tracing_context);
//
// NOTE: timestamps can have non-trivial side-effecting behavior and may
// introduce serialization in graph execution.
//
// TODO(benvanik): expose hipEvent_t reservation separate from recording. For
// graphs we will need to insert the events but in order to reuse the graphs
// we'll need to reserve and patch new events each graph launch. For now we
// don't instrument graphs.
//
// Thread-compatible: external synchronization is required if using from
// multiple threads (same as with hipStream_t itself).
typedef struct iree_hal_hip_tracing_context_t iree_hal_hip_tracing_context_t;

// Allocates a tracing context for the given HIP |stream|.
// Each context must only be used with the stream it was created for.
iree_status_t iree_hal_hip_tracing_context_allocate(
    const iree_hal_hip_dynamic_symbols_t* symbols,
    iree_string_view_t queue_name, hipStream_t stream,
    iree_arena_block_pool_t* block_pool, iree_allocator_t host_allocator,
    iree_hal_hip_tracing_context_t** out_context);

// Frees a tracing context and all associated HIP resources.
// All submissions using the resources must be completed prior to calling.
void iree_hal_hip_tracing_context_free(iree_hal_hip_tracing_context_t* context);

// Collects in-flight timestamp queries from the stream and feeds them to tracy.
// Must be called frequently (every submission, etc) to drain the backlog;
// tracing may start failing if the internal ringbuffer is exceeded.
void iree_hal_hip_tracing_context_collect(
    iree_hal_hip_tracing_context_t* context);

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION

// Begins a normal zone derived on the calling |src_loc|.
// Must be perfectly nested and paired with a corresponding zone end.
void iree_hal_hip_tracing_zone_begin_impl(
    iree_hal_hip_tracing_context_t* context, hipStream_t stream,
    const iree_tracing_location_t* src_loc);

// Begins an external zone using the given source information.
// The provided strings will be copied into the tracy buffer.
void iree_hal_hip_tracing_zone_begin_external_impl(
    iree_hal_hip_tracing_context_t* context, hipStream_t stream,
    const char* file_name, size_t file_name_length, uint32_t line,
    const char* function_name, size_t function_name_length, const char* name,
    size_t name_length);

void iree_hal_hip_tracing_zone_end_impl(iree_hal_hip_tracing_context_t* context,
                                        hipStream_t stream);

// Begins a new zone with the parent function name.
#define IREE_HIP_TRACE_ZONE_BEGIN(context, stream)                            \
  static const iree_tracing_location_t TracyConcat(                           \
      __tracy_source_location, __LINE__) = {name_literal, __FUNCTION__,       \
                                            __FILE__, (uint32_t)__LINE__, 0}; \
  iree_hal_hip_tracing_zone_begin_impl(                                       \
      context, stream, &TracyConcat(__tracy_source_location, __LINE__));

// Begins an externally defined zone with a dynamic source location.
// The |file_name|, |function_name|, and optional |name| strings will be copied
// into the trace buffer and do not need to persist.
#define IREE_HIP_TRACE_ZONE_BEGIN_EXTERNAL(                              \
    context, stream, file_name, file_name_length, line, function_name,   \
    function_name_length, name, name_length)                             \
  iree_hal_hip_tracing_zone_begin_external_impl(                         \
      context, stream, file_name, file_name_length, line, function_name, \
      function_name_length, name, name_length)

// Ends the current zone. Must be passed the |zone_id| from the _BEGIN.
#define IREE_HIP_TRACE_ZONE_END(context, stream) \
  iree_hal_hip_tracing_zone_end_impl(context, stream)

#else

#define IREE_HIP_TRACE_ZONE_BEGIN(context, stream)
#define IREE_HIP_TRACE_ZONE_BEGIN_EXTERNAL(                            \
    context, stream, file_name, file_name_length, line, function_name, \
    function_name_length, name, name_length)
#define IREE_HIP_TRACE_ZONE_END(context, stream)

#endif  // IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_EXPERIMENTAL_HIP_TRACING_H_
