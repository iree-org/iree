// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/cuda2/tracing.h"

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION

#include "iree/hal/drivers/cuda2/cuda_dynamic_symbols.h"
#include "iree/hal/drivers/cuda2/cuda_status_util.h"

// Total number of events per tracing context. This translates to the maximum
// number of outstanding timestamp queries before collection is required.
// To prevent spilling pages we leave some room for the context structure.
#define IREE_HAL_CUDA_TRACING_DEFAULT_QUERY_CAPACITY (16 * 1024 - 256)

struct iree_hal_cuda2_tracing_context_t {
  const iree_hal_cuda2_dynamic_symbols_t* symbols;

  CUstream stream;
  iree_arena_block_pool_t* block_pool;
  iree_allocator_t host_allocator;

  // A unique GPU zone ID allocated from Tracy.
  // There is a global limit of 255 GPU zones (ID 255 is special).
  uint8_t id;

  // Base event used for computing relative times for all recorded events.
  // This is required as CUDA (without CUPTI) only allows for relative timing
  // between events and we need a stable base event.
  CUevent base_event;

  // Indices into |event_pool| defining a ringbuffer.
  uint32_t query_head;
  uint32_t query_tail;
  uint32_t query_capacity;

  // Event pool reused to capture tracing timestamps.
  CUevent event_pool[IREE_HAL_CUDA_TRACING_DEFAULT_QUERY_CAPACITY];
};

static iree_status_t iree_hal_cuda2_tracing_context_initial_calibration(
    const iree_hal_cuda2_dynamic_symbols_t* symbols, CUstream stream,
    CUevent base_event, int64_t* out_cpu_timestamp, int64_t* out_gpu_timestamp,
    float* out_timestamp_period) {
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_cpu_timestamp = 0;
  *out_gpu_timestamp = 0;
  *out_timestamp_period = 1.0f;

  // Record event to the stream; in the absence of a synchronize this may not
  // flush immediately.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, IREE_CURESULT_TO_STATUS(symbols, cuEventRecord(base_event, stream)));

  // Force flush the event and wait for it to complete.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, IREE_CURESULT_TO_STATUS(symbols, cuEventSynchronize(base_event)));

  // Track when we know the event has completed and has a reasonable timestamp.
  // This may drift from the actual time differential between host/device but is
  // (maybe?) the best we can do without CUPTI.
  *out_cpu_timestamp = iree_tracing_time();

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_cuda2_tracing_context_allocate(
    const iree_hal_cuda2_dynamic_symbols_t* symbols,
    iree_string_view_t queue_name, CUstream stream,
    iree_arena_block_pool_t* block_pool, iree_allocator_t host_allocator,
    iree_hal_cuda2_tracing_context_t** out_context) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(symbols);
  IREE_ASSERT_ARGUMENT(stream);
  IREE_ASSERT_ARGUMENT(block_pool);
  IREE_ASSERT_ARGUMENT(out_context);
  *out_context = NULL;

  iree_hal_cuda2_tracing_context_t* context = NULL;
  iree_status_t status =
      iree_allocator_malloc(host_allocator, sizeof(*context), (void**)&context);
  if (iree_status_is_ok(status)) {
    context->symbols = symbols;
    context->stream = stream;
    context->block_pool = block_pool;
    context->host_allocator = host_allocator;
    context->query_capacity = IREE_ARRAYSIZE(context->event_pool);
  }

  // Pre-allocate all events in the event pool.
  if (iree_status_is_ok(status)) {
    IREE_TRACE_ZONE_BEGIN_NAMED(
        z_event_pool, "iree_hal_cuda2_tracing_context_allocate_event_pool");
    IREE_TRACE_ZONE_APPEND_VALUE_I64(z_event_pool,
                                     (int64_t)context->query_capacity);
    for (iree_host_size_t i = 0; i < context->query_capacity; ++i) {
      status = IREE_CURESULT_TO_STATUS(
          symbols, cuEventCreate(&context->event_pool[i], CU_EVENT_DEFAULT));
      if (!iree_status_is_ok(status)) break;
    }
    IREE_TRACE_ZONE_END(z_event_pool);
  }

  // Create the initial GPU event and insert it into the stream.
  // All events we record are relative to this event.
  int64_t cpu_timestamp = 0;
  int64_t gpu_timestamp = 0;
  float timestamp_period = 0.0f;
  if (iree_status_is_ok(status)) {
    status = IREE_CURESULT_TO_STATUS(
        symbols, cuEventCreate(&context->base_event, CU_EVENT_DEFAULT));
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_cuda2_tracing_context_initial_calibration(
        symbols, stream, context->base_event, &cpu_timestamp, &gpu_timestamp,
        &timestamp_period);
  }

  // Allocate the GPU context and pass initial calibration data.
  if (iree_status_is_ok(status)) {
    context->id = iree_tracing_gpu_context_allocate(
        IREE_TRACING_GPU_CONTEXT_TYPE_VULKAN, queue_name.data, queue_name.size,
        /*is_calibrated=*/false, cpu_timestamp, gpu_timestamp,
        timestamp_period);
  }

  if (iree_status_is_ok(status)) {
    *out_context = context;
  } else {
    iree_hal_cuda2_tracing_context_free(context);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_cuda2_tracing_context_free(
    iree_hal_cuda2_tracing_context_t* context) {
  if (!context) return;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Always perform a collection on shutdown.
  iree_hal_cuda2_tracing_context_collect(context);

  // Release all events; since collection completed they should all be unused.
  IREE_TRACE_ZONE_BEGIN_NAMED(z_event_pool,
                              "iree_hal_cuda2_tracing_context_free_event_pool");
  for (iree_host_size_t i = 0; i < context->query_capacity; ++i) {
    if (context->event_pool[i]) {
      IREE_CUDA_IGNORE_ERROR(context->symbols,
                             cuEventDestroy(context->event_pool[i]));
    }
  }
  IREE_TRACE_ZONE_END(z_event_pool);
  if (context->base_event) {
    IREE_CUDA_IGNORE_ERROR(context->symbols,
                           cuEventDestroy(context->base_event));
  }

  iree_allocator_t host_allocator = context->host_allocator;
  iree_allocator_free(host_allocator, context);

  IREE_TRACE_ZONE_END(z0);
}

void iree_hal_cuda2_tracing_context_collect(
    iree_hal_cuda2_tracing_context_t* context) {
  if (!context) return;
  if (context->query_tail == context->query_head) {
    // No outstanding queries.
    return;
  }
  IREE_TRACE_ZONE_BEGIN(z0);

  while (context->query_tail != context->query_head) {
    // Compute the contiguous range of queries ready to be read.
    // If the ringbuffer wraps around we'll handle that in the next loop.
    uint32_t try_query_count =
        context->query_head < context->query_tail
            ? context->query_capacity - context->query_tail
            : context->query_head - context->query_tail;
    IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)try_query_count);

    // Scan and feed the times to tracy, stopping when we hit the first
    // unavailable query.
    uint32_t query_base = context->query_tail;
    uint32_t read_query_count = 0;
    for (uint32_t i = 0; i < try_query_count; ++i) {
      // Ensure the event has completed; will return CUDA_ERROR_NOT_READY if
      // recorded but not retired or any other deferred error.
      uint16_t query_id = (uint16_t)(query_base + i);
      CUevent query_event = context->event_pool[query_id];
      CUresult result = context->symbols->cuEventQuery(query_event);
      if (result != CUDA_SUCCESS) break;

      // Calculate context-relative time and notify tracy.
      float relative_millis = 0.0f;
      IREE_CUDA_IGNORE_ERROR(
          context->symbols,
          cuEventElapsedTime(&relative_millis, context->base_event,
                             query_event));
      int64_t gpu_timestamp = (int64_t)((double)relative_millis * 1000000.0);
      iree_tracing_gpu_zone_notify(context->id, query_id, gpu_timestamp);

      read_query_count = i + 1;
    }
    IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)read_query_count);

    context->query_tail += read_query_count;
    if (context->query_tail >= context->query_capacity) {
      context->query_tail = 0;
    }
  }

  IREE_TRACE_ZONE_END(z0);
}

static uint16_t iree_hal_cuda2_tracing_context_insert_query(
    iree_hal_cuda2_tracing_context_t* context, CUstream stream) {
  // Allocate an event from the pool for use by the query.
  uint32_t query_id = context->query_head;
  context->query_head = (context->query_head + 1) % context->query_capacity;

  // TODO: check to see if the read and write heads of the ringbuffer have
  // overlapped. If they have we could try to collect but it's not guaranteed
  // that collection will complete (e.g. we may be reserving events for use in
  // graphs that haven't yet been launched).
  //
  // For now we just allow the overlap and tracing results will be inconsistent.
  IREE_ASSERT_NE(context->query_head, context->query_tail);

  CUevent event = context->event_pool[query_id];
  IREE_CUDA_IGNORE_ERROR(context->symbols, cuEventRecord(event, stream));

  return query_id;
}

// TODO: optimize this implementation to reduce the number of events required:
// today we insert 2 events per zone (one for begin and one for end) but in
// many cases we could reduce this by inserting events only between zones and
// using the differences between them.

void iree_hal_cuda2_tracing_zone_begin_impl(
    iree_hal_cuda2_tracing_context_t* context, CUstream stream,
    const iree_tracing_location_t* src_loc) {
  if (!context) return;
  uint16_t query_id =
      iree_hal_cuda2_tracing_context_insert_query(context, stream);
  iree_tracing_gpu_zone_begin(context->id, query_id, src_loc);
}

void iree_hal_cuda2_tracing_zone_begin_external_impl(
    iree_hal_cuda2_tracing_context_t* context, CUstream stream,
    const char* file_name, size_t file_name_length, uint32_t line,
    const char* function_name, size_t function_name_length, const char* name,
    size_t name_length) {
  if (!context) return;
  uint16_t query_id =
      iree_hal_cuda2_tracing_context_insert_query(context, stream);
  iree_tracing_gpu_zone_begin_external(context->id, query_id, file_name,
                                       file_name_length, line, function_name,
                                       function_name_length, name, name_length);
}

void iree_hal_cuda2_tracing_zone_end_impl(
    iree_hal_cuda2_tracing_context_t* context, CUstream stream) {
  if (!context) return;
  uint16_t query_id =
      iree_hal_cuda2_tracing_context_insert_query(context, stream);
  iree_tracing_gpu_zone_end(context->id, query_id);
}

#else

iree_status_t iree_hal_cuda2_tracing_context_allocate(
    const iree_hal_cuda2_dynamic_symbols_t* symbols,
    iree_string_view_t queue_name, CUstream stream,
    iree_arena_block_pool_t* block_pool, iree_allocator_t host_allocator,
    iree_hal_cuda2_tracing_context_t** out_context) {
  *out_context = NULL;
  return iree_ok_status();
}

void iree_hal_cuda2_tracing_context_free(
    iree_hal_cuda2_tracing_context_t* context) {}

void iree_hal_cuda2_tracing_context_collect(
    iree_hal_cuda2_tracing_context_t* context) {}

#endif  // IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION
