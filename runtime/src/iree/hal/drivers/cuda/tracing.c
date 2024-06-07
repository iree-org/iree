// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/cuda/tracing.h"

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION_DEVICE

#include "iree/hal/drivers/cuda/cuda_dynamic_symbols.h"
#include "iree/hal/drivers/cuda/cuda_status_util.h"

// Total number of events per tracing context. This translates to the maximum
// number of outstanding timestamp queries before collection is required.
// To prevent spilling pages we leave some room for the context structure.
#define IREE_HAL_CUDA_TRACING_DEFAULT_QUERY_CAPACITY (16 * 1024 - 256)

struct iree_hal_cuda_tracing_context_event_t {
  CUevent event;
  iree_hal_cuda_tracing_context_event_t* next_in_cb;
  iree_hal_cuda_tracing_context_event_t* next_submission;
};

struct iree_hal_cuda_tracing_context_t {
  const iree_hal_cuda_dynamic_symbols_t* symbols;

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

  // Unallocated events
  iree_hal_cuda_tracing_context_event_t* event_freelist_head;

  // Submitted events
  iree_hal_cuda_tracing_context_event_t* event_submitted_list_head;
  iree_hal_cuda_tracing_context_event_t* event_submitted_list_tail;

  uint32_t query_capacity;

  // Event pool reused to capture tracing timestamps.
  iree_hal_cuda_tracing_context_event_t
      event_pool[IREE_HAL_CUDA_TRACING_DEFAULT_QUERY_CAPACITY];
};

static iree_status_t iree_hal_cuda_tracing_context_initial_calibration(
    const iree_hal_cuda_dynamic_symbols_t* symbols, CUstream stream,
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

iree_status_t iree_hal_cuda_tracing_context_allocate(
    const iree_hal_cuda_dynamic_symbols_t* symbols,
    iree_string_view_t queue_name, CUstream stream,
    iree_arena_block_pool_t* block_pool, iree_allocator_t host_allocator,
    iree_hal_cuda_tracing_context_t** out_context) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(symbols);
  IREE_ASSERT_ARGUMENT(stream);
  IREE_ASSERT_ARGUMENT(block_pool);
  IREE_ASSERT_ARGUMENT(out_context);
  *out_context = NULL;

  iree_hal_cuda_tracing_context_t* context = NULL;
  iree_status_t status =
      iree_allocator_malloc(host_allocator, sizeof(*context), (void**)&context);
  if (iree_status_is_ok(status)) {
    context->symbols = symbols;
    context->stream = stream;
    context->block_pool = block_pool;
    context->host_allocator = host_allocator;
    context->query_capacity = IREE_ARRAYSIZE(context->event_pool);
    context->event_submitted_list_head = NULL;
    context->event_submitted_list_tail = NULL;
  }

  // Pre-allocate all events in the event pool.
  if (iree_status_is_ok(status)) {
    IREE_TRACE_ZONE_BEGIN_NAMED(
        z_event_pool, "iree_hal_cuda_tracing_context_allocate_event_pool");
    IREE_TRACE_ZONE_APPEND_VALUE_I64(z_event_pool,
                                     (int64_t)context->query_capacity);
    context->event_freelist_head = &context->event_pool[0];
    for (iree_host_size_t i = 0; i < context->query_capacity; ++i) {
      status = IREE_CURESULT_TO_STATUS(
          symbols,
          cuEventCreate(&context->event_pool[i].event, CU_EVENT_DEFAULT));
      if (!iree_status_is_ok(status)) break;
      if (i > 0) {
        context->event_pool[i - 1].next_in_cb = &context->event_pool[i];
      }
      context->event_pool[i].next_submission = NULL;
      if (i + 1 == context->query_capacity) {
        context->event_pool[i].next_in_cb = NULL;
      }
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
    status = iree_hal_cuda_tracing_context_initial_calibration(
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
    iree_hal_cuda_tracing_context_free(context);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_cuda_tracing_context_free(
    iree_hal_cuda_tracing_context_t* context) {
  if (!context) return;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Always perform a collection on shutdown.
  iree_hal_cuda_tracing_context_collect(context);

  // Release all events; since collection completed they should all be unused.
  IREE_TRACE_ZONE_BEGIN_NAMED(z_event_pool,
                              "iree_hal_cuda_tracing_context_free_event_pool");
  for (iree_host_size_t i = 0; i < context->query_capacity; ++i) {
    if (context->event_pool[i].event) {
      IREE_CUDA_IGNORE_ERROR(context->symbols,
                             cuEventDestroy(context->event_pool[i].event));
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

void iree_hal_cuda_tracing_context_collect(
    iree_hal_cuda_tracing_context_t* context) {
  if (!context) return;
  // No outstanding queries
  if (!context->event_submitted_list_head) {
    return;
  }
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_cuda_tracing_context_event_t* events =
      context->event_submitted_list_head;
  uint32_t read_query_count = 0;
  while (events) {
    iree_hal_cuda_tracing_context_event_t* event = events;
    while (event) {
      uint32_t query_id = (uint32_t)(event - &context->event_pool[0]);

      CUresult result = context->symbols->cuEventQuery(event->event);
      while (result == CUDA_ERROR_NOT_READY) {
        result = context->symbols->cuEventQuery(event->event);
      }
      if (result != CUDA_SUCCESS) break;

      // Calculate context-relative time and notify tracy.
      float relative_millis = 0.0f;
      IREE_CUDA_IGNORE_ERROR(
          context->symbols,
          cuEventElapsedTime(&relative_millis, context->base_event,
                             event->event));
      int64_t gpu_timestamp = (int64_t)((double)relative_millis * 1000000.0);
      iree_tracing_gpu_zone_notify(context->id, query_id, gpu_timestamp);

      read_query_count += 1;
      event = event->next_in_cb;
    }
    iree_hal_cuda_tracing_context_event_t* next = events->next_submission;
    events->next_submission = events;
    events = next;
    context->event_submitted_list_head = events;
  }
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)read_query_count);
  IREE_TRACE_ZONE_END(z0);
}

void iree_hal_cuda_tracing_notify_submitted(
    iree_hal_cuda_tracing_context_t* context,
    iree_hal_cuda_tracing_context_event_t** event_list_begin,
    iree_hal_cuda_tracing_context_event_t** event_list_end) {
  if (!context) return;

  IREE_ASSERT_ARGUMENT(event_list_begin);
  IREE_ASSERT_ARGUMENT(event_list_end);

  if (!*event_list_begin) {
    return;
  }

  iree_hal_cuda_tracing_context_event_t* evt = *event_list_begin;
  while (evt) {
    evt = evt->next_in_cb;
  }

  if (!context->event_submitted_list_head) {
    context->event_submitted_list_head = *event_list_begin;
    context->event_submitted_list_tail = *event_list_begin;
    return;
  }

  context->event_submitted_list_tail->next_submission = *event_list_begin;
  context->event_submitted_list_tail = *event_list_begin;
}

void iree_hal_cuda_tracing_free(
    iree_hal_cuda_tracing_context_t* context,
    iree_hal_cuda_tracing_context_event_t** event_list_begin,
    iree_hal_cuda_tracing_context_event_t** event_list_end) {
  if (!context) return;

  IREE_ASSERT_ARGUMENT(event_list_begin);
  IREE_ASSERT_ARGUMENT(event_list_end);

  if (!*event_list_begin) {
    return;
  }

  // If this event list has never been submitted,
  // we still need to add values to the timeline,
  // otherwise tracy will not behave correctly.
  if ((!(*event_list_begin)->next_submission)) {
    iree_hal_cuda_tracing_context_event_t* event = *event_list_begin;
    while (event) {
      uint32_t query_id = (uint32_t)(event - &context->event_pool[0]);
      iree_tracing_gpu_zone_notify(context->id, query_id, 0);
      event = event->next_in_cb;
    }
  }

  if (!context->event_freelist_head) {
    context->event_freelist_head = *event_list_begin;
    return;
  }
  (*event_list_begin)->next_submission = NULL;
  (*event_list_end)->next_in_cb = context->event_freelist_head;
  context->event_freelist_head = *event_list_begin;

  *event_list_begin = NULL;
  *event_list_end = NULL;
}

static uint16_t iree_hal_cuda_stream_tracing_context_insert_query(
    iree_hal_cuda_tracing_context_t* context,
    iree_hal_cuda_tracing_context_event_t** event_list_begin,
    iree_hal_cuda_tracing_context_event_t** event_list_end, CUstream stream) {
  IREE_ASSERT_ARGUMENT(event_list_begin);
  IREE_ASSERT_ARGUMENT(event_list_end);
  // Allocate an event from the pool for use by the query.
  // TODO: If we have run out of our freelist, then we
  //   need to try and recover allocate events.
  iree_hal_cuda_tracing_context_event_t* event = context->event_freelist_head;
  context->event_freelist_head = event->next_in_cb;
  uint32_t query_id = event - &context->event_pool[0];
  IREE_ASSERT(event->next_in_cb != NULL);
  event->next_in_cb = NULL;

  IREE_CUDA_IGNORE_ERROR(context->symbols, cuEventRecord(event->event, stream));

  if (!*event_list_begin) {
    *event_list_begin = event;
    *event_list_end = event;
  } else {
    (*event_list_end)->next_in_cb = event;
    *event_list_end = event;
  }

  return query_id;
}

static uint16_t iree_hal_cuda_graph_tracing_context_insert_query(
    iree_hal_cuda_tracing_context_t* context,
    iree_hal_cuda_tracing_context_event_t** event_list_begin,
    iree_hal_cuda_tracing_context_event_t** event_list_end,
    CUgraphNode* out_node, CUgraph graph, CUgraphNode* dependency_nodes,
    size_t dependency_nodes_count) {
  IREE_ASSERT_ARGUMENT(event_list_begin);
  IREE_ASSERT_ARGUMENT(event_list_end);
  // Allocate an event from the pool for use by the query.
  // TODO: If we have run out of our freelist, then we
  //   need to try and recover or allocate more
  //   events.
  iree_hal_cuda_tracing_context_event_t* event = context->event_freelist_head;
  context->event_freelist_head = event->next_in_cb;
  uint32_t query_id = event - &context->event_pool[0];
  IREE_ASSERT(event->next_in_cb != NULL);
  event->next_in_cb = NULL;

  iree_status_t status = IREE_CURESULT_TO_STATUS(
      context->symbols,
      cuGraphAddEventRecordNode(out_node, graph, dependency_nodes,
                                dependency_nodes_count, event->event));
  IREE_ASSERT(iree_status_is_ok(status));

  if (!*event_list_begin) {
    *event_list_begin = event;
    *event_list_end = event;
  } else {
    (*event_list_end)->next_in_cb = event;
    *event_list_end = event;
  }
  return query_id;
}

// TODO: optimize this implementation to reduce the number of events required:
// today we insert 2 events per zone (one for begin and one for end) but in
// many cases we could reduce this by inserting events only between zones and
// using the differences between them.

void iree_hal_cuda_stream_tracing_zone_begin_impl(
    iree_hal_cuda_tracing_context_t* context,
    iree_hal_cuda_tracing_context_event_t** event_list_begin,
    iree_hal_cuda_tracing_context_event_t** event_list_end, CUstream stream,
    const iree_tracing_location_t* src_loc) {
  IREE_ASSERT_ARGUMENT(context);
  uint16_t query_id = iree_hal_cuda_stream_tracing_context_insert_query(
      context, event_list_begin, event_list_end, stream);
  iree_tracing_gpu_zone_begin(context->id, query_id, src_loc);
}

void iree_hal_cuda_stream_tracing_zone_begin_external_impl(
    iree_hal_cuda_tracing_context_t* context,
    iree_hal_cuda_tracing_context_event_t** event_list_begin,
    iree_hal_cuda_tracing_context_event_t** event_list_end, CUstream stream,
    const char* file_name, size_t file_name_length, uint32_t line,
    const char* function_name, size_t function_name_length, const char* name,
    size_t name_length) {
  IREE_ASSERT_ARGUMENT(context);
  uint16_t query_id = iree_hal_cuda_stream_tracing_context_insert_query(
      context, event_list_begin, event_list_end, stream);
  iree_tracing_gpu_zone_begin_external(context->id, query_id, file_name,
                                       file_name_length, line, function_name,
                                       function_name_length, name, name_length);
}

void iree_hal_cuda_graph_tracing_zone_begin_external_impl(
    iree_hal_cuda_tracing_context_t* context,
    iree_hal_cuda_tracing_context_event_t** event_list_begin,
    iree_hal_cuda_tracing_context_event_t** event_list_end,
    CUgraphNode* out_node, CUgraph graph, CUgraphNode* dependency_nodes,
    size_t dependency_nodes_count, const char* file_name,
    size_t file_name_length, uint32_t line, const char* function_name,
    size_t function_name_length, const char* name, size_t name_length) {
  if (!context) return;
  uint16_t query_id = iree_hal_cuda_graph_tracing_context_insert_query(
      context, event_list_begin, event_list_end, out_node, graph,
      dependency_nodes, dependency_nodes_count);
  iree_tracing_gpu_zone_begin_external(context->id, query_id, file_name,
                                       file_name_length, line, function_name,
                                       function_name_length, name, name_length);
}

void iree_hal_cuda_stream_tracing_zone_end_impl(
    iree_hal_cuda_tracing_context_t* context,
    iree_hal_cuda_tracing_context_event_t** event_list_begin,
    iree_hal_cuda_tracing_context_event_t** event_list_end, CUstream stream) {
  if (!context) return;
  uint16_t query_id = iree_hal_cuda_stream_tracing_context_insert_query(
      context, event_list_begin, event_list_end, stream);
  iree_tracing_gpu_zone_end(context->id, query_id);
}

void iree_hal_cuda_graph_tracing_zone_end_impl(
    iree_hal_cuda_tracing_context_t* context,
    iree_hal_cuda_tracing_context_event_t** event_list_begin,
    iree_hal_cuda_tracing_context_event_t** event_list_end,
    CUgraphNode* out_node, CUgraph graph, CUgraphNode* dependency_nodes,
    size_t dependency_nodes_count) {
  if (!context) return;
  uint16_t query_id = iree_hal_cuda_graph_tracing_context_insert_query(
      context, event_list_begin, event_list_end, out_node, graph,
      dependency_nodes, dependency_nodes_count);
  iree_tracing_gpu_zone_end(context->id, query_id);
}

#else

iree_status_t iree_hal_cuda_tracing_context_allocate(
    const iree_hal_cuda_dynamic_symbols_t* symbols,
    iree_string_view_t queue_name, CUstream stream,
    iree_arena_block_pool_t* block_pool, iree_allocator_t host_allocator,
    iree_hal_cuda_tracing_context_t** out_context) {
  *out_context = NULL;
  return iree_ok_status();
}

void iree_hal_cuda_tracing_context_free(
    iree_hal_cuda_tracing_context_t* context) {}

void iree_hal_cuda_tracing_context_collect(
    iree_hal_cuda_tracing_context_t* context) {}

void iree_hal_cuda_tracing_notify_submitted(
    iree_hal_cuda_tracing_context_t* context,
    iree_hal_cuda_tracing_context_event_t** event_list_begin,
    iree_hal_cuda_tracing_context_event_t** event_list_end) {}

void iree_hal_cuda_tracing_free(
    iree_hal_cuda_tracing_context_t* context,
    iree_hal_cuda_tracing_context_event_t** event_list_begin,
    iree_hal_cuda_tracing_context_event_t** event_list_end) {}

#endif  // IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION_DEVICE
