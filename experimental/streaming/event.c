// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/streaming/internal.h"

//===----------------------------------------------------------------------===//
// Event management
//===----------------------------------------------------------------------===//

static void iree_hal_streaming_event_destroy(iree_hal_streaming_event_t* event);

iree_status_t iree_hal_streaming_event_create(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_event_flags_t flags, iree_allocator_t host_allocator,
    iree_hal_streaming_event_t** out_event) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_event);
  *out_event = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_streaming_event_t* event = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator, sizeof(*event), (void**)&event));

  // Initialize event.
  iree_atomic_ref_count_init(&event->ref_count);
  event->flags = flags;
  event->signal_value = 0;
  event->recording_stream = NULL;
  event->context = context;
  iree_hal_streaming_context_retain(context);
  event->record_time_ns = 0;
  event->ipc_handle = NULL;
  event->semaphore = NULL;
  event->host_allocator = host_allocator;

  // Create HAL semaphore for synchronization.
  iree_status_t status = iree_hal_semaphore_create(
      context->device, 0ULL, IREE_HAL_SEMAPHORE_FLAG_NONE, &event->semaphore);

  if (iree_status_is_ok(status)) {
    *out_event = event;
  } else {
    iree_hal_streaming_event_destroy(event);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_streaming_event_destroy(
    iree_hal_streaming_event_t* event) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Release semaphore.
  if (event->semaphore) {
    iree_hal_semaphore_release(event->semaphore);
  }

  // Release recording stream reference.
  if (event->recording_stream) {
    iree_hal_streaming_stream_release(event->recording_stream);
  }

  // Release context.
  iree_hal_streaming_context_release(event->context);

  // Free event memory.
  iree_allocator_t host_allocator = event->host_allocator;
  iree_allocator_free(host_allocator, event);

  IREE_TRACE_ZONE_END(z0);
}

void iree_hal_streaming_event_retain(iree_hal_streaming_event_t* event) {
  if (event) {
    iree_atomic_ref_count_inc(&event->ref_count);
  }
}

void iree_hal_streaming_event_release(iree_hal_streaming_event_t* event) {
  if (event && iree_atomic_ref_count_dec(&event->ref_count) == 1) {
    iree_hal_streaming_event_destroy(event);
  }
}

iree_status_t iree_hal_streaming_event_query(iree_hal_streaming_event_t* event,
                                             int* status) {
  IREE_ASSERT_ARGUMENT(event);
  IREE_ASSERT_ARGUMENT(status);
  IREE_TRACE_ZONE_BEGIN(z0);

  uint64_t current_value = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_semaphore_query(event->semaphore, &current_value));

  *status = (current_value >= event->signal_value)
                ? 0
                : 1;  // 0=complete, 1=not complete

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_event_record(
    iree_hal_streaming_event_t* event, iree_hal_streaming_stream_t* stream) {
  IREE_ASSERT_ARGUMENT(event);
  IREE_ASSERT_ARGUMENT(stream);
  IREE_TRACE_ZONE_BEGIN(z0);

  event->record_time_ns = iree_time_now();

  // Set recording stream.
  if (event->recording_stream != stream) {
    if (event->recording_stream) {
      iree_hal_streaming_stream_release(event->recording_stream);
    }
    event->recording_stream = stream;
    iree_hal_streaming_stream_retain(stream);
  }

  // Flush the stream to ensure all prior operations are submitted.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0,
                                    iree_hal_streaming_stream_flush(stream));

  // Use stream's current pending value as wait value and increment for signal.
  uint64_t wait_value = stream->pending_value;
  event->signal_value = wait_value + 1;
  stream->pending_value = event->signal_value;

  // Create a queue barrier to signal the event semaphore.
  // This waits for the stream's last submission to complete before signaling.
  iree_hal_semaphore_list_t wait_semaphores = {
      .count = 1,
      .semaphores = &stream->timeline_semaphore,
      .payload_values = &wait_value,
  };
  iree_hal_semaphore_list_t signal_semaphores = {
      .count = 1,
      .semaphores = &event->semaphore,
      .payload_values = &event->signal_value,
  };

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_device_queue_barrier(
              stream->context->device, stream->queue_affinity, wait_semaphores,
              signal_semaphores, IREE_HAL_EXECUTE_FLAG_NONE));

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_event_synchronize(
    iree_hal_streaming_event_t* event) {
  IREE_ASSERT_ARGUMENT(event);
  IREE_TRACE_ZONE_BEGIN(z0);

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_semaphore_wait(event->semaphore, event->signal_value,
                                  iree_infinite_timeout()));

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_event_elapsed_time(
    float* ms, iree_hal_streaming_event_t* start,
    iree_hal_streaming_event_t* stop) {
  IREE_ASSERT_ARGUMENT(ms);
  IREE_ASSERT_ARGUMENT(start);
  IREE_ASSERT_ARGUMENT(stop);

  // Ensure both events have been recorded.
  if (start->record_time_ns == 0 || stop->record_time_ns == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "events must be recorded before measuring elapsed time");
  }

  // Ensure both events have completed.
  int start_status = 0;
  IREE_RETURN_IF_ERROR(iree_hal_streaming_event_query(start, &start_status));
  if (start_status != 0) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "start event has not completed");
  }

  int stop_status = 0;
  IREE_RETURN_IF_ERROR(iree_hal_streaming_event_query(stop, &stop_status));
  if (stop_status != 0) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "stop event has not completed");
  }

  // Calculate elapsed time in milliseconds.
  int64_t elapsed_ns = stop->record_time_ns - start->record_time_ns;
  *ms = (float)elapsed_ns / 1000000.0f;  // Convert nanoseconds to milliseconds.

  return iree_ok_status();
}
