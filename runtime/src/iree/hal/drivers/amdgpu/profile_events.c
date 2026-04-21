// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/profile_events.h"

static void iree_hal_amdgpu_profile_event_stream_initialize(
    iree_hal_amdgpu_profile_event_stream_t* stream) {
  iree_slim_mutex_initialize(&stream->mutex);
}

static void iree_hal_amdgpu_profile_event_stream_deallocate(
    iree_hal_amdgpu_profile_event_stream_t* stream,
    iree_allocator_t host_allocator) {
  iree_allocator_free(host_allocator, stream->events);
  stream->events = NULL;
  stream->event_size = 0;
  stream->capacity = 0;
  stream->mask = 0;
  stream->read_position = 0;
  stream->write_position = 0;
  stream->dropped_count = 0;
  stream->next_event_id = 0;
}

static void iree_hal_amdgpu_profile_event_stream_deinitialize(
    iree_hal_amdgpu_profile_event_stream_t* stream,
    iree_allocator_t host_allocator) {
  iree_hal_amdgpu_profile_event_stream_deallocate(stream, host_allocator);
  iree_slim_mutex_deinitialize(&stream->mutex);
}

static iree_status_t iree_hal_amdgpu_profile_event_stream_ensure_storage(
    iree_hal_amdgpu_profile_event_stream_t* stream, iree_host_size_t event_size,
    iree_host_size_t event_capacity, iree_allocator_t host_allocator) {
  if (stream->events) return iree_ok_status();
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, event_capacity);

  if (IREE_UNLIKELY(event_capacity == 0 ||
                    !iree_host_size_is_power_of_two(event_capacity))) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                             "AMDGPU profile event stream capacity must be a "
                             "non-zero power of two (got %" PRIhsz ")",
                             event_capacity));
  }

  iree_host_size_t storage_size = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_host_size_checked_mul(event_capacity, event_size, &storage_size)
              ? iree_ok_status()
              : iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                 "AMDGPU profile event stream storage size "
                                 "overflows"));

  void* events = NULL;
  iree_status_t status =
      iree_allocator_malloc(host_allocator, storage_size, &events);
  if (iree_status_is_ok(status)) {
    memset(events, 0, storage_size);
    stream->events = events;
    stream->event_size = event_size;
    stream->capacity = event_capacity;
    stream->mask = event_capacity - 1;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_amdgpu_profile_event_stream_clear(
    iree_hal_amdgpu_profile_event_stream_t* stream) {
  iree_slim_mutex_lock(&stream->mutex);
  stream->read_position = 0;
  stream->write_position = 0;
  stream->dropped_count = 0;
  stream->next_event_id = 1;
  if (stream->events) {
    memset(stream->events, 0, stream->capacity * stream->event_size);
  }
  iree_slim_mutex_unlock(&stream->mutex);
}

static iree_status_t iree_hal_amdgpu_profile_event_stream_copy_pending(
    iree_hal_amdgpu_profile_event_stream_t* stream,
    iree_allocator_t host_allocator, void** out_events,
    iree_host_size_t* out_event_count, iree_host_size_t* out_event_storage_size,
    uint64_t* out_dropped_count, uint64_t* out_read_position) {
  *out_events = NULL;
  *out_event_count = 0;
  *out_event_storage_size = 0;
  *out_dropped_count = 0;
  *out_read_position = 0;

  if (!stream->events) return iree_ok_status();

  iree_slim_mutex_lock(&stream->mutex);
  const uint64_t read_position = stream->read_position;
  const uint64_t write_position = stream->write_position;
  const iree_host_size_t event_count =
      (iree_host_size_t)(write_position - read_position);
  const uint64_t dropped_count = stream->dropped_count;

  iree_status_t status = iree_ok_status();
  void* events = NULL;
  iree_host_size_t event_storage_size = 0;
  if (event_count > 0) {
    if (!iree_host_size_checked_mul(event_count, stream->event_size,
                                    &event_storage_size)) {
      status =
          iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                           "AMDGPU profile event stream copy size overflows");
    }
    if (iree_status_is_ok(status)) {
      status =
          iree_allocator_malloc(host_allocator, event_storage_size, &events);
    }
    if (iree_status_is_ok(status)) {
      uint8_t* target = (uint8_t*)events;
      const uint8_t* source = (const uint8_t*)stream->events;
      for (iree_host_size_t i = 0; i < event_count; ++i) {
        const iree_host_size_t source_offset =
            ((read_position + i) & stream->mask) * stream->event_size;
        memcpy(target + i * stream->event_size, source + source_offset,
               stream->event_size);
      }
    }
  }
  iree_slim_mutex_unlock(&stream->mutex);

  if (!iree_status_is_ok(status)) {
    iree_allocator_free(host_allocator, events);
    return status;
  }

  *out_events = events;
  *out_event_count = event_count;
  *out_event_storage_size = event_storage_size;
  *out_dropped_count = dropped_count;
  *out_read_position = read_position;
  return iree_ok_status();
}

static void iree_hal_amdgpu_profile_event_stream_commit_write(
    iree_hal_amdgpu_profile_event_stream_t* stream, uint64_t read_position,
    iree_host_size_t event_count, uint64_t dropped_count) {
  iree_slim_mutex_lock(&stream->mutex);
  stream->read_position = read_position + event_count;
  if (stream->dropped_count >= dropped_count) {
    stream->dropped_count -= dropped_count;
  } else {
    stream->dropped_count = 0;
  }
  iree_slim_mutex_unlock(&stream->mutex);
}

static iree_status_t iree_hal_amdgpu_profile_event_stream_write(
    iree_hal_amdgpu_profile_event_stream_t* stream,
    iree_hal_profile_sink_t* sink, iree_hal_profile_chunk_metadata_t metadata,
    iree_allocator_t host_allocator) {
  if (!sink || !stream->events) return iree_ok_status();
  IREE_TRACE_ZONE_BEGIN(z0);

  void* events = NULL;
  iree_host_size_t event_count = 0;
  iree_host_size_t event_storage_size = 0;
  uint64_t dropped_count = 0;
  uint64_t read_position = 0;
  iree_status_t status = iree_hal_amdgpu_profile_event_stream_copy_pending(
      stream, host_allocator, &events, &event_count, &event_storage_size,
      &dropped_count, &read_position);

  if (iree_status_is_ok(status) && (event_count != 0 || dropped_count != 0)) {
    if (dropped_count != 0) {
      metadata.flags |= IREE_HAL_PROFILE_CHUNK_FLAG_TRUNCATED;
      metadata.dropped_record_count = dropped_count;
    }

    iree_const_byte_span_t iovec =
        iree_make_const_byte_span(events, event_storage_size);
    status = iree_hal_profile_sink_write(sink, &metadata, event_count ? 1 : 0,
                                         event_count ? &iovec : NULL);
    if (iree_status_is_ok(status)) {
      iree_hal_amdgpu_profile_event_stream_commit_write(
          stream, read_position, event_count, dropped_count);
    }
  }

  iree_allocator_free(host_allocator, events);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_amdgpu_profile_event_streams_initialize(
    iree_hal_amdgpu_profile_event_streams_t* streams) {
  IREE_ASSERT_ARGUMENT(streams);
  iree_hal_amdgpu_profile_event_stream_initialize(&streams->memory.stream);
  iree_hal_amdgpu_profile_event_stream_initialize(&streams->queue.stream);
}

void iree_hal_amdgpu_profile_event_streams_deinitialize(
    iree_hal_amdgpu_profile_event_streams_t* streams,
    iree_allocator_t host_allocator) {
  if (!streams) return;
  iree_hal_amdgpu_profile_event_stream_deinitialize(&streams->memory.stream,
                                                    host_allocator);
  streams->memory.next_allocation_id = 0;
  iree_hal_amdgpu_profile_event_stream_deinitialize(&streams->queue.stream,
                                                    host_allocator);
}

bool iree_hal_amdgpu_profile_event_streams_has_memory_storage(
    const iree_hal_amdgpu_profile_event_streams_t* streams) {
  return streams->memory.stream.events != NULL;
}

bool iree_hal_amdgpu_profile_event_streams_has_queue_storage(
    const iree_hal_amdgpu_profile_event_streams_t* streams) {
  return streams->queue.stream.events != NULL;
}

iree_status_t iree_hal_amdgpu_profile_event_streams_ensure_memory_storage(
    iree_hal_amdgpu_profile_event_streams_t* streams,
    iree_host_size_t event_capacity, iree_allocator_t host_allocator) {
  return iree_hal_amdgpu_profile_event_stream_ensure_storage(
      &streams->memory.stream, sizeof(iree_hal_profile_memory_event_t),
      event_capacity, host_allocator);
}

iree_status_t iree_hal_amdgpu_profile_event_streams_ensure_queue_storage(
    iree_hal_amdgpu_profile_event_streams_t* streams,
    iree_host_size_t event_capacity, iree_allocator_t host_allocator) {
  return iree_hal_amdgpu_profile_event_stream_ensure_storage(
      &streams->queue.stream, sizeof(iree_hal_profile_queue_event_t),
      event_capacity, host_allocator);
}

void iree_hal_amdgpu_profile_event_streams_clear_memory(
    iree_hal_amdgpu_profile_event_streams_t* streams) {
  iree_hal_amdgpu_profile_event_stream_clear(&streams->memory.stream);
  streams->memory.next_allocation_id = 1;
}

void iree_hal_amdgpu_profile_event_streams_clear_queue(
    iree_hal_amdgpu_profile_event_streams_t* streams) {
  iree_hal_amdgpu_profile_event_stream_clear(&streams->queue.stream);
}

uint64_t iree_hal_amdgpu_profile_event_streams_allocate_memory_allocation_id(
    iree_hal_amdgpu_profile_event_streams_t* streams,
    uint64_t active_session_id, uint64_t* out_session_id) {
  *out_session_id = 0;
  iree_slim_mutex_lock(&streams->memory.stream.mutex);
  *out_session_id = active_session_id;
  const uint64_t allocation_id = streams->memory.next_allocation_id++;
  iree_slim_mutex_unlock(&streams->memory.stream.mutex);
  return allocation_id;
}

bool iree_hal_amdgpu_profile_event_streams_record_memory_event(
    iree_hal_amdgpu_profile_event_streams_t* streams,
    uint64_t active_session_id, uint64_t session_id,
    const iree_hal_profile_memory_event_t* event) {
  bool recorded = false;
  iree_hal_amdgpu_profile_event_stream_t* stream = &streams->memory.stream;
  if (!stream->events) return false;
  iree_slim_mutex_lock(&stream->mutex);
  const bool session_matches =
      session_id == 0 || active_session_id == session_id;
  if (session_matches) {
    const uint64_t read_position = stream->read_position;
    const uint64_t write_position = stream->write_position;
    const uint64_t occupied_count = write_position - read_position;
    if (occupied_count < stream->capacity) {
      iree_hal_profile_memory_event_t record = *event;
      record.record_length = sizeof(record);
      record.event_id = stream->next_event_id++;
      if (record.host_time_ns == 0) {
        record.host_time_ns = iree_time_now();
      }
      iree_hal_profile_memory_event_t* events =
          (iree_hal_profile_memory_event_t*)stream->events;
      events[write_position & stream->mask] = record;
      stream->write_position = write_position + 1;
      recorded = true;
    } else {
      ++stream->dropped_count;
    }
  }
  iree_slim_mutex_unlock(&stream->mutex);
  return recorded;
}

void iree_hal_amdgpu_profile_event_streams_record_queue_event(
    iree_hal_amdgpu_profile_event_streams_t* streams,
    const iree_hal_profile_queue_event_t* event) {
  iree_hal_amdgpu_profile_event_stream_t* stream = &streams->queue.stream;
  if (!stream->events) return;
  iree_slim_mutex_lock(&stream->mutex);
  const uint64_t read_position = stream->read_position;
  const uint64_t write_position = stream->write_position;
  const uint64_t occupied_count = write_position - read_position;
  if (occupied_count < stream->capacity) {
    iree_hal_profile_queue_event_t record = *event;
    record.record_length = sizeof(record);
    record.event_id = stream->next_event_id++;
    if (record.host_time_ns == 0) {
      record.host_time_ns = iree_time_now();
    }
    iree_hal_profile_queue_event_t* events =
        (iree_hal_profile_queue_event_t*)stream->events;
    events[write_position & stream->mask] = record;
    stream->write_position = write_position + 1;
  } else {
    ++stream->dropped_count;
  }
  iree_slim_mutex_unlock(&stream->mutex);
}

iree_status_t iree_hal_amdgpu_profile_event_streams_write_memory(
    iree_hal_amdgpu_profile_event_streams_t* streams,
    iree_hal_profile_sink_t* sink, uint64_t session_id,
    iree_allocator_t host_allocator) {
  iree_hal_profile_chunk_metadata_t metadata =
      iree_hal_profile_chunk_metadata_default();
  metadata.content_type = IREE_HAL_PROFILE_CONTENT_TYPE_MEMORY_EVENTS;
  metadata.name = iree_make_cstring_view("amdgpu.memory");
  metadata.session_id = session_id;
  return iree_hal_amdgpu_profile_event_stream_write(
      &streams->memory.stream, sink, metadata, host_allocator);
}

iree_status_t iree_hal_amdgpu_profile_event_streams_write_queue(
    iree_hal_amdgpu_profile_event_streams_t* streams,
    iree_hal_profile_sink_t* sink, uint64_t session_id,
    iree_allocator_t host_allocator) {
  iree_hal_profile_chunk_metadata_t metadata =
      iree_hal_profile_chunk_metadata_default();
  metadata.content_type = IREE_HAL_PROFILE_CONTENT_TYPE_QUEUE_EVENTS;
  metadata.name = iree_make_cstring_view("amdgpu.queue");
  metadata.session_id = session_id;
  return iree_hal_amdgpu_profile_event_stream_write(
      &streams->queue.stream, sink, metadata, host_allocator);
}
