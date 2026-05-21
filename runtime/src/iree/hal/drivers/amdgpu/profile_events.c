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
  iree_allocator_free(host_allocator, stream->ring.records);
  memset(&stream->ring, 0, sizeof(stream->ring));
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
  if (stream->ring.records) return iree_ok_status();
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
    iree_hal_profile_event_ring_initialize(events, event_size, event_capacity,
                                           &stream->ring);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_amdgpu_profile_event_stream_clear(
    iree_hal_amdgpu_profile_event_stream_t* stream) {
  iree_slim_mutex_lock(&stream->mutex);
  iree_hal_profile_event_ring_clear(&stream->ring);
  iree_slim_mutex_unlock(&stream->mutex);
}

static iree_status_t iree_hal_amdgpu_profile_event_stream_write(
    iree_hal_amdgpu_profile_event_stream_t* stream,
    iree_hal_profile_sink_t* sink, iree_hal_profile_chunk_metadata_t metadata,
    iree_allocator_t host_allocator) {
  (void)host_allocator;
  if (!sink || !stream->ring.records) return iree_ok_status();
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_profile_event_ring_snapshot_t snapshot;
  iree_slim_mutex_lock(&stream->mutex);
  iree_status_t status =
      iree_hal_profile_event_ring_snapshot(&stream->ring, &snapshot);
  iree_slim_mutex_unlock(&stream->mutex);

  if (iree_status_is_ok(status) &&
      (snapshot.record_count != 0 || snapshot.dropped_record_count != 0)) {
    if (snapshot.dropped_record_count != 0) {
      metadata.flags |= IREE_HAL_PROFILE_CHUNK_FLAG_TRUNCATED;
      metadata.dropped_record_count = snapshot.dropped_record_count;
    }

    status = iree_hal_profile_sink_write(
        sink, &metadata, snapshot.record_span_count,
        snapshot.record_span_count ? snapshot.record_spans : NULL);
    if (iree_status_is_ok(status)) {
      iree_slim_mutex_lock(&stream->mutex);
      iree_hal_profile_event_ring_commit_snapshot(&stream->ring, &snapshot);
      iree_slim_mutex_unlock(&stream->mutex);
    }
  }

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
  return streams->memory.stream.ring.records != NULL;
}

bool iree_hal_amdgpu_profile_event_streams_has_queue_storage(
    const iree_hal_amdgpu_profile_event_streams_t* streams) {
  return streams->queue.stream.ring.records != NULL;
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
  if (!stream->ring.records) return false;
  iree_slim_mutex_lock(&stream->mutex);
  const bool session_matches =
      session_id == 0 || active_session_id == session_id;
  if (session_matches) {
    uint64_t event_position = 0;
    uint64_t event_id = 0;
    if (iree_hal_profile_event_ring_try_append(&stream->ring, &event_position,
                                               &event_id)) {
      iree_hal_profile_memory_event_t record = *event;
      record.record_length = sizeof(record);
      record.event_id = event_id;
      if (record.host_time_ns == 0) {
        record.host_time_ns = iree_time_now();
      }
      iree_hal_profile_memory_event_t* target =
          (iree_hal_profile_memory_event_t*)
              iree_hal_profile_event_ring_record_at(&stream->ring,
                                                    event_position);
      *target = record;
      recorded = true;
    }
  }
  iree_slim_mutex_unlock(&stream->mutex);
  return recorded;
}

void iree_hal_amdgpu_profile_event_streams_record_queue_event(
    iree_hal_amdgpu_profile_event_streams_t* streams,
    const iree_hal_profile_queue_event_t* event) {
  iree_hal_amdgpu_profile_event_stream_t* stream = &streams->queue.stream;
  if (!stream->ring.records) return;
  iree_slim_mutex_lock(&stream->mutex);
  uint64_t event_position = 0;
  uint64_t event_id = 0;
  if (iree_hal_profile_event_ring_try_append(&stream->ring, &event_position,
                                             &event_id)) {
    iree_hal_profile_queue_event_t record = *event;
    record.record_length = sizeof(record);
    record.event_id = event_id;
    if (record.host_time_ns == 0) {
      record.host_time_ns = iree_time_now();
    }
    iree_hal_profile_queue_event_t* target =
        (iree_hal_profile_queue_event_t*)iree_hal_profile_event_ring_record_at(
            &stream->ring, event_position);
    *target = record;
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
