// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_PROFILE_EVENTS_H_
#define IREE_HAL_DRIVERS_AMDGPU_PROFILE_EVENTS_H_

#include "iree/base/api.h"
#include "iree/base/threading/mutex.h"
#include "iree/hal/api.h"
#include "iree/hal/profile_sink.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_profile_event_streams_t
//===----------------------------------------------------------------------===//

// Lossy fixed-capacity host-side profiling event stream.
//
// Producers append records until the stream reaches capacity. Once full, new
// records are dropped and accounted in |dropped_count| until a flush copies
// retained records to the sink and advances |read_position|. The stream writes
// a metadata-only TRUNCATED chunk if only dropped records are available.
typedef struct iree_hal_amdgpu_profile_event_stream_t {
  // Mutex protecting positions, dropped counts, and event id allocation.
  iree_slim_mutex_t mutex;

  // Event record storage, or NULL when the stream is disabled.
  void* events;

  // Byte size of one event record in |events|.
  iree_host_size_t event_size;

  // Allocated event capacity, always a power of two when nonzero.
  iree_host_size_t capacity;

  // Mask used to wrap absolute stream positions.
  iree_host_size_t mask;

  // Absolute read position of the first retained event.
  uint64_t read_position;

  // Absolute write position one past the last retained event.
  uint64_t write_position;

  // Number of events dropped since the last successful flush accounted them.
  uint64_t dropped_count;

  // Next nonzero event id assigned by this stream.
  uint64_t next_event_id;
} iree_hal_amdgpu_profile_event_stream_t;

// Host-side profiling event streams owned by an AMDGPU logical device.
typedef struct iree_hal_amdgpu_profile_event_streams_t {
  // Memory lifecycle event stream and allocation id state.
  struct {
    // Lossy ring for iree_hal_profile_memory_event_t records.
    iree_hal_amdgpu_profile_event_stream_t stream;

    // Next nonzero allocation id assigned to profiled memory objects.
    uint64_t next_allocation_id;
  } memory;

  // Queue operation event stream.
  struct {
    // Lossy ring for iree_hal_profile_queue_event_t records.
    iree_hal_amdgpu_profile_event_stream_t stream;
  } queue;
} iree_hal_amdgpu_profile_event_streams_t;

// Initializes stream mutexes in caller-owned zeroed storage.
void iree_hal_amdgpu_profile_event_streams_initialize(
    iree_hal_amdgpu_profile_event_streams_t* streams);

// Deinitializes streams and releases all event storage.
void iree_hal_amdgpu_profile_event_streams_deinitialize(
    iree_hal_amdgpu_profile_event_streams_t* streams,
    iree_allocator_t host_allocator);

// Returns true when memory event storage is allocated.
bool iree_hal_amdgpu_profile_event_streams_has_memory_storage(
    const iree_hal_amdgpu_profile_event_streams_t* streams);

// Returns true when queue event storage is allocated.
bool iree_hal_amdgpu_profile_event_streams_has_queue_storage(
    const iree_hal_amdgpu_profile_event_streams_t* streams);

// Allocates memory event storage if not already allocated.
iree_status_t iree_hal_amdgpu_profile_event_streams_ensure_memory_storage(
    iree_hal_amdgpu_profile_event_streams_t* streams,
    iree_host_size_t event_capacity, iree_allocator_t host_allocator);

// Allocates queue event storage if not already allocated.
iree_status_t iree_hal_amdgpu_profile_event_streams_ensure_queue_storage(
    iree_hal_amdgpu_profile_event_streams_t* streams,
    iree_host_size_t event_capacity, iree_allocator_t host_allocator);

// Clears the memory event stream and resets memory event/allocation ids.
void iree_hal_amdgpu_profile_event_streams_clear_memory(
    iree_hal_amdgpu_profile_event_streams_t* streams);

// Clears the queue event stream and resets queue event ids.
void iree_hal_amdgpu_profile_event_streams_clear_queue(
    iree_hal_amdgpu_profile_event_streams_t* streams);

// Allocates a memory allocation id for |active_session_id|.
uint64_t iree_hal_amdgpu_profile_event_streams_allocate_memory_allocation_id(
    iree_hal_amdgpu_profile_event_streams_t* streams,
    uint64_t active_session_id, uint64_t* out_session_id);

// Records one memory event if |session_id| matches |active_session_id|.
bool iree_hal_amdgpu_profile_event_streams_record_memory_event(
    iree_hal_amdgpu_profile_event_streams_t* streams,
    uint64_t active_session_id, uint64_t session_id,
    const iree_hal_profile_memory_event_t* event);

// Records one queue event.
void iree_hal_amdgpu_profile_event_streams_record_queue_event(
    iree_hal_amdgpu_profile_event_streams_t* streams,
    const iree_hal_profile_queue_event_t* event);

// Writes pending memory events to |sink|.
iree_status_t iree_hal_amdgpu_profile_event_streams_write_memory(
    iree_hal_amdgpu_profile_event_streams_t* streams,
    iree_hal_profile_sink_t* sink, uint64_t session_id,
    iree_allocator_t host_allocator);

// Writes pending queue events to |sink|.
iree_status_t iree_hal_amdgpu_profile_event_streams_write_queue(
    iree_hal_amdgpu_profile_event_streams_t* streams,
    iree_hal_profile_sink_t* sink, uint64_t session_id,
    iree_allocator_t host_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_PROFILE_EVENTS_H_
