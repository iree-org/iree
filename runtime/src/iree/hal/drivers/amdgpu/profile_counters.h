// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_PROFILE_COUNTERS_H_
#define IREE_HAL_DRIVERS_AMDGPU_PROFILE_COUNTERS_H_

#include "iree/hal/device.h"
#include "iree/hal/drivers/amdgpu/util/aql_emitter.h"
#include "iree/hal/drivers/amdgpu/util/aql_ring.h"
#include "iree/hal/profile_schema.h"
#include "iree/hal/profile_sink.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_amdgpu_host_queue_t iree_hal_amdgpu_host_queue_t;
typedef struct iree_hal_amdgpu_logical_device_t
    iree_hal_amdgpu_logical_device_t;
typedef struct iree_hal_amdgpu_profile_counter_sample_slot_t
    iree_hal_amdgpu_profile_counter_sample_slot_t;
typedef struct iree_hal_amdgpu_profile_counter_range_slot_t
    iree_hal_amdgpu_profile_counter_range_slot_t;
typedef struct iree_hal_amdgpu_profile_counter_session_t
    iree_hal_amdgpu_profile_counter_session_t;
typedef struct iree_hal_amdgpu_profile_dispatch_event_reservation_t
    iree_hal_amdgpu_profile_dispatch_event_reservation_t;

// Flags selecting which counter resources to enable on a host queue.
typedef uint32_t iree_hal_amdgpu_profile_counter_enable_flags_t;
enum iree_hal_amdgpu_profile_counter_enable_flag_bits_t {
  IREE_HAL_AMDGPU_PROFILE_COUNTER_ENABLE_FLAG_NONE = 0u,
  // Enables dispatch-attributed counter sample resources.
  IREE_HAL_AMDGPU_PROFILE_COUNTER_ENABLE_FLAG_DISPATCH_SAMPLES = 1u << 0,
  // Enables queue-carried physical-device range counter resources.
  IREE_HAL_AMDGPU_PROFILE_COUNTER_ENABLE_FLAG_QUEUE_RANGES = 1u << 1,
};

// Flags controlling queue-range counter flush behavior.
typedef uint32_t iree_hal_amdgpu_profile_counter_range_flush_flags_t;
enum iree_hal_amdgpu_profile_counter_range_flush_flag_bits_t {
  IREE_HAL_AMDGPU_PROFILE_COUNTER_RANGE_FLUSH_FLAG_NONE = 0u,
  // Starts a new range on the queue after stopping the current range.
  IREE_HAL_AMDGPU_PROFILE_COUNTER_RANGE_FLUSH_FLAG_RESTART = 1u << 0,
};

// Allocates a hardware counter profiling session from |options|.
//
// The returned session is immutable after creation except for its monotonically
// assigned sample identifiers. The logical-device profiling begin path owns the
// session and publishes a borrowed pointer to queues while profiling is active.
iree_status_t iree_hal_amdgpu_profile_counter_session_allocate(
    iree_hal_amdgpu_logical_device_t* logical_device,
    const iree_hal_device_profiling_options_t* options,
    iree_allocator_t host_allocator,
    iree_hal_amdgpu_profile_counter_session_t** out_session);

// Frees |session| and releases its aqlprofile library reference.
void iree_hal_amdgpu_profile_counter_session_free(
    iree_hal_amdgpu_profile_counter_session_t* session);

// Returns true when |session| contains counter sets to capture.
bool iree_hal_amdgpu_profile_counter_session_is_active(
    const iree_hal_amdgpu_profile_counter_session_t* session);

// Returns true when |session| captures dispatch-attributed samples.
bool iree_hal_amdgpu_profile_counter_session_captures_dispatch_samples(
    const iree_hal_amdgpu_profile_counter_session_t* session);

// Returns true when |session| captures queue-level counter ranges.
bool iree_hal_amdgpu_profile_counter_session_captures_queue_ranges(
    const iree_hal_amdgpu_profile_counter_session_t* session);

// Writes counter-set and counter metadata chunks for |session|.
iree_status_t iree_hal_amdgpu_profile_counter_session_write_metadata(
    const iree_hal_amdgpu_profile_counter_session_t* session,
    iree_hal_profile_sink_t* sink, uint64_t session_id,
    iree_string_view_t stream_name);

// Enables host-queue-carried counter sample storage for |queue|.
//
// |flags| selects which session capture resources are materialized on this
// queue. Dispatch slots create aqlprofile handles lazily and retain them until
// profiling is disabled so steady counter captures reuse packet/output storage
// after the dispatch event cursor advances past each slot. Range resources are
// pre-created because range flush/restart is cold but should not allocate while
// the queue is stopped waiting for samples.
iree_status_t iree_hal_amdgpu_host_queue_enable_profile_counters(
    iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_amdgpu_profile_counter_session_t* session,
    iree_hal_amdgpu_profile_counter_enable_flags_t flags);

// Disables queue-local counter sample storage and deletes all slot handles.
void iree_hal_amdgpu_host_queue_disable_profile_counters(
    iree_hal_amdgpu_host_queue_t* queue);

// Starts queue-carried physical-device counter ranges for |queue|.
iree_status_t iree_hal_amdgpu_host_queue_start_profile_counter_ranges(
    iree_hal_amdgpu_host_queue_t* queue);

// Stops, optionally writes, and optionally restarts physical-device ranges.
iree_status_t iree_hal_amdgpu_host_queue_flush_profile_counter_ranges(
    iree_hal_amdgpu_host_queue_t* queue, iree_hal_profile_sink_t* sink,
    uint64_t session_id,
    iree_hal_amdgpu_profile_counter_range_flush_flags_t flags);

// Returns the number of additional AQL packets needed for |reservation|.
uint32_t iree_hal_amdgpu_host_queue_profile_counter_packet_count(
    const iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_amdgpu_profile_dispatch_event_reservation_t reservation);

// Returns the number of counter sets captured for each profiled dispatch.
uint32_t iree_hal_amdgpu_host_queue_profile_counter_set_count(
    const iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_amdgpu_profile_dispatch_event_reservation_t reservation);

// Prepares counter sample slots for |reservation|.
//
// Caller must hold queue->locks.submission_mutex and must call this only after
// the dispatch profile events have been reserved. Handles are created lazily
// per event-ring slot and then reused only after the dispatch event cursor has
// advanced past the slot.
iree_status_t iree_hal_amdgpu_host_queue_prepare_profile_counter_samples(
    iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_amdgpu_profile_dispatch_event_reservation_t reservation);

// Emplaces counter start packets beginning at |first_packet_index|.
//
// Packet bodies are populated and commit metadata is written into
// |packet_headers|/|packet_setups|, but headers are not committed to the AQL
// ring. Command-buffer replay uses this so all packet bodies can be populated
// before publishing headers in order.
void iree_hal_amdgpu_host_queue_emplace_profile_counter_start_packets(
    iree_hal_amdgpu_host_queue_t* queue, uint64_t event_position,
    uint32_t counter_set_count, uint64_t first_packet_id,
    uint32_t first_packet_index,
    iree_hal_amdgpu_aql_packet_control_t packet_control,
    uint16_t* packet_headers, uint16_t* packet_setups);

// Emplaces counter read/stop packet pairs beginning at |first_packet_index|.
//
// Packets are emitted in rocprof/aqlprofile order: read first, then stop.
void iree_hal_amdgpu_host_queue_emplace_profile_counter_read_stop_packets(
    iree_hal_amdgpu_host_queue_t* queue, uint64_t event_position,
    uint32_t counter_set_count, uint64_t first_packet_id,
    uint32_t first_packet_index,
    iree_hal_amdgpu_aql_packet_control_t packet_control,
    uint16_t* packet_headers, uint16_t* packet_setups);

// Commits counter start packets beginning at |first_packet_id|.
//
// The caller owns all surrounding ordering and doorbell publication. Each
// packet references the prepared sample slot for |event_position| and has no
// completion signal; dispatch timestamp harvest remains the queue-visible
// completion point for the profiled dispatch.
void iree_hal_amdgpu_host_queue_commit_profile_counter_start_packets(
    iree_hal_amdgpu_host_queue_t* queue, uint64_t event_position,
    uint32_t counter_set_count, uint64_t first_packet_id,
    iree_hal_amdgpu_aql_packet_control_t packet_control);

// Commits counter read/stop packet pairs beginning at |first_packet_id|.
//
// Packets are emitted in rocprof/aqlprofile order: read first, then stop.
void iree_hal_amdgpu_host_queue_commit_profile_counter_read_stop_packets(
    iree_hal_amdgpu_host_queue_t* queue, uint64_t event_position,
    uint32_t counter_set_count, uint64_t first_packet_id,
    iree_hal_amdgpu_aql_packet_control_t packet_control);

// Writes counter sample chunks for retired dispatch events in |events|.
//
// The caller must not advance the dispatch event read cursor until this returns
// successfully; queue slot reuse is what makes the aqlprofile handles safe.
iree_status_t iree_hal_amdgpu_host_queue_write_profile_counter_samples(
    iree_hal_amdgpu_host_queue_t* queue, iree_hal_profile_sink_t* sink,
    uint64_t session_id, uint64_t event_read_position,
    iree_host_size_t event_count,
    const iree_hal_profile_dispatch_event_t* events);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_PROFILE_COUNTERS_H_
