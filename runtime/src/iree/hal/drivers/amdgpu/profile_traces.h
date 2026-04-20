// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_PROFILE_TRACES_H_
#define IREE_HAL_DRIVERS_AMDGPU_PROFILE_TRACES_H_

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
typedef struct iree_hal_amdgpu_profile_dispatch_event_reservation_t
    iree_hal_amdgpu_profile_dispatch_event_reservation_t;
typedef struct iree_hal_amdgpu_profile_trace_session_t
    iree_hal_amdgpu_profile_trace_session_t;
typedef struct iree_hal_amdgpu_profile_trace_slot_t
    iree_hal_amdgpu_profile_trace_slot_t;

// Allocates an executable trace profiling session from |options|.
//
// The returned session is immutable after creation except for its monotonically
// assigned trace identifiers. The logical-device profiling begin path owns the
// session and publishes a borrowed pointer to queues while profiling is active.
iree_status_t iree_hal_amdgpu_profile_trace_session_allocate(
    iree_hal_amdgpu_logical_device_t* logical_device,
    const iree_hal_device_profiling_options_t* options,
    iree_allocator_t host_allocator,
    iree_hal_amdgpu_profile_trace_session_t** out_session);

// Frees |session| and releases its aqlprofile library reference.
void iree_hal_amdgpu_profile_trace_session_free(
    iree_hal_amdgpu_profile_trace_session_t* session);

// Returns true when |session| should emit executable trace packets.
bool iree_hal_amdgpu_profile_trace_session_is_active(
    const iree_hal_amdgpu_profile_trace_session_t* session);

// Enables queue-local executable trace storage for |queue|.
//
// Allocates one host-side slot per dispatch event ring entry. Slots hold only
// small control metadata until selected dispatches prepare them; a prepared
// slot owns one aqlprofile ATT packet handle and its trace output buffer until
// the corresponding dispatch event is successfully flushed and released.
iree_status_t iree_hal_amdgpu_host_queue_enable_profile_traces(
    iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_amdgpu_profile_trace_session_t* session);

// Disables queue-local executable trace storage and deletes all remaining slot
// handles.
void iree_hal_amdgpu_host_queue_disable_profile_traces(
    iree_hal_amdgpu_host_queue_t* queue);

// Returns the number of additional AQL packets needed for |reservation|.
uint32_t iree_hal_amdgpu_host_queue_profile_trace_packet_count(
    const iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_amdgpu_profile_dispatch_event_reservation_t reservation);

// Returns the number of trace start packets emitted before profiled dispatches.
uint32_t iree_hal_amdgpu_host_queue_profile_trace_start_packet_count(
    const iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_amdgpu_profile_dispatch_event_reservation_t reservation);

// Prepares executable trace slots for |reservation|.
//
// Caller must hold queue->submission_mutex and must call this only after the
// dispatch profile events have been reserved. Start/stop handles are created
// lazily per event-ring slot and then reused only after the dispatch event
// cursor has advanced past the slot. Because ATT output buffers are large, the
// event flush path releases the slot handle after all sink writes for the
// corresponding dispatch event have succeeded instead of retaining it for the
// full profiling session.
iree_status_t iree_hal_amdgpu_host_queue_prepare_profile_traces(
    iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_amdgpu_profile_dispatch_event_reservation_t reservation);

// Prepares the ATT code-object marker packet for |event_position|.
//
// Caller must hold queue->submission_mutex and call
// iree_hal_amdgpu_host_queue_prepare_profile_traces first for the same event
// slot so the aqlprofile allocation context has been initialized.
iree_status_t iree_hal_amdgpu_host_queue_prepare_profile_trace_code_object(
    iree_hal_amdgpu_host_queue_t* queue, uint64_t event_position,
    uint64_t executable_id);

// Emplaces one ATT start packet for |event_position| at |first_packet_index|.
void iree_hal_amdgpu_host_queue_emplace_profile_trace_start_packet(
    iree_hal_amdgpu_host_queue_t* queue, uint64_t event_position,
    uint64_t first_packet_id, uint32_t first_packet_index,
    iree_hal_amdgpu_aql_packet_control_t packet_control,
    uint16_t* packet_headers, uint16_t* packet_setups);

// Emplaces one ATT code-object marker packet for |event_position| at
// |first_packet_index|.
void iree_hal_amdgpu_host_queue_emplace_profile_trace_code_object_packet(
    iree_hal_amdgpu_host_queue_t* queue, uint64_t event_position,
    uint64_t first_packet_id, uint32_t first_packet_index,
    iree_hal_amdgpu_aql_packet_control_t packet_control,
    uint16_t* packet_headers, uint16_t* packet_setups);

// Emplaces one ATT stop packet for |event_position| at |first_packet_index|.
void iree_hal_amdgpu_host_queue_emplace_profile_trace_stop_packet(
    iree_hal_amdgpu_host_queue_t* queue, uint64_t event_position,
    uint64_t first_packet_id, uint32_t first_packet_index,
    iree_hal_amdgpu_aql_packet_control_t packet_control,
    uint16_t* packet_headers, uint16_t* packet_setups);

// Commits one ATT start packet for |event_position| at |packet_id|.
void iree_hal_amdgpu_host_queue_commit_profile_trace_start_packet(
    iree_hal_amdgpu_host_queue_t* queue, uint64_t event_position,
    uint64_t packet_id, iree_hal_amdgpu_aql_packet_control_t packet_control);

// Commits one ATT code-object marker packet for |event_position| at
// |packet_id|.
void iree_hal_amdgpu_host_queue_commit_profile_trace_code_object_packet(
    iree_hal_amdgpu_host_queue_t* queue, uint64_t event_position,
    uint64_t packet_id, iree_hal_amdgpu_aql_packet_control_t packet_control);

// Commits one ATT stop packet for |event_position| at |packet_id|.
void iree_hal_amdgpu_host_queue_commit_profile_trace_stop_packet(
    iree_hal_amdgpu_host_queue_t* queue, uint64_t event_position,
    uint64_t packet_id, iree_hal_amdgpu_aql_packet_control_t packet_control);

// Writes executable trace chunks for retired dispatch events in |events|.
//
// The caller must not advance the dispatch event read cursor until this returns
// successfully. This only writes trace payloads; the caller must release the
// flushed slots after all sink writes associated with the same event positions
// have succeeded.
iree_status_t iree_hal_amdgpu_host_queue_write_profile_traces(
    iree_hal_amdgpu_host_queue_t* queue, iree_hal_profile_sink_t* sink,
    uint64_t session_id, uint64_t event_read_position,
    iree_host_size_t event_count,
    const iree_hal_profile_dispatch_event_t* events);

// Releases ATT handles for flushed dispatch event positions.
//
// The event flush path must call this only after every sink write associated
// with the event positions has succeeded and before advancing the dispatch
// event read cursor so those ring slots cannot be reused while they are being
// reset.
void iree_hal_amdgpu_host_queue_release_profile_trace_slots(
    iree_hal_amdgpu_host_queue_t* queue, uint64_t event_read_position,
    iree_host_size_t event_count);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_PROFILE_TRACES_H_
