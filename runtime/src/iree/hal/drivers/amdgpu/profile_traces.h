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

// Creates an executable trace profiling session from |options|.
//
// The returned session is immutable after creation except for its monotonically
// assigned trace identifiers. The logical-device profiling begin path owns the
// session and publishes a borrowed pointer to queues while profiling is active.
iree_status_t iree_hal_amdgpu_profile_trace_session_create(
    iree_hal_amdgpu_logical_device_t* logical_device,
    const iree_hal_device_profiling_options_t* options,
    iree_allocator_t host_allocator,
    iree_hal_amdgpu_profile_trace_session_t** out_session);

// Destroys |session| and releases its aqlprofile library reference.
void iree_hal_amdgpu_profile_trace_session_destroy(
    iree_hal_amdgpu_profile_trace_session_t* session);

// Returns true when |session| should emit executable trace packets.
bool iree_hal_amdgpu_profile_trace_session_is_active(
    const iree_hal_amdgpu_profile_trace_session_t* session);

// Enables queue-local executable trace storage for |queue|.
iree_status_t iree_hal_amdgpu_host_queue_enable_profile_traces(
    iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_amdgpu_profile_trace_session_t* session);

// Disables queue-local executable trace storage and deletes all slot handles.
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
// dispatch profile events have been reserved. Handles are created lazily per
// event-ring slot and then reused only after the dispatch event cursor has
// advanced past the slot.
iree_status_t iree_hal_amdgpu_host_queue_prepare_profile_traces(
    iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_amdgpu_profile_dispatch_event_reservation_t reservation);

// Emplaces one ATT start packet for |event_position| at |first_packet_index|.
void iree_hal_amdgpu_host_queue_emplace_profile_trace_start_packet(
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

// Commits one ATT stop packet for |event_position| at |packet_id|.
void iree_hal_amdgpu_host_queue_commit_profile_trace_stop_packet(
    iree_hal_amdgpu_host_queue_t* queue, uint64_t event_position,
    uint64_t packet_id, iree_hal_amdgpu_aql_packet_control_t packet_control);

// Writes executable trace chunks for retired dispatch events in |events|.
//
// The caller must not advance the dispatch event read cursor until this returns
// successfully; queue slot reuse is what makes the aqlprofile handles safe.
iree_status_t iree_hal_amdgpu_host_queue_write_profile_traces(
    iree_hal_amdgpu_host_queue_t* queue, iree_hal_profile_sink_t* sink,
    uint64_t session_id, uint64_t event_read_position,
    iree_host_size_t event_count,
    const iree_hal_profile_dispatch_event_t* events);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_PROFILE_TRACES_H_
