// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_PROFILE_EVENTS_H_
#define IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_PROFILE_EVENTS_H_

#include "iree/hal/drivers/amdgpu/host_queue.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Allocates queue-local device-visible event rings used by dispatch and
// queue-device timestamp profiling.
iree_status_t iree_hal_amdgpu_host_queue_ensure_profile_event_storage(
    iree_hal_amdgpu_host_queue_t* queue);

// Clears all event-ring cursors and records while preserving allocated storage.
void iree_hal_amdgpu_host_queue_clear_profile_events(
    iree_hal_amdgpu_host_queue_t* queue);

// Releases queue-local profile event ring storage.
void iree_hal_amdgpu_host_queue_deallocate_profile_events(
    iree_hal_amdgpu_host_queue_t* queue);

// Allocates queue-local completion signals paired with dispatch event slots.
iree_status_t iree_hal_amdgpu_host_queue_ensure_profiling_completion_signals(
    iree_hal_amdgpu_host_queue_t* queue);

// Releases queue-local profiling completion signals.
void iree_hal_amdgpu_host_queue_deallocate_profiling_completion_signals(
    iree_hal_amdgpu_host_queue_t* queue);

// Returns the configured dispatch event ring capacity.
static inline uint32_t
iree_hal_amdgpu_host_queue_profile_dispatch_event_capacity(
    const iree_hal_amdgpu_host_queue_t* queue) {
  return queue->profiling.dispatch_event_capacity;
}

// Returns the dispatch event ring slot index for |event_position|.
static inline uint32_t iree_hal_amdgpu_host_queue_profile_dispatch_event_index(
    const iree_hal_amdgpu_host_queue_t* queue, uint64_t event_position) {
  return (uint32_t)(event_position & queue->profiling.dispatch_event_mask);
}

// Returns the raw profiling completion signal paired with |event_position|'s
// dispatch event ring slot. The returned pointer references queue-owned
// iree_amd_signal_t storage, not a ROCR-created HSA signal, and must never be
// passed to host signal APIs except as an AQL packet completion_signal handle.
// Valid only while HSA queue timestamp profiling is enabled.
static inline iree_amd_signal_t*
iree_hal_amdgpu_host_queue_profiling_completion_signal_ptr(
    const iree_hal_amdgpu_host_queue_t* queue, uint64_t event_position) {
  const uint32_t signal_index =
      iree_hal_amdgpu_host_queue_profile_dispatch_event_index(queue,
                                                              event_position);
  const uint32_t block_index =
      signal_index / queue->profiling.signals_per_block;
  const uint32_t block_signal_index =
      signal_index - block_index * queue->profiling.signals_per_block;
  uint8_t* block_ptr =
      (uint8_t*)queue->profiling.signal_blocks[block_index]->ptr;
  iree_amd_signal_t* signal =
      (iree_amd_signal_t*)(block_ptr +
                           block_signal_index * sizeof(iree_amd_signal_t));
  return signal;
}

// Returns the raw profiling completion signal handle paired with
// |event_position|'s dispatch event ring slot.
static inline iree_hsa_signal_t
iree_hal_amdgpu_host_queue_profiling_completion_signal(
    const iree_hal_amdgpu_host_queue_t* queue, uint64_t event_position) {
  iree_amd_signal_t* signal =
      iree_hal_amdgpu_host_queue_profiling_completion_signal_ptr(
          queue, event_position);
  return (iree_hsa_signal_t){.handle = (uint64_t)(uintptr_t)signal};
}

// Reserves queue-local dispatch profile event records.
//
// Caller must hold submission_mutex. If the ring cannot hold |event_count|
// records the function fails with RESOURCE_EXHAUSTED. Callers that want to keep
// long captures exact must drain with iree_hal_device_profiling_flush before
// the ring fills.
iree_status_t iree_hal_amdgpu_host_queue_reserve_profile_dispatch_events(
    iree_hal_amdgpu_host_queue_t* queue, uint32_t event_count,
    iree_hal_amdgpu_profile_dispatch_event_reservation_t* out_reservation);

// Cancels a tail reservation before its packets have been published.
//
// Caller must hold submission_mutex. Only valid for the most recent successful
// reservation on a path that is failing before AQL publication.
void iree_hal_amdgpu_host_queue_cancel_profile_dispatch_events(
    iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_amdgpu_profile_dispatch_event_reservation_t reservation);

// Returns the dispatch event record at |event_position|.
iree_hal_amdgpu_profile_dispatch_event_t*
iree_hal_amdgpu_host_queue_profile_dispatch_event_at(
    const iree_hal_amdgpu_host_queue_t* queue, uint64_t event_position);

// Marks a completed event reservation ready for sink flush.
void iree_hal_amdgpu_host_queue_retire_profile_dispatch_events(
    iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_amdgpu_profile_dispatch_event_reservation_t reservation);

// Returns true when queue device timestamp packets should be emitted.
bool iree_hal_amdgpu_host_queue_should_profile_queue_device_events(
    const iree_hal_amdgpu_host_queue_t* queue);

// Reserves queue-local device-timestamped queue operation records.
//
// Caller must hold submission_mutex. If the ring cannot hold |event_count|
// records the function fails with RESOURCE_EXHAUSTED. The returned records live
// in device-visible memory so PM4 packets can write timestamp fields directly.
iree_status_t iree_hal_amdgpu_host_queue_reserve_profile_queue_device_events(
    iree_hal_amdgpu_host_queue_t* queue, uint32_t event_count,
    iree_hal_amdgpu_profile_queue_device_event_reservation_t* out_reservation);

// Cancels a tail queue-device-event reservation before AQL publication.
//
// Caller must hold submission_mutex. Only valid for the most recent successful
// reservation on a path that is failing before AQL publication.
void iree_hal_amdgpu_host_queue_cancel_profile_queue_device_events(
    iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_amdgpu_profile_queue_device_event_reservation_t reservation);

// Returns the queue device event record at |event_position|.
iree_hal_amdgpu_profile_queue_device_event_t*
iree_hal_amdgpu_host_queue_profile_queue_device_event_at(
    const iree_hal_amdgpu_host_queue_t* queue, uint64_t event_position);

// Marks a completed queue-device-event reservation ready for sink flush.
void iree_hal_amdgpu_host_queue_retire_profile_queue_device_events(
    iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_amdgpu_profile_queue_device_event_reservation_t reservation);

// Writes and clears buffered device-side profile events for this queue.
//
// Sink writes are cold profiling API operations and may block. The submission
// and completion paths only append to the queue-local batch.
iree_status_t iree_hal_amdgpu_host_queue_write_profile_events(
    iree_hal_amdgpu_host_queue_t* queue, iree_hal_profile_sink_t* sink,
    uint64_t session_id);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_PROFILE_EVENTS_H_
