// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_LOCAL_PROFILE_H_
#define IREE_HAL_LOCAL_PROFILE_H_

#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/hal/device.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_local_profile_recorder_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_local_profile_recorder_t
    iree_hal_local_profile_recorder_t;

// Returns the HAL-native profiling data families produced by local CPU
// profiling recorders.
static inline iree_hal_device_profiling_data_families_t
iree_hal_local_profile_recorder_supported_data_families(void) {
  return IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS |
         IREE_HAL_DEVICE_PROFILING_DATA_HOST_EXECUTION_EVENTS;
}

// Metadata needed to begin a local CPU profiling session.
//
// The recorder writes the supplied device and queue records during creation and
// does not retain the record arrays. |name| is copied into recorder-owned
// storage because flush/end can happen after profiling_begin returns.
typedef struct iree_hal_local_profile_recorder_options_t {
  // Human-readable producer name used on session and metadata chunks.
  iree_string_view_t name;

  // Process-local profiling session identifier assigned by the caller.
  uint64_t session_id;

  // Number of physical device metadata records in |device_records|.
  iree_host_size_t device_record_count;

  // Borrowed physical device metadata records emitted at session begin.
  const iree_hal_profile_device_record_t* device_records;

  // Number of queue metadata records in |queue_records|.
  iree_host_size_t queue_record_count;

  // Borrowed queue metadata records emitted at session begin.
  const iree_hal_profile_queue_record_t* queue_records;

  // Maximum queue events retained between flushes; 0 selects the default.
  iree_host_size_t queue_event_capacity;

  // Maximum host execution events retained between flushes; 0 selects the
  // default.
  iree_host_size_t host_execution_event_capacity;

  // Maximum event relationships retained between flushes; 0 selects the
  // default.
  iree_host_size_t event_relationship_capacity;
} iree_hal_local_profile_recorder_options_t;

// Queue identity shared by local profiling records.
typedef struct iree_hal_local_profile_queue_scope_t {
  // Session-local physical device ordinal associated with the record.
  uint32_t physical_device_ordinal;

  // Session-local queue ordinal associated with the record.
  uint32_t queue_ordinal;

  // Producer-defined stream identifier matching queue metadata.
  uint64_t stream_id;
} iree_hal_local_profile_queue_scope_t;

// Returns a queue scope with absent ordinals and no stream id.
static inline iree_hal_local_profile_queue_scope_t
iree_hal_local_profile_queue_scope_default(void) {
  iree_hal_local_profile_queue_scope_t scope;
  memset(&scope, 0, sizeof(scope));
  scope.physical_device_ordinal = UINT32_MAX;
  scope.queue_ordinal = UINT32_MAX;
  return scope;
}

// Queue operation data used to append one queue event record.
typedef struct iree_hal_local_profile_queue_event_info_t {
  // Kind of queue operation represented by the event.
  iree_hal_profile_queue_event_type_t type;

  // Flags describing queue operation properties.
  iree_hal_profile_queue_event_flags_t flags;

  // Strategy used for wait dependencies on this operation.
  iree_hal_profile_queue_dependency_strategy_t dependency_strategy;

  // Queue metadata identity shared by the appended record.
  iree_hal_local_profile_queue_scope_t scope;

  // IREE monotonic host timestamp when submitted, or 0 to sample now.
  iree_time_t host_time_ns;

  // IREE monotonic host timestamp when ready to execute, or 0 when unknown.
  iree_time_t ready_host_time_ns;

  // Queue submission epoch associated with this operation, or 0 when absent.
  uint64_t submission_id;

  // Session-local command-buffer identifier, or 0 when not applicable.
  uint64_t command_buffer_id;

  // Producer-defined allocation identifier, or 0 when not applicable.
  uint64_t allocation_id;

  // Number of wait semaphores supplied to the queue operation.
  uint32_t wait_count;

  // Number of signal semaphores supplied to the queue operation.
  uint32_t signal_count;

  // Number of dedicated dependency barrier packets emitted for this operation.
  uint32_t barrier_count;

  // Number of encoded payload operations represented by this queue operation.
  uint32_t operation_count;

  // Type-specific payload byte length, or 0 when not applicable.
  uint64_t payload_length;
} iree_hal_local_profile_queue_event_info_t;

// Returns default queue event append data.
static inline iree_hal_local_profile_queue_event_info_t
iree_hal_local_profile_queue_event_info_default(void) {
  iree_hal_local_profile_queue_event_info_t info;
  memset(&info, 0, sizeof(info));
  info.scope = iree_hal_local_profile_queue_scope_default();
  return info;
}

// Host execution span data used to append one host execution event record.
typedef struct iree_hal_local_profile_host_execution_event_info_t {
  // Kind of queue operation represented by this span.
  iree_hal_profile_queue_event_type_t type;

  // Flags describing host execution properties.
  iree_hal_profile_host_execution_event_flags_t flags;

  // IREE status code for terminal execution result, or UINT32_MAX if unknown.
  uint32_t status_code;

  // Queue metadata identity shared by the appended record.
  iree_hal_local_profile_queue_scope_t scope;

  // Queue event identifier corresponding to this span, or 0 when absent.
  uint64_t related_queue_event_id;

  // Queue submission epoch containing this span, or 0 when absent.
  uint64_t submission_id;

  // Session-local command-buffer identifier, or 0 when not applicable.
  uint64_t command_buffer_id;

  // Session-local executable identifier, or 0 when not applicable.
  uint64_t executable_id;

  // Producer-defined allocation identifier, or 0 when not applicable.
  uint64_t allocation_id;

  // Command ordinal within a command buffer, or UINT32_MAX when absent.
  uint32_t command_index;

  // Executable export ordinal, or UINT32_MAX when absent.
  uint32_t export_ordinal;

  // Workgroup counts submitted for dispatch-like spans.
  uint32_t workgroup_count[3];

  // Workgroup sizes submitted for dispatch-like spans.
  uint32_t workgroup_size[3];

  // IREE monotonic host timestamp when execution started, or 0 to sample now.
  iree_time_t start_host_time_ns;

  // IREE monotonic host timestamp when execution completed, or 0 to sample now.
  iree_time_t end_host_time_ns;

  // Type-specific payload byte length, or 0 when not applicable.
  uint64_t payload_length;

  // Number of encoded payload operations represented by this span.
  uint32_t operation_count;
} iree_hal_local_profile_host_execution_event_info_t;

// Returns default host execution event append data.
static inline iree_hal_local_profile_host_execution_event_info_t
iree_hal_local_profile_host_execution_event_info_default(void) {
  iree_hal_local_profile_host_execution_event_info_t info;
  memset(&info, 0, sizeof(info));
  info.status_code = UINT32_MAX;
  info.scope = iree_hal_local_profile_queue_scope_default();
  info.command_index = UINT32_MAX;
  info.export_ordinal = UINT32_MAX;
  return info;
}

// Begins a local CPU profiling session and returns its recorder.
//
// Returns OK with |out_recorder| set to NULL when
// |profiling_options->data_families| is NONE. Otherwise the requested families
// must be a subset of iree_hal_local_profile_recorder_supported_data_families()
// and |profiling_options->sink| must be non-NULL.
iree_status_t iree_hal_local_profile_recorder_create(
    const iree_hal_local_profile_recorder_options_t* recorder_options,
    const iree_hal_device_profiling_options_t* profiling_options,
    iree_allocator_t host_allocator,
    iree_hal_local_profile_recorder_t** out_recorder);

// Destroys |recorder| and releases retained session resources.
//
// Callers should end active sessions with iree_hal_local_profile_recorder_end
// before destroying the recorder so sink end-session failures can be observed.
void iree_hal_local_profile_recorder_destroy(
    iree_hal_local_profile_recorder_t* recorder);

// Returns true when |recorder| is active and any of |data_families| is enabled.
bool iree_hal_local_profile_recorder_is_enabled(
    const iree_hal_local_profile_recorder_t* recorder,
    iree_hal_device_profiling_data_families_t data_families);

// Appends one host-timestamped queue event to |recorder|.
//
// |out_event_id| may be NULL. When provided it receives the assigned event id,
// or 0 if queue events were not requested.
iree_status_t iree_hal_local_profile_recorder_append_queue_event(
    iree_hal_local_profile_recorder_t* recorder,
    const iree_hal_local_profile_queue_event_info_t* event_info,
    uint64_t* out_event_id);

// Appends one host execution span to |recorder|.
//
// When |event_info->related_queue_event_id| is nonzero and both queue and host
// execution events are enabled, the recorder also appends an explicit
// queue-event-to-host-execution relationship.
iree_status_t iree_hal_local_profile_recorder_append_host_execution_event(
    iree_hal_local_profile_recorder_t* recorder,
    const iree_hal_local_profile_host_execution_event_info_t* event_info,
    uint64_t* out_event_id);

// Consumes |status| and records it as a deferred recorder failure.
//
// This is for producer contexts that cannot return profiling failures to their
// immediate caller, such as async completion callbacks. The first failure is
// reported by later flush/end calls; additional failures are attached to that
// first failure when status annotations are enabled. OK statuses are consumed
// without touching the recorder.
void iree_hal_local_profile_recorder_consume_status(
    iree_hal_local_profile_recorder_t* recorder, iree_status_t status);

// Writes all buffered profile records to the session sink.
iree_status_t iree_hal_local_profile_recorder_flush(
    iree_hal_local_profile_recorder_t* recorder);

// Flushes and ends the profiling session.
//
// The sink receives the terminal status code derived from the first failing
// flush or end-session operation. A second call after a successful end is a
// no-op.
iree_status_t iree_hal_local_profile_recorder_end(
    iree_hal_local_profile_recorder_t* recorder);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_LOCAL_PROFILE_H_
