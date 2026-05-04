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
#include "iree/hal/api.h"

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
         IREE_HAL_DEVICE_PROFILING_DATA_HOST_EXECUTION_EVENTS |
         IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_METADATA |
         IREE_HAL_DEVICE_PROFILING_DATA_MEMORY_EVENTS |
         IREE_HAL_DEVICE_PROFILING_DATA_COMMAND_REGION_EVENTS;
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

  // Maximum memory events retained between flushes; 0 selects the default.
  iree_host_size_t memory_event_capacity;

  // Maximum command region events retained between flushes; 0 selects the
  // default.
  iree_host_size_t command_region_event_capacity;
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

  // Number of execution tiles represented by this span, or 0 when absent.
  uint64_t tile_count;

  // Sum of per-tile execution durations in nanoseconds. This may exceed the
  // span duration when tiles execute concurrently.
  int64_t tile_duration_sum_ns;

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

// Command-buffer region data used to append one command region event record.
typedef struct iree_hal_local_profile_command_region_event_info_t {
  // Flags describing command region transition properties.
  iree_hal_profile_command_region_event_flags_t flags;

  // Queue metadata identity shared by the appended record.
  iree_hal_local_profile_queue_scope_t scope;

  // Queue submission epoch containing this region, or 0 when absent.
  uint64_t submission_id;

  // Session-local command-buffer identifier.
  uint64_t command_buffer_id;

  // Scheduler-visible command-buffer region that completed.
  struct {
    // Producer-defined block sequence containing |index|.
    uint32_t block_sequence;
    // Producer-defined execution epoch active while this region was claimable.
    uint32_t epoch;
    // Producer-defined region index within |block_sequence|.
    int32_t index;
    // Number of encoded work commands in this region.
    uint32_t dispatch_count;
    // Initial execution tile count observed when this region was published.
    uint32_t tile_count;
    // Producer-defined worker-width bucket for this region.
    uint32_t width_bucket;
    // Producer-defined lookahead worker-width bucket for following regions.
    uint32_t lookahead_width_bucket;
    // Number of region drain attempts that executed one or more tiles.
    uint32_t useful_drain_count;
    // Number of region drain attempts that found no claimable tiles.
    uint32_t no_work_drain_count;
    // Tail no-work observations collected after checking region work commands.
    struct {
      // Number of active-region drains that found no claimable tile.
      uint32_t count;
      // Unfinished tile counts observed by tail no-work drains.
      struct {
        // Minimum unfinished tile count observed, or 0 when absent.
        uint32_t min;
        // Maximum unfinished tile count observed, or 0 when absent.
        uint32_t max;
        // Power-of-two bucket counts. See
        // IREE_HAL_PROFILE_COMMAND_REGION_REMAINING_TILE_BUCKET_COUNT.
        uint32_t bucket_counts
            [IREE_HAL_PROFILE_COMMAND_REGION_REMAINING_TILE_BUCKET_COUNT];
      } remaining_tiles;
      // IREE monotonic host timestamp when the first drain began, or 0.
      iree_time_t first_start_host_time_ns;
      // IREE monotonic host timestamp when the last drain ended, or 0.
      iree_time_t last_end_host_time_ns;
      // Accumulated region-relative time values for tail no-work drains.
      struct {
        // Sum of no-work drain start offsets from the region start.
        iree_time_t start_offset_ns;
        // Sum of no-work drain durations.
        iree_time_t drain_duration_ns;
      } time_sums;
    } tail_no_work;
    // IREE monotonic host timestamp when the first useful drain began, or 0.
    iree_time_t first_useful_drain_start_host_time_ns;
    // IREE monotonic host timestamp when the last useful drain ended, or 0.
    iree_time_t last_useful_drain_end_host_time_ns;
    // IREE monotonic host timestamp when the region became claimable.
    iree_time_t start_host_time_ns;
    // IREE monotonic host timestamp when the region completed.
    iree_time_t end_host_time_ns;
  } command_region;

  // Scheduler-visible command-buffer region published by this transition.
  struct {
    // Following region index, or -1 when the transition is terminal.
    int32_t index;
    // Initial execution tile count observed for the following region.
    uint32_t tile_count;
    // Producer-defined worker-width bucket for the following region.
    uint32_t width_bucket;
    // Producer-defined lookahead worker-width bucket for following regions.
    uint32_t lookahead_width_bucket;
  } next_command_region;

  // Scheduler wake state associated with the transition.
  struct {
    // Number of workers available to the producer's region scheduler.
    uint32_t worker_count;
    // Wake budget active before the transition, or 0 when unavailable.
    int32_t old_wake_budget;
    // Wake budget selected for the following region, or 0 when unavailable.
    int32_t new_wake_budget;
    // Additional wake credits published for the following region.
    int32_t wake_delta;
  } scheduler;

  // Warm-worker retention behavior observed while this region was active.
  struct {
    // No-work drains that kept the process active after observing advancement.
    uint32_t keep_active_count;
    // No-work drains that explicitly republished process activity.
    uint32_t publish_keep_active_count;
    // No-work drains that waited warm on the process retention epoch.
    uint32_t keep_warm_count;
  } retention;
} iree_hal_local_profile_command_region_event_info_t;

// Returns default command region event append data.
static inline iree_hal_local_profile_command_region_event_info_t
iree_hal_local_profile_command_region_event_info_default(void) {
  iree_hal_local_profile_command_region_event_info_t info;
  memset(&info, 0, sizeof(info));
  info.scope = iree_hal_local_profile_queue_scope_default();
  info.command_region.index = -1;
  info.next_command_region.index = -1;
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

// Emits executable/export metadata for |executable| once per recorder session.
//
// Returns OK without work when executable metadata is not enabled. This is a
// cold-path helper for queue submission/replay sites that produce events with
// executable ids and need offline tools to resolve those ids into export names.
iree_status_t iree_hal_local_profile_recorder_record_executable(
    iree_hal_local_profile_recorder_t* recorder,
    iree_hal_executable_t* executable);

// Emits command-buffer and operation metadata once per recorder session.
//
// Returns OK without work when metadata is not enabled. The caller owns
// |command_buffer| and |operations|; the recorder does not retain them.
iree_status_t iree_hal_local_profile_recorder_record_command_buffer(
    iree_hal_local_profile_recorder_t* recorder,
    const iree_hal_profile_command_buffer_record_t* command_buffer,
    iree_host_size_t operation_count,
    const iree_hal_profile_command_operation_record_t* operations);

// Appends one host-timestamped queue event to |recorder|.
//
// |out_event_id| may be NULL. When provided it receives the assigned event id,
// or 0 if queue events were not requested or the event ring was full. Ring
// capacity pressure is reported later as truncated chunks during flush.
void iree_hal_local_profile_recorder_append_queue_event(
    iree_hal_local_profile_recorder_t* recorder,
    const iree_hal_local_profile_queue_event_info_t* event_info,
    uint64_t* out_event_id);

// Appends one host execution span to |recorder|.
//
// |out_event_id| may be NULL. When provided it receives the assigned event id,
// or 0 if host execution events were not requested or the event ring was full.
// Ring capacity pressure is reported later as truncated chunks during flush.
void iree_hal_local_profile_recorder_append_host_execution_event(
    iree_hal_local_profile_recorder_t* recorder,
    const iree_hal_local_profile_host_execution_event_info_t* event_info,
    uint64_t* out_event_id);

// Appends one host-timestamped command-buffer region event to |recorder|.
//
// |out_event_id| may be NULL. When provided it receives the assigned event id,
// or 0 if command region events were not requested or the event ring was full.
// Ring capacity pressure is reported later as truncated chunks during flush.
void iree_hal_local_profile_recorder_append_command_region_event(
    iree_hal_local_profile_recorder_t* recorder,
    const iree_hal_local_profile_command_region_event_info_t* event_info,
    uint64_t* out_event_id);

// Appends one host-timestamped memory lifecycle event to |recorder|.
//
// |event| is copied into the recorder. The recorder overwrites record_length
// and event_id and samples host_time_ns when it is zero. Queue-operation memory
// events must provide a valid physical device and queue ordinal. |out_event_id|
// may be NULL. When provided it receives the assigned event id, or 0 if memory
// events were not requested or the event ring was full. Ring capacity pressure
// is reported later as truncated chunks during flush.
void iree_hal_local_profile_recorder_append_memory_event(
    iree_hal_local_profile_recorder_t* recorder,
    const iree_hal_profile_memory_event_t* event, uint64_t* out_event_id);

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
