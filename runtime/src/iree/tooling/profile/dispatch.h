// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOOLING_PROFILE_DISPATCH_H_
#define IREE_TOOLING_PROFILE_DISPATCH_H_

#include "iree/tooling/profile/common.h"
#include "iree/tooling/profile/model.h"
#include "iree/tooling/profile/queue_query.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#define IREE_PROFILE_DISPATCH_TOP_EVENT_COUNT 8

typedef enum iree_profile_projection_mode_e {
  // Dispatch projection; --id matches dispatch event ids.
  IREE_PROFILE_PROJECTION_MODE_DISPATCH = 0,
  // Executable projection; --id matches executable ids.
  IREE_PROFILE_PROJECTION_MODE_EXECUTABLE = 1,
  // Command-buffer projection; --id matches command-buffer ids.
  IREE_PROFILE_PROJECTION_MODE_COMMAND = 2,
  // Queue projection; --id matches queue submission ids.
  IREE_PROFILE_PROJECTION_MODE_QUEUE = 3,
} iree_profile_projection_mode_t;

typedef struct iree_profile_dispatch_aggregate_t {
  // Session-local physical device ordinal for this aggregate row.
  uint32_t physical_device_ordinal;
  // Producer-local executable identifier for this aggregate row.
  uint64_t executable_id;
  // Export ordinal for this aggregate row.
  uint32_t export_ordinal;
  // Total dispatch records matched for this aggregate row.
  uint64_t dispatch_count;
  // Dispatch records with valid start/end timestamps.
  uint64_t valid_count;
  // Dispatch records with missing or reversed timestamps.
  uint64_t invalid_count;
  // Earliest valid dispatch start tick in this aggregate row.
  uint64_t earliest_start_tick;
  // Latest valid dispatch end tick in this aggregate row.
  uint64_t latest_end_tick;
  // Minimum valid dispatch duration in raw device ticks.
  uint64_t minimum_ticks;
  // Maximum valid dispatch duration in raw device ticks.
  uint64_t maximum_ticks;
  // Sum of valid dispatch durations in raw device ticks.
  uint64_t total_ticks;
  // Running mean of valid dispatch durations in raw device ticks.
  double mean_ticks;
  // Running sum of squares of differences from |mean_ticks|.
  double m2_ticks;
  // Last observed workgroup count for this dispatch key.
  uint32_t last_workgroup_count[3];
  // Last observed workgroup size for this dispatch key.
  uint32_t last_workgroup_size[3];
} iree_profile_dispatch_aggregate_t;

typedef struct iree_profile_dispatch_command_aggregate_t {
  // Producer-local command-buffer identifier for this aggregate row.
  uint64_t command_buffer_id;
  // Queue submission epoch containing this command-buffer execution.
  uint64_t submission_id;
  // Session-local physical device ordinal for this aggregate row.
  uint32_t physical_device_ordinal;
  // Session-local queue ordinal for this aggregate row.
  uint32_t queue_ordinal;
  // Producer-defined stream identifier for this aggregate row.
  uint64_t stream_id;
  // Total dispatch records matched for this aggregate row.
  uint64_t dispatch_count;
  // Dispatch records with valid start/end timestamps.
  uint64_t valid_count;
  // Dispatch records with missing or reversed timestamps.
  uint64_t invalid_count;
  // Earliest valid dispatch start tick in this aggregate row.
  uint64_t earliest_start_tick;
  // Latest valid dispatch end tick in this aggregate row.
  uint64_t latest_end_tick;
  // Sum of valid dispatch durations in raw device ticks.
  uint64_t total_ticks;
} iree_profile_dispatch_command_aggregate_t;

typedef struct iree_profile_dispatch_queue_aggregate_t {
  // Session-local physical device ordinal for this aggregate row.
  uint32_t physical_device_ordinal;
  // Session-local queue ordinal for this aggregate row.
  uint32_t queue_ordinal;
  // Producer-defined stream identifier for this aggregate row.
  uint64_t stream_id;
  // Queue submission epoch for this aggregate row.
  uint64_t submission_id;
  // Total dispatch records matched for this aggregate row.
  uint64_t dispatch_count;
  // Dispatch records with valid start/end timestamps.
  uint64_t valid_count;
  // Dispatch records with missing or reversed timestamps.
  uint64_t invalid_count;
  // Earliest valid dispatch start tick in this aggregate row.
  uint64_t earliest_start_tick;
  // Latest valid dispatch end tick in this aggregate row.
  uint64_t latest_end_tick;
  // Sum of valid dispatch durations in raw device ticks.
  uint64_t total_ticks;
} iree_profile_dispatch_queue_aggregate_t;

typedef struct iree_profile_host_dispatch_aggregate_t {
  // Session-local physical device ordinal for this aggregate row.
  uint32_t physical_device_ordinal;
  // Producer-local executable identifier for this aggregate row.
  uint64_t executable_id;
  // Export ordinal for this aggregate row.
  uint32_t export_ordinal;
  // Total host execution dispatch records matched for this aggregate row.
  uint64_t dispatch_count;
  // Host execution dispatch records with valid start/end timestamps.
  uint64_t valid_count;
  // Host execution dispatch records with missing or reversed timestamps.
  uint64_t invalid_count;
  // Earliest valid dispatch start time in iree_host_time_ns.
  int64_t earliest_start_host_time_ns;
  // Latest valid dispatch end time in iree_host_time_ns.
  int64_t latest_end_host_time_ns;
  // Minimum valid dispatch duration in nanoseconds.
  int64_t minimum_ns;
  // Maximum valid dispatch duration in nanoseconds.
  int64_t maximum_ns;
  // Sum of valid dispatch durations in nanoseconds.
  int64_t total_ns;
  // Sum of tile counts reported by valid dispatch spans.
  uint64_t total_tile_count;
  // Sum of valid per-tile duration totals in nanoseconds.
  int64_t total_tile_duration_sum_ns;
  // Running mean of valid dispatch durations in nanoseconds.
  double mean_ns;
  // Running sum of squares of differences from |mean_ns|.
  double m2_ns;
  // Last observed workgroup count for this dispatch key.
  uint32_t last_workgroup_count[3];
  // Last observed workgroup size for this dispatch key.
  uint32_t last_workgroup_size[3];
} iree_profile_host_dispatch_aggregate_t;

typedef struct iree_profile_host_dispatch_command_aggregate_t {
  // Producer-local command-buffer identifier for this aggregate row.
  uint64_t command_buffer_id;
  // Queue submission epoch containing this command-buffer execution.
  uint64_t submission_id;
  // Session-local physical device ordinal for this aggregate row.
  uint32_t physical_device_ordinal;
  // Session-local queue ordinal for this aggregate row.
  uint32_t queue_ordinal;
  // Producer-defined stream identifier for this aggregate row.
  uint64_t stream_id;
  // Total host execution dispatch records matched for this aggregate row.
  uint64_t dispatch_count;
  // Host execution dispatch records with valid start/end timestamps.
  uint64_t valid_count;
  // Host execution dispatch records with missing or reversed timestamps.
  uint64_t invalid_count;
  // Earliest valid dispatch start time in iree_host_time_ns.
  int64_t earliest_start_host_time_ns;
  // Latest valid dispatch end time in iree_host_time_ns.
  int64_t latest_end_host_time_ns;
  // Sum of valid dispatch durations in nanoseconds.
  int64_t total_ns;
  // Sum of tile counts reported by valid dispatch spans.
  uint64_t total_tile_count;
  // Sum of valid per-tile duration totals in nanoseconds.
  int64_t total_tile_duration_sum_ns;
} iree_profile_host_dispatch_command_aggregate_t;

typedef struct iree_profile_dispatch_top_event_t {
  // Session-local physical device ordinal for this dispatch event.
  uint32_t physical_device_ordinal;
  // Session-local queue ordinal for this dispatch event.
  uint32_t queue_ordinal;
  // Producer-defined stream identifier for this dispatch event.
  uint64_t stream_id;
  // Device timestamp delta for this dispatch event.
  uint64_t duration_ticks;
  // Dispatch event record copied from the profile bundle.
  iree_hal_profile_dispatch_event_t event;
} iree_profile_dispatch_top_event_t;

typedef struct iree_profile_dispatch_event_row_t {
  // Profile chunk containing |event| and valid only for the callback duration.
  const iree_hal_profile_file_record_t* file_record;
  // Dispatch event record valid only for the callback duration.
  const iree_hal_profile_dispatch_event_t* event;
  // Resolved executable/export key valid only for the callback duration.
  iree_string_view_t key;
  // Device clock fit valid only when |has_clock_fit| is true.
  const iree_profile_model_clock_fit_t* clock_fit;
  // True when |clock_fit| can translate raw device ticks to nanoseconds.
  bool has_clock_fit;
} iree_profile_dispatch_event_row_t;

typedef struct iree_profile_host_dispatch_event_row_t {
  // Host dispatch event record valid only for the callback duration.
  const iree_hal_profile_host_execution_event_t* event;
  // Resolved executable/export key valid only for the callback duration.
  iree_string_view_t key;
} iree_profile_host_dispatch_event_row_t;

typedef iree_status_t (*iree_profile_dispatch_event_callback_fn_t)(
    void* user_data, const iree_profile_dispatch_event_row_t* row);

typedef iree_status_t (*iree_profile_host_dispatch_event_callback_fn_t)(
    void* user_data, const iree_profile_host_dispatch_event_row_t* row);

typedef struct iree_profile_dispatch_event_callback_t {
  // Optional callback invoked for each matched dispatch event row.
  iree_profile_dispatch_event_callback_fn_t fn;
  // Optional callback invoked for each matched host dispatch span row.
  iree_profile_host_dispatch_event_callback_fn_t host_fn;
  // Opaque user data passed to callback functions.
  void* user_data;
} iree_profile_dispatch_event_callback_t;

typedef struct iree_profile_dispatch_context_t {
  // Host allocator used for dynamic aggregate rows.
  iree_allocator_t host_allocator;
  // Shared metadata side tables.
  iree_profile_model_t model;
  // Filtered queue-operation query used by queue and explain projections.
  iree_profile_queue_event_query_t queue_query;
  // Dynamic array of aggregate dispatch rows.
  iree_profile_dispatch_aggregate_t* aggregates;
  // Number of valid entries in |aggregates|.
  iree_host_size_t aggregate_count;
  // Capacity of |aggregates| in entries.
  iree_host_size_t aggregate_capacity;
  // Lookup index from executable export key to |aggregates| entry index.
  iree_profile_index_t aggregate_index;
  // Dynamic array of command-buffer execution aggregate rows.
  iree_profile_dispatch_command_aggregate_t* command_aggregates;
  // Number of valid entries in |command_aggregates|.
  iree_host_size_t command_aggregate_count;
  // Capacity of |command_aggregates| in entries.
  iree_host_size_t command_aggregate_capacity;
  // Lookup index from command execution key to |command_aggregates| entry
  // index.
  iree_profile_index_t command_aggregate_index;
  // Dynamic array of queue submission aggregate rows.
  iree_profile_dispatch_queue_aggregate_t* queue_aggregates;
  // Number of valid entries in |queue_aggregates|.
  iree_host_size_t queue_aggregate_count;
  // Capacity of |queue_aggregates| in entries.
  iree_host_size_t queue_aggregate_capacity;
  // Lookup index from queue submission key to |queue_aggregates| entry index.
  iree_profile_index_t queue_aggregate_index;
  // Dynamic array of aggregate host dispatch rows.
  iree_profile_host_dispatch_aggregate_t* host_dispatch_aggregates;
  // Number of valid entries in |host_dispatch_aggregates|.
  iree_host_size_t host_dispatch_aggregate_count;
  // Capacity of |host_dispatch_aggregates| in entries.
  iree_host_size_t host_dispatch_aggregate_capacity;
  // Lookup index from executable export key to |host_dispatch_aggregates|.
  iree_profile_index_t host_dispatch_aggregate_index;
  // Dynamic array of command-buffer host dispatch aggregate rows.
  iree_profile_host_dispatch_command_aggregate_t* host_command_aggregates;
  // Number of valid entries in |host_command_aggregates|.
  iree_host_size_t host_command_aggregate_count;
  // Capacity of |host_command_aggregates| in entries.
  iree_host_size_t host_command_aggregate_capacity;
  // Lookup index from command execution key to |host_command_aggregates|.
  iree_profile_index_t host_command_aggregate_index;
  // Largest valid dispatch events observed while applying the active filter.
  iree_profile_dispatch_top_event_t
      top_dispatches[IREE_PROFILE_DISPATCH_TOP_EVENT_COUNT];
  // Number of valid entries in |top_dispatches|.
  iree_host_size_t top_dispatch_count;
  // Total dispatch records parsed before filtering.
  uint64_t total_dispatch_count;
  // Dispatch records matched by the active filter.
  uint64_t matched_dispatch_count;
  // Matched dispatch records with valid timestamps.
  uint64_t valid_dispatch_count;
  // Matched dispatch records with missing or reversed timestamps.
  uint64_t invalid_dispatch_count;
  // Total host execution dispatch records parsed before filtering.
  uint64_t total_host_dispatch_count;
  // Host execution dispatch records matched by the active filter.
  uint64_t matched_host_dispatch_count;
  // Matched host execution dispatch records with valid timestamps.
  uint64_t valid_host_dispatch_count;
  // Matched host execution dispatch records with missing or reversed
  // timestamps.
  uint64_t invalid_host_dispatch_count;
} iree_profile_dispatch_context_t;

// Initializes |out_context| for dispatch, queue, and executable aggregation.
void iree_profile_dispatch_context_initialize(
    iree_allocator_t host_allocator,
    iree_profile_dispatch_context_t* out_context);

// Releases dynamic arrays owned by |context|.
void iree_profile_dispatch_context_deinitialize(
    iree_profile_dispatch_context_t* context);

// Processes dispatch, host execution, and queue event records from one profile
// file record.
//
// When |aggregate_events| is true, matched dispatches update aggregate arrays
// owned by |context|. When a callback field is non-NULL, each matched event in
// that timing domain is delivered before returning from this call.
iree_status_t iree_profile_dispatch_process_events_record(
    iree_profile_dispatch_context_t* context,
    const iree_hal_profile_file_record_t* record, iree_string_view_t filter,
    iree_profile_projection_mode_t projection_mode, int64_t id_filter,
    bool aggregate_events,
    iree_profile_dispatch_event_callback_t event_callback);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_TOOLING_PROFILE_DISPATCH_H_
