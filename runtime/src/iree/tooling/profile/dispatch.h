// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOOLING_PROFILE_DISPATCH_H_
#define IREE_TOOLING_PROFILE_DISPATCH_H_

#include "iree/tooling/profile/common.h"

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

typedef struct iree_profile_dispatch_export_t {
  // Producer-local executable identifier owning this export.
  uint64_t executable_id;
  // Flags specifying which optional export fields are populated.
  iree_hal_profile_executable_export_flags_t flags;
  // Export ordinal used by dispatch event records.
  uint32_t export_ordinal;
  // Number of HAL ABI constant words expected by this export.
  uint32_t constant_count;
  // Number of HAL ABI binding pointer slots expected by this export.
  uint32_t binding_count;
  // Number of reflected parameters associated with this export.
  uint32_t parameter_count;
  // Static workgroup size for each dispatch dimension.
  uint32_t workgroup_size[3];
  // Deterministic executable-export identity hash words when present in
  // |flags|.
  uint64_t pipeline_hash[2];
  // Borrowed export name from the mapped profile bundle.
  iree_string_view_t name;
} iree_profile_dispatch_export_t;

typedef struct iree_profile_dispatch_executable_t {
  // Immutable executable metadata record borrowed from the profile bundle.
  iree_hal_profile_executable_record_t record;
} iree_profile_dispatch_executable_t;

typedef struct iree_profile_dispatch_command_buffer_t {
  // Immutable command-buffer metadata record borrowed from the profile bundle.
  iree_hal_profile_command_buffer_record_t record;
} iree_profile_dispatch_command_buffer_t;

typedef struct iree_profile_dispatch_command_operation_t {
  // Immutable command-operation metadata record copied from the profile bundle.
  iree_hal_profile_command_operation_record_t record;
} iree_profile_dispatch_command_operation_t;

typedef struct iree_profile_dispatch_queue_t {
  // Immutable queue metadata record borrowed from the profile bundle.
  iree_hal_profile_queue_record_t record;
} iree_profile_dispatch_queue_t;

typedef struct iree_profile_dispatch_queue_event_t {
  // Immutable queue event record copied from the profile bundle.
  iree_hal_profile_queue_event_t record;
} iree_profile_dispatch_queue_event_t;

typedef struct iree_profile_dispatch_queue_device_event_t {
  // Immutable queue device event record copied from the profile bundle.
  iree_hal_profile_queue_device_event_t record;
} iree_profile_dispatch_queue_device_event_t;

typedef struct iree_profile_dispatch_device_t {
  // Session-local physical device ordinal.
  uint32_t physical_device_ordinal;
  // Number of clock-correlation samples seen for this physical device.
  uint64_t clock_sample_count;
  // First clock-correlation sample seen for this physical device.
  iree_hal_profile_clock_correlation_record_t first_clock_sample;
  // Last clock-correlation sample seen for this physical device.
  iree_hal_profile_clock_correlation_record_t last_clock_sample;
} iree_profile_dispatch_device_t;

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
  double total_ticks;
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
  double total_ticks;
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
  double total_ticks;
} iree_profile_dispatch_queue_aggregate_t;

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

typedef struct iree_profile_dispatch_context_t {
  // Host allocator used for dynamic side tables and aggregate rows.
  iree_allocator_t host_allocator;
  // Dynamic array of executable side-table entries.
  iree_profile_dispatch_executable_t* executables;
  // Number of valid entries in |executables|.
  iree_host_size_t executable_count;
  // Capacity of |executables| in entries.
  iree_host_size_t executable_capacity;
  // Dynamic array of executable export side-table entries.
  iree_profile_dispatch_export_t* exports;
  // Number of valid entries in |exports|.
  iree_host_size_t export_count;
  // Capacity of |exports| in entries.
  iree_host_size_t export_capacity;
  // Dynamic array of command-buffer side-table entries.
  iree_profile_dispatch_command_buffer_t* command_buffers;
  // Number of valid entries in |command_buffers|.
  iree_host_size_t command_buffer_count;
  // Capacity of |command_buffers| in entries.
  iree_host_size_t command_buffer_capacity;
  // Dynamic array of command-operation side-table entries.
  iree_profile_dispatch_command_operation_t* command_operations;
  // Number of valid entries in |command_operations|.
  iree_host_size_t command_operation_count;
  // Capacity of |command_operations| in entries.
  iree_host_size_t command_operation_capacity;
  // Dynamic array of queue side-table entries.
  iree_profile_dispatch_queue_t* queues;
  // Number of valid entries in |queues|.
  iree_host_size_t queue_count;
  // Capacity of |queues| in entries.
  iree_host_size_t queue_capacity;
  // Dynamic array of per-physical-device clock samples.
  iree_profile_dispatch_device_t* devices;
  // Number of valid entries in |devices|.
  iree_host_size_t device_count;
  // Capacity of |devices| in entries.
  iree_host_size_t device_capacity;
  // Dynamic array of aggregate dispatch rows.
  iree_profile_dispatch_aggregate_t* aggregates;
  // Number of valid entries in |aggregates|.
  iree_host_size_t aggregate_count;
  // Capacity of |aggregates| in entries.
  iree_host_size_t aggregate_capacity;
  // Dynamic array of command-buffer execution aggregate rows.
  iree_profile_dispatch_command_aggregate_t* command_aggregates;
  // Number of valid entries in |command_aggregates|.
  iree_host_size_t command_aggregate_count;
  // Capacity of |command_aggregates| in entries.
  iree_host_size_t command_aggregate_capacity;
  // Dynamic array of queue submission aggregate rows.
  iree_profile_dispatch_queue_aggregate_t* queue_aggregates;
  // Number of valid entries in |queue_aggregates|.
  iree_host_size_t queue_aggregate_count;
  // Capacity of |queue_aggregates| in entries.
  iree_host_size_t queue_aggregate_capacity;
  // Dynamic array of queue operation event rows.
  iree_profile_dispatch_queue_event_t* queue_events;
  // Number of valid entries in |queue_events|.
  iree_host_size_t queue_event_count;
  // Capacity of |queue_events| in entries.
  iree_host_size_t queue_event_capacity;
  // Dynamic array of device-timestamped queue operation event rows.
  iree_profile_dispatch_queue_device_event_t* queue_device_events;
  // Number of valid entries in |queue_device_events|.
  iree_host_size_t queue_device_event_count;
  // Capacity of |queue_device_events| in entries.
  iree_host_size_t queue_device_event_capacity;
  // Largest valid dispatch events observed while applying the active filter.
  iree_profile_dispatch_top_event_t
      top_dispatches[IREE_PROFILE_DISPATCH_TOP_EVENT_COUNT];
  // Number of valid entries in |top_dispatches|.
  iree_host_size_t top_dispatch_count;
  // Total queue operation records parsed before filtering.
  uint64_t total_queue_event_count;
  // Queue operation records matched by the active filter.
  uint64_t matched_queue_event_count;
  // Device-timestamped queue operation records parsed before filtering.
  uint64_t total_queue_device_event_count;
  // Device-timestamped queue operation records matched by the active filter.
  uint64_t matched_queue_device_event_count;
  // Total dispatch records parsed before filtering.
  uint64_t total_dispatch_count;
  // Dispatch records matched by the active filter.
  uint64_t matched_dispatch_count;
  // Matched dispatch records with valid timestamps.
  uint64_t valid_dispatch_count;
  // Matched dispatch records with missing or reversed timestamps.
  uint64_t invalid_dispatch_count;
} iree_profile_dispatch_context_t;

// Initializes |out_context| for dispatch, queue, and executable aggregation.
void iree_profile_dispatch_context_initialize(
    iree_allocator_t host_allocator,
    iree_profile_dispatch_context_t* out_context);

// Releases dynamic arrays owned by |context|.
void iree_profile_dispatch_context_deinitialize(
    iree_profile_dispatch_context_t* context);

// Returns the physical-device metadata row for |physical_device_ordinal|.
const iree_profile_dispatch_device_t* iree_profile_dispatch_find_device(
    const iree_profile_dispatch_context_t* context,
    uint32_t physical_device_ordinal);

// Fits a linear device-tick to nanosecond conversion for |device|.
bool iree_profile_dispatch_device_try_fit_clock(
    const iree_profile_dispatch_device_t* device, double* out_ns_per_tick,
    double* out_tick_frequency_hz);

// Formats an executable export key into |numeric_buffer| when unnamed.
iree_string_view_t iree_profile_dispatch_format_export_key(
    const iree_profile_dispatch_export_t* export_info,
    uint32_t physical_device_ordinal, char* numeric_buffer,
    iree_host_size_t numeric_buffer_capacity);

// Resolves an executable/export pair into a human-readable dispatch key.
iree_status_t iree_profile_dispatch_resolve_key(
    const iree_profile_dispatch_context_t* context,
    uint32_t physical_device_ordinal, uint64_t executable_id,
    uint32_t export_ordinal, char* numeric_buffer,
    iree_host_size_t numeric_buffer_capacity, iree_string_view_t* out_key);

// Returns true when |filter| matches |key|, treating an empty filter as
// match-all.
bool iree_profile_dispatch_key_matches(iree_string_view_t key,
                                       iree_string_view_t filter);

// Processes one metadata record into the dispatch context side tables.
iree_status_t iree_profile_dispatch_process_metadata_record(
    iree_profile_dispatch_context_t* context,
    const iree_hal_profile_file_record_t* record);

// Processes one event record into dispatch/queue aggregates.
iree_status_t iree_profile_dispatch_process_events_record(
    iree_profile_dispatch_context_t* context,
    const iree_hal_profile_file_record_t* record, iree_string_view_t filter,
    iree_profile_projection_mode_t projection_mode, int64_t id_filter,
    bool emit_events, FILE* file);

// Processes host-timestamped queue operation records into |context|.
iree_status_t iree_profile_dispatch_process_queue_event_records(
    iree_profile_dispatch_context_t* context,
    const iree_hal_profile_file_record_t* record, iree_string_view_t filter,
    int64_t id_filter);

// Returns the unsigned distance between two device timestamp ticks.
double iree_profile_dispatch_span_ticks(uint64_t earliest_start_tick,
                                        uint64_t latest_end_tick);

// Returns the stable text name for a command operation type.
const char* iree_profile_command_operation_type_name(
    iree_hal_profile_command_operation_type_t type);

// Returns the stable text name for a queue event type.
const char* iree_profile_queue_event_type_name(
    iree_hal_profile_queue_event_type_t type);

// Returns the stable text name for a queue dependency strategy.
const char* iree_profile_queue_dependency_strategy_name(
    iree_hal_profile_queue_dependency_strategy_t strategy);

// Returns the stable text name for a profile event relationship type.
const char* iree_profile_event_relationship_type_name(
    iree_hal_profile_event_relationship_type_t type);

// Returns the stable text name for a profile event relationship endpoint type.
const char* iree_profile_event_endpoint_type_name(
    iree_hal_profile_event_endpoint_type_t type);

// Resolves a command operation into a human-readable operation key.
iree_status_t iree_profile_command_operation_resolve_key(
    const iree_profile_dispatch_context_t* context,
    const iree_hal_profile_command_operation_record_t* operation,
    char* numeric_buffer, iree_host_size_t numeric_buffer_capacity,
    iree_string_view_t* out_key);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_TOOLING_PROFILE_DISPATCH_H_
