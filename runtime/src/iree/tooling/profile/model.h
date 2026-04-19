// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOOLING_PROFILE_MODEL_H_
#define IREE_TOOLING_PROFILE_MODEL_H_

#include "iree/tooling/profile/common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Executable export metadata indexed by executable id and export ordinal.
typedef struct iree_profile_model_export_t {
  // Session-local executable identifier owning this export.
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
} iree_profile_model_export_t;

// Executable metadata indexed by session-local executable id.
typedef struct iree_profile_model_executable_t {
  // Immutable executable metadata record borrowed from the profile bundle.
  iree_hal_profile_executable_record_t record;
} iree_profile_model_executable_t;

// Command-buffer metadata indexed by session-local command-buffer id.
typedef struct iree_profile_model_command_buffer_t {
  // Immutable command-buffer metadata record borrowed from the profile bundle.
  iree_hal_profile_command_buffer_record_t record;
  // First command-operation row for this command buffer, or IREE_HOST_SIZE_MAX.
  iree_host_size_t first_operation_index;
  // Last command-operation row for this command buffer, or IREE_HOST_SIZE_MAX.
  iree_host_size_t last_operation_index;
  // Number of command-operation rows linked to this command buffer.
  iree_host_size_t operation_count;
} iree_profile_model_command_buffer_t;

// Command-operation metadata indexed by command-buffer id and command index.
typedef struct iree_profile_model_command_operation_t {
  // Immutable command-operation metadata record copied from the profile bundle.
  iree_hal_profile_command_operation_record_t record;
  // Next command-operation row in the owning command buffer, or
  // IREE_HOST_SIZE_MAX.
  iree_host_size_t next_operation_index;
} iree_profile_model_command_operation_t;

// Queue metadata indexed by physical device, queue ordinal, and stream id.
typedef struct iree_profile_model_queue_t {
  // Immutable queue metadata record borrowed from the profile bundle.
  iree_hal_profile_queue_record_t record;
} iree_profile_model_queue_t;

// Per-physical-device clock-correlation state.
typedef struct iree_profile_model_device_t {
  // Session-local physical device ordinal.
  uint32_t physical_device_ordinal;
  // Number of clock-correlation samples seen for this physical device.
  uint64_t clock_sample_count;
  // First clock-correlation sample seen for this physical device.
  iree_hal_profile_clock_correlation_record_t first_clock_sample;
  // Last clock-correlation sample seen for this physical device.
  iree_hal_profile_clock_correlation_record_t last_clock_sample;
} iree_profile_model_device_t;

// Host clock domain selected as the target for a device-clock fit.
typedef uint32_t iree_profile_model_clock_time_domain_t;
enum iree_profile_model_clock_time_domain_e {
  IREE_PROFILE_MODEL_CLOCK_TIME_DOMAIN_HOST_CPU_TIMESTAMP_NS = 0u,
  IREE_PROFILE_MODEL_CLOCK_TIME_DOMAIN_IREE_HOST_TIME_NS = 1u,
};

// Exact linear fit between a device clock and one host nanosecond clock.
typedef struct iree_profile_model_clock_fit_t {
  // Target host clock domain used by |first_time_ns| and |last_time_ns|.
  iree_profile_model_clock_time_domain_t time_domain;
  // First clock-correlation sample id used for the fit.
  uint64_t first_sample_id;
  // Last clock-correlation sample id used for the fit.
  uint64_t last_sample_id;
  // First device tick used for the fit.
  uint64_t first_device_tick;
  // Last device tick used for the fit.
  uint64_t last_device_tick;
  // First host timestamp used for the fit.
  int64_t first_time_ns;
  // Last host timestamp used for the fit.
  int64_t last_time_ns;
  // Device tick distance between the first and last samples.
  uint64_t device_tick_span;
  // Host nanosecond distance between the first and last samples.
  uint64_t time_span_ns;
} iree_profile_model_clock_fit_t;

// Shared profile metadata side tables.
typedef struct iree_profile_model_t {
  // Host allocator used for dynamic side tables.
  iree_allocator_t host_allocator;
  // Dynamic array of executable side-table entries.
  iree_profile_model_executable_t* executables;
  // Number of valid entries in |executables|.
  iree_host_size_t executable_count;
  // Capacity of |executables| in entries.
  iree_host_size_t executable_capacity;
  // Lookup index from executable id to |executables| entry index.
  iree_profile_index_t executable_index;
  // Dynamic array of executable export side-table entries.
  iree_profile_model_export_t* exports;
  // Number of valid entries in |exports|.
  iree_host_size_t export_count;
  // Capacity of |exports| in entries.
  iree_host_size_t export_capacity;
  // Lookup index from executable id and export ordinal to |exports| entry
  // index.
  iree_profile_index_t export_index;
  // Dynamic array of command-buffer side-table entries.
  iree_profile_model_command_buffer_t* command_buffers;
  // Number of valid entries in |command_buffers|.
  iree_host_size_t command_buffer_count;
  // Capacity of |command_buffers| in entries.
  iree_host_size_t command_buffer_capacity;
  // Lookup index from command-buffer id to |command_buffers| entry index.
  iree_profile_index_t command_buffer_index;
  // Dynamic array of command-operation side-table entries.
  iree_profile_model_command_operation_t* command_operations;
  // Number of valid entries in |command_operations|.
  iree_host_size_t command_operation_count;
  // Capacity of |command_operations| in entries.
  iree_host_size_t command_operation_capacity;
  // Dynamic array of queue side-table entries.
  iree_profile_model_queue_t* queues;
  // Number of valid entries in |queues|.
  iree_host_size_t queue_count;
  // Capacity of |queues| in entries.
  iree_host_size_t queue_capacity;
  // Lookup index from physical queue stream to |queues| entry index.
  iree_profile_index_t queue_index;
  // Dynamic array of per-physical-device clock-correlation state.
  iree_profile_model_device_t* devices;
  // Number of valid entries in |devices|.
  iree_host_size_t device_count;
  // Capacity of |devices| in entries.
  iree_host_size_t device_capacity;
  // Lookup index from physical-device ordinal to |devices| entry index.
  iree_profile_index_t device_index;
} iree_profile_model_t;

// Initializes |out_model| for profile metadata side tables.
void iree_profile_model_initialize(iree_allocator_t host_allocator,
                                   iree_profile_model_t* out_model);

// Releases dynamic arrays owned by |model|.
void iree_profile_model_deinitialize(iree_profile_model_t* model);

// Returns the physical-device model row for |physical_device_ordinal|, creating
// one if no clock-correlation sample has referenced it yet.
iree_status_t iree_profile_model_ensure_device(
    iree_profile_model_t* model, uint32_t physical_device_ordinal,
    iree_profile_model_device_t** out_device);

// Returns the physical-device model row for |physical_device_ordinal|.
const iree_profile_model_device_t* iree_profile_model_find_device(
    const iree_profile_model_t* model, uint32_t physical_device_ordinal);

// Returns the queue metadata row for a physical queue stream.
const iree_profile_model_queue_t* iree_profile_model_find_queue(
    const iree_profile_model_t* model, uint32_t physical_device_ordinal,
    uint32_t queue_ordinal, uint64_t stream_id);

// Returns the executable metadata row for |executable_id|.
const iree_profile_model_executable_t* iree_profile_model_find_executable(
    const iree_profile_model_t* model, uint64_t executable_id);

// Returns the executable export metadata row for |executable_id| and
// |export_ordinal|.
const iree_profile_model_export_t* iree_profile_model_find_export(
    const iree_profile_model_t* model, uint64_t executable_id,
    uint32_t export_ordinal);

// Returns the command-buffer metadata row for |command_buffer_id|.
const iree_profile_model_command_buffer_t*
iree_profile_model_find_command_buffer(const iree_profile_model_t* model,
                                       uint64_t command_buffer_id);

// Fits a linear device-tick to nanosecond conversion for |device|.
//
// The returned fit preserves the raw rational mapping:
//
//   time_ns = first_time_ns +
//       round((device_tick - first_device_tick) * time_span_ns /
//             device_tick_span)
//
// Consumers that need event timestamps should use
// iree_profile_model_clock_fit_map_tick so large timestamp values stay in
// integer arithmetic until the final rounded nanosecond.
bool iree_profile_model_device_try_fit_clock_exact(
    const iree_profile_model_device_t* device,
    iree_profile_model_clock_time_domain_t time_domain,
    iree_profile_model_clock_fit_t* out_fit);

// Fits a linear device-tick to driver host CPU nanosecond conversion for
// display-only aggregate statistics.
bool iree_profile_model_device_try_fit_clock(
    const iree_profile_model_device_t* device, double* out_ns_per_tick,
    double* out_tick_frequency_hz);

// Maps |device_tick| through |fit| and rounds to the nearest host nanosecond.
bool iree_profile_model_clock_fit_map_tick(
    const iree_profile_model_clock_fit_t* fit, uint64_t device_tick,
    int64_t* out_time_ns);

// Converts |device_tick_count| through |fit| and rounds to the nearest
// nanosecond duration.
bool iree_profile_model_clock_fit_scale_ticks_to_ns(
    const iree_profile_model_clock_fit_t* fit, uint64_t device_tick_count,
    int64_t* out_duration_ns);

// Returns the fitted nanoseconds per device tick as a display value.
double iree_profile_model_clock_fit_ns_per_tick(
    const iree_profile_model_clock_fit_t* fit);

// Returns the fitted device tick frequency in Hz as a display value.
double iree_profile_model_clock_fit_tick_frequency_hz(
    const iree_profile_model_clock_fit_t* fit);

// Formats an executable export key into |numeric_buffer| when unnamed.
iree_string_view_t iree_profile_model_format_export_key(
    const iree_profile_model_export_t* export_info,
    uint32_t physical_device_ordinal, char* numeric_buffer,
    iree_host_size_t numeric_buffer_capacity);

// Resolves an executable/export pair into a human-readable dispatch key.
iree_status_t iree_profile_model_resolve_dispatch_key(
    const iree_profile_model_t* model, uint32_t physical_device_ordinal,
    uint64_t executable_id, uint32_t export_ordinal, char* numeric_buffer,
    iree_host_size_t numeric_buffer_capacity, iree_string_view_t* out_key);

// Resolves a command operation into a human-readable operation key.
iree_status_t iree_profile_model_resolve_command_operation_key(
    const iree_profile_model_t* model,
    const iree_hal_profile_command_operation_record_t* operation,
    char* numeric_buffer, iree_host_size_t numeric_buffer_capacity,
    iree_string_view_t* out_key);

// Processes one metadata record into the shared profile model side tables.
iree_status_t iree_profile_model_process_metadata_record(
    iree_profile_model_t* model, const iree_hal_profile_file_record_t* record);

// Returns the unsigned distance between two device timestamp ticks.
double iree_profile_model_span_ticks(uint64_t earliest_start_tick,
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

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_TOOLING_PROFILE_MODEL_H_
