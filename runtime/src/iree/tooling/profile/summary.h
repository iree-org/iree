// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOOLING_PROFILE_SUMMARY_H_
#define IREE_TOOLING_PROFILE_SUMMARY_H_

#include "iree/tooling/profile/common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_profile_device_summary_t {
  // Session-local physical device ordinal.
  uint32_t physical_device_ordinal;
  // Number of device metadata records seen for this ordinal.
  uint32_t device_record_count;
  // Number of queues reported in device metadata.
  uint32_t queue_count;
  // Number of queue metadata records seen for this physical device.
  uint32_t queue_record_count;
  // Number of clock-correlation samples seen for this physical device.
  uint64_t clock_sample_count;
  // First clock-correlation sample seen for this physical device.
  iree_hal_profile_clock_correlation_record_t first_clock_sample;
  // Last clock-correlation sample seen for this physical device.
  iree_hal_profile_clock_correlation_record_t last_clock_sample;
  // Minimum host bracket length observed around KFD clock-counter samples.
  int64_t minimum_clock_uncertainty_ns;
  // Maximum host bracket length observed around KFD clock-counter samples.
  int64_t maximum_clock_uncertainty_ns;
  // Number of dispatch event records seen for this physical device.
  uint64_t dispatch_event_count;
  // Number of dispatch records with unusable or reversed timestamps.
  uint64_t invalid_dispatch_event_count;
  // Sum of valid dispatch durations in raw device ticks.
  uint64_t total_dispatch_ticks;
  // Earliest valid dispatch start tick seen for this physical device.
  uint64_t earliest_dispatch_start_tick;
  // Latest valid dispatch end tick seen for this physical device.
  uint64_t latest_dispatch_end_tick;
  // Minimum valid dispatch duration in raw device ticks.
  uint64_t minimum_dispatch_ticks;
  // Maximum valid dispatch duration in raw device ticks.
  uint64_t maximum_dispatch_ticks;
} iree_profile_device_summary_t;

typedef struct iree_profile_summary_t {
  // Host allocator used for dynamic summary arrays.
  iree_allocator_t host_allocator;
  // Dynamic array of per-device summaries.
  iree_profile_device_summary_t* devices;
  // Number of valid entries in |devices|.
  iree_host_size_t device_count;
  // Capacity of |devices| in entries.
  iree_host_size_t device_capacity;
  // Total file records parsed.
  uint64_t file_record_count;
  // Session-begin records parsed.
  uint64_t session_begin_count;
  // Session-end records parsed.
  uint64_t session_end_count;
  // Chunk records parsed.
  uint64_t chunk_count;
  // Records with unknown file record types.
  uint64_t unknown_record_count;
  // Chunks with the truncated flag set.
  uint64_t truncated_chunk_count;
  // Producer-reported typed records omitted from truncated chunks.
  uint64_t dropped_record_count;
  // Device metadata chunks parsed.
  uint64_t device_chunk_count;
  // Queue metadata chunks parsed.
  uint64_t queue_chunk_count;
  // Executable metadata chunks parsed.
  uint64_t executable_chunk_count;
  // Executable records parsed.
  uint64_t executable_record_count;
  // Executable code-object metadata chunks parsed.
  uint64_t executable_code_object_chunk_count;
  // Executable code-object image records parsed.
  uint64_t executable_code_object_record_count;
  // Executable code-object image bytes referenced by metadata records.
  uint64_t executable_code_object_data_bytes;
  // Executable code-object load metadata chunks parsed.
  uint64_t executable_code_object_load_chunk_count;
  // Executable code-object load records parsed.
  uint64_t executable_code_object_load_record_count;
  // Executable export metadata chunks parsed.
  uint64_t executable_export_chunk_count;
  // Executable export records parsed.
  uint64_t executable_export_record_count;
  // Command-buffer metadata chunks parsed.
  uint64_t command_buffer_chunk_count;
  // Command-buffer records parsed.
  uint64_t command_buffer_record_count;
  // Command-operation metadata chunks parsed.
  uint64_t command_operation_chunk_count;
  // Command-operation records parsed.
  uint64_t command_operation_record_count;
  // Clock-correlation chunks parsed.
  uint64_t clock_correlation_chunk_count;
  // Dispatch event chunks parsed.
  uint64_t dispatch_event_chunk_count;
  // Queue device event chunks parsed.
  uint64_t queue_device_event_chunk_count;
  // Queue device event records parsed.
  uint64_t queue_device_event_record_count;
  // Host execution event chunks parsed.
  uint64_t host_execution_event_chunk_count;
  // Host execution event records parsed.
  uint64_t host_execution_event_record_count;
  // Host execution records with unusable or reversed timestamps.
  uint64_t invalid_host_execution_event_record_count;
  // Sum of valid host execution durations in nanoseconds.
  uint64_t total_host_execution_duration_ns;
  // Profile event relationship chunks parsed.
  uint64_t event_relationship_chunk_count;
  // Profile event relationship records parsed.
  uint64_t event_relationship_record_count;
  // Queue event chunks parsed.
  uint64_t queue_event_chunk_count;
  // Queue event records parsed.
  uint64_t queue_event_record_count;
  // Memory event chunks parsed.
  uint64_t memory_event_chunk_count;
  // Memory event records parsed.
  uint64_t memory_event_record_count;
  // Hardware counter set metadata chunks parsed.
  uint64_t counter_set_chunk_count;
  // Hardware counter set metadata records parsed.
  uint64_t counter_set_record_count;
  // Hardware counter metadata chunks parsed.
  uint64_t counter_chunk_count;
  // Hardware counter metadata records parsed.
  uint64_t counter_record_count;
  // Hardware counter sample chunks parsed.
  uint64_t counter_sample_chunk_count;
  // Hardware counter sample records parsed.
  uint64_t counter_sample_record_count;
  // Executable trace chunks parsed.
  uint64_t executable_trace_chunk_count;
  // Executable trace records parsed.
  uint64_t executable_trace_record_count;
  // Raw executable trace bytes referenced by trace records.
  uint64_t executable_trace_data_bytes;
  // Chunk records with unknown content types.
  uint64_t unknown_chunk_count;
} iree_profile_summary_t;

// Initializes |out_summary| for bundle-wide summary aggregation.
void iree_profile_summary_initialize(iree_allocator_t host_allocator,
                                     iree_profile_summary_t* out_summary);

// Releases dynamic arrays owned by |summary|.
void iree_profile_summary_deinitialize(iree_profile_summary_t* summary);

// Processes one profile file record and updates bundle-wide counters.
iree_status_t iree_profile_summary_process_record(
    iree_profile_summary_t* summary,
    const iree_hal_profile_file_record_t* record);

// Reads a profile bundle from |path| and writes a summary report to |file|.
iree_status_t iree_profile_summary_file(iree_string_view_t path,
                                        iree_string_view_t format, FILE* file,
                                        iree_allocator_t host_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_TOOLING_PROFILE_SUMMARY_H_
