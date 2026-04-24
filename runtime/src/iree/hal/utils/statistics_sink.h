// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_UTILS_STATISTICS_SINK_H_
#define IREE_HAL_UTILS_STATISTICS_SINK_H_

#include <stdint.h>
#include <stdio.h>

#include "iree/base/api.h"
#include "iree/hal/profile_schema.h"
#include "iree/hal/profile_sink.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_profile_statistics_sink_t
//===----------------------------------------------------------------------===//

// In-memory aggregate reducer for HAL profiling chunks.
//
// The statistics sink is a utility view over HAL profiling data, not a durable
// profile format and not an independent device statistics API. Tooling can pass
// the sink to iree_hal_device_profiling_begin, let the selected HAL producer
// emit ordinary profile chunks, and then iterate the aggregate rows after
// profiling has ended.
typedef struct iree_hal_profile_statistics_sink_t
    iree_hal_profile_statistics_sink_t;

// Kind of aggregate statistic represented by one row.
typedef uint32_t iree_hal_profile_statistics_row_type_t;
enum iree_hal_profile_statistics_row_type_e {
  IREE_HAL_PROFILE_STATISTICS_ROW_TYPE_NONE = 0u,

  // Device-tick dispatch samples grouped by physical device and
  // executable/export pair.
  IREE_HAL_PROFILE_STATISTICS_ROW_TYPE_DISPATCH_EXPORT = 1u,

  // Device-tick dispatch samples grouped by physical device and command buffer.
  IREE_HAL_PROFILE_STATISTICS_ROW_TYPE_DISPATCH_COMMAND_BUFFER = 2u,

  // Device-tick dispatch samples grouped by physical device, command buffer,
  // and command index.
  IREE_HAL_PROFILE_STATISTICS_ROW_TYPE_DISPATCH_COMMAND_OPERATION = 3u,

  // Device-tick queue operation samples grouped by physical queue and queue
  // operation type.
  IREE_HAL_PROFILE_STATISTICS_ROW_TYPE_QUEUE_DEVICE_OPERATION = 4u,

  // Host-timestamped queue submissions grouped by physical queue and queue
  // operation type.
  IREE_HAL_PROFILE_STATISTICS_ROW_TYPE_QUEUE_HOST_OPERATION = 5u,

  // Host execution samples grouped by physical device and executable/export
  // pair.
  IREE_HAL_PROFILE_STATISTICS_ROW_TYPE_HOST_EXECUTION_EXPORT = 6u,

  // Host execution samples grouped by physical device and command buffer.
  IREE_HAL_PROFILE_STATISTICS_ROW_TYPE_HOST_EXECUTION_COMMAND_BUFFER = 7u,

  // Host execution samples grouped by physical device, command buffer, and
  // command index.
  IREE_HAL_PROFILE_STATISTICS_ROW_TYPE_HOST_EXECUTION_COMMAND_OPERATION = 8u,

  // Host execution samples grouped by physical queue and queue operation type.
  IREE_HAL_PROFILE_STATISTICS_ROW_TYPE_HOST_EXECUTION_QUEUE_OPERATION = 9u,

  // Memory lifecycle events grouped by physical device, queue, and memory event
  // type.
  IREE_HAL_PROFILE_STATISTICS_ROW_TYPE_MEMORY_LIFECYCLE = 10u,
};

// Time domain used by aggregate timing fields in a statistics row.
typedef uint32_t iree_hal_profile_statistics_time_domain_t;
enum iree_hal_profile_statistics_time_domain_e {
  IREE_HAL_PROFILE_STATISTICS_TIME_DOMAIN_NONE = 0u,

  // Timing values are raw device ticks.
  IREE_HAL_PROFILE_STATISTICS_TIME_DOMAIN_DEVICE_TICK = 1u,

  // Timing values are nonnegative nanoseconds from iree_time_now().
  IREE_HAL_PROFILE_STATISTICS_TIME_DOMAIN_IREE_HOST_TIME_NS = 2u,
};

// Bitfield specifying which optional aggregate fields are populated.
typedef uint32_t iree_hal_profile_statistics_row_flags_t;
enum iree_hal_profile_statistics_row_flag_bits_t {
  IREE_HAL_PROFILE_STATISTICS_ROW_FLAG_NONE = 0u,

  // The row carries valid start/end/duration aggregate timing fields.
  IREE_HAL_PROFILE_STATISTICS_ROW_FLAG_TIMING = 1u << 0,

  // The row carries byte totals in |payload_bytes|.
  IREE_HAL_PROFILE_STATISTICS_ROW_FLAG_PAYLOAD_BYTES = 1u << 1,

  // The row carries queue/command operation totals in |operation_count|.
  IREE_HAL_PROFILE_STATISTICS_ROW_FLAG_OPERATION_COUNT = 1u << 2,

  // The row carries tile totals in |tile_count| and |tile_duration_sum_ns|.
  IREE_HAL_PROFILE_STATISTICS_ROW_FLAG_TILE_TOTALS = 1u << 3,
};

// Aggregate statistic row.
//
// Rows are borrowed from the sink and remain valid until the sink is mutated,
// reset, or released. Absent key fields use the default sentinel from the
// source profile record schema: zero for optional identifiers and UINT32_MAX
// for optional ordinals.
typedef struct iree_hal_profile_statistics_row_t {
  // Kind of aggregate statistic represented by this row.
  iree_hal_profile_statistics_row_type_t row_type;
  // Time domain used by timing aggregate fields.
  iree_hal_profile_statistics_time_domain_t time_domain;
  // Flags specifying which aggregate fields are populated.
  iree_hal_profile_statistics_row_flags_t flags;
  // Session-local physical device ordinal associated with this row.
  uint32_t physical_device_ordinal;
  // Session-local queue ordinal associated with this row, or UINT32_MAX.
  uint32_t queue_ordinal;
  // Queue or memory event type for operation/lifecycle rows, or zero.
  uint32_t event_type;
  // Session-local executable identifier, or zero.
  uint64_t executable_id;
  // Session-local command-buffer identifier, or zero.
  uint64_t command_buffer_id;
  // Executable export ordinal, or UINT32_MAX.
  uint32_t export_ordinal;
  // Command ordinal within a command buffer, or UINT32_MAX.
  uint32_t command_index;
  // Number of source samples accumulated into this row.
  uint64_t sample_count;
  // Number of source samples rejected from timing aggregates.
  uint64_t invalid_sample_count;
  // Sum of source operation counts when available.
  uint64_t operation_count;
  // Sum of source payload byte lengths when available.
  uint64_t payload_bytes;
  // Sum of source tile counts when available.
  uint64_t tile_count;
  // Sum of source per-tile durations in nanoseconds when available.
  uint64_t tile_duration_sum_ns;
  // Earliest valid source start time in |time_domain| units.
  uint64_t first_start_time;
  // Latest valid source end time in |time_domain| units.
  uint64_t last_end_time;
  // Sum of valid source durations in |time_domain| units.
  uint64_t total_duration;
  // Minimum valid source duration in |time_domain| units.
  uint64_t minimum_duration;
  // Maximum valid source duration in |time_domain| units.
  uint64_t maximum_duration;
} iree_hal_profile_statistics_row_t;

// Function invoked for one aggregate statistics row.
typedef iree_status_t(
    IREE_API_PTR* iree_hal_profile_statistics_row_callback_fn_t)(
    void* user_data, const iree_hal_profile_statistics_row_t* row);

typedef struct iree_hal_profile_statistics_row_callback_t {
  // Function invoked for each aggregate statistics row.
  iree_hal_profile_statistics_row_callback_fn_t fn;
  // User data passed to |fn|.
  void* user_data;
} iree_hal_profile_statistics_row_callback_t;

// Creates an aggregate statistics sink.
IREE_API_EXPORT iree_status_t iree_hal_profile_statistics_sink_create(
    iree_allocator_t host_allocator,
    iree_hal_profile_statistics_sink_t** out_sink);

// Returns |sink| as the HAL profile sink interface it implements.
IREE_API_EXPORT iree_hal_profile_sink_t* iree_hal_profile_statistics_sink_base(
    iree_hal_profile_statistics_sink_t* sink);

// Retains |sink| for the caller.
IREE_API_EXPORT void iree_hal_profile_statistics_sink_retain(
    iree_hal_profile_statistics_sink_t* sink);

// Releases |sink| from the caller.
IREE_API_EXPORT void iree_hal_profile_statistics_sink_release(
    iree_hal_profile_statistics_sink_t* sink);

// Returns the number of aggregate rows currently available in |sink|.
IREE_API_EXPORT iree_host_size_t iree_hal_profile_statistics_sink_row_count(
    const iree_hal_profile_statistics_sink_t* sink);

// Returns the number of source records that producers reported as dropped.
IREE_API_EXPORT uint64_t iree_hal_profile_statistics_sink_dropped_record_count(
    const iree_hal_profile_statistics_sink_t* sink);

// Iterates aggregate rows accumulated in |sink|.
IREE_API_EXPORT iree_status_t iree_hal_profile_statistics_sink_for_each_row(
    const iree_hal_profile_statistics_sink_t* sink,
    iree_hal_profile_statistics_row_callback_t callback);

// Resolves an executable/export key to a borrowed export name when metadata was
// present in the consumed profile stream.
IREE_API_EXPORT bool iree_hal_profile_statistics_sink_find_export_name(
    const iree_hal_profile_statistics_sink_t* sink, uint64_t executable_id,
    uint32_t export_ordinal, iree_string_view_t* out_name);

// Scales |duration| from |row|'s time domain to nanoseconds when possible.
//
// Host-time rows are already in nanoseconds. Device-tick rows require clock
// correlation records for the row's physical device. Returns false when no
// reliable conversion is available.
IREE_API_EXPORT bool iree_hal_profile_statistics_sink_scale_duration_to_ns(
    const iree_hal_profile_statistics_sink_t* sink,
    const iree_hal_profile_statistics_row_t* row, uint64_t duration,
    uint64_t* out_duration_ns);

// Prints a compact human-readable aggregate statistics report to |file|.
IREE_API_EXPORT iree_status_t iree_hal_profile_statistics_sink_fprint(
    FILE* file, const iree_hal_profile_statistics_sink_t* sink);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_UTILS_STATISTICS_SINK_H_
