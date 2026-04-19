// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_STATISTICS_H_
#define IREE_HAL_STATISTICS_H_

#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/hal/command_buffer.h"
#include "iree/hal/executable.h"
#include "iree/hal/queue.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Runtime Statistics Query Model
//===----------------------------------------------------------------------===//

// Sentinel used for optional ordinals in filters and rows.
#define IREE_HAL_STATISTICS_ORDINAL_ANY UINT32_MAX

// Sentinel used for producer-local identifiers that are unavailable.
#define IREE_HAL_STATISTICS_ID_UNAVAILABLE 0ull

// Sentinel used for numeric fields whose producer did not track a value.
#define IREE_HAL_STATISTICS_VALUE_UNAVAILABLE UINT64_MAX

// Bitfield selecting aggregate runtime statistics row families.
typedef uint64_t iree_hal_statistics_families_t;
enum iree_hal_statistics_family_bits_t {
  IREE_HAL_STATISTICS_FAMILY_NONE = 0u,

  // Device-wide aggregate rows.
  IREE_HAL_STATISTICS_FAMILY_DEVICE = 1ull << 0,

  // Queue or stream aggregate rows.
  IREE_HAL_STATISTICS_FAMILY_QUEUE = 1ull << 1,

  // Executable export aggregate rows.
  IREE_HAL_STATISTICS_FAMILY_EXECUTABLE_EXPORT = 1ull << 2,

  // Command-buffer aggregate rows.
  IREE_HAL_STATISTICS_FAMILY_COMMAND_BUFFER = 1ull << 3,

  // Device memory pressure aggregate rows.
  IREE_HAL_STATISTICS_FAMILY_MEMORY = 1ull << 4,

  // All currently defined statistics row families.
  IREE_HAL_STATISTICS_FAMILY_ALL =
      IREE_HAL_STATISTICS_FAMILY_DEVICE | IREE_HAL_STATISTICS_FAMILY_QUEUE |
      IREE_HAL_STATISTICS_FAMILY_EXECUTABLE_EXPORT |
      IREE_HAL_STATISTICS_FAMILY_COMMAND_BUFFER |
      IREE_HAL_STATISTICS_FAMILY_MEMORY,
};

// Bitfield controlling how a statistics query is produced.
typedef uint32_t iree_hal_statistics_query_flags_t;
enum iree_hal_statistics_query_flag_bits_t {
  IREE_HAL_STATISTICS_QUERY_FLAG_NONE = 0u,

  // Requires rows to reflect an exact producer-supported snapshot.
  //
  // Producers that can only report approximate rows may omit them. Exact does
  // not imply synchronization; use IREE_HAL_STATISTICS_QUERY_FLAG_DRAIN to
  // request a producer-owned drain or synchronization step when available.
  IREE_HAL_STATISTICS_QUERY_FLAG_EXACT = 1u << 0,

  // Allows the producer to synchronize or drain work before taking a snapshot.
  //
  // This is a cold-path option. Producers that cannot provide drained rows may
  // omit them instead of returning stale data.
  IREE_HAL_STATISTICS_QUERY_FLAG_DRAIN = 1u << 1,

  // Resets the requested aggregate families after taking the snapshot.
  //
  // Producers that cannot reset a row may omit it. In-flight completions may
  // race into either the old or next generation according to the backend's
  // completion ordering. Snapshot metadata reports the generation captured and
  // reset so callers can make this explicit.
  IREE_HAL_STATISTICS_QUERY_FLAG_RESET_AFTER_QUERY = 1u << 2,
};

// Bitfield describing the snapshot a producer returned.
typedef uint32_t iree_hal_statistics_snapshot_flags_t;
enum iree_hal_statistics_snapshot_flag_bits_t {
  IREE_HAL_STATISTICS_SNAPSHOT_FLAG_NONE = 0u,

  // Snapshot satisfies an exact query for all emitted rows.
  IREE_HAL_STATISTICS_SNAPSHOT_FLAG_EXACT = 1u << 0,

  // Snapshot includes approximate rows.
  IREE_HAL_STATISTICS_SNAPSHOT_FLAG_APPROXIMATE = 1u << 1,

  // Snapshot was taken after a producer-owned drain/synchronization step.
  IREE_HAL_STATISTICS_SNAPSHOT_FLAG_DRAINED = 1u << 2,

  // Snapshot reset the emitted families after emission.
  IREE_HAL_STATISTICS_SNAPSHOT_FLAG_RESET_AFTER_QUERY = 1u << 3,

  // Snapshot was truncated and one or more matching rows were omitted.
  IREE_HAL_STATISTICS_SNAPSHOT_FLAG_TRUNCATED = 1u << 4,
};

// Time domain used by duration fields in statistics rows.
typedef uint32_t iree_hal_statistics_time_domain_t;
enum iree_hal_statistics_time_domain_bits_t {
  // Producer did not report a time domain for duration fields.
  IREE_HAL_STATISTICS_TIME_DOMAIN_UNKNOWN = 0u,

  // Durations are measured in the host iree_time_now nanosecond domain.
  IREE_HAL_STATISTICS_TIME_DOMAIN_HOST_TIME_NS = 1u,

  // Durations are measured in a device-calibrated nanosecond domain.
  IREE_HAL_STATISTICS_TIME_DOMAIN_DEVICE_TIME_NS = 2u,

  // Row combines values from multiple time domains.
  IREE_HAL_STATISTICS_TIME_DOMAIN_MIXED = 3u,
};

// Row type passed to iree_hal_statistics_row_callback_t.
typedef uint32_t iree_hal_statistics_row_type_t;
enum iree_hal_statistics_row_type_bits_t {
  IREE_HAL_STATISTICS_ROW_TYPE_NONE = 0u,
  IREE_HAL_STATISTICS_ROW_TYPE_DEVICE = 1u,
  IREE_HAL_STATISTICS_ROW_TYPE_QUEUE = 2u,
  IREE_HAL_STATISTICS_ROW_TYPE_EXECUTABLE_EXPORT = 3u,
  IREE_HAL_STATISTICS_ROW_TYPE_COMMAND_BUFFER = 4u,
  IREE_HAL_STATISTICS_ROW_TYPE_MEMORY = 5u,
};

// Common aggregate operation counters.
//
// Counts are disjoint by terminal state: a submitted operation may later
// contribute to exactly one of completed, failed, or cancelled. Operations that
// have been submitted but have not reached a terminal state are represented by
// submitted_count minus the sum of terminal counts.
typedef struct iree_hal_statistics_operation_counts_t {
  // Number of operations accepted by the device or queue.
  uint64_t submitted_count;

  // Number of operations completed successfully.
  uint64_t completed_count;

  // Number of operations completed with an error status.
  uint64_t failed_count;

  // Number of operations cancelled before normal completion.
  uint64_t cancelled_count;
} iree_hal_statistics_operation_counts_t;

// Aggregate duration distribution in nanoseconds.
//
// Each sample is an independent duration in the row's time domain.
// total_duration_ns is the sum of those samples and may double-count
// overlapping work. It is not elapsed wall time, device busy time, or occupancy
// unless the row field explicitly says that its samples are non-overlapping
// spans. When sample_count is zero all duration fields must be zero.
typedef struct iree_hal_statistics_timing_ns_t {
  // Number of measured durations contributing to this distribution.
  uint64_t sample_count;

  // Sum of all measured sample durations in nanoseconds.
  uint64_t total_duration_ns;

  // Minimum measured duration in nanoseconds.
  uint64_t minimum_duration_ns;

  // Maximum measured duration in nanoseconds.
  uint64_t maximum_duration_ns;

  // Most recently measured sample duration in nanoseconds.
  uint64_t last_duration_ns;
} iree_hal_statistics_timing_ns_t;

// Query filters and behavior flags.
typedef struct iree_hal_statistics_query_options_t {
  // Families the caller wants the device to emit.
  iree_hal_statistics_families_t requested_families;

  // Flags controlling exactness, synchronization, and reset behavior.
  iree_hal_statistics_query_flags_t flags;

  // Queue affinity filter, or IREE_HAL_QUEUE_AFFINITY_ANY for all queues.
  iree_hal_queue_affinity_t queue_affinity;

  // Physical device ordinal filter, or IREE_HAL_STATISTICS_ORDINAL_ANY.
  uint32_t physical_device_ordinal;

  // Queue ordinal filter within a physical device, or
  // IREE_HAL_STATISTICS_ORDINAL_ANY.
  uint32_t queue_ordinal;

  // Optional command-buffer filter.
  iree_hal_command_buffer_t* command_buffer;

  // Optional executable filter.
  iree_hal_executable_t* executable;

  // Export ordinal filter used when |executable| is non-NULL, or
  // IREE_HAL_STATISTICS_ORDINAL_ANY.
  iree_hal_executable_export_ordinal_t export_ordinal;

  // Reserved for future query flags; must be zero.
  uint32_t reserved0;

  // Reserved for future query data; must be zero.
  uint64_t reserved1;
} iree_hal_statistics_query_options_t;

// Returns default query options that match all devices/queues/exports.
static inline iree_hal_statistics_query_options_t
iree_hal_statistics_query_options_default(void) {
  iree_hal_statistics_query_options_t options;
  memset(&options, 0, sizeof(options));
  options.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  options.physical_device_ordinal = IREE_HAL_STATISTICS_ORDINAL_ANY;
  options.queue_ordinal = IREE_HAL_STATISTICS_ORDINAL_ANY;
  options.export_ordinal = IREE_HAL_STATISTICS_ORDINAL_ANY;
  return options;
}

// Metadata describing one statistics snapshot.
//
// Metadata is scoped to the row currently being emitted. A producer may emit
// rows with different metadata in one query when some rows are exact, others
// are approximate, or reset/drain behavior differs by family. Omitted rows are
// not represented by metadata and are not reset unless documented by the
// producer.
typedef struct iree_hal_statistics_snapshot_metadata_t {
  // Size of this metadata struct in bytes.
  uint32_t record_length;

  // Flags describing exactness, truncation, reset, and synchronization.
  iree_hal_statistics_snapshot_flags_t flags;

  // Families requested by the caller.
  iree_hal_statistics_families_t requested_families;

  // Families emitted by the producer.
  iree_hal_statistics_families_t emitted_families;

  // Producer-owned generation captured by this snapshot.
  uint64_t generation;

  // Producer-owned generation reset by this snapshot, or zero if no reset
  // occurred.
  uint64_t reset_generation;

  // Host timestamp when the snapshot was produced in iree_time_now
  // nanoseconds.
  iree_time_t host_time_ns;
} iree_hal_statistics_snapshot_metadata_t;

// Device-wide aggregate statistics row.
typedef struct iree_hal_statistics_device_row_t {
  // Size of this row struct in bytes.
  uint32_t record_length;

  // Physical device ordinal, or IREE_HAL_STATISTICS_ORDINAL_ANY when the row
  // aggregates multiple physical devices.
  uint32_t physical_device_ordinal;

  // Number of queues included in this row, or
  // IREE_HAL_STATISTICS_VALUE_UNAVAILABLE when unknown.
  uint64_t queue_count;

  // Aggregate queue-visible operation counts.
  iree_hal_statistics_operation_counts_t operation_counts;

  // Aggregate dependency wait duration.
  iree_hal_statistics_timing_ns_t queue_wait_time;

  // Aggregate ready-to-execute scheduling duration.
  iree_hal_statistics_timing_ns_t queue_ready_time;

  // Aggregate operation execution duration.
  iree_hal_statistics_timing_ns_t execution_time;

  // Time domain used by duration fields in this row.
  iree_hal_statistics_time_domain_t time_domain;
} iree_hal_statistics_device_row_t;

// Queue or stream aggregate statistics row.
typedef struct iree_hal_statistics_queue_row_t {
  // Size of this row struct in bytes.
  uint32_t record_length;

  // Physical device ordinal containing the queue.
  uint32_t physical_device_ordinal;

  // Queue ordinal within the physical device.
  uint32_t queue_ordinal;

  // Producer-local stream identifier, or IREE_HAL_STATISTICS_ID_UNAVAILABLE.
  uint64_t stream_id;

  // Aggregate queue-visible operation counts.
  iree_hal_statistics_operation_counts_t operation_counts;

  // Aggregate dependency wait duration.
  iree_hal_statistics_timing_ns_t queue_wait_time;

  // Aggregate ready-to-execute scheduling duration.
  iree_hal_statistics_timing_ns_t queue_ready_time;

  // Aggregate operation execution duration.
  iree_hal_statistics_timing_ns_t execution_time;

  // Time domain used by duration fields in this row.
  iree_hal_statistics_time_domain_t time_domain;
} iree_hal_statistics_queue_row_t;

// Executable export aggregate statistics row.
typedef struct iree_hal_statistics_executable_export_row_t {
  // Size of this row struct in bytes.
  uint32_t record_length;

  // Physical device ordinal containing the executable replica, or
  // IREE_HAL_STATISTICS_ORDINAL_ANY when merged across replicas.
  uint32_t physical_device_ordinal;

  // Producer-local executable identifier, or
  // IREE_HAL_STATISTICS_ID_UNAVAILABLE.
  uint64_t executable_id;

  // Export ordinal within the executable.
  iree_hal_executable_export_ordinal_t export_ordinal;

  // Number of dispatch invocations observed for this export.
  uint64_t invocation_count;

  // Aggregate dispatch duration for invocations with timing data.
  iree_hal_statistics_timing_ns_t dispatch_time;

  // Last observed workgroup count, or zero when no invocation was observed.
  uint32_t last_workgroup_count[3];

  // Last observed workgroup size, or zero when no invocation was observed.
  uint32_t last_workgroup_size[3];

  // Accumulated workgroup counts per dimension.
  uint64_t total_workgroup_count[3];

  // Accumulated work item count, or IREE_HAL_STATISTICS_VALUE_UNAVAILABLE.
  uint64_t total_work_item_count;

  // Time domain used by duration fields in this row.
  iree_hal_statistics_time_domain_t time_domain;
} iree_hal_statistics_executable_export_row_t;

// Command-buffer aggregate statistics row.
typedef struct iree_hal_statistics_command_buffer_row_t {
  // Size of this row struct in bytes.
  uint32_t record_length;

  // Physical device ordinal associated with the row, or
  // IREE_HAL_STATISTICS_ORDINAL_ANY when merged across devices.
  uint32_t physical_device_ordinal;

  // Producer-local command-buffer identifier, or
  // IREE_HAL_STATISTICS_ID_UNAVAILABLE.
  uint64_t command_buffer_id;

  // Number of command-buffer executions observed.
  uint64_t execution_count;

  // Number of dispatch operations recorded or executed.
  uint64_t dispatch_count;

  // Number of transfer operations recorded or executed.
  uint64_t transfer_count;

  // Total transfer bytes recorded or executed.
  uint64_t transfer_byte_count;

  // Aggregate whole-command-buffer latency samples.
  //
  // This is not the sum of child dispatch/transfer durations. Producers that
  // can only report child-duration sums should omit this field by leaving
  // sample_count zero or omit the row when the distinction is material to the
  // requested exactness.
  iree_hal_statistics_timing_ns_t execution_time;

  // Time domain used by duration fields in this row.
  iree_hal_statistics_time_domain_t time_domain;
} iree_hal_statistics_command_buffer_row_t;

// Device memory pressure aggregate statistics row.
typedef struct iree_hal_statistics_memory_row_t {
  // Size of this row struct in bytes.
  uint32_t record_length;

  // Physical device ordinal, or IREE_HAL_STATISTICS_ORDINAL_ANY when merged.
  uint32_t physical_device_ordinal;

  // Bytes reserved from the underlying memory provider.
  uint64_t reserved_byte_count;

  // Bytes currently live in user-visible allocations.
  uint64_t live_byte_count;

  // Bytes reserved for in-flight queue allocation requests.
  uint64_t in_flight_byte_count;

  // Peak live bytes since the current generation began.
  uint64_t peak_live_byte_count;

  // Number of live allocations.
  uint64_t live_allocation_count;

  // Number of allocation attempts delayed by pressure/backpressure.
  uint64_t pressure_wait_count;
} iree_hal_statistics_memory_row_t;

// Callback invoked for each statistics row in a snapshot.
typedef iree_status_t(IREE_API_PTR* iree_hal_statistics_row_callback_fn_t)(
    void* user_data, const iree_hal_statistics_snapshot_metadata_t* metadata,
    iree_hal_statistics_row_type_t row_type, const void* row,
    iree_host_size_t row_length);

// Callback target used by iree_hal_device_query_statistics.
typedef struct iree_hal_statistics_row_callback_t {
  // Function invoked for each row; required.
  iree_hal_statistics_row_callback_fn_t fn;

  // User data passed to |fn|.
  void* user_data;
} iree_hal_statistics_row_callback_t;

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_STATISTICS_H_
