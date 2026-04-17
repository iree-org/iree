// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <errno.h>
#include <float.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/tooling/flags.h"
#include "iree/hal/utils/profile_file.h"
#include "iree/io/file_contents.h"

IREE_FLAG(string, format, "text",
          "Output format for profile commands: one of `text` or `jsonl`.");
IREE_FLAG(string, filter, "*",
          "Name/key wildcard filter for profile projection output.");
IREE_FLAG(string, output, "-",
          "Output file path for export commands, or `-` for stdout.");
IREE_FLAG(int64_t, id, -1,
          "Optional id filter for projection commands: dispatch event id, "
          "executable id, command-buffer id, memory event/allocation id, or "
          "queue submission id.");
IREE_FLAG(bool, dispatch_events, false,
          "Emits individual dispatch event rows for projection commands with "
          "`--format=jsonl`.");
IREE_FLAG(bool, counter_samples, false,
          "Emits individual counter sample rows for the counter command with "
          "`--format=jsonl`.");
IREE_FLAG(bool, agent_md, false,
          "Prints an agent-oriented Markdown guide for iree-profile JSONL "
          "workflows and exits.");

#define IREE_PROFILE_EXPLAIN_TOP_DISPATCH_COUNT 8
#define IREE_PROFILE_EXPLAIN_TOP_EXPORT_COUNT 10

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
  double total_dispatch_ticks;
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
  // Device metadata chunks parsed.
  uint64_t device_chunk_count;
  // Queue metadata chunks parsed.
  uint64_t queue_chunk_count;
  // Executable metadata chunks parsed.
  uint64_t executable_chunk_count;
  // Executable records parsed.
  uint64_t executable_record_count;
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
  // Chunk records with unknown content types.
  uint64_t unknown_chunk_count;
} iree_profile_summary_t;

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

typedef struct iree_profile_memory_device_t {
  // Session-local physical device ordinal.
  uint32_t physical_device_ordinal;
  // Total memory events matched for this device.
  uint64_t event_count;
  // Slab acquire events matched for this device.
  uint64_t slab_acquire_count;
  // Slab release events matched for this device.
  uint64_t slab_release_count;
  // Current slab allocations after applying matched slab events.
  uint64_t current_slab_allocation_count;
  // Maximum current slab allocations observed while applying matched events.
  uint64_t high_water_slab_allocation_count;
  // Pool reservation events matched for this device.
  uint64_t pool_reserve_count;
  // Pool materialization events matched for this device.
  uint64_t pool_materialize_count;
  // Pool release events matched for this device.
  uint64_t pool_release_count;
  // Pool wait events matched for this device.
  uint64_t pool_wait_count;
  // Synchronous HAL buffer allocation events matched for this device.
  uint64_t buffer_allocate_count;
  // Synchronous HAL buffer free events matched for this device.
  uint64_t buffer_free_count;
  // Current live synchronous buffer allocations after matched events.
  uint64_t current_buffer_allocation_count;
  // Maximum live synchronous buffer allocations observed.
  uint64_t high_water_buffer_allocation_count;
  // Current live pool reservations after applying matched pool events.
  uint64_t current_pool_reservation_count;
  // Maximum live pool reservations observed while applying matched events.
  uint64_t high_water_pool_reservation_count;
  // Current pool reservation bytes after matched reserve/release events.
  uint64_t current_pool_reserved_bytes;
  // Maximum pool reservation bytes observed while applying matched events.
  uint64_t high_water_pool_reserved_bytes;
  // Total bytes reserved by matched pool reserve events.
  uint64_t total_pool_reserved_bytes;
  // Total bytes released by matched pool release events.
  uint64_t total_pool_released_bytes;
  // Queue alloca events matched for this device.
  uint64_t queue_alloca_count;
  // Queue dealloca events matched for this device.
  uint64_t queue_dealloca_count;
  // Current live queue allocations after applying matched queue events.
  uint64_t current_queue_allocation_count;
  // Maximum live queue allocations observed while applying matched events.
  uint64_t high_water_queue_allocation_count;
  // Current slab bytes after applying matched slab events.
  uint64_t current_slab_bytes;
  // Maximum current slab bytes observed while applying matched slab events.
  uint64_t high_water_slab_bytes;
  // Total slab bytes acquired by matched slab events.
  uint64_t total_slab_acquired_bytes;
  // Total slab bytes released by matched slab events.
  uint64_t total_slab_released_bytes;
  // Current queue-allocation bytes after matched alloca/dealloca events.
  uint64_t current_queue_bytes;
  // Maximum queue-allocation bytes observed while applying matched events.
  uint64_t high_water_queue_bytes;
  // Total bytes allocated by matched queue alloca events.
  uint64_t total_queue_alloca_bytes;
  // Total bytes deallocated by matched queue dealloca events.
  uint64_t total_queue_dealloca_bytes;
  // Current synchronous buffer bytes after matched allocate/free events.
  uint64_t current_buffer_bytes;
  // Maximum synchronous buffer bytes observed while applying matched events.
  uint64_t high_water_buffer_bytes;
  // Total synchronous buffer bytes allocated by matched events.
  uint64_t total_buffer_allocate_bytes;
  // Total synchronous buffer bytes freed by matched events.
  uint64_t total_buffer_free_bytes;
} iree_profile_memory_device_t;

typedef enum iree_profile_memory_lifecycle_kind_e {
  // Slab-provider backing allocation lifecycle.
  IREE_PROFILE_MEMORY_LIFECYCLE_KIND_SLAB = 0,
  // Pool reservation lifecycle.
  IREE_PROFILE_MEMORY_LIFECYCLE_KIND_POOL_RESERVATION = 1,
  // Queue-visible transient allocation lifecycle.
  IREE_PROFILE_MEMORY_LIFECYCLE_KIND_QUEUE_ALLOCATION = 2,
  // Direct synchronous HAL buffer lifecycle.
  IREE_PROFILE_MEMORY_LIFECYCLE_KIND_BUFFER_ALLOCATION = 3,
} iree_profile_memory_lifecycle_kind_t;

typedef struct iree_profile_memory_pool_t {
  // Lifecycle kind responsible for this aggregate.
  iree_profile_memory_lifecycle_kind_t kind;
  // Session-local physical device ordinal.
  uint32_t physical_device_ordinal;
  // Producer-defined pool/provider identifier.
  uint64_t pool_id;
  // HAL memory type bits observed for this pool/provider.
  uint64_t memory_type;
  // HAL buffer usage bits observed for this pool/provider.
  uint64_t buffer_usage;
  // Matched events attributed to this pool/provider.
  uint64_t event_count;
  // Wait events attributed to this pool/provider.
  uint64_t wait_count;
  // Materialize events attributed to this pool/provider.
  uint64_t materialize_count;
  // Current live allocation/reservation/slab count.
  uint64_t current_allocation_count;
  // Maximum live allocation/reservation/slab count.
  uint64_t high_water_allocation_count;
  // Current live bytes in this pool/provider aggregate.
  uint64_t current_bytes;
  // Maximum live bytes in this pool/provider aggregate.
  uint64_t high_water_bytes;
  // Cumulative bytes acquired/allocated/reserved.
  uint64_t total_allocate_bytes;
  // Cumulative bytes released/freed/deallocated.
  uint64_t total_free_bytes;
} iree_profile_memory_pool_t;

typedef struct iree_profile_memory_allocation_t {
  // Lifecycle kind represented by this allocation row.
  iree_profile_memory_lifecycle_kind_t kind;
  // Session-local physical device ordinal.
  uint32_t physical_device_ordinal;
  // Producer-defined allocation identifier.
  uint64_t allocation_id;
  // Producer-defined pool/provider identifier.
  uint64_t pool_id;
  // Producer-defined backing allocation or slab identifier.
  uint64_t backing_id;
  // HAL memory type bits observed for this allocation.
  uint64_t memory_type;
  // HAL buffer usage bits observed for this allocation.
  uint64_t buffer_usage;
  // First matched event id in this lifecycle.
  uint64_t first_event_id;
  // Last matched event id in this lifecycle.
  uint64_t last_event_id;
  // First matched event host timestamp.
  int64_t first_host_time_ns;
  // Last matched event host timestamp.
  int64_t last_host_time_ns;
  // First nonzero queue submission id associated with this allocation.
  uint64_t first_submission_id;
  // Last nonzero queue submission id associated with this allocation.
  uint64_t last_submission_id;
  // Number of matched events in this lifecycle.
  uint64_t event_count;
  // Wait events in this lifecycle.
  uint64_t wait_count;
  // Materialize events in this lifecycle.
  uint64_t materialize_count;
  // Current live bytes after matched events.
  uint64_t current_bytes;
  // Maximum live bytes observed after matched events.
  uint64_t high_water_bytes;
  // Cumulative bytes acquired/allocated/reserved.
  uint64_t total_allocate_bytes;
  // Cumulative bytes released/freed/deallocated.
  uint64_t total_free_bytes;
} iree_profile_memory_allocation_t;

typedef struct iree_profile_memory_context_t {
  // Host allocator used for dynamic memory summary arrays.
  iree_allocator_t host_allocator;
  // Dynamic array of per-device memory summaries.
  iree_profile_memory_device_t* devices;
  // Number of valid entries in |devices|.
  iree_host_size_t device_count;
  // Capacity of |devices| in entries.
  iree_host_size_t device_capacity;
  // Dynamic array of per-pool/provider memory summaries.
  iree_profile_memory_pool_t* pools;
  // Number of valid entries in |pools|.
  iree_host_size_t pool_count;
  // Capacity of |pools| in entries.
  iree_host_size_t pool_capacity;
  // Dynamic array of per-allocation memory lifecycles.
  iree_profile_memory_allocation_t* allocations;
  // Number of valid entries in |allocations|.
  iree_host_size_t allocation_count;
  // Capacity of |allocations| in entries.
  iree_host_size_t allocation_capacity;
  // Total memory events parsed before filtering.
  uint64_t total_event_count;
  // Memory events matched by --id and --filter.
  uint64_t matched_event_count;
  // Matched memory events from truncated chunks.
  uint64_t truncated_event_count;
} iree_profile_memory_context_t;

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

typedef struct iree_profile_counter_set_t {
  // Immutable counter set metadata record copied from the profile bundle.
  iree_hal_profile_counter_set_record_t record;
  // Borrowed counter set name from the mapped profile bundle.
  iree_string_view_t name;
} iree_profile_counter_set_t;

typedef struct iree_profile_counter_t {
  // Immutable counter metadata record copied from the profile bundle.
  iree_hal_profile_counter_record_t record;
  // Borrowed hardware block name from the mapped profile bundle.
  iree_string_view_t block_name;
  // Borrowed counter name from the mapped profile bundle.
  iree_string_view_t name;
  // Borrowed counter description from the mapped profile bundle.
  iree_string_view_t description;
} iree_profile_counter_t;

typedef struct iree_profile_counter_aggregate_t {
  // Session-local physical device ordinal for this aggregate row.
  uint32_t physical_device_ordinal;
  // Session-local queue ordinal for this aggregate row.
  uint32_t queue_ordinal;
  // Producer-defined stream identifier for this aggregate row.
  uint64_t stream_id;
  // Producer-local counter set identifier for this aggregate row.
  uint64_t counter_set_id;
  // Counter ordinal within |counter_set_id| for this aggregate row.
  uint32_t counter_ordinal;
  // Producer-local executable identifier for this aggregate row.
  uint64_t executable_id;
  // Export ordinal for this aggregate row.
  uint32_t export_ordinal;
  // Process-local command-buffer identifier for this aggregate row.
  uint64_t command_buffer_id;
  // Number of matched counter samples contributing to this aggregate row.
  uint64_t sample_count;
  // Number of raw uint64_t values contributing to this aggregate row.
  uint64_t raw_value_count;
  // First matched producer-local counter sample identifier.
  uint64_t first_sample_id;
  // Last matched producer-local counter sample identifier.
  uint64_t last_sample_id;
  // Minimum per-sample counter value sum.
  double minimum_value;
  // Maximum per-sample counter value sum.
  double maximum_value;
  // Sum of per-sample counter value sums.
  double total_value;
  // Running mean of per-sample counter value sums.
  double mean_value;
  // Running sum of squares of differences from |mean_value|.
  double m2_value;
} iree_profile_counter_aggregate_t;

typedef struct iree_profile_counter_context_t {
  // Host allocator used for dynamic counter metadata and aggregate rows.
  iree_allocator_t host_allocator;
  // Dynamic array of counter set metadata entries.
  iree_profile_counter_set_t* counter_sets;
  // Number of valid entries in |counter_sets|.
  iree_host_size_t counter_set_count;
  // Capacity of |counter_sets| in entries.
  iree_host_size_t counter_set_capacity;
  // Dynamic array of counter metadata entries.
  iree_profile_counter_t* counters;
  // Number of valid entries in |counters|.
  iree_host_size_t counter_count;
  // Capacity of |counters| in entries.
  iree_host_size_t counter_capacity;
  // Dynamic array of aggregate counter rows.
  iree_profile_counter_aggregate_t* aggregates;
  // Number of valid entries in |aggregates|.
  iree_host_size_t aggregate_count;
  // Capacity of |aggregates| in entries.
  iree_host_size_t aggregate_capacity;
  // Total counter sample records parsed before filtering.
  uint64_t total_sample_count;
  // Counter sample records matched by --id and --filter.
  uint64_t matched_sample_count;
  // Matched counter sample records from truncated chunks.
  uint64_t truncated_sample_count;
} iree_profile_counter_context_t;

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
  // Largest valid dispatch events observed while applying the active filter.
  iree_profile_dispatch_top_event_t
      top_dispatches[IREE_PROFILE_EXPLAIN_TOP_DISPATCH_COUNT];
  // Number of valid entries in |top_dispatches|.
  iree_host_size_t top_dispatch_count;
  // Total queue operation records parsed before filtering.
  uint64_t total_queue_event_count;
  // Queue operation records matched by the active filter.
  uint64_t matched_queue_event_count;
  // Total dispatch records parsed before filtering.
  uint64_t total_dispatch_count;
  // Dispatch records matched by the active filter.
  uint64_t matched_dispatch_count;
  // Matched dispatch records with valid timestamps.
  uint64_t valid_dispatch_count;
  // Matched dispatch records with missing or reversed timestamps.
  uint64_t invalid_dispatch_count;
} iree_profile_dispatch_context_t;

static const char* iree_profile_record_type_name(
    iree_hal_profile_file_record_type_t record_type) {
  switch (record_type) {
    case IREE_HAL_PROFILE_FILE_RECORD_TYPE_SESSION_BEGIN:
      return "session_begin";
    case IREE_HAL_PROFILE_FILE_RECORD_TYPE_CHUNK:
      return "chunk";
    case IREE_HAL_PROFILE_FILE_RECORD_TYPE_SESSION_END:
      return "session_end";
    default:
      return "unknown";
  }
}

static void iree_profile_fprint_json_string(FILE* file,
                                            iree_string_view_t value) {
  fputc('"', file);
  for (iree_host_size_t i = 0; i < value.size; ++i) {
    uint8_t c = (uint8_t)value.data[i];
    switch (c) {
      case '"':
        fputs("\\\"", file);
        break;
      case '\\':
        fputs("\\\\", file);
        break;
      case '\b':
        fputs("\\b", file);
        break;
      case '\f':
        fputs("\\f", file);
        break;
      case '\n':
        fputs("\\n", file);
        break;
      case '\r':
        fputs("\\r", file);
        break;
      case '\t':
        fputs("\\t", file);
        break;
      default:
        if (c < 0x20) {
          fprintf(file, "\\u%04x", c);
        } else {
          fputc(c, file);
        }
        break;
    }
  }
  fputc('"', file);
}

static void iree_profile_fprint_hash_hex(FILE* file, const uint64_t hash[2]) {
  fprintf(file, "%016" PRIx64 "%016" PRIx64, hash[0], hash[1]);
}

static void iree_profile_dump_header_text(
    const iree_hal_profile_file_header_t* header, FILE* file) {
  fprintf(file, "IREE HAL profile bundle\n");
  fprintf(file, "version: %u.%u\n", header->version_major,
          header->version_minor);
  fprintf(file, "header_length: %u\n", header->header_length);
  fprintf(file, "flags: 0x%08x\n", header->flags);
  fprintf(file, "records:\n");
}

static void iree_profile_dump_record_text(
    iree_host_size_t record_index, const iree_hal_profile_file_record_t* record,
    FILE* file) {
  const iree_hal_profile_file_record_header_t* header = &record->header;
  fprintf(file,
          "[%" PRIhsz "] %s record_length=%" PRIu64 " payload_length=%" PRIu64
          "\n",
          record_index, iree_profile_record_type_name(header->record_type),
          header->record_length, header->payload_length);
  fprintf(file, "  content_type: %.*s\n", (int)record->content_type.size,
          record->content_type.data);
  fprintf(file, "  name: %.*s\n", (int)record->name.size, record->name.data);
  fprintf(file,
          "  session_id=%" PRIu64 " stream_id=%" PRIu64 " event_id=%" PRIu64
          "\n",
          header->session_id, header->stream_id, header->event_id);
  fprintf(file, "  executable_id=%" PRIu64 " command_buffer_id=%" PRIu64 "\n",
          header->executable_id, header->command_buffer_id);
  fprintf(file, "  physical_device_ordinal=%u queue_ordinal=%u\n",
          header->physical_device_ordinal, header->queue_ordinal);
  fprintf(file, "  chunk_flags=0x%016" PRIx64 " session_status_code=%u\n",
          header->chunk_flags, header->session_status_code);
}

static void iree_profile_dump_header_jsonl(
    const iree_hal_profile_file_header_t* header, FILE* file) {
  fprintf(file,
          "{\"type\":\"file\",\"magic\":\"IRPF\","
          "\"version_major\":%u,\"version_minor\":%u,"
          "\"header_length\":%u,\"flags\":%u}\n",
          header->version_major, header->version_minor, header->header_length,
          header->flags);
}

static void iree_profile_dump_record_jsonl(
    iree_host_size_t record_index, const iree_hal_profile_file_record_t* record,
    FILE* file) {
  const iree_hal_profile_file_record_header_t* header = &record->header;
  fprintf(file, "{\"type\":\"record\",\"index\":%" PRIhsz, record_index);
  fprintf(file, ",\"record_type\":");
  iree_profile_fprint_json_string(
      file, iree_make_cstring_view(
                iree_profile_record_type_name(header->record_type)));
  fprintf(file, ",\"record_type_value\":%u", header->record_type);
  fprintf(file, ",\"record_length\":%" PRIu64, header->record_length);
  fprintf(file, ",\"payload_length\":%" PRIu64, header->payload_length);
  fprintf(file, ",\"content_type\":");
  iree_profile_fprint_json_string(file, record->content_type);
  fprintf(file, ",\"name\":");
  iree_profile_fprint_json_string(file, record->name);
  fprintf(file, ",\"session_id\":%" PRIu64, header->session_id);
  fprintf(file, ",\"stream_id\":%" PRIu64, header->stream_id);
  fprintf(file, ",\"event_id\":%" PRIu64, header->event_id);
  fprintf(file, ",\"executable_id\":%" PRIu64, header->executable_id);
  fprintf(file, ",\"command_buffer_id\":%" PRIu64, header->command_buffer_id);
  fprintf(file, ",\"physical_device_ordinal\":%u",
          header->physical_device_ordinal);
  fprintf(file, ",\"queue_ordinal\":%u", header->queue_ordinal);
  fprintf(file, ",\"chunk_flags\":%" PRIu64, header->chunk_flags);
  fprintf(file, ",\"session_status_code\":%u", header->session_status_code);
  fputs("}\n", file);
}

static void iree_profile_summary_initialize(
    iree_allocator_t host_allocator, iree_profile_summary_t* out_summary) {
  memset(out_summary, 0, sizeof(*out_summary));
  out_summary->host_allocator = host_allocator;
}

static void iree_profile_summary_deinitialize(iree_profile_summary_t* summary) {
  iree_allocator_free(summary->host_allocator, summary->devices);
  memset(summary, 0, sizeof(*summary));
}

static iree_status_t iree_profile_summary_get_device(
    iree_profile_summary_t* summary, uint32_t physical_device_ordinal,
    iree_profile_device_summary_t** out_device) {
  *out_device = NULL;

  for (iree_host_size_t i = 0; i < summary->device_count; ++i) {
    if (summary->devices[i].physical_device_ordinal ==
        physical_device_ordinal) {
      *out_device = &summary->devices[i];
      return iree_ok_status();
    }
  }

  if (summary->device_count + 1 > summary->device_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        summary->host_allocator,
        iree_max((iree_host_size_t)4, summary->device_count + 1),
        sizeof(summary->devices[0]), &summary->device_capacity,
        (void**)&summary->devices));
  }

  iree_profile_device_summary_t* device =
      &summary->devices[summary->device_count++];
  memset(device, 0, sizeof(*device));
  device->physical_device_ordinal = physical_device_ordinal;
  device->minimum_clock_uncertainty_ns = INT64_MAX;
  device->earliest_dispatch_start_tick = UINT64_MAX;
  device->minimum_dispatch_ticks = UINT64_MAX;
  *out_device = device;
  return iree_ok_status();
}

static iree_status_t iree_profile_payload_record_length(
    iree_string_view_t content_type, iree_const_byte_span_t payload,
    iree_host_size_t payload_offset, iree_host_size_t minimum_record_length,
    iree_host_size_t* out_record_length) {
  *out_record_length = 0;

  const iree_host_size_t remaining_length =
      payload.data_length - payload_offset;
  if (remaining_length < minimum_record_length) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "profile chunk '%.*s' has a truncated typed record",
                            (int)content_type.size, content_type.data);
  }

  uint32_t record_length = 0;
  memcpy(&record_length, payload.data + payload_offset, sizeof(record_length));
  if (record_length < minimum_record_length ||
      record_length > remaining_length) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "profile chunk '%.*s' has invalid typed record length %u",
        (int)content_type.size, content_type.data, record_length);
  }

  *out_record_length = record_length;
  return iree_ok_status();
}

static double iree_profile_sqrt_f64(double value) {
  if (value <= 0.0) return 0.0;
  // Keep this standalone C tool free of libm linkage.
  double estimate = value >= 1.0 ? value : 1.0;
  for (int i = 0; i < 32; ++i) {
    estimate = 0.5 * (estimate + value / estimate);
  }
  return estimate;
}

static void iree_profile_dispatch_context_initialize(
    iree_allocator_t host_allocator,
    iree_profile_dispatch_context_t* out_context) {
  memset(out_context, 0, sizeof(*out_context));
  out_context->host_allocator = host_allocator;
}

static void iree_profile_dispatch_context_deinitialize(
    iree_profile_dispatch_context_t* context) {
  iree_allocator_free(context->host_allocator, context->queue_events);
  iree_allocator_free(context->host_allocator, context->queue_aggregates);
  iree_allocator_free(context->host_allocator, context->command_aggregates);
  iree_allocator_free(context->host_allocator, context->aggregates);
  iree_allocator_free(context->host_allocator, context->devices);
  iree_allocator_free(context->host_allocator, context->queues);
  iree_allocator_free(context->host_allocator, context->command_operations);
  iree_allocator_free(context->host_allocator, context->command_buffers);
  iree_allocator_free(context->host_allocator, context->exports);
  iree_allocator_free(context->host_allocator, context->executables);
  memset(context, 0, sizeof(*context));
}

static void iree_profile_counter_context_initialize(
    iree_allocator_t host_allocator,
    iree_profile_counter_context_t* out_context) {
  memset(out_context, 0, sizeof(*out_context));
  out_context->host_allocator = host_allocator;
}

static void iree_profile_counter_context_deinitialize(
    iree_profile_counter_context_t* context) {
  iree_allocator_free(context->host_allocator, context->aggregates);
  iree_allocator_free(context->host_allocator, context->counters);
  iree_allocator_free(context->host_allocator, context->counter_sets);
  memset(context, 0, sizeof(*context));
}

static iree_status_t iree_profile_dispatch_get_device(
    iree_profile_dispatch_context_t* context, uint32_t physical_device_ordinal,
    iree_profile_dispatch_device_t** out_device) {
  *out_device = NULL;

  for (iree_host_size_t i = 0; i < context->device_count; ++i) {
    if (context->devices[i].physical_device_ordinal ==
        physical_device_ordinal) {
      *out_device = &context->devices[i];
      return iree_ok_status();
    }
  }

  if (context->device_count + 1 > context->device_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        context->host_allocator,
        iree_max((iree_host_size_t)4, context->device_count + 1),
        sizeof(context->devices[0]), &context->device_capacity,
        (void**)&context->devices));
  }

  iree_profile_dispatch_device_t* device =
      &context->devices[context->device_count++];
  memset(device, 0, sizeof(*device));
  device->physical_device_ordinal = physical_device_ordinal;
  *out_device = device;
  return iree_ok_status();
}

static void iree_profile_dispatch_record_clock_sample(
    iree_profile_dispatch_device_t* device,
    const iree_hal_profile_clock_correlation_record_t* record) {
  if (device->clock_sample_count == 0) {
    device->first_clock_sample = *record;
  }
  device->last_clock_sample = *record;
  ++device->clock_sample_count;
}

static const iree_profile_dispatch_queue_t* iree_profile_dispatch_find_queue(
    const iree_profile_dispatch_context_t* context,
    uint32_t physical_device_ordinal, uint32_t queue_ordinal,
    uint64_t stream_id) {
  for (iree_host_size_t i = 0; i < context->queue_count; ++i) {
    const iree_profile_dispatch_queue_t* queue_info = &context->queues[i];
    if (queue_info->record.physical_device_ordinal == physical_device_ordinal &&
        queue_info->record.queue_ordinal == queue_ordinal &&
        queue_info->record.stream_id == stream_id) {
      return queue_info;
    }
  }
  return NULL;
}

static const iree_profile_dispatch_executable_t*
iree_profile_dispatch_find_executable(
    const iree_profile_dispatch_context_t* context, uint64_t executable_id) {
  for (iree_host_size_t i = 0; i < context->executable_count; ++i) {
    const iree_profile_dispatch_executable_t* executable_info =
        &context->executables[i];
    if (executable_info->record.executable_id == executable_id) {
      return executable_info;
    }
  }
  return NULL;
}

static const iree_profile_dispatch_command_buffer_t*
iree_profile_dispatch_find_command_buffer(
    const iree_profile_dispatch_context_t* context,
    uint64_t command_buffer_id) {
  for (iree_host_size_t i = 0; i < context->command_buffer_count; ++i) {
    const iree_profile_dispatch_command_buffer_t* command_buffer_info =
        &context->command_buffers[i];
    if (command_buffer_info->record.command_buffer_id == command_buffer_id) {
      return command_buffer_info;
    }
  }
  return NULL;
}

static bool iree_profile_dispatch_device_try_fit_clock(
    const iree_profile_dispatch_device_t* device, double* out_ns_per_tick,
    double* out_tick_frequency_hz) {
  *out_ns_per_tick = 0.0;
  *out_tick_frequency_hz = 0.0;
  if (!device || device->clock_sample_count < 2) return false;

  const iree_hal_profile_clock_correlation_record_t* first =
      &device->first_clock_sample;
  const iree_hal_profile_clock_correlation_record_t* last =
      &device->last_clock_sample;
  if (!iree_all_bits_set(
          first->flags,
          IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_DEVICE_TICK |
              IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_HOST_CPU_TIMESTAMP) ||
      !iree_all_bits_set(
          last->flags,
          IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_DEVICE_TICK |
              IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_HOST_CPU_TIMESTAMP)) {
    return false;
  }
  if (last->device_tick <= first->device_tick ||
      last->host_cpu_timestamp_ns <= first->host_cpu_timestamp_ns) {
    return false;
  }

  const double device_delta_ticks =
      (double)(last->device_tick - first->device_tick);
  const double host_delta_ns =
      (double)(last->host_cpu_timestamp_ns - first->host_cpu_timestamp_ns);
  *out_ns_per_tick = host_delta_ns / device_delta_ticks;
  *out_tick_frequency_hz = 1000000000.0 / *out_ns_per_tick;
  return true;
}

static const iree_profile_dispatch_export_t* iree_profile_dispatch_find_export(
    const iree_profile_dispatch_context_t* context, uint64_t executable_id,
    uint32_t export_ordinal) {
  for (iree_host_size_t i = 0; i < context->export_count; ++i) {
    const iree_profile_dispatch_export_t* export_info = &context->exports[i];
    if (export_info->executable_id == executable_id &&
        export_info->export_ordinal == export_ordinal) {
      return export_info;
    }
  }
  return NULL;
}

static iree_string_view_t iree_profile_dispatch_format_numeric_key(
    uint32_t physical_device_ordinal, uint64_t executable_id,
    uint32_t export_ordinal, char* buffer, iree_host_size_t buffer_capacity) {
  if (buffer_capacity == 0) return iree_string_view_empty();
  int result = 0;
  if (physical_device_ordinal == UINT32_MAX) {
    result = snprintf(buffer, buffer_capacity, "executable%" PRIu64 "#%u",
                      executable_id, export_ordinal);
  } else {
    result =
        snprintf(buffer, buffer_capacity, "device%u/executable%" PRIu64 "#%u",
                 physical_device_ordinal, executable_id, export_ordinal);
  }
  if (result < 0) return iree_string_view_empty();
  iree_host_size_t length = (iree_host_size_t)result;
  if (length >= buffer_capacity) length = buffer_capacity - 1;
  return iree_make_string_view(buffer, length);
}

static iree_string_view_t iree_profile_dispatch_format_export_key(
    const iree_profile_dispatch_export_t* export_info,
    uint32_t physical_device_ordinal, char* numeric_buffer,
    iree_host_size_t numeric_buffer_capacity) {
  if (!iree_string_view_is_empty(export_info->name)) {
    return export_info->name;
  }
  return iree_profile_dispatch_format_numeric_key(
      physical_device_ordinal, export_info->executable_id,
      export_info->export_ordinal, numeric_buffer, numeric_buffer_capacity);
}

static iree_status_t iree_profile_dispatch_resolve_key(
    const iree_profile_dispatch_context_t* context,
    uint32_t physical_device_ordinal, uint64_t executable_id,
    uint32_t export_ordinal, char* numeric_buffer,
    iree_host_size_t numeric_buffer_capacity, iree_string_view_t* out_key) {
  *out_key = iree_string_view_empty();
  const iree_profile_dispatch_executable_t* executable_info =
      iree_profile_dispatch_find_executable(context, executable_id);
  if (!executable_info) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "dispatch event references missing executable metadata "
        "device=%u executable=%" PRIu64 " export=%u",
        physical_device_ordinal, executable_id, export_ordinal);
  }
  const iree_profile_dispatch_export_t* export_info =
      iree_profile_dispatch_find_export(context, executable_id, export_ordinal);
  if (!export_info) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "dispatch event references missing executable export metadata "
        "device=%u executable=%" PRIu64 " export=%u",
        physical_device_ordinal, executable_id, export_ordinal);
  }
  if (!iree_string_view_is_empty(export_info->name)) {
    *out_key = export_info->name;
  } else {
    *out_key = iree_profile_dispatch_format_export_key(
        export_info, physical_device_ordinal, numeric_buffer,
        numeric_buffer_capacity);
  }
  return iree_ok_status();
}

static iree_status_t iree_profile_dispatch_validate_event_metadata(
    const iree_profile_dispatch_context_t* context,
    const iree_hal_profile_file_record_t* record,
    const iree_hal_profile_dispatch_event_t* event,
    iree_profile_projection_mode_t projection_mode) {
  const iree_profile_dispatch_queue_t* queue_info =
      iree_profile_dispatch_find_queue(
          context, record->header.physical_device_ordinal,
          record->header.queue_ordinal, record->header.stream_id);
  if (!queue_info) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "dispatch event references missing queue metadata "
        "device=%u queue=%u stream=%" PRIu64 " submission=%" PRIu64,
        record->header.physical_device_ordinal, record->header.queue_ordinal,
        record->header.stream_id, event->submission_id);
  }
  if (projection_mode == IREE_PROFILE_PROJECTION_MODE_COMMAND &&
      event->command_buffer_id != 0 &&
      !iree_profile_dispatch_find_command_buffer(context,
                                                 event->command_buffer_id)) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "dispatch event references missing command-buffer metadata "
        "command_buffer=%" PRIu64 " submission=%" PRIu64,
        event->command_buffer_id, event->submission_id);
  }
  return iree_ok_status();
}

static bool iree_profile_dispatch_key_matches(iree_string_view_t key,
                                              iree_string_view_t filter) {
  if (iree_string_view_is_empty(filter) ||
      iree_string_view_equal(filter, IREE_SV("*"))) {
    return true;
  }
  return iree_string_view_match_pattern(key, filter);
}

static bool iree_profile_dispatch_event_matches_id(
    const iree_hal_profile_dispatch_event_t* event,
    iree_profile_projection_mode_t mode, int64_t id_filter) {
  if (id_filter < 0) return true;
  const uint64_t id = (uint64_t)id_filter;
  switch (mode) {
    case IREE_PROFILE_PROJECTION_MODE_DISPATCH:
      return event->event_id == id;
    case IREE_PROFILE_PROJECTION_MODE_EXECUTABLE:
      return event->executable_id == id;
    case IREE_PROFILE_PROJECTION_MODE_COMMAND:
      return event->command_buffer_id == id;
    case IREE_PROFILE_PROJECTION_MODE_QUEUE:
      return event->submission_id == id;
    default:
      return false;
  }
}

static iree_status_t iree_profile_dispatch_append_export(
    iree_profile_dispatch_context_t* context,
    const iree_profile_dispatch_export_t* export_info) {
  if (context->export_count + 1 > context->export_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        context->host_allocator,
        iree_max((iree_host_size_t)16, context->export_count + 1),
        sizeof(context->exports[0]), &context->export_capacity,
        (void**)&context->exports));
  }
  context->exports[context->export_count++] = *export_info;
  return iree_ok_status();
}

static iree_status_t iree_profile_dispatch_append_executable(
    iree_profile_dispatch_context_t* context,
    const iree_hal_profile_executable_record_t* record) {
  if (context->executable_count + 1 > context->executable_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        context->host_allocator,
        iree_max((iree_host_size_t)8, context->executable_count + 1),
        sizeof(context->executables[0]), &context->executable_capacity,
        (void**)&context->executables));
  }
  iree_profile_dispatch_executable_t* executable_info =
      &context->executables[context->executable_count++];
  executable_info->record = *record;
  return iree_ok_status();
}

static iree_status_t iree_profile_dispatch_append_command_buffer(
    iree_profile_dispatch_context_t* context,
    const iree_hal_profile_command_buffer_record_t* record) {
  if (context->command_buffer_count + 1 > context->command_buffer_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        context->host_allocator,
        iree_max((iree_host_size_t)16, context->command_buffer_count + 1),
        sizeof(context->command_buffers[0]), &context->command_buffer_capacity,
        (void**)&context->command_buffers));
  }
  iree_profile_dispatch_command_buffer_t* command_buffer_info =
      &context->command_buffers[context->command_buffer_count++];
  command_buffer_info->record = *record;
  return iree_ok_status();
}

static iree_status_t iree_profile_dispatch_append_command_operation(
    iree_profile_dispatch_context_t* context,
    const iree_hal_profile_command_operation_record_t* record) {
  if (context->command_operation_count + 1 >
      context->command_operation_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        context->host_allocator,
        iree_max((iree_host_size_t)64, context->command_operation_count + 1),
        sizeof(context->command_operations[0]),
        &context->command_operation_capacity,
        (void**)&context->command_operations));
  }
  iree_profile_dispatch_command_operation_t* operation_info =
      &context->command_operations[context->command_operation_count++];
  operation_info->record = *record;
  return iree_ok_status();
}

static iree_status_t iree_profile_dispatch_append_queue(
    iree_profile_dispatch_context_t* context,
    const iree_hal_profile_queue_record_t* record) {
  if (context->queue_count + 1 > context->queue_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        context->host_allocator,
        iree_max((iree_host_size_t)4, context->queue_count + 1),
        sizeof(context->queues[0]), &context->queue_capacity,
        (void**)&context->queues));
  }
  iree_profile_dispatch_queue_t* queue_info =
      &context->queues[context->queue_count++];
  queue_info->record = *record;
  return iree_ok_status();
}

static iree_status_t iree_profile_dispatch_append_queue_event(
    iree_profile_dispatch_context_t* context,
    const iree_hal_profile_queue_event_t* record) {
  if (context->queue_event_count + 1 > context->queue_event_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        context->host_allocator,
        iree_max((iree_host_size_t)64, context->queue_event_count + 1),
        sizeof(context->queue_events[0]), &context->queue_event_capacity,
        (void**)&context->queue_events));
  }
  iree_profile_dispatch_queue_event_t* event_info =
      &context->queue_events[context->queue_event_count++];
  event_info->record = *record;
  return iree_ok_status();
}

static iree_status_t iree_profile_counter_append_counter_set(
    iree_profile_counter_context_t* context,
    const iree_profile_counter_set_t* counter_set) {
  if (context->counter_set_count + 1 > context->counter_set_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        context->host_allocator,
        iree_max((iree_host_size_t)4, context->counter_set_count + 1),
        sizeof(context->counter_sets[0]), &context->counter_set_capacity,
        (void**)&context->counter_sets));
  }
  context->counter_sets[context->counter_set_count++] = *counter_set;
  return iree_ok_status();
}

static iree_status_t iree_profile_counter_append_counter(
    iree_profile_counter_context_t* context,
    const iree_profile_counter_t* counter) {
  if (context->counter_count + 1 > context->counter_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        context->host_allocator,
        iree_max((iree_host_size_t)8, context->counter_count + 1),
        sizeof(context->counters[0]), &context->counter_capacity,
        (void**)&context->counters));
  }
  context->counters[context->counter_count++] = *counter;
  return iree_ok_status();
}

static const iree_profile_counter_set_t* iree_profile_counter_find_counter_set(
    const iree_profile_counter_context_t* context, uint64_t counter_set_id) {
  for (iree_host_size_t i = 0; i < context->counter_set_count; ++i) {
    const iree_profile_counter_set_t* counter_set = &context->counter_sets[i];
    if (counter_set->record.counter_set_id == counter_set_id) {
      return counter_set;
    }
  }
  return NULL;
}

static const iree_profile_counter_t* iree_profile_counter_find_counter(
    const iree_profile_counter_context_t* context, uint64_t counter_set_id,
    uint32_t counter_ordinal) {
  for (iree_host_size_t i = 0; i < context->counter_count; ++i) {
    const iree_profile_counter_t* counter = &context->counters[i];
    if (counter->record.counter_set_id == counter_set_id &&
        counter->record.counter_ordinal == counter_ordinal) {
      return counter;
    }
  }
  return NULL;
}

static iree_status_t iree_profile_counter_get_aggregate(
    iree_profile_counter_context_t* context, uint32_t physical_device_ordinal,
    uint32_t queue_ordinal, uint64_t stream_id, uint64_t counter_set_id,
    uint32_t counter_ordinal, uint64_t executable_id, uint32_t export_ordinal,
    uint64_t command_buffer_id,
    iree_profile_counter_aggregate_t** out_aggregate) {
  *out_aggregate = NULL;

  for (iree_host_size_t i = 0; i < context->aggregate_count; ++i) {
    iree_profile_counter_aggregate_t* aggregate = &context->aggregates[i];
    if (aggregate->physical_device_ordinal == physical_device_ordinal &&
        aggregate->queue_ordinal == queue_ordinal &&
        aggregate->stream_id == stream_id &&
        aggregate->counter_set_id == counter_set_id &&
        aggregate->counter_ordinal == counter_ordinal &&
        aggregate->executable_id == executable_id &&
        aggregate->export_ordinal == export_ordinal &&
        aggregate->command_buffer_id == command_buffer_id) {
      *out_aggregate = aggregate;
      return iree_ok_status();
    }
  }

  if (context->aggregate_count + 1 > context->aggregate_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        context->host_allocator,
        iree_max((iree_host_size_t)16, context->aggregate_count + 1),
        sizeof(context->aggregates[0]), &context->aggregate_capacity,
        (void**)&context->aggregates));
  }

  iree_profile_counter_aggregate_t* aggregate =
      &context->aggregates[context->aggregate_count++];
  memset(aggregate, 0, sizeof(*aggregate));
  aggregate->physical_device_ordinal = physical_device_ordinal;
  aggregate->queue_ordinal = queue_ordinal;
  aggregate->stream_id = stream_id;
  aggregate->counter_set_id = counter_set_id;
  aggregate->counter_ordinal = counter_ordinal;
  aggregate->executable_id = executable_id;
  aggregate->export_ordinal = export_ordinal;
  aggregate->command_buffer_id = command_buffer_id;
  aggregate->minimum_value = DBL_MAX;
  *out_aggregate = aggregate;
  return iree_ok_status();
}

static iree_status_t iree_profile_dispatch_get_aggregate(
    iree_profile_dispatch_context_t* context, uint32_t physical_device_ordinal,
    uint64_t executable_id, uint32_t export_ordinal,
    iree_profile_dispatch_aggregate_t** out_aggregate) {
  *out_aggregate = NULL;

  for (iree_host_size_t i = 0; i < context->aggregate_count; ++i) {
    iree_profile_dispatch_aggregate_t* aggregate = &context->aggregates[i];
    if (aggregate->physical_device_ordinal == physical_device_ordinal &&
        aggregate->executable_id == executable_id &&
        aggregate->export_ordinal == export_ordinal) {
      *out_aggregate = aggregate;
      return iree_ok_status();
    }
  }

  if (context->aggregate_count + 1 > context->aggregate_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        context->host_allocator,
        iree_max((iree_host_size_t)16, context->aggregate_count + 1),
        sizeof(context->aggregates[0]), &context->aggregate_capacity,
        (void**)&context->aggregates));
  }

  iree_profile_dispatch_aggregate_t* aggregate =
      &context->aggregates[context->aggregate_count++];
  memset(aggregate, 0, sizeof(*aggregate));
  aggregate->physical_device_ordinal = physical_device_ordinal;
  aggregate->executable_id = executable_id;
  aggregate->export_ordinal = export_ordinal;
  aggregate->earliest_start_tick = UINT64_MAX;
  aggregate->minimum_ticks = UINT64_MAX;
  *out_aggregate = aggregate;
  return iree_ok_status();
}

static iree_status_t iree_profile_dispatch_get_command_aggregate(
    iree_profile_dispatch_context_t* context, uint64_t command_buffer_id,
    uint64_t submission_id, uint32_t physical_device_ordinal,
    uint32_t queue_ordinal, uint64_t stream_id,
    iree_profile_dispatch_command_aggregate_t** out_aggregate) {
  *out_aggregate = NULL;

  for (iree_host_size_t i = 0; i < context->command_aggregate_count; ++i) {
    iree_profile_dispatch_command_aggregate_t* aggregate =
        &context->command_aggregates[i];
    if (aggregate->command_buffer_id == command_buffer_id &&
        aggregate->submission_id == submission_id &&
        aggregate->physical_device_ordinal == physical_device_ordinal &&
        aggregate->queue_ordinal == queue_ordinal &&
        aggregate->stream_id == stream_id) {
      *out_aggregate = aggregate;
      return iree_ok_status();
    }
  }

  if (context->command_aggregate_count + 1 >
      context->command_aggregate_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        context->host_allocator,
        iree_max((iree_host_size_t)16, context->command_aggregate_count + 1),
        sizeof(context->command_aggregates[0]),
        &context->command_aggregate_capacity,
        (void**)&context->command_aggregates));
  }

  iree_profile_dispatch_command_aggregate_t* aggregate =
      &context->command_aggregates[context->command_aggregate_count++];
  memset(aggregate, 0, sizeof(*aggregate));
  aggregate->command_buffer_id = command_buffer_id;
  aggregate->submission_id = submission_id;
  aggregate->physical_device_ordinal = physical_device_ordinal;
  aggregate->queue_ordinal = queue_ordinal;
  aggregate->stream_id = stream_id;
  aggregate->earliest_start_tick = UINT64_MAX;
  *out_aggregate = aggregate;
  return iree_ok_status();
}

static iree_status_t iree_profile_dispatch_get_queue_aggregate(
    iree_profile_dispatch_context_t* context, uint32_t physical_device_ordinal,
    uint32_t queue_ordinal, uint64_t stream_id, uint64_t submission_id,
    iree_profile_dispatch_queue_aggregate_t** out_aggregate) {
  *out_aggregate = NULL;

  for (iree_host_size_t i = 0; i < context->queue_aggregate_count; ++i) {
    iree_profile_dispatch_queue_aggregate_t* aggregate =
        &context->queue_aggregates[i];
    if (aggregate->physical_device_ordinal == physical_device_ordinal &&
        aggregate->queue_ordinal == queue_ordinal &&
        aggregate->stream_id == stream_id &&
        aggregate->submission_id == submission_id) {
      *out_aggregate = aggregate;
      return iree_ok_status();
    }
  }

  if (context->queue_aggregate_count + 1 > context->queue_aggregate_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        context->host_allocator,
        iree_max((iree_host_size_t)16, context->queue_aggregate_count + 1),
        sizeof(context->queue_aggregates[0]),
        &context->queue_aggregate_capacity,
        (void**)&context->queue_aggregates));
  }

  iree_profile_dispatch_queue_aggregate_t* aggregate =
      &context->queue_aggregates[context->queue_aggregate_count++];
  memset(aggregate, 0, sizeof(*aggregate));
  aggregate->physical_device_ordinal = physical_device_ordinal;
  aggregate->queue_ordinal = queue_ordinal;
  aggregate->stream_id = stream_id;
  aggregate->submission_id = submission_id;
  aggregate->earliest_start_tick = UINT64_MAX;
  *out_aggregate = aggregate;
  return iree_ok_status();
}

static void iree_profile_dispatch_record_aggregate_event(
    iree_profile_dispatch_aggregate_t* aggregate,
    const iree_hal_profile_dispatch_event_t* event) {
  ++aggregate->dispatch_count;
  memcpy(aggregate->last_workgroup_count, event->workgroup_count,
         sizeof(aggregate->last_workgroup_count));
  memcpy(aggregate->last_workgroup_size, event->workgroup_size,
         sizeof(aggregate->last_workgroup_size));

  if (event->start_tick == 0 || event->end_tick == 0 ||
      event->end_tick < event->start_tick) {
    ++aggregate->invalid_count;
    return;
  }

  const uint64_t duration_ticks = event->end_tick - event->start_tick;
  ++aggregate->valid_count;
  aggregate->earliest_start_tick =
      iree_min(aggregate->earliest_start_tick, event->start_tick);
  aggregate->latest_end_tick =
      iree_max(aggregate->latest_end_tick, event->end_tick);
  aggregate->minimum_ticks = iree_min(aggregate->minimum_ticks, duration_ticks);
  aggregate->maximum_ticks = iree_max(aggregate->maximum_ticks, duration_ticks);
  aggregate->total_ticks += (double)duration_ticks;

  const double duration = (double)duration_ticks;
  const double delta = duration - aggregate->mean_ticks;
  aggregate->mean_ticks += delta / (double)aggregate->valid_count;
  const double delta2 = duration - aggregate->mean_ticks;
  aggregate->m2_ticks += delta * delta2;
}

static void iree_profile_dispatch_record_command_aggregate_event(
    iree_profile_dispatch_command_aggregate_t* aggregate,
    const iree_hal_profile_dispatch_event_t* event) {
  ++aggregate->dispatch_count;
  if (event->start_tick == 0 || event->end_tick == 0 ||
      event->end_tick < event->start_tick) {
    ++aggregate->invalid_count;
    return;
  }
  ++aggregate->valid_count;
  aggregate->earliest_start_tick =
      iree_min(aggregate->earliest_start_tick, event->start_tick);
  aggregate->latest_end_tick =
      iree_max(aggregate->latest_end_tick, event->end_tick);
  aggregate->total_ticks += (double)(event->end_tick - event->start_tick);
}

static void iree_profile_dispatch_record_queue_aggregate_event(
    iree_profile_dispatch_queue_aggregate_t* aggregate,
    const iree_hal_profile_dispatch_event_t* event) {
  ++aggregate->dispatch_count;
  if (event->start_tick == 0 || event->end_tick == 0 ||
      event->end_tick < event->start_tick) {
    ++aggregate->invalid_count;
    return;
  }
  ++aggregate->valid_count;
  aggregate->earliest_start_tick =
      iree_min(aggregate->earliest_start_tick, event->start_tick);
  aggregate->latest_end_tick =
      iree_max(aggregate->latest_end_tick, event->end_tick);
  aggregate->total_ticks += (double)(event->end_tick - event->start_tick);
}

static void iree_profile_dispatch_record_top_event(
    iree_profile_dispatch_context_t* context,
    const iree_hal_profile_file_record_t* file_record,
    const iree_hal_profile_dispatch_event_t* event) {
  if (event->start_tick == 0 || event->end_tick == 0 ||
      event->end_tick < event->start_tick) {
    return;
  }

  const uint64_t duration_ticks = event->end_tick - event->start_tick;
  iree_host_size_t target_index = context->top_dispatch_count;
  if (context->top_dispatch_count < IREE_PROFILE_EXPLAIN_TOP_DISPATCH_COUNT) {
    ++context->top_dispatch_count;
  } else {
    target_index = 0;
    for (iree_host_size_t i = 1; i < context->top_dispatch_count; ++i) {
      if (context->top_dispatches[i].duration_ticks <
          context->top_dispatches[target_index].duration_ticks) {
        target_index = i;
      }
    }
    if (duration_ticks <=
        context->top_dispatches[target_index].duration_ticks) {
      return;
    }
  }

  iree_profile_dispatch_top_event_t* top_event =
      &context->top_dispatches[target_index];
  top_event->physical_device_ordinal =
      file_record->header.physical_device_ordinal;
  top_event->queue_ordinal = file_record->header.queue_ordinal;
  top_event->stream_id = file_record->header.stream_id;
  top_event->duration_ticks = duration_ticks;
  top_event->event = *event;
}

static void iree_profile_summary_record_clock_sample(
    iree_profile_device_summary_t* device,
    const iree_hal_profile_clock_correlation_record_t* record) {
  if (device->clock_sample_count == 0) {
    device->first_clock_sample = *record;
  }
  device->last_clock_sample = *record;
  ++device->clock_sample_count;

  if (iree_all_bits_set(
          record->flags,
          IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_HOST_TIME_BRACKET) &&
      record->host_time_end_ns >= record->host_time_begin_ns) {
    const int64_t uncertainty_ns =
        record->host_time_end_ns - record->host_time_begin_ns;
    device->minimum_clock_uncertainty_ns =
        iree_min(device->minimum_clock_uncertainty_ns, uncertainty_ns);
    device->maximum_clock_uncertainty_ns =
        iree_max(device->maximum_clock_uncertainty_ns, uncertainty_ns);
  }
}

static void iree_profile_summary_record_dispatch_event(
    iree_profile_device_summary_t* device,
    const iree_hal_profile_dispatch_event_t* record) {
  ++device->dispatch_event_count;
  if (record->start_tick == 0 || record->end_tick == 0 ||
      record->end_tick < record->start_tick) {
    ++device->invalid_dispatch_event_count;
    return;
  }

  const uint64_t duration_ticks = record->end_tick - record->start_tick;
  device->total_dispatch_ticks += (double)duration_ticks;
  device->earliest_dispatch_start_tick =
      iree_min(device->earliest_dispatch_start_tick, record->start_tick);
  device->latest_dispatch_end_tick =
      iree_max(device->latest_dispatch_end_tick, record->end_tick);
  device->minimum_dispatch_ticks =
      iree_min(device->minimum_dispatch_ticks, duration_ticks);
  device->maximum_dispatch_ticks =
      iree_max(device->maximum_dispatch_ticks, duration_ticks);
}

static iree_status_t iree_profile_summary_process_device_records(
    iree_profile_summary_t* summary,
    const iree_hal_profile_file_record_t* record) {
  iree_host_size_t payload_offset = 0;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         payload_offset < record->payload.data_length) {
    iree_host_size_t record_length = 0;
    status = iree_profile_payload_record_length(
        record->content_type, record->payload, payload_offset,
        sizeof(iree_hal_profile_device_record_t), &record_length);
    if (iree_status_is_ok(status)) {
      iree_hal_profile_device_record_t device_record;
      memcpy(&device_record, record->payload.data + payload_offset,
             sizeof(device_record));

      iree_profile_device_summary_t* device = NULL;
      status = iree_profile_summary_get_device(
          summary, device_record.physical_device_ordinal, &device);
      if (iree_status_is_ok(status)) {
        ++device->device_record_count;
        device->queue_count = device_record.queue_count;
      }
      payload_offset += record_length;
    }
  }
  return status;
}

static iree_status_t iree_profile_summary_process_queue_records(
    iree_profile_summary_t* summary,
    const iree_hal_profile_file_record_t* record) {
  iree_host_size_t payload_offset = 0;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         payload_offset < record->payload.data_length) {
    iree_host_size_t record_length = 0;
    status = iree_profile_payload_record_length(
        record->content_type, record->payload, payload_offset,
        sizeof(iree_hal_profile_queue_record_t), &record_length);
    if (iree_status_is_ok(status)) {
      iree_hal_profile_queue_record_t queue_record;
      memcpy(&queue_record, record->payload.data + payload_offset,
             sizeof(queue_record));

      iree_profile_device_summary_t* device = NULL;
      status = iree_profile_summary_get_device(
          summary, queue_record.physical_device_ordinal, &device);
      if (iree_status_is_ok(status)) {
        ++device->queue_record_count;
      }
      payload_offset += record_length;
    }
  }
  return status;
}

static iree_status_t iree_profile_summary_process_executable_records(
    iree_profile_summary_t* summary,
    const iree_hal_profile_file_record_t* record) {
  iree_host_size_t payload_offset = 0;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         payload_offset < record->payload.data_length) {
    iree_host_size_t record_length = 0;
    status = iree_profile_payload_record_length(
        record->content_type, record->payload, payload_offset,
        sizeof(iree_hal_profile_executable_record_t), &record_length);
    if (iree_status_is_ok(status)) {
      ++summary->executable_record_count;
      payload_offset += record_length;
    }
  }
  return status;
}

static iree_status_t iree_profile_summary_process_executable_export_records(
    iree_profile_summary_t* summary,
    const iree_hal_profile_file_record_t* record) {
  iree_host_size_t payload_offset = 0;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         payload_offset < record->payload.data_length) {
    iree_host_size_t record_length = 0;
    status = iree_profile_payload_record_length(
        record->content_type, record->payload, payload_offset,
        sizeof(iree_hal_profile_executable_export_record_t), &record_length);
    if (iree_status_is_ok(status)) {
      ++summary->executable_export_record_count;
      payload_offset += record_length;
    }
  }
  return status;
}

static iree_status_t iree_profile_summary_process_command_buffer_records(
    iree_profile_summary_t* summary,
    const iree_hal_profile_file_record_t* record) {
  iree_host_size_t payload_offset = 0;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         payload_offset < record->payload.data_length) {
    iree_host_size_t record_length = 0;
    status = iree_profile_payload_record_length(
        record->content_type, record->payload, payload_offset,
        sizeof(iree_hal_profile_command_buffer_record_t), &record_length);
    if (iree_status_is_ok(status)) {
      ++summary->command_buffer_record_count;
      payload_offset += record_length;
    }
  }
  return status;
}

static iree_status_t iree_profile_summary_process_command_operation_records(
    iree_profile_summary_t* summary,
    const iree_hal_profile_file_record_t* record) {
  iree_host_size_t payload_offset = 0;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         payload_offset < record->payload.data_length) {
    iree_host_size_t record_length = 0;
    status = iree_profile_payload_record_length(
        record->content_type, record->payload, payload_offset,
        sizeof(iree_hal_profile_command_operation_record_t), &record_length);
    if (iree_status_is_ok(status)) {
      ++summary->command_operation_record_count;
      payload_offset += record_length;
    }
  }
  return status;
}

static iree_status_t iree_profile_summary_process_clock_correlation_records(
    iree_profile_summary_t* summary,
    const iree_hal_profile_file_record_t* record) {
  iree_host_size_t payload_offset = 0;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         payload_offset < record->payload.data_length) {
    iree_host_size_t record_length = 0;
    status = iree_profile_payload_record_length(
        record->content_type, record->payload, payload_offset,
        sizeof(iree_hal_profile_clock_correlation_record_t), &record_length);
    if (iree_status_is_ok(status)) {
      iree_hal_profile_clock_correlation_record_t clock_record;
      memcpy(&clock_record, record->payload.data + payload_offset,
             sizeof(clock_record));

      iree_profile_device_summary_t* device = NULL;
      status = iree_profile_summary_get_device(
          summary, clock_record.physical_device_ordinal, &device);
      if (iree_status_is_ok(status)) {
        iree_profile_summary_record_clock_sample(device, &clock_record);
      }
      payload_offset += record_length;
    }
  }
  return status;
}

static iree_status_t iree_profile_summary_process_dispatch_event_records(
    iree_profile_summary_t* summary,
    const iree_hal_profile_file_record_t* record) {
  iree_profile_device_summary_t* device = NULL;
  IREE_RETURN_IF_ERROR(iree_profile_summary_get_device(
      summary, record->header.physical_device_ordinal, &device));

  iree_host_size_t payload_offset = 0;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         payload_offset < record->payload.data_length) {
    iree_host_size_t record_length = 0;
    status = iree_profile_payload_record_length(
        record->content_type, record->payload, payload_offset,
        sizeof(iree_hal_profile_dispatch_event_t), &record_length);
    if (iree_status_is_ok(status)) {
      iree_hal_profile_dispatch_event_t dispatch_record;
      memcpy(&dispatch_record, record->payload.data + payload_offset,
             sizeof(dispatch_record));
      iree_profile_summary_record_dispatch_event(device, &dispatch_record);
      payload_offset += record_length;
    }
  }
  return status;
}

static iree_status_t iree_profile_summary_process_memory_event_records(
    iree_profile_summary_t* summary,
    const iree_hal_profile_file_record_t* record) {
  iree_host_size_t payload_offset = 0;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         payload_offset < record->payload.data_length) {
    iree_host_size_t record_length = 0;
    status = iree_profile_payload_record_length(
        record->content_type, record->payload, payload_offset,
        sizeof(iree_hal_profile_memory_event_t), &record_length);
    if (iree_status_is_ok(status)) {
      ++summary->memory_event_record_count;
      payload_offset += record_length;
    }
  }
  return status;
}

static iree_status_t iree_profile_summary_process_counter_set_records(
    iree_profile_summary_t* summary,
    const iree_hal_profile_file_record_t* record) {
  iree_host_size_t payload_offset = 0;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         payload_offset < record->payload.data_length) {
    iree_host_size_t record_length = 0;
    status = iree_profile_payload_record_length(
        record->content_type, record->payload, payload_offset,
        sizeof(iree_hal_profile_counter_set_record_t), &record_length);
    if (iree_status_is_ok(status)) {
      iree_hal_profile_counter_set_record_t counter_set_record;
      memcpy(&counter_set_record, record->payload.data + payload_offset,
             sizeof(counter_set_record));
      if (counter_set_record.name_length !=
          record_length - sizeof(counter_set_record)) {
        status = iree_make_status(
            IREE_STATUS_DATA_LOSS,
            "counter set name length is inconsistent with record length");
      }
      if (iree_status_is_ok(status)) {
        ++summary->counter_set_record_count;
        payload_offset += record_length;
      }
    }
  }
  return status;
}

static iree_status_t iree_profile_summary_process_counter_records(
    iree_profile_summary_t* summary,
    const iree_hal_profile_file_record_t* record) {
  iree_host_size_t payload_offset = 0;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         payload_offset < record->payload.data_length) {
    iree_host_size_t record_length = 0;
    status = iree_profile_payload_record_length(
        record->content_type, record->payload, payload_offset,
        sizeof(iree_hal_profile_counter_record_t), &record_length);
    if (iree_status_is_ok(status)) {
      iree_hal_profile_counter_record_t counter_record;
      memcpy(&counter_record, record->payload.data + payload_offset,
             sizeof(counter_record));
      iree_host_size_t trailing_length = 0;
      if (!iree_host_size_checked_add(counter_record.block_name_length,
                                      counter_record.name_length,
                                      &trailing_length) ||
          !iree_host_size_checked_add(trailing_length,
                                      counter_record.description_length,
                                      &trailing_length) ||
          trailing_length != record_length - sizeof(counter_record)) {
        status = iree_make_status(
            IREE_STATUS_DATA_LOSS,
            "counter string lengths are inconsistent with record length");
      }
      if (iree_status_is_ok(status)) {
        ++summary->counter_record_count;
        payload_offset += record_length;
      }
    }
  }
  return status;
}

static iree_status_t iree_profile_summary_process_counter_sample_records(
    iree_profile_summary_t* summary,
    const iree_hal_profile_file_record_t* record) {
  iree_host_size_t payload_offset = 0;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         payload_offset < record->payload.data_length) {
    iree_host_size_t record_length = 0;
    status = iree_profile_payload_record_length(
        record->content_type, record->payload, payload_offset,
        sizeof(iree_hal_profile_counter_sample_record_t), &record_length);
    if (iree_status_is_ok(status)) {
      iree_hal_profile_counter_sample_record_t sample_record;
      memcpy(&sample_record, record->payload.data + payload_offset,
             sizeof(sample_record));
      iree_host_size_t values_length = 0;
      if (!iree_host_size_checked_mul(sample_record.sample_value_count,
                                      sizeof(uint64_t), &values_length) ||
          values_length != record_length - sizeof(sample_record)) {
        status = iree_make_status(
            IREE_STATUS_DATA_LOSS,
            "counter sample value count is inconsistent with record length");
      }
      if (iree_status_is_ok(status)) {
        ++summary->counter_sample_record_count;
        payload_offset += record_length;
      }
    }
  }
  return status;
}

static iree_status_t iree_profile_summary_process_queue_event_records(
    iree_profile_summary_t* summary,
    const iree_hal_profile_file_record_t* record) {
  iree_host_size_t payload_offset = 0;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         payload_offset < record->payload.data_length) {
    iree_host_size_t record_length = 0;
    status = iree_profile_payload_record_length(
        record->content_type, record->payload, payload_offset,
        sizeof(iree_hal_profile_queue_event_t), &record_length);
    if (iree_status_is_ok(status)) {
      ++summary->queue_event_record_count;
      payload_offset += record_length;
    }
  }
  return status;
}

static iree_status_t iree_profile_summary_process_record(
    iree_profile_summary_t* summary,
    const iree_hal_profile_file_record_t* record) {
  ++summary->file_record_count;

  switch (record->header.record_type) {
    case IREE_HAL_PROFILE_FILE_RECORD_TYPE_SESSION_BEGIN:
      ++summary->session_begin_count;
      return iree_ok_status();
    case IREE_HAL_PROFILE_FILE_RECORD_TYPE_SESSION_END:
      ++summary->session_end_count;
      return iree_ok_status();
    case IREE_HAL_PROFILE_FILE_RECORD_TYPE_CHUNK:
      ++summary->chunk_count;
      break;
    default:
      ++summary->unknown_record_count;
      return iree_ok_status();
  }

  if (iree_any_bit_set(record->header.chunk_flags,
                       IREE_HAL_PROFILE_CHUNK_FLAG_TRUNCATED)) {
    ++summary->truncated_chunk_count;
  }

  if (iree_string_view_equal(record->content_type,
                             IREE_HAL_PROFILE_CONTENT_TYPE_DEVICES)) {
    ++summary->device_chunk_count;
    return iree_profile_summary_process_device_records(summary, record);
  } else if (iree_string_view_equal(record->content_type,
                                    IREE_HAL_PROFILE_CONTENT_TYPE_QUEUES)) {
    ++summary->queue_chunk_count;
    return iree_profile_summary_process_queue_records(summary, record);
  } else if (iree_string_view_equal(
                 record->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLES)) {
    ++summary->executable_chunk_count;
    return iree_profile_summary_process_executable_records(summary, record);
  } else if (iree_string_view_equal(
                 record->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLE_EXPORTS)) {
    ++summary->executable_export_chunk_count;
    return iree_profile_summary_process_executable_export_records(summary,
                                                                  record);
  } else if (iree_string_view_equal(
                 record->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_COMMAND_BUFFERS)) {
    ++summary->command_buffer_chunk_count;
    return iree_profile_summary_process_command_buffer_records(summary, record);
  } else if (iree_string_view_equal(
                 record->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_COMMAND_OPERATIONS)) {
    ++summary->command_operation_chunk_count;
    return iree_profile_summary_process_command_operation_records(summary,
                                                                  record);
  } else if (iree_string_view_equal(
                 record->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_CLOCK_CORRELATIONS)) {
    ++summary->clock_correlation_chunk_count;
    return iree_profile_summary_process_clock_correlation_records(summary,
                                                                  record);
  } else if (iree_string_view_equal(
                 record->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_DISPATCH_EVENTS)) {
    ++summary->dispatch_event_chunk_count;
    return iree_profile_summary_process_dispatch_event_records(summary, record);
  } else if (iree_string_view_equal(
                 record->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_QUEUE_EVENTS)) {
    ++summary->queue_event_chunk_count;
    return iree_profile_summary_process_queue_event_records(summary, record);
  } else if (iree_string_view_equal(
                 record->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_MEMORY_EVENTS)) {
    ++summary->memory_event_chunk_count;
    return iree_profile_summary_process_memory_event_records(summary, record);
  } else if (iree_string_view_equal(
                 record->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_COUNTER_SETS)) {
    ++summary->counter_set_chunk_count;
    return iree_profile_summary_process_counter_set_records(summary, record);
  } else if (iree_string_view_equal(record->content_type,
                                    IREE_HAL_PROFILE_CONTENT_TYPE_COUNTERS)) {
    ++summary->counter_chunk_count;
    return iree_profile_summary_process_counter_records(summary, record);
  } else if (iree_string_view_equal(
                 record->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_COUNTER_SAMPLES)) {
    ++summary->counter_sample_chunk_count;
    return iree_profile_summary_process_counter_sample_records(summary, record);
  }

  ++summary->unknown_chunk_count;
  return iree_ok_status();
}

static bool iree_profile_device_summary_try_fit_clock(
    const iree_profile_device_summary_t* device, double* out_ns_per_tick,
    double* out_tick_frequency_hz) {
  *out_ns_per_tick = 0.0;
  *out_tick_frequency_hz = 0.0;
  if (device->clock_sample_count < 2) return false;

  const iree_hal_profile_clock_correlation_record_t* first =
      &device->first_clock_sample;
  const iree_hal_profile_clock_correlation_record_t* last =
      &device->last_clock_sample;
  if (!iree_all_bits_set(
          first->flags,
          IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_DEVICE_TICK |
              IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_HOST_CPU_TIMESTAMP) ||
      !iree_all_bits_set(
          last->flags,
          IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_DEVICE_TICK |
              IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_HOST_CPU_TIMESTAMP)) {
    return false;
  }
  if (last->device_tick <= first->device_tick ||
      last->host_cpu_timestamp_ns <= first->host_cpu_timestamp_ns) {
    return false;
  }

  const double device_delta_ticks =
      (double)(last->device_tick - first->device_tick);
  const double host_delta_ns =
      (double)(last->host_cpu_timestamp_ns - first->host_cpu_timestamp_ns);
  *out_ns_per_tick = host_delta_ns / device_delta_ticks;
  *out_tick_frequency_hz = 1000000000.0 / *out_ns_per_tick;
  return true;
}

static bool iree_profile_device_summary_clock_covers_dispatches(
    const iree_profile_device_summary_t* device) {
  const uint64_t valid_dispatch_count =
      device->dispatch_event_count - device->invalid_dispatch_event_count;
  if (device->clock_sample_count < 2 || valid_dispatch_count == 0) {
    return false;
  }
  if (!iree_all_bits_set(device->first_clock_sample.flags,
                         IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_DEVICE_TICK) ||
      !iree_all_bits_set(device->last_clock_sample.flags,
                         IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_DEVICE_TICK)) {
    return false;
  }
  return device->first_clock_sample.device_tick <=
             device->earliest_dispatch_start_tick &&
         device->latest_dispatch_end_tick <=
             device->last_clock_sample.device_tick;
}

static void iree_profile_print_summary_text(
    const iree_profile_summary_t* summary, FILE* file) {
  fprintf(file, "IREE HAL profile summary\n");
  fprintf(file,
          "records: file=%" PRIu64 " session_begin=%" PRIu64 " chunks=%" PRIu64
          " session_end=%" PRIu64 " unknown=%" PRIu64 "\n",
          summary->file_record_count, summary->session_begin_count,
          summary->chunk_count, summary->session_end_count,
          summary->unknown_record_count);
  fprintf(file,
          "chunks: devices=%" PRIu64 " queues=%" PRIu64 " executables=%" PRIu64
          " executable_exports=%" PRIu64 " command_buffers=%" PRIu64
          " command_operations=%" PRIu64 " clock_correlations=%" PRIu64
          " dispatch_events=%" PRIu64 " queue_events=%" PRIu64
          " memory_events=%" PRIu64 " counter_sets=%" PRIu64
          " counters=%" PRIu64 " counter_samples=%" PRIu64 " unknown=%" PRIu64
          " truncated=%" PRIu64 "\n",
          summary->device_chunk_count, summary->queue_chunk_count,
          summary->executable_chunk_count,
          summary->executable_export_chunk_count,
          summary->command_buffer_chunk_count,
          summary->command_operation_chunk_count,
          summary->clock_correlation_chunk_count,
          summary->dispatch_event_chunk_count, summary->queue_event_chunk_count,
          summary->memory_event_chunk_count, summary->counter_set_chunk_count,
          summary->counter_chunk_count, summary->counter_sample_chunk_count,
          summary->unknown_chunk_count, summary->truncated_chunk_count);
  fprintf(
      file,
      "metadata_records: executables=%" PRIu64 " executable_exports=%" PRIu64
      " command_buffers=%" PRIu64 " command_operations=%" PRIu64 "\n",
      summary->executable_record_count, summary->executable_export_record_count,
      summary->command_buffer_record_count,
      summary->command_operation_record_count);
  fprintf(file,
          "event_records: queue_events=%" PRIu64 " memory_events=%" PRIu64
          " counter_samples=%" PRIu64 "\n",
          summary->queue_event_record_count, summary->memory_event_record_count,
          summary->counter_sample_record_count);
  fprintf(file,
          "counter_records: counter_sets=%" PRIu64 " counters=%" PRIu64 "\n",
          summary->counter_set_record_count, summary->counter_record_count);
  fprintf(file, "devices:\n");

  for (iree_host_size_t i = 0; i < summary->device_count; ++i) {
    const iree_profile_device_summary_t* device = &summary->devices[i];
    double ns_per_tick = 0.0;
    double tick_frequency_hz = 0.0;
    const bool has_clock_fit = iree_profile_device_summary_try_fit_clock(
        device, &ns_per_tick, &tick_frequency_hz);
    const bool clock_covers_dispatches =
        iree_profile_device_summary_clock_covers_dispatches(device);
    const uint64_t valid_dispatch_count =
        device->dispatch_event_count - device->invalid_dispatch_event_count;

    fprintf(file, "  device[%u]: device_records=%u queues=%u/%u\n",
            device->physical_device_ordinal, device->device_record_count,
            device->queue_record_count, device->queue_count);
    fprintf(file,
            "    clock_samples=%" PRIu64 " min_uncertainty_ns=%" PRId64
            " max_uncertainty_ns=%" PRId64 "\n",
            device->clock_sample_count,
            device->minimum_clock_uncertainty_ns == INT64_MAX
                ? 0
                : device->minimum_clock_uncertainty_ns,
            device->maximum_clock_uncertainty_ns);
    if (has_clock_fit) {
      fprintf(file,
              "    clock_fit: ns_per_tick=%.9f tick_frequency_hz=%.3f"
              " device_delta_ticks=%" PRIu64 " host_delta_ns=%" PRIu64 "\n",
              ns_per_tick, tick_frequency_hz,
              device->last_clock_sample.device_tick -
                  device->first_clock_sample.device_tick,
              device->last_clock_sample.host_cpu_timestamp_ns -
                  device->first_clock_sample.host_cpu_timestamp_ns);
    } else {
      fprintf(file, "    clock_fit: unavailable\n");
    }

    fprintf(file,
            "    dispatches=%" PRIu64 " valid=%" PRIu64 " invalid=%" PRIu64
            "\n",
            device->dispatch_event_count, valid_dispatch_count,
            device->invalid_dispatch_event_count);
    if (valid_dispatch_count != 0) {
      const double average_ticks =
          device->total_dispatch_ticks / (double)valid_dispatch_count;
      fprintf(file,
              "    dispatch_tick_range: start=%" PRIu64 " end=%" PRIu64
              " covered_by_clock_samples=%s\n",
              device->earliest_dispatch_start_tick,
              device->latest_dispatch_end_tick,
              clock_covers_dispatches ? "true" : "false");
      fprintf(file,
              "    dispatch_ticks: min=%" PRIu64 " avg=%.3f max=%" PRIu64
              " total=%.3f\n",
              device->minimum_dispatch_ticks, average_ticks,
              device->maximum_dispatch_ticks, device->total_dispatch_ticks);
      if (has_clock_fit) {
        fprintf(file,
                "    dispatch_time_ns: min=%.3f avg=%.3f max=%.3f"
                " total=%.3f\n",
                (double)device->minimum_dispatch_ticks * ns_per_tick,
                average_ticks * ns_per_tick,
                (double)device->maximum_dispatch_ticks * ns_per_tick,
                device->total_dispatch_ticks * ns_per_tick);
      }
    }
  }
}

static void iree_profile_print_summary_jsonl(
    const iree_profile_summary_t* summary, FILE* file) {
  fprintf(
      file,
      "{\"type\":\"summary\",\"file_records\":%" PRIu64
      ",\"session_begin_records\":%" PRIu64 ",\"chunk_records\":%" PRIu64
      ",\"session_end_records\":%" PRIu64 ",\"unknown_records\":%" PRIu64
      ",\"device_chunks\":%" PRIu64 ",\"queue_chunks\":%" PRIu64
      ",\"executable_chunks\":%" PRIu64 ",\"executable_records\":%" PRIu64
      ",\"executable_export_chunks\":%" PRIu64
      ",\"executable_export_records\":%" PRIu64
      ",\"command_buffer_chunks\":%" PRIu64
      ",\"command_buffer_records\":%" PRIu64
      ",\"command_operation_chunks\":%" PRIu64
      ",\"command_operation_records\":%" PRIu64
      ",\"clock_correlation_chunks\":%" PRIu64
      ",\"dispatch_event_chunks\":%" PRIu64 ",\"queue_event_chunks\":%" PRIu64
      ",\"queue_event_records\":%" PRIu64 ",\"memory_event_chunks\":%" PRIu64
      ",\"memory_event_records\":%" PRIu64 ",\"counter_set_chunks\":%" PRIu64
      ",\"counter_set_records\":%" PRIu64 ",\"counter_chunks\":%" PRIu64
      ",\"counter_records\":%" PRIu64 ",\"counter_sample_chunks\":%" PRIu64
      ",\"counter_sample_records\":%" PRIu64 ",\"unknown_chunks\":%" PRIu64
      ",\"truncated_chunks\":%" PRIu64 "}\n",
      summary->file_record_count, summary->session_begin_count,
      summary->chunk_count, summary->session_end_count,
      summary->unknown_record_count, summary->device_chunk_count,
      summary->queue_chunk_count, summary->executable_chunk_count,
      summary->executable_record_count, summary->executable_export_chunk_count,
      summary->executable_export_record_count,
      summary->command_buffer_chunk_count, summary->command_buffer_record_count,
      summary->command_operation_chunk_count,
      summary->command_operation_record_count,
      summary->clock_correlation_chunk_count,
      summary->dispatch_event_chunk_count, summary->queue_event_chunk_count,
      summary->queue_event_record_count, summary->memory_event_chunk_count,
      summary->memory_event_record_count, summary->counter_set_chunk_count,
      summary->counter_set_record_count, summary->counter_chunk_count,
      summary->counter_record_count, summary->counter_sample_chunk_count,
      summary->counter_sample_record_count, summary->unknown_chunk_count,
      summary->truncated_chunk_count);

  for (iree_host_size_t i = 0; i < summary->device_count; ++i) {
    const iree_profile_device_summary_t* device = &summary->devices[i];
    double ns_per_tick = 0.0;
    double tick_frequency_hz = 0.0;
    const bool has_clock_fit = iree_profile_device_summary_try_fit_clock(
        device, &ns_per_tick, &tick_frequency_hz);
    const bool clock_covers_dispatches =
        iree_profile_device_summary_clock_covers_dispatches(device);
    const uint64_t valid_dispatch_count =
        device->dispatch_event_count - device->invalid_dispatch_event_count;
    const double average_ticks =
        valid_dispatch_count
            ? device->total_dispatch_ticks / (double)valid_dispatch_count
            : 0.0;
    fprintf(file,
            "{\"type\":\"device_summary\",\"physical_device_ordinal\":%u"
            ",\"device_records\":%u,\"queue_records\":%u,\"queues\":%u"
            ",\"clock_samples\":%" PRIu64
            ",\"clock_fit_available\":%s"
            ",\"ns_per_tick\":%.9f,\"tick_frequency_hz\":%.3f"
            ",\"min_clock_uncertainty_ns\":%" PRId64
            ",\"max_clock_uncertainty_ns\":%" PRId64 ",\"dispatches\":%" PRIu64
            ",\"valid_dispatches\":%" PRIu64 ",\"invalid_dispatches\":%" PRIu64
            ",\"min_dispatch_ticks\":%" PRIu64
            ",\"avg_dispatch_ticks\":%.3f"
            ",\"max_dispatch_ticks\":%" PRIu64
            ",\"total_dispatch_ticks\":%.3f"
            ",\"earliest_dispatch_start_tick\":%" PRIu64
            ",\"latest_dispatch_end_tick\":%" PRIu64
            ",\"dispatch_ticks_covered_by_clock_samples\":%s"
            ",\"min_dispatch_ns\":%.3f,\"avg_dispatch_ns\":%.3f"
            ",\"max_dispatch_ns\":%.3f,\"total_dispatch_ns\":%.3f}\n",
            device->physical_device_ordinal, device->device_record_count,
            device->queue_record_count, device->queue_count,
            device->clock_sample_count, has_clock_fit ? "true" : "false",
            ns_per_tick, tick_frequency_hz,
            device->minimum_clock_uncertainty_ns == INT64_MAX
                ? 0
                : device->minimum_clock_uncertainty_ns,
            device->maximum_clock_uncertainty_ns, device->dispatch_event_count,
            valid_dispatch_count, device->invalid_dispatch_event_count,
            valid_dispatch_count ? device->minimum_dispatch_ticks : 0,
            average_ticks,
            valid_dispatch_count ? device->maximum_dispatch_ticks : 0,
            device->total_dispatch_ticks,
            valid_dispatch_count ? device->earliest_dispatch_start_tick : 0,
            valid_dispatch_count ? device->latest_dispatch_end_tick : 0,
            clock_covers_dispatches ? "true" : "false",
            has_clock_fit && valid_dispatch_count
                ? (double)device->minimum_dispatch_ticks * ns_per_tick
                : 0.0,
            has_clock_fit && valid_dispatch_count ? average_ticks * ns_per_tick
                                                  : 0.0,
            has_clock_fit && valid_dispatch_count
                ? (double)device->maximum_dispatch_ticks * ns_per_tick
                : 0.0,
            has_clock_fit ? device->total_dispatch_ticks * ns_per_tick : 0.0);
  }
}

static iree_status_t iree_profile_dispatch_process_queue_records(
    iree_profile_dispatch_context_t* context,
    const iree_hal_profile_file_record_t* record) {
  iree_host_size_t payload_offset = 0;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         payload_offset < record->payload.data_length) {
    iree_host_size_t record_length = 0;
    status = iree_profile_payload_record_length(
        record->content_type, record->payload, payload_offset,
        sizeof(iree_hal_profile_queue_record_t), &record_length);
    if (iree_status_is_ok(status)) {
      iree_hal_profile_queue_record_t record_value;
      memcpy(&record_value, record->payload.data + payload_offset,
             sizeof(record_value));
      status = iree_profile_dispatch_append_queue(context, &record_value);
      payload_offset += record_length;
    }
  }
  return status;
}

static iree_status_t iree_profile_dispatch_process_executable_records(
    iree_profile_dispatch_context_t* context,
    const iree_hal_profile_file_record_t* record) {
  iree_host_size_t payload_offset = 0;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         payload_offset < record->payload.data_length) {
    iree_host_size_t record_length = 0;
    status = iree_profile_payload_record_length(
        record->content_type, record->payload, payload_offset,
        sizeof(iree_hal_profile_executable_record_t), &record_length);
    if (iree_status_is_ok(status)) {
      iree_hal_profile_executable_record_t record_value;
      memcpy(&record_value, record->payload.data + payload_offset,
             sizeof(record_value));
      status = iree_profile_dispatch_append_executable(context, &record_value);
      payload_offset += record_length;
    }
  }
  return status;
}

static iree_status_t iree_profile_dispatch_process_export_records(
    iree_profile_dispatch_context_t* context,
    const iree_hal_profile_file_record_t* record) {
  iree_host_size_t payload_offset = 0;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         payload_offset < record->payload.data_length) {
    iree_host_size_t record_length = 0;
    status = iree_profile_payload_record_length(
        record->content_type, record->payload, payload_offset,
        sizeof(iree_hal_profile_executable_export_record_t), &record_length);
    if (iree_status_is_ok(status)) {
      iree_hal_profile_executable_export_record_t record_value;
      memcpy(&record_value, record->payload.data + payload_offset,
             sizeof(record_value));
      if (record_value.name_length != record_length - sizeof(record_value)) {
        status =
            iree_make_status(IREE_STATUS_DATA_LOSS,
                             "executable export name length is inconsistent");
      }
      if (iree_status_is_ok(status)) {
        iree_profile_dispatch_export_t export_info = {
            .executable_id = record_value.executable_id,
            .flags = record_value.flags,
            .export_ordinal = record_value.export_ordinal,
            .constant_count = record_value.constant_count,
            .binding_count = record_value.binding_count,
            .parameter_count = record_value.parameter_count,
            .workgroup_size = {record_value.workgroup_size[0],
                               record_value.workgroup_size[1],
                               record_value.workgroup_size[2]},
            .pipeline_hash = {record_value.pipeline_hash[0],
                              record_value.pipeline_hash[1]},
            .name =
                iree_make_string_view((const char*)record->payload.data +
                                          payload_offset + sizeof(record_value),
                                      record_value.name_length),
        };
        status = iree_profile_dispatch_append_export(context, &export_info);
      }
      payload_offset += record_length;
    }
  }
  return status;
}

static iree_status_t iree_profile_dispatch_process_command_buffer_records(
    iree_profile_dispatch_context_t* context,
    const iree_hal_profile_file_record_t* record) {
  iree_host_size_t payload_offset = 0;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         payload_offset < record->payload.data_length) {
    iree_host_size_t record_length = 0;
    status = iree_profile_payload_record_length(
        record->content_type, record->payload, payload_offset,
        sizeof(iree_hal_profile_command_buffer_record_t), &record_length);
    if (iree_status_is_ok(status)) {
      iree_hal_profile_command_buffer_record_t record_value;
      memcpy(&record_value, record->payload.data + payload_offset,
             sizeof(record_value));
      status =
          iree_profile_dispatch_append_command_buffer(context, &record_value);
      payload_offset += record_length;
    }
  }
  return status;
}

static iree_status_t iree_profile_dispatch_process_command_operation_records(
    iree_profile_dispatch_context_t* context,
    const iree_hal_profile_file_record_t* record) {
  iree_host_size_t payload_offset = 0;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         payload_offset < record->payload.data_length) {
    iree_host_size_t record_length = 0;
    status = iree_profile_payload_record_length(
        record->content_type, record->payload, payload_offset,
        sizeof(iree_hal_profile_command_operation_record_t), &record_length);
    if (iree_status_is_ok(status)) {
      iree_hal_profile_command_operation_record_t record_value;
      memcpy(&record_value, record->payload.data + payload_offset,
             sizeof(record_value));
      if (!iree_profile_dispatch_find_command_buffer(
              context, record_value.command_buffer_id)) {
        status = iree_make_status(
            IREE_STATUS_DATA_LOSS,
            "command operation references missing command-buffer metadata "
            "command_buffer=%" PRIu64 " command_index=%u",
            record_value.command_buffer_id, record_value.command_index);
      }
      if (iree_status_is_ok(status)) {
        status = iree_profile_dispatch_append_command_operation(context,
                                                                &record_value);
      }
      payload_offset += record_length;
    }
  }
  return status;
}

static iree_status_t iree_profile_dispatch_process_clock_records(
    iree_profile_dispatch_context_t* context,
    const iree_hal_profile_file_record_t* record) {
  iree_host_size_t payload_offset = 0;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         payload_offset < record->payload.data_length) {
    iree_host_size_t record_length = 0;
    status = iree_profile_payload_record_length(
        record->content_type, record->payload, payload_offset,
        sizeof(iree_hal_profile_clock_correlation_record_t), &record_length);
    if (iree_status_is_ok(status)) {
      iree_hal_profile_clock_correlation_record_t clock_record;
      memcpy(&clock_record, record->payload.data + payload_offset,
             sizeof(clock_record));

      iree_profile_dispatch_device_t* device = NULL;
      status = iree_profile_dispatch_get_device(
          context, clock_record.physical_device_ordinal, &device);
      if (iree_status_is_ok(status)) {
        iree_profile_dispatch_record_clock_sample(device, &clock_record);
      }
      payload_offset += record_length;
    }
  }
  return status;
}

static iree_status_t iree_profile_counter_process_counter_set_records(
    iree_profile_counter_context_t* context,
    const iree_hal_profile_file_record_t* record) {
  iree_host_size_t payload_offset = 0;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         payload_offset < record->payload.data_length) {
    iree_host_size_t record_length = 0;
    status = iree_profile_payload_record_length(
        record->content_type, record->payload, payload_offset,
        sizeof(iree_hal_profile_counter_set_record_t), &record_length);
    if (iree_status_is_ok(status)) {
      iree_hal_profile_counter_set_record_t record_value;
      memcpy(&record_value, record->payload.data + payload_offset,
             sizeof(record_value));
      if (record_value.name_length != record_length - sizeof(record_value)) {
        status = iree_make_status(
            IREE_STATUS_DATA_LOSS,
            "counter set name length is inconsistent with record length");
      }
      if (iree_status_is_ok(status)) {
        iree_profile_counter_set_t counter_set = {
            .record = record_value,
            .name =
                iree_make_string_view((const char*)record->payload.data +
                                          payload_offset + sizeof(record_value),
                                      record_value.name_length),
        };
        status = iree_profile_counter_append_counter_set(context, &counter_set);
      }
      payload_offset += record_length;
    }
  }
  return status;
}

static iree_status_t iree_profile_counter_process_counter_records(
    iree_profile_counter_context_t* context,
    const iree_hal_profile_file_record_t* record) {
  iree_host_size_t payload_offset = 0;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         payload_offset < record->payload.data_length) {
    iree_host_size_t record_length = 0;
    status = iree_profile_payload_record_length(
        record->content_type, record->payload, payload_offset,
        sizeof(iree_hal_profile_counter_record_t), &record_length);
    if (iree_status_is_ok(status)) {
      iree_hal_profile_counter_record_t record_value;
      memcpy(&record_value, record->payload.data + payload_offset,
             sizeof(record_value));
      iree_host_size_t trailing_length = 0;
      if (!iree_host_size_checked_add(record_value.block_name_length,
                                      record_value.name_length,
                                      &trailing_length) ||
          !iree_host_size_checked_add(trailing_length,
                                      record_value.description_length,
                                      &trailing_length) ||
          trailing_length != record_length - sizeof(record_value)) {
        status = iree_make_status(
            IREE_STATUS_DATA_LOSS,
            "counter string lengths are inconsistent with record length");
      }
      if (iree_status_is_ok(status) &&
          !iree_profile_counter_find_counter_set(context,
                                                 record_value.counter_set_id)) {
        status = iree_make_status(
            IREE_STATUS_DATA_LOSS,
            "counter references missing counter-set metadata "
            "counter_set=%" PRIu64 " counter_ordinal=%u",
            record_value.counter_set_id, record_value.counter_ordinal);
      }
      if (iree_status_is_ok(status)) {
        const char* string_base = (const char*)record->payload.data +
                                  payload_offset + sizeof(record_value);
        iree_profile_counter_t counter = {
            .record = record_value,
            .block_name = iree_make_string_view(string_base,
                                                record_value.block_name_length),
            .name = iree_make_string_view(
                string_base + record_value.block_name_length,
                record_value.name_length),
            .description = iree_make_string_view(
                string_base + record_value.block_name_length +
                    record_value.name_length,
                record_value.description_length),
        };
        status = iree_profile_counter_append_counter(context, &counter);
      }
      payload_offset += record_length;
    }
  }
  return status;
}

static iree_status_t iree_profile_counter_process_metadata_record(
    iree_profile_counter_context_t* context,
    const iree_hal_profile_file_record_t* record) {
  if (record->header.record_type != IREE_HAL_PROFILE_FILE_RECORD_TYPE_CHUNK) {
    return iree_ok_status();
  }
  if (iree_string_view_equal(record->content_type,
                             IREE_HAL_PROFILE_CONTENT_TYPE_COUNTER_SETS)) {
    return iree_profile_counter_process_counter_set_records(context, record);
  } else if (iree_string_view_equal(record->content_type,
                                    IREE_HAL_PROFILE_CONTENT_TYPE_COUNTERS)) {
    return iree_profile_counter_process_counter_records(context, record);
  }
  return iree_ok_status();
}

static iree_status_t iree_profile_dispatch_process_metadata_record(
    iree_profile_dispatch_context_t* context,
    const iree_hal_profile_file_record_t* record) {
  if (record->header.record_type != IREE_HAL_PROFILE_FILE_RECORD_TYPE_CHUNK) {
    return iree_ok_status();
  }
  if (iree_string_view_equal(record->content_type,
                             IREE_HAL_PROFILE_CONTENT_TYPE_QUEUES)) {
    return iree_profile_dispatch_process_queue_records(context, record);
  } else if (iree_string_view_equal(
                 record->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLES)) {
    return iree_profile_dispatch_process_executable_records(context, record);
  } else if (iree_string_view_equal(
                 record->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLE_EXPORTS)) {
    return iree_profile_dispatch_process_export_records(context, record);
  } else if (iree_string_view_equal(
                 record->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_COMMAND_BUFFERS)) {
    return iree_profile_dispatch_process_command_buffer_records(context,
                                                                record);
  } else if (iree_string_view_equal(
                 record->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_COMMAND_OPERATIONS)) {
    return iree_profile_dispatch_process_command_operation_records(context,
                                                                   record);
  } else if (iree_string_view_equal(
                 record->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_CLOCK_CORRELATIONS)) {
    return iree_profile_dispatch_process_clock_records(context, record);
  }
  return iree_ok_status();
}

static const char* iree_profile_counter_unit_name(
    iree_hal_profile_counter_unit_t unit) {
  switch (unit) {
    case IREE_HAL_PROFILE_COUNTER_UNIT_NONE:
      return "none";
    case IREE_HAL_PROFILE_COUNTER_UNIT_COUNT:
      return "count";
    case IREE_HAL_PROFILE_COUNTER_UNIT_CYCLES:
      return "cycles";
    case IREE_HAL_PROFILE_COUNTER_UNIT_BYTES:
      return "bytes";
    default:
      return "unknown";
  }
}

static bool iree_profile_counter_sample_matches_id(
    const iree_hal_profile_counter_sample_record_t* sample, int64_t id_filter) {
  if (id_filter < 0) return true;
  const uint64_t id = (uint64_t)id_filter;
  return sample->sample_id == id || sample->dispatch_event_id == id ||
         sample->submission_id == id || sample->command_buffer_id == id;
}

static bool iree_profile_counter_filter_matches(
    const iree_profile_counter_set_t* counter_set,
    const iree_profile_counter_t* counter, iree_string_view_t key,
    iree_string_view_t filter) {
  return iree_profile_dispatch_key_matches(key, filter) ||
         iree_profile_dispatch_key_matches(counter_set->name, filter) ||
         iree_profile_dispatch_key_matches(counter->block_name, filter) ||
         iree_profile_dispatch_key_matches(counter->name, filter);
}

static iree_status_t iree_profile_counter_resolve_sample_key(
    const iree_profile_dispatch_context_t* dispatch_context,
    const iree_hal_profile_counter_sample_record_t* sample,
    char* numeric_buffer, iree_host_size_t numeric_buffer_capacity,
    iree_string_view_t* out_key) {
  *out_key = IREE_SV("unattributed");
  if (sample->executable_id == 0 || sample->export_ordinal == UINT32_MAX) {
    return iree_ok_status();
  }
  return iree_profile_dispatch_resolve_key(
      dispatch_context, sample->physical_device_ordinal, sample->executable_id,
      sample->export_ordinal, numeric_buffer, numeric_buffer_capacity, out_key);
}

static iree_status_t iree_profile_counter_sum_value(
    const iree_profile_counter_t* counter,
    const iree_hal_profile_counter_sample_record_t* sample,
    const uint8_t* sample_values, double* out_value_sum) {
  *out_value_sum = 0.0;
  if (counter->record.sample_value_offset > sample->sample_value_count ||
      sample->sample_value_count - counter->record.sample_value_offset <
          counter->record.sample_value_count) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "counter sample value layout is inconsistent with counter metadata "
        "counter_set=%" PRIu64 " counter_ordinal=%u",
        counter->record.counter_set_id, counter->record.counter_ordinal);
  }

  double value_sum = 0.0;
  for (uint32_t i = 0; i < counter->record.sample_value_count; ++i) {
    uint64_t raw_value = 0;
    const iree_host_size_t value_offset =
        ((iree_host_size_t)counter->record.sample_value_offset + i) *
        sizeof(raw_value);
    memcpy(&raw_value, sample_values + value_offset, sizeof(raw_value));
    value_sum += (double)raw_value;
  }
  *out_value_sum = value_sum;
  return iree_ok_status();
}

static void iree_profile_counter_print_sample_values_jsonl(
    const iree_profile_counter_t* counter, const uint8_t* sample_values,
    FILE* file) {
  fputc('[', file);
  for (uint32_t i = 0; i < counter->record.sample_value_count; ++i) {
    uint64_t raw_value = 0;
    const iree_host_size_t value_offset =
        ((iree_host_size_t)counter->record.sample_value_offset + i) *
        sizeof(raw_value);
    memcpy(&raw_value, sample_values + value_offset, sizeof(raw_value));
    if (i != 0) fputc(',', file);
    fprintf(file, "%" PRIu64, raw_value);
  }
  fputc(']', file);
}

static void iree_profile_counter_record_aggregate_sample(
    iree_profile_counter_aggregate_t* aggregate,
    const iree_profile_counter_t* counter,
    const iree_hal_profile_counter_sample_record_t* sample, double value_sum) {
  ++aggregate->sample_count;
  aggregate->raw_value_count += counter->record.sample_value_count;
  if (aggregate->first_sample_id == 0) {
    aggregate->first_sample_id = sample->sample_id;
  }
  aggregate->last_sample_id = sample->sample_id;
  aggregate->minimum_value = iree_min(aggregate->minimum_value, value_sum);
  aggregate->maximum_value = iree_max(aggregate->maximum_value, value_sum);
  aggregate->total_value += value_sum;

  const double delta = value_sum - aggregate->mean_value;
  aggregate->mean_value += delta / (double)aggregate->sample_count;
  const double delta2 = value_sum - aggregate->mean_value;
  aggregate->m2_value += delta * delta2;
}

static void iree_profile_counter_print_sample_jsonl(
    const iree_hal_profile_counter_sample_record_t* sample,
    const iree_profile_counter_set_t* counter_set,
    const iree_profile_counter_t* counter, iree_string_view_t key,
    const uint8_t* sample_values, double value_sum, FILE* file) {
  fprintf(file,
          "{\"type\":\"counter_sample\",\"sample_id\":%" PRIu64
          ",\"counter_set_id\":%" PRIu64 ",\"counter_set\":",
          sample->sample_id, sample->counter_set_id);
  iree_profile_fprint_json_string(file, counter_set->name);
  fprintf(file, ",\"counter_ordinal\":%u,\"counter\":",
          counter->record.counter_ordinal);
  iree_profile_fprint_json_string(file, counter->name);
  fprintf(file, ",\"block\":");
  iree_profile_fprint_json_string(file, counter->block_name);
  fprintf(file, ",\"unit\":");
  iree_profile_fprint_json_string(
      file, iree_make_cstring_view(
                iree_profile_counter_unit_name(counter->record.unit)));
  fprintf(file,
          ",\"dispatch_event_id\":%" PRIu64 ",\"submission_id\":%" PRIu64
          ",\"command_buffer_id\":%" PRIu64
          ",\"command_index\":%u,\"executable_id\":%" PRIu64
          ",\"export_ordinal\":%u,\"key\":",
          sample->dispatch_event_id, sample->submission_id,
          sample->command_buffer_id, sample->command_index,
          sample->executable_id, sample->export_ordinal);
  iree_profile_fprint_json_string(file, key);
  fprintf(file,
          ",\"physical_device_ordinal\":%u,\"queue_ordinal\":%u"
          ",\"stream_id\":%" PRIu64
          ",\"flags\":%u,\"value\":%.3f"
          ",\"values\":",
          sample->physical_device_ordinal, sample->queue_ordinal,
          sample->stream_id, sample->flags, value_sum);
  iree_profile_counter_print_sample_values_jsonl(counter, sample_values, file);
  fputs("}\n", file);
}

static iree_status_t iree_profile_counter_process_sample_records(
    iree_profile_counter_context_t* counter_context,
    const iree_profile_dispatch_context_t* dispatch_context,
    const iree_hal_profile_file_record_t* record, iree_string_view_t filter,
    int64_t id_filter, bool emit_samples, FILE* file) {
  if (record->header.record_type != IREE_HAL_PROFILE_FILE_RECORD_TYPE_CHUNK) {
    return iree_ok_status();
  }
  if (!iree_string_view_equal(record->content_type,
                              IREE_HAL_PROFILE_CONTENT_TYPE_COUNTER_SAMPLES)) {
    return iree_ok_status();
  }

  const bool is_truncated = iree_any_bit_set(
      record->header.chunk_flags, IREE_HAL_PROFILE_CHUNK_FLAG_TRUNCATED);
  iree_host_size_t payload_offset = 0;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         payload_offset < record->payload.data_length) {
    iree_host_size_t record_length = 0;
    status = iree_profile_payload_record_length(
        record->content_type, record->payload, payload_offset,
        sizeof(iree_hal_profile_counter_sample_record_t), &record_length);
    if (iree_status_is_ok(status)) {
      iree_hal_profile_counter_sample_record_t sample;
      memcpy(&sample, record->payload.data + payload_offset, sizeof(sample));
      ++counter_context->total_sample_count;

      iree_host_size_t values_length = 0;
      if (!iree_host_size_checked_mul(sample.sample_value_count,
                                      sizeof(uint64_t), &values_length) ||
          values_length != record_length - sizeof(sample)) {
        status = iree_make_status(
            IREE_STATUS_DATA_LOSS,
            "counter sample value count is inconsistent with record length");
      }

      const iree_profile_counter_set_t* counter_set = NULL;
      if (iree_status_is_ok(status)) {
        counter_set = iree_profile_counter_find_counter_set(
            counter_context, sample.counter_set_id);
        if (!counter_set) {
          status = iree_make_status(
              IREE_STATUS_DATA_LOSS,
              "counter sample references missing counter-set metadata "
              "sample=%" PRIu64 " counter_set=%" PRIu64,
              sample.sample_id, sample.counter_set_id);
        }
      }
      if (iree_status_is_ok(status) &&
          counter_set->record.sample_value_count != sample.sample_value_count) {
        status = iree_make_status(
            IREE_STATUS_DATA_LOSS,
            "counter sample value count does not match counter-set metadata "
            "sample=%" PRIu64 " counter_set=%" PRIu64,
            sample.sample_id, sample.counter_set_id);
      }

      char numeric_buffer[128];
      iree_string_view_t key = iree_string_view_empty();
      if (iree_status_is_ok(status)) {
        status = iree_profile_counter_resolve_sample_key(
            dispatch_context, &sample, numeric_buffer, sizeof(numeric_buffer),
            &key);
      }

      bool matched_sample = false;
      const uint8_t* sample_values =
          record->payload.data + payload_offset + sizeof(sample);
      if (iree_status_is_ok(status) &&
          iree_profile_counter_sample_matches_id(&sample, id_filter)) {
        bool found_counter = false;
        for (iree_host_size_t i = 0;
             i < counter_context->counter_count && iree_status_is_ok(status);
             ++i) {
          const iree_profile_counter_t* counter = &counter_context->counters[i];
          if (counter->record.counter_set_id != sample.counter_set_id) {
            continue;
          }
          found_counter = true;
          if (!iree_profile_counter_filter_matches(counter_set, counter, key,
                                                   filter)) {
            continue;
          }

          double value_sum = 0.0;
          status = iree_profile_counter_sum_value(counter, &sample,
                                                  sample_values, &value_sum);
          if (iree_status_is_ok(status)) {
            matched_sample = true;
            iree_profile_counter_aggregate_t* aggregate = NULL;
            status = iree_profile_counter_get_aggregate(
                counter_context, sample.physical_device_ordinal,
                sample.queue_ordinal, sample.stream_id, sample.counter_set_id,
                counter->record.counter_ordinal, sample.executable_id,
                sample.export_ordinal, sample.command_buffer_id, &aggregate);
            if (iree_status_is_ok(status)) {
              iree_profile_counter_record_aggregate_sample(aggregate, counter,
                                                           &sample, value_sum);
            }
            if (iree_status_is_ok(status) && emit_samples) {
              iree_profile_counter_print_sample_jsonl(
                  &sample, counter_set, counter, key, sample_values, value_sum,
                  file);
            }
          }
        }
        if (iree_status_is_ok(status) && !found_counter) {
          status = iree_make_status(
              IREE_STATUS_DATA_LOSS,
              "counter sample references counter set with no counter metadata "
              "sample=%" PRIu64 " counter_set=%" PRIu64,
              sample.sample_id, sample.counter_set_id);
        }
      }
      if (iree_status_is_ok(status) && matched_sample) {
        ++counter_context->matched_sample_count;
        if (is_truncated) ++counter_context->truncated_sample_count;
      }
      payload_offset += record_length;
    }
  }
  return status;
}

static void iree_profile_dispatch_print_event_jsonl(
    const iree_hal_profile_file_record_t* file_record,
    const iree_hal_profile_dispatch_event_t* event, iree_string_view_t key,
    double ns_per_tick, bool has_clock_fit, FILE* file) {
  const bool is_valid = event->start_tick != 0 && event->end_tick != 0 &&
                        event->end_tick >= event->start_tick;
  const uint64_t duration_ticks =
      is_valid ? event->end_tick - event->start_tick : 0;

  fprintf(file,
          "{\"type\":\"dispatch_event\",\"physical_device_ordinal\":%u"
          ",\"queue_ordinal\":%u,\"stream_id\":%" PRIu64
          ",\"event_id\":%" PRIu64 ",\"submission_id\":%" PRIu64,
          file_record->header.physical_device_ordinal,
          file_record->header.queue_ordinal, file_record->header.stream_id,
          event->event_id, event->submission_id);
  fprintf(file,
          ",\"command_buffer_id\":%" PRIu64
          ",\"command_index\":%u"
          ",\"executable_id\":%" PRIu64 ",\"export_ordinal\":%u",
          event->command_buffer_id, event->command_index, event->executable_id,
          event->export_ordinal);
  fprintf(file, ",\"key\":");
  iree_profile_fprint_json_string(file, key);
  fprintf(file,
          ",\"flags\":%u,\"workgroup_count\":[%u,%u,%u]"
          ",\"workgroup_size\":[%u,%u,%u]",
          event->flags, event->workgroup_count[0], event->workgroup_count[1],
          event->workgroup_count[2], event->workgroup_size[0],
          event->workgroup_size[1], event->workgroup_size[2]);
  fprintf(file,
          ",\"start_tick\":%" PRIu64 ",\"end_tick\":%" PRIu64
          ",\"duration_ticks\":%" PRIu64 ",\"valid\":%s",
          event->start_tick, event->end_tick, duration_ticks,
          is_valid ? "true" : "false");
  fprintf(file, ",\"clock_fit_available\":%s",
          has_clock_fit ? "true" : "false");
  fprintf(
      file, ",\"duration_ns\":%.3f",
      has_clock_fit && is_valid ? (double)duration_ticks * ns_per_tick : 0.0);
  fputs("}\n", file);
}

static const char* iree_profile_command_operation_type_name(
    iree_hal_profile_command_operation_type_t type) {
  switch (type) {
    case IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_BARRIER:
      return "barrier";
    case IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_DISPATCH:
      return "dispatch";
    case IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_FILL:
      return "fill";
    case IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_COPY:
      return "copy";
    case IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_UPDATE:
      return "update";
    case IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_PROFILE_MARKER:
      return "profile_marker";
    case IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_BRANCH:
      return "branch";
    case IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_COND_BRANCH:
      return "cond_branch";
    case IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_RETURN:
      return "return";
    default:
      return "unknown";
  }
}

static const char* iree_profile_queue_event_type_name(
    iree_hal_profile_queue_event_type_t type) {
  switch (type) {
    case IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_BARRIER:
      return "barrier";
    case IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_DISPATCH:
      return "dispatch";
    case IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_EXECUTE:
      return "execute";
    case IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_COPY:
      return "copy";
    case IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_FILL:
      return "fill";
    case IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_UPDATE:
      return "update";
    case IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_READ:
      return "read";
    case IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_WRITE:
      return "write";
    case IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_ALLOCA:
      return "alloca";
    case IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_DEALLOCA:
      return "dealloca";
    case IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_HOST_CALL:
      return "host_call";
    default:
      return "unknown";
  }
}

static const char* iree_profile_queue_dependency_strategy_name(
    iree_hal_profile_queue_dependency_strategy_t strategy) {
  switch (strategy) {
    case IREE_HAL_PROFILE_QUEUE_DEPENDENCY_STRATEGY_NONE:
      return "none";
    case IREE_HAL_PROFILE_QUEUE_DEPENDENCY_STRATEGY_INLINE:
      return "inline";
    case IREE_HAL_PROFILE_QUEUE_DEPENDENCY_STRATEGY_DEVICE_BARRIER:
      return "device_barrier";
    case IREE_HAL_PROFILE_QUEUE_DEPENDENCY_STRATEGY_SOFTWARE_DEFER:
      return "software_defer";
    default:
      return "unknown";
  }
}

static bool iree_profile_queue_event_matches(
    const iree_hal_profile_queue_event_t* event, int64_t id_filter,
    iree_string_view_t filter) {
  if (id_filter >= 0 && event->submission_id != (uint64_t)id_filter) {
    return false;
  }
  iree_string_view_t type_name =
      iree_make_cstring_view(iree_profile_queue_event_type_name(event->type));
  return iree_profile_dispatch_key_matches(type_name, filter);
}

static iree_status_t iree_profile_dispatch_process_event_records(
    iree_profile_dispatch_context_t* context,
    const iree_hal_profile_file_record_t* record, iree_string_view_t filter,
    iree_profile_projection_mode_t projection_mode, int64_t id_filter,
    bool emit_events, FILE* file) {
  iree_profile_dispatch_device_t* device = NULL;
  IREE_RETURN_IF_ERROR(iree_profile_dispatch_get_device(
      context, record->header.physical_device_ordinal, &device));
  double ns_per_tick = 0.0;
  double tick_frequency_hz = 0.0;
  const bool has_clock_fit = iree_profile_dispatch_device_try_fit_clock(
      device, &ns_per_tick, &tick_frequency_hz);
  (void)tick_frequency_hz;

  iree_host_size_t payload_offset = 0;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         payload_offset < record->payload.data_length) {
    iree_host_size_t record_length = 0;
    status = iree_profile_payload_record_length(
        record->content_type, record->payload, payload_offset,
        sizeof(iree_hal_profile_dispatch_event_t), &record_length);
    if (iree_status_is_ok(status)) {
      iree_hal_profile_dispatch_event_t event;
      memcpy(&event, record->payload.data + payload_offset, sizeof(event));
      ++context->total_dispatch_count;

      char numeric_buffer[128];
      iree_string_view_t key = iree_string_view_empty();
      const bool id_matches = iree_profile_dispatch_event_matches_id(
          &event, projection_mode, id_filter);
      if (id_matches) {
        status = iree_profile_dispatch_validate_event_metadata(
            context, record, &event, projection_mode);
      }
      if (iree_status_is_ok(status) && id_matches) {
        status = iree_profile_dispatch_resolve_key(
            context, record->header.physical_device_ordinal,
            event.executable_id, event.export_ordinal, numeric_buffer,
            sizeof(numeric_buffer), &key);
      }
      if (iree_status_is_ok(status) && !iree_string_view_is_empty(key) &&
          iree_profile_dispatch_key_matches(key, filter)) {
        ++context->matched_dispatch_count;
        const bool is_valid = event.start_tick != 0 && event.end_tick != 0 &&
                              event.end_tick >= event.start_tick;
        if (is_valid) {
          ++context->valid_dispatch_count;
          iree_profile_dispatch_record_top_event(context, record, &event);
        } else {
          ++context->invalid_dispatch_count;
        }
        if (emit_events) {
          iree_profile_dispatch_print_event_jsonl(
              record, &event, key, ns_per_tick, has_clock_fit, file);
        } else {
          iree_profile_dispatch_aggregate_t* aggregate = NULL;
          status = iree_profile_dispatch_get_aggregate(
              context, record->header.physical_device_ordinal,
              event.executable_id, event.export_ordinal, &aggregate);
          if (iree_status_is_ok(status)) {
            iree_profile_dispatch_record_aggregate_event(aggregate, &event);
          }
          if (iree_status_is_ok(status) && event.command_buffer_id != 0) {
            iree_profile_dispatch_command_aggregate_t* command_aggregate = NULL;
            status = iree_profile_dispatch_get_command_aggregate(
                context, event.command_buffer_id, event.submission_id,
                record->header.physical_device_ordinal,
                record->header.queue_ordinal, record->header.stream_id,
                &command_aggregate);
            if (iree_status_is_ok(status)) {
              iree_profile_dispatch_record_command_aggregate_event(
                  command_aggregate, &event);
            }
          }
          if (iree_status_is_ok(status)) {
            iree_profile_dispatch_queue_aggregate_t* queue_aggregate = NULL;
            status = iree_profile_dispatch_get_queue_aggregate(
                context, record->header.physical_device_ordinal,
                record->header.queue_ordinal, record->header.stream_id,
                event.submission_id, &queue_aggregate);
            if (iree_status_is_ok(status)) {
              iree_profile_dispatch_record_queue_aggregate_event(
                  queue_aggregate, &event);
            }
          }
        }
      }
      payload_offset += record_length;
    }
  }
  return status;
}

static iree_status_t iree_profile_dispatch_process_queue_event_records(
    iree_profile_dispatch_context_t* context,
    const iree_hal_profile_file_record_t* record, iree_string_view_t filter,
    int64_t id_filter) {
  iree_host_size_t payload_offset = 0;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         payload_offset < record->payload.data_length) {
    iree_host_size_t record_length = 0;
    status = iree_profile_payload_record_length(
        record->content_type, record->payload, payload_offset,
        sizeof(iree_hal_profile_queue_event_t), &record_length);
    if (iree_status_is_ok(status)) {
      iree_hal_profile_queue_event_t event;
      memcpy(&event, record->payload.data + payload_offset, sizeof(event));
      ++context->total_queue_event_count;
      if (iree_profile_queue_event_matches(&event, id_filter, filter)) {
        ++context->matched_queue_event_count;
        const iree_profile_dispatch_queue_t* queue_info =
            iree_profile_dispatch_find_queue(
                context, event.physical_device_ordinal, event.queue_ordinal,
                event.stream_id);
        if (!queue_info) {
          status = iree_make_status(
              IREE_STATUS_DATA_LOSS,
              "queue event references missing queue metadata "
              "device=%u queue=%u stream=%" PRIu64 " submission=%" PRIu64,
              event.physical_device_ordinal, event.queue_ordinal,
              event.stream_id, event.submission_id);
        }
        if (iree_status_is_ok(status)) {
          status = iree_profile_dispatch_append_queue_event(context, &event);
        }
      }
      payload_offset += record_length;
    }
  }
  return status;
}

static iree_status_t iree_profile_dispatch_process_events_record(
    iree_profile_dispatch_context_t* context,
    const iree_hal_profile_file_record_t* record, iree_string_view_t filter,
    iree_profile_projection_mode_t projection_mode, int64_t id_filter,
    bool emit_events, FILE* file) {
  if (record->header.record_type != IREE_HAL_PROFILE_FILE_RECORD_TYPE_CHUNK) {
    return iree_ok_status();
  }
  if (!iree_string_view_equal(record->content_type,
                              IREE_HAL_PROFILE_CONTENT_TYPE_DISPATCH_EVENTS)) {
    if (projection_mode == IREE_PROFILE_PROJECTION_MODE_QUEUE &&
        iree_string_view_equal(record->content_type,
                               IREE_HAL_PROFILE_CONTENT_TYPE_QUEUE_EVENTS)) {
      return iree_profile_dispatch_process_queue_event_records(
          context, record, filter, id_filter);
    }
    return iree_ok_status();
  }
  return iree_profile_dispatch_process_event_records(
      context, record, filter, projection_mode, id_filter, emit_events, file);
}

static const iree_profile_dispatch_device_t* iree_profile_dispatch_find_device(
    const iree_profile_dispatch_context_t* context,
    uint32_t physical_device_ordinal) {
  for (iree_host_size_t i = 0; i < context->device_count; ++i) {
    if (context->devices[i].physical_device_ordinal ==
        physical_device_ordinal) {
      return &context->devices[i];
    }
  }
  return NULL;
}

static double iree_profile_dispatch_span_ticks(uint64_t earliest_start_tick,
                                               uint64_t latest_end_tick) {
  if (earliest_start_tick == UINT64_MAX || latest_end_tick == 0 ||
      latest_end_tick < earliest_start_tick) {
    return 0.0;
  }
  return (double)(latest_end_tick - earliest_start_tick);
}

static iree_status_t iree_profile_command_operation_resolve_key(
    const iree_profile_dispatch_context_t* context,
    const iree_hal_profile_command_operation_record_t* operation,
    char* numeric_buffer, iree_host_size_t numeric_buffer_capacity,
    iree_string_view_t* out_key) {
  *out_key = iree_make_cstring_view(
      iree_profile_command_operation_type_name(operation->type));
  if (operation->type != IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_DISPATCH ||
      operation->executable_id == 0 ||
      operation->export_ordinal == UINT32_MAX) {
    return iree_ok_status();
  }

  const iree_profile_dispatch_command_buffer_t* command_buffer =
      iree_profile_dispatch_find_command_buffer(context,
                                                operation->command_buffer_id);
  if (!command_buffer) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "command operation references missing command-buffer metadata "
        "command_buffer=%" PRIu64 " command_index=%u",
        operation->command_buffer_id, operation->command_index);
  }
  return iree_profile_dispatch_resolve_key(
      context, command_buffer->record.physical_device_ordinal,
      operation->executable_id, operation->export_ordinal, numeric_buffer,
      numeric_buffer_capacity, out_key);
}

static bool iree_profile_command_operation_filter_matches(
    iree_string_view_t operation_name, iree_string_view_t key,
    iree_string_view_t filter) {
  return iree_profile_dispatch_key_matches(operation_name, filter) ||
         iree_profile_dispatch_key_matches(key, filter);
}

static iree_status_t iree_profile_command_count_matching_operations(
    const iree_profile_dispatch_context_t* context, iree_string_view_t filter,
    int64_t id_filter, iree_host_size_t* out_operation_count) {
  *out_operation_count = 0;
  iree_host_size_t operation_count = 0;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < context->command_operation_count && iree_status_is_ok(status); ++i) {
    const iree_hal_profile_command_operation_record_t* operation =
        &context->command_operations[i].record;
    if (id_filter >= 0 && operation->command_buffer_id != (uint64_t)id_filter) {
      continue;
    }
    const char* operation_name =
        iree_profile_command_operation_type_name(operation->type);
    char numeric_buffer[128];
    iree_string_view_t key = iree_string_view_empty();
    status = iree_profile_command_operation_resolve_key(
        context, operation, numeric_buffer, sizeof(numeric_buffer), &key);
    if (iree_status_is_ok(status) &&
        iree_profile_command_operation_filter_matches(
            iree_make_cstring_view(operation_name), key, filter)) {
      ++operation_count;
    }
  }
  if (iree_status_is_ok(status)) {
    *out_operation_count = operation_count;
  }
  return status;
}

static iree_status_t iree_profile_command_print_operation_text(
    const iree_profile_dispatch_context_t* context,
    const iree_hal_profile_command_operation_record_t* operation,
    iree_string_view_t filter, FILE* file) {
  const char* operation_name =
      iree_profile_command_operation_type_name(operation->type);
  char numeric_buffer[128];
  iree_string_view_t key = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_profile_command_operation_resolve_key(
      context, operation, numeric_buffer, sizeof(numeric_buffer), &key));
  if (!iree_profile_command_operation_filter_matches(
          iree_make_cstring_view(operation_name), key, filter)) {
    return iree_ok_status();
  }

  fprintf(file, "    command[%u]: op=%s block=%u local=%u flags=0x%x",
          operation->command_index, operation_name, operation->block_ordinal,
          operation->block_command_ordinal, operation->flags);
  if (operation->type == IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_DISPATCH) {
    fprintf(file,
            " executable=%" PRIu64
            " export=%u key=%.*s bindings=%u"
            " workgroups=[%u,%u,%u] workgroup_size=[%u,%u,%u]",
            operation->executable_id, operation->export_ordinal, (int)key.size,
            key.data, operation->binding_count, operation->workgroup_count[0],
            operation->workgroup_count[1], operation->workgroup_count[2],
            operation->workgroup_size[0], operation->workgroup_size[1],
            operation->workgroup_size[2]);
  } else if (operation->type == IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_FILL ||
             operation->type ==
                 IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_UPDATE) {
    fprintf(file, " target=%u target_offset=%" PRIu64 " length=%" PRIu64,
            operation->target_ordinal, operation->target_offset,
            operation->length);
  } else if (operation->type == IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_COPY) {
    fprintf(file,
            " source=%u source_offset=%" PRIu64
            " target=%u target_offset=%" PRIu64 " length=%" PRIu64,
            operation->source_ordinal, operation->source_offset,
            operation->target_ordinal, operation->target_offset,
            operation->length);
  } else if (operation->type ==
                 IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_BRANCH ||
             operation->type ==
                 IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_COND_BRANCH) {
    fprintf(file, " target_block=%u alternate_block=%u",
            operation->target_block_ordinal,
            operation->alternate_block_ordinal);
  }
  fputc('\n', file);
  return iree_ok_status();
}

static iree_status_t iree_profile_command_print_operation_jsonl(
    const iree_profile_dispatch_context_t* context,
    const iree_hal_profile_command_operation_record_t* operation,
    iree_string_view_t filter, FILE* file) {
  const char* operation_name =
      iree_profile_command_operation_type_name(operation->type);
  char numeric_buffer[128];
  iree_string_view_t key = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_profile_command_operation_resolve_key(
      context, operation, numeric_buffer, sizeof(numeric_buffer), &key));
  if (!iree_profile_command_operation_filter_matches(
          iree_make_cstring_view(operation_name), key, filter)) {
    return iree_ok_status();
  }

  fprintf(file,
          "{\"type\":\"command_operation\",\"command_buffer_id\":%" PRIu64
          ",\"command_index\":%u,\"op\":\"%s\",\"flags\":%u"
          ",\"block_ordinal\":%u,\"block_command_ordinal\":%u",
          operation->command_buffer_id, operation->command_index,
          operation_name, operation->flags, operation->block_ordinal,
          operation->block_command_ordinal);
  fprintf(file, ",\"key\":");
  iree_profile_fprint_json_string(file, key);
  fprintf(file,
          ",\"executable_id\":%" PRIu64
          ",\"export_ordinal\":%u"
          ",\"binding_count\":%u,\"workgroup_count\":[%u,%u,%u]"
          ",\"workgroup_size\":[%u,%u,%u]"
          ",\"source_ordinal\":%u,\"target_ordinal\":%u"
          ",\"source_offset\":%" PRIu64 ",\"target_offset\":%" PRIu64
          ",\"length\":%" PRIu64
          ",\"target_block_ordinal\":%u,\"alternate_block_ordinal\":%u}\n",
          operation->executable_id, operation->export_ordinal,
          operation->binding_count, operation->workgroup_count[0],
          operation->workgroup_count[1], operation->workgroup_count[2],
          operation->workgroup_size[0], operation->workgroup_size[1],
          operation->workgroup_size[2], operation->source_ordinal,
          operation->target_ordinal, operation->source_offset,
          operation->target_offset, operation->length,
          operation->target_block_ordinal, operation->alternate_block_ordinal);
  return iree_ok_status();
}

static iree_status_t iree_profile_dispatch_print_text(
    const iree_profile_dispatch_context_t* context, iree_string_view_t filter,
    FILE* file) {
  fprintf(file, "IREE HAL profile dispatch summary\n");
  fprintf(file, "filter: %.*s\n", (int)filter.size, filter.data);
  fprintf(file,
          "dispatches: total=%" PRIu64 " matched=%" PRIu64 " valid=%" PRIu64
          " invalid=%" PRIu64 " groups=%" PRIhsz "\n",
          context->total_dispatch_count, context->matched_dispatch_count,
          context->valid_dispatch_count, context->invalid_dispatch_count,
          context->aggregate_count);
  fprintf(file, "groups:\n");

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < context->aggregate_count && iree_status_is_ok(status); ++i) {
    const iree_profile_dispatch_aggregate_t* aggregate =
        &context->aggregates[i];
    const iree_profile_dispatch_device_t* device = NULL;
    for (iree_host_size_t j = 0; j < context->device_count; ++j) {
      if (context->devices[j].physical_device_ordinal ==
          aggregate->physical_device_ordinal) {
        device = &context->devices[j];
        break;
      }
    }
    double ns_per_tick = 0.0;
    double tick_frequency_hz = 0.0;
    const bool has_clock_fit = iree_profile_dispatch_device_try_fit_clock(
        device, &ns_per_tick, &tick_frequency_hz);

    char numeric_buffer[128];
    iree_string_view_t key = iree_string_view_empty();
    status = iree_profile_dispatch_resolve_key(
        context, aggregate->physical_device_ordinal, aggregate->executable_id,
        aggregate->export_ordinal, numeric_buffer, sizeof(numeric_buffer),
        &key);
    if (iree_status_is_ok(status)) {
      const double variance_ticks =
          aggregate->valid_count > 1
              ? aggregate->m2_ticks / (double)(aggregate->valid_count - 1)
              : 0.0;
      const double stddev_ticks = iree_profile_sqrt_f64(variance_ticks);
      fprintf(file, "  %.*s\n", (int)key.size, key.data);
      fprintf(file,
              "    device=%u executable=%" PRIu64 " export=%u count=%" PRIu64
              " valid=%" PRIu64 " invalid=%" PRIu64 "\n",
              aggregate->physical_device_ordinal, aggregate->executable_id,
              aggregate->export_ordinal, aggregate->dispatch_count,
              aggregate->valid_count, aggregate->invalid_count);
      if (aggregate->valid_count != 0) {
        fprintf(file,
                "    ticks: min=%" PRIu64 " avg=%.3f stddev=%.3f max=%" PRIu64
                " total=%.3f\n",
                aggregate->minimum_ticks, aggregate->mean_ticks, stddev_ticks,
                aggregate->maximum_ticks, aggregate->total_ticks);
        if (has_clock_fit) {
          fprintf(file,
                  "    time_ns: min=%.3f avg=%.3f stddev=%.3f max=%.3f"
                  " total=%.3f\n",
                  (double)aggregate->minimum_ticks * ns_per_tick,
                  aggregate->mean_ticks * ns_per_tick,
                  stddev_ticks * ns_per_tick,
                  (double)aggregate->maximum_ticks * ns_per_tick,
                  aggregate->total_ticks * ns_per_tick);
        } else {
          fprintf(file, "    time_ns: unavailable\n");
        }
      }
      fprintf(
          file,
          "    last_geometry: workgroup_count=%ux%ux%u"
          " workgroup_size=%ux%ux%u\n",
          aggregate->last_workgroup_count[0],
          aggregate->last_workgroup_count[1],
          aggregate->last_workgroup_count[2], aggregate->last_workgroup_size[0],
          aggregate->last_workgroup_size[1], aggregate->last_workgroup_size[2]);
    }
  }
  return status;
}

static void iree_profile_dispatch_print_jsonl_summary(
    const iree_profile_dispatch_context_t* context, iree_string_view_t filter,
    bool emit_events, FILE* file) {
  fprintf(file, "{\"type\":\"dispatch_summary\",\"mode\":\"%s\",\"filter\":",
          emit_events ? "events" : "aggregate");
  iree_profile_fprint_json_string(file, filter);
  fprintf(file,
          ",\"total_dispatches\":%" PRIu64 ",\"matched_dispatches\":%" PRIu64
          ",\"valid_dispatches\":%" PRIu64 ",\"invalid_dispatches\":%" PRIu64
          ",\"aggregate_groups\":%" PRIhsz "}\n",
          context->total_dispatch_count, context->matched_dispatch_count,
          context->valid_dispatch_count, context->invalid_dispatch_count,
          context->aggregate_count);
}

static iree_status_t iree_profile_dispatch_print_jsonl_aggregates(
    const iree_profile_dispatch_context_t* context, FILE* file) {
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < context->aggregate_count && iree_status_is_ok(status); ++i) {
    const iree_profile_dispatch_aggregate_t* aggregate =
        &context->aggregates[i];
    const iree_profile_dispatch_device_t* device = NULL;
    for (iree_host_size_t j = 0; j < context->device_count; ++j) {
      if (context->devices[j].physical_device_ordinal ==
          aggregate->physical_device_ordinal) {
        device = &context->devices[j];
        break;
      }
    }
    double ns_per_tick = 0.0;
    double tick_frequency_hz = 0.0;
    const bool has_clock_fit = iree_profile_dispatch_device_try_fit_clock(
        device, &ns_per_tick, &tick_frequency_hz);
    (void)tick_frequency_hz;

    char numeric_buffer[128];
    iree_string_view_t key = iree_string_view_empty();
    status = iree_profile_dispatch_resolve_key(
        context, aggregate->physical_device_ordinal, aggregate->executable_id,
        aggregate->export_ordinal, numeric_buffer, sizeof(numeric_buffer),
        &key);
    if (iree_status_is_ok(status)) {
      const double variance_ticks =
          aggregate->valid_count > 1
              ? aggregate->m2_ticks / (double)(aggregate->valid_count - 1)
              : 0.0;
      const double stddev_ticks = iree_profile_sqrt_f64(variance_ticks);
      fprintf(file,
              "{\"type\":\"dispatch_group\",\"physical_device_ordinal\":%u"
              ",\"executable_id\":%" PRIu64 ",\"export_ordinal\":%u,\"key\":",
              aggregate->physical_device_ordinal, aggregate->executable_id,
              aggregate->export_ordinal);
      iree_profile_fprint_json_string(file, key);
      fprintf(file,
              ",\"count\":%" PRIu64 ",\"valid\":%" PRIu64
              ",\"invalid\":%" PRIu64 ",\"min_ticks\":%" PRIu64
              ",\"avg_ticks\":%.3f,\"stddev_ticks\":%.3f"
              ",\"max_ticks\":%" PRIu64 ",\"total_ticks\":%.3f",
              aggregate->dispatch_count, aggregate->valid_count,
              aggregate->invalid_count,
              aggregate->valid_count ? aggregate->minimum_ticks : 0,
              aggregate->valid_count ? aggregate->mean_ticks : 0.0,
              stddev_ticks,
              aggregate->valid_count ? aggregate->maximum_ticks : 0,
              aggregate->total_ticks);
      fprintf(file, ",\"clock_fit_available\":%s",
              has_clock_fit ? "true" : "false");
      fprintf(file,
              ",\"min_ns\":%.3f,\"avg_ns\":%.3f,\"stddev_ns\":%.3f"
              ",\"max_ns\":%.3f,\"total_ns\":%.3f",
              has_clock_fit && aggregate->valid_count
                  ? (double)aggregate->minimum_ticks * ns_per_tick
                  : 0.0,
              has_clock_fit && aggregate->valid_count
                  ? aggregate->mean_ticks * ns_per_tick
                  : 0.0,
              has_clock_fit ? stddev_ticks * ns_per_tick : 0.0,
              has_clock_fit && aggregate->valid_count
                  ? (double)aggregate->maximum_ticks * ns_per_tick
                  : 0.0,
              has_clock_fit ? aggregate->total_ticks * ns_per_tick : 0.0);
      fprintf(
          file,
          ",\"last_workgroup_count\":[%u,%u,%u]"
          ",\"last_workgroup_size\":[%u,%u,%u]}\n",
          aggregate->last_workgroup_count[0],
          aggregate->last_workgroup_count[1],
          aggregate->last_workgroup_count[2], aggregate->last_workgroup_size[0],
          aggregate->last_workgroup_size[1], aggregate->last_workgroup_size[2]);
    }
  }
  return status;
}

static iree_status_t iree_profile_executable_print_text(
    const iree_profile_dispatch_context_t* context, iree_string_view_t filter,
    int64_t id_filter, FILE* file) {
  fprintf(file, "IREE HAL profile executable summary\n");
  fprintf(file, "filter: %.*s\n", (int)filter.size, filter.data);
  fprintf(file,
          "executables=%" PRIhsz " exports=%" PRIhsz
          " matched_dispatches=%" PRIu64 " valid=%" PRIu64 " invalid=%" PRIu64
          "\n",
          context->executable_count, context->export_count,
          context->matched_dispatch_count, context->valid_dispatch_count,
          context->invalid_dispatch_count);

  for (iree_host_size_t i = 0; i < context->executable_count; ++i) {
    const iree_hal_profile_executable_record_t* executable =
        &context->executables[i].record;
    if (id_filter >= 0 && executable->executable_id != (uint64_t)id_filter) {
      continue;
    }
    const bool has_code_object_hash = iree_all_bits_set(
        executable->flags, IREE_HAL_PROFILE_EXECUTABLE_FLAG_CODE_OBJECT_HASH);
    fprintf(file, "executable %" PRIu64 ": exports=%u flags=%u ",
            executable->executable_id, executable->export_count,
            executable->flags);
    fprintf(file, "code_object_hash=");
    if (has_code_object_hash) {
      iree_profile_fprint_hash_hex(file, executable->code_object_hash);
    } else {
      fprintf(file, "unavailable");
    }
    fputc('\n', file);
    for (iree_host_size_t j = 0; j < context->export_count; ++j) {
      const iree_profile_dispatch_export_t* export_info = &context->exports[j];
      if (export_info->executable_id != executable->executable_id) continue;

      char numeric_buffer[128];
      iree_string_view_t key = iree_profile_dispatch_format_export_key(
          export_info, UINT32_MAX, numeric_buffer, sizeof(numeric_buffer));
      if (!iree_profile_dispatch_key_matches(key, filter)) continue;

      const bool has_pipeline_hash = iree_all_bits_set(
          export_info->flags,
          IREE_HAL_PROFILE_EXECUTABLE_EXPORT_FLAG_PIPELINE_HASH);
      fprintf(file,
              "  export %u: %.*s flags=%u constants=%u bindings=%u "
              "parameters=%u workgroup_size=%ux%ux%u pipeline_hash=",
              export_info->export_ordinal, (int)key.size, key.data,
              export_info->flags, export_info->constant_count,
              export_info->binding_count, export_info->parameter_count,
              export_info->workgroup_size[0], export_info->workgroup_size[1],
              export_info->workgroup_size[2]);
      if (has_pipeline_hash) {
        iree_profile_fprint_hash_hex(file, export_info->pipeline_hash);
      } else {
        fprintf(file, "unavailable");
      }
      fputc('\n', file);
      bool has_aggregate = false;
      for (iree_host_size_t k = 0; k < context->aggregate_count; ++k) {
        const iree_profile_dispatch_aggregate_t* aggregate =
            &context->aggregates[k];
        if (aggregate->executable_id != export_info->executable_id ||
            aggregate->export_ordinal != export_info->export_ordinal) {
          continue;
        }
        has_aggregate = true;
        const iree_profile_dispatch_device_t* device =
            iree_profile_dispatch_find_device(
                context, aggregate->physical_device_ordinal);
        double ns_per_tick = 0.0;
        double tick_frequency_hz = 0.0;
        const bool has_clock_fit = iree_profile_dispatch_device_try_fit_clock(
            device, &ns_per_tick, &tick_frequency_hz);
        const double average_ticks =
            aggregate->valid_count
                ? aggregate->total_ticks / (double)aggregate->valid_count
                : 0.0;
        fprintf(file,
                "    device=%u dispatches=%" PRIu64 " valid=%" PRIu64
                " invalid=%" PRIu64 " ticks[min/avg/max/total]=%" PRIu64
                "/%.3f/%" PRIu64 "/%.3f",
                aggregate->physical_device_ordinal, aggregate->dispatch_count,
                aggregate->valid_count, aggregate->invalid_count,
                aggregate->valid_count ? aggregate->minimum_ticks : 0,
                average_ticks,
                aggregate->valid_count ? aggregate->maximum_ticks : 0,
                aggregate->total_ticks);
        if (has_clock_fit) {
          fprintf(file, " ns[min/avg/max/total]=%.3f/%.3f/%.3f/%.3f",
                  aggregate->valid_count
                      ? (double)aggregate->minimum_ticks * ns_per_tick
                      : 0.0,
                  average_ticks * ns_per_tick,
                  aggregate->valid_count
                      ? (double)aggregate->maximum_ticks * ns_per_tick
                      : 0.0,
                  aggregate->total_ticks * ns_per_tick);
        }
        fputc('\n', file);
      }
      if (!has_aggregate) {
        fprintf(file, "    dispatches=0\n");
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t iree_profile_executable_print_jsonl(
    const iree_profile_dispatch_context_t* context, iree_string_view_t filter,
    int64_t id_filter, FILE* file) {
  fprintf(file, "{\"type\":\"executable_summary\",\"filter\":");
  iree_profile_fprint_json_string(file, filter);
  fprintf(file,
          ",\"executables\":%" PRIhsz ",\"exports\":%" PRIhsz
          ",\"matched_dispatches\":%" PRIu64 ",\"valid_dispatches\":%" PRIu64
          ",\"invalid_dispatches\":%" PRIu64 "}\n",
          context->executable_count, context->export_count,
          context->matched_dispatch_count, context->valid_dispatch_count,
          context->invalid_dispatch_count);

  for (iree_host_size_t i = 0; i < context->executable_count; ++i) {
    const iree_hal_profile_executable_record_t* executable =
        &context->executables[i].record;
    if (id_filter >= 0 && executable->executable_id != (uint64_t)id_filter) {
      continue;
    }
    const bool has_code_object_hash = iree_all_bits_set(
        executable->flags, IREE_HAL_PROFILE_EXECUTABLE_FLAG_CODE_OBJECT_HASH);
    fprintf(file,
            "{\"type\":\"executable\",\"executable_id\":%" PRIu64
            ",\"flags\":%u,\"export_count\":%u"
            ",\"code_object_hash_present\":%s,\"code_object_hash\":",
            executable->executable_id, executable->flags,
            executable->export_count, has_code_object_hash ? "true" : "false");
    if (has_code_object_hash) {
      fputc('"', file);
      iree_profile_fprint_hash_hex(file, executable->code_object_hash);
      fputc('"', file);
    } else {
      fprintf(file, "null");
    }
    fputs("}\n", file);
    for (iree_host_size_t j = 0; j < context->export_count; ++j) {
      const iree_profile_dispatch_export_t* export_info = &context->exports[j];
      if (export_info->executable_id != executable->executable_id) continue;

      char numeric_buffer[128];
      iree_string_view_t key = iree_profile_dispatch_format_export_key(
          export_info, UINT32_MAX, numeric_buffer, sizeof(numeric_buffer));
      if (!iree_profile_dispatch_key_matches(key, filter)) continue;

      const bool has_pipeline_hash = iree_all_bits_set(
          export_info->flags,
          IREE_HAL_PROFILE_EXECUTABLE_EXPORT_FLAG_PIPELINE_HASH);
      fprintf(file,
              "{\"type\":\"executable_export\",\"executable_id\":%" PRIu64
              ",\"export_ordinal\":%u,\"flags\":%u,\"key\":",
              export_info->executable_id, export_info->export_ordinal,
              export_info->flags);
      iree_profile_fprint_json_string(file, key);
      fprintf(file,
              ",\"constant_count\":%u,\"binding_count\":%u"
              ",\"parameter_count\":%u,\"workgroup_size\":[%u,%u,%u]"
              ",\"pipeline_hash_present\":%s,\"pipeline_hash\":",
              export_info->constant_count, export_info->binding_count,
              export_info->parameter_count, export_info->workgroup_size[0],
              export_info->workgroup_size[1], export_info->workgroup_size[2],
              has_pipeline_hash ? "true" : "false");
      if (has_pipeline_hash) {
        fputc('"', file);
        iree_profile_fprint_hash_hex(file, export_info->pipeline_hash);
        fputc('"', file);
      } else {
        fprintf(file, "null");
      }
      fputs("}\n", file);
      for (iree_host_size_t k = 0; k < context->aggregate_count; ++k) {
        const iree_profile_dispatch_aggregate_t* aggregate =
            &context->aggregates[k];
        if (aggregate->executable_id != export_info->executable_id ||
            aggregate->export_ordinal != export_info->export_ordinal) {
          continue;
        }
        const iree_profile_dispatch_device_t* device =
            iree_profile_dispatch_find_device(
                context, aggregate->physical_device_ordinal);
        double ns_per_tick = 0.0;
        double tick_frequency_hz = 0.0;
        const bool has_clock_fit = iree_profile_dispatch_device_try_fit_clock(
            device, &ns_per_tick, &tick_frequency_hz);
        const double average_ticks =
            aggregate->valid_count
                ? aggregate->total_ticks / (double)aggregate->valid_count
                : 0.0;
        fprintf(file,
                "{\"type\":\"executable_export_dispatch_group\""
                ",\"physical_device_ordinal\":%u"
                ",\"executable_id\":%" PRIu64
                ",\"export_ordinal\":%u"
                ",\"key\":",
                aggregate->physical_device_ordinal, aggregate->executable_id,
                aggregate->export_ordinal);
        iree_profile_fprint_json_string(file, key);
        fprintf(file,
                ",\"dispatches\":%" PRIu64 ",\"valid\":%" PRIu64
                ",\"invalid\":%" PRIu64 ",\"min_ticks\":%" PRIu64
                ",\"avg_ticks\":%.3f,\"max_ticks\":%" PRIu64
                ",\"total_ticks\":%.3f,\"clock_fit_available\":%s"
                ",\"min_ns\":%.3f,\"avg_ns\":%.3f,\"max_ns\":%.3f"
                ",\"total_ns\":%.3f}\n",
                aggregate->dispatch_count, aggregate->valid_count,
                aggregate->invalid_count,
                aggregate->valid_count ? aggregate->minimum_ticks : 0,
                average_ticks,
                aggregate->valid_count ? aggregate->maximum_ticks : 0,
                aggregate->total_ticks, has_clock_fit ? "true" : "false",
                has_clock_fit && aggregate->valid_count
                    ? (double)aggregate->minimum_ticks * ns_per_tick
                    : 0.0,
                has_clock_fit ? average_ticks * ns_per_tick : 0.0,
                has_clock_fit && aggregate->valid_count
                    ? (double)aggregate->maximum_ticks * ns_per_tick
                    : 0.0,
                has_clock_fit ? aggregate->total_ticks * ns_per_tick : 0.0);
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t iree_profile_command_print_text(
    const iree_profile_dispatch_context_t* context, iree_string_view_t filter,
    int64_t id_filter, FILE* file) {
  iree_host_size_t matched_command_operation_count = 0;
  IREE_RETURN_IF_ERROR(iree_profile_command_count_matching_operations(
      context, filter, id_filter, &matched_command_operation_count));

  fprintf(file, "IREE HAL profile command-buffer summary\n");
  fprintf(file, "filter: %.*s\n", (int)filter.size, filter.data);
  fprintf(file,
          "command_buffers=%" PRIhsz " executions=%" PRIhsz
          " command_operations=%" PRIhsz " matched_command_operations=%" PRIhsz
          " matched_dispatches=%" PRIu64 " valid=%" PRIu64 " invalid=%" PRIu64
          "\n",
          context->command_buffer_count, context->command_aggregate_count,
          context->command_operation_count, matched_command_operation_count,
          context->matched_dispatch_count, context->valid_dispatch_count,
          context->invalid_dispatch_count);

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < context->command_buffer_count; ++i) {
    const iree_hal_profile_command_buffer_record_t* command_buffer =
        &context->command_buffers[i].record;
    if (id_filter >= 0 &&
        command_buffer->command_buffer_id != (uint64_t)id_filter) {
      continue;
    }
    fprintf(file,
            "command_buffer %" PRIu64 ": device=%u mode=%" PRIu64
            " categories=%" PRIu64 " queue_affinity=%" PRIu64 "\n",
            command_buffer->command_buffer_id,
            command_buffer->physical_device_ordinal, command_buffer->mode,
            command_buffer->command_categories, command_buffer->queue_affinity);
    for (iree_host_size_t j = 0;
         j < context->command_operation_count && iree_status_is_ok(status);
         ++j) {
      const iree_hal_profile_command_operation_record_t* operation =
          &context->command_operations[j].record;
      if (operation->command_buffer_id != command_buffer->command_buffer_id) {
        continue;
      }
      status = iree_profile_command_print_operation_text(context, operation,
                                                         filter, file);
    }
    for (iree_host_size_t j = 0;
         j < context->command_aggregate_count && iree_status_is_ok(status);
         ++j) {
      const iree_profile_dispatch_command_aggregate_t* aggregate =
          &context->command_aggregates[j];
      if (aggregate->command_buffer_id != command_buffer->command_buffer_id) {
        continue;
      }
      const iree_profile_dispatch_device_t* device =
          iree_profile_dispatch_find_device(context,
                                            aggregate->physical_device_ordinal);
      double ns_per_tick = 0.0;
      double tick_frequency_hz = 0.0;
      const bool has_clock_fit = iree_profile_dispatch_device_try_fit_clock(
          device, &ns_per_tick, &tick_frequency_hz);
      const double span_ticks = iree_profile_dispatch_span_ticks(
          aggregate->earliest_start_tick, aggregate->latest_end_tick);
      fprintf(file,
              "  submission=%" PRIu64 " queue=%u stream=%" PRIu64
              " dispatches=%" PRIu64 " valid=%" PRIu64 " invalid=%" PRIu64
              " span_ticks=%.3f total_dispatch_ticks=%.3f",
              aggregate->submission_id, aggregate->queue_ordinal,
              aggregate->stream_id, aggregate->dispatch_count,
              aggregate->valid_count, aggregate->invalid_count, span_ticks,
              aggregate->total_ticks);
      if (has_clock_fit) {
        fprintf(file, " span_ns=%.3f total_dispatch_ns=%.3f",
                span_ticks * ns_per_tick, aggregate->total_ticks * ns_per_tick);
      }
      fputc('\n', file);
    }
  }
  return status;
}

static iree_status_t iree_profile_command_print_jsonl(
    const iree_profile_dispatch_context_t* context, iree_string_view_t filter,
    int64_t id_filter, FILE* file) {
  iree_host_size_t matched_command_operation_count = 0;
  IREE_RETURN_IF_ERROR(iree_profile_command_count_matching_operations(
      context, filter, id_filter, &matched_command_operation_count));

  fprintf(file, "{\"type\":\"command_summary\",\"filter\":");
  iree_profile_fprint_json_string(file, filter);
  fprintf(file,
          ",\"command_buffers\":%" PRIhsz ",\"executions\":%" PRIhsz
          ",\"command_operations\":%" PRIhsz
          ",\"matched_command_operations\":%" PRIhsz
          ",\"matched_dispatches\":%" PRIu64 ",\"valid_dispatches\":%" PRIu64
          ",\"invalid_dispatches\":%" PRIu64 "}\n",
          context->command_buffer_count, context->command_aggregate_count,
          context->command_operation_count, matched_command_operation_count,
          context->matched_dispatch_count, context->valid_dispatch_count,
          context->invalid_dispatch_count);

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < context->command_buffer_count; ++i) {
    const iree_hal_profile_command_buffer_record_t* command_buffer =
        &context->command_buffers[i].record;
    if (id_filter >= 0 &&
        command_buffer->command_buffer_id != (uint64_t)id_filter) {
      continue;
    }
    fprintf(file,
            "{\"type\":\"command_buffer\",\"command_buffer_id\":%" PRIu64
            ",\"physical_device_ordinal\":%u,\"mode\":%" PRIu64
            ",\"command_categories\":%" PRIu64 ",\"queue_affinity\":%" PRIu64
            "}\n",
            command_buffer->command_buffer_id,
            command_buffer->physical_device_ordinal, command_buffer->mode,
            command_buffer->command_categories, command_buffer->queue_affinity);
  }
  for (iree_host_size_t i = 0;
       i < context->command_operation_count && iree_status_is_ok(status); ++i) {
    const iree_hal_profile_command_operation_record_t* operation =
        &context->command_operations[i].record;
    if (id_filter >= 0 && operation->command_buffer_id != (uint64_t)id_filter) {
      continue;
    }
    status = iree_profile_command_print_operation_jsonl(context, operation,
                                                        filter, file);
  }
  for (iree_host_size_t i = 0;
       i < context->command_aggregate_count && iree_status_is_ok(status); ++i) {
    const iree_profile_dispatch_command_aggregate_t* aggregate =
        &context->command_aggregates[i];
    if (id_filter >= 0 && aggregate->command_buffer_id != (uint64_t)id_filter) {
      continue;
    }
    const iree_profile_dispatch_device_t* device =
        iree_profile_dispatch_find_device(context,
                                          aggregate->physical_device_ordinal);
    double ns_per_tick = 0.0;
    double tick_frequency_hz = 0.0;
    const bool has_clock_fit = iree_profile_dispatch_device_try_fit_clock(
        device, &ns_per_tick, &tick_frequency_hz);
    const double span_ticks = iree_profile_dispatch_span_ticks(
        aggregate->earliest_start_tick, aggregate->latest_end_tick);
    fprintf(file,
            "{\"type\":\"command_execution\",\"command_buffer_id\":%" PRIu64
            ",\"submission_id\":%" PRIu64
            ",\"physical_device_ordinal\":%u"
            ",\"queue_ordinal\":%u,\"stream_id\":%" PRIu64
            ",\"dispatches\":%" PRIu64 ",\"valid\":%" PRIu64
            ",\"invalid\":%" PRIu64
            ",\"span_ticks\":%.3f"
            ",\"total_dispatch_ticks\":%.3f,\"clock_fit_available\":%s"
            ",\"span_ns\":%.3f,\"total_dispatch_ns\":%.3f}\n",
            aggregate->command_buffer_id, aggregate->submission_id,
            aggregate->physical_device_ordinal, aggregate->queue_ordinal,
            aggregate->stream_id, aggregate->dispatch_count,
            aggregate->valid_count, aggregate->invalid_count, span_ticks,
            aggregate->total_ticks, has_clock_fit ? "true" : "false",
            has_clock_fit ? span_ticks * ns_per_tick : 0.0,
            has_clock_fit ? aggregate->total_ticks * ns_per_tick : 0.0);
  }
  return status;
}

static iree_status_t iree_profile_queue_print_text(
    const iree_profile_dispatch_context_t* context, iree_string_view_t filter,
    int64_t id_filter, FILE* file) {
  fprintf(file, "IREE HAL profile queue summary\n");
  fprintf(file, "filter: %.*s\n", (int)filter.size, filter.data);
  fprintf(file,
          "queues=%" PRIhsz " queue_events=%" PRIhsz " submissions=%" PRIhsz
          " matched_dispatches=%" PRIu64 " valid=%" PRIu64 " invalid=%" PRIu64
          "\n",
          context->queue_count, context->queue_event_count,
          context->queue_aggregate_count, context->matched_dispatch_count,
          context->valid_dispatch_count, context->invalid_dispatch_count);

  for (iree_host_size_t i = 0; i < context->queue_count; ++i) {
    const iree_hal_profile_queue_record_t* queue = &context->queues[i].record;
    fprintf(file, "queue device=%u ordinal=%u stream=%" PRIu64 "\n",
            queue->physical_device_ordinal, queue->queue_ordinal,
            queue->stream_id);
    for (iree_host_size_t j = 0; j < context->queue_aggregate_count; ++j) {
      const iree_profile_dispatch_queue_aggregate_t* aggregate =
          &context->queue_aggregates[j];
      if (aggregate->physical_device_ordinal !=
              queue->physical_device_ordinal ||
          aggregate->queue_ordinal != queue->queue_ordinal ||
          aggregate->stream_id != queue->stream_id) {
        continue;
      }
      if (id_filter >= 0 && aggregate->submission_id != (uint64_t)id_filter) {
        continue;
      }
      const iree_profile_dispatch_device_t* device =
          iree_profile_dispatch_find_device(context,
                                            aggregate->physical_device_ordinal);
      double ns_per_tick = 0.0;
      double tick_frequency_hz = 0.0;
      const bool has_clock_fit = iree_profile_dispatch_device_try_fit_clock(
          device, &ns_per_tick, &tick_frequency_hz);
      const double span_ticks = iree_profile_dispatch_span_ticks(
          aggregate->earliest_start_tick, aggregate->latest_end_tick);
      fprintf(file,
              "  submission=%" PRIu64 " dispatches=%" PRIu64 " valid=%" PRIu64
              " invalid=%" PRIu64 " span_ticks=%.3f total_dispatch_ticks=%.3f",
              aggregate->submission_id, aggregate->dispatch_count,
              aggregate->valid_count, aggregate->invalid_count, span_ticks,
              aggregate->total_ticks);
      if (has_clock_fit) {
        fprintf(file, " span_ns=%.3f total_dispatch_ns=%.3f",
                span_ticks * ns_per_tick, aggregate->total_ticks * ns_per_tick);
      }
      fputc('\n', file);
    }
    for (iree_host_size_t j = 0; j < context->queue_event_count; ++j) {
      const iree_hal_profile_queue_event_t* event =
          &context->queue_events[j].record;
      if (event->physical_device_ordinal != queue->physical_device_ordinal ||
          event->queue_ordinal != queue->queue_ordinal ||
          event->stream_id != queue->stream_id) {
        continue;
      }
      if (id_filter >= 0 && event->submission_id != (uint64_t)id_filter) {
        continue;
      }
      fprintf(
          file,
          "  event=%" PRIu64 " type=%s submission=%" PRIu64
          " strategy=%s waits=%u signals=%u barriers=%u ops=%u bytes=%" PRIu64
          "\n",
          event->event_id, iree_profile_queue_event_type_name(event->type),
          event->submission_id,
          iree_profile_queue_dependency_strategy_name(
              event->dependency_strategy),
          event->wait_count, event->signal_count, event->barrier_count,
          event->operation_count, event->payload_length);
    }
  }
  return iree_ok_status();
}

static iree_status_t iree_profile_queue_print_jsonl(
    const iree_profile_dispatch_context_t* context, iree_string_view_t filter,
    int64_t id_filter, FILE* file) {
  fprintf(file, "{\"type\":\"queue_summary\",\"filter\":");
  iree_profile_fprint_json_string(file, filter);
  fprintf(file,
          ",\"queues\":%" PRIhsz ",\"queue_events\":%" PRIhsz
          ",\"submissions\":%" PRIhsz ",\"matched_dispatches\":%" PRIu64
          ",\"valid_dispatches\":%" PRIu64 ",\"invalid_dispatches\":%" PRIu64
          "}\n",
          context->queue_count, context->queue_event_count,
          context->queue_aggregate_count, context->matched_dispatch_count,
          context->valid_dispatch_count, context->invalid_dispatch_count);

  for (iree_host_size_t i = 0; i < context->queue_count; ++i) {
    const iree_hal_profile_queue_record_t* queue = &context->queues[i].record;
    fprintf(file,
            "{\"type\":\"queue\",\"physical_device_ordinal\":%u"
            ",\"queue_ordinal\":%u,\"stream_id\":%" PRIu64 "}\n",
            queue->physical_device_ordinal, queue->queue_ordinal,
            queue->stream_id);
  }
  for (iree_host_size_t i = 0; i < context->queue_aggregate_count; ++i) {
    const iree_profile_dispatch_queue_aggregate_t* aggregate =
        &context->queue_aggregates[i];
    if (id_filter >= 0 && aggregate->submission_id != (uint64_t)id_filter) {
      continue;
    }
    const iree_profile_dispatch_device_t* device =
        iree_profile_dispatch_find_device(context,
                                          aggregate->physical_device_ordinal);
    double ns_per_tick = 0.0;
    double tick_frequency_hz = 0.0;
    const bool has_clock_fit = iree_profile_dispatch_device_try_fit_clock(
        device, &ns_per_tick, &tick_frequency_hz);
    const double span_ticks = iree_profile_dispatch_span_ticks(
        aggregate->earliest_start_tick, aggregate->latest_end_tick);
    fprintf(file,
            "{\"type\":\"queue_submission\",\"submission_id\":%" PRIu64
            ",\"physical_device_ordinal\":%u,\"queue_ordinal\":%u"
            ",\"stream_id\":%" PRIu64 ",\"dispatches\":%" PRIu64
            ",\"valid\":%" PRIu64 ",\"invalid\":%" PRIu64
            ",\"span_ticks\":%.3f,\"total_dispatch_ticks\":%.3f"
            ",\"clock_fit_available\":%s,\"span_ns\":%.3f"
            ",\"total_dispatch_ns\":%.3f}\n",
            aggregate->submission_id, aggregate->physical_device_ordinal,
            aggregate->queue_ordinal, aggregate->stream_id,
            aggregate->dispatch_count, aggregate->valid_count,
            aggregate->invalid_count, span_ticks, aggregate->total_ticks,
            has_clock_fit ? "true" : "false",
            has_clock_fit ? span_ticks * ns_per_tick : 0.0,
            has_clock_fit ? aggregate->total_ticks * ns_per_tick : 0.0);
  }
  for (iree_host_size_t i = 0; i < context->queue_event_count; ++i) {
    const iree_hal_profile_queue_event_t* event =
        &context->queue_events[i].record;
    if (id_filter >= 0 && event->submission_id != (uint64_t)id_filter) {
      continue;
    }
    fprintf(
        file,
        "{\"type\":\"queue_event\",\"event_id\":%" PRIu64
        ",\"op\":\"%s\",\"flags\":%u,\"dependency_strategy\":\"%s\""
        ",\"submission_id\":%" PRIu64 ",\"command_buffer_id\":%" PRIu64
        ",\"allocation_id\":%" PRIu64
        ",\"physical_device_ordinal\":%u,\"queue_ordinal\":%u"
        ",\"stream_id\":%" PRIu64 ",\"host_time_ns\":%" PRId64
        ",\"wait_count\":%u,\"signal_count\":%u,\"barrier_count\":%u"
        ",\"operation_count\":%u,\"payload_length\":%" PRIu64 "}\n",
        event->event_id, iree_profile_queue_event_type_name(event->type),
        event->flags,
        iree_profile_queue_dependency_strategy_name(event->dependency_strategy),
        event->submission_id, event->command_buffer_id, event->allocation_id,
        event->physical_device_ordinal, event->queue_ordinal, event->stream_id,
        event->host_time_ns, event->wait_count, event->signal_count,
        event->barrier_count, event->operation_count, event->payload_length);
  }
  return iree_ok_status();
}

static iree_status_t iree_profile_counter_resolve_aggregate_key(
    const iree_profile_dispatch_context_t* dispatch_context,
    const iree_profile_counter_aggregate_t* aggregate, char* numeric_buffer,
    iree_host_size_t numeric_buffer_capacity, iree_string_view_t* out_key) {
  *out_key = IREE_SV("unattributed");
  if (aggregate->executable_id == 0 ||
      aggregate->export_ordinal == UINT32_MAX) {
    return iree_ok_status();
  }
  return iree_profile_dispatch_resolve_key(
      dispatch_context, aggregate->physical_device_ordinal,
      aggregate->executable_id, aggregate->export_ordinal, numeric_buffer,
      numeric_buffer_capacity, out_key);
}

static iree_status_t iree_profile_counter_print_metadata_text(
    const iree_profile_counter_context_t* counter_context, FILE* file) {
  fprintf(file, "counter_sets:\n");
  for (iree_host_size_t i = 0; i < counter_context->counter_set_count; ++i) {
    const iree_profile_counter_set_t* counter_set =
        &counter_context->counter_sets[i];
    fprintf(file,
            "  counter_set %" PRIu64
            ": device=%u counters=%u sample_values=%u flags=%u name=%.*s\n",
            counter_set->record.counter_set_id,
            counter_set->record.physical_device_ordinal,
            counter_set->record.counter_count,
            counter_set->record.sample_value_count, counter_set->record.flags,
            (int)counter_set->name.size, counter_set->name.data);
  }

  fprintf(file, "counters:\n");
  for (iree_host_size_t i = 0; i < counter_context->counter_count; ++i) {
    const iree_profile_counter_t* counter = &counter_context->counters[i];
    if (!iree_profile_counter_find_counter_set(
            counter_context, counter->record.counter_set_id)) {
      return iree_make_status(IREE_STATUS_DATA_LOSS,
                              "counter references missing counter-set metadata "
                              "counter_set=%" PRIu64 " counter_ordinal=%u",
                              counter->record.counter_set_id,
                              counter->record.counter_ordinal);
    }
    fprintf(file,
            "  counter %" PRIu64
            "#%u: device=%u block=%.*s name=%.*s "
            "unit=%s values=[%u,%u) flags=%u description=%.*s\n",
            counter->record.counter_set_id, counter->record.counter_ordinal,
            counter->record.physical_device_ordinal,
            (int)counter->block_name.size, counter->block_name.data,
            (int)counter->name.size, counter->name.data,
            iree_profile_counter_unit_name(counter->record.unit),
            counter->record.sample_value_offset,
            counter->record.sample_value_offset +
                counter->record.sample_value_count,
            counter->record.flags, (int)counter->description.size,
            counter->description.data);
  }
  return iree_ok_status();
}

static void iree_profile_counter_print_metadata_jsonl(
    const iree_profile_counter_context_t* counter_context, FILE* file) {
  for (iree_host_size_t i = 0; i < counter_context->counter_set_count; ++i) {
    const iree_profile_counter_set_t* counter_set =
        &counter_context->counter_sets[i];
    fprintf(file,
            "{\"type\":\"counter_set\",\"counter_set_id\":%" PRIu64
            ",\"physical_device_ordinal\":%u,\"flags\":%u"
            ",\"counter_count\":%u,\"sample_value_count\":%u,\"name\":",
            counter_set->record.counter_set_id,
            counter_set->record.physical_device_ordinal,
            counter_set->record.flags, counter_set->record.counter_count,
            counter_set->record.sample_value_count);
    iree_profile_fprint_json_string(file, counter_set->name);
    fputs("}\n", file);
  }

  for (iree_host_size_t i = 0; i < counter_context->counter_count; ++i) {
    const iree_profile_counter_t* counter = &counter_context->counters[i];
    fprintf(file,
            "{\"type\":\"counter\",\"counter_set_id\":%" PRIu64
            ",\"counter_ordinal\":%u,\"physical_device_ordinal\":%u"
            ",\"flags\":%u,\"unit\":",
            counter->record.counter_set_id, counter->record.counter_ordinal,
            counter->record.physical_device_ordinal, counter->record.flags);
    iree_profile_fprint_json_string(
        file, iree_make_cstring_view(
                  iree_profile_counter_unit_name(counter->record.unit)));
    fprintf(file, ",\"unit_value\":%u,\"sample_value_offset\":%u",
            counter->record.unit, counter->record.sample_value_offset);
    fprintf(file, ",\"sample_value_count\":%u,\"block\":",
            counter->record.sample_value_count);
    iree_profile_fprint_json_string(file, counter->block_name);
    fprintf(file, ",\"name\":");
    iree_profile_fprint_json_string(file, counter->name);
    fprintf(file, ",\"description\":");
    iree_profile_fprint_json_string(file, counter->description);
    fputs("}\n", file);
  }
}

static iree_status_t iree_profile_counter_print_text(
    const iree_profile_counter_context_t* counter_context,
    const iree_profile_dispatch_context_t* dispatch_context,
    iree_string_view_t filter, int64_t id_filter, FILE* file) {
  fprintf(file, "IREE HAL profile counter summary\n");
  fprintf(file, "filter: %.*s\n", (int)filter.size, filter.data);
  if (id_filter >= 0) {
    fprintf(file, "id_filter: %" PRId64 "\n", id_filter);
  }
  fprintf(file,
          "samples: total=%" PRIu64 " matched=%" PRIu64
          " truncated_matched=%" PRIu64 " counter_sets=%" PRIhsz
          " counters=%" PRIhsz " groups=%" PRIhsz "\n",
          counter_context->total_sample_count,
          counter_context->matched_sample_count,
          counter_context->truncated_sample_count,
          counter_context->counter_set_count, counter_context->counter_count,
          counter_context->aggregate_count);

  IREE_RETURN_IF_ERROR(
      iree_profile_counter_print_metadata_text(counter_context, file));

  fprintf(file, "groups:\n");
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < counter_context->aggregate_count && iree_status_is_ok(status); ++i) {
    const iree_profile_counter_aggregate_t* aggregate =
        &counter_context->aggregates[i];
    const iree_profile_counter_set_t* counter_set =
        iree_profile_counter_find_counter_set(counter_context,
                                              aggregate->counter_set_id);
    const iree_profile_counter_t* counter = iree_profile_counter_find_counter(
        counter_context, aggregate->counter_set_id, aggregate->counter_ordinal);
    if (!counter_set || !counter) {
      status = iree_make_status(
          IREE_STATUS_DATA_LOSS,
          "counter aggregate references missing metadata counter_set=%" PRIu64
          " counter_ordinal=%u",
          aggregate->counter_set_id, aggregate->counter_ordinal);
      continue;
    }

    char numeric_buffer[128];
    iree_string_view_t key = iree_string_view_empty();
    status = iree_profile_counter_resolve_aggregate_key(
        dispatch_context, aggregate, numeric_buffer, sizeof(numeric_buffer),
        &key);
    if (iree_status_is_ok(status)) {
      const double variance =
          aggregate->sample_count > 1
              ? aggregate->m2_value / (double)(aggregate->sample_count - 1)
              : 0.0;
      const double stddev = iree_profile_sqrt_f64(variance);
      fprintf(file,
              "  %.*s / %.*s.%.*s\n"
              "    device=%u queue=%u stream=%" PRIu64
              " command_buffer=%" PRIu64 " executable=%" PRIu64
              " export=%u key=%.*s\n"
              "    samples=%" PRIu64 " raw_values=%" PRIu64
              " value[min/avg/stddev/max/total]=%.3f/%.3f/%.3f/%.3f/%.3f "
              "unit=%s first_sample=%" PRIu64 " last_sample=%" PRIu64 "\n",
              (int)counter_set->name.size, counter_set->name.data,
              (int)counter->block_name.size, counter->block_name.data,
              (int)counter->name.size, counter->name.data,
              aggregate->physical_device_ordinal, aggregate->queue_ordinal,
              aggregate->stream_id, aggregate->command_buffer_id,
              aggregate->executable_id, aggregate->export_ordinal,
              (int)key.size, key.data, aggregate->sample_count,
              aggregate->raw_value_count,
              aggregate->sample_count ? aggregate->minimum_value : 0.0,
              aggregate->sample_count ? aggregate->mean_value : 0.0, stddev,
              aggregate->sample_count ? aggregate->maximum_value : 0.0,
              aggregate->total_value,
              iree_profile_counter_unit_name(counter->record.unit),
              aggregate->first_sample_id, aggregate->last_sample_id);
    }
  }
  return status;
}

static iree_status_t iree_profile_counter_print_jsonl(
    const iree_profile_counter_context_t* counter_context,
    const iree_profile_dispatch_context_t* dispatch_context,
    iree_string_view_t filter, int64_t id_filter, bool emit_samples,
    FILE* file) {
  fprintf(file, "{\"type\":\"counter_summary\",\"filter\":");
  iree_profile_fprint_json_string(file, filter);
  fprintf(file,
          ",\"id_filter\":%" PRId64
          ",\"mode\":\"%s\",\"total_samples\":%" PRIu64
          ",\"matched_samples\":%" PRIu64
          ",\"truncated_matched_samples\":%" PRIu64 ",\"counter_sets\":%" PRIhsz
          ",\"counters\":%" PRIhsz ",\"aggregate_groups\":%" PRIhsz "}\n",
          id_filter, emit_samples ? "samples" : "aggregate",
          counter_context->total_sample_count,
          counter_context->matched_sample_count,
          counter_context->truncated_sample_count,
          counter_context->counter_set_count, counter_context->counter_count,
          counter_context->aggregate_count);

  iree_profile_counter_print_metadata_jsonl(counter_context, file);

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < counter_context->aggregate_count && iree_status_is_ok(status); ++i) {
    const iree_profile_counter_aggregate_t* aggregate =
        &counter_context->aggregates[i];
    const iree_profile_counter_set_t* counter_set =
        iree_profile_counter_find_counter_set(counter_context,
                                              aggregate->counter_set_id);
    const iree_profile_counter_t* counter = iree_profile_counter_find_counter(
        counter_context, aggregate->counter_set_id, aggregate->counter_ordinal);
    if (!counter_set || !counter) {
      status = iree_make_status(
          IREE_STATUS_DATA_LOSS,
          "counter aggregate references missing metadata counter_set=%" PRIu64
          " counter_ordinal=%u",
          aggregate->counter_set_id, aggregate->counter_ordinal);
      continue;
    }

    char numeric_buffer[128];
    iree_string_view_t key = iree_string_view_empty();
    status = iree_profile_counter_resolve_aggregate_key(
        dispatch_context, aggregate, numeric_buffer, sizeof(numeric_buffer),
        &key);
    if (iree_status_is_ok(status)) {
      const double variance =
          aggregate->sample_count > 1
              ? aggregate->m2_value / (double)(aggregate->sample_count - 1)
              : 0.0;
      const double stddev = iree_profile_sqrt_f64(variance);
      fprintf(file,
              "{\"type\":\"counter_group\",\"physical_device_ordinal\":%u"
              ",\"queue_ordinal\":%u,\"stream_id\":%" PRIu64
              ",\"counter_set_id\":%" PRIu64 ",\"counter_set\":",
              aggregate->physical_device_ordinal, aggregate->queue_ordinal,
              aggregate->stream_id, aggregate->counter_set_id);
      iree_profile_fprint_json_string(file, counter_set->name);
      fprintf(file, ",\"counter_ordinal\":%u,\"counter\":",
              aggregate->counter_ordinal);
      iree_profile_fprint_json_string(file, counter->name);
      fprintf(file, ",\"block\":");
      iree_profile_fprint_json_string(file, counter->block_name);
      fprintf(file, ",\"unit\":");
      iree_profile_fprint_json_string(
          file, iree_make_cstring_view(
                    iree_profile_counter_unit_name(counter->record.unit)));
      fprintf(file,
              ",\"command_buffer_id\":%" PRIu64 ",\"executable_id\":%" PRIu64
              ",\"export_ordinal\":%u"
              ",\"key\":",
              aggregate->command_buffer_id, aggregate->executable_id,
              aggregate->export_ordinal);
      iree_profile_fprint_json_string(file, key);
      fprintf(file,
              ",\"samples\":%" PRIu64 ",\"raw_values\":%" PRIu64
              ",\"min\":%.3f,\"avg\":%.3f,\"stddev\":%.3f"
              ",\"max\":%.3f,\"sum\":%.3f"
              ",\"first_sample_id\":%" PRIu64 ",\"last_sample_id\":%" PRIu64
              "}\n",
              aggregate->sample_count, aggregate->raw_value_count,
              aggregate->sample_count ? aggregate->minimum_value : 0.0,
              aggregate->sample_count ? aggregate->mean_value : 0.0, stddev,
              aggregate->sample_count ? aggregate->maximum_value : 0.0,
              aggregate->total_value, aggregate->first_sample_id,
              aggregate->last_sample_id);
    }
  }
  return status;
}

static iree_status_t iree_profile_counter_file(
    iree_string_view_t path, iree_string_view_t format,
    iree_string_view_t filter, int64_t id_filter, bool emit_samples, FILE* file,
    iree_allocator_t host_allocator) {
  bool is_text = iree_string_view_equal(format, IREE_SV("text"));
  bool is_jsonl = iree_string_view_equal(format, IREE_SV("jsonl"));
  if (!is_text && !is_jsonl) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported profile output format '%.*s'",
                            (int)format.size, format.data);
  }
  if (emit_samples && !is_jsonl) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "--counter_samples requires --format=jsonl");
  }

  iree_io_file_contents_t* file_contents = NULL;
  iree_status_t status = iree_io_file_contents_map(
      path, IREE_IO_FILE_ACCESS_READ, host_allocator, &file_contents);

  iree_hal_profile_file_header_t header;
  iree_host_size_t first_record_offset = 0;
  if (iree_status_is_ok(status)) {
    status = iree_hal_profile_file_parse_header(file_contents->const_buffer,
                                                &header, &first_record_offset);
  }

  iree_profile_dispatch_context_t dispatch_context;
  iree_profile_dispatch_context_initialize(host_allocator, &dispatch_context);
  iree_profile_counter_context_t counter_context;
  iree_profile_counter_context_initialize(host_allocator, &counter_context);

  iree_host_size_t record_offset = first_record_offset;
  while (iree_status_is_ok(status) &&
         record_offset < file_contents->const_buffer.data_length) {
    iree_hal_profile_file_record_t record;
    iree_host_size_t next_record_offset = 0;
    status = iree_hal_profile_file_parse_record(file_contents->const_buffer,
                                                record_offset, &record,
                                                &next_record_offset);
    if (iree_status_is_ok(status)) {
      status = iree_profile_dispatch_process_metadata_record(&dispatch_context,
                                                             &record);
    }
    if (iree_status_is_ok(status)) {
      status = iree_profile_counter_process_metadata_record(&counter_context,
                                                            &record);
    }
    if (iree_status_is_ok(status)) {
      record_offset = next_record_offset;
    }
  }

  record_offset = first_record_offset;
  while (iree_status_is_ok(status) &&
         record_offset < file_contents->const_buffer.data_length) {
    iree_hal_profile_file_record_t record;
    iree_host_size_t next_record_offset = 0;
    status = iree_hal_profile_file_parse_record(file_contents->const_buffer,
                                                record_offset, &record,
                                                &next_record_offset);
    if (iree_status_is_ok(status)) {
      status = iree_profile_counter_process_sample_records(
          &counter_context, &dispatch_context, &record, filter, id_filter,
          emit_samples, file);
      record_offset = next_record_offset;
    }
  }

  if (iree_status_is_ok(status)) {
    if (is_text) {
      status = iree_profile_counter_print_text(
          &counter_context, &dispatch_context, filter, id_filter, file);
    } else {
      status = iree_profile_counter_print_jsonl(&counter_context,
                                                &dispatch_context, filter,
                                                id_filter, emit_samples, file);
    }
  }

  iree_profile_counter_context_deinitialize(&counter_context);
  iree_profile_dispatch_context_deinitialize(&dispatch_context);
  iree_io_file_contents_free(file_contents);
  return status;
}

static const char* iree_profile_memory_event_type_name(
    iree_hal_profile_memory_event_type_t type) {
  switch (type) {
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_SLAB_ACQUIRE:
      return "slab_acquire";
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_SLAB_RELEASE:
      return "slab_release";
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_RESERVE:
      return "pool_reserve";
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_MATERIALIZE:
      return "pool_materialize";
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_RELEASE:
      return "pool_release";
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_WAIT:
      return "pool_wait";
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_ALLOCA:
      return "queue_alloca";
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_DEALLOCA:
      return "queue_dealloca";
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_BUFFER_ALLOCATE:
      return "buffer_allocate";
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_BUFFER_FREE:
      return "buffer_free";
    default:
      return "unknown";
  }
}

static const char* iree_profile_memory_lifecycle_kind_name(
    iree_profile_memory_lifecycle_kind_t kind) {
  switch (kind) {
    case IREE_PROFILE_MEMORY_LIFECYCLE_KIND_SLAB:
      return "slab";
    case IREE_PROFILE_MEMORY_LIFECYCLE_KIND_POOL_RESERVATION:
      return "pool_reservation";
    case IREE_PROFILE_MEMORY_LIFECYCLE_KIND_QUEUE_ALLOCATION:
      return "queue_allocation";
    case IREE_PROFILE_MEMORY_LIFECYCLE_KIND_BUFFER_ALLOCATION:
      return "buffer_allocation";
    default:
      return "unknown";
  }
}

static bool iree_profile_memory_event_allocation_kind(
    const iree_hal_profile_memory_event_t* event,
    iree_profile_memory_lifecycle_kind_t* out_kind) {
  switch (event->type) {
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_SLAB_ACQUIRE:
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_SLAB_RELEASE:
      *out_kind = IREE_PROFILE_MEMORY_LIFECYCLE_KIND_SLAB;
      return true;
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_RESERVE:
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_MATERIALIZE:
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_RELEASE:
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_WAIT:
      if (iree_all_bits_set(
              event->flags,
              IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_POOL_RESERVATION)) {
        *out_kind = IREE_PROFILE_MEMORY_LIFECYCLE_KIND_POOL_RESERVATION;
        return true;
      }
      return false;
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_ALLOCA:
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_DEALLOCA:
      *out_kind = IREE_PROFILE_MEMORY_LIFECYCLE_KIND_QUEUE_ALLOCATION;
      return true;
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_BUFFER_ALLOCATE:
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_BUFFER_FREE:
      *out_kind = IREE_PROFILE_MEMORY_LIFECYCLE_KIND_BUFFER_ALLOCATION;
      return true;
    default:
      return false;
  }
}

static bool iree_profile_memory_event_pool_kind(
    const iree_hal_profile_memory_event_t* event,
    iree_profile_memory_lifecycle_kind_t* out_kind) {
  switch (event->type) {
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_SLAB_ACQUIRE:
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_SLAB_RELEASE:
      *out_kind = IREE_PROFILE_MEMORY_LIFECYCLE_KIND_SLAB;
      return true;
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_RESERVE:
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_MATERIALIZE:
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_RELEASE:
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_WAIT:
      *out_kind = IREE_PROFILE_MEMORY_LIFECYCLE_KIND_POOL_RESERVATION;
      return true;
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_ALLOCA:
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_DEALLOCA:
      *out_kind = IREE_PROFILE_MEMORY_LIFECYCLE_KIND_QUEUE_ALLOCATION;
      return true;
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_BUFFER_ALLOCATE:
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_BUFFER_FREE:
      *out_kind = IREE_PROFILE_MEMORY_LIFECYCLE_KIND_BUFFER_ALLOCATION;
      return true;
    default:
      return false;
  }
}

static bool iree_profile_memory_event_increases_live_bytes(
    const iree_hal_profile_memory_event_t* event) {
  switch (event->type) {
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_SLAB_ACQUIRE:
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_BUFFER_ALLOCATE:
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_ALLOCA:
      return true;
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_RESERVE:
      return iree_all_bits_set(
          event->flags, IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_POOL_RESERVATION);
    default:
      return false;
  }
}

static bool iree_profile_memory_event_decreases_live_bytes(
    const iree_hal_profile_memory_event_t* event) {
  switch (event->type) {
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_SLAB_RELEASE:
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_RELEASE:
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_DEALLOCA:
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_BUFFER_FREE:
      return true;
    default:
      return false;
  }
}

static void iree_profile_memory_context_initialize(
    iree_allocator_t host_allocator,
    iree_profile_memory_context_t* out_context) {
  memset(out_context, 0, sizeof(*out_context));
  out_context->host_allocator = host_allocator;
}

static void iree_profile_memory_context_deinitialize(
    iree_profile_memory_context_t* context) {
  iree_allocator_free(context->host_allocator, context->devices);
  iree_allocator_free(context->host_allocator, context->pools);
  iree_allocator_free(context->host_allocator, context->allocations);
  memset(context, 0, sizeof(*context));
}

static iree_status_t iree_profile_memory_get_device(
    iree_profile_memory_context_t* context, uint32_t physical_device_ordinal,
    iree_profile_memory_device_t** out_device) {
  *out_device = NULL;

  for (iree_host_size_t i = 0; i < context->device_count; ++i) {
    if (context->devices[i].physical_device_ordinal ==
        physical_device_ordinal) {
      *out_device = &context->devices[i];
      return iree_ok_status();
    }
  }

  if (context->device_count + 1 > context->device_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        context->host_allocator,
        iree_max((iree_host_size_t)4, context->device_count + 1),
        sizeof(context->devices[0]), &context->device_capacity,
        (void**)&context->devices));
  }

  iree_profile_memory_device_t* device =
      &context->devices[context->device_count++];
  memset(device, 0, sizeof(*device));
  device->physical_device_ordinal = physical_device_ordinal;
  *out_device = device;
  return iree_ok_status();
}

static iree_status_t iree_profile_memory_get_pool(
    iree_profile_memory_context_t* context,
    iree_profile_memory_lifecycle_kind_t kind, uint32_t physical_device_ordinal,
    uint64_t pool_id, uint64_t memory_type,
    iree_profile_memory_pool_t** out_pool) {
  *out_pool = NULL;

  for (iree_host_size_t i = context->pool_count; i > 0; --i) {
    iree_profile_memory_pool_t* pool = &context->pools[i - 1];
    if (pool->kind == kind &&
        pool->physical_device_ordinal == physical_device_ordinal &&
        pool->pool_id == pool_id && pool->memory_type == memory_type) {
      *out_pool = pool;
      return iree_ok_status();
    }
  }

  if (context->pool_count + 1 > context->pool_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        context->host_allocator,
        iree_max((iree_host_size_t)8, context->pool_count + 1),
        sizeof(context->pools[0]), &context->pool_capacity,
        (void**)&context->pools));
  }

  iree_profile_memory_pool_t* pool = &context->pools[context->pool_count++];
  memset(pool, 0, sizeof(*pool));
  pool->kind = kind;
  pool->physical_device_ordinal = physical_device_ordinal;
  pool->pool_id = pool_id;
  pool->memory_type = memory_type;
  *out_pool = pool;
  return iree_ok_status();
}

static iree_status_t iree_profile_memory_get_allocation(
    iree_profile_memory_context_t* context,
    iree_profile_memory_lifecycle_kind_t kind, uint32_t physical_device_ordinal,
    uint64_t allocation_id, uint64_t pool_id,
    iree_profile_memory_allocation_t** out_allocation) {
  *out_allocation = NULL;

  for (iree_host_size_t i = context->allocation_count; i > 0; --i) {
    iree_profile_memory_allocation_t* allocation = &context->allocations[i - 1];
    if (allocation->kind == kind &&
        allocation->physical_device_ordinal == physical_device_ordinal &&
        allocation->allocation_id == allocation_id &&
        allocation->pool_id == pool_id) {
      *out_allocation = allocation;
      return iree_ok_status();
    }
  }

  if (context->allocation_count + 1 > context->allocation_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        context->host_allocator,
        iree_max((iree_host_size_t)16, context->allocation_count + 1),
        sizeof(context->allocations[0]), &context->allocation_capacity,
        (void**)&context->allocations));
  }

  iree_profile_memory_allocation_t* allocation =
      &context->allocations[context->allocation_count++];
  memset(allocation, 0, sizeof(*allocation));
  allocation->kind = kind;
  allocation->physical_device_ordinal = physical_device_ordinal;
  allocation->allocation_id = allocation_id;
  allocation->pool_id = pool_id;
  *out_allocation = allocation;
  return iree_ok_status();
}

static uint64_t iree_profile_memory_resolve_pool_id(
    const iree_profile_memory_context_t* context,
    const iree_hal_profile_memory_event_t* event,
    iree_profile_memory_lifecycle_kind_t kind) {
  if (event->pool_id != 0) return event->pool_id;
  if (kind != IREE_PROFILE_MEMORY_LIFECYCLE_KIND_QUEUE_ALLOCATION) {
    return event->pool_id;
  }

  for (iree_host_size_t i = context->allocation_count; i > 0; --i) {
    const iree_profile_memory_allocation_t* allocation =
        &context->allocations[i - 1];
    if (allocation->kind == kind &&
        allocation->physical_device_ordinal == event->physical_device_ordinal &&
        allocation->allocation_id == event->allocation_id) {
      return allocation->pool_id;
    }
  }
  return event->pool_id;
}

static bool iree_profile_memory_event_matches(
    const iree_hal_profile_memory_event_t* event, int64_t id_filter,
    iree_string_view_t filter) {
  if (id_filter >= 0 && event->event_id != (uint64_t)id_filter &&
      event->allocation_id != (uint64_t)id_filter) {
    return false;
  }
  iree_string_view_t type_name =
      iree_make_cstring_view(iree_profile_memory_event_type_name(event->type));
  return iree_profile_dispatch_key_matches(type_name, filter);
}

static void iree_profile_memory_record_event(
    iree_profile_memory_device_t* device,
    const iree_hal_profile_memory_event_t* event) {
  ++device->event_count;
  switch (event->type) {
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_SLAB_ACQUIRE:
      ++device->slab_acquire_count;
      ++device->current_slab_allocation_count;
      device->high_water_slab_allocation_count =
          iree_max(device->high_water_slab_allocation_count,
                   device->current_slab_allocation_count);
      device->total_slab_acquired_bytes += event->length;
      device->current_slab_bytes += event->length;
      device->high_water_slab_bytes =
          iree_max(device->high_water_slab_bytes, device->current_slab_bytes);
      break;
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_SLAB_RELEASE:
      ++device->slab_release_count;
      device->current_slab_allocation_count =
          device->current_slab_allocation_count == 0
              ? 0
              : device->current_slab_allocation_count - 1;
      device->total_slab_released_bytes += event->length;
      device->current_slab_bytes =
          event->length > device->current_slab_bytes
              ? 0
              : device->current_slab_bytes - event->length;
      break;
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_RESERVE:
      ++device->pool_reserve_count;
      if (iree_all_bits_set(
              event->flags,
              IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_POOL_RESERVATION)) {
        ++device->current_pool_reservation_count;
        device->high_water_pool_reservation_count =
            iree_max(device->high_water_pool_reservation_count,
                     device->current_pool_reservation_count);
        device->total_pool_reserved_bytes += event->length;
        device->current_pool_reserved_bytes += event->length;
        device->high_water_pool_reserved_bytes =
            iree_max(device->high_water_pool_reserved_bytes,
                     device->current_pool_reserved_bytes);
      }
      break;
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_MATERIALIZE:
      ++device->pool_materialize_count;
      break;
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_RELEASE:
      ++device->pool_release_count;
      device->current_pool_reservation_count =
          device->current_pool_reservation_count == 0
              ? 0
              : device->current_pool_reservation_count - 1;
      device->total_pool_released_bytes += event->length;
      device->current_pool_reserved_bytes =
          event->length > device->current_pool_reserved_bytes
              ? 0
              : device->current_pool_reserved_bytes - event->length;
      break;
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_WAIT:
      ++device->pool_wait_count;
      break;
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_BUFFER_ALLOCATE:
      ++device->buffer_allocate_count;
      ++device->current_buffer_allocation_count;
      device->high_water_buffer_allocation_count =
          iree_max(device->high_water_buffer_allocation_count,
                   device->current_buffer_allocation_count);
      device->total_buffer_allocate_bytes += event->length;
      device->current_buffer_bytes += event->length;
      device->high_water_buffer_bytes = iree_max(
          device->high_water_buffer_bytes, device->current_buffer_bytes);
      break;
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_BUFFER_FREE:
      ++device->buffer_free_count;
      device->current_buffer_allocation_count =
          device->current_buffer_allocation_count == 0
              ? 0
              : device->current_buffer_allocation_count - 1;
      device->total_buffer_free_bytes += event->length;
      device->current_buffer_bytes =
          event->length > device->current_buffer_bytes
              ? 0
              : device->current_buffer_bytes - event->length;
      break;
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_ALLOCA:
      ++device->queue_alloca_count;
      ++device->current_queue_allocation_count;
      device->high_water_queue_allocation_count =
          iree_max(device->high_water_queue_allocation_count,
                   device->current_queue_allocation_count);
      device->total_queue_alloca_bytes += event->length;
      device->current_queue_bytes += event->length;
      device->high_water_queue_bytes =
          iree_max(device->high_water_queue_bytes, device->current_queue_bytes);
      break;
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_DEALLOCA:
      ++device->queue_dealloca_count;
      device->current_queue_allocation_count =
          device->current_queue_allocation_count == 0
              ? 0
              : device->current_queue_allocation_count - 1;
      device->total_queue_dealloca_bytes += event->length;
      device->current_queue_bytes =
          event->length > device->current_queue_bytes
              ? 0
              : device->current_queue_bytes - event->length;
      break;
    default:
      break;
  }
}

static void iree_profile_memory_apply_live_increase(
    uint64_t length, uint64_t* current_count, uint64_t* high_water_count,
    uint64_t* current_bytes, uint64_t* high_water_bytes,
    uint64_t* total_allocate_bytes) {
  *current_count += 1;
  *high_water_count = iree_max(*high_water_count, *current_count);
  *current_bytes += length;
  *high_water_bytes = iree_max(*high_water_bytes, *current_bytes);
  *total_allocate_bytes += length;
}

static void iree_profile_memory_apply_live_decrease(
    uint64_t length, uint64_t* current_count, uint64_t* current_bytes,
    uint64_t* total_free_bytes) {
  *current_count = *current_count == 0 ? 0 : *current_count - 1;
  *current_bytes = length > *current_bytes ? 0 : *current_bytes - length;
  *total_free_bytes += length;
}

static iree_status_t iree_profile_memory_record_pool_event(
    iree_profile_memory_context_t* context,
    const iree_hal_profile_memory_event_t* event) {
  iree_profile_memory_lifecycle_kind_t kind = 0;
  if (!iree_profile_memory_event_pool_kind(event, &kind)) {
    return iree_ok_status();
  }

  const uint64_t pool_id =
      iree_profile_memory_resolve_pool_id(context, event, kind);
  iree_profile_memory_pool_t* pool = NULL;
  IREE_RETURN_IF_ERROR(iree_profile_memory_get_pool(
      context, kind, event->physical_device_ordinal, pool_id,
      event->memory_type, &pool));
  ++pool->event_count;
  pool->buffer_usage |= event->buffer_usage;
  if (event->type == IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_WAIT) {
    ++pool->wait_count;
  } else if (event->type ==
             IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_MATERIALIZE) {
    ++pool->materialize_count;
  }

  if (iree_profile_memory_event_increases_live_bytes(event)) {
    iree_profile_memory_apply_live_increase(
        event->length, &pool->current_allocation_count,
        &pool->high_water_allocation_count, &pool->current_bytes,
        &pool->high_water_bytes, &pool->total_allocate_bytes);
  } else if (iree_profile_memory_event_decreases_live_bytes(event)) {
    iree_profile_memory_apply_live_decrease(
        event->length, &pool->current_allocation_count, &pool->current_bytes,
        &pool->total_free_bytes);
  }
  return iree_ok_status();
}

static iree_status_t iree_profile_memory_record_allocation_event(
    iree_profile_memory_context_t* context,
    const iree_hal_profile_memory_event_t* event) {
  iree_profile_memory_lifecycle_kind_t kind = 0;
  if (!iree_profile_memory_event_allocation_kind(event, &kind)) {
    return iree_ok_status();
  }

  const uint64_t pool_id =
      iree_profile_memory_resolve_pool_id(context, event, kind);
  iree_profile_memory_allocation_t* allocation = NULL;
  IREE_RETURN_IF_ERROR(iree_profile_memory_get_allocation(
      context, kind, event->physical_device_ordinal, event->allocation_id,
      pool_id, &allocation));
  if (allocation->first_event_id == 0) {
    allocation->first_event_id = event->event_id;
    allocation->first_host_time_ns = event->host_time_ns;
  }
  allocation->last_event_id = event->event_id;
  allocation->last_host_time_ns = event->host_time_ns;
  if (allocation->first_submission_id == 0 && event->submission_id != 0) {
    allocation->first_submission_id = event->submission_id;
  }
  if (event->submission_id != 0) {
    allocation->last_submission_id = event->submission_id;
  }
  allocation->backing_id =
      allocation->backing_id ? allocation->backing_id : event->backing_id;
  allocation->memory_type |= event->memory_type;
  allocation->buffer_usage |= event->buffer_usage;
  ++allocation->event_count;
  if (event->type == IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_WAIT) {
    ++allocation->wait_count;
  } else if (event->type ==
             IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_MATERIALIZE) {
    ++allocation->materialize_count;
  }

  if (iree_profile_memory_event_increases_live_bytes(event)) {
    allocation->total_allocate_bytes += event->length;
    allocation->current_bytes += event->length;
    allocation->high_water_bytes =
        iree_max(allocation->high_water_bytes, allocation->current_bytes);
  } else if (iree_profile_memory_event_decreases_live_bytes(event)) {
    allocation->total_free_bytes += event->length;
    allocation->current_bytes = event->length > allocation->current_bytes
                                    ? 0
                                    : allocation->current_bytes - event->length;
  }
  return iree_ok_status();
}

static void iree_profile_memory_print_event_jsonl(
    const iree_hal_profile_memory_event_t* event, FILE* file) {
  fprintf(file,
          "{\"type\":\"memory_event\",\"event_id\":%" PRIu64 ",\"event_type\":",
          event->event_id);
  iree_profile_fprint_json_string(
      file,
      iree_make_cstring_view(iree_profile_memory_event_type_name(event->type)));
  fprintf(file,
          ",\"event_type_value\":%u,\"flags\":%u,\"result\":%u"
          ",\"host_time_ns\":%" PRId64 ",\"allocation_id\":%" PRIu64
          ",\"pool_id\":%" PRIu64 ",\"backing_id\":%" PRIu64
          ",\"submission_id\":%" PRIu64
          ",\"physical_device_ordinal\":%u,\"queue_ordinal\":%u"
          ",\"frontier_entry_count\":%u,\"memory_type\":%" PRIu64
          ",\"buffer_usage\":%" PRIu64 ",\"offset\":%" PRIu64
          ",\"length\":%" PRIu64 ",\"alignment\":%" PRIu64 "}\n",
          event->type, event->flags, event->result, event->host_time_ns,
          event->allocation_id, event->pool_id, event->backing_id,
          event->submission_id, event->physical_device_ordinal,
          event->queue_ordinal, event->frontier_entry_count, event->memory_type,
          event->buffer_usage, event->offset, event->length, event->alignment);
}

static iree_status_t iree_profile_memory_process_event_records(
    iree_profile_memory_context_t* context,
    const iree_hal_profile_file_record_t* record, iree_string_view_t filter,
    int64_t id_filter, bool emit_events, FILE* file) {
  if (record->header.record_type != IREE_HAL_PROFILE_FILE_RECORD_TYPE_CHUNK) {
    return iree_ok_status();
  }
  if (!iree_string_view_equal(record->content_type,
                              IREE_HAL_PROFILE_CONTENT_TYPE_MEMORY_EVENTS)) {
    return iree_ok_status();
  }

  const bool is_truncated = iree_any_bit_set(
      record->header.chunk_flags, IREE_HAL_PROFILE_CHUNK_FLAG_TRUNCATED);
  iree_host_size_t payload_offset = 0;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         payload_offset < record->payload.data_length) {
    iree_host_size_t record_length = 0;
    status = iree_profile_payload_record_length(
        record->content_type, record->payload, payload_offset,
        sizeof(iree_hal_profile_memory_event_t), &record_length);
    if (iree_status_is_ok(status)) {
      iree_hal_profile_memory_event_t event;
      memcpy(&event, record->payload.data + payload_offset, sizeof(event));
      ++context->total_event_count;
      if (iree_profile_memory_event_matches(&event, id_filter, filter)) {
        ++context->matched_event_count;
        if (is_truncated) ++context->truncated_event_count;
        iree_profile_memory_device_t* device = NULL;
        status = iree_profile_memory_get_device(
            context, event.physical_device_ordinal, &device);
        if (iree_status_is_ok(status)) {
          iree_profile_memory_record_event(device, &event);
          status = iree_profile_memory_record_pool_event(context, &event);
        }
        if (iree_status_is_ok(status)) {
          status = iree_profile_memory_record_allocation_event(context, &event);
        }
        if (iree_status_is_ok(status)) {
          if (emit_events) {
            iree_profile_memory_print_event_jsonl(&event, file);
          }
        }
      }
      payload_offset += record_length;
    }
  }
  return status;
}

static iree_status_t iree_profile_memory_print_text(
    const iree_profile_memory_context_t* context, iree_string_view_t filter,
    int64_t id_filter, FILE* file) {
  uint64_t live_allocation_count = 0;
  for (iree_host_size_t i = 0; i < context->allocation_count; ++i) {
    if (context->allocations[i].current_bytes != 0) {
      ++live_allocation_count;
    }
  }

  fprintf(file, "IREE HAL profile memory summary\n");
  fprintf(file, "filter: %.*s\n", (int)filter.size, filter.data);
  if (id_filter >= 0) {
    fprintf(file, "id_filter: %" PRId64 "\n", id_filter);
  }
  fprintf(file,
          "events: total=%" PRIu64 " matched=%" PRIu64
          " truncated_matched=%" PRIu64 " devices=%" PRIhsz " pools=%" PRIhsz
          " allocation_lifecycles=%" PRIhsz " live_lifecycles=%" PRIu64 "\n",
          context->total_event_count, context->matched_event_count,
          context->truncated_event_count, context->device_count,
          context->pool_count, context->allocation_count,
          live_allocation_count);
  for (iree_host_size_t i = 0; i < context->device_count; ++i) {
    const iree_profile_memory_device_t* device = &context->devices[i];
    fprintf(file,
            "device[%u]: events=%" PRIu64 " slab_acquire/release=%" PRIu64
            "/%" PRIu64 " pool_reserve/materialize/release/wait=%" PRIu64
            "/%" PRIu64 "/%" PRIu64 "/%" PRIu64
            " queue_alloca/dealloca=%" PRIu64 "/%" PRIu64
            " buffer_allocate/free=%" PRIu64 "/%" PRIu64 "\n",
            device->physical_device_ordinal, device->event_count,
            device->slab_acquire_count, device->slab_release_count,
            device->pool_reserve_count, device->pool_materialize_count,
            device->pool_release_count, device->pool_wait_count,
            device->queue_alloca_count, device->queue_dealloca_count,
            device->buffer_allocate_count, device->buffer_free_count);
    fprintf(file,
            "  slab_provider_events: live_at_end=%" PRIu64 " peak_live=%" PRIu64
            " current_bytes=%" PRIu64 " high_water_bytes=%" PRIu64
            " acquired_bytes=%" PRIu64 " released_bytes=%" PRIu64 "\n",
            device->current_slab_allocation_count,
            device->high_water_slab_allocation_count,
            device->current_slab_bytes, device->high_water_slab_bytes,
            device->total_slab_acquired_bytes,
            device->total_slab_released_bytes);
    fprintf(file,
            "  pool_reservations: live_at_end=%" PRIu64 " peak_live=%" PRIu64
            " current_bytes=%" PRIu64 " high_water_bytes=%" PRIu64
            " reserved_bytes=%" PRIu64 " released_bytes=%" PRIu64 "\n",
            device->current_pool_reservation_count,
            device->high_water_pool_reservation_count,
            device->current_pool_reserved_bytes,
            device->high_water_pool_reserved_bytes,
            device->total_pool_reserved_bytes,
            device->total_pool_released_bytes);
    fprintf(file,
            "  queue_allocations: live_at_end=%" PRIu64 " peak_live=%" PRIu64
            " current_bytes=%" PRIu64 " high_water_bytes=%" PRIu64
            " alloca_bytes=%" PRIu64 " dealloca_bytes=%" PRIu64 "\n",
            device->current_queue_allocation_count,
            device->high_water_queue_allocation_count,
            device->current_queue_bytes, device->high_water_queue_bytes,
            device->total_queue_alloca_bytes,
            device->total_queue_dealloca_bytes);
    fprintf(file,
            "  buffer_allocations: live_at_end=%" PRIu64 " peak_live=%" PRIu64
            " current_bytes=%" PRIu64 " high_water_bytes=%" PRIu64
            " allocate_bytes=%" PRIu64 " free_bytes=%" PRIu64 "\n",
            device->current_buffer_allocation_count,
            device->high_water_buffer_allocation_count,
            device->current_buffer_bytes, device->high_water_buffer_bytes,
            device->total_buffer_allocate_bytes,
            device->total_buffer_free_bytes);
  }
  for (iree_host_size_t i = 0; i < context->pool_count; ++i) {
    const iree_profile_memory_pool_t* pool = &context->pools[i];
    fprintf(file,
            "pool[%s device=%u id=%" PRIu64 " memory_type=%" PRIu64
            "]: events=%" PRIu64 " waits=%" PRIu64 " materializes=%" PRIu64
            " live_at_end=%" PRIu64 " peak_live=%" PRIu64
            " current_bytes=%" PRIu64 " high_water_bytes=%" PRIu64
            " allocate_bytes=%" PRIu64 " free_bytes=%" PRIu64 "\n",
            iree_profile_memory_lifecycle_kind_name(pool->kind),
            pool->physical_device_ordinal, pool->pool_id, pool->memory_type,
            pool->event_count, pool->wait_count, pool->materialize_count,
            pool->current_allocation_count, pool->high_water_allocation_count,
            pool->current_bytes, pool->high_water_bytes,
            pool->total_allocate_bytes, pool->total_free_bytes);
  }
  if (id_filter >= 0) {
    for (iree_host_size_t i = 0; i < context->allocation_count; ++i) {
      const iree_profile_memory_allocation_t* allocation =
          &context->allocations[i];
      const int64_t duration_ns =
          allocation->last_host_time_ns >= allocation->first_host_time_ns
              ? allocation->last_host_time_ns - allocation->first_host_time_ns
              : 0;
      fprintf(
          file,
          "allocation[%s device=%u id=%" PRIu64 " pool=%" PRIu64
          " backing=%" PRIu64 "]: events=%" PRIu64 " waits=%" PRIu64
          " materializes=%" PRIu64 " live_at_end=%s current_bytes=%" PRIu64
          " high_water_bytes=%" PRIu64 " allocate_bytes=%" PRIu64
          " free_bytes=%" PRIu64 " first_event=%" PRIu64 " last_event=%" PRIu64
          " duration_ns=%" PRId64 "\n",
          iree_profile_memory_lifecycle_kind_name(allocation->kind),
          allocation->physical_device_ordinal, allocation->allocation_id,
          allocation->pool_id, allocation->backing_id, allocation->event_count,
          allocation->wait_count, allocation->materialize_count,
          allocation->current_bytes != 0 ? "true" : "false",
          allocation->current_bytes, allocation->high_water_bytes,
          allocation->total_allocate_bytes, allocation->total_free_bytes,
          allocation->first_event_id, allocation->last_event_id, duration_ns);
    }
  }
  return iree_ok_status();
}

static void iree_profile_memory_print_jsonl_summary(
    const iree_profile_memory_context_t* context, iree_string_view_t filter,
    int64_t id_filter, FILE* file) {
  uint64_t live_allocation_count = 0;
  for (iree_host_size_t i = 0; i < context->allocation_count; ++i) {
    if (context->allocations[i].current_bytes != 0) {
      ++live_allocation_count;
    }
  }

  fprintf(file, "{\"type\":\"memory_summary\",\"filter\":");
  iree_profile_fprint_json_string(file, filter);
  fprintf(file,
          ",\"id_filter\":%" PRId64 ",\"total_events\":%" PRIu64
          ",\"matched_events\":%" PRIu64
          ",\"truncated_matched_events\":%" PRIu64 ",\"devices\":%" PRIhsz
          ",\"pools\":%" PRIhsz ",\"allocation_lifecycles\":%" PRIhsz
          ",\"live_allocation_lifecycles\":%" PRIu64 "}\n",
          id_filter, context->total_event_count, context->matched_event_count,
          context->truncated_event_count, context->device_count,
          context->pool_count, context->allocation_count,
          live_allocation_count);
  for (iree_host_size_t i = 0; i < context->device_count; ++i) {
    const iree_profile_memory_device_t* device = &context->devices[i];
    fprintf(
        file,
        "{\"type\":\"memory_device\",\"physical_device_ordinal\":%u"
        ",\"events\":%" PRIu64 ",\"slab_acquires\":%" PRIu64
        ",\"slab_releases\":%" PRIu64 ",\"pool_reserves\":%" PRIu64
        ",\"pool_materializes\":%" PRIu64 ",\"pool_releases\":%" PRIu64
        ",\"pool_waits\":%" PRIu64 ",\"queue_allocas\":%" PRIu64
        ",\"queue_deallocas\":%" PRIu64 ",\"buffer_allocates\":%" PRIu64
        ",\"buffer_frees\":%" PRIu64 ",\"current_slab_allocations\":%" PRIu64
        ",\"high_water_slab_allocations\":%" PRIu64
        ",\"current_slab_bytes\":%" PRIu64 ",\"high_water_slab_bytes\":%" PRIu64
        ",\"total_slab_acquired_bytes\":%" PRIu64
        ",\"total_slab_released_bytes\":%" PRIu64
        ",\"current_pool_reservations\":%" PRIu64
        ",\"high_water_pool_reservations\":%" PRIu64
        ",\"current_pool_reserved_bytes\":%" PRIu64
        ",\"high_water_pool_reserved_bytes\":%" PRIu64
        ",\"total_pool_reserved_bytes\":%" PRIu64
        ",\"total_pool_released_bytes\":%" PRIu64
        ",\"current_queue_allocations\":%" PRIu64
        ",\"high_water_queue_allocations\":%" PRIu64
        ",\"current_queue_bytes\":%" PRIu64
        ",\"high_water_queue_bytes\":%" PRIu64
        ",\"total_queue_alloca_bytes\":%" PRIu64
        ",\"total_queue_dealloca_bytes\":%" PRIu64
        ",\"current_buffer_allocations\":%" PRIu64
        ",\"high_water_buffer_allocations\":%" PRIu64
        ",\"current_buffer_bytes\":%" PRIu64
        ",\"high_water_buffer_bytes\":%" PRIu64
        ",\"total_buffer_allocate_bytes\":%" PRIu64
        ",\"total_buffer_free_bytes\":%" PRIu64 "}\n",
        device->physical_device_ordinal, device->event_count,
        device->slab_acquire_count, device->slab_release_count,
        device->pool_reserve_count, device->pool_materialize_count,
        device->pool_release_count, device->pool_wait_count,
        device->queue_alloca_count, device->queue_dealloca_count,
        device->buffer_allocate_count, device->buffer_free_count,
        device->current_slab_allocation_count,
        device->high_water_slab_allocation_count, device->current_slab_bytes,
        device->high_water_slab_bytes, device->total_slab_acquired_bytes,
        device->total_slab_released_bytes,
        device->current_pool_reservation_count,
        device->high_water_pool_reservation_count,
        device->current_pool_reserved_bytes,
        device->high_water_pool_reserved_bytes,
        device->total_pool_reserved_bytes, device->total_pool_released_bytes,
        device->current_queue_allocation_count,
        device->high_water_queue_allocation_count, device->current_queue_bytes,
        device->high_water_queue_bytes, device->total_queue_alloca_bytes,
        device->total_queue_dealloca_bytes,
        device->current_buffer_allocation_count,
        device->high_water_buffer_allocation_count,
        device->current_buffer_bytes, device->high_water_buffer_bytes,
        device->total_buffer_allocate_bytes, device->total_buffer_free_bytes);
  }
  for (iree_host_size_t i = 0; i < context->pool_count; ++i) {
    const iree_profile_memory_pool_t* pool = &context->pools[i];
    fprintf(file, "{\"type\":\"memory_pool\",\"kind\":");
    iree_profile_fprint_json_string(
        file, iree_make_cstring_view(
                  iree_profile_memory_lifecycle_kind_name(pool->kind)));
    fprintf(file,
            ",\"physical_device_ordinal\":%u,\"pool_id\":%" PRIu64
            ",\"memory_type\":%" PRIu64 ",\"buffer_usage\":%" PRIu64
            ",\"events\":%" PRIu64 ",\"waits\":%" PRIu64
            ",\"materializes\":%" PRIu64 ",\"current_allocations\":%" PRIu64
            ",\"high_water_allocations\":%" PRIu64 ",\"current_bytes\":%" PRIu64
            ",\"high_water_bytes\":%" PRIu64
            ",\"total_allocate_bytes\":%" PRIu64
            ",\"total_free_bytes\":%" PRIu64 "}\n",
            pool->physical_device_ordinal, pool->pool_id, pool->memory_type,
            pool->buffer_usage, pool->event_count, pool->wait_count,
            pool->materialize_count, pool->current_allocation_count,
            pool->high_water_allocation_count, pool->current_bytes,
            pool->high_water_bytes, pool->total_allocate_bytes,
            pool->total_free_bytes);
  }
  for (iree_host_size_t i = 0; i < context->allocation_count; ++i) {
    const iree_profile_memory_allocation_t* allocation =
        &context->allocations[i];
    const int64_t duration_ns =
        allocation->last_host_time_ns >= allocation->first_host_time_ns
            ? allocation->last_host_time_ns - allocation->first_host_time_ns
            : 0;
    fprintf(file, "{\"type\":\"memory_allocation\",\"kind\":");
    iree_profile_fprint_json_string(
        file, iree_make_cstring_view(
                  iree_profile_memory_lifecycle_kind_name(allocation->kind)));
    fprintf(
        file,
        ",\"physical_device_ordinal\":%u,\"allocation_id\":%" PRIu64
        ",\"pool_id\":%" PRIu64 ",\"backing_id\":%" PRIu64
        ",\"memory_type\":%" PRIu64 ",\"buffer_usage\":%" PRIu64
        ",\"events\":%" PRIu64 ",\"waits\":%" PRIu64
        ",\"materializes\":%" PRIu64
        ",\"live_at_end\":%s"
        ",\"current_bytes\":%" PRIu64 ",\"high_water_bytes\":%" PRIu64
        ",\"total_allocate_bytes\":%" PRIu64 ",\"total_free_bytes\":%" PRIu64
        ",\"first_event_id\":%" PRIu64 ",\"last_event_id\":%" PRIu64
        ",\"first_host_time_ns\":%" PRId64 ",\"last_host_time_ns\":%" PRId64
        ",\"duration_ns\":%" PRId64 ",\"first_submission_id\":%" PRIu64
        ",\"last_submission_id\":%" PRIu64 "}\n",
        allocation->physical_device_ordinal, allocation->allocation_id,
        allocation->pool_id, allocation->backing_id, allocation->memory_type,
        allocation->buffer_usage, allocation->event_count,
        allocation->wait_count, allocation->materialize_count,
        allocation->current_bytes != 0 ? "true" : "false",
        allocation->current_bytes, allocation->high_water_bytes,
        allocation->total_allocate_bytes, allocation->total_free_bytes,
        allocation->first_event_id, allocation->last_event_id,
        allocation->first_host_time_ns, allocation->last_host_time_ns,
        duration_ns, allocation->first_submission_id,
        allocation->last_submission_id);
  }
}

static iree_status_t iree_profile_memory_file(iree_string_view_t path,
                                              iree_string_view_t format,
                                              iree_string_view_t filter,
                                              int64_t id_filter, FILE* file,
                                              iree_allocator_t host_allocator) {
  bool is_text = iree_string_view_equal(format, IREE_SV("text"));
  bool is_jsonl = iree_string_view_equal(format, IREE_SV("jsonl"));
  if (!is_text && !is_jsonl) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported profile output format '%.*s'",
                            (int)format.size, format.data);
  }

  iree_io_file_contents_t* file_contents = NULL;
  iree_status_t status = iree_io_file_contents_map(
      path, IREE_IO_FILE_ACCESS_READ, host_allocator, &file_contents);

  iree_hal_profile_file_header_t header;
  iree_host_size_t record_offset = 0;
  if (iree_status_is_ok(status)) {
    status = iree_hal_profile_file_parse_header(file_contents->const_buffer,
                                                &header, &record_offset);
  }

  iree_profile_memory_context_t context;
  iree_profile_memory_context_initialize(host_allocator, &context);
  while (iree_status_is_ok(status) &&
         record_offset < file_contents->const_buffer.data_length) {
    iree_hal_profile_file_record_t record;
    iree_host_size_t next_record_offset = 0;
    status = iree_hal_profile_file_parse_record(file_contents->const_buffer,
                                                record_offset, &record,
                                                &next_record_offset);
    if (iree_status_is_ok(status)) {
      status = iree_profile_memory_process_event_records(
          &context, &record, filter, id_filter, is_jsonl, file);
      record_offset = next_record_offset;
    }
  }

  if (iree_status_is_ok(status)) {
    if (is_text) {
      status =
          iree_profile_memory_print_text(&context, filter, id_filter, file);
    } else {
      iree_profile_memory_print_jsonl_summary(&context, filter, id_filter,
                                              file);
    }
  }

  iree_profile_memory_context_deinitialize(&context);
  iree_io_file_contents_free(file_contents);
  return status;
}

static iree_status_t iree_profile_summary_file(
    iree_string_view_t path, iree_string_view_t format, FILE* file,
    iree_allocator_t host_allocator) {
  bool is_text = iree_string_view_equal(format, IREE_SV("text"));
  bool is_jsonl = iree_string_view_equal(format, IREE_SV("jsonl"));
  if (!is_text && !is_jsonl) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported profile output format '%.*s'",
                            (int)format.size, format.data);
  }

  iree_io_file_contents_t* file_contents = NULL;
  iree_status_t status = iree_io_file_contents_map(
      path, IREE_IO_FILE_ACCESS_READ, host_allocator, &file_contents);

  iree_hal_profile_file_header_t header;
  iree_host_size_t record_offset = 0;
  if (iree_status_is_ok(status)) {
    status = iree_hal_profile_file_parse_header(file_contents->const_buffer,
                                                &header, &record_offset);
  }

  iree_profile_summary_t summary;
  iree_profile_summary_initialize(host_allocator, &summary);
  while (iree_status_is_ok(status) &&
         record_offset < file_contents->const_buffer.data_length) {
    iree_hal_profile_file_record_t record;
    iree_host_size_t next_record_offset = 0;
    status = iree_hal_profile_file_parse_record(file_contents->const_buffer,
                                                record_offset, &record,
                                                &next_record_offset);
    if (iree_status_is_ok(status)) {
      status = iree_profile_summary_process_record(&summary, &record);
      record_offset = next_record_offset;
    }
  }

  if (iree_status_is_ok(status)) {
    if (is_text) {
      iree_profile_print_summary_text(&summary, file);
    } else {
      iree_profile_print_summary_jsonl(&summary, file);
    }
  }

  iree_profile_summary_deinitialize(&summary);
  iree_io_file_contents_free(file_contents);
  return status;
}

typedef struct iree_profile_explain_export_rank_t {
  // Session-local physical device ordinal for this export aggregate.
  uint32_t physical_device_ordinal;
  // Producer-local executable identifier for this export aggregate.
  uint64_t executable_id;
  // Export ordinal for this export aggregate.
  uint32_t export_ordinal;
  // Total dispatch count for this export aggregate.
  uint64_t dispatch_count;
  // Valid dispatch count for this export aggregate.
  uint64_t valid_count;
  // Invalid dispatch count for this export aggregate.
  uint64_t invalid_count;
  // Maximum valid dispatch duration in raw device ticks.
  uint64_t maximum_ticks;
  // Total valid dispatch duration in raw device ticks.
  double total_ticks;
  // Average valid dispatch duration in nanoseconds when clock fit is available.
  double average_ns;
  // Maximum valid dispatch duration in nanoseconds when clock fit is available.
  double maximum_ns;
  // Total valid dispatch duration in nanoseconds when clock fit is available.
  double total_ns;
  // True when nanosecond values were computed from a device clock fit.
  bool has_clock_fit;
} iree_profile_explain_export_rank_t;

typedef struct iree_profile_explain_interval_t {
  // Inclusive dispatch start tick for this interval.
  uint64_t start_tick;
  // Exclusive dispatch end tick for this interval.
  uint64_t end_tick;
} iree_profile_explain_interval_t;

static double iree_profile_explain_export_rank_score(
    const iree_profile_explain_export_rank_t* rank) {
  return rank->has_clock_fit ? rank->total_ns : rank->total_ticks;
}

static int iree_profile_explain_compare_export_rank(const void* lhs,
                                                    const void* rhs) {
  const iree_profile_explain_export_rank_t* a =
      (const iree_profile_explain_export_rank_t*)lhs;
  const iree_profile_explain_export_rank_t* b =
      (const iree_profile_explain_export_rank_t*)rhs;
  const double a_score = iree_profile_explain_export_rank_score(a);
  const double b_score = iree_profile_explain_export_rank_score(b);
  if (a_score < b_score) return 1;
  if (a_score > b_score) return -1;
  return 0;
}

static int iree_profile_explain_compare_top_event(const void* lhs,
                                                  const void* rhs) {
  const iree_profile_dispatch_top_event_t* a =
      (const iree_profile_dispatch_top_event_t*)lhs;
  const iree_profile_dispatch_top_event_t* b =
      (const iree_profile_dispatch_top_event_t*)rhs;
  if (a->duration_ticks < b->duration_ticks) return 1;
  if (a->duration_ticks > b->duration_ticks) return -1;
  return 0;
}

static int iree_profile_explain_compare_interval(const void* lhs,
                                                 const void* rhs) {
  const iree_profile_explain_interval_t* a =
      (const iree_profile_explain_interval_t*)lhs;
  const iree_profile_explain_interval_t* b =
      (const iree_profile_explain_interval_t*)rhs;
  if (a->start_tick < b->start_tick) return -1;
  if (a->start_tick > b->start_tick) return 1;
  if (a->end_tick < b->end_tick) return -1;
  if (a->end_tick > b->end_tick) return 1;
  return 0;
}

static iree_status_t iree_profile_explain_collect_export_ranks(
    const iree_profile_dispatch_context_t* context,
    iree_allocator_t host_allocator,
    iree_profile_explain_export_rank_t** out_ranks,
    iree_host_size_t* out_rank_count) {
  *out_ranks = NULL;
  *out_rank_count = 0;
  if (context->aggregate_count == 0) return iree_ok_status();

  iree_profile_explain_export_rank_t* ranks = NULL;
  iree_status_t status = iree_allocator_malloc_array_uninitialized(
      host_allocator, context->aggregate_count, sizeof(ranks[0]),
      (void**)&ranks);

  iree_host_size_t rank_count = 0;
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < context->aggregate_count; ++i) {
      const iree_profile_dispatch_aggregate_t* aggregate =
          &context->aggregates[i];
      if (aggregate->valid_count == 0) continue;

      const iree_profile_dispatch_device_t* device =
          iree_profile_dispatch_find_device(context,
                                            aggregate->physical_device_ordinal);
      double ns_per_tick = 0.0;
      double tick_frequency_hz = 0.0;
      const bool has_clock_fit = iree_profile_dispatch_device_try_fit_clock(
          device, &ns_per_tick, &tick_frequency_hz);
      (void)tick_frequency_hz;

      iree_profile_explain_export_rank_t* rank = &ranks[rank_count++];
      memset(rank, 0, sizeof(*rank));
      rank->physical_device_ordinal = aggregate->physical_device_ordinal;
      rank->executable_id = aggregate->executable_id;
      rank->export_ordinal = aggregate->export_ordinal;
      rank->dispatch_count = aggregate->dispatch_count;
      rank->valid_count = aggregate->valid_count;
      rank->invalid_count = aggregate->invalid_count;
      rank->maximum_ticks = aggregate->maximum_ticks;
      rank->total_ticks = aggregate->total_ticks;
      rank->has_clock_fit = has_clock_fit;
      if (has_clock_fit) {
        rank->average_ns = aggregate->mean_ticks * ns_per_tick;
        rank->maximum_ns = (double)aggregate->maximum_ticks * ns_per_tick;
        rank->total_ns = aggregate->total_ticks * ns_per_tick;
      }
    }
  }

  if (iree_status_is_ok(status)) {
    qsort(ranks, rank_count, sizeof(ranks[0]),
          iree_profile_explain_compare_export_rank);
    *out_ranks = ranks;
    *out_rank_count = rank_count;
  } else {
    iree_allocator_free(host_allocator, ranks);
  }
  return status;
}

static void iree_profile_explain_accumulate_queue_operation_totals(
    const iree_profile_dispatch_context_t* context, uint64_t event_counts[],
    uint64_t payload_bytes[], uint64_t strategy_counts[]) {
  for (iree_host_size_t i = 0; i < context->queue_event_count; ++i) {
    const iree_hal_profile_queue_event_t* event =
        &context->queue_events[i].record;
    if (event->type <= IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_HOST_CALL) {
      ++event_counts[event->type];
      payload_bytes[event->type] += event->payload_length;
    }
    if (event->dependency_strategy <=
        IREE_HAL_PROFILE_QUEUE_DEPENDENCY_STRATEGY_SOFTWARE_DEFER) {
      ++strategy_counts[event->dependency_strategy];
    }
  }
}

static double iree_profile_explain_visible_span_ticks(
    const iree_profile_dispatch_context_t* context,
    uint32_t physical_device_ordinal) {
  uint64_t earliest_start_tick = UINT64_MAX;
  uint64_t latest_end_tick = 0;
  for (iree_host_size_t i = 0; i < context->queue_aggregate_count; ++i) {
    const iree_profile_dispatch_queue_aggregate_t* aggregate =
        &context->queue_aggregates[i];
    if (aggregate->physical_device_ordinal != physical_device_ordinal ||
        aggregate->valid_count == 0) {
      continue;
    }
    earliest_start_tick =
        iree_min(earliest_start_tick, aggregate->earliest_start_tick);
    latest_end_tick = iree_max(latest_end_tick, aggregate->latest_end_tick);
  }
  return iree_profile_dispatch_span_ticks(earliest_start_tick, latest_end_tick);
}

static double iree_profile_explain_total_dispatch_ticks_for_device(
    const iree_profile_dispatch_context_t* context,
    uint32_t physical_device_ordinal) {
  double total_ticks = 0.0;
  for (iree_host_size_t i = 0; i < context->aggregate_count; ++i) {
    const iree_profile_dispatch_aggregate_t* aggregate =
        &context->aggregates[i];
    if (aggregate->physical_device_ordinal == physical_device_ordinal) {
      total_ticks += aggregate->total_ticks;
    }
  }
  return total_ticks;
}

static iree_status_t iree_profile_explain_summarize_queue(
    const iree_profile_dispatch_context_t* context,
    const iree_hal_profile_queue_record_t* queue,
    iree_allocator_t host_allocator, uint64_t* out_submission_count,
    uint64_t* out_valid_submission_count,
    uint64_t* out_invalid_submission_count, double* out_busy_ticks,
    double* out_total_dispatch_ticks, uint64_t* out_gap_count,
    double* out_total_gap_ticks, double* out_max_gap_ticks) {
  *out_submission_count = 0;
  *out_valid_submission_count = 0;
  *out_invalid_submission_count = 0;
  *out_busy_ticks = 0.0;
  *out_total_dispatch_ticks = 0.0;
  *out_gap_count = 0;
  *out_total_gap_ticks = 0.0;
  *out_max_gap_ticks = 0.0;
  if (context->queue_aggregate_count == 0) return iree_ok_status();

  iree_profile_explain_interval_t* intervals = NULL;
  iree_status_t status = iree_allocator_malloc_array_uninitialized(
      host_allocator, context->queue_aggregate_count, sizeof(intervals[0]),
      (void**)&intervals);

  iree_host_size_t interval_count = 0;
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < context->queue_aggregate_count; ++i) {
      const iree_profile_dispatch_queue_aggregate_t* aggregate =
          &context->queue_aggregates[i];
      if (aggregate->physical_device_ordinal !=
              queue->physical_device_ordinal ||
          aggregate->queue_ordinal != queue->queue_ordinal ||
          aggregate->stream_id != queue->stream_id) {
        continue;
      }
      *out_submission_count += 1;
      *out_total_dispatch_ticks += aggregate->total_ticks;
      if (aggregate->valid_count != 0) {
        *out_valid_submission_count += 1;
        intervals[interval_count++] = (iree_profile_explain_interval_t){
            aggregate->earliest_start_tick, aggregate->latest_end_tick};
      }
      if (aggregate->invalid_count != 0) {
        *out_invalid_submission_count += 1;
      }
    }
  }

  if (iree_status_is_ok(status) && interval_count != 0) {
    qsort(intervals, interval_count, sizeof(intervals[0]),
          iree_profile_explain_compare_interval);

    uint64_t merged_start_tick = intervals[0].start_tick;
    uint64_t merged_end_tick = intervals[0].end_tick;
    for (iree_host_size_t i = 1; i < interval_count; ++i) {
      if (intervals[i].start_tick <= merged_end_tick) {
        merged_end_tick = iree_max(merged_end_tick, intervals[i].end_tick);
        continue;
      }
      *out_busy_ticks += (double)(merged_end_tick - merged_start_tick);
      const uint64_t gap_ticks = intervals[i].start_tick - merged_end_tick;
      ++*out_gap_count;
      *out_total_gap_ticks += (double)gap_ticks;
      *out_max_gap_ticks = iree_max(*out_max_gap_ticks, (double)gap_ticks);
      merged_start_tick = intervals[i].start_tick;
      merged_end_tick = intervals[i].end_tick;
    }
    *out_busy_ticks += (double)(merged_end_tick - merged_start_tick);
  }

  iree_allocator_free(host_allocator, intervals);
  return status;
}

static void iree_profile_explain_print_hint_text(const char* severity,
                                                 const char* category,
                                                 const char* message,
                                                 FILE* file) {
  fprintf(file, "  [%s] %s: %s\n", severity, category, message);
}

static void iree_profile_explain_print_hint_jsonl(const char* severity,
                                                  const char* category,
                                                  const char* message,
                                                  FILE* file) {
  fprintf(file, "{\"type\":\"explain_hint\",\"severity\":");
  iree_profile_fprint_json_string(file, iree_make_cstring_view(severity));
  fprintf(file, ",\"category\":");
  iree_profile_fprint_json_string(file, iree_make_cstring_view(category));
  fprintf(file, ",\"message\":");
  iree_profile_fprint_json_string(file, iree_make_cstring_view(message));
  fputs("}\n", file);
}

static iree_status_t iree_profile_explain_print_text(
    const iree_profile_summary_t* summary,
    const iree_profile_dispatch_context_t* dispatch_context,
    const iree_profile_memory_context_t* memory_context,
    iree_allocator_t host_allocator, FILE* file) {
  iree_profile_explain_export_rank_t* export_ranks = NULL;
  iree_host_size_t export_rank_count = 0;
  iree_status_t status = iree_profile_explain_collect_export_ranks(
      dispatch_context, host_allocator, &export_ranks, &export_rank_count);

  if (iree_status_is_ok(status)) {
    fprintf(file, "IREE HAL profile explain\n");
    fprintf(file,
            "health: records=%" PRIu64 " chunks=%" PRIu64
            " unknown_records=%" PRIu64 " unknown_chunks=%" PRIu64
            " truncated_chunks=%" PRIu64 "\n",
            summary->file_record_count, summary->chunk_count,
            summary->unknown_record_count, summary->unknown_chunk_count,
            summary->truncated_chunk_count);
    fprintf(file,
            "coverage: dispatches=%" PRIu64 " valid=%" PRIu64
            " invalid=%" PRIu64 " queues=%" PRIhsz " queue_events=%" PRIhsz
            " command_buffers=%" PRIhsz " command_operations=%" PRIhsz
            " memory_events=%" PRIu64 "\n",
            dispatch_context->matched_dispatch_count,
            dispatch_context->valid_dispatch_count,
            dispatch_context->invalid_dispatch_count,
            dispatch_context->queue_count, dispatch_context->queue_event_count,
            dispatch_context->command_buffer_count,
            dispatch_context->command_operation_count,
            memory_context->matched_event_count);

    fprintf(file, "devices:\n");
    for (iree_host_size_t i = 0; i < dispatch_context->device_count; ++i) {
      const iree_profile_dispatch_device_t* device =
          &dispatch_context->devices[i];
      double ns_per_tick = 0.0;
      double tick_frequency_hz = 0.0;
      const bool has_clock_fit = iree_profile_dispatch_device_try_fit_clock(
          device, &ns_per_tick, &tick_frequency_hz);
      const double visible_span_ticks = iree_profile_explain_visible_span_ticks(
          dispatch_context, device->physical_device_ordinal);
      const double total_dispatch_ticks =
          iree_profile_explain_total_dispatch_ticks_for_device(
              dispatch_context, device->physical_device_ordinal);
      const double active_ratio =
          visible_span_ticks > 0.0 ? total_dispatch_ticks / visible_span_ticks
                                   : 0.0;
      fprintf(file,
              "  device[%u]: clock_fit=%s clock_samples=%" PRIu64
              " visible_span_ticks=%.3f active_dispatch_ticks=%.3f"
              " active_over_visible=%.3f\n",
              device->physical_device_ordinal, has_clock_fit ? "true" : "false",
              device->clock_sample_count, visible_span_ticks,
              total_dispatch_ticks, active_ratio);
      if (has_clock_fit) {
        fprintf(file,
                "    visible_span_ns=%.3f active_dispatch_ns=%.3f"
                " tick_frequency_hz=%.3f\n",
                visible_span_ticks * ns_per_tick,
                total_dispatch_ticks * ns_per_tick, tick_frequency_hz);
      }
    }

    fprintf(file, "queues:\n");
    for (iree_host_size_t i = 0;
         i < dispatch_context->queue_count && iree_status_is_ok(status); ++i) {
      const iree_hal_profile_queue_record_t* queue =
          &dispatch_context->queues[i].record;
      uint64_t submission_count = 0;
      uint64_t valid_submission_count = 0;
      uint64_t invalid_submission_count = 0;
      double busy_ticks = 0.0;
      double total_dispatch_ticks = 0.0;
      uint64_t gap_count = 0;
      double total_gap_ticks = 0.0;
      double max_gap_ticks = 0.0;
      status = iree_profile_explain_summarize_queue(
          dispatch_context, queue, host_allocator, &submission_count,
          &valid_submission_count, &invalid_submission_count, &busy_ticks,
          &total_dispatch_ticks, &gap_count, &total_gap_ticks, &max_gap_ticks);
      if (iree_status_is_ok(status)) {
        const iree_profile_dispatch_device_t* device =
            iree_profile_dispatch_find_device(dispatch_context,
                                              queue->physical_device_ordinal);
        double ns_per_tick = 0.0;
        double tick_frequency_hz = 0.0;
        const bool has_clock_fit = iree_profile_dispatch_device_try_fit_clock(
            device, &ns_per_tick, &tick_frequency_hz);
        (void)tick_frequency_hz;
        fprintf(file,
                "  queue device=%u ordinal=%u stream=%" PRIu64
                ": submissions=%" PRIu64 " valid=%" PRIu64 " invalid=%" PRIu64
                " busy_ticks=%.3f"
                " total_dispatch_ticks=%.3f",
                queue->physical_device_ordinal, queue->queue_ordinal,
                queue->stream_id, submission_count, valid_submission_count,
                invalid_submission_count, busy_ticks, total_dispatch_ticks);
        if (has_clock_fit) {
          fprintf(file, " busy_ns=%.3f total_dispatch_ns=%.3f",
                  busy_ticks * ns_per_tick, total_dispatch_ticks * ns_per_tick);
        }
        if (dispatch_context->queue_event_count != 0) {
          fprintf(file,
                  " gaps=%" PRIu64 " total_gap_ticks=%.3f max_gap_ticks=%.3f",
                  gap_count, total_gap_ticks, max_gap_ticks);
          if (has_clock_fit) {
            fprintf(file, " total_gap_ns=%.3f max_gap_ns=%.3f",
                    total_gap_ticks * ns_per_tick, max_gap_ticks * ns_per_tick);
          }
        } else {
          fprintf(file, " gaps=unavailable_without_queue_events");
        }
        fputc('\n', file);
      }
    }

    fprintf(file, "top exports by total dispatch time:\n");
    const iree_host_size_t top_export_count =
        iree_min(export_rank_count,
                 (iree_host_size_t)IREE_PROFILE_EXPLAIN_TOP_EXPORT_COUNT);
    for (iree_host_size_t i = 0;
         i < top_export_count && iree_status_is_ok(status); ++i) {
      const iree_profile_explain_export_rank_t* rank = &export_ranks[i];
      char numeric_buffer[128];
      iree_string_view_t key = iree_string_view_empty();
      status = iree_profile_dispatch_resolve_key(
          dispatch_context, rank->physical_device_ordinal, rank->executable_id,
          rank->export_ordinal, numeric_buffer, sizeof(numeric_buffer), &key);
      if (iree_status_is_ok(status)) {
        fprintf(file,
                "  #%" PRIhsz " %.*s device=%u executable=%" PRIu64
                " export=%u count=%" PRIu64 " valid=%" PRIu64
                " invalid=%" PRIu64 " total_ticks=%.3f max_ticks=%" PRIu64,
                i + 1, (int)key.size, key.data, rank->physical_device_ordinal,
                rank->executable_id, rank->export_ordinal, rank->dispatch_count,
                rank->valid_count, rank->invalid_count, rank->total_ticks,
                rank->maximum_ticks);
        if (rank->has_clock_fit) {
          fprintf(file, " total_ns=%.3f avg_ns=%.3f max_ns=%.3f",
                  rank->total_ns, rank->average_ns, rank->maximum_ns);
        }
        fputc('\n', file);
      }
    }

    iree_profile_dispatch_top_event_t
        top_dispatches[IREE_PROFILE_EXPLAIN_TOP_DISPATCH_COUNT];
    memcpy(top_dispatches, dispatch_context->top_dispatches,
           dispatch_context->top_dispatch_count * sizeof(top_dispatches[0]));
    qsort(top_dispatches, dispatch_context->top_dispatch_count,
          sizeof(top_dispatches[0]), iree_profile_explain_compare_top_event);
    fprintf(file, "top individual dispatches:\n");
    for (iree_host_size_t i = 0;
         i < dispatch_context->top_dispatch_count && iree_status_is_ok(status);
         ++i) {
      const iree_profile_dispatch_top_event_t* top_event = &top_dispatches[i];
      const iree_profile_dispatch_device_t* device =
          iree_profile_dispatch_find_device(dispatch_context,
                                            top_event->physical_device_ordinal);
      double ns_per_tick = 0.0;
      double tick_frequency_hz = 0.0;
      const bool has_clock_fit = iree_profile_dispatch_device_try_fit_clock(
          device, &ns_per_tick, &tick_frequency_hz);
      (void)tick_frequency_hz;
      char numeric_buffer[128];
      iree_string_view_t key = iree_string_view_empty();
      status = iree_profile_dispatch_resolve_key(
          dispatch_context, top_event->physical_device_ordinal,
          top_event->event.executable_id, top_event->event.export_ordinal,
          numeric_buffer, sizeof(numeric_buffer), &key);
      if (iree_status_is_ok(status)) {
        fprintf(file,
                "  #%" PRIhsz " event=%" PRIu64 " submission=%" PRIu64
                " %.*s device=%u queue=%u stream=%" PRIu64
                " duration_ticks=%" PRIu64,
                i + 1, top_event->event.event_id,
                top_event->event.submission_id, (int)key.size, key.data,
                top_event->physical_device_ordinal, top_event->queue_ordinal,
                top_event->stream_id, top_event->duration_ticks);
        if (has_clock_fit) {
          fprintf(file, " duration_ns=%.3f",
                  (double)top_event->duration_ticks * ns_per_tick);
        }
        fputc('\n', file);
      }
    }

    uint64_t event_counts[IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_HOST_CALL + 1] = {
        0};
    uint64_t payload_bytes[IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_HOST_CALL + 1] = {
        0};
    uint64_t strategy_counts
        [IREE_HAL_PROFILE_QUEUE_DEPENDENCY_STRATEGY_SOFTWARE_DEFER + 1] = {0};
    iree_profile_explain_accumulate_queue_operation_totals(
        dispatch_context, event_counts, payload_bytes, strategy_counts);
    fprintf(file,
            "queue operations: events=%" PRIhsz
            " strategies none/inline/device_barrier/software_defer=%" PRIu64
            "/%" PRIu64 "/%" PRIu64 "/%" PRIu64 "\n",
            dispatch_context->queue_event_count,
            strategy_counts[IREE_HAL_PROFILE_QUEUE_DEPENDENCY_STRATEGY_NONE],
            strategy_counts[IREE_HAL_PROFILE_QUEUE_DEPENDENCY_STRATEGY_INLINE],
            strategy_counts
                [IREE_HAL_PROFILE_QUEUE_DEPENDENCY_STRATEGY_DEVICE_BARRIER],
            strategy_counts
                [IREE_HAL_PROFILE_QUEUE_DEPENDENCY_STRATEGY_SOFTWARE_DEFER]);
    fprintf(file,
            "  transfer_payload_bytes copy/fill/update/read/write=%" PRIu64
            "/%" PRIu64 "/%" PRIu64 "/%" PRIu64 "/%" PRIu64 "\n",
            payload_bytes[IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_COPY],
            payload_bytes[IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_FILL],
            payload_bytes[IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_UPDATE],
            payload_bytes[IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_READ],
            payload_bytes[IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_WRITE]);
    fprintf(
        file,
        "  operation_counts dispatch/execute/alloca/dealloca/host_call=%" PRIu64
        "/%" PRIu64 "/%" PRIu64 "/%" PRIu64 "/%" PRIu64 "\n",
        event_counts[IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_DISPATCH],
        event_counts[IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_EXECUTE],
        event_counts[IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_ALLOCA],
        event_counts[IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_DEALLOCA],
        event_counts[IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_HOST_CALL]);

    fprintf(file, "memory pressure:\n");
    for (iree_host_size_t i = 0; i < memory_context->device_count; ++i) {
      const iree_profile_memory_device_t* device = &memory_context->devices[i];
      fprintf(file,
              "  device[%u]: slab_high_water=%" PRIu64
              " pool_high_water=%" PRIu64 " queue_high_water=%" PRIu64
              " buffer_high_water=%" PRIu64 " pool_waits=%" PRIu64 "\n",
              device->physical_device_ordinal, device->high_water_slab_bytes,
              device->high_water_pool_reserved_bytes,
              device->high_water_queue_bytes, device->high_water_buffer_bytes,
              device->pool_wait_count);
    }

    fprintf(file, "hints:\n");
    if (summary->unknown_record_count != 0 ||
        summary->unknown_chunk_count != 0) {
      iree_profile_explain_print_hint_text(
          "warning", "format",
          "unknown records or chunks were present; newer producer data may be "
          "missing from this analysis",
          file);
    }
    if (summary->truncated_chunk_count != 0) {
      iree_profile_explain_print_hint_text(
          "error", "format",
          "truncated chunks were present; timing and lifecycle totals may be "
          "incomplete",
          file);
    }
    if (dispatch_context->invalid_dispatch_count != 0) {
      iree_profile_explain_print_hint_text(
          "warning", "dispatch",
          "some dispatch events had missing or reversed timestamps and were "
          "excluded from timing totals",
          file);
    }
    if (dispatch_context->queue_event_count == 0) {
      iree_profile_explain_print_hint_text(
          "info", "queue",
          "queue event records are absent; queue dependency and gap hints are "
          "limited to dispatch timestamp intervals",
          file);
    }
    for (iree_host_size_t i = 0; i < memory_context->device_count; ++i) {
      if (memory_context->devices[i].pool_wait_count != 0) {
        iree_profile_explain_print_hint_text(
            "warning", "memory",
            "pool wait events were observed; allocation readiness affected the "
            "captured queue schedule",
            file);
        break;
      }
    }
    if (dispatch_context->valid_dispatch_count == 0) {
      iree_profile_explain_print_hint_text(
          "info", "dispatch",
          "no valid dispatch events were captured; enable queue profiling on a "
          "producer that emits device timestamps",
          file);
    }
  }

  iree_allocator_free(host_allocator, export_ranks);
  return status;
}

static iree_status_t iree_profile_explain_print_jsonl(
    const iree_profile_summary_t* summary,
    const iree_profile_dispatch_context_t* dispatch_context,
    const iree_profile_memory_context_t* memory_context,
    iree_allocator_t host_allocator, FILE* file) {
  iree_profile_explain_export_rank_t* export_ranks = NULL;
  iree_host_size_t export_rank_count = 0;
  iree_status_t status = iree_profile_explain_collect_export_ranks(
      dispatch_context, host_allocator, &export_ranks, &export_rank_count);

  if (iree_status_is_ok(status)) {
    fprintf(file,
            "{\"type\":\"explain_summary\",\"file_records\":%" PRIu64
            ",\"chunk_records\":%" PRIu64 ",\"unknown_records\":%" PRIu64
            ",\"unknown_chunks\":%" PRIu64 ",\"truncated_chunks\":%" PRIu64
            ",\"dispatches\":%" PRIu64 ",\"valid_dispatches\":%" PRIu64
            ",\"invalid_dispatches\":%" PRIu64 ",\"queues\":%" PRIhsz
            ",\"queue_events\":%" PRIhsz ",\"command_buffers\":%" PRIhsz
            ",\"command_operations\":%" PRIhsz ",\"memory_events\":%" PRIu64
            "}\n",
            summary->file_record_count, summary->chunk_count,
            summary->unknown_record_count, summary->unknown_chunk_count,
            summary->truncated_chunk_count,
            dispatch_context->matched_dispatch_count,
            dispatch_context->valid_dispatch_count,
            dispatch_context->invalid_dispatch_count,
            dispatch_context->queue_count, dispatch_context->queue_event_count,
            dispatch_context->command_buffer_count,
            dispatch_context->command_operation_count,
            memory_context->matched_event_count);

    for (iree_host_size_t i = 0; i < dispatch_context->device_count; ++i) {
      const iree_profile_dispatch_device_t* device =
          &dispatch_context->devices[i];
      double ns_per_tick = 0.0;
      double tick_frequency_hz = 0.0;
      const bool has_clock_fit = iree_profile_dispatch_device_try_fit_clock(
          device, &ns_per_tick, &tick_frequency_hz);
      const double visible_span_ticks = iree_profile_explain_visible_span_ticks(
          dispatch_context, device->physical_device_ordinal);
      const double total_dispatch_ticks =
          iree_profile_explain_total_dispatch_ticks_for_device(
              dispatch_context, device->physical_device_ordinal);
      fprintf(
          file,
          "{\"type\":\"explain_device\",\"physical_device_ordinal\":%u"
          ",\"clock_fit_available\":%s,\"clock_samples\":%" PRIu64
          ",\"visible_span_ticks\":%.3f"
          ",\"active_dispatch_ticks\":%.3f"
          ",\"active_over_visible\":%.6f"
          ",\"tick_frequency_hz\":%.3f,\"visible_span_ns\":%.3f"
          ",\"active_dispatch_ns\":%.3f}\n",
          device->physical_device_ordinal, has_clock_fit ? "true" : "false",
          device->clock_sample_count, visible_span_ticks, total_dispatch_ticks,
          visible_span_ticks > 0.0 ? total_dispatch_ticks / visible_span_ticks
                                   : 0.0,
          has_clock_fit ? tick_frequency_hz : 0.0,
          has_clock_fit ? visible_span_ticks * ns_per_tick : 0.0,
          has_clock_fit ? total_dispatch_ticks * ns_per_tick : 0.0);
    }

    for (iree_host_size_t i = 0;
         i < dispatch_context->queue_count && iree_status_is_ok(status); ++i) {
      const iree_hal_profile_queue_record_t* queue =
          &dispatch_context->queues[i].record;
      uint64_t submission_count = 0;
      uint64_t valid_submission_count = 0;
      uint64_t invalid_submission_count = 0;
      double busy_ticks = 0.0;
      double total_dispatch_ticks = 0.0;
      uint64_t gap_count = 0;
      double total_gap_ticks = 0.0;
      double max_gap_ticks = 0.0;
      status = iree_profile_explain_summarize_queue(
          dispatch_context, queue, host_allocator, &submission_count,
          &valid_submission_count, &invalid_submission_count, &busy_ticks,
          &total_dispatch_ticks, &gap_count, &total_gap_ticks, &max_gap_ticks);
      if (iree_status_is_ok(status)) {
        const iree_profile_dispatch_device_t* device =
            iree_profile_dispatch_find_device(dispatch_context,
                                              queue->physical_device_ordinal);
        double ns_per_tick = 0.0;
        double tick_frequency_hz = 0.0;
        const bool has_clock_fit = iree_profile_dispatch_device_try_fit_clock(
            device, &ns_per_tick, &tick_frequency_hz);
        (void)tick_frequency_hz;
        const bool gap_analysis_available =
            dispatch_context->queue_event_count != 0;
        fprintf(file,
                "{\"type\":\"explain_queue\",\"physical_device_ordinal\":%u"
                ",\"queue_ordinal\":%u,\"stream_id\":%" PRIu64
                ",\"submissions\":%" PRIu64 ",\"valid_submissions\":%" PRIu64
                ",\"invalid_submissions\":%" PRIu64
                ",\"busy_ticks\":%.3f,\"total_dispatch_ticks\":%.3f"
                ",\"clock_fit_available\":%s,\"busy_ns\":%.3f"
                ",\"total_dispatch_ns\":%.3f"
                ",\"gap_analysis_available\":%s,\"gaps\":%" PRIu64
                ",\"total_gap_ticks\":%.3f,\"max_gap_ticks\":%.3f"
                ",\"total_gap_ns\":%.3f,\"max_gap_ns\":%.3f}\n",
                queue->physical_device_ordinal, queue->queue_ordinal,
                queue->stream_id, submission_count, valid_submission_count,
                invalid_submission_count, busy_ticks, total_dispatch_ticks,
                has_clock_fit ? "true" : "false",
                has_clock_fit ? busy_ticks * ns_per_tick : 0.0,
                has_clock_fit ? total_dispatch_ticks * ns_per_tick : 0.0,
                gap_analysis_available ? "true" : "false",
                gap_analysis_available ? gap_count : 0,
                gap_analysis_available ? total_gap_ticks : 0.0,
                gap_analysis_available ? max_gap_ticks : 0.0,
                gap_analysis_available && has_clock_fit
                    ? total_gap_ticks * ns_per_tick
                    : 0.0,
                gap_analysis_available && has_clock_fit
                    ? max_gap_ticks * ns_per_tick
                    : 0.0);
      }
    }

    const iree_host_size_t top_export_count =
        iree_min(export_rank_count,
                 (iree_host_size_t)IREE_PROFILE_EXPLAIN_TOP_EXPORT_COUNT);
    for (iree_host_size_t i = 0;
         i < top_export_count && iree_status_is_ok(status); ++i) {
      const iree_profile_explain_export_rank_t* rank = &export_ranks[i];
      char numeric_buffer[128];
      iree_string_view_t key = iree_string_view_empty();
      status = iree_profile_dispatch_resolve_key(
          dispatch_context, rank->physical_device_ordinal, rank->executable_id,
          rank->export_ordinal, numeric_buffer, sizeof(numeric_buffer), &key);
      if (iree_status_is_ok(status)) {
        fprintf(file,
                "{\"type\":\"explain_top_export\",\"rank\":%" PRIhsz
                ",\"physical_device_ordinal\":%u"
                ",\"executable_id\":%" PRIu64
                ",\"export_ordinal\":%u"
                ",\"key\":",
                i + 1, rank->physical_device_ordinal, rank->executable_id,
                rank->export_ordinal);
        iree_profile_fprint_json_string(file, key);
        fprintf(file,
                ",\"dispatches\":%" PRIu64 ",\"valid\":%" PRIu64
                ",\"invalid\":%" PRIu64
                ",\"total_ticks\":%.3f"
                ",\"max_ticks\":%" PRIu64
                ",\"clock_fit_available\":%s"
                ",\"total_ns\":%.3f,\"avg_ns\":%.3f,\"max_ns\":%.3f}\n",
                rank->dispatch_count, rank->valid_count, rank->invalid_count,
                rank->total_ticks, rank->maximum_ticks,
                rank->has_clock_fit ? "true" : "false", rank->total_ns,
                rank->average_ns, rank->maximum_ns);
      }
    }

    iree_profile_dispatch_top_event_t
        top_dispatches[IREE_PROFILE_EXPLAIN_TOP_DISPATCH_COUNT];
    memcpy(top_dispatches, dispatch_context->top_dispatches,
           dispatch_context->top_dispatch_count * sizeof(top_dispatches[0]));
    qsort(top_dispatches, dispatch_context->top_dispatch_count,
          sizeof(top_dispatches[0]), iree_profile_explain_compare_top_event);
    for (iree_host_size_t i = 0;
         i < dispatch_context->top_dispatch_count && iree_status_is_ok(status);
         ++i) {
      const iree_profile_dispatch_top_event_t* top_event = &top_dispatches[i];
      const iree_profile_dispatch_device_t* device =
          iree_profile_dispatch_find_device(dispatch_context,
                                            top_event->physical_device_ordinal);
      double ns_per_tick = 0.0;
      double tick_frequency_hz = 0.0;
      const bool has_clock_fit = iree_profile_dispatch_device_try_fit_clock(
          device, &ns_per_tick, &tick_frequency_hz);
      (void)tick_frequency_hz;
      char numeric_buffer[128];
      iree_string_view_t key = iree_string_view_empty();
      status = iree_profile_dispatch_resolve_key(
          dispatch_context, top_event->physical_device_ordinal,
          top_event->event.executable_id, top_event->event.export_ordinal,
          numeric_buffer, sizeof(numeric_buffer), &key);
      if (iree_status_is_ok(status)) {
        fprintf(file,
                "{\"type\":\"explain_top_dispatch\",\"rank\":%" PRIhsz
                ",\"event_id\":%" PRIu64 ",\"submission_id\":%" PRIu64
                ",\"physical_device_ordinal\":%u,\"queue_ordinal\":%u"
                ",\"stream_id\":%" PRIu64 ",\"key\":",
                i + 1, top_event->event.event_id,
                top_event->event.submission_id,
                top_event->physical_device_ordinal, top_event->queue_ordinal,
                top_event->stream_id);
        iree_profile_fprint_json_string(file, key);
        fprintf(file,
                ",\"duration_ticks\":%" PRIu64
                ",\"clock_fit_available\":%s,\"duration_ns\":%.3f}\n",
                top_event->duration_ticks, has_clock_fit ? "true" : "false",
                has_clock_fit ? (double)top_event->duration_ticks * ns_per_tick
                              : 0.0);
      }
    }

    uint64_t event_counts[IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_HOST_CALL + 1] = {
        0};
    uint64_t payload_bytes[IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_HOST_CALL + 1] = {
        0};
    uint64_t strategy_counts
        [IREE_HAL_PROFILE_QUEUE_DEPENDENCY_STRATEGY_SOFTWARE_DEFER + 1] = {0};
    iree_profile_explain_accumulate_queue_operation_totals(
        dispatch_context, event_counts, payload_bytes, strategy_counts);
    fprintf(file,
            "{\"type\":\"explain_queue_operations\""
            ",\"events\":%" PRIhsz ",\"strategy_none\":%" PRIu64
            ",\"strategy_inline\":%" PRIu64
            ",\"strategy_device_barrier\":%" PRIu64
            ",\"strategy_software_defer\":%" PRIu64 ",\"copy_bytes\":%" PRIu64
            ",\"fill_bytes\":%" PRIu64 ",\"update_bytes\":%" PRIu64
            ",\"read_bytes\":%" PRIu64 ",\"write_bytes\":%" PRIu64
            ",\"dispatches\":%" PRIu64 ",\"executes\":%" PRIu64
            ",\"allocas\":%" PRIu64 ",\"deallocas\":%" PRIu64
            ",\"host_calls\":%" PRIu64 "}\n",
            dispatch_context->queue_event_count,
            strategy_counts[IREE_HAL_PROFILE_QUEUE_DEPENDENCY_STRATEGY_NONE],
            strategy_counts[IREE_HAL_PROFILE_QUEUE_DEPENDENCY_STRATEGY_INLINE],
            strategy_counts
                [IREE_HAL_PROFILE_QUEUE_DEPENDENCY_STRATEGY_DEVICE_BARRIER],
            strategy_counts
                [IREE_HAL_PROFILE_QUEUE_DEPENDENCY_STRATEGY_SOFTWARE_DEFER],
            payload_bytes[IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_COPY],
            payload_bytes[IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_FILL],
            payload_bytes[IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_UPDATE],
            payload_bytes[IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_READ],
            payload_bytes[IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_WRITE],
            event_counts[IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_DISPATCH],
            event_counts[IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_EXECUTE],
            event_counts[IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_ALLOCA],
            event_counts[IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_DEALLOCA],
            event_counts[IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_HOST_CALL]);

    for (iree_host_size_t i = 0; i < memory_context->device_count; ++i) {
      const iree_profile_memory_device_t* device = &memory_context->devices[i];
      fprintf(file,
              "{\"type\":\"explain_memory_pressure\""
              ",\"physical_device_ordinal\":%u"
              ",\"slab_high_water_bytes\":%" PRIu64
              ",\"pool_high_water_bytes\":%" PRIu64
              ",\"queue_high_water_bytes\":%" PRIu64
              ",\"buffer_high_water_bytes\":%" PRIu64 ",\"pool_waits\":%" PRIu64
              "}\n",
              device->physical_device_ordinal, device->high_water_slab_bytes,
              device->high_water_pool_reserved_bytes,
              device->high_water_queue_bytes, device->high_water_buffer_bytes,
              device->pool_wait_count);
    }

    if (summary->unknown_record_count != 0 ||
        summary->unknown_chunk_count != 0) {
      iree_profile_explain_print_hint_jsonl(
          "warning", "format",
          "unknown records or chunks were present; newer producer data may be "
          "missing from this analysis",
          file);
    }
    if (summary->truncated_chunk_count != 0) {
      iree_profile_explain_print_hint_jsonl(
          "error", "format",
          "truncated chunks were present; timing and lifecycle totals may be "
          "incomplete",
          file);
    }
    if (dispatch_context->invalid_dispatch_count != 0) {
      iree_profile_explain_print_hint_jsonl(
          "warning", "dispatch",
          "some dispatch events had missing or reversed timestamps and were "
          "excluded from timing totals",
          file);
    }
    if (dispatch_context->queue_event_count == 0) {
      iree_profile_explain_print_hint_jsonl(
          "info", "queue",
          "queue event records are absent; queue dependency and gap hints are "
          "limited to dispatch timestamp intervals",
          file);
    }
    for (iree_host_size_t i = 0; i < memory_context->device_count; ++i) {
      if (memory_context->devices[i].pool_wait_count != 0) {
        iree_profile_explain_print_hint_jsonl(
            "warning", "memory",
            "pool wait events were observed; allocation readiness affected the "
            "captured queue schedule",
            file);
        break;
      }
    }
    if (dispatch_context->valid_dispatch_count == 0) {
      iree_profile_explain_print_hint_jsonl(
          "info", "dispatch",
          "no valid dispatch events were captured; enable queue profiling on a "
          "producer that emits device timestamps",
          file);
    }
  }

  iree_allocator_free(host_allocator, export_ranks);
  return status;
}

static iree_status_t iree_profile_explain_file(
    iree_string_view_t path, iree_string_view_t format,
    iree_string_view_t filter, int64_t id_filter, FILE* file,
    iree_allocator_t host_allocator) {
  bool is_text = iree_string_view_equal(format, IREE_SV("text"));
  bool is_jsonl = iree_string_view_equal(format, IREE_SV("jsonl"));
  if (!is_text && !is_jsonl) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported profile output format '%.*s'",
                            (int)format.size, format.data);
  }

  iree_io_file_contents_t* file_contents = NULL;
  iree_status_t status = iree_io_file_contents_map(
      path, IREE_IO_FILE_ACCESS_READ, host_allocator, &file_contents);

  iree_hal_profile_file_header_t header;
  iree_host_size_t first_record_offset = 0;
  if (iree_status_is_ok(status)) {
    status = iree_hal_profile_file_parse_header(file_contents->const_buffer,
                                                &header, &first_record_offset);
  }

  iree_profile_summary_t summary;
  iree_profile_summary_initialize(host_allocator, &summary);
  iree_profile_dispatch_context_t dispatch_context;
  iree_profile_dispatch_context_initialize(host_allocator, &dispatch_context);
  iree_profile_memory_context_t memory_context;
  iree_profile_memory_context_initialize(host_allocator, &memory_context);

  iree_host_size_t record_offset = first_record_offset;
  while (iree_status_is_ok(status) &&
         record_offset < file_contents->const_buffer.data_length) {
    iree_hal_profile_file_record_t record;
    iree_host_size_t next_record_offset = 0;
    status = iree_hal_profile_file_parse_record(file_contents->const_buffer,
                                                record_offset, &record,
                                                &next_record_offset);
    if (iree_status_is_ok(status)) {
      status = iree_profile_summary_process_record(&summary, &record);
    }
    if (iree_status_is_ok(status)) {
      status = iree_profile_dispatch_process_metadata_record(&dispatch_context,
                                                             &record);
    }
    if (iree_status_is_ok(status)) {
      record_offset = next_record_offset;
    }
  }

  record_offset = first_record_offset;
  while (iree_status_is_ok(status) &&
         record_offset < file_contents->const_buffer.data_length) {
    iree_hal_profile_file_record_t record;
    iree_host_size_t next_record_offset = 0;
    status = iree_hal_profile_file_parse_record(file_contents->const_buffer,
                                                record_offset, &record,
                                                &next_record_offset);
    if (iree_status_is_ok(status)) {
      status = iree_profile_dispatch_process_events_record(
          &dispatch_context, &record, filter,
          IREE_PROFILE_PROJECTION_MODE_DISPATCH, id_filter,
          /*emit_events=*/false, file);
    }
    if (iree_status_is_ok(status) &&
        record.header.record_type == IREE_HAL_PROFILE_FILE_RECORD_TYPE_CHUNK &&
        iree_string_view_equal(record.content_type,
                               IREE_HAL_PROFILE_CONTENT_TYPE_QUEUE_EVENTS)) {
      status = iree_profile_dispatch_process_queue_event_records(
          &dispatch_context, &record, IREE_SV("*"), /*id_filter=*/-1);
    }
    if (iree_status_is_ok(status)) {
      status = iree_profile_memory_process_event_records(
          &memory_context, &record, IREE_SV("*"), /*id_filter=*/-1,
          /*emit_events=*/false, file);
    }
    if (iree_status_is_ok(status)) {
      record_offset = next_record_offset;
    }
  }

  if (iree_status_is_ok(status)) {
    if (is_text) {
      status = iree_profile_explain_print_text(
          &summary, &dispatch_context, &memory_context, host_allocator, file);
    } else {
      status = iree_profile_explain_print_jsonl(
          &summary, &dispatch_context, &memory_context, host_allocator, file);
    }
  }

  iree_profile_memory_context_deinitialize(&memory_context);
  iree_profile_dispatch_context_deinitialize(&dispatch_context);
  iree_profile_summary_deinitialize(&summary);
  iree_io_file_contents_free(file_contents);
  return status;
}

#define IREE_PROFILE_EXPORT_SCHEMA_VERSION 1

static void iree_profile_export_print_prefix(FILE* file,
                                             const char* record_type,
                                             iree_host_size_t record_index) {
  fprintf(file,
          "{\"schema_version\":%d,\"record_type\":\"%s\""
          ",\"source_record_index\":%" PRIhsz,
          IREE_PROFILE_EXPORT_SCHEMA_VERSION, record_type, record_index);
}

static void iree_profile_export_fprint_hex_bytes(FILE* file,
                                                 const uint8_t* data,
                                                 iree_host_size_t length) {
  static const char kHexDigits[] = "0123456789abcdef";
  for (iree_host_size_t i = 0; i < length; ++i) {
    fputc(kHexDigits[data[i] >> 4], file);
    fputc(kHexDigits[data[i] & 0x0F], file);
  }
}

static void iree_profile_export_fprint_nullable_hash(FILE* file, bool has_hash,
                                                     const uint64_t hash[2]) {
  if (has_hash) {
    fputc('"', file);
    iree_profile_fprint_hash_hex(file, hash);
    fputc('"', file);
  } else {
    fprintf(file, "null");
  }
}

static bool iree_profile_export_try_normalize_tick(
    const iree_profile_dispatch_context_t* context,
    uint32_t physical_device_ordinal, uint64_t tick, double* out_time_ns) {
  *out_time_ns = 0.0;
  if (tick == 0) return false;

  const iree_profile_dispatch_device_t* device =
      iree_profile_dispatch_find_device(context, physical_device_ordinal);
  double ns_per_tick = 0.0;
  double tick_frequency_hz = 0.0;
  if (!iree_profile_dispatch_device_try_fit_clock(device, &ns_per_tick,
                                                  &tick_frequency_hz)) {
    return false;
  }
  (void)tick_frequency_hz;

  const double tick_delta =
      (double)tick - (double)device->first_clock_sample.device_tick;
  *out_time_ns = (double)device->first_clock_sample.host_cpu_timestamp_ns +
                 tick_delta * ns_per_tick;
  return true;
}

static void iree_profile_export_print_session_record(
    const iree_hal_profile_file_record_t* record, iree_host_size_t record_index,
    FILE* file) {
  const char* event_name =
      record->header.record_type ==
              IREE_HAL_PROFILE_FILE_RECORD_TYPE_SESSION_BEGIN
          ? "begin"
          : "end";
  iree_profile_export_print_prefix(file, "session", record_index);
  fprintf(file,
          ",\"event\":\"%s\",\"session_id\":%" PRIu64 ",\"stream_id\":%" PRIu64
          ",\"event_id\":%" PRIu64 ",\"session_status_code\":%u}\n",
          event_name, record->header.session_id, record->header.stream_id,
          record->header.event_id, record->header.session_status_code);
}

static void iree_profile_export_print_diagnostic(
    iree_host_size_t record_index, const char* severity, const char* category,
    iree_string_view_t content_type, const char* message, FILE* file) {
  iree_profile_export_print_prefix(file, "diagnostic", record_index);
  fprintf(file, ",\"severity\":");
  iree_profile_fprint_json_string(file, iree_make_cstring_view(severity));
  fprintf(file, ",\"category\":");
  iree_profile_fprint_json_string(file, iree_make_cstring_view(category));
  fprintf(file, ",\"content_type\":");
  iree_profile_fprint_json_string(file, content_type);
  fprintf(file, ",\"message\":");
  iree_profile_fprint_json_string(file, iree_make_cstring_view(message));
  fputs("}\n", file);
}

static iree_status_t iree_profile_export_process_device_records(
    const iree_hal_profile_file_record_t* record, iree_host_size_t record_index,
    FILE* file) {
  iree_host_size_t payload_offset = 0;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         payload_offset < record->payload.data_length) {
    iree_host_size_t record_length = 0;
    status = iree_profile_payload_record_length(
        record->content_type, record->payload, payload_offset,
        sizeof(iree_hal_profile_device_record_t), &record_length);
    if (iree_status_is_ok(status)) {
      iree_hal_profile_device_record_t device_record;
      memcpy(&device_record, record->payload.data + payload_offset,
             sizeof(device_record));
      const bool has_uuid =
          iree_all_bits_set(device_record.flags,
                            IREE_HAL_PROFILE_DEVICE_FLAG_PHYSICAL_DEVICE_UUID);
      iree_profile_export_print_prefix(file, "device", record_index);
      fprintf(file,
              ",\"physical_device_ordinal\":%u,\"flags\":%u"
              ",\"queue_count\":%u,\"physical_device_uuid_present\":%s"
              ",\"physical_device_uuid\":",
              device_record.physical_device_ordinal, device_record.flags,
              device_record.queue_count, has_uuid ? "true" : "false");
      if (has_uuid) {
        fputc('"', file);
        iree_profile_export_fprint_hex_bytes(
            file, device_record.physical_device_uuid,
            sizeof(device_record.physical_device_uuid));
        fputc('"', file);
      } else {
        fprintf(file, "null");
      }
      fputs("}\n", file);
      payload_offset += record_length;
    }
  }
  return status;
}

static iree_status_t iree_profile_export_process_queue_records(
    const iree_hal_profile_file_record_t* record, iree_host_size_t record_index,
    FILE* file) {
  iree_host_size_t payload_offset = 0;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         payload_offset < record->payload.data_length) {
    iree_host_size_t record_length = 0;
    status = iree_profile_payload_record_length(
        record->content_type, record->payload, payload_offset,
        sizeof(iree_hal_profile_queue_record_t), &record_length);
    if (iree_status_is_ok(status)) {
      iree_hal_profile_queue_record_t queue_record;
      memcpy(&queue_record, record->payload.data + payload_offset,
             sizeof(queue_record));
      iree_profile_export_print_prefix(file, "queue", record_index);
      fprintf(file,
              ",\"physical_device_ordinal\":%u,\"queue_ordinal\":%u"
              ",\"stream_id\":%" PRIu64 "}\n",
              queue_record.physical_device_ordinal, queue_record.queue_ordinal,
              queue_record.stream_id);
      payload_offset += record_length;
    }
  }
  return status;
}

static iree_status_t iree_profile_export_process_executable_records(
    const iree_hal_profile_file_record_t* record, iree_host_size_t record_index,
    FILE* file) {
  iree_host_size_t payload_offset = 0;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         payload_offset < record->payload.data_length) {
    iree_host_size_t record_length = 0;
    status = iree_profile_payload_record_length(
        record->content_type, record->payload, payload_offset,
        sizeof(iree_hal_profile_executable_record_t), &record_length);
    if (iree_status_is_ok(status)) {
      iree_hal_profile_executable_record_t executable_record;
      memcpy(&executable_record, record->payload.data + payload_offset,
             sizeof(executable_record));
      const bool has_code_object_hash =
          iree_all_bits_set(executable_record.flags,
                            IREE_HAL_PROFILE_EXECUTABLE_FLAG_CODE_OBJECT_HASH);
      iree_profile_export_print_prefix(file, "executable", record_index);
      fprintf(file,
              ",\"executable_id\":%" PRIu64
              ",\"flags\":%u"
              ",\"export_count\":%u,\"code_object_hash_present\":%s"
              ",\"code_object_hash\":",
              executable_record.executable_id, executable_record.flags,
              executable_record.export_count,
              has_code_object_hash ? "true" : "false");
      iree_profile_export_fprint_nullable_hash(
          file, has_code_object_hash, executable_record.code_object_hash);
      fputs("}\n", file);
      payload_offset += record_length;
    }
  }
  return status;
}

static iree_status_t iree_profile_export_process_executable_export_records(
    const iree_hal_profile_file_record_t* record, iree_host_size_t record_index,
    FILE* file) {
  iree_host_size_t payload_offset = 0;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         payload_offset < record->payload.data_length) {
    iree_host_size_t record_length = 0;
    status = iree_profile_payload_record_length(
        record->content_type, record->payload, payload_offset,
        sizeof(iree_hal_profile_executable_export_record_t), &record_length);
    if (iree_status_is_ok(status)) {
      iree_hal_profile_executable_export_record_t export_record;
      memcpy(&export_record, record->payload.data + payload_offset,
             sizeof(export_record));
      if (export_record.name_length != record_length - sizeof(export_record)) {
        status =
            iree_make_status(IREE_STATUS_DATA_LOSS,
                             "executable export name length is inconsistent");
      }
      if (iree_status_is_ok(status)) {
        const bool has_pipeline_hash = iree_all_bits_set(
            export_record.flags,
            IREE_HAL_PROFILE_EXECUTABLE_EXPORT_FLAG_PIPELINE_HASH);
        iree_string_view_t name =
            iree_make_string_view((const char*)record->payload.data +
                                      payload_offset + sizeof(export_record),
                                  export_record.name_length);
        iree_profile_export_print_prefix(file, "executable_export",
                                         record_index);
        fprintf(file,
                ",\"executable_id\":%" PRIu64
                ",\"export_ordinal\":%u"
                ",\"flags\":%u,\"name\":",
                export_record.executable_id, export_record.export_ordinal,
                export_record.flags);
        iree_profile_fprint_json_string(file, name);
        fprintf(file,
                ",\"constant_count\":%u,\"binding_count\":%u"
                ",\"parameter_count\":%u,\"workgroup_size\":[%u,%u,%u]"
                ",\"pipeline_hash_present\":%s,\"pipeline_hash\":",
                export_record.constant_count, export_record.binding_count,
                export_record.parameter_count, export_record.workgroup_size[0],
                export_record.workgroup_size[1],
                export_record.workgroup_size[2],
                has_pipeline_hash ? "true" : "false");
        iree_profile_export_fprint_nullable_hash(file, has_pipeline_hash,
                                                 export_record.pipeline_hash);
        fputs("}\n", file);
      }
      payload_offset += record_length;
    }
  }
  return status;
}

static iree_status_t iree_profile_export_process_command_buffer_records(
    const iree_hal_profile_file_record_t* record, iree_host_size_t record_index,
    FILE* file) {
  iree_host_size_t payload_offset = 0;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         payload_offset < record->payload.data_length) {
    iree_host_size_t record_length = 0;
    status = iree_profile_payload_record_length(
        record->content_type, record->payload, payload_offset,
        sizeof(iree_hal_profile_command_buffer_record_t), &record_length);
    if (iree_status_is_ok(status)) {
      iree_hal_profile_command_buffer_record_t command_buffer_record;
      memcpy(&command_buffer_record, record->payload.data + payload_offset,
             sizeof(command_buffer_record));
      iree_profile_export_print_prefix(file, "command_buffer", record_index);
      fprintf(
          file,
          ",\"command_buffer_id\":%" PRIu64
          ",\"flags\":%u"
          ",\"physical_device_ordinal\":%u,\"mode\":%" PRIu64
          ",\"command_categories\":%" PRIu64 ",\"queue_affinity\":%" PRIu64
          "}\n",
          command_buffer_record.command_buffer_id, command_buffer_record.flags,
          command_buffer_record.physical_device_ordinal,
          command_buffer_record.mode, command_buffer_record.command_categories,
          command_buffer_record.queue_affinity);
      payload_offset += record_length;
    }
  }
  return status;
}

static iree_status_t iree_profile_export_process_command_operation_records(
    const iree_profile_dispatch_context_t* context,
    const iree_hal_profile_file_record_t* record, iree_host_size_t record_index,
    FILE* file) {
  iree_host_size_t payload_offset = 0;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         payload_offset < record->payload.data_length) {
    iree_host_size_t record_length = 0;
    status = iree_profile_payload_record_length(
        record->content_type, record->payload, payload_offset,
        sizeof(iree_hal_profile_command_operation_record_t), &record_length);
    if (iree_status_is_ok(status)) {
      iree_hal_profile_command_operation_record_t operation_record;
      memcpy(&operation_record, record->payload.data + payload_offset,
             sizeof(operation_record));
      const char* operation_name =
          iree_profile_command_operation_type_name(operation_record.type);
      char numeric_buffer[128];
      iree_string_view_t key = iree_string_view_empty();
      status = iree_profile_command_operation_resolve_key(
          context, &operation_record, numeric_buffer, sizeof(numeric_buffer),
          &key);
      if (iree_status_is_ok(status)) {
        iree_profile_export_print_prefix(file, "command_operation",
                                         record_index);
        fprintf(file,
                ",\"command_buffer_id\":%" PRIu64
                ",\"command_index\":%u"
                ",\"op\":",
                operation_record.command_buffer_id,
                operation_record.command_index);
        iree_profile_fprint_json_string(file,
                                        iree_make_cstring_view(operation_name));
        fprintf(file, ",\"key\":");
        iree_profile_fprint_json_string(file, key);
        fprintf(
            file,
            ",\"flags\":%u,\"block_ordinal\":%u"
            ",\"block_command_ordinal\":%u"
            ",\"executable_id\":%" PRIu64
            ",\"export_ordinal\":%u"
            ",\"binding_count\":%u,\"workgroup_count\":[%u,%u,%u]"
            ",\"workgroup_size\":[%u,%u,%u]"
            ",\"source_ordinal\":%u,\"target_ordinal\":%u"
            ",\"source_offset\":%" PRIu64 ",\"target_offset\":%" PRIu64
            ",\"length\":%" PRIu64
            ",\"target_block_ordinal\":%u"
            ",\"alternate_block_ordinal\":%u}\n",
            operation_record.flags, operation_record.block_ordinal,
            operation_record.block_command_ordinal,
            operation_record.executable_id, operation_record.export_ordinal,
            operation_record.binding_count, operation_record.workgroup_count[0],
            operation_record.workgroup_count[1],
            operation_record.workgroup_count[2],
            operation_record.workgroup_size[0],
            operation_record.workgroup_size[1],
            operation_record.workgroup_size[2], operation_record.source_ordinal,
            operation_record.target_ordinal, operation_record.source_offset,
            operation_record.target_offset, operation_record.length,
            operation_record.target_block_ordinal,
            operation_record.alternate_block_ordinal);
        payload_offset += record_length;
      }
    }
  }
  return status;
}

static iree_status_t iree_profile_export_process_clock_records(
    const iree_hal_profile_file_record_t* record, iree_host_size_t record_index,
    FILE* file) {
  iree_host_size_t payload_offset = 0;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         payload_offset < record->payload.data_length) {
    iree_host_size_t record_length = 0;
    status = iree_profile_payload_record_length(
        record->content_type, record->payload, payload_offset,
        sizeof(iree_hal_profile_clock_correlation_record_t), &record_length);
    if (iree_status_is_ok(status)) {
      iree_hal_profile_clock_correlation_record_t clock_record;
      memcpy(&clock_record, record->payload.data + payload_offset,
             sizeof(clock_record));
      iree_profile_export_print_prefix(file, "clock_correlation", record_index);
      fprintf(
          file,
          ",\"sample_id\":%" PRIu64
          ",\"flags\":%u"
          ",\"physical_device_ordinal\":%u"
          ",\"device_tick\":%" PRIu64 ",\"host_cpu_timestamp_ns\":%" PRIu64
          ",\"host_system_timestamp\":%" PRIu64
          ",\"host_system_frequency_hz\":%" PRIu64
          ",\"host_time_begin_ns\":%" PRId64 ",\"host_time_end_ns\":%" PRId64
          ",\"host_time_uncertainty_ns\":%" PRId64 "}\n",
          clock_record.sample_id, clock_record.flags,
          clock_record.physical_device_ordinal, clock_record.device_tick,
          clock_record.host_cpu_timestamp_ns,
          clock_record.host_system_timestamp,
          clock_record.host_system_frequency_hz,
          clock_record.host_time_begin_ns, clock_record.host_time_end_ns,
          clock_record.host_time_end_ns >= clock_record.host_time_begin_ns
              ? clock_record.host_time_end_ns - clock_record.host_time_begin_ns
              : 0);
      payload_offset += record_length;
    }
  }
  return status;
}

static iree_status_t iree_profile_export_process_dispatch_records(
    const iree_profile_dispatch_context_t* context,
    const iree_hal_profile_file_record_t* record, iree_host_size_t record_index,
    FILE* file) {
  iree_host_size_t payload_offset = 0;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         payload_offset < record->payload.data_length) {
    iree_host_size_t record_length = 0;
    status = iree_profile_payload_record_length(
        record->content_type, record->payload, payload_offset,
        sizeof(iree_hal_profile_dispatch_event_t), &record_length);
    if (iree_status_is_ok(status)) {
      iree_hal_profile_dispatch_event_t dispatch_record;
      memcpy(&dispatch_record, record->payload.data + payload_offset,
             sizeof(dispatch_record));
      const bool is_valid =
          dispatch_record.start_tick != 0 && dispatch_record.end_tick != 0 &&
          dispatch_record.end_tick >= dispatch_record.start_tick;
      const uint64_t duration_ticks =
          is_valid ? dispatch_record.end_tick - dispatch_record.start_tick : 0;
      double start_ns = 0.0;
      double end_ns = 0.0;
      const bool has_normalized_start = iree_profile_export_try_normalize_tick(
          context, record->header.physical_device_ordinal,
          dispatch_record.start_tick, &start_ns);
      const bool has_normalized_end = iree_profile_export_try_normalize_tick(
          context, record->header.physical_device_ordinal,
          dispatch_record.end_tick, &end_ns);
      const bool has_normalized_time =
          is_valid && has_normalized_start && has_normalized_end;
      char numeric_buffer[128];
      iree_string_view_t key = iree_string_view_empty();
      status = iree_profile_dispatch_resolve_key(
          context, record->header.physical_device_ordinal,
          dispatch_record.executable_id, dispatch_record.export_ordinal,
          numeric_buffer, sizeof(numeric_buffer), &key);
      if (iree_status_is_ok(status)) {
        iree_profile_export_print_prefix(file, "dispatch_event", record_index);
        fprintf(file,
                ",\"physical_device_ordinal\":%u,\"queue_ordinal\":%u"
                ",\"stream_id\":%" PRIu64 ",\"event_id\":%" PRIu64
                ",\"submission_id\":%" PRIu64 ",\"command_buffer_id\":%" PRIu64
                ",\"command_index\":%u,\"executable_id\":%" PRIu64
                ",\"export_ordinal\":%u,\"key\":",
                record->header.physical_device_ordinal,
                record->header.queue_ordinal, record->header.stream_id,
                dispatch_record.event_id, dispatch_record.submission_id,
                dispatch_record.command_buffer_id,
                dispatch_record.command_index, dispatch_record.executable_id,
                dispatch_record.export_ordinal);
        iree_profile_fprint_json_string(file, key);
        fprintf(file,
                ",\"flags\":%u,\"workgroup_count\":[%u,%u,%u]"
                ",\"workgroup_size\":[%u,%u,%u]"
                ",\"start_tick\":%" PRIu64 ",\"end_tick\":%" PRIu64
                ",\"duration_ticks\":%" PRIu64
                ",\"valid\":%s"
                ",\"normalized_time_available\":%s"
                ",\"start_ns\":%.3f,\"end_ns\":%.3f"
                ",\"duration_ns\":%.3f}\n",
                dispatch_record.flags, dispatch_record.workgroup_count[0],
                dispatch_record.workgroup_count[1],
                dispatch_record.workgroup_count[2],
                dispatch_record.workgroup_size[0],
                dispatch_record.workgroup_size[1],
                dispatch_record.workgroup_size[2], dispatch_record.start_tick,
                dispatch_record.end_tick, duration_ticks,
                is_valid ? "true" : "false",
                has_normalized_time ? "true" : "false",
                has_normalized_time ? start_ns : 0.0,
                has_normalized_time ? end_ns : 0.0,
                has_normalized_time ? end_ns - start_ns : 0.0);
        payload_offset += record_length;
      }
    }
  }
  return status;
}

static iree_status_t iree_profile_export_process_queue_event_records(
    const iree_hal_profile_file_record_t* record, iree_host_size_t record_index,
    FILE* file) {
  iree_host_size_t payload_offset = 0;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         payload_offset < record->payload.data_length) {
    iree_host_size_t record_length = 0;
    status = iree_profile_payload_record_length(
        record->content_type, record->payload, payload_offset,
        sizeof(iree_hal_profile_queue_event_t), &record_length);
    if (iree_status_is_ok(status)) {
      iree_hal_profile_queue_event_t queue_event;
      memcpy(&queue_event, record->payload.data + payload_offset,
             sizeof(queue_event));
      iree_profile_export_print_prefix(file, "queue_event", record_index);
      fprintf(file, ",\"event_id\":%" PRIu64 ",\"op\":", queue_event.event_id);
      iree_profile_fprint_json_string(
          file, iree_make_cstring_view(
                    iree_profile_queue_event_type_name(queue_event.type)));
      fprintf(file, ",\"type_value\":%u,\"flags\":%u,\"dependency_strategy\":",
              queue_event.type, queue_event.flags);
      iree_profile_fprint_json_string(
          file,
          iree_make_cstring_view(iree_profile_queue_dependency_strategy_name(
              queue_event.dependency_strategy)));
      fprintf(file,
              ",\"submission_id\":%" PRIu64 ",\"command_buffer_id\":%" PRIu64
              ",\"allocation_id\":%" PRIu64
              ",\"physical_device_ordinal\":%u,\"queue_ordinal\":%u"
              ",\"stream_id\":%" PRIu64 ",\"host_time_ns\":%" PRId64
              ",\"wait_count\":%u,\"signal_count\":%u"
              ",\"barrier_count\":%u,\"operation_count\":%u"
              ",\"payload_length\":%" PRIu64 "}\n",
              queue_event.submission_id, queue_event.command_buffer_id,
              queue_event.allocation_id, queue_event.physical_device_ordinal,
              queue_event.queue_ordinal, queue_event.stream_id,
              queue_event.host_time_ns, queue_event.wait_count,
              queue_event.signal_count, queue_event.barrier_count,
              queue_event.operation_count, queue_event.payload_length);
      payload_offset += record_length;
    }
  }
  return status;
}

static iree_status_t iree_profile_export_process_memory_event_records(
    const iree_hal_profile_file_record_t* record, iree_host_size_t record_index,
    FILE* file) {
  iree_host_size_t payload_offset = 0;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         payload_offset < record->payload.data_length) {
    iree_host_size_t record_length = 0;
    status = iree_profile_payload_record_length(
        record->content_type, record->payload, payload_offset,
        sizeof(iree_hal_profile_memory_event_t), &record_length);
    if (iree_status_is_ok(status)) {
      iree_hal_profile_memory_event_t memory_event;
      memcpy(&memory_event, record->payload.data + payload_offset,
             sizeof(memory_event));
      iree_profile_export_print_prefix(file, "memory_event", record_index);
      fprintf(file, ",\"event_id\":%" PRIu64 ",\"event_type\":",
              memory_event.event_id);
      iree_profile_fprint_json_string(
          file, iree_make_cstring_view(
                    iree_profile_memory_event_type_name(memory_event.type)));
      fprintf(file,
              ",\"event_type_value\":%u,\"flags\":%u,\"result\":%u"
              ",\"host_time_ns\":%" PRId64 ",\"allocation_id\":%" PRIu64
              ",\"pool_id\":%" PRIu64 ",\"backing_id\":%" PRIu64
              ",\"submission_id\":%" PRIu64
              ",\"physical_device_ordinal\":%u,\"queue_ordinal\":%u"
              ",\"frontier_entry_count\":%u,\"memory_type\":%" PRIu64
              ",\"buffer_usage\":%" PRIu64 ",\"offset\":%" PRIu64
              ",\"length\":%" PRIu64 ",\"alignment\":%" PRIu64 "}\n",
              memory_event.type, memory_event.flags, memory_event.result,
              memory_event.host_time_ns, memory_event.allocation_id,
              memory_event.pool_id, memory_event.backing_id,
              memory_event.submission_id, memory_event.physical_device_ordinal,
              memory_event.queue_ordinal, memory_event.frontier_entry_count,
              memory_event.memory_type, memory_event.buffer_usage,
              memory_event.offset, memory_event.length, memory_event.alignment);
      payload_offset += record_length;
    }
  }
  return status;
}

static iree_status_t iree_profile_export_process_counter_set_records(
    const iree_hal_profile_file_record_t* record, iree_host_size_t record_index,
    FILE* file) {
  iree_host_size_t payload_offset = 0;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         payload_offset < record->payload.data_length) {
    iree_host_size_t record_length = 0;
    status = iree_profile_payload_record_length(
        record->content_type, record->payload, payload_offset,
        sizeof(iree_hal_profile_counter_set_record_t), &record_length);
    if (iree_status_is_ok(status)) {
      iree_hal_profile_counter_set_record_t counter_set_record;
      memcpy(&counter_set_record, record->payload.data + payload_offset,
             sizeof(counter_set_record));
      if (counter_set_record.name_length !=
          record_length - sizeof(counter_set_record)) {
        status = iree_make_status(
            IREE_STATUS_DATA_LOSS,
            "counter set name length is inconsistent with record length");
      }
      if (iree_status_is_ok(status)) {
        iree_string_view_t name = iree_make_string_view(
            (const char*)record->payload.data + payload_offset +
                sizeof(counter_set_record),
            counter_set_record.name_length);
        iree_profile_export_print_prefix(file, "counter_set", record_index);
        fprintf(file,
                ",\"counter_set_id\":%" PRIu64
                ",\"physical_device_ordinal\":%u"
                ",\"flags\":%u,\"counter_count\":%u"
                ",\"sample_value_count\":%u,\"name\":",
                counter_set_record.counter_set_id,
                counter_set_record.physical_device_ordinal,
                counter_set_record.flags, counter_set_record.counter_count,
                counter_set_record.sample_value_count);
        iree_profile_fprint_json_string(file, name);
        fputs("}\n", file);
        payload_offset += record_length;
      }
    }
  }
  return status;
}

static iree_status_t iree_profile_export_process_counter_records(
    const iree_hal_profile_file_record_t* record, iree_host_size_t record_index,
    FILE* file) {
  iree_host_size_t payload_offset = 0;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         payload_offset < record->payload.data_length) {
    iree_host_size_t record_length = 0;
    status = iree_profile_payload_record_length(
        record->content_type, record->payload, payload_offset,
        sizeof(iree_hal_profile_counter_record_t), &record_length);
    if (iree_status_is_ok(status)) {
      iree_hal_profile_counter_record_t counter_record;
      memcpy(&counter_record, record->payload.data + payload_offset,
             sizeof(counter_record));
      iree_host_size_t trailing_length = 0;
      if (!iree_host_size_checked_add(counter_record.block_name_length,
                                      counter_record.name_length,
                                      &trailing_length) ||
          !iree_host_size_checked_add(trailing_length,
                                      counter_record.description_length,
                                      &trailing_length) ||
          trailing_length != record_length - sizeof(counter_record)) {
        status = iree_make_status(
            IREE_STATUS_DATA_LOSS,
            "counter string lengths are inconsistent with record length");
      }
      if (iree_status_is_ok(status)) {
        const char* string_base = (const char*)record->payload.data +
                                  payload_offset + sizeof(counter_record);
        iree_string_view_t block_name = iree_make_string_view(
            string_base, counter_record.block_name_length);
        iree_string_view_t name = iree_make_string_view(
            string_base + counter_record.block_name_length,
            counter_record.name_length);
        iree_string_view_t description = iree_make_string_view(
            string_base + counter_record.block_name_length +
                counter_record.name_length,
            counter_record.description_length);
        iree_profile_export_print_prefix(file, "counter", record_index);
        fprintf(file,
                ",\"counter_set_id\":%" PRIu64
                ",\"counter_ordinal\":%u"
                ",\"physical_device_ordinal\":%u,\"flags\":%u"
                ",\"unit\":%u,\"sample_value_offset\":%u"
                ",\"sample_value_count\":%u,\"block\":",
                counter_record.counter_set_id, counter_record.counter_ordinal,
                counter_record.physical_device_ordinal, counter_record.flags,
                counter_record.unit, counter_record.sample_value_offset,
                counter_record.sample_value_count);
        iree_profile_fprint_json_string(file, block_name);
        fprintf(file, ",\"name\":");
        iree_profile_fprint_json_string(file, name);
        fprintf(file, ",\"description\":");
        iree_profile_fprint_json_string(file, description);
        fputs("}\n", file);
        payload_offset += record_length;
      }
    }
  }
  return status;
}

static iree_status_t iree_profile_export_process_counter_sample_records(
    const iree_hal_profile_file_record_t* record, iree_host_size_t record_index,
    FILE* file) {
  iree_host_size_t payload_offset = 0;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         payload_offset < record->payload.data_length) {
    iree_host_size_t record_length = 0;
    status = iree_profile_payload_record_length(
        record->content_type, record->payload, payload_offset,
        sizeof(iree_hal_profile_counter_sample_record_t), &record_length);
    if (iree_status_is_ok(status)) {
      iree_hal_profile_counter_sample_record_t sample_record;
      memcpy(&sample_record, record->payload.data + payload_offset,
             sizeof(sample_record));
      iree_host_size_t values_length = 0;
      if (!iree_host_size_checked_mul(sample_record.sample_value_count,
                                      sizeof(uint64_t), &values_length) ||
          values_length != record_length - sizeof(sample_record)) {
        status = iree_make_status(
            IREE_STATUS_DATA_LOSS,
            "counter sample value count is inconsistent with record length");
      }
      if (iree_status_is_ok(status)) {
        const uint64_t* values =
            (const uint64_t*)(record->payload.data + payload_offset +
                              sizeof(sample_record));
        iree_profile_export_print_prefix(file, "counter_sample", record_index);
        fprintf(file,
                ",\"sample_id\":%" PRIu64 ",\"counter_set_id\":%" PRIu64
                ",\"dispatch_event_id\":%" PRIu64 ",\"submission_id\":%" PRIu64
                ",\"command_buffer_id\":%" PRIu64
                ",\"command_index\":%u,\"executable_id\":%" PRIu64
                ",\"export_ordinal\":%u"
                ",\"physical_device_ordinal\":%u,\"queue_ordinal\":%u"
                ",\"stream_id\":%" PRIu64 ",\"flags\":%u,\"values\":[",
                sample_record.sample_id, sample_record.counter_set_id,
                sample_record.dispatch_event_id, sample_record.submission_id,
                sample_record.command_buffer_id, sample_record.command_index,
                sample_record.executable_id, sample_record.export_ordinal,
                sample_record.physical_device_ordinal,
                sample_record.queue_ordinal, sample_record.stream_id,
                sample_record.flags);
        for (uint32_t i = 0; i < sample_record.sample_value_count; ++i) {
          if (i != 0) fputc(',', file);
          fprintf(file, "%" PRIu64, values[i]);
        }
        fputs("]}\n", file);
        payload_offset += record_length;
      }
    }
  }
  return status;
}

static iree_status_t iree_profile_export_process_decoded_record(
    const iree_profile_dispatch_context_t* context,
    const iree_hal_profile_file_record_t* record, iree_host_size_t record_index,
    FILE* file) {
  if (record->header.record_type ==
          IREE_HAL_PROFILE_FILE_RECORD_TYPE_SESSION_BEGIN ||
      record->header.record_type ==
          IREE_HAL_PROFILE_FILE_RECORD_TYPE_SESSION_END) {
    iree_profile_export_print_session_record(record, record_index, file);
    return iree_ok_status();
  }
  if (record->header.record_type != IREE_HAL_PROFILE_FILE_RECORD_TYPE_CHUNK) {
    iree_profile_export_print_diagnostic(
        record_index, "warning", "unknown_record", record->content_type,
        "unknown profile file record type", file);
    return iree_ok_status();
  }

  if (iree_any_bit_set(record->header.chunk_flags,
                       IREE_HAL_PROFILE_CHUNK_FLAG_TRUNCATED)) {
    iree_profile_export_print_diagnostic(
        record_index, "error", "truncated_chunk", record->content_type,
        "chunk was marked truncated by the producer", file);
  }

  if (iree_string_view_equal(record->content_type,
                             IREE_HAL_PROFILE_CONTENT_TYPE_DEVICES)) {
    return iree_profile_export_process_device_records(record, record_index,
                                                      file);
  } else if (iree_string_view_equal(record->content_type,
                                    IREE_HAL_PROFILE_CONTENT_TYPE_QUEUES)) {
    return iree_profile_export_process_queue_records(record, record_index,
                                                     file);
  } else if (iree_string_view_equal(
                 record->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLES)) {
    return iree_profile_export_process_executable_records(record, record_index,
                                                          file);
  } else if (iree_string_view_equal(
                 record->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLE_EXPORTS)) {
    return iree_profile_export_process_executable_export_records(
        record, record_index, file);
  } else if (iree_string_view_equal(
                 record->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_COMMAND_BUFFERS)) {
    return iree_profile_export_process_command_buffer_records(
        record, record_index, file);
  } else if (iree_string_view_equal(
                 record->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_COMMAND_OPERATIONS)) {
    return iree_profile_export_process_command_operation_records(
        context, record, record_index, file);
  } else if (iree_string_view_equal(
                 record->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_CLOCK_CORRELATIONS)) {
    return iree_profile_export_process_clock_records(record, record_index,
                                                     file);
  } else if (iree_string_view_equal(
                 record->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_DISPATCH_EVENTS)) {
    return iree_profile_export_process_dispatch_records(context, record,
                                                        record_index, file);
  } else if (iree_string_view_equal(
                 record->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_QUEUE_EVENTS)) {
    return iree_profile_export_process_queue_event_records(record, record_index,
                                                           file);
  } else if (iree_string_view_equal(
                 record->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_MEMORY_EVENTS)) {
    return iree_profile_export_process_memory_event_records(record,
                                                            record_index, file);
  } else if (iree_string_view_equal(
                 record->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_COUNTER_SETS)) {
    return iree_profile_export_process_counter_set_records(record, record_index,
                                                           file);
  } else if (iree_string_view_equal(record->content_type,
                                    IREE_HAL_PROFILE_CONTENT_TYPE_COUNTERS)) {
    return iree_profile_export_process_counter_records(record, record_index,
                                                       file);
  } else if (iree_string_view_equal(
                 record->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_COUNTER_SAMPLES)) {
    return iree_profile_export_process_counter_sample_records(
        record, record_index, file);
  }

  iree_profile_export_print_diagnostic(
      record_index, "warning", "unknown_chunk", record->content_type,
      "unknown profile chunk content type", file);
  return iree_ok_status();
}

static iree_status_t iree_profile_export_ireeperf_jsonl_file(
    iree_string_view_t path, FILE* file, iree_allocator_t host_allocator) {
  iree_io_file_contents_t* file_contents = NULL;
  iree_status_t status = iree_io_file_contents_map(
      path, IREE_IO_FILE_ACCESS_READ, host_allocator, &file_contents);

  iree_hal_profile_file_header_t header;
  iree_host_size_t first_record_offset = 0;
  if (iree_status_is_ok(status)) {
    status = iree_hal_profile_file_parse_header(file_contents->const_buffer,
                                                &header, &first_record_offset);
  }

  iree_profile_dispatch_context_t context;
  iree_profile_dispatch_context_initialize(host_allocator, &context);
  iree_host_size_t record_offset = first_record_offset;
  while (iree_status_is_ok(status) &&
         record_offset < file_contents->const_buffer.data_length) {
    iree_hal_profile_file_record_t record;
    iree_host_size_t next_record_offset = 0;
    status = iree_hal_profile_file_parse_record(file_contents->const_buffer,
                                                record_offset, &record,
                                                &next_record_offset);
    if (iree_status_is_ok(status)) {
      status = iree_profile_dispatch_process_metadata_record(&context, &record);
      record_offset = next_record_offset;
    }
  }

  if (iree_status_is_ok(status)) {
    fprintf(file,
            "{\"schema_version\":%d,\"record_type\":\"schema\""
            ",\"format\":\"ireeperf-jsonl\""
            ",\"source_format\":\"ireeprof\""
            ",\"source_version_major\":%u,\"source_version_minor\":%u}\n",
            IREE_PROFILE_EXPORT_SCHEMA_VERSION, header.version_major,
            header.version_minor);
  }

  record_offset = first_record_offset;
  iree_host_size_t record_index = 0;
  while (iree_status_is_ok(status) &&
         record_offset < file_contents->const_buffer.data_length) {
    iree_hal_profile_file_record_t record;
    iree_host_size_t next_record_offset = 0;
    status = iree_hal_profile_file_parse_record(file_contents->const_buffer,
                                                record_offset, &record,
                                                &next_record_offset);
    if (iree_status_is_ok(status)) {
      status = iree_profile_export_process_decoded_record(&context, &record,
                                                          record_index, file);
      record_offset = next_record_offset;
      ++record_index;
    }
  }

  iree_profile_dispatch_context_deinitialize(&context);
  iree_io_file_contents_free(file_contents);
  return status;
}

static iree_status_t iree_profile_export_file(iree_string_view_t path,
                                              iree_string_view_t format,
                                              iree_string_view_t output_path,
                                              iree_allocator_t host_allocator) {
  if (!iree_string_view_equal(format, IREE_SV("ireeperf-jsonl"))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported profile export format '%.*s'",
                            (int)format.size, format.data);
  }

  FILE* file = stdout;
  bool should_close_file = false;
  if (!iree_string_view_equal(output_path, IREE_SV("-"))) {
    char* c_path = NULL;
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(
        host_allocator, output_path.size + 1, (void**)&c_path));
    iree_string_view_to_cstring(output_path, c_path, output_path.size + 1);
    file = fopen(c_path, "wb");
    const int open_errno = errno;
    iree_allocator_free(host_allocator, c_path);
    if (!file) {
      return iree_make_status(IREE_STATUS_UNAVAILABLE,
                              "failed to open export output file: %s",
                              strerror(open_errno));
    }
    should_close_file = true;
  }

  iree_status_t status =
      iree_profile_export_ireeperf_jsonl_file(path, file, host_allocator);
  if (should_close_file) {
    if (fclose(file) != 0 && iree_status_is_ok(status)) {
      status = iree_make_status(IREE_STATUS_UNAVAILABLE,
                                "failed to close export output file: %s",
                                strerror(errno));
    }
  }
  return status;
}

static iree_status_t iree_profile_projection_file(
    iree_string_view_t path, iree_string_view_t format,
    iree_string_view_t filter, iree_profile_projection_mode_t projection_mode,
    int64_t id_filter, bool emit_events, FILE* file,
    iree_allocator_t host_allocator) {
  bool is_text = iree_string_view_equal(format, IREE_SV("text"));
  bool is_jsonl = iree_string_view_equal(format, IREE_SV("jsonl"));
  if (!is_text && !is_jsonl) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported profile output format '%.*s'",
                            (int)format.size, format.data);
  }
  if (emit_events && !is_jsonl) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "--dispatch_events requires --format=jsonl");
  }

  iree_io_file_contents_t* file_contents = NULL;
  iree_status_t status = iree_io_file_contents_map(
      path, IREE_IO_FILE_ACCESS_READ, host_allocator, &file_contents);

  iree_hal_profile_file_header_t header;
  iree_host_size_t first_record_offset = 0;
  if (iree_status_is_ok(status)) {
    status = iree_hal_profile_file_parse_header(file_contents->const_buffer,
                                                &header, &first_record_offset);
  }

  iree_profile_dispatch_context_t context;
  iree_profile_dispatch_context_initialize(host_allocator, &context);

  iree_host_size_t record_offset = first_record_offset;
  while (iree_status_is_ok(status) &&
         record_offset < file_contents->const_buffer.data_length) {
    iree_hal_profile_file_record_t record;
    iree_host_size_t next_record_offset = 0;
    status = iree_hal_profile_file_parse_record(file_contents->const_buffer,
                                                record_offset, &record,
                                                &next_record_offset);
    if (iree_status_is_ok(status)) {
      status = iree_profile_dispatch_process_metadata_record(&context, &record);
      record_offset = next_record_offset;
    }
  }

  record_offset = first_record_offset;
  while (iree_status_is_ok(status) &&
         record_offset < file_contents->const_buffer.data_length) {
    iree_hal_profile_file_record_t record;
    iree_host_size_t next_record_offset = 0;
    status = iree_hal_profile_file_parse_record(file_contents->const_buffer,
                                                record_offset, &record,
                                                &next_record_offset);
    if (iree_status_is_ok(status)) {
      status = iree_profile_dispatch_process_events_record(
          &context, &record, filter, projection_mode, id_filter, emit_events,
          file);
      record_offset = next_record_offset;
    }
  }

  if (iree_status_is_ok(status)) {
    switch (projection_mode) {
      case IREE_PROFILE_PROJECTION_MODE_DISPATCH:
        if (is_text) {
          status = iree_profile_dispatch_print_text(&context, filter, file);
        } else {
          iree_profile_dispatch_print_jsonl_summary(&context, filter,
                                                    emit_events, file);
          if (!emit_events) {
            status =
                iree_profile_dispatch_print_jsonl_aggregates(&context, file);
          }
        }
        break;
      case IREE_PROFILE_PROJECTION_MODE_EXECUTABLE:
        if (emit_events) {
          iree_profile_dispatch_print_jsonl_summary(&context, filter,
                                                    emit_events, file);
        } else if (is_text) {
          status = iree_profile_executable_print_text(&context, filter,
                                                      id_filter, file);
        } else {
          status = iree_profile_executable_print_jsonl(&context, filter,
                                                       id_filter, file);
        }
        break;
      case IREE_PROFILE_PROJECTION_MODE_COMMAND:
        if (emit_events) {
          iree_profile_dispatch_print_jsonl_summary(&context, filter,
                                                    emit_events, file);
        } else if (is_text) {
          status = iree_profile_command_print_text(&context, filter, id_filter,
                                                   file);
        } else {
          status = iree_profile_command_print_jsonl(&context, filter, id_filter,
                                                    file);
        }
        break;
      case IREE_PROFILE_PROJECTION_MODE_QUEUE:
        if (emit_events) {
          iree_profile_dispatch_print_jsonl_summary(&context, filter,
                                                    emit_events, file);
        } else if (is_text) {
          status =
              iree_profile_queue_print_text(&context, filter, id_filter, file);
        } else {
          status =
              iree_profile_queue_print_jsonl(&context, filter, id_filter, file);
        }
        break;
      default:
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "unsupported profile projection mode %d",
                                  (int)projection_mode);
        break;
    }
  }

  iree_profile_dispatch_context_deinitialize(&context);
  iree_io_file_contents_free(file_contents);
  return status;
}

static iree_status_t iree_profile_cat_file(iree_string_view_t path,
                                           iree_string_view_t format,
                                           FILE* file,
                                           iree_allocator_t host_allocator) {
  bool is_text = iree_string_view_equal(format, IREE_SV("text"));
  bool is_jsonl = iree_string_view_equal(format, IREE_SV("jsonl"));
  if (!is_text && !is_jsonl) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported profile output format '%.*s'",
                            (int)format.size, format.data);
  }

  iree_io_file_contents_t* file_contents = NULL;
  iree_status_t status = iree_io_file_contents_map(
      path, IREE_IO_FILE_ACCESS_READ, host_allocator, &file_contents);

  iree_hal_profile_file_header_t header;
  iree_host_size_t record_offset = 0;
  if (iree_status_is_ok(status)) {
    status = iree_hal_profile_file_parse_header(file_contents->const_buffer,
                                                &header, &record_offset);
  }
  if (iree_status_is_ok(status)) {
    if (is_text) {
      iree_profile_dump_header_text(&header, file);
    } else {
      iree_profile_dump_header_jsonl(&header, file);
    }
  }

  iree_host_size_t record_index = 0;
  while (iree_status_is_ok(status) &&
         record_offset < file_contents->const_buffer.data_length) {
    iree_hal_profile_file_record_t record;
    iree_host_size_t next_record_offset = 0;
    status = iree_hal_profile_file_parse_record(file_contents->const_buffer,
                                                record_offset, &record,
                                                &next_record_offset);
    if (iree_status_is_ok(status)) {
      if (is_text) {
        iree_profile_dump_record_text(record_index, &record, file);
      } else {
        iree_profile_dump_record_jsonl(record_index, &record, file);
      }
      record_offset = next_record_offset;
      ++record_index;
    }
  }

  iree_io_file_contents_free(file_contents);
  return status;
}

static const char kIreeProfileUsage[] =
    "Inspects IREE HAL profile bundles.\n"
    "\n"
    "A .ireeprof bundle is a HAL-native profiling artifact. It contains\n"
    "session metadata, physical-device and queue metadata, executable/export\n"
    "metadata, command-buffer metadata and operations, clock-correlation\n"
    "samples, host queue-operation events, device-timestamped dispatch "
    "events,\n"
    "optional memory lifecycle events, and optional hardware counter samples. "
    "The projection commands below let you enter "
    "from\n"
    "the object you care about and then cross-reference ids in JSONL.\n"
    "\n"
    "Usage:\n"
    "  iree-profile summary [--format=text|jsonl] <file.ireeprof>\n"
    "  iree-profile explain [--format=text|jsonl] [--filter=pattern]\n"
    "      [--id=dispatch_event_id] <file.ireeprof>\n"
    "  iree-profile executable [--format=text|jsonl] [--id=executable_id]\n"
    "      [--filter=pattern] [--dispatch_events] <file.ireeprof>\n"
    "  iree-profile dispatch [--format=text|jsonl] [--id=event_id]\n"
    "      [--filter=pattern] [--dispatch_events] <file.ireeprof>\n"
    "  iree-profile command [--format=text|jsonl] [--id=command_buffer_id]\n"
    "      [--filter=pattern] [--dispatch_events] <file.ireeprof>\n"
    "  iree-profile counter [--format=text|jsonl] "
    "[--id=sample_or_related_id]\n"
    "      [--filter=pattern] [--counter_samples] <file.ireeprof>\n"
    "  iree-profile memory [--format=text|jsonl] "
    "[--id=event_or_allocation_id]\n"
    "      [--filter=event_type_pattern] <file.ireeprof>\n"
    "  iree-profile queue [--format=text|jsonl] [--id=submission_id]\n"
    "      [--filter=pattern] [--dispatch_events] <file.ireeprof>\n"
    "  iree-profile export --format=ireeperf-jsonl [--output=path|-]\n"
    "      <file.ireeprof>\n"
    "  iree-profile cat [--format=text|jsonl] <file.ireeprof>\n"
    "  iree-profile --agent_md\n"
    "\n"
    "Commands:\n"
    "  summary      Bundle health, metadata counts, clock fit, and per-device\n"
    "               dispatch timing totals.\n"
    "  explain      Opinionated bottleneck summary: device spans, queue busy\n"
    "               intervals, top exports/dispatches, transfer/memory totals,"
    "\n"
    "               and evidence-backed hints.\n"
    "  executable   Executable/export catalog joined with dispatch timing. "
    "Use\n"
    "               this when optimizing a specific kernel name or export. "
    "Includes\n"
    "               code_object_hash and pipeline_hash when producers provide "
    "them.\n"
    "  dispatch     Per-export dispatch timing aggregates, or individual "
    "events\n"
    "               with --dispatch_events --format=jsonl.\n"
    "  command      Recorded command-buffer operations, metadata, and "
    "per-execution\n"
    "               dispatch spans. Use this to inspect copy/fill/update/"
    "barrier\n"
    "               structure and drill into dispatch timings.\n"
    "  queue        Queue operation events and dispatch-derived submission "
    "spans.\n"
    "  counter      Hardware counter metadata and sample aggregates joined "
    "to\n"
    "               dispatch/export/command/queue ids. Use "
    "--counter_samples\n"
    "               with JSONL for per-dispatch sample rows.\n"
    "  memory       Memory lifecycle events and high-water summaries for "
    "slabs,\n"
    "               pools, async queue allocations, and HAL buffers. "
    "Live-at-end counts\n"
    "               are capture-window state, not required leaks.\n"
    "  export       Decoded tooling interchange export. The first supported\n"
    "               format is schema-versioned ireeperf-jsonl.\n"
    "  cat          Raw bundle record dump for format archaeology/debugging.\n"
    "\n"
    "Important flags:\n"
    "  --format=text|jsonl     Text for humans, JSONL for tools and agents.\n"
    "  --filter=pattern        Wildcard over dispatch/export keys, command op\n"
    "                          names/keys, or queue/memory event type names,\n"
    "                          such as '*softmax*' or 'copy'.\n"
    "  --id=N                  dispatch: event_id; executable: executable_id;\n"
    "                          command: command_buffer_id; memory: "
    "event_id or\n"
    "                          allocation_id; queue: submission_id; "
    "counter:\n"
    "                          sample_id, dispatch_event_id, "
    "submission_id, or\n"
    "                          command_buffer_id; explain:\n"
    "                          dispatch event id.\n"
    "  --dispatch_events       Stream individual dispatch_event rows instead "
    "of\n"
    "                          only aggregate projection rows. Requires "
    "JSONL.\n"
    "  --counter_samples       Stream individual counter_sample rows instead "
    "of\n"
    "                          only aggregate counter rows. Requires JSONL.\n"
    "  --output=path|-         Export destination path, or `-` for stdout.\n"
    "  --agent_md              Print a Markdown guide optimized for "
    "AGENTS.md.\n"
    "\n"
    "Capture examples:\n"
    "  iree-benchmark-module --device=amdgpu --module=model.vmfb \\\n"
    "      --function=main --benchmark_min_time=20x \\\n"
    "      --device_profiling_mode=queue \\\n"
    "      --device_profiling_output=/tmp/model.ireeprof\n"
    "\n"
    "Embedding examples:\n"
    "  Use iree_hal_profile_file_sink_create to create a file-backed sink,\n"
    "  set iree_hal_device_profiling_options_t::sink, then call\n"
    "  iree_hal_device_profiling_begin/end around the work to capture.\n"
    "\n"
    "Analysis examples:\n"
    "  iree-profile summary /tmp/model.ireeprof\n"
    "  iree-profile explain /tmp/model.ireeprof\n"
    "  iree-profile explain --format=jsonl /tmp/model.ireeprof | \\\n"
    "      jq 'select(.type | startswith(\"explain_top_\"))'\n"
    "  iree-profile export --format=ireeperf-jsonl \\\n"
    "      --output=/tmp/model.ireeperf.jsonl /tmp/model.ireeprof\n"
    "  iree-profile executable --filter='*matmul*' /tmp/model.ireeprof\n"
    "  iree-profile dispatch --format=jsonl /tmp/model.ireeprof | \\\n"
    "      jq 'select(.type==\"dispatch_group\") | {key,avg_ns,count}'\n"
    "  iree-profile queue --format=jsonl /tmp/model.ireeprof | \\\n"
    "      jq 'select(.type==\"queue_event\" or "
    ".type==\"queue_submission\")'\n"
    "  iree-profile counter --format=jsonl /tmp/model.ireeprof | \\\n"
    "      jq 'select(.type==\"counter_group\") | "
    "{key,counter,avg,sum,samples}'\n"
    "  iree-profile counter --format=jsonl --counter_samples \\\n"
    "      /tmp/model.ireeprof | jq 'select(.type==\"counter_sample\")'\n"
    "  iree-profile memory --format=jsonl /tmp/model.ireeprof | \\\n"
    "      jq 'select(.type==\"memory_pool\") | \\\n"
    "          {kind,physical_device_ordinal,pool_id,"
    "high_water_bytes,waits}'\n"
    "  iree-profile command --id=1 --format=jsonl --dispatch_events \\\n"
    "      /tmp/model.ireeprof | jq 'select(.type==\"dispatch_event\")'\n"
    "\n"
    "Use `iree-profile --agent_md` for a Markdown playbook with JSONL record\n"
    "types and cross-reference recipes.\n";

static void iree_profile_fprint_usage(FILE* file) {
  fputs("iree-profile\n", file);
  fputs(kIreeProfileUsage, file);
}

static void iree_profile_print_agent_markdown(FILE* file) {
  fputs(
      "# IREE HAL Profiling With iree-profile\n"
      "\n"
      "Use `iree-profile` to inspect `.ireeprof` bundles emitted by IREE HAL\n"
      "device profiling. Prefer `--format=jsonl` for automated analysis: each\n"
      "line is one independent JSON record that can be filtered with `jq`.\n"
      "\n"
      "## Capture\n"
      "\n"
      "Capture from `iree-benchmark-module` with HAL queue profiling "
      "enabled:\n"
      "\n"
      "```bash\n"
      "iree-benchmark-module --device=amdgpu --module=model.vmfb \\\n"
      "  --function=main --benchmark_min_time=20x \\\n"
      "  --device_profiling_mode=queue \\\n"
      "  --device_profiling_output=/tmp/model.ireeprof\n"
      "```\n"
      "\n"
      "Embedding applications can capture the same bundle through the HAL "
      "API:\n"
      "\n"
      "```c\n"
      "iree_io_file_handle_t* file_handle = NULL;\n"
      "iree_hal_profile_sink_t* sink = NULL;\n"
      "IREE_RETURN_IF_ERROR(iree_io_file_handle_create(\n"
      "    IREE_IO_FILE_MODE_WRITE | IREE_IO_FILE_MODE_SEQUENTIAL_SCAN,\n"
      "    IREE_SV(\"/tmp/model.ireeprof\"), 0, host_allocator, "
      "&file_handle));\n"
      "IREE_RETURN_IF_ERROR(iree_hal_profile_file_sink_create(\n"
      "    file_handle, host_allocator, &sink));\n"
      "iree_hal_device_profiling_options_t options = {0};\n"
      "options.mode = IREE_HAL_DEVICE_PROFILING_MODE_QUEUE_OPERATIONS;\n"
      "options.sink = sink;\n"
      "IREE_RETURN_IF_ERROR(iree_hal_device_profiling_begin(device, "
      "&options));\n"
      "/* Run workload. */\n"
      "IREE_RETURN_IF_ERROR(iree_hal_device_profiling_end(device));\n"
      "iree_hal_profile_sink_release(sink);\n"
      "iree_io_file_handle_release(file_handle);\n"
      "```\n"
      "\n"
      "## Commands\n"
      "\n"
      "- `summary` checks bundle health, metadata counts, dispatch counts, "
      "clock\n"
      "  correlation, and per-device timing totals.\n"
      "- `explain` gives an opinionated first-pass bottleneck view: visible "
      "device\n"
      "  spans, summed active dispatch time, merged per-queue busy intervals, "
      "top\n"
      "  exports, top individual dispatches, transfer totals, memory "
      "pressure,\n"
      "  and evidence-backed hints. Queue gap hints require queue event "
      "records;\n"
      "  do not infer queue dependency causes from dispatch record order.\n"
      "- `executable` lists executable/export metadata and joins export ids "
      "to\n"
      "  dispatch timing aggregates. Hash-capable producers include "
      "`code_object_hash`\n"
      "  on `executable` rows and `pipeline_hash` on `executable_export` "
      "rows for\n"
      "  cross-run and external-tool correlation.\n"
      "- `dispatch` groups timings by executable export, or emits every "
      "dispatch\n"
      "  event with `--dispatch_events`.\n"
      "- `command` lists command-buffer metadata, static operation records, "
      "and\n"
      "  execution spans grouped by `command_buffer_id` and `submission_id`.\n"
      "- `queue` groups dispatch-derived queue submissions by physical queue "
      "and\n"
      "  `submission_id`, and emits host queue-operation events when the "
      "producer\n"
      "  provides them. Queue events include operation type, wait/signal "
      "counts,\n"
      "  dependency strategy, payload bytes, and related object ids.\n"
      "- `counter` joins hardware counter metadata and samples to "
      "dispatch/export,\n"
      "  command-buffer, queue, and submission ids. Aggregate mode emits "
      "per-counter\n"
      "  statistics; `--counter_samples --format=jsonl` emits every matched "
      "sample\n"
      "  row with raw values for drilldown.\n"
      "- `memory` summarizes slab/provider events, pool reservation events, "
      "async\n"
      "  queue alloca/dealloca, and synchronous HAL buffer allocation/free "
      "high-water behavior. With `--format=jsonl` it emits individual\n"
      "  `memory_event` rows before the summary and then emits `memory_pool` "
      "and\n"
      "  `memory_allocation` rows for pool/provider and lifecycle drilldown. "
      "Live\n"
      "  allocations at profile end are capture-window state and may be "
      "normal\n"
      "  for retained outputs or embedding-managed buffers.\n"
      "- `export --format=ireeperf-jsonl` decodes `.ireeprof` into a "
      "schema-versioned\n"
      "  tooling interchange stream. This is the stable boundary for jq, "
      "agents,\n"
      "  telemetry upload, and optional Python adapters such as Perfetto or "
      "SQLite.\n"
      "- `cat` dumps raw bundle records and is mainly for format debugging.\n"
      "\n"
      "## JSONL Record Types\n"
      "\n"
      "- `summary` emits `summary` and `device_summary`.\n"
      "- `explain --format=jsonl` emits `explain_summary`, "
      "`explain_device`,\n"
      "  `explain_queue`, `explain_top_export`, "
      "`explain_top_dispatch`,\n"
      "  `explain_queue_operations`, `explain_memory_pressure`, and "
      "`explain_hint`.\n"
      "- `executable --format=jsonl` emits `executable_summary`, "
      "`executable`,\n"
      "  `executable_export`, and `executable_export_dispatch_group`.\n"
      "- `dispatch --format=jsonl` emits `dispatch_summary` and "
      "`dispatch_group`.\n"
      "- `dispatch --format=jsonl --dispatch_events` emits `dispatch_event` "
      "rows\n"
      "  followed by `dispatch_summary`.\n"
      "- `command --format=jsonl` emits `command_summary`, `command_buffer`,\n"
      "  `command_operation`, and `command_execution`.\n"
      "- `queue --format=jsonl` emits `queue_summary`, `queue`,\n"
      "  `queue_submission`, and `queue_event`.\n"
      "- `counter --format=jsonl` emits `counter_summary`, `counter_set`,\n"
      "  `counter`, and `counter_group`.\n"
      "- `counter --format=jsonl --counter_samples` additionally emits\n"
      "  `counter_sample` rows before the summary.\n"
      "- `memory --format=jsonl` emits `memory_event`, `memory_summary`,\n"
      "  `memory_device`, `memory_pool`, and `memory_allocation`.\n"
      "- `export --format=ireeperf-jsonl` emits records with "
      "`schema_version`\n"
      "  and `record_type`: `schema`, `session`, `device`, `queue`,\n"
      "  `executable`, `executable_export`, `command_buffer`,\n"
      "  `command_operation`, `clock_correlation`, `dispatch_event`,\n"
      "  `queue_event`, `memory_event`, `counter_set`, `counter`,\n"
      "  `counter_sample`, and `diagnostic`.\n"
      "\n"
      "## Cross-Reference Keys\n"
      "\n"
      "- `dispatch_event.event_id` uniquely identifies one dispatch event in "
      "the\n"
      "  profile stream. Use `dispatch --id=<event_id>` to isolate it.\n"
      "- `dispatch_event.submission_id` joins dispatches to "
      "`queue_submission`.\n"
      "  Use `queue --id=<submission_id>` or "
      "`queue --id=<submission_id> --dispatch_events` to inspect a "
      "submission.\n"
      "- `dispatch_event.command_buffer_id` joins dispatches to "
      "`command_buffer`\n"
      "  and `command_execution`. Use `command --id=<command_buffer_id>`.\n"
      "- `dispatch_event.command_index` is the ordinal of the dispatch within "
      "the\n"
      "  command buffer when the event came from reusable command-buffer "
      "replay.\n"
      "- `command_operation.command_buffer_id` and "
      "`command_operation.command_index`\n"
      "  join recorded command-buffer operations to dispatch events. "
      "`op` and\n"
      "  `key` can be filtered with `command --filter=<pattern>`.\n"
      "- `dispatch_event.executable_id` plus `dispatch_event.export_ordinal` "
      "joins\n"
      "  to `executable_export`. The `key` field is the export name when "
      "present,\n"
      "  otherwise a stable numeric fallback.\n"
      "- `physical_device_ordinal`, `queue_ordinal`, and `stream_id` identify "
      "the\n"
      "  producing physical queue within the session.\n"
      "- `queue_event.submission_id` joins queue operation metadata to "
      "`queue_submission`\n"
      "  and dispatch events. `queue_event.command_buffer_id` joins execute "
      "events\n"
      "  to `command_buffer`; `queue_event.allocation_id` joins alloca/"
      "dealloca\n"
      "  events to memory events.\n"
      "- `counter_sample.sample_id` identifies one producer sample. "
      "`counter_sample.dispatch_event_id`, `submission_id`, "
      "`command_buffer_id`,\n"
      "  `command_index`, `executable_id`, `export_ordinal`, "
      "`physical_device_ordinal`,\n"
      "  `queue_ordinal`, and `stream_id` join counter data to the matching\n"
      "  dispatch, command, executable, and queue projections. "
      "`counter_set_id`\n"
      "  plus `counter_ordinal` joins samples to counter metadata.\n"
      "- `memory_event.submission_id` joins queue alloca/dealloca events to "
      "queue\n"
      "  submissions when nonzero. `memory_event.allocation_id` joins all "
      "events\n"
      "  associated with one producer-defined allocation handle. "
      "`memory_pool`\n"
      "  groups by producer pool/provider id and memory type. "
      "`memory_allocation`\n"
      "  groups capture-window lifecycle rows by kind, allocation id, and "
      "pool id.\n"
      "  `memory_device`\n"
      "  live/high-water fields are computed over the capture window, so "
      "they\n"
      "  need not balance to zero if buffers outlive profiling.\n"
      "\n"
      "## Recipes\n"
      "\n"
      "Find the slowest kernel/export groups:\n"
      "\n"
      "```bash\n"
      "iree-profile explain --format=jsonl /tmp/model.ireeprof | \\\n"
      "  jq 'select(.type==\"explain_top_export\") | \\\n"
      "      {rank,key,total_ns,avg_ns,max_ns,dispatches}'\n"
      "```\n"
      "\n"
      "Find the slowest individual dispatches:\n"
      "\n"
      "```bash\n"
      "iree-profile explain --format=jsonl /tmp/model.ireeprof | \\\n"
      "  jq 'select(.type==\"explain_top_dispatch\") | \\\n"
      "      {rank,event_id,submission_id,key,duration_ns}'\n"
      "```\n"
      "\n"
      "Build a full custom export table:\n"
      "\n"
      "```bash\n"
      "iree-profile dispatch --format=jsonl /tmp/model.ireeprof | \\\n"
      "  jq -s 'map(select(.type==\"dispatch_group\")) | \\\n"
      "         sort_by(.total_ns) | reverse[:20] | \\\n"
      "         map({key,count,avg_ns,total_ns,max_ns})'\n"
      "```\n"
      "\n"
      "Export the full decoded interchange stream for downstream tools:\n"
      "\n"
      "```bash\n"
      "iree-profile export --format=ireeperf-jsonl \\\n"
      "  --output=/tmp/model.ireeperf.jsonl /tmp/model.ireeprof\n"
      "jq 'select(.record_type==\"dispatch_event\") | \\\n"
      "    {event_id,key,duration_ns,normalized_time_available}' \\\n"
      "  /tmp/model.ireeperf.jsonl\n"
      "```\n"
      "\n"
      "Show every dispatch of one kernel name pattern:\n"
      "\n"
      "```bash\n"
      "iree-profile dispatch --format=jsonl --dispatch_events \\\n"
      "  --filter='*softmax*' /tmp/model.ireeprof | \\\n"
      "  jq 'select(.type==\"dispatch_event\") | \\\n"
      "      {event_id,submission_id,command_buffer_id,command_index,key,"
      "duration_ns}'\n"
      "```\n"
      "\n"
      "Find expensive command-buffer executions:\n"
      "\n"
      "```bash\n"
      "iree-profile command --format=jsonl /tmp/model.ireeprof | \\\n"
      "  jq -s 'map(select(.type==\"command_execution\")) | \\\n"
      "         sort_by(.span_ns) | reverse[:20] | \\\n"
      "         map({command_buffer_id,submission_id,span_ns,dispatches})'\n"
      "```\n"
      "\n"
      "Drill into all dispatches for command buffer 1:\n"
      "\n"
      "```bash\n"
      "iree-profile command --id=1 --format=jsonl --dispatch_events \\\n"
      "  /tmp/model.ireeprof | jq 'select(.type==\"dispatch_event\")'\n"
      "```\n"
      "\n"
      "Show the recorded operation stream for command buffer 1:\n"
      "\n"
      "```bash\n"
      "iree-profile command --id=1 --format=jsonl /tmp/model.ireeprof | \\\n"
      "  jq 'select(.type==\"command_operation\") | \\\n"
      "      {command_index,op,key,length,workgroup_count}'\n"
      "```\n"
      "\n"
      "Find queue submissions with long device spans:\n"
      "\n"
      "```bash\n"
      "iree-profile queue --format=jsonl /tmp/model.ireeprof | \\\n"
      "  jq -s 'map(select(.type==\"queue_submission\")) | \\\n"
      "         sort_by(.span_ns) | reverse[:20] | \\\n"
      "         map({submission_id,physical_device_ordinal,queue_ordinal,"
      "span_ns,dispatches})'\n"
      "```\n"
      "\n"
      "Drill from queue submission 4 to its dispatch events:\n"
      "\n"
      "```bash\n"
      "iree-profile queue --id=4 --format=jsonl --dispatch_events \\\n"
      "  /tmp/model.ireeprof | jq 'select(.type==\"dispatch_event\")'\n"
      "```\n"
      "\n"
      "Summarize hardware counters by kernel/export:\n"
      "\n"
      "```bash\n"
      "iree-profile counter --format=jsonl /tmp/model.ireeprof | \\\n"
      "  jq -s 'map(select(.type==\"counter_group\")) | \\\n"
      "         sort_by(.sum) | reverse[:20] | \\\n"
      "         map({key,counter,samples,avg,sum})'\n"
      "```\n"
      "\n"
      "Show raw counter samples for one dispatch event:\n"
      "\n"
      "```bash\n"
      "iree-profile counter --format=jsonl --counter_samples \\\n"
      "  --id=<dispatch_event_id> /tmp/model.ireeprof | \\\n"
      "  jq 'select(.type==\"counter_sample\") | \\\n"
      "      {sample_id,dispatch_event_id,key,counter,value,values}'\n"
      "```\n"
      "\n"
      "Inspect allocation high-water by physical device:\n"
      "\n"
      "```bash\n"
      "iree-profile memory --format=jsonl /tmp/model.ireeprof | \\\n"
      "  jq 'select(.type==\"memory_device\") | \\\n"
      "      {physical_device_ordinal,current_queue_allocations,"
      "high_water_queue_bytes,current_buffer_allocations,"
      "high_water_buffer_bytes}'\n"
      "```\n"
      "\n"
      "Find pools/providers with the largest memory high-water:\n"
      "\n"
      "```bash\n"
      "iree-profile memory --format=jsonl /tmp/model.ireeprof | \\\n"
      "  jq -s 'map(select(.type==\"memory_pool\")) | \\\n"
      "         sort_by(.high_water_bytes) | reverse[:20] | \\\n"
      "         map({kind,physical_device_ordinal,pool_id,memory_type,"
      "high_water_bytes,waits})'\n"
      "```\n"
      "\n"
      "Trace one allocation lifecycle:\n"
      "\n"
      "```bash\n"
      "iree-profile memory --format=jsonl --id=<allocation_id> \\\n"
      "  /tmp/model.ireeprof | \\\n"
      "  jq 'select(.type==\"memory_event\" or "
      ".type==\"memory_allocation\")'\n"
      "```\n"
      "\n"
      "Map executable ids and export ordinals to names:\n"
      "\n"
      "```bash\n"
      "iree-profile executable --format=jsonl /tmp/model.ireeprof | \\\n"
      "  jq 'select(.type==\"executable_export\") | \\\n"
      "      {executable_id,export_ordinal,key,binding_count,workgroup_size}'\n"
      "```\n"
      "\n"
      "## Notes For Agents\n"
      "\n"
      "Treat `summary` as the first sanity check: if clock fitting is "
      "unavailable\n"
      "or dispatch timestamps are invalid, timing conclusions are suspect. "
      "For\n"
      "automated analysis, prefer entering through `queue`, `command`,\n"
      "`executable`, or `dispatch` with JSONL and then drilling into\n"
      "`--dispatch_events` using the ids above. Do not infer queue ordering "
      "from\n"
      "record order alone; HAL queue ordering is established by semaphore "
      "edges,\n"
      "and full queue-edge records require producer support beyond the "
      "current\n"
      "dispatch-derived projections.\n",
      file);
}

int main(int argc, char** argv) {
  IREE_TRACE_APP_ENTER();
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_t host_allocator = iree_allocator_system();
  int exit_code = EXIT_SUCCESS;

  iree_flags_set_usage("iree-profile", kIreeProfileUsage);
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);
  if (FLAG_agent_md) {
    iree_profile_print_agent_markdown(stdout);
    fflush(stdout);
    IREE_TRACE_ZONE_END(z0);
    IREE_TRACE_APP_EXIT(exit_code);
    return exit_code;
  }

  iree_string_view_t command = IREE_SV("cat");
  iree_string_view_t path = iree_string_view_empty();
  if (argc == 2) {
    path = iree_make_cstring_view(argv[1]);
  } else if (argc == 3) {
    command = iree_make_cstring_view(argv[1]);
    path = iree_make_cstring_view(argv[2]);
  }

  iree_status_t status = iree_ok_status();
  if (argc != 2 && argc != 3) {
    fprintf(stderr, "Error: expected profile bundle path.\n");
    iree_profile_fprint_usage(stderr);
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "expected profile bundle path");
  } else if (iree_string_view_is_empty(path)) {
    fprintf(stderr, "Error: missing profile bundle path.\n");
    iree_profile_fprint_usage(stderr);
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "missing profile bundle path");
  } else if (!iree_string_view_equal(command, IREE_SV("cat")) &&
             !iree_string_view_equal(command, IREE_SV("command")) &&
             !iree_string_view_equal(command, IREE_SV("counter")) &&
             !iree_string_view_equal(command, IREE_SV("dispatch")) &&
             !iree_string_view_equal(command, IREE_SV("explain")) &&
             !iree_string_view_equal(command, IREE_SV("export")) &&
             !iree_string_view_equal(command, IREE_SV("executable")) &&
             !iree_string_view_equal(command, IREE_SV("memory")) &&
             !iree_string_view_equal(command, IREE_SV("queue")) &&
             !iree_string_view_equal(command, IREE_SV("summary"))) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported iree-profile command '%.*s'",
                              (int)command.size, command.data);
  }

  if (iree_status_is_ok(status)) {
    if (iree_string_view_equal(command, IREE_SV("summary"))) {
      status = iree_profile_summary_file(
          path, iree_make_cstring_view(FLAG_format), stdout, host_allocator);
    } else if (iree_string_view_equal(command, IREE_SV("explain"))) {
      status = iree_profile_explain_file(
          path, iree_make_cstring_view(FLAG_format),
          iree_make_cstring_view(FLAG_filter), FLAG_id, stdout, host_allocator);
    } else if (iree_string_view_equal(command, IREE_SV("export"))) {
      status = iree_profile_export_file(
          path, iree_make_cstring_view(FLAG_format),
          iree_make_cstring_view(FLAG_output), host_allocator);
    } else if (iree_string_view_equal(command, IREE_SV("command"))) {
      status = iree_profile_projection_file(
          path, iree_make_cstring_view(FLAG_format),
          iree_make_cstring_view(FLAG_filter),
          IREE_PROFILE_PROJECTION_MODE_COMMAND, FLAG_id, FLAG_dispatch_events,
          stdout, host_allocator);
    } else if (iree_string_view_equal(command, IREE_SV("counter"))) {
      status = iree_profile_counter_file(
          path, iree_make_cstring_view(FLAG_format),
          iree_make_cstring_view(FLAG_filter), FLAG_id, FLAG_counter_samples,
          stdout, host_allocator);
    } else if (iree_string_view_equal(command, IREE_SV("dispatch"))) {
      status = iree_profile_projection_file(
          path, iree_make_cstring_view(FLAG_format),
          iree_make_cstring_view(FLAG_filter),
          IREE_PROFILE_PROJECTION_MODE_DISPATCH, FLAG_id, FLAG_dispatch_events,
          stdout, host_allocator);
    } else if (iree_string_view_equal(command, IREE_SV("executable"))) {
      status = iree_profile_projection_file(
          path, iree_make_cstring_view(FLAG_format),
          iree_make_cstring_view(FLAG_filter),
          IREE_PROFILE_PROJECTION_MODE_EXECUTABLE, FLAG_id,
          FLAG_dispatch_events, stdout, host_allocator);
    } else if (iree_string_view_equal(command, IREE_SV("memory"))) {
      status = iree_profile_memory_file(
          path, iree_make_cstring_view(FLAG_format),
          iree_make_cstring_view(FLAG_filter), FLAG_id, stdout, host_allocator);
    } else if (iree_string_view_equal(command, IREE_SV("queue"))) {
      status = iree_profile_projection_file(
          path, iree_make_cstring_view(FLAG_format),
          iree_make_cstring_view(FLAG_filter),
          IREE_PROFILE_PROJECTION_MODE_QUEUE, FLAG_id, FLAG_dispatch_events,
          stdout, host_allocator);
    } else {
      status = iree_profile_cat_file(path, iree_make_cstring_view(FLAG_format),
                                     stdout, host_allocator);
    }
  }

  fflush(stdout);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    exit_code = EXIT_FAILURE;
  }
  fflush(stderr);

  IREE_TRACE_ZONE_END(z0);
  IREE_TRACE_APP_EXIT(exit_code);
  return exit_code;
}
