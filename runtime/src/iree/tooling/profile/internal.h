// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOOLING_PROFILE_INTERNAL_H_
#define IREE_TOOLING_PROFILE_INTERNAL_H_

#include <errno.h>
#include <float.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/hal/utils/profile_file.h"
#include "iree/io/file_contents.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#define IREE_PROFILE_EXPLAIN_TOP_DISPATCH_COUNT 8
#define IREE_PROFILE_EXPLAIN_TOP_EXPORT_COUNT 10

typedef struct iree_profile_file_t {
  // Mapped profile file contents owned by this wrapper.
  iree_io_file_contents_t* contents;
  // Parsed profile file header.
  iree_hal_profile_file_header_t header;
  // Byte offset of the first record after the file header.
  iree_host_size_t first_record_offset;
} iree_profile_file_t;

typedef iree_status_t (*iree_profile_file_record_callback_t)(
    void* user_data, const iree_hal_profile_file_record_t* record,
    iree_host_size_t record_index);

typedef struct iree_profile_typed_record_t {
  // Source chunk containing this typed record.
  const iree_hal_profile_file_record_t* chunk;
  // Zero-based ordinal of this typed record within |chunk|.
  iree_host_size_t record_index;
  // Byte offset of this typed record within |chunk->payload|.
  iree_host_size_t payload_offset;
  // Minimum fixed header length requested by the parser.
  iree_host_size_t minimum_record_length;
  // Total byte length reported by the typed record prefix.
  iree_host_size_t record_length;
  // Full bytes covered by |record_length|.
  iree_const_byte_span_t contents;
  // Bytes after |minimum_record_length| and before |record_length|.
  iree_const_byte_span_t inline_payload;
  // Bytes after |record_length| through the end of the containing chunk.
  iree_const_byte_span_t following_payload;
} iree_profile_typed_record_t;

typedef struct iree_profile_typed_record_iterator_t {
  // Source chunk being iterated.
  const iree_hal_profile_file_record_t* chunk;
  // Minimum fixed header length to validate for each typed record.
  iree_host_size_t minimum_record_length;
  // Current byte offset into |chunk->payload|.
  iree_host_size_t payload_offset;
  // Zero-based ordinal for the next typed record.
  iree_host_size_t record_index;
} iree_profile_typed_record_iterator_t;

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
  // Dynamic array of device-timestamped queue operation event rows.
  iree_profile_dispatch_queue_device_event_t* queue_device_events;
  // Number of valid entries in |queue_device_events|.
  iree_host_size_t queue_device_event_count;
  // Capacity of |queue_device_events| in entries.
  iree_host_size_t queue_device_event_capacity;
  // Largest valid dispatch events observed while applying the active filter.
  iree_profile_dispatch_top_event_t
      top_dispatches[IREE_PROFILE_EXPLAIN_TOP_DISPATCH_COUNT];
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

const char* iree_profile_record_type_name(
    iree_hal_profile_file_record_type_t record_type);
void iree_profile_fprint_json_string(FILE* file, iree_string_view_t value);
void iree_profile_fprint_hash_hex(FILE* file, const uint64_t hash[2]);
double iree_profile_sqrt_f64(double value);
iree_status_t iree_profile_typed_record_parse(
    const iree_hal_profile_file_record_t* chunk,
    iree_host_size_t payload_offset, iree_host_size_t minimum_record_length,
    iree_host_size_t record_index, iree_profile_typed_record_t* out_record);
void iree_profile_typed_record_iterator_initialize(
    const iree_hal_profile_file_record_t* chunk,
    iree_host_size_t minimum_record_length,
    iree_profile_typed_record_iterator_t* out_iterator);
iree_status_t iree_profile_typed_record_iterator_next(
    iree_profile_typed_record_iterator_t* iterator,
    iree_profile_typed_record_t* out_record, bool* out_has_record);
iree_status_t iree_profile_file_open(iree_string_view_t path,
                                     iree_allocator_t host_allocator,
                                     iree_profile_file_t* out_profile_file);
void iree_profile_file_close(iree_profile_file_t* profile_file);
iree_status_t iree_profile_file_for_each_record(
    const iree_profile_file_t* profile_file,
    iree_profile_file_record_callback_t callback, void* user_data);

void iree_profile_summary_initialize(iree_allocator_t host_allocator,
                                     iree_profile_summary_t* out_summary);
void iree_profile_summary_deinitialize(iree_profile_summary_t* summary);
iree_status_t iree_profile_summary_process_record(
    iree_profile_summary_t* summary,
    const iree_hal_profile_file_record_t* record);
iree_status_t iree_profile_summary_file(iree_string_view_t path,
                                        iree_string_view_t format, FILE* file,
                                        iree_allocator_t host_allocator);

void iree_profile_dispatch_context_initialize(
    iree_allocator_t host_allocator,
    iree_profile_dispatch_context_t* out_context);
void iree_profile_dispatch_context_deinitialize(
    iree_profile_dispatch_context_t* context);
const iree_profile_dispatch_device_t* iree_profile_dispatch_find_device(
    const iree_profile_dispatch_context_t* context,
    uint32_t physical_device_ordinal);
bool iree_profile_dispatch_device_try_fit_clock(
    const iree_profile_dispatch_device_t* device, double* out_ns_per_tick,
    double* out_tick_frequency_hz);
iree_string_view_t iree_profile_dispatch_format_export_key(
    const iree_profile_dispatch_export_t* export_info,
    uint32_t physical_device_ordinal, char* numeric_buffer,
    iree_host_size_t numeric_buffer_capacity);
iree_status_t iree_profile_dispatch_resolve_key(
    const iree_profile_dispatch_context_t* context,
    uint32_t physical_device_ordinal, uint64_t executable_id,
    uint32_t export_ordinal, char* numeric_buffer,
    iree_host_size_t numeric_buffer_capacity, iree_string_view_t* out_key);
bool iree_profile_dispatch_key_matches(iree_string_view_t key,
                                       iree_string_view_t filter);
iree_status_t iree_profile_dispatch_process_metadata_record(
    iree_profile_dispatch_context_t* context,
    const iree_hal_profile_file_record_t* record);
iree_status_t iree_profile_dispatch_process_events_record(
    iree_profile_dispatch_context_t* context,
    const iree_hal_profile_file_record_t* record, iree_string_view_t filter,
    iree_profile_projection_mode_t projection_mode, int64_t id_filter,
    bool emit_events, FILE* file);
iree_status_t iree_profile_dispatch_process_queue_event_records(
    iree_profile_dispatch_context_t* context,
    const iree_hal_profile_file_record_t* record, iree_string_view_t filter,
    int64_t id_filter);
double iree_profile_dispatch_span_ticks(uint64_t earliest_start_tick,
                                        uint64_t latest_end_tick);
const char* iree_profile_command_operation_type_name(
    iree_hal_profile_command_operation_type_t type);
const char* iree_profile_queue_event_type_name(
    iree_hal_profile_queue_event_type_t type);
const char* iree_profile_queue_dependency_strategy_name(
    iree_hal_profile_queue_dependency_strategy_t strategy);
const char* iree_profile_event_relationship_type_name(
    iree_hal_profile_event_relationship_type_t type);
const char* iree_profile_event_endpoint_type_name(
    iree_hal_profile_event_endpoint_type_t type);
iree_status_t iree_profile_command_operation_resolve_key(
    const iree_profile_dispatch_context_t* context,
    const iree_hal_profile_command_operation_record_t* operation,
    char* numeric_buffer, iree_host_size_t numeric_buffer_capacity,
    iree_string_view_t* out_key);

const char* iree_profile_memory_event_type_name(
    iree_hal_profile_memory_event_type_t type);
void iree_profile_memory_context_initialize(
    iree_allocator_t host_allocator,
    iree_profile_memory_context_t* out_context);
void iree_profile_memory_context_deinitialize(
    iree_profile_memory_context_t* context);
iree_status_t iree_profile_memory_process_event_records(
    iree_profile_memory_context_t* context,
    const iree_hal_profile_file_record_t* record, iree_string_view_t filter,
    int64_t id_filter, bool emit_events, FILE* file);
iree_status_t iree_profile_memory_file(iree_string_view_t path,
                                       iree_string_view_t format,
                                       iree_string_view_t filter,
                                       int64_t id_filter, FILE* file,
                                       iree_allocator_t host_allocator);

iree_status_t iree_profile_projection_file(
    iree_string_view_t path, iree_string_view_t format,
    iree_string_view_t filter, iree_profile_projection_mode_t projection_mode,
    int64_t id_filter, bool emit_events, FILE* file,
    iree_allocator_t host_allocator);
iree_status_t iree_profile_counter_file(iree_string_view_t path,
                                        iree_string_view_t format,
                                        iree_string_view_t filter,
                                        int64_t id_filter, bool emit_samples,
                                        FILE* file,
                                        iree_allocator_t host_allocator);
iree_status_t iree_profile_explain_file(iree_string_view_t path,
                                        iree_string_view_t format,
                                        iree_string_view_t filter,
                                        int64_t id_filter, FILE* file,
                                        iree_allocator_t host_allocator);
iree_status_t iree_profile_export_file(iree_string_view_t path,
                                       iree_string_view_t format,
                                       iree_string_view_t output_path,
                                       iree_allocator_t host_allocator);
iree_status_t iree_profile_cat_file(iree_string_view_t path,
                                    iree_string_view_t format, FILE* file,
                                    iree_allocator_t host_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_TOOLING_PROFILE_INTERNAL_H_
