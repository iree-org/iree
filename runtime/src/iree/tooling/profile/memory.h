// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOOLING_PROFILE_MEMORY_H_
#define IREE_TOOLING_PROFILE_MEMORY_H_

#include "iree/tooling/profile/common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_profile_memory_balance_t {
  // Number of lifecycle instances open after applying matched events.
  uint64_t current_count;
  // Maximum number of simultaneously open lifecycle instances.
  uint64_t high_water_count;
  // Number of opening transitions observed in the matched event stream.
  uint64_t total_open_count;
  // Number of closing transitions observed in the matched event stream.
  uint64_t total_close_count;
  // Closing transitions whose matching open was outside the matched stream.
  uint64_t partial_close_count;
  // Bytes open after applying matched events.
  uint64_t current_bytes;
  // Maximum simultaneously open bytes.
  uint64_t high_water_bytes;
  // Cumulative bytes opened by matched events.
  uint64_t total_open_bytes;
  // Cumulative bytes closed by matched events.
  uint64_t total_close_bytes;
  // Closing bytes whose matching open was outside the matched stream.
  uint64_t partial_close_bytes;
} iree_profile_memory_balance_t;

typedef struct iree_profile_memory_device_t {
  // Session-local physical device ordinal.
  uint32_t physical_device_ordinal;
  // Total memory events matched for this device.
  uint64_t event_count;
  // Slab acquire events matched for this device.
  uint64_t slab_acquire_count;
  // Slab release events matched for this device.
  uint64_t slab_release_count;
  // Slab-provider backing allocations opened and closed in the capture window.
  iree_profile_memory_balance_t slab_allocation_balance;
  // Pool reservation events matched for this device.
  uint64_t pool_reserve_count;
  // Pool materialization events matched for this device.
  uint64_t pool_materialize_count;
  // Pool release events matched for this device.
  uint64_t pool_release_count;
  // Pool wait events matched for this device.
  uint64_t pool_wait_count;
  // Pool bytes reserved from acquire until release.
  iree_profile_memory_balance_t pool_reservation_balance;
  // Pool reservations observed materialized into concrete HAL buffer views.
  iree_profile_memory_balance_t pool_materialization_balance;
  // Synchronous HAL buffer allocation events matched for this device.
  uint64_t buffer_allocate_count;
  // Synchronous HAL buffer free events matched for this device.
  uint64_t buffer_free_count;
  // Synchronous HAL buffer allocations opened and closed in the capture window.
  iree_profile_memory_balance_t buffer_allocation_balance;
  // Externally-owned HAL buffer import events matched for this device.
  uint64_t buffer_import_count;
  // Externally-owned HAL buffer unimport events matched for this device.
  uint64_t buffer_unimport_count;
  // Imported external buffer visibility opened and closed in the capture.
  iree_profile_memory_balance_t buffer_import_balance;
  // Queue alloca events matched for this device.
  uint64_t queue_alloca_count;
  // Queue dealloca events matched for this device.
  uint64_t queue_dealloca_count;
  // Queue-visible allocation bytes between queue_alloca and queue_dealloca.
  iree_profile_memory_balance_t queue_inflight_balance;
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
  // Externally-owned HAL buffer import lifecycle.
  IREE_PROFILE_MEMORY_LIFECYCLE_KIND_IMPORTED_BUFFER = 4,
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
  // Number of pool-stat snapshots captured for this pool row.
  uint64_t pool_stats_sample_count;
  // Last sampled pool bytes occupied by live reservations.
  uint64_t pool_bytes_reserved;
  // Highest sampled pool bytes occupied by live reservations.
  uint64_t pool_bytes_reserved_high_water;
  // Last sampled pool bytes available for reservation.
  uint64_t pool_bytes_free;
  // Lowest sampled pool bytes available for reservation.
  uint64_t pool_bytes_free_low_water;
  // Last sampled pool physical bytes committed.
  uint64_t pool_bytes_committed;
  // Highest sampled pool physical bytes committed.
  uint64_t pool_bytes_committed_high_water;
  // Last sampled pool budget limit in bytes, or 0 for unlimited.
  uint64_t pool_budget_limit;
  // Last sampled live reservation count.
  uint32_t pool_reservation_count;
  // Highest sampled live reservation count.
  uint32_t pool_reservation_high_water_count;
  // Last sampled committed slab count.
  uint32_t pool_slab_count;
  // Highest sampled committed slab count.
  uint32_t pool_slab_high_water_count;
  // Lifecycle balance for this pool/provider row's |kind|.
  iree_profile_memory_balance_t lifecycle_balance;
  // Materialization balance for pool reservation rows.
  iree_profile_memory_balance_t materialization_balance;
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
  // Combined memory-event flags observed for this allocation row.
  uint32_t flags;
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
  // Queue ordinal associated with |first_submission_id|, or UINT32_MAX.
  uint32_t first_queue_ordinal;
  // Queue ordinal associated with |last_submission_id|, or UINT32_MAX.
  uint32_t last_queue_ordinal;
  // Number of matched events in this lifecycle.
  uint64_t event_count;
  // Wait events in this lifecycle.
  uint64_t wait_count;
  // Materialize events in this lifecycle.
  uint64_t materialize_count;
  // Lifecycle balance for this allocation row's |kind|.
  iree_profile_memory_balance_t lifecycle_balance;
  // Materialization balance for pool reservation rows.
  iree_profile_memory_balance_t materialization_balance;
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

// Returns the stable text spelling used for memory event JSON/text output.
const char* iree_profile_memory_event_type_name(
    iree_hal_profile_memory_event_type_t type);

//===----------------------------------------------------------------------===//
// Memory event aggregation
//===----------------------------------------------------------------------===//

// Initializes |out_context| for memory event aggregation.
void iree_profile_memory_context_initialize(
    iree_allocator_t host_allocator,
    iree_profile_memory_context_t* out_context);

// Releases dynamic arrays owned by |context|.
void iree_profile_memory_context_deinitialize(
    iree_profile_memory_context_t* context);

// Accumulates memory events from one profile file record into |context|.
//
// Non-memory records are ignored. |filter| matches the memory event key fields,
// and |id_filter| restricts the matched allocation/event id when non-negative.
// If |emit_events| is true then each matched memory event is also written as a
// JSONL row to |file| before the aggregate report is generated.
iree_status_t iree_profile_memory_context_accumulate_record(
    iree_profile_memory_context_t* context,
    const iree_hal_profile_file_record_t* record, iree_string_view_t filter,
    int64_t id_filter, bool emit_events, FILE* file);

//===----------------------------------------------------------------------===//
// File-level reporting
//===----------------------------------------------------------------------===//

// Opens |path|, aggregates memory events, and writes a memory report to |file|.
//
// Text output emits summary tables. JSONL output emits matched raw memory event
// rows followed by aggregate summary rows for devices, pools, and allocations.
iree_status_t iree_profile_memory_report_file(iree_string_view_t path,
                                              iree_string_view_t format,
                                              iree_string_view_t filter,
                                              int64_t id_filter, FILE* file,
                                              iree_allocator_t host_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_TOOLING_PROFILE_MEMORY_H_
