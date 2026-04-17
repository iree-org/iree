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

const char* iree_profile_memory_event_type_name(
    iree_hal_profile_memory_event_type_t type);

// Initializes |out_context| for memory event aggregation.
void iree_profile_memory_context_initialize(
    iree_allocator_t host_allocator,
    iree_profile_memory_context_t* out_context);

// Releases dynamic arrays owned by |context|.
void iree_profile_memory_context_deinitialize(
    iree_profile_memory_context_t* context);

// Processes one profile file record and accumulates matching memory events.
//
// If |emit_events| is true, each matched memory event is written as JSONL to
// |file| in addition to updating aggregate state.
iree_status_t iree_profile_memory_process_event_records(
    iree_profile_memory_context_t* context,
    const iree_hal_profile_file_record_t* record, iree_string_view_t filter,
    int64_t id_filter, bool emit_events, FILE* file);

// Reads a profile bundle from |path| and writes a memory report to |file|.
iree_status_t iree_profile_memory_report_file(iree_string_view_t path,
                                              iree_string_view_t format,
                                              iree_string_view_t filter,
                                              int64_t id_filter, FILE* file,
                                              iree_allocator_t host_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_TOOLING_PROFILE_MEMORY_H_
