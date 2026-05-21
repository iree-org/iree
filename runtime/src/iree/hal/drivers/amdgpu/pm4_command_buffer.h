// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_PM4_COMMAND_BUFFER_H_
#define IREE_HAL_DRIVERS_AMDGPU_PM4_COMMAND_BUFFER_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/amdgpu/abi/command_buffer.h"
#include "iree/hal/drivers/amdgpu/abi/timestamp.h"
#include "iree/hal/drivers/amdgpu/profile_metadata.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"
#include "iree/hal/drivers/amdgpu/util/pm4_capabilities.h"
#include "iree/hal/drivers/amdgpu/util/pm4_program.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef enum iree_hal_amdgpu_pm4_command_buffer_flag_bits_e {
  IREE_HAL_AMDGPU_PM4_COMMAND_BUFFER_FLAG_NONE = 0u,
  // Collects finalize-time host timing counters. This calls iree_time_now()
  // during command-buffer finalization and must only be enabled for profiling
  // or benchmark accounting runs.
  IREE_HAL_AMDGPU_PM4_COMMAND_BUFFER_FLAG_COLLECT_FINALIZE_TIMINGS = 1u << 0,
  // Materializes PM4 IB/template/fixup bytes into host staging builders and
  // publishes them into resident memory with hsa_memory_copy.
  IREE_HAL_AMDGPU_PM4_COMMAND_BUFFER_FLAG_MATERIALIZE_TO_HOST_COPY = 1u << 1,
  // Materializes PM4 IB/template/fixup bytes into a host staging image and
  // publishes it into resident memory with hsa_amd_memory_async_copy.
  IREE_HAL_AMDGPU_PM4_COMMAND_BUFFER_FLAG_MATERIALIZE_TO_HOST_ASYNC_COPY = 1u
                                                                           << 2,
  // Leaves async-copy publication pending at end() and makes queue submission
  // depend on the copy completion signal before executing the resident IB.
  IREE_HAL_AMDGPU_PM4_COMMAND_BUFFER_FLAG_NONBLOCKING_PUBLICATION = 1u << 3,
  // Materializes a dispatch-attributed profiling PM4 IB in addition to the
  // normal execution IB. This must only be set for active dispatch profiling
  // sessions because it adds timestamp packets and profile fixup metadata.
  IREE_HAL_AMDGPU_PM4_COMMAND_BUFFER_FLAG_MATERIALIZE_PROFILE_DISPATCH_TIMESTAMPS =
      1u << 4,
} iree_hal_amdgpu_pm4_command_buffer_flag_bits_t;

typedef uint32_t iree_hal_amdgpu_pm4_command_buffer_flags_t;

typedef struct iree_hal_amdgpu_pm4_command_buffer_resident_pool_t
    iree_hal_amdgpu_pm4_command_buffer_resident_pool_t;

// Device-resident command-buffer storage and dynamic fixup records owned by a
// finalized PM4 command buffer.
typedef struct iree_hal_amdgpu_pm4_command_buffer_fixup_plan_t {
  // Device pointer to immutable fixup records, or NULL when no fixup runs.
  IREE_AMDGPU_DEVICE_PTR const iree_hal_amdgpu_command_buffer_pm4_fixup_entry_t*
      entries;
  // Number of entries in |entries|.
  uint32_t entry_count;
  // Reserved padding that must be zero.
  uint32_t reserved0;
  // Device pointer to the resident allocation base patched by fixup offsets.
  IREE_AMDGPU_DEVICE_PTR uint8_t* target_base;
  // Allocated byte length of |target_base|.
  iree_host_size_t target_byte_length;
  // Device pointer to resident kernarg-template storage referenced by PM4.
  IREE_AMDGPU_DEVICE_PTR uint8_t* template_base;
  // Allocated byte length of |template_base|.
  iree_host_size_t template_byte_length;
} iree_hal_amdgpu_pm4_command_buffer_fixup_plan_t;

// Device-resident PM4 program and fixup plan used only while
// dispatch-attributed profiling is enabled.
typedef struct iree_hal_amdgpu_pm4_command_buffer_profile_plan_t {
  // Device-visible PM4 IB with per-dispatch timestamp packets.
  iree_hal_amdgpu_pm4_program_t program;
  // Device pointer to immutable profile fixup records.
  IREE_AMDGPU_DEVICE_PTR const iree_hal_amdgpu_command_buffer_pm4_fixup_entry_t*
      entries;
  // Number of entries in |entries|.
  uint32_t entry_count;
  // First synthetic binding slot used for dispatch timestamp destinations.
  uint32_t timestamp_binding_base;
  // Total binding-table entries consumed by profile fixup.
  uint32_t binding_count;
  // Number of profile-visible dispatch operations in |program|.
  uint32_t dispatch_count;
  // Device pointer to the resident allocation base patched by fixup offsets.
  IREE_AMDGPU_DEVICE_PTR uint8_t* target_base;
  // Device-visible fallback timestamp range used for unselected dispatches.
  IREE_AMDGPU_DEVICE_PTR iree_hal_amdgpu_timestamp_range_t* dummy_ticks;
} iree_hal_amdgpu_pm4_command_buffer_profile_plan_t;

// Host timing and byte counters captured while finalizing PM4 storage.
typedef struct iree_hal_amdgpu_pm4_command_buffer_publish_stats_t {
  // Total nanoseconds spent in command-buffer end/finalize.
  uint64_t total_finalize_ns;
  // Nanoseconds spent materializing compact PM4 records into resident storage.
  uint64_t materialize_ns;
  // Nanoseconds spent allocating resident PM4/template/fixup storage.
  uint64_t resident_allocate_ns;
  // Nanoseconds spent granting resident storage access to the device agent.
  uint64_t resident_allow_access_ns;
  // Nanoseconds spent copying bytes into resident storage after allocation.
  uint64_t resident_copy_ns;
  // Nanoseconds spent allocating host staging storage for publication.
  uint64_t host_staging_allocate_ns;
  // Nanoseconds spent granting device access to host staging storage.
  uint64_t host_staging_allow_access_ns;
  // Host compact record byte length consumed by materialization.
  uint64_t host_record_bytes;
  // Host staging byte length used for publication.
  uint64_t host_staging_bytes;
  // Total resident allocation byte length.
  uint64_t resident_bytes;
  // Bytes copied into resident storage after allocation.
  uint64_t resident_copy_bytes;
  // Number of resident HSA allocations performed.
  uint64_t resident_allocation_count;
  // Number of GPU agents passed to allow-access for resident storage.
  uint64_t resident_allow_access_agent_count;
  // Number of host staging HSA allocations performed.
  uint64_t host_staging_allocation_count;
  // Number of GPU agents passed to allow-access for host staging storage.
  uint64_t host_staging_allow_access_agent_count;
  // PM4 dwords used for non-terminal execution barriers.
  uint64_t execution_barrier_dwords;
  // PM4 dwords used for fixup-to-IB visibility barriers.
  uint64_t fixup_barrier_dwords;
  // PM4 dwords used for dispatch setup packets.
  uint64_t dispatch_setup_dwords;
  // PM4 dwords used for dispatch user-data packets.
  uint64_t dispatch_user_data_dwords;
  // PM4 dwords used for dispatch-direct packets.
  uint64_t dispatch_direct_dwords;
  // PM4 dwords used for the terminal execution barrier.
  uint64_t terminal_barrier_dwords;
  // Resident PM4 IB byte length.
  uint64_t program_bytes;
  // Resident kernarg-template byte length.
  uint64_t template_bytes;
  // Resident fixup-entry byte length.
  uint64_t fixup_entry_bytes;
} iree_hal_amdgpu_pm4_command_buffer_publish_stats_t;

// Creates a per-GPU pool of executable resident PM4 command-buffer storage.
//
// The pool owns whole HSA executable allocations and grants access only to
// |device_agent|. Command buffers borrow one allocation while finalized and
// return it to this pool on destruction. The pool must remain live until all
// command buffers created from it have been destroyed.
iree_status_t iree_hal_amdgpu_pm4_command_buffer_resident_pool_create(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_agent_t host_agent,
    hsa_agent_t device_agent, hsa_amd_memory_pool_t resident_memory_pool,
    hsa_amd_memory_pool_t host_staging_memory_pool,
    iree_allocator_t host_allocator,
    iree_hal_amdgpu_pm4_command_buffer_resident_pool_t** out_pool);

// Destroys |resident_pool| and releases all cached HSA allocations.
void iree_hal_amdgpu_pm4_command_buffer_resident_pool_destroy(
    iree_hal_amdgpu_pm4_command_buffer_resident_pool_t* resident_pool);

// Releases unused resident allocations cached in |resident_pool|.
void iree_hal_amdgpu_pm4_command_buffer_resident_pool_trim(
    iree_hal_amdgpu_pm4_command_buffer_resident_pool_t* resident_pool);

// Creates a host-recorded PM4 command buffer.
//
// The PM4 command-buffer path records dispatch-only HAL commands directly into
// a resident PM4 indirect buffer. Static dispatches reference resident kernarg
// templates; dynamic dispatches reference resident kernarg templates patched by
// a queue_execute binding-table fixup dispatch before the PM4 IB runs. The
// command buffer borrows |resident_pool| and |resource_set_block_pool|; callers
// must keep them live until all created command buffers are destroyed.
iree_status_t iree_hal_amdgpu_pm4_command_buffer_create(
    iree_hal_allocator_t* device_allocator, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_host_size_t device_ordinal,
    iree_hal_amdgpu_pm4_command_buffer_flags_t flags,
    iree_hal_amdgpu_vendor_packet_capability_flags_t vendor_packet_capabilities,
    iree_hal_amdgpu_pm4_timestamp_strategy_t pm4_timestamp_strategy,
    iree_hal_amdgpu_pm4_command_buffer_resident_pool_t* resident_pool,
    iree_hal_amdgpu_profile_metadata_registry_t* profile_metadata,
    iree_arena_block_pool_t* resource_set_block_pool,
    iree_allocator_t host_allocator,
    iree_hal_command_buffer_t** out_command_buffer);

// Returns true if |command_buffer| is an AMDGPU PM4 command buffer.
bool iree_hal_amdgpu_pm4_command_buffer_isa(
    iree_hal_command_buffer_t* command_buffer);

// Returns the physical device ordinal this command buffer was recorded for.
iree_host_size_t iree_hal_amdgpu_pm4_command_buffer_device_ordinal(
    iree_hal_command_buffer_t* command_buffer);

// Returns the immutable resident PM4 program produced by end().
const iree_hal_amdgpu_pm4_program_t* iree_hal_amdgpu_pm4_command_buffer_program(
    iree_hal_command_buffer_t* command_buffer);

// Returns the session-local profile id for this command buffer, or 0 when it
// was not created with retained profile metadata.
uint64_t iree_hal_amdgpu_pm4_command_buffer_profile_id(
    iree_hal_command_buffer_t* command_buffer);

// Returns retained profile command operations, or NULL when not retained.
const iree_hal_profile_command_operation_record_t*
iree_hal_amdgpu_pm4_command_buffer_profile_operations(
    iree_hal_command_buffer_t* command_buffer, uint32_t* out_count);

// Returns the number of recorded profile-visible operations.
uint32_t iree_hal_amdgpu_pm4_command_buffer_operation_count(
    iree_hal_command_buffer_t* command_buffer);

// Returns the immutable resident kernarg template and fixup plan from end().
const iree_hal_amdgpu_pm4_command_buffer_fixup_plan_t*
iree_hal_amdgpu_pm4_command_buffer_fixup_plan(
    iree_hal_command_buffer_t* command_buffer);

// Returns the immutable resident profile PM4 IB and fixup plan from end().
const iree_hal_amdgpu_pm4_command_buffer_profile_plan_t*
iree_hal_amdgpu_pm4_command_buffer_profile_plan(
    iree_hal_command_buffer_t* command_buffer);

// Returns finalize-time publication stats captured by the command buffer.
const iree_hal_amdgpu_pm4_command_buffer_publish_stats_t*
iree_hal_amdgpu_pm4_command_buffer_publish_stats(
    iree_hal_command_buffer_t* command_buffer);

// Acquires one queue-submission reference to the publication signal required
// before the resident PM4 IB may execute.
//
// Returns a null signal if no nonblocking publication is pending. When a
// non-null signal is returned the caller must either cancel the reference if no
// AQL barrier packet is published, or retire the reference after every AQL
// packet that names the signal has retired.
hsa_signal_t iree_hal_amdgpu_pm4_command_buffer_acquire_publication_reference(
    iree_hal_command_buffer_t* command_buffer);

// Cancels one acquired publication reference when no AQL packet was published.
void iree_hal_amdgpu_pm4_command_buffer_cancel_publication_reference(
    iree_hal_command_buffer_t* command_buffer);

// Retires one acquired publication reference after a queue submission that
// waited on the publication signal has retired.
void iree_hal_amdgpu_pm4_command_buffer_retire_publication_reference(
    iree_hal_command_buffer_t* command_buffer, iree_status_t status);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_PM4_COMMAND_BUFFER_H_
