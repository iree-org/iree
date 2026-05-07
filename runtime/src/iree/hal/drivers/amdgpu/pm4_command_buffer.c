// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/pm4_command_buffer.h"

#include <inttypes.h>
#include <string.h>

#include "iree/base/threading/mutex.h"
#include "iree/hal/drivers/amdgpu/abi/kernel_descriptor.h"
#include "iree/hal/drivers/amdgpu/buffer.h"
#include "iree/hal/drivers/amdgpu/device/dispatch.h"
#include "iree/hal/drivers/amdgpu/executable.h"
#include "iree/hal/drivers/amdgpu/transient_buffer.h"
#include "iree/hal/drivers/amdgpu/util/pm4_barrier.h"
#include "iree/hal/drivers/amdgpu/util/pm4_dispatch.h"
#include "iree/hal/drivers/amdgpu/util/pm4_emitter.h"
#include "iree/hal/drivers/amdgpu/util/signal_pool.h"
#include "iree/hal/utils/resource_set.h"

//===----------------------------------------------------------------------===//
// PM4 command-buffer storage
//===----------------------------------------------------------------------===//

static const iree_hal_command_buffer_vtable_t
    iree_hal_amdgpu_pm4_command_buffer_vtable;

typedef enum iree_hal_amdgpu_pm4_command_buffer_recording_state_e {
  IREE_HAL_AMDGPU_PM4_COMMAND_BUFFER_RECORDING_STATE_INITIAL = 0,
  IREE_HAL_AMDGPU_PM4_COMMAND_BUFFER_RECORDING_STATE_RECORDING = 1,
  IREE_HAL_AMDGPU_PM4_COMMAND_BUFFER_RECORDING_STATE_FINALIZED = 2,
  IREE_HAL_AMDGPU_PM4_COMMAND_BUFFER_RECORDING_STATE_FAILED = 3,
} iree_hal_amdgpu_pm4_command_buffer_recording_state_t;

typedef enum iree_hal_amdgpu_pm4_command_record_opcode_e {
  IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_OPCODE_DISPATCH = 1,
} iree_hal_amdgpu_pm4_command_record_opcode_t;

typedef enum iree_hal_amdgpu_pm4_dispatch_record_flag_bits_e {
  IREE_HAL_AMDGPU_PM4_DISPATCH_RECORD_FLAG_NONE = 0u,
  IREE_HAL_AMDGPU_PM4_DISPATCH_RECORD_FLAG_EXECUTION_BARRIER = 1u << 0,
  IREE_HAL_AMDGPU_PM4_DISPATCH_RECORD_FLAG_FIXUP_BARRIER = 1u << 1,
} iree_hal_amdgpu_pm4_dispatch_record_flag_bits_t;

typedef uint32_t iree_hal_amdgpu_pm4_dispatch_record_flags_t;

typedef enum iree_hal_amdgpu_pm4_binding_record_flag_bits_e {
  IREE_HAL_AMDGPU_PM4_BINDING_RECORD_FLAG_NONE = 0u,
  IREE_HAL_AMDGPU_PM4_BINDING_RECORD_FLAG_DYNAMIC = 1u << 0,
} iree_hal_amdgpu_pm4_binding_record_flag_bits_t;

typedef uint32_t iree_hal_amdgpu_pm4_binding_record_flags_t;

typedef struct iree_hal_amdgpu_pm4_command_record_header_t {
  // Total byte length of this record including inline payload.
  uint32_t length;
  // iree_hal_amdgpu_pm4_command_record_opcode_t value.
  uint16_t opcode;
  // Reserved padding; must be zero.
  uint16_t reserved0;
} iree_hal_amdgpu_pm4_command_record_header_t;

typedef struct iree_hal_amdgpu_pm4_dispatch_record_t {
  // Common command-record header.
  iree_hal_amdgpu_pm4_command_record_header_t header;
  // Host descriptor with executable-load PM4 metadata for this dispatch.
  const iree_hal_amdgpu_executable_dispatch_descriptor_t* descriptor;
  // Session-local executable identifier, or 0 when unavailable.
  uint64_t executable_id;
  // Dispatch workgroup counts.
  uint32_t workgroup_count[3];
  // HAL command ordinal within this command buffer.
  uint32_t command_index;
  // Executable export ordinal dispatched by this command.
  uint32_t export_ordinal;
  // Byte offset of this dispatch's kernarg template in resident template
  // memory.
  uint32_t template_offset;
  // Byte length of this dispatch's kernarg template.
  uint32_t template_length;
  // Byte length of inline constant data following the record.
  uint32_t constant_length;
  // Number of inline native binding records following constants.
  uint32_t binding_record_count;
  // iree_hal_amdgpu_pm4_dispatch_record_flag_bits_t mask.
  iree_hal_amdgpu_pm4_dispatch_record_flags_t flags;
  // Acquire fence scope for a pending execution barrier.
  iree_hsa_fence_scope_t barrier_acquire_scope;
  // Release fence scope for a pending execution barrier.
  iree_hsa_fence_scope_t barrier_release_scope;
} iree_hal_amdgpu_pm4_dispatch_record_t;

typedef struct iree_hal_amdgpu_pm4_binding_record_t {
  // Static binding device pointer, or dynamic binding byte offset.
  uint64_t value;
  // Dynamic binding table slot when the dynamic flag is set.
  uint32_t binding_slot;
  // Byte offset in resident kernarg template storage.
  uint32_t target_offset;
  // iree_hal_amdgpu_pm4_binding_record_flag_bits_t mask.
  iree_hal_amdgpu_pm4_binding_record_flags_t flags;
  // Reserved padding; must be zero.
  uint32_t reserved0;
} iree_hal_amdgpu_pm4_binding_record_t;

typedef struct iree_hal_amdgpu_pm4_dword_builder_t {
  // Host allocator used to grow |dwords|.
  iree_allocator_t host_allocator;
  // PM4 dwords being emitted into host-owned or resident storage.
  uint32_t* dwords;
  // Number of populated PM4 dwords in |dwords|.
  uint32_t dword_count;
  // Number of allocated PM4 dwords in |dwords|.
  uint32_t capacity;
  // True when |dwords| must be freed by the builder.
  bool owns_storage;
} iree_hal_amdgpu_pm4_dword_builder_t;

typedef struct iree_hal_amdgpu_pm4_byte_builder_t {
  // Host allocator used to grow |bytes|.
  iree_allocator_t host_allocator;
  // Bytes being emitted into host-owned or resident storage.
  uint8_t* bytes;
  // Number of populated bytes in |bytes|.
  iree_host_size_t length;
  // Number of allocated bytes in |bytes|.
  iree_host_size_t capacity;
  // True when |bytes| must be freed by the builder.
  bool owns_storage;
} iree_hal_amdgpu_pm4_byte_builder_t;

typedef struct iree_hal_amdgpu_pm4_fixup_entry_builder_t {
  // Host allocator used to grow |entries|.
  iree_allocator_t host_allocator;
  // Fixup entries being emitted into host-owned or resident storage.
  iree_hal_amdgpu_command_buffer_pm4_fixup_entry_t* entries;
  // Number of populated fixup entries.
  uint32_t count;
  // Number of allocated fixup entries.
  uint32_t capacity;
  // True when |entries| must be freed by the builder.
  bool owns_storage;
} iree_hal_amdgpu_pm4_fixup_entry_builder_t;

enum {
  IREE_HAL_AMDGPU_PM4_RETAINED_RESOURCE_INLINE_CAPACITY = 64,
};

typedef struct iree_hal_amdgpu_pm4_retained_resource_table_t {
  // Open-addressed table of retained resources seen during recording.
  iree_hal_resource_t** resources;
  // Number of occupied resource slots.
  iree_host_size_t count;
  // Power-of-two capacity of |resources|.
  iree_host_size_t capacity;
  // Inline slots used before recording sees enough unique resources to spill.
  iree_hal_resource_t*
      inline_resources[IREE_HAL_AMDGPU_PM4_RETAINED_RESOURCE_INLINE_CAPACITY];
} iree_hal_amdgpu_pm4_retained_resource_table_t;

typedef struct iree_hal_amdgpu_pm4_command_buffer_resident_allocation_t {
  // Next allocation in the resident pool free list.
  struct iree_hal_amdgpu_pm4_command_buffer_resident_allocation_t* next;
  // Device-visible executable allocation base pointer.
  IREE_AMDGPU_DEVICE_PTR uint8_t* base;
  // Allocated byte capacity of |base|.
  iree_host_size_t capacity;
} iree_hal_amdgpu_pm4_command_buffer_resident_allocation_t;

struct iree_hal_amdgpu_pm4_command_buffer_resident_pool_t {
  // HSA API table used to allocate and free resident storage.
  const iree_hal_amdgpu_libhsa_t* libhsa;
  // CPU agent associated with host-side materialization buffers.
  hsa_agent_t host_agent;
  // GPU agent allowed to fetch resident PM4 storage.
  hsa_agent_t device_agent;
  // HSA memory pool used for executable PM4 storage.
  hsa_amd_memory_pool_t memory_pool;
  // HSA host memory pool used for async-copy staging sources.
  hsa_amd_memory_pool_t host_staging_memory_pool;
  // Host allocator used for allocation metadata and this pool.
  iree_allocator_t host_allocator;
  // Recommended HSA allocation granule used to round backing allocations.
  iree_host_size_t allocation_granule;
  // Recommended HSA allocation granule used to round host staging allocations.
  iree_host_size_t host_staging_allocation_granule;
  // Mutex guarding the free list and outstanding allocation count.
  iree_slim_mutex_t mutex;
  // HSA signals used for synchronous waits on async publication copies.
  iree_hal_amdgpu_host_signal_pool_t copy_signal_pool;
  // Cached allocations available for command-buffer publication.
  iree_hal_amdgpu_pm4_command_buffer_resident_allocation_t* free_list
      IREE_GUARDED_BY(mutex);
  // Cached host staging allocations available for async-copy publication.
  iree_hal_amdgpu_pm4_command_buffer_resident_allocation_t* staging_free_list
      IREE_GUARDED_BY(mutex);
  // Number of allocations currently borrowed by finalized command buffers.
  iree_host_size_t outstanding_count IREE_GUARDED_BY(mutex);
  // Number of staging allocations currently borrowed during finalization.
  iree_host_size_t outstanding_staging_count IREE_GUARDED_BY(mutex);
};

static void iree_hal_amdgpu_pm4_command_buffer_resident_allocation_free(
    iree_hal_amdgpu_pm4_command_buffer_resident_pool_t* resident_pool,
    iree_hal_amdgpu_pm4_command_buffer_resident_allocation_t* allocation) {
  iree_hal_amdgpu_hsa_cleanup_assert_success(iree_hsa_amd_memory_pool_free_raw(
      resident_pool->libhsa, allocation->base));
  iree_allocator_free(resident_pool->host_allocator, allocation);
}

iree_status_t iree_hal_amdgpu_pm4_command_buffer_resident_pool_create(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_agent_t host_agent,
    hsa_agent_t device_agent, hsa_amd_memory_pool_t resident_memory_pool,
    hsa_amd_memory_pool_t host_staging_memory_pool,
    iree_allocator_t host_allocator,
    iree_hal_amdgpu_pm4_command_buffer_resident_pool_t** out_pool) {
  IREE_ASSERT_ARGUMENT(out_pool);
  *out_pool = NULL;
  if (IREE_UNLIKELY(!libhsa)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "PM4 command-buffer resident pool HSA API table is required");
  }
  if (IREE_UNLIKELY(!resident_memory_pool.handle)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "PM4 command-buffer resident pool memory pool is required");
  }
  if (IREE_UNLIKELY(!host_staging_memory_pool.handle)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "PM4 command-buffer host staging memory pool is required");
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdgpu_pm4_command_buffer_resident_pool_t* resident_pool = NULL;
  iree_status_t status = iree_allocator_malloc(
      host_allocator, sizeof(*resident_pool), (void**)&resident_pool);
  if (iree_status_is_ok(status)) {
    memset(resident_pool, 0, sizeof(*resident_pool));
    resident_pool->libhsa = libhsa;
    resident_pool->host_agent = host_agent;
    resident_pool->device_agent = device_agent;
    resident_pool->memory_pool = resident_memory_pool;
    resident_pool->host_staging_memory_pool = host_staging_memory_pool;
    resident_pool->host_allocator = host_allocator;
    iree_slim_mutex_initialize(&resident_pool->mutex);

    size_t allocation_granule = 0;
    status = iree_hsa_amd_memory_pool_get_info(
        IREE_LIBHSA(libhsa), resident_memory_pool,
        HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_REC_GRANULE,
        &allocation_granule);
    if (iree_status_is_ok(status)) {
      if (IREE_UNLIKELY(allocation_granule == 0 ||
                        !iree_host_size_is_power_of_two(
                            (iree_host_size_t)allocation_granule))) {
        status = iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "PM4 command-buffer resident pool allocation granule %zu is "
            "invalid",
            allocation_granule);
      } else {
        resident_pool->allocation_granule =
            (iree_host_size_t)allocation_granule;
      }
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_amdgpu_host_signal_pool_initialize(
          libhsa, /*initial_capacity=*/0,
          IREE_HAL_AMDGPU_HOST_SIGNAL_POOL_BATCH_SIZE_DEFAULT, host_allocator,
          &resident_pool->copy_signal_pool);
    }
    if (iree_status_is_ok(status)) {
      size_t host_staging_allocation_granule = 0;
      status = iree_hsa_amd_memory_pool_get_info(
          IREE_LIBHSA(libhsa), host_staging_memory_pool,
          HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_REC_GRANULE,
          &host_staging_allocation_granule);
      if (iree_status_is_ok(status)) {
        if (IREE_UNLIKELY(
                host_staging_allocation_granule == 0 ||
                !iree_host_size_is_power_of_two(
                    (iree_host_size_t)host_staging_allocation_granule))) {
          status = iree_make_status(
              IREE_STATUS_FAILED_PRECONDITION,
              "PM4 command-buffer host staging allocation granule %zu is "
              "invalid",
              host_staging_allocation_granule);
        } else {
          resident_pool->host_staging_allocation_granule =
              (iree_host_size_t)host_staging_allocation_granule;
        }
      }
    }
  }

  if (iree_status_is_ok(status)) {
    *out_pool = resident_pool;
  } else if (resident_pool) {
    if (resident_pool->copy_signal_pool.libhsa) {
      iree_hal_amdgpu_host_signal_pool_deinitialize(
          &resident_pool->copy_signal_pool);
    }
    iree_slim_mutex_deinitialize(&resident_pool->mutex);
    iree_allocator_free(host_allocator, resident_pool);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_amdgpu_pm4_command_buffer_resident_pool_trim(
    iree_hal_amdgpu_pm4_command_buffer_resident_pool_t* resident_pool) {
  if (!resident_pool) return;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_slim_mutex_lock(&resident_pool->mutex);
  iree_hal_amdgpu_pm4_command_buffer_resident_allocation_t* allocation =
      resident_pool->free_list;
  resident_pool->free_list = NULL;
  iree_hal_amdgpu_pm4_command_buffer_resident_allocation_t* staging_allocation =
      resident_pool->staging_free_list;
  resident_pool->staging_free_list = NULL;
  iree_slim_mutex_unlock(&resident_pool->mutex);

  while (allocation) {
    iree_hal_amdgpu_pm4_command_buffer_resident_allocation_t* next_allocation =
        allocation->next;
    iree_hal_amdgpu_pm4_command_buffer_resident_allocation_free(resident_pool,
                                                                allocation);
    allocation = next_allocation;
  }
  while (staging_allocation) {
    iree_hal_amdgpu_pm4_command_buffer_resident_allocation_t* next_allocation =
        staging_allocation->next;
    iree_hal_amdgpu_pm4_command_buffer_resident_allocation_free(
        resident_pool, staging_allocation);
    staging_allocation = next_allocation;
  }

  IREE_TRACE_ZONE_END(z0);
}

void iree_hal_amdgpu_pm4_command_buffer_resident_pool_destroy(
    iree_hal_amdgpu_pm4_command_buffer_resident_pool_t* resident_pool) {
  if (!resident_pool) return;
  IREE_TRACE_ZONE_BEGIN(z0);

  IREE_ASSERT(resident_pool->outstanding_count == 0,
              "PM4 command-buffer resident pool destroyed with %" PRIhsz
              " outstanding allocations",
              resident_pool->outstanding_count);
  IREE_ASSERT(resident_pool->outstanding_staging_count == 0,
              "PM4 command-buffer resident pool destroyed with %" PRIhsz
              " outstanding staging allocations",
              resident_pool->outstanding_staging_count);
  iree_hal_amdgpu_pm4_command_buffer_resident_pool_trim(resident_pool);
  iree_hal_amdgpu_host_signal_pool_deinitialize(
      &resident_pool->copy_signal_pool);
  iree_slim_mutex_deinitialize(&resident_pool->mutex);
  iree_allocator_t host_allocator = resident_pool->host_allocator;
  iree_allocator_free(host_allocator, resident_pool);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_amdgpu_pm4_command_buffer_round_pooled_capacity(
    iree_host_size_t required_byte_length, iree_host_size_t allocation_granule,
    iree_host_size_t* out_capacity) {
  *out_capacity = 0;
  if (IREE_UNLIKELY(required_byte_length == 0)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "PM4 command-buffer resident allocation cannot be empty");
  }
  if (IREE_UNLIKELY(!iree_host_size_checked_align(
          required_byte_length, allocation_granule, out_capacity))) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "PM4 command-buffer resident allocation size overflows");
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_pm4_command_buffer_resident_pool_acquire(
    iree_hal_amdgpu_pm4_command_buffer_resident_pool_t* resident_pool,
    iree_host_size_t required_byte_length, bool collect_timings,
    iree_hal_amdgpu_pm4_command_buffer_publish_stats_t* publish_stats,
    iree_hal_amdgpu_pm4_command_buffer_resident_allocation_t** out_allocation) {
  IREE_ASSERT_ARGUMENT(resident_pool);
  IREE_ASSERT_ARGUMENT(out_allocation);
  *out_allocation = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, required_byte_length);

  iree_host_size_t capacity = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hal_amdgpu_pm4_command_buffer_round_pooled_capacity(
          required_byte_length, resident_pool->allocation_granule, &capacity));

  iree_slim_mutex_lock(&resident_pool->mutex);
  iree_hal_amdgpu_pm4_command_buffer_resident_allocation_t** inout_link =
      &resident_pool->free_list;
  iree_hal_amdgpu_pm4_command_buffer_resident_allocation_t* allocation =
      resident_pool->free_list;
  while (allocation) {
    if (allocation->capacity >= required_byte_length) {
      *inout_link = allocation->next;
      allocation->next = NULL;
      ++resident_pool->outstanding_count;
      break;
    }
    inout_link = &allocation->next;
    allocation = allocation->next;
  }
  iree_slim_mutex_unlock(&resident_pool->mutex);

  if (allocation) {
    *out_allocation = allocation;
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  iree_status_t status = iree_allocator_malloc(
      resident_pool->host_allocator, sizeof(*allocation), (void**)&allocation);
  if (iree_status_is_ok(status)) {
    memset(allocation, 0, sizeof(*allocation));
    allocation->capacity = capacity;

    const bool should_collect_stats = collect_timings && publish_stats;
    iree_time_t time_start = should_collect_stats ? iree_time_now() : 0;
    status = iree_hsa_amd_memory_pool_allocate(
        IREE_LIBHSA(resident_pool->libhsa), resident_pool->memory_pool,
        allocation->capacity, HSA_AMD_MEMORY_POOL_EXECUTABLE_FLAG,
        (void**)&allocation->base);
    if (should_collect_stats) {
      publish_stats->resident_allocate_ns += iree_time_now() - time_start;
    }
    if (iree_status_is_ok(status)) {
      time_start = should_collect_stats ? iree_time_now() : 0;
      status = iree_hsa_amd_agents_allow_access(
          IREE_LIBHSA(resident_pool->libhsa),
          /*num_agents=*/1, &resident_pool->device_agent,
          /*flags=*/NULL, allocation->base);
      if (should_collect_stats) {
        publish_stats->resident_allow_access_ns += iree_time_now() - time_start;
      }
    }
    if (iree_status_is_ok(status) && publish_stats) {
      publish_stats->resident_allocation_count += 1;
      publish_stats->resident_allow_access_agent_count += 1;
    }
  }

  if (iree_status_is_ok(status)) {
    iree_slim_mutex_lock(&resident_pool->mutex);
    ++resident_pool->outstanding_count;
    iree_slim_mutex_unlock(&resident_pool->mutex);
    *out_allocation = allocation;
  } else if (allocation) {
    if (allocation->base) {
      status = iree_status_join(
          status, iree_hsa_amd_memory_pool_free(
                      IREE_LIBHSA(resident_pool->libhsa), allocation->base));
    }
    iree_allocator_free(resident_pool->host_allocator, allocation);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_amdgpu_pm4_command_buffer_resident_pool_release(
    iree_hal_amdgpu_pm4_command_buffer_resident_pool_t* resident_pool,
    iree_hal_amdgpu_pm4_command_buffer_resident_allocation_t* allocation) {
  IREE_ASSERT_ARGUMENT(resident_pool);
  if (!allocation) return;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, allocation->capacity);

  iree_slim_mutex_lock(&resident_pool->mutex);
  IREE_ASSERT(resident_pool->outstanding_count > 0,
              "PM4 command-buffer resident allocation released without an "
              "outstanding borrow");
  allocation->next = resident_pool->free_list;
  resident_pool->free_list = allocation;
  --resident_pool->outstanding_count;
  iree_slim_mutex_unlock(&resident_pool->mutex);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t
iree_hal_amdgpu_pm4_command_buffer_resident_pool_acquire_staging(
    iree_hal_amdgpu_pm4_command_buffer_resident_pool_t* resident_pool,
    iree_host_size_t required_byte_length, bool collect_timings,
    iree_hal_amdgpu_pm4_command_buffer_publish_stats_t* publish_stats,
    iree_hal_amdgpu_pm4_command_buffer_resident_allocation_t** out_allocation) {
  IREE_ASSERT_ARGUMENT(resident_pool);
  IREE_ASSERT_ARGUMENT(out_allocation);
  *out_allocation = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, required_byte_length);

  iree_host_size_t capacity = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdgpu_pm4_command_buffer_round_pooled_capacity(
              required_byte_length,
              resident_pool->host_staging_allocation_granule, &capacity));

  iree_slim_mutex_lock(&resident_pool->mutex);
  iree_hal_amdgpu_pm4_command_buffer_resident_allocation_t** inout_link =
      &resident_pool->staging_free_list;
  iree_hal_amdgpu_pm4_command_buffer_resident_allocation_t* allocation =
      resident_pool->staging_free_list;
  while (allocation) {
    if (allocation->capacity >= required_byte_length) {
      *inout_link = allocation->next;
      allocation->next = NULL;
      ++resident_pool->outstanding_staging_count;
      break;
    }
    inout_link = &allocation->next;
    allocation = allocation->next;
  }
  iree_slim_mutex_unlock(&resident_pool->mutex);

  if (allocation) {
    *out_allocation = allocation;
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  iree_status_t status = iree_allocator_malloc(
      resident_pool->host_allocator, sizeof(*allocation), (void**)&allocation);
  if (iree_status_is_ok(status)) {
    memset(allocation, 0, sizeof(*allocation));
    allocation->capacity = capacity;

    const bool should_collect_stats = collect_timings && publish_stats;
    iree_time_t time_start = should_collect_stats ? iree_time_now() : 0;
    status = iree_hsa_amd_memory_pool_allocate(
        IREE_LIBHSA(resident_pool->libhsa),
        resident_pool->host_staging_memory_pool, allocation->capacity,
        HSA_AMD_MEMORY_POOL_STANDARD_FLAG, (void**)&allocation->base);
    if (should_collect_stats) {
      publish_stats->host_staging_allocate_ns += iree_time_now() - time_start;
    }
    if (iree_status_is_ok(status)) {
      time_start = should_collect_stats ? iree_time_now() : 0;
      status = iree_hsa_amd_agents_allow_access(
          IREE_LIBHSA(resident_pool->libhsa),
          /*num_agents=*/1, &resident_pool->device_agent,
          /*flags=*/NULL, allocation->base);
      if (should_collect_stats) {
        publish_stats->host_staging_allow_access_ns +=
            iree_time_now() - time_start;
      }
    }
    if (iree_status_is_ok(status) && publish_stats) {
      publish_stats->host_staging_allocation_count += 1;
      publish_stats->host_staging_allow_access_agent_count += 1;
    }
  }

  if (iree_status_is_ok(status)) {
    iree_slim_mutex_lock(&resident_pool->mutex);
    ++resident_pool->outstanding_staging_count;
    iree_slim_mutex_unlock(&resident_pool->mutex);
    *out_allocation = allocation;
  } else if (allocation) {
    if (allocation->base) {
      status = iree_status_join(
          status, iree_hsa_amd_memory_pool_free(
                      IREE_LIBHSA(resident_pool->libhsa), allocation->base));
    }
    iree_allocator_free(resident_pool->host_allocator, allocation);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_amdgpu_pm4_command_buffer_resident_pool_release_staging(
    iree_hal_amdgpu_pm4_command_buffer_resident_pool_t* resident_pool,
    iree_hal_amdgpu_pm4_command_buffer_resident_allocation_t* allocation) {
  IREE_ASSERT_ARGUMENT(resident_pool);
  if (!allocation) return;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, allocation->capacity);

  iree_slim_mutex_lock(&resident_pool->mutex);
  IREE_ASSERT(resident_pool->outstanding_staging_count > 0,
              "PM4 command-buffer host staging allocation released without an "
              "outstanding borrow");
  allocation->next = resident_pool->staging_free_list;
  resident_pool->staging_free_list = allocation;
  --resident_pool->outstanding_staging_count;
  iree_slim_mutex_unlock(&resident_pool->mutex);

  IREE_TRACE_ZONE_END(z0);
}

typedef struct iree_hal_amdgpu_pm4_barrier_state_t {
  // True when a recorded barrier must be emitted before the next dispatch.
  bool pending;
  // Maximum pending acquire fence scope.
  iree_hsa_fence_scope_t acquire_scope;
  // Maximum pending release fence scope.
  iree_hsa_fence_scope_t release_scope;
} iree_hal_amdgpu_pm4_barrier_state_t;

typedef struct iree_hal_amdgpu_pm4_command_buffer_t {
  // Base HAL command-buffer resource.
  iree_hal_command_buffer_t base;
  // Host allocator used for command-buffer-owned host storage.
  iree_allocator_t host_allocator;
  // Borrowed block pool used for retained resource sets.
  iree_arena_block_pool_t* resource_set_block_pool;
  // Resource set retaining static buffers and executables unless unretained.
  iree_hal_resource_set_t* resource_set;
  // Exact first-use table for resources retained into |resource_set|.
  iree_hal_amdgpu_pm4_retained_resource_table_t retained_resources;
  // Last executable retained into |resource_set| during this recording.
  iree_hal_executable_t* last_retained_executable;
  // Pool that owns resident PM4 storage allocations.
  iree_hal_amdgpu_pm4_command_buffer_resident_pool_t* resident_pool;
  // Borrowed resident storage allocation returned to |resident_pool|.
  iree_hal_amdgpu_pm4_command_buffer_resident_allocation_t* resident_allocation;
  // Borrowed host staging allocation returned to |resident_pool|.
  iree_hal_amdgpu_pm4_command_buffer_resident_allocation_t*
      host_staging_allocation;
  // Mutex guarding nonblocking publication signal ownership.
  iree_slim_mutex_t publication_mutex;
  // Pending async-copy publication signal, or null when resident bytes no
  // longer need queue-side publication ordering.
  hsa_signal_t publication_signal IREE_GUARDED_BY(publication_mutex);
  // Number of queued AQL barriers that may still reference
  // |publication_signal|.
  uint32_t publication_reference_count IREE_GUARDED_BY(publication_mutex);
  // HSA API table used to copy materialized resident PM4 storage.
  const iree_hal_amdgpu_libhsa_t* libhsa;
  // PM4 packet-family capabilities validated for this physical device.
  iree_hal_amdgpu_vendor_packet_capability_flags_t vendor_packet_capabilities;
  // iree_hal_amdgpu_pm4_command_buffer_flag_bits_t mask.
  iree_hal_amdgpu_pm4_command_buffer_flags_t flags;
  // Physical device ordinal this command buffer was recorded for.
  uint32_t device_ordinal;
  // Recording lifecycle state.
  iree_hal_amdgpu_pm4_command_buffer_recording_state_t recording_state;
  // Pending execution/visibility barrier debt.
  iree_hal_amdgpu_pm4_barrier_state_t barrier_state;
  // Previous dispatch launch state emitted into the current IB.
  iree_hal_amdgpu_pm4_dispatch_launch_state_t previous_launch_state;
  // True once |previous_launch_state| contains a valid value.
  bool has_previous_launch_state;
  // True once the fixup-to-IB visibility barrier has been emitted.
  bool has_emitted_fixup_barrier;
  // Next HAL command ordinal assigned while recording.
  uint32_t record_command_count;
  // Profile metadata retained for iree-profile command-buffer records.
  struct {
    // Borrowed metadata registry owned by the logical device.
    iree_hal_amdgpu_profile_metadata_registry_t* metadata;
    // Session-local command-buffer identifier, or 0 when metadata is not
    // retained.
    uint64_t id;
  } profile;
  // Compact host command records accumulated while recording.
  iree_hal_amdgpu_pm4_byte_builder_t record_builder;
  // PM4 dwords materialized from compact records during end().
  iree_hal_amdgpu_pm4_dword_builder_t dword_builder;
  // Kernarg-template bytes materialized from compact records during end().
  iree_hal_amdgpu_pm4_byte_builder_t template_builder;
  // Dynamic binding fixup entries materialized from compact records during
  // end().
  iree_hal_amdgpu_pm4_fixup_entry_builder_t fixup_builder;
  // Expected resident PM4 IB dword count computed while appending records.
  uint32_t record_ib_dword_count;
  // Expected resident kernarg-template byte length computed while appending
  // records.
  iree_host_size_t record_template_byte_length;
  // Expected resident fixup entry count computed while appending records.
  uint32_t record_fixup_entry_count;
  // Finalize-time publication stats.
  iree_hal_amdgpu_pm4_command_buffer_publish_stats_t publish_stats;
  // Resident immutable PM4 indirect buffer produced by end().
  iree_hal_amdgpu_pm4_program_t program;
  // Resident kernarg template and fixup storage produced by end().
  struct {
    // Device-visible resident template and fixup plan.
    iree_hal_amdgpu_pm4_command_buffer_fixup_plan_t plan;
    // Allocated byte length of |plan.entries|.
    iree_host_size_t entry_byte_length;
  } fixup;
} iree_hal_amdgpu_pm4_command_buffer_t;

static iree_hal_amdgpu_pm4_command_buffer_t*
iree_hal_amdgpu_pm4_command_buffer_cast(iree_hal_command_buffer_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_amdgpu_pm4_command_buffer_vtable);
  return (iree_hal_amdgpu_pm4_command_buffer_t*)base_value;
}

static bool iree_hal_amdgpu_pm4_command_buffer_retains_resources(
    const iree_hal_amdgpu_pm4_command_buffer_t* command_buffer) {
  return !iree_all_bits_set(command_buffer->base.mode,
                            IREE_HAL_COMMAND_BUFFER_MODE_UNRETAINED);
}

static bool iree_hal_amdgpu_pm4_command_buffer_retains_profile_metadata(
    const iree_hal_amdgpu_pm4_command_buffer_t* command_buffer) {
  return iree_all_bits_set(
      command_buffer->base.mode,
      IREE_HAL_COMMAND_BUFFER_MODE_RETAIN_PROFILE_METADATA);
}

static void iree_hal_amdgpu_pm4_command_buffer_host_staging_reset(
    iree_hal_amdgpu_pm4_command_buffer_t* command_buffer) {
  if (!command_buffer->host_staging_allocation) return;
  iree_hal_amdgpu_pm4_command_buffer_resident_pool_release_staging(
      command_buffer->resident_pool, command_buffer->host_staging_allocation);
  command_buffer->host_staging_allocation = NULL;
}

static bool iree_hal_amdgpu_pm4_command_buffer_has_pending_publication(
    iree_hal_amdgpu_pm4_command_buffer_t* command_buffer) {
  iree_slim_mutex_lock(&command_buffer->publication_mutex);
  const bool has_pending_publication =
      !iree_hsa_signal_is_null(command_buffer->publication_signal);
  iree_slim_mutex_unlock(&command_buffer->publication_mutex);
  return has_pending_publication;
}

static void
iree_hal_amdgpu_pm4_command_buffer_release_publication_resources_locked(
    iree_hal_amdgpu_pm4_command_buffer_t* command_buffer) {
  IREE_ASSERT(!iree_hsa_signal_is_null(command_buffer->publication_signal),
              "release requires a pending publication signal");
  iree_hal_amdgpu_host_signal_pool_release(
      &command_buffer->resident_pool->copy_signal_pool,
      command_buffer->publication_signal);
  command_buffer->publication_signal = iree_hsa_signal_null();
  iree_hal_amdgpu_pm4_command_buffer_host_staging_reset(command_buffer);
}

static void iree_hal_amdgpu_pm4_command_buffer_wait_for_publication(
    iree_hal_amdgpu_pm4_command_buffer_t* command_buffer) {
  iree_slim_mutex_lock(&command_buffer->publication_mutex);
  IREE_ASSERT(command_buffer->publication_reference_count == 0,
              "PM4 command-buffer publication has %u live AQL references at "
              "destruction",
              command_buffer->publication_reference_count);
  if (!iree_hsa_signal_is_null(command_buffer->publication_signal)) {
    const hsa_signal_value_t completion_value = iree_hsa_signal_wait_scacquire(
        IREE_LIBHSA(command_buffer->libhsa), command_buffer->publication_signal,
        HSA_SIGNAL_CONDITION_LT, /*compare_value=*/1, UINT64_MAX,
        HSA_WAIT_STATE_BLOCKED);
    IREE_ASSERT(completion_value == 0,
                "PM4 command-buffer async resident copy failed with signal "
                "value %" PRId64,
                (int64_t)completion_value);
    iree_hal_amdgpu_pm4_command_buffer_release_publication_resources_locked(
        command_buffer);
  } else {
    iree_hal_amdgpu_pm4_command_buffer_host_staging_reset(command_buffer);
  }
  iree_slim_mutex_unlock(&command_buffer->publication_mutex);
}

static bool iree_hal_amdgpu_pm4_command_buffer_validates(
    const iree_hal_amdgpu_pm4_command_buffer_t* command_buffer) {
#if IREE_HAL_COMMAND_BUFFER_VALIDATION_ENABLE
  return !iree_any_bit_set(command_buffer->base.mode,
                           IREE_HAL_COMMAND_BUFFER_MODE_UNVALIDATED);
#else
  (void)command_buffer;
  return false;
#endif  // IREE_HAL_COMMAND_BUFFER_VALIDATION_ENABLE
}

static bool iree_hal_amdgpu_pm4_command_buffer_collects_finalize_timings(
    const iree_hal_amdgpu_pm4_command_buffer_t* command_buffer) {
  return iree_any_bit_set(
      command_buffer->flags,
      IREE_HAL_AMDGPU_PM4_COMMAND_BUFFER_FLAG_COLLECT_FINALIZE_TIMINGS);
}

static bool iree_hal_amdgpu_pm4_command_buffer_materializes_to_host(
    const iree_hal_amdgpu_pm4_command_buffer_t* command_buffer) {
  return iree_any_bit_set(
      command_buffer->flags,
      IREE_HAL_AMDGPU_PM4_COMMAND_BUFFER_FLAG_MATERIALIZE_TO_HOST_COPY |
          IREE_HAL_AMDGPU_PM4_COMMAND_BUFFER_FLAG_MATERIALIZE_TO_HOST_ASYNC_COPY);
}

static bool iree_hal_amdgpu_pm4_command_buffer_uses_host_async_copy(
    const iree_hal_amdgpu_pm4_command_buffer_t* command_buffer) {
  return iree_any_bit_set(
      command_buffer->flags,
      IREE_HAL_AMDGPU_PM4_COMMAND_BUFFER_FLAG_MATERIALIZE_TO_HOST_ASYNC_COPY);
}

static bool iree_hal_amdgpu_pm4_command_buffer_uses_nonblocking_publication(
    const iree_hal_amdgpu_pm4_command_buffer_t* command_buffer) {
  return iree_any_bit_set(
      command_buffer->flags,
      IREE_HAL_AMDGPU_PM4_COMMAND_BUFFER_FLAG_NONBLOCKING_PUBLICATION);
}

static iree_status_t iree_hal_amdgpu_pm4_command_buffer_ensure_resource_set(
    iree_hal_amdgpu_pm4_command_buffer_t* command_buffer) {
  if (!iree_hal_amdgpu_pm4_command_buffer_retains_resources(command_buffer) ||
      command_buffer->resource_set) {
    return iree_ok_status();
  }
  return iree_hal_resource_set_allocate(command_buffer->resource_set_block_pool,
                                        &command_buffer->resource_set);
}

static void iree_hal_amdgpu_pm4_dword_builder_initialize(
    iree_allocator_t host_allocator,
    iree_hal_amdgpu_pm4_dword_builder_t* out_builder) {
  memset(out_builder, 0, sizeof(*out_builder));
  out_builder->host_allocator = host_allocator;
  out_builder->owns_storage = true;
}

static void iree_hal_amdgpu_pm4_dword_builder_deinitialize(
    iree_hal_amdgpu_pm4_dword_builder_t* builder) {
  if (builder->owns_storage) {
    iree_allocator_free(builder->host_allocator, builder->dwords);
  }
  memset(builder, 0, sizeof(*builder));
}

static void iree_hal_amdgpu_pm4_dword_builder_borrow_storage(
    iree_hal_amdgpu_pm4_dword_builder_t* builder, uint32_t* dwords,
    uint32_t capacity) {
  if (builder->owns_storage) {
    iree_allocator_free(builder->host_allocator, builder->dwords);
  }
  builder->dwords = dwords;
  builder->dword_count = 0;
  builder->capacity = capacity;
  builder->owns_storage = false;
}

static void iree_hal_amdgpu_pm4_byte_builder_initialize(
    iree_allocator_t host_allocator,
    iree_hal_amdgpu_pm4_byte_builder_t* out_builder) {
  memset(out_builder, 0, sizeof(*out_builder));
  out_builder->host_allocator = host_allocator;
  out_builder->owns_storage = true;
}

static void iree_hal_amdgpu_pm4_byte_builder_deinitialize(
    iree_hal_amdgpu_pm4_byte_builder_t* builder) {
  if (builder->owns_storage) {
    iree_allocator_free(builder->host_allocator, builder->bytes);
  }
  memset(builder, 0, sizeof(*builder));
}

static void iree_hal_amdgpu_pm4_byte_builder_borrow_storage(
    iree_hal_amdgpu_pm4_byte_builder_t* builder, uint8_t* bytes,
    iree_host_size_t capacity) {
  if (builder->owns_storage) {
    iree_allocator_free(builder->host_allocator, builder->bytes);
  }
  builder->bytes = bytes;
  builder->length = 0;
  builder->capacity = capacity;
  builder->owns_storage = false;
}

static void iree_hal_amdgpu_pm4_fixup_entry_builder_initialize(
    iree_allocator_t host_allocator,
    iree_hal_amdgpu_pm4_fixup_entry_builder_t* out_builder) {
  memset(out_builder, 0, sizeof(*out_builder));
  out_builder->host_allocator = host_allocator;
  out_builder->owns_storage = true;
}

static void iree_hal_amdgpu_pm4_fixup_entry_builder_deinitialize(
    iree_hal_amdgpu_pm4_fixup_entry_builder_t* builder) {
  if (builder->owns_storage) {
    iree_allocator_free(builder->host_allocator, builder->entries);
  }
  memset(builder, 0, sizeof(*builder));
}

static void iree_hal_amdgpu_pm4_fixup_entry_builder_borrow_storage(
    iree_hal_amdgpu_pm4_fixup_entry_builder_t* builder,
    iree_hal_amdgpu_command_buffer_pm4_fixup_entry_t* entries,
    uint32_t capacity) {
  if (builder->owns_storage) {
    iree_allocator_free(builder->host_allocator, builder->entries);
  }
  builder->entries = entries;
  builder->count = 0;
  builder->capacity = capacity;
  builder->owns_storage = false;
}

static void iree_hal_amdgpu_pm4_retained_resource_table_initialize(
    iree_hal_amdgpu_pm4_retained_resource_table_t* table) {
  memset(table->inline_resources, 0, sizeof(table->inline_resources));
  table->resources = table->inline_resources;
  table->count = 0;
  table->capacity = IREE_ARRAYSIZE(table->inline_resources);
}

static void iree_hal_amdgpu_pm4_retained_resource_table_deinitialize(
    iree_allocator_t host_allocator,
    iree_hal_amdgpu_pm4_retained_resource_table_t* table) {
  if (table->resources != table->inline_resources) {
    iree_allocator_free(host_allocator, table->resources);
  }
  table->resources = NULL;
  table->count = 0;
  table->capacity = 0;
  memset(table->inline_resources, 0, sizeof(table->inline_resources));
}

static iree_host_size_t iree_hal_amdgpu_pm4_retained_resource_table_hash(
    iree_hal_resource_t* resource) {
  uint64_t value = (uint64_t)((uintptr_t)resource >> 4);
  value ^= value >> 33;
  value *= UINT64_C(0xff51afd7ed558ccd);
  value ^= value >> 33;
  value *= UINT64_C(0xc4ceb9fe1a85ec53);
  value ^= value >> 33;
  return (iree_host_size_t)value;
}

static iree_host_size_t iree_hal_amdgpu_pm4_retained_resource_table_find_slot(
    const iree_hal_amdgpu_pm4_retained_resource_table_t* table,
    iree_hal_resource_t* resource, bool* out_found) {
  const iree_host_size_t mask = table->capacity - 1;
  iree_host_size_t slot =
      iree_hal_amdgpu_pm4_retained_resource_table_hash(resource) & mask;
  while (true) {
    iree_hal_resource_t* existing_resource = table->resources[slot];
    if (existing_resource == resource) {
      *out_found = true;
      return slot;
    }
    if (!existing_resource) {
      *out_found = false;
      return slot;
    }
    slot = (slot + 1) & mask;
  }
}

static iree_status_t iree_hal_amdgpu_pm4_retained_resource_table_reserve(
    iree_allocator_t host_allocator,
    iree_hal_amdgpu_pm4_retained_resource_table_t* table,
    iree_host_size_t required_count) {
  if (required_count <= table->capacity - table->capacity / 4) {
    return iree_ok_status();
  }

  iree_host_size_t new_capacity = table->capacity * 2;
  while (required_count > new_capacity - new_capacity / 4) {
    if (IREE_UNLIKELY(new_capacity > IREE_HOST_SIZE_MAX / 2)) {
      return iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "PM4 command-buffer retained resource table exceeds host size");
    }
    new_capacity *= 2;
  }

  iree_hal_resource_t** new_resources = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(host_allocator, new_capacity,
                                                   sizeof(*new_resources),
                                                   (void**)&new_resources));
  iree_hal_amdgpu_pm4_retained_resource_table_t new_table = {
      .resources = new_resources,
      .capacity = new_capacity,
  };
  for (iree_host_size_t i = 0; i < table->capacity; ++i) {
    iree_hal_resource_t* resource = table->resources[i];
    if (!resource) continue;
    bool found = false;
    const iree_host_size_t slot =
        iree_hal_amdgpu_pm4_retained_resource_table_find_slot(&new_table,
                                                              resource, &found);
    IREE_ASSERT(!found);
    new_resources[slot] = resource;
    ++new_table.count;
  }
  IREE_ASSERT(new_table.count == table->count);
  if (table->resources != table->inline_resources) {
    iree_allocator_free(host_allocator, table->resources);
  }
  table->resources = new_resources;
  table->capacity = new_capacity;
  return iree_ok_status();
}

static bool iree_hal_amdgpu_pm4_retained_resource_table_contains(
    const iree_hal_amdgpu_pm4_retained_resource_table_t* table,
    iree_hal_resource_t* resource) {
  bool found = false;
  iree_hal_amdgpu_pm4_retained_resource_table_find_slot(table, resource,
                                                        &found);
  return found;
}

static void iree_hal_amdgpu_pm4_retained_resource_table_insert_prepared(
    iree_hal_amdgpu_pm4_retained_resource_table_t* table,
    iree_hal_resource_t* resource) {
  bool found = false;
  const iree_host_size_t slot =
      iree_hal_amdgpu_pm4_retained_resource_table_find_slot(table, resource,
                                                            &found);
  IREE_ASSERT(!found);
  table->resources[slot] = resource;
  ++table->count;
}

static iree_status_t iree_hal_amdgpu_pm4_dword_builder_reserve(
    iree_hal_amdgpu_pm4_dword_builder_t* builder, uint32_t required_capacity) {
  if (required_capacity <= builder->capacity) return iree_ok_status();
  if (IREE_UNLIKELY(!builder->owns_storage)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "PM4 command-buffer resident program storage is undersized");
  }

  uint32_t new_capacity = builder->capacity ? builder->capacity : 256u;
  while (new_capacity < required_capacity) {
    if (IREE_UNLIKELY(new_capacity >
                      IREE_HAL_AMDGPU_PM4_IB_MAX_DWORD_COUNT / 2u)) {
      new_capacity = IREE_HAL_AMDGPU_PM4_IB_MAX_DWORD_COUNT;
      break;
    }
    new_capacity *= 2u;
  }
  if (IREE_UNLIKELY(new_capacity < required_capacity)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "PM4 command buffer requires %u dwords, exceeding PM4-IB maximum %u",
        required_capacity, IREE_HAL_AMDGPU_PM4_IB_MAX_DWORD_COUNT);
  }

  uint32_t* new_dwords = builder->dwords;
  IREE_RETURN_IF_ERROR(
      iree_allocator_realloc_array(builder->host_allocator, new_capacity,
                                   sizeof(*new_dwords), (void**)&new_dwords));
  builder->dwords = new_dwords;
  builder->capacity = new_capacity;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_pm4_dword_builder_append(
    iree_hal_amdgpu_pm4_dword_builder_t* builder, uint32_t dword_count,
    uint32_t** out_dwords) {
  *out_dwords = NULL;
  if (IREE_UNLIKELY(dword_count > IREE_HAL_AMDGPU_PM4_IB_MAX_DWORD_COUNT -
                                      builder->dword_count)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "PM4 command buffer requires more than the PM4-IB maximum %u dwords",
        IREE_HAL_AMDGPU_PM4_IB_MAX_DWORD_COUNT);
  }
  const uint32_t required_capacity = builder->dword_count + dword_count;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_pm4_dword_builder_reserve(builder, required_capacity));
  *out_dwords = &builder->dwords[builder->dword_count];
  builder->dword_count = required_capacity;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_pm4_byte_builder_reserve(
    iree_hal_amdgpu_pm4_byte_builder_t* builder,
    iree_host_size_t required_capacity) {
  if (required_capacity <= builder->capacity) return iree_ok_status();
  if (IREE_UNLIKELY(!builder->owns_storage)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "PM4 command-buffer resident template storage is undersized");
  }

  iree_host_size_t new_capacity = builder->capacity ? builder->capacity : 4096;
  while (new_capacity < required_capacity) {
    if (IREE_UNLIKELY(new_capacity > UINT32_MAX / 2u)) {
      new_capacity = UINT32_MAX;
      break;
    }
    new_capacity *= 2u;
  }
  if (IREE_UNLIKELY(new_capacity < required_capacity)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "PM4 command-buffer dynamic template storage requires %" PRIhsz
        " bytes, exceeding uint32_t fixup offsets",
        required_capacity);
  }

  uint8_t* new_bytes = builder->bytes;
  IREE_RETURN_IF_ERROR(iree_allocator_realloc(
      builder->host_allocator, new_capacity, (void**)&new_bytes));
  builder->bytes = new_bytes;
  builder->capacity = new_capacity;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_pm4_byte_builder_append_aligned(
    iree_hal_amdgpu_pm4_byte_builder_t* builder, iree_host_size_t alignment,
    iree_host_size_t byte_length, uint32_t* out_offset, uint8_t** out_bytes) {
  *out_offset = 0;
  *out_bytes = NULL;
  if (IREE_UNLIKELY(!iree_host_size_is_power_of_two(alignment))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "PM4 template alignment must be a power-of-two");
  }
  const iree_host_size_t aligned_length =
      iree_host_align(builder->length, alignment);
  if (IREE_UNLIKELY(aligned_length > UINT32_MAX ||
                    byte_length > UINT32_MAX - aligned_length)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "PM4 command-buffer dynamic template offset exceeds uint32_t storage");
  }
  const iree_host_size_t required_capacity = aligned_length + byte_length;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_pm4_byte_builder_reserve(builder, required_capacity));
  if (aligned_length > builder->length) {
    memset(builder->bytes + builder->length, 0,
           aligned_length - builder->length);
  }
  uint8_t* bytes = builder->bytes + aligned_length;
  memset(bytes, 0, byte_length);
  builder->length = required_capacity;
  *out_offset = (uint32_t)aligned_length;
  *out_bytes = bytes;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_pm4_byte_builder_append_record(
    iree_hal_amdgpu_pm4_byte_builder_t* builder, iree_host_size_t byte_length,
    uint8_t** out_bytes) {
  *out_bytes = NULL;
  if (IREE_UNLIKELY(byte_length > UINT32_MAX - builder->length)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "PM4 command-buffer host record storage exceeds uint32_t offsets");
  }
  const iree_host_size_t required_capacity = builder->length + byte_length;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_pm4_byte_builder_reserve(builder, required_capacity));
  *out_bytes = builder->bytes + builder->length;
  builder->length = required_capacity;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_pm4_fixup_entry_builder_reserve(
    iree_hal_amdgpu_pm4_fixup_entry_builder_t* builder,
    uint32_t required_capacity) {
  if (required_capacity <= builder->capacity) return iree_ok_status();
  if (IREE_UNLIKELY(!builder->owns_storage)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "PM4 command-buffer resident fixup storage is undersized");
  }

  uint32_t new_capacity = builder->capacity ? builder->capacity : 256u;
  while (new_capacity < required_capacity) {
    if (IREE_UNLIKELY(new_capacity > UINT32_MAX / 2u)) {
      new_capacity = UINT32_MAX;
      break;
    }
    new_capacity *= 2u;
  }
  if (IREE_UNLIKELY(new_capacity < required_capacity)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "PM4 command-buffer fixup entry count exceeds uint32_t storage");
  }

  iree_hal_amdgpu_command_buffer_pm4_fixup_entry_t* new_entries =
      builder->entries;
  IREE_RETURN_IF_ERROR(
      iree_allocator_realloc_array(builder->host_allocator, new_capacity,
                                   sizeof(*new_entries), (void**)&new_entries));
  builder->entries = new_entries;
  builder->capacity = new_capacity;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_pm4_fixup_entry_builder_append(
    iree_hal_amdgpu_pm4_fixup_entry_builder_t* builder,
    iree_hal_amdgpu_command_buffer_pm4_fixup_entry_t entry) {
  if (IREE_UNLIKELY(builder->count == UINT32_MAX)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "PM4 command-buffer fixup entry count exceeds uint32_t storage");
  }
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_fixup_entry_builder_reserve(
      builder, builder->count + 1u));
  builder->entries[builder->count++] = entry;
  return iree_ok_status();
}

static void iree_hal_amdgpu_pm4_recording_builders_initialize(
    iree_hal_amdgpu_pm4_command_buffer_t* command_buffer) {
  iree_hal_amdgpu_pm4_retained_resource_table_initialize(
      &command_buffer->retained_resources);
  iree_hal_amdgpu_pm4_byte_builder_initialize(command_buffer->host_allocator,
                                              &command_buffer->record_builder);
  iree_hal_amdgpu_pm4_dword_builder_initialize(command_buffer->host_allocator,
                                               &command_buffer->dword_builder);
  iree_hal_amdgpu_pm4_byte_builder_initialize(
      command_buffer->host_allocator, &command_buffer->template_builder);
  iree_hal_amdgpu_pm4_fixup_entry_builder_initialize(
      command_buffer->host_allocator, &command_buffer->fixup_builder);
}

static void iree_hal_amdgpu_pm4_recording_builders_deinitialize(
    iree_hal_amdgpu_pm4_command_buffer_t* command_buffer) {
  iree_hal_amdgpu_pm4_retained_resource_table_deinitialize(
      command_buffer->host_allocator, &command_buffer->retained_resources);
  iree_hal_amdgpu_pm4_fixup_entry_builder_deinitialize(
      &command_buffer->fixup_builder);
  iree_hal_amdgpu_pm4_byte_builder_deinitialize(
      &command_buffer->template_builder);
  iree_hal_amdgpu_pm4_dword_builder_deinitialize(
      &command_buffer->dword_builder);
  iree_hal_amdgpu_pm4_byte_builder_deinitialize(
      &command_buffer->record_builder);
  if (!iree_hal_amdgpu_pm4_command_buffer_has_pending_publication(
          command_buffer)) {
    iree_hal_amdgpu_pm4_command_buffer_host_staging_reset(command_buffer);
  }
  command_buffer->record_ib_dword_count = 0;
  command_buffer->record_template_byte_length = 0;
  command_buffer->record_fixup_entry_count = 0;
}

//===----------------------------------------------------------------------===//
// PM4 packet emission
//===----------------------------------------------------------------------===//

static void iree_hal_amdgpu_pm4_barrier_state_accumulate(
    iree_hal_amdgpu_pm4_barrier_state_t* barrier_state,
    iree_hsa_fence_scope_t acquire_scope,
    iree_hsa_fence_scope_t release_scope) {
  barrier_state->pending = true;
  barrier_state->acquire_scope = iree_hal_amdgpu_pm4_max_fence_scope(
      barrier_state->acquire_scope, acquire_scope);
  barrier_state->release_scope = iree_hal_amdgpu_pm4_max_fence_scope(
      barrier_state->release_scope, release_scope);
}

static void iree_hal_amdgpu_pm4_barrier_state_reset(
    iree_hal_amdgpu_pm4_barrier_state_t* barrier_state) {
  *barrier_state = (iree_hal_amdgpu_pm4_barrier_state_t){0};
}

static iree_status_t iree_hal_amdgpu_pm4_dword_builder_emit_barrier(
    iree_hal_amdgpu_pm4_dword_builder_t* builder,
    iree_hal_amdgpu_vendor_packet_capability_flags_t capabilities,
    iree_hal_amdgpu_pm4_barrier_flags_t barrier_flags,
    iree_hsa_fence_scope_t acquire_scope,
    iree_hsa_fence_scope_t release_scope) {
  const uint32_t barrier_dword_count =
      iree_hal_amdgpu_pm4_barrier_dword_count_gfx10(
          capabilities, barrier_flags, acquire_scope, release_scope);
  if (IREE_UNLIKELY(barrier_dword_count == 0)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "PM4 command-buffer barrier cannot be emitted with capabilities "
        "0x%08" PRIx32 ", flags 0x%08" PRIx32,
        capabilities, barrier_flags);
  }

  uint32_t* barrier_dwords = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_dword_builder_append(
      builder, barrier_dword_count, &barrier_dwords));
  uint32_t emitted_dword_count = 0;
  if (IREE_UNLIKELY(!iree_hal_amdgpu_pm4_barrier_emit_gfx10(
                        capabilities, barrier_flags, acquire_scope,
                        release_scope, barrier_dword_count, barrier_dwords,
                        &emitted_dword_count) ||
                    emitted_dword_count != barrier_dword_count)) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "PM4 command-buffer barrier emission changed size");
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_pm4_dword_builder_emit_dispatch_setup(
    iree_hal_amdgpu_pm4_dword_builder_t* builder,
    const uint32_t
        source_dwords[IREE_HAL_AMDGPU_PM4_DISPATCH_SETUP_DWORD_COUNT],
    uint32_t setup_dword_count) {
  if (IREE_UNLIKELY(setup_dword_count !=
                    IREE_HAL_AMDGPU_PM4_DISPATCH_SETUP_DWORD_COUNT)) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "PM4 dispatch setup dword count is invalid");
  }
  uint32_t* target_dwords = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_dword_builder_append(
      builder, setup_dword_count, &target_dwords));
  memcpy(target_dwords, source_dwords,
         setup_dword_count * sizeof(source_dwords[0]));
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_pm4_dword_builder_emit_user_data(
    iree_hal_amdgpu_pm4_dword_builder_t* builder,
    const iree_hal_amdgpu_pm4_dispatch_launch_state_t* launch_state,
    uint64_t kernarg_address) {
  if (launch_state->user_data_dword_count == 0) return iree_ok_status();

  const uint32_t user_data_dword_count =
      2u + launch_state->user_data_dword_count;
  uint32_t* user_data_dwords = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_dword_builder_append(
      builder, user_data_dword_count, &user_data_dwords));
  uint32_t emitted_dword_count = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_dispatch_emit_user_data(
      launch_state, kernarg_address, user_data_dword_count, user_data_dwords,
      &emitted_dword_count));
  if (IREE_UNLIKELY(emitted_dword_count != user_data_dword_count)) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "PM4 dispatch user-data emission changed size");
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_pm4_dword_builder_emit_dispatch_direct(
    iree_hal_amdgpu_pm4_dword_builder_t* builder,
    const uint32_t workgroup_count[3], uint32_t dispatch_initiator) {
  uint32_t* dispatch_dwords = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_dword_builder_append(
      builder, IREE_HAL_AMDGPU_PM4_DISPATCH_DIRECT_DWORD_COUNT,
      &dispatch_dwords));
  dispatch_dwords[0] = iree_hal_amdgpu_pm4_make_compute_header(
      IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_DISPATCH_DIRECT,
      IREE_HAL_AMDGPU_PM4_DISPATCH_DIRECT_DWORD_COUNT);
  dispatch_dwords[1] = workgroup_count[0];
  dispatch_dwords[2] = workgroup_count[1];
  dispatch_dwords[3] = workgroup_count[2];
  dispatch_dwords[4] = dispatch_initiator;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Resident storage publication
//===----------------------------------------------------------------------===//

static void iree_hal_amdgpu_pm4_command_buffer_fixup_reset(
    iree_hal_amdgpu_pm4_command_buffer_t* command_buffer) {
  memset(&command_buffer->fixup, 0, sizeof(command_buffer->fixup));
}

static iree_status_t iree_hal_amdgpu_pm4_align_host_size(
    iree_host_size_t value, iree_host_size_t alignment,
    iree_host_size_t* out_aligned_value) {
  *out_aligned_value = 0;
  if (IREE_UNLIKELY(!iree_host_size_is_power_of_two(alignment))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "PM4 resident alignment must be a power-of-two");
  }
  if (IREE_UNLIKELY(value > UINTPTR_MAX - (alignment - 1))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "PM4 resident layout alignment overflows");
  }
  *out_aligned_value = iree_host_align(value, alignment);
  return iree_ok_status();
}

static iree_status_t
iree_hal_amdgpu_pm4_command_buffer_allocate_host_staging_storage(
    iree_hal_amdgpu_pm4_command_buffer_t* command_buffer,
    uint32_t resident_dword_count, iree_host_size_t program_byte_length,
    iree_host_size_t template_offset, iree_host_size_t fixup_offset,
    iree_host_size_t entry_byte_length) {
  iree_host_size_t total_byte_length = 0;
  if (!iree_host_size_checked_add(fixup_offset, entry_byte_length,
                                  &total_byte_length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "PM4 host staging allocation size overflows");
  }
  command_buffer->publish_stats.host_staging_bytes = total_byte_length;

  const bool collect_timings =
      iree_hal_amdgpu_pm4_command_buffer_collects_finalize_timings(
          command_buffer);
  iree_hal_amdgpu_pm4_command_buffer_resident_allocation_t* allocation = NULL;
  iree_status_t status =
      iree_hal_amdgpu_pm4_command_buffer_resident_pool_acquire_staging(
          command_buffer->resident_pool, total_byte_length, collect_timings,
          &command_buffer->publish_stats, &allocation);
  if (iree_status_is_ok(status)) {
    command_buffer->host_staging_allocation = allocation;
    uint8_t* staging_base = allocation->base;
    iree_hal_amdgpu_pm4_dword_builder_borrow_storage(
        &command_buffer->dword_builder, (uint32_t*)staging_base,
        resident_dword_count);
    iree_hal_amdgpu_pm4_byte_builder_borrow_storage(
        &command_buffer->template_builder, staging_base + template_offset,
        command_buffer->record_template_byte_length);
    iree_hal_amdgpu_pm4_fixup_entry_builder_borrow_storage(
        &command_buffer->fixup_builder,
        entry_byte_length > 0
            ? (iree_hal_amdgpu_command_buffer_pm4_fixup_entry_t*)(staging_base +
                                                                  fixup_offset)
            : NULL,
        command_buffer->record_fixup_entry_count);
  }
  return status;
}

static iree_status_t
iree_hal_amdgpu_pm4_command_buffer_allocate_resident_storage(
    iree_hal_amdgpu_pm4_command_buffer_t* command_buffer,
    uint32_t resident_dword_count) {
  if (IREE_UNLIKELY(resident_dword_count == 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "PM4 command-buffer resident program is empty");
  }
  if (IREE_UNLIKELY(command_buffer->record_template_byte_length == 0 &&
                    command_buffer->record_fixup_entry_count != 0)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "PM4 command-buffer fixup entries require target storage");
  }

  iree_host_size_t program_byte_length = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      0, &program_byte_length,
      IREE_STRUCT_FIELD(resident_dword_count, uint32_t, NULL)));
  command_buffer->publish_stats.program_bytes = program_byte_length;
  command_buffer->publish_stats.template_bytes =
      command_buffer->record_template_byte_length;

  iree_host_size_t template_offset = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_align_host_size(
      program_byte_length, iree_max_align_t, &template_offset));

  iree_host_size_t template_end = 0;
  if (!iree_host_size_checked_add(template_offset,
                                  command_buffer->record_template_byte_length,
                                  &template_end)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "PM4 resident template layout overflows");
  }
  iree_host_size_t fixup_offset = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_align_host_size(
      template_end,
      iree_alignof(iree_hal_amdgpu_command_buffer_pm4_fixup_entry_t),
      &fixup_offset));

  iree_host_size_t entry_byte_length = 0;
  if (command_buffer->record_fixup_entry_count > 0) {
    IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
        0, &entry_byte_length,
        IREE_STRUCT_FIELD(command_buffer->record_fixup_entry_count,
                          iree_hal_amdgpu_command_buffer_pm4_fixup_entry_t,
                          NULL)));
  }
  command_buffer->publish_stats.fixup_entry_bytes = entry_byte_length;

  iree_host_size_t total_byte_length = 0;
  if (!iree_host_size_checked_add(fixup_offset, entry_byte_length,
                                  &total_byte_length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "PM4 resident allocation size overflows");
  }
  command_buffer->publish_stats.resident_bytes = total_byte_length;

  const bool collect_timings =
      iree_hal_amdgpu_pm4_command_buffer_collects_finalize_timings(
          command_buffer);
  iree_hal_amdgpu_pm4_command_buffer_resident_allocation_t*
      resident_allocation = NULL;
  iree_status_t status =
      iree_hal_amdgpu_pm4_command_buffer_resident_pool_acquire(
          command_buffer->resident_pool, total_byte_length, collect_timings,
          &command_buffer->publish_stats, &resident_allocation);
  uint8_t* resident_base =
      iree_status_is_ok(status) ? resident_allocation->base : NULL;

  if (iree_status_is_ok(status)) {
    command_buffer->resident_allocation = resident_allocation;
    command_buffer->program.libhsa = command_buffer->libhsa;
    command_buffer->program.memory_pool =
        command_buffer->resident_pool->memory_pool;
    command_buffer->program.dwords = (uint32_t*)resident_base;
    command_buffer->program.dword_count = resident_dword_count;
    command_buffer->program.byte_length = total_byte_length;
    command_buffer->fixup.plan.entries =
        entry_byte_length > 0
            ? (const iree_hal_amdgpu_command_buffer_pm4_fixup_entry_t*)(resident_base +
                                                                        fixup_offset)
            : NULL;
    command_buffer->fixup.plan.entry_count =
        command_buffer->record_fixup_entry_count;
    command_buffer->fixup.plan.reserved0 = 0;
    command_buffer->fixup.plan.target_base =
        command_buffer->record_template_byte_length > 0
            ? resident_base + template_offset
            : NULL;
    command_buffer->fixup.plan.target_byte_length =
        command_buffer->record_template_byte_length;
    command_buffer->fixup.entry_byte_length = entry_byte_length;
    if (iree_hal_amdgpu_pm4_command_buffer_uses_host_async_copy(
            command_buffer)) {
      status = iree_hal_amdgpu_pm4_command_buffer_allocate_host_staging_storage(
          command_buffer, resident_dword_count, program_byte_length,
          template_offset, fixup_offset, entry_byte_length);
    } else if (iree_hal_amdgpu_pm4_command_buffer_materializes_to_host(
                   command_buffer)) {
      status = iree_hal_amdgpu_pm4_dword_builder_reserve(
          &command_buffer->dword_builder, resident_dword_count);
      if (iree_status_is_ok(status)) {
        status = iree_hal_amdgpu_pm4_byte_builder_reserve(
            &command_buffer->template_builder,
            command_buffer->record_template_byte_length);
      }
      if (iree_status_is_ok(status)) {
        status = iree_hal_amdgpu_pm4_fixup_entry_builder_reserve(
            &command_buffer->fixup_builder,
            command_buffer->record_fixup_entry_count);
      }
    } else {
      iree_hal_amdgpu_pm4_dword_builder_borrow_storage(
          &command_buffer->dword_builder, (uint32_t*)resident_base,
          resident_dword_count);
      iree_hal_amdgpu_pm4_byte_builder_borrow_storage(
          &command_buffer->template_builder,
          command_buffer->fixup.plan.target_base,
          command_buffer->record_template_byte_length);
      iree_hal_amdgpu_pm4_fixup_entry_builder_borrow_storage(
          &command_buffer->fixup_builder,
          (iree_hal_amdgpu_command_buffer_pm4_fixup_entry_t*)
              command_buffer->fixup.plan.entries,
          command_buffer->record_fixup_entry_count);
    }
  }

  if (!iree_status_is_ok(status) && resident_allocation) {
    iree_hal_amdgpu_pm4_command_buffer_resident_pool_release(
        command_buffer->resident_pool, resident_allocation);
    command_buffer->resident_allocation = NULL;
    memset(&command_buffer->program, 0, sizeof(command_buffer->program));
    memset(&command_buffer->fixup, 0, sizeof(command_buffer->fixup));
  }
  return status;
}

static iree_status_t
iree_hal_amdgpu_pm4_command_buffer_copy_materialized_segment_sync(
    iree_hal_amdgpu_pm4_command_buffer_t* command_buffer, void* target,
    const void* source, iree_host_size_t byte_length) {
  if (byte_length == 0) return iree_ok_status();
  IREE_RETURN_IF_ERROR(iree_hsa_memory_copy(IREE_LIBHSA(command_buffer->libhsa),
                                            target, source, byte_length));
  command_buffer->publish_stats.resident_copy_bytes += byte_length;
  return iree_ok_status();
}

static iree_status_t
iree_hal_amdgpu_pm4_command_buffer_launch_materialized_segment_async(
    iree_hal_amdgpu_pm4_command_buffer_t* command_buffer, void* target,
    const void* source, iree_host_size_t byte_length,
    hsa_signal_t* out_completion_signal) {
  IREE_ASSERT_ARGUMENT(out_completion_signal);
  *out_completion_signal = iree_hsa_signal_null();
  if (byte_length == 0) return iree_ok_status();

  hsa_signal_t completion_signal = {0};
  iree_status_t status = iree_hal_amdgpu_host_signal_pool_acquire(
      &command_buffer->resident_pool->copy_signal_pool, /*initial_value=*/1,
      &completion_signal);
  if (iree_status_is_ok(status)) {
    status = iree_hsa_amd_memory_async_copy(
        IREE_LIBHSA(command_buffer->libhsa), target,
        command_buffer->resident_pool->device_agent, source,
        command_buffer->resident_pool->host_agent, byte_length,
        /*num_dep_signals=*/0, /*dep_signals=*/NULL, completion_signal);
  }
  if (iree_status_is_ok(status)) {
    *out_completion_signal = completion_signal;
  } else if (completion_signal.handle) {
    iree_hal_amdgpu_host_signal_pool_release(
        &command_buffer->resident_pool->copy_signal_pool, completion_signal);
  }
  return status;
}

static iree_status_t
iree_hal_amdgpu_pm4_command_buffer_copy_materialized_segment_async(
    iree_hal_amdgpu_pm4_command_buffer_t* command_buffer, void* target,
    const void* source, iree_host_size_t byte_length) {
  hsa_signal_t completion_signal = {0};
  iree_status_t status =
      iree_hal_amdgpu_pm4_command_buffer_launch_materialized_segment_async(
          command_buffer, target, source, byte_length, &completion_signal);
  if (iree_status_is_ok(status) && completion_signal.handle) {
    const hsa_signal_value_t completion_value = iree_hsa_signal_wait_scacquire(
        IREE_LIBHSA(command_buffer->libhsa), completion_signal,
        HSA_SIGNAL_CONDITION_LT, /*compare_value=*/1, UINT64_MAX,
        HSA_WAIT_STATE_BLOCKED);
    if (IREE_UNLIKELY(completion_value != 0)) {
      status = iree_make_status(IREE_STATUS_INTERNAL,
                                "PM4 command-buffer async resident copy failed "
                                "with signal value %" PRId64,
                                (int64_t)completion_value);
    }
  }
  if (completion_signal.handle) {
    iree_hal_amdgpu_host_signal_pool_release(
        &command_buffer->resident_pool->copy_signal_pool, completion_signal);
  }
  if (iree_status_is_ok(status)) {
    command_buffer->publish_stats.resident_copy_bytes += byte_length;
  }
  return status;
}

static iree_status_t
iree_hal_amdgpu_pm4_command_buffer_copy_materialized_segment(
    iree_hal_amdgpu_pm4_command_buffer_t* command_buffer, void* target,
    const void* source, iree_host_size_t byte_length) {
  if (iree_hal_amdgpu_pm4_command_buffer_uses_host_async_copy(command_buffer)) {
    return iree_hal_amdgpu_pm4_command_buffer_copy_materialized_segment_async(
        command_buffer, target, source, byte_length);
  }
  return iree_hal_amdgpu_pm4_command_buffer_copy_materialized_segment_sync(
      command_buffer, target, source, byte_length);
}

static iree_status_t
iree_hal_amdgpu_pm4_command_buffer_copy_materialized_storage(
    iree_hal_amdgpu_pm4_command_buffer_t* command_buffer) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (iree_hal_amdgpu_pm4_command_buffer_uses_host_async_copy(command_buffer)) {
    iree_status_t status = iree_ok_status();
    if (iree_hal_amdgpu_pm4_command_buffer_uses_nonblocking_publication(
            command_buffer)) {
      hsa_signal_t completion_signal = {0};
      status =
          iree_hal_amdgpu_pm4_command_buffer_launch_materialized_segment_async(
              command_buffer, command_buffer->program.dwords,
              command_buffer->host_staging_allocation->base,
              command_buffer->program.byte_length, &completion_signal);
      if (iree_status_is_ok(status) && completion_signal.handle) {
        iree_slim_mutex_lock(&command_buffer->publication_mutex);
        IREE_ASSERT(iree_hsa_signal_is_null(command_buffer->publication_signal),
                    "PM4 command buffer already has a pending publication");
        IREE_ASSERT(command_buffer->publication_reference_count == 0,
                    "PM4 command buffer already has pending publication "
                    "references");
        command_buffer->publication_signal = completion_signal;
        iree_slim_mutex_unlock(&command_buffer->publication_mutex);
        command_buffer->publish_stats.resident_copy_bytes +=
            command_buffer->program.byte_length;
      }
    } else {
      status =
          iree_hal_amdgpu_pm4_command_buffer_copy_materialized_segment_async(
              command_buffer, command_buffer->program.dwords,
              command_buffer->host_staging_allocation->base,
              command_buffer->program.byte_length);
    }
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  const iree_host_size_t program_byte_length =
      (iree_host_size_t)command_buffer->dword_builder.dword_count *
      sizeof(command_buffer->dword_builder.dwords[0]);
  iree_status_t status =
      iree_hal_amdgpu_pm4_command_buffer_copy_materialized_segment(
          command_buffer, command_buffer->program.dwords,
          command_buffer->dword_builder.dwords, program_byte_length);
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_pm4_command_buffer_copy_materialized_segment(
        command_buffer, command_buffer->fixup.plan.target_base,
        command_buffer->template_builder.bytes,
        command_buffer->template_builder.length);
  }
  const iree_host_size_t fixup_entry_byte_length =
      (iree_host_size_t)command_buffer->fixup_builder.count *
      sizeof(command_buffer->fixup_builder.entries[0]);
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_pm4_command_buffer_copy_materialized_segment(
        command_buffer,
        (iree_hal_amdgpu_command_buffer_pm4_fixup_entry_t*)
            command_buffer->fixup.plan.entries,
        command_buffer->fixup_builder.entries, fixup_entry_byte_length);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static uint64_t iree_hal_amdgpu_pm4_command_buffer_host_record_bytes(
    const iree_hal_amdgpu_pm4_command_buffer_t* command_buffer) {
  return command_buffer->record_builder.length;
}

//===----------------------------------------------------------------------===//
// Dispatch recording
//===----------------------------------------------------------------------===//

static bool iree_hal_amdgpu_dispatch_config_has_workgroup_size_override(
    const iree_hal_dispatch_config_t config) {
  return config.workgroup_size[0] || config.workgroup_size[1] ||
         config.workgroup_size[2];
}

static iree_status_t iree_hal_amdgpu_pm4_command_buffer_check_dispatch_flags(
    iree_hal_dispatch_flags_t flags) {
  if (iree_hal_dispatch_uses_indirect_arguments(flags) ||
      iree_hal_dispatch_uses_indirect_parameters(flags)) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "PM4 command-buffer indirect dispatch is not "
                            "implemented");
  }
  const iree_hal_dispatch_flags_t supported_flags =
      IREE_HAL_DISPATCH_FLAG_ALLOW_INLINE_EXECUTION;
  if (IREE_UNLIKELY(iree_any_bit_set(flags, ~supported_flags))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported PM4 dispatch flags: 0x%" PRIx64,
                            flags);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_pm4_command_buffer_validate_dispatch_shape(
    const iree_hal_amdgpu_executable_dispatch_descriptor_t* descriptor,
    const iree_hal_dispatch_config_t config) {
  if (iree_hal_amdgpu_dispatch_config_has_workgroup_size_override(config)) {
    for (iree_host_size_t i = 0; i < 3; ++i) {
      if (IREE_UNLIKELY(!config.workgroup_size[i])) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "dispatch workgroup size override must specify all dimensions");
      }
      if (IREE_UNLIKELY(config.workgroup_size[i] > UINT16_MAX)) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "dispatch workgroup size override dimension %" PRIhsz
            " value %u exceeds %u",
            i, config.workgroup_size[i], UINT16_MAX);
      }
      const uint64_t grid_size =
          (uint64_t)config.workgroup_count[i] * config.workgroup_size[i];
      if (IREE_UNLIKELY(grid_size > UINT32_MAX)) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "dispatch grid dimension %" PRIhsz " overflows uint32_t", i);
      }
    }
  } else {
    for (iree_host_size_t i = 0; i < 3; ++i) {
      if (IREE_UNLIKELY(config.workgroup_count[i] >
                        descriptor->max_workgroup_count[i])) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "dispatch grid dimension %" PRIhsz
            " overflows uint32_t (workgroup_count=%u, workgroup_size=%u)",
            i, config.workgroup_count[i],
            descriptor->kernel_args.workgroup_size[i]);
      }
    }
  }
  if (IREE_UNLIKELY(config.dynamic_workgroup_local_memory >
                    descriptor->max_dynamic_workgroup_local_memory)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "dispatch group segment size overflows uint32_t");
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_pm4_command_buffer_resolve_buffer_ref(
    const iree_hal_buffer_ref_t* buffer_ref, uint64_t* out_device_pointer) {
  *out_device_pointer = 0;
  iree_hal_buffer_t* allocated_buffer =
      iree_hal_buffer_allocated_buffer(buffer_ref->buffer);
  void* device_ptr = iree_hal_amdgpu_buffer_device_pointer(allocated_buffer);
  if (IREE_UNLIKELY(!device_ptr)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "PM4 command-buffer buffer reference must be backed by an AMDGPU "
        "allocation");
  }
  iree_device_size_t device_offset = 0;
  if (IREE_UNLIKELY(!iree_device_size_checked_add(
          iree_hal_buffer_byte_offset(buffer_ref->buffer), buffer_ref->offset,
          &device_offset))) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "PM4 command-buffer buffer reference device pointer offset overflows");
  }
  if (IREE_UNLIKELY(device_offset > UINTPTR_MAX)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "PM4 command-buffer buffer reference device pointer offset exceeds "
        "host pointer size");
  }
  *out_device_pointer =
      (uint64_t)((uintptr_t)device_ptr + (uintptr_t)device_offset);
  return iree_ok_status();
}

static void iree_hal_amdgpu_pm4_command_buffer_write_implicit_args(
    const iree_hal_amdgpu_device_kernel_args_t* kernel_args,
    const iree_hal_dispatch_config_t config,
    iree_amdgpu_kernel_implicit_args_t* implicit_args) {
  memset(implicit_args, 0, sizeof(*implicit_args));
  implicit_args->block_count[0] = config.workgroup_count[0];
  implicit_args->block_count[1] = config.workgroup_count[1];
  implicit_args->block_count[2] = config.workgroup_count[2];
  implicit_args->group_size[0] = kernel_args->workgroup_size[0];
  implicit_args->group_size[1] = kernel_args->workgroup_size[1];
  implicit_args->group_size[2] = kernel_args->workgroup_size[2];
  implicit_args->grid_dims = 3;
  implicit_args->printf_buffer = NULL;
  implicit_args->hostcall_buffer = NULL;
  implicit_args->dynamic_lds_size = config.dynamic_workgroup_local_memory;
}

static iree_status_t iree_hal_amdgpu_pm4_command_buffer_write_template(
    iree_hal_amdgpu_pm4_command_buffer_t* command_buffer,
    const iree_hal_amdgpu_device_kernel_args_t* kernel_args,
    const iree_hal_amdgpu_device_dispatch_kernarg_layout_t* layout,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    const iree_hal_amdgpu_pm4_binding_record_t* binding_records,
    uint32_t binding_record_count, uint8_t* template_bytes) {
  uint64_t* binding_dst = (uint64_t*)template_bytes;
  for (uint32_t i = 0; i < binding_record_count; ++i) {
    const iree_hal_amdgpu_pm4_binding_record_t* binding_record =
        &binding_records[i];
    if (iree_any_bit_set(binding_record->flags,
                         IREE_HAL_AMDGPU_PM4_BINDING_RECORD_FLAG_DYNAMIC)) {
      IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_fixup_entry_builder_append(
          &command_buffer->fixup_builder,
          (iree_hal_amdgpu_command_buffer_pm4_fixup_entry_t){
              .target_offset = binding_record->target_offset,
              .binding_slot = binding_record->binding_slot,
              .binding_offset = binding_record->value,
          }));
      continue;
    }
    binding_dst[i] = binding_record->value;
  }

  const iree_host_size_t binding_bytes =
      (iree_host_size_t)kernel_args->binding_count * sizeof(uint64_t);
  if (constants.data_length > 0) {
    memcpy(template_bytes + binding_bytes, constants.data,
           constants.data_length);
  }
  if (layout->has_implicit_args) {
    iree_amdgpu_kernel_implicit_args_t* implicit_args =
        (iree_amdgpu_kernel_implicit_args_t*)(template_bytes +
                                              layout->implicit_args_offset);
    iree_hal_amdgpu_pm4_command_buffer_write_implicit_args(kernel_args, config,
                                                           implicit_args);
  }
  return iree_ok_status();
}

static const uint8_t* iree_hal_amdgpu_pm4_dispatch_record_constants(
    const iree_hal_amdgpu_pm4_dispatch_record_t* record) {
  return (const uint8_t*)record + sizeof(*record);
}

static const iree_hal_amdgpu_pm4_binding_record_t*
iree_hal_amdgpu_pm4_dispatch_record_bindings(
    const iree_hal_amdgpu_pm4_dispatch_record_t* record) {
  const uintptr_t constants_end =
      (uintptr_t)iree_hal_amdgpu_pm4_dispatch_record_constants(record) +
      record->constant_length;
  return (const iree_hal_amdgpu_pm4_binding_record_t*)iree_host_align(
      constants_end, iree_alignof(iree_hal_amdgpu_pm4_binding_record_t));
}

static iree_status_t iree_hal_amdgpu_pm4_command_buffer_retain_resource_once(
    iree_hal_amdgpu_pm4_command_buffer_t* command_buffer,
    iree_hal_resource_t* resource) {
  if (!resource) return iree_ok_status();
  iree_hal_amdgpu_pm4_retained_resource_table_t* retained_resources =
      &command_buffer->retained_resources;
  if (iree_hal_amdgpu_pm4_retained_resource_table_contains(retained_resources,
                                                           resource)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_retained_resource_table_reserve(
      command_buffer->host_allocator, retained_resources,
      retained_resources->count + 1));
  IREE_RETURN_IF_ERROR(
      iree_hal_resource_set_insert(command_buffer->resource_set,
                                   /*count=*/1, &resource));
  iree_hal_amdgpu_pm4_retained_resource_table_insert_prepared(
      retained_resources, resource);
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_pm4_command_buffer_retain_dispatch(
    iree_hal_amdgpu_pm4_command_buffer_t* command_buffer,
    iree_hal_executable_t* executable, iree_hal_buffer_ref_list_t bindings) {
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_pm4_command_buffer_ensure_resource_set(command_buffer));
  if (!command_buffer->resource_set) return iree_ok_status();

  if (command_buffer->last_retained_executable != executable) {
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_pm4_command_buffer_retain_resource_once(
            command_buffer, (iree_hal_resource_t*)executable));
    command_buffer->last_retained_executable = executable;
  }
  for (iree_host_size_t i = 0; i < bindings.count; ++i) {
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_pm4_command_buffer_retain_resource_once(
            command_buffer, (iree_hal_resource_t*)bindings.values[i].buffer));
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_pm4_command_buffer_append_dispatch_record(
    iree_hal_amdgpu_pm4_command_buffer_t* command_buffer,
    const iree_hal_amdgpu_executable_dispatch_descriptor_t* descriptor,
    uint64_t executable_id, uint32_t command_index, uint32_t export_ordinal,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    iree_hal_buffer_ref_list_t bindings,
    iree_hal_amdgpu_pm4_dispatch_record_flags_t flags,
    iree_hsa_fence_scope_t barrier_acquire_scope,
    iree_hsa_fence_scope_t barrier_release_scope) {
  const iree_hal_amdgpu_device_kernel_args_t* kernel_args =
      &descriptor->kernel_args;
  if (IREE_UNLIKELY(constants.data_length > UINT32_MAX)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "PM4 command-buffer dispatch constants exceed uint32_t storage");
  }
  if (IREE_UNLIKELY(bindings.count > UINT32_MAX)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "PM4 command-buffer dispatch bindings exceed uint32_t storage");
  }
  if (IREE_UNLIKELY(bindings.count > kernel_args->binding_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "PM4 command-buffer dispatch binding count %" PRIhsz
                            " exceeds executable binding count %u",
                            bindings.count,
                            (uint32_t)kernel_args->binding_count);
  }
  if (IREE_UNLIKELY(bindings.count > 0 && !bindings.values)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "PM4 command-buffer dispatch bindings must be non-null when count is "
        "non-zero");
  }
  const iree_host_size_t expected_constant_bytes =
      (iree_host_size_t)kernel_args->constant_count * sizeof(uint32_t);
  if (IREE_UNLIKELY(constants.data_length > expected_constant_bytes)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "PM4 command-buffer dispatch constants exceed executable constant "
        "storage");
  }

  iree_host_size_t template_offset = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_align_host_size(
      command_buffer->record_template_byte_length,
      kernel_args->kernarg_alignment, &template_offset));
  if (IREE_UNLIKELY(template_offset > UINT32_MAX)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "PM4 command-buffer kernarg template offset exceeds uint32_t storage");
  }
  const iree_hal_amdgpu_device_dispatch_kernarg_layout_t layout =
      descriptor->hal_kernarg_layout;
  iree_host_size_t new_record_template_byte_length = 0;
  if (!iree_host_size_checked_add(template_offset, layout.total_kernarg_size,
                                  &new_record_template_byte_length)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "PM4 command-buffer kernarg template storage overflows");
  }
  if (IREE_UNLIKELY(layout.total_kernarg_size > UINT32_MAX)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "PM4 command-buffer kernarg template length exceeds uint32_t storage");
  }

  uint32_t dynamic_fixup_count = 0;
  for (iree_host_size_t i = 0; i < bindings.count; ++i) {
    const iree_hal_buffer_ref_t* binding = &bindings.values[i];
    if (!binding->buffer) {
      if (IREE_UNLIKELY(binding->buffer_slot == UINT32_MAX)) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "PM4 command-buffer dispatch binding slot %u exceeds binding "
            "count storage",
            binding->buffer_slot);
      }
      command_buffer->base.binding_count = iree_max(
          command_buffer->base.binding_count, binding->buffer_slot + 1);
      ++dynamic_fixup_count;
    }
  }
  if (IREE_UNLIKELY(dynamic_fixup_count >
                    UINT32_MAX - command_buffer->record_fixup_entry_count)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "PM4 command-buffer dynamic fixup entry count exceeds uint32_t "
        "storage");
  }
  const uint32_t new_record_fixup_entry_count =
      command_buffer->record_fixup_entry_count + dynamic_fixup_count;

  uint32_t dispatch_dwords = IREE_HAL_AMDGPU_PM4_DISPATCH_DIRECT_DWORD_COUNT;
  if (iree_any_bit_set(
          flags, IREE_HAL_AMDGPU_PM4_DISPATCH_RECORD_FLAG_EXECUTION_BARRIER)) {
    dispatch_dwords += iree_hal_amdgpu_pm4_barrier_dword_count_gfx10(
        command_buffer->vendor_packet_capabilities,
        IREE_HAL_AMDGPU_PM4_BARRIER_FLAG_EXECUTION, barrier_acquire_scope,
        barrier_release_scope);
  }
  if (iree_any_bit_set(
          flags, IREE_HAL_AMDGPU_PM4_DISPATCH_RECORD_FLAG_FIXUP_BARRIER)) {
    dispatch_dwords += iree_hal_amdgpu_pm4_barrier_dword_count_gfx10(
        command_buffer->vendor_packet_capabilities,
        IREE_HAL_AMDGPU_PM4_BARRIER_FLAG_FIXUP_TO_IB, IREE_HSA_FENCE_SCOPE_NONE,
        IREE_HSA_FENCE_SCOPE_NONE);
  }
  const iree_hal_amdgpu_pm4_dispatch_launch_state_t* launch_state =
      &descriptor->pm4_launch_state;
  const bool needs_dispatch_setup =
      !command_buffer->has_previous_launch_state ||
      memcmp(&command_buffer->previous_launch_state, launch_state,
             sizeof(*launch_state)) != 0;
  if (needs_dispatch_setup) {
    dispatch_dwords += descriptor->pm4_setup_dword_count;
  }
  if (launch_state->user_data_dword_count != 0) {
    dispatch_dwords += 2u + launch_state->user_data_dword_count;
  }
  if (IREE_UNLIKELY(dispatch_dwords >
                    IREE_HAL_AMDGPU_PM4_IB_MAX_DWORD_COUNT -
                        command_buffer->record_ib_dword_count)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "PM4 command buffer requires more than the PM4-IB maximum %u dwords",
        IREE_HAL_AMDGPU_PM4_IB_MAX_DWORD_COUNT);
  }
  const uint32_t new_record_ib_dword_count =
      command_buffer->record_ib_dword_count + dispatch_dwords;

  const iree_host_size_t binding_record_count =
      (iree_host_size_t)kernel_args->binding_count;
  iree_host_size_t binding_record_bytes = 0;
  if (!iree_host_size_checked_mul(binding_record_count,
                                  sizeof(iree_hal_amdgpu_pm4_binding_record_t),
                                  &binding_record_bytes)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "PM4 binding records overflow");
  }

  iree_host_size_t constants_end = 0;
  if (!iree_host_size_checked_add(sizeof(iree_hal_amdgpu_pm4_dispatch_record_t),
                                  constants.data_length, &constants_end)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "PM4 dispatch record constants overflow");
  }
  const iree_host_size_t bindings_offset = iree_host_align(
      constants_end, iree_alignof(iree_hal_amdgpu_pm4_binding_record_t));
  iree_host_size_t record_length = 0;
  if (!iree_host_size_checked_add(bindings_offset, binding_record_bytes,
                                  &record_length) ||
      record_length > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "PM4 dispatch record length overflows");
  }

  uint8_t* record_bytes = NULL;
  const iree_host_size_t record_builder_base_length =
      command_buffer->record_builder.length;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_byte_builder_append_record(
      &command_buffer->record_builder, record_length, &record_bytes));
  memset(record_bytes, 0, record_length);
  iree_hal_amdgpu_pm4_dispatch_record_t* record =
      (iree_hal_amdgpu_pm4_dispatch_record_t*)record_bytes;
  record->header.length = (uint32_t)record_length;
  record->header.opcode = IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_OPCODE_DISPATCH;
  record->descriptor = descriptor;
  record->executable_id = executable_id;
  record->workgroup_count[0] = config.workgroup_count[0];
  record->workgroup_count[1] = config.workgroup_count[1];
  record->workgroup_count[2] = config.workgroup_count[2];
  record->command_index = command_index;
  record->export_ordinal = export_ordinal;
  record->template_offset = (uint32_t)template_offset;
  record->template_length = (uint32_t)layout.total_kernarg_size;
  record->constant_length = (uint32_t)constants.data_length;
  record->binding_record_count = (uint32_t)binding_record_count;
  record->flags = flags;
  record->barrier_acquire_scope = barrier_acquire_scope;
  record->barrier_release_scope = barrier_release_scope;
  if (constants.data_length > 0) {
    memcpy(record_bytes + sizeof(*record), constants.data,
           constants.data_length);
  }
  if (binding_record_bytes > 0) {
    iree_hal_amdgpu_pm4_binding_record_t* binding_records =
        (iree_hal_amdgpu_pm4_binding_record_t*)(record_bytes + bindings_offset);
    memset(binding_records, 0, binding_record_bytes);
    for (iree_host_size_t i = 0; i < bindings.count; ++i) {
      const iree_hal_buffer_ref_t* binding = &bindings.values[i];
      iree_hal_amdgpu_pm4_binding_record_t* binding_record =
          &binding_records[i];
      if (IREE_UNLIKELY((uint64_t)template_offset >
                        UINT32_MAX - i * sizeof(uint64_t))) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "PM4 command-buffer binding target offset overflows");
      }
      binding_record->target_offset =
          (uint32_t)template_offset + (uint32_t)i * sizeof(uint64_t);
      if (!binding->buffer) {
        binding_record->value = binding->offset;
        binding_record->binding_slot = binding->buffer_slot;
        binding_record->flags = IREE_HAL_AMDGPU_PM4_BINDING_RECORD_FLAG_DYNAMIC;
        continue;
      }
      iree_status_t status =
          iree_hal_amdgpu_pm4_command_buffer_resolve_buffer_ref(
              binding, &binding_record->value);
      if (!iree_status_is_ok(status)) {
        command_buffer->record_builder.length = record_builder_base_length;
        return status;
      }
    }
  }
  command_buffer->record_template_byte_length = new_record_template_byte_length;
  command_buffer->record_fixup_entry_count = new_record_fixup_entry_count;
  command_buffer->record_ib_dword_count = new_record_ib_dword_count;
  if (needs_dispatch_setup) {
    command_buffer->previous_launch_state = *launch_state;
    command_buffer->has_previous_launch_state = true;
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_pm4_command_buffer_materialize_dispatch(
    iree_hal_amdgpu_pm4_command_buffer_t* command_buffer,
    const iree_hal_amdgpu_pm4_dispatch_record_t* record) {
  const iree_hal_amdgpu_executable_dispatch_descriptor_t* descriptor =
      record->descriptor;
  const iree_hal_amdgpu_device_kernel_args_t* kernel_args =
      &descriptor->kernel_args;
  const iree_hal_amdgpu_device_dispatch_kernarg_layout_t* layout =
      &descriptor->hal_kernarg_layout;
  iree_const_byte_span_t constants = iree_make_const_byte_span(
      iree_hal_amdgpu_pm4_dispatch_record_constants(record),
      record->constant_length);
  const iree_hal_amdgpu_pm4_binding_record_t* binding_records =
      iree_hal_amdgpu_pm4_dispatch_record_bindings(record);
  const iree_hal_dispatch_config_t config = {
      .workgroup_count = {record->workgroup_count[0],
                          record->workgroup_count[1],
                          record->workgroup_count[2]},
  };

  if (iree_any_bit_set(
          record->flags,
          IREE_HAL_AMDGPU_PM4_DISPATCH_RECORD_FLAG_EXECUTION_BARRIER)) {
    const uint32_t dword_count_before =
        command_buffer->dword_builder.dword_count;
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_dword_builder_emit_barrier(
        &command_buffer->dword_builder,
        command_buffer->vendor_packet_capabilities,
        IREE_HAL_AMDGPU_PM4_BARRIER_FLAG_EXECUTION,
        record->barrier_acquire_scope, record->barrier_release_scope));
    command_buffer->publish_stats.execution_barrier_dwords +=
        command_buffer->dword_builder.dword_count - dword_count_before;
  }

  uint8_t* template_bytes = NULL;
  uint32_t template_offset = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_byte_builder_append_aligned(
      &command_buffer->template_builder, kernel_args->kernarg_alignment,
      layout->total_kernarg_size, &template_offset, &template_bytes));
  if (IREE_UNLIKELY(template_offset != record->template_offset)) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "PM4 template materialization offset changed");
  }
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_command_buffer_write_template(
      command_buffer, kernel_args, layout, config, constants, binding_records,
      record->binding_record_count, template_bytes));

  if (iree_any_bit_set(
          record->flags,
          IREE_HAL_AMDGPU_PM4_DISPATCH_RECORD_FLAG_FIXUP_BARRIER)) {
    const uint32_t dword_count_before =
        command_buffer->dword_builder.dword_count;
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_dword_builder_emit_barrier(
        &command_buffer->dword_builder,
        command_buffer->vendor_packet_capabilities,
        IREE_HAL_AMDGPU_PM4_BARRIER_FLAG_FIXUP_TO_IB, IREE_HSA_FENCE_SCOPE_NONE,
        IREE_HSA_FENCE_SCOPE_NONE));
    command_buffer->publish_stats.fixup_barrier_dwords +=
        command_buffer->dword_builder.dword_count - dword_count_before;
  }

  const iree_hal_amdgpu_pm4_dispatch_launch_state_t* launch_state =
      &descriptor->pm4_launch_state;
  if (!command_buffer->has_previous_launch_state ||
      memcmp(&command_buffer->previous_launch_state, launch_state,
             sizeof(*launch_state)) != 0) {
    const uint32_t dword_count_before =
        command_buffer->dword_builder.dword_count;
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_dword_builder_emit_dispatch_setup(
        &command_buffer->dword_builder, descriptor->pm4_setup_dwords,
        descriptor->pm4_setup_dword_count));
    command_buffer->publish_stats.dispatch_setup_dwords +=
        command_buffer->dword_builder.dword_count - dword_count_before;
    command_buffer->previous_launch_state = *launch_state;
    command_buffer->has_previous_launch_state = true;
  }

  if (IREE_UNLIKELY(!command_buffer->fixup.plan.target_base)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "PM4 command-buffer dispatch requires resident kernarg storage");
  }
  if (IREE_UNLIKELY(launch_state->user_data_dword_count == 0)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "PM4 command-buffer dispatch has no kernarg user-data dwords");
  }
  const uintptr_t kernarg_address =
      (uintptr_t)command_buffer->fixup.plan.target_base +
      record->template_offset;
  uint32_t dword_count_before = command_buffer->dword_builder.dword_count;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_dword_builder_emit_user_data(
      &command_buffer->dword_builder, launch_state, kernarg_address));
  command_buffer->publish_stats.dispatch_user_data_dwords +=
      command_buffer->dword_builder.dword_count - dword_count_before;
  dword_count_before = command_buffer->dword_builder.dword_count;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_dword_builder_emit_dispatch_direct(
      &command_buffer->dword_builder, record->workgroup_count,
      launch_state->dispatch_initiator));
  command_buffer->publish_stats.dispatch_direct_dwords +=
      command_buffer->dword_builder.dword_count - dword_count_before;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_pm4_command_buffer_materialize_records(
    iree_hal_amdgpu_pm4_command_buffer_t* command_buffer) {
  command_buffer->has_previous_launch_state = false;
  command_buffer->has_emitted_fixup_barrier = false;
  const uint8_t* cursor = command_buffer->record_builder.bytes;
  const uint8_t* const end = cursor + command_buffer->record_builder.length;
  while (cursor < end) {
    if (IREE_UNLIKELY((iree_host_size_t)(end - cursor) <
                      sizeof(iree_hal_amdgpu_pm4_command_record_header_t))) {
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "PM4 command record is truncated");
    }
    const iree_hal_amdgpu_pm4_command_record_header_t* header =
        (const iree_hal_amdgpu_pm4_command_record_header_t*)cursor;
    if (IREE_UNLIKELY(header->length == 0 ||
                      (iree_host_size_t)(end - cursor) < header->length)) {
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "PM4 command record length is invalid");
    }
    switch ((iree_hal_amdgpu_pm4_command_record_opcode_t)header->opcode) {
      case IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_OPCODE_DISPATCH: {
        IREE_RETURN_IF_ERROR(
            iree_hal_amdgpu_pm4_command_buffer_materialize_dispatch(
                command_buffer,
                (const iree_hal_amdgpu_pm4_dispatch_record_t*)cursor));
        break;
      }
      default:
        return iree_make_status(IREE_STATUS_INTERNAL,
                                "unknown PM4 command record opcode %u",
                                header->opcode);
    }
    cursor += header->length;
  }
  return iree_ok_status();
}

static iree_hal_profile_command_operation_flags_t
iree_hal_amdgpu_pm4_command_buffer_profile_dispatch_binding_flags(
    const iree_hal_amdgpu_pm4_dispatch_record_t* record) {
  iree_hal_profile_command_operation_flags_t flags =
      IREE_HAL_PROFILE_COMMAND_OPERATION_FLAG_NONE;
  const iree_hal_amdgpu_pm4_binding_record_t* binding_records =
      iree_hal_amdgpu_pm4_dispatch_record_bindings(record);
  for (uint32_t i = 0; i < record->binding_record_count; ++i) {
    flags |= iree_any_bit_set(binding_records[i].flags,
                              IREE_HAL_AMDGPU_PM4_BINDING_RECORD_FLAG_DYNAMIC)
                 ? IREE_HAL_PROFILE_COMMAND_OPERATION_FLAG_DYNAMIC_BINDINGS
                 : IREE_HAL_PROFILE_COMMAND_OPERATION_FLAG_STATIC_BINDINGS;
  }
  return flags;
}

static void iree_hal_amdgpu_pm4_command_buffer_initialize_profile_operation(
    uint64_t command_buffer_id,
    const iree_hal_amdgpu_pm4_dispatch_record_t* dispatch_record,
    iree_hal_profile_command_operation_record_t* out_record) {
  iree_hal_profile_command_operation_record_t record =
      iree_hal_profile_command_operation_record_default();
  record.type = IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_DISPATCH;
  record.command_buffer_id = command_buffer_id;
  record.command_index = dispatch_record->command_index;
  if (iree_any_bit_set(
          dispatch_record->flags,
          IREE_HAL_AMDGPU_PM4_DISPATCH_RECORD_FLAG_EXECUTION_BARRIER)) {
    record.flags |= IREE_HAL_PROFILE_COMMAND_OPERATION_FLAG_EXECUTION_BARRIER;
  }
  const iree_hal_profile_command_operation_flags_t binding_flags =
      iree_hal_amdgpu_pm4_command_buffer_profile_dispatch_binding_flags(
          dispatch_record);
  record.flags |= binding_flags;
  if (!iree_any_bit_set(
          binding_flags,
          IREE_HAL_PROFILE_COMMAND_OPERATION_FLAG_DYNAMIC_BINDINGS)) {
    record.flags |=
        IREE_HAL_PROFILE_COMMAND_OPERATION_FLAG_PREPUBLISHED_ARGUMENTS;
  }
  record.executable_id = dispatch_record->executable_id;
  record.export_ordinal = dispatch_record->export_ordinal;
  record.binding_count = dispatch_record->binding_record_count;
  record.workgroup_count[0] = dispatch_record->workgroup_count[0];
  record.workgroup_count[1] = dispatch_record->workgroup_count[1];
  record.workgroup_count[2] = dispatch_record->workgroup_count[2];
  const iree_hal_amdgpu_device_kernel_args_t* kernel_args =
      &dispatch_record->descriptor->kernel_args;
  record.workgroup_size[0] = kernel_args->workgroup_size[0];
  record.workgroup_size[1] = kernel_args->workgroup_size[1];
  record.workgroup_size[2] = kernel_args->workgroup_size[2];
  *out_record = record;
}

static iree_status_t
iree_hal_amdgpu_pm4_command_buffer_register_profile_operations(
    iree_hal_amdgpu_pm4_command_buffer_t* command_buffer) {
  if (command_buffer->record_command_count == 0) return iree_ok_status();

  iree_host_size_t byte_length = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      0, &byte_length,
      IREE_STRUCT_FIELD(command_buffer->record_command_count,
                        iree_hal_profile_command_operation_record_t, NULL)));

  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, command_buffer->record_command_count);

  iree_hal_profile_command_operation_record_t* records = NULL;
  iree_status_t status = iree_allocator_malloc(command_buffer->host_allocator,
                                               byte_length, (void**)&records);

  iree_host_size_t record_count = 0;
  if (iree_status_is_ok(status)) {
    const uint8_t* cursor = command_buffer->record_builder.bytes;
    const uint8_t* const end = cursor + command_buffer->record_builder.length;
    while (cursor < end &&
           record_count < command_buffer->record_command_count) {
      if (IREE_UNLIKELY((iree_host_size_t)(end - cursor) <
                        sizeof(iree_hal_amdgpu_pm4_command_record_header_t))) {
        status = iree_make_status(IREE_STATUS_INTERNAL,
                                  "PM4 command record header is truncated");
        break;
      }
      const iree_hal_amdgpu_pm4_command_record_header_t* header =
          (const iree_hal_amdgpu_pm4_command_record_header_t*)cursor;
      if (IREE_UNLIKELY(header->length == 0 ||
                        (iree_host_size_t)(end - cursor) < header->length)) {
        status = iree_make_status(IREE_STATUS_INTERNAL,
                                  "PM4 command record length is invalid");
        break;
      }
      switch ((iree_hal_amdgpu_pm4_command_record_opcode_t)header->opcode) {
        case IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_OPCODE_DISPATCH:
          iree_hal_amdgpu_pm4_command_buffer_initialize_profile_operation(
              command_buffer->profile.id,
              (const iree_hal_amdgpu_pm4_dispatch_record_t*)cursor,
              &records[record_count++]);
          break;
        default:
          status = iree_make_status(IREE_STATUS_INTERNAL,
                                    "unknown PM4 command record opcode %u",
                                    header->opcode);
          break;
      }
      cursor += header->length;
    }
  }
  if (iree_status_is_ok(status) &&
      record_count != command_buffer->record_command_count) {
    status =
        iree_make_status(IREE_STATUS_INTERNAL,
                         "PM4 profile command-operation count mismatch: "
                         "expected %u but got %" PRIhsz,
                         command_buffer->record_command_count, record_count);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_profile_metadata_register_command_operations(
        command_buffer->profile.metadata, record_count, records);
  }

  iree_allocator_free(command_buffer->host_allocator, records);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_amdgpu_pm4_command_buffer_record_dispatch(
    iree_hal_amdgpu_pm4_command_buffer_t* command_buffer,
    iree_hal_executable_t* executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    iree_hal_buffer_ref_list_t bindings, iree_hal_dispatch_flags_t flags) {
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_pm4_command_buffer_check_dispatch_flags(flags));

  const iree_hal_amdgpu_executable_dispatch_descriptor_t* descriptor = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_executable_lookup_dispatch_descriptor_for_device(
          executable, export_ordinal, command_buffer->device_ordinal,
          &descriptor));
  if (IREE_UNLIKELY(!descriptor->pm4_launch_state_valid)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "PM4 command-buffer dispatch requires executable-load PM4 metadata");
  }
  if (IREE_UNLIKELY(iree_hal_amdgpu_dispatch_config_has_workgroup_size_override(
          config))) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "PM4 command-buffer dispatch workgroup-size override is not "
        "implemented");
  }

  const bool validates =
      iree_hal_amdgpu_pm4_command_buffer_validates(command_buffer);
  if (validates) {
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_pm4_command_buffer_validate_dispatch_shape(descriptor,
                                                                   config));
    if (IREE_UNLIKELY(constants.data_length !=
                      (iree_host_size_t)descriptor->kernel_args.constant_count *
                          sizeof(uint32_t))) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "dispatch constant count mismatch; expected %u but got %" PRIhsz,
          (uint32_t)descriptor->kernel_args.constant_count,
          constants.data_length / sizeof(uint32_t));
    }
    if (IREE_UNLIKELY(bindings.count !=
                      descriptor->kernel_args.binding_count)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "dispatch binding count mismatch; expected %u but got %" PRIhsz,
          (uint32_t)descriptor->kernel_args.binding_count, bindings.count);
    }
    if (IREE_UNLIKELY(bindings.count > 0 && !bindings.values)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "dispatch bindings must be non-null when count is non-zero");
    }
  }

  if (IREE_UNLIKELY(config.dynamic_workgroup_local_memory != 0)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "PM4 command-buffer dynamic LDS is not implemented");
  }

  uint32_t workgroup_count[3] = {
      config.workgroup_count[0],
      config.workgroup_count[1],
      config.workgroup_count[2],
  };
  if (workgroup_count[0] == 0 || workgroup_count[1] == 0 ||
      workgroup_count[2] == 0) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_command_buffer_retain_dispatch(
      command_buffer, executable, bindings));
  if (IREE_UNLIKELY(descriptor->pm4_group_segment_fixed_size !=
                    descriptor->kernel_args.group_segment_size)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "PM4 command-buffer dynamic LDS is not implemented; descriptor LDS "
        "size %" PRIu32 " differs from dispatch group segment size %" PRIu32,
        descriptor->pm4_group_segment_fixed_size,
        descriptor->kernel_args.group_segment_size);
  }

  iree_hal_amdgpu_pm4_dispatch_record_flags_t record_flags =
      IREE_HAL_AMDGPU_PM4_DISPATCH_RECORD_FLAG_NONE;
  iree_hsa_fence_scope_t barrier_acquire_scope = IREE_HSA_FENCE_SCOPE_NONE;
  iree_hsa_fence_scope_t barrier_release_scope = IREE_HSA_FENCE_SCOPE_NONE;
  if (command_buffer->barrier_state.pending) {
    record_flags |= IREE_HAL_AMDGPU_PM4_DISPATCH_RECORD_FLAG_EXECUTION_BARRIER;
    barrier_acquire_scope = command_buffer->barrier_state.acquire_scope;
    barrier_release_scope = command_buffer->barrier_state.release_scope;
    iree_hal_amdgpu_pm4_barrier_state_reset(&command_buffer->barrier_state);
  }
  bool has_dynamic_binding = false;
  for (iree_host_size_t i = 0; i < bindings.count; ++i) {
    if (!bindings.values[i].buffer) {
      has_dynamic_binding = true;
      break;
    }
  }
  if (has_dynamic_binding && !command_buffer->has_emitted_fixup_barrier) {
    record_flags |= IREE_HAL_AMDGPU_PM4_DISPATCH_RECORD_FLAG_FIXUP_BARRIER;
    command_buffer->has_emitted_fixup_barrier = true;
  }
  if (IREE_UNLIKELY(command_buffer->record_command_count == UINT32_MAX)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "PM4 command-buffer command index exceeds uint32_t storage");
  }
  const uint32_t command_index = command_buffer->record_command_count;
  iree_status_t status =
      iree_hal_amdgpu_pm4_command_buffer_append_dispatch_record(
          command_buffer, descriptor,
          iree_hal_amdgpu_executable_profile_id(executable), command_index,
          export_ordinal, config, constants, bindings, record_flags,
          barrier_acquire_scope, barrier_release_scope);
  if (iree_status_is_ok(status)) {
    ++command_buffer->record_command_count;
  }
  return status;
}

//===----------------------------------------------------------------------===//
// Lifecycle
//===----------------------------------------------------------------------===//

static void iree_hal_amdgpu_pm4_command_buffer_destroy(
    iree_hal_command_buffer_t* base_command_buffer);

static iree_status_t iree_hal_amdgpu_pm4_command_buffer_verify_create(
    iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_amdgpu_vendor_packet_capability_flags_t vendor_packet_capabilities,
    iree_hal_amdgpu_pm4_command_buffer_resident_pool_t* resident_pool,
    iree_hal_amdgpu_profile_metadata_registry_t* profile_metadata,
    iree_arena_block_pool_t* resource_set_block_pool) {
  if (iree_any_bit_set(mode, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "PM4 command buffers require reusable command-buffer mode");
  }
  const iree_hal_command_buffer_mode_t unsupported_modes =
      IREE_HAL_COMMAND_BUFFER_MODE_ALLOW_INLINE_EXECUTION |
      IREE_HAL_COMMAND_BUFFER_MODE_RETAIN_DISPATCH_METADATA;
  if (iree_any_bit_set(mode, unsupported_modes)) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "PM4 command-buffer mode bits 0x%08" PRIx32
                            " are not implemented",
                            mode & unsupported_modes);
  }
  if (!iree_all_bits_set(command_categories,
                         IREE_HAL_COMMAND_CATEGORY_DISPATCH) ||
      iree_any_bit_set(command_categories,
                       ~IREE_HAL_COMMAND_CATEGORY_DISPATCH)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "PM4 command buffers require dispatch-only command categories");
  }
  if (!iree_hal_amdgpu_vendor_packet_capabilities_support_pm4_dispatch_command_buffers(
          vendor_packet_capabilities)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "PM4 command buffers require PM4-IB, EVENT_WRITE, SET_SH_REG, "
        "ACQUIRE_MEM, and DISPATCH_DIRECT capabilities");
  }
  if (IREE_UNLIKELY(!resident_pool)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "PM4 command-buffer resident pool is required");
  }
  if (IREE_UNLIKELY(
          iree_all_bits_set(
              mode, IREE_HAL_COMMAND_BUFFER_MODE_RETAIN_PROFILE_METADATA) &&
          !profile_metadata)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "PM4 command-buffer profile metadata is required");
  }
  if (IREE_UNLIKELY(!resource_set_block_pool)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "PM4 command-buffer resource set block pool is required");
  }
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_pm4_command_buffer_create(
    iree_hal_allocator_t* device_allocator, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_host_size_t device_ordinal,
    iree_hal_amdgpu_pm4_command_buffer_flags_t flags,
    iree_hal_amdgpu_vendor_packet_capability_flags_t vendor_packet_capabilities,
    iree_hal_amdgpu_pm4_command_buffer_resident_pool_t* resident_pool,
    iree_hal_amdgpu_profile_metadata_registry_t* profile_metadata,
    iree_arena_block_pool_t* resource_set_block_pool,
    iree_allocator_t host_allocator,
    iree_hal_command_buffer_t** out_command_buffer) {
  IREE_ASSERT_ARGUMENT(device_allocator);
  IREE_ASSERT_ARGUMENT(out_command_buffer);
  *out_command_buffer = NULL;

  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_command_buffer_verify_create(
      mode, command_categories, vendor_packet_capabilities, resident_pool,
      profile_metadata, resource_set_block_pool));
  if (IREE_UNLIKELY(device_ordinal > UINT32_MAX)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "PM4 command-buffer device ordinal %" PRIhsz
                            " exceeds uint32_t storage",
                            device_ordinal);
  }

  iree_host_size_t total_size = 0;
  iree_host_size_t validation_state_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_amdgpu_pm4_command_buffer_t), &total_size,
      IREE_STRUCT_FIELD(
          iree_hal_command_buffer_validation_state_size(mode, binding_capacity),
          uint8_t, &validation_state_offset)));

  iree_hal_amdgpu_pm4_command_buffer_t* command_buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, total_size,
                                             (void**)&command_buffer));
  memset(command_buffer, 0, sizeof(*command_buffer));
  iree_slim_mutex_initialize(&command_buffer->publication_mutex);
  iree_hal_command_buffer_initialize(
      device_allocator, mode, command_categories, queue_affinity,
      binding_capacity, (uint8_t*)command_buffer + validation_state_offset,
      &iree_hal_amdgpu_pm4_command_buffer_vtable, &command_buffer->base);
  command_buffer->host_allocator = host_allocator;
  command_buffer->resource_set_block_pool = resource_set_block_pool;
  command_buffer->resident_pool = resident_pool;
  command_buffer->profile.metadata = profile_metadata;
  command_buffer->libhsa = resident_pool->libhsa;
  command_buffer->vendor_packet_capabilities = vendor_packet_capabilities;
  command_buffer->flags = flags;
  command_buffer->device_ordinal = (uint32_t)device_ordinal;

  iree_status_t status = iree_ok_status();
  if (iree_hal_amdgpu_pm4_command_buffer_retains_profile_metadata(
          command_buffer)) {
    status = iree_hal_amdgpu_profile_metadata_register_command_buffer(
        profile_metadata, mode, command_categories, queue_affinity,
        device_ordinal, &command_buffer->profile.id);
  }
  if (iree_status_is_ok(status)) {
    *out_command_buffer = &command_buffer->base;
  } else {
    iree_hal_amdgpu_pm4_command_buffer_destroy(&command_buffer->base);
  }
  return status;
}

static void iree_hal_amdgpu_pm4_command_buffer_destroy(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_amdgpu_pm4_command_buffer_t* command_buffer =
      iree_hal_amdgpu_pm4_command_buffer_cast(base_command_buffer);
  iree_allocator_t host_allocator = command_buffer->host_allocator;

  iree_hal_amdgpu_pm4_command_buffer_wait_for_publication(command_buffer);
  if (command_buffer->resident_allocation) {
    iree_hal_amdgpu_pm4_command_buffer_resident_pool_release(
        command_buffer->resident_pool, command_buffer->resident_allocation);
    command_buffer->resident_allocation = NULL;
    memset(&command_buffer->program, 0, sizeof(command_buffer->program));
  } else {
    iree_status_t release_status =
        iree_hal_amdgpu_pm4_program_release(&command_buffer->program);
    IREE_ASSERT(iree_status_is_ok(release_status),
                "PM4 command-buffer program release failed");
    iree_status_free(release_status);
  }
  iree_hal_amdgpu_pm4_command_buffer_fixup_reset(command_buffer);
  if (command_buffer->recording_state ==
      IREE_HAL_AMDGPU_PM4_COMMAND_BUFFER_RECORDING_STATE_RECORDING) {
    iree_hal_amdgpu_pm4_recording_builders_deinitialize(command_buffer);
  }
  iree_hal_resource_set_free(command_buffer->resource_set);
  iree_slim_mutex_deinitialize(&command_buffer->publication_mutex);
  iree_allocator_free(host_allocator, command_buffer);
}

bool iree_hal_amdgpu_pm4_command_buffer_isa(
    iree_hal_command_buffer_t* command_buffer) {
  return iree_hal_resource_is(&command_buffer->resource,
                              &iree_hal_amdgpu_pm4_command_buffer_vtable);
}

iree_host_size_t iree_hal_amdgpu_pm4_command_buffer_device_ordinal(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_amdgpu_pm4_command_buffer_t* command_buffer =
      iree_hal_amdgpu_pm4_command_buffer_cast(base_command_buffer);
  return command_buffer->device_ordinal;
}

const iree_hal_amdgpu_pm4_program_t* iree_hal_amdgpu_pm4_command_buffer_program(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_amdgpu_pm4_command_buffer_t* command_buffer =
      iree_hal_amdgpu_pm4_command_buffer_cast(base_command_buffer);
  return &command_buffer->program;
}

uint64_t iree_hal_amdgpu_pm4_command_buffer_profile_id(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_amdgpu_pm4_command_buffer_t* command_buffer =
      iree_hal_amdgpu_pm4_command_buffer_cast(base_command_buffer);
  return command_buffer->profile.id;
}

uint32_t iree_hal_amdgpu_pm4_command_buffer_operation_count(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_amdgpu_pm4_command_buffer_t* command_buffer =
      iree_hal_amdgpu_pm4_command_buffer_cast(base_command_buffer);
  return command_buffer->record_command_count;
}

const iree_hal_amdgpu_pm4_command_buffer_fixup_plan_t*
iree_hal_amdgpu_pm4_command_buffer_fixup_plan(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_amdgpu_pm4_command_buffer_t* command_buffer =
      iree_hal_amdgpu_pm4_command_buffer_cast(base_command_buffer);
  return &command_buffer->fixup.plan;
}

const iree_hal_amdgpu_pm4_command_buffer_publish_stats_t*
iree_hal_amdgpu_pm4_command_buffer_publish_stats(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_amdgpu_pm4_command_buffer_t* command_buffer =
      iree_hal_amdgpu_pm4_command_buffer_cast(base_command_buffer);
  return &command_buffer->publish_stats;
}

hsa_signal_t iree_hal_amdgpu_pm4_command_buffer_acquire_publication_reference(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_amdgpu_pm4_command_buffer_t* command_buffer =
      iree_hal_amdgpu_pm4_command_buffer_cast(base_command_buffer);
  hsa_signal_t publication_signal = iree_hsa_signal_null();
  iree_slim_mutex_lock(&command_buffer->publication_mutex);
  if (!iree_hsa_signal_is_null(command_buffer->publication_signal)) {
    publication_signal = command_buffer->publication_signal;
    IREE_ASSERT(command_buffer->publication_reference_count != UINT32_MAX,
                "PM4 command-buffer publication reference count overflowed");
    ++command_buffer->publication_reference_count;
  }
  iree_slim_mutex_unlock(&command_buffer->publication_mutex);
  return publication_signal;
}

void iree_hal_amdgpu_pm4_command_buffer_cancel_publication_reference(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_amdgpu_pm4_command_buffer_t* command_buffer =
      iree_hal_amdgpu_pm4_command_buffer_cast(base_command_buffer);
  iree_slim_mutex_lock(&command_buffer->publication_mutex);
  IREE_ASSERT(command_buffer->publication_reference_count > 0,
              "PM4 command-buffer publication reference cancelled without a "
              "matching acquire");
  --command_buffer->publication_reference_count;
  iree_slim_mutex_unlock(&command_buffer->publication_mutex);
}

void iree_hal_amdgpu_pm4_command_buffer_retire_publication_reference(
    iree_hal_command_buffer_t* base_command_buffer, iree_status_t status) {
  iree_hal_amdgpu_pm4_command_buffer_t* command_buffer =
      iree_hal_amdgpu_pm4_command_buffer_cast(base_command_buffer);
  iree_slim_mutex_lock(&command_buffer->publication_mutex);
  IREE_ASSERT(command_buffer->publication_reference_count > 0,
              "PM4 command-buffer publication reference retired without a "
              "matching acquire");
  --command_buffer->publication_reference_count;
  if (command_buffer->publication_reference_count == 0 &&
      iree_status_is_ok(status) &&
      !iree_hsa_signal_is_null(command_buffer->publication_signal)) {
    iree_hal_amdgpu_pm4_command_buffer_release_publication_resources_locked(
        command_buffer);
  }
  iree_slim_mutex_unlock(&command_buffer->publication_mutex);
}

//===----------------------------------------------------------------------===//
// Recording operations
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_amdgpu_pm4_command_buffer_begin(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_amdgpu_pm4_command_buffer_t* command_buffer =
      iree_hal_amdgpu_pm4_command_buffer_cast(base_command_buffer);
  switch (command_buffer->recording_state) {
    case IREE_HAL_AMDGPU_PM4_COMMAND_BUFFER_RECORDING_STATE_INITIAL:
      break;
    case IREE_HAL_AMDGPU_PM4_COMMAND_BUFFER_RECORDING_STATE_RECORDING:
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "PM4 command buffer is already recording");
    case IREE_HAL_AMDGPU_PM4_COMMAND_BUFFER_RECORDING_STATE_FINALIZED:
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "PM4 command buffer has already been recorded");
    case IREE_HAL_AMDGPU_PM4_COMMAND_BUFFER_RECORDING_STATE_FAILED:
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "PM4 command buffer recording failed and cannot be reused");
    default:
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "invalid PM4 command-buffer recording state %d",
                              (int)command_buffer->recording_state);
  }
  iree_hal_amdgpu_pm4_recording_builders_initialize(command_buffer);
  memset(&command_buffer->publish_stats, 0,
         sizeof(command_buffer->publish_stats));
  command_buffer->barrier_state = (iree_hal_amdgpu_pm4_barrier_state_t){0};
  command_buffer->previous_launch_state =
      (iree_hal_amdgpu_pm4_dispatch_launch_state_t){0};
  command_buffer->last_retained_executable = NULL;
  command_buffer->has_previous_launch_state = false;
  command_buffer->has_emitted_fixup_barrier = false;
  command_buffer->record_command_count = 0;
  command_buffer->recording_state =
      IREE_HAL_AMDGPU_PM4_COMMAND_BUFFER_RECORDING_STATE_RECORDING;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_pm4_command_buffer_end(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_amdgpu_pm4_command_buffer_t* command_buffer =
      iree_hal_amdgpu_pm4_command_buffer_cast(base_command_buffer);
  if (IREE_UNLIKELY(
          command_buffer->recording_state !=
          IREE_HAL_AMDGPU_PM4_COMMAND_BUFFER_RECORDING_STATE_RECORDING)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "PM4 command buffer is not recording");
  }

  const bool collect_timings =
      iree_hal_amdgpu_pm4_command_buffer_collects_finalize_timings(
          command_buffer);
  const iree_time_t finalize_start = collect_timings ? iree_time_now() : 0;
  const uint32_t terminal_barrier_dword_count =
      iree_hal_amdgpu_pm4_barrier_dword_count_gfx10(
          command_buffer->vendor_packet_capabilities,
          IREE_HAL_AMDGPU_PM4_BARRIER_FLAG_EXECUTION,
          command_buffer->barrier_state.acquire_scope,
          command_buffer->barrier_state.release_scope);
  iree_status_t status = iree_ok_status();
  if (IREE_UNLIKELY(terminal_barrier_dword_count == 0)) {
    status = iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "PM4 command-buffer terminal barrier cannot be emitted with "
        "capabilities 0x%08" PRIx32,
        command_buffer->vendor_packet_capabilities);
  } else if (IREE_UNLIKELY(terminal_barrier_dword_count >
                           IREE_HAL_AMDGPU_PM4_IB_MAX_DWORD_COUNT -
                               command_buffer->record_ib_dword_count)) {
    status = iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "PM4 command buffer requires more than the PM4-IB maximum %u dwords",
        IREE_HAL_AMDGPU_PM4_IB_MAX_DWORD_COUNT);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_pm4_command_buffer_allocate_resident_storage(
        command_buffer,
        command_buffer->record_ib_dword_count + terminal_barrier_dword_count);
  }
  if (iree_status_is_ok(status)) {
    iree_time_t time_start = collect_timings ? iree_time_now() : 0;
    status =
        iree_hal_amdgpu_pm4_command_buffer_materialize_records(command_buffer);
    if (collect_timings) {
      command_buffer->publish_stats.materialize_ns +=
          iree_time_now() - time_start;
    }
  }
  if (iree_status_is_ok(status)) {
    if (IREE_UNLIKELY(command_buffer->template_builder.length !=
                      command_buffer->record_template_byte_length)) {
      status = iree_make_status(IREE_STATUS_INTERNAL,
                                "PM4 template materialization produced %" PRIhsz
                                " bytes, expected %" PRIhsz,
                                command_buffer->template_builder.length,
                                command_buffer->record_template_byte_length);
    } else if (IREE_UNLIKELY(command_buffer->fixup_builder.count !=
                             command_buffer->record_fixup_entry_count)) {
      status = iree_make_status(
          IREE_STATUS_INTERNAL,
          "PM4 fixup materialization produced %u entries, expected %u",
          command_buffer->fixup_builder.count,
          command_buffer->record_fixup_entry_count);
    } else if (IREE_UNLIKELY(command_buffer->dword_builder.dword_count !=
                             command_buffer->record_ib_dword_count)) {
      status = iree_make_status(
          IREE_STATUS_INTERNAL,
          "PM4 IB materialization produced %u dwords, expected %u",
          command_buffer->dword_builder.dword_count,
          command_buffer->record_ib_dword_count);
    }
  }
  if (iree_status_is_ok(status)) {
    const uint32_t dword_count_before =
        command_buffer->dword_builder.dword_count;
    status = iree_hal_amdgpu_pm4_dword_builder_emit_barrier(
        &command_buffer->dword_builder,
        command_buffer->vendor_packet_capabilities,
        IREE_HAL_AMDGPU_PM4_BARRIER_FLAG_EXECUTION,
        command_buffer->barrier_state.acquire_scope,
        command_buffer->barrier_state.release_scope);
    if (iree_status_is_ok(status)) {
      command_buffer->publish_stats.terminal_barrier_dwords +=
          command_buffer->dword_builder.dword_count - dword_count_before;
    }
  }
  if (iree_status_is_ok(status) &&
      IREE_UNLIKELY(command_buffer->dword_builder.dword_count !=
                    command_buffer->program.dword_count)) {
    status = iree_make_status(IREE_STATUS_INTERNAL,
                              "PM4 resident IB contains %u dwords, expected %u",
                              command_buffer->dword_builder.dword_count,
                              command_buffer->program.dword_count);
  }
  if (iree_status_is_ok(status)) {
    command_buffer->publish_stats.host_record_bytes =
        iree_hal_amdgpu_pm4_command_buffer_host_record_bytes(command_buffer);
  }
  if (iree_status_is_ok(status) &&
      iree_hal_amdgpu_pm4_command_buffer_materializes_to_host(command_buffer)) {
    iree_time_t time_start = collect_timings ? iree_time_now() : 0;
    status = iree_hal_amdgpu_pm4_command_buffer_copy_materialized_storage(
        command_buffer);
    if (collect_timings) {
      command_buffer->publish_stats.resident_copy_ns +=
          iree_time_now() - time_start;
    }
  }
  if (iree_status_is_ok(status) &&
      iree_hal_amdgpu_pm4_command_buffer_retains_profile_metadata(
          command_buffer)) {
    status = iree_hal_amdgpu_pm4_command_buffer_register_profile_operations(
        command_buffer);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_resource_set_freeze(command_buffer->resource_set);
  }

  iree_hal_amdgpu_pm4_recording_builders_deinitialize(command_buffer);
  if (collect_timings) {
    command_buffer->publish_stats.total_finalize_ns =
        iree_time_now() - finalize_start;
  }
  if (iree_status_is_ok(status)) {
    command_buffer->recording_state =
        IREE_HAL_AMDGPU_PM4_COMMAND_BUFFER_RECORDING_STATE_FINALIZED;
  } else {
    command_buffer->recording_state =
        IREE_HAL_AMDGPU_PM4_COMMAND_BUFFER_RECORDING_STATE_FAILED;
  }
  return status;
}

static iree_status_t iree_hal_amdgpu_pm4_command_buffer_begin_debug_group(
    iree_hal_command_buffer_t* base_command_buffer, iree_string_view_t label,
    iree_hal_label_color_t label_color,
    const iree_hal_label_location_t* location) {
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_pm4_command_buffer_end_debug_group(
    iree_hal_command_buffer_t* base_command_buffer) {
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_pm4_command_buffer_execution_barrier(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_execution_barrier_flags_t flags,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  iree_hal_amdgpu_pm4_command_buffer_t* command_buffer =
      iree_hal_amdgpu_pm4_command_buffer_cast(base_command_buffer);
  if (IREE_UNLIKELY(flags != IREE_HAL_EXECUTION_BARRIER_FLAG_NONE)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported PM4 command-buffer barrier flags: "
                            "0x%" PRIx64,
                            flags);
  }
  (void)source_stage_mask;
  (void)target_stage_mask;
  (void)memory_barrier_count;
  (void)memory_barriers;
  (void)buffer_barrier_count;
  (void)buffer_barriers;
  iree_hal_amdgpu_pm4_barrier_state_accumulate(&command_buffer->barrier_state,
                                               IREE_HSA_FENCE_SCOPE_AGENT,
                                               IREE_HSA_FENCE_SCOPE_AGENT);
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_pm4_command_buffer_signal_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "PM4 command-buffer events are not implemented");
}

static iree_status_t iree_hal_amdgpu_pm4_command_buffer_reset_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "PM4 command-buffer events are not implemented");
}

static iree_status_t iree_hal_amdgpu_pm4_command_buffer_wait_events(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_host_size_t event_count, const iree_hal_event_t** events,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "PM4 command-buffer events are not implemented");
}

static iree_status_t iree_hal_amdgpu_pm4_command_buffer_advise_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t buffer_ref, iree_hal_memory_advise_flags_t flags,
    uint64_t arg0, uint64_t arg1) {
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_pm4_command_buffer_fill_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t target_ref, const void* pattern,
    iree_host_size_t pattern_length, iree_hal_fill_flags_t flags) {
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "PM4 command-buffer fill operations are not implemented");
}

static iree_status_t iree_hal_amdgpu_pm4_command_buffer_update_buffer(
    iree_hal_command_buffer_t* base_command_buffer, const void* source_buffer,
    iree_host_size_t source_offset, iree_hal_buffer_ref_t target_ref,
    iree_hal_update_flags_t flags) {
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "PM4 command-buffer update operations are not implemented");
}

static iree_status_t iree_hal_amdgpu_pm4_command_buffer_copy_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t source_ref, iree_hal_buffer_ref_t target_ref,
    iree_hal_copy_flags_t flags) {
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "PM4 command-buffer copy operations are not implemented");
}

static iree_status_t iree_hal_amdgpu_pm4_command_buffer_collective(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_channel_t* channel,
    iree_hal_collective_op_t op, uint32_t param, iree_hal_buffer_ref_t send_ref,
    iree_hal_buffer_ref_t recv_ref, iree_device_size_t element_count) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "PM4 command-buffer collectives are not implemented");
}

static iree_status_t iree_hal_amdgpu_pm4_command_buffer_dispatch(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_t* executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    iree_hal_buffer_ref_list_t bindings, iree_hal_dispatch_flags_t flags) {
  iree_hal_amdgpu_pm4_command_buffer_t* command_buffer =
      iree_hal_amdgpu_pm4_command_buffer_cast(base_command_buffer);
  if (IREE_UNLIKELY(
          command_buffer->recording_state !=
          IREE_HAL_AMDGPU_PM4_COMMAND_BUFFER_RECORDING_STATE_RECORDING)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "PM4 command buffer is not recording");
  }
  return iree_hal_amdgpu_pm4_command_buffer_record_dispatch(
      command_buffer, executable, export_ordinal, config, constants, bindings,
      flags);
}

//===----------------------------------------------------------------------===//
// Vtable
//===----------------------------------------------------------------------===//

static const iree_hal_command_buffer_vtable_t
    iree_hal_amdgpu_pm4_command_buffer_vtable = {
        .destroy = iree_hal_amdgpu_pm4_command_buffer_destroy,
        .begin = iree_hal_amdgpu_pm4_command_buffer_begin,
        .end = iree_hal_amdgpu_pm4_command_buffer_end,
        .begin_debug_group =
            iree_hal_amdgpu_pm4_command_buffer_begin_debug_group,
        .end_debug_group = iree_hal_amdgpu_pm4_command_buffer_end_debug_group,
        .execution_barrier =
            iree_hal_amdgpu_pm4_command_buffer_execution_barrier,
        .signal_event = iree_hal_amdgpu_pm4_command_buffer_signal_event,
        .reset_event = iree_hal_amdgpu_pm4_command_buffer_reset_event,
        .wait_events = iree_hal_amdgpu_pm4_command_buffer_wait_events,
        .advise_buffer = iree_hal_amdgpu_pm4_command_buffer_advise_buffer,
        .fill_buffer = iree_hal_amdgpu_pm4_command_buffer_fill_buffer,
        .update_buffer = iree_hal_amdgpu_pm4_command_buffer_update_buffer,
        .copy_buffer = iree_hal_amdgpu_pm4_command_buffer_copy_buffer,
        .collective = iree_hal_amdgpu_pm4_command_buffer_collective,
        .dispatch = iree_hal_amdgpu_pm4_command_buffer_dispatch,
};
