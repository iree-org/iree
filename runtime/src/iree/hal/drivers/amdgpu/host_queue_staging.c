// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/host_queue_staging.h"

#include <string.h>

#include "iree/async/operations/file.h"
#include "iree/hal/drivers/amdgpu/access_policy.h"
#include "iree/hal/drivers/amdgpu/buffer.h"
#include "iree/hal/drivers/amdgpu/host_queue.h"
#include "iree/hal/drivers/amdgpu/host_queue_blit.h"
#include "iree/hal/drivers/amdgpu/host_queue_profile.h"
#include "iree/hal/drivers/amdgpu/host_queue_submission.h"
#include "iree/hal/drivers/amdgpu/slab_provider.h"

typedef struct iree_hal_amdgpu_staging_slot_t {
  // Slot ordinal in the physical-device staging pool.
  uint32_t ordinal;
  // Byte offset of the slot inside the staging buffer.
  iree_device_size_t buffer_offset;
  // Host-accessible bytes for file I/O.
  iree_byte_span_t host_span;
  // HAL buffer wrapping the complete staging allocation. Borrowed from the
  // pool; queue copy submissions retain it while the GPU owns the slot.
  iree_hal_buffer_t* buffer;
} iree_hal_amdgpu_staging_slot_t;

typedef struct iree_hal_amdgpu_staging_allocation_t {
  // Borrowed HSA API table used to free |allocation_base|.
  const iree_hal_amdgpu_libhsa_t* libhsa;
  // Host allocator used to allocate this release state.
  iree_allocator_t host_allocator;
  // Original pointer returned by hsa_amd_memory_pool_allocate.
  void* allocation_base;
} iree_hal_amdgpu_staging_allocation_t;

typedef uint32_t iree_hal_amdgpu_staging_pool_waiter_flags_t;
enum iree_hal_amdgpu_staging_pool_waiter_flag_bits_e {
  IREE_HAL_AMDGPU_STAGING_POOL_WAITER_FLAG_NONE = 0u,
  IREE_HAL_AMDGPU_STAGING_POOL_WAITER_FLAG_QUEUED = 1u << 0,
};

// Callback invoked when a staging slot may be available.
typedef void(IREE_API_PTR* iree_hal_amdgpu_staging_pool_waiter_fn_t)(
    void* user_data);

// Intrusive waiter used by transfers that cannot acquire a staging slot.
struct iree_hal_amdgpu_staging_pool_waiter_t {
  // Next waiter in the pool-owned FIFO list.
  iree_hal_amdgpu_staging_pool_waiter_t* next;
  // Callback invoked after the waiter is removed from the FIFO list.
  iree_hal_amdgpu_staging_pool_waiter_fn_t fn;
  // User data passed to |fn|.
  void* user_data;
  // Slot reserved for this waiter when it is dequeued.
  iree_hal_amdgpu_staging_slot_t slot;
  // Waiter lifecycle flags from iree_hal_amdgpu_staging_pool_waiter_flags_t.
  iree_hal_amdgpu_staging_pool_waiter_flags_t flags;
};

typedef enum iree_hal_amdgpu_staging_pool_wait_result_e {
  // The waiter was newly queued and will receive a future callback.
  IREE_HAL_AMDGPU_STAGING_POOL_WAIT_QUEUED = 0,
  // The waiter was already queued by an earlier pump attempt.
  IREE_HAL_AMDGPU_STAGING_POOL_WAIT_ALREADY_QUEUED = 1,
  // A slot became available before the waiter was queued; retry acquire.
  IREE_HAL_AMDGPU_STAGING_POOL_WAIT_RETRY = 2,
} iree_hal_amdgpu_staging_pool_wait_result_t;

void iree_hal_amdgpu_staging_pool_options_initialize(
    iree_hal_amdgpu_staging_pool_options_t* out_options) {
  IREE_ASSERT_ARGUMENT(out_options);
  memset(out_options, 0, sizeof(*out_options));
  out_options->slot_size = IREE_HAL_AMDGPU_STAGING_SLOT_SIZE_DEFAULT;
  out_options->slot_count = IREE_HAL_AMDGPU_STAGING_SLOT_COUNT_DEFAULT;
}

iree_status_t iree_hal_amdgpu_staging_pool_options_verify(
    const iree_hal_amdgpu_staging_pool_options_t* options) {
  IREE_ASSERT_ARGUMENT(options);
  if (options->slot_size == 0 ||
      !iree_host_size_is_power_of_two(options->slot_size)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "staging slot size must be a non-zero power of two (got %" PRIhsz ")",
        options->slot_size);
  }
  if (options->slot_size < IREE_HAL_AMDGPU_STAGING_SLOT_ALIGNMENT) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "staging slot size must be at least %" PRIhsz
        " bytes to preserve slot "
        "alignment (got %" PRIhsz ")",
        (iree_host_size_t)IREE_HAL_AMDGPU_STAGING_SLOT_ALIGNMENT,
        options->slot_size);
  }
  if (options->slot_count == 0 ||
      !iree_host_size_is_power_of_two(options->slot_count)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "staging slot count must be a non-zero power of two (got %u)",
        options->slot_count);
  }
  iree_host_size_t total_size = 0;
  if (!iree_host_size_checked_mul(options->slot_size, options->slot_count,
                                  &total_size) ||
      total_size > (iree_host_size_t)IREE_DEVICE_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "staging pool size overflows (slot_size=%" PRIhsz
                            ", slot_count=%u)",
                            options->slot_size, options->slot_count);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_staging_pool_resolve_access_agents(
    const iree_hal_amdgpu_topology_t* topology,
    iree_hal_queue_affinity_t queue_affinity_mask,
    iree_hal_amdgpu_access_agent_list_t* out_agent_list) {
  const iree_hal_amdgpu_queue_affinity_domain_t domain = {
      .supported_affinity = queue_affinity_mask,
      .physical_device_count = topology->gpu_agent_count,
      .queue_count_per_physical_device = topology->gpu_agent_queue_count,
  };
  return iree_hal_amdgpu_access_agent_list_resolve(
      topology, domain, queue_affinity_mask, out_agent_list);
}

static void iree_hal_amdgpu_staging_allocation_release(
    void* user_data, iree_hal_buffer_t* buffer) {
  (void)buffer;
  iree_hal_amdgpu_staging_allocation_t* allocation =
      (iree_hal_amdgpu_staging_allocation_t*)user_data;
  IREE_TRACE_ZONE_BEGIN(z0);
  if (allocation->allocation_base) {
    iree_hal_amdgpu_hsa_cleanup_assert_success(
        iree_hsa_amd_memory_pool_free_raw(allocation->libhsa,
                                          allocation->allocation_base));
  }
  iree_allocator_free(allocation->host_allocator, allocation);
  IREE_TRACE_ZONE_END(z0);
}

static iree_hal_amdgpu_staging_pool_waiter_t*
iree_hal_amdgpu_staging_pool_pop_waiter_locked(
    iree_hal_amdgpu_staging_pool_t* pool,
    const iree_hal_amdgpu_staging_slot_t* slot) {
  iree_hal_amdgpu_staging_pool_waiter_t* waiter = pool->waiter_head;
  if (waiter) {
    pool->waiter_head = waiter->next;
    if (!pool->waiter_head) {
      pool->waiter_tail = NULL;
    }
    waiter->next = NULL;
    waiter->slot = *slot;
    waiter->flags &= ~IREE_HAL_AMDGPU_STAGING_POOL_WAITER_FLAG_QUEUED;
  }
  return waiter;
}

static bool iree_hal_amdgpu_staging_pool_try_acquire(
    iree_hal_amdgpu_staging_pool_t* pool,
    iree_hal_amdgpu_staging_slot_t* out_slot) {
  bool did_acquire = false;
  iree_slim_mutex_lock(&pool->mutex);
  if (pool->available_count > 0) {
    const uint32_t slot_ordinal =
        pool->free_slots[pool->free_read++ & pool->slot_mask];
    --pool->available_count;
    out_slot->ordinal = slot_ordinal;
    out_slot->buffer_offset =
        (iree_device_size_t)slot_ordinal * pool->slot_size;
    out_slot->host_span = iree_make_byte_span(
        pool->host_base + (iree_host_size_t)out_slot->buffer_offset,
        pool->slot_size);
    out_slot->buffer = pool->buffer;
    did_acquire = true;
  }
  iree_slim_mutex_unlock(&pool->mutex);
  return did_acquire;
}

static bool iree_hal_amdgpu_staging_pool_take_waiter_slot(
    iree_hal_amdgpu_staging_pool_t* pool,
    iree_hal_amdgpu_staging_pool_waiter_t* waiter,
    iree_hal_amdgpu_staging_slot_t* out_slot) {
  bool did_take = false;
  iree_slim_mutex_lock(&pool->mutex);
  if (waiter->slot.buffer) {
    *out_slot = waiter->slot;
    memset(&waiter->slot, 0, sizeof(waiter->slot));
    did_take = true;
  }
  iree_slim_mutex_unlock(&pool->mutex);
  return did_take;
}

static iree_hal_amdgpu_staging_pool_wait_result_t
iree_hal_amdgpu_staging_pool_queue_waiter(
    iree_hal_amdgpu_staging_pool_t* pool,
    iree_hal_amdgpu_staging_pool_waiter_t* waiter,
    iree_hal_amdgpu_staging_pool_waiter_fn_t fn, void* user_data) {
  iree_hal_amdgpu_staging_pool_wait_result_t result =
      IREE_HAL_AMDGPU_STAGING_POOL_WAIT_QUEUED;
  iree_slim_mutex_lock(&pool->mutex);
  if (iree_any_bit_set(waiter->flags,
                       IREE_HAL_AMDGPU_STAGING_POOL_WAITER_FLAG_QUEUED)) {
    result = IREE_HAL_AMDGPU_STAGING_POOL_WAIT_ALREADY_QUEUED;
  } else if (pool->available_count > 0) {
    result = IREE_HAL_AMDGPU_STAGING_POOL_WAIT_RETRY;
  } else {
    waiter->next = NULL;
    waiter->fn = fn;
    waiter->user_data = user_data;
    waiter->flags |= IREE_HAL_AMDGPU_STAGING_POOL_WAITER_FLAG_QUEUED;
    if (pool->waiter_tail) {
      pool->waiter_tail->next = waiter;
    } else {
      pool->waiter_head = waiter;
    }
    pool->waiter_tail = waiter;
  }
  iree_slim_mutex_unlock(&pool->mutex);
  return result;
}

static bool iree_hal_amdgpu_staging_pool_cancel_waiter(
    iree_hal_amdgpu_staging_pool_t* pool,
    iree_hal_amdgpu_staging_pool_waiter_t* waiter) {
  bool did_cancel = false;
  iree_slim_mutex_lock(&pool->mutex);
  iree_hal_amdgpu_staging_pool_waiter_t* previous = NULL;
  for (iree_hal_amdgpu_staging_pool_waiter_t* current = pool->waiter_head;
       current != NULL; current = current->next) {
    if (current == waiter) {
      if (previous) {
        previous->next = current->next;
      } else {
        pool->waiter_head = current->next;
      }
      if (pool->waiter_tail == current) {
        pool->waiter_tail = previous;
      }
      waiter->next = NULL;
      waiter->flags &= ~IREE_HAL_AMDGPU_STAGING_POOL_WAITER_FLAG_QUEUED;
      did_cancel = true;
      break;
    }
    previous = current;
  }
  iree_slim_mutex_unlock(&pool->mutex);
  return did_cancel;
}

static void iree_hal_amdgpu_staging_pool_release(
    iree_hal_amdgpu_staging_pool_t* pool, uint32_t slot_ordinal) {
  iree_hal_amdgpu_staging_slot_t slot = {
      .ordinal = slot_ordinal,
      .buffer_offset = (iree_device_size_t)slot_ordinal * pool->slot_size,
      .host_span = iree_make_byte_span(
          pool->host_base + (iree_host_size_t)slot_ordinal * pool->slot_size,
          pool->slot_size),
      .buffer = pool->buffer,
  };
  iree_hal_amdgpu_staging_pool_waiter_t* waiter = NULL;
  iree_slim_mutex_lock(&pool->mutex);
  waiter = iree_hal_amdgpu_staging_pool_pop_waiter_locked(pool, &slot);
  if (!waiter) {
    pool->free_slots[pool->free_write++ & pool->slot_mask] = slot_ordinal;
    ++pool->available_count;
  }
  iree_slim_mutex_unlock(&pool->mutex);
  if (waiter) {
    waiter->fn(waiter->user_data);
  }
}

iree_status_t iree_hal_amdgpu_staging_pool_initialize(
    iree_hal_device_t* logical_device, const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_topology_t* topology,
    const iree_hal_amdgpu_host_memory_pools_t* host_memory_pools,
    iree_hal_queue_affinity_t queue_affinity_mask,
    const iree_hal_amdgpu_staging_pool_options_t* options,
    iree_allocator_t host_allocator, iree_hal_amdgpu_staging_pool_t* out_pool) {
  IREE_ASSERT_ARGUMENT(logical_device);
  IREE_ASSERT_ARGUMENT(libhsa);
  IREE_ASSERT_ARGUMENT(topology);
  IREE_ASSERT_ARGUMENT(host_memory_pools);
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(out_pool);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, options->slot_size);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, options->slot_count);

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdgpu_staging_pool_options_verify(options));

  memset(out_pool, 0, sizeof(*out_pool));
  out_pool->host_allocator = host_allocator;
  out_pool->slot_size = options->slot_size;
  out_pool->slot_count = options->slot_count;
  out_pool->slot_mask = options->slot_count - 1u;
  iree_slim_mutex_initialize(&out_pool->mutex);

  hsa_amd_memory_pool_t memory_pool = host_memory_pools->coarse_pool;
  if (options->force_fine_host_memory || !memory_pool.handle) {
    memory_pool = host_memory_pools->fine_pool;
  }
  iree_status_t status = iree_ok_status();
  if (IREE_UNLIKELY(!memory_pool.handle)) {
    status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "AMDGPU staging requires a host memory pool");
  }

  iree_hal_amdgpu_slab_provider_memory_pool_properties_t memory_pool_properties;
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_slab_provider_query_memory_pool_properties(
        libhsa, memory_pool, &memory_pool_properties);
  }

  iree_host_size_t total_size = 0;
  if (iree_status_is_ok(status) &&
      !iree_host_size_checked_mul(options->slot_size, options->slot_count,
                                  &total_size)) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "staging pool size overflows (slot_size=%" PRIhsz
                              ", slot_count=%u)",
                              options->slot_size, options->slot_count);
  }

  iree_host_size_t free_slots_size = 0;
  if (iree_status_is_ok(status) &&
      !iree_host_size_checked_mul(options->slot_count, sizeof(uint32_t),
                                  &free_slots_size)) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "staging free-slot table size overflows");
  }

  iree_host_size_t allocation_size = total_size;
  if (iree_status_is_ok(status) && memory_pool_properties.allocation_alignment <
                                       IREE_HAL_AMDGPU_STAGING_SLOT_ALIGNMENT) {
    if (!iree_host_size_checked_add(total_size,
                                    IREE_HAL_AMDGPU_STAGING_SLOT_ALIGNMENT - 1,
                                    &allocation_size)) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "staging aligned allocation size overflows");
    }
  }

  uint32_t* free_slots = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(host_allocator, free_slots_size,
                                   (void**)&free_slots);
  }

  void* allocation_base = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hsa_amd_memory_pool_allocate(
        IREE_LIBHSA(libhsa), memory_pool, allocation_size,
        HSA_AMD_MEMORY_POOL_STANDARD_FLAG, &allocation_base);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_amdgpu_access_agent_list_t access_agents;
    status = iree_hal_amdgpu_staging_pool_resolve_access_agents(
        topology, queue_affinity_mask, &access_agents);
    if (iree_status_is_ok(status)) {
      status = iree_hal_amdgpu_access_allow_agent_list(libhsa, &access_agents,
                                                       allocation_base);
    }
  }

  void* host_ptr = NULL;
  if (iree_status_is_ok(status)) {
    const uintptr_t allocation_begin = (uintptr_t)allocation_base;
    const uintptr_t allocation_end = allocation_begin + allocation_size;
    iree_host_size_t aligned_host_base = 0;
    if (!iree_host_size_checked_align((iree_host_size_t)allocation_begin,
                                      IREE_HAL_AMDGPU_STAGING_SLOT_ALIGNMENT,
                                      &aligned_host_base)) {
      status = iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "HSA staging allocation base overflowed while aligning to %" PRIhsz
          " bytes (base=%p)",
          (iree_host_size_t)IREE_HAL_AMDGPU_STAGING_SLOT_ALIGNMENT,
          allocation_base);
    }
    if (iree_status_is_ok(status)) {
      const uintptr_t aligned_base = (uintptr_t)aligned_host_base;
      const uintptr_t aligned_end = aligned_base + total_size;
      if (allocation_end < allocation_begin ||
          aligned_base < allocation_begin || aligned_end < aligned_base ||
          aligned_end > allocation_end ||
          !iree_host_ptr_has_alignment(
              (void*)aligned_base, IREE_HAL_AMDGPU_STAGING_SLOT_ALIGNMENT)) {
        status = iree_make_status(
            IREE_STATUS_INTERNAL,
            "HSA staging allocation could not satisfy %" PRIhsz
            "-byte alignment (base=%p, allocation_size=%" PRIhsz
            ", total_size=%" PRIhsz ")",
            (iree_host_size_t)IREE_HAL_AMDGPU_STAGING_SLOT_ALIGNMENT,
            allocation_base, allocation_size, total_size);
      } else {
        host_ptr = (void*)aligned_base;
      }
    }
  }

  iree_hal_amdgpu_staging_allocation_t* release_state = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(host_allocator, sizeof(*release_state),
                                   (void**)&release_state);
  }
  if (iree_status_is_ok(status)) {
    release_state->libhsa = libhsa;
    release_state->host_allocator = host_allocator;
    release_state->allocation_base = allocation_base;
  }

  iree_hal_buffer_t* buffer = NULL;
  if (iree_status_is_ok(status)) {
    const iree_hal_buffer_placement_t placement = {
        .device = logical_device,
        .queue_affinity = queue_affinity_mask,
        .flags = IREE_HAL_BUFFER_PLACEMENT_FLAG_NONE,
    };
    status = iree_hal_amdgpu_buffer_create(
        libhsa, placement,
        IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
        IREE_HAL_MEMORY_ACCESS_ALL, IREE_HAL_BUFFER_USAGE_TRANSFER,
        (iree_device_size_t)total_size, (iree_device_size_t)total_size,
        host_ptr,
        (iree_hal_buffer_release_callback_t){
            .fn = iree_hal_amdgpu_staging_allocation_release,
            .user_data = release_state,
        },
        host_allocator, &buffer);
  }

  if (iree_status_is_ok(status)) {
    for (uint32_t i = 0; i < options->slot_count; ++i) {
      free_slots[i] = i;
    }
    out_pool->buffer = buffer;
    out_pool->host_base = (uint8_t*)host_ptr;
    out_pool->available_count = options->slot_count;
    out_pool->free_slots = free_slots;
    out_pool->free_write = options->slot_count;
    release_state = NULL;
    allocation_base = NULL;
  } else {
    iree_hal_buffer_release(buffer);
    if (allocation_base) {
      status = iree_status_join(
          status,
          iree_hsa_amd_memory_pool_free(IREE_LIBHSA(libhsa), allocation_base));
    }
    iree_allocator_free(host_allocator, release_state);
    iree_allocator_free(host_allocator, free_slots);
    iree_slim_mutex_deinitialize(&out_pool->mutex);
    memset(out_pool, 0, sizeof(*out_pool));
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_amdgpu_staging_pool_deinitialize(
    iree_hal_amdgpu_staging_pool_t* pool) {
  IREE_ASSERT_ARGUMENT(pool);
  if (!pool->buffer && !pool->free_slots && pool->slot_count == 0) {
    return;
  }
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_buffer_release(pool->buffer);
  iree_allocator_free(pool->host_allocator, pool->free_slots);
  iree_slim_mutex_deinitialize(&pool->mutex);
  memset(pool, 0, sizeof(*pool));
  IREE_TRACE_ZONE_END(z0);
}

typedef enum iree_hal_amdgpu_staging_transfer_kind_e {
  // File data flows from the proactor into staging and then GPU copy writes the
  // target buffer.
  IREE_HAL_AMDGPU_STAGING_TRANSFER_READ = 0,
  // GPU copy writes staging and then file data flows from staging into the
  // proactor.
  IREE_HAL_AMDGPU_STAGING_TRANSFER_WRITE = 1,
} iree_hal_amdgpu_staging_transfer_kind_t;

static iree_hal_profile_queue_event_type_t
iree_hal_amdgpu_staging_transfer_profile_event_type(
    iree_hal_amdgpu_staging_transfer_kind_t kind) {
  return kind == IREE_HAL_AMDGPU_STAGING_TRANSFER_READ
             ? IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_READ
             : IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_WRITE;
}

typedef uint32_t iree_hal_amdgpu_staging_transfer_flags_t;
enum iree_hal_amdgpu_staging_transfer_flag_bits_e {
  IREE_HAL_AMDGPU_STAGING_TRANSFER_FLAG_NONE = 0u,
  IREE_HAL_AMDGPU_STAGING_TRANSFER_FLAG_FINISHING = 1u << 0,
};

typedef enum iree_hal_amdgpu_staging_chunk_state_e {
  // Chunk is available for a new file subrange.
  IREE_HAL_AMDGPU_STAGING_CHUNK_IDLE = 0,
  // Chunk has an in-flight async file read into its staging slot.
  IREE_HAL_AMDGPU_STAGING_CHUNK_READING = 1,
  // Chunk has an in-flight GPU copy from staging to the user buffer.
  IREE_HAL_AMDGPU_STAGING_CHUNK_COPYING_TO_DEVICE = 2,
  // Chunk has an in-flight GPU copy from the user buffer to staging.
  IREE_HAL_AMDGPU_STAGING_CHUNK_COPYING_TO_HOST = 3,
  // Chunk has an in-flight async file write from its staging slot.
  IREE_HAL_AMDGPU_STAGING_CHUNK_WRITING = 4,
} iree_hal_amdgpu_staging_chunk_state_t;

typedef struct iree_hal_amdgpu_staging_transfer_t
    iree_hal_amdgpu_staging_transfer_t;

typedef struct iree_hal_amdgpu_staging_chunk_t {
  // Owning transfer.
  iree_hal_amdgpu_staging_transfer_t* transfer;
  // Current lifecycle state.
  iree_hal_amdgpu_staging_chunk_state_t state;
  // Staging slot owned by this chunk while |state| is not IDLE.
  iree_hal_amdgpu_staging_slot_t slot;
  // Byte offset from the transfer start.
  iree_device_size_t transfer_offset;
  // Byte length assigned to this chunk.
  iree_host_size_t length;
  // Bytes completed by the current partial file operation.
  iree_host_size_t file_progress;
  // Owned status captured by the GPU copy pre-signal action for post-drain use.
  iree_status_t copy_status;
  // Post-drain continuation queued by the GPU copy pre-signal action.
  iree_hal_amdgpu_host_queue_post_drain_action_t post_drain_action;
  // Async read operation storage.
  iree_async_file_read_operation_t read_op;
  // Async write operation storage.
  iree_async_file_write_operation_t write_op;
} iree_hal_amdgpu_staging_chunk_t;

struct iree_hal_amdgpu_staging_transfer_t {
  // Resource header retained by host actions, async file callbacks, and GPU
  // copy reclaim entries.
  iree_hal_resource_t resource;
  // Host allocator used for this transfer and cloned semaphore-list storage.
  iree_allocator_t host_allocator;
  // Serializes transfer counters and terminal status ownership.
  iree_slim_mutex_t mutex;
  // Queue used for internal GPU copies and final user signal publication.
  iree_hal_amdgpu_host_queue_t* queue;
  // Physical-device staging pool used by this transfer.
  iree_hal_amdgpu_staging_pool_t* pool;
  // Logical device retained while asynchronous transfer work is pending.
  iree_hal_device_t* logical_device;
  // File being read or written.
  iree_hal_file_t* file;
  // Async file handle borrowed from |file|.
  iree_async_file_t* async_file;
  // User buffer being copied to or from.
  iree_hal_buffer_t* buffer;
  // File byte offset for the first requested byte.
  uint64_t file_offset;
  // User buffer byte offset for the first requested byte.
  iree_device_size_t buffer_offset;
  // Total requested transfer length.
  iree_device_size_t requested_length;
  // Number of bytes assigned to chunks.
  iree_device_size_t submitted_length;
  // Number of bytes fully transferred through all stages.
  iree_device_size_t completed_length;
  // Number of chunks currently owning a staging slot or in-flight operation.
  uint32_t active_chunk_count;
  // Number of chunk records in |chunks|.
  uint32_t chunk_count;
  // Number of wait semaphores supplied to the queue_read/write operation.
  uint32_t profile_wait_count;
  // Direction of this transfer.
  iree_hal_amdgpu_staging_transfer_kind_t kind;
  // Transfer lifecycle flags from iree_hal_amdgpu_staging_transfer_flags_t.
  iree_hal_amdgpu_staging_transfer_flags_t flags;
  // Owned first failure status, or OK if no failure has occurred.
  iree_status_t failure_status;
  // Waiter queued when all staging slots are temporarily unavailable.
  iree_hal_amdgpu_staging_pool_waiter_t slot_waiter;
  // Completion-thread retry queued when the final signal barrier is blocked by
  // temporary queue capacity pressure.
  iree_hal_amdgpu_host_queue_post_drain_action_t signal_capacity_retry;
  // Cloned signal list published after the transfer completes.
  iree_hal_semaphore_list_t signal_semaphore_list;
  // Chunk records used to pipeline file I/O and GPU copies.
  iree_hal_amdgpu_staging_chunk_t* chunks;
};

static void iree_hal_amdgpu_staging_transfer_pump(
    iree_hal_amdgpu_staging_transfer_t* transfer);

static void iree_hal_amdgpu_staging_copy_post_drain(void* user_data);
static void iree_hal_amdgpu_staging_copy_capacity_post_drain(void* user_data);
static void iree_hal_amdgpu_staging_signal_capacity_post_drain(void* user_data);

static void iree_hal_amdgpu_staging_transfer_destroy(
    iree_hal_resource_t* resource) {
  iree_hal_amdgpu_staging_transfer_t* transfer =
      (iree_hal_amdgpu_staging_transfer_t*)resource;
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!iree_hal_semaphore_list_is_empty(transfer->signal_semaphore_list)) {
    iree_hal_semaphore_list_free(transfer->signal_semaphore_list,
                                 transfer->host_allocator);
  }
  iree_hal_buffer_release(transfer->buffer);
  iree_hal_file_release(transfer->file);
  iree_hal_device_release(transfer->logical_device);
  iree_slim_mutex_deinitialize(&transfer->mutex);
  iree_allocator_free(transfer->host_allocator, transfer);
  IREE_TRACE_ZONE_END(z0);
}

static const iree_hal_resource_vtable_t
    iree_hal_amdgpu_staging_transfer_vtable = {
        .destroy = iree_hal_amdgpu_staging_transfer_destroy,
};

static iree_status_t iree_hal_amdgpu_staging_transfer_clone_queue_error(
    iree_hal_amdgpu_staging_transfer_t* transfer) {
  iree_status_t error = (iree_status_t)iree_atomic_load(
      &transfer->queue->error_status, iree_memory_order_acquire);
  return iree_status_is_ok(error) ? iree_ok_status() : iree_status_clone(error);
}

static void iree_hal_amdgpu_staging_transfer_record_failure(
    iree_hal_amdgpu_staging_transfer_t* transfer, iree_status_t status) {
  if (iree_status_is_ok(status)) return;
  iree_slim_mutex_lock(&transfer->mutex);
  if (iree_status_is_ok(transfer->failure_status)) {
    transfer->failure_status = status;
    status = iree_ok_status();
  }
  iree_slim_mutex_unlock(&transfer->mutex);
  iree_status_free(status);
}

static iree_status_t iree_hal_amdgpu_staging_transfer_submit_signal_barrier(
    iree_hal_amdgpu_staging_transfer_t* transfer) {
  if (iree_hal_semaphore_list_is_empty(transfer->signal_semaphore_list)) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_staging_transfer_clone_queue_error(transfer));

  iree_hal_amdgpu_wait_resolution_t resolution;
  memset(&resolution, 0, sizeof(resolution));
  resolution.inline_acquire_scope = IREE_HSA_FENCE_SCOPE_SYSTEM;
  resolution.barrier_acquire_scope = IREE_HSA_FENCE_SCOPE_SYSTEM;

  iree_slim_mutex_lock(&transfer->queue->locks.submission_mutex);
  bool ready = false;
  uint64_t submission_id = 0;
  iree_hal_amdgpu_host_queue_profile_event_info_t profile_event_info = {
      .type =
          iree_hal_amdgpu_staging_transfer_profile_event_type(transfer->kind),
      .payload_length = transfer->requested_length,
      .operation_count = 1,
  };
  iree_status_t status = iree_hal_amdgpu_host_queue_try_submit_barrier(
      transfer->queue, &resolution, transfer->signal_semaphore_list,
      (iree_hal_amdgpu_reclaim_action_t){0},
      /*operation_resources=*/NULL, /*operation_resource_count=*/0,
      &profile_event_info,
      iree_hal_amdgpu_host_queue_post_commit_callback_null(),
      /*resource_set=*/NULL,
      IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_RETAIN_RESOURCES, &ready,
      &submission_id);
  if (iree_status_is_ok(status) && ready) {
    iree_hal_amdgpu_wait_resolution_t profile_resolution = resolution;
    profile_resolution.wait_count = transfer->profile_wait_count;
    profile_event_info.submission_id = submission_id;
    iree_hal_amdgpu_host_queue_record_profile_queue_event(
        transfer->queue, &profile_resolution, transfer->signal_semaphore_list,
        &profile_event_info);
  }
  if (iree_status_is_ok(status) && !ready) {
    iree_hal_resource_retain(&transfer->resource);
    iree_hal_amdgpu_host_queue_enqueue_post_drain_action(
        transfer->queue, &transfer->signal_capacity_retry,
        iree_hal_amdgpu_staging_signal_capacity_post_drain, transfer);
  }
  iree_slim_mutex_unlock(&transfer->queue->locks.submission_mutex);
  return status;
}

static void iree_hal_amdgpu_staging_transfer_fail_signals(
    iree_hal_amdgpu_staging_transfer_t* transfer, iree_status_t status) {
  if (iree_status_is_ok(status)) return;
  if (iree_hal_semaphore_list_is_empty(transfer->signal_semaphore_list)) {
    iree_status_free(status);
    return;
  }
  iree_hal_semaphore_list_fail(transfer->signal_semaphore_list, status);
}

static void iree_hal_amdgpu_staging_transfer_fail_signals_with_borrowed_status(
    iree_hal_amdgpu_staging_transfer_t* transfer, iree_status_t status) {
  if (iree_status_is_ok(status) ||
      iree_hal_semaphore_list_is_empty(transfer->signal_semaphore_list)) {
    return;
  }
  iree_hal_semaphore_list_fail(transfer->signal_semaphore_list,
                               iree_status_clone(status));
}

static void iree_hal_amdgpu_staging_transfer_complete(
    iree_hal_amdgpu_staging_transfer_t* transfer, iree_status_t status) {
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_staging_transfer_submit_signal_barrier(transfer);
  }
  iree_hal_amdgpu_staging_transfer_fail_signals(transfer, status);
  iree_hal_resource_release(&transfer->resource);
}

static void iree_hal_amdgpu_staging_signal_capacity_post_drain(
    void* user_data) {
  iree_hal_amdgpu_staging_transfer_complete(
      (iree_hal_amdgpu_staging_transfer_t*)user_data, iree_ok_status());
}

static void iree_hal_amdgpu_staging_transfer_try_finish(
    iree_hal_amdgpu_staging_transfer_t* transfer) {
  bool should_complete = false;
  bool should_release_waiter_ref = false;
  iree_status_t status = iree_ok_status();

  iree_slim_mutex_lock(&transfer->mutex);
  const bool has_failure = !iree_status_is_ok(transfer->failure_status);
  const bool is_complete =
      transfer->completed_length == transfer->requested_length;
  if (!iree_any_bit_set(transfer->flags,
                        IREE_HAL_AMDGPU_STAGING_TRANSFER_FLAG_FINISHING) &&
      transfer->active_chunk_count == 0 && (has_failure || is_complete)) {
    transfer->flags |= IREE_HAL_AMDGPU_STAGING_TRANSFER_FLAG_FINISHING;
    status = transfer->failure_status;
    transfer->failure_status = iree_ok_status();
    should_complete = true;
  }
  iree_slim_mutex_unlock(&transfer->mutex);

  if (should_complete && iree_hal_amdgpu_staging_pool_cancel_waiter(
                             transfer->pool, &transfer->slot_waiter)) {
    should_release_waiter_ref = true;
  }
  if (should_release_waiter_ref) {
    iree_hal_resource_release(&transfer->resource);
  }
  if (should_complete) {
    iree_hal_amdgpu_staging_transfer_complete(transfer, status);
  }
}

static void iree_hal_amdgpu_staging_chunk_return_slot(
    iree_hal_amdgpu_staging_chunk_t* chunk) {
  iree_hal_amdgpu_staging_pool_t* pool = chunk->transfer->pool;
  const uint32_t slot_ordinal = chunk->slot.ordinal;
  memset(&chunk->slot, 0, sizeof(chunk->slot));
  iree_hal_amdgpu_staging_pool_release(pool, slot_ordinal);
}

static void iree_hal_amdgpu_staging_chunk_finish(
    iree_hal_amdgpu_staging_chunk_t* chunk, bool did_transfer_bytes) {
  iree_hal_amdgpu_staging_transfer_t* transfer = chunk->transfer;
  iree_slim_mutex_lock(&transfer->mutex);
  if (did_transfer_bytes) {
    transfer->completed_length += chunk->length;
  }
  chunk->state = IREE_HAL_AMDGPU_STAGING_CHUNK_IDLE;
  chunk->length = 0;
  chunk->file_progress = 0;
  --transfer->active_chunk_count;
  iree_slim_mutex_unlock(&transfer->mutex);
  iree_hal_amdgpu_staging_chunk_return_slot(chunk);
  iree_hal_amdgpu_staging_transfer_pump(transfer);
  iree_hal_amdgpu_staging_transfer_try_finish(transfer);
}

static void iree_hal_amdgpu_staging_chunk_fail(
    iree_hal_amdgpu_staging_chunk_t* chunk, iree_status_t status) {
  iree_hal_amdgpu_staging_transfer_record_failure(chunk->transfer, status);
  iree_hal_amdgpu_staging_chunk_finish(chunk, /*did_transfer_bytes=*/false);
}

static iree_status_t iree_hal_amdgpu_staging_chunk_submit_read(
    iree_hal_amdgpu_staging_chunk_t* chunk);

static iree_status_t iree_hal_amdgpu_staging_chunk_submit_write(
    iree_hal_amdgpu_staging_chunk_t* chunk);

static void iree_hal_amdgpu_staging_copy_pre_signal(
    iree_hal_amdgpu_reclaim_entry_t* entry, void* user_data,
    iree_status_t status) {
  (void)entry;
  iree_hal_amdgpu_staging_chunk_t* chunk =
      (iree_hal_amdgpu_staging_chunk_t*)user_data;
  chunk->copy_status =
      iree_status_is_ok(status) ? iree_ok_status() : iree_status_clone(status);
  iree_hal_resource_retain(&chunk->transfer->resource);
  iree_hal_amdgpu_host_queue_enqueue_post_drain_action(
      chunk->transfer->queue, &chunk->post_drain_action,
      iree_hal_amdgpu_staging_copy_post_drain, chunk);
}

static iree_status_t iree_hal_amdgpu_staging_chunk_submit_copy(
    iree_hal_amdgpu_staging_chunk_t* chunk) {
  iree_hal_amdgpu_staging_transfer_t* transfer = chunk->transfer;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_staging_transfer_clone_queue_error(transfer));

  iree_hal_amdgpu_wait_resolution_t resolution;
  memset(&resolution, 0, sizeof(resolution));
  resolution.inline_acquire_scope = IREE_HSA_FENCE_SCOPE_NONE;
  resolution.barrier_acquire_scope = IREE_HSA_FENCE_SCOPE_NONE;

  iree_hal_buffer_t* source_buffer = NULL;
  iree_device_size_t source_offset = 0;
  iree_hal_buffer_t* target_buffer = NULL;
  iree_device_size_t target_offset = 0;
  iree_hsa_fence_scope_t minimum_acquire_scope = IREE_HSA_FENCE_SCOPE_NONE;
  iree_hsa_fence_scope_t minimum_release_scope = IREE_HSA_FENCE_SCOPE_NONE;
  if (transfer->kind == IREE_HAL_AMDGPU_STAGING_TRANSFER_READ) {
    source_buffer = chunk->slot.buffer;
    source_offset = chunk->slot.buffer_offset;
    target_buffer = transfer->buffer;
    target_offset = transfer->buffer_offset + chunk->transfer_offset;
    minimum_acquire_scope = IREE_HSA_FENCE_SCOPE_SYSTEM;
    chunk->state = IREE_HAL_AMDGPU_STAGING_CHUNK_COPYING_TO_DEVICE;
  } else {
    source_buffer = transfer->buffer;
    source_offset = transfer->buffer_offset + chunk->transfer_offset;
    target_buffer = chunk->slot.buffer;
    target_offset = chunk->slot.buffer_offset;
    minimum_release_scope = IREE_HSA_FENCE_SCOPE_SYSTEM;
    chunk->state = IREE_HAL_AMDGPU_STAGING_CHUNK_COPYING_TO_HOST;
  }

  iree_hal_resource_t* extra_resources[1] = {&transfer->resource};
  iree_slim_mutex_lock(&transfer->queue->locks.submission_mutex);
  bool ready = false;
  iree_status_t status = iree_hal_amdgpu_host_queue_submit_copy_with_action(
      transfer->queue, &resolution, iree_hal_semaphore_list_empty(),
      source_buffer, source_offset, target_buffer, target_offset, chunk->length,
      IREE_HAL_COPY_FLAG_NONE, minimum_acquire_scope, minimum_release_scope,
      (iree_hal_amdgpu_reclaim_action_t){
          .fn = iree_hal_amdgpu_staging_copy_pre_signal,
          .user_data = chunk,
      },
      extra_resources, IREE_ARRAYSIZE(extra_resources),
      IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_RETAIN_RESOURCES, &ready);
  if (iree_status_is_ok(status) && !ready) {
    iree_hal_resource_retain(&transfer->resource);
    iree_hal_amdgpu_host_queue_enqueue_post_drain_action(
        transfer->queue, &chunk->post_drain_action,
        iree_hal_amdgpu_staging_copy_capacity_post_drain, chunk);
  }
  iree_slim_mutex_unlock(&transfer->queue->locks.submission_mutex);
  return status;
}

static void iree_hal_amdgpu_staging_read_complete(
    void* user_data, iree_async_operation_t* base_operation,
    iree_status_t status, iree_async_completion_flags_t flags) {
  (void)base_operation;
  (void)flags;
  iree_hal_amdgpu_staging_chunk_t* chunk =
      (iree_hal_amdgpu_staging_chunk_t*)user_data;

  if (iree_status_is_ok(status) && chunk->read_op.bytes_read > 0) {
    chunk->file_progress += chunk->read_op.bytes_read;
    if (chunk->file_progress < chunk->length) {
      status = iree_hal_amdgpu_staging_chunk_submit_read(chunk);
      if (iree_status_is_ok(status)) {
        iree_hal_resource_release(&chunk->transfer->resource);
        return;
      }
    }
  } else if (iree_status_is_ok(status) &&
             chunk->file_progress < chunk->length) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "short read: requested %" PRIhsz
                              " bytes, got %" PRIhsz,
                              chunk->length, chunk->file_progress);
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_staging_chunk_submit_copy(chunk);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_amdgpu_staging_chunk_fail(chunk, status);
  }
  iree_hal_resource_release(&chunk->transfer->resource);
}

static void iree_hal_amdgpu_staging_write_complete(
    void* user_data, iree_async_operation_t* base_operation,
    iree_status_t status, iree_async_completion_flags_t flags) {
  (void)base_operation;
  (void)flags;
  iree_hal_amdgpu_staging_chunk_t* chunk =
      (iree_hal_amdgpu_staging_chunk_t*)user_data;

  if (iree_status_is_ok(status) && chunk->write_op.bytes_written > 0) {
    chunk->file_progress += chunk->write_op.bytes_written;
    if (chunk->file_progress < chunk->length) {
      status = iree_hal_amdgpu_staging_chunk_submit_write(chunk);
      if (iree_status_is_ok(status)) {
        iree_hal_resource_release(&chunk->transfer->resource);
        return;
      }
    }
  } else if (iree_status_is_ok(status) &&
             chunk->file_progress < chunk->length) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "short write: requested %" PRIhsz
                              " bytes, wrote %" PRIhsz,
                              chunk->length, chunk->file_progress);
  }

  if (!iree_status_is_ok(status)) {
    iree_hal_amdgpu_staging_chunk_fail(chunk, status);
  } else {
    iree_hal_amdgpu_staging_chunk_finish(chunk, /*did_transfer_bytes=*/true);
  }
  iree_hal_resource_release(&chunk->transfer->resource);
}

static iree_status_t iree_hal_amdgpu_staging_chunk_submit_read(
    iree_hal_amdgpu_staging_chunk_t* chunk) {
  iree_hal_amdgpu_staging_transfer_t* transfer = chunk->transfer;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_staging_transfer_clone_queue_error(transfer));

  iree_async_operation_zero(&chunk->read_op.base, sizeof(chunk->read_op));
  iree_async_operation_initialize(&chunk->read_op.base,
                                  IREE_ASYNC_OPERATION_TYPE_FILE_READ,
                                  IREE_ASYNC_OPERATION_FLAG_NONE,
                                  iree_hal_amdgpu_staging_read_complete, chunk);
  chunk->read_op.file = transfer->async_file;
  chunk->read_op.offset =
      transfer->file_offset + chunk->transfer_offset + chunk->file_progress;
  chunk->read_op.buffer = iree_async_span_from_ptr(
      chunk->slot.host_span.data + chunk->file_progress,
      chunk->length - chunk->file_progress);
  iree_hal_resource_retain(&transfer->resource);
  iree_status_t status = iree_async_proactor_submit_one(
      transfer->queue->proactor, &chunk->read_op.base);
  if (!iree_status_is_ok(status)) {
    iree_hal_resource_release(&transfer->resource);
  }
  return status;
}

static iree_status_t iree_hal_amdgpu_staging_chunk_submit_write(
    iree_hal_amdgpu_staging_chunk_t* chunk) {
  iree_hal_amdgpu_staging_transfer_t* transfer = chunk->transfer;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_staging_transfer_clone_queue_error(transfer));

  iree_async_operation_zero(&chunk->write_op.base, sizeof(chunk->write_op));
  iree_async_operation_initialize(
      &chunk->write_op.base, IREE_ASYNC_OPERATION_TYPE_FILE_WRITE,
      IREE_ASYNC_OPERATION_FLAG_NONE, iree_hal_amdgpu_staging_write_complete,
      chunk);
  chunk->write_op.file = transfer->async_file;
  chunk->write_op.offset =
      transfer->file_offset + chunk->transfer_offset + chunk->file_progress;
  chunk->write_op.buffer = iree_async_span_from_ptr(
      chunk->slot.host_span.data + chunk->file_progress,
      chunk->length - chunk->file_progress);
  iree_hal_resource_retain(&transfer->resource);
  iree_status_t status = iree_async_proactor_submit_one(
      transfer->queue->proactor, &chunk->write_op.base);
  if (!iree_status_is_ok(status)) {
    iree_hal_resource_release(&transfer->resource);
  }
  return status;
}

static void iree_hal_amdgpu_staging_copy_post_drain(void* user_data) {
  iree_hal_amdgpu_staging_chunk_t* chunk =
      (iree_hal_amdgpu_staging_chunk_t*)user_data;
  iree_status_t status = chunk->copy_status;
  chunk->copy_status = iree_ok_status();

  if (iree_status_is_ok(status) &&
      chunk->transfer->kind == IREE_HAL_AMDGPU_STAGING_TRANSFER_WRITE) {
    status = iree_hal_amdgpu_staging_chunk_submit_write(chunk);
    if (iree_status_is_ok(status)) {
      iree_hal_resource_release(&chunk->transfer->resource);
      return;
    }
  }

  if (!iree_status_is_ok(status)) {
    iree_hal_amdgpu_staging_chunk_fail(chunk, status);
  } else {
    iree_hal_amdgpu_staging_chunk_finish(chunk, /*did_transfer_bytes=*/true);
  }
  iree_hal_resource_release(&chunk->transfer->resource);
}

static void iree_hal_amdgpu_staging_copy_capacity_post_drain(void* user_data) {
  iree_hal_amdgpu_staging_chunk_t* chunk =
      (iree_hal_amdgpu_staging_chunk_t*)user_data;
  iree_status_t status = iree_hal_amdgpu_staging_chunk_submit_copy(chunk);
  if (!iree_status_is_ok(status)) {
    iree_hal_amdgpu_staging_chunk_fail(chunk, status);
  }
  iree_hal_resource_release(&chunk->transfer->resource);
}

static void iree_hal_amdgpu_staging_transfer_slot_available(void* user_data) {
  iree_hal_amdgpu_staging_transfer_t* transfer =
      (iree_hal_amdgpu_staging_transfer_t*)user_data;
  iree_hal_amdgpu_staging_transfer_pump(transfer);
  iree_hal_amdgpu_staging_transfer_try_finish(transfer);
  iree_hal_resource_release(&transfer->resource);
}

static iree_hal_amdgpu_staging_chunk_t*
iree_hal_amdgpu_staging_transfer_find_idle_chunk(
    iree_hal_amdgpu_staging_transfer_t* transfer) {
  for (uint32_t i = 0; i < transfer->chunk_count; ++i) {
    if (transfer->chunks[i].state == IREE_HAL_AMDGPU_STAGING_CHUNK_IDLE) {
      return &transfer->chunks[i];
    }
  }
  return NULL;
}

static void iree_hal_amdgpu_staging_transfer_pump(
    iree_hal_amdgpu_staging_transfer_t* transfer) {
  for (;;) {
    iree_hal_amdgpu_staging_slot_t slot;
    memset(&slot, 0, sizeof(slot));
    const bool has_waiter_slot = iree_hal_amdgpu_staging_pool_take_waiter_slot(
        transfer->pool, &transfer->slot_waiter, &slot);

    iree_slim_mutex_lock(&transfer->mutex);
    const bool can_submit_more =
        !iree_any_bit_set(transfer->flags,
                          IREE_HAL_AMDGPU_STAGING_TRANSFER_FLAG_FINISHING) &&
        iree_status_is_ok(transfer->failure_status) &&
        transfer->submitted_length < transfer->requested_length;
    iree_slim_mutex_unlock(&transfer->mutex);
    if (!can_submit_more) {
      if (has_waiter_slot) {
        iree_hal_amdgpu_staging_pool_release(transfer->pool, slot.ordinal);
      }
      return;
    }

    if (!has_waiter_slot &&
        !iree_hal_amdgpu_staging_pool_try_acquire(transfer->pool, &slot)) {
      iree_hal_amdgpu_staging_pool_wait_result_t wait_result =
          iree_hal_amdgpu_staging_pool_queue_waiter(
              transfer->pool, &transfer->slot_waiter,
              iree_hal_amdgpu_staging_transfer_slot_available, transfer);
      if (wait_result == IREE_HAL_AMDGPU_STAGING_POOL_WAIT_QUEUED) {
        iree_hal_resource_retain(&transfer->resource);
      }
      if (wait_result != IREE_HAL_AMDGPU_STAGING_POOL_WAIT_RETRY) {
        return;
      }
      continue;
    }

    iree_hal_amdgpu_staging_chunk_t* chunk = NULL;
    iree_device_size_t chunk_offset = 0;
    iree_host_size_t chunk_length = 0;
    bool should_release_slot = false;
    iree_slim_mutex_lock(&transfer->mutex);
    const bool has_failure = !iree_status_is_ok(transfer->failure_status);
    const bool has_more_bytes =
        transfer->submitted_length < transfer->requested_length;
    if (!iree_any_bit_set(transfer->flags,
                          IREE_HAL_AMDGPU_STAGING_TRANSFER_FLAG_FINISHING) &&
        !has_failure && has_more_bytes) {
      chunk = iree_hal_amdgpu_staging_transfer_find_idle_chunk(transfer);
    }
    if (chunk) {
      const iree_device_size_t remaining_length =
          transfer->requested_length - transfer->submitted_length;
      chunk_length = (iree_host_size_t)iree_min(
          (iree_device_size_t)transfer->pool->slot_size, remaining_length);
      chunk_offset = transfer->submitted_length;
      chunk->state = transfer->kind == IREE_HAL_AMDGPU_STAGING_TRANSFER_READ
                         ? IREE_HAL_AMDGPU_STAGING_CHUNK_READING
                         : IREE_HAL_AMDGPU_STAGING_CHUNK_COPYING_TO_HOST;
      chunk->slot = slot;
      chunk->transfer_offset = chunk_offset;
      chunk->length = chunk_length;
      chunk->file_progress = 0;
      chunk->copy_status = iree_ok_status();
      transfer->submitted_length += chunk_length;
      ++transfer->active_chunk_count;
    } else {
      should_release_slot = true;
    }
    iree_slim_mutex_unlock(&transfer->mutex);

    if (should_release_slot) {
      iree_hal_amdgpu_staging_pool_release(transfer->pool, slot.ordinal);
      return;
    }

    iree_status_t status =
        transfer->kind == IREE_HAL_AMDGPU_STAGING_TRANSFER_READ
            ? iree_hal_amdgpu_staging_chunk_submit_read(chunk)
            : iree_hal_amdgpu_staging_chunk_submit_copy(chunk);
    if (!iree_status_is_ok(status)) {
      iree_hal_amdgpu_staging_chunk_fail(chunk, status);
      return;
    }
  }
}

static iree_status_t iree_hal_amdgpu_staging_transfer_start(
    iree_hal_amdgpu_staging_transfer_t* transfer) {
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_staging_transfer_clone_queue_error(transfer));
  // The transfer buffer may be a queue_alloca result whose backing is only
  // staged when the operation is submitted. Validate the device pointer here,
  // after the host action's wait set has been satisfied.
  iree_hal_buffer_t* allocated_buffer =
      iree_hal_buffer_allocated_buffer(transfer->buffer);
  if (IREE_UNLIKELY(!iree_hal_amdgpu_buffer_device_pointer(allocated_buffer))) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "staged AMDGPU file transfer buffer was not backed by an AMDGPU "
        "allocation after queue waits completed");
  }
  // The host action's reclaim entry owns |transfer| only until this callback
  // returns. Keep a transfer-owned self reference across the async file/GPU
  // copy pipeline; the terminal completion path releases it.
  iree_hal_resource_retain(&transfer->resource);
  iree_hal_amdgpu_staging_transfer_pump(transfer);
  iree_hal_amdgpu_staging_transfer_try_finish(transfer);
  return iree_ok_status();
}

static void iree_hal_amdgpu_staging_transfer_execute(
    iree_hal_amdgpu_reclaim_entry_t* entry, void* user_data,
    iree_status_t status) {
  (void)entry;
  iree_hal_amdgpu_staging_transfer_t* transfer =
      (iree_hal_amdgpu_staging_transfer_t*)user_data;

  if (!iree_status_is_ok(status)) {
    iree_hal_amdgpu_staging_transfer_fail_signals_with_borrowed_status(transfer,
                                                                       status);
    return;
  }

  iree_status_t start_status = iree_hal_amdgpu_staging_transfer_start(transfer);
  if (!iree_status_is_ok(start_status)) {
    iree_hal_amdgpu_staging_transfer_fail_signals(transfer, start_status);
  }
}

static iree_status_t iree_hal_amdgpu_staging_transfer_validate_buffer(
    iree_hal_amdgpu_staging_transfer_kind_t kind, iree_hal_buffer_t* buffer,
    iree_device_size_t buffer_offset, iree_device_size_t length) {
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_validate_range(buffer, buffer_offset, length));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_usage(
      iree_hal_buffer_allowed_usage(buffer),
      kind == IREE_HAL_AMDGPU_STAGING_TRANSFER_READ
          ? IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET
          : IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_access(
      iree_hal_buffer_allowed_access(buffer),
      kind == IREE_HAL_AMDGPU_STAGING_TRANSFER_READ
          ? IREE_HAL_MEMORY_ACCESS_WRITE
          : IREE_HAL_MEMORY_ACCESS_READ));
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_staging_transfer_create(
    iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_amdgpu_staging_transfer_kind_t kind, iree_hal_file_t* file,
    uint64_t file_offset, iree_hal_buffer_t* buffer,
    iree_device_size_t buffer_offset, iree_device_size_t length,
    uint32_t profile_wait_count,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_amdgpu_staging_transfer_t** out_transfer) {
  *out_transfer = NULL;
  if (IREE_UNLIKELY(!queue->staging_pool || !queue->staging_pool->buffer)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU queue file staging pool is not initialized");
  }
  if (IREE_UNLIKELY(!iree_hal_file_async_handle(file))) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "AMDGPU staged queue file transfers require a proactor-backed async "
        "file handle");
  }
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_staging_transfer_validate_buffer(
      kind, buffer, buffer_offset, length));

  iree_host_size_t chunks_size = 0;
  if (!iree_host_size_checked_mul(queue->staging_pool->slot_count,
                                  sizeof(iree_hal_amdgpu_staging_chunk_t),
                                  &chunks_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "staging transfer chunk table size overflows");
  }
  iree_host_size_t total_size = 0;
  if (!iree_host_size_checked_add(sizeof(iree_hal_amdgpu_staging_transfer_t),
                                  chunks_size, &total_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "staging transfer allocation size overflows");
  }

  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, length);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, queue->staging_pool->slot_count);
  iree_hal_amdgpu_staging_transfer_t* transfer = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(queue->host_allocator, total_size,
                                (void**)&transfer));
  memset(transfer, 0, total_size);
  iree_hal_resource_initialize(&iree_hal_amdgpu_staging_transfer_vtable,
                               &transfer->resource);
  transfer->host_allocator = queue->host_allocator;
  iree_slim_mutex_initialize(&transfer->mutex);
  transfer->queue = queue;
  transfer->pool = queue->staging_pool;
  transfer->logical_device = queue->logical_device;
  iree_hal_device_retain(transfer->logical_device);
  transfer->file = file;
  iree_hal_file_retain(transfer->file);
  transfer->async_file = iree_hal_file_async_handle(file);
  transfer->buffer = buffer;
  iree_hal_buffer_retain(transfer->buffer);
  transfer->file_offset = file_offset;
  transfer->buffer_offset = buffer_offset;
  transfer->requested_length = length;
  transfer->chunk_count = queue->staging_pool->slot_count;
  transfer->profile_wait_count = profile_wait_count;
  transfer->kind = kind;
  transfer->chunks = (iree_hal_amdgpu_staging_chunk_t*)(transfer + 1);
  for (uint32_t i = 0; i < transfer->chunk_count; ++i) {
    transfer->chunks[i].transfer = transfer;
  }

  iree_status_t status = iree_hal_semaphore_list_clone(
      &signal_semaphore_list, transfer->host_allocator,
      &transfer->signal_semaphore_list);
  if (iree_status_is_ok(status)) {
    *out_transfer = transfer;
  } else {
    iree_hal_resource_release(&transfer->resource);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_amdgpu_host_queue_submit_staged_transfer(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_amdgpu_staging_transfer_kind_t kind, iree_hal_file_t* file,
    uint64_t file_offset, iree_hal_buffer_t* buffer,
    iree_device_size_t buffer_offset, iree_device_size_t length) {
  iree_hal_amdgpu_staging_transfer_t* transfer = NULL;
  const uint32_t profile_wait_count =
      iree_hal_amdgpu_host_queue_profile_semaphore_count(wait_semaphore_list);
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_staging_transfer_create(
      queue, kind, file, file_offset, buffer, buffer_offset, length,
      profile_wait_count, signal_semaphore_list, &transfer));

  iree_hal_resource_t* resources[1] = {&transfer->resource};
  iree_status_t status = iree_hal_amdgpu_host_queue_enqueue_host_action(
      queue, wait_semaphore_list,
      (iree_hal_amdgpu_reclaim_action_t){
          .fn = iree_hal_amdgpu_staging_transfer_execute,
          .user_data = transfer,
      },
      resources, IREE_ARRAYSIZE(resources));
  iree_hal_resource_release(&transfer->resource);
  return status;
}

iree_status_t iree_hal_amdgpu_host_queue_submit_staged_read(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_file_t* source_file, uint64_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length) {
  return iree_hal_amdgpu_host_queue_submit_staged_transfer(
      queue, wait_semaphore_list, signal_semaphore_list,
      IREE_HAL_AMDGPU_STAGING_TRANSFER_READ, source_file, source_offset,
      target_buffer, target_offset, length);
}

iree_status_t iree_hal_amdgpu_host_queue_submit_staged_write(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_file_t* target_file, uint64_t target_offset,
    iree_device_size_t length) {
  return iree_hal_amdgpu_host_queue_submit_staged_transfer(
      queue, wait_semaphore_list, signal_semaphore_list,
      IREE_HAL_AMDGPU_STAGING_TRANSFER_WRITE, target_file, target_offset,
      source_buffer, source_offset, length);
}
