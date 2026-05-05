// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/semaphore.h"

#include <string.h>

#include "iree/base/internal/atomics.h"

//===----------------------------------------------------------------------===//
// iree_hal_vulkan_last_signal_t
//===----------------------------------------------------------------------===//

void iree_hal_vulkan_last_signal_store(
    iree_hal_vulkan_last_signal_t* cache,
    iree_hal_vulkan_last_signal_flags_t flags, iree_async_axis_t producer_axis,
    uint64_t epoch, uint64_t value) {
  iree_atomic_fetch_add(&cache->sequence, 1, iree_memory_order_acquire);
  cache->flags = flags;
  memset(cache->reserved, 0, sizeof(cache->reserved));
  cache->producer_axis = producer_axis;
  cache->epoch = epoch;
  cache->value = value;
  iree_atomic_fetch_add(&cache->sequence, 1, iree_memory_order_release);
}

bool iree_hal_vulkan_last_signal_load(
    const iree_hal_vulkan_last_signal_t* cache,
    iree_hal_vulkan_last_signal_flags_t* out_flags,
    iree_async_axis_t* out_producer_axis, uint64_t* out_epoch,
    uint64_t* out_value) {
  int32_t sequence = 0;
  do {
    sequence = iree_atomic_load(&cache->sequence, iree_memory_order_acquire);
    if (IREE_UNLIKELY(sequence & 1)) continue;
    *out_flags = cache->flags;
    *out_producer_axis = cache->producer_axis;
    *out_epoch = cache->epoch;
    *out_value = cache->value;
  } while (
      IREE_UNLIKELY(iree_atomic_load(&cache->sequence,
                                     iree_memory_order_acquire) != sequence));
  return (*out_flags & IREE_HAL_VULKAN_LAST_SIGNAL_FLAG_VALID) != 0;
}

//===----------------------------------------------------------------------===//
// iree_hal_vulkan_semaphore_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_vulkan_semaphore_t {
  // Embedded async semaphore at offset 0 for HAL/async toll-free bridging.
  iree_async_semaphore_t async;

  // Host allocator used to free this semaphore.
  iree_allocator_t host_allocator;

  // Device-level Vulkan dispatch table copied from the creating device.
  iree_hal_vulkan_device_syms_t syms;

  // Back-pointer to the logical device that created this semaphore.
  iree_hal_vulkan_logical_device_t* device;

  // Vulkan logical device that owns the semaphore handle.
  VkDevice logical_device;

  // Native Vulkan timeline semaphore handle.
  VkSemaphore handle;

  // Queue affinity provided at creation.
  iree_hal_queue_affinity_t queue_affinity;

  // Creation flags controlling synchronization behavior.
  iree_hal_semaphore_flags_t flags;

  // Seqlock-protected cache of the most recent queue signal.
  iree_hal_vulkan_last_signal_t last_signal;
} iree_hal_vulkan_semaphore_t;

static const iree_hal_semaphore_vtable_t iree_hal_vulkan_semaphore_vtable;

static iree_hal_vulkan_semaphore_t* iree_hal_vulkan_semaphore_cast(
    iree_hal_semaphore_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_vulkan_semaphore_vtable);
  return (iree_hal_vulkan_semaphore_t*)base_value;
}

static iree_status_code_t iree_hal_vulkan_semaphore_failure_code(
    VkResult result) {
  switch (result) {
    default:
      return IREE_STATUS_INTERNAL;
    case VK_ERROR_DEVICE_LOST:
      return IREE_STATUS_DATA_LOSS;
    case VK_ERROR_OUT_OF_HOST_MEMORY:
    case VK_ERROR_OUT_OF_DEVICE_MEMORY:
      return IREE_STATUS_RESOURCE_EXHAUSTED;
  }
}

iree_status_t iree_hal_vulkan_semaphore_create(
    iree_hal_vulkan_logical_device_t* device,
    const iree_hal_vulkan_device_syms_t* syms, VkDevice logical_device,
    iree_async_proactor_t* proactor, iree_hal_queue_affinity_t queue_affinity,
    uint64_t initial_value, iree_hal_semaphore_flags_t flags,
    iree_allocator_t host_allocator, iree_hal_semaphore_t** out_semaphore) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(syms);
  IREE_ASSERT_ARGUMENT(proactor);
  IREE_ASSERT_ARGUMENT(out_semaphore);
  *out_semaphore = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  if (initial_value > IREE_HAL_SEMAPHORE_MAX_VALUE) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Vulkan semaphore initial value %" PRIu64
                            " exceeds the maximum HAL semaphore value %" PRIu64,
                            initial_value,
                            (uint64_t)IREE_HAL_SEMAPHORE_MAX_VALUE);
  }

  VkSemaphoreTypeCreateInfo timeline_create_info = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
      .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
      .initialValue = initial_value,
  };
  VkSemaphoreCreateInfo create_info = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
      .pNext = &timeline_create_info,
  };

  VkSemaphore handle = VK_NULL_HANDLE;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_vkCreateSemaphore(IREE_VULKAN_DEVICE(syms), logical_device,
                                 &create_info, /*pAllocator=*/NULL, &handle));

  iree_hal_vulkan_semaphore_t* semaphore = NULL;
  iree_host_size_t frontier_offset = 0;
  iree_host_size_t total_size = 0;
  iree_status_t status = iree_async_semaphore_layout(
      sizeof(*semaphore), 0, &frontier_offset, &total_size);
  if (iree_status_is_ok(status)) {
    status =
        iree_allocator_malloc(host_allocator, total_size, (void**)&semaphore);
  }
  if (iree_status_is_ok(status)) {
    memset(semaphore, 0, total_size);
    iree_async_semaphore_initialize(
        (const iree_async_semaphore_vtable_t*)&iree_hal_vulkan_semaphore_vtable,
        proactor, initial_value, frontier_offset, 0, &semaphore->async);
    semaphore->host_allocator = host_allocator;
    semaphore->syms = *syms;
    semaphore->device = device;
    semaphore->logical_device = logical_device;
    semaphore->handle = handle;
    semaphore->queue_affinity = queue_affinity;
    semaphore->flags = flags;
    memset(&semaphore->last_signal, 0, sizeof(semaphore->last_signal));
    *out_semaphore = iree_hal_semaphore_cast(&semaphore->async);
  } else {
    iree_vkDestroySemaphore(IREE_VULKAN_DEVICE(syms), logical_device, handle,
                            /*pAllocator=*/NULL);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_vulkan_semaphore_destroy(
    iree_async_semaphore_t* base_semaphore) {
  iree_hal_vulkan_semaphore_t* semaphore =
      iree_hal_vulkan_semaphore_cast(iree_hal_semaphore_cast(base_semaphore));
  iree_allocator_t host_allocator = semaphore->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_vkDestroySemaphore(IREE_VULKAN_DEVICE(&semaphore->syms),
                          semaphore->logical_device, semaphore->handle,
                          /*pAllocator=*/NULL);
  iree_async_semaphore_deinitialize(&semaphore->async);
  iree_allocator_free(host_allocator, semaphore);

  IREE_TRACE_ZONE_END(z0);
}

bool iree_hal_vulkan_semaphore_isa(iree_hal_semaphore_t* semaphore) {
  return iree_hal_resource_is((const iree_hal_resource_t*)semaphore,
                              &iree_hal_vulkan_semaphore_vtable);
}

bool iree_hal_vulkan_semaphore_is_local(
    iree_hal_semaphore_t* semaphore,
    const iree_hal_vulkan_logical_device_t* device) {
  return iree_hal_resource_is((const iree_hal_resource_t*)semaphore,
                              &iree_hal_vulkan_semaphore_vtable) &&
         ((const iree_hal_vulkan_semaphore_t*)semaphore)->device == device;
}

iree_hal_semaphore_flags_t iree_hal_vulkan_semaphore_flags(
    iree_hal_semaphore_t* semaphore) {
  return ((const iree_hal_vulkan_semaphore_t*)semaphore)->flags;
}

iree_hal_queue_affinity_t iree_hal_vulkan_semaphore_queue_affinity(
    iree_hal_semaphore_t* semaphore) {
  return ((const iree_hal_vulkan_semaphore_t*)semaphore)->queue_affinity;
}

iree_status_t iree_hal_vulkan_semaphore_handle(
    iree_hal_semaphore_t* base_semaphore, VkSemaphore* out_handle) {
  IREE_ASSERT_ARGUMENT(base_semaphore);
  IREE_ASSERT_ARGUMENT(out_handle);
  iree_hal_vulkan_semaphore_t* semaphore =
      iree_hal_vulkan_semaphore_cast(base_semaphore);
  *out_handle = semaphore->handle;
  return iree_ok_status();
}

iree_hal_vulkan_last_signal_t* iree_hal_vulkan_semaphore_last_signal(
    iree_hal_semaphore_t* base_semaphore) {
  iree_hal_vulkan_semaphore_t* semaphore =
      iree_hal_vulkan_semaphore_cast(base_semaphore);
  return &semaphore->last_signal;
}

bool iree_hal_vulkan_semaphore_publish_signal(
    iree_hal_semaphore_t* base_semaphore, iree_async_axis_t producer_axis,
    const iree_async_frontier_t* producer_frontier, uint64_t producer_epoch,
    uint64_t producer_value) {
  IREE_ASSERT_ARGUMENT(producer_frontier);
  iree_hal_vulkan_semaphore_t* semaphore =
      iree_hal_vulkan_semaphore_cast(base_semaphore);

  iree_hal_vulkan_last_signal_flags_t flags =
      IREE_HAL_VULKAN_LAST_SIGNAL_FLAG_VALID;
  bool source_dominates_frontier = false;
  iree_slim_mutex_lock(&semaphore->async.mutex);
  const bool merged = iree_async_frontier_merge_and_test_source_dominance(
      semaphore->async.frontier, semaphore->async.frontier_capacity,
      producer_frontier, &source_dominates_frontier);
  if (merged && source_dominates_frontier) {
    flags |= IREE_HAL_VULKAN_LAST_SIGNAL_FLAG_PRODUCER_FRONTIER_EXACT;
  }
  iree_hal_vulkan_last_signal_store(
      &semaphore->last_signal, merged ? flags : 0,
      merged ? producer_axis : (iree_async_axis_t)0,
      merged ? producer_epoch : 0, merged ? producer_value : 0);
  iree_slim_mutex_unlock(&semaphore->async.mutex);

  return merged;
}

static uint64_t iree_hal_vulkan_semaphore_observe_native_value(
    iree_async_semaphore_t* base_semaphore, uint64_t value) {
  int64_t current_raw = 0;
  do {
    current_raw = iree_atomic_load(&base_semaphore->timeline_value,
                                   iree_memory_order_acquire);
    if (value <= (uint64_t)current_raw) return (uint64_t)current_raw;
  } while (!iree_atomic_compare_exchange_weak(
      &base_semaphore->timeline_value, &current_raw, (int64_t)value,
      iree_memory_order_release, iree_memory_order_relaxed));

  iree_async_semaphore_dispatch_timepoints(base_semaphore, value);
  return value;
}

iree_status_t iree_hal_vulkan_semaphore_retire_signal(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    const iree_async_frontier_t* frontier) {
  iree_hal_vulkan_semaphore_t* semaphore =
      iree_hal_vulkan_semaphore_cast(base_semaphore);
  iree_status_t status =
      iree_async_semaphore_signal_untainted(&semaphore->async, value, frontier);
  if (!iree_status_is_ok(status)) {
    iree_async_semaphore_fail(&semaphore->async, iree_status_clone(status));
    return status;
  }
  return iree_ok_status();
}

static uint64_t iree_hal_vulkan_semaphore_query(
    iree_async_semaphore_t* base_semaphore) {
  iree_hal_vulkan_semaphore_t* semaphore =
      iree_hal_vulkan_semaphore_cast(iree_hal_semaphore_cast(base_semaphore));

  iree_status_t failure_status = (iree_status_t)iree_atomic_load(
      &base_semaphore->failure_status, iree_memory_order_acquire);
  if (!iree_status_is_ok(failure_status)) {
    return iree_hal_status_as_semaphore_failure(failure_status);
  }

  uint64_t value = 0;
  VkResult result = iree_vkGetSemaphoreCounterValue_raw(
      &semaphore->syms, semaphore->logical_device, semaphore->handle, &value);
  if (IREE_UNLIKELY(result != VK_SUCCESS)) {
    return iree_hal_status_as_semaphore_failure(
        iree_status_from_code(iree_hal_vulkan_semaphore_failure_code(result)));
  }

  if (value >= IREE_HAL_SEMAPHORE_FAILURE_VALUE) {
    failure_status = (iree_status_t)iree_atomic_load(
        &base_semaphore->failure_status, iree_memory_order_acquire);
    if (!iree_status_is_ok(failure_status)) {
      return iree_hal_status_as_semaphore_failure(failure_status);
    }
    return IREE_HAL_SEMAPHORE_FAILURE_VALUE;
  }

  return iree_hal_vulkan_semaphore_observe_native_value(base_semaphore, value);
}

static iree_status_t iree_hal_vulkan_semaphore_signal(
    iree_async_semaphore_t* base_semaphore, uint64_t new_value,
    const iree_async_frontier_t* frontier) {
  if (new_value > IREE_HAL_SEMAPHORE_MAX_VALUE) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Vulkan semaphore signal value %" PRIu64
                            " exceeds the maximum HAL semaphore value %" PRIu64,
                            new_value, (uint64_t)IREE_HAL_SEMAPHORE_MAX_VALUE);
  }

  iree_hal_vulkan_semaphore_t* semaphore =
      iree_hal_vulkan_semaphore_cast(iree_hal_semaphore_cast(base_semaphore));
  iree_status_t status = iree_async_semaphore_advance_timeline(
      base_semaphore, new_value, frontier);
  if (!iree_status_is_ok(status)) return status;

  VkSemaphoreSignalInfo signal_info = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO,
      .semaphore = semaphore->handle,
      .value = new_value,
  };
  status = iree_vkSignalSemaphore(IREE_VULKAN_DEVICE(&semaphore->syms),
                                  semaphore->logical_device, &signal_info);
  if (!iree_status_is_ok(status)) {
    iree_async_semaphore_fail(base_semaphore, iree_status_clone(status));
    return status;
  }

  iree_async_semaphore_dispatch_timepoints(base_semaphore, new_value);
  return iree_ok_status();
}

static void iree_hal_vulkan_semaphore_on_fail(
    iree_async_semaphore_t* base_semaphore, iree_status_code_t status_code) {
  (void)status_code;
  iree_hal_vulkan_semaphore_t* semaphore =
      iree_hal_vulkan_semaphore_cast(iree_hal_semaphore_cast(base_semaphore));

  VkSemaphoreSignalInfo signal_info = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO,
      .semaphore = semaphore->handle,
      .value = IREE_HAL_SEMAPHORE_FAILURE_VALUE,
  };
  VkResult result = iree_vkSignalSemaphore_raw(
      &semaphore->syms, semaphore->logical_device, &signal_info);
  IREE_ASSERT(result == VK_SUCCESS || result == VK_ERROR_DEVICE_LOST);
  (void)result;
}

static uint64_t iree_hal_vulkan_timeout_to_nanoseconds(iree_timeout_t timeout) {
  iree_time_t deadline_ns = iree_timeout_as_deadline_ns(timeout);
  if (deadline_ns == IREE_TIME_INFINITE_FUTURE) return UINT64_MAX;
  if (deadline_ns == IREE_TIME_INFINITE_PAST) return 0;
  iree_time_t now_ns = iree_time_now();
  return deadline_ns < now_ns ? 0 : (uint64_t)(deadline_ns - now_ns);
}

static iree_status_t iree_hal_vulkan_semaphore_wait(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    iree_timeout_t timeout, iree_async_wait_flags_t flags) {
  (void)flags;
  iree_hal_vulkan_semaphore_t* semaphore =
      iree_hal_vulkan_semaphore_cast(base_semaphore);
  IREE_TRACE_ZONE_BEGIN(z0);

  VkSemaphore vk_semaphore = semaphore->handle;
  VkSemaphoreWaitInfo wait_info = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
      .semaphoreCount = 1,
      .pSemaphores = &vk_semaphore,
      .pValues = &value,
  };
  VkResult result = iree_vkWaitSemaphores_raw(
      &semaphore->syms, semaphore->logical_device, &wait_info,
      iree_hal_vulkan_timeout_to_nanoseconds(timeout));

  iree_status_t status = iree_ok_status();
  if (result == VK_SUCCESS) {
    iree_status_t failure_status = (iree_status_t)iree_atomic_load(
        &((iree_async_semaphore_t*)base_semaphore)->failure_status,
        iree_memory_order_acquire);
    if (!iree_status_is_ok(failure_status)) {
      status = iree_status_clone(failure_status);
    } else {
      uint64_t current_value = 0;
      result = iree_vkGetSemaphoreCounterValue_raw(
          &semaphore->syms, semaphore->logical_device, semaphore->handle,
          &current_value);
      if (result == VK_SUCCESS &&
          current_value < IREE_HAL_SEMAPHORE_FAILURE_VALUE) {
        iree_hal_vulkan_semaphore_observe_native_value(
            (iree_async_semaphore_t*)base_semaphore, current_value);
      } else if (result != VK_SUCCESS) {
        status = iree_status_from_vk_result(__FILE__, __LINE__, result,
                                            "vkGetSemaphoreCounterValue");
      }
    }
  } else {
    status = iree_status_from_vk_result(__FILE__, __LINE__, result,
                                        "vkWaitSemaphores");
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_vulkan_semaphore_import_timepoint(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    iree_hal_queue_affinity_t queue_affinity,
    iree_hal_external_timepoint_t external_timepoint) {
  (void)base_semaphore;
  (void)value;
  (void)queue_affinity;
  (void)external_timepoint;
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "Vulkan semaphore timepoint import requires external semaphore interop");
}

static iree_status_t iree_hal_vulkan_semaphore_export_timepoint(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    iree_hal_queue_affinity_t queue_affinity,
    iree_hal_external_timepoint_type_t requested_type,
    iree_hal_external_timepoint_flags_t requested_flags,
    iree_hal_external_timepoint_t* IREE_RESTRICT out_external_timepoint) {
  IREE_ASSERT_ARGUMENT(out_external_timepoint);
  (void)base_semaphore;
  (void)value;
  (void)queue_affinity;
  (void)requested_type;
  (void)requested_flags;
  memset(out_external_timepoint, 0, sizeof(*out_external_timepoint));
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "Vulkan semaphore timepoint export requires external semaphore interop");
}

static const iree_hal_semaphore_vtable_t iree_hal_vulkan_semaphore_vtable = {
    .async =
        {
            .destroy = iree_hal_vulkan_semaphore_destroy,
            .query = iree_hal_vulkan_semaphore_query,
            .signal = iree_hal_vulkan_semaphore_signal,
            .on_fail = iree_hal_vulkan_semaphore_on_fail,
        },
    .wait = iree_hal_vulkan_semaphore_wait,
    .import_timepoint = iree_hal_vulkan_semaphore_import_timepoint,
    .export_timepoint = iree_hal_vulkan_semaphore_export_timepoint,
};
