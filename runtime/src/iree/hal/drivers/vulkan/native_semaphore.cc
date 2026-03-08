// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/native_semaphore.h"

#include <cstddef>

#include "iree/base/api.h"
#include "iree/hal/drivers/vulkan/dynamic_symbol_tables.h"
#include "iree/hal/drivers/vulkan/dynamic_symbols.h"
#include "iree/hal/drivers/vulkan/status_util.h"
#include "iree/hal/drivers/vulkan/util/ref_ptr.h"

using namespace iree::hal::vulkan;

typedef struct iree_hal_vulkan_native_semaphore_t {
  iree_async_semaphore_t async;
  VkDeviceHandle* logical_device;
  VkSemaphore handle;
} iree_hal_vulkan_native_semaphore_t;

namespace {
extern const iree_hal_semaphore_vtable_t
    iree_hal_vulkan_native_semaphore_vtable;
}  // namespace

static iree_hal_vulkan_native_semaphore_t*
iree_hal_vulkan_native_semaphore_cast(iree_hal_semaphore_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_vulkan_native_semaphore_vtable);
  return (iree_hal_vulkan_native_semaphore_t*)base_value;
}

iree_status_t iree_hal_vulkan_native_semaphore_create(
    iree::hal::vulkan::VkDeviceHandle* logical_device, uint64_t initial_value,
    iree_hal_semaphore_t** out_semaphore) {
  IREE_ASSERT_ARGUMENT(logical_device);
  IREE_ASSERT_ARGUMENT(out_semaphore);
  *out_semaphore = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  VkSemaphoreTypeCreateInfo timeline_create_info;
  timeline_create_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
  timeline_create_info.pNext = NULL;
  timeline_create_info.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
  timeline_create_info.initialValue = initial_value;

  VkSemaphoreCreateInfo create_info;
  create_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  create_info.pNext = &timeline_create_info;
  create_info.flags = 0;
  VkSemaphore handle = VK_NULL_HANDLE;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, VK_RESULT_TO_STATUS(logical_device->syms()->vkCreateSemaphore(
                                  *logical_device, &create_info,
                                  logical_device->allocator(), &handle),
                              "vkCreateSemaphore"));

  iree_hal_vulkan_native_semaphore_t* semaphore = NULL;
  iree_host_size_t frontier_offset = 0, total_size = 0;
  iree_status_t status = iree_async_semaphore_layout(
      sizeof(*semaphore), 0, &frontier_offset, &total_size);
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(logical_device->host_allocator(), total_size,
                                   (void**)&semaphore);
  }
  if (iree_status_is_ok(status)) {
    iree_async_semaphore_initialize(
        (const iree_async_semaphore_vtable_t*)&iree_hal_vulkan_native_semaphore_vtable,
        initial_value, frontier_offset, 0, &semaphore->async);
    semaphore->logical_device = logical_device;
    semaphore->handle = handle;
    *out_semaphore = iree_hal_semaphore_cast(&semaphore->async);
  } else {
    logical_device->syms()->vkDestroySemaphore(*logical_device, handle,
                                               logical_device->allocator());
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_vulkan_native_semaphore_destroy(
    iree_async_semaphore_t* base_semaphore) {
  iree_hal_vulkan_native_semaphore_t* semaphore =
      iree_hal_vulkan_native_semaphore_cast(
          iree_hal_semaphore_cast(base_semaphore));
  iree_allocator_t host_allocator = semaphore->logical_device->host_allocator();
  IREE_TRACE_ZONE_BEGIN(z0);

  semaphore->logical_device->syms()->vkDestroySemaphore(
      *semaphore->logical_device, semaphore->handle,
      semaphore->logical_device->allocator());

  iree_async_semaphore_deinitialize(&semaphore->async);
  iree_allocator_free(host_allocator, semaphore);

  IREE_TRACE_ZONE_END(z0);
}

bool iree_hal_vulkan_native_semaphore_isa(iree_hal_semaphore_t* semaphore) {
  return iree_hal_resource_is((const iree_hal_resource_t*)semaphore,
                              &iree_hal_vulkan_native_semaphore_vtable);
}

VkSemaphore iree_hal_vulkan_native_semaphore_handle(
    iree_hal_semaphore_t* base_semaphore) {
  iree_hal_vulkan_native_semaphore_t* semaphore =
      iree_hal_vulkan_native_semaphore_cast(base_semaphore);
  return semaphore->handle;
}

static uint64_t iree_hal_vulkan_native_semaphore_query(
    iree_async_semaphore_t* base_semaphore) {
  iree_hal_vulkan_native_semaphore_t* semaphore =
      iree_hal_vulkan_native_semaphore_cast(
          iree_hal_semaphore_cast(base_semaphore));
  iree_async_semaphore_t* async_sem = base_semaphore;

  // Check for failure first so we can encode the actual failure status.
  iree_status_t failure_status = (iree_status_t)iree_atomic_load(
      &async_sem->failure_status, iree_memory_order_acquire);
  if (!iree_status_is_ok(failure_status)) {
    return iree_hal_status_as_semaphore_failure(failure_status);
  }

  // Query from Vulkan source-of-truth.
  uint64_t value = 0;
  VkResult result =
      semaphore->logical_device->syms()->vkGetSemaphoreCounterValue(
          *semaphore->logical_device, semaphore->handle, &value);
  if (result != VK_SUCCESS) {
    return iree_hal_status_as_semaphore_failure(
        iree_status_from_code(IREE_STATUS_INTERNAL));
  }

  // If the semaphore has failed return the encoded failure status.
  if (value >= IREE_HAL_SEMAPHORE_FAILURE_VALUE) {
    // Re-check the failure status as it may have been set between our first
    // check and the Vulkan query.
    failure_status = (iree_status_t)iree_atomic_load(&async_sem->failure_status,
                                                     iree_memory_order_acquire);
    if (!iree_status_is_ok(failure_status)) {
      return iree_hal_status_as_semaphore_failure(failure_status);
    }
    return IREE_HAL_SEMAPHORE_FAILURE_VALUE;
  }

  // Sync the timeline to the Vulkan-queried value. We use an atomic store
  // rather than the normal signal path because queries can return the same
  // value on consecutive calls, which is not a monotonicity violation. This
  // flushes any satisfied timepoints, keeping latency low on the device-side
  // signal path (Vulkan does not notify us of device-side signals).
  iree_atomic_store(&async_sem->timeline_value, (int64_t)value,
                    iree_memory_order_release);
  iree_async_semaphore_dispatch_timepoints(base_semaphore, value);

  return value;
}

static iree_status_t iree_hal_vulkan_native_semaphore_signal(
    iree_async_semaphore_t* base_semaphore, uint64_t new_value,
    const iree_async_frontier_t* frontier) {
  iree_hal_vulkan_native_semaphore_t* semaphore =
      iree_hal_vulkan_native_semaphore_cast(
          iree_hal_semaphore_cast(base_semaphore));
  iree_async_semaphore_t* async_sem = base_semaphore;

  VkSemaphoreSignalInfo signal_info;
  signal_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
  signal_info.pNext = NULL;
  signal_info.semaphore = semaphore->handle;
  signal_info.value = new_value;
  iree_status_t status =
      VK_RESULT_TO_STATUS(semaphore->logical_device->syms()->vkSignalSemaphore(
                              *semaphore->logical_device, &signal_info),
                          "vkSignalSemaphore");

  if (iree_status_is_ok(status)) {
    // Advance the software timeline (CAS) and merge frontier. Each timeline
    // value must be signaled exactly once — CAS failure here indicates a
    // structural error (duplicate signal or non-monotonic scheduling).
    iree_status_t advance_status = iree_async_semaphore_advance_timeline(
        base_semaphore, new_value, frontier);
    if (IREE_UNLIKELY(!iree_status_is_ok(advance_status))) {
      iree_async_semaphore_fail(base_semaphore, advance_status);
      // The VkSemaphore was already signaled — return OK for the Vulkan side
      // but the async semaphore is now failed so waiters get the diagnostic.
      return iree_ok_status();
    }
    iree_async_semaphore_dispatch_timepoints(base_semaphore, new_value);
  } else {
    // Vulkan signal failed — fail the async semaphore to notify waiters.
    intptr_t expected = 0;
    iree_atomic_compare_exchange_strong(
        &async_sem->failure_status, &expected,
        (intptr_t)iree_status_from_code(iree_status_code(status)),
        iree_memory_order_release, iree_memory_order_relaxed);
    iree_async_semaphore_dispatch_timepoints_failed(
        base_semaphore, iree_status_from_code(iree_status_code(status)));
  }

  return status;
}

static void iree_hal_vulkan_native_semaphore_fail(
    iree_async_semaphore_t* base_semaphore, iree_status_t status) {
  iree_hal_vulkan_native_semaphore_t* semaphore =
      iree_hal_vulkan_native_semaphore_cast(
          iree_hal_semaphore_cast(base_semaphore));
  iree_async_semaphore_t* async_sem = base_semaphore;

  // First failure wins via CAS. Clone for storage, pass original to dispatch.
  iree_status_t stored = iree_status_clone(status);
  intptr_t expected = 0;
  if (!iree_atomic_compare_exchange_strong(
          &async_sem->failure_status, &expected, (intptr_t)stored,
          iree_memory_order_release, iree_memory_order_acquire)) {
    iree_status_free(stored);
    iree_status_free(status);
    return;
  }

  // Signal the VkSemaphore with the failure sentinel so Vulkan-side waiters
  // wake. We don't care about the result — we're already failing.
  VkSemaphoreSignalInfo signal_info;
  signal_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
  signal_info.pNext = NULL;
  signal_info.semaphore = semaphore->handle;
  signal_info.value = IREE_HAL_SEMAPHORE_FAILURE_VALUE;
  semaphore->logical_device->syms()->vkSignalSemaphore(
      *semaphore->logical_device, &signal_info);

  // Dispatch all pending timepoints with the failure status.
  // Takes ownership of |status| (clones per-timepoint, frees original).
  iree_async_semaphore_dispatch_timepoints_failed(base_semaphore, status);
}

iree_status_t iree_hal_vulkan_native_semaphore_multi_wait(
    iree::hal::vulkan::VkDeviceHandle* logical_device,
    const iree_hal_semaphore_list_t* semaphore_list, iree_timeout_t timeout,
    iree_hal_wait_flags_t flags, VkSemaphoreWaitFlags wait_flags) {
  if (semaphore_list->count == 0) return iree_ok_status();

  iree_time_t deadline_ns = iree_timeout_as_deadline_ns(timeout);

  uint64_t timeout_ns;
  if (deadline_ns == IREE_TIME_INFINITE_FUTURE) {
    timeout_ns = UINT64_MAX;
  } else if (deadline_ns == IREE_TIME_INFINITE_PAST) {
    timeout_ns = 0;
  } else {
    iree_time_t now_ns = iree_time_now();
    timeout_ns = deadline_ns < now_ns ? 0 : (uint64_t)(deadline_ns - now_ns);
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  VkSemaphore* semaphore_handles =
      (VkSemaphore*)iree_alloca(semaphore_list->count * sizeof(VkSemaphore));
  for (iree_host_size_t i = 0; i < semaphore_list->count; ++i) {
    semaphore_handles[i] =
        iree_hal_vulkan_native_semaphore_handle(semaphore_list->semaphores[i]);
  }

  VkSemaphoreWaitInfo wait_info;
  wait_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
  wait_info.pNext = nullptr;
  wait_info.flags = wait_flags;
  wait_info.semaphoreCount = semaphore_list->count;
  wait_info.pSemaphores = semaphore_handles;
  wait_info.pValues = semaphore_list->payload_values;
  static_assert(
      sizeof(wait_info.pValues[0]) == sizeof(semaphore_list->payload_values[0]),
      "payload value type must match vulkan expected size");

  // NOTE: this may fail with a timeout (VK_TIMEOUT) or in the case of a
  // device loss event may return either VK_SUCCESS *or* VK_ERROR_DEVICE_LOST.
  // We may want to explicitly query for device loss after a successful wait
  // to ensure we consistently return errors.
  VkResult result = logical_device->syms()->vkWaitSemaphores(
      *logical_device, &wait_info, timeout_ns);

  IREE_TRACE_ZONE_END(z0);

  // Poll semaphores and sync async semaphore state. We aren't notified of
  // device-side signals by Vulkan, so we query each semaphore and sync the
  // timeline to dispatch any satisfied timepoints.
  //
  // TODO(benvanik): on success optimize this to sync to the waited values
  // directly instead of a full poll; it'll avoid additional API queries.
  bool any_failed = false;
  for (iree_host_size_t i = 0; i < semaphore_list->count; ++i) {
    iree_hal_vulkan_native_semaphore_t* semaphore =
        iree_hal_vulkan_native_semaphore_cast(semaphore_list->semaphores[i]);
    iree_async_semaphore_t* async_sem =
        (iree_async_semaphore_t*)semaphore_list->semaphores[i];

    uint64_t value = 0;
    IREE_RETURN_IF_ERROR(VK_RESULT_TO_STATUS(
        semaphore->logical_device->syms()->vkGetSemaphoreCounterValue(
            *semaphore->logical_device, semaphore->handle, &value),
        "vkGetSemaphoreCounterValue"));

    if (value >= IREE_HAL_SEMAPHORE_FAILURE_VALUE) {
      any_failed = true;
      // Sync failure status to the async semaphore.
      intptr_t expected = 0;
      iree_atomic_compare_exchange_strong(
          &async_sem->failure_status, &expected,
          (intptr_t)iree_status_from_code(IREE_STATUS_ABORTED),
          iree_memory_order_release, iree_memory_order_relaxed);
      iree_async_semaphore_dispatch_timepoints_failed(
          async_sem, iree_status_from_code(IREE_STATUS_ABORTED));
    } else {
      // Sync the timeline and dispatch satisfied timepoints.
      iree_atomic_store(&async_sem->timeline_value, (int64_t)value,
                        iree_memory_order_release);
      iree_async_semaphore_dispatch_timepoints(async_sem, value);
    }
  }

  if (any_failed) {
    return iree_make_status(IREE_STATUS_ABORTED,
                            "one or more semaphores have failed");
  } else if (result == VK_SUCCESS) {
    return iree_ok_status();
  } else if (result == VK_ERROR_DEVICE_LOST) {
    // Nothing we do now matters.
    return VK_RESULT_TO_STATUS(result, "vkWaitSemaphores");
  } else if (result == VK_TIMEOUT) {
    return iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED);
  }
  return VK_RESULT_TO_STATUS(result, "vkWaitSemaphores");
}

static iree_status_t iree_hal_vulkan_native_semaphore_wait(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    iree_timeout_t timeout, iree_hal_wait_flags_t flags) {
  iree_hal_vulkan_native_semaphore_t* semaphore =
      iree_hal_vulkan_native_semaphore_cast(base_semaphore);
  iree_hal_semaphore_list_t semaphore_list = {
      /*.count=*/1,
      /*.semaphores=*/&base_semaphore,
      /*.payload_values=*/&value,
  };
  return iree_hal_vulkan_native_semaphore_multi_wait(
      semaphore->logical_device, &semaphore_list, timeout, flags, 0);
}

IREE_API_EXPORT iree_status_t iree_hal_vulkan_semaphore_handle(
    iree_hal_semaphore_t* base_semaphore, VkSemaphore* out_handle) {
  IREE_ASSERT_ARGUMENT(base_semaphore);
  IREE_ASSERT_ARGUMENT(out_handle);
  iree_hal_vulkan_native_semaphore_t* semaphore =
      iree_hal_vulkan_native_semaphore_cast(base_semaphore);
  *out_handle = semaphore->handle;
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_semaphore_import_timepoint(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    iree_hal_queue_affinity_t queue_affinity,
    iree_hal_external_timepoint_t external_timepoint) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "timepoint import is not yet implemented");
}

static iree_status_t iree_hal_vulkan_semaphore_export_timepoint(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    iree_hal_queue_affinity_t queue_affinity,
    iree_hal_external_timepoint_type_t requested_type,
    iree_hal_external_timepoint_flags_t requested_flags,
    iree_hal_external_timepoint_t* IREE_RESTRICT out_external_timepoint) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "timepoint export is not yet implemented");
}

static uint8_t iree_hal_vulkan_native_semaphore_query_frontier(
    iree_async_semaphore_t* semaphore, iree_async_frontier_t* out_frontier,
    uint8_t capacity) {
  (void)semaphore;
  (void)out_frontier;
  (void)capacity;
  return 0;
}

static iree_status_t iree_hal_vulkan_native_semaphore_acquire_timepoint(
    iree_async_semaphore_t* semaphore, uint64_t minimum_value,
    iree_async_semaphore_timepoint_t* timepoint) {
  (void)semaphore;
  (void)minimum_value;
  (void)timepoint;
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "async timepoints not supported");
}

static void iree_hal_vulkan_native_semaphore_cancel_timepoint(
    iree_async_semaphore_t* semaphore,
    iree_async_semaphore_timepoint_t* timepoint) {
  (void)semaphore;
  (void)timepoint;
}

static iree_status_t iree_hal_vulkan_native_semaphore_export_primitive(
    iree_async_semaphore_t* semaphore, uint64_t minimum_value,
    iree_async_primitive_t* out_primitive) {
  (void)semaphore;
  (void)minimum_value;
  (void)out_primitive;
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "primitive export not supported");
}

namespace {
const iree_hal_semaphore_vtable_t iree_hal_vulkan_native_semaphore_vtable = {
    /*.async=*/
    {
        /*.destroy=*/iree_hal_vulkan_native_semaphore_destroy,
        /*.query=*/iree_hal_vulkan_native_semaphore_query,
        /*.signal=*/iree_hal_vulkan_native_semaphore_signal,
        /*.query_frontier=*/iree_hal_vulkan_native_semaphore_query_frontier,
        /*.fail=*/iree_hal_vulkan_native_semaphore_fail,
        /*.acquire_timepoint=*/
        iree_hal_vulkan_native_semaphore_acquire_timepoint,
        /*.cancel_timepoint=*/
        iree_hal_vulkan_native_semaphore_cancel_timepoint,
        /*.export_primitive=*/
        iree_hal_vulkan_native_semaphore_export_primitive,
    },
    /*.wait=*/iree_hal_vulkan_native_semaphore_wait,
    /*.import_timepoint=*/iree_hal_vulkan_semaphore_import_timepoint,
    /*.export_timepoint=*/iree_hal_vulkan_semaphore_export_timepoint,
};
}  // namespace
