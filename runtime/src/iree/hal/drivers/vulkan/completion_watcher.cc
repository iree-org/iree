// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/completion_watcher.h"

#include <cstddef>
#include <cstring>

#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"
#include "iree/base/threading/mutex.h"
#include "iree/base/threading/thread.h"
#include "iree/hal/drivers/vulkan/dynamic_symbols.h"
#include "iree/hal/drivers/vulkan/native_semaphore.h"
#include "iree/hal/drivers/vulkan/status_util.h"

using namespace iree::hal::vulkan;

//===----------------------------------------------------------------------===//
// Constants
//===----------------------------------------------------------------------===//

// Initial capacity for the pending and active entry arrays.
#define IREE_HAL_VULKAN_COMPLETION_WATCHER_INITIAL_CAPACITY 64

// Timeout for vkWaitSemaphores in nanoseconds. Periodic wakeups serve as a
// liveness check (some drivers don't return VK_ERROR_DEVICE_LOST from wait).
#define IREE_HAL_VULKAN_COMPLETION_WATCHER_WAIT_TIMEOUT_NS \
  (5ull * 1000000000ull)  // 5 seconds

//===----------------------------------------------------------------------===//
// Completion entry
//===----------------------------------------------------------------------===//

typedef struct iree_hal_vulkan_completion_entry_t {
  // HAL semaphore being watched. Retained by the watcher.
  iree_hal_semaphore_t* semaphore;
  // VkSemaphore handle (cached from semaphore for direct use in wait arrays).
  VkSemaphore vk_semaphore;
  // Target timeline value that indicates this entry's work is complete.
  uint64_t target_value;
  // Resources retained until this entry completes. Freed on completion.
  // May be NULL if no resources need to be retained (earlier entries in a
  // multi-semaphore submission where the resource set is on the last entry).
  iree_hal_resource_set_t* resource_set;
} iree_hal_vulkan_completion_entry_t;

//===----------------------------------------------------------------------===//
// Watcher state
//===----------------------------------------------------------------------===//

struct iree_hal_vulkan_completion_watcher_t {
  iree_thread_t* thread;
  iree_allocator_t host_allocator;

  // Logical device handle for Vulkan API calls. Borrowed — the device outlives
  // the watcher.
  VkDeviceHandle* logical_device;

  // Poke semaphore: a dedicated VkTimelineSemaphore used solely to wake the
  // watcher thread when new entries are registered or shutdown is requested.
  VkSemaphore poke_semaphore;
  // Next value to signal on the poke semaphore. Incremented atomically by
  // registration calls.
  iree_atomic_int64_t poke_next_value;
  // Value the watcher last observed on the poke semaphore. Used to compute
  // the wait target (poke_last_value + 1). Watcher-local, no sync needed.
  uint64_t poke_last_value;

  // Protects pending_entries, pending_count, and do_exit. Held briefly by
  // submitters to append entries and by the watcher to drain them.
  iree_slim_mutex_t mutex;
  iree_hal_vulkan_completion_entry_t* pending_entries;
  iree_host_size_t pending_count;
  iree_host_size_t pending_capacity;
  bool do_exit;

  // Active entries being watched. Owned exclusively by the watcher thread —
  // no lock needed.
  iree_hal_vulkan_completion_entry_t* active_entries;
  iree_host_size_t active_count;
  iree_host_size_t active_capacity;

  // Scratch arrays rebuilt each iteration for vkWaitSemaphores. Sized to
  // active_capacity + 1 (the extra slot is for the poke semaphore).
  VkSemaphore* wait_semaphores;
  uint64_t* wait_values;
};

//===----------------------------------------------------------------------===//
// Internal helpers
//===----------------------------------------------------------------------===//

// Ensures the active arrays have room for at least |required_capacity| entries
// plus 1 (for the poke semaphore slot in the wait arrays). Grows by doubling.
// All three arrays (active_entries, wait_semaphores, wait_values) are allocated
// together in a single allocation to avoid partial-update bugs.
static iree_status_t iree_hal_vulkan_completion_watcher_grow_active(
    iree_hal_vulkan_completion_watcher_t* watcher,
    iree_host_size_t required_capacity) {
  if (required_capacity <= watcher->active_capacity) return iree_ok_status();

  iree_host_size_t new_capacity =
      iree_max(watcher->active_capacity * 2,
               iree_max(required_capacity,
                        IREE_HAL_VULKAN_COMPLETION_WATCHER_INITIAL_CAPACITY));

  // Single allocation for all three arrays. Wait arrays have +1 for poke.
  iree_host_size_t wait_count = new_capacity + 1;
  iree_host_size_t entries_size =
      new_capacity * sizeof(iree_hal_vulkan_completion_entry_t);
  iree_host_size_t semaphores_size = wait_count * sizeof(VkSemaphore);
  iree_host_size_t values_size = wait_count * sizeof(uint64_t);
  iree_host_size_t total_size = entries_size + semaphores_size + values_size;
  uint8_t* buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(watcher->host_allocator,
                                             total_size, (void**)&buffer));

  iree_hal_vulkan_completion_entry_t* new_active =
      (iree_hal_vulkan_completion_entry_t*)buffer;
  VkSemaphore* new_wait_semaphores = (VkSemaphore*)(buffer + entries_size);
  uint64_t* new_wait_values =
      (uint64_t*)(buffer + entries_size + semaphores_size);

  if (watcher->active_count > 0) {
    memcpy(new_active, watcher->active_entries,
           watcher->active_count * sizeof(iree_hal_vulkan_completion_entry_t));
  }

  // Free old combined allocation (active_entries is the base pointer).
  iree_allocator_free(watcher->host_allocator, watcher->active_entries);
  watcher->active_entries = new_active;
  watcher->wait_semaphores = new_wait_semaphores;
  watcher->wait_values = new_wait_values;
  watcher->active_capacity = new_capacity;
  return iree_ok_status();
}

// Drains pending entries into the active set. Caller must NOT hold the mutex.
static iree_status_t iree_hal_vulkan_completion_watcher_drain_pending(
    iree_hal_vulkan_completion_watcher_t* watcher) {
  iree_slim_mutex_lock(&watcher->mutex);
  iree_host_size_t pending_count = watcher->pending_count;
  if (pending_count == 0) {
    iree_slim_mutex_unlock(&watcher->mutex);
    return iree_ok_status();
  }

  // Ensure active has room.
  iree_host_size_t required = watcher->active_count + pending_count;
  iree_status_t status =
      iree_hal_vulkan_completion_watcher_grow_active(watcher, required);
  if (iree_status_is_ok(status)) {
    memcpy(&watcher->active_entries[watcher->active_count],
           watcher->pending_entries,
           pending_count * sizeof(iree_hal_vulkan_completion_entry_t));
    watcher->active_count += pending_count;
    watcher->pending_count = 0;
  }
  iree_slim_mutex_unlock(&watcher->mutex);
  return status;
}

// Releases a single completion entry: frees the resource set and releases the
// semaphore.
static void iree_hal_vulkan_completion_entry_release(
    iree_hal_vulkan_completion_entry_t* entry) {
  iree_hal_resource_set_free(entry->resource_set);
  iree_hal_semaphore_release(entry->semaphore);
  memset(entry, 0, sizeof(*entry));
}

// Swap-removes the entry at |index| from the active array.
static void iree_hal_vulkan_completion_watcher_swap_remove(
    iree_hal_vulkan_completion_watcher_t* watcher, iree_host_size_t index) {
  IREE_ASSERT(index < watcher->active_count);
  watcher->active_count--;
  if (index < watcher->active_count) {
    watcher->active_entries[index] =
        watcher->active_entries[watcher->active_count];
  }
}

// Fails all active and pending entries with the given status. Used during
// device loss or forced shutdown.
static void iree_hal_vulkan_completion_watcher_fail_all(
    iree_hal_vulkan_completion_watcher_t* watcher, iree_status_code_t code) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Fail all active entries.
  for (iree_host_size_t i = 0; i < watcher->active_count; ++i) {
    iree_hal_semaphore_fail(watcher->active_entries[i].semaphore,
                            iree_status_from_code(code));
    iree_hal_vulkan_completion_entry_release(&watcher->active_entries[i]);
  }
  watcher->active_count = 0;

  // Drain and fail pending entries.
  iree_slim_mutex_lock(&watcher->mutex);
  for (iree_host_size_t i = 0; i < watcher->pending_count; ++i) {
    iree_hal_semaphore_fail(watcher->pending_entries[i].semaphore,
                            iree_status_from_code(code));
    iree_hal_vulkan_completion_entry_release(&watcher->pending_entries[i]);
  }
  watcher->pending_count = 0;
  iree_slim_mutex_unlock(&watcher->mutex);

  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// Watcher thread
//===----------------------------------------------------------------------===//

static int iree_hal_vulkan_completion_watcher_thread_main(void* param) {
  iree_hal_vulkan_completion_watcher_t* watcher =
      (iree_hal_vulkan_completion_watcher_t*)param;
  auto* syms = watcher->logical_device->syms().get();
  VkDevice vk_device = *watcher->logical_device;

  while (true) {
    IREE_TRACE_ZONE_BEGIN_NAMED(z_iteration,
                                "iree_hal_vulkan_completion_watcher_iteration");

    // Drain pending entries into the active set.
    iree_status_t status =
        iree_hal_vulkan_completion_watcher_drain_pending(watcher);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      iree_hal_vulkan_completion_watcher_fail_all(watcher,
                                                  IREE_STATUS_INTERNAL);
      IREE_TRACE_ZONE_END(z_iteration);
      break;
    }

    // Check exit condition.
    iree_slim_mutex_lock(&watcher->mutex);
    bool do_exit = watcher->do_exit;
    iree_slim_mutex_unlock(&watcher->mutex);
    if (do_exit && watcher->active_count == 0) {
      IREE_TRACE_ZONE_END(z_iteration);
      break;
    }

    // Build wait set.
    iree_host_size_t wait_count = 0;
    VkSemaphoreWaitFlags wait_flags = 0;

    if (watcher->active_count == 0) {
      // No active entries — wait only on the poke semaphore.
      if (do_exit) {
        IREE_TRACE_ZONE_END(z_iteration);
        break;
      }
      watcher->wait_semaphores[0] = watcher->poke_semaphore;
      watcher->wait_values[0] = watcher->poke_last_value + 1;
      wait_count = 1;
      // No ANY flag needed for single semaphore.
    } else {
      // Active entries + poke semaphore.
      for (iree_host_size_t i = 0; i < watcher->active_count; ++i) {
        watcher->wait_semaphores[i] = watcher->active_entries[i].vk_semaphore;
        watcher->wait_values[i] = watcher->active_entries[i].target_value;
      }
      watcher->wait_semaphores[watcher->active_count] = watcher->poke_semaphore;
      watcher->wait_values[watcher->active_count] =
          watcher->poke_last_value + 1;
      wait_count = watcher->active_count + 1;
      wait_flags = VK_SEMAPHORE_WAIT_ANY_BIT;
    }

    VkSemaphoreWaitInfo wait_info;
    wait_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
    wait_info.pNext = NULL;
    wait_info.flags = wait_flags;
    wait_info.semaphoreCount = (uint32_t)wait_count;
    wait_info.pSemaphores = watcher->wait_semaphores;
    wait_info.pValues = watcher->wait_values;

    {
      IREE_TRACE_ZONE_BEGIN_NAMED(z_wait,
                                  "iree_hal_vulkan_completion_watcher_wait");
      VkResult result = syms->vkWaitSemaphores(
          vk_device, &wait_info,
          IREE_HAL_VULKAN_COMPLETION_WATCHER_WAIT_TIMEOUT_NS);
      IREE_TRACE_ZONE_END(z_wait);

      if (result == VK_ERROR_DEVICE_LOST) {
        iree_hal_vulkan_completion_watcher_fail_all(watcher,
                                                    IREE_STATUS_UNAVAILABLE);
        IREE_TRACE_ZONE_END(z_iteration);
        break;
      }
      if (result == VK_TIMEOUT) {
        // Periodic liveness wake. If we're exiting and have active entries,
        // check if they've timed out (GPU may be hung).
        if (do_exit && watcher->active_count > 0) {
          // On shutdown, don't wait forever — fail remaining entries.
          iree_hal_vulkan_completion_watcher_fail_all(watcher,
                                                      IREE_STATUS_ABORTED);
          IREE_TRACE_ZONE_END(z_iteration);
          break;
        }
        IREE_TRACE_ZONE_END(z_iteration);
        continue;
      }
      // VK_SUCCESS — at least one semaphore made progress.
    }

    // Catch up poke value so next iteration waits at the right target.
    watcher->poke_last_value = (uint64_t)iree_atomic_load(
        &watcher->poke_next_value, iree_memory_order_acquire);

    // Scan active entries for completion.
    {
      IREE_TRACE_ZONE_BEGIN_NAMED(z_scan,
                                  "iree_hal_vulkan_completion_watcher_scan");
      iree_host_size_t i = 0;
      while (i < watcher->active_count) {
        uint64_t value = 0;
        VkResult query_result = syms->vkGetSemaphoreCounterValue(
            vk_device, watcher->active_entries[i].vk_semaphore, &value);
        if (query_result != VK_SUCCESS) {
          // Query failure — treat as device loss.
          iree_hal_vulkan_completion_watcher_fail_all(watcher,
                                                      IREE_STATUS_UNAVAILABLE);
          IREE_TRACE_ZONE_END(z_scan);
          IREE_TRACE_ZONE_END(z_iteration);
          return 0;
        }

        if (value >= IREE_HAL_SEMAPHORE_FAILURE_VALUE) {
          // Semaphore has been failed by another thread. The failing thread
          // already dispatched timepoints, so we just free resources.
          iree_hal_vulkan_completion_entry_release(&watcher->active_entries[i]);
          iree_hal_vulkan_completion_watcher_swap_remove(watcher, i);
          continue;
        }

        if (value >= watcher->active_entries[i].target_value) {
          // Completed. Sync host-side async timeline and dispatch timepoints.
          iree_async_semaphore_t* async_sem =
              (iree_async_semaphore_t*)watcher->active_entries[i].semaphore;
          iree_atomic_store(&async_sem->timeline_value, (int64_t)value,
                            iree_memory_order_release);
          iree_async_semaphore_dispatch_timepoints(async_sem, value);
          iree_hal_vulkan_completion_entry_release(&watcher->active_entries[i]);
          iree_hal_vulkan_completion_watcher_swap_remove(watcher, i);
          continue;
        }

        ++i;
      }
      IREE_TRACE_ZONE_END(z_scan);
    }

    IREE_TRACE_ZONE_END(z_iteration);
  }

  return 0;
}

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_vulkan_completion_watcher_create(
    VkDeviceHandle* logical_device, iree_allocator_t host_allocator,
    iree_hal_vulkan_completion_watcher_t** out_watcher) {
  IREE_ASSERT_ARGUMENT(logical_device);
  IREE_ASSERT_ARGUMENT(out_watcher);
  *out_watcher = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_vulkan_completion_watcher_t* watcher = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*watcher),
                                (void**)&watcher));
  memset(watcher, 0, sizeof(*watcher));
  watcher->host_allocator = host_allocator;
  watcher->logical_device = logical_device;
  iree_atomic_store(&watcher->poke_next_value, 0, iree_memory_order_release);
  watcher->poke_last_value = 0;
  iree_slim_mutex_initialize(&watcher->mutex);
  watcher->do_exit = false;

  // Create the poke semaphore (timeline, initial value 0).
  VkSemaphoreTypeCreateInfo timeline_create_info;
  timeline_create_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
  timeline_create_info.pNext = NULL;
  timeline_create_info.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
  timeline_create_info.initialValue = 0;
  VkSemaphoreCreateInfo create_info;
  create_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  create_info.pNext = &timeline_create_info;
  create_info.flags = 0;
  iree_status_t status = VK_RESULT_TO_STATUS(
      logical_device->syms()->vkCreateSemaphore(*logical_device, &create_info,
                                                logical_device->allocator(),
                                                &watcher->poke_semaphore),
      "vkCreateSemaphore(poke)");

  // Allocate initial pending array.
  if (iree_status_is_ok(status)) {
    watcher->pending_capacity =
        IREE_HAL_VULKAN_COMPLETION_WATCHER_INITIAL_CAPACITY;
    status = iree_allocator_malloc(
        host_allocator,
        watcher->pending_capacity * sizeof(iree_hal_vulkan_completion_entry_t),
        (void**)&watcher->pending_entries);
  }

  // Allocate initial active + wait arrays.
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_completion_watcher_grow_active(
        watcher, IREE_HAL_VULKAN_COMPLETION_WATCHER_INITIAL_CAPACITY);
  }

  // Spawn the watcher thread.
  if (iree_status_is_ok(status)) {
    iree_thread_create_params_t params;
    memset(&params, 0, sizeof(params));
    params.name = iree_make_cstring_view("iree-hal-vulkan-watcher");
    params.priority_class = IREE_THREAD_PRIORITY_CLASS_NORMAL;
    status = iree_thread_create(
        (iree_thread_entry_t)iree_hal_vulkan_completion_watcher_thread_main,
        watcher, params, host_allocator, &watcher->thread);
  }

  if (iree_status_is_ok(status)) {
    *out_watcher = watcher;
  } else {
    // Cleanup on failure.
    if (watcher->poke_semaphore != VK_NULL_HANDLE) {
      logical_device->syms()->vkDestroySemaphore(*logical_device,
                                                 watcher->poke_semaphore,
                                                 logical_device->allocator());
    }
    iree_allocator_free(host_allocator, watcher->pending_entries);
    // active_entries is the base of the combined allocation that also contains
    // wait_semaphores and wait_values.
    iree_allocator_free(host_allocator, watcher->active_entries);
    iree_slim_mutex_deinitialize(&watcher->mutex);
    iree_allocator_free(host_allocator, watcher);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_vulkan_completion_watcher_destroy(
    iree_hal_vulkan_completion_watcher_t* watcher) {
  if (!watcher) return;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Signal the watcher to exit.
  iree_slim_mutex_lock(&watcher->mutex);
  watcher->do_exit = true;
  iree_slim_mutex_unlock(&watcher->mutex);

  // Poke the watcher to wake it from vkWaitSemaphores.
  int64_t poke_value = iree_atomic_fetch_add(&watcher->poke_next_value, 1,
                                             iree_memory_order_acq_rel) +
                       1;
  VkSemaphoreSignalInfo signal_info;
  signal_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
  signal_info.pNext = NULL;
  signal_info.semaphore = watcher->poke_semaphore;
  signal_info.value = (uint64_t)poke_value;
  watcher->logical_device->syms()->vkSignalSemaphore(*watcher->logical_device,
                                                     &signal_info);

  // Release joins the thread (ref count drops to 0 → internal join).
  iree_thread_release(watcher->thread);

  // Destroy the poke semaphore.
  watcher->logical_device->syms()->vkDestroySemaphore(
      *watcher->logical_device, watcher->poke_semaphore,
      watcher->logical_device->allocator());

  // Free arrays. active_entries is the base of the combined allocation that
  // also contains wait_semaphores and wait_values.
  iree_allocator_free(watcher->host_allocator, watcher->pending_entries);
  iree_allocator_free(watcher->host_allocator, watcher->active_entries);
  iree_slim_mutex_deinitialize(&watcher->mutex);

  iree_allocator_t host_allocator = watcher->host_allocator;
  iree_allocator_free(host_allocator, watcher);

  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_hal_vulkan_completion_watcher_register(
    iree_hal_vulkan_completion_watcher_t* watcher,
    const iree_hal_semaphore_list_t* signal_semaphores,
    iree_hal_resource_set_t* resource_set) {
  IREE_ASSERT_ARGUMENT(watcher);
  IREE_ASSERT_ARGUMENT(signal_semaphores);
  if (signal_semaphores->count == 0) {
    // Nothing to watch — free the resource set immediately.
    iree_hal_resource_set_free(resource_set);
    return iree_ok_status();
  }
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_slim_mutex_lock(&watcher->mutex);

  // Check for failure/exit before accepting new work.
  if (watcher->do_exit) {
    iree_slim_mutex_unlock(&watcher->mutex);
    iree_hal_resource_set_free(resource_set);
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_ABORTED,
                            "completion watcher is shutting down");
  }

  // Grow pending array if needed.
  iree_host_size_t required = watcher->pending_count + signal_semaphores->count;
  if (required > watcher->pending_capacity) {
    iree_host_size_t new_capacity =
        iree_max(watcher->pending_capacity * 2, required);
    iree_hal_vulkan_completion_entry_t* new_pending = NULL;
    iree_status_t status = iree_allocator_malloc(
        watcher->host_allocator,
        new_capacity * sizeof(iree_hal_vulkan_completion_entry_t),
        (void**)&new_pending);
    if (!iree_status_is_ok(status)) {
      iree_slim_mutex_unlock(&watcher->mutex);
      iree_hal_resource_set_free(resource_set);
      IREE_TRACE_ZONE_END(z0);
      return status;
    }
    if (watcher->pending_count > 0) {
      memcpy(
          new_pending, watcher->pending_entries,
          watcher->pending_count * sizeof(iree_hal_vulkan_completion_entry_t));
    }
    iree_allocator_free(watcher->host_allocator, watcher->pending_entries);
    watcher->pending_entries = new_pending;
    watcher->pending_capacity = new_capacity;
  }

  // Append entries. The resource set attaches to the last entry.
  for (iree_host_size_t i = 0; i < signal_semaphores->count; ++i) {
    iree_hal_vulkan_completion_entry_t* entry =
        &watcher->pending_entries[watcher->pending_count++];
    entry->semaphore = signal_semaphores->semaphores[i];
    iree_hal_semaphore_retain(entry->semaphore);
    entry->vk_semaphore =
        iree_hal_vulkan_native_semaphore_handle(entry->semaphore);
    entry->target_value = signal_semaphores->payload_values[i];
    entry->resource_set =
        (i == signal_semaphores->count - 1) ? resource_set : NULL;
  }

  iree_slim_mutex_unlock(&watcher->mutex);

  // Poke the watcher to wake it from vkWaitSemaphores. The atomic increment
  // and vkSignalSemaphore are both thread-safe without the mutex.
  int64_t poke_value = iree_atomic_fetch_add(&watcher->poke_next_value, 1,
                                             iree_memory_order_acq_rel) +
                       1;
  VkSemaphoreSignalInfo signal_info;
  signal_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
  signal_info.pNext = NULL;
  signal_info.semaphore = watcher->poke_semaphore;
  signal_info.value = (uint64_t)poke_value;
  iree_status_t status =
      VK_RESULT_TO_STATUS(watcher->logical_device->syms()->vkSignalSemaphore(
                              *watcher->logical_device, &signal_info),
                          "vkSignalSemaphore(poke)");

  IREE_TRACE_ZONE_END(z0);
  return status;
}
