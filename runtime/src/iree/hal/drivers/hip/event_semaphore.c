// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/hip/event_semaphore.h"

#include "iree/base/internal/math.h"
#include "iree/base/internal/synchronization.h"
#include "iree/base/internal/wait_handle.h"
#include "iree/base/status.h"
#include "iree/hal/drivers/hip/dynamic_symbols.h"
#include "iree/hal/drivers/hip/event_pool.h"
#include "iree/hal/drivers/hip/status_util.h"
#include "iree/hal/drivers/hip/util/tree.h"
#include "iree/hal/utils/semaphore_base.h"

typedef struct iree_hal_hip_cpu_event_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  iree_event_t event;
} iree_hal_hip_cpu_event_t;

static void iree_hal_hip_cpu_event_destroy(iree_hal_resource_t* resource) {
  iree_hal_hip_cpu_event_t* event = (iree_hal_hip_cpu_event_t*)(resource);
  iree_event_deinitialize(&event->event);
  iree_allocator_free(event->host_allocator, event);
}

static const iree_hal_resource_vtable_t iree_hal_hip_cpu_event_vtable = {
    .destroy = &iree_hal_hip_cpu_event_destroy,
};

typedef struct iree_hal_hip_cpu_event_vtable_t {
  void(IREE_API_PTR* destroy)(iree_hal_resource_t* resource);
} iree_hal_hip_cpu_event_vtable_t;

typedef struct iree_hal_hip_semaphore_work_item_t {
  iree_hal_hip_event_semaphore_scheduled_callback_t scheduled_callback;
  void* user_data;
  struct iree_hal_hip_semaphore_work_item_t* next;
} iree_hal_hip_semaphore_work_item_t;

// The type of the queue item.
typedef enum {
  // A standard queue item.
  // This is the bulk of the semaphore timepoint types.
  IREE_HAL_HIP_SEMAPHORE_QUEUE_ITEM_INTERNAL,
  // A queue item attached to an imported timepoint.
  IREE_HAL_HIP_SEMAPHORE_QUEUE_ITEM_IMPORTED,
  // A queue item attached to an exported timepoint.
  IREE_HAL_HIP_SEMAPHORE_QUEUE_ITEM_EXPORTED,
} iree_hal_hip_semaphore_queue_item_type_t;

// Work associated with a particular point in the semaphore timeline.
//
// The |work_item| is a set of callbacks to be made when the semaphore
// is guaranteed to make forward progress the associated key value. They
// will also be cleaned up at this time. If the semaphore is failed,
// the callbacks will be called with the status code of the failure.
// If the semaphore is destroyed while callbacks are active,
// they will be called with the CANCELLED erorr.
// The |cpu_event| is a value for the CPU to wait on when
// we may not have to wait infinitely. For example with a multi
// wait or a non-infinite timeout.
// The |event| is a hip event that is used for GPU waits or
// infinite CPU waits.
// The |imported_event| is an event that has been imported
// from an external source, that will be waited on.
typedef struct iree_hal_hip_semaphore_queue_item_t {
  iree_hal_hip_semaphore_queue_item_type_t type;
  iree_hal_hip_event_t* event;
  iree_hal_hip_event_t* secondary_event;
  iree_hal_hip_cpu_event_t* cpu_event;
  iree_hal_hip_semaphore_work_item_t* work_item;
  iree_hal_hip_per_device_info_t* created_device;
  bool has_been_waited_on : 1;
  bool has_been_signaled : 1;
  bool is_external : 1;
} iree_hal_hip_semaphore_queue_item_t;

typedef struct iree_hal_hip_semaphore_t {
  // Abstract resource used for injecting reference counting and vtable;
  // must be at offset 0.
  iree_hal_resource_t base;

  // The allocator used to create this semaphore.
  iree_allocator_t host_allocator;
  // The symbols used to issue HIP API calls.
  const iree_hal_hip_dynamic_symbols_t* symbols;

  // This queue represents the values in the timeline.
  // The keys in the queue are the timeline values that
  // are being signaled/waited on in the semaphore
  // The values are |iree_hal_hip_semaphore_queue_item_t| values.
  struct {
    iree_hal_hip_util_tree_t tree;
    // Inline storage for this tree. We expect the normal number of
    // nodes in use for a single semaphore to be relatively small.
    uint8_t inline_storage[sizeof(iree_hal_hip_util_tree_node_t) * 16];
  } event_queue;

  // Notify any potential CPU waiters that this semaphore
  // has changed state.
  iree_notification_t state_notification;

  iree_notification_t external_event_notification;

  iree_hal_hip_device_topology_t devices;

  iree_slim_mutex_t mutex;
  // The maximum value that this semaphore has been signaled to.
  // This means this semaphore is guaranteed to make forward progress
  // until that semaphore is hit, as all signaling operations have
  // been made available.
  uint64_t max_value_to_be_signaled IREE_GUARDED_BY(mutex);

  // The largest value that has been observed by the host.
  uint64_t current_visible_value IREE_GUARDED_BY(mutex);

  // OK or the status passed to iree_hal_semaphore_fail. Owned by the semaphore.
  iree_status_t failure_status IREE_GUARDED_BY(mutex);
} iree_hal_hip_semaphore_t;

static const iree_hal_semaphore_vtable_t iree_hal_hip_semaphore_vtable;

static iree_hal_hip_semaphore_t* iree_hal_hip_semaphore_cast(
    iree_hal_semaphore_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_hip_semaphore_vtable);
  return (iree_hal_hip_semaphore_t*)base_value;
}

iree_status_t iree_hal_hip_event_semaphore_create(
    uint64_t initial_value, const iree_hal_hip_dynamic_symbols_t* symbols,
    iree_allocator_t host_allocator, iree_hal_hip_device_topology_t topology,
    iree_hal_semaphore_t** out_semaphore) {
  IREE_ASSERT_ARGUMENT(symbols);
  IREE_ASSERT_ARGUMENT(out_semaphore);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_semaphore = NULL;

  iree_hal_hip_semaphore_t* semaphore = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*semaphore),
                                (void**)&semaphore));

  iree_hal_semaphore_initialize(&iree_hal_hip_semaphore_vtable,
                                (iree_hal_semaphore_t*)semaphore);
  semaphore->host_allocator = host_allocator;
  semaphore->symbols = symbols;
  semaphore->devices = topology;
  iree_hal_hip_util_tree_initialize(
      host_allocator, sizeof(iree_hal_hip_semaphore_queue_item_t),
      semaphore->event_queue.inline_storage,
      sizeof(semaphore->event_queue.inline_storage),
      &semaphore->event_queue.tree);
  iree_notification_initialize(&semaphore->state_notification);
  iree_notification_initialize(&semaphore->external_event_notification);

  iree_slim_mutex_initialize(&semaphore->mutex);
  semaphore->current_visible_value = initial_value;
  semaphore->max_value_to_be_signaled = initial_value;
  semaphore->failure_status = iree_ok_status();

  *out_semaphore = (iree_hal_semaphore_t*)semaphore;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_hip_semaphore_destroy(
    iree_hal_semaphore_t* base_semaphore) {
  iree_hal_hip_semaphore_t* semaphore =
      iree_hal_hip_semaphore_cast(base_semaphore);
  iree_allocator_t host_allocator = semaphore->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_ignore(semaphore->failure_status);
  iree_slim_mutex_deinitialize(&semaphore->mutex);

  iree_notification_deinitialize(&semaphore->external_event_notification);
  iree_notification_deinitialize(&semaphore->state_notification);
  for (iree_hal_hip_util_tree_node_t* i =
           iree_hal_hip_util_tree_first(&semaphore->event_queue.tree);
       i != NULL; i = iree_hal_hip_util_tree_node_next(i)) {
    iree_hal_hip_semaphore_queue_item_t* queue_item =
        (iree_hal_hip_semaphore_queue_item_t*)
            iree_hal_hip_util_tree_node_get_value(i);
    iree_hal_hip_event_release(queue_item->event);
    iree_hal_hip_event_release(queue_item->secondary_event);
    iree_hal_resource_release(queue_item->cpu_event);
    iree_hal_hip_semaphore_work_item_t* work_item = queue_item->work_item;
    while (work_item) {
      work_item->scheduled_callback(
          work_item->user_data, base_semaphore,
          iree_make_status(
              IREE_STATUS_CANCELLED,
              "semaphore was destroyed while callback is in flight"));
      iree_hal_hip_semaphore_work_item_t* next = work_item->next;
      iree_allocator_free(host_allocator, work_item);
      work_item = next;
    }
  }
  iree_hal_hip_util_tree_deinitialize(&semaphore->event_queue.tree);
  iree_allocator_free(host_allocator, semaphore);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_hip_semaphore_get_cpu_event(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    iree_hal_hip_cpu_event_t** out_event) {
  IREE_ASSERT_ARGUMENT(out_event);
  *out_event = NULL;
  iree_hal_hip_semaphore_t* semaphore =
      iree_hal_hip_semaphore_cast(base_semaphore);
  iree_slim_mutex_lock(&semaphore->mutex);
  if (value <= semaphore->current_visible_value) {
    iree_slim_mutex_unlock(&semaphore->mutex);
    return iree_ok_status();
  }
  iree_status_t status = iree_ok_status();
  iree_hal_hip_util_tree_node_t* node =
      iree_hal_hip_util_tree_get(&semaphore->event_queue.tree, value);
  if (!node) {
    status = iree_hal_hip_util_tree_insert(&semaphore->event_queue.tree, value,
                                           &node);
  }

  iree_hal_hip_semaphore_queue_item_t* item = NULL;
  if (iree_status_is_ok(status)) {
    item = (iree_hal_hip_semaphore_queue_item_t*)
        iree_hal_hip_util_tree_node_get_value(node);

    if (!item->cpu_event) {
      status = iree_allocator_malloc(semaphore->host_allocator,
                                     sizeof(*item->cpu_event),
                                     (void**)&item->cpu_event);
      if (iree_status_is_ok(status)) {
        iree_hal_resource_initialize(&iree_hal_hip_cpu_event_vtable,
                                     (iree_hal_resource_t*)item->cpu_event);
        item->cpu_event->host_allocator = semaphore->host_allocator;

        status = iree_event_initialize(false, &item->cpu_event->event);
        if (!iree_status_is_ok(status)) {
          // Clear out the cpu_event here, so that we dont have to
          // special case cleanup later.
          iree_allocator_free(semaphore->host_allocator, item->cpu_event);
          item->cpu_event = NULL;
        }
      }
    }

    if (iree_status_is_ok(status)) {
      iree_hal_resource_retain(&item->cpu_event->resource);
      *out_event = item->cpu_event;
    }
  }
  iree_slim_mutex_unlock(&semaphore->mutex);
  if (!iree_status_is_ok(status)) {
    if (item && item->cpu_event) {
      iree_event_deinitialize(&item->cpu_event->event);
      iree_allocator_free(semaphore->host_allocator, item->cpu_event);
    }
  }
  return status;
}

static bool iree_hal_hip_semaphore_is_aborted(
    iree_hal_semaphore_t* base_semaphore) {
  iree_hal_hip_semaphore_t* semaphore =
      iree_hal_hip_semaphore_cast(base_semaphore);

  iree_slim_mutex_lock(&semaphore->mutex);
  bool aborted =
      semaphore->current_visible_value >= IREE_HAL_SEMAPHORE_FAILURE_VALUE;
  iree_slim_mutex_unlock(&semaphore->mutex);
  return aborted;
}

iree_status_t iree_hal_hip_semaphore_multi_wait(
    const iree_hal_semaphore_list_t semaphore_list,
    iree_hal_wait_mode_t wait_mode, iree_timeout_t timeout,
    iree_allocator_t host_allocator) {
  if (semaphore_list.count == 0) return iree_ok_status();
  IREE_TRACE_ZONE_BEGIN(z0);

  const iree_time_t deadline_ns = iree_timeout_as_deadline_ns(timeout);
  iree_status_t status = iree_ok_status();

  // If we have to wait on "all" semaphores then we can
  // fast-path this to just a normal wait.
  if (semaphore_list.count == 1 || wait_mode == IREE_HAL_WAIT_MODE_ALL) {
    // Fast-path if we don't have to wait on only a subset of the semaphores.
    for (iree_host_size_t i = 0; i < semaphore_list.count; ++i) {
      iree_timeout_t t = iree_make_deadline(deadline_ns);
      status = iree_status_join(
          status, iree_hal_semaphore_wait(semaphore_list.semaphores[0],
                                          semaphore_list.payload_values[0], t));
      if (!iree_status_is_ok(status)) {
        break;
      }
    }
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  iree_hal_hip_cpu_event_t** cpu_events = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator,
                                semaphore_list.count * sizeof(*cpu_events),
                                (void**)&cpu_events));
  bool semaphore_hit = false;
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < semaphore_list.count; ++i) {
      status = iree_hal_hip_semaphore_get_cpu_event(
          semaphore_list.semaphores[i], semaphore_list.payload_values[i],
          &cpu_events[i]);
      if (!iree_status_is_ok(status)) {
        break;
      }
      // If we can not get a CPU event for a given value BUT it returns success
      // it is because the event has already been signaled to that value.
      if (!cpu_events[i]) {
        semaphore_hit = true;
        if (iree_hal_hip_semaphore_is_aborted(semaphore_list.semaphores[i])) {
          status = iree_make_status(IREE_STATUS_ABORTED,
                                    "the semaphore was aborted");
        }
        break;
      }
    }
  }

  iree_wait_set_t* wait_set = NULL;
  if (iree_status_is_ok(status) && !semaphore_hit) {
    status =
        iree_wait_set_allocate(semaphore_list.count, host_allocator, &wait_set);
  }

  if (iree_status_is_ok(status) && !semaphore_hit) {
    for (iree_host_size_t i = 0;
         i < semaphore_list.count && iree_status_is_ok(status); ++i) {
      status = iree_wait_set_insert(wait_set, cpu_events[i]->event);
    }
  }

  if (iree_status_is_ok(status) && !semaphore_hit) {
    status = iree_wait_any(wait_set, deadline_ns, NULL);
    iree_wait_set_free(wait_set);
    if (iree_status_is_ok(status)) {
      // Now we have to walk all of the semaphores to propagate
      // any errors that we find.
      for (iree_host_size_t i = 0; i < semaphore_list.count; ++i) {
        if (iree_hal_hip_semaphore_is_aborted(semaphore_list.semaphores[i])) {
          status = iree_make_status(IREE_STATUS_ABORTED,
                                    "the semaphore was aborted");
          break;
        }
      }
    }
  }

  for (iree_host_size_t i = 0; i < semaphore_list.count; ++i) {
    iree_hal_resource_release(&cpu_events[i]->resource);
  }
  iree_allocator_free(host_allocator, cpu_events);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_hip_event_semaphore_run_scheduled_callbacks(
    iree_hal_semaphore_t* base_semaphore) {
  iree_hal_hip_semaphore_t* semaphore =
      iree_hal_hip_semaphore_cast(base_semaphore);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_hip_semaphore_work_item_t* work_item = NULL;
  iree_hal_hip_semaphore_work_item_t* last_work_item = NULL;

  // Take out all of the values from the queue that are less than the
  // current visible value, and make sure we advance any work needed
  // on them.
  do {
    iree_slim_mutex_lock(&semaphore->mutex);
    iree_hal_hip_util_tree_node_t* node =
        iree_hal_hip_util_tree_first(&semaphore->event_queue.tree);
    if (node == NULL) {
      iree_slim_mutex_unlock(&semaphore->mutex);
      break;
    }
    if (iree_hal_hip_util_tree_node_get_key(node) >
        semaphore->current_visible_value) {
      iree_slim_mutex_unlock(&semaphore->mutex);
      break;
    }
    iree_hal_hip_semaphore_queue_item_t copy =
        *(iree_hal_hip_semaphore_queue_item_t*)
            iree_hal_hip_util_tree_node_get_value(node);
    iree_hal_hip_util_tree_erase(&semaphore->event_queue.tree, node);
    iree_slim_mutex_unlock(&semaphore->mutex);
    iree_hal_hip_event_release(copy.event);
    iree_hal_hip_event_release(copy.secondary_event);
    if (copy.cpu_event) {
      iree_event_set(&copy.cpu_event->event);
      iree_hal_resource_release(&copy.cpu_event->resource);
    }

    iree_hal_hip_semaphore_work_item_t* next_work_item = copy.work_item;
    while (next_work_item) {
      if (!work_item) {
        work_item = next_work_item;
      }
      if (last_work_item && !last_work_item->next) {
        last_work_item->next = next_work_item;
      }
      last_work_item = next_work_item;
      next_work_item = next_work_item->next;
    }
  } while (true);

  iree_slim_mutex_lock(&semaphore->mutex);
  if (semaphore->max_value_to_be_signaled < semaphore->current_visible_value) {
    semaphore->max_value_to_be_signaled = iree_max(
        semaphore->max_value_to_be_signaled, semaphore->current_visible_value);
    iree_notification_post(&semaphore->external_event_notification,
                           IREE_ALL_WAITERS);
  }

  iree_status_t status = iree_status_clone(semaphore->failure_status);

  iree_slim_mutex_unlock(&semaphore->mutex);
  // Now that we have accumulated all of the work items, and we have
  // unlocked the semaphore, start running through the work items.
  while (work_item) {
    iree_hal_hip_semaphore_work_item_t* next_work_item = work_item->next;
    iree_status_ignore(work_item->scheduled_callback(
        work_item->user_data, base_semaphore, iree_status_clone(status)));
    iree_allocator_free(semaphore->host_allocator, work_item);
    work_item = next_work_item;
  }

  iree_notification_post(&semaphore->state_notification, IREE_ALL_WAITERS);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_hip_semaphore_notify_work(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    iree_hal_hip_event_pool_t* event_pool,
    iree_hal_hip_event_semaphore_scheduled_callback_t callback,
    void* user_data) {
  iree_hal_hip_semaphore_t* semaphore =
      iree_hal_hip_semaphore_cast(base_semaphore);
  iree_slim_mutex_lock(&semaphore->mutex);
  iree_status_t status = iree_status_clone(semaphore->failure_status);

  if (iree_status_is_ok(status) &&
      value > semaphore->max_value_to_be_signaled) {
    iree_hal_hip_util_tree_node_t* node =
        iree_hal_hip_util_tree_get(&semaphore->event_queue.tree, value);
    if (node == NULL) {
      status = iree_hal_hip_util_tree_insert(&semaphore->event_queue.tree,
                                             value, &node);
      if (iree_status_is_ok(status)) {
        iree_hal_hip_semaphore_queue_item_t* item =
            (iree_hal_hip_semaphore_queue_item_t*)
                iree_hal_hip_util_tree_node_get_value(node);
        item->type = IREE_HAL_HIP_SEMAPHORE_QUEUE_ITEM_INTERNAL;
        item->event = NULL;
        item->cpu_event = NULL;
        item->work_item = NULL;
        item->created_device = NULL;
      }
    }
    if (iree_status_is_ok(status)) {
      iree_hal_hip_semaphore_queue_item_t* item =
          (iree_hal_hip_semaphore_queue_item_t*)
              iree_hal_hip_util_tree_node_get_value(node);
      iree_hal_hip_semaphore_work_item_t* work_item = NULL;
      status = iree_allocator_malloc(semaphore->host_allocator,
                                     sizeof(*work_item), (void**)&work_item);
      if (iree_status_is_ok(status)) {
        work_item->scheduled_callback = callback;
        work_item->user_data = user_data;
        work_item->next = item->work_item;
        item->work_item = work_item;
        callback = NULL;
      }
    }
  }
  iree_slim_mutex_unlock(&semaphore->mutex);

  // If this semaphore requirement has already been satisfied,
  // of if this semaphore has failed then we can just run the callback right
  // now.
  if (callback) {
    status = callback(user_data, base_semaphore, status);
  }
  return status;
}

iree_status_t iree_hal_hip_semaphore_notify_forward_progress_to(
    iree_hal_semaphore_t* base_semaphore, uint64_t value) {
  iree_hal_hip_semaphore_t* semaphore =
      iree_hal_hip_semaphore_cast(base_semaphore);
  iree_slim_mutex_lock(&semaphore->mutex);
  iree_status_t status = iree_status_clone(semaphore->failure_status);
  if (!iree_status_is_ok(status)) {
    iree_slim_mutex_unlock(&semaphore->mutex);
    return status;
  }
  iree_hal_hip_semaphore_work_item_t* work_item = NULL;
  iree_hal_hip_semaphore_work_item_t* last_work_item = NULL;

  iree_hal_hip_util_tree_node_t* node = iree_hal_hip_util_tree_upper_bound(
      &semaphore->event_queue.tree, semaphore->max_value_to_be_signaled);
  // Collect all of the things to schedule now that we know we can safely make
  // it to a given value.
  while (node) {
    iree_hal_hip_semaphore_queue_item_t* queue_item =
        (iree_hal_hip_semaphore_queue_item_t*)
            iree_hal_hip_util_tree_node_get_value(node);

    // If a semaphore is imported then we consider it valid
    // as soon as the semaphore preceding it is valid.
    if (iree_hal_hip_util_tree_node_get_key(node) > value &&
        queue_item->type != IREE_HAL_HIP_SEMAPHORE_QUEUE_ITEM_IMPORTED) {
      break;
    }

    value =
        iree_max(value, iree_max(semaphore->max_value_to_be_signaled,
                                 iree_hal_hip_util_tree_node_get_key(node)));

    iree_hal_hip_semaphore_work_item_t* next_work_item = queue_item->work_item;
    while (next_work_item) {
      if (!work_item) {
        work_item = next_work_item;
      }
      if (last_work_item && !last_work_item->next) {
        last_work_item->next = next_work_item;
      }
      last_work_item = next_work_item;
      next_work_item = next_work_item->next;
    }
    queue_item->work_item = NULL;
    iree_hal_hip_util_tree_node_t* last_node = node;
    node = iree_hal_hip_util_tree_node_next(node);
    if (!queue_item->event) {
      iree_hal_hip_util_tree_erase(&semaphore->event_queue.tree, last_node);
    }
  }

  if (value > semaphore->max_value_to_be_signaled) {
    semaphore->max_value_to_be_signaled =
        iree_max(semaphore->max_value_to_be_signaled, value);
    iree_notification_post(&semaphore->external_event_notification,
                           IREE_ALL_WAITERS);
  }
  iree_slim_mutex_unlock(&semaphore->mutex);

  // Now that we have accumulated all of the work items, and we have
  // unlocked the semaphore, start running through the work items.
  while (work_item) {
    iree_hal_hip_semaphore_work_item_t* next_work_item = work_item->next;
    work_item->scheduled_callback(work_item->user_data, base_semaphore, status);
    iree_allocator_free(semaphore->host_allocator, work_item);
    work_item = next_work_item;
  }
  return status;
}

iree_status_t iree_hal_hip_semaphore_wait_hip_events(
    iree_hal_semaphore_t* base_semaphore, uint64_t value, hipStream_t stream) {
  iree_hal_hip_semaphore_t* semaphore =
      iree_hal_hip_semaphore_cast(base_semaphore);

  iree_slim_mutex_lock(&semaphore->mutex);
  if (value <= semaphore->current_visible_value) {
    iree_slim_mutex_unlock(&semaphore->mutex);
    return iree_ok_status();
  }
  iree_status_t status = iree_status_clone(semaphore->failure_status);
  if (iree_status_is_ok(status)) {
    iree_hal_hip_util_tree_node_t* node =
        iree_hal_hip_util_tree_get(&semaphore->event_queue.tree, value);

    if (node == NULL) {
      status = iree_hal_hip_util_tree_insert(&semaphore->event_queue.tree,
                                             value, &node);
      if (iree_status_is_ok(status)) {
        iree_hal_hip_semaphore_queue_item_t* item =
            (iree_hal_hip_semaphore_queue_item_t*)
                iree_hal_hip_util_tree_node_get_value(node);
        item->cpu_event = NULL;
        item->work_item = NULL;
        item->type = IREE_HAL_HIP_SEMAPHORE_QUEUE_ITEM_INTERNAL;
        item->event = NULL;
        item->created_device = NULL;
      }
    }

    if (iree_status_is_ok(status)) {
      // Walk backwards in time and make sure we have at least
      // waited for all events once.
      // This is especially important as some events are
      // external, and we cannot rely on hip ordering.

      iree_hal_hip_util_tree_node_t* prev_node =
          iree_hal_hip_util_tree_node_prev(node);
      while (prev_node && iree_status_is_ok(status)) {
        iree_hal_hip_semaphore_queue_item_t* item =
            (iree_hal_hip_semaphore_queue_item_t*)
                iree_hal_hip_util_tree_node_get_value(prev_node);
        if (item->has_been_waited_on) {
          break;
        }
        if (item->event) {
          status = IREE_HIP_CALL_TO_STATUS(
              semaphore->symbols,
              hipStreamWaitEvent(stream, iree_hal_hip_event_handle(item->event),
                                 0));
        }

        item->has_been_waited_on = true;
        prev_node = iree_hal_hip_util_tree_node_prev(prev_node);
      }

      iree_hal_hip_event_t* event =
          ((iree_hal_hip_semaphore_queue_item_t*)
               iree_hal_hip_util_tree_node_get_value(node))
              ->event;
      if (!event) {
        do {
          node = iree_hal_hip_util_tree_node_next(node);
          if (!node) {
            status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                      "there was no event that could be valid");
            break;
          }
          event = ((iree_hal_hip_semaphore_queue_item_t*)
                       iree_hal_hip_util_tree_node_get_value(node))
                      ->event;
        } while (!event);
      }

      status = IREE_HIP_CALL_TO_STATUS(
          semaphore->symbols,
          hipStreamWaitEvent(stream, iree_hal_hip_event_handle(event), 0));
    }
  }
  iree_slim_mutex_unlock(&semaphore->mutex);

  return status;
}

iree_status_t iree_hal_hip_semaphore_create_event_and_record_if_necessary(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    iree_hal_hip_per_device_info_t* device, hipStream_t dispatch_stream,
    iree_hal_hip_event_pool_t* event_pool) {
  iree_hal_hip_semaphore_t* semaphore =
      iree_hal_hip_semaphore_cast(base_semaphore);
  iree_slim_mutex_lock(&semaphore->mutex);
  if (value <= semaphore->current_visible_value) {
    iree_slim_mutex_unlock(&semaphore->mutex);
    return iree_ok_status();
  }
  iree_status_t status = iree_status_clone(semaphore->failure_status);
  if (iree_status_is_ok(status)) {
    iree_hal_hip_util_tree_node_t* node =
        iree_hal_hip_util_tree_get(&semaphore->event_queue.tree, value);

    if (node == NULL) {
      status = iree_hal_hip_util_tree_insert(&semaphore->event_queue.tree,
                                             value, &node);
      if (iree_status_is_ok(status)) {
        iree_hal_hip_semaphore_queue_item_t* item =
            (iree_hal_hip_semaphore_queue_item_t*)
                iree_hal_hip_util_tree_node_get_value(node);
        item->cpu_event = NULL;
        item->work_item = NULL;
        item->event = NULL;
        item->has_been_waited_on = false;
        item->type = IREE_HAL_HIP_SEMAPHORE_QUEUE_ITEM_INTERNAL;
      }
    }

    if (iree_status_is_ok(status)) {
      iree_hal_hip_semaphore_queue_item_t* value =
          (iree_hal_hip_semaphore_queue_item_t*)
              iree_hal_hip_util_tree_node_get_value(node);
      if (!value->event) {
        status = iree_hal_hip_event_pool_acquire(event_pool, 1, &value->event);
        value->created_device = device;
      }
      if (iree_status_is_ok(status) && !value->has_been_signaled) {
        // If the event was created on a different device, then we
        // have to actually signal a secondary event first.
        if (device != value->created_device) {
          status = iree_hal_hip_event_pool_acquire(event_pool, 1,
                                                   &value->secondary_event);
          if (iree_status_is_ok(status)) {
            status = IREE_HIP_CALL_TO_STATUS(
                semaphore->symbols, hipEventRecord(iree_hal_hip_event_handle(
                                                       value->secondary_event),
                                                   dispatch_stream));
          }
          if (iree_status_is_ok(status)) {
            status = IREE_HIP_CALL_TO_STATUS(
                semaphore->symbols,
                hipStreamWaitEvent(
                    value->created_device->hip_async_memory_stream,
                    iree_hal_hip_event_handle(value->secondary_event), 0));
          }
          if (iree_status_is_ok(status)) {
            status = IREE_HIP_CALL_TO_STATUS(
                semaphore->symbols,
                hipEventRecord(iree_hal_hip_event_handle(value->event),
                               value->created_device->hip_async_memory_stream));
          }
        } else {
          status = IREE_HIP_CALL_TO_STATUS(
              semaphore->symbols,
              hipEventRecord(iree_hal_hip_event_handle(value->event),
                             dispatch_stream));
        }
        value->has_been_signaled = true;
      }
    }
  }
  iree_slim_mutex_unlock(&semaphore->mutex);

  return status;
}

static iree_status_t iree_hal_hip_semaphore_query_locked(
    iree_hal_hip_semaphore_t* semaphore, uint64_t* out_value) {
  iree_status_t status = iree_ok_status();
  *out_value = semaphore->current_visible_value;
  iree_hal_hip_util_tree_node_t* node =
      iree_hal_hip_util_tree_first(&semaphore->event_queue.tree);
  while (node) {
    if (!((iree_hal_hip_semaphore_queue_item_t*)
              iree_hal_hip_util_tree_node_get_value(node))
             ->event) {
      node = iree_hal_hip_util_tree_node_next(node);
      continue;
    }
    // If we have an event but it has yet to be signaled
    // then we can't even query it (as it will return the wrong answer)
    // This happens when we have an external semaphore.
    if (!((iree_hal_hip_semaphore_queue_item_t*)
              iree_hal_hip_util_tree_node_get_value(node))
             ->has_been_signaled) {
      break;
    }

    hipError_t err =
        semaphore->symbols->hipEventQuery(iree_hal_hip_event_handle(
            ((iree_hal_hip_semaphore_queue_item_t*)
                 iree_hal_hip_util_tree_node_get_value(node))
                ->event));
    if (err == hipErrorNotReady) {
      break;
    }
    if (err != hipSuccess) {
      status = IREE_HIP_RESULT_TO_STATUS(semaphore->symbols, err);
      break;
    }

    *out_value = iree_hal_hip_util_tree_node_get_key(node);
    node = iree_hal_hip_util_tree_node_next(node);
  }

  if (iree_status_is_ok(status)) {
    if (semaphore->current_visible_value < *out_value) {
      semaphore->current_visible_value = *out_value;
      iree_notification_post(&semaphore->state_notification, IREE_ALL_WAITERS);
    }

    if (*out_value >= IREE_HAL_SEMAPHORE_FAILURE_VALUE) {
      status =
          iree_make_status(IREE_STATUS_ABORTED, "the semaphore was aborted");
    }
  }

  return status;
}

static iree_status_t iree_hal_hip_semaphore_query(
    iree_hal_semaphore_t* base_semaphore, uint64_t* out_value) {
  iree_hal_hip_semaphore_t* semaphore =
      iree_hal_hip_semaphore_cast(base_semaphore);

  iree_slim_mutex_lock(&semaphore->mutex);
  *out_value = semaphore->current_visible_value;

  iree_status_t status =
      iree_hal_hip_semaphore_query_locked(semaphore, out_value);
  iree_slim_mutex_unlock(&semaphore->mutex);
  // If the status is aborted, we will pick up the real status from
  // semaphore_advance.
  if (iree_status_is_aborted(status)) {
    iree_status_ignore(status);
    status = iree_ok_status();
  }

  return iree_status_join(
      status,
      iree_hal_hip_event_semaphore_run_scheduled_callbacks(base_semaphore));
}

iree_status_t iree_hal_hip_event_semaphore_advance(
    iree_hal_semaphore_t* base_semaphore) {
  iree_hal_hip_semaphore_t* semaphore =
      iree_hal_hip_semaphore_cast(base_semaphore);

  iree_slim_mutex_lock(&semaphore->mutex);

  iree_status_t status = iree_ok_status();
  iree_hal_hip_util_tree_node_t* node =
      iree_hal_hip_util_tree_first(&semaphore->event_queue.tree);

  iree_host_size_t highest_value = 0;
  while (node) {
    if (!((iree_hal_hip_semaphore_queue_item_t*)
              iree_hal_hip_util_tree_node_get_value(node))
             ->event) {
      node = iree_hal_hip_util_tree_node_next(node);
      continue;
    }

    // If we have an event but it has yet to be signaled
    // then we can't even query it (as it will return the wrong answer)
    // This happens when we have an external semaphore.
    if (!((iree_hal_hip_semaphore_queue_item_t*)
              iree_hal_hip_util_tree_node_get_value(node))
             ->has_been_signaled) {
      break;
    }

    hipError_t err =
        semaphore->symbols->hipEventQuery(iree_hal_hip_event_handle(
            ((iree_hal_hip_semaphore_queue_item_t*)
                 iree_hal_hip_util_tree_node_get_value(node))
                ->event));
    if (err == hipErrorNotReady) {
      break;
    }
    if (err != hipSuccess) {
      status = IREE_HIP_RESULT_TO_STATUS(semaphore->symbols, err);
      break;
    }

    highest_value = iree_hal_hip_util_tree_node_get_key(node);
    node = iree_hal_hip_util_tree_node_next(node);
  }

  if (iree_status_is_ok(status)) {
    if (semaphore->current_visible_value < highest_value) {
      semaphore->current_visible_value = highest_value;
      iree_notification_post(&semaphore->state_notification, IREE_ALL_WAITERS);
    }

    if (highest_value >= IREE_HAL_SEMAPHORE_FAILURE_VALUE) {
      status =
          iree_make_status(IREE_STATUS_ABORTED, "the semaphore was aborted");
    }
  }

  iree_slim_mutex_unlock(&semaphore->mutex);
  // If the status is aborted, we will pick up the real status from
  // iree_hal_hip_event_semaphore_run_scheduled_callbacks.
  if (iree_status_is_aborted(status)) {
    iree_status_ignore(status);
    status = iree_ok_status();
  }
  status = iree_status_join(
      status,
      iree_hal_hip_event_semaphore_run_scheduled_callbacks(base_semaphore));
  return status;
}

static iree_status_t iree_hal_hip_semaphore_signal(
    iree_hal_semaphore_t* base_semaphore, uint64_t new_value) {
  iree_hal_hip_semaphore_t* semaphore =
      iree_hal_hip_semaphore_cast(base_semaphore);
  iree_slim_mutex_lock(&semaphore->mutex);

  iree_status_t status = iree_ok_status();
  if (new_value <= semaphore->current_visible_value) {
    uint64_t current_value IREE_ATTRIBUTE_UNUSED =
        semaphore->current_visible_value;
    iree_slim_mutex_unlock(&semaphore->mutex);
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "semaphore values must be monotonically "
                              "increasing; current_value=%" PRIu64
                              ", new_value=%" PRIu64,
                              current_value, new_value);
  }

  if (iree_status_is_ok(status)) {
    semaphore->current_visible_value = new_value;
  }

  iree_slim_mutex_unlock(&semaphore->mutex);

  if (iree_status_is_ok(status)) {
    status =
        iree_hal_hip_event_semaphore_run_scheduled_callbacks(base_semaphore);
  }
  return status;
}

static void iree_hal_hip_semaphore_fail(iree_hal_semaphore_t* base_semaphore,
                                        iree_status_t status) {
  iree_hal_hip_semaphore_t* semaphore =
      iree_hal_hip_semaphore_cast(base_semaphore);

  iree_slim_mutex_lock(&semaphore->mutex);

  // Try to set our local status - we only preserve the first failure so only
  // do this if we are going from a valid semaphore to a failed one.
  if (!iree_status_is_ok(semaphore->failure_status)) {
    // Previous sta-tus was not OK; drop our new status.
    iree_slim_mutex_unlock(&semaphore->mutex);
    return;
  }

  // Signal to our failure sentinel value.
  semaphore->current_visible_value = IREE_HAL_SEMAPHORE_FAILURE_VALUE;
  semaphore->failure_status = status;

  iree_slim_mutex_unlock(&semaphore->mutex);
  iree_status_ignore(
      iree_hal_hip_event_semaphore_run_scheduled_callbacks(base_semaphore));
}

static iree_status_t iree_hal_hip_semaphore_wait(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    iree_timeout_t timeout) {
  iree_hal_hip_semaphore_t* semaphore =
      iree_hal_hip_semaphore_cast(base_semaphore);
  IREE_TRACE_ZONE_BEGIN(z0);

  const iree_time_t deadline_ns = iree_timeout_as_deadline_ns(timeout);
  iree_slim_mutex_lock(&semaphore->mutex);
  uint64_t current_value = 0;

  // query_locked to make sure our count is up to date.
  iree_status_t status =
      iree_hal_hip_semaphore_query_locked(semaphore, &current_value);

  if (iree_status_is_ok(status)) {
    while (semaphore->max_value_to_be_signaled < value) {
      if (iree_time_now() > deadline_ns) {
        status = iree_make_status(IREE_STATUS_DEADLINE_EXCEEDED);
        break;
      }
      iree_wait_token_t wait =
          iree_notification_prepare_wait(&semaphore->state_notification);
      iree_slim_mutex_unlock(&semaphore->mutex);

      // We have to wait for the semaphore to catch up.
      bool committed =
          iree_notification_commit_wait(&semaphore->state_notification, wait,
                                        IREE_DURATION_ZERO, deadline_ns);

      iree_slim_mutex_lock(&semaphore->mutex);
      if (!committed) {
        status = iree_make_status(IREE_STATUS_DEADLINE_EXCEEDED);
        break;
      }

      // query_locked to make sure our count is up to date.
      status = iree_hal_hip_semaphore_query_locked(semaphore, &current_value);
      if (!iree_status_is_ok(status)) {
        break;
      }
    }
  }

  if (iree_status_is_ok(status)) {
    // The current value stored in the semaphore is greater than the current
    // value, so we can return.
    if (semaphore->current_visible_value >= value) {
      iree_slim_mutex_unlock(&semaphore->mutex);
      iree_hal_hip_event_semaphore_run_scheduled_callbacks(base_semaphore);
      iree_slim_mutex_lock(&semaphore->mutex);
    } else if (iree_timeout_is_infinite(timeout)) {
      // This is the fast-path. Since we have an infinite timeout, we can
      // wait directly on the hip event.

      // The current value is not enough, but we have at least submitted
      // the work that will increment the semaphore to the value we need.
      // Use iree_hal_hip_util_tree_lower_bound to find the first element in the
      // tree that would signal our semaphore to at least the given value.
      iree_hal_hip_util_tree_node_t* node = iree_hal_hip_util_tree_lower_bound(
          &semaphore->event_queue.tree, value);
      IREE_ASSERT(
          node,
          "We really should either have an event in the queue that will satisfy"
          "this semaphore (we checked max_value_to_be_signaled above) or we"
          "should already have signaled (current_visible_value above)");
      iree_hal_hip_semaphore_queue_item_t* item =
          (iree_hal_hip_semaphore_queue_item_t*)
              iree_hal_hip_util_tree_node_get_value(node);

      iree_hal_hip_event_t* event = item->event;

      // Retain the event, as the event may be removed from the tree
      // while we sleep on the event.
      iree_hal_hip_event_retain(event);
      iree_slim_mutex_unlock(&semaphore->mutex);
      iree_hal_hip_event_semaphore_run_scheduled_callbacks(base_semaphore);
      status = IREE_HIP_CALL_TO_STATUS(
          semaphore->symbols,
          hipEventSynchronize(iree_hal_hip_event_handle(event)));
      iree_hal_hip_event_release(event);
      iree_slim_mutex_lock(&semaphore->mutex);
    } else {
      // If we have a non-infinite timeout, this is the slow-path.
      // because we will end up having to wait for either the
      // cleanup thread, or someone else to advance the
      // semaphore.
      iree_slim_mutex_unlock(&semaphore->mutex);
      iree_hal_hip_cpu_event_t* cpu_event = NULL;
      status = iree_hal_hip_semaphore_get_cpu_event(base_semaphore, value,
                                                    &cpu_event);
      if (iree_status_is_ok(status)) {
        // If there is no cpu event the semaphore has hit the value already.
        if (cpu_event) {
          status = iree_wait_one(&cpu_event->event, deadline_ns);
          iree_hal_resource_release(&cpu_event->resource);
        }
      }
      iree_slim_mutex_lock(&semaphore->mutex);
    }
  }

  if (iree_status_is_ok(status)) {
    if (semaphore->current_visible_value >= IREE_HAL_SEMAPHORE_FAILURE_VALUE) {
      status =
          iree_make_status(IREE_STATUS_ABORTED, "the semaphore was aborted");
    }
  }
  iree_slim_mutex_unlock(&semaphore->mutex);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_hip_semaphore_import_timepoint(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    iree_hal_queue_affinity_t queue_affinity,
    iree_hal_external_timepoint_t external_timepoint) {
  if (external_timepoint.type != IREE_HAL_EXTERNAL_TIMEPOINT_TYPE_HIP_EVENT) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid timepoint type for HIP semaphore");
  }

  if ((external_timepoint.compatibility &
       IREE_HAL_SEMAPHORE_COMPATIBILITY_DEVICE_WAIT) == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "HIP only supports the import of DEVICE_WAIT timepoints");
  }

  int device_ordinal = iree_math_count_trailing_zeros_u64(queue_affinity);

  iree_hal_hip_semaphore_t* semaphore =
      iree_hal_hip_semaphore_cast(base_semaphore);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_slim_mutex_lock(&semaphore->mutex);
  iree_hal_hip_util_tree_node_t* node;
  iree_status_t status =
      iree_hal_hip_util_tree_insert(&semaphore->event_queue.tree, value, &node);
  if (iree_status_is_ok(status)) {
    iree_hal_hip_semaphore_queue_item_t* item =
        (iree_hal_hip_semaphore_queue_item_t*)
            iree_hal_hip_util_tree_node_get_value(node);
    item->cpu_event = NULL;
    item->work_item = NULL;
    item->type = IREE_HAL_HIP_SEMAPHORE_QUEUE_ITEM_IMPORTED;
    item->created_device = &semaphore->devices.devices[device_ordinal];
    status = iree_hal_hip_event_pool_import(
        semaphore->devices.devices[device_ordinal].device_event_pool,
        external_timepoint.handle.hip_event, &item->event);
  }

  uint64_t current_value = semaphore->max_value_to_be_signaled;
  iree_slim_mutex_unlock(&semaphore->mutex);
  iree_hal_hip_semaphore_notify_forward_progress_to(base_semaphore,
                                                    current_value);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_hip_semaphore_export_timepoint(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    iree_hal_queue_affinity_t queue_affinity,
    iree_hal_external_timepoint_type_t requested_type,
    iree_hal_external_timepoint_flags_t requested_flags,
    iree_hal_external_timepoint_t* IREE_RESTRICT out_external_timepoint) {
  memset(out_external_timepoint, 0x00, sizeof(*out_external_timepoint));
  if ((requested_type & IREE_HAL_EXTERNAL_TIMEPOINT_TYPE_HIP_EVENT) == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "HIP only supports the export of DEVICE_WAIT timepoints");
  }

  int device_ordinal = iree_math_count_trailing_zeros_u64(queue_affinity);
  iree_hal_hip_semaphore_t* semaphore =
      iree_hal_hip_semaphore_cast(base_semaphore);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_slim_mutex_lock(&semaphore->mutex);
  iree_hal_hip_util_tree_node_t* node;
  iree_status_t status =
      iree_hal_hip_util_tree_insert(&semaphore->event_queue.tree, value, &node);
  if (iree_status_is_ok(status)) {
    iree_hal_hip_semaphore_queue_item_t* item =
        (iree_hal_hip_semaphore_queue_item_t*)
            iree_hal_hip_util_tree_node_get_value(node);
    item->cpu_event = NULL;
    item->work_item = NULL;
    item->type = IREE_HAL_HIP_SEMAPHORE_QUEUE_ITEM_EXPORTED;
    item->created_device = &semaphore->devices.devices[device_ordinal];
    item->is_external = true;
    status = iree_hal_hip_event_pool_acquire(
        semaphore->devices.devices[device_ordinal].device_event_pool, 1,
        &item->event);
    if (iree_status_is_ok(status)) {
      status = iree_hal_hip_event_export(item->event);
    }
    if (iree_status_is_ok(status)) {
      out_external_timepoint->compatibility =
          IREE_HAL_SEMAPHORE_COMPATIBILITY_HOST_WAIT |
          IREE_HAL_SEMAPHORE_COMPATIBILITY_DEVICE_WAIT;
      out_external_timepoint->flags = requested_flags;
      out_external_timepoint->handle.hip_event =
          iree_hal_hip_event_handle(item->event);
      out_external_timepoint->type = IREE_HAL_EXTERNAL_TIMEPOINT_TYPE_HIP_EVENT;
    }
  }
  iree_slim_mutex_unlock(&semaphore->mutex);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

typedef struct iree_hal_hip_semaphore_external_timepoint_wait_data_t {
  iree_hal_hip_semaphore_t* semaphore;
  uint64_t value;
} iree_hal_hip_semaphore_external_timepoint_wait_data_t;

bool iree_hal_hip_semaphore_timepoint_already_exported(
    iree_hal_hip_semaphore_external_timepoint_wait_data_t* data) {
  iree_slim_mutex_lock(&data->semaphore->mutex);
  bool ret = data->semaphore->max_value_to_be_signaled >= data->value;
  iree_slim_mutex_unlock(&data->semaphore->mutex);
  return ret;
}

iree_status_t iree_hal_hip_semaphore_for_exported_timepoints(
    iree_hal_semaphore_t* base_semaphore, uint64_t value) {
  int64_t value_to_wait_for = 0;
  iree_hal_hip_semaphore_t* semaphore =
      iree_hal_hip_semaphore_cast(base_semaphore);
  iree_slim_mutex_lock(&semaphore->mutex);
  for (iree_hal_hip_util_tree_node_t* i =
           iree_hal_hip_util_tree_first(&semaphore->event_queue.tree);
       i != NULL; i = iree_hal_hip_util_tree_node_next(i)) {
    if (iree_hal_hip_util_tree_node_get_key(i) > value) {
      break;
    }
    iree_hal_hip_semaphore_queue_item_t* item =
        (iree_hal_hip_semaphore_queue_item_t*)
            iree_hal_hip_util_tree_node_get_value(i);
    if (!item->is_external) {
      continue;
    }
    value_to_wait_for = iree_hal_hip_util_tree_node_get_key(i);
  }
  bool do_wait = value_to_wait_for > semaphore->max_value_to_be_signaled;
  iree_slim_mutex_unlock(&semaphore->mutex);
  // If we have any exported timepoints, we want to block this
  // thread until they at LEAST have been recorded. This is because
  // once we return it is legal for any other operation to then submit
  // the exported event, which because of hip ordering must occur
  // after the record.
  if (do_wait) {
    IREE_TRACE_ZONE_BEGIN(z0);
    iree_hal_hip_semaphore_external_timepoint_wait_data_t dat = {
        .semaphore = semaphore, .value = value_to_wait_for};
    iree_notification_await(
        &semaphore->external_event_notification,
        (iree_condition_fn_t)iree_hal_hip_semaphore_timepoint_already_exported,
        &dat, iree_infinite_timeout());
    IREE_TRACE_ZONE_END(z0);
  }
  return iree_ok_status();
}

static const iree_hal_semaphore_vtable_t iree_hal_hip_semaphore_vtable = {
    .destroy = iree_hal_hip_semaphore_destroy,
    .query = iree_hal_hip_semaphore_query,
    .signal = iree_hal_hip_semaphore_signal,
    .fail = iree_hal_hip_semaphore_fail,
    .wait = iree_hal_hip_semaphore_wait,
    .import_timepoint = iree_hal_hip_semaphore_import_timepoint,
    .export_timepoint = iree_hal_hip_semaphore_export_timepoint,
};
