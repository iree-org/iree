// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/utils/deferred_work_queue.h"

#include <stdbool.h>
#include <stddef.h>

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/base/internal/synchronization.h"
#include "iree/base/internal/threading.h"
#include "iree/hal/api.h"
#include "iree/hal/utils/deferred_command_buffer.h"
#include "iree/hal/utils/resource_set.h"

// The maximal number of events a command buffer can wait on.
#define IREE_HAL_MAX_WAIT_EVENT_COUNT 32

//===----------------------------------------------------------------------===//
// Queue action
//===----------------------------------------------------------------------===//

typedef enum iree_hal_deferred_work_queue_action_kind_e {
  IREE_HAL_QUEUE_ACTION_TYPE_EXECUTION,
  // TODO: Add support for queue alloca and dealloca.
} iree_hal_deferred_work_queue_action_kind_t;

typedef enum iree_hal_deferred_work_queue_action_state_e {
  // The current action is active as waiting for or under execution.
  IREE_HAL_QUEUE_ACTION_STATE_ALIVE,
} iree_hal_deferred_work_queue_action_state_t;

// A work queue action.
// Note that this struct does not have internal synchronization; it's expected
// to work together with the deferred work queue, which synchronizes accesses.
typedef struct iree_hal_deferred_work_queue_action_t {
  // Intrusive doubly-linked list next entry pointer.
  struct iree_hal_deferred_work_queue_action_t* next;
  // Intrusive doubly-linked list previous entry pointer.
  struct iree_hal_deferred_work_queue_action_t* prev;

  // The owning deferred work queue. We use its allocators and pools.
  // Retained to make sure it outlives the current action.
  iree_hal_deferred_work_queue_t* owning_actions;

  // The current state of this action. When an action is initially created it
  // will be alive and enqueued to wait for releasing to the GPU. After done
  // execution, it will be flipped into zombie state and enqueued again for
  // destruction.
  iree_hal_deferred_work_queue_action_state_t state;
  // The callback to run after completing this action and before freeing
  // all resources. Can be NULL.
  iree_hal_deferred_work_queue_cleanup_callback_t cleanup_callback;
  // User data to pass into the callback.
  void* callback_user_data;

  iree_hal_deferred_work_queue_device_interface_t* device_interface;

  iree_hal_deferred_work_queue_action_kind_t kind;
  union {
    struct {
      iree_host_size_t count;
      iree_hal_command_buffer_t** command_buffers;
      iree_hal_buffer_binding_table_t* binding_tables;
    } execution;
  } payload;

  // Resource set to retain all associated resources by the payload.
  iree_hal_resource_set_t* resource_set;

  // Semaphore list to wait on for the payload to start on the GPU.
  iree_hal_semaphore_list_t wait_semaphore_list;
  // Semaphore list to signal after the payload completes on the GPU.
  iree_hal_semaphore_list_t signal_semaphore_list;

  // Scratch fields for analyzing whether actions are ready to issue.
  iree_hal_deferred_work_queue_host_device_event_t
      wait_events[IREE_HAL_MAX_WAIT_EVENT_COUNT];
  iree_host_size_t event_count;
  // Whether the current action is still not ready for releasing to the GPU.
  bool is_pending;
} iree_hal_deferred_work_queue_action_t;

static void iree_hal_deferred_work_queue_action_fail_locked(
    iree_hal_deferred_work_queue_action_t* action, iree_status_t status);

static void iree_hal_deferred_work_queue_action_clear_events(
    iree_hal_deferred_work_queue_action_t* action) {
  for (iree_host_size_t i = 0; i < action->event_count; ++i) {
    action->device_interface->vtable->release_wait_event(
        action->device_interface, action->wait_events[i]);
  }
  action->event_count = 0;
}

static void iree_hal_deferred_work_queue_action_destroy(
    iree_hal_deferred_work_queue_action_t* action);

//===----------------------------------------------------------------------===//
// Queue action list
//===----------------------------------------------------------------------===//

typedef struct iree_hal_deferred_work_queue_action_list_t {
  iree_hal_deferred_work_queue_action_t* head;
  iree_hal_deferred_work_queue_action_t* tail;
} iree_hal_deferred_work_queue_action_list_t;

// Returns true if the action list is empty.
static inline bool iree_hal_deferred_work_queue_action_list_is_empty(
    const iree_hal_deferred_work_queue_action_list_t* list) {
  return list->head == NULL;
}

static iree_hal_deferred_work_queue_action_t*
iree_hal_deferred_work_queue_action_list_pop_front(
    iree_hal_deferred_work_queue_action_list_t* list) {
  IREE_ASSERT(list->head && list->tail);

  iree_hal_deferred_work_queue_action_t* action = list->head;
  IREE_ASSERT(!action->prev);
  list->head = action->next;
  if (action->next) {
    action->next->prev = NULL;
    action->next = NULL;
  }
  if (list->tail == action) {
    list->tail = NULL;
  }

  return action;
}

// Pushes |action| on to the end of the given action |list|.
static void iree_hal_deferred_work_queue_action_list_push_back(
    iree_hal_deferred_work_queue_action_list_t* list,
    iree_hal_deferred_work_queue_action_t* action) {
  IREE_ASSERT(!action->next && !action->prev);
  if (list->tail) {
    list->tail->next = action;
  } else {
    list->head = action;
  }
  action->prev = list->tail;
  list->tail = action;
}

// Takes all actions from |available_list| and moves them into |ready_list|.
static void iree_hal_deferred_work_queue_action_list_take_all(
    iree_hal_deferred_work_queue_action_list_t* available_list,
    iree_hal_deferred_work_queue_action_list_t* ready_list) {
  IREE_ASSERT_NE(available_list, ready_list);
  ready_list->head = available_list->head;
  ready_list->tail = available_list->tail;
  available_list->head = NULL;
  available_list->tail = NULL;
}

static void iree_hal_deferred_work_queue_action_list_destroy(
    iree_hal_deferred_work_queue_action_t* head_action) {
  while (head_action) {
    iree_hal_deferred_work_queue_action_t* next_action = head_action->next;
    iree_hal_deferred_work_queue_action_destroy(head_action);
    head_action = next_action;
  }
}

//===----------------------------------------------------------------------===//
// Ready-list processing
//===----------------------------------------------------------------------===//

// Ready action entry struct.
typedef struct iree_hal_deferred_work_queue_entry_list_node_t {
  iree_hal_deferred_work_queue_action_t* ready_list_head;
  struct iree_hal_deferred_work_queue_entry_list_node_t* next;
} iree_hal_deferred_work_queue_entry_list_node_t;

typedef struct iree_hal_deferred_work_queue_entry_list_t {
  iree_slim_mutex_t guard_mutex;

  iree_hal_deferred_work_queue_entry_list_node_t* head
      IREE_GUARDED_BY(guard_mutex);
  iree_hal_deferred_work_queue_entry_list_node_t* tail
      IREE_GUARDED_BY(guard_mutex);
} iree_hal_deferred_work_queue_entry_list_t;

static iree_hal_deferred_work_queue_entry_list_node_t*
iree_hal_deferred_work_queue_entry_list_pop(
    iree_hal_deferred_work_queue_entry_list_t* list) {
  iree_hal_deferred_work_queue_entry_list_node_t* out = NULL;
  iree_slim_mutex_lock(&list->guard_mutex);
  if (list->head) {
    out = list->head;
    list->head = list->head->next;
    if (out == list->tail) {
      list->tail = NULL;
    }
  }
  iree_slim_mutex_unlock(&list->guard_mutex);
  return out;
}

void iree_hal_deferred_work_queue_entry_list_push(
    iree_hal_deferred_work_queue_entry_list_t* list,
    iree_hal_deferred_work_queue_entry_list_node_t* next) {
  iree_slim_mutex_lock(&list->guard_mutex);
  next->next = NULL;
  if (list->tail) {
    list->tail->next = next;
    list->tail = next;
  } else {
    list->head = next;
    list->tail = next;
  }
  iree_slim_mutex_unlock(&list->guard_mutex);
}

static void iree_hal_deferred_work_queue_ready_action_list_deinitialize(
    iree_hal_deferred_work_queue_entry_list_t* list,
    iree_allocator_t host_allocator) {
  iree_hal_deferred_work_queue_entry_list_node_t* head = list->head;
  while (head) {
    if (!head) break;
    iree_hal_deferred_work_queue_action_list_destroy(head->ready_list_head);
    list->head = head->next;
    iree_allocator_free(host_allocator, head);
  }
  iree_slim_mutex_deinitialize(&list->guard_mutex);
}

static void iree_hal_deferred_work_queue_ready_action_list_initialize(
    iree_hal_deferred_work_queue_entry_list_t* list) {
  list->head = NULL;
  list->tail = NULL;
  iree_slim_mutex_initialize(&list->guard_mutex);
}

// Ready action entry struct.
typedef struct iree_hal_deferred_work_queue_completion_list_node_t {
  // The callback and user data for that callback. To be called
  // when the associated event has completed.
  iree_status_t (*callback)(iree_status_t, void* user_data);
  void* user_data;
  // The event to wait for on the completion thread.
  iree_hal_deferred_work_queue_native_event_t native_event;
  // If this event was created just for the completion thread, and therefore
  // needs to be cleaned up.
  bool created_event;
  struct iree_hal_deferred_work_queue_completion_list_node_t* next;
} iree_hal_deferred_work_queue_completion_list_node_t;

typedef struct iree_hal_deferred_work_queue_completion_list_t {
  iree_slim_mutex_t guard_mutex;
  iree_hal_deferred_work_queue_completion_list_node_t* head
      IREE_GUARDED_BY(guard_mutex);
  iree_hal_deferred_work_queue_completion_list_node_t* tail
      IREE_GUARDED_BY(guard_mutex);
} iree_hal_deferred_work_queue_completion_list_t;

static iree_hal_deferred_work_queue_completion_list_node_t*
iree_hal_deferred_work_queue_completion_list_pop(
    iree_hal_deferred_work_queue_completion_list_t* list) {
  iree_hal_deferred_work_queue_completion_list_node_t* out = NULL;
  iree_slim_mutex_lock(&list->guard_mutex);
  if (list->head) {
    out = list->head;
    list->head = list->head->next;
    if (out == list->tail) {
      list->tail = NULL;
    }
  }
  iree_slim_mutex_unlock(&list->guard_mutex);
  return out;
}

void iree_hal_deferred_work_queue_completion_list_push(
    iree_hal_deferred_work_queue_completion_list_t* list,
    iree_hal_deferred_work_queue_completion_list_node_t* next) {
  iree_slim_mutex_lock(&list->guard_mutex);
  next->next = NULL;
  if (list->tail) {
    list->tail->next = next;
    list->tail = next;
  } else {
    list->head = next;
    list->tail = next;
  }
  iree_slim_mutex_unlock(&list->guard_mutex);
}

static void iree_hal_deferred_work_queue_completion_list_initialize(
    iree_hal_deferred_work_queue_completion_list_t* list) {
  list->head = NULL;
  list->tail = NULL;
  iree_slim_mutex_initialize(&list->guard_mutex);
}

static void iree_hal_deferred_work_queue_completion_list_deinitialize(
    iree_hal_deferred_work_queue_completion_list_t* list,
    iree_hal_deferred_work_queue_device_interface_t* device_interface,
    iree_allocator_t host_allocator) {
  iree_hal_deferred_work_queue_completion_list_node_t* head = list->head;
  while (head) {
    if (head->created_event) {
      device_interface->vtable->destroy_native_event(device_interface,
                                                     head->native_event);
    }
    list->head = list->head->next;
    iree_allocator_free(host_allocator, head);
  }
  iree_slim_mutex_deinitialize(&list->guard_mutex);
}

static iree_hal_deferred_work_queue_action_t*
iree_hal_deferred_work_queue_entry_list_node_pop_front(
    iree_hal_deferred_work_queue_entry_list_node_t* list) {
  IREE_ASSERT(list->ready_list_head);

  iree_hal_deferred_work_queue_action_t* action = list->ready_list_head;
  IREE_ASSERT(!action->prev);
  list->ready_list_head = action->next;
  if (action->next) {
    action->next->prev = NULL;
    action->next = NULL;
  }

  return action;
}

static void iree_hal_deferred_work_queue_entry_list_node_push_front(
    iree_hal_deferred_work_queue_entry_list_node_t* entry,
    iree_hal_deferred_work_queue_action_t* action) {
  IREE_ASSERT(!action->next && !action->prev);

  iree_hal_deferred_work_queue_action_t* head = entry->ready_list_head;
  entry->ready_list_head = action;
  if (head) {
    action->next = head;
    head->prev = action;
  }
}

// The ready-list processing worker's working/exiting state.
//
// States in the list has increasing priorities--meaning normally ones appearing
// earlier can overwrite ones appearing later without checking; but not the
// reverse order.
typedef enum iree_hal_deferred_work_queue_worker_state_e {
  IREE_HAL_WORKER_STATE_IDLE_WAITING = 0,      // Worker to any thread
  IREE_HAL_WORKER_STATE_WORKLOAD_PENDING = 1,  // Any to worker thread
} iree_hal_deferred_work_queue_worker_state_t;

// The data structure needed by a ready-list processing worker thread to issue
// ready actions to the GPU.
//
// This data structure is shared between the parent thread, which owns the
// whole deferred work queue, and the worker thread; so proper synchronization
// is needed to touch it from both sides.
//
// The parent thread should push a list of ready actions to ready_worklist,
// update worker_state, and give state_notification accordingly.
// The worker thread waits on the state_notification and checks worker_state,
// and pops from the ready_worklist to process. The worker thread also monitors
// worker_state and stops processing if requested by the parent thread.
typedef struct iree_hal_deferred_work_queue_working_area_t {
  // Notification from the parent thread to request worker state changes.
  iree_notification_t state_notification;
  iree_hal_deferred_work_queue_entry_list_t ready_worklist;  // atomic
  iree_atomic_int32_t worker_state;                          // atomic
} iree_hal_deferred_work_queue_working_area_t;

// This data structure is shared by the parent thread. It is responsible
// for dispatching callbacks when work items complete.

// This replaces the use of Launch Host Function APIs, which cause
// streams to block and wait for the CPU work to complete.
// It also picks up completed events with significantly less latency than
// Launch Host Function APIs.
typedef struct iree_hal_deferred_work_queue_completion_area_t {
  // Notification from the parent thread to request completion state changes.
  iree_notification_t state_notification;
  iree_hal_deferred_work_queue_completion_list_t completion_list;  // atomic
  iree_atomic_int32_t worker_state;                                // atomic
} iree_hal_deferred_work_queue_completion_area_t;

static void iree_hal_deferred_work_queue_working_area_initialize(
    iree_allocator_t host_allocator,
    const iree_hal_deferred_work_queue_device_interface_t* device_interface,
    iree_hal_deferred_work_queue_working_area_t* working_area) {
  iree_notification_initialize(&working_area->state_notification);
  iree_hal_deferred_work_queue_ready_action_list_deinitialize(
      &working_area->ready_worklist, host_allocator);
  iree_atomic_store_int32(&working_area->worker_state,
                          IREE_HAL_WORKER_STATE_IDLE_WAITING,
                          iree_memory_order_release);
}

static void iree_hal_deferred_work_queue_working_area_deinitialize(
    iree_hal_deferred_work_queue_working_area_t* working_area,
    iree_allocator_t host_allocator) {
  iree_hal_deferred_work_queue_ready_action_list_deinitialize(
      &working_area->ready_worklist, host_allocator);
  iree_notification_deinitialize(&working_area->state_notification);
}

static void iree_hal_deferred_work_queue_completion_area_initialize(
    iree_allocator_t host_allocator,
    iree_hal_deferred_work_queue_device_interface_t* device_interface,
    iree_hal_deferred_work_queue_completion_area_t* completion_area) {
  iree_notification_initialize(&completion_area->state_notification);
  iree_hal_deferred_work_queue_completion_list_initialize(
      &completion_area->completion_list);
  iree_atomic_store_int32(&completion_area->worker_state,
                          IREE_HAL_WORKER_STATE_IDLE_WAITING,
                          iree_memory_order_release);
}

static void iree_hal_deferred_work_queue_completion_area_deinitialize(
    iree_hal_deferred_work_queue_completion_area_t* completion_area,
    iree_hal_deferred_work_queue_device_interface_t* device_interface,
    iree_allocator_t host_allocator) {
  iree_hal_deferred_work_queue_completion_list_deinitialize(
      &completion_area->completion_list, device_interface, host_allocator);
  iree_notification_deinitialize(&completion_area->state_notification);
}

// The main function for the ready-list processing worker thread.
static int iree_hal_deferred_work_queue_worker_execute(
    iree_hal_deferred_work_queue_t* actions);

static int iree_hal_deferred_work_queue_completion_execute(
    iree_hal_deferred_work_queue_t* actions);

//===----------------------------------------------------------------------===//
// Deferred work queue
//===----------------------------------------------------------------------===//

struct iree_hal_deferred_work_queue_t {
  // Abstract resource used for injecting reference counting and vtable;
  // must be at offset 0.
  iree_hal_resource_t resource;

  // The allocator used to create the timepoint pool.
  iree_allocator_t host_allocator;
  // The block pool to allocate resource sets from.
  iree_arena_block_pool_t* block_pool;

  // The device interface used to interact with the native driver.
  iree_hal_deferred_work_queue_device_interface_t* device_interface;

  // Non-recursive mutex guarding access.
  iree_slim_mutex_t action_mutex;

  // The double-linked list of deferred work.
  iree_hal_deferred_work_queue_action_list_t action_list
      IREE_GUARDED_BY(action_mutex);

  // The worker thread that monitors incoming requests and issues ready actions
  // to the GPU.
  iree_thread_t* worker_thread;

  // Worker thread to wait on completion events instead of running
  // synchronous completion callbacks
  iree_thread_t* completion_thread;

  // The worker's working area; data exchange place with the parent thread.
  iree_hal_deferred_work_queue_working_area_t working_area;

  // Completion thread's working area.
  iree_hal_deferred_work_queue_completion_area_t completion_area;

  // Atomic of type iree_status_t. It is a sticky error.
  // Once set with an error, all subsequent actions that have not completed
  // will fail with this error.
  iree_status_t status IREE_GUARDED_BY(action_mutex);

  // The number of asynchronous work items that are scheduled and not
  // complete.
  // These are
  // * the number of actions issued.
  // * the number of pending action cleanups.
  // The work and completion threads can exit only when there are no more
  // pending work items.
  iree_host_size_t pending_work_items_count IREE_GUARDED_BY(action_mutex);

  // The owner can request an exit of the worker threads.
  // Once all pending enqueued work is complete the threads will exit.
  // No actions can be enqueued after requesting an exit.
  bool exit_requested IREE_GUARDED_BY(action_mutex);
};

iree_status_t iree_hal_deferred_work_queue_create(
    iree_hal_deferred_work_queue_device_interface_t* device_interface,
    iree_arena_block_pool_t* block_pool, iree_allocator_t host_allocator,
    iree_hal_deferred_work_queue_t** out_actions) {
  IREE_ASSERT_ARGUMENT(device_interface);
  IREE_ASSERT_ARGUMENT(block_pool);
  IREE_ASSERT_ARGUMENT(out_actions);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_deferred_work_queue_t* actions = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*actions),
                                (void**)&actions));
  actions->host_allocator = host_allocator;
  actions->block_pool = block_pool;
  actions->device_interface = device_interface;

  iree_slim_mutex_initialize(&actions->action_mutex);
  memset(&actions->action_list, 0, sizeof(actions->action_list));

  // Initialize the working area for the ready-list processing worker.
  iree_hal_deferred_work_queue_working_area_t* working_area =
      &actions->working_area;
  iree_hal_deferred_work_queue_working_area_initialize(
      host_allocator, device_interface, working_area);

  iree_hal_deferred_work_queue_completion_area_t* completion_area =
      &actions->completion_area;
  iree_hal_deferred_work_queue_completion_area_initialize(
      host_allocator, device_interface, completion_area);

  // Create the ready-list processing worker itself.
  iree_thread_create_params_t params;
  memset(&params, 0, sizeof(params));
  params.name = IREE_SV("iree-hip-queue-worker");
  params.create_suspended = false;
  iree_status_t status = iree_thread_create(
      (iree_thread_entry_t)iree_hal_deferred_work_queue_worker_execute, actions,
      params, actions->host_allocator, &actions->worker_thread);

  params.name = IREE_SV("iree-hip-queue-completion");
  params.create_suspended = false;
  if (iree_status_is_ok(status)) {
    status = iree_thread_create(
        (iree_thread_entry_t)iree_hal_deferred_work_queue_completion_execute,
        actions, params, actions->host_allocator, &actions->completion_thread);
  }

  if (iree_status_is_ok(status)) {
    *out_actions = actions;
  } else {
    iree_hal_deferred_work_queue_destroy(actions);
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_hal_deferred_work_queue_t* iree_hal_deferred_work_queue_cast(
    iree_hal_resource_t* base_value) {
  return (iree_hal_deferred_work_queue_t*)base_value;
}

static void iree_hal_deferred_work_queue_notify_worker_thread(
    iree_hal_deferred_work_queue_working_area_t* working_area) {
  iree_atomic_store_int32(&working_area->worker_state,
                          IREE_HAL_WORKER_STATE_WORKLOAD_PENDING,
                          iree_memory_order_release);
  iree_notification_post(&working_area->state_notification, IREE_ALL_WAITERS);
}

static void iree_hal_deferred_work_queue_notify_completion_thread(
    iree_hal_deferred_work_queue_completion_area_t* completion_area) {
  iree_atomic_store_int32(&completion_area->worker_state,
                          IREE_HAL_WORKER_STATE_WORKLOAD_PENDING,
                          iree_memory_order_release);
  iree_notification_post(&completion_area->state_notification,
                         IREE_ALL_WAITERS);
}

// Notifies worker and completion threads that there is work available to
// process.
static void iree_hal_deferred_work_queue_notify_threads(
    iree_hal_deferred_work_queue_t* actions) {
  iree_hal_deferred_work_queue_notify_worker_thread(&actions->working_area);
  iree_hal_deferred_work_queue_notify_completion_thread(
      &actions->completion_area);
}

static void iree_hal_deferred_work_queue_request_exit(
    iree_hal_deferred_work_queue_t* actions) {
  iree_slim_mutex_lock(&actions->action_mutex);
  actions->exit_requested = true;
  iree_slim_mutex_unlock(&actions->action_mutex);

  iree_hal_deferred_work_queue_notify_threads(actions);
}

void iree_hal_deferred_work_queue_destroy(
    iree_hal_deferred_work_queue_t* work_queue) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_allocator_t host_allocator = work_queue->host_allocator;

  // Request the workers to exit.
  iree_hal_deferred_work_queue_request_exit(work_queue);

  iree_thread_release(work_queue->worker_thread);
  iree_thread_release(work_queue->completion_thread);

  iree_hal_deferred_work_queue_working_area_deinitialize(
      &work_queue->working_area, work_queue->host_allocator);
  iree_hal_deferred_work_queue_completion_area_deinitialize(
      &work_queue->completion_area, work_queue->device_interface,
      work_queue->host_allocator);

  iree_slim_mutex_deinitialize(&work_queue->action_mutex);
  iree_hal_deferred_work_queue_action_list_destroy(
      work_queue->action_list.head);

  work_queue->device_interface->vtable->destroy(work_queue->device_interface);
  iree_allocator_free(host_allocator, work_queue);

  IREE_TRACE_ZONE_END(z0);
}

static void iree_hal_deferred_work_queue_action_destroy(
    iree_hal_deferred_work_queue_action_t* action) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_deferred_work_queue_t* actions = action->owning_actions;
  iree_allocator_t host_allocator = actions->host_allocator;

  // Call user provided callback before releasing any resource.
  if (action->cleanup_callback) {
    action->cleanup_callback(action->callback_user_data);
  }

  // Only release resources after callbacks have been issued.
  iree_hal_resource_set_free(action->resource_set);

  iree_hal_deferred_work_queue_action_clear_events(action);

  iree_hal_resource_release(actions);

  iree_allocator_free(host_allocator, action);

  IREE_TRACE_ZONE_END(z0);
}

static void iree_hal_deferred_work_queue_decrement_work_items_count(
    iree_hal_deferred_work_queue_t* actions) {
  iree_slim_mutex_lock(&actions->action_mutex);
  --actions->pending_work_items_count;
  iree_slim_mutex_unlock(&actions->action_mutex);
}

iree_status_t iree_hal_deferred_work_queue_enqueue(
    iree_hal_deferred_work_queue_t* actions,
    iree_hal_deferred_work_queue_cleanup_callback_t cleanup_callback,
    void* callback_user_data,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_host_size_t command_buffer_count,
    iree_hal_command_buffer_t* const* command_buffers,
    iree_hal_buffer_binding_table_t const* binding_tables) {
  IREE_ASSERT_ARGUMENT(actions);
  IREE_ASSERT_ARGUMENT(command_buffer_count == 0 || command_buffers);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Embed captured tables in the action allocation.
  iree_hal_deferred_work_queue_action_t* action = NULL;
  const iree_host_size_t wait_semaphore_list_size =
      wait_semaphore_list.count * sizeof(*wait_semaphore_list.semaphores) +
      wait_semaphore_list.count * sizeof(*wait_semaphore_list.payload_values);
  const iree_host_size_t signal_semaphore_list_size =
      signal_semaphore_list.count * sizeof(*signal_semaphore_list.semaphores) +
      signal_semaphore_list.count *
          sizeof(*signal_semaphore_list.payload_values);
  const iree_host_size_t command_buffers_size =
      command_buffer_count * sizeof(*action->payload.execution.command_buffers);
  iree_host_size_t binding_tables_size = 0;
  iree_host_size_t binding_table_elements_size = 0;
  if (binding_tables) {
    binding_tables_size = command_buffer_count * sizeof(*binding_tables);
    for (iree_host_size_t i = 0; i < command_buffer_count; ++i) {
      binding_table_elements_size +=
          binding_tables[i].count * sizeof(*binding_tables[i].bindings);
    }
  }
  const iree_host_size_t payload_size =
      command_buffers_size + binding_tables_size + binding_table_elements_size;
  const iree_host_size_t total_action_size =
      sizeof(*action) + wait_semaphore_list_size + signal_semaphore_list_size +
      payload_size;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(actions->host_allocator, total_action_size,
                                (void**)&action));
  uint8_t* action_ptr = (uint8_t*)action + sizeof(*action);

  action->owning_actions = actions;
  action->device_interface = actions->device_interface;
  action->state = IREE_HAL_QUEUE_ACTION_STATE_ALIVE;
  action->cleanup_callback = cleanup_callback;
  action->callback_user_data = callback_user_data;
  action->kind = IREE_HAL_QUEUE_ACTION_TYPE_EXECUTION;

  // Initialize scratch fields.
  action->event_count = 0;
  action->is_pending = true;

  // Copy wait list for later access.
  action->wait_semaphore_list.count = wait_semaphore_list.count;
  action->wait_semaphore_list.semaphores = (iree_hal_semaphore_t**)action_ptr;
  memcpy(action->wait_semaphore_list.semaphores, wait_semaphore_list.semaphores,
         wait_semaphore_list.count * sizeof(*wait_semaphore_list.semaphores));
  action->wait_semaphore_list.payload_values =
      (uint64_t*)(action_ptr + wait_semaphore_list.count *
                                   sizeof(*wait_semaphore_list.semaphores));
  memcpy(
      action->wait_semaphore_list.payload_values,
      wait_semaphore_list.payload_values,
      wait_semaphore_list.count * sizeof(*wait_semaphore_list.payload_values));
  action_ptr += wait_semaphore_list_size;

  // Copy signal list for later access.
  action->signal_semaphore_list.count = signal_semaphore_list.count;
  action->signal_semaphore_list.semaphores = (iree_hal_semaphore_t**)action_ptr;
  memcpy(
      action->signal_semaphore_list.semaphores,
      signal_semaphore_list.semaphores,
      signal_semaphore_list.count * sizeof(*signal_semaphore_list.semaphores));
  action->signal_semaphore_list.payload_values =
      (uint64_t*)(action_ptr + signal_semaphore_list.count *
                                   sizeof(*signal_semaphore_list.semaphores));
  memcpy(action->signal_semaphore_list.payload_values,
         signal_semaphore_list.payload_values,
         signal_semaphore_list.count *
             sizeof(*signal_semaphore_list.payload_values));
  action_ptr += signal_semaphore_list_size;

  // Copy the execution resources for later access.
  action->payload.execution.count = command_buffer_count;
  action->payload.execution.command_buffers =
      (iree_hal_command_buffer_t**)action_ptr;
  memcpy(action->payload.execution.command_buffers, command_buffers,
         command_buffers_size);
  action_ptr += command_buffers_size;

  // Retain all command buffers and semaphores.
  iree_status_t status = iree_hal_resource_set_allocate(actions->block_pool,
                                                        &action->resource_set);
  if (iree_status_is_ok(status)) {
    status = iree_hal_resource_set_insert(action->resource_set,
                                          wait_semaphore_list.count,
                                          wait_semaphore_list.semaphores);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_resource_set_insert(action->resource_set,
                                          signal_semaphore_list.count,
                                          signal_semaphore_list.semaphores);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_resource_set_insert(
        action->resource_set, command_buffer_count, command_buffers);
  }

  // Copy binding tables and retain all bindings.
  if (iree_status_is_ok(status) && binding_table_elements_size > 0) {
    action->payload.execution.binding_tables =
        (iree_hal_buffer_binding_table_t*)action_ptr;
    action_ptr += binding_tables_size;
    iree_hal_buffer_binding_t* binding_element_ptr =
        (iree_hal_buffer_binding_t*)action_ptr;
    for (iree_host_size_t i = 0; i < command_buffer_count; ++i) {
      iree_host_size_t element_count = binding_tables[i].count;
      iree_hal_buffer_binding_table_t* target_table =
          &action->payload.execution.binding_tables[i];
      target_table->count = element_count;
      target_table->bindings = binding_element_ptr;
      memcpy((void*)target_table->bindings, binding_tables[i].bindings,
             element_count * sizeof(*binding_element_ptr));
      binding_element_ptr += element_count;

      // Bulk insert all bindings into the resource set. This will keep the
      // referenced buffers live until the action has completed. Note that if we
      // fail here we need to clean up the resource set below before returning.
      status = iree_hal_resource_set_insert_strided(
          action->resource_set, element_count, target_table->bindings,
          offsetof(iree_hal_buffer_binding_t, buffer),
          sizeof(iree_hal_buffer_binding_t));
      if (!iree_status_is_ok(status)) break;
    }
  } else {
    action->payload.execution.binding_tables = NULL;
  }

  if (iree_status_is_ok(status)) {
    // Now everything is okay and we can enqueue the action.
    iree_slim_mutex_lock(&actions->action_mutex);
    if (actions->exit_requested) {
      status = iree_make_status(
          IREE_STATUS_ABORTED,
          "can not issue more executions, exit already requested");
      iree_hal_deferred_work_queue_action_fail_locked(action, status);
    } else {
      iree_hal_deferred_work_queue_action_list_push_back(&actions->action_list,
                                                         action);
      // One work item is the callback that makes it across from the
      // completion thread.
      actions->pending_work_items_count += 1;
    }
    iree_slim_mutex_unlock(&actions->action_mutex);
  } else {
    iree_hal_resource_set_free(action->resource_set);
    iree_allocator_free(actions->host_allocator, action);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Does not consume |status|.
static void iree_hal_deferred_work_queue_fail_status_locked(
    iree_hal_deferred_work_queue_t* actions, iree_status_t status) {
  if (iree_status_is_ok(actions->status) && status != actions->status) {
    actions->status = iree_status_clone(status);
  }
}

// Fails and destroys the action.
// Does not consume |status|.
// Decrements pending work items count accordingly based on the unfulfilled
// number of work items.
static void iree_hal_deferred_work_queue_action_fail_locked(
    iree_hal_deferred_work_queue_action_t* action, iree_status_t status) {
  IREE_ASSERT(!iree_status_is_ok(status));
  iree_hal_deferred_work_queue_t* actions = action->owning_actions;

  // Unlock since failing the semaphore will use |actions|.
  iree_slim_mutex_unlock(&actions->action_mutex);
  iree_hal_semaphore_list_fail(action->signal_semaphore_list,
                               iree_status_clone(status));

  iree_slim_mutex_lock(&actions->action_mutex);
  action->owning_actions->pending_work_items_count -= 1;
  iree_hal_deferred_work_queue_fail_status_locked(actions, status);
  iree_hal_deferred_work_queue_action_destroy(action);
}

// Fails and destroys all actions.
// Does not consume |status|.
static void iree_hal_deferred_work_queue_action_fail(
    iree_hal_deferred_work_queue_action_t* action, iree_status_t status) {
  iree_hal_deferred_work_queue_t* actions = action->owning_actions;
  iree_slim_mutex_lock(&actions->action_mutex);
  iree_hal_deferred_work_queue_action_fail_locked(action, status);
  iree_slim_mutex_unlock(&actions->action_mutex);
}

// Fails and destroys all actions.
// Does not consume |status|.
static void iree_hal_deferred_work_queue_action_raw_list_fail_locked(
    iree_hal_deferred_work_queue_action_t* head_action, iree_status_t status) {
  while (head_action) {
    iree_hal_deferred_work_queue_action_t* next_action = head_action->next;
    iree_hal_deferred_work_queue_action_fail_locked(head_action, status);
    head_action = next_action;
  }
}

// Fails and destroys all actions.
// Does not consume |status|.
static void iree_hal_deferred_work_queue_ready_action_list_fail_locked(
    iree_hal_deferred_work_queue_entry_list_t* list, iree_status_t status) {
  iree_hal_deferred_work_queue_entry_list_node_t* entry =
      iree_hal_deferred_work_queue_entry_list_pop(list);
  while (entry) {
    iree_hal_deferred_work_queue_action_raw_list_fail_locked(
        entry->ready_list_head, status);
    entry = iree_hal_deferred_work_queue_entry_list_pop(list);
  }
}

// Fails and destroys all actions.
// Does not consume |status|.
static void iree_hal_deferred_work_queue_action_list_fail_locked(
    iree_hal_deferred_work_queue_action_list_t* list, iree_status_t status) {
  iree_hal_deferred_work_queue_action_t* action;
  if (iree_hal_deferred_work_queue_action_list_is_empty(list)) {
    return;
  }
  do {
    action = iree_hal_deferred_work_queue_action_list_pop_front(list);
    iree_hal_deferred_work_queue_action_fail_locked(action, status);
  } while (action);
}

// Fails and destroys all actions and sets status of |actions|.
// Does not consume |status|.
// Assumes the caller is holding the action_mutex.
static void iree_hal_deferred_work_queue_fail_locked(
    iree_hal_deferred_work_queue_t* actions, iree_status_t status) {
  iree_hal_deferred_work_queue_fail_status_locked(actions, status);
  iree_hal_deferred_work_queue_action_list_fail_locked(&actions->action_list,
                                                       status);
  iree_hal_deferred_work_queue_ready_action_list_fail_locked(
      &actions->working_area.ready_worklist, status);
}

// Does not consume |status|.
static void iree_hal_deferred_work_queue_fail(
    iree_hal_deferred_work_queue_t* actions, iree_status_t status) {
  iree_slim_mutex_lock(&actions->action_mutex);
  iree_hal_deferred_work_queue_fail_locked(actions, status);
  iree_slim_mutex_unlock(&actions->action_mutex);
}

// Releases resources after action completion on the GPU and advances timeline
// and deferred work queue.
static iree_status_t
iree_hal_deferred_work_queue_execution_device_signal_host_callback(
    iree_status_t status, void* user_data) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_deferred_work_queue_action_t* action =
      (iree_hal_deferred_work_queue_action_t*)user_data;
  IREE_ASSERT_EQ(action->kind, IREE_HAL_QUEUE_ACTION_TYPE_EXECUTION);
  IREE_ASSERT_EQ(action->state, IREE_HAL_QUEUE_ACTION_STATE_ALIVE);
  if (IREE_UNLIKELY(!iree_status_is_ok(status))) {
    iree_hal_deferred_work_queue_action_fail(action, status);
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  // Need to signal the list before zombifying the action, because in the mean
  // time someone else may issue the pending queue actions.
  // If we push first to the deferred work list, the cleanup of this action
  // may run while we are still using the semaphore list, causing a crash.
  status = iree_hal_semaphore_list_signal(action->signal_semaphore_list);
  if (IREE_UNLIKELY(!iree_status_is_ok(status))) {
    iree_hal_deferred_work_queue_action_fail(action, status);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  iree_hal_deferred_work_queue_action_destroy(action);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Issues the given kernel dispatch |action| to the GPU.
static iree_status_t iree_hal_deferred_work_queue_issue_execution(
    iree_hal_deferred_work_queue_action_t* action) {
  IREE_ASSERT_EQ(action->kind, IREE_HAL_QUEUE_ACTION_TYPE_EXECUTION);
  IREE_ASSERT_EQ(action->is_pending, false);
  iree_hal_deferred_work_queue_t* actions = action->owning_actions;
  iree_hal_deferred_work_queue_device_interface_t* device_interface =
      actions->device_interface;
  IREE_TRACE_ZONE_BEGIN(z0);

  // No need to lock given that this action is already detched from the pending
  // actions list; so only this thread is seeing it now.

  // First wait all the device events in the dispatch stream.
  for (iree_host_size_t i = 0; i < action->event_count; ++i) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, device_interface->vtable->device_wait_on_host_event(
                device_interface, action->wait_events[i]));
  }

  // Then launch all command buffers to the dispatch stream.
  IREE_TRACE_ZONE_BEGIN(z_dispatch_command_buffers);
  IREE_TRACE_ZONE_APPEND_TEXT(z_dispatch_command_buffers,
                              "dispatch_command_buffers");

  for (iree_host_size_t i = 0; i < action->payload.execution.count; ++i) {
    iree_hal_command_buffer_t* command_buffer =
        action->payload.execution.command_buffers[i];
    iree_hal_buffer_binding_table_t binding_table =
        action->payload.execution.binding_tables
            ? action->payload.execution.binding_tables[i]
            : iree_hal_buffer_binding_table_empty();
    if (iree_hal_deferred_command_buffer_isa(command_buffer)) {
      iree_hal_command_buffer_t* stream_command_buffer = NULL;
      iree_hal_command_buffer_mode_t mode =
          iree_hal_command_buffer_mode(command_buffer) |
          IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT |
          // NOTE: we need to validate if a binding table is provided as the
          // bindings were not known when it was originally recorded.
          (iree_hal_buffer_binding_table_is_empty(binding_table)
               ? IREE_HAL_COMMAND_BUFFER_MODE_UNVALIDATED
               : 0);
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, device_interface->vtable->create_stream_command_buffer(
                  device_interface, mode, IREE_HAL_COMMAND_CATEGORY_ANY,
                  &stream_command_buffer))
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_hal_resource_set_insert(action->resource_set, 1,
                                           &stream_command_buffer));

      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_hal_deferred_command_buffer_apply(
                  command_buffer, stream_command_buffer, binding_table));
      command_buffer = stream_command_buffer;
    } else {
      iree_hal_resource_retain(command_buffer);
    }

    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, device_interface->vtable->submit_command_buffer(device_interface,
                                                            command_buffer));

    // The stream_command_buffer is going to be retained by
    // the action->resource_set and deleted after the action
    // completes.
    iree_hal_resource_release(command_buffer);
  }

  IREE_TRACE_ZONE_END(z_dispatch_command_buffers);

  iree_hal_deferred_work_queue_native_event_t completion_event = NULL;
  // Last record event signals in the dispatch stream.
  for (iree_host_size_t i = 0; i < action->signal_semaphore_list.count; ++i) {
    // Grab an event for this semaphore value signaling.
    iree_hal_deferred_work_queue_native_event_t event = NULL;

    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0,
        device_interface->vtable
            ->semaphore_acquire_timepoint_device_signal_native_event(
                device_interface, action->signal_semaphore_list.semaphores[i],
                action->signal_semaphore_list.payload_values[i], &event));

    // Record the event signaling in the dispatch stream.
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0,
        device_interface->vtable->record_native_event(device_interface, event));
    completion_event = event;
  }

  bool created_event = false;
  // In the case where we issue an execution and there are signal semaphores
  // we can re-use those as a wait event. However if there are no signals
  // then we create one. In my testing this is not a common case.
  if (IREE_UNLIKELY(!completion_event)) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, device_interface->vtable->create_native_event(device_interface,
                                                          &completion_event));
    created_event = true;
  }

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, device_interface->vtable->record_native_event(device_interface,
                                                        completion_event));

  iree_hal_deferred_work_queue_completion_list_node_t* entry = NULL;
  // TODO: avoid host allocator malloc; use some pool for the allocation.
  iree_status_t status = iree_allocator_malloc(actions->host_allocator,
                                               sizeof(*entry), (void**)&entry);

  if (IREE_UNLIKELY(!iree_status_is_ok(status))) {
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  // Now push the ready list to the worker and have it to issue the actions to
  // the GPU.
  entry->native_event = completion_event;
  entry->created_event = created_event;
  entry->callback =
      iree_hal_deferred_work_queue_execution_device_signal_host_callback;
  entry->user_data = action;
  iree_hal_deferred_work_queue_completion_list_push(
      &actions->completion_area.completion_list, entry);

  iree_hal_deferred_work_queue_notify_completion_thread(
      &actions->completion_area);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_deferred_work_queue_issue(
    iree_hal_deferred_work_queue_t* actions) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_deferred_work_queue_action_list_t pending_list = {NULL, NULL};
  iree_hal_deferred_work_queue_action_list_t ready_list = {NULL, NULL};

  iree_slim_mutex_lock(&actions->action_mutex);

  if (iree_hal_deferred_work_queue_action_list_is_empty(
          &actions->action_list)) {
    iree_slim_mutex_unlock(&actions->action_mutex);
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  if (IREE_UNLIKELY(!iree_status_is_ok(actions->status))) {
    iree_hal_deferred_work_queue_action_list_fail_locked(&actions->action_list,
                                                         actions->status);
    iree_slim_mutex_unlock(&actions->action_mutex);
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  iree_status_t status = iree_ok_status();
  // Scan through the list and categorize actions into pending and ready lists.
  while (!iree_hal_deferred_work_queue_action_list_is_empty(
      &actions->action_list)) {
    iree_hal_deferred_work_queue_action_t* action =
        iree_hal_deferred_work_queue_action_list_pop_front(
            &actions->action_list);

    iree_hal_semaphore_t** semaphores = action->wait_semaphore_list.semaphores;
    uint64_t* values = action->wait_semaphore_list.payload_values;

    action->is_pending = false;
    bool action_failed = false;

    // Cleanup actions are immediately ready to release. Otherwise, look at all
    // wait semaphores to make sure that they are either already ready or we can
    // wait on a device event.
    if (action->state == IREE_HAL_QUEUE_ACTION_STATE_ALIVE) {
      for (iree_host_size_t i = 0; i < action->wait_semaphore_list.count; ++i) {
        // If this semaphore has already signaled past the desired value, we can
        // just ignore it.
        uint64_t value = 0;
        iree_status_t semaphore_status =
            iree_hal_semaphore_query(semaphores[i], &value);
        if (IREE_UNLIKELY(!iree_status_is_ok(semaphore_status))) {
          iree_hal_deferred_work_queue_action_fail_locked(action,
                                                          semaphore_status);
          iree_status_ignore(semaphore_status);
          action_failed = true;
          break;
        }
        if (value >= values[i]) {
          // No need to wait on this timepoint as it has already occurred and
          // we can remove it from the wait list.
          iree_hal_semaphore_list_erase(&action->wait_semaphore_list, i);
          --i;
          continue;
        }

        // Try to acquire an event from an existing device signal timepoint.
        // If so, we can use that event to wait on the device.
        // Otherwise, this action is still not ready for execution.
        // Before issuing recording on a stream, an event represents an empty
        // set of work so waiting on it will just return success.
        // Here we must guarantee the event is indeed recorded, which means
        // it's associated with some already present device signal timepoint on
        // the semaphore timeline.
        iree_hal_deferred_work_queue_host_device_event_t wait_event = NULL;
        if (!action->device_interface->vtable->acquire_host_wait_event(
                action->device_interface, semaphores[i], values[i],
                &wait_event)) {
          action->is_pending = true;
          break;
        }
        if (IREE_UNLIKELY(action->event_count >=
                          IREE_HAL_MAX_WAIT_EVENT_COUNT)) {
          status = iree_make_status(
              IREE_STATUS_RESOURCE_EXHAUSTED,
              "exceeded maximum queue action wait event limit");
          action->device_interface->vtable->release_wait_event(
              action->device_interface, wait_event);
          if (iree_status_is_ok(actions->status)) {
            actions->status = status;
          }
          iree_hal_deferred_work_queue_action_fail_locked(action, status);
          break;
        }
        action->wait_events[action->event_count++] = wait_event;

        // Remove the wait timepoint as we have a corresponding event that we
        // will wait on.
        iree_hal_semaphore_list_erase(&action->wait_semaphore_list, i);
        --i;
      }
    }

    if (IREE_UNLIKELY(!iree_status_is_ok(actions->status))) {
      if (!action_failed) {
        iree_hal_deferred_work_queue_action_fail_locked(action,
                                                        actions->status);
      }
      iree_hal_deferred_work_queue_action_list_fail_locked(
          &actions->action_list, actions->status);
      break;
    }

    if (action_failed) {
      break;
    }

    if (action->is_pending) {
      iree_hal_deferred_work_queue_action_list_push_back(&pending_list, action);
    } else {
      iree_hal_deferred_work_queue_action_list_push_back(&ready_list, action);
    }
  }

  // Preserve pending timepoints.
  actions->action_list = pending_list;

  iree_slim_mutex_unlock(&actions->action_mutex);

  if (ready_list.head == NULL) {
    // Nothing ready yet. Just return.
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  iree_hal_deferred_work_queue_entry_list_node_t* entry = NULL;
  // TODO: avoid host allocator malloc; use some pool for the allocation.
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(actions->host_allocator, sizeof(*entry),
                                   (void**)&entry);
  }

  if (IREE_UNLIKELY(!iree_status_is_ok(status))) {
    iree_slim_mutex_lock(&actions->action_mutex);
    iree_hal_deferred_work_queue_fail_status_locked(actions, status);
    iree_hal_deferred_work_queue_action_list_fail_locked(&ready_list, status);
    iree_slim_mutex_unlock(&actions->action_mutex);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  // Now push the ready list to the worker and have it to issue the actions to
  // the GPU.
  entry->ready_list_head = ready_list.head;
  iree_hal_deferred_work_queue_entry_list_push(
      &actions->working_area.ready_worklist, entry);

  iree_hal_deferred_work_queue_notify_worker_thread(&actions->working_area);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Worker routines
//===----------------------------------------------------------------------===//

static bool iree_hal_deferred_work_queue_worker_has_incoming_request(
    iree_hal_deferred_work_queue_working_area_t* working_area) {
  iree_hal_deferred_work_queue_worker_state_t value = iree_atomic_load_int32(
      &working_area->worker_state, iree_memory_order_acquire);
  return value == IREE_HAL_WORKER_STATE_WORKLOAD_PENDING;
}

static bool iree_hal_deferred_work_queue_completion_has_incoming_request(
    iree_hal_deferred_work_queue_completion_area_t* completion_area) {
  iree_hal_deferred_work_queue_worker_state_t value = iree_atomic_load_int32(
      &completion_area->worker_state, iree_memory_order_acquire);
  return value == IREE_HAL_WORKER_STATE_WORKLOAD_PENDING;
}

// Processes all ready actions in the given |worklist|.
static void iree_hal_deferred_work_queue_worker_process_ready_list(
    iree_hal_deferred_work_queue_t* actions) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_slim_mutex_lock(&actions->action_mutex);
  iree_status_t status = actions->status;
  if (IREE_UNLIKELY(!iree_status_is_ok(status))) {
    iree_hal_deferred_work_queue_ready_action_list_fail_locked(
        &actions->working_area.ready_worklist, status);
    iree_slim_mutex_unlock(&actions->action_mutex);
    iree_status_ignore(status);
    return;
  }
  iree_slim_mutex_unlock(&actions->action_mutex);

  while (true) {
    iree_hal_deferred_work_queue_entry_list_node_t* entry =
        iree_hal_deferred_work_queue_entry_list_pop(
            &actions->working_area.ready_worklist);
    if (!entry) break;

    // Process the current batch of ready actions.
    while (entry->ready_list_head) {
      iree_hal_deferred_work_queue_action_t* action =
          iree_hal_deferred_work_queue_entry_list_node_pop_front(entry);
      switch (action->state) {
        case IREE_HAL_QUEUE_ACTION_STATE_ALIVE:
          status = iree_hal_deferred_work_queue_issue_execution(action);
          break;
      }

      if (IREE_UNLIKELY(!iree_status_is_ok(status))) {
        iree_hal_deferred_work_queue_entry_list_node_push_front(entry, action);
        iree_hal_deferred_work_queue_entry_list_push(
            &actions->working_area.ready_worklist, entry);
        break;
      }
      iree_hal_deferred_work_queue_decrement_work_items_count(actions);
    }

    if (IREE_UNLIKELY(!iree_status_is_ok(status))) {
      break;
    }

    iree_allocator_free(actions->host_allocator, entry);
  }

  if (IREE_UNLIKELY(!iree_status_is_ok(status))) {
    iree_hal_deferred_work_queue_fail(actions, status);
    iree_status_ignore(status);
  }

  IREE_TRACE_ZONE_END(z0);
}

static void iree_hal_deferred_work_queue_worker_process_completion(
    iree_hal_deferred_work_queue_t* actions) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_deferred_work_queue_completion_list_t* worklist =
      &actions->completion_area.completion_list;
  iree_slim_mutex_lock(&actions->action_mutex);
  iree_status_t status = iree_status_clone(actions->status);
  iree_slim_mutex_unlock(&actions->action_mutex);

  while (true) {
    iree_hal_deferred_work_queue_completion_list_node_t* entry =
        iree_hal_deferred_work_queue_completion_list_pop(worklist);
    if (!entry) break;

    if (IREE_LIKELY(iree_status_is_ok(status))) {
      IREE_TRACE_ZONE_BEGIN_NAMED(z1, "synchronize_native_event");
      status = actions->device_interface->vtable->synchronize_native_event(
          actions->device_interface, entry->native_event);
      IREE_TRACE_ZONE_END(z1);
    }

    status =
        iree_status_join(status, entry->callback(status, entry->user_data));

    if (IREE_UNLIKELY(entry->created_event)) {
      status = iree_status_join(
          status, actions->device_interface->vtable->destroy_native_event(
                      actions->device_interface, entry->native_event));
    }
    iree_allocator_free(actions->host_allocator, entry);
  }

  if (IREE_UNLIKELY(!iree_status_is_ok(status))) {
    iree_hal_deferred_work_queue_fail(actions, status);
    iree_status_ignore(status);
  }

  IREE_TRACE_ZONE_END(z0);
}

// The main function for the completion worker thread.
static int iree_hal_deferred_work_queue_completion_execute(
    iree_hal_deferred_work_queue_t* actions) {
  iree_hal_deferred_work_queue_completion_area_t* completion_area =
      &actions->completion_area;

  iree_status_t status = actions->device_interface->vtable->bind_to_thread(
      actions->device_interface);
  if (IREE_UNLIKELY(!iree_status_is_ok(status))) {
    iree_hal_deferred_work_queue_fail(actions, status);
    iree_status_ignore(status);
  }

  while (true) {
    iree_notification_await(
        &completion_area->state_notification,
        (iree_condition_fn_t)
            iree_hal_deferred_work_queue_completion_has_incoming_request,
        completion_area, iree_infinite_timeout());

    // Immediately flip the state to idle waiting if and only if the previous
    // state is workload pending. We do it before processing ready list to make
    // sure that we don't accidentally ignore new workload pushed after done
    // ready list processing but before overwriting the state from this worker
    // thread.
    iree_atomic_store_int32(&completion_area->worker_state,
                            IREE_HAL_WORKER_STATE_IDLE_WAITING,
                            iree_memory_order_release);
    iree_hal_deferred_work_queue_worker_process_completion(actions);

    iree_slim_mutex_lock(&actions->action_mutex);
    if (IREE_UNLIKELY(actions->exit_requested &&
                      !actions->pending_work_items_count)) {
      iree_slim_mutex_unlock(&actions->action_mutex);
      return 0;
    }
    iree_slim_mutex_unlock(&actions->action_mutex);
  }

  return 0;
}

// The main function for the ready-list processing worker thread.
static int iree_hal_deferred_work_queue_worker_execute(
    iree_hal_deferred_work_queue_t* actions) {
  iree_hal_deferred_work_queue_working_area_t* working_area =
      &actions->working_area;

  // Some APIs store thread-local data. Allow the interface to bind
  // the thread-local data once for this thread rather than having to
  // do it every call.
  iree_status_t status = actions->device_interface->vtable->bind_to_thread(
      actions->device_interface);

  if (IREE_UNLIKELY(!iree_status_is_ok(status))) {
    iree_hal_deferred_work_queue_fail(actions, status);
    iree_status_ignore(status);
    // We can safely exit here because there are no actions in flight yet.
    return -1;
  }

  while (true) {
    // Block waiting for incoming requests.
    //
    // TODO: When exit is requested with
    // IREE_HAL_WORKER_STATE_EXIT_REQUESTED
    // we will return immediately causing a busy wait and hogging the CPU.
    // We need to properly wait for action cleanups to be scheduled from the
    // host stream callbacks.
    iree_notification_await(
        &working_area->state_notification,
        (iree_condition_fn_t)
            iree_hal_deferred_work_queue_worker_has_incoming_request,
        working_area, iree_infinite_timeout());

    // Immediately flip the state to idle waiting if and only if the previous
    // state is workload pending. We do it before processing ready list to make
    // sure that we don't accidentally ignore new workload pushed after done
    // ready list processing but before overwriting the state from this worker
    // thread.
    iree_atomic_store_int32(&working_area->worker_state,
                            IREE_HAL_WORKER_STATE_IDLE_WAITING,
                            iree_memory_order_release);

    iree_hal_deferred_work_queue_worker_process_ready_list(actions);

    iree_slim_mutex_lock(&actions->action_mutex);
    if (IREE_UNLIKELY(actions->exit_requested &&
                      !actions->pending_work_items_count)) {
      iree_slim_mutex_unlock(&actions->action_mutex);
      iree_hal_deferred_work_queue_notify_completion_thread(
          &actions->completion_area);
      return 0;
    }
    iree_slim_mutex_unlock(&actions->action_mutex);
  }
  return 0;
}
