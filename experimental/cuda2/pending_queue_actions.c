// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/cuda2/pending_queue_actions.h"

#include <stdbool.h>
#include <stddef.h>

#include "experimental/cuda2/cuda_device.h"
#include "experimental/cuda2/cuda_dynamic_symbols.h"
#include "experimental/cuda2/cuda_status_util.h"
#include "experimental/cuda2/event_semaphore.h"
#include "experimental/cuda2/graph_command_buffer.h"
#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/base/internal/atomic_slist.h"
#include "iree/base/internal/synchronization.h"
#include "iree/base/internal/threading.h"
#include "iree/hal/api.h"
#include "iree/hal/utils/deferred_command_buffer.h"
#include "iree/hal/utils/resource_set.h"

// The maximal number of CUevent objects a command buffer can wait.
#define IREE_HAL_CUDA_MAX_WAIT_EVENT_COUNT 32

//===----------------------------------------------------------------------===//
// Queue action
//===----------------------------------------------------------------------===//

typedef enum iree_hal_cuda2_queue_action_kind_e {
  IREE_HAL_CUDA2_QUEUE_ACTION_TYPE_EXECUTION,
  // TODO: Add support for queue alloca and dealloca.
} iree_hal_cuda2_queue_action_kind_t;

// A pending queue action.
//
// Note that this struct does not have internal synchronization; it's expected
// to work together with the pending action queue, which synchronizes accesses.
typedef struct iree_hal_cuda2_queue_action_t {
  // Intrusive doubly-linked list next entry pointer.
  struct iree_hal_cuda2_queue_action_t* next;
  // Intrusive doubly-linked list previous entry pointer.
  struct iree_hal_cuda2_queue_action_t* prev;

  // The owning pending actions queue. We use its allocators and pools.
  // Retained to make sure it outlives the current action.
  iree_hal_cuda2_pending_queue_actions_t* owning_actions;

  iree_hal_cuda2_queue_action_kind_t kind;
  union {
    struct {
      iree_host_size_t count;
      iree_hal_command_buffer_t* const* ptr;
    } command_buffers;
  } payload;

  // The device from which to allocate CUDA stream-based command buffers for
  // applying deferred command buffers.
  iree_hal_device_t* device;

  // The stream to launch main GPU workload.
  CUstream dispatch_cu_stream;
  // The stream to launch CUDA host function callbacks.
  CUstream callback_cu_stream;

  // Resource set to retain all associated resources by the payload.
  iree_hal_resource_set_t* resource_set;

  // Semaphore list to wait on for the payload to start on the GPU.
  iree_hal_semaphore_list_t wait_semaphore_list;
  // Semaphore list to signal after the payload completes on the GPU.
  iree_hal_semaphore_list_t signal_semaphore_list;

  // Scratch fields for analyzing whether actions are ready to issue.
  CUevent events[IREE_HAL_CUDA_MAX_WAIT_EVENT_COUNT];
  iree_host_size_t event_count;
  bool is_pending;
} iree_hal_cuda2_queue_action_t;

//===----------------------------------------------------------------------===//
// Queue action list
//===----------------------------------------------------------------------===//

typedef struct iree_hal_cuda2_queue_action_list_t {
  iree_hal_cuda2_queue_action_t* head;
  iree_hal_cuda2_queue_action_t* tail;
} iree_hal_cuda2_queue_action_list_t;

// Returns true if the action list is empty.
static inline bool iree_hal_cuda2_queue_action_list_is_empty(
    const iree_hal_cuda2_queue_action_list_t* list) {
  return list->head == NULL;
}

// Pushes |action| on to the end of the given action |list|.
static void iree_hal_cuda2_queue_action_list_push_back(
    iree_hal_cuda2_queue_action_list_t* list,
    iree_hal_cuda2_queue_action_t* action) {
  if (list->tail) {
    list->tail->next = action;
  } else {
    list->head = action;
  }
  action->next = NULL;
  action->prev = list->tail;
  list->tail = action;
}

// Erases |action| from |list|.
static void iree_hal_cuda2_queue_action_list_erase(
    iree_hal_cuda2_queue_action_list_t* list,
    iree_hal_cuda2_queue_action_t* action) {
  iree_hal_cuda2_queue_action_t* next = action->next;
  iree_hal_cuda2_queue_action_t* prev = action->prev;
  if (prev) {
    prev->next = next;
    action->prev = NULL;
  } else {
    list->head = next;
  }
  if (next) {
    next->prev = prev;
    action->next = NULL;
  } else {
    list->tail = prev;
  }
}

// Takes all actions from |available_list| and moves them into |ready_list|.
static void iree_hal_cuda2_queue_action_list_take_all(
    iree_hal_cuda2_queue_action_list_t* available_list,
    iree_hal_cuda2_queue_action_list_t* ready_list) {
  IREE_ASSERT(available_list != ready_list);
  ready_list->head = available_list->head;
  ready_list->tail = available_list->tail;
  available_list->head = NULL;
  available_list->tail = NULL;
}

//===----------------------------------------------------------------------===//
// Ready-list processing
//===----------------------------------------------------------------------===//

// Ready action atomic slist entry struct.
typedef struct iree_hal_cuda2_atomic_slist_entry_t {
  iree_hal_cuda2_queue_action_t* ready_list_head;
  iree_atomic_slist_intrusive_ptr_t slist_next;
} iree_hal_cuda2_atomic_slist_entry_t;

// Ready action atomic slist.
IREE_TYPED_ATOMIC_SLIST_WRAPPER(iree_hal_cuda2_ready_action,
                                iree_hal_cuda2_atomic_slist_entry_t,
                                offsetof(iree_hal_cuda2_atomic_slist_entry_t,
                                         slist_next));

// The ready-list processing worker's working/exiting state.
typedef enum iree_hal_cuda2_worker_state_e {
  IREE_HAL_CUDA_WORKER_STATE_IDLE_WAITING = 0,
  IREE_HAL_CUDA_WORKER_STATE_WORKLOAD_PENDING = 1,
  IREE_HAL_CUDA_WORKER_STATE_EXIT_REQUESTED = -1,
  IREE_HAL_CUDA_WORKER_STATE_EXIT_COMMITTED = -2,
} iree_hal_cuda2_worker_state_t;

// The data structure needed by a ready-list processing worker thread to issue
// ready actions to the GPU.
//
// This data structure is shared between the parent thread, which owns the
// whole pending actions queue, and the worker thread; so proper synchronization
// is needed to touch it from both sides.
//
// The parent thread should push a list of ready actions to ready_worklist,
// update working_state, and give notification naccordingly.
// The worker thread waits on the notification and checks working_state, and
// pops from the ready_worklist to process. THe worker thread also monintors
// exiting_state and stops processing if requested by the parent thread.
typedef struct iree_hal_cuda2_working_area_t {
  iree_notification_t notification;
  iree_hal_cuda2_ready_action_slist_t ready_worklist;  // atomic
  iree_atomic_int32_t working_state;                   // atomic
  iree_atomic_int32_t exiting_state;                   // atomic
  iree_allocator_t host_allocator;                     // const
} iree_hal_cuda2_working_area_t;

static void iree_hal_cuda2_working_area_initialize(
    iree_allocator_t host_allocator,
    iree_hal_cuda2_working_area_t* working_area) {
  iree_notification_initialize(&working_area->notification);
  iree_hal_cuda2_ready_action_slist_initialize(&working_area->ready_worklist);
  iree_atomic_store_int32(&working_area->working_state,
                          IREE_HAL_CUDA_WORKER_STATE_IDLE_WAITING,
                          iree_memory_order_relaxed);
  iree_atomic_store_int32(&working_area->exiting_state,
                          IREE_HAL_CUDA_WORKER_STATE_IDLE_WAITING,
                          iree_memory_order_relaxed);
  working_area->host_allocator = host_allocator;
}

static void iree_hal_cuda2_working_area_deinitialize(
    iree_hal_cuda2_working_area_t* working_area) {
  iree_hal_cuda2_ready_action_slist_deinitialize(&working_area->ready_worklist);
  iree_notification_deinitialize(&working_area->notification);
}

// The main function for the ready-list processing worker thread.
static int iree_hal_cuda2_worker_execute(void* args);

//===----------------------------------------------------------------------===//
// Pending queue actions
//===----------------------------------------------------------------------===//

struct iree_hal_cuda2_pending_queue_actions_t {
  // Abstract resource used for injecting reference counting and vtable;
  // must be at offset 0.
  iree_hal_resource_t resource;

  // The allocator used to create the timepoint pool.
  iree_allocator_t host_allocator;
  // The block pool to allocate resource sets from.
  iree_arena_block_pool_t* block_pool;

  // The symbols used to create and destroy CUevent objects.
  const iree_hal_cuda2_dynamic_symbols_t* symbols;

  // Non-recursive mutex guarding access to the action list.
  iree_slim_mutex_t action_mutex;

  // The double-linked list of pending actions.
  iree_hal_cuda2_queue_action_list_t action_list IREE_GUARDED_BY(action_mutex);

  // The worker thread that monitors incoming requests and issues ready actions
  // to the GPU.
  iree_thread_t* worker_thread;
  // The worker's working area; data exchange place with the parent thread.
  iree_hal_cuda2_working_area_t working_area;
};

static const iree_hal_resource_vtable_t
    iree_hal_cuda2_pending_queue_actions_vtable;

iree_status_t iree_hal_cuda2_pending_queue_actions_create(
    const iree_hal_cuda2_dynamic_symbols_t* symbols,
    iree_arena_block_pool_t* block_pool, iree_allocator_t host_allocator,
    iree_hal_cuda2_pending_queue_actions_t** out_actions) {
  IREE_ASSERT_ARGUMENT(symbols);
  IREE_ASSERT_ARGUMENT(block_pool);
  IREE_ASSERT_ARGUMENT(out_actions);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_cuda2_pending_queue_actions_t* actions = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*actions),
                                (void**)&actions));
  iree_hal_resource_initialize(&iree_hal_cuda2_pending_queue_actions_vtable,
                               &actions->resource);
  actions->host_allocator = host_allocator;
  actions->block_pool = block_pool;
  actions->symbols = symbols;
  iree_slim_mutex_initialize(&actions->action_mutex);
  memset(&actions->action_list, 0, sizeof(actions->action_list));

  // Initialize the working area for the ready-list processing worker.
  iree_hal_cuda2_working_area_t* working_area = &actions->working_area;
  iree_hal_cuda2_working_area_initialize(host_allocator, working_area);

  // Create the ready-list processing worker itself.
  iree_thread_create_params_t params;
  memset(&params, 0, sizeof(params));
  params.name = IREE_SV("deferred_queue_worker");
  params.create_suspended = false;
  iree_status_t status =
      iree_thread_create(iree_hal_cuda2_worker_execute, working_area, params,
                         actions->host_allocator, &actions->worker_thread);

  if (iree_status_is_ok(status)) {
    *out_actions = actions;
  } else {
    iree_hal_cuda2_pending_queue_actions_destroy((iree_hal_resource_t*)actions);
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_hal_cuda2_pending_queue_actions_t*
iree_hal_cuda2_pending_queue_actions_cast(iree_hal_resource_t* base_value) {
  return (iree_hal_cuda2_pending_queue_actions_t*)base_value;
}

void iree_hal_cuda2_pending_queue_actions_destroy(
    iree_hal_resource_t* base_actions) {
  iree_hal_cuda2_pending_queue_actions_t* actions =
      iree_hal_cuda2_pending_queue_actions_cast(base_actions);
  iree_allocator_t host_allocator = actions->host_allocator;
  iree_hal_cuda2_working_area_t* working_area = &actions->working_area;
  IREE_TRACE_ZONE_BEGIN(z0);

  IREE_ASSERT(iree_hal_cuda2_queue_action_list_is_empty(&actions->action_list));

  iree_atomic_int32_t* exiting_state = &working_area->exiting_state;
  iree_atomic_store_int32(exiting_state,
                          IREE_HAL_CUDA_WORKER_STATE_EXIT_REQUESTED,
                          iree_memory_order_relaxed);
  while (iree_atomic_load_int32(exiting_state, iree_memory_order_relaxed) !=
         IREE_HAL_CUDA_WORKER_STATE_EXIT_COMMITTED) {
    // Busy wait until the worker thread exits.
  }
  iree_thread_release(actions->worker_thread);

  iree_hal_cuda2_working_area_deinitialize(working_area);
  iree_slim_mutex_deinitialize(&actions->action_mutex);
  iree_allocator_free(host_allocator, actions);

  IREE_TRACE_ZONE_END(z0);
}

static const iree_hal_resource_vtable_t
    iree_hal_cuda2_pending_queue_actions_vtable = {
        .destroy = iree_hal_cuda2_pending_queue_actions_destroy,
};

// Copies of the given |in_list| to |out_list| to retain the command buffer
// list.
static iree_status_t iree_hal_cuda2_copy_command_buffer_list(
    iree_host_size_t command_buffer_count,
    iree_hal_command_buffer_t* const* in_list, iree_allocator_t host_allocator,
    iree_hal_command_buffer_t* const** out_list) {
  if (command_buffer_count == 0) {
    *out_list = NULL;
  } else {
    iree_host_size_t total_size = command_buffer_count * sizeof(*in_list);
    IREE_RETURN_IF_ERROR(
        iree_allocator_malloc(host_allocator, total_size, (void**)out_list));
    memcpy((void*)*out_list, in_list, total_size);
  }
  return iree_ok_status();
}

// Frees the semaphore and value list inside |semaphore_list|.
static void iree_hal_cuda2_free_command_buffer_list(
    iree_allocator_t host_allocator,
    iree_hal_command_buffer_t* const* command_buffer_list) {
  iree_allocator_free(host_allocator, (void*)command_buffer_list);
}

// Copies of the given |in_list| to |out_list| to retain the semaphore and value
// list.
static iree_status_t iree_hal_cuda2_copy_semaphore_list(
    iree_hal_semaphore_list_t in_list, iree_allocator_t host_allocator,
    iree_hal_semaphore_list_t* out_list) {
  if (in_list.count == 0) {
    memset(out_list, 0, sizeof(*out_list));
  } else {
    out_list->count = in_list.count;

    iree_host_size_t semaphore_size =
        in_list.count * sizeof(*in_list.semaphores);
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, semaphore_size,
                                               (void**)&out_list->semaphores));
    memcpy(out_list->semaphores, in_list.semaphores, semaphore_size);

    iree_host_size_t value_size =
        in_list.count * sizeof(*in_list.payload_values);
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(
        host_allocator, value_size, (void**)&out_list->payload_values));
    memcpy(out_list->payload_values, in_list.payload_values, value_size);
  }
  return iree_ok_status();
}

// Frees the semaphore and value list inside |semaphore_list|.
static void iree_hal_cuda2_free_semaphore_list(
    iree_allocator_t host_allocator,
    iree_hal_semaphore_list_t* semaphore_list) {
  iree_allocator_free(host_allocator, semaphore_list->semaphores);
  iree_allocator_free(host_allocator, semaphore_list->payload_values);
}

iree_status_t iree_hal_cuda2_pending_queue_actions_enqueue_execution(
    iree_hal_device_t* device, CUstream dispatch_stream,
    CUstream callback_stream, iree_hal_cuda2_pending_queue_actions_t* actions,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_host_size_t command_buffer_count,
    iree_hal_command_buffer_t* const* command_buffers) {
  IREE_ASSERT_ARGUMENT(actions);
  IREE_ASSERT_ARGUMENT(command_buffer_count == 0 || command_buffers);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_cuda2_queue_action_t* action = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(actions->host_allocator, sizeof(*action),
                                (void**)&action));

  action->kind = IREE_HAL_CUDA2_QUEUE_ACTION_TYPE_EXECUTION;
  action->device = device;
  action->dispatch_cu_stream = dispatch_stream;
  action->callback_cu_stream = callback_stream;
  action->event_count = 0;
  action->is_pending = true;

  // Retain all command buffers and semaphores.
  iree_hal_resource_set_t* resource_set = NULL;
  iree_status_t status =
      iree_hal_resource_set_allocate(actions->block_pool, &resource_set);
  if (IREE_LIKELY(iree_status_is_ok(status))) {
    status = iree_hal_resource_set_insert(resource_set, command_buffer_count,
                                          command_buffers);
  }
  if (IREE_LIKELY(iree_status_is_ok(status))) {
    status =
        iree_hal_resource_set_insert(resource_set, wait_semaphore_list.count,
                                     wait_semaphore_list.semaphores);
  }
  if (IREE_LIKELY(iree_status_is_ok(status))) {
    status =
        iree_hal_resource_set_insert(resource_set, signal_semaphore_list.count,
                                     signal_semaphore_list.semaphores);
  }

  // Copy the command buffer list for later access.
  // TODO: avoid host allocator malloc; use some pool for the allocation.
  if (IREE_LIKELY(iree_status_is_ok(status))) {
    action->payload.command_buffers.count = command_buffer_count;
    status = iree_hal_cuda2_copy_command_buffer_list(
        command_buffer_count, command_buffers, actions->host_allocator,
        &action->payload.command_buffers.ptr);
  }

  // Copy the semaphore and value list for later access.
  // TODO: avoid host allocator malloc; use some pool for the allocation.
  if (IREE_LIKELY(iree_status_is_ok(status))) {
    status = iree_hal_cuda2_copy_semaphore_list(wait_semaphore_list,
                                                actions->host_allocator,
                                                &action->wait_semaphore_list);
  }
  if (IREE_LIKELY(iree_status_is_ok(status))) {
    status = iree_hal_cuda2_copy_semaphore_list(signal_semaphore_list,
                                                actions->host_allocator,
                                                &action->signal_semaphore_list);
  }

  if (IREE_LIKELY(iree_status_is_ok(status))) {
    action->owning_actions = actions;
    iree_hal_resource_retain(actions);

    action->resource_set = resource_set;

    iree_slim_mutex_lock(&actions->action_mutex);
    iree_hal_cuda2_queue_action_list_push_back(&actions->action_list, action);
    iree_slim_mutex_unlock(&actions->action_mutex);
  } else {
    iree_hal_cuda2_free_semaphore_list(actions->host_allocator,
                                       &action->wait_semaphore_list);
    iree_hal_cuda2_free_semaphore_list(actions->host_allocator,
                                       &action->signal_semaphore_list);
    iree_hal_cuda2_free_command_buffer_list(
        actions->host_allocator, action->payload.command_buffers.ptr);
    iree_hal_resource_set_free(resource_set);
    iree_allocator_free(actions->host_allocator, action);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_cuda2_pending_queue_actions_cleanup_execution(
    iree_hal_cuda2_queue_action_t* action);

// Releases resources after action completion on the GPU and advances timeline
// and pending actions queue.
//
// This is the CUDA host function callback to cudaLaunchHostFunc, invoked by a
// CUDA driver thread.
static void iree_hal_cuda2_execution_device_signal_host_callback(
    void* user_data) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_cuda2_queue_action_t* action =
      (iree_hal_cuda2_queue_action_t*)user_data;
  iree_hal_cuda2_pending_queue_actions_t* actions = action->owning_actions;
  // Advance semaphore timelines by calling into the host signaling function.
  IREE_IGNORE_ERROR(
      iree_hal_semaphore_list_signal(action->signal_semaphore_list));
  // Destroy the current action given its done now--this also frees all retained
  // resources.
  iree_hal_cuda2_pending_queue_actions_cleanup_execution(action);
  // Try to release more pending actions to the GPU now.
  IREE_IGNORE_ERROR(iree_hal_cuda2_pending_queue_actions_issue(actions));
  IREE_TRACE_ZONE_END(z0);
}

// Issues the given kernel dispatch |action| to the GPU.
static iree_status_t iree_hal_cuda2_pending_queue_actions_issue_execution(
    iree_hal_cuda2_queue_action_t* action) {
  IREE_ASSERT(action->is_pending == false);
  const iree_hal_cuda2_dynamic_symbols_t* symbols =
      action->owning_actions->symbols;
  IREE_TRACE_ZONE_BEGIN(z0);

  // No need to lock given that this action is already detched from the pending
  // actions list; so only this thread is seeing it now.

  // First wait all the device CUevent in the dispatch stream.
  for (iree_host_size_t i = 0; i < action->event_count; ++i) {
    IREE_CUDA_RETURN_AND_END_ZONE_IF_ERROR(
        z0, symbols,
        cuStreamWaitEvent(action->dispatch_cu_stream, action->events[i],
                          CU_EVENT_WAIT_DEFAULT),
        "cuStreamWaitEvent");
  }

  // Then launch all command buffers to the dispatch stream.
  for (iree_host_size_t i = 0; i < action->payload.command_buffers.count; ++i) {
    iree_hal_command_buffer_t* command_buffer =
        action->payload.command_buffers.ptr[i];
    if (iree_hal_cuda2_graph_command_buffer_isa(command_buffer)) {
      CUgraphExec exec = iree_hal_cuda2_graph_command_buffer_handle(
          action->payload.command_buffers.ptr[i]);
      IREE_CUDA_RETURN_AND_END_ZONE_IF_ERROR(
          z0, symbols, cuGraphLaunch(exec, action->dispatch_cu_stream),
          "cuGraphLaunch");
    } else {
      iree_hal_command_buffer_t* stream_command_buffer = NULL;
      iree_hal_command_buffer_mode_t mode =
          IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT |
          IREE_HAL_COMMAND_BUFFER_MODE_ALLOW_INLINE_EXECUTION |
          IREE_HAL_COMMAND_BUFFER_MODE_UNVALIDATED;
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_hal_cuda2_device_create_stream_command_buffer(
                  action->device, mode, IREE_HAL_COMMAND_CATEGORY_ANY,
                  /*binding_capacity=*/0, &stream_command_buffer));
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_hal_resource_set_insert(action->resource_set, 1,
                                           &stream_command_buffer));
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_hal_deferred_command_buffer_apply(
                  command_buffer, stream_command_buffer,
                  iree_hal_buffer_binding_table_empty()));
    }
  }

  // Last record CUevent signals in the dispatch stream.
  for (iree_host_size_t i = 0; i < action->signal_semaphore_list.count; ++i) {
    // Grab a CUevent for this semaphore value signaling.
    CUevent event = NULL;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_cuda2_event_semaphore_acquire_timepoint_device_signal(
                action->signal_semaphore_list.semaphores[i],
                action->signal_semaphore_list.payload_values[i], &event));

    // Record the event signaling in the dispatch stream.
    IREE_CUDA_RETURN_AND_END_ZONE_IF_ERROR(
        z0, symbols, cuEventRecord(event, action->dispatch_cu_stream),
        "cuEventRecord");
    // Let the callback stream to wait on the CUevent.
    IREE_CUDA_RETURN_AND_END_ZONE_IF_ERROR(
        z0, symbols,
        cuStreamWaitEvent(action->callback_cu_stream, event,
                          CU_EVENT_WAIT_DEFAULT),
        "cuStreamWaitEvent");
  }

  // Now launch a host function on the callback stream to advance the semaphore
  // timeline.
  IREE_CUDA_RETURN_AND_END_ZONE_IF_ERROR(
      z0, symbols,
      cuLaunchHostFunc(action->callback_cu_stream,
                       iree_hal_cuda2_execution_device_signal_host_callback,
                       action),
      "cuLaunchHostFunc");

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

// Releases resources after completing the given kernel dispatch |action|.
static void iree_hal_cuda2_pending_queue_actions_cleanup_execution(
    iree_hal_cuda2_queue_action_t* action) {
  iree_hal_cuda2_pending_queue_actions_t* actions = action->owning_actions;
  iree_allocator_t host_allocator = actions->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_resource_set_free(action->resource_set);
  iree_hal_cuda2_free_semaphore_list(host_allocator,
                                     &action->wait_semaphore_list);
  iree_hal_cuda2_free_semaphore_list(host_allocator,
                                     &action->signal_semaphore_list);
  iree_hal_resource_release(actions);

  iree_allocator_free(host_allocator, action);

  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_hal_cuda2_pending_queue_actions_issue(
    iree_hal_cuda2_pending_queue_actions_t* actions) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_cuda2_queue_action_list_t pending_list = {NULL, NULL};
  iree_hal_cuda2_queue_action_list_t ready_list = {NULL, NULL};

  iree_slim_mutex_lock(&actions->action_mutex);

  if (iree_hal_cuda2_queue_action_list_is_empty(&actions->action_list)) {
    iree_slim_mutex_unlock(&actions->action_mutex);
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  // Scan through the list and categorize actions into pending and ready lists.
  for (iree_hal_cuda2_queue_action_t* action = actions->action_list.head;
       action != NULL;) {
    iree_hal_cuda2_queue_action_t* next_action = action->next;
    action->next = NULL;

    iree_host_size_t semaphore_count = action->wait_semaphore_list.count;
    iree_hal_semaphore_t** semaphores = action->wait_semaphore_list.semaphores;
    uint64_t* values = action->wait_semaphore_list.payload_values;

    action->event_count = 0;
    action->is_pending = false;

    // Look at all wait semaphores.
    for (iree_host_size_t i = 0; i < semaphore_count; ++i) {
      // If this semaphore has already signaled past the desired value, we can
      // just ignore it.
      uint64_t value = 0;
      iree_status_t status = iree_hal_semaphore_query(semaphores[i], &value);
      if (!iree_status_is_ok(status)) {
        iree_slim_mutex_unlock(&actions->action_mutex);
        IREE_TRACE_ZONE_END(z0);
        return status;
      }
      if (value >= values[i]) continue;

      // Try to acquire a CUevent from a device wait timepoint. If so, we can
      // use that CUevent to wait on the device. Otherwise, this action is still
      // not ready.
      CUevent event = NULL;
      status = iree_hal_cuda2_event_semaphore_acquire_timepoint_device_wait(
          semaphores[i], values[i], &event);
      if (!iree_status_is_ok(status)) {
        iree_slim_mutex_unlock(&actions->action_mutex);
        IREE_TRACE_ZONE_END(z0);
        return status;
      }
      if (!event) {
        // Clear the scratch fields.
        action->event_count = 0;
        action->is_pending = true;
        break;
      }
      if (action->event_count >= IREE_HAL_CUDA_MAX_WAIT_EVENT_COUNT) {
        iree_slim_mutex_unlock(&actions->action_mutex);
        IREE_TRACE_ZONE_END(z0);
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "exceeded max wait CUevent limit");
      }
      action->events[action->event_count++] = event;
    }

    if (action->is_pending) {
      iree_hal_cuda2_queue_action_list_push_back(&pending_list, action);
    } else {
      iree_hal_cuda2_queue_action_list_push_back(&ready_list, action);
    }

    action = next_action;
  }

  // Preserve pending timepoints.
  actions->action_list = pending_list;

  iree_slim_mutex_unlock(&actions->action_mutex);

  iree_hal_cuda2_atomic_slist_entry_t* entry = NULL;
  // TODO: avoid host allocator malloc; use some pool for the allocation.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(actions->host_allocator, sizeof(*entry),
                                (void**)&entry));
  entry->ready_list_head = ready_list.head;

  // Now push the ready list to the worker and have it to issue the actions to
  // the GPU.
  iree_hal_cuda2_working_area_t* working_area = &actions->working_area;
  iree_hal_cuda2_ready_action_slist_push(&working_area->ready_worklist, entry);
  iree_atomic_store_int32(&working_area->working_state,
                          IREE_HAL_CUDA_WORKER_STATE_WORKLOAD_PENDING,
                          iree_memory_order_relaxed);
  iree_notification_post(&working_area->notification, 1);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Worker routines
//===----------------------------------------------------------------------===//

static bool iree_hal_cuda2_worker_has_incoming_request(void* args) {
  iree_hal_cuda2_working_area_t* working_area =
      (iree_hal_cuda2_working_area_t*)args;
  return iree_atomic_load_int32(&working_area->working_state,
                                iree_memory_order_relaxed) ==
         IREE_HAL_CUDA_WORKER_STATE_WORKLOAD_PENDING;
}

// Processes all ready actions in the given |worklist|.
static int iree_hal_cuda2_worker_process_ready_list(
    iree_allocator_t host_allocator,
    iree_hal_cuda2_ready_action_slist_t* worklist) {
  iree_hal_cuda2_atomic_slist_entry_t* entry =
      iree_hal_cuda2_ready_action_slist_pop(worklist);
  while (entry) {
    // Process the current batch of ready actions.
    for (iree_hal_cuda2_queue_action_t* action = entry->ready_list_head;
         action != NULL;) {
      iree_hal_cuda2_queue_action_t* next_action = action->next;
      action->next = NULL;

      iree_status_t status =
          iree_hal_cuda2_pending_queue_actions_issue_execution(action);
      if (!iree_status_is_ok(status)) {
        // TODO: surface better error back from this thread.
        iree_status_ignore(status);
        return 1;
      }
      action->event_count = 0;

      action = next_action;
    }

    iree_allocator_free(host_allocator, entry);
    // Try to see if we have the next batch.
    entry = iree_hal_cuda2_ready_action_slist_pop(worklist);
  }
  return 0;
}

// The main function for the ready-list processing worker thread.
static int iree_hal_cuda2_worker_execute(void* args) {
  iree_hal_cuda2_working_area_t* working_area =
      (iree_hal_cuda2_working_area_t*)args;

  iree_hal_cuda2_ready_action_slist_t* worklist = &working_area->ready_worklist;
  iree_atomic_int32_t* exiting_state = &working_area->exiting_state;

  while (true) {
    // Check if we received request to stop processing and exit this thread.
    if (iree_atomic_load_int32(exiting_state, iree_memory_order_relaxed) ==
        IREE_HAL_CUDA_WORKER_STATE_EXIT_REQUESTED) {
      // Process pending requests for the last time.
      int return_value = iree_hal_cuda2_worker_process_ready_list(
          working_area->host_allocator, worklist);

      // Signal that this thread is committed to exit.
      iree_atomic_store_int32(exiting_state,
                              IREE_HAL_CUDA_WORKER_STATE_EXIT_COMMITTED,
                              iree_memory_order_relaxed);
      return return_value;
    }

    // Block waiting for incoming requests.
    iree_notification_await(&working_area->notification,
                            iree_hal_cuda2_worker_has_incoming_request, args,
                            iree_infinite_timeout());

    int return_value = iree_hal_cuda2_worker_process_ready_list(
        working_area->host_allocator, worklist);
    if (return_value != 0) return return_value;

    // Signal that this thread is done processing and now waiting for more.
    iree_atomic_store_int32(&working_area->working_state,
                            IREE_HAL_CUDA_WORKER_STATE_IDLE_WAITING,
                            iree_memory_order_relaxed);
  }
  return 0;
}
