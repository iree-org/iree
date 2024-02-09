// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/local_task/task_command_buffer.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/hal/local/executable_environment.h"
#include "iree/hal/local/executable_library.h"
#include "iree/hal/local/local_executable.h"
#include "iree/hal/local/local_pipeline_layout.h"
#include "iree/hal/utils/resource_set.h"
#include "iree/task/affinity_set.h"
#include "iree/task/list.h"
#include "iree/task/submission.h"
#include "iree/task/task.h"

//===----------------------------------------------------------------------===//
// iree_hal_task_command_buffer_t
//===----------------------------------------------------------------------===//

// iree/task/-based command buffer.
// We track a minimal amount of state here and incrementally build out the task
// DAG that we can submit to the task system directly. There's no intermediate
// data structures and we produce the iree_task_ts directly. In the steady state
// all allocations are served from a shared per-device block pool with no
// additional allocations required during recording or execution. That means our
// command buffer here is essentially just a builder for the task system types
// and manager of the lifetime of the tasks.
typedef struct iree_hal_task_command_buffer_t {
  iree_hal_command_buffer_t base;
  iree_allocator_t host_allocator;

  iree_task_scope_t* scope;

  // Arena used for all allocations; references the shared device block pool.
  iree_arena_allocator_t arena;

  // Maintains a reference to all resources used within the command buffer.
  // Reset on each begin.
  iree_hal_resource_set_t* resource_set;

  // One or more tasks at the root of the command buffer task DAG.
  // These tasks are all able to execute concurrently and will be the initial
  // ready task set in the submission.
  iree_task_list_t root_tasks;

  // One or more tasks at the leaves of the DAG.
  // Only once all these tasks have completed execution will the command buffer
  // be considered completed as a whole.
  //
  // An empty list indicates that root_tasks are also the leaves.
  iree_task_list_t leaf_tasks;

  // TODO(benvanik): move this out of the struct and allocate from the arena -
  // we only need this during recording and it's ~4KB of waste otherwise.
  // State tracked within the command buffer during recording only.
  struct {
    // The last global barrier that was inserted, if any.
    // The barrier is allocated and inserted into the DAG when requested but the
    // actual barrier dependency list is only allocated and set on flushes.
    // This lets us allocate the appropriately sized barrier task list from the
    // arena even though when the barrier is recorded we don't yet know what
    // other tasks we'll be emitting as we walk the command stream.
    iree_task_barrier_t* open_barrier;

    // The number of tasks in the open barrier (|open_tasks|), used to quickly
    // allocate storage for the task list without needing to walk the list.
    iree_host_size_t open_task_count;

    // All execution tasks emitted that must execute after |open_barrier|.
    iree_task_list_t open_tasks;

    // A flattened list of all available descriptor set bindings.
    // As descriptor sets are pushed/bound the bindings will be updated to
    // represent the fully-translated binding data pointer.
    // TODO(benvanik): support proper mapping semantics and track the
    // iree_hal_buffer_mapping_t and map/unmap where appropriate.
    void* bindings[IREE_HAL_LOCAL_MAX_DESCRIPTOR_SET_COUNT *
                   IREE_HAL_LOCAL_MAX_DESCRIPTOR_BINDING_COUNT];
    iree_device_size_t
        binding_lengths[IREE_HAL_LOCAL_MAX_DESCRIPTOR_SET_COUNT *
                        IREE_HAL_LOCAL_MAX_DESCRIPTOR_BINDING_COUNT];

    // All available push constants updated each time push_constants is called.
    // Reset only with the command buffer and otherwise will maintain its values
    // during recording to allow for partial push_constants updates.
    uint32_t push_constants[IREE_HAL_LOCAL_MAX_PUSH_CONSTANT_COUNT];
  } state;
} iree_hal_task_command_buffer_t;

static const iree_hal_command_buffer_vtable_t
    iree_hal_task_command_buffer_vtable;

static iree_hal_task_command_buffer_t* iree_hal_task_command_buffer_cast(
    iree_hal_command_buffer_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_task_command_buffer_vtable);
  return (iree_hal_task_command_buffer_t*)base_value;
}

iree_status_t iree_hal_task_command_buffer_create(
    iree_hal_device_t* device, iree_task_scope_t* scope,
    iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_arena_block_pool_t* block_pool, iree_allocator_t host_allocator,
    iree_hal_command_buffer_t** out_command_buffer) {
  IREE_ASSERT_ARGUMENT(out_command_buffer);
  *out_command_buffer = NULL;

  if (!iree_all_bits_set(mode, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT)) {
    // If we want reuse we'd need to support duplicating the task DAG after
    // recording or have some kind of copy-on-submit behavior that does so if
    // a command buffer is submitted for execution twice. Allowing for the same
    // command buffer to be enqueued multiple times would be fine so long as
    // execution doesn't overlap (`cmdbuf|cmdbuf` vs
    // `cmdbuf -> semaphore -> cmdbuf`) though we'd still need to be careful
    // that we did the enqueuing and reset of the task structures at the right
    // times. Definitely something that'll be useful in the future... but not
    // today :)
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "only one-shot command buffer usage is supported");
  }
  if (binding_capacity > 0) {
    // TODO(#10144): support indirect command buffers with binding tables.
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "indirect command buffers not yet implemented");
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_task_command_buffer_t* command_buffer = NULL;
  iree_status_t status = iree_allocator_malloc(
      host_allocator, sizeof(*command_buffer), (void**)&command_buffer);
  if (iree_status_is_ok(status)) {
    iree_hal_command_buffer_initialize(
        device, mode, command_categories, queue_affinity, binding_capacity,
        &iree_hal_task_command_buffer_vtable, &command_buffer->base);
    command_buffer->host_allocator = host_allocator;
    command_buffer->scope = scope;
    iree_arena_initialize(block_pool, &command_buffer->arena);
    iree_task_list_initialize(&command_buffer->root_tasks);
    iree_task_list_initialize(&command_buffer->leaf_tasks);
    memset(&command_buffer->state, 0, sizeof(command_buffer->state));
    status = iree_hal_resource_set_allocate(block_pool,
                                            &command_buffer->resource_set);
  }
  if (iree_status_is_ok(status)) {
    *out_command_buffer = &command_buffer->base;
  } else {
    iree_hal_command_buffer_release(&command_buffer->base);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_task_command_buffer_destroy(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_task_command_buffer_t* command_buffer =
      iree_hal_task_command_buffer_cast(base_command_buffer);
  iree_allocator_t host_allocator = command_buffer->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  memset(&command_buffer->state, 0, sizeof(command_buffer->state));
  iree_task_list_discard(&command_buffer->root_tasks);
  iree_task_list_discard(&command_buffer->leaf_tasks);
  iree_arena_deinitialize(&command_buffer->arena);
  iree_hal_resource_set_free(command_buffer->resource_set);
  iree_allocator_free(host_allocator, command_buffer);

  IREE_TRACE_ZONE_END(z0);
}

bool iree_hal_task_command_buffer_isa(
    iree_hal_command_buffer_t* command_buffer) {
  return iree_hal_resource_is(&command_buffer->resource,
                              &iree_hal_task_command_buffer_vtable);
}

//===----------------------------------------------------------------------===//
// iree_hal_task_command_buffer_t recording
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_task_command_buffer_flush_tasks(
    iree_hal_task_command_buffer_t* command_buffer);

static iree_status_t iree_hal_task_command_buffer_begin(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_task_command_buffer_t* command_buffer =
      iree_hal_task_command_buffer_cast(base_command_buffer);
  if (!iree_task_list_is_empty(&command_buffer->root_tasks)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "command buffer cannot be re-recorded");
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_task_command_buffer_end(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_task_command_buffer_t* command_buffer =
      iree_hal_task_command_buffer_cast(base_command_buffer);

  // Flush any open barriers.
  IREE_RETURN_IF_ERROR(
      iree_hal_task_command_buffer_flush_tasks(command_buffer));

  // Move the tasks from the leaf list (tail) to the root list (head) if this
  // was the first set of tasks recorded.
  if (iree_task_list_is_empty(&command_buffer->root_tasks) &&
      !iree_task_list_is_empty(&command_buffer->leaf_tasks)) {
    iree_task_list_move(&command_buffer->leaf_tasks,
                        &command_buffer->root_tasks);
  }

  iree_hal_resource_set_freeze(command_buffer->resource_set);

  return iree_ok_status();
}

// Flushes all open tasks to the previous barrier and prepares for more
// recording. The root tasks are also populated here when required as this is
// the one place where we can see both halves of the most recent synchronization
// event: those tasks recorded prior (if any) and the task that marks the set of
// tasks that will be recorded after (if any).
static iree_status_t iree_hal_task_command_buffer_flush_tasks(
    iree_hal_task_command_buffer_t* command_buffer) {
  iree_task_barrier_t* open_barrier = command_buffer->state.open_barrier;
  if (open_barrier != NULL) {
    // There is an open barrier we need to fixup the fork out to all of the open
    // tasks that were recorded after it.
    iree_task_t* task_head =
        iree_task_list_front(&command_buffer->state.open_tasks);
    iree_host_size_t dependent_task_count =
        command_buffer->state.open_task_count;
    if (dependent_task_count == 1) {
      // Special-case: only one open task so we can avoid the additional barrier
      // overhead by reusing the completion task.
      iree_task_set_completion_task(&open_barrier->header, task_head);
    } else if (dependent_task_count > 1) {
      // Allocate the list of tasks we'll stash back on the previous barrier.
      // Since we couldn't know at the time how many tasks would end up in the
      // barrier we had to defer it until now.
      iree_task_t** dependent_tasks = NULL;
      IREE_RETURN_IF_ERROR(iree_arena_allocate(
          &command_buffer->arena, dependent_task_count * sizeof(iree_task_t*),
          (void**)&dependent_tasks));
      iree_task_t* task = task_head;
      for (iree_host_size_t i = 0; i < dependent_task_count; ++i) {
        dependent_tasks[i] = task;
        task = task->next_task;
      }
      iree_task_barrier_set_dependent_tasks(open_barrier, dependent_task_count,
                                            dependent_tasks);
    }
  }
  command_buffer->state.open_barrier = NULL;

  // Move the open tasks to the tail as they represent the first half of the
  // *next* barrier that will be inserted.
  if (command_buffer->state.open_task_count > 0) {
    iree_task_list_move(&command_buffer->state.open_tasks,
                        &command_buffer->leaf_tasks);
    command_buffer->state.open_task_count = 0;
  }

  return iree_ok_status();
}

// Emits a global barrier, splitting execution into all prior recorded tasks
// and all subsequent recorded tasks. This is currently the critical piece that
// limits our concurrency: changing to fine-grained barriers (via barrier
// buffers or events) will allow more work to overlap at the cost of more brain
// to build out the proper task graph.
static iree_status_t iree_hal_task_command_buffer_emit_global_barrier(
    iree_hal_task_command_buffer_t* command_buffer) {
  // Flush open tasks to the previous barrier. This resets our state such that
  // we can assign the new open barrier and start recording tasks for it.
  // Previous tasks will be moved into the leaf_tasks list.
  IREE_RETURN_IF_ERROR(
      iree_hal_task_command_buffer_flush_tasks(command_buffer));

  // Allocate the new open barrier.
  // As we are recording forward we can't yet assign the dependent tasks (the
  // second half of the synchronization domain) and instead are just inserting
  // it so we can setup the join from previous tasks (the first half of the
  // synchronization domain).
  iree_task_barrier_t* barrier = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(&command_buffer->arena,
                                           sizeof(*barrier), (void**)&barrier));
  iree_task_barrier_initialize_empty(command_buffer->scope, barrier);

  // If there were previous tasks then join them to the barrier.
  for (iree_task_t* task = iree_task_list_front(&command_buffer->leaf_tasks);
       task != NULL; task = task->next_task) {
    iree_task_set_completion_task(task, &barrier->header);
  }

  // Move the tasks from the leaf list (tail) to the root list (head) if this
  // was the first set of tasks recorded.
  if (iree_task_list_is_empty(&command_buffer->root_tasks) &&
      !iree_task_list_is_empty(&command_buffer->leaf_tasks)) {
    iree_task_list_move(&command_buffer->leaf_tasks,
                        &command_buffer->root_tasks);
  }

  // Reset the tail of the command buffer to the barrier. This leaves us in a
  // consistent state if the recording ends immediate after this (the barrier
  // will be the last task).
  iree_task_list_t* target_task_list =
      iree_task_list_is_empty(&command_buffer->root_tasks)
          ? &command_buffer->root_tasks
          : &command_buffer->leaf_tasks;
  iree_task_list_initialize(target_task_list);
  iree_task_list_push_back(target_task_list, &barrier->header);

  // NOTE: all new tasks emitted will be executed after this barrier.
  command_buffer->state.open_barrier = barrier;
  command_buffer->state.open_task_count = 0;

  return iree_ok_status();
}

// Emits a the given execution |task| into the current open synchronization
// scope (after state.open_barrier and before the next barrier).
static iree_status_t iree_hal_task_command_buffer_emit_execution_task(
    iree_hal_task_command_buffer_t* command_buffer, iree_task_t* task) {
  if (command_buffer->state.open_barrier == NULL) {
    // If there is no open barrier then we are at the head and going right into
    // the task DAG.
    iree_task_list_push_back(&command_buffer->leaf_tasks, task);
  } else {
    // Append to the open task list that will be flushed to the open barrier.
    iree_task_list_push_back(&command_buffer->state.open_tasks, task);
    ++command_buffer->state.open_task_count;
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_task_command_buffer_t execution
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_task_command_buffer_issue(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_task_queue_state_t* queue_state, iree_task_t* retire_task,
    iree_arena_allocator_t* arena, iree_task_submission_t* pending_submission) {
  iree_hal_task_command_buffer_t* command_buffer =
      iree_hal_task_command_buffer_cast(base_command_buffer);
  IREE_ASSERT_TRUE(command_buffer);

  // If the command buffer is empty (valid!) then we are a no-op.
  bool has_root_tasks = !iree_task_list_is_empty(&command_buffer->root_tasks);
  if (!has_root_tasks) {
    return iree_ok_status();
  }

  bool has_leaf_tasks = !iree_task_list_is_empty(&command_buffer->leaf_tasks);
  if (has_leaf_tasks) {
    // Chain the retire task onto the leaf tasks as their completion indicates
    // that all commands have completed.
    for (iree_task_t* task = command_buffer->leaf_tasks.head; task != NULL;
         task = task->next_task) {
      iree_task_set_completion_task(task, retire_task);
    }
  } else {
    // If we have no leaf tasks it means that this is a single layer DAG and
    // after the root tasks complete the entire command buffer has completed.
    for (iree_task_t* task = command_buffer->root_tasks.head; task != NULL;
         task = task->next_task) {
      iree_task_set_completion_task(task, retire_task);
    }
  }

  // Enqueue all root tasks that are ready to run immediately.
  // After this all of the command buffer tasks are owned by the submission and
  // we need to ensure the command buffer doesn't try to discard them.
  iree_task_submission_enqueue_list(pending_submission,
                                    &command_buffer->root_tasks);
  iree_task_list_initialize(&command_buffer->leaf_tasks);

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_task_command_buffer_t debug utilities
//===----------------------------------------------------------------------===//

static void iree_hal_task_command_buffer_begin_debug_group(
    iree_hal_command_buffer_t* base_command_buffer, iree_string_view_t label,
    iree_hal_label_color_t label_color,
    const iree_hal_label_location_t* location) {
  // TODO(benvanik): tracy event stack.
}

static void iree_hal_task_command_buffer_end_debug_group(
    iree_hal_command_buffer_t* base_command_buffer) {
  // TODO(benvanik): tracy event stack.
}

//===----------------------------------------------------------------------===//
// iree_hal_command_buffer_execution_barrier
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_task_command_buffer_execution_barrier(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_execution_barrier_flags_t flags,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  iree_hal_task_command_buffer_t* command_buffer =
      iree_hal_task_command_buffer_cast(base_command_buffer);

  // TODO(benvanik): actual DAG construction. Right now we are just doing simple
  // global barriers each time and forcing a join-fork point.
  return iree_hal_task_command_buffer_emit_global_barrier(command_buffer);
}

//===----------------------------------------------------------------------===//
// iree_hal_command_buffer_signal_event
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_task_command_buffer_signal_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  // TODO(#4518): implement events. For now we just insert global barriers.
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_command_buffer_reset_event
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_task_command_buffer_reset_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  // TODO(#4518): implement events. For now we just insert global barriers.
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_command_buffer_wait_events
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_task_command_buffer_wait_events(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_host_size_t event_count, const iree_hal_event_t** events,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  iree_hal_task_command_buffer_t* command_buffer =
      iree_hal_task_command_buffer_cast(base_command_buffer);
  // TODO(#4518): implement events. For now we just insert global barriers.
  return iree_hal_task_command_buffer_emit_global_barrier(command_buffer);
}

//===----------------------------------------------------------------------===//
// iree_hal_command_buffer_discard_buffer
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_task_command_buffer_discard_buffer(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_buffer_t* buffer) {
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_command_buffer_fill_buffer
//===----------------------------------------------------------------------===//
// NOTE: for large copies we dispatch this as tiles for parallelism.
// We'd want to do some measurement for when it's worth it; filling a 200KB
// buffer: maybe not, filling a 200MB buffer: yeah. For now we just do
// arbitrarily sized chunks.

// TODO(benvanik): make this a configurable setting. Must be aligned to pattern
// length so pick a power of two.
#define IREE_HAL_CMD_FILL_SLICE_LENGTH (128 * 1024)

typedef struct iree_hal_cmd_fill_buffer_t {
  iree_task_dispatch_t task;
  iree_hal_buffer_t* target_buffer;
  iree_device_size_t target_offset;
  iree_device_size_t length;
  uint32_t pattern_length;
  uint8_t pattern[8];
} iree_hal_cmd_fill_buffer_t;

static iree_status_t iree_hal_cmd_fill_tile(
    void* user_context, const iree_task_tile_context_t* tile_context,
    iree_task_submission_t* pending_submission) {
  const iree_hal_cmd_fill_buffer_t* cmd =
      (const iree_hal_cmd_fill_buffer_t*)user_context;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_device_size_t length_per_slice = tile_context->workgroup_size[0];
  iree_device_size_t slice_offset =
      (iree_device_size_t)tile_context->workgroup_xyz[0] * length_per_slice;
  iree_device_size_t remaining_length = cmd->length - slice_offset;
  iree_device_size_t slice_length =
      iree_min(length_per_slice, remaining_length);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (uint64_t)slice_length);

  iree_status_t status = iree_hal_buffer_map_fill(
      cmd->target_buffer, cmd->target_offset + slice_offset, slice_length,
      cmd->pattern, cmd->pattern_length);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_task_command_buffer_fill_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, const void* pattern,
    iree_host_size_t pattern_length) {
  iree_hal_task_command_buffer_t* command_buffer =
      iree_hal_task_command_buffer_cast(base_command_buffer);

  IREE_RETURN_IF_ERROR(iree_hal_resource_set_insert(
      command_buffer->resource_set, 1, &target_buffer));

  iree_hal_cmd_fill_buffer_t* cmd = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(&command_buffer->arena, sizeof(*cmd), (void**)&cmd));

  const uint32_t workgroup_size[3] = {
      /*x=*/IREE_HAL_CMD_FILL_SLICE_LENGTH,
      /*y=*/1,
      /*z=*/1,
  };
  const uint32_t workgroup_count[3] = {
      /*x=*/iree_device_size_ceil_div(length, workgroup_size[0]),
      /*y=*/1,
      /*z=*/1,
  };
  iree_task_dispatch_initialize(
      command_buffer->scope,
      iree_task_make_dispatch_closure(iree_hal_cmd_fill_tile, (void*)cmd),
      workgroup_size, workgroup_count, &cmd->task);
  cmd->target_buffer = target_buffer;
  cmd->target_offset = target_offset;
  cmd->length = length;
  memcpy(cmd->pattern, pattern, pattern_length);
  cmd->pattern_length = pattern_length;

  return iree_hal_task_command_buffer_emit_execution_task(command_buffer,
                                                          &cmd->task.header);
}

//===----------------------------------------------------------------------===//
// iree_hal_command_buffer_update_buffer
//===----------------------------------------------------------------------===//

typedef struct iree_hal_cmd_update_buffer_t {
  iree_task_call_t task;
  iree_hal_buffer_t* target_buffer;
  iree_device_size_t target_offset;
  iree_device_size_t length;
  uint8_t source_buffer[];
} iree_hal_cmd_update_buffer_t;

static iree_status_t iree_hal_cmd_update_buffer(
    void* user_context, iree_task_t* task,
    iree_task_submission_t* pending_submission) {
  const iree_hal_cmd_update_buffer_t* cmd =
      (const iree_hal_cmd_update_buffer_t*)user_context;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = iree_hal_buffer_map_write(
      cmd->target_buffer, cmd->target_offset, cmd->source_buffer, cmd->length);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_task_command_buffer_update_buffer(
    iree_hal_command_buffer_t* base_command_buffer, const void* source_buffer,
    iree_host_size_t source_offset, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_device_size_t length) {
  iree_hal_task_command_buffer_t* command_buffer =
      iree_hal_task_command_buffer_cast(base_command_buffer);

  IREE_RETURN_IF_ERROR(iree_hal_resource_set_insert(
      command_buffer->resource_set, 1, &target_buffer));

  iree_host_size_t total_cmd_size =
      sizeof(iree_hal_cmd_update_buffer_t) + length;

  iree_hal_cmd_update_buffer_t* cmd = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(&command_buffer->arena,
                                           total_cmd_size, (void**)&cmd));

  iree_task_call_initialize(
      command_buffer->scope,
      iree_task_make_call_closure(iree_hal_cmd_update_buffer, (void*)cmd),
      &cmd->task);
  cmd->target_buffer = target_buffer;
  cmd->target_offset = target_offset;
  cmd->length = length;

  memcpy(cmd->source_buffer, (const uint8_t*)source_buffer + source_offset,
         cmd->length);

  return iree_hal_task_command_buffer_emit_execution_task(command_buffer,
                                                          &cmd->task.header);
}

//===----------------------------------------------------------------------===//
// iree_hal_command_buffer_copy_buffer
//===----------------------------------------------------------------------===//
// NOTE: for large copies we dispatch this as tiles for parallelism.
// We'd want to do some measurement for when it's worth it; copying a 200KB
// buffer: maybe not, copying a 200MB buffer: yeah. For now we just do
// arbitrarily sized chunks.

// TODO(benvanik): make this a configurable setting. Must be aligned to pattern
// length so pick a power of two.
#define IREE_HAL_CMD_COPY_SLICE_LENGTH (128 * 1024)

typedef struct iree_hal_cmd_copy_buffer_t {
  iree_task_dispatch_t task;
  iree_hal_buffer_t* source_buffer;
  iree_device_size_t source_offset;
  iree_hal_buffer_t* target_buffer;
  iree_device_size_t target_offset;
  iree_device_size_t length;
} iree_hal_cmd_copy_buffer_t;

static iree_status_t iree_hal_cmd_copy_tile(
    void* user_context, const iree_task_tile_context_t* tile_context,
    iree_task_submission_t* pending_submission) {
  const iree_hal_cmd_copy_buffer_t* cmd =
      (const iree_hal_cmd_copy_buffer_t*)user_context;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_device_size_t length_per_slice = tile_context->workgroup_size[0];
  iree_device_size_t slice_offset =
      (iree_device_size_t)tile_context->workgroup_xyz[0] * length_per_slice;
  iree_device_size_t remaining_length = cmd->length - slice_offset;
  iree_device_size_t slice_length =
      iree_min(length_per_slice, remaining_length);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (uint64_t)slice_length);

  iree_status_t status = iree_hal_buffer_map_copy(
      cmd->source_buffer, cmd->source_offset + slice_offset, cmd->target_buffer,
      cmd->target_offset + slice_offset, slice_length);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_task_command_buffer_copy_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length) {
  iree_hal_task_command_buffer_t* command_buffer =
      iree_hal_task_command_buffer_cast(base_command_buffer);

  const iree_hal_buffer_t* buffers[2] = {source_buffer, target_buffer};
  IREE_RETURN_IF_ERROR(
      iree_hal_resource_set_insert(command_buffer->resource_set, 2, buffers));

  iree_hal_cmd_copy_buffer_t* cmd = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(&command_buffer->arena, sizeof(*cmd), (void**)&cmd));

  const uint32_t workgroup_size[3] = {
      /*x=*/IREE_HAL_CMD_COPY_SLICE_LENGTH,
      /*y=*/1,
      /*z=*/1,
  };
  const uint32_t workgroup_count[3] = {
      /*x=*/iree_device_size_ceil_div(length, workgroup_size[0]),
      /*y=*/1,
      /*z=*/1,
  };
  iree_task_dispatch_initialize(
      command_buffer->scope,
      iree_task_make_dispatch_closure(iree_hal_cmd_copy_tile, (void*)cmd),
      workgroup_size, workgroup_count, &cmd->task);
  cmd->source_buffer = source_buffer;
  cmd->source_offset = source_offset;
  cmd->target_buffer = target_buffer;
  cmd->target_offset = target_offset;
  cmd->length = length;

  return iree_hal_task_command_buffer_emit_execution_task(command_buffer,
                                                          &cmd->task.header);
}

//===----------------------------------------------------------------------===//
// iree_hal_command_buffer_collective
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_task_command_buffer_collective(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_channel_t* channel,
    iree_hal_collective_op_t op, uint32_t param,
    iree_hal_buffer_binding_t send_binding,
    iree_hal_buffer_binding_t recv_binding, iree_device_size_t element_count) {
  // The channel can be used as a vtable if we want to inject collective APIs -
  // the device creation function would set up the channel once and we'll
  // receive it here each time. When interacting with the task system we want to
  // get wait handles we can model with iree_task_wait_t.
  //
  // An example basic flow:
  //   insert iree_task_call_t:
  //     chains with prior commands and makes the collective API call
  //   insert iree_task_wait_t with API wait handle or our event:
  //     chains with call
  //
  // What we probably want to do, though, is group the commands based on
  // execution barriers. When a new collective command comes in we should
  // reserve an event from the event pool, create the call to issue the
  // collective operation, and then track the event in the command buffer state.
  // When another collective call comes in we'll do the same and append the
  // event. At the next execution barrier (or non-collective command) we'd
  // flush to a multi-wait on all of the pending events.
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "collectives not yet implemented on the task system");
}

//===----------------------------------------------------------------------===//
// iree_hal_command_buffer_push_constants
//===----------------------------------------------------------------------===//
// NOTE: command buffer state change only; enqueues no tasks.

static iree_status_t iree_hal_task_command_buffer_push_constants(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_pipeline_layout_t* pipeline_layout, iree_host_size_t offset,
    const void* values, iree_host_size_t values_length) {
  iree_hal_task_command_buffer_t* command_buffer =
      iree_hal_task_command_buffer_cast(base_command_buffer);

  if (IREE_UNLIKELY(offset + values_length >=
                    sizeof(command_buffer->state.push_constants))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "push constant range %" PRIhsz " (length=%" PRIhsz
                            ") out of range",
                            offset, values_length);
  }

  memcpy((uint8_t*)&command_buffer->state.push_constants + offset, values,
         values_length);

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_command_buffer_push_descriptor_set
//===----------------------------------------------------------------------===//
// NOTE: command buffer state change only; enqueues no tasks.

static iree_status_t iree_hal_task_command_buffer_push_descriptor_set(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_pipeline_layout_t* pipeline_layout, uint32_t set,
    iree_host_size_t binding_count,
    const iree_hal_descriptor_set_binding_t* bindings) {
  iree_hal_task_command_buffer_t* command_buffer =
      iree_hal_task_command_buffer_cast(base_command_buffer);

  if (IREE_UNLIKELY(set >= IREE_HAL_LOCAL_MAX_DESCRIPTOR_SET_COUNT)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "set %u out of bounds", set);
  }

  iree_host_size_t binding_base =
      set * IREE_HAL_LOCAL_MAX_DESCRIPTOR_BINDING_COUNT;
  for (iree_host_size_t i = 0; i < binding_count; ++i) {
    if (IREE_UNLIKELY(bindings[i].binding >=
                      IREE_HAL_LOCAL_MAX_DESCRIPTOR_BINDING_COUNT)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "buffer binding index out of bounds");
    }
    iree_host_size_t binding_ordinal = binding_base + bindings[i].binding;

    // TODO(benvanik): batch insert by getting the resources in their own list.
    IREE_RETURN_IF_ERROR(iree_hal_resource_set_insert(
        command_buffer->resource_set, 1, &bindings[i].buffer));

    // TODO(benvanik): track mapping so we can properly map/unmap/flush/etc.
    iree_hal_buffer_mapping_t buffer_mapping = {{0}};
    if (bindings[i].buffer) {
      IREE_RETURN_IF_ERROR(iree_hal_buffer_map_range(
          bindings[i].buffer, IREE_HAL_MAPPING_MODE_PERSISTENT,
          IREE_HAL_MEMORY_ACCESS_ANY, bindings[i].offset, bindings[i].length,
          &buffer_mapping));
      command_buffer->state.bindings[binding_ordinal] =
          buffer_mapping.contents.data;
      command_buffer->state.binding_lengths[binding_ordinal] =
          buffer_mapping.contents.data_length;
    } else {
      // TODO(#10144): stash indirect binding reference in the state table.
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "binding table lookup not yet supported");
    }
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_command_buffer_dispatch
//===----------------------------------------------------------------------===//

typedef struct iree_hal_cmd_dispatch_t {
  iree_task_dispatch_t task;
  iree_hal_local_executable_t* executable;
  int32_t ordinal;

  // Total number of available 4 byte push constant values in |push_constants|.
  uint16_t push_constant_count;

  // Total number of binding base pointers in |binding_ptrs| and
  // |binding_lengths|. The set is packed densely based on which binidngs are
  // used (known at compile-time).
  uint16_t binding_count;

  // Following this structure in memory there are 3 tables:
  // - const uint32_t push_constants[push_constant_count];
  // - void* binding_ptrs[binding_count];
  // - const size_t binding_lengths[binding_count];
} iree_hal_cmd_dispatch_t;

static iree_status_t iree_hal_cmd_dispatch_tile(
    void* user_context, const iree_task_tile_context_t* tile_context,
    iree_task_submission_t* pending_submission) {
  const iree_hal_cmd_dispatch_t* cmd =
      (const iree_hal_cmd_dispatch_t*)user_context;
  IREE_TRACE_ZONE_BEGIN(z0);

  // We could share this across all workgroups in a dispatch and reduce cache
  // pressure as all cores would be hitting the same hot read-only cache line.
  // It'd grow the size of iree_hal_cmd_dispatch_t by a few dozen bytes, though,
  // and so we'd need some profiling to see if it's worth it (fixed command
  // buffer cost vs potential for saving a cache miss or two).
  iree_alignas(64) iree_hal_executable_dispatch_state_v0_t dispatch_state = {
      .workgroup_size_x = tile_context->workgroup_size[0],
      .workgroup_size_y = tile_context->workgroup_size[1],
      .workgroup_size_z = tile_context->workgroup_size[2],
      .push_constant_count = cmd->push_constant_count,
      .workgroup_count_x = tile_context->workgroup_count[0],
      .workgroup_count_y = tile_context->workgroup_count[1],
      .workgroup_count_z = tile_context->workgroup_count[2],
      .max_concurrency =
          iree_task_affinity_set_count_ones(cmd->task.header.affinity_set),
      .binding_count = cmd->binding_count,
  };
  uint8_t* cmd_ptr = (uint8_t*)cmd + sizeof(*cmd);
  dispatch_state.push_constants = (uint32_t*)cmd_ptr;
  cmd_ptr += cmd->push_constant_count * sizeof(*dispatch_state.push_constants);
  dispatch_state.binding_ptrs = (void**)cmd_ptr;
  cmd_ptr += cmd->binding_count * sizeof(*dispatch_state.binding_ptrs);
  dispatch_state.binding_lengths = (size_t*)cmd_ptr;
  cmd_ptr += cmd->binding_count * sizeof(*dispatch_state.binding_lengths);

  const iree_alignas(64)
      iree_hal_executable_workgroup_state_v0_t workgroup_state = {
          .workgroup_id_x = tile_context->workgroup_xyz[0],
          .workgroup_id_y = tile_context->workgroup_xyz[1],
          .workgroup_id_z = tile_context->workgroup_xyz[2],
          .reserved = 0,
          .processor_id = tile_context->processor_id,
          .local_memory = tile_context->local_memory.data,
          .local_memory_size = (size_t)tile_context->local_memory.data_length,
      };
  iree_status_t status = iree_hal_local_executable_issue_call(
      cmd->executable, cmd->ordinal, &dispatch_state, &workgroup_state,
      tile_context->worker_id);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_task_command_buffer_build_dispatch(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_t* executable, int32_t entry_point,
    uint32_t workgroup_x, uint32_t workgroup_y, uint32_t workgroup_z,
    iree_hal_cmd_dispatch_t** out_cmd) {
  iree_hal_task_command_buffer_t* command_buffer =
      iree_hal_task_command_buffer_cast(base_command_buffer);

  iree_hal_local_executable_t* local_executable =
      iree_hal_local_executable_cast(executable);
  if (IREE_UNLIKELY(!local_executable->pipeline_layouts)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "layouts not provided during executable creation; cannot dispatch");
  }

  iree_hal_local_pipeline_layout_t* local_layout =
      (iree_hal_local_pipeline_layout_t*)
          local_executable->pipeline_layouts[entry_point];
  iree_host_size_t push_constant_count = local_layout->push_constants;
  iree_hal_local_binding_mask_t used_binding_mask = local_layout->used_bindings;
  iree_host_size_t used_binding_count =
      iree_math_count_ones_u64(used_binding_mask);

  // To save a few command buffer bytes we narrow these:
  if (IREE_UNLIKELY(push_constant_count >= UINT16_MAX) ||
      IREE_UNLIKELY(used_binding_count >= UINT16_MAX)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "too many bindings/push constants");
  }

  iree_hal_cmd_dispatch_t* cmd = NULL;
  iree_host_size_t total_cmd_size =
      sizeof(*cmd) + push_constant_count * sizeof(uint32_t) +
      used_binding_count * sizeof(void*) +
      used_binding_count * sizeof(iree_device_size_t);
  IREE_RETURN_IF_ERROR(iree_arena_allocate(&command_buffer->arena,
                                           total_cmd_size, (void**)&cmd));

  cmd->executable = local_executable;
  cmd->ordinal = entry_point;
  cmd->push_constant_count = push_constant_count;
  cmd->binding_count = used_binding_count;

  const uint32_t workgroup_count[3] = {workgroup_x, workgroup_y, workgroup_z};
  // TODO(benvanik): expose on API or keep fixed on executable.
  const uint32_t workgroup_size[3] = {1, 1, 1};
  iree_task_dispatch_initialize(
      command_buffer->scope,
      iree_task_make_dispatch_closure(iree_hal_cmd_dispatch_tile, (void*)cmd),
      workgroup_size, workgroup_count, &cmd->task);

  // Tell the task system how much workgroup local memory is required for the
  // dispatch; each invocation of the entry point will have at least as much
  // scratch memory available during execution.
  cmd->task.local_memory_size =
      local_executable->dispatch_attrs
          ? local_executable->dispatch_attrs[entry_point].local_memory_pages *
                IREE_HAL_WORKGROUP_LOCAL_MEMORY_PAGE_SIZE
          : 0;

  // Copy only the push constant range used by the executable.
  uint8_t* cmd_ptr = (uint8_t*)cmd + sizeof(*cmd);
  uint32_t* push_constants = (uint32_t*)cmd_ptr;
  memcpy(push_constants, command_buffer->state.push_constants,
         push_constant_count * sizeof(*push_constants));
  cmd_ptr += push_constant_count * sizeof(*push_constants);

  // Produce the dense binding list based on the declared bindings used.
  // This allows us to change the descriptor sets and bindings counts supported
  // in the HAL independent of any executable as each executable just gets the
  // flat dense list and doesn't care about our descriptor set stuff.
  //
  // Note that we are just directly setting the binding data pointers here with
  // no ownership/retaining/etc - it's part of the HAL contract that buffers are
  // kept valid for the duration they may be in use.
  void** binding_ptrs = (void**)cmd_ptr;
  cmd_ptr += used_binding_count * sizeof(*binding_ptrs);
  size_t* binding_lengths = (size_t*)cmd_ptr;
  cmd_ptr += used_binding_count * sizeof(*binding_lengths);
  iree_host_size_t binding_base = 0;
  for (iree_host_size_t i = 0; i < used_binding_count; ++i) {
    int mask_offset = iree_math_count_trailing_zeros_u64(used_binding_mask);
    int binding_ordinal = binding_base + mask_offset;
    binding_base += mask_offset + 1;
    used_binding_mask = iree_shr(used_binding_mask, mask_offset + 1);
    binding_ptrs[i] = command_buffer->state.bindings[binding_ordinal];
    binding_lengths[i] = command_buffer->state.binding_lengths[binding_ordinal];
    if (!binding_ptrs[i]) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "(flat) binding %d is NULL", binding_ordinal);
    }
  }

  *out_cmd = cmd;
  return iree_hal_task_command_buffer_emit_execution_task(command_buffer,
                                                          &cmd->task.header);
}

static iree_status_t iree_hal_task_command_buffer_dispatch(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_t* executable, int32_t entry_point,
    uint32_t workgroup_x, uint32_t workgroup_y, uint32_t workgroup_z) {
  iree_hal_task_command_buffer_t* command_buffer =
      iree_hal_task_command_buffer_cast(base_command_buffer);
  IREE_RETURN_IF_ERROR(iree_hal_resource_set_insert(
      command_buffer->resource_set, 1, &executable));
  iree_hal_cmd_dispatch_t* cmd = NULL;
  return iree_hal_task_command_buffer_build_dispatch(
      base_command_buffer, executable, entry_point, workgroup_x, workgroup_y,
      workgroup_z, &cmd);
}

static iree_status_t iree_hal_task_command_buffer_dispatch_indirect(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_t* executable, int32_t entry_point,
    iree_hal_buffer_t* workgroups_buffer,
    iree_device_size_t workgroups_offset) {
  iree_hal_task_command_buffer_t* command_buffer =
      iree_hal_task_command_buffer_cast(base_command_buffer);

  const void* resources[2] = {executable, workgroups_buffer};
  IREE_RETURN_IF_ERROR(
      iree_hal_resource_set_insert(command_buffer->resource_set, 2, resources));

  // TODO(benvanik): track mapping so we can properly map/unmap/flush/etc.
  iree_hal_buffer_mapping_t buffer_mapping = {{0}};
  IREE_RETURN_IF_ERROR(iree_hal_buffer_map_range(
      workgroups_buffer, IREE_HAL_MAPPING_MODE_PERSISTENT,
      IREE_HAL_MEMORY_ACCESS_READ, workgroups_offset, 3 * sizeof(uint32_t),
      &buffer_mapping));

  iree_hal_cmd_dispatch_t* cmd = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_task_command_buffer_build_dispatch(
      base_command_buffer, executable, entry_point, 0, 0, 0, &cmd));
  cmd->task.workgroup_count.ptr = (const uint32_t*)buffer_mapping.contents.data;
  cmd->task.header.flags |= IREE_TASK_FLAG_DISPATCH_INDIRECT;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_command_buffer_execute_commands
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_task_command_buffer_execute_commands(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_command_buffer_t* base_commands,
    iree_hal_buffer_binding_table_t binding_table) {
  // TODO(#10144): support indirect command buffers by using deferred command
  // buffers or caching the task topology (probably not worth the tracking).
  // If we could separate the topology that referenced the binding table we'd
  // be able to reissue but not concurrently (as each task can only be in flight
  // as a singleton) - which may be enough in many cases but adds complexity to
  // tracking as we'd need to either enforce serialization of subsequent
  // submissions or copy-on-write-style clone the topology for each additional
  // concurrent submission.
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "indirect command buffers not yet implemented");
}

//===----------------------------------------------------------------------===//
// iree_hal_command_buffer_vtable_t
//===----------------------------------------------------------------------===//

static const iree_hal_command_buffer_vtable_t
    iree_hal_task_command_buffer_vtable = {
        .destroy = iree_hal_task_command_buffer_destroy,
        .begin = iree_hal_task_command_buffer_begin,
        .end = iree_hal_task_command_buffer_end,
        .begin_debug_group = iree_hal_task_command_buffer_begin_debug_group,
        .end_debug_group = iree_hal_task_command_buffer_end_debug_group,
        .execution_barrier = iree_hal_task_command_buffer_execution_barrier,
        .signal_event = iree_hal_task_command_buffer_signal_event,
        .reset_event = iree_hal_task_command_buffer_reset_event,
        .wait_events = iree_hal_task_command_buffer_wait_events,
        .discard_buffer = iree_hal_task_command_buffer_discard_buffer,
        .fill_buffer = iree_hal_task_command_buffer_fill_buffer,
        .update_buffer = iree_hal_task_command_buffer_update_buffer,
        .copy_buffer = iree_hal_task_command_buffer_copy_buffer,
        .collective = iree_hal_task_command_buffer_collective,
        .push_constants = iree_hal_task_command_buffer_push_constants,
        .push_descriptor_set = iree_hal_task_command_buffer_push_descriptor_set,
        .dispatch = iree_hal_task_command_buffer_dispatch,
        .dispatch_indirect = iree_hal_task_command_buffer_dispatch_indirect,
        .execute_commands = iree_hal_task_command_buffer_execute_commands,
};
