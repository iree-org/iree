// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/task/task.h"

#include <stdio.h>
#include <string.h>

#include "iree/task/list.h"
#include "iree/task/pool.h"
#include "iree/task/post_batch.h"
#include "iree/task/scope.h"
#include "iree/task/submission.h"
#include "iree/task/task_impl.h"
#include "iree/task/tuning.h"

//==============================================================================
// Task bookkeeping
//==============================================================================

void iree_task_initialize(iree_task_type_t type, iree_task_scope_t* scope,
                          iree_task_t* out_task) {
  // NOTE: only clears the header, not the task body.
  memset(out_task, 0, sizeof(*out_task));
  out_task->scope = scope;
  out_task->affinity_set = iree_task_affinity_for_any_worker();
  out_task->type = type;
}

void iree_task_set_cleanup_fn(iree_task_t* task,
                              iree_task_cleanup_fn_t cleanup_fn) {
  task->cleanup_fn = cleanup_fn;
}

void iree_task_set_completion_task(iree_task_t* task,
                                   iree_task_t* completion_task) {
  IREE_ASSERT(!task->completion_task);
  task->completion_task = completion_task;
  iree_atomic_fetch_add_int32(&completion_task->pending_dependency_count, 1,
                              iree_memory_order_acq_rel);
}

bool iree_task_is_ready(iree_task_t* task) {
  if (iree_atomic_load_int32(&task->pending_dependency_count,
                             iree_memory_order_acquire) > 0) {
    // At least one dependency is still pending.
    return false;
  }
  return true;
}

static void iree_task_try_set_status(iree_atomic_intptr_t* permanent_status,
                                     iree_status_t new_status) {
  if (IREE_UNLIKELY(iree_status_is_ok(new_status))) return;

  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, "failed: ");
  IREE_TRACE_ZONE_APPEND_TEXT(
      z0, iree_status_code_string(iree_status_code(new_status)));

  iree_status_t old_status = iree_ok_status();
  if (!iree_atomic_compare_exchange_strong_intptr(
          permanent_status, (intptr_t*)&old_status, (intptr_t)new_status,
          iree_memory_order_acq_rel,
          iree_memory_order_relaxed /* old_status is unused */)) {
    // Previous status was not OK; drop our new status.
    IREE_IGNORE_ERROR(new_status);
  }

  IREE_TRACE_ZONE_END(z0);
}

static void iree_task_cleanup(iree_task_t* task,
                              iree_status_code_t status_code) {
  // Call the (optional) cleanup function.
  // NOTE: this may free the memory of the task itself!
  iree_task_pool_t* pool = task->pool;
  iree_task_cleanup_fn_t cleanup_fn = task->cleanup_fn;
  if (cleanup_fn) {
    cleanup_fn(task, status_code);
  }

  // Return the task to the pool it was allocated from.
  // Some tasks are allocated as part of arenas/ringbuffers and won't have a
  // pool as they'll be cleaned up as part of a larger operation.
  if (pool) {
    iree_task_pool_release(pool, task);
  }
}

static void iree_task_barrier_discard(iree_task_barrier_t* task,
                                      iree_task_list_t* discard_worklist);

void iree_task_discard(iree_task_t* task, iree_task_list_t* discard_worklist) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // This models a BFS discard in our non-recursive approach.
  // We must ensure that we only discard each task once and that we discard the
  // tasks in the appropriate order: if we had a DAG of A -> B, C -> D we must
  // discard respecting the same topological ordering.

  IREE_ASSERT_EQ(0, iree_atomic_load_int32(&task->pending_dependency_count,
                                           iree_memory_order_acquire));

  // Almost all tasks will have a completion task; some may have additional
  // dependent tasks (like barriers) that will be handled below.
  const bool completion_task_ready =
      task->completion_task &&
      iree_atomic_fetch_sub_int32(
          &task->completion_task->pending_dependency_count, 1,
          iree_memory_order_acq_rel) == 1;
  if (completion_task_ready) {
    iree_task_list_push_back(discard_worklist, task->completion_task);
  }

  iree_task_scope_t* end_scope = NULL;
  switch (task->type) {
    default:
    case IREE_TASK_TYPE_NOP:
    case IREE_TASK_TYPE_CALL:
      break;
    case IREE_TASK_TYPE_BARRIER:
      iree_task_barrier_discard((iree_task_barrier_t*)task, discard_worklist);
      break;
    case IREE_TASK_TYPE_FENCE:
      end_scope = task->scope;  // need to clean up the task first
      break;
    case IREE_TASK_TYPE_WAIT:
    case IREE_TASK_TYPE_DISPATCH:
      break;
  }

  iree_task_cleanup(task, IREE_STATUS_ABORTED);
  // NOTE: task is invalidated here and cannot be used!
  task = NULL;

  if (end_scope) {
    iree_task_scope_end(end_scope);
  }

  IREE_TRACE_ZONE_END(z0);
}

static void iree_task_retire(iree_task_t* task,
                             iree_task_submission_t* pending_submission,
                             iree_status_t status) {
  IREE_ASSERT_EQ(0, iree_atomic_load_int32(&task->pending_dependency_count,
                                           iree_memory_order_acquire));

  // Decrement the pending count on the completion task, if any.
  iree_task_t* completion_task = task->completion_task;
  task->completion_task = NULL;

  if (iree_status_is_ok(status)) {
    // Task completed successfully.
    iree_task_cleanup(task, IREE_STATUS_OK);
    bool completion_task_ready =
        completion_task &&
        iree_atomic_fetch_sub_int32(&completion_task->pending_dependency_count,
                                    1, iree_memory_order_acq_rel) == 1;
    if (completion_task_ready) {
      // This was the last pending dependency and the completion task is ready
      // to run.
      iree_task_submission_enqueue(pending_submission, completion_task);
    }
  } else {
    // Task failed: notify the scope.
    iree_task_scope_t* scope = task->scope;
    iree_task_scope_fail(scope, status);
    status = iree_ok_status();  // consumed by the fail

    // We need to carefully clean up the task: if we go discarding fences we'll
    // end up waking waiters before we're done. To ensure this doesn't happen
    // we retain the scope until we've finished cleaning things up.
    iree_task_scope_begin(scope);
    iree_task_cleanup(task, IREE_STATUS_ABORTED);

    bool completion_task_ready =
        completion_task &&
        iree_atomic_fetch_sub_int32(&completion_task->pending_dependency_count,
                                    1, iree_memory_order_acq_rel) == 1;
    if (completion_task_ready) {
      // This was the last pending dependency and we know that we can safely
      // abort the completion task by discarding.
      iree_task_list_t discard_worklist;
      iree_task_list_initialize(&discard_worklist);
      iree_task_discard(completion_task, &discard_worklist);
      iree_task_list_discard(&discard_worklist);
    } else if (completion_task) {
      // One or more pending dependencies are not yet satisfied and the
      // completion task must stay alive. We can mark it as aborted, though,
      // so that it knows not to execute when it is ready to run.
      // TODO(benvanik): make this atomic? we only ever add bits and it's safe
      // for it to run if we got this far.
      completion_task->flags |= IREE_TASK_FLAG_ABORTED;
    }

    // Unlock the scope; it may immediately be freed before this returns!
    iree_task_scope_end(scope);
  }

  // NOTE: task is invalidated here and cannot be used!
  task = NULL;
}

//==============================================================================
// IREE_TASK_TYPE_NOP
//==============================================================================

void iree_task_nop_initialize(iree_task_scope_t* scope,
                              iree_task_nop_t* out_task) {
  iree_task_initialize(IREE_TASK_TYPE_NOP, scope, &out_task->header);
}

void iree_task_nop_retire(iree_task_nop_t* task,
                          iree_task_submission_t* pending_submission) {
  iree_task_retire(&task->header, pending_submission, iree_ok_status());
}

//==============================================================================
// IREE_TASK_TYPE_CALL
//==============================================================================

// Returns an XXBBGGRR color (red in the lowest bits).
// Must not be 0 (tracy will ignore).
static uint32_t iree_math_ptr_to_xrgb(const void* ptr) {
  // This is just a simple hack to give us a unique(ish) per-pointer color.
  // It's only to make it easier to distinguish which tiles are from the same
  // dispatch.
  uint64_t ptr64 = (uintptr_t)ptr;
  return (uint32_t)ptr64 ^ (uint32_t)(ptr64 >> 32);
}

void iree_task_call_initialize(iree_task_scope_t* scope,
                               iree_task_call_closure_t closure,
                               iree_task_call_t* out_task) {
  iree_task_initialize(IREE_TASK_TYPE_CALL, scope, &out_task->header);
  out_task->closure = closure;
  iree_atomic_store_intptr(&out_task->status, 0, iree_memory_order_release);
}

void iree_task_call_execute(iree_task_call_t* task,
                            iree_task_submission_t* pending_submission) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_SET_COLOR(z0,
                            iree_math_ptr_to_xrgb(task->closure.user_context));

  if (IREE_LIKELY(
          !iree_any_bit_set(task->header.flags, IREE_TASK_FLAG_ABORTED))) {
    // Execute the user callback.
    // Note that this may enqueue more nested tasks, including tasks that
    // prevent this task from retiring.
    iree_status_t status = task->closure.fn(task->closure.user_context,
                                            &task->header, pending_submission);
    if (!iree_status_is_ok(status)) {
      // Stash the failure status on the task.
      // If there's still pending dependencies we won't be able to discard
      // immediately and need to keep the status around until they all complete.
      iree_task_try_set_status(&task->status, status);
      status = iree_ok_status();  // consumed by try_set_status

      // TODO(benvanik): discard pending_submission? As we may have pending work
      // from multiple scopes it's dangerous to discard all. We could filter
      // based on scope, though, and if we did that we (probably) wouldn't need
      // to handle the permanent status on the task and could discard
      // immediately.
    }
  }

  // Check to see if there are no pending dependencies before retiring; the
  // dependency count can go up if new nested tasks were enqueued.
  if (iree_atomic_load_int32(&task->header.pending_dependency_count,
                             iree_memory_order_acquire) == 0) {
    iree_status_t status = (iree_status_t)iree_atomic_exchange_intptr(
        &task->status, 0, iree_memory_order_acq_rel);
    iree_task_retire(&task->header, pending_submission, status);
  }

  IREE_TRACE_ZONE_END(z0);
}

//==============================================================================
// IREE_TASK_TYPE_BARRIER
//==============================================================================

void iree_task_barrier_initialize(iree_task_scope_t* scope,
                                  iree_host_size_t dependent_task_count,
                                  iree_task_t* const* dependent_tasks,
                                  iree_task_barrier_t* out_task) {
  iree_task_initialize(IREE_TASK_TYPE_BARRIER, scope, &out_task->header);
  out_task->dependent_task_count = dependent_task_count;
  out_task->dependent_tasks = dependent_tasks;
  for (iree_host_size_t i = 0; i < out_task->dependent_task_count; ++i) {
    iree_task_t* dependent_task = out_task->dependent_tasks[i];
    iree_atomic_fetch_add_int32(&dependent_task->pending_dependency_count, 1,
                                iree_memory_order_acq_rel);
  }
}

void iree_task_barrier_initialize_empty(iree_task_scope_t* scope,
                                        iree_task_barrier_t* out_task) {
  iree_task_initialize(IREE_TASK_TYPE_BARRIER, scope, &out_task->header);
  out_task->dependent_task_count = 0;
  out_task->dependent_tasks = NULL;
}

void iree_task_barrier_set_dependent_tasks(
    iree_task_barrier_t* task, iree_host_size_t dependent_task_count,
    iree_task_t* const* dependent_tasks) {
  task->dependent_task_count = dependent_task_count;
  task->dependent_tasks = dependent_tasks;
  for (iree_host_size_t i = 0; i < task->dependent_task_count; ++i) {
    iree_task_t* dependent_task = task->dependent_tasks[i];
    iree_atomic_fetch_add_int32(&dependent_task->pending_dependency_count, 1,
                                iree_memory_order_acq_rel);
  }
}

static void iree_task_barrier_discard(iree_task_barrier_t* task,
                                      iree_task_list_t* discard_worklist) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Discard all of the tasks after the barrier.
  // Note that we need to ensure we only enqueue them for discard after all of
  // their dependencies have been met - otherwise we'll double-discard.
  for (iree_host_size_t i = 0; i < task->dependent_task_count; ++i) {
    iree_task_t* dependent_task = task->dependent_tasks[i];
    const bool dependent_task_ready =
        iree_atomic_fetch_sub_int32(&dependent_task->pending_dependency_count,
                                    1, iree_memory_order_acq_rel) == 1;
    if (dependent_task_ready) {
      // The dependent task has retired and can now be discard.
      iree_task_list_push_back(discard_worklist, dependent_task);
    }
  }

  IREE_TRACE_ZONE_END(z0);
}

void iree_task_barrier_retire(iree_task_barrier_t* task,
                              iree_task_submission_t* pending_submission) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // NOTE: we walk in reverse so that we enqueue in LIFO order.
  for (iree_host_size_t i = 0; i < task->dependent_task_count; ++i) {
    iree_task_t* dependent_task =
        task->dependent_tasks[task->dependent_task_count - i - 1];
    if (iree_atomic_fetch_sub_int32(&dependent_task->pending_dependency_count,
                                    1, iree_memory_order_acq_rel) == 1) {
      // The dependent task has retired and can now be made ready.
      iree_task_submission_enqueue(pending_submission, dependent_task);
    }
  }

  iree_task_retire(&task->header, pending_submission, iree_ok_status());

  IREE_TRACE_ZONE_END(z0);
}

//==============================================================================
// IREE_TASK_TYPE_FENCE
//==============================================================================

void iree_task_fence_initialize(iree_task_scope_t* scope,
                                iree_wait_primitive_t signal_handle,
                                iree_task_fence_t* out_task) {
  iree_task_initialize(IREE_TASK_TYPE_FENCE, scope, &out_task->header);
  out_task->signal_handle = signal_handle;
  iree_task_scope_begin(scope);
}

void iree_task_fence_retire(iree_task_fence_t* task,
                            iree_task_submission_t* pending_submission) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Need to wait until after we clean up the task before ending the scope.
  // This way anyone waiting on the scope to go idle will be able to ensure the
  // scope is actually idle - otherwise it may try to free the task memory
  // while we are still using it.
  iree_task_scope_t* end_scope = task->header.scope;

  // TODO(benvanik): better API that doesn't require wrapping or requiring that
  // iree_event_t is an iree_wait_handle_t.
  iree_wait_handle_t signal_handle = {
      .type = task->signal_handle.type,
      .value = task->signal_handle.value,
  };
  iree_event_set(&signal_handle);

  iree_task_retire(&task->header, pending_submission, iree_ok_status());

  if (end_scope) {
    iree_task_scope_end(end_scope);
  }

  IREE_TRACE_ZONE_END(z0);
}

//==============================================================================
// IREE_TASK_TYPE_WAIT
//==============================================================================

void iree_task_wait_initialize(iree_task_scope_t* scope,
                               iree_wait_source_t wait_source,
                               iree_time_t deadline_ns,
                               iree_task_wait_t* out_task) {
  iree_task_initialize(IREE_TASK_TYPE_WAIT, scope, &out_task->header);
  out_task->wait_source = wait_source;
  out_task->deadline_ns = deadline_ns;
  out_task->cancellation_flag = NULL;
}

void iree_task_wait_initialize_delay(iree_task_scope_t* scope,
                                     iree_time_t deadline_ns,
                                     iree_task_wait_t* out_task) {
  iree_task_wait_initialize(scope, iree_wait_source_delay(deadline_ns),
                            IREE_TIME_INFINITE_FUTURE, out_task);
}

void iree_task_wait_set_wait_any(iree_task_wait_t* task,
                                 iree_atomic_int32_t* cancellation_flag) {
  task->header.flags |= IREE_TASK_FLAG_WAIT_ANY;
  task->cancellation_flag = cancellation_flag;
}

void iree_task_wait_retire(iree_task_wait_t* task,
                           iree_task_submission_t* pending_submission,
                           iree_status_t status) {
  IREE_TRACE_ZONE_BEGIN(z0);

  task->header.flags &= ~IREE_TASK_FLAG_WAIT_COMPLETED;  // reset for future use

  // TODO(benvanik): allow deinit'ing the wait handle (if transient/from the
  // executor event pool).
  iree_task_retire(&task->header, pending_submission, status);

  IREE_TRACE_ZONE_END(z0);
}

//==============================================================================
// IREE_TASK_TYPE_DISPATCH_* utilities
//==============================================================================

// Returns an XXBBGGRR color (red in the lowest bits).
// Must not be 0 (tracy will ignore).
static uint32_t iree_task_tile_to_color(
    const iree_task_tile_context_t* tile_context);

#if defined(IREE_TASK_TRACING_PER_TILE_COLORS)

// TODO(#4017): optimize this to compute entire slices at once and fold in the
// work grid location code.
static uint32_t iree_math_hsv_to_xrgb(const uint8_t h, const uint8_t s,
                                      const uint8_t v) {
  // NOTE: this is matching with tracy's TracyColor.cpp implementation so that
  // our colors fit nicely in the UI.
  const uint8_t reg = h / 43;
  const uint8_t rem = (h - (reg * 43)) * 6;
  const uint8_t p = (v * (255 - s)) >> 8;
  const uint8_t q = (v * (255 - ((s * rem) >> 8))) >> 8;
  const uint8_t t = (v * (255 - ((s * (255 - rem)) >> 8))) >> 8;

  // clang-format off
  uint8_t r, g, b;
  switch (reg) {
    case 0:  r = v; g = t; b = p; break;
    case 1:  r = q; g = v; b = p; break;
    case 2:  r = p; g = v; b = t; break;
    case 3:  r = p; g = q; b = v; break;
    case 4:  r = t; g = p; b = v; break;
    default: r = v; g = p; b = q; break;
  }
  // clang-format on

  uint32_t xrgb = (r << 16) | (g << 8) | b;
  xrgb |= (xrgb ? 0 : 1);  // ensure never zero
  return xrgb;
}

static uint32_t iree_task_tile_to_color(
    const iree_task_tile_context_t* tile_context) {
  // TODO(#4017): optimize such that it's always on when tracing is
  // enabled by amortizing the cost across the entire slice.

  // Picked to try to make it easy to see gradients from tiles along the same x,
  // y, and z (in that order). x is the fastest changing dimension and as such
  // should all have the same hue, while z is the slowest changing dimension and
  // should have different hues.
  uint8_t h = (tile_context->workgroup_xyz[1] /
               (float)(tile_context->workgroup_count[1])) *
              255;
  h = (h * 11400714819323198485ull) & 0xFF;
  uint8_t s = 100 - (tile_context->workgroup_xyz[2] /
                     (float)(tile_context->workgroup_count[2])) *
                        100;
  uint8_t v = (tile_context->workgroup_xyz[0] /
               (float)(tile_context->workgroup_count[0])) *
                  50 +
              50;
  return iree_math_hsv_to_xrgb(h, s, v);
}

#else

static uint32_t iree_task_tile_to_color(
    const iree_task_tile_context_t* tile_context) {
  return 0;  // use default tracy colors
}

#endif  // IREE_TASK_TRACING_PER_TILE_COLORS

void iree_task_dispatch_statistics_merge(
    const iree_task_dispatch_statistics_t* source,
    iree_task_dispatch_statistics_t* target) {
  // TODO(benvanik): statistics.
}

//==============================================================================
// IREE_TASK_TYPE_DISPATCH
//==============================================================================

static void iree_task_dispatch_initialize_base(
    iree_task_scope_t* scope, iree_task_dispatch_closure_t closure,
    const uint32_t workgroup_size[3], iree_task_dispatch_t* out_task) {
  iree_task_initialize(IREE_TASK_TYPE_DISPATCH, scope, &out_task->header);
  out_task->closure = closure;
  memcpy(out_task->workgroup_size, workgroup_size,
         sizeof(out_task->workgroup_size));
  out_task->local_memory_size = 0;
  iree_atomic_store_intptr(&out_task->status, 0, iree_memory_order_release);
  memset(&out_task->statistics, 0, sizeof(out_task->statistics));

  IREE_TRACE({
    static iree_atomic_int64_t next_dispatch_id = IREE_ATOMIC_VAR_INIT(0);
    out_task->dispatch_id = iree_atomic_fetch_add_int64(
        &next_dispatch_id, 1ll, iree_memory_order_acq_rel);
  });
}

void iree_task_dispatch_initialize(iree_task_scope_t* scope,
                                   iree_task_dispatch_closure_t closure,
                                   const uint32_t workgroup_size[3],
                                   const uint32_t workgroup_count[3],
                                   iree_task_dispatch_t* out_task) {
  iree_task_dispatch_initialize_base(scope, closure, workgroup_size, out_task);
  memcpy(out_task->workgroup_count.value, workgroup_count,
         sizeof(out_task->workgroup_count.value));
}

void iree_task_dispatch_initialize_indirect(
    iree_task_scope_t* scope, iree_task_dispatch_closure_t closure,
    const uint32_t workgroup_size[3], const uint32_t* workgroup_count_ptr,
    iree_task_dispatch_t* out_task) {
  iree_task_dispatch_initialize_base(scope, closure, workgroup_size, out_task);
  out_task->header.flags |= IREE_TASK_FLAG_DISPATCH_INDIRECT;
  out_task->workgroup_count.ptr = workgroup_count_ptr;
}

void iree_task_dispatch_issue(iree_task_dispatch_t* dispatch_task,
                              iree_task_pool_t* shard_task_pool,
                              iree_task_submission_t* pending_submission,
                              iree_task_post_batch_t* post_batch) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, dispatch_task->dispatch_id);

  // Mark the dispatch as having been issued; the next time it retires it'll be
  // because all work has completed.
  dispatch_task->header.flags |= IREE_TASK_FLAG_DISPATCH_RETIRE;

  // Fetch the workgroup count (directly or indirectly).
  if (dispatch_task->header.flags & IREE_TASK_FLAG_DISPATCH_INDIRECT) {
    // By the task being ready to execute we know any dependencies on the
    // indirection buffer have been satisfied and its safe to read. We perform
    // the indirection here and convert the dispatch to a direct one such that
    // following code can read the value.
    // TODO(benvanik): non-one-shot command buffers won't be able to do this as
    // the intent is that they can be dynamic per execution.
    const uint32_t* source_ptr = dispatch_task->workgroup_count.ptr;
    memcpy(dispatch_task->workgroup_count.value, source_ptr,
           sizeof(dispatch_task->workgroup_count.value));
    dispatch_task->header.flags ^= IREE_TASK_FLAG_DISPATCH_INDIRECT;
  }
  const uint32_t* workgroup_count = dispatch_task->workgroup_count.value;

#if IREE_HAL_VERBOSE_TRACING_ENABLE
  // TODO(benvanik): tracing.h helper that speeds this up; too slow.
  IREE_TRACE({
    char xyz_string[32];
    int xyz_string_length =
        snprintf(xyz_string, IREE_ARRAYSIZE(xyz_string), "%ux%ux%u",
                 workgroup_count[0], workgroup_count[1], workgroup_count[2]);
    IREE_TRACE_ZONE_APPEND_TEXT_STRING_VIEW(z0, xyz_string, xyz_string_length);
  });
#endif  // IREE_HAL_VERBOSE_TRACING_ENABLE

  // Setup the iteration space for shards to pull work from the complete grid.
  iree_atomic_store_int32(&dispatch_task->tile_index, 0,
                          iree_memory_order_relaxed);
  dispatch_task->tile_count =
      workgroup_count[0] * workgroup_count[1] * workgroup_count[2];

  // Compute shard count - almost always worker_count unless we are a very small
  // dispatch (1x1x1, etc).
  iree_host_size_t worker_count = iree_task_post_batch_worker_count(post_batch);
  iree_host_size_t shard_count =
      iree_min(dispatch_task->tile_count, worker_count);

  // Compute how many tiles we want each shard to reserve at a time from the
  // larger grid. A higher number reduces overhead and improves locality while
  // a lower number reduces maximum worst-case latency (coarser work stealing).
  if (dispatch_task->tile_count <
      worker_count * IREE_TASK_DISPATCH_MAX_TILES_PER_SHARD_RESERVATION) {
    // Grid is small - allow it to be eagerly sliced up.
    dispatch_task->tiles_per_reservation = 1;
  } else {
    dispatch_task->tiles_per_reservation =
        IREE_TASK_DISPATCH_MAX_TILES_PER_SHARD_RESERVATION;
  }

  // Randomize starting worker.
  iree_host_size_t worker_offset = iree_task_post_batch_select_worker(
      post_batch, dispatch_task->header.affinity_set);
  iree_host_size_t worker_index = worker_offset;

  for (iree_host_size_t i = 0; i < shard_count; ++i) {
    // Allocate and initialize the shard.
    iree_task_dispatch_shard_t* shard_task =
        iree_task_dispatch_shard_allocate(dispatch_task, shard_task_pool);

    // Enqueue on the worker selected for the task.
    iree_task_post_batch_enqueue(post_batch, worker_index % worker_count,
                                 &shard_task->header);
    ++worker_index;
  }

  // NOTE: the dispatch is not retired until all shards complete. Upon the last
  // shard completing the lucky worker will retire the task inline and
  // potentially queue up more ready tasks that follow.
  //
  // The gotcha here is that it's possible for there to be zero shards within
  // a dispatch (if, for example, and indirect dispatch had its workgroup counts
  // set to zero to prevent it from running). We check for that here.
  if (shard_count == 0) {
    iree_task_dispatch_retire(dispatch_task, pending_submission);
  }

  IREE_TRACE_ZONE_END(z0);
}

void iree_task_dispatch_retire(iree_task_dispatch_t* dispatch_task,
                               iree_task_submission_t* pending_submission) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, dispatch_task->dispatch_id);

  // TODO(benvanik): attach statistics to the tracy zone.

  // Merge the statistics from the dispatch into the scope so we can track all
  // of the work without tracking all the dispatches at a global level.
  iree_task_dispatch_statistics_merge(
      &dispatch_task->statistics,
      &dispatch_task->header.scope->dispatch_statistics);

  // Consume the status of the dispatch that may have been set from a workgroup
  // and notify the scope. We need to do this here so that each shard retires
  // before we discard any subsequent tasks: otherwise a failure of one shard
  // would discard the shared dispatch task (and potentially everything) while
  // other shards were still running. We also want to avoid fine-grained
  // synchronization across shards that would occur by each checking to see if
  // any other has hit an error; failure in a dispatch should be so exceedingly
  // rare that allowing some shards to complete after one encounters an error is
  // not a problem.
  iree_status_t status = (iree_status_t)iree_atomic_exchange_intptr(
      &dispatch_task->status, 0, iree_memory_order_acq_rel);

  iree_task_retire(&dispatch_task->header, pending_submission, status);
  IREE_TRACE_ZONE_END(z0);
}

//==============================================================================
// IREE_TASK_TYPE_DISPATCH_SHARD
//==============================================================================

static inline iree_task_dispatch_t* iree_task_dispatch_shard_parent(
    iree_task_dispatch_shard_t* task) {
  return (iree_task_dispatch_t*)task->header.completion_task;
}

void iree_task_dispatch_shard_initialize(iree_task_dispatch_t* dispatch_task,
                                         iree_task_dispatch_shard_t* out_task) {
  iree_task_initialize(IREE_TASK_TYPE_DISPATCH_SHARD,
                       dispatch_task->header.scope, &out_task->header);
  iree_task_set_completion_task(&out_task->header, &dispatch_task->header);
}

iree_task_dispatch_shard_t* iree_task_dispatch_shard_allocate(
    iree_task_dispatch_t* dispatch_task, iree_task_pool_t* shard_task_pool) {
  iree_task_dispatch_shard_t* shard_task = NULL;
  iree_status_t status =
      iree_task_pool_acquire(shard_task_pool, (iree_task_t**)&shard_task);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    return NULL;
  }
  iree_task_dispatch_shard_initialize(dispatch_task, shard_task);
  shard_task->header.pool = shard_task_pool;
  return shard_task;
}

void iree_task_dispatch_shard_execute(
    iree_task_dispatch_shard_t* task, iree_cpu_processor_id_t processor_id,
    uint32_t worker_id, iree_byte_span_t worker_local_memory,
    iree_task_submission_t* pending_submission) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_task_dispatch_t* dispatch_task = iree_task_dispatch_shard_parent(task);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, dispatch_task->dispatch_id);
  IREE_TRACE_ZONE_SET_COLOR(
      z0, iree_math_ptr_to_xrgb(dispatch_task->closure.user_context));

  // Require at least the requested amount of worker local memory but pass all
  // of the available memory. This allows dispatches to use more when available
  // but still get nice validation here when the minimums aren't met.
  if (IREE_UNLIKELY(dispatch_task->local_memory_size >
                    worker_local_memory.data_length)) {
    iree_task_try_set_status(
        &dispatch_task->status,
        iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                         "dispatch requires %ub of local memory but only "
                         "%" PRIhsz "b is available per-worker",
                         dispatch_task->local_memory_size,
                         worker_local_memory.data_length));
    iree_task_retire(&task->header, pending_submission, iree_ok_status());
    IREE_TRACE_ZONE_END(z0);
    return;
  }

  // Prepare context shared for all tiles in the shard.
  iree_task_tile_context_t tile_context;
  memcpy(&tile_context.workgroup_size, dispatch_task->workgroup_size,
         sizeof(tile_context.workgroup_size));
  memcpy(&tile_context.workgroup_count, dispatch_task->workgroup_count.value,
         sizeof(tile_context.workgroup_count));
  uint32_t workgroup_count_x = tile_context.workgroup_count[0];
  uint32_t workgroup_count_y = tile_context.workgroup_count[1];
  tile_context.worker_id = worker_id;
  tile_context.local_memory = worker_local_memory;

  // We perform all our shard statistics work locally here and only push back to
  // the dispatch at the end; this avoids contention from each shard trying to
  // update the statistics together.
  iree_task_dispatch_statistics_t shard_statistics;
  memset(&shard_statistics, 0, sizeof(shard_statistics));
  tile_context.statistics = &shard_statistics;

  // Hint as to which processor we are running on.
  tile_context.processor_id = processor_id;

  // Loop over all tiles until they are all processed.
  const uint32_t tile_count = dispatch_task->tile_count;
  const uint32_t tiles_per_reservation = dispatch_task->tiles_per_reservation;
  // relaxed order because we only care about atomic increments, not about
  // ordering of tile_index accesses w.r.t. other memory accesses.
  uint32_t tile_base = iree_atomic_fetch_add_int32(&dispatch_task->tile_index,
                                                   tiles_per_reservation,
                                                   iree_memory_order_relaxed);
  while (tile_base < tile_count) {
    const uint32_t tile_range =
        iree_min(tile_base + tiles_per_reservation, tile_count);
    for (uint32_t tile_index = tile_base; tile_index < tile_range;
         ++tile_index) {
      // TODO(benvanik): faster math here, especially knowing we pull off N
      // sequential indices per reservation.
      uint32_t tile_i = tile_index;
      tile_context.workgroup_xyz[0] = tile_i % workgroup_count_x;
      tile_i /= workgroup_count_x;
      tile_context.workgroup_xyz[1] = tile_i % workgroup_count_y;
      tile_i /= workgroup_count_y;
      tile_context.workgroup_xyz[2] = tile_i;

      IREE_TRACE_ZONE_BEGIN_NAMED(z_tile,
                                  "iree_task_dispatch_shard_execute_tile");
      IREE_TRACE_ZONE_SET_COLOR(z_tile, iree_task_tile_to_color(&tile_context));

#ifndef NDEBUG
      // NOTE: these are useful for debugging but dramatically increase our
      // cost here; only enable if needed for tracking work distribution:
      IREE_TRACE_ZONE_APPEND_VALUE_I64(z_tile, tile_context.workgroup_xyz[0]);
      IREE_TRACE_ZONE_APPEND_VALUE_I64(z_tile, tile_context.workgroup_xyz[1]);
      IREE_TRACE_ZONE_APPEND_VALUE_I64(z_tile, tile_context.workgroup_xyz[2]);
      // IREE_TRACE_ZONE_APPEND_VALUE_I64(z_tile, (uint64_t)task->closure.fn);
#endif  // !NDEBUG

      iree_status_t status =
          dispatch_task->closure.fn(dispatch_task->closure.user_context,
                                    &tile_context, pending_submission);

      IREE_TRACE_ZONE_END(z_tile);

      // If any tile fails we bail early from the loop. This doesn't match
      // what an accelerator would do but saves some unneeded work.
      // Note that other shards may have completed execution, be executing
      // concurrently with this one, or still be pending - this does not
      // have any influence on them and they may continue to execute even
      // after we bail from here.
      if (!iree_status_is_ok(status)) {
        // Propagate failures to the dispatch task.
        iree_task_try_set_status(&dispatch_task->status, status);
        goto abort_shard;  // out of the while-for nest
      }
    }

    // Try to grab the next slice of tiles.
    tile_base = iree_atomic_fetch_add_int32(&dispatch_task->tile_index,
                                            tiles_per_reservation,
                                            iree_memory_order_relaxed);
  }
abort_shard:

  // Push aggregate statistics up to the dispatch.
  // Note that we may have partial information here if we errored out of the
  // loop but that's still useful to know.
  iree_task_dispatch_statistics_merge(&shard_statistics,
                                      &dispatch_task->statistics);

  // NOTE: even if an error was hit we retire OK - the error has already been
  // propagated to the dispatch and it'll clean up after all shards are joined.
  iree_task_retire(&task->header, pending_submission, iree_ok_status());
  IREE_TRACE_ZONE_END(z0);
}
