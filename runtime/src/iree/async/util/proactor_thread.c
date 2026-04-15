// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/util/proactor_thread.h"

#include "iree/base/internal/atomics.h"
#include "iree/base/threading/notification.h"

//===----------------------------------------------------------------------===//
// iree_async_proactor_thread_t
//===----------------------------------------------------------------------===//

struct iree_async_proactor_thread_t {
  iree_atomic_ref_count_t ref_count;
  iree_allocator_t allocator;

  // Retained proactor driven by this thread.
  iree_async_proactor_t* proactor;

  // Platform thread handle.
  iree_thread_t* thread;

  // Set by request_stop(), checked each poll iteration.
  iree_atomic_int32_t stop_requested;

  // Signaled by the thread main function just before returning.
  // Used by join() to implement timed waits (iree_thread_join is untimed).
  iree_notification_t exited;

  // Maximum time the thread blocks in each poll() call.
  // 0 means infinite (block until work arrives or wake() is called).
  iree_duration_t poll_timeout;

  // Optional error callback (fires once on fatal proactor error).
  iree_async_proactor_thread_error_callback_t error_callback;

  // Fatal status from poll(). Written once by the poll thread, read after
  // join() by consume_status(). Protected by the exit ordering: the thread
  // writes this before signaling |exited|, and consume_status() is only valid
  // after join() returns OK.
  iree_status_t fatal_status;
};

static bool iree_async_proactor_thread_has_exited(void* user_data) {
  iree_async_proactor_thread_t* thread =
      (iree_async_proactor_thread_t*)user_data;
  // The thread stores 1 into stop_requested after exiting the poll loop, right
  // before signaling the notification. But we use a separate atomic for
  // clarity: the notification's own epoch mechanism handles the actual wake,
  // and this condition function is just the predicate. We check that the thread
  // has written its fatal_status and is about to (or has already) signaled
  // exit.
  //
  // We piggyback on stop_requested >= 2 to mean "exited" (vs 1 = "stop
  // requested but still running"). This avoids adding another atomic.
  return iree_atomic_load(&thread->stop_requested, iree_memory_order_acquire) >=
         2;
}

static int iree_async_proactor_thread_main(void* entry_arg) {
  iree_async_proactor_thread_t* thread =
      (iree_async_proactor_thread_t*)entry_arg;
  {
    IREE_TRACE_ZONE_BEGIN_NAMED(z0, "iree_async_proactor_thread_start");
    IREE_TRACE_ZONE_END(z0);
  }

  iree_status_t status = iree_ok_status();
  while (
      iree_status_is_ok(status) &&
      !iree_atomic_load(&thread->stop_requested, iree_memory_order_acquire)) {
    iree_timeout_t timeout;
    if (thread->poll_timeout == 0) {
      timeout = iree_infinite_timeout();
    } else {
      timeout = iree_make_timeout_ns(thread->poll_timeout);
    }

    iree_host_size_t completed_count = 0;
    status =
        iree_async_proactor_poll(thread->proactor, timeout, &completed_count);
    if (iree_status_is_deadline_exceeded(status)) {
      iree_status_ignore(status);
      status = iree_ok_status();
      continue;
    }
    {
      IREE_TRACE_ZONE_BEGIN_NAMED(z0, "iree_async_proactor_thread_poll_wake");
      IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, completed_count);
      IREE_TRACE_ZONE_END(z0);
    }
  }

  // Store fatal status (if any) before signaling exit.
  if (!iree_status_is_ok(status)) {
    thread->fatal_status = status;
    if (thread->error_callback.fn) {
      // The error callback takes ownership of a clone; we keep the original
      // for consume_status().
      thread->error_callback.fn(thread->error_callback.user_data,
                                iree_status_clone(status));
    }
  }

  // Mark as exited and signal waiters.
  iree_atomic_store(&thread->stop_requested, 2, iree_memory_order_release);
  iree_notification_post(&thread->exited, IREE_ALL_WAITERS);

  {
    IREE_TRACE_ZONE_BEGIN_NAMED(z0, "iree_async_proactor_thread_exit");
    IREE_TRACE_ZONE_END(z0);
  }
  return 0;
}

iree_status_t iree_async_proactor_thread_create(
    iree_async_proactor_t* proactor,
    iree_async_proactor_thread_options_t options, iree_allocator_t allocator,
    iree_async_proactor_thread_t** out_thread) {
  IREE_ASSERT_ARGUMENT(proactor);
  IREE_ASSERT_ARGUMENT(out_thread);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_thread = NULL;

  iree_async_proactor_thread_t* thread = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(allocator, sizeof(*thread), (void**)&thread));

  iree_atomic_ref_count_init(&thread->ref_count);
  thread->allocator = allocator;
  thread->proactor = proactor;
  iree_async_proactor_retain(proactor);
  iree_atomic_store(&thread->stop_requested, 0, iree_memory_order_relaxed);
  iree_notification_initialize(&thread->exited);
  thread->poll_timeout = options.poll_timeout;
  thread->error_callback = options.error_callback;
  thread->fatal_status = iree_ok_status();

  iree_thread_create_params_t params;
  memset(&params, 0, sizeof(params));
  params.name = options.debug_name;
  params.initial_affinity = options.affinity;

  iree_status_t status =
      iree_thread_create(iree_async_proactor_thread_main, thread, params,
                         allocator, &thread->thread);
  if (iree_status_is_ok(status)) {
    *out_thread = thread;
  } else {
    iree_async_proactor_release(thread->proactor);
    iree_notification_deinitialize(&thread->exited);
    iree_allocator_free(allocator, thread);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_async_proactor_thread_retain(iree_async_proactor_thread_t* thread) {
  if (IREE_LIKELY(thread)) {
    iree_atomic_ref_count_inc(&thread->ref_count);
  }
}

void iree_async_proactor_thread_release(iree_async_proactor_thread_t* thread) {
  if (IREE_LIKELY(thread) &&
      iree_atomic_ref_count_dec(&thread->ref_count) == 1) {
    iree_allocator_t allocator = thread->allocator;
    // Thread must be stopped before release (contract from header).
    iree_status_ignore(thread->fatal_status);
    // Join the OS thread before destroying the notification — the thread may
    // still be inside iree_notification_post when it signals exit.
    iree_thread_release(thread->thread);
    iree_notification_deinitialize(&thread->exited);
    iree_async_proactor_release(thread->proactor);
    iree_allocator_free(allocator, thread);
  }
}

void iree_async_proactor_thread_request_stop(
    iree_async_proactor_thread_t* thread) {
  if (!thread) return;
  // CAS from 0 → 1 so we don't overwrite the "exited" value of 2.
  int32_t expected = 0;
  iree_atomic_compare_exchange_strong(&thread->stop_requested, &expected, 1,
                                      iree_memory_order_release,
                                      iree_memory_order_relaxed);
  // Wake the proactor so the thread sees the flag promptly even if poll() was
  // blocking.
  iree_async_proactor_wake(thread->proactor);
}

iree_status_t iree_async_proactor_thread_join(
    iree_async_proactor_thread_t* thread, iree_duration_t timeout) {
  IREE_ASSERT_ARGUMENT(thread);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_timeout_t timed;
  if (timeout == IREE_DURATION_INFINITE) {
    timed = iree_infinite_timeout();
  } else {
    timed = iree_make_timeout_ns(timeout);
  }

  bool exited = iree_notification_await(
      &thread->exited, iree_async_proactor_thread_has_exited, thread, timed);
  if (!exited) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_DEADLINE_EXCEEDED,
                            "proactor thread did not exit within timeout");
  }

  // The thread has exited its main function. We do NOT call iree_thread_join()
  // here because iree_thread_release() (called from our release()) does
  // pthread_join internally when the thread handle's ref count reaches zero.
  // Calling both would be a double-join (undefined behavior / ASAN abort).
  //
  // The notification guarantees the thread has finished executing our entry
  // function and will not touch any of our state after this point. The actual
  // OS thread cleanup (stack deallocation etc.) happens when the thread handle
  // is released.

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_async_proactor_thread_consume_status(
    iree_async_proactor_thread_t* thread) {
  IREE_ASSERT_ARGUMENT(thread);
  iree_status_t status = thread->fatal_status;
  thread->fatal_status = iree_ok_status();
  return status;
}
