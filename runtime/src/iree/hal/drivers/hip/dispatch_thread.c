// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/hip/dispatch_thread.h"

#include "iree/base/internal/synchronization.h"
#include "iree/base/internal/threading.h"
#include "iree/hal/drivers/hip/event_pool.h"
#include "iree/hal/drivers/hip/status_util.h"
#include "iree/hal/drivers/hip/util/queue.h"

#define iree_hal_hip_dispatch_thread_default_queue_size 64

typedef struct iree_hal_hip_dispatch_thread_dispatch_t {
  iree_hal_hip_dispatch_callback_t dispatch;
  void* user_data;
} iree_hal_hip_dispatch_thread_dispatch_t;

IREE_HAL_HIP_UTIL_TYPED_QUEUE_WRAPPER(
    iree_hal_hip_dispatch_queue, iree_hal_hip_dispatch_thread_dispatch_t,
    iree_hal_hip_dispatch_thread_default_queue_size);

typedef struct iree_hal_hip_dispatch_thread_t {
  iree_thread_t* thread;
  iree_allocator_t host_allocator;
  iree_slim_mutex_t mutex;

  iree_hal_hip_dispatch_queue_t queue;
  iree_status_t failure_status;
  iree_notification_t notification;
  bool do_exit;
} iree_hal_hip_dispatch_thread_t;

static bool iree_hal_hip_dispatch_thread_has_request(void* user_data) {
  iree_hal_hip_dispatch_thread_t* thread =
      (iree_hal_hip_dispatch_thread_t*)user_data;
  iree_slim_mutex_lock(&thread->mutex);
  bool has_request = !iree_hal_hip_dispatch_queue_empty(&thread->queue);
  has_request |= thread->do_exit;
  iree_slim_mutex_unlock(&thread->mutex);
  return has_request;
}

static int iree_hal_hip_dispatch_thread_main(void* param) {
  iree_hal_hip_dispatch_thread_t* thread =
      (iree_hal_hip_dispatch_thread_t*)param;
  bool exit = false;
  while (true) {
    iree_notification_await(&thread->notification,
                            &iree_hal_hip_dispatch_thread_has_request, thread,
                            iree_infinite_timeout());

    iree_slim_mutex_lock(&thread->mutex);
    exit |= thread->do_exit;
    iree_status_t status = iree_status_clone(thread->failure_status);
    while (!iree_hal_hip_dispatch_queue_empty(&thread->queue)) {
      iree_hal_hip_dispatch_thread_dispatch_t dispatch =
          iree_hal_hip_dispatch_queue_at(&thread->queue, 0);
      iree_hal_hip_dispatch_queue_pop_front(&thread->queue, 1);
      iree_slim_mutex_unlock(&thread->mutex);

      // The status passed will be joined and returned.
      status = dispatch.dispatch(dispatch.user_data, status);
      iree_slim_mutex_lock(&thread->mutex);
      if (!iree_status_is_ok(status)) {
        // We don't join here as the failure status was already
        // included here.
        iree_status_ignore(thread->failure_status);
        thread->failure_status = iree_status_clone(status);
      }
    }
    iree_slim_mutex_unlock(&thread->mutex);

    if (!iree_status_is_ok(status) || exit) {
      // Drop the status as it was cloned into thread->failure_status
      // if needed.
      iree_status_ignore(status);
      break;
    }
  }
  return 0;
}

iree_status_t iree_hal_hip_dispatch_thread_initialize(
    iree_allocator_t host_allocator,
    iree_hal_hip_dispatch_thread_t** out_thread) {
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_thread = NULL;
  iree_hal_hip_dispatch_thread_t* thread = NULL;

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator, sizeof(*thread), (void**)&thread));

  thread->do_exit = false;
  iree_slim_mutex_initialize(&thread->mutex);
  iree_hal_hip_dispatch_queue_initialize(host_allocator, &thread->queue);
  thread->failure_status = iree_ok_status();
  thread->host_allocator = host_allocator;
  iree_notification_initialize(&thread->notification);

  iree_thread_create_params_t params;
  memset(&params, 0x00, sizeof(params));
  params.name = iree_make_cstring_view("iree-hal-hip-dispatch");
  iree_status_t status =
      iree_thread_create((iree_thread_entry_t)iree_hal_hip_dispatch_thread_main,
                         thread, params, host_allocator, &thread->thread);

  if (iree_status_is_ok(status)) {
    *out_thread = thread;
  } else {
    iree_hal_hip_dispatch_queue_deinitialize(&thread->queue);
    iree_slim_mutex_deinitialize(&thread->mutex);
    iree_allocator_free(host_allocator, thread);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_hip_dispatch_thread_deinitialize(
    iree_hal_hip_dispatch_thread_t* thread) {
  if (!thread) {
    return;
  }
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_slim_mutex_lock(&thread->mutex);
  thread->do_exit = true;
  iree_slim_mutex_unlock(&thread->mutex);

  iree_notification_post(&thread->notification, IREE_ALL_WAITERS);
  // There is only one owner for the thread, so this also joins the thread.
  iree_thread_release(thread->thread);

  iree_status_ignore(thread->failure_status);
  iree_hal_hip_dispatch_queue_deinitialize(&thread->queue);
  iree_slim_mutex_deinitialize(&thread->mutex);
  iree_allocator_free(thread->host_allocator, thread);
  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_hal_hip_dispatch_thread_add_dispatch(
    iree_hal_hip_dispatch_thread_t* thread,
    iree_hal_hip_dispatch_callback_t dispatch, void* user_data) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_slim_mutex_lock(&thread->mutex);
  iree_status_t status = iree_status_clone(thread->failure_status);

  iree_hal_hip_dispatch_thread_dispatch_t dispatch_data = {
      .dispatch = dispatch,
      .user_data = user_data,
  };

  if (iree_status_is_ok(status)) {
    status =
        iree_hal_hip_dispatch_queue_push_back(&thread->queue, dispatch_data);
  }
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(thread->failure_status);
    thread->failure_status = iree_status_clone(status);
  }
  iree_slim_mutex_unlock(&thread->mutex);
  iree_notification_post(&thread->notification, IREE_ALL_WAITERS);

  if (!iree_status_is_ok(status)) {
    iree_status_ignore(dispatch(user_data, iree_status_clone(status)));
  }
  IREE_TRACE_ZONE_END(z0);

  // If this was a failure then it was put into thread->failure_status.
  return status;
}
