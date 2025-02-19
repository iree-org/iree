// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/hip/cleanup_thread.h"

#include "iree/base/internal/synchronization.h"
#include "iree/base/internal/threading.h"
#include "iree/hal/drivers/hip/event_pool.h"
#include "iree/hal/drivers/hip/status_util.h"
#include "iree/hal/drivers/hip/util/queue.h"

#define iree_hal_hip_cleanup_thread_default_queue_size 64

typedef struct iree_hal_hip_cleanup_thread_callback_t {
  iree_hal_hip_cleanup_callback_t callback;
  void* user_data;
  iree_hal_hip_event_t* event;
} iree_hal_hip_cleanup_thread_callback_t;

IREE_HAL_HIP_UTIL_TYPED_QUEUE_WRAPPER(
    iree_hal_hip_callback_queue, iree_hal_hip_cleanup_thread_callback_t,
    iree_hal_hip_cleanup_thread_default_queue_size);

typedef struct iree_hal_hip_cleanup_thread_t {
  iree_thread_t* thread;
  iree_allocator_t host_allocator;
  const iree_hal_hip_dynamic_symbols_t* symbols;
  iree_slim_mutex_t mutex;

  iree_hal_hip_callback_queue_t queue;
  iree_status_t failure_status;
  iree_notification_t notification;
  bool do_exit;
} iree_hal_hip_cleanup_thread_t;

static bool iree_hal_hip_cleanup_thread_has_request(void* user_data) {
  iree_hal_hip_cleanup_thread_t* thread =
      (iree_hal_hip_cleanup_thread_t*)user_data;
  iree_slim_mutex_lock(&thread->mutex);
  bool has_request = !iree_hal_hip_callback_queue_empty(&thread->queue);
  has_request |= thread->do_exit;
  iree_slim_mutex_unlock(&thread->mutex);
  return has_request;
}

static int iree_hal_hip_cleanup_thread_main(void* param) {
  iree_hal_hip_cleanup_thread_t* thread = (iree_hal_hip_cleanup_thread_t*)param;
  bool exit = false;
  while (true) {
    iree_notification_await(&thread->notification,
                            &iree_hal_hip_cleanup_thread_has_request, thread,
                            iree_infinite_timeout());

    iree_slim_mutex_lock(&thread->mutex);
    exit |= thread->do_exit;
    iree_status_t status = thread->failure_status;
    while (!iree_hal_hip_callback_queue_empty(&thread->queue)) {
      iree_hal_hip_cleanup_thread_callback_t callback =
          iree_hal_hip_callback_queue_at(&thread->queue, 0);
      iree_hal_hip_callback_queue_pop_front(&thread->queue, 1);
      iree_slim_mutex_unlock(&thread->mutex);

      // If we have a null event then we don't have to wait
      // on the GPU to synchronize.
      if (iree_status_is_ok(status) && callback.event) {
        status = IREE_HIP_CALL_TO_STATUS(
            thread->symbols,
            hipEventSynchronize(iree_hal_hip_event_handle(callback.event)));
      }

      status = iree_status_join(
          status,
          callback.callback(callback.user_data, callback.event, status));
      iree_slim_mutex_lock(&thread->mutex);
      if (!iree_status_is_ok(status)) {
        thread->failure_status = status;
      }
      if (!iree_status_is_ok(thread->failure_status)) {
        status = iree_status_clone(thread->failure_status);
      }
    }
    iree_slim_mutex_unlock(&thread->mutex);

    if (!iree_status_is_ok(status) || exit) {
      break;
    }
  }
  return 0;
}

iree_status_t iree_hal_hip_cleanup_thread_initialize(
    const iree_hal_hip_dynamic_symbols_t* symbols,
    iree_allocator_t host_allocator,
    iree_hal_hip_cleanup_thread_t** out_thread) {
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_thread = NULL;
  iree_hal_hip_cleanup_thread_t* thread = NULL;

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator, sizeof(*thread), (void**)&thread));

  thread->symbols = symbols;
  thread->do_exit = false;
  iree_slim_mutex_initialize(&thread->mutex);
  iree_hal_hip_callback_queue_initialize(host_allocator, &thread->queue);
  thread->failure_status = iree_ok_status();
  thread->host_allocator = host_allocator;
  iree_notification_initialize(&thread->notification);

  iree_thread_create_params_t params;
  memset(&params, 0x00, sizeof(params));
  params.name = iree_make_cstring_view("iree-hal-hip-cleanup");
  iree_status_t status =
      iree_thread_create((iree_thread_entry_t)iree_hal_hip_cleanup_thread_main,
                         thread, params, host_allocator, &thread->thread);
  if (iree_status_is_ok(status)) {
    *out_thread = thread;
  } else {
    iree_hal_hip_callback_queue_deinitialize(&thread->queue);
    iree_slim_mutex_deinitialize(&thread->mutex);
    iree_allocator_free(host_allocator, thread);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_hip_cleanup_thread_deinitialize(
    iree_hal_hip_cleanup_thread_t* thread) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_slim_mutex_lock(&thread->mutex);
  thread->do_exit = true;
  iree_slim_mutex_unlock(&thread->mutex);

  iree_notification_post(&thread->notification, IREE_ALL_WAITERS);
  // There is only one owner for the thread, so this also joins the thread.
  iree_thread_release(thread->thread);

  iree_hal_hip_callback_queue_deinitialize(&thread->queue);
  iree_slim_mutex_deinitialize(&thread->mutex);
  iree_allocator_free(thread->host_allocator, thread);
  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_hal_hip_cleanup_thread_add_cleanup(
    iree_hal_hip_cleanup_thread_t* thread, iree_hal_hip_event_t* event,
    iree_hal_hip_cleanup_callback_t callback, void* user_data) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_slim_mutex_lock(&thread->mutex);
  if (!iree_status_is_ok(thread->failure_status)) {
    IREE_TRACE_ZONE_END(z0);
    iree_slim_mutex_unlock(&thread->mutex);
    return thread->failure_status;
  }

  iree_hal_hip_cleanup_thread_callback_t callback_data = {
      .callback = callback,
      .user_data = user_data,
      .event = event,
  };
  iree_hal_hip_callback_queue_push_back(&thread->queue, callback_data);
  iree_slim_mutex_unlock(&thread->mutex);
  iree_notification_post(&thread->notification, IREE_ALL_WAITERS);

  IREE_TRACE_ZONE_END(z0);

  return iree_ok_status();
}
