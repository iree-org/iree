// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/metal/shared_event.h"

#import <Metal/Metal.h>

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/base/internal/synchronization.h"
#include "iree/base/tracing.h"
#include "iree/hal/api.h"

typedef struct iree_hal_metal_shared_event_t {
  // Abstract resource used for injecting reference counting and vtable; must be at offset 0.
  iree_hal_resource_t resource;

  id<MTLSharedEvent> shared_event;
  // A listener object used for dispatching notifications; owned by the device.
  MTLSharedEventListener* event_listener;

  iree_allocator_t host_allocator;

  // Permanently failure state of the current semaphore, if failed.
  iree_status_t failure_state;
  // Mutex guarding access to the failure state.
  iree_slim_mutex_t state_mutex;
} iree_hal_metal_shared_event_t;

static const iree_hal_semaphore_vtable_t iree_hal_metal_shared_event_vtable;

static iree_hal_metal_shared_event_t* iree_hal_metal_shared_event_cast(
    iree_hal_semaphore_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_metal_shared_event_vtable);
  return (iree_hal_metal_shared_event_t*)base_value;
}

static const iree_hal_metal_shared_event_t* iree_hal_metal_shared_event_const_cast(
    const iree_hal_semaphore_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_metal_shared_event_vtable);
  return (const iree_hal_metal_shared_event_t*)base_value;
}

bool iree_hal_metal_shared_event_isa(iree_hal_semaphore_t* semaphore) {
  return iree_hal_resource_is(semaphore, &iree_hal_metal_shared_event_vtable);
}

id<MTLSharedEvent> iree_hal_metal_shared_event_handle(const iree_hal_semaphore_t* base_semaphore) {
  const iree_hal_metal_shared_event_t* semaphore =
      iree_hal_metal_shared_event_const_cast(base_semaphore);
  return semaphore->shared_event;
}

iree_status_t iree_hal_metal_shared_event_create(id<MTLDevice> device, uint64_t initial_value,
                                                 MTLSharedEventListener* listener,
                                                 iree_allocator_t host_allocator,
                                                 iree_hal_semaphore_t** out_semaphore) {
  IREE_ASSERT_ARGUMENT(out_semaphore);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_metal_shared_event_t* semaphore = NULL;
  iree_status_t status =
      iree_allocator_malloc(host_allocator, sizeof(*semaphore), (void**)&semaphore);
  if (iree_status_is_ok(status)) {
    iree_hal_resource_initialize(&iree_hal_metal_shared_event_vtable, &semaphore->resource);
    semaphore->shared_event = [device newSharedEvent];  // +1
    semaphore->shared_event.signaledValue = initial_value;
    semaphore->event_listener = listener;
    semaphore->host_allocator = host_allocator;
    iree_slim_mutex_initialize(&semaphore->state_mutex);
    semaphore->failure_state = iree_ok_status();
    *out_semaphore = (iree_hal_semaphore_t*)semaphore;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_metal_shared_event_destroy(iree_hal_semaphore_t* base_semaphore) {
  iree_hal_metal_shared_event_t* semaphore = iree_hal_metal_shared_event_cast(base_semaphore);
  IREE_TRACE_ZONE_BEGIN(z0);

  [semaphore->shared_event release];  // -1
  iree_slim_mutex_deinitialize(&semaphore->state_mutex);
  iree_allocator_free(semaphore->host_allocator, semaphore);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_metal_shared_event_query(iree_hal_semaphore_t* base_semaphore,
                                                       uint64_t* out_value) {
  iree_hal_metal_shared_event_t* semaphore = iree_hal_metal_shared_event_cast(base_semaphore);
  uint64_t value = semaphore->shared_event.signaledValue;
  if (IREE_UNLIKELY(value >= IREE_HAL_SEMAPHORE_FAILURE_VALUE)) {
    iree_status_t status = iree_ok_status();
    iree_slim_mutex_lock(&semaphore->state_mutex);
    status = semaphore->failure_state;
    iree_slim_mutex_unlock(&semaphore->state_mutex);
    return status;
  }
  *out_value = value;
  return iree_ok_status();
}

static iree_status_t iree_hal_metal_shared_event_signal(iree_hal_semaphore_t* base_semaphore,
                                                        uint64_t new_value) {
  iree_hal_metal_shared_event_t* semaphore = iree_hal_metal_shared_event_cast(base_semaphore);
  uint64_t value = semaphore->shared_event.signaledValue;
  if (IREE_UNLIKELY(value >= IREE_HAL_SEMAPHORE_FAILURE_VALUE)) {
    iree_status_t status = iree_ok_status();
    iree_slim_mutex_lock(&semaphore->state_mutex);
    status = semaphore->failure_state;
    iree_slim_mutex_unlock(&semaphore->state_mutex);
    return status;
  }
  semaphore->shared_event.signaledValue = new_value;
  return iree_ok_status();
}

static void iree_hal_metal_shared_event_fail(iree_hal_semaphore_t* base_semaphore,
                                             iree_status_t status) {
  iree_hal_metal_shared_event_t* semaphore = iree_hal_metal_shared_event_cast(base_semaphore);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_slim_mutex_lock(&semaphore->state_mutex);
  semaphore->failure_state = status;
  semaphore->shared_event.signaledValue = IREE_HAL_SEMAPHORE_FAILURE_VALUE;
  iree_slim_mutex_unlock(&semaphore->state_mutex);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_metal_shared_event_wait(iree_hal_semaphore_t* base_semaphore,
                                                      uint64_t value, iree_timeout_t timeout) {
  iree_hal_metal_shared_event_t* semaphore = iree_hal_metal_shared_event_cast(base_semaphore);

  iree_time_t deadline_ns = iree_timeout_as_deadline_ns(timeout);
  uint64_t timeout_ns;
  dispatch_time_t apple_timeout_ns;
  if (deadline_ns == IREE_TIME_INFINITE_FUTURE) {
    timeout_ns = UINT64_MAX;
    apple_timeout_ns = DISPATCH_TIME_FOREVER;
  } else if (deadline_ns == IREE_TIME_INFINITE_PAST) {
    timeout_ns = 0;
    apple_timeout_ns = DISPATCH_TIME_NOW;
  } else {
    iree_time_t now_ns = iree_time_now();
    if (deadline_ns < now_ns) {
      return iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED);
    }
    timeout_ns = (uint64_t)(deadline_ns - now_ns);
    apple_timeout_ns = dispatch_time(DISPATCH_TIME_NOW, timeout_ns);
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  // Quick path for impatient waiting to avoid all the overhead of dispatch queues and semaphores.
  if (timeout_ns == 0) {
    uint64_t current_value = 0;
    iree_status_t status = iree_hal_metal_shared_event_query(base_semaphore, &current_value);
    if (iree_status_is_ok(status) && current_value < value) {
      status = iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED);
    }
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  // Theoretically we don't really need to mark the semaphore handle as __block given that the
  // handle itself is not modified and there is only one block and it will copy the handle.
  // But marking it as __block serves as good documentation purpose.
  __block dispatch_semaphore_t work_done = dispatch_semaphore_create(0);

  __block bool did_fail = false;

  // Use a listener to the MTLSharedEvent to notify us when the work is done on GPU by signaling a
  // semaphore. The signaling will happen in a new dispatch queue; the current thread will wait on
  // the semaphore.
  [semaphore->shared_event notifyListener:semaphore->event_listener
                                  atValue:value
                                    block:^(id<MTLSharedEvent> se, uint64_t v) {
                                      if (v >= IREE_HAL_SEMAPHORE_FAILURE_VALUE) did_fail = true;

                                      dispatch_semaphore_signal(work_done);
                                    }];

  // If the work is not done immediately, dispatch_semaphore_wait decreases the semaphore value to
  // less than zero first and then puts the current thread into wait state.
  intptr_t timed_out = dispatch_semaphore_wait(work_done, apple_timeout_ns);
  dispatch_release(work_done);

  IREE_TRACE_ZONE_END(z0);
  if (IREE_UNLIKELY(did_fail)) return iree_status_from_code(IREE_STATUS_ABORTED);
  if (timed_out) return iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED);
  return iree_ok_status();
}

iree_status_t iree_hal_metal_shared_event_multi_wait(
    iree_hal_wait_mode_t wait_mode, const iree_hal_semaphore_list_t* semaphore_list,
    iree_timeout_t timeout) {
  if (semaphore_list->count == 0) return iree_ok_status();
  // If there is only one semaphore, just wait on it.
  if (semaphore_list->count == 1) {
    return iree_hal_metal_shared_event_wait(semaphore_list->semaphores[0],
                                            semaphore_list->payload_values[0], timeout);
  }

  iree_time_t deadline_ns = iree_timeout_as_deadline_ns(timeout);
  uint64_t timeout_ns;
  dispatch_time_t apple_timeout_ns;
  if (deadline_ns == IREE_TIME_INFINITE_FUTURE) {
    timeout_ns = UINT64_MAX;
    apple_timeout_ns = DISPATCH_TIME_FOREVER;
  } else if (deadline_ns == IREE_TIME_INFINITE_PAST) {
    timeout_ns = 0;
    apple_timeout_ns = DISPATCH_TIME_NOW;
  } else {
    iree_time_t now_ns = iree_time_now();
    if (deadline_ns < now_ns) {
      return iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED);
    }
    timeout_ns = (uint64_t)(deadline_ns - now_ns);
    apple_timeout_ns = dispatch_time(DISPATCH_TIME_NOW, timeout_ns);
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  // Create an atomic to count how many semaphores have signaled. Mark it as `__block` so different
  // threads are sharing the same data via reference.
  __block iree_atomic_int32_t wait_count;
  iree_atomic_store_int32(&wait_count, 0, iree_memory_order_release);
  // The total count we are expecting to see.
  iree_host_size_t total_count = (wait_mode == IREE_HAL_WAIT_MODE_ALL) ? semaphore_list->count : 1;
  // Theoretically we don't really need to mark the semaphore handle as __block given that the
  // handle itself is not modified and there is only one block and it will copy the handle.
  // But marking it as __block serves as good documentation purpose.
  __block dispatch_semaphore_t work_done = dispatch_semaphore_create(0);

  __block bool did_fail = false;

  for (iree_host_size_t i = 0; i < semaphore_list->count; ++i) {
    // Use a listener to the MTLSharedEvent to notify us when the work is done on GPU by signaling a
    // semaphore. The signaling will happen in a new dispatch queue; the current thread will wait on
    // the semaphore.
    iree_hal_metal_shared_event_t* semaphore =
        iree_hal_metal_shared_event_cast(semaphore_list->semaphores[i]);
    [semaphore->shared_event notifyListener:semaphore->event_listener
                                    atValue:semaphore_list->payload_values[i]
                                      block:^(id<MTLSharedEvent> se, uint64_t v) {
                                        // Fail as a whole if any participating semaphore failed.
                                        if (v >= IREE_HAL_SEMAPHORE_FAILURE_VALUE) did_fail = true;

                                        int32_t old_value = iree_atomic_fetch_add_int32(
                                            &wait_count, 1, iree_memory_order_release);
                                        // The last signaled semaphore send out the notification.
                                        // Atomic fetch add returns the old value, so need to +1.
                                        if (old_value + 1 == total_count) {
                                          dispatch_semaphore_signal(work_done);
                                        }
                                      }];
  }

  // If the work is not done immediately, dispatch_semaphore_wait decreases the semaphore value by
  // one first and then puts the current thread into wait state.
  intptr_t timed_out = dispatch_semaphore_wait(work_done, apple_timeout_ns);
  dispatch_release(work_done);

  IREE_TRACE_ZONE_END(z0);
  if (IREE_UNLIKELY(did_fail)) return iree_status_from_code(IREE_STATUS_ABORTED);
  if (timed_out) return iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED);
  return iree_ok_status();
}

static const iree_hal_semaphore_vtable_t iree_hal_metal_shared_event_vtable = {
    .destroy = iree_hal_metal_shared_event_destroy,
    .query = iree_hal_metal_shared_event_query,
    .signal = iree_hal_metal_shared_event_signal,
    .fail = iree_hal_metal_shared_event_fail,
    .wait = iree_hal_metal_shared_event_wait,
};
