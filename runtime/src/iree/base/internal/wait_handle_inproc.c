// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// clang-format off: must be included before all other headers.
#include "iree/base/internal/wait_handle_impl.h"
// clang-format on

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/internal/wait_handle.h"
#include "iree/base/threading/notification.h"

// This implementation uses iree_notification_t - backed by a futex in most
// cases - to simulate system wait handles. Threads can block and wait for an
// event to be signaled via iree_wait_one.
#if IREE_WAIT_API == IREE_WAIT_API_INPROC

//===----------------------------------------------------------------------===//
// iree_wait_primitive_* raw calls
//===----------------------------------------------------------------------===//

typedef struct iree_futex_handle_t {
  iree_atomic_int64_t value;
  iree_notification_t notification;
} iree_futex_handle_t;

void iree_wait_handle_close(iree_wait_handle_t* handle) {
  switch (handle->type) {
#if defined(IREE_HAVE_WAIT_TYPE_LOCAL_FUTEX)
    case IREE_WAIT_PRIMITIVE_TYPE_LOCAL_FUTEX: {
      iree_futex_handle_t* futex =
          (iree_futex_handle_t*)handle->value.local_futex;
      iree_notification_deinitialize(&futex->notification);
      iree_allocator_free(iree_allocator_system(), futex);
      break;
    }
#endif  // IREE_HAVE_WAIT_TYPE_LOCAL_FUTEX
    default:
      break;
  }
  iree_wait_handle_deinitialize(handle);
}

static bool iree_futex_handle_check(iree_futex_handle_t* futex) {
  return iree_atomic_load(&futex->value, iree_memory_order_acquire) != 0;
}

static bool iree_futex_handle_check_thunk(void* arg) {
  return iree_futex_handle_check((iree_futex_handle_t*)arg);
}

iree_status_t iree_wait_one(iree_wait_handle_t* handle,
                            iree_time_t deadline_ns) {
  if (handle->type == IREE_WAIT_PRIMITIVE_TYPE_NONE) {
    return iree_ok_status();
  }
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_ok_status();
  if (handle->type == IREE_WAIT_PRIMITIVE_TYPE_LOCAL_FUTEX) {
    iree_futex_handle_t* futex =
        (iree_futex_handle_t*)handle->value.local_futex;
    if (!iree_notification_await(&futex->notification,
                                 iree_futex_handle_check_thunk, futex,
                                 iree_make_deadline(deadline_ns))) {
      status = iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED);
    }
  } else {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "unhandled primitive type");
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// iree_event_t
//===----------------------------------------------------------------------===//

iree_status_t iree_event_initialize(bool initial_state,
                                    iree_event_t* out_event) {
  IREE_TRACE_ZONE_BEGIN(z0);
  memset(out_event, 0, sizeof(*out_event));

  iree_futex_handle_t* futex = NULL;
  iree_status_t status = iree_allocator_malloc(iree_allocator_system(),
                                               sizeof(*futex), (void**)&futex);
  if (iree_status_is_ok(status)) {
    out_event->type = IREE_WAIT_PRIMITIVE_TYPE_LOCAL_FUTEX;
    out_event->value.local_futex = (void*)futex;
    iree_atomic_store(&futex->value, initial_state ? 1 : 0,
                      iree_memory_order_release);
    iree_notification_initialize(&futex->notification);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_event_deinitialize(iree_event_t* event) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_wait_handle_close(event);
  IREE_TRACE_ZONE_END(z0);
}

void iree_event_set(iree_event_t* event) {
  if (!event) return;
  iree_futex_handle_t* futex = (iree_futex_handle_t*)event->value.local_futex;
  if (!futex) return;

  // Try to transition from unset -> set.
  // No-op if already set and otherwise we successfully signaled the event and
  // need to notify all waiters.
  if (iree_atomic_exchange(&futex->value, 1, iree_memory_order_release) == 0) {
    iree_notification_post(&futex->notification, IREE_ALL_WAITERS);
  }
}

void iree_event_reset(iree_event_t* event) {
  if (!event) return;
  iree_futex_handle_t* futex = (iree_futex_handle_t*)event->value.local_futex;
  if (!futex) return;
  iree_atomic_store(&futex->value, 0, iree_memory_order_release);
}

#endif  // IREE_WAIT_API == IREE_WAIT_API_INPROC
