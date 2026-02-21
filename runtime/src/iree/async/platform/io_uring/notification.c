// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/platform/io_uring/notification.h"

#include <errno.h>
#include <poll.h>
#include <string.h>
#include <sys/eventfd.h>
#include <unistd.h>

#include "iree/async/platform/io_uring/proactor.h"
#include "iree/async/proactor.h"
#include "iree/base/threading/futex.h"

//===----------------------------------------------------------------------===//
// Creation and destruction
//===----------------------------------------------------------------------===//

iree_status_t iree_async_io_uring_notification_create(
    iree_async_proactor_io_uring_t* proactor,
    iree_async_notification_flags_t flags,
    iree_async_notification_t** out_notification) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(out_notification);
  *out_notification = NULL;

  iree_allocator_t allocator = proactor->base.allocator;

  iree_async_notification_t* notification = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(allocator, sizeof(*notification),
                                (void**)&notification));
  memset(notification, 0, sizeof(*notification));

  iree_atomic_ref_count_init(&notification->ref_count);
  notification->proactor = &proactor->base;
  iree_atomic_store(&notification->epoch, 0, iree_memory_order_release);

  // Select mode based on proactor capabilities.
  iree_status_t status = iree_ok_status();
#if defined(IREE_RUNTIME_USE_FUTEX)
  if (iree_any_bit_set(proactor->capabilities,
                       IREE_ASYNC_PROACTOR_CAPABILITY_FUTEX_OPERATIONS)) {
    notification->mode = IREE_ASYNC_NOTIFICATION_MODE_FUTEX;
  } else
#endif  // IREE_RUNTIME_USE_FUTEX
  {
    notification->mode = IREE_ASYNC_NOTIFICATION_MODE_EVENT;
    // EFD_SEMAPHORE makes read() decrement by 1 instead of draining the
    // counter. This allows wake_count to control how many waiters are woken.
    int eventfd_result = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK | EFD_SEMAPHORE);
    if (eventfd_result >= 0) {
      notification->platform.io_uring.primitive =
          iree_async_primitive_from_fd(eventfd_result);
      notification->platform.io_uring.drain_buffer = 0;
    } else {
      status = iree_make_status(iree_status_code_from_errno(errno),
                                "eventfd creation failed (%d)", errno);
    }
  }

  if (iree_status_is_ok(status)) {
    *out_notification = notification;
  } else {
    iree_allocator_free(allocator, notification);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_async_io_uring_notification_destroy(
    iree_async_proactor_io_uring_t* proactor,
    iree_async_notification_t* notification) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!notification) {
    IREE_TRACE_ZONE_END(z0);
    return;
  }

  iree_allocator_t allocator = proactor->base.allocator;

  if (notification->mode == IREE_ASYNC_NOTIFICATION_MODE_EVENT) {
    if (notification->platform.io_uring.primitive.value.fd >= 0) {
      close(notification->platform.io_uring.primitive.value.fd);
    }
  }

  iree_allocator_free(allocator, notification);
  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// Vtable implementations for signal and wait
//===----------------------------------------------------------------------===//

void iree_async_io_uring_notification_signal(
    iree_async_proactor_t* base_proactor,
    iree_async_notification_t* notification, int32_t wake_count) {
#if defined(IREE_RUNTIME_USE_FUTEX)
  if (notification->mode == IREE_ASYNC_NOTIFICATION_MODE_FUTEX) {
    // Add the count of relays with in-flight FUTEX_WAIT operations on this
    // notification. Each relay is an implicit kernel-side futex waiter that the
    // caller cannot account for. Without this, signaling with wake_count=1
    // when N relays are attached would only wake 1 of the N kernel waiters,
    // leaving N-1 relays permanently stuck.
    int32_t relay_count =
        iree_atomic_load(&notification->platform.io_uring.futex_relay_count,
                         iree_memory_order_acquire);
    int32_t total_wake_count = wake_count;
    if (relay_count > 0) {
      if (wake_count > INT32_MAX - relay_count) {
        total_wake_count = INT32_MAX;
      } else {
        total_wake_count = wake_count + relay_count;
      }
    }
    iree_futex_wake(&notification->epoch, total_wake_count);
    return;
  }
#endif  // IREE_RUNTIME_USE_FUTEX

  // With EFD_SEMAPHORE, write(N) allows N read()s to succeed.
  uint64_t value = (wake_count > 0) ? (uint64_t)wake_count : UINT32_MAX;
  ssize_t result = write(notification->platform.io_uring.primitive.value.fd,
                         &value, sizeof(value));
  // The vtable signature returns void so we cannot propagate errors, but an
  // eventfd write failure means waiters will not be woken — assert in debug.
  IREE_ASSERT(result == sizeof(value),
              "eventfd write failed during notification signal: %zd (errno=%d)",
              result, errno);
}

bool iree_async_io_uring_notification_wait(
    iree_async_proactor_t* base_proactor,
    iree_async_notification_t* notification, iree_timeout_t timeout) {
  iree_time_t deadline_ns = iree_timeout_as_deadline_ns(timeout);

  // Capture current epoch before waiting.
  uint32_t wait_epoch =
      iree_atomic_load(&notification->epoch, iree_memory_order_acquire);

#if defined(IREE_RUNTIME_USE_FUTEX)
  if (notification->mode == IREE_ASYNC_NOTIFICATION_MODE_FUTEX) {
    while (iree_time_now() < deadline_ns) {
      uint32_t current_epoch =
          iree_atomic_load(&notification->epoch, iree_memory_order_acquire);
      if (current_epoch != wait_epoch) return true;

      iree_status_code_t status_code =
          iree_futex_wait(&notification->epoch, wait_epoch, deadline_ns);
      if (status_code == IREE_STATUS_DEADLINE_EXCEEDED) return false;
    }

    uint32_t final_epoch =
        iree_atomic_load(&notification->epoch, iree_memory_order_acquire);
    return final_epoch != wait_epoch;
  }
#endif  // IREE_RUNTIME_USE_FUTEX

  // Event mode: poll on eventfd, check epoch after wakeup.
  int fd = notification->platform.io_uring.primitive.value.fd;

  while (iree_time_now() < deadline_ns) {
    uint32_t current_epoch =
        iree_atomic_load(&notification->epoch, iree_memory_order_acquire);
    if (current_epoch != wait_epoch) return true;

    iree_duration_t remaining_ns = deadline_ns - iree_time_now();
    if (remaining_ns <= 0) break;
    int timeout_ms = (int)(remaining_ns / 1000000);
    if (timeout_ms <= 0) timeout_ms = 1;

    struct pollfd pfd = {.fd = fd, .events = POLLIN, .revents = 0};
    int result = poll(&pfd, 1, timeout_ms);
    if (result < 0) {
      if (errno == EINTR) continue;
      return false;
    }

    if (pfd.revents & POLLIN) {
      uint64_t value;
      ssize_t result = read(fd, &value, sizeof(value));
      // EAGAIN means counter is already 0 (spurious wake, benign).
      IREE_ASSERT(result >= 0 || errno == EAGAIN);
    }

    if (pfd.revents & (POLLHUP | POLLERR)) return false;
  }

  uint32_t final_epoch =
      iree_atomic_load(&notification->epoch, iree_memory_order_acquire);
  return final_epoch != wait_epoch;
}
