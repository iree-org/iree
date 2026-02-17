// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Event set implementation using BSD/macOS kqueue.
//
// kqueue provides O(1) event notification similar to Linux epoll. It uses
// a filter-based model where EVFILT_READ and EVFILT_WRITE correspond to
// poll()'s POLLIN and POLLOUT.
//
// Implementation notes:
// - kqueue is level-triggered by default, matching poll() semantics.
// - kevent() combines registration and waiting in a single syscall.
// - Closing an fd automatically removes it from the kqueue.
// - kqueue returns separate events for READ and WRITE on the same fd, but
//   poll() returns a combined bitmask. We coalesce events in next_ready()
//   to maintain poll() semantics.
//
// References:
// - https://man.openbsd.org/kqueue.2
// - https://github.com/libuv/libuv/blob/v1.x/src/unix/kqueue.c
// - https://github.com/libevent/libevent/blob/master/kqueue.c

#include "iree/async/platform/posix/event_set.h"

#if defined(IREE_PLATFORM_APPLE) || defined(IREE_PLATFORM_BSD)

#include <errno.h>
#include <fcntl.h>
#include <string.h>
#include <sys/event.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

// Fixed capacity for the kevent results buffer.
#define IREE_EVENT_SET_KQUEUE_EVENTS_CAPACITY 64

typedef struct iree_async_posix_event_set_kqueue_t {
  iree_async_posix_event_set_t base;
  iree_allocator_t allocator;
  int kqueue_fd;

  // Ready event iteration state.
  iree_host_size_t ready_count;
  iree_host_size_t iteration_index;

  // Fixed buffer for kevent() results.
  struct kevent events[IREE_EVENT_SET_KQUEUE_EVENTS_CAPACITY];
} iree_async_posix_event_set_kqueue_t;

static iree_async_posix_event_set_kqueue_t*
iree_async_posix_event_set_kqueue_cast(
    iree_async_posix_event_set_t* event_set) {
  return (iree_async_posix_event_set_kqueue_t*)event_set;
}

//===----------------------------------------------------------------------===//
// Event translation
//===----------------------------------------------------------------------===//

static short iree_kevent_to_poll_events(const struct kevent* event) {
  short poll_events = 0;
  switch (event->filter) {
    case EVFILT_READ:
      poll_events |= POLLIN;
      break;
    case EVFILT_WRITE:
      poll_events |= POLLOUT;
      break;
  }
  if (event->flags & EV_EOF) poll_events |= POLLHUP;
  if (event->flags & EV_ERROR) poll_events |= POLLERR;
  return poll_events;
}

// Deletes a single filter from the kqueue. Returns OK if deleted or if the
// filter wasn't registered (ENOENT) or fd was closed (EBADF).
static iree_status_t iree_kqueue_delete_filter(int kqueue_fd, int fd,
                                               int16_t filter) {
  struct kevent kev;
  EV_SET(&kev, fd, filter, EV_DELETE, 0, 0, NULL);
  if (kevent(kqueue_fd, &kev, 1, NULL, 0, NULL) < 0) {
    int error = errno;
    // ENOENT: filter not registered - already removed, success.
    // EBADF: fd closed - automatically removed from kqueue, success.
    if (error == ENOENT || error == EBADF) {
      return iree_ok_status();
    }
    return iree_make_status(iree_status_code_from_errno(error),
                            "kevent(DELETE) failed for fd %d filter %d: %s", fd,
                            filter, strerror(error));
  }
  return iree_ok_status();
}

// Adds or updates a filter on the kqueue. EV_ADD updates existing filters.
static iree_status_t iree_kqueue_add_filter(int kqueue_fd, int fd,
                                            int16_t filter) {
  struct kevent kev;
  EV_SET(&kev, fd, filter, EV_ADD, 0, 0, NULL);
  if (kevent(kqueue_fd, &kev, 1, NULL, 0, NULL) < 0) {
    int error = errno;
    return iree_make_status(iree_status_code_from_errno(error),
                            "kevent(ADD) failed for fd %d filter %d: %s", fd,
                            filter, strerror(error));
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Vtable implementations
//===----------------------------------------------------------------------===//

static void iree_async_posix_event_set_kqueue_free(
    iree_async_posix_event_set_t* base_event_set) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_async_posix_event_set_kqueue_t* event_set =
      iree_async_posix_event_set_kqueue_cast(base_event_set);
  iree_allocator_t allocator = event_set->allocator;
  if (event_set->kqueue_fd >= 0) {
    close(event_set->kqueue_fd);
  }
  iree_allocator_free(allocator, event_set);
  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_async_posix_event_set_kqueue_add(
    iree_async_posix_event_set_t* base_event_set, int fd, short events) {
  iree_async_posix_event_set_kqueue_t* event_set =
      iree_async_posix_event_set_kqueue_cast(base_event_set);

  iree_status_t status = iree_ok_status();
  if (iree_status_is_ok(status) && (events & POLLIN)) {
    status = iree_kqueue_add_filter(event_set->kqueue_fd, fd, EVFILT_READ);
  }
  if (iree_status_is_ok(status) && (events & POLLOUT)) {
    status = iree_kqueue_add_filter(event_set->kqueue_fd, fd, EVFILT_WRITE);
  }
  return status;
}

static iree_status_t iree_async_posix_event_set_kqueue_modify(
    iree_async_posix_event_set_t* base_event_set, int fd, short events) {
  iree_async_posix_event_set_kqueue_t* event_set =
      iree_async_posix_event_set_kqueue_cast(base_event_set);

  // EV_ADD updates existing filters, so we just need to:
  // 1. Delete filters that are NOT in the new mask
  // 2. Add filters that ARE in the new mask (EV_ADD handles updates)
  iree_status_t status = iree_ok_status();

  // Delete READ if not wanted.
  if (iree_status_is_ok(status) && !(events & POLLIN)) {
    status = iree_kqueue_delete_filter(event_set->kqueue_fd, fd, EVFILT_READ);
  }
  // Delete WRITE if not wanted.
  if (iree_status_is_ok(status) && !(events & POLLOUT)) {
    status = iree_kqueue_delete_filter(event_set->kqueue_fd, fd, EVFILT_WRITE);
  }
  // Add/update READ if wanted.
  if (iree_status_is_ok(status) && (events & POLLIN)) {
    status = iree_kqueue_add_filter(event_set->kqueue_fd, fd, EVFILT_READ);
  }
  // Add/update WRITE if wanted.
  if (iree_status_is_ok(status) && (events & POLLOUT)) {
    status = iree_kqueue_add_filter(event_set->kqueue_fd, fd, EVFILT_WRITE);
  }
  return status;
}

static iree_status_t iree_async_posix_event_set_kqueue_remove(
    iree_async_posix_event_set_t* base_event_set, int fd) {
  iree_async_posix_event_set_kqueue_t* event_set =
      iree_async_posix_event_set_kqueue_cast(base_event_set);

  // Delete both filters individually to avoid batch failure.
  iree_status_t status = iree_ok_status();
  if (iree_status_is_ok(status)) {
    status = iree_kqueue_delete_filter(event_set->kqueue_fd, fd, EVFILT_READ);
  }
  if (iree_status_is_ok(status)) {
    status = iree_kqueue_delete_filter(event_set->kqueue_fd, fd, EVFILT_WRITE);
  }
  return status;
}

static iree_status_t iree_async_posix_event_set_kqueue_wait(
    iree_async_posix_event_set_t* base_event_set, int timeout_ms,
    iree_host_size_t* out_ready_count, bool* out_timed_out) {
  iree_async_posix_event_set_kqueue_t* event_set =
      iree_async_posix_event_set_kqueue_cast(base_event_set);
  if (out_ready_count) *out_ready_count = 0;
  *out_timed_out = false;

  event_set->ready_count = 0;
  event_set->iteration_index = 0;

  struct timespec timeout_spec;
  struct timespec* timeout_ptr = NULL;
  if (timeout_ms >= 0) {
    timeout_spec.tv_sec = timeout_ms / 1000;
    timeout_spec.tv_nsec = (timeout_ms % 1000) * 1000000;
    timeout_ptr = &timeout_spec;
  }

  int result = kevent(event_set->kqueue_fd, NULL, 0, event_set->events,
                      IREE_EVENT_SET_KQUEUE_EVENTS_CAPACITY, timeout_ptr);
  if (result < 0) {
    int error = errno;
    if (error == EINTR) {
      // Interrupted by signal - not a timeout, no events ready.
      return iree_ok_status();
    }
    return iree_make_status(iree_status_code_from_errno(error),
                            "kevent() failed: %s", strerror(error));
  }

  if (result == 0) {
    *out_timed_out = true;
    return iree_ok_status();
  }

  event_set->ready_count = (iree_host_size_t)result;
  if (out_ready_count) *out_ready_count = event_set->ready_count;
  return iree_ok_status();
}

static void iree_async_posix_event_set_kqueue_reset_ready_iteration(
    iree_async_posix_event_set_t* base_event_set) {
  iree_async_posix_event_set_kqueue_t* event_set =
      iree_async_posix_event_set_kqueue_cast(base_event_set);
  event_set->iteration_index = 0;
}

static bool iree_async_posix_event_set_kqueue_next_ready(
    iree_async_posix_event_set_t* base_event_set, int* out_fd,
    short* out_revents) {
  iree_async_posix_event_set_kqueue_t* event_set =
      iree_async_posix_event_set_kqueue_cast(base_event_set);

  if (event_set->iteration_index >= event_set->ready_count) {
    return false;
  }

  // Coalesce events for the same fd to match poll() semantics.
  // kqueue returns separate events for READ and WRITE, but poll returns
  // a combined bitmask per fd.
  iree_host_size_t index = event_set->iteration_index++;
  const struct kevent* event = &event_set->events[index];
  int fd = (int)event->ident;
  short revents = iree_kevent_to_poll_events(event);

  // Look ahead for additional events on the same fd and coalesce them.
  while (event_set->iteration_index < event_set->ready_count) {
    const struct kevent* next = &event_set->events[event_set->iteration_index];
    if ((int)next->ident != fd) break;
    revents |= iree_kevent_to_poll_events(next);
    event_set->iteration_index++;
  }

  *out_fd = fd;
  *out_revents = revents;
  return true;
}

//===----------------------------------------------------------------------===//
// Vtable
//===----------------------------------------------------------------------===//

static const iree_async_posix_event_set_vtable_t
    iree_async_posix_event_set_kqueue_vtable = {
        .free = iree_async_posix_event_set_kqueue_free,
        .add = iree_async_posix_event_set_kqueue_add,
        .modify = iree_async_posix_event_set_kqueue_modify,
        .remove = iree_async_posix_event_set_kqueue_remove,
        .wait = iree_async_posix_event_set_kqueue_wait,
        .reset_ready_iteration =
            iree_async_posix_event_set_kqueue_reset_ready_iteration,
        .next_ready = iree_async_posix_event_set_kqueue_next_ready,
};

iree_status_t iree_async_posix_event_set_allocate_kqueue(
    iree_allocator_t allocator, iree_async_posix_event_set_t** out_event_set) {
  IREE_ASSERT_ARGUMENT(out_event_set);
  *out_event_set = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_async_posix_event_set_kqueue_t* event_set = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(allocator, sizeof(*event_set), (void**)&event_set));
  memset(event_set, 0, sizeof(*event_set));
  event_set->base.vtable = &iree_async_posix_event_set_kqueue_vtable;
  event_set->allocator = allocator;
  event_set->kqueue_fd = -1;

  event_set->kqueue_fd = kqueue();
  if (event_set->kqueue_fd < 0) {
    int error = errno;
    iree_allocator_free(allocator, event_set);
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(iree_status_code_from_errno(error),
                            "kqueue() failed: %s", strerror(error));
  }

  // Set close-on-exec to prevent fd leakage across fork/exec.
  fcntl(event_set->kqueue_fd, F_SETFD, FD_CLOEXEC);

  *out_event_set = &event_set->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

#endif  // IREE_PLATFORM_APPLE || IREE_PLATFORM_BSD
