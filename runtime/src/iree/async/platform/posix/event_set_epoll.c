// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Event set implementation using Linux epoll.
//
// epoll provides O(1) event notification for ready fds, compared to poll()'s
// O(n) scanning. The epoll fd is level-triggered by default, matching poll()
// semantics.
//
// Implementation notes:
// - Uses epoll_create1(EPOLL_CLOEXEC) to avoid fd leaks across fork/exec.
// - Translates between poll events (POLLIN/OUT/ERR) and epoll events.
// - Stores ready events in a buffer for iteration after epoll_wait().
// - The events buffer is sized to max_events and reused across waits.

#include "iree/base/api.h"

#if defined(IREE_PLATFORM_LINUX)

#include <errno.h>
#include <string.h>
#include <sys/epoll.h>
#include <unistd.h>

#include "iree/async/platform/posix/event_set.h"

// Fixed capacity for the epoll events buffer.
// If more fds are ready than this, level-triggered epoll will return them on
// the next wait() call.
#define IREE_EVENT_SET_EPOLL_EVENTS_CAPACITY 64

// Epoll-based event set implementation.
typedef struct iree_async_posix_event_set_epoll_t {
  iree_async_posix_event_set_t base;  // Must be first.
  iree_allocator_t allocator;
  int epoll_fd;

  // Ready event iteration state.
  iree_host_size_t ready_count;      // Number of ready events from last wait.
  iree_host_size_t iteration_index;  // Current position in iteration.

  // Fixed buffer for epoll_wait() results.
  struct epoll_event events[IREE_EVENT_SET_EPOLL_EVENTS_CAPACITY];
} iree_async_posix_event_set_epoll_t;

static iree_async_posix_event_set_epoll_t*
iree_async_posix_event_set_epoll_cast(iree_async_posix_event_set_t* event_set) {
  return (iree_async_posix_event_set_epoll_t*)event_set;
}

//===----------------------------------------------------------------------===//
// Event translation
//===----------------------------------------------------------------------===//

// Converts poll() event flags to epoll event flags.
static uint32_t iree_poll_events_to_epoll(short poll_events) {
  uint32_t epoll_events = 0;
  if (poll_events & POLLIN) epoll_events |= EPOLLIN;
  if (poll_events & POLLOUT) epoll_events |= EPOLLOUT;
  if (poll_events & POLLPRI) epoll_events |= EPOLLPRI;
  // POLLERR and POLLHUP are always reported by epoll, no need to set them.
  return epoll_events;
}

// Converts epoll event flags to poll() event flags.
static short iree_epoll_events_to_poll(uint32_t epoll_events) {
  short poll_events = 0;
  if (epoll_events & EPOLLIN) poll_events |= POLLIN;
  if (epoll_events & EPOLLOUT) poll_events |= POLLOUT;
  if (epoll_events & EPOLLPRI) poll_events |= POLLPRI;
  if (epoll_events & EPOLLERR) poll_events |= POLLERR;
  if (epoll_events & EPOLLHUP) poll_events |= POLLHUP;
  return poll_events;
}

//===----------------------------------------------------------------------===//
// Vtable implementations
//===----------------------------------------------------------------------===//

static void iree_async_posix_event_set_epoll_free(
    iree_async_posix_event_set_t* base_event_set) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_async_posix_event_set_epoll_t* event_set =
      iree_async_posix_event_set_epoll_cast(base_event_set);
  iree_allocator_t allocator = event_set->allocator;

  if (event_set->epoll_fd >= 0) {
    close(event_set->epoll_fd);
  }
  iree_allocator_free(allocator, event_set);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_async_posix_event_set_epoll_add(
    iree_async_posix_event_set_t* base_event_set, int fd, short events) {
  iree_async_posix_event_set_epoll_t* event_set =
      iree_async_posix_event_set_epoll_cast(base_event_set);

  struct epoll_event ev = {0};
  ev.events = iree_poll_events_to_epoll(events);
  ev.data.fd = fd;

  if (epoll_ctl(event_set->epoll_fd, EPOLL_CTL_ADD, fd, &ev) < 0) {
    int error = errno;
    return iree_make_status(iree_status_code_from_errno(error),
                            "epoll_ctl(ADD) failed for fd %d: %s", fd,
                            strerror(error));
  }
  return iree_ok_status();
}

static iree_status_t iree_async_posix_event_set_epoll_modify(
    iree_async_posix_event_set_t* base_event_set, int fd, short events) {
  iree_async_posix_event_set_epoll_t* event_set =
      iree_async_posix_event_set_epoll_cast(base_event_set);

  struct epoll_event ev = {0};
  ev.events = iree_poll_events_to_epoll(events);
  ev.data.fd = fd;

  if (epoll_ctl(event_set->epoll_fd, EPOLL_CTL_MOD, fd, &ev) < 0) {
    int error = errno;
    if (error == ENOENT) {
      return iree_make_status(IREE_STATUS_NOT_FOUND, "fd %d not in event set",
                              fd);
    }
    return iree_make_status(iree_status_code_from_errno(error),
                            "epoll_ctl(MOD) failed for fd %d: %s", fd,
                            strerror(error));
  }
  return iree_ok_status();
}

static iree_status_t iree_async_posix_event_set_epoll_remove(
    iree_async_posix_event_set_t* base_event_set, int fd) {
  iree_async_posix_event_set_epoll_t* event_set =
      iree_async_posix_event_set_epoll_cast(base_event_set);

  // epoll_ctl with EPOLL_CTL_DEL ignores the event parameter (can be NULL on
  // Linux 2.6.9+), but we pass a valid pointer for older kernel compatibility.
  struct epoll_event ev = {0};
  if (epoll_ctl(event_set->epoll_fd, EPOLL_CTL_DEL, fd, &ev) < 0) {
    int error = errno;
    if (error == ENOENT) {
      // Already removed - not an error (matches poll implementation).
      return iree_ok_status();
    }
    return iree_make_status(iree_status_code_from_errno(error),
                            "epoll_ctl(DEL) failed for fd %d: %s", fd,
                            strerror(error));
  }
  return iree_ok_status();
}

static iree_status_t iree_async_posix_event_set_epoll_wait(
    iree_async_posix_event_set_t* base_event_set, int timeout_ms,
    iree_host_size_t* out_ready_count, bool* out_timed_out) {
  iree_async_posix_event_set_epoll_t* event_set =
      iree_async_posix_event_set_epoll_cast(base_event_set);
  if (out_ready_count) *out_ready_count = 0;
  *out_timed_out = false;

  // Reset iteration state.
  event_set->ready_count = 0;
  event_set->iteration_index = 0;

  int result = epoll_wait(event_set->epoll_fd, event_set->events,
                          IREE_EVENT_SET_EPOLL_EVENTS_CAPACITY, timeout_ms);
  if (result < 0) {
    int error = errno;
    if (error == EINTR) {
      // Interrupted by signal - not a timeout, no events ready.
      return iree_ok_status();
    }
    return iree_make_status(iree_status_code_from_errno(error),
                            "epoll_wait() failed: %s", strerror(error));
  }

  if (result == 0) {
    *out_timed_out = true;
    return iree_ok_status();
  }

  event_set->ready_count = (iree_host_size_t)result;
  if (out_ready_count) *out_ready_count = event_set->ready_count;
  return iree_ok_status();
}

static void iree_async_posix_event_set_epoll_reset_ready_iteration(
    iree_async_posix_event_set_t* base_event_set) {
  iree_async_posix_event_set_epoll_t* event_set =
      iree_async_posix_event_set_epoll_cast(base_event_set);
  event_set->iteration_index = 0;
}

static bool iree_async_posix_event_set_epoll_next_ready(
    iree_async_posix_event_set_t* base_event_set, int* out_fd,
    short* out_revents) {
  iree_async_posix_event_set_epoll_t* event_set =
      iree_async_posix_event_set_epoll_cast(base_event_set);

  if (event_set->iteration_index >= event_set->ready_count) {
    return false;
  }

  iree_host_size_t index = event_set->iteration_index++;
  *out_fd = event_set->events[index].data.fd;
  *out_revents = iree_epoll_events_to_poll(event_set->events[index].events);
  return true;
}

//===----------------------------------------------------------------------===//
// Vtable definition
//===----------------------------------------------------------------------===//

static const iree_async_posix_event_set_vtable_t
    iree_async_posix_event_set_epoll_vtable = {
        .free = iree_async_posix_event_set_epoll_free,
        .add = iree_async_posix_event_set_epoll_add,
        .modify = iree_async_posix_event_set_epoll_modify,
        .remove = iree_async_posix_event_set_epoll_remove,
        .wait = iree_async_posix_event_set_epoll_wait,
        .reset_ready_iteration =
            iree_async_posix_event_set_epoll_reset_ready_iteration,
        .next_ready = iree_async_posix_event_set_epoll_next_ready,
};

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

iree_status_t iree_async_posix_event_set_allocate_epoll(
    iree_allocator_t allocator, iree_async_posix_event_set_t** out_event_set) {
  IREE_ASSERT_ARGUMENT(out_event_set);
  *out_event_set = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_async_posix_event_set_epoll_t* event_set = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(allocator, sizeof(*event_set), (void**)&event_set));
  memset(event_set, 0, sizeof(*event_set));
  event_set->base.vtable = &iree_async_posix_event_set_epoll_vtable;
  event_set->allocator = allocator;
  event_set->epoll_fd = -1;

  // Create epoll fd with close-on-exec flag.
  event_set->epoll_fd = epoll_create1(EPOLL_CLOEXEC);
  if (event_set->epoll_fd < 0) {
    int error = errno;
    iree_allocator_free(allocator, event_set);
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(iree_status_code_from_errno(error),
                            "epoll_create1() failed: %s", strerror(error));
  }

  *out_event_set = &event_set->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

#endif  // IREE_PLATFORM_LINUX
