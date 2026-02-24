// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Abstract interface for fd event monitoring.
//
// The event_set provides a vtable-based abstraction over different event
// notification mechanisms:
// - poll(): O(n) portable baseline (event_set_poll.c)
// - epoll: O(1) Linux-specific (event_set_epoll.c)
// - kqueue: O(1) BSD/macOS (event_set_kqueue.c)
//
// All implementations use POLLIN/POLLOUT/POLLERR event masks for portability.
// The vtable allows runtime selection of the backend for testing.

#ifndef IREE_ASYNC_PLATFORM_POSIX_EVENT_SET_H_
#define IREE_ASYNC_PLATFORM_POSIX_EVENT_SET_H_

#include <poll.h>

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_async_posix_event_set_t iree_async_posix_event_set_t;

typedef struct iree_async_posix_event_set_vtable_t {
  void (*free)(iree_async_posix_event_set_t* event_set);
  iree_status_t (*add)(iree_async_posix_event_set_t* event_set, int fd,
                       short events);
  iree_status_t (*modify)(iree_async_posix_event_set_t* event_set, int fd,
                          short events);
  iree_status_t (*remove)(iree_async_posix_event_set_t* event_set, int fd);
  iree_status_t (*wait)(iree_async_posix_event_set_t* event_set, int timeout_ms,
                        iree_host_size_t* out_ready_count, bool* out_timed_out);
  void (*reset_ready_iteration)(iree_async_posix_event_set_t* event_set);
  bool (*next_ready)(iree_async_posix_event_set_t* event_set, int* out_fd,
                     short* out_revents);
} iree_async_posix_event_set_vtable_t;

struct iree_async_posix_event_set_t {
  const iree_async_posix_event_set_vtable_t* vtable;
};

//===----------------------------------------------------------------------===//
// Event set operations
//===----------------------------------------------------------------------===//

// Frees an event set allocated with one of the allocate functions.
static inline void iree_async_posix_event_set_free(
    iree_async_posix_event_set_t* event_set) {
  if (event_set) event_set->vtable->free(event_set);
}

// Adds |fd| to the event set with the given event interest mask.
// |events| uses poll() constants: POLLIN, POLLOUT, POLLERR, etc.
static inline iree_status_t iree_async_posix_event_set_add(
    iree_async_posix_event_set_t* event_set, int fd, short events) {
  return event_set->vtable->add(event_set, fd, events);
}

// Modifies the event interest mask for |fd|.
// Returns IREE_STATUS_NOT_FOUND if |fd| was not previously added.
static inline iree_status_t iree_async_posix_event_set_modify(
    iree_async_posix_event_set_t* event_set, int fd, short events) {
  return event_set->vtable->modify(event_set, fd, events);
}

// Removes |fd| from the event set.
// Silently succeeds if |fd| was already removed or never added.
static inline iree_status_t iree_async_posix_event_set_remove(
    iree_async_posix_event_set_t* event_set, int fd) {
  return event_set->vtable->remove(event_set, fd);
}

// Waits for events on fds in the event set.
//
// |timeout_ms| controls blocking behavior:
//   - < 0: Block indefinitely until an event occurs.
//   - 0: Non-blocking poll, returns immediately.
//   - > 0: Block for at most this many milliseconds.
//
// On success, |out_ready_count| (if non-NULL) receives the number of fds with
// events. Use reset_ready_iteration() and next_ready() to enumerate them.
//
// On timeout, sets |*out_timed_out| to true and returns iree_ok_status().
// Timeouts are expected conditions and not errors.
// On signal interruption, returns iree_ok_status() with ready_count 0 and
// |*out_timed_out| false.
static inline iree_status_t iree_async_posix_event_set_wait(
    iree_async_posix_event_set_t* event_set, int timeout_ms,
    iree_host_size_t* out_ready_count, bool* out_timed_out) {
  return event_set->vtable->wait(event_set, timeout_ms, out_ready_count,
                                 out_timed_out);
}

// Resets the ready fd iterator to the beginning.
// Call this before iterating with next_ready() after wait() returns.
static inline void iree_async_posix_event_set_reset_ready_iteration(
    iree_async_posix_event_set_t* event_set) {
  event_set->vtable->reset_ready_iteration(event_set);
}

// Returns the next ready fd and the events that occurred on it.
// Returns true if a ready fd was returned, false if iteration is complete.
// |out_revents| receives the events: POLLIN, POLLOUT, POLLERR, POLLHUP, etc.
// The iteration order is unspecified.
static inline bool iree_async_posix_event_set_next_ready(
    iree_async_posix_event_set_t* event_set, int* out_fd, short* out_revents) {
  return event_set->vtable->next_ready(event_set, out_fd, out_revents);
}

//===----------------------------------------------------------------------===//
// Backend selection
//===----------------------------------------------------------------------===//

typedef enum iree_async_posix_event_backend_e {
  // Selects the best backend for the platform.
  // Linux: epoll, BSD/macOS: kqueue, Other: poll
  IREE_ASYNC_POSIX_EVENT_BACKEND_DEFAULT = 0,
  IREE_ASYNC_POSIX_EVENT_BACKEND_POLL,
  IREE_ASYNC_POSIX_EVENT_BACKEND_EPOLL,
  IREE_ASYNC_POSIX_EVENT_BACKEND_KQUEUE,
} iree_async_posix_event_backend_t;

// Allocates an event set using the specified |backend|.
// Returns IREE_STATUS_UNAVAILABLE if |backend| is not supported on this
// platform (e.g., EPOLL on macOS returns UNAVAILABLE).
iree_status_t iree_async_posix_event_set_allocate(
    iree_async_posix_event_backend_t backend, iree_allocator_t allocator,
    iree_async_posix_event_set_t** out_event_set);

// Allocates an event set using poll(). Available on all POSIX platforms.
iree_status_t iree_async_posix_event_set_allocate_poll(
    iree_allocator_t allocator, iree_async_posix_event_set_t** out_event_set);

#if defined(IREE_PLATFORM_LINUX)
// Allocates an event set using Linux epoll for O(1) event notification.
iree_status_t iree_async_posix_event_set_allocate_epoll(
    iree_allocator_t allocator, iree_async_posix_event_set_t** out_event_set);
#endif  // IREE_PLATFORM_LINUX

#if defined(IREE_PLATFORM_APPLE) || defined(IREE_PLATFORM_BSD)
// Allocates an event set using BSD/macOS kqueue for O(1) event notification.
iree_status_t iree_async_posix_event_set_allocate_kqueue(
    iree_allocator_t allocator, iree_async_posix_event_set_t** out_event_set);
#endif  // IREE_PLATFORM_APPLE || IREE_PLATFORM_BSD

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_ASYNC_PLATFORM_POSIX_EVENT_SET_H_
