// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Event set implementation using poll().
//
// This is the portable baseline implementation. It maintains a dynamic array
// of struct pollfd and calls poll() for event notification. Lookup and
// removal are O(n) operations.

#include <errno.h>
#include <string.h>

#include "iree/async/platform/posix/event_set.h"

// Initial capacity for the pollfd array.
#define IREE_EVENT_SET_POLL_INITIAL_CAPACITY 16

// Minimum capacity for grow_array.
#define IREE_EVENT_SET_POLL_MIN_GROW 32

// Poll-based event set implementation.
typedef struct iree_async_posix_event_set_poll_t {
  iree_async_posix_event_set_t base;  // Must be first.
  iree_allocator_t allocator;
  struct pollfd* fds;
  iree_host_size_t count;
  iree_host_size_t capacity;
  iree_host_size_t iteration_index;  // For next_ready iteration.
} iree_async_posix_event_set_poll_t;

static iree_async_posix_event_set_poll_t* iree_async_posix_event_set_poll_cast(
    iree_async_posix_event_set_t* event_set) {
  return (iree_async_posix_event_set_poll_t*)event_set;
}

//===----------------------------------------------------------------------===//
// Vtable implementations
//===----------------------------------------------------------------------===//

static void iree_async_posix_event_set_poll_free(
    iree_async_posix_event_set_t* base_event_set) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_async_posix_event_set_poll_t* event_set =
      iree_async_posix_event_set_poll_cast(base_event_set);
  iree_allocator_t allocator = event_set->allocator;
  iree_allocator_free(allocator, event_set->fds);
  iree_allocator_free(allocator, event_set);
  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_async_posix_event_set_poll_ensure_capacity(
    iree_async_posix_event_set_poll_t* event_set) {
  if (event_set->count < event_set->capacity) {
    return iree_ok_status();
  }
  iree_host_size_t new_capacity = event_set->capacity;
  IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
      event_set->allocator, IREE_EVENT_SET_POLL_MIN_GROW, sizeof(struct pollfd),
      &new_capacity, (void**)&event_set->fds));
  event_set->capacity = new_capacity;
  return iree_ok_status();
}

static iree_status_t iree_async_posix_event_set_poll_add(
    iree_async_posix_event_set_t* base_event_set, int fd, short events) {
  iree_async_posix_event_set_poll_t* event_set =
      iree_async_posix_event_set_poll_cast(base_event_set);
  IREE_RETURN_IF_ERROR(
      iree_async_posix_event_set_poll_ensure_capacity(event_set));

  iree_host_size_t index = event_set->count;
  event_set->fds[index].fd = fd;
  event_set->fds[index].events = events;
  event_set->fds[index].revents = 0;
  event_set->count++;
  return iree_ok_status();
}

// Finds the index of fd in the event set, or IREE_HOST_SIZE_MAX if not found.
static iree_host_size_t iree_async_posix_event_set_poll_find(
    iree_async_posix_event_set_poll_t* event_set, int fd) {
  for (iree_host_size_t i = 0; i < event_set->count; ++i) {
    if (event_set->fds[i].fd == fd) {
      return i;
    }
  }
  return IREE_HOST_SIZE_MAX;
}

static iree_status_t iree_async_posix_event_set_poll_modify(
    iree_async_posix_event_set_t* base_event_set, int fd, short events) {
  iree_async_posix_event_set_poll_t* event_set =
      iree_async_posix_event_set_poll_cast(base_event_set);
  iree_host_size_t index = iree_async_posix_event_set_poll_find(event_set, fd);
  if (index == IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_NOT_FOUND, "fd %d not in event set",
                            fd);
  }
  event_set->fds[index].events = events;
  return iree_ok_status();
}

static iree_status_t iree_async_posix_event_set_poll_remove(
    iree_async_posix_event_set_t* base_event_set, int fd) {
  iree_async_posix_event_set_poll_t* event_set =
      iree_async_posix_event_set_poll_cast(base_event_set);
  iree_host_size_t index = iree_async_posix_event_set_poll_find(event_set, fd);
  if (index == IREE_HOST_SIZE_MAX) {
    // Already removed - not an error.
    return iree_ok_status();
  }

  // Swap-remove: move last element to fill the gap.
  iree_host_size_t last_index = event_set->count - 1;
  if (index != last_index) {
    event_set->fds[index] = event_set->fds[last_index];
  }
  event_set->count--;

  // If we're removing an entry that the iterator has already passed, the
  // swap brought a not-yet-seen entry (from the tail) into the removed slot.
  // Rewind the iterator so the swapped entry is visited on the next call to
  // next_ready.
  if (index < event_set->iteration_index) {
    event_set->iteration_index--;
  }
  return iree_ok_status();
}

static iree_status_t iree_async_posix_event_set_poll_wait(
    iree_async_posix_event_set_t* base_event_set, int timeout_ms,
    iree_host_size_t* out_ready_count, bool* out_timed_out) {
  iree_async_posix_event_set_poll_t* event_set =
      iree_async_posix_event_set_poll_cast(base_event_set);
  if (out_ready_count) *out_ready_count = 0;
  *out_timed_out = false;

  // Clear revents before polling.
  for (iree_host_size_t i = 0; i < event_set->count; ++i) {
    event_set->fds[i].revents = 0;
  }

  int result = poll(event_set->fds, (nfds_t)event_set->count, timeout_ms);
  if (result < 0) {
    int error = errno;
    if (error == EINTR) {
      // Interrupted by signal - not a timeout, no events ready.
      return iree_ok_status();
    }
    return iree_make_status(iree_status_code_from_errno(error),
                            "poll() failed: %s", strerror(error));
  }

  if (result == 0) {
    *out_timed_out = true;
    return iree_ok_status();
  }

  if (out_ready_count) *out_ready_count = (iree_host_size_t)result;
  return iree_ok_status();
}

static void iree_async_posix_event_set_poll_reset_ready_iteration(
    iree_async_posix_event_set_t* base_event_set) {
  iree_async_posix_event_set_poll_t* event_set =
      iree_async_posix_event_set_poll_cast(base_event_set);
  event_set->iteration_index = 0;
}

static bool iree_async_posix_event_set_poll_next_ready(
    iree_async_posix_event_set_t* base_event_set, int* out_fd,
    short* out_revents) {
  iree_async_posix_event_set_poll_t* event_set =
      iree_async_posix_event_set_poll_cast(base_event_set);

  // Find the next fd with events.
  while (event_set->iteration_index < event_set->count) {
    iree_host_size_t index = event_set->iteration_index++;
    short revents = event_set->fds[index].revents;
    if (revents != 0) {
      *out_fd = event_set->fds[index].fd;
      *out_revents = revents;
      return true;
    }
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Vtable definition
//===----------------------------------------------------------------------===//

static const iree_async_posix_event_set_vtable_t
    iree_async_posix_event_set_poll_vtable = {
        .free = iree_async_posix_event_set_poll_free,
        .add = iree_async_posix_event_set_poll_add,
        .modify = iree_async_posix_event_set_poll_modify,
        .remove = iree_async_posix_event_set_poll_remove,
        .wait = iree_async_posix_event_set_poll_wait,
        .reset_ready_iteration =
            iree_async_posix_event_set_poll_reset_ready_iteration,
        .next_ready = iree_async_posix_event_set_poll_next_ready,
};

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

iree_status_t iree_async_posix_event_set_allocate_poll(
    iree_allocator_t allocator, iree_async_posix_event_set_t** out_event_set) {
  IREE_ASSERT_ARGUMENT(out_event_set);
  *out_event_set = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_async_posix_event_set_poll_t* event_set = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(allocator, sizeof(*event_set), (void**)&event_set));
  memset(event_set, 0, sizeof(*event_set));
  event_set->base.vtable = &iree_async_posix_event_set_poll_vtable;
  event_set->allocator = allocator;

  // Allocate initial capacity.
  iree_status_t status = iree_allocator_malloc_array_uninitialized(
      allocator, IREE_EVENT_SET_POLL_INITIAL_CAPACITY, sizeof(struct pollfd),
      (void**)&event_set->fds);
  if (iree_status_is_ok(status)) {
    event_set->capacity = IREE_EVENT_SET_POLL_INITIAL_CAPACITY;
    *out_event_set = &event_set->base;
  } else {
    iree_allocator_free(allocator, event_set);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_async_posix_event_set_allocate(
    iree_async_posix_event_backend_t backend, iree_allocator_t allocator,
    iree_async_posix_event_set_t** out_event_set) {
  switch (backend) {
    case IREE_ASYNC_POSIX_EVENT_BACKEND_DEFAULT:
#if defined(IREE_PLATFORM_LINUX)
      // Use epoll on Linux for O(1) performance.
      return iree_async_posix_event_set_allocate_epoll(allocator,
                                                       out_event_set);
#elif defined(IREE_PLATFORM_APPLE) || defined(IREE_PLATFORM_BSD)
      // Use kqueue on BSD/macOS for O(1) performance.
      return iree_async_posix_event_set_allocate_kqueue(allocator,
                                                        out_event_set);
#else
      // Fall through to poll on other platforms.
      return iree_async_posix_event_set_allocate_poll(allocator, out_event_set);
#endif
    case IREE_ASYNC_POSIX_EVENT_BACKEND_POLL:
      return iree_async_posix_event_set_allocate_poll(allocator, out_event_set);
    case IREE_ASYNC_POSIX_EVENT_BACKEND_EPOLL:
#if defined(IREE_PLATFORM_LINUX)
      return iree_async_posix_event_set_allocate_epoll(allocator,
                                                       out_event_set);
#else
      return iree_make_status(IREE_STATUS_UNAVAILABLE,
                              "epoll is only available on Linux");
#endif
    case IREE_ASYNC_POSIX_EVENT_BACKEND_KQUEUE:
#if defined(IREE_PLATFORM_APPLE) || defined(IREE_PLATFORM_BSD)
      return iree_async_posix_event_set_allocate_kqueue(allocator,
                                                        out_event_set);
#else
      return iree_make_status(IREE_STATUS_UNAVAILABLE,
                              "kqueue is only available on BSD and macOS");
#endif
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown event backend %d", (int)backend);
  }
}
