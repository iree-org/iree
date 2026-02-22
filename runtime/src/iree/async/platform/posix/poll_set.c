// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/platform/posix/poll_set.h"

#include <errno.h>

// Initial capacity for the pollfd array.
#define IREE_ASYNC_POSIX_POLL_SET_INITIAL_CAPACITY 16

// Minimum capacity for grow_array.
#define IREE_ASYNC_POSIX_POLL_SET_MIN_CAPACITY 32

iree_status_t iree_async_posix_poll_set_initialize(
    iree_allocator_t allocator, iree_async_posix_poll_set_t* out_poll_set) {
  IREE_TRACE_ZONE_BEGIN(z0);

  memset(out_poll_set, 0, sizeof(*out_poll_set));
  out_poll_set->allocator = allocator;

  // Allocate initial capacity.
  iree_status_t status = iree_allocator_malloc_array_uninitialized(
      allocator, IREE_ASYNC_POSIX_POLL_SET_INITIAL_CAPACITY,
      sizeof(struct pollfd), (void**)&out_poll_set->fds);
  if (iree_status_is_ok(status)) {
    out_poll_set->capacity = IREE_ASYNC_POSIX_POLL_SET_INITIAL_CAPACITY;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_async_posix_poll_set_deinitialize(
    iree_async_posix_poll_set_t* poll_set) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_allocator_free(poll_set->allocator, poll_set->fds);
  memset(poll_set, 0, sizeof(*poll_set));
  IREE_TRACE_ZONE_END(z0);
}

// Ensures capacity for at least one more fd.
static iree_status_t iree_async_posix_poll_set_ensure_capacity(
    iree_async_posix_poll_set_t* poll_set) {
  if (poll_set->count < poll_set->capacity) {
    return iree_ok_status();
  }

  // Grow the array using the standard grow pattern.
  iree_host_size_t new_capacity = poll_set->capacity;
  IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
      poll_set->allocator, IREE_ASYNC_POSIX_POLL_SET_MIN_CAPACITY,
      sizeof(struct pollfd), &new_capacity, (void**)&poll_set->fds));
  poll_set->capacity = new_capacity;

  return iree_ok_status();
}

iree_status_t iree_async_posix_poll_set_add(
    iree_async_posix_poll_set_t* poll_set, int fd, short events,
    iree_host_size_t* out_index) {
  IREE_RETURN_IF_ERROR(iree_async_posix_poll_set_ensure_capacity(poll_set));

  iree_host_size_t index = poll_set->count;
  poll_set->fds[index].fd = fd;
  poll_set->fds[index].events = events;
  poll_set->fds[index].revents = 0;
  poll_set->count++;

  if (out_index) *out_index = index;
  return iree_ok_status();
}

int iree_async_posix_poll_set_remove_at(iree_async_posix_poll_set_t* poll_set,
                                        iree_host_size_t index) {
  if (index >= poll_set->count) {
    return -1;  // Invalid index.
  }

  int removed_fd = poll_set->fds[index].fd;

  // Swap with last element (if not already last).
  iree_host_size_t last_index = poll_set->count - 1;
  if (index != last_index) {
    poll_set->fds[index] = poll_set->fds[last_index];
  }
  poll_set->count--;

  return removed_fd;
}

iree_host_size_t iree_async_posix_poll_set_find(
    const iree_async_posix_poll_set_t* poll_set, int fd) {
  for (iree_host_size_t i = 0; i < poll_set->count; ++i) {
    if (poll_set->fds[i].fd == fd) {
      return i;
    }
  }
  return IREE_HOST_SIZE_MAX;
}

iree_status_t iree_async_posix_poll_set_wait(
    iree_async_posix_poll_set_t* poll_set, int timeout_ms,
    iree_host_size_t* out_ready_count) {
  if (out_ready_count) *out_ready_count = 0;

  // Clear revents before polling.
  for (iree_host_size_t i = 0; i < poll_set->count; ++i) {
    poll_set->fds[i].revents = 0;
  }

  // Call poll().
  int result = poll(poll_set->fds, (nfds_t)poll_set->count, timeout_ms);

  if (result < 0) {
    int error = errno;
    if (error == EINTR) {
      // Interrupted by signal - treat as timeout with no events.
      return iree_ok_status();
    }
    return iree_make_status(iree_status_code_from_errno(error),
                            "poll() failed: %s", strerror(error));
  }

  if (result == 0) {
    // Timeout with no events.
    return iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED);
  }

  if (out_ready_count) *out_ready_count = (iree_host_size_t)result;
  return iree_ok_status();
}

void iree_async_posix_poll_set_clear(iree_async_posix_poll_set_t* poll_set) {
  poll_set->count = 0;
}
