// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Wrapper around poll() system call for fd readiness monitoring.
//
// Provides a dynamic set of file descriptors to monitor with poll().
// Uses swap-remove for O(1) removal since fd ordering doesn't matter.

#ifndef IREE_ASYNC_PLATFORM_POSIX_POLL_SET_H_
#define IREE_ASYNC_PLATFORM_POSIX_POLL_SET_H_

#include <poll.h>

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Dynamic set of file descriptors for poll().
//
// Thread-safety: NOT thread-safe. All operations must be serialized
// (typically called only from the poll thread).
typedef struct iree_async_posix_poll_set_t {
  struct pollfd* fds;         // Dynamic array of pollfds.
  iree_host_size_t count;     // Current number of fds.
  iree_host_size_t capacity;  // Allocated capacity.
  iree_allocator_t allocator;
} iree_async_posix_poll_set_t;

// Initializes an empty poll set.
iree_status_t iree_async_posix_poll_set_initialize(
    iree_allocator_t allocator, iree_async_posix_poll_set_t* out_poll_set);

// Deinitializes the poll set and frees storage.
void iree_async_posix_poll_set_deinitialize(
    iree_async_posix_poll_set_t* poll_set);

// Returns true if the poll set is empty.
static inline bool iree_async_posix_poll_set_is_empty(
    const iree_async_posix_poll_set_t* poll_set) {
  return poll_set->count == 0;
}

// Returns the number of fds in the poll set.
static inline iree_host_size_t iree_async_posix_poll_set_size(
    const iree_async_posix_poll_set_t* poll_set) {
  return poll_set->count;
}

// Adds an fd to the poll set with the given events (POLLIN, POLLOUT, etc).
// Returns the index where the fd was added (for use with remove_at).
// Note: The same fd can be added multiple times (for different operations).
iree_status_t iree_async_posix_poll_set_add(
    iree_async_posix_poll_set_t* poll_set, int fd, short events,
    iree_host_size_t* out_index);

// Removes the fd at the given index using swap-remove.
// The last element is moved to fill the gap, so indices of other fds may
// change. Callers tracking indices must update them accordingly.
// Returns the fd that was removed (for verification).
int iree_async_posix_poll_set_remove_at(iree_async_posix_poll_set_t* poll_set,
                                        iree_host_size_t index);

// Finds the first occurrence of |fd| in the poll set.
// Returns the index, or IREE_HOST_SIZE_MAX if not found.
iree_host_size_t iree_async_posix_poll_set_find(
    const iree_async_posix_poll_set_t* poll_set, int fd);

// Waits for events on the poll set.
// |timeout_ms|: -1 for infinite, 0 for non-blocking, >0 for milliseconds.
// |out_ready_count|: Receives the number of fds with events (may be NULL).
//
// Returns:
//   IREE_STATUS_OK: poll() returned successfully.
//   IREE_STATUS_DEADLINE_EXCEEDED: Timeout with no events.
//   Other status: poll() error.
iree_status_t iree_async_posix_poll_set_wait(
    iree_async_posix_poll_set_t* poll_set, int timeout_ms,
    iree_host_size_t* out_ready_count);

// Returns true if the fd at |index| has events ready, and writes the
// revents to |out_revents|.
static inline bool iree_async_posix_poll_set_is_ready(
    const iree_async_posix_poll_set_t* poll_set, iree_host_size_t index,
    short* out_revents) {
  if (index >= poll_set->count) return false;
  short revents = poll_set->fds[index].revents;
  if (out_revents) *out_revents = revents;
  return revents != 0;
}

// Returns the fd at the given index.
static inline int iree_async_posix_poll_set_fd_at(
    const iree_async_posix_poll_set_t* poll_set, iree_host_size_t index) {
  return poll_set->fds[index].fd;
}

// Clears all fds from the poll set without deallocating storage.
void iree_async_posix_poll_set_clear(iree_async_posix_poll_set_t* poll_set);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_ASYNC_PLATFORM_POSIX_POLL_SET_H_
