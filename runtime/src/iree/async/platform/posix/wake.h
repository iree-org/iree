// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Cross-thread wake mechanism using eventfd (Linux) or pipe (portable).
//
// Provides a way for any thread to wake a blocked poll() call. The wake fd
// is added to the poll set; writing to it causes poll() to return.

#ifndef IREE_ASYNC_PLATFORM_POSIX_WAKE_H_
#define IREE_ASYNC_PLATFORM_POSIX_WAKE_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Wake mechanism state. Uses eventfd on Linux, pipe elsewhere.
typedef struct iree_async_posix_wake_t {
  // File descriptor to add to poll set (eventfd or pipe[0]).
  int read_fd;
  // File descriptor to write to (same as read_fd for eventfd, pipe[1] for
  // pipe).
  int write_fd;
} iree_async_posix_wake_t;

// Initializes the wake mechanism.
// On Linux, tries eventfd first; falls back to pipe on failure or other
// platforms.
iree_status_t iree_async_posix_wake_initialize(
    iree_async_posix_wake_t* out_wake);

// Deinitializes the wake mechanism and closes fds.
void iree_async_posix_wake_deinitialize(iree_async_posix_wake_t* wake);

// Returns the fd to add to the poll set (monitors for readability).
static inline int iree_async_posix_wake_fd(
    const iree_async_posix_wake_t* wake) {
  return wake->read_fd;
}

// Triggers a wake (thread-safe, async-signal-safe).
// Writes to the wake fd, causing poll() to return if blocked.
// Multiple concurrent triggers coalesce into a single wake.
void iree_async_posix_wake_trigger(iree_async_posix_wake_t* wake);

// Drains pending wake signals (call after poll() returns).
// Reads and discards data from the wake fd to reset for next poll.
void iree_async_posix_wake_drain(iree_async_posix_wake_t* wake);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_ASYNC_PLATFORM_POSIX_WAKE_H_
