// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/platform/posix/wake.h"

#include <errno.h>
#include <fcntl.h>
#include <string.h>
#include <unistd.h>

#if defined(IREE_PLATFORM_LINUX)
#include <sys/eventfd.h>
#endif  // IREE_PLATFORM_LINUX

iree_status_t iree_async_posix_wake_initialize(
    iree_async_posix_wake_t* out_wake) {
  IREE_TRACE_ZONE_BEGIN(z0);

  out_wake->read_fd = -1;
  out_wake->write_fd = -1;

#if defined(IREE_PLATFORM_LINUX)
  // Try eventfd first (more efficient - single fd, no pipe buffer).
  int efd = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
  if (efd >= 0) {
    out_wake->read_fd = efd;
    out_wake->write_fd = efd;
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }
  // Fall through to pipe on failure.
#endif  // IREE_PLATFORM_LINUX

  // Portable fallback: pipe.
  int pipe_fds[2];
  if (pipe(pipe_fds) < 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(iree_status_code_from_errno(errno),
                            "pipe() failed: %s", strerror(errno));
  }

  // Set non-blocking and close-on-exec. Failure to set O_NONBLOCK would cause
  // the wake drain loop to block the poll thread.
  for (int i = 0; i < 2; ++i) {
    int flags = fcntl(pipe_fds[i], F_GETFL);
    if (flags < 0 || fcntl(pipe_fds[i], F_SETFL, flags | O_NONBLOCK) < 0 ||
        fcntl(pipe_fds[i], F_SETFD, FD_CLOEXEC) < 0) {
      int saved_errno = errno;
      close(pipe_fds[0]);
      close(pipe_fds[1]);
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(iree_status_code_from_errno(saved_errno),
                              "fcntl() on wake pipe failed: %s",
                              strerror(saved_errno));
    }
  }

  out_wake->read_fd = pipe_fds[0];
  out_wake->write_fd = pipe_fds[1];

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

void iree_async_posix_wake_deinitialize(iree_async_posix_wake_t* wake) {
  IREE_TRACE_ZONE_BEGIN(z0);

  if (wake->read_fd >= 0) {
    close(wake->read_fd);
  }
  if (wake->write_fd >= 0 && wake->write_fd != wake->read_fd) {
    close(wake->write_fd);
  }
  wake->read_fd = -1;
  wake->write_fd = -1;

  IREE_TRACE_ZONE_END(z0);
}

void iree_async_posix_wake_trigger(iree_async_posix_wake_t* wake) {
  // Write a single byte (or eventfd counter increment).
  // This is async-signal-safe and thread-safe.
  uint64_t value = 1;
  ssize_t ret = write(wake->write_fd, &value, sizeof(value));
  (void)ret;  // Ignore errors - EAGAIN means already pending, which is fine.
}

void iree_async_posix_wake_drain(iree_async_posix_wake_t* wake) {
  // Read and discard all pending data.
  uint64_t buffer;
  while (read(wake->read_fd, &buffer, sizeof(buffer)) > 0) {
    // Keep draining.
  }
  // EAGAIN/EWOULDBLOCK means drained, other errors we ignore.
}
