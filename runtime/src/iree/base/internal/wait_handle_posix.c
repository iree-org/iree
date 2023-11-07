// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/wait_handle_posix.h"

#if defined(IREE_WAIT_API_POSIX_LIKE)

#include <errno.h>
#include <fcntl.h>
#include <unistd.h>

#if defined(IREE_HAVE_WAIT_TYPE_EVENTFD)
#include <sys/eventfd.h>
#endif  // IREE_HAVE_WAIT_TYPE_EVENTFD
#if defined(IREE_HAVE_WAIT_TYPE_SYNC_FILE)
#include <android/sync.h>
#endif  // IREE_HAVE_WAIT_TYPE_SYNC_FILE

//===----------------------------------------------------------------------===//
// iree_wait_primitive_* raw calls
//===----------------------------------------------------------------------===//

#if defined(IREE_HAVE_WAIT_TYPE_EVENTFD)
static iree_status_t iree_wait_primitive_create_eventfd(
    bool initial_state, iree_wait_handle_t* out_handle) {
  memset(out_handle, 0, sizeof(*out_handle));
  out_handle->type = IREE_WAIT_PRIMITIVE_TYPE_EVENT_FD;

  // https://man7.org/linux/man-pages/man2/eventfd.2.html
  out_handle->value.event.fd =
      eventfd(initial_state ? 1 : 0, EFD_CLOEXEC | EFD_NONBLOCK);
  if (IREE_UNLIKELY(out_handle->value.event.fd == -1)) {
    return iree_make_status(iree_status_code_from_errno(errno),
                            "failed to create eventfd (%d)", errno);
  }

  return iree_ok_status();
}
#endif  // IREE_HAVE_WAIT_TYPE_EVENTFD

#if defined(IREE_HAVE_WAIT_TYPE_PIPE)
IREE_ATTRIBUTE_UNUSED static iree_status_t iree_wait_primitive_create_pipe(
    bool initial_state, iree_wait_handle_t* out_handle) {
  memset(out_handle, 0, sizeof(*out_handle));
  out_handle->type = IREE_WAIT_PRIMITIVE_TYPE_PIPE;

  // Create read (fds[0]) and write (fds[1]) handles.
  // https://man7.org/linux/man-pages/man2/pipe.2.html
  if (IREE_UNLIKELY(pipe(out_handle->value.pipe.fds) < 0)) {
    return iree_make_status(iree_status_code_from_errno(errno),
                            "failed to create pipe (%d)", errno);
  }

  // Set both fds to non-blocking.
  // NOTE: we could use pipe2 when available on linux to avoid the need for the
  // fcntl, but BSD/darwin/etc don't have it so we'd still need a fallback. This
  // is effectively the same as passing O_NONBLOCK to pipe2.
  for (int i = 0; i < 2; ++i) {
    if (IREE_UNLIKELY(
            fcntl(out_handle->value.pipe.fds[i], F_SETFL, O_NONBLOCK) < 0)) {
      return iree_make_status(iree_status_code_from_errno(errno),
                              "failed to set pipe fd %d to non-blocking (%d)",
                              i, errno);
    }
  }

  // Initially triggered means we just write once to the pipe.
  // This write must not fail as if the caller requested the state they would
  // likely deadlock if the first read would block.
  if (initial_state) {
    iree_status_t status = iree_wait_primitive_write(out_handle);
    if (!iree_status_is_ok(status)) {
      iree_wait_handle_close(out_handle);
      return status;
    }
  }

  return iree_ok_status();
}
#endif  // IREE_HAVE_WAIT_TYPE_PIPE

iree_status_t iree_wait_primitive_create_native(
    bool initial_state, iree_wait_handle_t* out_handle) {
  memset(out_handle, 0, sizeof(*out_handle));
#if defined(IREE_HAVE_WAIT_TYPE_EVENTFD)
  // Always prefer eventfd when present; they rock.
  return iree_wait_primitive_create_eventfd(initial_state, out_handle);
#elif defined(IREE_HAVE_WAIT_TYPE_PIPE)
  // Pipes are fine but much heavier than eventfds.
  return iree_wait_primitive_create_pipe(initial_state, out_handle);
#else
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "no native wait handle type supported");
#endif  // IREE_HAVE_WAIT_TYPE_*
}

static void iree_wait_handle_close_fd(int fd) {
  int rv;
  IREE_SYSCALL(rv, close(fd));
  // NOTE: we could fail to close if the handle is invalid/already closed/etc.
  // As Windows has undefined behavior when handles are closed while there are
  // active waits we don't use fd closes as load-bearing operations and it's
  // fine to ignore the error.
}

void iree_wait_handle_close(iree_wait_handle_t* handle) {
  switch (handle->type) {
#if defined(IREE_HAVE_WAIT_TYPE_EVENTFD)
    case IREE_WAIT_PRIMITIVE_TYPE_EVENT_FD: {
      iree_wait_handle_close_fd(handle->value.event.fd);
      break;
    }
#endif  // IREE_HAVE_WAIT_TYPE_EVENTFD
#if defined(IREE_HAVE_WAIT_TYPE_SYNC_FILE)
    case IREE_WAIT_PRIMITIVE_TYPE_SYNC_FILE:
      iree_wait_handle_close_fd(handle->value.sync_file.fd);
      break;
#endif  // IREE_HAVE_WAIT_TYPE_SYNC_FILE
#if defined(IREE_HAVE_WAIT_TYPE_PIPE)
    case IREE_WAIT_PRIMITIVE_TYPE_PIPE: {
      iree_wait_handle_close_fd(handle->value.pipe.read_fd);
      iree_wait_handle_close_fd(handle->value.pipe.write_fd);
      break;
    }
#endif  // IREE_HAVE_WAIT_TYPE_PIPE
    default:
      break;
  }
  iree_wait_handle_deinitialize(handle);
}

bool iree_wait_primitive_compare_identical(const iree_wait_handle_t* lhs,
                                           const iree_wait_handle_t* rhs) {
  return lhs->type == rhs->type &&
         memcmp(&lhs->value, &rhs->value, sizeof(lhs->value)) == 0;
}

int iree_wait_primitive_get_read_fd(const iree_wait_handle_t* handle) {
  switch (handle->type) {
#if defined(IREE_HAVE_WAIT_TYPE_EVENTFD)
    case IREE_WAIT_PRIMITIVE_TYPE_EVENT_FD:
      return handle->value.event.fd;
#endif  // IREE_HAVE_WAIT_TYPE_EVENTFD
#if defined(IREE_HAVE_WAIT_TYPE_SYNC_FILE)
    case IREE_WAIT_PRIMITIVE_TYPE_SYNC_FILE:
      return handle->value.sync_file.fd;
#endif  // IREE_HAVE_WAIT_TYPE_SYNC_FILE
#if defined(IREE_HAVE_WAIT_TYPE_PIPE)
    case IREE_WAIT_PRIMITIVE_TYPE_PIPE:
      return handle->value.pipe.read_fd;
#endif  // IREE_HAVE_WAIT_TYPE_PIPE
    default:
      return -1;
  }
}

iree_status_t iree_wait_primitive_read(iree_wait_handle_t* handle,
                                       iree_time_t deadline_ns) {
  // Until we need it this does not support anything but polling.
  // If we want to support auto reset events we'd want to implement blocking.
  if (deadline_ns != IREE_TIME_INFINITE_PAST) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "reads are just polls today");
  }

  int rv = -1;
  switch (handle->type) {
    case IREE_WAIT_PRIMITIVE_TYPE_NONE:
      return iree_ok_status();  // no-op
#if defined(IREE_HAVE_WAIT_TYPE_EVENTFD)
    case IREE_WAIT_PRIMITIVE_TYPE_EVENT_FD: {
      eventfd_t val = 0;
      IREE_SYSCALL(rv, eventfd_read(handle->value.event.fd, &val));
      break;
    }
#endif  // IREE_HAVE_WAIT_TYPE_EVENTFD
#if defined(IREE_HAVE_WAIT_TYPE_SYNC_FILE)
    case IREE_WAIT_PRIMITIVE_TYPE_SYNC_FILE:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "sync files not yet implemented");
#endif  // IREE_HAVE_WAIT_TYPE_SYNC_FILE
#if defined(IREE_HAVE_WAIT_TYPE_PIPE)
    case IREE_WAIT_PRIMITIVE_TYPE_PIPE: {
      char buf;
      IREE_SYSCALL(rv, read(handle->value.pipe.read_fd, &buf, 1));
      break;
    }
#endif  // IREE_HAVE_WAIT_TYPE_PIPE
    default:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "unhandled wait type %d", (int)handle->type);
  }
  if (rv >= 0) {
    // Read completed successfully.
    return iree_ok_status();
  } else if (errno == EWOULDBLOCK) {
    // Would have blocked meaning that there's no data waiting.
    // NOTE: we purposefully avoid a full status result here as this is a
    // non-exceptional result.
    return iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED);
  } else {
    return iree_make_status(iree_status_code_from_errno(errno),
                            "fd read failure %d", errno);
  }
}

iree_status_t iree_wait_primitive_write(iree_wait_handle_t* handle) {
  int rv = -1;
  switch (handle->type) {
    case IREE_WAIT_PRIMITIVE_TYPE_NONE:
      return iree_ok_status();  // no-op
#if defined(IREE_HAVE_WAIT_TYPE_EVENTFD)
    case IREE_WAIT_PRIMITIVE_TYPE_EVENT_FD: {
      IREE_SYSCALL(rv, eventfd_write(handle->value.event.fd, 1ull));
      break;
    }
#endif  // IREE_HAVE_WAIT_TYPE_EVENTFD
#if defined(IREE_HAVE_WAIT_TYPE_SYNC_FILE)
    case IREE_WAIT_PRIMITIVE_TYPE_SYNC_FILE:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "sync files not yet implemented");
#endif  // IREE_HAVE_WAIT_TYPE_SYNC_FILE
#if defined(IREE_HAVE_WAIT_TYPE_PIPE)
    case IREE_WAIT_PRIMITIVE_TYPE_PIPE: {
      char buf = '\n';
      IREE_SYSCALL(rv, write(handle->value.pipe.write_fd, &buf, 1));
      break;
    }
#endif  // IREE_HAVE_WAIT_TYPE_PIPE
    default:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "unhandled wait type");
  }
  if (rv >= 0) {
    // Write completed successfully.
    return iree_ok_status();
  } else {
    return iree_make_status(iree_status_code_from_errno(errno),
                            "fd write failure %d", errno);
  }
}

iree_status_t iree_wait_primitive_clear(iree_wait_handle_t* handle) {
  // No-op for null handles.
  if (handle->type == IREE_WAIT_PRIMITIVE_TYPE_NONE) return iree_ok_status();

  // Read in a loop until the read would block.
  // Depending on how the user setup the fd the act of reading may reset the
  // entire handle (such as with the default eventfd mode) or multiple reads may
  // be required (such as with semaphores).
  while (true) {
    iree_status_t status =
        iree_wait_primitive_read(handle, IREE_TIME_INFINITE_PAST);
    if (iree_status_is_deadline_exceeded(status)) {
      // Would have blocked reading which means we've cleared the fd.
      return iree_ok_status();
    } else if (!iree_status_is_ok(status)) {
      return status;
    }
  }
}

//===----------------------------------------------------------------------===//
// iree_event_t
//===----------------------------------------------------------------------===//

iree_status_t iree_event_initialize(bool initial_state,
                                    iree_event_t* out_event) {
  return iree_wait_primitive_create_native(initial_state, out_event);
}

void iree_event_deinitialize(iree_event_t* event) {
  iree_wait_handle_close(event);
}

void iree_event_set(iree_event_t* event) {
  IREE_IGNORE_ERROR(iree_wait_primitive_write(event));
}

void iree_event_reset(iree_event_t* event) {
  IREE_IGNORE_ERROR(iree_wait_primitive_clear(event));
}

#endif  // IREE_WAIT_API_POSIX_LIKE
