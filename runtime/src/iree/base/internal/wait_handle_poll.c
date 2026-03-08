// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// NOTE: must be first to ensure that we can define settings for all includes.
#include "iree/base/internal/wait_handle_impl.h"

#if IREE_WAIT_API == IREE_WAIT_API_POLL || IREE_WAIT_API == IREE_WAIT_API_PPOLL

#include <errno.h>
#include <poll.h>
#include <time.h>

#include "iree/base/internal/wait_handle_posix.h"

//===----------------------------------------------------------------------===//
// Platform utilities
//===----------------------------------------------------------------------===//

// ppoll is preferred as it has a much better timing mechanism; poll can have a
// large slop on the deadline as not only is it at ms timeout granularity but
// in general tends to round more.
//
// poll/ppoll may spuriously wake with an EINTR. We don't do anything with that
// opportunity (no fancy signal stuff), but we do need to retry the poll and
// ensure that we do so with an updated timeout based on the deadline.
//
// Documentation: https://linux.die.net/man/2/poll

#if IREE_WAIT_API == IREE_WAIT_API_POLL
static iree_status_t iree_syscall_poll(struct pollfd* fds, nfds_t nfds,
                                       iree_time_t deadline_ns,
                                       int* out_signaled_count) {
  *out_signaled_count = 0;
  int rv = -1;
  do {
    uint32_t timeout_ms = iree_absolute_deadline_to_timeout_ms(deadline_ns);
    rv = poll(fds, nfds, (int)timeout_ms);
  } while (rv < 0 && errno == EINTR);
  if (rv > 0) {
    // One or more events set.
    *out_signaled_count = rv;
    return iree_ok_status();
  } else if (IREE_UNLIKELY(rv < 0)) {
    return iree_make_status(iree_status_code_from_errno(errno),
                            "poll failure %d", errno);
  }
  // rv == 0
  // Timeout; no events set.
  return iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED);
}
#elif IREE_WAIT_API == IREE_WAIT_API_PPOLL
static iree_status_t iree_syscall_poll(struct pollfd* fds, nfds_t nfds,
                                       iree_time_t deadline_ns,
                                       int* out_signaled_count) {
  *out_signaled_count = 0;
  int rv = -1;
  do {
    // Convert the deadline into a tmo_p struct for ppoll that controls whether
    // the call is blocking or non-blocking. Note that we must do this every
    // iteration of the loop as a previous ppoll may have taken some of the
    // time.
    //
    // See the ppoll docs for more information as to what the expected value is:
    // http://man7.org/linux/man-pages/man2/poll.2.html
    struct timespec timeout_ts;
    struct timespec* tmo_p = &timeout_ts;
    if (deadline_ns == IREE_TIME_INFINITE_PAST) {
      // Block never.
      memset(&timeout_ts, 0, sizeof(timeout_ts));
    } else if (deadline_ns == IREE_TIME_INFINITE_FUTURE) {
      // Block forever (NULL timeout to ppoll).
      tmo_p = NULL;
    } else {
      // Wait only for as much time as we have before the deadline is exceeded.
      iree_duration_t timeout_ns = deadline_ns - iree_time_now();
      if (timeout_ns < 0) {
        // We've reached the deadline; we'll still perform the poll though as
        // the caller is likely expecting that behavior (intentional context
        // switch/thread yield/etc).
        memset(&timeout_ts, 0, sizeof(timeout_ts));
      } else {
        timeout_ts.tv_sec = (time_t)(timeout_ns / 1000000000ull);
        timeout_ts.tv_nsec = (long)(timeout_ns % 1000000000ull);
      }
    }
    rv = ppoll(fds, nfds, tmo_p, NULL);
  } while (rv < 0 && errno == EINTR);
  if (rv > 0) {
    // One or more events set.
    *out_signaled_count = rv;
    return iree_ok_status();
  } else if (rv < 0) {
    return iree_make_status(iree_status_code_from_errno(errno),
                            "ppoll failure %d", errno);
  }
  // rv == 0
  // Timeout; no events set.
  return iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED);
}
#else
#error "unsupported IREE_WAIT_API value"
#endif  // IREE_WAIT_API

iree_status_t iree_wait_one(iree_wait_handle_t* handle,
                            iree_time_t deadline_ns) {
  struct pollfd poll_fds;
  poll_fds.fd = iree_wait_primitive_get_read_fd(handle);
  if (poll_fds.fd == -1) {
    return iree_ok_status();  // no-op wait
  }
  poll_fds.events = POLLIN;
  poll_fds.revents = 0;

  IREE_TRACE_ZONE_BEGIN(z0);

  // TODO(benvanik): see if we can use tracy's mutex tracking to make waits
  // nicer (at least showing signal->wait relations).

  // Just check for our single handle/event.
  // The benefit of this is that we didn't need to heap alloc the pollfds and
  // the cache should all stay hot. Reusing the same iree_syscall_pool as the
  // multi-wait variants ensures consistent handling (and the same syscall
  // showing in strace/tracy/etc).
  int signaled_count = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_syscall_poll(&poll_fds, 1, deadline_ns, &signaled_count));

  IREE_TRACE_ZONE_END(z0);
  return signaled_count ? iree_ok_status()
                        : iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED);
}

#endif  // IREE_WAIT_API == IREE_WAIT_API_POLL ||
        // IREE_WAIT_API == IREE_WAIT_API_PPOLL
