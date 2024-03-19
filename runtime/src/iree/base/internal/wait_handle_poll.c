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

//===----------------------------------------------------------------------===//
// iree_wait_set_t
//===----------------------------------------------------------------------===//

struct iree_wait_set_t {
  iree_allocator_t allocator;

  // Total capacity of each handle list.
  iree_host_size_t handle_capacity;

  // Total number of valid user_handles/poll_fds.
  iree_host_size_t handle_count;

  // User-provided handles.
  // We only really need to track these so that we can preserve the handle
  // types; we could either just do that (a few bytes) or keep them here as-is
  // where they are a bit easier to debug.
  iree_wait_handle_t* user_handles;

  // Native list of fds+req we can pass to poll/ppoll/etc and that will receive
  // the output information like which events were triggered during the wait.
  //
  // pollfd::events is specified when the fds are added to the set and then each
  // wait pollfd::revents is modified during the poll syscall.
  struct pollfd* poll_fds;
};

iree_status_t iree_wait_set_allocate(iree_host_size_t capacity,
                                     iree_allocator_t allocator,
                                     iree_wait_set_t** out_set) {
  IREE_ASSERT_ARGUMENT(out_set);

  // Be reasonable; 64K objects is too high (even if poll supports it, which is
  // hard to tell if it does).
  if (capacity >= UINT16_MAX) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "wait set capacity of %" PRIhsz " is unreasonably large", capacity);
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  iree_host_size_t user_handle_list_size =
      capacity * iree_sizeof_struct(iree_wait_handle_t);
  iree_host_size_t poll_fd_list_size = capacity * sizeof(struct pollfd);
  iree_host_size_t total_size = iree_sizeof_struct(iree_wait_set_t) +
                                user_handle_list_size + poll_fd_list_size;

  iree_wait_set_t* set = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(allocator, total_size, (void**)&set));
  set->allocator = allocator;
  set->handle_capacity = capacity;
  iree_wait_set_clear(set);

  set->user_handles =
      (iree_wait_handle_t*)((uint8_t*)set +
                            iree_sizeof_struct(iree_wait_set_t));
  set->poll_fds =
      (struct pollfd*)((uint8_t*)set->user_handles + user_handle_list_size);

  *out_set = set;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

void iree_wait_set_free(iree_wait_set_t* set) {
  if (!set) return;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_allocator_free(set->allocator, set);
  IREE_TRACE_ZONE_END(z0);
}

bool iree_wait_set_is_empty(const iree_wait_set_t* set) {
  return set->handle_count != 0;
}

iree_status_t iree_wait_set_insert(iree_wait_set_t* set,
                                   iree_wait_handle_t handle) {
  if (set->handle_count + 1 > set->handle_capacity) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "wait set capacity reached");
  }

  iree_host_size_t index = set->handle_count++;

  iree_wait_handle_t* user_handle = &set->user_handles[index];
  iree_wait_handle_wrap_primitive(handle.type, handle.value, user_handle);

  // NOTE: poll will ignore any negative fds.
  struct pollfd* poll_fd = &set->poll_fds[index];
  poll_fd->fd = iree_wait_primitive_get_read_fd(&handle);
  poll_fd->events = POLLIN | POLLPRI;  // implicit POLLERR | POLLHUP | POLLNVAL
  poll_fd->revents = 0;

  return iree_ok_status();
}

void iree_wait_set_erase(iree_wait_set_t* set, iree_wait_handle_t handle) {
  // Find the user handle in the set. This either requires a linear scan to
  // find the matching user handle or - if valid - we can use the native index
  // set after an iree_wait_any wake to do a quick lookup.
  iree_host_size_t index = handle.set_internal.index;
  if (IREE_UNLIKELY(index >= set->handle_count) ||
      IREE_UNLIKELY(!iree_wait_primitive_compare_identical(
          &set->user_handles[index], &handle))) {
    // Fallback to a linear scan of (hopefully) a small list.
    for (iree_host_size_t i = 0; i < set->handle_count; ++i) {
      if (iree_wait_primitive_compare_identical(&set->user_handles[i],
                                                &handle)) {
        index = i;
        break;
      }
    }
  }

  // Remove from both handle lists.
  // Since we make no guarantees about the order of the lists we can just swap
  // with the last value.
  int tail_index = (int)set->handle_count - 1;
  if (tail_index > index) {
    memcpy(&set->poll_fds[index], &set->poll_fds[tail_index],
           sizeof(*set->poll_fds));
    memcpy(&set->user_handles[index], &set->user_handles[tail_index],
           sizeof(*set->user_handles));
  }
  --set->handle_count;
}

void iree_wait_set_clear(iree_wait_set_t* set) { set->handle_count = 0; }

// Maps a poll revent bitfield result to a status (on failure) and an indicator
// of whether the event was signaled.
static iree_status_t iree_wait_set_resolve_poll_events(short revents,
                                                       bool* out_signaled) {
  if (revents & POLLERR) {
    return iree_make_status(IREE_STATUS_INTERNAL, "POLLERR on fd");
  } else if (revents & POLLHUP) {
    return iree_make_status(IREE_STATUS_CANCELLED, "POLLHUP on fd");
  } else if (revents & POLLNVAL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "POLLNVAL on fd");
  }
  *out_signaled = (revents & POLLIN) != 0;
  return iree_ok_status();
}

iree_status_t iree_wait_all(iree_wait_set_t* set, iree_time_t deadline_ns) {
  // Make the syscall only when we have at least one valid fd.
  // Don't use this as a sleep.
  if (set->handle_count <= 0) {
    return iree_ok_status();
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  // TODO(benvanik): see if we can use tracy's mutex tracking to make waits
  // nicer (at least showing signal->wait relations).

  // Certain poll implementations have a nasty behavior where they allow
  // negative fds to ignore entries... except for at [0]. To avoid any
  // additional tracking here we manage a local pollfd list that we keep offset
  // to the first non-negative fd.
  //
  // Gotcha is buried in here (and various spooky bug reports on the web):
  // https://manpages.debian.org/buster/manpages-dev/poll.2.en.html
  //   This provides an easy way of ignoring a file descriptor for a single
  //   poll() call: simply negate the fd field. Note, however, that this
  //   technique can't be used to ignore file descriptor 0.
  //
  // Thanks guys 🙄
  struct pollfd* poll_fd_base = set->poll_fds;
  nfds_t poll_fd_count = set->handle_count;

  // Wait-all requires that we repeatedly poll until all handles have been
  // signaled. To reduce overhead (and not miss events) we mark any handle we
  // have successfully polled as invalid (fd<0) so that the kernel ignores it.
  // Only when all handles are invalid does it mean that we've actually waited
  // for all of them.
  iree_status_t status = iree_ok_status();
  int unsignaled_count = poll_fd_count;
  do {
    // Eat any negative handles at the start to avoid the mentioned fd[0] bug.
    while (poll_fd_base[0].fd < 0) {
      ++poll_fd_base;
      --poll_fd_count;
    }

    int signaled_count = 0;
    status = iree_syscall_poll(poll_fd_base, poll_fd_count, deadline_ns,
                               &signaled_count);
    if (!iree_status_is_ok(status)) {
      // Failed during the poll itself. Ensure that we fall-through and refresh
      // the poll_fds handle list.
      break;
    }
    unsignaled_count -= signaled_count;

    // Neuter any that have successfully resolved.
    for (nfds_t i = 0; i < poll_fd_count; ++i) {
      if (poll_fd_base[i].fd < 0) continue;
      bool signaled = false;
      status =
          iree_wait_set_resolve_poll_events(poll_fd_base[i].revents, &signaled);
      if (!iree_status_is_ok(status)) {
        // One (or more) fds had an issue. Ensure that we fall-through and
        // refresh the poll_fds handle list.
        break;
      }
      if (signaled) {
        // Negate fd so that we ignore it in the next poll.
        poll_fd_base[i].fd = -poll_fd_base[i].fd;
      }
    }
  } while (unsignaled_count > 0);

  // Since we destroyed the list of handles during the operation we need to
  // refresh them with their fds so that the next wait can happen. This is the
  // kind of thing kqueue/epoll solves (mutable in-place updates on polls) and
  // an unfortunate reality of using an ancient API. Thankfully most waits are
  // wait-any so a little loop isn't the worst thing in the wait-all case.
  for (nfds_t i = 0; i < set->handle_count; ++i) {
    set->poll_fds[i].fd = -set->poll_fds[i].fd;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_wait_any(iree_wait_set_t* set, iree_time_t deadline_ns,
                            iree_wait_handle_t* out_wake_handle) {
  // Make the syscall only when we have at least one valid fd.
  // Don't use this as a sleep.
  if (set->handle_count <= 0) {
    if (out_wake_handle) {
      memset(out_wake_handle, 0, sizeof(*out_wake_handle));
    }
    return iree_ok_status();
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  // TODO(benvanik): see if we can use tracy's mutex tracking to make waits
  // nicer (at least showing signal->wait relations).

  // Wait-any lets us just poll all the handles we have without needing to worry
  // about whether all of them were signaled.
  int signaled_count = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_syscall_poll(set->poll_fds, set->handle_count, deadline_ns,
                            &signaled_count));

  // Find at least one signaled handle.
  if (out_wake_handle) {
    memset(out_wake_handle, 0, sizeof(*out_wake_handle));
    if (signaled_count > 0) {
      for (iree_host_size_t i = 0; i < set->handle_count; ++i) {
        bool signaled = false;
        IREE_RETURN_AND_END_ZONE_IF_ERROR(
            z0, iree_wait_set_resolve_poll_events(set->poll_fds[i].revents,
                                                  &signaled));
        if (signaled) {
          memcpy(out_wake_handle, &set->user_handles[i],
                 sizeof(*out_wake_handle));
          out_wake_handle->set_internal.index = i;
          break;
        }
      }
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_wait_one(iree_wait_handle_t* handle,
                            iree_time_t deadline_ns) {
  struct pollfd poll_fds;
  poll_fds.fd = iree_wait_primitive_get_read_fd(handle);
  if (poll_fds.fd == -1) return false;
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
