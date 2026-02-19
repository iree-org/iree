// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Event set implementation using BSD/macOS kqueue.
//
// kqueue provides O(1) event notification similar to Linux epoll. It uses
// a filter-based model where EVFILT_READ and EVFILT_WRITE correspond to
// poll()'s POLLIN and POLLOUT.
//
// Implementation notes:
// - kqueue is level-triggered by default, matching poll() semantics.
// - kevent() combines registration and waiting in a single syscall.
// - Closing an fd automatically removes it from the kqueue.
// - kqueue returns separate events for READ and WRITE on the same fd.
//   Unlike poll(), which returns a combined bitmask, the proactor may see the
//   same fd multiple times per poll iteration. The proactor's dispatch loop
//   handles this correctly by filtering operations against per-event revents.
// - Filter add/modify/remove operations are batched into a single kevent()
//   call to ensure atomicity: either all filters for an fd are registered or
//   none are (no orphaned filters on partial failure).
//
// References:
// - https://man.openbsd.org/kqueue.2
// - https://github.com/libuv/libuv/blob/v1.x/src/unix/kqueue.c
// - https://github.com/libevent/libevent/blob/master/kqueue.c

#include "iree/async/platform/posix/event_set.h"

#if defined(IREE_PLATFORM_APPLE) || defined(IREE_PLATFORM_BSD)

#include <errno.h>
#include <fcntl.h>
#include <string.h>
#include <sys/event.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

// Fixed capacity for the kevent results buffer.
#define IREE_EVENT_SET_KQUEUE_EVENTS_CAPACITY 64

typedef struct iree_async_posix_event_set_kqueue_t {
  iree_async_posix_event_set_t base;
  iree_allocator_t allocator;
  int kqueue_fd;

  // Ready event iteration state.
  iree_host_size_t ready_count;
  iree_host_size_t iteration_index;

  // Fixed buffer for kevent() results.
  struct kevent events[IREE_EVENT_SET_KQUEUE_EVENTS_CAPACITY];
} iree_async_posix_event_set_kqueue_t;

static iree_async_posix_event_set_kqueue_t*
iree_async_posix_event_set_kqueue_cast(
    iree_async_posix_event_set_t* event_set) {
  return (iree_async_posix_event_set_kqueue_t*)event_set;
}

//===----------------------------------------------------------------------===//
// Event translation
//===----------------------------------------------------------------------===//

static short iree_kevent_to_poll_events(const struct kevent* event) {
  short poll_events = 0;
  switch (event->filter) {
    case EVFILT_READ:
      poll_events |= POLLIN;
      break;
    case EVFILT_WRITE:
      poll_events |= POLLOUT;
      break;
  }
  if (event->flags & EV_EOF) poll_events |= POLLHUP;
  if (event->flags & EV_ERROR) poll_events |= POLLERR;
  return poll_events;
}

// Submits a batch of changelist entries to the kqueue atomically.
// On failure, individual entries report EV_ERROR with the errno in |data|.
// For DELETE operations, ENOENT (not registered) and EBADF (fd closed) are
// silently ignored since the filter is already gone.
static iree_status_t iree_kqueue_submit_changes(int kqueue_fd, int fd,
                                                struct kevent* changelist,
                                                int change_count) {
  // kevent() with a zero timeout applies all changelist entries and returns
  // immediately. The eventlist receives EV_ERROR entries for any changelist
  // items that failed, plus any already-ready events. We size the eventlist
  // to the changelist length — we only care about error events.
  struct kevent eventlist[4];  // max 4 changes (2 deletes + 2 adds in modify)
  IREE_ASSERT(change_count <= 4);

  struct timespec zero_timeout = {0, 0};
  int result = kevent(kqueue_fd, changelist, change_count, eventlist,
                      change_count, &zero_timeout);
  if (result < 0) {
    int error = errno;
    return iree_make_status(iree_status_code_from_errno(error),
                            "kevent() changelist submission failed for fd %d: "
                            "%s",
                            fd, strerror(error));
  }

  // The returned eventlist is a mix of EV_ERROR entries (for failed changelist
  // items) and normal ready events (for already-monitored fds). We only care
  // about EV_ERROR entries. Each EV_ERROR event carries the filter and flags
  // from the original changelist entry that failed.
  for (int i = 0; i < result; ++i) {
    if (!(eventlist[i].flags & EV_ERROR)) continue;
    int error = (int)eventlist[i].data;
    // For DELETE operations, ENOENT (filter not registered) and EBADF (fd
    // already closed) are expected and harmless — the filter is already gone.
    // We identify delete operations by matching the error event back to the
    // changelist: the event's ident and filter correspond to the failed entry.
    bool is_delete = false;
    for (int j = 0; j < change_count; ++j) {
      if (changelist[j].ident == eventlist[i].ident &&
          changelist[j].filter == eventlist[i].filter &&
          (changelist[j].flags & EV_DELETE)) {
        is_delete = true;
        break;
      }
    }
    if (is_delete && (error == ENOENT || error == EBADF)) {
      continue;
    }
    const char* action_name = is_delete ? "DELETE" : "ADD";
    return iree_make_status(iree_status_code_from_errno(error),
                            "kevent(%s) failed for fd %d filter %d: %s",
                            action_name, fd, (int)eventlist[i].filter,
                            strerror(error));
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Vtable implementations
//===----------------------------------------------------------------------===//

static void iree_async_posix_event_set_kqueue_free(
    iree_async_posix_event_set_t* base_event_set) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_async_posix_event_set_kqueue_t* event_set =
      iree_async_posix_event_set_kqueue_cast(base_event_set);
  iree_allocator_t allocator = event_set->allocator;
  if (event_set->kqueue_fd >= 0) {
    close(event_set->kqueue_fd);
  }
  iree_allocator_free(allocator, event_set);
  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_async_posix_event_set_kqueue_add(
    iree_async_posix_event_set_t* base_event_set, int fd, short events) {
  iree_async_posix_event_set_kqueue_t* event_set =
      iree_async_posix_event_set_kqueue_cast(base_event_set);

  // Batch all filter additions into a single kevent() call so that either
  // both READ and WRITE are registered or neither is (no orphaned filters).
  struct kevent changelist[2];
  int change_count = 0;
  if (events & POLLIN) {
    EV_SET(&changelist[change_count++], fd, EVFILT_READ, EV_ADD, 0, 0, NULL);
  }
  if (events & POLLOUT) {
    EV_SET(&changelist[change_count++], fd, EVFILT_WRITE, EV_ADD, 0, 0, NULL);
  }
  if (change_count == 0) return iree_ok_status();
  return iree_kqueue_submit_changes(event_set->kqueue_fd, fd, changelist,
                                    change_count);
}

static iree_status_t iree_async_posix_event_set_kqueue_modify(
    iree_async_posix_event_set_t* base_event_set, int fd, short events) {
  iree_async_posix_event_set_kqueue_t* event_set =
      iree_async_posix_event_set_kqueue_cast(base_event_set);

  // Batch all filter changes into a single kevent() call.
  // EV_ADD updates existing filters, EV_DELETE removes unwanted ones.
  // Deletions are listed first so that any stale filters are cleaned up
  // before new ones are applied within the same atomic submission.
  struct kevent changelist[4];
  int change_count = 0;

  // Delete filters that are NOT in the new mask.
  if (!(events & POLLIN)) {
    EV_SET(&changelist[change_count++], fd, EVFILT_READ, EV_DELETE, 0, 0, NULL);
  }
  if (!(events & POLLOUT)) {
    EV_SET(&changelist[change_count++], fd, EVFILT_WRITE, EV_DELETE, 0, 0,
           NULL);
  }
  // Add/update filters that ARE in the new mask.
  if (events & POLLIN) {
    EV_SET(&changelist[change_count++], fd, EVFILT_READ, EV_ADD, 0, 0, NULL);
  }
  if (events & POLLOUT) {
    EV_SET(&changelist[change_count++], fd, EVFILT_WRITE, EV_ADD, 0, 0, NULL);
  }
  if (change_count == 0) return iree_ok_status();
  return iree_kqueue_submit_changes(event_set->kqueue_fd, fd, changelist,
                                    change_count);
}

static iree_status_t iree_async_posix_event_set_kqueue_remove(
    iree_async_posix_event_set_t* base_event_set, int fd) {
  iree_async_posix_event_set_kqueue_t* event_set =
      iree_async_posix_event_set_kqueue_cast(base_event_set);

  // Delete both filters in a single batched kevent() call.
  // ENOENT/EBADF on individual filters is silently ignored (the filter
  // was already gone or the fd was already closed).
  struct kevent changelist[2];
  EV_SET(&changelist[0], fd, EVFILT_READ, EV_DELETE, 0, 0, NULL);
  EV_SET(&changelist[1], fd, EVFILT_WRITE, EV_DELETE, 0, 0, NULL);
  return iree_kqueue_submit_changes(event_set->kqueue_fd, fd, changelist, 2);
}

static iree_status_t iree_async_posix_event_set_kqueue_wait(
    iree_async_posix_event_set_t* base_event_set, int timeout_ms,
    iree_host_size_t* out_ready_count, bool* out_timed_out) {
  iree_async_posix_event_set_kqueue_t* event_set =
      iree_async_posix_event_set_kqueue_cast(base_event_set);
  if (out_ready_count) *out_ready_count = 0;
  *out_timed_out = false;

  event_set->ready_count = 0;
  event_set->iteration_index = 0;

  struct timespec timeout_spec;
  struct timespec* timeout_ptr = NULL;
  if (timeout_ms >= 0) {
    timeout_spec.tv_sec = timeout_ms / 1000;
    timeout_spec.tv_nsec = (timeout_ms % 1000) * 1000000;
    timeout_ptr = &timeout_spec;
  }

  int result = kevent(event_set->kqueue_fd, NULL, 0, event_set->events,
                      IREE_EVENT_SET_KQUEUE_EVENTS_CAPACITY, timeout_ptr);
  if (result < 0) {
    int error = errno;
    if (error == EINTR) {
      // Interrupted by signal - not a timeout, no events ready.
      return iree_ok_status();
    }
    return iree_make_status(iree_status_code_from_errno(error),
                            "kevent() failed: %s", strerror(error));
  }

  if (result == 0) {
    *out_timed_out = true;
    return iree_ok_status();
  }

  event_set->ready_count = (iree_host_size_t)result;
  if (out_ready_count) *out_ready_count = event_set->ready_count;
  return iree_ok_status();
}

static void iree_async_posix_event_set_kqueue_reset_ready_iteration(
    iree_async_posix_event_set_t* base_event_set) {
  iree_async_posix_event_set_kqueue_t* event_set =
      iree_async_posix_event_set_kqueue_cast(base_event_set);
  event_set->iteration_index = 0;
}

static bool iree_async_posix_event_set_kqueue_next_ready(
    iree_async_posix_event_set_t* base_event_set, int* out_fd,
    short* out_revents) {
  iree_async_posix_event_set_kqueue_t* event_set =
      iree_async_posix_event_set_kqueue_cast(base_event_set);

  if (event_set->iteration_index >= event_set->ready_count) {
    return false;
  }

  // Return each kevent individually. kqueue returns separate events for
  // EVFILT_READ and EVFILT_WRITE, so the same fd may appear multiple times.
  // The proactor's dispatch loop handles this correctly: each handler filters
  // operations against the per-event revents bitmask.
  iree_host_size_t index = event_set->iteration_index++;
  const struct kevent* event = &event_set->events[index];
  *out_fd = (int)event->ident;
  *out_revents = iree_kevent_to_poll_events(event);
  return true;
}

//===----------------------------------------------------------------------===//
// Vtable
//===----------------------------------------------------------------------===//

static const iree_async_posix_event_set_vtable_t
    iree_async_posix_event_set_kqueue_vtable = {
        .free = iree_async_posix_event_set_kqueue_free,
        .add = iree_async_posix_event_set_kqueue_add,
        .modify = iree_async_posix_event_set_kqueue_modify,
        .remove = iree_async_posix_event_set_kqueue_remove,
        .wait = iree_async_posix_event_set_kqueue_wait,
        .reset_ready_iteration =
            iree_async_posix_event_set_kqueue_reset_ready_iteration,
        .next_ready = iree_async_posix_event_set_kqueue_next_ready,
};

iree_status_t iree_async_posix_event_set_allocate_kqueue(
    iree_allocator_t allocator, iree_async_posix_event_set_t** out_event_set) {
  IREE_ASSERT_ARGUMENT(out_event_set);
  *out_event_set = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_async_posix_event_set_kqueue_t* event_set = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(allocator, sizeof(*event_set), (void**)&event_set));
  memset(event_set, 0, sizeof(*event_set));
  event_set->base.vtable = &iree_async_posix_event_set_kqueue_vtable;
  event_set->allocator = allocator;
  event_set->kqueue_fd = -1;

  // Create the kqueue fd with close-on-exec where possible.
#if defined(IREE_PLATFORM_FREEBSD) || defined(IREE_PLATFORM_NETBSD)
  // FreeBSD 12+ and NetBSD 10+ provide kqueue1() for atomic CLOEXEC.
  event_set->kqueue_fd = kqueue1(O_CLOEXEC);
#else
  // macOS and OpenBSD lack kqueue1(). We use kqueue() + fcntl() which has
  // a theoretical TOCTOU race if another thread forks between the two calls.
  // This is acceptable: IREE processes do not fork, and the fcntl is a
  // defense-in-depth measure for fd hygiene across exec.
  event_set->kqueue_fd = kqueue();
#endif  // IREE_PLATFORM_FREEBSD || IREE_PLATFORM_NETBSD
  if (event_set->kqueue_fd < 0) {
    int error = errno;
    iree_allocator_free(allocator, event_set);
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(iree_status_code_from_errno(error),
                            "kqueue() failed: %s", strerror(error));
  }
#if !defined(IREE_PLATFORM_FREEBSD) && !defined(IREE_PLATFORM_NETBSD)
  fcntl(event_set->kqueue_fd, F_SETFD, FD_CLOEXEC);
#endif  // !IREE_PLATFORM_FREEBSD && !IREE_PLATFORM_NETBSD

  *out_event_set = &event_set->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

#endif  // IREE_PLATFORM_APPLE || IREE_PLATFORM_BSD
