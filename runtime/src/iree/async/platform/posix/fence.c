// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/platform/posix/fence.h"

#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <unistd.h>

#if defined(IREE_PLATFORM_LINUX)
#include <sys/eventfd.h>
#endif  // IREE_PLATFORM_LINUX

#include "iree/async/platform/posix/event_set.h"
#include "iree/async/platform/posix/fd_map.h"
#include "iree/async/platform/posix/proactor.h"
#include "iree/async/platform/posix/wake.h"

//===----------------------------------------------------------------------===//
// Import fence implementation
//===----------------------------------------------------------------------===//

iree_status_t iree_async_proactor_posix_import_fence(
    iree_async_proactor_t* base_proactor, iree_async_primitive_t fence,
    iree_async_semaphore_t* semaphore, uint64_t signal_value) {
  iree_async_proactor_posix_t* proactor =
      iree_async_proactor_posix_cast(base_proactor);

  // Validate: must be a POSIX fd with a valid descriptor.
  // On validation error the caller retains fd ownership.
  if (fence.type != IREE_ASYNC_PRIMITIVE_TYPE_FD) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "import_fence requires an FD primitive (got type %d)", (int)fence.type);
  }
  if (fence.value.fd < 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "import_fence fd must be >= 0 (got %d)",
                            fence.value.fd);
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  // Allocate a tracker to associate the fd with the semaphore.
  iree_async_posix_fence_import_tracker_t* tracker = NULL;
  iree_status_t alloc_status = iree_allocator_malloc(
      base_proactor->allocator, sizeof(*tracker), (void**)&tracker);
  if (!iree_status_is_ok(alloc_status)) {
    IREE_TRACE_ZONE_END(z0);
    return alloc_status;
  }

  // Retain the semaphore so it survives until the completion handler fires.
  iree_async_semaphore_retain(semaphore);
  tracker->semaphore = semaphore;
  tracker->signal_value = signal_value;
  tracker->fence_fd = fence.value.fd;
  tracker->allocator = base_proactor->allocator;

  // Defer fd_map/event_set registration to the poll thread. The tracker is
  // pushed to the pending_fence_imports MPSC queue and the poll thread drains
  // it at the top of each poll() iteration, performing the actual fd_map
  // insert and event_set add. This is thread-safe: any thread may call
  // import_fence without synchronizing with poll().
  iree_atomic_slist_push(&proactor->pending_fence_imports,
                         &tracker->slist_entry);
  iree_async_proactor_posix_wake_poll_thread(proactor);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

void iree_async_proactor_posix_drain_pending_fence_imports(
    iree_async_proactor_posix_t* proactor) {
  iree_atomic_slist_entry_t* head = NULL;
  if (!iree_atomic_slist_flush(&proactor->pending_fence_imports,
                               IREE_ATOMIC_SLIST_FLUSH_ORDER_APPROXIMATE_FIFO,
                               &head, /*out_tail=*/NULL)) {
    return;
  }

  while (head) {
    iree_async_posix_fence_import_tracker_t* tracker =
        (iree_async_posix_fence_import_tracker_t*)head;
    head = head->next;

    // Register fd in fd_map. This runs on the poll thread, so no race.
    iree_status_t status = iree_async_posix_fd_map_insert(
        &proactor->fd_map, tracker->fence_fd,
        IREE_ASYNC_POSIX_FD_HANDLER_FENCE_IMPORT, tracker);
    if (iree_status_is_ok(status)) {
      status = iree_async_posix_event_set_add(
          proactor->event_set, tracker->fence_fd, POLLIN | POLLPRI);
      if (!iree_status_is_ok(status)) {
        iree_async_posix_fd_map_remove(&proactor->fd_map, tracker->fence_fd);
      }
    }

    if (!iree_status_is_ok(status)) {
      // Registration failed. Clean up: close fd, release semaphore, free.
      iree_status_ignore(status);
      close(tracker->fence_fd);
      iree_async_semaphore_release(tracker->semaphore);
      iree_allocator_free(tracker->allocator, tracker);
    }
  }
}

void iree_async_proactor_posix_handle_fence_import(
    iree_async_proactor_posix_t* proactor,
    iree_async_posix_fence_import_tracker_t* tracker, short revents) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Check revents to determine success or failure.
  if (iree_any_bit_set((uint32_t)revents, POLLIN | POLLPRI)) {
    // POLLIN/POLLPRI set: fd became readable, fence signaled.
    iree_status_t signal_status = iree_async_semaphore_signal(
        tracker->semaphore, tracker->signal_value, /*frontier=*/NULL);
    if (!iree_status_is_ok(signal_status)) {
      // Signal failed (e.g., non-monotonic value). Fail the semaphore.
      iree_async_semaphore_fail(tracker->semaphore, signal_status);
    }
  } else {
    // No POLLIN: pure POLLHUP or POLLERR (fd error condition).
    iree_async_semaphore_fail(
        tracker->semaphore,
        iree_make_status(IREE_STATUS_DATA_LOSS,
                         "fence fd signaled error (poll events=0x%04x)",
                         revents));
  }

  // One-shot cleanup: remove from fd_map + event_set, then clean up.
  // Order: fd_map first (prevents stale lookups), then event_set.
  int fd = tracker->fence_fd;
  iree_async_posix_fd_map_remove(&proactor->fd_map, fd);
  iree_status_ignore(
      iree_async_posix_event_set_remove(proactor->event_set, fd));

  // Cleanup: close fd, release semaphore, free tracker.
  close(fd);
  iree_async_semaphore_release(tracker->semaphore);
  iree_allocator_free(tracker->allocator, tracker);

  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// Export fence implementation
//===----------------------------------------------------------------------===//

iree_status_t iree_async_proactor_posix_export_fence(
    iree_async_proactor_t* base_proactor, iree_async_semaphore_t* semaphore,
    uint64_t wait_value, iree_async_primitive_t* out_fence) {
  IREE_ASSERT_ARGUMENT(out_fence);
  *out_fence = iree_async_primitive_none();

  IREE_TRACE_ZONE_BEGIN(z0);

  // Create a signalable fd using wake.c pattern.
  // Linux: eventfd (single fd, read/write same fd)
  // macOS/BSD: pipe (separate read/write ends)
  iree_async_posix_wake_t wake;
  iree_status_t status = iree_async_posix_wake_initialize(&wake);
  if (!iree_status_is_ok(status)) {
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  // Allocate a tracker to bridge the semaphore timepoint to the fd.
  iree_async_posix_fence_export_tracker_t* tracker = NULL;
  status = iree_allocator_malloc(base_proactor->allocator, sizeof(*tracker),
                                 (void**)&tracker);
  if (!iree_status_is_ok(status)) {
    iree_async_posix_wake_deinitialize(&wake);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  // Retain the semaphore so it survives until the timepoint callback fires.
  // The timepoint API does not hold a reference.
  iree_async_semaphore_retain(semaphore);
  tracker->semaphore = semaphore;
  tracker->eventfd = wake.write_fd;  // Store write end for callback.
  tracker->allocator = base_proactor->allocator;

  // Set callback on the embedded timepoint before registering.
  tracker->timepoint.callback = iree_async_posix_fence_export_callback;
  tracker->timepoint.user_data = tracker;

  // Register the timepoint. If the semaphore has already reached wait_value,
  // the callback fires synchronously (writing to fd before we return).
  // If the semaphore is already failed, the callback fires with the failure
  // status (fd stays unreadable). acquire_timepoint always returns OK.
  status = iree_async_semaphore_acquire_timepoint(semaphore, wait_value,
                                                  &tracker->timepoint);

  if (iree_status_is_ok(status)) {
    // Return the read end as the exported fence. Caller owns the fd.
    // On Linux (eventfd), read_fd == write_fd. On macOS/BSD, read_fd is
    // pipe[0].
    *out_fence = iree_async_primitive_from_fd(wake.read_fd);
  } else {
    // Should not happen (acquire_timepoint only fails on NULL semaphore), but
    // handle defensively.
    iree_async_posix_wake_deinitialize(&wake);
    iree_async_semaphore_release(semaphore);
    iree_allocator_free(base_proactor->allocator, tracker);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_async_posix_fence_export_callback(
    void* user_data, iree_async_semaphore_timepoint_t* timepoint,
    iree_status_t status) {
  iree_async_posix_fence_export_tracker_t* tracker =
      (iree_async_posix_fence_export_tracker_t*)user_data;

  if (iree_status_is_ok(status)) {
    // Semaphore reached the target value. Write to fd to make it readable.
    uint64_t value = 1;
    ssize_t result = write(tracker->eventfd, &value, sizeof(value));
    // EAGAIN means already signaled (redundant, benign). Any other failure
    // (EBADF, EPIPE) indicates the eventfd/pipe was closed while a timepoint
    // callback was still pending — a lifecycle bug.
    IREE_ASSERT(result >= 0 || errno == EAGAIN);
  } else {
    // Semaphore failed or was cancelled. Leave fd unreadable so consumers
    // observe a timeout or check the semaphore for failure status.
    iree_status_ignore(status);
  }

  // Cleanup: release semaphore, free tracker. The eventfd/pipe is caller-owned
  // and must not be closed here.
  iree_async_semaphore_release(tracker->semaphore);
  iree_allocator_free(tracker->allocator, tracker);
}
