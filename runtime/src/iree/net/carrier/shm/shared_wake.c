// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/carrier/shm/shared_wake.h"

#if defined(IREE_PLATFORM_WINDOWS)
// Windows: Event objects for wake/signal primitives.
#else
// POSIX: eventfd (Linux) or pipe (macOS/BSD) for wake/signal primitives.
#include <fcntl.h>
#include <unistd.h>
#if defined(IREE_PLATFORM_LINUX) || defined(IREE_PLATFORM_ANDROID)
#include <sys/eventfd.h>
#endif  // IREE_PLATFORM_LINUX || IREE_PLATFORM_ANDROID
#endif  // IREE_PLATFORM_WINDOWS

static void iree_net_shm_shared_wake_scan(void* user_data,
                                          iree_async_operation_t* operation,
                                          iree_status_t status,
                                          iree_async_completion_flags_t flags);

// Posts the NOTIFICATION_WAIT if not already posted and there are sleeping
// carriers.
static iree_status_t iree_net_shm_shared_wake_ensure_wait_posted(
    iree_net_shm_shared_wake_t* shared_wake) {
  if (shared_wake->wait_posted) return iree_ok_status();
  if (!shared_wake->sleeping_head) return iree_ok_status();

  memset(&shared_wake->wait_operation, 0, sizeof(shared_wake->wait_operation));
  iree_async_operation_initialize(&shared_wake->wait_operation.base,
                                  IREE_ASYNC_OPERATION_TYPE_NOTIFICATION_WAIT,
                                  IREE_ASYNC_OPERATION_FLAG_NONE,
                                  iree_net_shm_shared_wake_scan, shared_wake);
  shared_wake->wait_operation.notification = shared_wake->notification;
  iree_status_t status = iree_async_proactor_submit_one(
      shared_wake->proactor, &shared_wake->wait_operation.base);
  if (iree_status_is_ok(status)) {
    shared_wake->wait_posted = true;
  }
  return status;
}

// Scan callback: fires when the shared notification's NOTIFICATION_WAIT
// completes. Iterates all sleeping carriers, dispatches per-carrier drain,
// and re-posts the wait if carriers remain in the list.
static void iree_net_shm_shared_wake_scan(void* user_data,
                                          iree_async_operation_t* operation,
                                          iree_status_t status,
                                          iree_async_completion_flags_t flags) {
  (void)operation;
  (void)flags;
  iree_net_shm_shared_wake_t* shared_wake =
      (iree_net_shm_shared_wake_t*)user_data;
  iree_status_ignore(status);
  shared_wake->wait_posted = false;

  // Iterate sleeping list with save-next-before-processing: drain_from_wake
  // may cause the carrier to transition out of the sleeping list.
  iree_net_shm_carrier_t* carrier = shared_wake->sleeping_head;
  iree_net_shm_carrier_t* previous = NULL;
  while (carrier) {
    iree_net_shm_carrier_t* next = iree_net_shm_carrier_sleeping_next(carrier);
    bool keep_sleeping = iree_net_shm_carrier_drain_from_wake(carrier);
    if (keep_sleeping) {
      previous = carrier;
    } else {
      // Remove from sleeping list. Do NOT touch the carrier after this —
      // drain_from_wake may have completed deactivation inline, which fires
      // the user's deactivation callback. That callback may release/free the
      // carrier. (Same pattern as progress callback's deferred on_remove.)
      if (previous) {
        iree_net_shm_carrier_set_sleeping_next(previous, next);
      } else {
        shared_wake->sleeping_head = next;
      }
    }
    carrier = next;
  }

  // Re-post if carriers remain sleeping.
  if (shared_wake->sleeping_head) {
    iree_status_t wait_status =
        iree_net_shm_shared_wake_ensure_wait_posted(shared_wake);
    if (IREE_UNLIKELY(!iree_status_is_ok(wait_status))) {
      iree_status_ignore(wait_status);
      // Cannot re-post — all sleeping carriers are orphaned. This is a
      // catastrophic failure (fd/proactor exhaustion). Each carrier's
      // deactivation will eventually time out or be forced externally.
    }
  }
}

//===----------------------------------------------------------------------===//
// API
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t iree_net_shm_shared_wake_create(
    iree_async_proactor_t* proactor, iree_allocator_t allocator,
    iree_net_shm_shared_wake_t** out_shared_wake) {
  IREE_ASSERT_ARGUMENT(proactor);
  IREE_ASSERT_ARGUMENT(out_shared_wake);
  *out_shared_wake = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_net_shm_shared_wake_t* shared_wake = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(allocator, sizeof(*shared_wake),
                                (void**)&shared_wake));
  memset(shared_wake, 0, sizeof(*shared_wake));
  iree_atomic_ref_count_init(&shared_wake->ref_count);
  shared_wake->proactor = proactor;
  iree_async_proactor_retain(proactor);
  shared_wake->allocator = allocator;

  // Create a local notification on the proactor. This uses whatever the
  // proactor's optimal mode is (futex, eventfd, pipe, etc.).
  iree_status_t notification_status = iree_async_notification_create(
      proactor, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &shared_wake->notification);

  if (iree_status_is_ok(notification_status)) {
    *out_shared_wake = shared_wake;
  } else {
    iree_async_proactor_release(proactor);
    iree_allocator_free(allocator, shared_wake);
  }
  IREE_TRACE_ZONE_END(z0);
  return notification_status;
}

// Creates platform-specific wake/signal primitives for shared mode.
// On success, both primitives are set and must be closed by the caller on
// error cleanup. On failure, any partially-created primitives are closed.
static iree_status_t iree_net_shm_shared_wake_create_primitives(
    iree_async_primitive_t* out_wake, iree_async_primitive_t* out_signal) {
  *out_wake = iree_async_primitive_none();
  *out_signal = iree_async_primitive_none();

#if defined(IREE_PLATFORM_WINDOWS)
  // Windows: create an auto-reset Event. Same Event is used for both
  // monitoring (wake) and signaling (SetEvent).
  HANDLE event = CreateEventW(NULL, /*bManualReset=*/FALSE,
                              /*bInitialState=*/FALSE, NULL);
  if (!event) {
    return iree_make_status(iree_status_code_from_win32_error(GetLastError()),
                            "CreateEvent failed for shared wake");
  }
  *out_wake = iree_async_primitive_from_win32_handle((uintptr_t)event);
  *out_signal = iree_async_primitive_from_win32_handle((uintptr_t)event);
  return iree_ok_status();

#elif defined(IREE_PLATFORM_LINUX) || defined(IREE_PLATFORM_ANDROID)
  // Linux/Android: eventfd (bidirectional, same fd for wake and signal).
  int efd = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
  if (efd < 0) {
    return iree_make_status(iree_status_code_from_errno(errno),
                            "eventfd creation failed for shared wake");
  }
  *out_wake = iree_async_primitive_from_fd(efd);
  *out_signal = iree_async_primitive_from_fd(efd);
  return iree_ok_status();

#else
  // macOS/BSD: pipe. Read end for wake (POLLIN), write end for signal.
  int pipe_fds[2];
  if (pipe(pipe_fds) < 0) {
    return iree_make_status(iree_status_code_from_errno(errno),
                            "pipe creation failed for shared wake");
  }
  for (int i = 0; i < 2; ++i) {
    int current_flags = fcntl(pipe_fds[i], F_GETFL);
    if (current_flags >= 0) {
      fcntl(pipe_fds[i], F_SETFL, current_flags | O_NONBLOCK);
    }
    fcntl(pipe_fds[i], F_SETFD, FD_CLOEXEC);
  }
  *out_wake = iree_async_primitive_from_fd(pipe_fds[0]);
  *out_signal = iree_async_primitive_from_fd(pipe_fds[1]);
  return iree_ok_status();
#endif
}

// Closes the owned wake/signal primitives. On Linux/Android the wake and
// signal primitives are the same eventfd — only close once.
static void iree_net_shm_shared_wake_close_primitives(
    iree_async_primitive_t* wake, iree_async_primitive_t* signal) {
#if defined(IREE_PLATFORM_LINUX) || defined(IREE_PLATFORM_ANDROID)
  // Eventfd: wake == signal (same fd). Close once via wake.
  iree_async_primitive_close(wake);
  *signal = iree_async_primitive_none();
#elif defined(IREE_PLATFORM_WINDOWS)
  // Windows: wake == signal (same Event HANDLE). Close once via wake.
  iree_async_primitive_close(wake);
  *signal = iree_async_primitive_none();
#else
  // macOS/BSD: pipe has two independent fds.
  iree_async_primitive_close(wake);
  iree_async_primitive_close(signal);
#endif
}

IREE_API_EXPORT iree_status_t iree_net_shm_shared_wake_create_shared(
    iree_async_proactor_t* proactor, iree_allocator_t allocator,
    iree_net_shm_shared_wake_t** out_shared_wake) {
  IREE_ASSERT_ARGUMENT(proactor);
  IREE_ASSERT_ARGUMENT(out_shared_wake);
  *out_shared_wake = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_net_shm_shared_wake_t* shared_wake = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(allocator, sizeof(*shared_wake),
                                (void**)&shared_wake));
  memset(shared_wake, 0, sizeof(*shared_wake));
  iree_atomic_ref_count_init(&shared_wake->ref_count);
  shared_wake->proactor = proactor;
  iree_async_proactor_retain(proactor);
  shared_wake->allocator = allocator;
  shared_wake->is_shared = true;
  shared_wake->epoch_mapping.handle = IREE_SHM_HANDLE_INVALID;

  // Allocate a SHM page for the epoch counter. This is what gets exported
  // to remote peers so they can create proxy notifications whose epoch_ptr
  // points at the same physical page (enabling cross-process futex).
  iree_status_t status =
      iree_shm_create(iree_shm_options_default(), sizeof(iree_atomic_int32_t),
                      &shared_wake->epoch_mapping);

  // Create platform wake/signal primitives.
  if (iree_status_is_ok(status)) {
    status = iree_net_shm_shared_wake_create_primitives(
        &shared_wake->owned_wake_primitive,
        &shared_wake->owned_signal_primitive);
  }

  // Create a shared notification backed by our SHM epoch and owned primitives.
  if (iree_status_is_ok(status)) {
    iree_async_notification_shared_options_t notification_options;
    memset(&notification_options, 0, sizeof(notification_options));
    notification_options.epoch_address =
        (iree_atomic_int32_t*)shared_wake->epoch_mapping.base;
    notification_options.wake_primitive = shared_wake->owned_wake_primitive;
    notification_options.signal_primitive = shared_wake->owned_signal_primitive;
    status = iree_async_notification_create_shared(
        proactor, &notification_options, &shared_wake->notification);
  }

  if (iree_status_is_ok(status)) {
    *out_shared_wake = shared_wake;
  } else {
    // All cleanup functions are NULL/invalid-handle safe. The fields are
    // zero-initialized and only populated on success, so closing them
    // unconditionally is safe regardless of which step failed.
    iree_net_shm_shared_wake_close_primitives(
        &shared_wake->owned_wake_primitive,
        &shared_wake->owned_signal_primitive);
    iree_shm_close(&shared_wake->epoch_mapping);
    iree_async_proactor_release(proactor);
    iree_allocator_free(allocator, shared_wake);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t
iree_net_shm_shared_wake_export(iree_net_shm_shared_wake_t* shared_wake,
                                iree_net_shm_shared_wake_export_t* out_export) {
  IREE_ASSERT_ARGUMENT(shared_wake);
  IREE_ASSERT_ARGUMENT(out_export);
  memset(out_export, 0, sizeof(*out_export));

  if (!shared_wake->is_shared) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "cannot export from a local (non-shared) "
                            "shared_wake; use create_shared()");
  }

  // Duplicate the SHM handle for the epoch region.
  iree_status_t status = iree_shm_handle_dup(shared_wake->epoch_mapping.handle,
                                             &out_export->epoch_shm_handle);
  if (!iree_status_is_ok(status)) return status;
  out_export->epoch_shm_size = shared_wake->epoch_mapping.size;

  // Duplicate the signal primitive. The remote peer writes to this to wake
  // our proactor's poll loop.
  status = iree_async_primitive_dup(shared_wake->owned_signal_primitive,
                                    &out_export->signal_primitive);
  if (!iree_status_is_ok(status)) {
    iree_shm_handle_close(&out_export->epoch_shm_handle);
    return status;
  }

  return iree_ok_status();
}

static void iree_net_shm_shared_wake_destroy(
    iree_net_shm_shared_wake_t* shared_wake) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT(!shared_wake->sleeping_head,
              "shared_wake destroyed with carriers still in the sleeping "
              "list; unregister all carriers before releasing");
  IREE_ASSERT(!shared_wake->wait_posted,
              "shared_wake destroyed with NOTIFICATION_WAIT in flight");
  iree_async_notification_release(shared_wake->notification);
  if (shared_wake->is_shared) {
    // The shared notification does not own the primitives (SHARED flag).
    // We own them and must close them here.
    iree_net_shm_shared_wake_close_primitives(
        &shared_wake->owned_wake_primitive,
        &shared_wake->owned_signal_primitive);
    iree_shm_close(&shared_wake->epoch_mapping);
  }
  iree_async_proactor_release(shared_wake->proactor);
  iree_allocator_t allocator = shared_wake->allocator;
  iree_allocator_free(allocator, shared_wake);
  IREE_TRACE_ZONE_END(z0);
}

IREE_API_EXPORT void iree_net_shm_shared_wake_retain(
    iree_net_shm_shared_wake_t* shared_wake) {
  if (shared_wake) {
    iree_atomic_ref_count_inc(&shared_wake->ref_count);
  }
}

IREE_API_EXPORT void iree_net_shm_shared_wake_release(
    iree_net_shm_shared_wake_t* shared_wake) {
  if (!shared_wake) return;
  if (iree_atomic_ref_count_dec(&shared_wake->ref_count) == 1) {
    iree_net_shm_shared_wake_destroy(shared_wake);
  }
}

IREE_API_EXPORT iree_status_t iree_net_shm_shared_wake_register(
    iree_net_shm_shared_wake_t* shared_wake, iree_net_shm_carrier_t* carrier) {
  IREE_ASSERT_ARGUMENT(shared_wake);
  IREE_ASSERT_ARGUMENT(carrier);
  // Prepend to sleeping list.
  iree_net_shm_carrier_set_sleeping_next(carrier, shared_wake->sleeping_head);
  shared_wake->sleeping_head = carrier;
  // Ensure the NOTIFICATION_WAIT is posted to catch signals.
  iree_status_t status =
      iree_net_shm_shared_wake_ensure_wait_posted(shared_wake);
  if (IREE_UNLIKELY(!iree_status_is_ok(status))) {
    // Remove the carrier we just prepended — it can never be woken without a
    // functioning NOTIFICATION_WAIT.
    shared_wake->sleeping_head = iree_net_shm_carrier_sleeping_next(carrier);
    iree_net_shm_carrier_set_sleeping_next(carrier, NULL);
  }
  return status;
}

IREE_API_EXPORT void iree_net_shm_shared_wake_unregister(
    iree_net_shm_shared_wake_t* shared_wake, iree_net_shm_carrier_t* carrier) {
  IREE_ASSERT_ARGUMENT(shared_wake);
  IREE_ASSERT_ARGUMENT(carrier);
  // Linear scan to find and remove. O(N) but N is small (typically 1-4
  // carriers per proactor) and this runs on the poll thread only.
  iree_net_shm_carrier_t* current = shared_wake->sleeping_head;
  iree_net_shm_carrier_t* previous = NULL;
  while (current) {
    if (current == carrier) {
      iree_net_shm_carrier_t* next =
          iree_net_shm_carrier_sleeping_next(carrier);
      if (previous) {
        iree_net_shm_carrier_set_sleeping_next(previous, next);
      } else {
        shared_wake->sleeping_head = next;
      }
      iree_net_shm_carrier_set_sleeping_next(carrier, NULL);
      return;
    }
    previous = current;
    current = iree_net_shm_carrier_sleeping_next(current);
  }
  // Not found — no-op (carrier may have already been removed by scan).
}
