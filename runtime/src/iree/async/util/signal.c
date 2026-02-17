// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/util/signal.h"

#include "iree/base/internal/atomics.h"

#if !defined(IREE_PLATFORM_WINDOWS)
#include <signal.h>
#endif

//===----------------------------------------------------------------------===//
// iree_async_signal_name
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_string_view_t
iree_async_signal_name(iree_async_signal_t signal) {
  switch (signal) {
    case IREE_ASYNC_SIGNAL_NONE:
      return iree_make_cstring_view("NONE");
    case IREE_ASYNC_SIGNAL_INTERRUPT:
      return iree_make_cstring_view("INTERRUPT");
    case IREE_ASYNC_SIGNAL_TERMINATE:
      return iree_make_cstring_view("TERMINATE");
    case IREE_ASYNC_SIGNAL_HANGUP:
      return iree_make_cstring_view("HANGUP");
    case IREE_ASYNC_SIGNAL_QUIT:
      return iree_make_cstring_view("QUIT");
    case IREE_ASYNC_SIGNAL_USER1:
      return iree_make_cstring_view("USER1");
    case IREE_ASYNC_SIGNAL_USER2:
      return iree_make_cstring_view("USER2");
    default:
      return iree_make_cstring_view("UNKNOWN");
  }
}

//===----------------------------------------------------------------------===//
// iree_async_signal_block_default / iree_async_signal_ignore_broken_pipe
//===----------------------------------------------------------------------===//

#if defined(IREE_PLATFORM_WINDOWS)

// Windows doesn't have POSIX signals. These are no-ops.

IREE_API_EXPORT iree_status_t iree_async_signal_block_default(void) {
  // No-op on Windows; signals don't exist in the POSIX sense.
  // Console control events are handled via SetConsoleCtrlHandler.
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_async_signal_ignore_broken_pipe(void) {
  // No-op on Windows; SIGPIPE doesn't exist.
  // Socket errors are returned from send/recv calls directly.
  return iree_ok_status();
}

#elif defined(IREE_PLATFORM_EMSCRIPTEN)

// Emscripten doesn't have real signal handling.

IREE_API_EXPORT iree_status_t iree_async_signal_block_default(void) {
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_async_signal_ignore_broken_pipe(void) {
  return iree_ok_status();
}

#else  // POSIX (Linux, macOS, BSD, etc.)

#include <errno.h>
#include <pthread.h>

IREE_API_EXPORT iree_status_t iree_async_signal_block_default(void) {
  // Block signals that will be handled via proactor subscriptions.
  // Child threads inherit this mask, so call BEFORE spawning any threads.
  sigset_t mask;
  iree_async_signal_build_sigset(&mask);

  if (pthread_sigmask(SIG_BLOCK, &mask, NULL) != 0) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "pthread_sigmask(SIG_BLOCK) failed: %d", errno);
  }
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_async_signal_ignore_broken_pipe(void) {
  // Ignore SIGPIPE so that writes to closed sockets return EPIPE instead of
  // killing the process. This is standard practice for network servers.
  struct sigaction sa;
  memset(&sa, 0, sizeof(sa));
  sa.sa_handler = SIG_IGN;
  sigemptyset(&sa.sa_mask);
  sa.sa_flags = 0;

  if (sigaction(SIGPIPE, &sa, NULL) != 0) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "sigaction(SIGPIPE, SIG_IGN) failed: %d", errno);
  }
  return iree_ok_status();
}

#endif  // POSIX

//===----------------------------------------------------------------------===//
// Global signal ownership
//===----------------------------------------------------------------------===//

// Process-global atomic tracking which proactor owns signals.
static iree_atomic_intptr_t g_signal_owner_proactor = IREE_ATOMIC_VAR_INIT(0);

iree_status_t iree_async_signal_claim_ownership(
    iree_async_proactor_t* proactor) {
  intptr_t proactor_value = (intptr_t)proactor;
  intptr_t expected = 0;
  if (iree_atomic_compare_exchange_strong(
          &g_signal_owner_proactor, &expected, proactor_value,
          iree_memory_order_acq_rel, iree_memory_order_acquire)) {
    // Successfully claimed ownership.
    return iree_ok_status();
  }
  if (expected == proactor_value) {
    // Already own it.
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                          "signals already owned by another proactor");
}

void iree_async_signal_release_ownership(iree_async_proactor_t* proactor) {
  intptr_t proactor_value = (intptr_t)proactor;
  intptr_t expected = proactor_value;
  // Only release if we're the actual owner. CAS handles concurrent release
  // attempts safely (only one succeeds).
  iree_atomic_compare_exchange_strong(&g_signal_owner_proactor, &expected, 0,
                                      iree_memory_order_release,
                                      iree_memory_order_relaxed);
}

//===----------------------------------------------------------------------===//
// Subscription list management
//===----------------------------------------------------------------------===//

void iree_async_signal_subscription_initialize(
    iree_async_signal_subscription_t* subscription,
    iree_async_proactor_t* proactor, iree_async_signal_t signal,
    iree_async_signal_callback_t callback) {
  subscription->next = NULL;
  subscription->prev = NULL;
  subscription->pending_next = NULL;
  subscription->proactor = proactor;
  subscription->signal = signal;
  subscription->callback = callback;
}

void iree_async_signal_subscription_link(
    iree_async_signal_subscription_t** head_ptr,
    iree_async_signal_subscription_t* subscription) {
  if (!*head_ptr) {
    // Empty list: subscription becomes the head.
    *head_ptr = subscription;
    subscription->next = NULL;
    subscription->prev = NULL;
  } else {
    // Walk to the tail and append.
    iree_async_signal_subscription_t* tail = *head_ptr;
    while (tail->next) {
      tail = tail->next;
    }
    tail->next = subscription;
    subscription->prev = tail;
    subscription->next = NULL;
  }
}

bool iree_async_signal_subscription_unlink(
    iree_async_signal_subscription_t** head_ptr,
    iree_async_signal_subscription_t* subscription) {
  if (!subscription) return false;

  // Check if already unlinked (both pointers NULL and not the head).
  if (!subscription->prev && !subscription->next && *head_ptr != subscription) {
    return false;
  }

  // Unlink from doubly-linked list.
  if (subscription->prev) {
    subscription->prev->next = subscription->next;
  } else {
    // subscription was the head.
    *head_ptr = subscription->next;
  }
  if (subscription->next) {
    subscription->next->prev = subscription->prev;
  }

  // Clear pointers to mark as unlinked.
  subscription->prev = NULL;
  subscription->next = NULL;
  return true;
}

void iree_async_signal_subscription_defer_unsubscribe(
    iree_async_signal_dispatch_state_t* dispatch_state,
    iree_async_signal_subscription_t* subscription) {
  // Prevent future invocations.
  subscription->callback.fn = NULL;

  // Add to pending unsubscribes list using dedicated pending_next pointer.
  // This preserves the main list's next/prev pointers for correct unlinking.
  subscription->pending_next = dispatch_state->pending_unsubscribes;
  dispatch_state->pending_unsubscribes = subscription;
}

iree_async_signal_subscription_t* iree_async_signal_subscription_dispatch(
    iree_async_signal_subscription_t* head,
    iree_async_signal_dispatch_state_t* dispatch_state,
    iree_async_signal_t signal) {
  // Enter dispatch mode: unsubscribes will be deferred.
  dispatch_state->dispatching = true;

  // Iterate and invoke callbacks. Cache next before calling in case the
  // callback triggers an unsubscribe (which would modify the list).
  iree_async_signal_subscription_t* current = head;
  while (current) {
    iree_async_signal_subscription_t* next = current->next;

    // Only invoke if callback hasn't been cleared (deferred unsubscribe).
    if (current->callback.fn) {
      current->callback.fn(current->callback.user_data, signal);
    }

    current = next;
  }

  // Exit dispatch mode.
  dispatch_state->dispatching = false;

  // Return pending unsubscribes for caller to free.
  iree_async_signal_subscription_t* to_free =
      dispatch_state->pending_unsubscribes;
  dispatch_state->pending_unsubscribes = NULL;
  return to_free;
}

//===----------------------------------------------------------------------===//
// POSIX signal number conversion
//===----------------------------------------------------------------------===//

#if !defined(IREE_PLATFORM_WINDOWS) && !defined(IREE_PLATFORM_EMSCRIPTEN)

int iree_async_signal_to_posix(iree_async_signal_t signal) {
  switch (signal) {
    case IREE_ASYNC_SIGNAL_INTERRUPT:
      return SIGINT;
    case IREE_ASYNC_SIGNAL_TERMINATE:
      return SIGTERM;
    case IREE_ASYNC_SIGNAL_HANGUP:
      return SIGHUP;
    case IREE_ASYNC_SIGNAL_QUIT:
      return SIGQUIT;
    case IREE_ASYNC_SIGNAL_USER1:
      return SIGUSR1;
    case IREE_ASYNC_SIGNAL_USER2:
      return SIGUSR2;
    default:
      return 0;
  }
}

iree_async_signal_t iree_async_signal_from_posix(int signo) {
  switch (signo) {
    case SIGINT:
      return IREE_ASYNC_SIGNAL_INTERRUPT;
    case SIGTERM:
      return IREE_ASYNC_SIGNAL_TERMINATE;
    case SIGHUP:
      return IREE_ASYNC_SIGNAL_HANGUP;
    case SIGQUIT:
      return IREE_ASYNC_SIGNAL_QUIT;
    case SIGUSR1:
      return IREE_ASYNC_SIGNAL_USER1;
    case SIGUSR2:
      return IREE_ASYNC_SIGNAL_USER2;
    default:
      return IREE_ASYNC_SIGNAL_NONE;
  }
}

void iree_async_signal_build_sigset(sigset_t* mask) {
  sigemptyset(mask);
  sigaddset(mask, SIGINT);
  sigaddset(mask, SIGTERM);
  sigaddset(mask, SIGHUP);
  sigaddset(mask, SIGQUIT);
  sigaddset(mask, SIGUSR1);
  sigaddset(mask, SIGUSR2);
}

#endif  // !IREE_PLATFORM_WINDOWS && !IREE_PLATFORM_EMSCRIPTEN
