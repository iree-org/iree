// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_ASYNC_PLATFORM_LINUX_SIGNAL_H_
#define IREE_ASYNC_PLATFORM_LINUX_SIGNAL_H_

#include "iree/async/proactor.h"
#include "iree/async/util/signal.h"
#include "iree/base/api.h"

#if defined(IREE_PLATFORM_LINUX) || defined(IREE_PLATFORM_ANDROID)

#include <signal.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Linux signalfd-based signal handling
//===----------------------------------------------------------------------===//
// Shared infrastructure for io_uring and epoll proactor backends.
//
// Linux delivers signals via signalfd: a file descriptor that becomes readable
// when signals are pending. This integrates cleanly with poll-based event loops
// and avoids the complexity of signal handlers.
//
// Signals are managed dynamically: only signals with active subscriptions are
// blocked and routed to signalfd. When the last subscription for a signal is
// removed, that signal is unblocked and reverts to default behavior.

// State for Linux signal handling via signalfd.
typedef struct iree_async_linux_signal_state_t {
  // signalfd for active signals. -1 if no signals are subscribed.
  int signal_fd;

  // Currently active signals (those with subscriptions).
  sigset_t active_mask;

  // Original signal mask saved before any modifications, restored on
  // deinitialize.
  sigset_t saved_sigmask;

  // Whether saved_sigmask is valid (at least one signal has been added).
  bool sigmask_saved;
} iree_async_linux_signal_state_t;

// Initializes signal state to defaults. Call before any add/remove operations.
static inline void iree_async_linux_signal_state_initialize(
    iree_async_linux_signal_state_t* state) {
  state->signal_fd = -1;
  sigemptyset(&state->active_mask);
  sigemptyset(&state->saved_sigmask);
  state->sigmask_saved = false;
}

// Adds |signal| to the set of handled signals.
//
// Blocks |signal| via pthread_sigmask and updates the signalfd to monitor it.
// On the first call, creates the signalfd and saves the original signal mask.
//
// Returns the signalfd file descriptor (for registration with poll). The fd
// remains the same across all add_signal calls.
//
// Thread safety: Call from main thread, serialized with poll.
iree_status_t iree_async_linux_signal_add_signal(
    iree_async_linux_signal_state_t* state, iree_async_signal_t signal,
    int* out_signal_fd);

// Removes |signal| from the set of handled signals.
//
// Unblocks |signal| via pthread_sigmask and updates the signalfd to stop
// monitoring it. When the last signal is removed, closes the signalfd and
// restores the original signal mask.
//
// Thread safety: Call from main thread, serialized with poll.
void iree_async_linux_signal_remove_signal(
    iree_async_linux_signal_state_t* state, iree_async_signal_t signal);

// Deinitializes all signal handling unconditionally.
//
// Closes the signalfd and restores the original signal mask. Use this during
// proactor destruction to clean up any remaining signal state.
//
// Thread safety: Call from main thread after poll loop exits.
void iree_async_linux_signal_deinitialize(
    iree_async_linux_signal_state_t* state);

// Reads all pending signals from the signalfd and dispatches them.
//
// Drains the signalfd completely (reads until EAGAIN) to prevent signals from
// lagging. Invokes |callback| for each signal received, in the order they
// were delivered.
//
// Call this when the signalfd becomes readable (e.g., from an event source
// callback or epoll event handler).
//
// Thread safety: Call from polling thread (serialized with other poll work).
iree_status_t iree_async_linux_signal_read_signalfd(
    iree_async_linux_signal_state_t* state,
    iree_async_signal_dispatch_callback_t callback);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_PLATFORM_LINUX || IREE_PLATFORM_ANDROID

#endif  // IREE_ASYNC_PLATFORM_LINUX_SIGNAL_H_
