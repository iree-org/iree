// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_ASYNC_PLATFORM_POSIX_SIGNAL_H_
#define IREE_ASYNC_PLATFORM_POSIX_SIGNAL_H_

#include "iree/async/proactor.h"
#include "iree/async/util/signal.h"
#include "iree/base/api.h"

// Self-pipe is the fallback for non-Linux POSIX platforms and for testing on
// Linux when IREE_ASYNC_SIGNAL_FORCE_SELFPIPE is defined.
#if !defined(IREE_PLATFORM_WINDOWS) && !defined(IREE_PLATFORM_EMSCRIPTEN)

#include <signal.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Self-pipe signal handling
//===----------------------------------------------------------------------===//
// Portable POSIX signal handling using the self-pipe trick.
//
// A pipe is created and signal handlers installed via sigaction(). When a
// signal arrives, the handler writes the signal number as a single byte to the
// pipe's write end. The read end is registered as an event source with the
// proactor's poll loop, so signal delivery integrates into the normal I/O path.
//
// This works on all POSIX platforms (macOS, BSD, Linux) and avoids the
// Linux-specific signalfd mechanism. On Linux, signalfd is preferred for
// efficiency (no signal handler overhead, kernel-managed buffering), but
// self-pipe is used as a fallback and for cross-platform testing.
//
// Signals are managed dynamically: only signals with active subscriptions have
// custom handlers installed. When the last subscription for a signal is
// removed, the original handler is restored.

// State for self-pipe signal handling.
typedef struct iree_async_selfpipe_signal_state_t {
  // Pipe file descriptors. -1 when inactive.
  int pipe_read_fd;
  int pipe_write_fd;

  // Currently active signals (those with subscriptions).
  sigset_t active_mask;

  // Original signal mask saved before any modifications, restored on
  // deinitialize.
  sigset_t saved_sigmask;

  // Whether saved_sigmask is valid (at least one signal has been added).
  bool sigmask_saved;

  // Original signal actions, saved per-signal for restoration on remove.
  // Indexed by iree_async_signal_t enum value.
  struct sigaction saved_actions[IREE_ASYNC_SIGNAL_COUNT];

  // Whether each saved_actions entry is valid.
  bool action_saved[IREE_ASYNC_SIGNAL_COUNT];
} iree_async_selfpipe_signal_state_t;

// Initializes signal state to defaults. Call before any add/remove operations.
static inline void iree_async_selfpipe_signal_state_initialize(
    iree_async_selfpipe_signal_state_t* state) {
  state->pipe_read_fd = -1;
  state->pipe_write_fd = -1;
  sigemptyset(&state->active_mask);
  sigemptyset(&state->saved_sigmask);
  state->sigmask_saved = false;
  memset(state->saved_actions, 0, sizeof(state->saved_actions));
  memset(state->action_saved, 0, sizeof(state->action_saved));
}

// Adds |signal| to the set of handled signals.
//
// On the first call, creates the pipe and stores the original signal mask.
// Installs a sigaction handler that writes the signal number to the pipe, then
// unblocks the signal so the handler can fire (signals may have been blocked by
// an earlier call to iree_async_signal_block_default()).
//
// Returns the pipe read fd (for registration with poll). The fd remains the
// same across all add_signal calls.
//
// Thread safety: Call from main thread, serialized with poll.
iree_status_t iree_async_selfpipe_signal_add_signal(
    iree_async_selfpipe_signal_state_t* state, iree_async_signal_t signal,
    int* out_fd);

// Removes |signal| from the set of handled signals.
//
// Blocks |signal| and restores the original sigaction handler. When the last
// signal is removed, closes the pipe and restores the original signal mask.
//
// Thread safety: Call from main thread, serialized with poll.
void iree_async_selfpipe_signal_remove_signal(
    iree_async_selfpipe_signal_state_t* state, iree_async_signal_t signal);

// Deinitializes all signal handling unconditionally.
//
// Restores all original sigaction handlers, closes the pipe, and restores the
// original signal mask. Use during proactor destruction to clean up any
// remaining signal state.
//
// Thread safety: Call from main thread after poll loop exits.
void iree_async_selfpipe_signal_deinitialize(
    iree_async_selfpipe_signal_state_t* state);

// Reads all pending signals from the pipe and dispatches them.
//
// Drains the pipe completely (reads until EAGAIN) to prevent signals from
// lagging. Invokes |callback| for each signal received, in the order they
// were written.
//
// Call this when the pipe read fd becomes readable (e.g., from an event source
// callback).
//
// Thread safety: Call from polling thread (serialized with other poll work).
iree_status_t iree_async_selfpipe_signal_read_pipe(
    iree_async_selfpipe_signal_state_t* state,
    iree_async_signal_dispatch_callback_t callback);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // !IREE_PLATFORM_WINDOWS && !IREE_PLATFORM_EMSCRIPTEN

#endif  // IREE_ASYNC_PLATFORM_POSIX_SIGNAL_H_
