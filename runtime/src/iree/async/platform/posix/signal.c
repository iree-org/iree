// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/platform/posix/signal.h"

#if !defined(IREE_PLATFORM_WINDOWS) && !defined(IREE_PLATFORM_EMSCRIPTEN)

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <unistd.h>

#include "iree/async/util/signal.h"

//===----------------------------------------------------------------------===//
// Self-pipe global state
//===----------------------------------------------------------------------===//
// Signal handlers are process-global, so the pipe write fd must be accessible
// from the handler. Only one proactor can own signals at a time (enforced by
// iree_async_signal_claim_ownership), so a single global is sufficient.
//
// Lifetime safety: the write fd is set BEFORE unblocking signals and cleared
// AFTER blocking them, so the handler always sees a valid fd or -1.

static volatile int g_selfpipe_write_fd = -1;

// Signal handler installed via sigaction. Writes the POSIX signal number as a
// single byte to the pipe. write() is async-signal-safe per POSIX.
// O_NONBLOCK on the write end prevents deadlock if the pipe buffer is full;
// a dropped write means a dropped signal, which is acceptable (signals are
// inherently coalesceable).
static void iree_selfpipe_signal_handler(int signo) {
  int fd = g_selfpipe_write_fd;
  if (fd >= 0) {
    uint8_t byte = (uint8_t)signo;
    // Async-signal-safe: cannot log, assert, or handle errors.
    // EAGAIN (pipe full) means signal already pending (coalesced by design).
    // EBADF (teardown race) means system is shutting down.
    ssize_t ignored = write(fd, &byte, 1);
    (void)ignored;
  }
}

//===----------------------------------------------------------------------===//
// Pipe creation
//===----------------------------------------------------------------------===//

// Creates a pipe with both ends set to O_NONBLOCK and FD_CLOEXEC.
// Uses pipe() + fcntl() for portability (pipe2 is Linux-specific).
static iree_status_t iree_selfpipe_create_pipe(int* out_read_fd,
                                               int* out_write_fd) {
  int fds[2];
  if (pipe(fds) != 0) {
    return iree_make_status(IREE_STATUS_INTERNAL, "pipe() failed: %d", errno);
  }

  // Set both ends to non-blocking and close-on-exec.
  for (int i = 0; i < 2; ++i) {
    int flags = fcntl(fds[i], F_GETFL);
    if (flags == -1 || fcntl(fds[i], F_SETFL, flags | O_NONBLOCK) == -1) {
      IREE_ATTRIBUTE_UNUSED int error = errno;
      close(fds[0]);
      close(fds[1]);
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "fcntl(F_SETFL, O_NONBLOCK) failed: %d", error);
    }
    if (fcntl(fds[i], F_SETFD, FD_CLOEXEC) == -1) {
      IREE_ATTRIBUTE_UNUSED int error = errno;
      close(fds[0]);
      close(fds[1]);
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "fcntl(F_SETFD, FD_CLOEXEC) failed: %d", error);
    }
  }

  *out_read_fd = fds[0];
  *out_write_fd = fds[1];
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Signal mask helpers
//===----------------------------------------------------------------------===//

// Returns true if the sigset contains no signals we care about.
static bool iree_sigset_is_empty(const sigset_t* set) {
  return !sigismember(set, SIGINT) && !sigismember(set, SIGTERM) &&
         !sigismember(set, SIGHUP) && !sigismember(set, SIGQUIT) &&
         !sigismember(set, SIGUSR1) && !sigismember(set, SIGUSR2);
}

//===----------------------------------------------------------------------===//
// Implementation
//===----------------------------------------------------------------------===//

iree_status_t iree_async_selfpipe_signal_add_signal(
    iree_async_selfpipe_signal_state_t* state, iree_async_signal_t signal,
    int* out_fd) {
  int posix_signal = iree_async_signal_to_posix(signal);
  if (posix_signal == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid signal type %d", (int)signal);
  }

  // Already handling this signal — return the existing pipe read fd.
  if (sigismember(&state->active_mask, posix_signal)) {
    *out_fd = state->pipe_read_fd;
    return iree_ok_status();
  }

  // First signal being added — save original mask and create the pipe.
  if (!state->sigmask_saved) {
    if (pthread_sigmask(SIG_BLOCK, NULL, &state->saved_sigmask) != 0) {
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "pthread_sigmask(query) failed: %d", errno);
    }
    state->sigmask_saved = true;
  }

  // Create pipe on first signal (pipe_read_fd == -1 means no pipe yet).
  if (state->pipe_read_fd < 0) {
    iree_status_t status =
        iree_selfpipe_create_pipe(&state->pipe_read_fd, &state->pipe_write_fd);
    if (!iree_status_is_ok(status)) return status;
    // Publish the write fd for the signal handler.
    g_selfpipe_write_fd = state->pipe_write_fd;
  }

  // Save the original sigaction for this signal.
  struct sigaction new_action;
  memset(&new_action, 0, sizeof(new_action));
  new_action.sa_handler = iree_selfpipe_signal_handler;
  sigemptyset(&new_action.sa_mask);
  // SA_RESTART: interrupted syscalls are restarted, minimizing disruption to
  // other threads that may be in blocking calls.
  new_action.sa_flags = SA_RESTART;

  if (sigaction(posix_signal, &new_action, &state->saved_actions[signal]) !=
      0) {
    IREE_ATTRIBUTE_UNUSED int error = errno;
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "sigaction(install) failed: %d", error);
  }
  state->action_saved[signal] = true;

  // Unblock this signal so our handler fires. Signals may have been blocked by
  // iree_async_signal_block_default() called early in the process.
  sigset_t unblock_mask;
  sigemptyset(&unblock_mask);
  sigaddset(&unblock_mask, posix_signal);
  if (pthread_sigmask(SIG_UNBLOCK, &unblock_mask, NULL) != 0) {
    IREE_ATTRIBUTE_UNUSED int error = errno;
    // Rollback: restore original handler.
    sigaction(posix_signal, &state->saved_actions[signal], NULL);
    state->action_saved[signal] = false;
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "pthread_sigmask(SIG_UNBLOCK) failed: %d", error);
  }

  // Track in active mask.
  sigaddset(&state->active_mask, posix_signal);

  *out_fd = state->pipe_read_fd;
  return iree_ok_status();
}

void iree_async_selfpipe_signal_remove_signal(
    iree_async_selfpipe_signal_state_t* state, iree_async_signal_t signal) {
  int posix_signal = iree_async_signal_to_posix(signal);
  if (posix_signal == 0) return;

  // Not handling this signal.
  if (!sigismember(&state->active_mask, posix_signal)) return;

  // Block the signal first — prevents new handler invocations. Any in-flight
  // handler on this thread has completed by the time pthread_sigmask returns.
  sigset_t block_mask;
  sigemptyset(&block_mask);
  sigaddset(&block_mask, posix_signal);
  pthread_sigmask(SIG_BLOCK, &block_mask, NULL);

  // Restore the original sigaction handler.
  if (state->action_saved[signal]) {
    sigaction(posix_signal, &state->saved_actions[signal], NULL);
    state->action_saved[signal] = false;
  }

  // Remove from active mask.
  sigdelset(&state->active_mask, posix_signal);

  // If no signals remain, close the pipe.
  if (iree_sigset_is_empty(&state->active_mask)) {
    g_selfpipe_write_fd = -1;
    if (state->pipe_write_fd >= 0) {
      close(state->pipe_write_fd);
      state->pipe_write_fd = -1;
    }
    if (state->pipe_read_fd >= 0) {
      close(state->pipe_read_fd);
      state->pipe_read_fd = -1;
    }
  }
}

void iree_async_selfpipe_signal_deinitialize(
    iree_async_selfpipe_signal_state_t* state) {
  // Block all handled signals to prevent handler invocations during
  // deinitialization.
  if (!iree_sigset_is_empty(&state->active_mask)) {
    pthread_sigmask(SIG_BLOCK, &state->active_mask, NULL);
  }

  // Restore all original sigaction handlers.
  for (int i = 1; i < IREE_ASYNC_SIGNAL_COUNT; ++i) {
    if (state->action_saved[i]) {
      int posix_signal = iree_async_signal_to_posix((iree_async_signal_t)i);
      if (posix_signal != 0) {
        sigaction(posix_signal, &state->saved_actions[i], NULL);
      }
      state->action_saved[i] = false;
    }
  }

  // Clear global write fd and close pipe.
  g_selfpipe_write_fd = -1;
  if (state->pipe_write_fd >= 0) {
    close(state->pipe_write_fd);
    state->pipe_write_fd = -1;
  }
  if (state->pipe_read_fd >= 0) {
    close(state->pipe_read_fd);
    state->pipe_read_fd = -1;
  }

  // Restore original signal mask.
  if (state->sigmask_saved) {
    pthread_sigmask(SIG_SETMASK, &state->saved_sigmask, NULL);
    state->sigmask_saved = false;
  }

  // Clear active mask.
  sigemptyset(&state->active_mask);
}

iree_status_t iree_async_selfpipe_signal_read_pipe(
    iree_async_selfpipe_signal_state_t* state,
    iree_async_signal_dispatch_callback_t callback) {
  // Drain all pending signal bytes from the pipe.
  uint8_t buffer[64];
  while (true) {
    ssize_t bytes_read = read(state->pipe_read_fd, buffer, sizeof(buffer));
    if (bytes_read > 0) {
      // Dispatch each signal byte.
      for (ssize_t i = 0; i < bytes_read; ++i) {
        iree_async_signal_t signal = iree_async_signal_from_posix(buffer[i]);
        if (signal != IREE_ASYNC_SIGNAL_NONE) {
          callback.fn(callback.user_data, signal);
        }
      }
    } else if (bytes_read < 0) {
      if (errno == EAGAIN) {
        // No more data pending. Normal exit path.
        break;
      } else if (errno == EINTR) {
        // Interrupted by signal. Retry.
        continue;
      } else {
        return iree_make_status(IREE_STATUS_INTERNAL,
                                "self-pipe read() failed: %d", errno);
      }
    } else {
      // EOF (bytes_read == 0): write end closed unexpectedly.
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "self-pipe read() returned EOF");
    }
  }

  return iree_ok_status();
}

#endif  // !IREE_PLATFORM_WINDOWS && !IREE_PLATFORM_EMSCRIPTEN
