// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/platform/linux/signal.h"

#if defined(IREE_PLATFORM_LINUX) || defined(IREE_PLATFORM_ANDROID)

#include <errno.h>
#include <pthread.h>
#include <sys/signalfd.h>
#include <unistd.h>

#include "iree/async/util/signal.h"

// Returns true if the sigset is empty (no signals set).
static bool iree_sigset_is_empty(const sigset_t* set) {
  // Check each signal we care about.
  return !sigismember(set, SIGINT) && !sigismember(set, SIGTERM) &&
         !sigismember(set, SIGHUP) && !sigismember(set, SIGQUIT) &&
         !sigismember(set, SIGUSR1) && !sigismember(set, SIGUSR2);
}

iree_status_t iree_async_linux_signal_add_signal(
    iree_async_linux_signal_state_t* state, iree_async_signal_t signal,
    int* out_signal_fd) {
  int posix_signal = iree_async_signal_to_posix(signal);
  if (posix_signal == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid signal type %d", (int)signal);
  }

  // Check if this signal is already in our active mask.
  if (sigismember(&state->active_mask, posix_signal)) {
    // Already handling this signal, just return the existing fd.
    *out_signal_fd = state->signal_fd;
    return iree_ok_status();
  }

  // First signal being added - save the original mask.
  if (!state->sigmask_saved) {
    if (pthread_sigmask(SIG_BLOCK, NULL, &state->saved_sigmask) != 0) {
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "pthread_sigmask(SIG_BLOCK) failed: %d", errno);
    }
    state->sigmask_saved = true;
  }

  // Block this signal so it doesn't invoke the default handler.
  sigset_t block_mask;
  sigemptyset(&block_mask);
  sigaddset(&block_mask, posix_signal);
  if (pthread_sigmask(SIG_BLOCK, &block_mask, NULL) != 0) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "pthread_sigmask(SIG_BLOCK) failed: %d", errno);
  }

  // Add to our active mask.
  sigaddset(&state->active_mask, posix_signal);

  // Create or update the signalfd.
  // signalfd() with an existing fd updates the mask; with -1 creates new fd.
  int fd = signalfd(state->signal_fd, &state->active_mask,
                    SFD_NONBLOCK | SFD_CLOEXEC);
  if (fd < 0) {
    IREE_ATTRIBUTE_UNUSED int error = errno;
    // Rollback: remove from active mask and unblock.
    sigdelset(&state->active_mask, posix_signal);
    pthread_sigmask(SIG_UNBLOCK, &block_mask, NULL);
    return iree_make_status(IREE_STATUS_INTERNAL, "signalfd() failed: %d",
                            error);
  }

  state->signal_fd = fd;
  *out_signal_fd = fd;
  return iree_ok_status();
}

void iree_async_linux_signal_remove_signal(
    iree_async_linux_signal_state_t* state, iree_async_signal_t signal) {
  int posix_signal = iree_async_signal_to_posix(signal);
  if (posix_signal == 0) return;

  // Check if this signal is in our active mask.
  if (!sigismember(&state->active_mask, posix_signal)) {
    return;  // Not handling this signal.
  }

  // Remove from active mask.
  sigdelset(&state->active_mask, posix_signal);

  // Unblock this signal so default handler takes over.
  sigset_t unblock_mask;
  sigemptyset(&unblock_mask);
  sigaddset(&unblock_mask, posix_signal);
  pthread_sigmask(SIG_UNBLOCK, &unblock_mask, NULL);

  // Check if any signals remain.
  if (iree_sigset_is_empty(&state->active_mask)) {
    // No more signals - close the signalfd entirely.
    if (state->signal_fd >= 0) {
      close(state->signal_fd);
      state->signal_fd = -1;
    }
    // Note: We keep sigmask_saved true and saved_sigmask valid in case
    // signals are added again later. Full restore happens in deinitialize().
  } else {
    // Update the signalfd to stop monitoring this signal.
    // Note: signalfd() with updated mask may leave the removed signal's events
    // in the buffer - the dispatch code handles this by ignoring signals
    // without subscribers.
    signalfd(state->signal_fd, &state->active_mask, SFD_NONBLOCK | SFD_CLOEXEC);
  }
}

void iree_async_linux_signal_deinitialize(
    iree_async_linux_signal_state_t* state) {
  // Close signalfd.
  if (state->signal_fd >= 0) {
    close(state->signal_fd);
    state->signal_fd = -1;
  }

  // Restore original signal mask.
  if (state->sigmask_saved) {
    pthread_sigmask(SIG_SETMASK, &state->saved_sigmask, NULL);
    state->sigmask_saved = false;
  }

  // Clear active mask.
  sigemptyset(&state->active_mask);
}

iree_status_t iree_async_linux_signal_read_signalfd(
    iree_async_linux_signal_state_t* state,
    iree_async_signal_dispatch_callback_t callback) {
  struct signalfd_siginfo info;

  // Drain all pending signals from the signalfd. This prevents signals from
  // appearing to lag when multiple arrive between polls.
  while (true) {
    ssize_t bytes_read = read(state->signal_fd, &info, sizeof(info));
    if (bytes_read == sizeof(info)) {
      // Successfully read a signal. Convert to IREE signal type and dispatch.
      iree_async_signal_t signal = iree_async_signal_from_posix(info.ssi_signo);
      if (signal != IREE_ASYNC_SIGNAL_NONE) {
        callback.fn(callback.user_data, signal);
      }
    } else if (bytes_read < 0) {
      if (errno == EAGAIN) {
        // No more signals pending. This is the normal exit path.
        break;
      } else if (errno == EINTR) {
        // Interrupted by signal (ironic). Retry.
        continue;
      } else {
        return iree_make_status(IREE_STATUS_INTERNAL,
                                "signalfd read() failed: %d", errno);
      }
    } else {
      // Partial read should never happen with signalfd.
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "signalfd partial read: %zd bytes", bytes_read);
    }
  }

  return iree_ok_status();
}

#endif  // IREE_PLATFORM_LINUX || IREE_PLATFORM_ANDROID
