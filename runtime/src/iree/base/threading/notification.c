// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/threading/notification.h"

#include <string.h>

#if !IREE_SYNCHRONIZATION_DISABLE_UNSAFE && defined(IREE_PLATFORM_WINDOWS)
#include <intrin.h>
#endif  // IREE_PLATFORM_WINDOWS

//==============================================================================
// Pthreads notification clock configuration
//==============================================================================
// When using pthreads for notifications (macOS, BSD, Linux+TSan, etc.),
// pthread_cond_timedwait() expects an absolute time in a specific clock domain.
// Since iree_time_now() returns CLOCK_MONOTONIC time, we need to handle the
// clock domain mismatch.

#if !defined(IREE_RUNTIME_USE_FUTEX) && !IREE_SYNCHRONIZATION_DISABLE_UNSAFE

#if defined(IREE_PLATFORM_APPLE)
// Apple has pthread_cond_timedwait_relative_np() but NOT
// pthread_condattr_setclock.
#define IREE_NOTIFICATION_USE_RELATIVE_TIMEDWAIT 1
#elif defined(_POSIX_CLOCK_SELECTION) && (_POSIX_CLOCK_SELECTION >= 0)
// Standard POSIX platforms with clock selection support.
#define IREE_NOTIFICATION_USE_CONDATTR_CLOCK 1
#else
// Fallback: convert to relative timeout then to realtime absolute.
#define IREE_NOTIFICATION_USE_RELATIVE_FALLBACK 1
#endif

#endif  // !IREE_RUNTIME_USE_FUTEX && !IREE_SYNCHRONIZATION_DISABLE_UNSAFE

#if defined(NDEBUG)
#define SYNC_ASSERT(x) (void)(x)
#else
#include <assert.h>
#define SYNC_ASSERT(x) assert(x)
#endif  // NDEBUG

//==============================================================================
// Cross-platform processor yield (where supported)
//==============================================================================

#if defined(IREE_COMPILER_MSVC_COMPAT)

static inline void iree_processor_yield(void) {
#if defined(IREE_ARCH_X86_32) || defined(IREE_ARCH_X86_64)
  _mm_pause();
#elif defined(IREE_ARCH_ARM_64)
  __yield();
#else
  // None available; we'll spin hard.
#endif  // IREE_ARCH_*
}

#else

static inline void iree_processor_yield(void) {
#if defined(IREE_ARCH_X86_32) || defined(IREE_ARCH_X86_64)
  asm volatile("pause");
#elif defined(IREE_ARCH_ARM_32) || defined(IREE_ARCH_ARM_64)
  asm volatile("yield");
#elif (defined(IREE_ARCH_RISCV_32) || defined(IREE_ARCH_RISCV_64)) && \
    defined(__riscv_zihintpause)
  asm volatile("pause");
#else
  // None available; we'll spin hard.
#endif  // IREE_ARCH_*
}

#endif  // IREE_COMPILER_*

//==============================================================================
// iree_notification_t
//==============================================================================

#if IREE_SYNCHRONIZATION_DISABLE_UNSAFE

// No-op implementation that is only used when there is guaranteed to be one
// thread at a time touching IREE-related code. It is unsafe to use in any
// situation where either IREE or a user of IREE has multiple threads!

void iree_notification_initialize(iree_notification_t* out_notification) {
  memset(out_notification, 0, sizeof(*out_notification));
}

void iree_notification_deinitialize(iree_notification_t* notification) {}

void iree_notification_post(iree_notification_t* notification, int32_t count) {}

iree_wait_token_t iree_notification_prepare_wait(
    iree_notification_t* notification) {
  return (iree_wait_token_t)0;
}

bool iree_notification_commit_wait(iree_notification_t* notification,
                                   iree_wait_token_t wait_token,
                                   iree_duration_t spin_ns,
                                   iree_time_t deadline_ns) {
  return true;
}

void iree_notification_cancel_wait(iree_notification_t* notification) {}

#elif !defined(IREE_RUNTIME_USE_FUTEX)

// Emulation of a lock-free futex-backed notification using pthreads.
// This is a normal cond-var-like usage with support for our prepare/cancel API
// so that users can still perform their own wait logic.

void iree_notification_initialize(iree_notification_t* out_notification) {
  memset(out_notification, 0, sizeof(*out_notification));
  pthread_mutex_init(&out_notification->mutex, NULL);

#if defined(IREE_NOTIFICATION_USE_CONDATTR_CLOCK)
  // Configure condition variable to use CLOCK_MONOTONIC so that
  // pthread_cond_timedwait() accepts monotonic absolute times directly.
  pthread_condattr_t cond_attr;
  pthread_condattr_init(&cond_attr);
  pthread_condattr_setclock(&cond_attr, CLOCK_MONOTONIC);
  pthread_cond_init(&out_notification->cond, &cond_attr);
  pthread_condattr_destroy(&cond_attr);
#else
  pthread_cond_init(&out_notification->cond, NULL);
#endif  // IREE_NOTIFICATION_USE_CONDATTR_CLOCK
}

void iree_notification_deinitialize(iree_notification_t* notification) {
  // Assert no more waiters (callers must tear down waiters first).
  pthread_mutex_lock(&notification->mutex);
  SYNC_ASSERT(notification->waiters == 0);
  pthread_cond_destroy(&notification->cond);
  pthread_mutex_unlock(&notification->mutex);
  pthread_mutex_destroy(&notification->mutex);
}

void iree_notification_post(iree_notification_t* notification, int32_t count) {
  pthread_mutex_lock(&notification->mutex);
  ++notification->epoch;
  if (notification->waiters > 0) {
    // NOTE: we only do the signal if we have waiters - this avoids a syscall
    // in cases where no one is actively listening.
    if (count == IREE_ALL_WAITERS) {
      pthread_cond_broadcast(&notification->cond);
    } else {
      for (int32_t i = 0; i < count; ++i) {
        pthread_cond_signal(&notification->cond);
      }
    }
  }
  pthread_mutex_unlock(&notification->mutex);
}

iree_wait_token_t iree_notification_prepare_wait(
    iree_notification_t* notification) {
  pthread_mutex_lock(&notification->mutex);
  iree_wait_token_t epoch = notification->epoch;
  ++notification->waiters;
  pthread_mutex_unlock(&notification->mutex);
  return epoch;
}

bool iree_notification_commit_wait(iree_notification_t* notification,
                                   iree_wait_token_t wait_token,
                                   iree_duration_t spin_ns,
                                   iree_time_t deadline_ns) {
  pthread_mutex_lock(&notification->mutex);

  bool result = true;
  while (notification->epoch == wait_token) {
    int ret;

    if (deadline_ns == IREE_TIME_INFINITE_FUTURE) {
      // No timeout - wait forever.
      ret = pthread_cond_wait(&notification->cond, &notification->mutex);
    } else if (deadline_ns == IREE_TIME_INFINITE_PAST) {
      // Already past deadline - immediate timeout.
      ret = ETIMEDOUT;
    } else {
#if defined(IREE_NOTIFICATION_USE_CONDATTR_CLOCK)
      // Condition variable configured with CLOCK_MONOTONIC - use absolute time
      // from iree_time_now() directly.
      struct timespec abs_ts = {
          .tv_sec = (time_t)(deadline_ns / 1000000000ull),
          .tv_nsec = (long)(deadline_ns % 1000000000ull),
      };
      ret = pthread_cond_timedwait(&notification->cond, &notification->mutex,
                                   &abs_ts);

#elif defined(IREE_NOTIFICATION_USE_RELATIVE_TIMEDWAIT)
      // Apple: use relative timeout API (pthread_cond_timedwait_relative_np).
      iree_duration_t timeout_ns =
          iree_absolute_deadline_to_timeout_ns(deadline_ns);
      if (timeout_ns <= 0) {
        ret = ETIMEDOUT;
      } else {
        struct timespec rel_ts = {
            .tv_sec = (time_t)(timeout_ns / 1000000000ull),
            .tv_nsec = (long)(timeout_ns % 1000000000ull),
        };
        ret = pthread_cond_timedwait_relative_np(&notification->cond,
                                                 &notification->mutex, &rel_ts);
      }

#else   // IREE_NOTIFICATION_USE_RELATIVE_FALLBACK
      // Convert monotonic deadline to realtime absolute.
      // This has some clock drift but is acceptable for timeout precision.
      iree_duration_t timeout_ns =
          iree_absolute_deadline_to_timeout_ns(deadline_ns);
      if (timeout_ns <= 0) {
        ret = ETIMEDOUT;
      } else {
        struct timespec now_realtime;
        clock_gettime(CLOCK_REALTIME, &now_realtime);
        int64_t deadline_realtime_ns =
            (int64_t)now_realtime.tv_sec * 1000000000ll + now_realtime.tv_nsec +
            timeout_ns;
        struct timespec abs_ts = {
            .tv_sec = (time_t)(deadline_realtime_ns / 1000000000ll),
            .tv_nsec = (long)(deadline_realtime_ns % 1000000000ll),
        };
        ret = pthread_cond_timedwait(&notification->cond, &notification->mutex,
                                     &abs_ts);
      }
#endif  // IREE_NOTIFICATION_USE_*
    }

    if (ret != 0) {
      // Wait failed (timeout/etc); cancel the wait.
      result = false;
      break;
    }
  }

  // Remove us from the waiter list.
  SYNC_ASSERT(notification->waiters > 0);
  --notification->waiters;

  pthread_mutex_unlock(&notification->mutex);

  return result;
}

void iree_notification_cancel_wait(iree_notification_t* notification) {
  pthread_mutex_lock(&notification->mutex);
  SYNC_ASSERT(notification->waiters > 0);
  --notification->waiters;
  pthread_mutex_unlock(&notification->mutex);
}

#else

// The 64-bit value used to atomically read-modify-write (RMW) the state is
// split in two and treated as independent 32-bit ints:
//
//  MSB (63)                           32                               LSB (0)
// +-------------------------------------+-------------------------------------+
// |            epoch/notification count |                        waiter count |
// +-------------------------------------+-------------------------------------+
//
// We use the epoch to wait/wake the futex (which is 32-bits), and as such when
// we pass the value address to the futex APIs we need to ensure we are only
// passing the most significant 32-bit value regardless of endianness.
#if defined(IREE_ENDIANNESS_LITTLE)
#define IREE_NOTIFICATION_EPOCH_OFFSET (/*words=*/1)
#else
#define IREE_NOTIFICATION_EPOCH_OFFSET (/*words=*/0)
#endif  // IREE_ENDIANNESS_*
#define iree_notification_epoch_address(notification) \
  ((iree_atomic_int32_t*)(&(notification)->value) +   \
   IREE_NOTIFICATION_EPOCH_OFFSET)
#define IREE_NOTIFICATION_WAITER_INC 0x0000000000000001ull
#define IREE_NOTIFICATION_WAITER_DEC 0xFFFFFFFFFFFFFFFFull
#define IREE_NOTIFICATION_WAITER_MASK 0x00000000FFFFFFFFull
#define IREE_NOTIFICATION_EPOCH_SHIFT 32
#define IREE_NOTIFICATION_EPOCH_INC \
  (0x00000001ull << IREE_NOTIFICATION_EPOCH_SHIFT)

void iree_notification_initialize(iree_notification_t* out_notification) {
  memset(out_notification, 0, sizeof(*out_notification));
}

void iree_notification_deinitialize(iree_notification_t* notification) {
  // Assert no more waiters (callers must tear down waiters first).
  SYNC_ASSERT(
      (iree_atomic_load(&notification->value, iree_memory_order_acquire) &
       IREE_NOTIFICATION_WAITER_MASK) == 0);
}

void iree_notification_post(iree_notification_t* notification, int32_t count) {
  uint64_t previous_value =
      iree_atomic_fetch_add(&notification->value, IREE_NOTIFICATION_EPOCH_INC,
                            iree_memory_order_acq_rel);
  // Ensure we have at least one waiter; wake up to |count| of them.
  if (IREE_UNLIKELY(previous_value & IREE_NOTIFICATION_WAITER_MASK)) {
    iree_futex_wake(iree_notification_epoch_address(notification), count);
  }
}

iree_wait_token_t iree_notification_prepare_wait(
    iree_notification_t* notification) {
  uint64_t previous_value =
      iree_atomic_fetch_add(&notification->value, IREE_NOTIFICATION_WAITER_INC,
                            iree_memory_order_acq_rel);
  return (iree_wait_token_t)(previous_value >> IREE_NOTIFICATION_EPOCH_SHIFT);
}

typedef enum iree_notification_result_e {
  IREE_NOTIFICATION_RESULT_UNRESOLVED = 0,
  IREE_NOTIFICATION_RESULT_RESOLVED,
  IREE_NOTIFICATION_RESULT_REJECTED,
} iree_notification_result_t;

static iree_notification_result_t iree_notification_test_wait_condition(
    iree_notification_t* notification, iree_wait_token_t wait_token) {
  return (iree_atomic_load(&notification->value, iree_memory_order_acquire) >>
          IREE_NOTIFICATION_EPOCH_SHIFT) != wait_token
             ? IREE_NOTIFICATION_RESULT_RESOLVED
             : IREE_NOTIFICATION_RESULT_UNRESOLVED;
}

bool iree_notification_commit_wait(iree_notification_t* notification,
                                   iree_wait_token_t wait_token,
                                   iree_duration_t spin_ns,
                                   iree_time_t deadline_ns) {
  // Quick check to see if the wait has already succeeded.
  iree_notification_result_t result =
      iree_notification_test_wait_condition(notification, wait_token);

  // If not already reached and spinning is enabled then we'll try that first.
  if (result == IREE_NOTIFICATION_RESULT_UNRESOLVED &&
      spin_ns != IREE_DURATION_ZERO) {
    const iree_time_t spin_deadline_ns = iree_time_now() + spin_ns;
    IREE_TRACE_ZONE_BEGIN_NAMED(z0, "iree_notification_commit_wait_spin");
    do {
      iree_processor_yield();
      result = iree_notification_test_wait_condition(notification, wait_token);
    } while (result == IREE_NOTIFICATION_RESULT_UNRESOLVED &&
             iree_time_now() < spin_deadline_ns);
    IREE_TRACE_ZONE_END(z0);
  }

  // If spinning failed let the kernel do what it does ... okish at.
  if (deadline_ns != IREE_TIME_INFINITE_PAST) {
    while (result == IREE_NOTIFICATION_RESULT_UNRESOLVED) {
      iree_status_code_t status_code =
          iree_futex_wait(iree_notification_epoch_address(notification),
                          wait_token, deadline_ns);
      if (status_code != IREE_STATUS_OK) {
        result = IREE_NOTIFICATION_RESULT_REJECTED;
        break;
      }
      result = iree_notification_test_wait_condition(notification, wait_token);
    }
  }

  uint64_t previous_value =
      iree_atomic_fetch_add(&notification->value, IREE_NOTIFICATION_WAITER_DEC,
                            iree_memory_order_acq_rel);
  SYNC_ASSERT((previous_value & IREE_NOTIFICATION_WAITER_MASK) != 0);

  return result == IREE_NOTIFICATION_RESULT_RESOLVED;
}

void iree_notification_cancel_wait(iree_notification_t* notification) {
  uint64_t previous_value =
      iree_atomic_fetch_add(&notification->value, IREE_NOTIFICATION_WAITER_DEC,
                            iree_memory_order_acq_rel);
  SYNC_ASSERT((previous_value & IREE_NOTIFICATION_WAITER_MASK) != 0);
}

#endif  // DISABLED / HAS_FUTEX

bool iree_notification_await(iree_notification_t* notification,
                             iree_condition_fn_t condition_fn,
                             void* condition_arg, iree_timeout_t timeout) {
  if (IREE_LIKELY(condition_fn(condition_arg))) {
    // Fast-path with condition already met.
    return true;
  }

  // If a (silly) query then bail immediately after our first condition check.
  if (iree_timeout_is_immediate(timeout)) return false;
  const iree_time_t deadline_ns = iree_timeout_as_deadline_ns(timeout);

  // Slow-path: try-wait until the condition is met.
  while (true) {
    iree_wait_token_t wait_token = iree_notification_prepare_wait(notification);
    if (condition_fn(condition_arg)) {
      // Condition is now met; no need to wait on the futex.
      iree_notification_cancel_wait(notification);
      return true;
    } else {
      if (!iree_notification_commit_wait(notification, wait_token,
                                         /*spin_ns=*/IREE_DURATION_ZERO,
                                         deadline_ns)) {
        // Wait hit the deadline before we hit the condition.
        return false;
      }
    }
  }

  return true;
}
