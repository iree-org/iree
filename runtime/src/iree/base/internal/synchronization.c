// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/synchronization.h"

#include <assert.h>
#include <string.h>

#if IREE_SYNCHRONIZATION_DISABLE_UNSAFE

// Disabled.

#elif defined(IREE_PLATFORM_WINDOWS)

#include <intrin.h>

#elif defined(IREE_PLATFORM_EMSCRIPTEN)

#include <emscripten/threading.h>
#include <errno.h>

#elif defined(IREE_PLATFORM_ANDROID) || defined(IREE_PLATFORM_LINUX)

#include <errno.h>
#include <linux/futex.h>
#include <sys/syscall.h>
#include <unistd.h>

#if defined(IREE_ARCH_RISCV_32) && defined(__NR_futex_time64) && \
    !defined(__NR_futex)
// RV32 uses 64-bit times by default (unlike other 32-bit archs).
#define __NR_futex __NR_futex_time64
#endif  // IREE_ARCH_RISCV_32

// Oh Android...
#ifndef SYS_futex
#define SYS_futex __NR_futex
#endif  // !SYS_futex
#ifndef FUTEX_PRIVATE_FLAG
#define FUTEX_PRIVATE_FLAG 128
#endif  // !FUTEX_PRIVATE_FLAG

#endif  // IREE_PLATFORM_*

#if defined(NDEBUG)
#define SYNC_ASSERT(x) (void)(x)
#else
#define SYNC_ASSERT(x) assert(x)
#endif  // NDEBUG

// Tag functions in .c files with this to indicate that thread safety analysis
// warnings should not show. This is useful on our implementation functions as
// clang cannot reason about lock-free magic.
#define IREE_DISABLE_THREAD_SAFETY_ANALYSIS \
  IREE_THREAD_ANNOTATION_ATTRIBUTE(no_thread_safety_analysis)

//==============================================================================
// Cross-platform processor yield (where supported)
//==============================================================================

#if defined(IREE_COMPILER_MSVC)

// MSVC uses architecture-specific intrinsics.

IREE_ATTRIBUTE_UNUSED static inline void iree_processor_yield(void) {
#if defined(IREE_ARCH_X86_32) || defined(IREE_ARCH_X86_64)
  // https://docs.microsoft.com/en-us/cpp/intrinsics/x86-intrinsics-list
  _mm_pause();
#elif defined(IREE_ARCH_ARM_64)
  // https://docs.microsoft.com/en-us/cpp/intrinsics/arm64-intrinsics
  __yield();
#else
  // None available; we'll spin hard.
#endif  // IREE_ARCH_*
}

#else

// Clang/GCC and compatibles use architecture-specific inline assembly.

IREE_ATTRIBUTE_UNUSED static inline void iree_processor_yield(void) {
#if defined(IREE_ARCH_X86_32) || defined(IREE_ARCH_X86_64)
  asm volatile("pause");
#elif defined(IREE_ARCH_ARM_32) || defined(IREE_ARCH_ARM_64)
  asm volatile("yield");
#else
  // None available; we'll spin hard.
#endif  // IREE_ARCH_*
}

#endif  // IREE_COMPILER_*

//==============================================================================
// Cross-platform futex mappings (where supported)
//==============================================================================

#if defined(IREE_PLATFORM_HAS_FUTEX)

// Waits in the OS for the value at the specified |address| to change.
// If the contents of |address| do not match |expected_value| the wait will
// fail and return IREE_STATUS_UNAVAILABLE and should be retried.
//
// |deadline_ns| can be either IREE_TIME_INFINITE_FUTURE to wait forever or an
// absolute time to wait until prior to returning early with
// IREE_STATUS_DEADLINE_EXCEEDED.
static inline iree_status_code_t iree_futex_wait(void* address,
                                                 uint32_t expected_value,
                                                 iree_time_t deadline_ns);

// Wakes at most |count| threads waiting for the |address| to change.
// Use IREE_ALL_WAITERS to wake all waiters. Which waiters are woken is
// undefined and it is not guaranteed that higher priority waiters will be woken
// over lower priority waiters.
static inline void iree_futex_wake(void* address, int32_t count);

#if defined(IREE_PLATFORM_EMSCRIPTEN)

static inline iree_status_code_t iree_futex_wait(void* address,
                                                 uint32_t expected_value,
                                                 iree_time_t deadline_ns) {
  uint32_t timeout_ms = iree_absolute_deadline_to_timeout_ms(deadline_ns);
  int rc = emscripten_futex_wait(address, expected_value, (double)timeout_ms);
  switch (rc) {
    default:
      return IREE_STATUS_OK;
    case -ETIMEDOUT:
      return IREE_STATUS_DEADLINE_EXCEEDED;
    case -EWOULDBLOCK:
      return IREE_STATUS_UNAVAILABLE;
  }
}

static inline void iree_futex_wake(void* address, int32_t count) {
  emscripten_futex_wake(address, count);
}

#elif defined(IREE_PLATFORM_WINDOWS)

#pragma comment(lib, "Synchronization.lib")

static inline iree_status_code_t iree_futex_wait(void* address,
                                                 uint32_t expected_value,
                                                 iree_time_t deadline_ns) {
  uint32_t timeout_ms = iree_absolute_deadline_to_timeout_ms(deadline_ns);
  if (IREE_LIKELY(WaitOnAddress(address, &expected_value,
                                sizeof(expected_value), timeout_ms) == TRUE)) {
    return IREE_STATUS_OK;
  }
  if (GetLastError() == ERROR_TIMEOUT) {
    return IREE_STATUS_DEADLINE_EXCEEDED;
  }
  return IREE_STATUS_UNAVAILABLE;
}

static inline void iree_futex_wake(void* address, int32_t count) {
  if (count == INT32_MAX) {
    WakeByAddressAll(address);
    return;
  }
  for (; count > 0; --count) {
    WakeByAddressSingle(address);
  }
}

#elif defined(IREE_PLATFORM_ANDROID) || defined(IREE_PLATFORM_LINUX)

static inline iree_status_code_t iree_futex_wait(void* address,
                                                 uint32_t expected_value,
                                                 iree_time_t deadline_ns) {
  uint32_t timeout_ms = iree_absolute_deadline_to_timeout_ms(deadline_ns);
  struct timespec timeout = {
      .tv_sec = timeout_ms / 1000,
      .tv_nsec = (timeout_ms % 1000) * 1000000,
  };
  int rc = syscall(
      SYS_futex, address, FUTEX_WAIT | FUTEX_PRIVATE_FLAG, expected_value,
      timeout_ms == IREE_INFINITE_TIMEOUT_MS ? NULL : &timeout, NULL, 0);
  if (IREE_LIKELY(rc == 0) || errno == EAGAIN || errno == EINTR) {
    return IREE_STATUS_OK;
  } else if (errno == ETIMEDOUT) {
    return IREE_STATUS_DEADLINE_EXCEEDED;
  }
  return IREE_STATUS_UNAVAILABLE;
}

static inline void iree_futex_wake(void* address, int32_t count) {
  syscall(SYS_futex, address, FUTEX_WAKE | FUTEX_PRIVATE_FLAG, count, NULL,
          NULL, 0);
}

#endif  // IREE_PLATFORM_*

#endif  // IREE_PLATFORM_HAS_FUTEX

//==============================================================================
// iree_mutex_t
//==============================================================================

#if IREE_SYNCHRONIZATION_DISABLE_UNSAFE

#define iree_mutex_impl_initialize(mutex)
#define iree_mutex_impl_deinitialize(mutex)
#define iree_mutex_impl_lock(mutex)
#define iree_mutex_impl_try_lock(mutex) true
#define iree_mutex_impl_unlock(mutex)

#elif defined(IREE_PLATFORM_WINDOWS) && defined(IREE_MUTEX_USE_WIN32_SRW)

// Win32 Slim Reader/Writer (SRW) Lock (same as std::mutex)
#define iree_mutex_impl_initialize(mutex) InitializeSRWLock(&(mutex)->value)
#define iree_mutex_impl_deinitialize(mutex)
#define iree_mutex_impl_lock(mutex) AcquireSRWLockExclusive(&(mutex)->value)
#define iree_mutex_impl_try_lock(mutex) \
  (TryAcquireSRWLockExclusive(&(mutex)->value) == TRUE)
#define iree_mutex_impl_unlock(mutex) ReleaseSRWLockExclusive(&(mutex)->value)

#elif defined(IREE_PLATFORM_WINDOWS)

// Win32 CRITICAL_SECTION
#define IREE_WIN32_CRITICAL_SECTION_FLAG_DYNAMIC_SPIN 0x02000000
#define iree_mutex_impl_initialize(mutex)            \
  InitializeCriticalSectionEx(&(mutex)->value, 4000, \
                              IREE_WIN32_CRITICAL_SECTION_FLAG_DYNAMIC_SPIN)
#define iree_mutex_impl_deinitialize(mutex) \
  DeleteCriticalSection(&(mutex)->value)
#define iree_mutex_impl_lock(mutex) EnterCriticalSection(&(mutex)->value)
#define iree_mutex_impl_try_lock(mutex) \
  (TryEnterCriticalSection(&(mutex)->value) == TRUE)
#define iree_mutex_impl_unlock(mutex) LeaveCriticalSection(&(mutex)->value)

#else

// pthreads pthread_mutex_t
#define iree_mutex_impl_initialize(mutex) \
  pthread_mutex_init(&(mutex)->value, NULL)
#define iree_mutex_impl_deinitialize(mutex) \
  pthread_mutex_destroy(&(mutex)->value)
#define iree_mutex_impl_lock(mutex) pthread_mutex_lock(&(mutex)->value)
#define iree_mutex_impl_try_lock(mutex) \
  (pthread_mutex_trylock(&(mutex)->value) == 0)
#define iree_mutex_impl_unlock(mutex) pthread_mutex_unlock(&(mutex)->value)

#endif  // IREE_PLATFORM_*

#if (IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_SLOW_LOCKS)

// NOTE: the tracy mutex tracing code takes locks itself (which makes it slower
// and may cause deadlocks).

void iree_mutex_initialize_impl(const iree_tracing_location_t* src_loc,
                                iree_mutex_t* out_mutex) {
  memset(out_mutex, 0, sizeof(*out_mutex));
  iree_tracing_mutex_announce(src_loc, &out_mutex->lock_id);
  iree_mutex_impl_initialize(out_mutex);
}

void iree_mutex_deinitialize(iree_mutex_t* mutex) {
  iree_mutex_impl_deinitialize(mutex);
  iree_tracing_mutex_terminate(mutex->lock_id);
  memset(mutex, 0, sizeof(*mutex));
}

void iree_mutex_lock(iree_mutex_t* mutex) IREE_DISABLE_THREAD_SAFETY_ANALYSIS {
  iree_tracing_mutex_before_lock(mutex->lock_id);
  iree_mutex_impl_lock(mutex);
  iree_tracing_mutex_after_lock(mutex->lock_id);
}

bool iree_mutex_try_lock(iree_mutex_t* mutex)
    IREE_DISABLE_THREAD_SAFETY_ANALYSIS {
  bool was_acquired = iree_mutex_impl_try_lock(mutex);
  iree_tracing_mutex_after_try_lock(mutex->lock_id, was_acquired);
  return was_acquired;
}

void iree_mutex_unlock(iree_mutex_t* mutex)
    IREE_DISABLE_THREAD_SAFETY_ANALYSIS {
  iree_mutex_impl_unlock(mutex);
  iree_tracing_mutex_after_unlock(mutex->lock_id);
}

#else

void iree_mutex_initialize(iree_mutex_t* out_mutex) {
  memset(out_mutex, 0, sizeof(*out_mutex));
  iree_mutex_impl_initialize(out_mutex);
}

void iree_mutex_deinitialize(iree_mutex_t* mutex) {
  iree_mutex_impl_deinitialize(mutex);
  memset(mutex, 0, sizeof(*mutex));
}

void iree_mutex_lock(iree_mutex_t* mutex) IREE_DISABLE_THREAD_SAFETY_ANALYSIS {
  iree_mutex_impl_lock(mutex);
}

bool iree_mutex_try_lock(iree_mutex_t* mutex)
    IREE_DISABLE_THREAD_SAFETY_ANALYSIS {
  return iree_mutex_impl_try_lock(mutex);
}

void iree_mutex_unlock(iree_mutex_t* mutex)
    IREE_DISABLE_THREAD_SAFETY_ANALYSIS {
  iree_mutex_impl_unlock(mutex);
}

#endif  // IREE_TRACING_FEATURE_SLOW_LOCKS

//==============================================================================
// iree_slim_mutex_t
//==============================================================================

#if (IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_FAST_LOCKS)

// Turn fast locks into slow locks.
// This lets us just reuse that code at the cost of obscuring our lock
// performance; but at the time you are recording 2+ tracy messages per lock use
// there's not much interesting to gain from that level of granularity anyway.
// If these start showing up in traces it means that the higher-level algorithm
// is taking too many locks and not that this taking time is the core issue.

void iree_slim_mutex_initialize_impl(const iree_tracing_location_t* src_loc,
                                     iree_slim_mutex_t* out_mutex) {
  iree_mutex_initialize_impl(src_loc, &out_mutex->impl);
}

void iree_slim_mutex_deinitialize(iree_slim_mutex_t* mutex) {
  iree_mutex_deinitialize(&mutex->impl);
}

void iree_slim_mutex_lock(iree_slim_mutex_t* mutex)
    IREE_DISABLE_THREAD_SAFETY_ANALYSIS {
  iree_mutex_lock(&mutex->impl);
}

bool iree_slim_mutex_try_lock(iree_slim_mutex_t* mutex)
    IREE_DISABLE_THREAD_SAFETY_ANALYSIS {
  return iree_mutex_try_lock(&mutex->impl);
}

void iree_slim_mutex_unlock(iree_slim_mutex_t* mutex)
    IREE_DISABLE_THREAD_SAFETY_ANALYSIS {
  iree_mutex_unlock(&mutex->impl);
}

#else

#if IREE_SYNCHRONIZATION_DISABLE_UNSAFE

void iree_slim_mutex_initialize(iree_slim_mutex_t* out_mutex) {}

void iree_slim_mutex_deinitialize(iree_slim_mutex_t* mutex) {}

void iree_slim_mutex_lock(iree_slim_mutex_t* mutex)
    IREE_DISABLE_THREAD_SAFETY_ANALYSIS {}

bool iree_slim_mutex_try_lock(iree_slim_mutex_t* mutex)
    IREE_DISABLE_THREAD_SAFETY_ANALYSIS {
  return iree_mutex_try_lock((iree_mutex_t*)&mutex->reserved);
}

void iree_slim_mutex_unlock(iree_slim_mutex_t* mutex)
    IREE_DISABLE_THREAD_SAFETY_ANALYSIS {}

#elif defined(IREE_PLATFORM_APPLE)

void iree_slim_mutex_initialize(iree_slim_mutex_t* out_mutex) {
  out_mutex->value = OS_UNFAIR_LOCK_INIT;
}

void iree_slim_mutex_deinitialize(iree_slim_mutex_t* mutex) {
  os_unfair_lock_assert_not_owner(&mutex->value);
}

void iree_slim_mutex_lock(iree_slim_mutex_t* mutex)
    IREE_DISABLE_THREAD_SAFETY_ANALYSIS {
  os_unfair_lock_lock(&mutex->value);
}

bool iree_slim_mutex_try_lock(iree_slim_mutex_t* mutex)
    IREE_DISABLE_THREAD_SAFETY_ANALYSIS {
  return os_unfair_lock_trylock(&mutex->value);
}

void iree_slim_mutex_unlock(iree_slim_mutex_t* mutex)
    IREE_DISABLE_THREAD_SAFETY_ANALYSIS {
  os_unfair_lock_unlock(&mutex->value);
}

#elif defined(IREE_PLATFORM_WINDOWS) && defined(IREE_MUTEX_USE_WIN32_SRW)

// The SRW on Windows is pointer-sized and slightly better than what we emulate
// with the futex so let's just use that.

void iree_slim_mutex_initialize(iree_slim_mutex_t* out_mutex) {
  iree_mutex_impl_initialize(out_mutex);
}

void iree_slim_mutex_deinitialize(iree_slim_mutex_t* mutex) {
  iree_mutex_impl_deinitialize(mutex);
}

void iree_slim_mutex_lock(iree_slim_mutex_t* mutex)
    IREE_DISABLE_THREAD_SAFETY_ANALYSIS {
  iree_mutex_impl_lock(mutex);
}

bool iree_slim_mutex_try_lock(iree_slim_mutex_t* mutex)
    IREE_DISABLE_THREAD_SAFETY_ANALYSIS {
  return iree_mutex_impl_try_lock(mutex);
}

void iree_slim_mutex_unlock(iree_slim_mutex_t* mutex)
    IREE_DISABLE_THREAD_SAFETY_ANALYSIS {
  iree_mutex_impl_unlock(mutex);
}

#elif defined(IREE_PLATFORM_HAS_FUTEX)

// This implementation is a combo of several sources:
//
// Basics of Futexes by Eli Bendersky:
// https://eli.thegreenplace.net/2018/basics-of-futexes/
//
// Futex based locks for C11’s generic atomics by Jens Gustedt:
// https://hal.inria.fr/hal-01236734/document
//
// Mutexes and Condition Variables using Futexes:
// http://locklessinc.com/articles/mutex_cv_futex/
//
// The high bit of the atomic value indicates whether the lock is held; each
// thread tries to transition the bit from 0->1 to acquire the lock and 1->0 to
// release it. The lower bits of the value are whether there are any interested
// waiters. We track these waiters so that we know when we can avoid performing
// the futex wake syscall.

#define iree_slim_mutex_value(value) (0x80000000u | (value))
#define iree_slim_mutex_is_locked(value) (0x80000000u & (value))

void iree_slim_mutex_initialize(iree_slim_mutex_t* out_mutex) {
  memset(out_mutex, 0, sizeof(*out_mutex));
}

void iree_slim_mutex_deinitialize(iree_slim_mutex_t* mutex) {
  // Assert unlocked (callers must ensure the mutex is no longer in use).
  SYNC_ASSERT(
      iree_atomic_load_int32(&mutex->value, iree_memory_order_acquire) == 0);
}

// Helper to perform a compare_exchange operation on mutex->value, internally
// used by iree_slim_mutex_try_lock and iree_slim_mutex_lock.
static bool iree_slim_mutex_try_lock_compare_exchange(
    iree_slim_mutex_t* mutex, int32_t* expected,
    int32_t desired) IREE_DISABLE_THREAD_SAFETY_ANALYSIS {
  // Refer to the iree_slim_mutex_t struct comment, "Notes on atomics",
  // particularly regarding why the comparison-success case has 'acquire' order
  // and not the perhaps more intuitive 'acq_rel'.
  // The comparison-failure case has 'relaxed' order because in that case,
  // we don't need ordering with other memory operations. Some callers won't use
  // the 'expected' value loaded in that case at all, and some other callers
  // will use it but won't rely on ordering w.r.t other memory operations.
  // The choice of the 'weak' form of compare_exchange is because callers care
  // more about efficiency in the uncontended case than we care about avoiding
  // spurious failure. Also, some callers are calling this in a loop, where they
  // would want the weak form anyway.
  return iree_atomic_compare_exchange_weak_int32(
      &mutex->value, expected, desired, iree_memory_order_acquire,
      iree_memory_order_relaxed);
}

void iree_slim_mutex_lock(iree_slim_mutex_t* mutex)
    IREE_DISABLE_THREAD_SAFETY_ANALYSIS {
  // Refer to the iree_slim_mutex_t struct comment, "Notes on atomics".
  // Try first to acquire the lock from an unlocked state.
  int32_t value = 0;
  if (iree_slim_mutex_try_lock_compare_exchange(mutex, &value,
                                                iree_slim_mutex_value(1))) {
    // Successfully took the lock and there were no other waiters.
    return;
  }

  // Increment the count bits to indicate that we want the lock and are willing
  // to wait for it to be available. Note that between the CAS above and this
  // the lock could have been made available and we want to ensure we don't
  // change the lock bit.
  // This uses relaxed order because this is an internal intermediate step and
  // we only need atomicity here.
  value =
      iree_atomic_fetch_add_int32(&mutex->value, 1, iree_memory_order_relaxed) +
      1;

  while (true) {
    // While the lock is available: try to acquire it for this thread.
    while (!iree_slim_mutex_is_locked(value)) {
      // See the above 'Explanation of memory orders' comment.
      if (iree_slim_mutex_try_lock_compare_exchange(
              mutex, &value, iree_slim_mutex_value(value))) {
        // Successfully took the lock.
        return;
      }

      // Spin a small amount to give us a tiny chance of falling through to the
      // wait. We can tune this value based on likely contention, however 10-60
      // is the recommended value and we should keep it in that order of
      // magnitude. A way to think of this is "how many spins would we have to
      // do to equal one call to iree_futex_wait" - if it's faster just to do
      // a futex wait then we shouldn't be spinning!
      // TODO(benvanik): measure on real workload on ARM; maybe remove entirely.
      int spin_count = 100;
      for (int i = 0; i < spin_count && iree_slim_mutex_is_locked(value); ++i) {
        iree_processor_yield();
        value =
            iree_atomic_load_int32(&mutex->value, iree_memory_order_relaxed);
      }
    }

    // While the lock is unavailable: wait for it to become available.
    while (iree_slim_mutex_is_locked(value)) {
      // NOTE: we don't care about wait failure here as we are going to loop
      // and check again anyway.
      iree_futex_wait(&mutex->value, value, IREE_TIME_INFINITE_FUTURE);
      value = iree_atomic_load_int32(&mutex->value, iree_memory_order_relaxed);
    }
  }
}

bool iree_slim_mutex_try_lock(iree_slim_mutex_t* mutex)
    IREE_DISABLE_THREAD_SAFETY_ANALYSIS {
  // Refer to the iree_slim_mutex_t struct comment, "Notes on atomics".
  // Attempt to acquire the lock from an unlocked state.
  int32_t value = 0;
  return iree_slim_mutex_try_lock_compare_exchange(mutex, &value,
                                                   iree_slim_mutex_value(1));
}

void iree_slim_mutex_unlock(iree_slim_mutex_t* mutex)
    IREE_DISABLE_THREAD_SAFETY_ANALYSIS {
  // Refer to the iree_slim_mutex_t struct comment, "Notes on atomics".
  // Transition 1->0 (unlocking with no waiters) or 2->1 (with waiters).
  if (iree_atomic_fetch_sub_int32(&mutex->value, iree_slim_mutex_value(1),
                                  iree_memory_order_release) !=
      iree_slim_mutex_value(1)) {
    // One (or more) waiters; wake a single one to avoid a thundering herd of
    // multiple threads all waking and trying to grab the lock (as only one will
    // win).
    //
    // Note that futexes (futeces? futices? futii?) are unfair and what thread
    // gets woken is undefined (not FIFO on waiters).
    iree_futex_wake(&mutex->value, 1);
  }
}

#else

// Pass-through to iree_mutex_t as a fallback for platforms without a futex we
// can use to implement a slim lock. Note that since we are reusing iree_mutex_t
// when tracing all slim mutexes will be traced along with the fat mutexes.

void iree_slim_mutex_initialize(iree_slim_mutex_t* out_mutex) {
  iree_mutex_initialize(&out_mutex->impl);
}

void iree_slim_mutex_deinitialize(iree_slim_mutex_t* mutex) {
  iree_mutex_deinitialize(&mutex->impl);
}

void iree_slim_mutex_lock(iree_slim_mutex_t* mutex)
    IREE_DISABLE_THREAD_SAFETY_ANALYSIS {
  iree_mutex_lock(&mutex->impl);
}

bool iree_slim_mutex_try_lock(iree_slim_mutex_t* mutex)
    IREE_DISABLE_THREAD_SAFETY_ANALYSIS {
  return iree_mutex_try_lock(&mutex->impl);
}

void iree_slim_mutex_unlock(iree_slim_mutex_t* mutex)
    IREE_DISABLE_THREAD_SAFETY_ANALYSIS {
  iree_mutex_unlock(&mutex->impl);
}

#endif  // IREE_PLATFORM_*

#endif  //  IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_SLOW_LOCKS

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

#elif !defined(IREE_PLATFORM_HAS_FUTEX)

// Emulation of a lock-free futex-backed notification using pthreads.
// This is a normal cond-var-like usage with support for our prepare/cancel API
// so that users can still perform their own wait logic.

void iree_notification_initialize(iree_notification_t* out_notification) {
  memset(out_notification, 0, sizeof(*out_notification));
  pthread_mutex_init(&out_notification->mutex, NULL);
  pthread_cond_init(&out_notification->cond, NULL);
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
  struct timespec abs_ts = {
      .tv_sec = (time_t)(deadline_ns / 1000000000ull),
      .tv_nsec = (long)(deadline_ns % 1000000000ull),
  };

  pthread_mutex_lock(&notification->mutex);

  // Spin until notified and the epoch increments from what we captured during
  // iree_notification_prepare_wait.
  bool result = true;
  while (notification->epoch == wait_token) {
    int ret = pthread_cond_timedwait(&notification->cond, &notification->mutex,
                                     &abs_ts);
    if (ret != 0) {
      // Wait failed (timeout/etc); cancel the wait.
      // This may happen in spurious wakes but that's fine - the caller is
      // designed to handle looping again and may want the chance to do some
      // bookkeeping while it has the thread.
      result = false;
      break;
    }
  }

  // Remove us from the waiter list - the caller will need to reacquire a wait
  // token if it wants to wait again.
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
//
// We use signed addition on the full 64-bit value to increment/decrement the
// waiter count. This means that an add of -1ll will decrement the waiter count
// and do nothing to the epoch count.
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
      (iree_atomic_load_int64(&notification->value, iree_memory_order_acquire) &
       IREE_NOTIFICATION_WAITER_MASK) == 0);
}

void iree_notification_post(iree_notification_t* notification, int32_t count) {
  uint64_t previous_value = iree_atomic_fetch_add_int64(
      &notification->value, IREE_NOTIFICATION_EPOCH_INC,
      iree_memory_order_acq_rel);
  // Ensure we have at least one waiter; wake up to |count| of them.
  if (IREE_UNLIKELY(previous_value & IREE_NOTIFICATION_WAITER_MASK)) {
    iree_futex_wake(iree_notification_epoch_address(notification), count);
  }
}

iree_wait_token_t iree_notification_prepare_wait(
    iree_notification_t* notification) {
  uint64_t previous_value = iree_atomic_fetch_add_int64(
      &notification->value, IREE_NOTIFICATION_WAITER_INC,
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
  return (iree_atomic_load_int64(&notification->value,
                                 iree_memory_order_acquire) >>
          IREE_NOTIFICATION_EPOCH_SHIFT) != wait_token
             ? IREE_NOTIFICATION_RESULT_RESOLVED
             : IREE_NOTIFICATION_RESULT_UNRESOLVED;
}

bool iree_notification_commit_wait(iree_notification_t* notification,
                                   iree_wait_token_t wait_token,
                                   iree_duration_t spin_ns,
                                   iree_time_t deadline_ns) {
  // Quick check to see if the wait has already succeeded (the epoch advances
  // from when it was captured in iree_notification_prepare_wait).
  iree_notification_result_t result =
      iree_notification_test_wait_condition(notification, wait_token);

  // If not already reached and spinning is enabled then we'll try that first.
  if (result == IREE_NOTIFICATION_RESULT_UNRESOLVED &&
      spin_ns != IREE_DURATION_ZERO) {
    // If spinning we need to compute the absolute deadline that we'll spin
    // until (as we may be descheduled while spinning and time may drift).
    const iree_time_t spin_deadline_ns = iree_time_now() + spin_ns;
    IREE_TRACE_ZONE_BEGIN_NAMED(z0, "iree_notification_commit_wait_spin");
    do {
      // Try to be nice to the processor when using SMT.
      iree_processor_yield();
      result = iree_notification_test_wait_condition(notification, wait_token);
    } while (result == IREE_NOTIFICATION_RESULT_UNRESOLVED &&
             iree_time_now() < spin_deadline_ns);
    IREE_TRACE_ZONE_END(z0);
  }

  // If spinning failed let the kernel do what it does ... okish at.
  // We loop until notified and the epoch increments from what we captured
  // during iree_notification_prepare_wait.
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

  // TODO(benvanik): benchmark under real workloads.
  // iree_memory_order_relaxed would suffice for correctness but the faster
  // the waiter count gets to 0 the less likely we'll wake on the futex.
  uint64_t previous_value = iree_atomic_fetch_add_int64(
      &notification->value, IREE_NOTIFICATION_WAITER_DEC,
      iree_memory_order_acq_rel);
  SYNC_ASSERT((previous_value & IREE_NOTIFICATION_WAITER_MASK) != 0);

  return result == IREE_NOTIFICATION_RESULT_RESOLVED;
}

void iree_notification_cancel_wait(iree_notification_t* notification) {
  // TODO(benvanik): benchmark under real workloads.
  // iree_memory_order_relaxed would suffice for correctness but the faster
  // the waiter count gets to 0 the less likely we'll wake on the futex.
  uint64_t previous_value = iree_atomic_fetch_add_int64(
      &notification->value, IREE_NOTIFICATION_WAITER_DEC,
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
  // Otherwise we may have a real deadline and want it in absolute form so that
  // we can easily handle spurious wakes.
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
