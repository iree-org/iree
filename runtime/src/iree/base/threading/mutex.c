// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/threading/mutex.h"

#include <string.h>

#if !IREE_SYNCHRONIZATION_DISABLE_UNSAFE && defined(IREE_PLATFORM_WINDOWS)
#include <intrin.h>
#endif  // IREE_PLATFORM_WINDOWS

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

void iree_mutex_lock(iree_mutex_t* mutex) {
  iree_tracing_mutex_before_lock(mutex->lock_id);
  iree_mutex_impl_lock(mutex);
  iree_tracing_mutex_after_lock(mutex->lock_id);
}

bool iree_mutex_try_lock(iree_mutex_t* mutex) {
  bool was_acquired = iree_mutex_impl_try_lock(mutex);
  iree_tracing_mutex_after_try_lock(mutex->lock_id, was_acquired);
  return was_acquired;
}

void iree_mutex_unlock(iree_mutex_t* mutex) {
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

void iree_mutex_lock(iree_mutex_t* mutex) { iree_mutex_impl_lock(mutex); }

bool iree_mutex_try_lock(iree_mutex_t* mutex) {
  return iree_mutex_impl_try_lock(mutex);
}

void iree_mutex_unlock(iree_mutex_t* mutex) { iree_mutex_impl_unlock(mutex); }

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

void iree_slim_mutex_lock(iree_slim_mutex_t* mutex) {
  iree_mutex_lock(&mutex->impl);
}

bool iree_slim_mutex_try_lock(iree_slim_mutex_t* mutex) {
  return iree_mutex_try_lock(&mutex->impl);
}

void iree_slim_mutex_unlock(iree_slim_mutex_t* mutex) {
  iree_mutex_unlock(&mutex->impl);
}

#else

#if IREE_SYNCHRONIZATION_DISABLE_UNSAFE

void iree_slim_mutex_initialize(iree_slim_mutex_t* out_mutex) {}

void iree_slim_mutex_deinitialize(iree_slim_mutex_t* mutex) {}

void iree_slim_mutex_lock(iree_slim_mutex_t* mutex) {}

bool iree_slim_mutex_try_lock(iree_slim_mutex_t* mutex) {
  return iree_mutex_try_lock((iree_mutex_t*)&mutex->reserved);
}

void iree_slim_mutex_unlock(iree_slim_mutex_t* mutex) {}

#elif defined(IREE_PLATFORM_APPLE)

void iree_slim_mutex_initialize(iree_slim_mutex_t* out_mutex) {
  out_mutex->value = OS_UNFAIR_LOCK_INIT;
}

void iree_slim_mutex_deinitialize(iree_slim_mutex_t* mutex) {
  os_unfair_lock_assert_not_owner(&mutex->value);
}

void iree_slim_mutex_lock(iree_slim_mutex_t* mutex) {
  os_unfair_lock_lock(&mutex->value);
}

bool iree_slim_mutex_try_lock(iree_slim_mutex_t* mutex) {
  return os_unfair_lock_trylock(&mutex->value);
}

void iree_slim_mutex_unlock(iree_slim_mutex_t* mutex) {
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

void iree_slim_mutex_lock(iree_slim_mutex_t* mutex) {
  iree_mutex_impl_lock(mutex);
}

bool iree_slim_mutex_try_lock(iree_slim_mutex_t* mutex) {
  return iree_mutex_impl_try_lock(mutex);
}

void iree_slim_mutex_unlock(iree_slim_mutex_t* mutex) {
  iree_mutex_impl_unlock(mutex);
}

#elif defined(IREE_RUNTIME_USE_FUTEX)

// This implementation is a combo of several sources:
//
// Basics of Futexes by Eli Bendersky:
// https://eli.thegreenplace.net/2018/basics-of-futexes/
//
// Futex based locks for C11's generic atomics by Jens Gustedt:
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
  SYNC_ASSERT(iree_atomic_load(&mutex->value, iree_memory_order_acquire) == 0);
}

// Helper to perform a compare_exchange operation on mutex->value.
static bool iree_slim_mutex_try_lock_compare_exchange(iree_slim_mutex_t* mutex,
                                                      int32_t* expected,
                                                      int32_t desired) {
  return iree_atomic_compare_exchange_weak(&mutex->value, expected, desired,
                                           iree_memory_order_acquire,
                                           iree_memory_order_relaxed);
}

void iree_slim_mutex_lock(iree_slim_mutex_t* mutex) {
  // Try first to acquire the lock from an unlocked state.
  int32_t value = 0;
  if (iree_slim_mutex_try_lock_compare_exchange(mutex, &value,
                                                iree_slim_mutex_value(1))) {
    return;
  }

  // Increment the count bits to indicate that we want the lock and are willing
  // to wait for it to be available.
  value =
      iree_atomic_fetch_add(&mutex->value, 1, iree_memory_order_relaxed) + 1;

  while (true) {
    // While the lock is available: try to acquire it for this thread.
    while (!iree_slim_mutex_is_locked(value)) {
      if (iree_slim_mutex_try_lock_compare_exchange(
              mutex, &value, iree_slim_mutex_value(value))) {
        return;
      }

      // Spin a small amount to give us a tiny chance of falling through to the
      // wait.
      int spin_count = 100;
      for (int i = 0; i < spin_count && iree_slim_mutex_is_locked(value); ++i) {
        iree_processor_yield();
        value = iree_atomic_load(&mutex->value, iree_memory_order_relaxed);
      }
    }

    // While the lock is unavailable: wait for it to become available.
    while (iree_slim_mutex_is_locked(value)) {
      iree_futex_wait(&mutex->value, value, IREE_TIME_INFINITE_FUTURE);
      value = iree_atomic_load(&mutex->value, iree_memory_order_relaxed);
    }
  }
}

bool iree_slim_mutex_try_lock(iree_slim_mutex_t* mutex) {
  int32_t value = 0;
  return iree_slim_mutex_try_lock_compare_exchange(mutex, &value,
                                                   iree_slim_mutex_value(1));
}

void iree_slim_mutex_unlock(iree_slim_mutex_t* mutex) {
  // Transition 1->0 (unlocking with no waiters) or 2->1 (with waiters).
  if (iree_atomic_fetch_sub(&mutex->value, iree_slim_mutex_value(1),
                            iree_memory_order_release) !=
      iree_slim_mutex_value(1)) {
    // One (or more) waiters; wake a single one to avoid a thundering herd.
    iree_futex_wake(&mutex->value, 1);
  }
}

#else

// Pass-through to iree_mutex_t as a fallback for platforms without a futex.

void iree_slim_mutex_initialize(iree_slim_mutex_t* out_mutex) {
  iree_mutex_initialize(&out_mutex->impl);
}

void iree_slim_mutex_deinitialize(iree_slim_mutex_t* mutex) {
  iree_mutex_deinitialize(&mutex->impl);
}

void iree_slim_mutex_lock(iree_slim_mutex_t* mutex) {
  iree_mutex_lock(&mutex->impl);
}

bool iree_slim_mutex_try_lock(iree_slim_mutex_t* mutex) {
  return iree_mutex_try_lock(&mutex->impl);
}

void iree_slim_mutex_unlock(iree_slim_mutex_t* mutex) {
  iree_mutex_unlock(&mutex->impl);
}

#endif  // IREE_PLATFORM_*

#endif  //  IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_SLOW_LOCKS
