// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// NOTE: the best kind of synchronization is no synchronization; always try to
// design your algorithm so that you don't need anything from this file :)
// See https://travisdowns.github.io/blog/2020/07/06/concurrency-costs.html

#ifndef IREE_BASE_INTERNAL_SYNCHRONIZATION_H_
#define IREE_BASE_INTERNAL_SYNCHRONIZATION_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"

// NOTE: clang cannot support thread annotations in C code due to some
// representational bugs... which means that we can't use it here. Boo.
// There's some workarounds I've seen but getting TSAN working would be much
// easier as a starting point.
#if 0  // defined(IREE_COMPILER_CLANG)
#define IREE_THREAD_ANNOTATION_ATTRIBUTE(x) __attribute__((x))
#else
#define IREE_THREAD_ANNOTATION_ATTRIBUTE(x)
#endif  // IREE_COMPILER_CLANG

#ifdef __cplusplus
// Documents if a shared field or global variable needs to be protected by a
// mutex. IREE_GUARDED_BY() allows the user to specify a particular mutex that
// should be held when accessing the annotated variable.
#define IREE_GUARDED_BY(x) IREE_THREAD_ANNOTATION_ATTRIBUTE(guarded_by(x))
#else
#define IREE_GUARDED_BY(x)
#endif  // __cplusplus

#ifdef __cplusplus
// Like IREE_GUARDED_BY but specifies that the contents of a pointer are guarded
// by a mutex instead of the pointer itself.
#define IREE_PTR_GUARDED_BY(x) \
  IREE_THREAD_ANNOTATION_ATTRIBUTE(pt_guarded_by(x))
#else
#define IREE_PTR_GUARDED_BY(x)
#endif  // __cplusplus

// Allow users to fully disable all synchronization for systems that are known
// to never need it. This removes our dependency on pthreads.
#if !IREE_SYNCHRONIZATION_DISABLE_UNSAFE

#if defined(IREE_PLATFORM_ANDROID) || defined(IREE_PLATFORM_EMSCRIPTEN) || \
    defined(IREE_PLATFORM_LINUX) || defined(IREE_PLATFORM_WINDOWS)
#define IREE_PLATFORM_HAS_FUTEX 1
#endif  // IREE_PLATFORM_*

#if defined(IREE_PLATFORM_APPLE)
#include <os/lock.h>
#endif  // IREE_PLATFORM_APPLE

#if !defined(IREE_PLATFORM_WINDOWS)
#include <pthread.h>
#endif  // !IREE_PLATFORM_WINDOWS

// We have the CRITICAL_SECTION path for now but Slim Reader/Writer lock (SRW)
// is much better (and what std::mutex uses). SRW doesn't spin, though, and has
// some other implications that don't quite line up with pthread_mutex_t on most
// platforms. Once we have larger end-to-end benchmarks we should choose based
// on workloads.
#define IREE_MUTEX_USE_WIN32_SRW 1

#endif  // !IREE_SYNCHRONIZATION_DISABLE_UNSAFE

#ifdef __cplusplus
extern "C" {
#endif

#define IREE_ALL_WAITERS INT32_MAX
#define IREE_INFINITE_TIMEOUT_MS UINT32_MAX

//==============================================================================
// iree_mutex_t
//==============================================================================

// A normal fat mutex (ala std::mutex).
// This may be implemented as a slim mutex on certain platforms but in the worst
// case will be the native platform primitive (like pthread_mutex_t) and as such
// should not be embedded in structures meant to be kept small.
//
// Windows: Slim Reader/Writer (SRW) Locks
// All others: pthread_mutex_t
typedef struct iree_mutex_t IREE_THREAD_ANNOTATION_ATTRIBUTE(
    capability("mutex")) {
#if IREE_SYNCHRONIZATION_DISABLE_UNSAFE
  int reserved;
#elif defined(IREE_PLATFORM_WINDOWS) && defined(IREE_MUTEX_USE_WIN32_SRW)
  SRWLOCK value;
#elif defined(IREE_PLATFORM_WINDOWS)
  CRITICAL_SECTION value;
#else
  pthread_mutex_t value;
#endif  // IREE_PLATFORM_*
#if (IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_SLOW_LOCKS)
  uint32_t lock_id;
#endif  // IREE_TRACING_FEATURE_SLOW_LOCKS
} iree_mutex_t;

#if (IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_SLOW_LOCKS)
// Initializes |out_mutex| to the well-defined unlocked contents.
// Must be called prior to using any other iree_mutex_* method.
#define iree_mutex_initialize(out_mutex)                                      \
  static const iree_tracing_location_t TracyConcat(                           \
      __tracy_source_location, __LINE__) = {NULL, __FUNCTION__, __FILE__,     \
                                            (uint32_t)__LINE__, 0};           \
  iree_mutex_initialize_impl(&TracyConcat(__tracy_source_location, __LINE__), \
                             out_mutex);
void iree_mutex_initialize_impl(const iree_tracing_location_t* src_loc,
                                iree_mutex_t* out_mutex);
#else
// Initializes |out_mutex| to the well-defined unlocked contents.
// Must be called prior to using any other iree_mutex_* method.
void iree_mutex_initialize(iree_mutex_t* out_mutex);
#endif  // IREE_TRACING_FEATURE_SLOW_LOCKS

// Deinitializes |mutex| (after a prior call to iree_mutex_initialize).
// The mutex must not be held by any thread.
void iree_mutex_deinitialize(iree_mutex_t* mutex)
    IREE_THREAD_ANNOTATION_ATTRIBUTE(locks_excluded(mutex));

// Locks the |mutex| and returns when held by the caller.
void iree_mutex_lock(iree_mutex_t* mutex)
    IREE_THREAD_ANNOTATION_ATTRIBUTE(acquire_capability(mutex));

// Tries to lock the |mutex| and returns true if the caller holds the lock.
bool iree_mutex_try_lock(iree_mutex_t* mutex)
    IREE_THREAD_ANNOTATION_ATTRIBUTE(try_acquire_capability(true, mutex));

// Unlocks the |mutex|, which must be held by the caller.
void iree_mutex_unlock(iree_mutex_t* mutex)
    IREE_THREAD_ANNOTATION_ATTRIBUTE(release_capability(mutex));

//==============================================================================
// iree_slim_mutex_t
//==============================================================================

// TODO(benvanik): instrument with tracy; need to capture source location on
// init and add storage for ID.

// A lightweight unfair lock.
// Depending on platform this is significantly smaller than a mutex (4-8 bytes
// vs 64+ bytes), can always be statically initialized/requires no allocations,
// and performs the minimal amount of work possible while still playing nicely
// with the OS thread scheduler.
//
// Unlike a full mutex these don't have the ability to be shared across
// processes (not something we care about), don't have a way to define timeouts,
// and have only a binary held/unheld state. They are often an order of
// magnitude faster in uncontended/lightly-contended code and the same
// performance in highly-contended code, though, so it's worth it for locks that
// be guarding small data structures (queue pointers, etc) and touched from many
// threads. Since they are so lightweight it's possible to embed them per-object
// instead of per-manager and change from a single highly-contended lock to
// thousands of almost completely uncontended slim locks.
//
// Though these locks have lightweight return paths (using atomics ops, see
// the below paragraph, with or without spinning), they always have a
// heavyweight fallback path that ends up calling into the kernel to properly
// let the thread wait. This is critical to avoid pathological cases under
// contention and allowing for thread priority inheritance when there are
// multiple threads competing that may otherwise be scheduled in a potentially
// livelocking order.
//
// The "unfair" here comes from the fact that it's possible on certain platforms
// for certain threads to never be able to acquire the lock in cases of
// extremely high contention or widely disparate thread priority levels. This is
// mitigated by ensuring only very small regions of code are guarded and that
// there's enough work happening outside of the lock on any particular thread to
// ensure that there's some chance of other threads being able to acquire it.
//
// Notes on weakly-ordered atomics (for the lightweight return paths)
// ------------------------------------------------------------------
//
// The lightweight return paths (avoiding waiting in the kernel) are typically
// implemented using acquire/release weakly-ordered atomics. When these code
// paths are taken:
//
//   * iree_slim_mutex_lock is a read-modify-write operation on the
//     mutex with memory_order_acquire.
//   * iree_slim_mutex_unlock is a read-modify-write operation on the
//     mutex with memory_order_release.
//
// This means the following guarantee for the caller of iree_slim_mutex_lock:
//
//   When iree_slim_mutex_lock returns on this thread T1 from having waited on
//   another thread T2 calling iree_slim_mutex_unlock, all memory read and write
//   operations performed on thread T2 prior to calling iree_slim_mutex_unlock
//   are guaranteed to "happen-before" any subsequent memory read or write on
//   thread T1.
//
// This is meant to be an implementation detail within the standard contract
// for anything called a "mutex". The C++ standard is a good reference for what
// it means to be a mutexin the context of the C11/C++11 memory model:
// https://eel.is/c++draft/thread.mutex#requirements.mutex
//
// It is not trivial why the above-described memory orders happen to be
// sufficient to meet these mutex requirements, so here is a FAQ:
//
// Q: Is it really OK with the memory model for mutex lock and unlock to be mere
//    atomic operations, and wouldn't they at least need to have sequentially-
//    consistent order?
// A: https://eel.is/c++draft/thread.mutex.requirements.mutex.general  says:
//      > The implementation provides lock and unlock operations, as described
//      > below. For purposes of determining the existence of a data race, these
//      > behave as atomic operations ([intro.multithread]). The lock and unlock
//      > operations on a single mutex appears to occur in a single total order.
//    Key here is "on a single mutex". There is no ordering requirement across
//    operations on separate mutex objects. That is what sequentially-consistent
//    atomics would be needed for. The only ordering requirement is among ops
//    "on a single mutex" and that is what one can implement with careful use
//    of acquire/release atomics.
//
// Q: Since iree_slim_mutex_{lock,unlock} are read-modify-write operations,
//    shouldn't they have at least memory_order_acq_rel so that both the
//    read and write aspects of each of them construct happens-before
//    relationships with each other?
// A: Separate answer regarting iree_slim_mutex_lock and iree_slim_mutex_unlock:
//    * Why iree_slim_mutex_lock only needs acquire order, not release:
//        When iree_slim_mutex_lock returns, the calling thread T1 holds the
//        lock. A call to iree_slim_mutex_lock on another thread T2 can't return
//        until thread T1 calls iree_slim_mutex_unlock, which already has
//        memory_order_release.
//    * Why iree_slim_mutex_unlock only needs release order, not acquire:
//        By requirement of iree_slim_mutex_unlock, the calling thread had
//        already called iree_slim_mutex_lock prior to calling
//        iree_slim_mutex_unlock. Mutual exclusion implies that no other thread
//        can have operated on this mutex object since that iree_slim_mutex_lock
//        call on the calling thread.
//
// OS-specific implementation aspects (for the heavyweight return paths)
// ---------------------------------------------------------------------
//
// MacOS/iOS: os_unfair_lock
//   Spins and after a short backoff drops to a futex-like behavior of waiting
//   in the kernel. Unfortunately real futexes aren't supported.
// See:
//   https://developer.apple.com/documentation/os/synchronization
//   https://opensource.apple.com/source/libplatform/libplatform-125/src/os/lock.c.auto.html
//
// Emscripten: emscripten_futex_wait/emscripten_futex_wake
//   Spins and after a short backoff drops to a futex-like behavior of waiting
//   in the kernel.
// See:
//   https://github.com/emscripten-core/emscripten/blob/b43474f55aeb49083b9df74fdd0e52ec8decf788/system/include/emscripten/threading.h#L114-L120
//   https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Atomics/wait
//   https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Atomics/notify
//
// Windows: WaitOnAddress/WakeByAddress*
//   Spins and after a short backoff drops to a futex and waits in the kernel.
// See:
//   https://docs.microsoft.com/en-us/windows/win32/api/synchapi/nf-synchapi-waitonaddress
//   https://devblogs.microsoft.com/oldnewthing/20170601-00/?p=96265
//
// Linux/Android/others: futex
//   Spins and after a short backoff drops to a futex and waits in the kernel.
// See:
//   http://locklessinc.com/articles/futex_cheat_sheet/
//   https://man7.org/linux/man-pages/man2/futex.2.html
//   https://eli.thegreenplace.net/2018/basics-of-futexes/
//   https://bartoszmilewski.com/2008/09/01/thin-lock-vs-futex/
typedef struct iree_slim_mutex_t IREE_THREAD_ANNOTATION_ATTRIBUTE(
    capability("mutex")) {
#if IREE_SYNCHRONIZATION_DISABLE_UNSAFE
  int reserved;
#elif (IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_FAST_LOCKS)
  iree_mutex_t impl;  // re-route to slow mutex
#elif defined(IREE_PLATFORM_APPLE)
  os_unfair_lock value;
#elif defined(IREE_PLATFORM_WINDOWS) && defined(IREE_MUTEX_USE_WIN32_SRW)
  SRWLOCK value;
#elif defined(IREE_PLATFORM_HAS_FUTEX)
  iree_atomic_int32_t value;
#else
  iree_mutex_t impl;  // fallback
#endif  // IREE_PLATFORM_*
} iree_slim_mutex_t;

#if (IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_FAST_LOCKS)
// Initializes |out_mutex| to the well-defined unlocked contents.
// Must be called prior to using any other iree_slim_mutex_* method.
#define iree_slim_mutex_initialize(out_mutex)                             \
  static const iree_tracing_location_t TracyConcat(                       \
      __tracy_source_location, __LINE__) = {NULL, __FUNCTION__, __FILE__, \
                                            (uint32_t)__LINE__, 0};       \
  iree_slim_mutex_initialize_impl(                                        \
      &TracyConcat(__tracy_source_location, __LINE__), out_mutex);
void iree_slim_mutex_initialize_impl(const iree_tracing_location_t* src_loc,
                                     iree_slim_mutex_t* out_mutex);
#else
// Initializes |out_mutex| to the well-defined unlocked contents.
// Must be called prior to using any other iree_slim_mutex_* method.
//
// Though optional (static initialization is fine) this is required to support
// lock tracing. Assume it's (mostly) free and always call it if possible. This
// also allows us to swap in a non-slim lock for enhanced debugging if we run
// into threading issues.
void iree_slim_mutex_initialize(iree_slim_mutex_t* out_mutex);
#endif  // IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_FAST_LOCKS

// Deinitializes |mutex| (after a prior call to iree_slim_mutex_initialize).
// The mutex must not be held by any thread.
void iree_slim_mutex_deinitialize(iree_slim_mutex_t* mutex)
    IREE_THREAD_ANNOTATION_ATTRIBUTE(locks_excluded(mutex));

// Locks the |mutex| and returns when held by the caller.
void iree_slim_mutex_lock(iree_slim_mutex_t* mutex)
    IREE_THREAD_ANNOTATION_ATTRIBUTE(acquire_capability(mutex));

// Tries to lock the |mutex| and returns true if the caller holds the lock.
//
// This function is allowed to fail spuriously, i.e. even if the lock isn't
// held by another thread.
bool iree_slim_mutex_try_lock(iree_slim_mutex_t* mutex)
    IREE_THREAD_ANNOTATION_ATTRIBUTE(try_acquire_capability(true, mutex));

// Unlocks the |mutex|, which must be held by the caller.
void iree_slim_mutex_unlock(iree_slim_mutex_t* mutex)
    IREE_THREAD_ANNOTATION_ATTRIBUTE(release_capability(mutex));

//==============================================================================
// iree_notification_t
//==============================================================================

// TODO(benvanik): add tracy support for watching the waits.

// A lightweight wait-free cross-thread notification mechanism.
// Classically called an 'event counter', these replace the use of condvars in
// lock-free code where you wouldn't want to guard a lock-free data structure
// with a lock.
//
// See:
// http://www.1024cores.net/home/lock-free-algorithms/eventcounts
// https://software.intel.com/en-us/forums/intel-threading-building-blocks/topic/299245
// https://github.com/r10a/Event-Counts
// https://github.com/facebook/folly/blob/main/folly/experimental/EventCount.h
// https://github.com/concurrencykit/ck/blob/master/include/ck_ec.h
typedef struct iree_notification_t {
#if IREE_SYNCHRONIZATION_DISABLE_UNSAFE
  // Nothing required. Unused field to make compilers happy.
  int reserved;
#elif !defined(IREE_PLATFORM_HAS_FUTEX)
  // No futex on darwin/when using TSAN, so use mutex/condvar instead.
  pthread_mutex_t mutex;
  pthread_cond_t cond;
  uint32_t epoch;
  uint32_t waiters;
#else
  iree_atomic_int64_t value;
#endif  // IREE_PLATFORM_*
} iree_notification_t;

#if IREE_SYNCHRONIZATION_DISABLE_UNSAFE
#define IREE_NOTIFICATION_INIT \
  { IREE_ATOMIC_VAR_INIT(0) }
#elif !defined(IREE_PLATFORM_HAS_FUTEX)
#define IREE_NOTIFICATION_INIT \
  { PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER, 0, 0 }
#else
#define IREE_NOTIFICATION_INIT \
  { IREE_ATOMIC_VAR_INIT(0) }
#endif  // notification type

// Initializes a notification to no waiters and an initial epoch of 0.
void iree_notification_initialize(iree_notification_t* out_notification);

// Deinitializes |notification| (after a prior call to
// iree_notification_initialize). No threads may be waiting on the notification.
void iree_notification_deinitialize(iree_notification_t* notification);

// Notifies up to |count| waiters of a change. Each waiter will wake and can
// check to see if they need to do any additional work.
// To notify all potential waiters pass IREE_ALL_WAITERS.
//
// Acts as (at least) a memory_order_release operation on the
// notification object. See the comment on iree_notification_commit_wait, which
// is the memory_order_acquire operation that is meant to pair with that.
void iree_notification_post(iree_notification_t* notification, int32_t count);

typedef uint32_t iree_wait_token_t;  // opaque

// Prepares for a wait operation, returning a token that must be passed to
// iree_notification_commit_wait to perform the actual wait.
//
// Acts as a memory_order_acq_rel read-modify-write operation on the
// notification object. See the comment on iree_notification_commit_wait for a
// general explanation of acquire/release semantics in this context.
iree_wait_token_t iree_notification_prepare_wait(
    iree_notification_t* notification);

// Commits a pending wait operation when the caller has ensured it must wait.
// Waiting will continue until a notification has been posted or |deadline_ns|
// is reached. Returns false if the deadline is reached before a notification is
// posted.
//
// If |spin_ns| is not IREE_DURATION_ZERO the wait _may_ spin for at least the
// specified duration before entering the system wait API.
//
// Acts as (at least) a memory_order_acquire operation on the notification
// object. This is meant to be paired with iree_notification_post, which is a
// memory_order_release operation. This means the following guarantee:
// When iree_notification_commit_wait returns on this thread T1 from having
// waited on another thread T2 calling iree_notification_post, all memory read
// and write operations performed on thread T2 prior to calling
// iree_notification_post are guaranteed to "happen-before" any subsequent
// memory read or write on thread T1.
bool iree_notification_commit_wait(iree_notification_t* notification,
                                   iree_wait_token_t wait_token,
                                   iree_duration_t spin_ns,
                                   iree_time_t deadline_ns);

// Cancels a pending wait operation without blocking.
//
// Acts as (at least) a memory_order_relaxed barrier:
//   Relaxed operation: there are no synchronization or ordering constraints
//   imposed on other reads or writes, only this operation's atomicity is
//   guaranteed.
void iree_notification_cancel_wait(iree_notification_t* notification);

// Returns true if the condition is true.
// |arg| is the |condition_arg| passed to the await function.
// Implementations must ensure they are coherent with their state values.
typedef bool (*iree_condition_fn_t)(void* arg);

// Blocks and waits until |condition_fn| returns true. Other threads must modify
// state checked by the |condition_fn| and post the notification.
// Returns true if the condition is true before |timeout| is reached. If the
// timeout is infinite then the return will always be true.
//
// Example:
//  thread 1:
//   bool check_flag_pred(void* arg) {
//     return iree_atomic_int32_load((iree_atomic_int32_t*)arg,
//                                   iree_memory_order_acquire) == 1;
//   }
//   iree_atomic_int32_t* flag = ...;
//   iree_notification_await(&notification, check_flag_pred, flag);
//  thread 2:
//   iree_atomic_int32_store(flag, 1, iree_memory_order_release);
//   iree_notification_post(&notification, IREE_ALL_WAITERS);
bool iree_notification_await(iree_notification_t* notification,
                             iree_condition_fn_t condition_fn,
                             void* condition_arg, iree_timeout_t timeout);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_BASE_INTERNAL_SYNCHRONIZATION_H_
