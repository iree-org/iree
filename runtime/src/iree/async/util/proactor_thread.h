// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_ASYNC_UTIL_PROACTOR_THREAD_H_
#define IREE_ASYNC_UTIL_PROACTOR_THREAD_H_

#include "iree/async/proactor.h"
#include "iree/base/api.h"
#include "iree/base/threading/thread.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Proactor thread
//===----------------------------------------------------------------------===//

// Callback invoked when the proactor encounters an unrecoverable error during
// poll(). Fires exactly once, then the thread stops.
//
// The error callback fires from the proactor thread. Heavy cleanup should be
// deferred (e.g., post work to another queue). The proactor thread will exit
// after this callback returns regardless of what the callback does — it exists
// purely to notify the application that the poll loop is dead.
//
// |status| is the fatal error from poll(). Ownership transfers to the callback
// (the callback must consume or ignore it).
typedef void (*iree_async_proactor_thread_error_fn_t)(void* user_data,
                                                      iree_status_t status);

// Configuration for a proactor thread.
//
// All fields have sensible defaults via _default(). Callers typically only
// customize poll_timeout and debug_name; the other fields are for production
// tuning (NUMA pinning, error monitoring).
typedef struct iree_async_proactor_thread_options_t {
  // Maximum time the thread blocks in each poll() call. Shorter values give
  // faster response to stop requests but waste CPU on empty polls. Longer
  // values save power at the cost of stop latency.
  //
  // Default (0): infinite — the thread blocks until work arrives or wake() is
  // called. This is correct for most uses: request_stop() calls wake()
  // internally, so the thread always responds promptly to stop requests.
  iree_duration_t poll_timeout;

  // Thread CPU affinity for NUMA-aware pinning. The proactor thread handles
  // all completions for its proactor, so pinning it near the NIC or GPU
  // that generates most completions reduces cross-NUMA traffic.
  //
  // Default: no affinity (OS scheduler chooses).
  iree_thread_affinity_t affinity;

  // Debug name for tracing and diagnostics. Shows in Tracy as the fiber/thread
  // name. Keep it short and descriptive (e.g., "io-main", "net-rx").
  //
  // Default: empty (thread gets a generic name).
  iree_string_view_t debug_name;

  // Error callback invoked on fatal proactor failure. If NULL, the thread stops
  // silently and the error can be retrieved later via consume_status().
  iree_async_proactor_thread_error_fn_t error_fn;
  void* error_user_data;
} iree_async_proactor_thread_options_t;

// Returns default proactor thread options (infinite poll timeout, no affinity,
// no error callback).
static inline iree_async_proactor_thread_options_t
iree_async_proactor_thread_options_default(void) {
  iree_async_proactor_thread_options_t options = {0};
  return options;
}

// A dedicated thread that drives a proactor's poll loop.
//
// This is pure convenience: the proactor is caller-driven (it only makes
// progress when poll() is called). This utility automates the calling by
// running poll() in a loop on a dedicated thread, dispatching completions as
// they arrive.
//
// The thread retains the proactor for its lifetime. Multiple threads can share
// a proactor if the backend supports it, but typically each proactor has one
// dedicated thread — this matches the io_uring model where each ring is
// single-threaded for submission/completion.
//
// ## Lifecycle
//
//   // Create proactor and thread:
//   iree_async_proactor_t* proactor = ...;
//   iree_async_proactor_thread_t* thread = NULL;
//   iree_async_proactor_thread_options_t options =
//       iree_async_proactor_thread_options_default();
//   options.debug_name = iree_make_cstring_view("io-main");
//   IREE_RETURN_IF_ERROR(iree_async_proactor_thread_create(
//       proactor, options, allocator, &thread));
//
//   // ... submit operations to proactor from any thread ...
//   // Completions fire on the proactor thread automatically.
//
//   // Shutdown:
//   iree_async_proactor_thread_request_stop(thread);
//   IREE_RETURN_IF_ERROR(iree_async_proactor_thread_join(
//       thread, IREE_DURATION_INFINITE));
//   iree_async_proactor_thread_release(thread);
//
// ## Shutdown protocol
//
// The proactor thread does NOT cancel pending operations on stop. It is the
// caller's responsibility to ensure all operations are either completed or
// cancelled before joining the thread. If operations remain in-flight when
// the thread exits, their callbacks will never fire.
//
// The recommended shutdown sequence:
//   - Cancel or drain all pending operations.
//   - Call request_stop() (sets internal flag, wakes poll loop).
//   - Call join() (blocks until thread exits).
//   - Call release() (releases proactor reference, frees thread).
//
// ## Thread safety
//
// request_stop():   Safe from any thread.
// join():           Must not be called from the proactor thread itself.
// consume_status(): Safe from any thread after join() returns.
// retain/release:   Safe from any thread (atomic ref count).
typedef struct iree_async_proactor_thread_t iree_async_proactor_thread_t;

// Creates a proactor thread that polls |proactor| in a loop.
// The thread starts immediately after creation. |proactor| is retained for
// the lifetime of the thread.
iree_status_t iree_async_proactor_thread_create(
    iree_async_proactor_t* proactor,
    iree_async_proactor_thread_options_t options, iree_allocator_t allocator,
    iree_async_proactor_thread_t** out_thread);

// Retains a reference to the proactor thread.
void iree_async_proactor_thread_retain(iree_async_proactor_thread_t* thread);

// Releases a reference to the proactor thread. When the count reaches zero,
// the thread must already be stopped (join must have returned). Releases the
// retained proactor reference and frees the thread.
void iree_async_proactor_thread_release(iree_async_proactor_thread_t* thread);

// Requests the thread to stop after completing its current poll iteration.
// Non-blocking: sets an internal flag and calls wake() on the proactor so
// the thread sees the flag promptly even if poll() was blocking.
//
// After this call, the thread will exit its poll loop. Pending operations
// already in-flight may still complete (their callbacks will fire before the
// thread exits).
void iree_async_proactor_thread_request_stop(
    iree_async_proactor_thread_t* thread);

// Blocks the calling thread until the proactor thread exits.
// Returns OK if the thread exited cleanly (either no work remaining or stop
// requested). Returns IREE_STATUS_DEADLINE_EXCEEDED if |timeout| expires
// before the thread exits.
//
// Must not be called from the proactor thread itself (deadlock). After this
// returns OK, the thread is guaranteed to have exited and no more callbacks
// will fire from it.
iree_status_t iree_async_proactor_thread_join(
    iree_async_proactor_thread_t* thread, iree_duration_t timeout);

// Retrieves and clears the stored failure status.
// If the thread stopped due to a fatal proactor error, returns that error
// (ownership transfers to caller). If the thread stopped normally (via
// request_stop or natural completion), returns iree_ok_status().
//
// May only be called after join() returns. Calling before join is undefined.
// Calling multiple times is safe: first call gets the error, subsequent calls
// return OK.
iree_status_t iree_async_proactor_thread_consume_status(
    iree_async_proactor_thread_t* thread);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_ASYNC_UTIL_PROACTOR_THREAD_H_
