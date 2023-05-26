// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BASE_LOOP_H_
#define IREE_BASE_LOOP_H_

#include <inttypes.h>

#include "iree/base/allocator.h"
#include "iree/base/attributes.h"
#include "iree/base/status.h"
#include "iree/base/time.h"
#include "iree/base/wait_source.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_loop_t public API
//===----------------------------------------------------------------------===//

typedef struct iree_loop_t iree_loop_t;
typedef uint32_t iree_loop_command_t;

// TODO(benvanik): define prioritization. This is useful for ensuring fast
// coroutine switching by avoiding the current coroutine being set to the back
// of the loop. It's easy to shoot yourself in the foot, though: cooperative
// scheduling can be tricky.
typedef enum iree_loop_priority_e {
  IREE_LOOP_PRIORITY_DEFAULT = 0u,
} iree_loop_priority_t;

// Callback to execute user code used by the loop.
// |user_data| contains the value provided to the callback when enqueuing the
// operation and must remain live until the callback is made.
//
// If the callback is to be executed as normal |status| will be OK.
// A non-fatal error case of IREE_STATUS_DEADLINE_EXCEEDED can occur if the
// operation had a deadline specified and it elapsed prior to the condition
// being met.
//
// |status| otherwise indicates that the operation failed (such as a failed wait
// or a failed workgroup callback).
//
// Callbacks may reentrantly queue work on the |loop| _unless_ the passed
// |status| is IREE_STATUS_ABORTED indicating that the loop is shutting down or
// the operation is being aborted because of a prior failure.
//
// Any non-OK result will be routed to a loop-global error handler (depending on
// implementation) or otherwise ignored; users must set their own exit bits.
typedef iree_status_t(IREE_API_PTR* iree_loop_callback_fn_t)(
    void* user_data, iree_loop_t loop, iree_status_t status);

// Callback to execute a single workgroup in a grid dispatch.
// Each call receives the XYZ location in the grid and may run concurrently with
// any other workgroup call.
//
// Any non-OK result will be routed to the completion callback of the dispatch
// operation but not otherwise trigger loop failure. Other workgroups may
// continue to run up until the completion callback is issued.
typedef iree_status_t(IREE_API_PTR* iree_loop_workgroup_fn_t)(
    void* user_data, iree_loop_t loop, uint32_t workgroup_x,
    uint32_t workgroup_y, uint32_t workgroup_z);

// Function pointer for an iree_loop_t control function.
// |command| provides the operation to perform. Commands may use |params| to
// pass additional operation-specific parameters. |inout_ptr| usage is defined
// by each operation.
typedef iree_status_t(IREE_API_PTR* iree_loop_ctl_fn_t)(
    void* self, iree_loop_command_t command, const void* params,
    void** inout_ptr);

// An event system for executing queued asynchronous work.
// Implementations are allowed to execute operations in any order but generally
// runs FIFO and will only ever execute one operation at a time. The thread used
// for execution may change from operation to operation. Usage that has order
// requirements is required to perform the ordering themselves.
//
// This is a form of cooperative scheduling and the loop _may_ not make forward
// progress if a callback issues a blocking operation. All blocking operations
// should either be done on user-controlled threads or via the loop primitives
// such as iree_loop_wait_one. Callbacks may enqueue zero or more operations
// with 2+ performing a conceptual fork. The iree_loop_dispatch operation allows
// for a constrained style of concurrency matching a GPU grid dispatch and can
// be used as a primitive to implement other kinds of parallel loops.
//
// User data passed to callbacks is unowned and must be kept live by the
// requester. All callbacks are guaranteed to be issued even on failure and
// allocations made when enqueuing operations are safe to free in the callbacks.
//
// The rough behavior of the loop matches that of the web event loop
// dispatching events/promises/timeouts/etc. It's a stackless design where the
// owner of the primary control loop is hidden from the users of the loop. This
// allows implementations to integrate into existing scheduling mechanisms
// (ALooper, libuv, io_uring, the browser main event loop, etc) in a generic
// way. The design of the API here is meant to make it easy to put the
// implementation in external code (python/javascript/rust/java/etc) as only a
// single method with a fixed interface is used to cross the boundaries.
//
// Note that by default this implementation is only intended for host-level
// synchronization and scheduling: fairly coarse events performed fairly
// infrequently. Optimized multi-threaded workloads are intended to execute on
// the iree/task/ system via command buffers.
typedef struct iree_loop_t {
  // Control function data.
  void* self;
  // ioctl-style control function servicing all loop-related commands.
  // See iree_loop_command_t for more information.
  iree_loop_ctl_fn_t ctl;
} iree_loop_t;

// A loop that can do no work. Attempts to enqueue work will fail.
static inline iree_loop_t iree_loop_null() {
  iree_loop_t loop = {NULL, NULL};
  return loop;
}

// Executes |callback| from the loop at some point in the future.
//
// The callback is guaranteed to be issued but in an undefined order.
// |user_data| is not retained and must be live until the callback is issued.
IREE_API_EXPORT iree_status_t iree_loop_call(iree_loop_t loop,
                                             iree_loop_priority_t priority,
                                             iree_loop_callback_fn_t callback,
                                             void* user_data);

// Executes |workgroup_callback| from the loop at some point in the future
// with grid dispatch of |workgroup_count_xyz| workgroups. Each
// |workgroup_callback| will receive its XYZ location in the grid and
// |completion_callback| will be issued upon completion (or failure).
// The dispatched workgroups are not guaranteed to run concurrently and must
// not perform blocking operations.
//
// The completion callback is guaranteed to be issued but in an undefined order.
// The workgroup callback runs serially or concurrently from multiple threads.
// |user_data| is not retained and must be live until the callback is issued.
IREE_API_EXPORT iree_status_t iree_loop_dispatch(
    iree_loop_t loop, const uint32_t workgroup_count_xyz[3],
    iree_loop_workgroup_fn_t workgroup_callback,
    iree_loop_callback_fn_t completion_callback, void* user_data);

// Waits until |timeout| is reached and then issues |callback|.
// There may be a significant latency between |timeout| and when the |callback|
// is executed.
//
// The callback is guaranteed to be issued.
// |user_data| is not retained and must be live until the callback is issued.
IREE_API_EXPORT iree_status_t
iree_loop_wait_until(iree_loop_t loop, iree_timeout_t timeout,
                     iree_loop_callback_fn_t callback, void* user_data);

// Waits until the |wait_source| is satisfied or |timeout| is reached and then
// issues |callback|.
//
// The callback is guaranteed to be issued.
// |user_data| is not retained and must be live until the callback is issued.
IREE_API_EXPORT iree_status_t iree_loop_wait_one(
    iree_loop_t loop, iree_wait_source_t wait_source, iree_timeout_t timeout,
    iree_loop_callback_fn_t callback, void* user_data);

// Waits until one or more of the |wait_sources| is satisfied or |timeout| is
// reached and then issues |callback|.
//
// The callback is guaranteed to be issued.
// |wait_sources| and |user_data| is not retained and must be live until the
// callback is issued.
IREE_API_EXPORT iree_status_t iree_loop_wait_any(
    iree_loop_t loop, iree_host_size_t count, iree_wait_source_t* wait_sources,
    iree_timeout_t timeout, iree_loop_callback_fn_t callback, void* user_data);

// Waits until all of the |wait_sources| is satisfied or |timeout| is reached
// and then issues |callback|.
//
// The callback is guaranteed to be issued.
// |wait_sources| and |user_data| is not retained and must be live until the
// callback is issued.
IREE_API_EXPORT iree_status_t iree_loop_wait_all(
    iree_loop_t loop, iree_host_size_t count, iree_wait_source_t* wait_sources,
    iree_timeout_t timeout, iree_loop_callback_fn_t callback, void* user_data);

// Blocks the caller and waits until the loop is idle or |timeout| is reached.
//
// Not all implementations support this and may return
// IREE_STATUS_DEADLINE_EXCEEDED immediately when work is still pending.
// |user_data| is not retained and must be live until the callback is issued.
IREE_API_EXPORT iree_status_t iree_loop_drain(iree_loop_t loop,
                                              iree_timeout_t timeout);

//===----------------------------------------------------------------------===//
// iree_loop_t implementation details
//===----------------------------------------------------------------------===//
// These are exposed so that user applications can implement their own loops and
// are otherwise private to the API.

// Controls the behavior of an iree_loop_ctl_fn_t callback function.
enum iree_loop_command_e {
  // Issues the callback from the loop at some point in the future.
  // The callback will always be called (including when aborted).
  //
  // iree_loop_ctl_fn_t:
  //   params: iree_loop_call_params_t
  //   inout_ptr: unused
  IREE_LOOP_COMMAND_CALL = 0u,

  // Issues a workgroup callback across a grid and then issues the callback.
  // The completion callback will always be called (including when aborted).
  //
  // iree_loop_ctl_fn_t:
  //   params: iree_loop_dispatch_params_t
  //   inout_ptr: unused
  IREE_LOOP_COMMAND_DISPATCH,

  // TODO(benvanik): open/read/write/close/etc with iovecs.
  // Our iree_byte_span_t matches with `struct iovec` and if we share that we
  // can do scatter/gather I/O with io_uring.
  // Want something with an fd, flags, count, and iree_byte_span_t's.

  // TODO(benvanik): IREE_LOOP_COMMAND_WAIT_IDLE to get idle callbacks.

  // Sleeps until the timeout is reached then issues the callback.
  // The callback will always be called (including when aborted).
  //
  // iree_loop_ctl_fn_t:
  //   params: iree_loop_wait_until_params_t
  //   inout_ptr: unused
  IREE_LOOP_COMMAND_WAIT_UNTIL,

  // Waits until the wait source has resolved then issues the callback.
  // The callback will always be called (including when aborted).
  //
  // iree_loop_ctl_fn_t:
  //   params: iree_loop_wait_one_params_t
  //   inout_ptr: unused
  IREE_LOOP_COMMAND_WAIT_ONE,

  // Waits until one or more wait sources have resolved then issues the
  // callback. The callback will always be called (including when aborted).
  //
  // iree_loop_ctl_fn_t:
  //   params: iree_loop_wait_multi_params_t
  //   inout_ptr: unused
  IREE_LOOP_COMMAND_WAIT_ANY,

  // Waits until all of the wait sources have resolved then issues the
  // callback. The callback will always be called (including when aborted).
  //
  // iree_loop_ctl_fn_t:
  //   params: iree_loop_wait_multi_params_t
  //   inout_ptr: unused
  IREE_LOOP_COMMAND_WAIT_ALL,

  // Waits until the loop has no more pending work.
  // Resolves early with IREE_STATUS_DEADLINE_EXCEEDED if the timeout is reached
  // before the loop is idle or if the platform does not support the operation.
  //
  // iree_loop_ctl_fn_t:
  //   params: iree_loop_drain_params_t
  //   inout_ptr: unused
  IREE_LOOP_COMMAND_DRAIN,

  IREE_LOOP_COMMAND_MAX = IREE_LOOP_COMMAND_DRAIN,
};

typedef struct iree_loop_callback_t {
  // Callback function pointer.
  iree_loop_callback_fn_t fn;
  // User data passed to the callback function. Unowned.
  void* user_data;
} iree_loop_callback_t;

// Parameters for IREE_LOOP_COMMAND_CALL.
typedef struct iree_loop_call_params_t {
  // Callback issued to perform the call.
  iree_loop_callback_t callback;
  // Controls the scheduling of the call.
  iree_loop_priority_t priority;
} iree_loop_call_params_t;

// Parameters for IREE_LOOP_COMMAND_DISPATCH.
typedef struct iree_loop_dispatch_params_t {
  // Callback issued when the call completes (successfully or otherwise).
  iree_loop_callback_t callback;
  // Callback issued for each workgroup.
  iree_loop_workgroup_fn_t workgroup_fn;
  // 3D workgroup count.
  uint32_t workgroup_count_xyz[3];
} iree_loop_dispatch_params_t;

// Parameters for IREE_LOOP_COMMAND_WAIT_UTIL.
typedef struct iree_loop_wait_until_params_t {
  // Callback issued after the deadline has passed.
  iree_loop_callback_t callback;
  // Minimum time to wait before issueing the callback.
  iree_time_t deadline_ns;
} iree_loop_wait_until_params_t;

// Parameters for IREE_LOOP_COMMAND_WAIT_ONE.
typedef struct iree_loop_wait_one_params_t {
  // Callback issued after the wait condition is satisfied.
  iree_loop_callback_t callback;
  // Maximum time to wait before failing the wait with
  // IREE_STATUS_DEADLINE_EXCEEDED.
  iree_time_t deadline_ns;
  // Wait source to wait on.
  iree_wait_source_t wait_source;
} iree_loop_wait_one_params_t;

// Parameters for IREE_LOOP_COMMAND_WAIT_ANY / IREE_LOOP_COMMAND_WAIT_ALL.
typedef struct iree_loop_wait_multi_params_t {
  // Callback issued after any/all wait conditions are satisfied.
  iree_loop_callback_t callback;
  // Maximum time to wait before failing the wait with
  // IREE_STATUS_DEADLINE_EXCEEDED.
  iree_time_t deadline_ns;
  // Total number of wait sources.
  iree_host_size_t count;
  // List of wait source to wait on.
  // Ownership remains with the issuer and must remain live until the callback.
  iree_wait_source_t* wait_sources;
} iree_loop_wait_multi_params_t;

// Parameters for IREE_LOOP_COMMAND_DRAIN.
typedef struct iree_loop_drain_params_t {
  // Time when the wait will abort.
  iree_time_t deadline_ns;
} iree_loop_drain_params_t;

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_BASE_LOOP_H_
