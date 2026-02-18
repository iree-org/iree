// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_ASYNC_OPERATIONS_SCHEDULING_H_
#define IREE_ASYNC_OPERATIONS_SCHEDULING_H_

#include "iree/async/operation.h"
#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Nop
//===----------------------------------------------------------------------===//

// No-operation. Completes immediately on the next poll().
//
// Useful for:
//   - Testing: Verify callback dispatch without side effects.
//   - Sequence placeholder: Reserve a slot to be filled later.
//   - Poll context callback: Trigger application logic from the poll thread.
//   - Batch completion notification: Know when a batch of operations was
//     submitted (nop completes in submission order).
//
// Availability:
//   generic | io_uring | IOCP | kqueue
//   yes     | yes      | yes  | yes
//
// Performance:
//   Minimal overhead—no syscall, just queue manipulation. On io_uring,
//   translates to IORING_OP_NOP which completes without kernel work.
typedef struct iree_async_nop_operation_t {
  iree_async_operation_t base;
} iree_async_nop_operation_t;

//===----------------------------------------------------------------------===//
// Timer
//===----------------------------------------------------------------------===//

// Completes when the deadline is reached.
//
// Timers use absolute monotonic time to avoid drift from scheduling delays.
// On backends without native absolute timeout support, the deadline is
// converted to relative at submission time (introducing potential drift
// if submission is delayed).
//
// Availability:
//   generic | io_uring | IOCP | kqueue
//   yes     | yes      | yes  | yes
//
// Absolute timeout support:
//   generic | io_uring | IOCP | kqueue
//   emul    | 5.4+     | yes  | emul
//
// Threading model:
//   Callback fires on the poll thread when the deadline is reached (or
//   shortly after, depending on poll granularity and system load).
//
// Cancellation:
//   Timers may be cancelled via iree_async_proactor_cancel(). The callback
//   fires with IREE_STATUS_CANCELLED.
//
// Example:
//   iree_async_timer_operation_t timer = {0};
//   timer.base.type = IREE_ASYNC_OPERATION_TYPE_TIMER;
//   timer.base.completion_fn = on_timeout;
//   timer.deadline_ns = iree_time_now() + 5 * IREE_DURATION_SECOND;
//   iree_async_proactor_submit_one(proactor, &timer.base);
typedef struct iree_async_timer_operation_t {
  iree_async_operation_t base;

  // Absolute monotonic time at which the timer fires. Compute as
  // iree_time_now() + duration for consistent behavior across all backends.
  // Use IREE_DURATION_* constants for readability.
  iree_time_t deadline_ns;

  // Platform-specific storage. Each proactor backend uses a different member:
  // - io_uring: timespec for kernel timeout (struct __kernel_timespec layout)
  // - POSIX (poll/epoll/kqueue): posix for intrusive timer list linkage
  // - IOCP: iocp for intrusive timer list linkage
  // Callers should zero-initialize and not access these fields.
  union {
    // io_uring: timespec for kernel timeout. Pointed to by the SQE's addr
    // field and must remain valid until the CQE fires.
    struct {
      int64_t tv_sec;
      int64_t tv_nsec;
    } timespec;
    // POSIX proactor: intrusive doubly-linked list pointers for userspace
    // timer management. The proactor maintains a sorted list by deadline;
    // this is mutually exclusive with io_uring's timespec usage.
    struct {
      struct iree_async_timer_operation_t* next;
      struct iree_async_timer_operation_t* prev;
    } posix;
    // IOCP proactor: intrusive doubly-linked list pointers for userspace
    // timer management. Same algorithm as POSIX (sorted by deadline).
    struct {
      struct iree_async_timer_operation_t* next;
      struct iree_async_timer_operation_t* prev;
    } iocp;
  } platform;
} iree_async_timer_operation_t;

//===----------------------------------------------------------------------===//
// Event wait
//===----------------------------------------------------------------------===//

typedef struct iree_async_event_t iree_async_event_t;

// Waits for an event to become signaled. Completes when the event is set.
//
// Events are lightweight, cross-thread signaling primitives. Use this
// operation to integrate event-based synchronization into async I/O flows.
//
// Availability:
//   generic | io_uring | IOCP | kqueue
//   yes     | yes      | yes  | yes
//
// Implementation:
//   io_uring: IORING_OP_POLL_ADD on the event's eventfd.
//   IOCP: Thread pool wait or completion port association.
//   kqueue: EVFILT_USER.
//   generic: poll/select on the event's fd.
//
// Threading model:
//   Callback fires on the poll thread when the event is signaled.
//   The event may be signaled from any thread via iree_async_event_set().
//
// Lifetime:
//   The event must remain valid until the operation completes. The proactor
//   retains the event during execution to prevent premature destruction.
typedef struct iree_async_event_wait_operation_t {
  iree_async_operation_t base;

  // The event to wait on. The event must remain valid until the operation
  // completes. The proactor retains the event during execution to prevent
  // premature destruction.
  iree_async_event_t* event;
} iree_async_event_wait_operation_t;

//===----------------------------------------------------------------------===//
// Sequence
//===----------------------------------------------------------------------===//

// Callback invoked between sequence steps.
//
// |user_data| is the base operation's user_data (same context as the
//   completion callback). If the sequence is embedded in a larger struct,
//   use container_of to recover the enclosing context.
// |completed_step| is the step that just finished (inspect results here).
// |next_step| is the next step to execute (NULL if this was the last step).
//   The callback may modify next_step's parameters based on completed_step's
//   results before the proactor submits it.
//
// Returns OK to continue the sequence, or an error to abort (the sequence's
// base callback fires with that error).
typedef iree_status_t (*iree_async_step_fn_t)(
    void* user_data, iree_async_operation_t* completed_step,
    iree_async_operation_t* next_step);

// Chains multiple operations into a pipeline.
//
// The proactor executes steps in order, advancing when each completes.
// This enables patterns like "wait for semaphore → recv → signal semaphore"
// to be expressed as a single logical operation.
//
// Availability:
//   generic | io_uring | IOCP | kqueue
//   yes     | yes      | yes  | yes
//
// Linked SQE optimization (io_uring):
//   Linked SQE support (step_fn == NULL):
//     generic | io_uring | IOCP | kqueue
//     emul    | 5.3+     | emul | emul
//
//   When |step_fn| is NULL and all steps are pre-filled, the io_uring backend
//   submits all steps as linked SQEs (IOSQE_IO_LINK) for kernel-chained
//   execution with no user-space round-trips between steps. This is the
//   optimal path for pre-planned pipelines.
//
//   When |step_fn| is set, the proactor calls it between steps (one poll
//   round-trip per step), allowing dynamic construction of later steps based
//   on earlier results.
//
// Lifecycle:
//   - All steps complete successfully → base callback fires with OK.
//   - Any step fails → remaining steps are skipped (or short-circuited in
//     linked mode), base callback fires with that step's error.
//   - Cancelled → current step is cancelled, base callback fires with
//     IREE_STATUS_CANCELLED.
//
// Variable-size allocation:
//   Use iree_async_sequence_operation_size() to compute the total slab size
//   including trailing step pointer storage, then initialize with
//   iree_async_sequence_operation_initialize(). Alternatively, embed the
//   struct in a caller-defined type and point |steps| at caller-managed
//   storage.
//
// Example (pre-planned pipeline for linked SQE optimization):
//   iree_host_size_t size = 0;
//   IREE_RETURN_IF_ERROR(iree_async_sequence_operation_size(3, &size));
//   iree_async_sequence_operation_t* seq = slab_alloc(size);
//   iree_async_sequence_operation_initialize(seq, 3, NULL);
//   seq->steps[0] = &wait_op.base;   // Wait for semaphore.
//   seq->steps[1] = &recv_op.base;   // Recv data.
//   seq->steps[2] = &signal_op.base; // Signal completion.
//   seq->base.completion_fn = on_pipeline_complete;
//   iree_async_proactor_submit_one(proactor, &seq->base);
typedef struct iree_async_sequence_operation_t {
  iree_async_operation_t base;

  // Array of step operations to execute in order. Points to trailing data
  // when slab-allocated via _initialize(), or to caller-managed storage
  // when the struct is embedded in a larger type.
  iree_async_operation_t** steps;
  iree_host_size_t step_count;

  // Current step index (advanced by the proactor during execution).
  iree_host_size_t current_step;

  // Optional inter-step callback. If NULL, io_uring may use linked SQEs.
  iree_async_step_fn_t step_fn;

  // Internal state managed exclusively by the sequence_emulation
  // implementation during execution. Callers must not access these fields.
  union {
    // Emulation path (step_fn != NULL): pointer to the backend's emulator.
    void* emulator;
    // LINK path (step_fn == NULL): first error status from a failing step,
    // buffered until all step CQEs are processed.
    iree_status_t stashed_error;
  } internal;
} iree_async_sequence_operation_t;

// Computes the total allocation size needed for a sequence with |step_count|
// steps using overflow-checked arithmetic. Includes the struct and trailing
// step pointer storage.
static inline iree_status_t iree_async_sequence_operation_size(
    iree_host_size_t step_count, iree_host_size_t* out_size) {
  return IREE_STRUCT_LAYOUT(
      sizeof(iree_async_sequence_operation_t), out_size,
      IREE_STRUCT_FIELD_FAM(step_count, iree_async_operation_t*));
}

// Initializes a slab-allocated sequence operation. Sets |steps| to point at
// the trailing data within the slab. The slab must be at least
// iree_async_sequence_operation_size(step_count) bytes.
// Caller must still fill the step pointers and set base fields.
static inline void iree_async_sequence_operation_initialize(
    iree_async_sequence_operation_t* sequence, iree_host_size_t step_count,
    iree_async_step_fn_t step_fn) {
  sequence->steps =
      (iree_async_operation_t**)((uint8_t*)sequence +
                                 sizeof(iree_async_sequence_operation_t));
  sequence->step_count = step_count;
  sequence->current_step = 0;
  sequence->step_fn = step_fn;
}

//===----------------------------------------------------------------------===//
// Notification wait
//===----------------------------------------------------------------------===//

typedef struct iree_async_notification_t iree_async_notification_t;

// Waits for a notification to be signaled.
// The wait completes when the notification's epoch advances past the captured
// token (signal was called after the wait was submitted).
//
// Availability:
//   generic | io_uring | IOCP | kqueue
//   yes     | yes      | yes  | yes
//
// Implementation:
//   io_uring 6.7+: IORING_OP_FUTEX_WAIT on epoch word.
//   io_uring <6.7: Linked POLL_ADD + READ on eventfd.
//   Others: Platform-specific (eventfd + poll, WaitOnAddress, etc.).
//
// Threading model:
//   Callback fires on the poll thread when the notification is signaled.
//   The notification may be signaled from any thread via
//   iree_async_notification_signal().
//
// Lifetime:
//   The notification must remain valid until the operation completes. The
//   proactor retains the notification during execution to prevent premature
//   destruction.
typedef struct iree_async_notification_wait_operation_t {
  iree_async_operation_t base;

  // The notification to wait on. Retained by the proactor during execution.
  iree_async_notification_t* notification;

  // Platform-internal: epoch token captured at submit time.
  // Used to detect signals that occur after submit.
  uint32_t wait_token;
} iree_async_notification_wait_operation_t;

//===----------------------------------------------------------------------===//
// Notification signal
//===----------------------------------------------------------------------===//

// Signals a notification, waking up to wake_count waiters.
// Use INT32_MAX to wake all waiters (broadcast).
//
// Availability:
//   generic | io_uring | IOCP | kqueue
//   yes     | yes      | yes  | yes
//
// Implementation:
//   io_uring 6.7+: IORING_OP_FUTEX_WAKE on epoch word.
//   io_uring <6.7: IORING_OP_WRITE to eventfd.
//   Others: Platform-specific (eventfd write, SetEvent, etc.).
//
// Threading model:
//   Callback fires on the poll thread after the signal is processed.
//   The signal itself happens synchronously in kernel space before the
//   CQE is posted, so woken waiters may begin running before the
//   signal operation's callback fires.
//
// Use in LINK chains:
//   NOTIFICATION_SIGNAL is commonly used as the final step in a linked
//   sequence:
//     RECV -> NOTIFICATION_SIGNAL
//   This wakes waiting consumer threads without returning to userspace
//   between the I/O completion and the wake.
typedef struct iree_async_notification_signal_operation_t {
  iree_async_operation_t base;

  // The notification to signal. Retained by the proactor during execution.
  iree_async_notification_t* notification;

  // Maximum number of waiters to wake. Common values:
  //   1: Wake a single waiter (e.g., producer/consumer handoff)
  //   INT32_MAX: Wake all waiters (broadcast)
  int32_t wake_count;

  // Result: actual number of waiters woken. Populated on completion.
  // May be less than wake_count if fewer waiters were blocked.
  int32_t woken_count;

  // Platform-internal: buffer for eventfd write in event mode.
  uint64_t write_value;
} iree_async_notification_signal_operation_t;

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_ASYNC_OPERATIONS_SCHEDULING_H_
