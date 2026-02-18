// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Base operation type for proactor-driven async I/O.
//
// All async operations extend iree_async_operation_t with type-specific
// parameters and result fields. Operations are caller-owned (intrusive — no
// proactor allocation on submit) and the proactor invokes the completion
// callback when the operation finishes.
//
// The ownership rule: the caller owns the operation storage after (and only
// after) the final completion callback fires. During execution the proactor
// may read/write any field. For multishot operations the callback fires
// multiple times; the caller regains ownership only on the final invocation
// (the one without IREE_ASYNC_COMPLETION_FLAG_MORE).
//
// Operation subtypes are defined in iree/async/operations/*.h. Each subtype
// embeds iree_async_operation_t as its first member for safe base-to-subtype
// casts via the type discriminator.

#ifndef IREE_ASYNC_OPERATION_H_
#define IREE_ASYNC_OPERATION_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_async_operation_t iree_async_operation_t;
typedef struct iree_async_operation_pool_t iree_async_operation_pool_t;

//===----------------------------------------------------------------------===//
// Completion callback
//===----------------------------------------------------------------------===//

// Flags passed to the completion callback alongside the status.
enum iree_async_completion_flag_bits_e {
  IREE_ASYNC_COMPLETION_FLAG_NONE = 0u,

  // More completions will follow for this operation (multishot).
  // The proactor retains ownership of the operation; the caller must not
  // modify, resubmit, or release it. When this flag is NOT set the callback
  // is final and the caller owns the operation again.
  IREE_ASYNC_COMPLETION_FLAG_MORE = 1u << 0,

  // Zero-copy was achieved for this send operation.
  // Set when the socket has IREE_ASYNC_SOCKET_OPTION_ZERO_COPY AND the kernel
  // actually achieved zero-copy (no fallback to copying). This allows callers
  // to monitor ZC effectiveness for diagnostics or adaptive behavior.
  // Only meaningful for socket send completions; other operations ignore this.
  IREE_ASYNC_COMPLETION_FLAG_ZERO_COPY_ACHIEVED = 1u << 1,
};
typedef uint32_t iree_async_completion_flags_t;

// Completion callback invoked by the proactor during poll().
//
// |user_data| is the value set on the operation at submit time.
// |operation| is the completed operation (cast to subtype via operation->type).
// |status| is the result: OK on success, CANCELLED on cancellation, or an
//   error status describing the failure. Ownership of the status is transferred
//   to the callback; the callback must consume or ignore it.
// |flags| is a bitmask of iree_async_completion_flag_e values.
//
// Callbacks fire from within iree_async_proactor_poll() on the polling thread.
// Heavy work should be deferred to avoid stalling completion dispatch.
typedef void (*iree_async_completion_fn_t)(void* user_data,
                                           iree_async_operation_t* operation,
                                           iree_status_t status,
                                           iree_async_completion_flags_t flags);

//===----------------------------------------------------------------------===//
// Operation type discriminator
//===----------------------------------------------------------------------===//

// Identifies the concrete subtype of an iree_async_operation_t for safe
// downcasting. Each value corresponds to a specific *_operation_t struct
// in iree/async/operations/.
enum iree_async_operation_type_e {
  // Scheduling and synchronization.
  IREE_ASYNC_OPERATION_TYPE_NOP = 0u,
  IREE_ASYNC_OPERATION_TYPE_TIMER,
  IREE_ASYNC_OPERATION_TYPE_EVENT_WAIT,
  IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_WAIT,
  IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_SIGNAL,
  IREE_ASYNC_OPERATION_TYPE_SEQUENCE,

  // Socket I/O.
  IREE_ASYNC_OPERATION_TYPE_SOCKET_ACCEPT,
  IREE_ASYNC_OPERATION_TYPE_SOCKET_CONNECT,
  IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV,
  IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV_POOL,
  IREE_ASYNC_OPERATION_TYPE_SOCKET_SEND,
  IREE_ASYNC_OPERATION_TYPE_SOCKET_SENDTO,
  IREE_ASYNC_OPERATION_TYPE_SOCKET_RECVFROM,
  IREE_ASYNC_OPERATION_TYPE_SOCKET_CLOSE,

  // File I/O.
  IREE_ASYNC_OPERATION_TYPE_FILE_OPEN,
  IREE_ASYNC_OPERATION_TYPE_FILE_READ,
  IREE_ASYNC_OPERATION_TYPE_FILE_WRITE,
  IREE_ASYNC_OPERATION_TYPE_FILE_CLOSE,

  // Futex operations (requires
  // IREE_ASYNC_PROACTOR_CAPABILITY_FUTEX_OPERATIONS).
  IREE_ASYNC_OPERATION_TYPE_FUTEX_WAIT,
  IREE_ASYNC_OPERATION_TYPE_FUTEX_WAKE,

  // Notification operations (uses futex or eventfd based on capabilities).
  IREE_ASYNC_OPERATION_TYPE_NOTIFICATION_WAIT,
  IREE_ASYNC_OPERATION_TYPE_NOTIFICATION_SIGNAL,

  // Cross-proactor messaging (requires
  // IREE_ASYNC_PROACTOR_CAPABILITY_PROACTOR_MESSAGING).
  IREE_ASYNC_OPERATION_TYPE_MESSAGE,
};
typedef uint8_t iree_async_operation_type_t;

//===----------------------------------------------------------------------===//
// Operation flags
//===----------------------------------------------------------------------===//

// Behavioral flags set by the caller before submitting an operation.
enum iree_async_operation_flag_bits_e {
  IREE_ASYNC_OPERATION_FLAG_NONE = 0u,

  // Request multishot behavior: the proactor will deliver completions
  // repeatedly until explicitly cancelled or the resource is closed.
  // Supported on: ACCEPT, RECV.
  // Unsupported operations will fail with IREE_STATUS_INVALID_ARGUMENT.
  IREE_ASYNC_OPERATION_FLAG_MULTISHOT = 1u << 0,

  // Link this operation to the next operation in the submission batch.
  // When set, the kernel will not begin the next operation until this one
  // completes successfully. If this operation fails or is cancelled, all
  // subsequent linked operations receive IREE_STATUS_CANCELLED.
  //
  // Semantics: "link TO next" - set on all operations EXCEPT the last in a
  // chain. The last operation must NOT have this flag set.
  //
  // Requires IREE_ASYNC_PROACTOR_CAPABILITY_LINKED_OPERATIONS capability.
  IREE_ASYNC_OPERATION_FLAG_LINKED = 1u << 1,
};
typedef uint32_t iree_async_operation_flags_t;

//===----------------------------------------------------------------------===//
// Wait mode
//===----------------------------------------------------------------------===//

// Determines when a multi-semaphore wait operation is satisfied.
enum iree_async_wait_mode_e {
  // Satisfied when ALL semaphores reach their target values.
  IREE_ASYNC_WAIT_MODE_ALL = 0u,
  // Satisfied when ANY semaphore reaches its target value.
  IREE_ASYNC_WAIT_MODE_ANY = 1u,
};
typedef uint8_t iree_async_wait_mode_t;

//===----------------------------------------------------------------------===//
// Internal flags
//===----------------------------------------------------------------------===//

// Proactor-private flags for operation state management during execution.
// These are not part of the public API; callers should not read or write them.
// Flag values are defined per-backend as enums (e.g.,
// iree_async_posix_operation_internal_flags_e in the POSIX proactor).
//
// The value type is used for flag constants; the struct field is atomic because
// cancel (any thread) sets CANCELLED while poll (single thread) reads it.
typedef uint32_t iree_async_operation_internal_flags_t;

//===----------------------------------------------------------------------===//
// Base operation
//===----------------------------------------------------------------------===//

// Base type for all async operations. Extended by subtype structs that embed
// this as their first member.
//
// Fields are written by the caller before submission:
//   type, flags, completion_fn, user_data, pool
//
// Fields are managed by the proactor during execution:
//   next, internal_flags, linked_next, submit_time_ns (tracing only)
//
// After the final completion callback, all fields are caller-owned again.
typedef struct iree_async_operation_t {
  // Intrusive linked list pointer (proactor-internal tracking).
  // Must not be touched by the caller while the operation is in flight.
  struct iree_async_operation_t* next;

  // Discriminates the concrete subtype for safe downcasting.
  iree_async_operation_type_t type;

  // Proactor-private flags for cancellation, iteration safety, etc.
  // Initialized to 0; callers must not access this field.
  // Atomic: cancel (any thread) writes CANCELLED, poll thread reads.
  iree_atomic_int32_t internal_flags;

  // Behavioral flags (MULTISHOT, etc.) set by caller before submit.
  iree_async_operation_flags_t flags;

  // Callback invoked on completion (from poll context).
  // Must not be NULL.
  iree_async_completion_fn_t completion_fn;
  void* user_data;

  // Optional: pool to release this operation to after the final callback.
  // If non-NULL, the proactor calls iree_async_operation_pool_release()
  // after the final callback returns. The caller must not access the
  // operation after the callback in this case.
  iree_async_operation_pool_t* pool;

  // LINKED chain continuation pointer (proactor-internal).
  // When this operation has IREE_ASYNC_OPERATION_FLAG_LINKED set, points to
  // the next operation in the chain. On completion, the proactor submits
  // continuations (on success) or cancels them (on failure).
  // Callers must not access this field.
  struct iree_async_operation_t* linked_next;

  // Tracing: timestamp of submission for latency measurement.
  IREE_TRACE(iree_time_t submit_time_ns;)
} iree_async_operation_t;

//===----------------------------------------------------------------------===//
// Operation list (batch submit)
//===----------------------------------------------------------------------===//

// A list of operations for batch submission.
// Maps to a single io_uring_submit() or kevent() call for efficiency.
// The values pointer and count must remain valid only for the duration of the
// submit call (the proactor copies or enqueues internally).
typedef struct iree_async_operation_list_t {
  iree_async_operation_t** values;
  iree_host_size_t count;
} iree_async_operation_list_t;

//===----------------------------------------------------------------------===//
// Resource retain/release
//===----------------------------------------------------------------------===//

// Retains resources referenced by an operation entering proactor management.
// Called at submit time to prevent premature destruction while the operation is
// in flight. Each retained resource gets exactly one reference increment.
//
// Close operations (SOCKET_CLOSE, FILE_CLOSE) are intentionally absent: they
// consume the caller's reference rather than retaining a new one. The
// corresponding release in iree_async_operation_release_resources IS the
// consumption, with no prior retain to balance it.
IREE_API_EXPORT void iree_async_operation_retain_resources(
    iree_async_operation_t* operation);

// Releases resources retained by iree_async_operation_retain_resources, plus
// consumes caller references for close operations. Called during completion
// dispatch, after linked continuation dispatch but before the user's callback.
// Must happen BEFORE the callback since the callback may free the operation.
IREE_API_EXPORT void iree_async_operation_release_resources(
    iree_async_operation_t* operation);

//===----------------------------------------------------------------------===//
// Inline helpers
//===----------------------------------------------------------------------===//

// Initializes the base fields of an operation.
// Caller must still fill subtype-specific fields after this call.
static inline void iree_async_operation_initialize(
    iree_async_operation_t* operation, iree_async_operation_type_t type,
    iree_async_operation_flags_t flags,
    iree_async_completion_fn_t completion_fn, void* user_data) {
  operation->next = NULL;
  operation->type = type;
  iree_atomic_store(&operation->internal_flags, 0, iree_memory_order_relaxed);
  operation->flags = flags;
  operation->completion_fn = completion_fn;
  operation->user_data = user_data;
  operation->pool = NULL;
  operation->linked_next = NULL;
  IREE_TRACE({ operation->submit_time_ns = 0; });
}

// Atomically loads the internal flags of an operation (acquire ordering).
static inline iree_async_operation_internal_flags_t
iree_async_operation_load_internal_flags(iree_async_operation_t* operation) {
  return (iree_async_operation_internal_flags_t)iree_atomic_load(
      &operation->internal_flags, iree_memory_order_acquire);
}

// Atomically sets (ORs) internal flags on an operation (release ordering).
// Used by cancel (any thread) to set CANCELLED, by poll thread to set state.
static inline void iree_async_operation_set_internal_flags(
    iree_async_operation_t* operation,
    iree_async_operation_internal_flags_t flags_to_set) {
  iree_atomic_fetch_or(&operation->internal_flags, (int32_t)flags_to_set,
                       iree_memory_order_release);
}

// Atomically clears (resets to 0) the internal flags of an operation.
// Used at submission time to prepare an operation for a fresh execution cycle.
static inline void iree_async_operation_clear_internal_flags(
    iree_async_operation_t* operation) {
  iree_atomic_store(&operation->internal_flags, 0, iree_memory_order_release);
}

// Creates an operation list from a pointer and count.
static inline iree_async_operation_list_t iree_async_operation_list_make(
    iree_async_operation_t** values, iree_host_size_t count) {
  iree_async_operation_list_t list;
  list.values = values;
  list.count = count;
  return list;
}

// Creates an operation list containing a single operation.
static inline iree_async_operation_list_t iree_async_operation_list_from_one(
    iree_async_operation_t* operation) {
  iree_async_operation_list_t list;
  list.values = &operation;
  list.count = 1;
  return list;
}

// Returns an empty operation list.
static inline iree_async_operation_list_t iree_async_operation_list_empty(
    void) {
  iree_async_operation_list_t list;
  list.values = NULL;
  list.count = 0;
  return list;
}

// Returns true if the operation list is empty.
static inline bool iree_async_operation_list_is_empty(
    iree_async_operation_list_t list) {
  return list.count == 0;
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_ASYNC_OPERATION_H_
