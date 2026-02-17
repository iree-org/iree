// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_ASYNC_OPERATIONS_SEMAPHORE_H_
#define IREE_ASYNC_OPERATIONS_SEMAPHORE_H_

#include "iree/async/frontier.h"
#include "iree/async/operation.h"
#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_async_semaphore_t iree_async_semaphore_t;

//===----------------------------------------------------------------------===//
// Semaphore wait
//===----------------------------------------------------------------------===//

// Completes when one or all of the specified semaphores reach their target
// values (determined by |mode|). If any semaphore is failed, the operation
// completes with that failure status.
//
// Availability:
//   generic | io_uring | IOCP | kqueue
//   yes     | yes      | yes  | yes
//
// Wait mode support:
//   Mode   generic | io_uring | IOCP | kqueue
//   ──────────────────────────────────────────
//   ALL    yes     | yes      | yes  | yes
//   ANY    yes     | futex*   | yes  | yes
//
//   * io_uring uses futex_waitv (5.16+) for efficient multi-wait when
//     available; falls back to per-semaphore polling otherwise.
//
// Wait modes:
//   IREE_ASYNC_WAIT_MODE_ALL: Completes when ALL semaphores reach their
//     target values. |satisfied_index| is not meaningful in this mode.
//   IREE_ASYNC_WAIT_MODE_ANY: Completes when ANY semaphore reaches its
//     target value. |satisfied_index| indicates which semaphore was
//     satisfied first (useful for select-style dispatch).
//
// Failure propagation:
//   If any semaphore enters a failed state (e.g., device loss), the wait
//   completes immediately with that semaphore's failure status.
//
// Variable-size allocation:
//   Use iree_async_semaphore_wait_operation_size() for slab allocation,
//   or point |semaphores| and |values| at caller-managed storage.
//
// Example (wait for GPU work to complete):
//   iree_async_semaphore_wait_operation_t wait = {0};
//   wait.base.type = IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_WAIT;
//   wait.base.completion_fn = on_gpu_complete;
//   wait.semaphores = &gpu_semaphore;
//   wait.values = &target_value;
//   wait.count = 1;
//   wait.mode = IREE_ASYNC_WAIT_MODE_ALL;
//   iree_async_proactor_submit_one(proactor, &wait.base);
typedef struct iree_async_semaphore_wait_operation_t {
  iree_async_operation_t base;

  // Parallel arrays: semaphores[i] must reach values[i].
  // Points to trailing slab data or caller-managed storage.
  iree_async_semaphore_t** semaphores;
  uint64_t* values;
  iree_host_size_t count;

  // Whether to wait for ALL or ANY.
  iree_async_wait_mode_t mode;

  // Result (for ANY mode): index of the first semaphore that was satisfied.
  // For ALL mode, this field is not meaningful.
  iree_host_size_t satisfied_index;
} iree_async_semaphore_wait_operation_t;

// Computes the total allocation size for a wait operation with |count|
// semaphores using overflow-checked arithmetic. Includes the struct and
// trailing semaphore pointer + value arrays.
static inline iree_status_t iree_async_semaphore_wait_operation_size(
    iree_host_size_t count, iree_host_size_t* out_size) {
  return IREE_STRUCT_LAYOUT(
      sizeof(iree_async_semaphore_wait_operation_t), out_size,
      IREE_STRUCT_FIELD_FAM(count, iree_async_semaphore_t*),
      IREE_STRUCT_FIELD_FAM(count, uint64_t));
}

// Initializes a slab-allocated wait operation. Sets |semaphores| and |values|
// to point at trailing data within the slab.
static inline void iree_async_semaphore_wait_operation_initialize(
    iree_async_semaphore_wait_operation_t* operation, iree_host_size_t count,
    iree_async_wait_mode_t mode) {
  uint8_t* trailing = (uint8_t*)operation + sizeof(*operation);
  operation->semaphores = (iree_async_semaphore_t**)trailing;
  trailing += count * sizeof(iree_async_semaphore_t*);
  operation->values = (uint64_t*)trailing;
  operation->count = count;
  operation->mode = mode;
  operation->satisfied_index = 0;
}

//===----------------------------------------------------------------------===//
// Semaphore signal
//===----------------------------------------------------------------------===//

// Signals one or more semaphores to specified values.
//
// Completes after all signals are applied. Useful as a sequence step to
// coordinate GPU/network handoffs without user-space round-trips between
// the I/O completion and the signal.
//
// Availability:
//   generic | io_uring | IOCP | kqueue
//   yes     | yes      | yes  | yes
//
// Frontier propagation:
//   Each signal carries the operation's causal frontier, enabling the
//   remoting layer to propagate ordering guarantees across machines. This
//   is how distributed semaphore semantics are maintained—a signal on one
//   machine carries enough information for the remote to reconstruct the
//   causal ordering.
//
// Use in sequences:
//   Signal operations are commonly the final step in linked sequences:
//     RECV → SIGNAL
//   This allows a network receive to directly wake GPU work waiting on
//   the semaphore without returning to userspace.
//
// Variable-size allocation:
//   Use iree_async_semaphore_signal_operation_size() for slab allocation,
//   or point |semaphores| and |values| at caller-managed storage.
//
// Example (signal after recv completes):
//   iree_async_sequence_operation_t* seq = ...;
//   seq->steps[0] = &recv_op.base;
//   seq->steps[1] = &signal_op.base;
//   // On io_uring 5.3+, this executes as linked SQEs with no userspace
//   // round-trip between recv completion and semaphore signal.
typedef struct iree_async_semaphore_signal_operation_t {
  iree_async_operation_t base;

  // Parallel arrays: semaphores[i] is signaled to values[i].
  // Points to trailing slab data or caller-managed storage.
  iree_async_semaphore_t** semaphores;
  uint64_t* values;
  iree_host_size_t count;

  // Causal context passed to each signal call. May be NULL for local-only
  // signals. Not owned by this operation—must remain valid until the
  // operation completes. Typically points to inline frontier data in the
  // protocol message or to a stack-allocated frontier.
  const iree_async_frontier_t* frontier;
} iree_async_semaphore_signal_operation_t;

// Computes the total allocation size for a signal operation with |count|
// semaphores using overflow-checked arithmetic. Includes the struct and
// trailing semaphore pointer + value arrays.
static inline iree_status_t iree_async_semaphore_signal_operation_size(
    iree_host_size_t count, iree_host_size_t* out_size) {
  return IREE_STRUCT_LAYOUT(
      sizeof(iree_async_semaphore_signal_operation_t), out_size,
      IREE_STRUCT_FIELD_FAM(count, iree_async_semaphore_t*),
      IREE_STRUCT_FIELD_FAM(count, uint64_t));
}

// Initializes a slab-allocated signal operation. Sets |semaphores| and
// |values| to point at trailing data within the slab.
static inline void iree_async_semaphore_signal_operation_initialize(
    iree_async_semaphore_signal_operation_t* operation,
    iree_host_size_t count) {
  uint8_t* trailing = (uint8_t*)operation + sizeof(*operation);
  operation->semaphores = (iree_async_semaphore_t**)trailing;
  trailing += count * sizeof(iree_async_semaphore_t*);
  operation->values = (uint64_t*)trailing;
  operation->count = count;
  operation->frontier = NULL;
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_ASYNC_OPERATIONS_SEMAPHORE_H_
