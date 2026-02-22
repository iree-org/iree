// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Continuation dispatcher for LINKED operation chains.
//
// When an operation with IREE_ASYNC_OPERATION_FLAG_LINKED completes, its
// continuation operations must be handled based on the result:
//   - On success: Submit the continuations to the proactor.
//   - On failure: Invoke continuation callbacks directly with CANCELLED.
//
// This utility provides the common dispatch logic used by all proactor
// backends for SEMAPHORE_WAIT and SEMAPHORE_SIGNAL continuations.
//
// Note: TIMER continuations use a different pattern on some backends (submit
// then cancel to get proper CQEs) and are handled separately.

#ifndef IREE_ASYNC_UTIL_CONTINUATION_H_
#define IREE_ASYNC_UTIL_CONTINUATION_H_

#include "iree/async/operation.h"
#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Continuation list
//===----------------------------------------------------------------------===//

// Reference to a list of continuation operations.
//
// Points into caller-owned storage (typically an operation's platform union or
// a wait tracker struct). The dispatcher clears |count| before dispatching to
// prevent re-entrancy issues with recursive submits.
typedef struct iree_async_continuation_list_t {
  // Array of continuation operations (not owned).
  iree_async_operation_t** operations;
  // First valid index in the array.
  iree_host_size_t start;
  // Number of continuation operations.
  iree_host_size_t count;
} iree_async_continuation_list_t;

// Returns an empty continuation list.
static inline iree_async_continuation_list_t iree_async_continuation_list_empty(
    void) {
  iree_async_continuation_list_t list = {NULL, 0, 0};
  return list;
}

// Constructs a continuation list from the given arguments.
static inline iree_async_continuation_list_t iree_async_make_continuation_list(
    iree_async_operation_t** operations, iree_host_size_t start,
    iree_host_size_t count) {
  iree_async_continuation_list_t list = {operations, start, count};
  return list;
}

// Returns true if the continuation list is empty.
static inline bool iree_async_continuation_list_is_empty(
    const iree_async_continuation_list_t* list) {
  return list->count == 0;
}

//===----------------------------------------------------------------------===//
// Continuation submit function
//===----------------------------------------------------------------------===//

// Function pointer type for submitting operations to a proactor.
//
// The implementation should be the backend's submit function (or a wrapper).
// |context| is backend-specific (typically the proactor pointer).
// Returns OK on success, or an error if submission failed.
typedef iree_status_t (*iree_async_continuation_submit_fn_t)(
    void* context, iree_async_operation_list_t operations);

//===----------------------------------------------------------------------===//
// Continuation dispatch
//===----------------------------------------------------------------------===//

// Dispatches continuation operations based on the triggering operation's
// result.
//
// On success (|status| is OK):
//   Submits continuation operations via |submit_fn|. If submit fails, invokes
//   callbacks directly with the submit error.
//
// On failure (|status| is not OK):
//   Invokes all continuation callbacks directly with CANCELLED.
//
// IMPORTANT: This function does NOT consume |status|. The caller retains
// ownership and must pass it to the triggering operation's completion callback.
// Continuations receive fresh CANCELLED statuses, not the original error.
//
// |submit_fn|: Backend's submit function.
// |submit_context|: Context for submit_fn (typically proactor pointer).
// |continuations|: List of continuations (count cleared before dispatch).
// |status|: Result of the triggering operation (not consumed).
//
// Returns the number of callbacks invoked directly (not via CQE). This is used
// for completion counting on backends that track completions.
//
// Thread-safety: Must be called from the proactor's poll thread.
iree_host_size_t iree_async_continuation_dispatch(
    iree_async_continuation_submit_fn_t submit_fn, void* submit_context,
    iree_async_continuation_list_t* continuations, iree_status_t status);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_ASYNC_UTIL_CONTINUATION_H_
