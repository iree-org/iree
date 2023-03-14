// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BASE_LOOP_SYNC_H_
#define IREE_BASE_LOOP_SYNC_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_loop_sync_t
//===----------------------------------------------------------------------===//

// Configuration options for the synchronous loop implementation.
typedef struct iree_loop_sync_options_t {
  // Specifies the maximum operation queue depth in number of operations.
  // Growth is not currently supported and if the capacity is reached during
  // execution then IREE_STATUS_RESOURCE_EXHAUSTED will be returned when new
  // operations are enqueued.
  iree_host_size_t max_queue_depth;

  // Specifies how many pending waits are allowed at the same time.
  // Growth is not currently supported and if the capacity is reached during
  // execution then IREE_STATUS_RESOURCE_EXHAUSTED will be returned when new
  // waits are enqueued.
  iree_host_size_t max_wait_count;
} iree_loop_sync_options_t;

// A lightweight loop that greedily runs operations as they are available.
// This does not require any system threading support and has deterministic
// behavior unless multi-waits are used.
//
// Thread-compatible: the loop only performs work when iree_loop_drain is
// called and must not be used from multiple threads concurrently.
typedef struct iree_loop_sync_t iree_loop_sync_t;

// Allocates a synchronous loop using |allocator| stored into |out_loop_sync|.
IREE_API_EXPORT iree_status_t iree_loop_sync_allocate(
    iree_loop_sync_options_t options, iree_allocator_t allocator,
    iree_loop_sync_t** out_loop_sync);

// Frees a synchronous |loop_sync|, aborting all pending operations.
IREE_API_EXPORT void iree_loop_sync_free(iree_loop_sync_t* loop_sync);

// Waits until the loop is idle (all operations in all scopes have retired).
// Returns IREE_STATUS_DEADLINE_EXCEEDED if |timeout| is reached before the
// loop is idle.
IREE_API_EXPORT iree_status_t
iree_loop_sync_wait_idle(iree_loop_sync_t* loop_sync, iree_timeout_t timeout);

// Handles scope errors returned from loop callback operations.
// Ownership of |status| is passed to the handler and must be freed.
// All operations of the same scope will be aborted.
typedef void(IREE_API_PTR* iree_loop_sync_error_fn_t)(void* user_data,
                                                      iree_status_t status);

// A scope of execution within a loop.
// Each scope has a dedicated error handler that is notified when an error
// propagates from a loop operation scheduled against the scope. When an error
// arises all other operations in the same scope will be aborted.
typedef struct iree_loop_sync_scope_t {
  // Target loop for execution.
  iree_loop_sync_t* loop_sync;

  // Total number of pending operations in the scope.
  // When 0 the scope is considered idle.
  int32_t pending_count;

  // Optional function used to report errors that occur during execution.
  iree_loop_sync_error_fn_t error_fn;
  void* error_user_data;
} iree_loop_sync_scope_t;

// Initializes a loop scope that runs operations against |loop_sync|.
IREE_API_EXPORT void iree_loop_sync_scope_initialize(
    iree_loop_sync_t* loop_sync, iree_loop_sync_error_fn_t error_fn,
    void* error_user_data, iree_loop_sync_scope_t* out_scope);

// Deinitializes a loop |scope| and aborts any pending operations.
IREE_API_EXPORT void iree_loop_sync_scope_deinitialize(
    iree_loop_sync_scope_t* scope);

IREE_API_EXPORT iree_status_t iree_loop_sync_ctl(void* self,
                                                 iree_loop_command_t command,
                                                 const void* params,
                                                 void** inout_ptr);

// Returns a loop that schedules operations against |scope|.
// The scope must remain valid until all operations scheduled against it have
// completed.
static inline iree_loop_t iree_loop_sync_scope(iree_loop_sync_scope_t* scope) {
  iree_loop_t loop = {
      scope,
      iree_loop_sync_ctl,
  };
  return loop;
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_BASE_LOOP_SYNC_H_
