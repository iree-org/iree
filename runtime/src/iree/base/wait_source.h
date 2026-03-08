// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BASE_WAIT_SOURCE_H_
#define IREE_BASE_WAIT_SOURCE_H_

#include "iree/base/attributes.h"
#include "iree/base/status.h"
#include "iree/base/time.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_wait_source_t
//===----------------------------------------------------------------------===//

typedef struct iree_wait_source_t iree_wait_source_t;

// Callback invoked when a wait source resolves.
// |status| is OK if the wait source reached its target value, or the failure
// status if the underlying primitive failed. Ownership of |status| transfers
// to the callback (the callback must consume or ignore it).
typedef void (*iree_wait_source_resolve_callback_t)(void* user_data,
                                                    iree_status_t status);

// Resolves a wait source by checking the current state and either invoking
// |callback| synchronously or registering for asynchronous notification.
//
// |wait_source| carries the target object and value. |timeout| bounds blocking
// for synchronous callers; asynchronous implementations ignore it (the caller
// manages deadlines externally).
//
// When |callback| is NULL the function operates synchronously: it blocks (up
// to |timeout|) and returns OK when the condition is met, or
// IREE_STATUS_DEADLINE_EXCEEDED if it is not met within the timeout, or the
// failure status of the underlying primitive.
//
// When |callback| is non-NULL the function may operate asynchronously: it
// returns OK if the callback was invoked synchronously or was successfully
// registered for later notification (the callback WILL fire eventually).
// Returns an error if registration failed — the callback will NOT fire.
//
// The callback may fire before this function returns (if the condition is
// already satisfied or the primitive has already failed). Callers must be
// prepared for reentrant callback invocation.
typedef iree_status_t (*iree_wait_source_resolve_fn_t)(
    iree_wait_source_t wait_source, iree_timeout_t timeout,
    iree_wait_source_resolve_callback_t callback, void* user_data);

// A wait source represents a future point in time on some primitive
// (semaphore timeline value, delay deadline, etc.).
typedef struct iree_wait_source_t {
  // The object being waited on (e.g. a semaphore pointer).
  void* self;
  // Implementation-defined data (e.g. timeline value, deadline).
  uint64_t data;
  // Resolution function. NULL for immediate wait sources.
  iree_wait_source_resolve_fn_t resolve;
} iree_wait_source_t;

// Returns a wait source that will always immediately return as resolved.
static inline iree_wait_source_t iree_wait_source_immediate(void) {
  iree_wait_source_t v = {NULL, 0ull, NULL};
  return v;
}

// Returns true if the |wait_source| is immediately resolved.
// This can be used to neuter waits in lists/sets.
static inline bool iree_wait_source_is_immediate(
    iree_wait_source_t wait_source) {
  return wait_source.resolve == NULL;
}

// Resolve function for iree_wait_source_delay.
IREE_API_EXPORT iree_status_t iree_wait_source_delay_resolve(
    iree_wait_source_t wait_source, iree_timeout_t timeout,
    iree_wait_source_resolve_callback_t callback, void* user_data);

// Returns a wait source that indicates a delay until a point in time.
// The source will remain unresolved until the |deadline_ns| is reached or
// exceeded and afterward return resolved.
static inline iree_wait_source_t iree_wait_source_delay(
    iree_time_t deadline_ns) {
  iree_wait_source_t v = {
      NULL,
      (uint64_t)deadline_ns,
      iree_wait_source_delay_resolve,
  };
  return v;
}

// Returns true if the |wait_source| is a timed delay.
static inline bool iree_wait_source_is_delay(iree_wait_source_t wait_source) {
  return wait_source.resolve == iree_wait_source_delay_resolve;
}

// Queries the state of a |wait_source| without waiting.
// |out_wait_status_code| will indicate the status of the source while the
// returned value indicates the status of the query. |out_wait_status_code| will
// be set to IREE_STATUS_DEFERRED if the wait source has not yet resolved and
// IREE_STATUS_OK otherwise.
IREE_API_EXPORT iree_status_t iree_wait_source_query(
    iree_wait_source_t wait_source, iree_status_code_t* out_wait_status_code);

// Blocks the caller and waits for a |wait_source| to resolve.
// Returns IREE_STATUS_DEADLINE_EXCEEDED if |timeout| is reached before the
// wait source resolves. If the wait source resolved with a failure then the
// error status will be returned.
IREE_API_EXPORT iree_status_t iree_wait_source_wait_one(
    iree_wait_source_t wait_source, iree_timeout_t timeout);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_BASE_WAIT_SOURCE_H_
