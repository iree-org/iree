// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Types shared between proactor_pool.h and runner implementations.

#ifndef IREE_ASYNC_UTIL_PROACTOR_POOL_TYPES_H_
#define IREE_ASYNC_UTIL_PROACTOR_POOL_TYPES_H_

#include "iree/async/proactor.h"
#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Factory callbacks for creating poll runners that drive proactors.
//
// When a proactor is first accessed via pool_get(), the pool calls |create| to
// optionally create a runner that drives the proactor's poll loop. If the
// factory's |create| is NULL, no runner is created and the caller (or host
// event loop) is responsible for polling.
//
// The standard implementation (proactor_runner_thread.h) creates a dedicated
// poll thread for each proactor.
typedef struct iree_async_proactor_pool_runner_factory_t {
  void* user_data;

  // Creates a poll runner for |proactor|.
  // |node_id| is the NUMA node for the proactor (UINT32_MAX if unspecified).
  // Stores an opaque runner handle in |out_runner| (may be NULL if the runner
  // is self-managing). Called under pool mutex — must not call back into the
  // pool.
  iree_status_t (*create)(void* user_data, iree_async_proactor_t* proactor,
                          uint32_t node_id, iree_allocator_t allocator,
                          void** out_runner);

  // Requests all runners to stop. Called once before any destroy calls.
  // Non-blocking: signals each runner to stop but does not wait.
  // |runners| and |count| are the opaque handles returned by create.
  // NULL entries are skipped.
  void (*request_stop)(void* user_data, void** runners, iree_host_size_t count);

  // Destroys a single runner, blocking until it has fully stopped.
  // Called after request_stop has been called for all runners.
  void (*destroy)(void* user_data, void* runner);
} iree_async_proactor_pool_runner_factory_t;

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_ASYNC_UTIL_PROACTOR_POOL_TYPES_H_
