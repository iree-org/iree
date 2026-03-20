// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Thread-based poll runner factory for the proactor pool.
//
// Creates a dedicated poll thread for each proactor, driving the proactor's
// poll loop automatically. This is the standard runner for native platforms
// (Linux, macOS, Windows) where C threads are available.
//
// The factory is returned by iree_async_proactor_pool_thread_runner_factory()
// and injected into pool options. The pool calls the factory callbacks during
// pool_get() (create) and pool_release() (request_stop, destroy).

#ifndef IREE_ASYNC_UTIL_PROACTOR_THREAD_RUNNER_H_
#define IREE_ASYNC_UTIL_PROACTOR_THREAD_RUNNER_H_

#include "iree/async/util/proactor_pool_types.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Returns a runner factory that creates a proactor_thread for each proactor.
// Threads are configured with NUMA affinity based on the pool entry's node_id.
//
// This is the default runner factory on native platforms. It is set
// automatically by iree_async_proactor_pool_options_default() when
// IREE_ASYNC_PROACTOR_POOL_HAVE_RUNNER_THREAD is defined.
iree_async_proactor_pool_runner_factory_t
iree_async_proactor_pool_thread_runner_factory(void);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_ASYNC_UTIL_PROACTOR_THREAD_RUNNER_H_
