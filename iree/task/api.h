// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TASK_API_H_
#define IREE_TASK_API_H_

#include "iree/base/api.h"
#include "iree/task/executor.h"  // IWYU pragma: export
#include "iree/task/topology.h"  // IWYU pragma: export

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Task system factory functions
//===----------------------------------------------------------------------===//

// Creates a task system executor from the current command line flags.
// This configures a topology and all of the executor parameters and returns
// a newly created instance in |out_executor| that must be released by the
// caller.
//
// This utility method is useful when only a single executor exists within a
// process as the flags are globals. When multiple executors may exist or
// programmatic configuration is needed use the iree_task_executor_create method
// directly.
iree_status_t iree_task_executor_create_from_flags(
    iree_allocator_t host_allocator, iree_task_executor_t** out_executor);

//===----------------------------------------------------------------------===//
// Task system simple invocation utilities
//===----------------------------------------------------------------------===//

// TODO(benvanik): simple IO completion event callback.
// TODO(benvanik): simple async function call dispatch.
// TODO(benvanik): simple parallel-for grid-style function call dispatch.

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_TASK_API_H_
