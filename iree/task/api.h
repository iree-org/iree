// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef IREE_TASK_API_H_
#define IREE_TASK_API_H_

#include "iree/task/executor.h"
#include "iree/task/topology.h"

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
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_task_executor_create_from_flags(iree_allocator_t host_allocator,
                                     iree_task_executor_t** out_executor);

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
