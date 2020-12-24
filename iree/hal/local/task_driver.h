// Copyright 2020 Google LLC
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

#ifndef IREE_HAL_LOCAL_TASK_DRIVER_H_
#define IREE_HAL_LOCAL_TASK_DRIVER_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/local/executable_loader.h"
#include "iree/hal/local/task_device.h"
#include "iree/task/executor.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates a new iree/task/-based local CPU driver that creates devices sharing
// the same |executor| for scheduling tasks. |loaders| is the set of executable
// loaders that are available for loading in each device context.
iree_status_t iree_hal_task_driver_create(
    iree_string_view_t identifier,
    const iree_hal_task_device_params_t* default_params,
    iree_task_executor_t* executor, iree_host_size_t loader_count,
    iree_hal_executable_loader_t** loaders, iree_allocator_t host_allocator,
    iree_hal_driver_t** out_driver);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_LOCAL_TASK_DRIVER_H_
