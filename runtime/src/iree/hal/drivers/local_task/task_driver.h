// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_LOCAL_TASK_TASK_DRIVER_H_
#define IREE_HAL_DRIVERS_LOCAL_TASK_TASK_DRIVER_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/local_task/task_device.h"
#include "iree/hal/local/executable_loader.h"
#include "iree/task/executor.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates a new iree/task/-based local CPU driver that creates devices sharing
// the provided executors for scheduling tasks.
//
// |queue_count| specifies the number of logical device queues exposed to
// programs with one entry in |queue_executors| providing the scheduling scope.
// Multiple queues may share the same executor. When multiple executors are used
// queries for device capabilities will always report from the first.
//
// |loaders| is the set of executable loaders that are available for loading in
// the device context. The loaders are retained for the lifetime of the device.
iree_status_t iree_hal_task_driver_create(
    iree_string_view_t identifier,
    const iree_hal_task_device_params_t* default_params,
    iree_host_size_t queue_count, iree_task_executor_t* const* queue_executors,
    iree_host_size_t loader_count, iree_hal_executable_loader_t** loaders,
    iree_hal_allocator_t* device_allocator, iree_allocator_t host_allocator,
    iree_hal_driver_t** out_driver);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_LOCAL_TASK_TASK_DRIVER_H_
