// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/local_task/registration/driver_module.h"

#include <inttypes.h>
#include <stddef.h>

#include "iree/base/api.h"
#include "iree/base/internal/flags.h"
#include "iree/hal/drivers/local_task/task_driver.h"
#include "iree/hal/local/loaders/registration/init.h"
#include "iree/hal/local/plugins/registration/init.h"
#include "iree/task/api.h"

IREE_FLAG(
    bool, task_abort_on_failure, false,
    "Aborts the program on the first failure within a task system queue.");

static iree_status_t iree_hal_local_task_driver_factory_enumerate(
    void* self, iree_host_size_t* out_driver_info_count,
    const iree_hal_driver_info_t** out_driver_infos) {
  static const iree_hal_driver_info_t driver_infos[1] = {
      {
          .driver_name = IREE_SVL("local-task"),
          .full_name = IREE_SVL("Local execution using the "
                                "IREE multithreading task system"),
      },
  };
  *out_driver_info_count = IREE_ARRAYSIZE(driver_infos);
  *out_driver_infos = driver_infos;
  return iree_ok_status();
}

static iree_status_t iree_hal_local_task_driver_factory_try_create(
    void* self, iree_string_view_t driver_name, iree_allocator_t host_allocator,
    iree_hal_driver_t** out_driver) {
  if (!iree_string_view_equal(driver_name, IREE_SV("local-task"))) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "no driver '%.*s' is provided by this factory",
                            (int)driver_name.size, driver_name.data);
  }

  iree_hal_task_device_params_t default_params;
  iree_hal_task_device_params_initialize(&default_params);
  if (FLAG_task_abort_on_failure) {
    default_params.queue_scope_flags |= IREE_TASK_SCOPE_FLAG_ABORT_ON_FAILURE;
  }

  // Create executors for each topology specified by flags.
  // Stack allocated storage today but we can query for the total count and
  // grow if needed in the future (16 NUMA nodes is enough for anyone, right?).
  iree_task_executor_t* executor_storage[16] = {NULL};
  iree_task_executor_t** executors = executor_storage;
  iree_host_size_t executor_count = 0;
  IREE_RETURN_IF_ERROR(iree_task_executors_create_from_flags(
      host_allocator, IREE_ARRAYSIZE(executor_storage), executors,
      &executor_count));

  iree_hal_executable_plugin_manager_t* plugin_manager = NULL;
  iree_status_t status = iree_hal_executable_plugin_manager_create_from_flags(
      host_allocator, &plugin_manager);

  // Create all executable loaders linked into the binary.
  iree_hal_executable_loader_t* loaders[8] = {NULL};
  iree_host_size_t loader_count = 0;
  if (iree_status_is_ok(status)) {
    status = iree_hal_create_all_available_executable_loaders(
        plugin_manager, IREE_ARRAYSIZE(loaders), &loader_count, loaders,
        host_allocator);
  }

  // TODO(benvanik): allow this to be injected to share across drivers.
  iree_hal_allocator_t* device_allocator = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_allocator_create_heap(iree_make_cstring_view("local"),
                                            host_allocator, host_allocator,
                                            &device_allocator);
  }

  // Create a task driver that will use the given executors for scheduling work
  // and loaders for loading executables.
  if (iree_status_is_ok(status)) {
    status = iree_hal_task_driver_create(
        driver_name, &default_params, executor_count, executors, loader_count,
        loaders, device_allocator, host_allocator, out_driver);
  }

  for (iree_host_size_t i = 0; i < executor_count; ++i) {
    iree_task_executor_release(executors[i]);
  }
  for (iree_host_size_t i = 0; i < loader_count; ++i) {
    iree_hal_executable_loader_release(loaders[i]);
  }
  iree_hal_executable_plugin_manager_release(plugin_manager);
  iree_hal_allocator_release(device_allocator);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_local_task_driver_module_register(
    iree_hal_driver_registry_t* registry) {
  static const iree_hal_driver_factory_t factory = {
      .self = NULL,
      .enumerate = iree_hal_local_task_driver_factory_enumerate,
      .try_create = iree_hal_local_task_driver_factory_try_create,
  };
  return iree_hal_driver_registry_register_factory(registry, &factory);
}
