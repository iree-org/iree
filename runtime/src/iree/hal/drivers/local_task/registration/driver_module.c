// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/local_task/registration/driver_module.h"

#include <inttypes.h>
#include <stddef.h>

#include "iree/base/api.h"
#include "iree/hal/drivers/local_task/task_driver.h"
#include "iree/hal/local/loaders/registration/init.h"
#include "iree/task/api.h"

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

  iree_hal_executable_loader_t* loaders[8] = {NULL};
  iree_host_size_t loader_count = 0;
  iree_status_t status = iree_hal_create_all_available_executable_loaders(
      IREE_ARRAYSIZE(loaders), &loader_count, loaders, host_allocator);

  iree_task_executor_t* executor = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_task_executor_create_from_flags(host_allocator, &executor);
  }

  iree_hal_allocator_t* device_allocator = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_allocator_create_heap(iree_make_cstring_view("local"),
                                            host_allocator, host_allocator,
                                            &device_allocator);
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_task_driver_create(
        driver_name, &default_params, executor, loader_count, loaders,
        device_allocator, host_allocator, out_driver);
  }

  iree_hal_allocator_release(device_allocator);
  iree_task_executor_release(executor);
  for (iree_host_size_t i = 0; i < loader_count; ++i) {
    iree_hal_executable_loader_release(loaders[i]);
  }
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
