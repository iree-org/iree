// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/dylib/registration/driver_module.h"

#include <inttypes.h>
#include <stddef.h>

#include "iree/base/api.h"
#include "iree/base/internal/flags.h"
#include "iree/hal/local/executable_loader.h"
#include "iree/hal/local/loaders/embedded_library_loader.h"
#include "iree/hal/local/loaders/legacy_library_loader.h"
#include "iree/hal/local/task_device.h"
#include "iree/hal/local/task_driver.h"
#include "iree/task/api.h"

// TODO(#4298): remove this driver registration and wrapper.
// By having a single iree/hal/local/registration that then has the loaders
// added to it based on compilation settings we can have a single set of flags
// for everything. We can also have API helper methods that register the driver
// using an existing executor so that we can entirely externalize the task
// system configuration from the HAL.

#define IREE_HAL_DYLIB_DRIVER_ID 0x58444C4Cu  // XDLL

static iree_status_t iree_hal_dylib_driver_factory_enumerate(
    void* self, const iree_hal_driver_info_t** out_driver_infos,
    iree_host_size_t* out_driver_info_count) {
  static const iree_hal_driver_info_t driver_infos[1] = {
      {
          .driver_id = IREE_HAL_DYLIB_DRIVER_ID,
          .driver_name = iree_string_view_literal("dylib"),
          .full_name =
              iree_string_view_literal("AOT compiled dynamic libraries"),
      },
  };
  *out_driver_info_count = IREE_ARRAYSIZE(driver_infos);
  *out_driver_infos = driver_infos;
  return iree_ok_status();
}

static iree_status_t iree_hal_dylib_driver_factory_try_create(
    void* self, iree_hal_driver_id_t driver_id, iree_allocator_t allocator,
    iree_hal_driver_t** out_driver) {
  if (driver_id != IREE_HAL_DYLIB_DRIVER_ID) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "no driver with ID %016" PRIu64
                            " is provided by this factory",
                            driver_id);
  }

  iree_hal_task_device_params_t default_params;
  iree_hal_task_device_params_initialize(&default_params);

  iree_status_t status = iree_ok_status();

  iree_hal_executable_loader_t* loaders[2] = {NULL, NULL};
  iree_host_size_t loader_count = 0;
  if (iree_status_is_ok(status)) {
    status = iree_hal_embedded_library_loader_create(
        iree_hal_executable_import_provider_null(), allocator,
        &loaders[loader_count++]);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_legacy_library_loader_create(
        iree_hal_executable_import_provider_null(), allocator,
        &loaders[loader_count++]);
  }

  iree_task_executor_t* executor = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_task_executor_create_from_flags(allocator, &executor);
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_task_driver_create(
        iree_make_cstring_view("dylib"), &default_params, executor,
        loader_count, loaders, allocator, out_driver);
  }

  iree_task_executor_release(executor);
  for (iree_host_size_t i = 0; i < loader_count; ++i) {
    iree_hal_executable_loader_release(loaders[i]);
  }
  return status;
}

IREE_API_EXPORT iree_status_t
iree_hal_dylib_driver_module_register(iree_hal_driver_registry_t* registry) {
  static const iree_hal_driver_factory_t factory = {
      .self = NULL,
      .enumerate = iree_hal_dylib_driver_factory_enumerate,
      .try_create = iree_hal_dylib_driver_factory_try_create,
  };
  return iree_hal_driver_registry_register_factory(registry, &factory);
}
