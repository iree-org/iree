// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CTS backend registration for the local-task HAL driver.
//
// Registers a single "local_task" backend that creates a multithreaded
// task-system-based HAL device using the default driver configuration.
// The factory uses the driver registry to create drivers identically to how
// applications create them (via iree_hal_register_all_available_drivers).

#include "iree/hal/api.h"
#include "iree/hal/cts/util/registry.h"
#include "iree/hal/drivers/local_task/registration/driver_module.h"

namespace iree::hal::cts {

static iree_status_t CreateLocalTaskDevice(iree_hal_driver_t** out_driver,
                                           iree_hal_device_t** out_device) {
  // Register the driver module with the global registry. Subsequent calls
  // return ALREADY_EXISTS which we ignore — only true errors propagate.
  iree_status_t status = iree_hal_local_task_driver_module_register(
      iree_hal_driver_registry_default());
  if (iree_status_is_already_exists(status)) {
    iree_status_ignore(status);
    status = iree_ok_status();
  }

  // Create the driver. This sets up the task executor pool, executable loaders,
  // and heap allocator via the driver module's flag-based configuration.
  iree_hal_driver_t* driver = nullptr;
  if (iree_status_is_ok(status)) {
    status = iree_hal_driver_registry_try_create(
        iree_hal_driver_registry_default(),
        iree_make_cstring_view("local-task"), iree_allocator_system(), &driver);
  }

  // Create the default device from the driver.
  iree_hal_device_t* device = nullptr;
  if (iree_status_is_ok(status)) {
    status = iree_hal_driver_create_default_device(
        driver, iree_allocator_system(), &device);
  }

  if (iree_status_is_ok(status)) {
    *out_driver = driver;
    *out_device = device;
  } else {
    iree_hal_device_release(device);
    iree_hal_driver_release(driver);
  }
  return status;
}

// Registration at static init time. The comma operator evaluates
// RegisterBackend() for its side effect and yields true for the bool.
static bool local_task_registered_ =
    (CtsRegistry::RegisterBackend({
         "local_task",
         {.name = "local_task", .factory = CreateLocalTaskDevice},
         {"async_queue", "events", "file_io", "host_calls", "mapping",
          "indirect"},
     }),
     true);

}  // namespace iree::hal::cts
