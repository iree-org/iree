// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "driver_module.h"

#include <inttypes.h>
#include <stddef.h>

#include "iree/base/api.h"
#include "iree/hal/drivers/nonlocal_sync/driver.h"
#include "iree/hal/nonlocal/embedded_elf_loader.h"
#include "iree/hal/local/plugins/registration/init.h"

static iree_status_t iree_hal_nonlocal_sync_driver_factory_enumerate(
    void* self, iree_host_size_t* out_driver_info_count,
    const iree_hal_driver_info_t** out_driver_infos) {
  static const iree_hal_driver_info_t default_driver_info = {
      .driver_name = IREE_SVL("nonlocal-sync"),
      .full_name = IREE_SVL("Local execution using a lightweight "
                            "inline synchronous queue"),
  };
  *out_driver_info_count = 1;
  *out_driver_infos = &default_driver_info;
  return iree_ok_status();
}

static iree_status_t iree_hal_nonlocal_sync_driver_factory_try_create(
    void* self, iree_string_view_t driver_name, iree_allocator_t host_allocator,
    iree_hal_driver_t** out_driver) {
  if (!iree_string_view_equal(driver_name, IREE_SV("nonlocal-sync"))) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "no driver '%.*s' is provided by this factory",
                            (int)driver_name.size, driver_name.data);
  }

  iree_hal_nl_sync_device_params_t default_params;
  iree_hal_nl_sync_device_params_initialize(&default_params);

  iree_hal_executable_plugin_manager_t* plugin_manager = NULL;
  iree_status_t status = iree_hal_executable_plugin_manager_create_from_flags(
      host_allocator, &plugin_manager);

  iree_hal_executable_loader_t* loader = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_nonlocal_embedded_elf_loader_create(plugin_manager,
      host_allocator, &loader));

  if (iree_status_is_ok(status)) {
    status = iree_hal_nl_sync_driver_create(
        driver_name, &default_params, 1, &loader,
        host_allocator, out_driver);
  }

  iree_hal_executable_loader_release(loader);
  iree_hal_executable_plugin_manager_release(plugin_manager);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_nonlocal_sync_driver_module_register(
    iree_hal_driver_registry_t* registry) {
  static const iree_hal_driver_factory_t factory = {
      .self = NULL,
      .enumerate = iree_hal_nonlocal_sync_driver_factory_enumerate,
      .try_create = iree_hal_nonlocal_sync_driver_factory_try_create,
  };
  return iree_hal_driver_registry_register_factory(registry, &factory);
}
