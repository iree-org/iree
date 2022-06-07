// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vmvx_sync/registration/driver_module.h"

#include <inttypes.h>
#include <stddef.h>

#include "iree/base/api.h"
#include "iree/hal/drivers/local_sync/sync_device.h"
#include "iree/hal/drivers/local_sync/sync_driver.h"
#include "iree/hal/local/executable_loader.h"
#include "iree/hal/local/loaders/vmvx_module_loader.h"
#include "iree/vm/api.h"

// TODO(#4298): remove this driver registration and wrapper.

#define IREE_HAL_VMVX_SYNC_DRIVER_ID 0x53564D58u  // SVMX

static iree_status_t iree_hal_vmvx_sync_driver_factory_enumerate(
    void* self, const iree_hal_driver_info_t** out_driver_infos,
    iree_host_size_t* out_driver_info_count) {
  static const iree_hal_driver_info_t driver_infos[1] = {
      {
          .driver_id = IREE_HAL_VMVX_SYNC_DRIVER_ID,
          .driver_name = iree_string_view_literal("vmvx-sync"),
          .full_name = iree_string_view_literal(
              "synchronous VM-based reference backend"),
      },
  };
  *out_driver_info_count = IREE_ARRAYSIZE(driver_infos);
  *out_driver_infos = driver_infos;
  return iree_ok_status();
}

static iree_status_t iree_hal_vmvx_sync_driver_factory_try_create(
    void* self, iree_hal_driver_id_t driver_id, iree_allocator_t host_allocator,
    iree_hal_driver_t** out_driver) {
  if (driver_id != IREE_HAL_VMVX_SYNC_DRIVER_ID) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "no driver with ID %016" PRIu64
                            " is provided by this factory",
                            driver_id);
  }

  iree_vm_instance_t* instance = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_instance_create(host_allocator, &instance));

  iree_hal_executable_loader_t* vmvx_loader = NULL;
  iree_status_t status = iree_hal_vmvx_module_loader_create(
      instance, host_allocator, &vmvx_loader);
  iree_hal_executable_loader_t* loaders[1] = {vmvx_loader};

  iree_hal_allocator_t* device_allocator = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_allocator_create_heap(iree_make_cstring_view("vmvx"),
                                            host_allocator, host_allocator,
                                            &device_allocator);
  }

  // Set parameters for the device created in the next step.
  iree_hal_sync_device_params_t default_params;
  iree_hal_sync_device_params_initialize(&default_params);
  if (iree_status_is_ok(status)) {
    status = iree_hal_sync_driver_create(
        iree_make_cstring_view("vmvx"), &default_params,
        IREE_ARRAYSIZE(loaders), loaders, device_allocator, host_allocator,
        out_driver);
  }

  iree_hal_allocator_release(device_allocator);
  iree_hal_executable_loader_release(vmvx_loader);
  iree_vm_instance_release(instance);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_vmvx_sync_driver_module_register(
    iree_hal_driver_registry_t* registry) {
  static const iree_hal_driver_factory_t factory = {
      .self = NULL,
      .enumerate = iree_hal_vmvx_sync_driver_factory_enumerate,
      .try_create = iree_hal_vmvx_sync_driver_factory_try_create,
  };
  return iree_hal_driver_registry_register_factory(registry, &factory);
}
