// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/metal/registration/driver_module.h"

#include <inttypes.h>
#include <stddef.h>

#include "iree/base/api.h"
#include "iree/base/internal/flags.h"
#include "iree/base/status.h"
#include "iree/base/tracing.h"
#include "iree/hal/drivers/metal/api.h"

IREE_FLAG(bool, metal_serial_command_dispatch, false,
          "Serializes all commands within command buffers as if there were "
          "barriers between each");
IREE_FLAG(bool, metal_command_buffer_retain_resources, false,
          "Enables automatic Metal resource reference counting for diagnosing "
          "resource lifetime issues");
IREE_FLAG(bool, metal_resource_hazard_tracking, false,
          "Enables automatic Metal hazard tracking for diagnosing concurrency "
          "issues");

static iree_status_t iree_hal_metal_driver_factory_enumerate(
    void* self, iree_host_size_t* out_driver_info_count,
    const iree_hal_driver_info_t** out_driver_infos) {
  IREE_ASSERT_ARGUMENT(out_driver_info_count);
  IREE_ASSERT_ARGUMENT(out_driver_infos);

  static const iree_hal_driver_info_t driver_infos[1] = {
      {
          .driver_name = IREE_SVL("metal"),
          .full_name = IREE_SVL("Apple Metal"),
      },
  };
  *out_driver_info_count = IREE_ARRAYSIZE(driver_infos);
  *out_driver_infos = driver_infos;

  return iree_ok_status();
}

static iree_status_t iree_hal_metal_driver_factory_try_create(
    void* self, iree_string_view_t driver_name, iree_allocator_t host_allocator,
    iree_hal_driver_t** out_driver) {
  IREE_ASSERT_ARGUMENT(out_driver);

  if (!iree_string_view_equal(driver_name, IREE_SV("metal"))) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "no driver '%.*s' is provided by this factory",
                            (int)driver_name.size, driver_name.data);
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_metal_device_params_t device_params;
  iree_hal_metal_device_params_initialize(&device_params);
  device_params.command_dispatch_type =
      FLAG_metal_serial_command_dispatch
          ? IREE_HAL_METAL_COMMAND_DISPATCH_TYPE_SERIAL
          : IREE_HAL_METAL_COMMAND_DISPATCH_TYPE_CONCURRENT;
  device_params.command_buffer_resource_reference_mode =
      FLAG_metal_command_buffer_retain_resources
          ? IREE_HAL_METAL_COMMAND_BUFFER_RESOURCE_REFERENCE_MODE_RETAINED
          : IREE_HAL_METAL_COMMAND_BUFFER_RESOURCE_REFERENCE_MODE_UNRETAINED;
  device_params.resource_hazard_tracking_mode =
      FLAG_metal_resource_hazard_tracking
          ? IREE_HAL_METAL_RESOURCE_HAZARD_TRACKING_MODE_TRACKED
          : IREE_HAL_METAL_RESOURCE_HAZARD_TRACKING_MODE_UNTRACKED;

  iree_status_t status = iree_hal_metal_driver_create(
      driver_name, &device_params, host_allocator, out_driver);

  IREE_TRACE_ZONE_END(z0);

  return status;
}

IREE_API_EXPORT iree_status_t
iree_hal_metal_driver_module_register(iree_hal_driver_registry_t* registry) {
  static const iree_hal_driver_factory_t factory = {
      .self = NULL,
      .enumerate = iree_hal_metal_driver_factory_enumerate,
      .try_create = iree_hal_metal_driver_factory_try_create,
  };
  return iree_hal_driver_registry_register_factory(registry, &factory);
}
