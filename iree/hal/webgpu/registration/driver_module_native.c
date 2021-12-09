// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <inttypes.h>
#include <stddef.h>

#include "iree/base/api.h"
#include "iree/base/internal/flags.h"
#include "iree/hal/webgpu/platform/native/native_driver.h"
#include "iree/hal/webgpu/registration/driver_module.h"

// TODO(#4298): remove this driver registration and wrapper.

#define IREE_HAL_WEBGPU_DRIVER_ID 0x57475055u  // WGPU

IREE_FLAG(string, webgpu_log_level, "warning",
          "[off, error, warning, info, debug, trace]; controls the verbosity "
          "level of the logging from the WebGPU "
          "implementation.");

IREE_FLAG(string, webgpu_power_preference, "high-performance",
          "[(empty), low-power, high-performance]; biases adapter selection "
          "based on the assumed power usage of the device.");

static iree_status_t iree_hal_webgpu_native_driver_factory_enumerate(
    void* self, const iree_hal_driver_info_t** out_driver_infos,
    iree_host_size_t* out_driver_info_count) {
  static const iree_hal_driver_info_t driver_infos[1] = {
      {
          .driver_id = IREE_HAL_WEBGPU_DRIVER_ID,
          .driver_name = iree_string_view_literal("webgpu-native"),
          .full_name = iree_string_view_literal("Experimental WebGPU"),
      },
  };
  *out_driver_info_count = IREE_ARRAYSIZE(driver_infos);
  *out_driver_infos = driver_infos;
  return iree_ok_status();
}

static iree_status_t iree_hal_webgpu_native_driver_factory_try_create(
    void* self, iree_hal_driver_id_t driver_id, iree_allocator_t host_allocator,
    iree_hal_driver_t** out_driver) {
  if (driver_id != IREE_HAL_WEBGPU_DRIVER_ID) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "no driver with ID %016" PRIu64
                            " is provided by this factory",
                            driver_id);
  }

  iree_hal_webgpu_driver_options_t options;
  iree_hal_webgpu_driver_options_initialize(&options);

  if (strcmp(FLAG_webgpu_log_level, "error") == 0) {
    options.log_level = IREE_HAL_WEBGPU_DRIVER_LOG_LEVEL_ERROR;
  } else if (strcmp(FLAG_webgpu_log_level, "warning") == 0) {
    options.log_level = IREE_HAL_WEBGPU_DRIVER_LOG_LEVEL_WARNING;
  } else if (strcmp(FLAG_webgpu_log_level, "info") == 0) {
    options.log_level = IREE_HAL_WEBGPU_DRIVER_LOG_LEVEL_INFO;
  } else if (strcmp(FLAG_webgpu_log_level, "debug") == 0) {
    options.log_level = IREE_HAL_WEBGPU_DRIVER_LOG_LEVEL_DEBUG;
  } else if (strcmp(FLAG_webgpu_log_level, "trace") == 0) {
    options.log_level = IREE_HAL_WEBGPU_DRIVER_LOG_LEVEL_TRACE;
  } else {
    options.log_level = IREE_HAL_WEBGPU_DRIVER_LOG_LEVEL_OFF;
  }

  if (strcmp(FLAG_webgpu_power_preference, "low-power") == 0) {
    options.power_preference = WGPUPowerPreference_LowPower;
  } else if (strcmp(FLAG_webgpu_power_preference, "high-performance") == 0) {
    options.power_preference = WGPUPowerPreference_HighPerformance;
  } else {
    // TODO(benvanik): coming in future spec update. For now go high-perf.
    // options.power_preference = WGPUPowerPreference_Undefined;
    options.power_preference = WGPUPowerPreference_HighPerformance;
  }

  return iree_hal_webgpu_native_driver_create(
      iree_make_cstring_view("webgpu-native"), &options, host_allocator,
      out_driver);
}

IREE_API_EXPORT iree_status_t
iree_hal_webgpu_driver_module_register(iree_hal_driver_registry_t* registry) {
  static const iree_hal_driver_factory_t factory = {
      .self = NULL,
      .enumerate = iree_hal_webgpu_native_driver_factory_enumerate,
      .try_create = iree_hal_webgpu_native_driver_factory_try_create,
  };
  return iree_hal_driver_registry_register_factory(registry, &factory);
}
