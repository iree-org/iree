// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/webgpu/platform/native/native_driver.h"

#include "iree/base/tracing.h"

//===----------------------------------------------------------------------===//
// Driver and device options
//===----------------------------------------------------------------------===//

IREE_API_EXPORT void iree_hal_webgpu_driver_options_initialize(
    iree_hal_webgpu_driver_options_t* out_options) {
  IREE_ASSERT_ARGUMENT(out_options);
  memset(out_options, 0, sizeof(*out_options));

  out_options->backend_preference = IREE_HAL_WEBGPU_DRIVER_BACKEND_ANY;

  out_options->log_level = IREE_HAL_WEBGPU_DRIVER_LOG_LEVEL_OFF;

  // TODO(benvanik): coming in future spec update. For now go high-perf.
  // out_options->power_preference = WGPUPowerPreference_Undefined;
  out_options->power_preference = WGPUPowerPreference_HighPerformance;

  iree_hal_webgpu_device_options_initialize(&out_options->device_options);
}
