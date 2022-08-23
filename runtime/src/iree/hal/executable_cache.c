// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/executable_cache.h"

#include <stddef.h>
#include <string.h>

#include "iree/base/tracing.h"
#include "iree/hal/detail.h"
#include "iree/hal/device.h"
#include "iree/hal/resource.h"

void iree_hal_executable_params_initialize(
    iree_hal_executable_params_t* out_executable_params) {
  memset(out_executable_params, 0, sizeof(*out_executable_params));
  out_executable_params->caching_mode =
      IREE_HAL_EXECUTABLE_CACHING_MODE_ALLOW_PERSISTENT_CACHING |
      IREE_HAL_EXECUTABLE_CACHING_MODE_ALLOW_OPTIMIZATION;
}

#define _VTABLE_DISPATCH(executable_cache, method_name)                 \
  IREE_HAL_VTABLE_DISPATCH(executable_cache, iree_hal_executable_cache, \
                           method_name)

IREE_HAL_API_RETAIN_RELEASE(executable_cache);

IREE_API_EXPORT iree_status_t iree_hal_executable_cache_create(
    iree_hal_device_t* device, iree_string_view_t identifier, iree_loop_t loop,
    iree_hal_executable_cache_t** out_executable_cache) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_executable_cache);
  *out_executable_cache = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = IREE_HAL_VTABLE_DISPATCH(device, iree_hal_device,
                                                  create_executable_cache)(
      device, identifier, loop, out_executable_cache);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT bool iree_hal_executable_cache_can_prepare_format(
    iree_hal_executable_cache_t* executable_cache,
    iree_hal_executable_caching_mode_t caching_mode,
    iree_string_view_t executable_format) {
  IREE_ASSERT_ARGUMENT(executable_cache);
  return _VTABLE_DISPATCH(executable_cache, can_prepare_format)(
      executable_cache, caching_mode, executable_format);
}

IREE_API_EXPORT iree_status_t iree_hal_executable_cache_prepare_executable(
    iree_hal_executable_cache_t* executable_cache,
    const iree_hal_executable_params_t* executable_params,
    iree_hal_executable_t** out_executable) {
  IREE_ASSERT_ARGUMENT(executable_cache);
  IREE_ASSERT_ARGUMENT(executable_params);
  IREE_ASSERT_ARGUMENT(!executable_params->pipeline_layout_count ||
                       executable_params->pipeline_layouts);
  IREE_ASSERT_ARGUMENT(out_executable);
  *out_executable = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = _VTABLE_DISPATCH(executable_cache, prepare_executable)(
      executable_cache, executable_params, out_executable);
  IREE_TRACE_ZONE_END(z0);
  return status;
}
