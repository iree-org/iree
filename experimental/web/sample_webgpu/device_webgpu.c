// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <emscripten/html5.h>
#include <emscripten/html5_webgpu.h>

#include "experimental/webgpu/api.h"
#include "experimental/webgpu/platform/webgpu.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"

iree_status_t create_device(iree_allocator_t host_allocator,
                            iree_hal_device_t** out_device) {
  WGPUDevice wgpu_device = emscripten_webgpu_get_device();
  if (!wgpu_device) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "emscripten_webgpu_get_device() failed to return a WGPUDevice");
  }

  iree_hal_webgpu_device_options_t default_options;
  iree_hal_webgpu_device_options_initialize(&default_options);

  return iree_hal_webgpu_wrap_device(IREE_SV("webgpu-emscripten"),
                                     &default_options, wgpu_device,
                                     host_allocator, out_device);
}
