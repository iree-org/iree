// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/webgpu/platform/emscripten/emscripten_driver.h"
#include "experimental/webgpu/registration/driver_module.h"
#include "iree/base/api.h"

// TODO(#4298): remove this driver registration and wrapper.

static iree_status_t iree_hal_webgpu_emscripten_driver_factory_enumerate(
    void* self, iree_host_size_t* out_driver_info_count,
    const iree_hal_driver_info_t** out_driver_infos) {
  static const iree_hal_driver_info_t driver_infos[1] = {
      {
          .driver_name = iree_string_view_literal("webgpu"),
          .full_name = iree_string_view_literal("Experimental WebGPU"),
      },
  };
  *out_driver_info_count = IREE_ARRAYSIZE(driver_infos);
  *out_driver_infos = driver_infos;
  return iree_ok_status();
}

static iree_status_t iree_hal_webgpu_emscripten_driver_factory_try_create(
    void* self, iree_string_view_t driver_name, iree_allocator_t host_allocator,
    iree_hal_driver_t** out_driver) {
  if (!iree_string_view_equal(driver_name, IREE_SV("webgpu"))) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "no driver '%.*s' is provided by this factory",
                            (int)driver_name.size, driver_name.data);
  }

  iree_hal_webgpu_driver_options_t options;
  iree_hal_webgpu_driver_options_initialize(&options);

  return iree_hal_webgpu_emscripten_driver_create(
      iree_make_cstring_view("webgpu-emscripten"), &options, host_allocator,
      out_driver);
}

IREE_API_EXPORT iree_status_t
iree_hal_webgpu_driver_module_register(iree_hal_driver_registry_t* registry) {
  static const iree_hal_driver_factory_t factory = {
      .self = NULL,
      .enumerate = iree_hal_webgpu_emscripten_driver_factory_enumerate,
      .try_create = iree_hal_webgpu_emscripten_driver_factory_try_create,
  };
  return iree_hal_driver_registry_register_factory(registry, &factory);
}
