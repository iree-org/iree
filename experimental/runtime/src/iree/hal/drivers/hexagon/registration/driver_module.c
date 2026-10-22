// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/hexagon/registration/driver_module.h"

#include "iree/base/api.h"
#include "iree/hal/drivers/hexagon/api.h"

static iree_status_t iree_hal_hexagon_driver_factory_enumerate(
    void* self, iree_host_size_t* out_driver_info_count,
    const iree_hal_driver_info_t** out_driver_infos) {
  static const iree_hal_driver_info_t default_driver_info = {
      .driver_name = IREE_SVL("hexagon"),
      .full_name = IREE_SVL("Qualcomm Hexagon DSP (experimental)"),
  };
  *out_driver_info_count = 1;
  *out_driver_infos = &default_driver_info;
  return iree_ok_status();
}

static iree_status_t iree_hal_hexagon_driver_factory_try_create(
    void* self, iree_string_view_t driver_name, iree_allocator_t host_allocator,
    iree_hal_driver_t** out_driver) {
  if (!iree_string_view_equal(driver_name, IREE_SV("hexagon"))) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "no driver '%.*s' is provided by this factory",
                            (int)driver_name.size, driver_name.data);
  }

  iree_hal_hexagon_driver_options_t options;
  iree_hal_hexagon_driver_options_initialize(&options);
  return iree_hal_hexagon_driver_create(driver_name, &options, host_allocator,
                                        out_driver);
}

IREE_API_EXPORT iree_status_t
iree_hal_hexagon_driver_module_register(iree_hal_driver_registry_t* registry) {
  static const iree_hal_driver_factory_t factory = {
      .self = NULL,
      .enumerate = iree_hal_hexagon_driver_factory_enumerate,
      .try_create = iree_hal_hexagon_driver_factory_try_create,
  };
  return iree_hal_driver_registry_register_factory(registry, &factory);
}
