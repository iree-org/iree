// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/null/registration/driver_module.h"

#include "iree/base/api.h"
#include "iree/hal/drivers/null/api.h"

static iree_status_t iree_hal_null_driver_factory_enumerate(
    void* self, iree_host_size_t* out_driver_info_count,
    const iree_hal_driver_info_t** out_driver_infos) {
  // TODO(null): return multiple drivers if desired. This information must be
  // static. The list here is just what is compiled into the binary and not
  // expected to actually try to load or initialize drivers.
  static const iree_hal_driver_info_t default_driver_info = {
      .driver_name = IREE_SVL("null"),
      .full_name = IREE_SVL("NULL Skeleton Driver"),
  };
  *out_driver_info_count = 1;
  *out_driver_infos = &default_driver_info;
  return iree_ok_status();
}

static iree_status_t iree_hal_null_driver_factory_try_create(
    void* self, iree_string_view_t driver_name, iree_allocator_t host_allocator,
    iree_hal_driver_t** out_driver) {
  // TODO(null): use your driver name - this will be the prefix when the user
  // specifies the device (`--device=null://foo`). A single driver can support
  // multiple prefixes if it wants.
  if (!iree_string_view_equal(driver_name, IREE_SV("null"))) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "no driver '%.*s' is provided by this factory",
                            (int)driver_name.size, driver_name.data);
  }

  // TODO(null): populate options from flags. This driver module file is only
  // used in native tools that have access to the flags library. Programmatic
  // creation of the driver and devices will bypass this file and pass the
  // options via this struct or key-value string parameters.
  iree_hal_null_driver_options_t options;
  iree_hal_null_driver_options_initialize(&options);

  iree_status_t status = iree_hal_null_driver_create(
      driver_name, &options, host_allocator, out_driver);

  return status;
}

IREE_API_EXPORT iree_status_t
iree_hal_null_driver_module_register(iree_hal_driver_registry_t* registry) {
  static const iree_hal_driver_factory_t factory = {
      .self = NULL,
      .enumerate = iree_hal_null_driver_factory_enumerate,
      .try_create = iree_hal_null_driver_factory_try_create,
  };
  return iree_hal_driver_registry_register_factory(registry, &factory);
}
