// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/webgpu/platform/native/native_driver.h"
#include "experimental/webgpu/registration/driver_module.h"
#include "iree/base/api.h"

// TODO(#4298): remove this driver registration and wrapper.

static iree_status_t iree_hal_webgpu_native_driver_factory_enumerate(
    void* self, const iree_hal_driver_info_t** out_driver_infos,
    iree_host_size_t* out_driver_info_count) {
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "WebGPU native driver is only for testing compilation");
}

static iree_status_t iree_hal_webgpu_native_driver_factory_try_create(
    void* self, iree_hal_driver_id_t driver_id, iree_allocator_t host_allocator,
    iree_hal_driver_t** out_driver) {
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "WebGPU native driver is only for testing compilation");
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
