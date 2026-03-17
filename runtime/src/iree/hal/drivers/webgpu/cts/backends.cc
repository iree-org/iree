// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CTS backend registration for the WebGPU HAL driver.
//
// Registers a single "webgpu" backend that creates a HAL device from a
// GPUDevice pre-configured by the JS host. The JS entry point
// (webgpu_cts_main.mjs) creates the GPUDevice via dawn before wasm starts and
// stores the handle on context.preConfiguredDevice. The driver's
// create_device_by_id detects this and uses the pre-configured device directly,
// bypassing the async adapter → device request path.

#include "iree/async/util/proactor_pool.h"
#include "iree/hal/api.h"
#include "iree/hal/cts/util/registry.h"
#include "iree/hal/drivers/webgpu/registration/driver_module.h"

namespace iree::hal::cts {

static iree_status_t CreateWebGPUDevice(iree_hal_driver_t** out_driver,
                                        iree_hal_device_t** out_device) {
  // Register the driver module with the global registry.
  iree_status_t status = iree_hal_webgpu_driver_module_register(
      iree_hal_driver_registry_default());
  if (iree_status_is_already_exists(status)) {
    iree_status_ignore(status);
    status = iree_ok_status();
  }

  // Create the driver.
  iree_hal_driver_t* driver = nullptr;
  if (iree_status_is_ok(status)) {
    status = iree_hal_driver_registry_try_create(
        iree_hal_driver_registry_default(), iree_make_cstring_view("webgpu"),
        iree_allocator_system(), &driver);
  }

  // Create a proactor pool for async I/O. WebGPU has a single device and
  // no NUMA topology — use a single-node pool.
  iree_async_proactor_pool_t* proactor_pool = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_async_proactor_pool_create(
        /*node_count=*/1, /*node_ids=*/NULL,
        iree_async_proactor_pool_options_default(), iree_allocator_system(),
        &proactor_pool);
  }

  // Create the default device. The driver's create_device_by_id checks for a
  // pre-configured device handle (set by the JS entry point) and uses it
  // directly, bypassing the async request path.
  iree_hal_device_t* device = nullptr;
  if (iree_status_is_ok(status)) {
    iree_hal_device_create_params_t create_params =
        iree_hal_device_create_params_default();
    create_params.proactor_pool = proactor_pool;
    status = iree_hal_driver_create_default_device(
        driver, &create_params, iree_allocator_system(), &device);
  }

  // The device retains the pool — caller can release immediately.
  iree_async_proactor_pool_release(proactor_pool);

  if (iree_status_is_ok(status)) {
    *out_driver = driver;
    *out_device = device;
  } else {
    iree_hal_device_release(device);
    iree_hal_driver_release(driver);
  }
  return status;
}

static bool webgpu_registered_ = (CtsRegistry::RegisterBackend({
                                      "webgpu",
                                      {
                                          "webgpu",
                                          CreateWebGPUDevice,
                                          /*executable_format=*/nullptr,
                                          /*executable_data=*/nullptr,
                                          RecordingMode::kDirect,
                                          /*unsupported_tests=*/
                                          {
                                              {"EventTest.*",
                                               "WebGPU does not support HAL "
                                               "events; commands within a "
                                               "queue have implicit ordering"},
                                          },
                                      },
                                      {"async_queue"},
                                  }),
                                  true);

}  // namespace iree::hal::cts
