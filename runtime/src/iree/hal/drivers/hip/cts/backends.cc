// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CTS backend registration for the HIP HAL driver.

#include "iree/hal/api.h"
#include "iree/hal/cts/util/registry.h"
#include "iree/hal/drivers/hip/registration/driver_module.h"

namespace iree::hal::cts {

static iree_status_t CreateHipDevice(iree_hal_driver_t** out_driver,
                                     iree_hal_device_t** out_device) {
  iree_status_t status =
      iree_hal_hip_driver_module_register(iree_hal_driver_registry_default());
  if (iree_status_is_already_exists(status)) {
    iree_status_ignore(status);
    status = iree_ok_status();
  }

  iree_hal_driver_t* driver = nullptr;
  if (iree_status_is_ok(status)) {
    status = iree_hal_driver_registry_try_create(
        iree_hal_driver_registry_default(), iree_make_cstring_view("hip"),
        iree_allocator_system(), &driver);
  }

  iree_hal_device_t* device = nullptr;
  if (iree_status_is_ok(status)) {
    status = iree_hal_driver_create_default_device(
        driver, iree_allocator_system(), &device);
  }

  if (iree_status_is_ok(status)) {
    *out_driver = driver;
    *out_device = device;
  } else {
    iree_hal_device_release(device);
    iree_hal_driver_release(driver);
  }
  return status;
}

static bool hip_registered_ =
    (CtsRegistry::RegisterBackend({
         "hip",
         {"hip",
          CreateHipDevice,
          /*executable_format=*/nullptr,
          /*executable_data=*/nullptr,
          RecordingMode::kDirect,
          /*unsupported_tests=*/
          {
              {"EventTest.*", "HIP does not implement HAL events"},
              {"ExecutableTest.*",
               "HIP does not implement executable reflection"},
              {"BufferMappingTest.*",
               "HIP graph command buffers lack memcpy node support"},
              {"CommandBufferCopyBufferTest.*",
               "HIP graph command buffers lack memcpy node support"},
              {"CommandBufferFillBufferTest.*",
               "HIP graph command buffers lack fill node support"},
              {"CommandBufferUpdateBufferTest.*",
               "HIP graph command buffers lack memcpy node support"},
              {"DispatchTest.*",
               "HIP CTS dispatch tests crash on GPU memory reservation "
               "failures"},
              {"DispatchMultiEntrypointTest.*",
               "HIP CTS dispatch tests crash on GPU memory reservation "
               "failures"},
              {"DispatchMultiWorkgroupTest.*",
               "HIP CTS dispatch tests crash on GPU memory reservation "
               "failures"},
              {"DispatchConstantsTest.*",
               "HIP graph command buffers lack fill node support"},
              {"DispatchConstantsBindingsTest.*",
               "HIP CTS dispatch tests crash on GPU memory reservation "
               "failures"},
          }},
         {"async_queue"},
     }),
     true);

}  // namespace iree::hal::cts
