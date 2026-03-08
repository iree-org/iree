// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CTS backend registration for the AMDGPU HAL driver.

#include "iree/async/util/proactor_pool.h"
#include "iree/base/threading/numa.h"
#include "iree/hal/api.h"
#include "iree/hal/cts/util/registry.h"
#include "iree/hal/drivers/amdgpu/registration/driver_module.h"

namespace iree::hal::cts {

static iree_status_t CreateAmdgpuDevice(iree_hal_driver_t** out_driver,
                                        iree_hal_device_t** out_device) {
  iree_status_t status = iree_hal_amdgpu_driver_module_register(
      iree_hal_driver_registry_default());
  if (iree_status_is_already_exists(status)) {
    iree_status_ignore(status);
    status = iree_ok_status();
  }

  iree_hal_driver_t* driver = nullptr;
  if (iree_status_is_ok(status)) {
    status = iree_hal_driver_registry_try_create(
        iree_hal_driver_registry_default(), iree_make_cstring_view("amdgpu"),
        iree_allocator_system(), &driver);
  }

  iree_async_proactor_pool_t* proactor_pool = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_async_proactor_pool_create(
        iree_numa_node_count(), /*node_ids=*/NULL,
        iree_async_proactor_pool_options_default(), iree_allocator_system(),
        &proactor_pool);
  }

  iree_hal_device_t* device = nullptr;
  if (iree_status_is_ok(status)) {
    iree_hal_device_create_params_t create_params =
        iree_hal_device_create_params_default();
    create_params.proactor_pool = proactor_pool;
    status = iree_hal_driver_create_default_device(
        driver, &create_params, iree_allocator_system(), &device);
  }

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

static bool amdgpu_registered_ =
    (CtsRegistry::RegisterBackend({
         "amdgpu",
         {"amdgpu",
          CreateAmdgpuDevice,
          /*executable_format=*/nullptr,
          /*executable_data=*/nullptr,
          RecordingMode::kDirect,
          /*unsupported_tests=*/
          {
              {"AllocatorTest.*",
               "AMDGPU driver does not yet implement allocator CTS tests"},
              {"BufferMappingTest.*",
               "AMDGPU driver does not yet implement buffer mapping"},
              {"CommandBufferTest.*",
               "AMDGPU driver does not yet implement command buffer CTS "
               "tests"},
              {"CommandBufferCopyBufferTest.*",
               "AMDGPU driver does not yet implement buffer copy"},
              {"CommandBufferFillBufferTest.*",
               "AMDGPU driver does not yet implement buffer fill"},
              {"CommandBufferUpdateBufferTest.*",
               "AMDGPU driver does not yet implement buffer update"},
              {"DispatchTest.*",
               "AMDGPU driver does not yet implement dispatch CTS tests"},
              {"DispatchMultiEntrypointTest.*",
               "AMDGPU driver does not yet implement dispatch CTS tests"},
              {"DispatchMultiWorkgroupTest.*",
               "AMDGPU driver does not yet implement dispatch CTS tests"},
              {"DispatchConstantsTest.*",
               "AMDGPU driver does not yet implement dispatch CTS tests"},
              {"DispatchConstantsBindingsTest.*",
               "AMDGPU driver does not yet implement dispatch CTS tests"},
              {"EventTest.*", "AMDGPU does not implement HAL events"},
              {"ExecutableTest.*",
               "AMDGPU does not implement executable reflection"},
              {"FileTest.*",
               "AMDGPU driver does not yet implement file CTS tests"},
          }},
         {"async_queue"},
     }),
     true);

}  // namespace iree::hal::cts
