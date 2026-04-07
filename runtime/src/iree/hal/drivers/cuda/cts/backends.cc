// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CTS backend registration for the CUDA HAL driver.

#include "iree/async/util/proactor_pool.h"
#include "iree/base/threading/numa.h"
#include "iree/hal/api.h"
#include "iree/hal/cts/util/registry.h"
#include "iree/hal/drivers/cuda/registration/driver_module.h"

namespace iree::hal::cts {

static iree_status_t CreateCudaDevice(iree_hal_driver_t** out_driver,
                                      iree_hal_device_t** out_device) {
  iree_status_t status =
      iree_hal_cuda_driver_module_register(iree_hal_driver_registry_default());
  if (iree_status_is_already_exists(status)) {
    iree_status_ignore(status);
    status = iree_ok_status();
  }

  iree_hal_driver_t* driver = nullptr;
  if (iree_status_is_ok(status)) {
    status = iree_hal_driver_registry_try_create(
        iree_hal_driver_registry_default(), iree_make_cstring_view("cuda"),
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

static bool cuda_registered_ =
    (CtsRegistry::RegisterBackend({
         "cuda",
         {"cuda",
          CreateCudaDevice,
          /*executable_format=*/nullptr,
          /*executable_data=*/nullptr,
          RecordingMode::kDirect,
          /*unsupported_tests=*/
          {
              {"QueueAllocaTest.AllocaWithWaitSemaphores",
               "CUDA queue_alloca waits synchronously on wait semaphores"},
              {"QueueAllocaTest.ExplicitPassthroughPoolAllocaDealloca",
               "CUDA custom queue alloca pools not yet implemented"},
              {"EventTest.*", "CUDA does not implement HAL events"},
              {"ExecutableTest.*",
               "CUDA does not implement executable reflection"},
          }},
         {"async_queue"},
     }),
     true);

}  // namespace iree::hal::cts
