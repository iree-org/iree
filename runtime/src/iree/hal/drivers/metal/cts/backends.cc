// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CTS backend registration for the Metal HAL driver.

#include "iree/hal/api.h"
#include "iree/hal/cts/util/registry.h"
#include "iree/hal/drivers/metal/registration/driver_module.h"

namespace iree::hal::cts {

static iree_status_t CreateMetalDevice(
    const iree_hal_device_create_params_t* create_params,
    iree_hal_driver_t** out_driver, iree_hal_device_t** out_device) {
  iree_status_t status =
      iree_hal_metal_driver_module_register(iree_hal_driver_registry_default());
  if (iree_status_is_already_exists(status)) {
    iree_status_ignore(status);
    status = iree_ok_status();
  }

  iree_hal_driver_t* driver = nullptr;
  if (iree_status_is_ok(status)) {
    status = iree_hal_driver_registry_try_create(
        iree_hal_driver_registry_default(), iree_make_cstring_view("metal"),
        iree_allocator_system(), &driver);
  }

  iree_hal_device_t* device = nullptr;
  if (iree_status_is_ok(status)) {
    status = iree_hal_driver_create_default_device(
        driver, create_params, iree_allocator_system(), &device);
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

static bool metal_registered_ =
    (CtsRegistry::RegisterBackend({
         "metal",
         {"metal",
          CreateMetalDevice,
          /*executable_format=*/nullptr,
          /*executable_data=*/nullptr,
          RecordingMode::kDirect,
          /*unsupported_tests=*/
          {
              {"QueueAllocaTest.AllocaWithWaitSemaphores",
               "Metal queue_alloca waits synchronously on wait semaphores"},
              {"AsyncTransientBufferTest.*",
               "Metal queue_alloca waits synchronously on wait semaphores "
               "and cannot park a transient allocation for command-buffer "
               "recording or binding-table resolution before commit."},
              {"QueueAllocaTest.ExplicitPassthroughPoolAllocaDealloca",
               "iree_hal_metal_device_queue_alloca rejects any non-NULL pool "
               "argument with UNIMPLEMENTED; the existing path waits on the "
               "wait list synchronously and forwards to "
               "iree_hal_allocator_allocate_buffer on the device allocator. "
               "Caller-supplied pools require a transient-buffer wrapper that "
               "bridges pool_acquire_reservation/release_reservation through "
               "the device allocator."},
              {"QueueAllocaTest.ExplicitTLSFPoolTransferAllocaDealloca",
               "Blocked by the same iree_hal_metal_device_queue_alloca "
               "non-NULL pool rejection as "
               "ExplicitPassthroughPoolAllocaDealloca."},
              {"QueueAllocaTest.ExplicitFixedBlockPoolCrossQueueWaitFrontier",
               "Blocked by the same iree_hal_metal_device_queue_alloca "
               "non-NULL pool rejection as "
               "ExplicitPassthroughPoolAllocaDealloca."},
              {"QueueAllocaTest."
               "ExplicitFixedBlockPoolPendingDeallocaWaitFrontier",
               "Blocked by the same iree_hal_metal_device_queue_alloca "
               "non-NULL pool rejection as "
               "ExplicitPassthroughPoolAllocaDealloca."},
              {"QueueAllocaTest.ExplicitFixedBlockPoolRequiresWaitFrontierFlag",
               "Blocked by the same iree_hal_metal_device_queue_alloca "
               "non-NULL pool rejection as "
               "ExplicitPassthroughPoolAllocaDealloca."},
              {"QueueAllocaTest.ExplicitTLSFPoolCrossQueueWaitFrontier",
               "Blocked by the same iree_hal_metal_device_queue_alloca "
               "non-NULL pool rejection as "
               "ExplicitPassthroughPoolAllocaDealloca."},
              {"QueueAllocaTest.ExplicitTLSFPoolCrossQueueStaleBlockGrows",
               "Blocked by the same Metal queue pool backend UNIMPLEMENTED "
               "path as ExplicitTLSFPoolCrossQueueWaitFrontier."},
              {"QueueAllocaTest.ExplicitFixedBlockPoolNotificationRetry",
               "Blocked by the same iree_hal_metal_device_queue_alloca "
               "non-NULL pool rejection as "
               "ExplicitPassthroughPoolAllocaDealloca."},
              {"QueueAllocaTest.BufferMetadata",
               "Metal queue_alloca forwards to the synchronous device "
               "allocator and does not return an asynchronous transient "
               "placement."},
              {"QueueAllocaTest.DeallocaReleasesMemory",
               "Metal queue_dealloca is currently a barrier and does not "
               "decommit the underlying allocation before buffer release."},
              {"EventTest.*", "Metal does not implement HAL events"},
              {"ExecutableTest.*",
               "Metal does not implement executable reflection"},
              {"SemaphoreTest.*",
               "Metal semaphore failure tests disabled pending fix"},
              {"SemaphoreSubmissionTest.*",
               "Metal semaphore submission tests disabled pending fix"},
          },
          /*expected_failures=*/
          {
              {"QueueAllocaTest.FailedDeallocaWaitDoesNotDealloca",
               "Metal queue_execute does not yet propagate failed wait "
               "dependencies before encoding GPU waits, so a failed wait can "
               "wake the MTLSharedEvent path and let the barrier signal "
               "complete successfully."},
          }},
         {"async_queue"},
     }),
     true);

}  // namespace iree::hal::cts
