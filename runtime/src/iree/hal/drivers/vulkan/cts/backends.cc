// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CTS backend registration for the Vulkan HAL driver.

#include "iree/hal/api.h"
#include "iree/hal/cts/util/registry.h"
#include "iree/hal/drivers/vulkan/registration/driver_module.h"

namespace iree::hal::cts {

static iree_status_t CreateVulkanDevice(
    const iree_hal_device_create_params_t* create_params,
    iree_hal_driver_t** out_driver, iree_hal_device_t** out_device) {
  iree_status_t status = iree_hal_vulkan_driver_module_register(
      iree_hal_driver_registry_default());
  if (iree_status_is_already_exists(status)) {
    iree_status_ignore(status);
    status = iree_ok_status();
  }

  iree_hal_driver_t* driver = nullptr;
  if (iree_status_is_ok(status)) {
    status = iree_hal_driver_registry_try_create(
        iree_hal_driver_registry_default(), iree_make_cstring_view("vulkan"),
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

static bool vulkan_registered_ =
    (CtsRegistry::RegisterBackend({
         "vulkan",
         {"vulkan",
          CreateVulkanDevice,
          /*executable_format=*/nullptr,
          /*executable_data=*/nullptr,
          RecordingMode::kDirect,
          /*unsupported_tests=*/
          {
              {"QueueAllocaTest.AllocaWithWaitSemaphores",
               "Vulkan queue_alloca waits synchronously on wait semaphores"},
              {"AsyncTransientBufferTest.*",
               "Vulkan queue_alloca waits synchronously on wait semaphores "
               "and cannot park a transient allocation for command-buffer "
               "recording or binding-table resolution before commit."},
              {"QueueAllocaTest.BufferMetadata",
               "Vulkan queue_alloca is implemented as synchronous allocator "
               "allocation without async queue placement metadata"},
              {"QueueAllocaTest.DeallocaReleasesMemory",
               "Vulkan queue_dealloca is currently only a queue barrier and "
               "does not decommit transient buffer backing"},
              {"QueueAllocaTest.ExplicitPassthroughPoolAllocaDealloca",
               "iree_hal_vulkan_device_queue_alloca rejects any non-NULL pool "
               "argument with UNIMPLEMENTED; the existing path waits on the "
               "wait list synchronously and forwards to "
               "iree_hal_allocator_allocate_buffer on the device allocator. "
               "Caller-supplied pools require a transient-buffer wrapper that "
               "bridges pool_acquire_reservation/release_reservation through "
               "the device allocator."},
              {"QueueAllocaTest.ExplicitTLSFPoolTransferAllocaDealloca",
               "Blocked by the same iree_hal_vulkan_device_queue_alloca "
               "non-NULL pool rejection as "
               "ExplicitPassthroughPoolAllocaDealloca."},
              {"QueueAllocaTest.ExplicitFixedBlockPoolCrossQueueWaitFrontier",
               "Blocked by the same iree_hal_vulkan_device_queue_alloca "
               "non-NULL pool rejection as "
               "ExplicitPassthroughPoolAllocaDealloca."},
              {"QueueAllocaTest."
               "ExplicitFixedBlockPoolPendingDeallocaWaitFrontier",
               "Blocked by the same iree_hal_vulkan_device_queue_alloca "
               "non-NULL pool rejection as "
               "ExplicitPassthroughPoolAllocaDealloca."},
              {"QueueAllocaTest.ExplicitFixedBlockPoolRequiresWaitFrontierFlag",
               "Blocked by the same iree_hal_vulkan_device_queue_alloca "
               "non-NULL pool rejection as "
               "ExplicitPassthroughPoolAllocaDealloca."},
              {"QueueAllocaTest.ExplicitTLSFPoolCrossQueueWaitFrontier",
               "Blocked by the same iree_hal_vulkan_device_queue_alloca "
               "non-NULL pool rejection as "
               "ExplicitPassthroughPoolAllocaDealloca."},
              {"QueueAllocaTest.ExplicitFixedBlockPoolNotificationRetry",
               "Blocked by the same iree_hal_vulkan_device_queue_alloca "
               "non-NULL pool rejection as "
               "ExplicitPassthroughPoolAllocaDealloca."},
              {"ExecutableTest.*",
               "Vulkan does not implement executable reflection"},
              {"SemaphoreTest.WaitThenFail",
               "Vulkan does not support semaphore failure signaling"},
              {"SemaphoreTest.FailThenWait",
               "Vulkan does not support semaphore failure signaling"},
              {"SemaphoreTest.MultiWaitThenFail",
               "Vulkan does not support semaphore failure signaling"},
              {"SemaphoreTest.DeviceMultiWaitThenFail",
               "Vulkan does not support semaphore failure signaling"},
              {"SemaphoreSubmissionTest.*",
               "Vulkan timeline semaphore waits hang without async queue "
               "submission"},
          },
          /*expected_failures=*/
          {
              {"QueueAllocaTest.FailedDeallocaWaitDoesNotDealloca",
               "Vulkan queue_execute does not yet propagate failed wait "
               "dependencies before encoding GPU waits, so a failed wait can "
               "let the barrier signal complete successfully."},
          }},
         {"async_queue"},
     }),
     true);

}  // namespace iree::hal::cts
