// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CTS backend registration for the local-sync HAL driver.
//
// Registers a single "local_sync" backend that creates a synchronous
// single-threaded HAL device. All command buffers execute inline during
// queue submission — there is no background scheduling or async semaphore
// advancement.

#include "iree/hal/api.h"
#include "iree/hal/cts/util/registry.h"
#include "iree/hal/drivers/local_sync/registration/driver_module.h"

namespace iree::hal::cts {

static iree_status_t CreateLocalSyncDevice(
    const iree_hal_device_create_params_t* create_params,
    iree_hal_driver_t** out_driver, iree_hal_device_t** out_device) {
  // Register the driver module with the global registry. Subsequent calls
  // return ALREADY_EXISTS which we ignore — only true errors propagate.
  iree_status_t status = iree_hal_local_sync_driver_module_register(
      iree_hal_driver_registry_default());
  if (iree_status_is_already_exists(status)) {
    iree_status_ignore(status);
    status = iree_ok_status();
  }

  // Create the driver. This sets up executable loaders and a heap allocator
  // via the driver module's flag-based configuration.
  iree_hal_driver_t* driver = nullptr;
  if (iree_status_is_ok(status)) {
    status = iree_hal_driver_registry_try_create(
        iree_hal_driver_registry_default(),
        iree_make_cstring_view("local-sync"), iree_allocator_system(), &driver);
  }

  // Create the default device from the driver.
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

// Registration at static init time. The comma operator evaluates
// RegisterBackend() for its side effect and yields true for the bool.
static bool local_sync_registered_ =
    (CtsRegistry::RegisterBackend({
         "local_sync",
         {"local_sync",
          CreateLocalSyncDevice,
          /*executable_format=*/nullptr,
          /*executable_data=*/nullptr,
          RecordingMode::kDirect,
          /*unsupported_tests=*/
          {
              {"QueueAllocaTest.BufferMetadata",
               "sync driver uses heap allocation, no ASYNCHRONOUS placement"},
              {"QueueAllocaTest.DeallocaReleasesMemory",
               "sync driver dealloca is a barrier, heap buffers stay valid"},
              {"QueueAllocaTest.ExplicitFixedBlockPoolCrossQueueWaitFrontier",
               "sync driver routes queue_alloca through "
               "iree_hal_pool_allocate_buffer (synchronous helper) and the "
               "transient buffer's release callback frees the reservation "
               "with a NULL frontier, so the pool always returns OK_FRESH on "
               "the next acquire instead of the OK_NEEDS_WAIT path this test "
               "asserts. The dependency model under test is only meaningful "
               "for queues that release while the freed work is still in "
               "flight."},
              {"QueueAllocaTest."
               "ExplicitFixedBlockPoolPendingDeallocaWaitFrontier",
               "sync driver routes queue_alloca through the synchronous pool "
               "helper and cannot submit a second alloca while the first "
               "queue-ordered dealloca is still in flight."},
              {"QueueAllocaTest.ExplicitFixedBlockPoolRequiresWaitFrontierFlag",
               "sync driver routes queue_alloca through the synchronous pool "
               "helper and cannot distinguish async queue-owned hidden "
               "frontier waits from pool-notification retries."},
              {"QueueAllocaTest.ExplicitTLSFPoolCrossQueueStaleBlockGrows",
               "sync driver routes queue_alloca through "
               "iree_hal_pool_allocate_buffer (synchronous helper) and the "
               "transient buffer's release callback frees the reservation "
               "with a NULL frontier, so the next acquire can reuse the block "
               "instead of observing a stale cross-queue block and growing. "
               "The stale-frontier growth behavior is only meaningful for "
               "queues that release while the freed work is still in flight."},
              {"QueueAllocaTest.ExplicitFixedBlockPoolNotificationRetry",
               "sync driver routes queue_alloca through the synchronous pool "
               "helper and cannot submit the dealloca that releases the first "
               "block while the second alloca is waiting on pool "
               "notification."},
              {"QueueAllocaTest.AllocaWithWaitSemaphores",
               "sync driver blocks on semaphore waits during queue_alloca "
               "and cannot model wait-before-signal queue submission."},
              {"QueueDispatchTest.DeferredNoopDispatch",
               "sync driver blocks on semaphore waits during queue_dispatch "
               "and cannot model wait-before-signal queue submission."},
              {"QueueDispatchTest.DeferredWaitBeforeSignalDispatch",
               "sync driver blocks on semaphore waits during queue_dispatch "
               "and cannot model wait-before-signal queue submission."},
              {"DispatchReuseTest.DeferredExecuteRetainsDispatchBindingTable",
               "sync driver blocks on semaphore waits during queue_execute "
               "and cannot model deferred command-buffer submission behind an "
               "unresolved wait."},
          },
          /*expected_failures=*/{}},
         {"events", "file_io", "host_calls", "mapping", "indirect"},
     }),
     true);

}  // namespace iree::hal::cts
