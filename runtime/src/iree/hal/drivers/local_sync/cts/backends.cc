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

#include "iree/async/util/proactor_pool.h"
#include "iree/base/threading/numa.h"
#include "iree/hal/api.h"
#include "iree/hal/cts/util/registry.h"
#include "iree/hal/drivers/local_sync/registration/driver_module.h"

namespace iree::hal::cts {

static iree_status_t CreateLocalSyncDevice(iree_hal_driver_t** out_driver,
                                           iree_hal_device_t** out_device) {
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

  // Create a proactor pool for async I/O. The pool is lazy — no threads are
  // spawned until a driver actually requests a proactor.
  iree_async_proactor_pool_t* proactor_pool = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_async_proactor_pool_create(
        iree_numa_node_count(), /*node_ids=*/NULL,
        iree_async_proactor_pool_options_default(), iree_allocator_system(),
        &proactor_pool);
  }

  // Create the default device from the driver.
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
          },
          /*expected_failures=*/
          {
              {"QueueAllocaTest.AllocaWithWaitSemaphores",
               "background thread signaling deadlocks sync queue wait "
               "(sync driver blocks on semaphore wait in queue_alloca)"},
          }},
         {"events", "file_io", "host_calls", "mapping", "indirect"},
     }),
     true);

}  // namespace iree::hal::cts
