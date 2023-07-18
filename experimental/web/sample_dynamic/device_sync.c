// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/local_sync/sync_device.h"
#include "iree/hal/local/executable_plugin_manager.h"
#include "iree/hal/local/loaders/system_library_loader.h"
#include "iree/hal/local/loaders/vmvx_module_loader.h"

iree_status_t create_device_with_loaders(iree_allocator_t host_allocator,
                                         iree_hal_device_t** out_device) {
  iree_hal_sync_device_params_t params;
  iree_hal_sync_device_params_initialize(&params);

  iree_status_t status = iree_ok_status();

  iree_hal_executable_plugin_manager_t* plugin_manager = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_executable_plugin_manager_create(
        /*capacity=*/0, host_allocator, &plugin_manager);
  }

  iree_hal_executable_loader_t* loaders[2] = {NULL, NULL};
  iree_host_size_t loader_count = 0;
  if (iree_status_is_ok(status)) {
    status = iree_hal_system_library_loader_create(
        plugin_manager, host_allocator, &loaders[loader_count++]);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vmvx_module_loader_create_isolated(
        /*user_module_count=*/0, /*user_modules=*/NULL, host_allocator,
        &loaders[loader_count++]);
  }

  iree_string_view_t identifier = iree_make_cstring_view("sync");
  iree_hal_allocator_t* device_allocator = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_allocator_create_heap(identifier, host_allocator,
                                            host_allocator, &device_allocator);
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_sync_device_create(identifier, &params, loader_count,
                                         loaders, device_allocator,
                                         host_allocator, out_device);
  }

  iree_hal_allocator_release(device_allocator);
  for (iree_host_size_t i = 0; i < loader_count; ++i) {
    iree_hal_executable_loader_release(loaders[i]);
  }
  return status;
}
