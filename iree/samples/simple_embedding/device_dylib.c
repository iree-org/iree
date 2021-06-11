// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// A example of setting up the the dylib driver.

#include <stddef.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/local/executable_loader.h"
#include "iree/hal/local/loaders/embedded_library_loader.h"
#include "iree/hal/local/loaders/legacy_library_loader.h"
#include "iree/hal/local/task_device.h"
#include "iree/task/api.h"

iree_status_t create_sample_device(iree_hal_device_t** device) {
  // Set paramters for the device created in the next step.
  iree_hal_task_device_params_t params;
  iree_hal_task_device_params_initialize(&params);

  iree_hal_executable_loader_t* loaders[2] = {NULL, NULL};
  iree_host_size_t loader_count = 0;
  IREE_RETURN_IF_ERROR(iree_hal_embedded_library_loader_create(
      iree_hal_executable_import_provider_null(), iree_allocator_system(),
      &loaders[loader_count++]));
  IREE_RETURN_IF_ERROR(iree_hal_legacy_library_loader_create(
      iree_hal_executable_import_provider_null(), iree_allocator_system(),
      &loaders[loader_count++]));

  iree_task_executor_t* executor = NULL;
  IREE_RETURN_IF_ERROR(
      iree_task_executor_create_from_flags(iree_allocator_system(), &executor));

  iree_string_view_t identifier = iree_make_cstring_view("dylib");

  // Create the device and release the executor and loader afterwards.
  IREE_RETURN_IF_ERROR(iree_hal_task_device_create(
      identifier, &params, executor, IREE_ARRAYSIZE(loaders), loaders,
      iree_allocator_system(), device));
  iree_task_executor_release(executor);
  for (iree_host_size_t i = 0; i < loader_count; ++i) {
    iree_hal_executable_loader_release(loaders[i]);
  }
  return iree_ok_status();
}
