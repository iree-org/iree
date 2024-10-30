// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <emscripten/threading.h>

#include "iree/hal/drivers/local_task/task_device.h"
#include "iree/hal/local/loaders/static_library_loader.h"
#include "iree/task/api.h"
#include "mnist_static.h"

iree_status_t create_device_with_static_loader(iree_allocator_t host_allocator,
                                               iree_hal_device_t** out_device) {
  iree_hal_task_device_params_t params;
  iree_hal_task_device_params_initialize(&params);

  // Register the statically linked executable library.
  const iree_hal_executable_library_query_fn_t libraries[] = {
      mnist_linked_library_query,
  };
  iree_hal_executable_loader_t* library_loader = NULL;
  iree_status_t status = iree_hal_static_library_loader_create(
      IREE_ARRAYSIZE(libraries), libraries,
      iree_hal_executable_import_provider_null(), host_allocator,
      &library_loader);

  // Create a task executor.
  iree_task_executor_options_t options;
  iree_task_executor_options_initialize(&options);
  options.worker_local_memory_size = 0;
  iree_task_topology_t topology;
  iree_task_topology_initialize(&topology);
  iree_task_topology_initialize_from_group_count(
      /*group_count=*/4, &topology);
  // Note: threads increase memory usage. If using a high thread count, consider
  // passing in a larger WebAssembly.Memory object, increasing Emscripten's
  // INITIAL_MEMORY, or setting Emscripten's ALLOW_MEMORY_GROWTH.
  // iree_task_topology_initialize_from_group_count(
  //     /*group_count=*/emscripten_num_logical_cores(), &topology);
  iree_task_executor_t* executor = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_task_executor_create(options, &topology, host_allocator,
                                       &executor);
  }
  iree_task_topology_deinitialize(&topology);

  iree_string_view_t identifier = iree_make_cstring_view("task");
  iree_hal_allocator_t* device_allocator = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_allocator_create_heap(identifier, host_allocator,
                                            host_allocator, &device_allocator);
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_task_device_create(
        identifier, &params, /*queue_count=*/1, &executor, /*loader_count=*/1,
        &library_loader, device_allocator, host_allocator, out_device);
  }

  iree_hal_allocator_release(device_allocator);
  iree_task_executor_release(executor);
  iree_hal_executable_loader_release(library_loader);
  return status;
}
