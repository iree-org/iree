// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree_pjrt/cpu/client.h"

#include "iree/hal/drivers/local_sync/sync_driver.h"
#include "iree/hal/drivers/local_task/task_driver.h"
#include "iree/hal/local/plugins/registration/init.h"
#include "iree/task/api.h"

namespace iree::pjrt::cpu {

CPUClientInstance::CPUClientInstance(std::unique_ptr<Platform> platform)
    : ClientInstance(std::move(platform)) {
  // Seems that it must match how registered. Action at a distance not
  // great.
  // TODO: Get this when constructing the client so it is guaranteed to
  // match.
  cached_platform_name_ = "iree_cpu";
  iree_task_topology_initialize(&task_topology_options_);
}

CPUClientInstance::~CPUClientInstance() {
  iree_hal_allocator_release(device_allocator_);
  if (executor_) iree_task_executor_release(executor_);
  for (iree_host_size_t i = 0; i < loader_count_; ++i) {
    iree_hal_executable_loader_release(loaders_[i]);
  }
  if (plugin_manager_)
    iree_hal_executable_plugin_manager_release(plugin_manager_);
  iree_task_topology_deinitialize(&task_topology_options_);
}

iree_status_t CPUClientInstance::InitializeDeps() {
  // plugin_manager_
  IREE_RETURN_IF_ERROR(iree_hal_executable_plugin_manager_create(
      /*capacity=*/0, host_allocator_, &plugin_manager_));

  // loaders_
  IREE_RETURN_IF_ERROR(iree_hal_create_all_available_executable_loaders(
      plugin_manager_, IREE_ARRAYSIZE(loaders_), &loader_count_, loaders_,
      host_allocator_));

  // device_allocator_
  IREE_RETURN_IF_ERROR(iree_hal_allocator_create_heap(
      iree_make_cstring_view("local"), host_allocator_, host_allocator_,
      &device_allocator_));
  return iree_ok_status();
}

iree_status_t CPUClientInstance::CreateDriver(iree_hal_driver_t** out_driver) {
  // TODO: There is substantial configuration available.
  // We choose to use explicit instantiation (vs registration) because
  // it is assumed that for server-library oriented cases, we are going to
  // want non-default control.
  IREE_RETURN_IF_ERROR(InitializeDeps());

  // driver
  if (single_threaded_debug_) {
    logger().debug("Creating single threaded CPU driver (debugging)");
    iree_hal_sync_device_params_t sync_params;
    iree_hal_sync_device_params_initialize(&sync_params);
    IREE_RETURN_IF_ERROR(iree_hal_sync_driver_create(
        IREE_SV("local-sync"), &sync_params, loader_count_, loaders_,
        device_allocator_, host_allocator_, out_driver));
  } else {
    iree_task_executor_options_t task_executor_options;
    iree_hal_task_device_params_t task_params;
    iree_task_executor_options_initialize(&task_executor_options);
    iree_hal_task_device_params_initialize(&task_params);

    // executor options and topology options. Getting these from flags is not
    // great for this use since there is no way to set the flags :/
    IREE_RETURN_IF_ERROR(iree_task_executor_options_initialize_from_flags(
        &task_executor_options));
    // TODO: Do something smarter than pinning to NUMA node 0.
    IREE_RETURN_IF_ERROR(iree_task_topology_initialize_from_flags(
        /*node_id=*/0, &task_topology_options_));

    IREE_RETURN_IF_ERROR(iree_task_executor_create(
        task_executor_options, &task_topology_options_, host_allocator_,
        &executor_));

    IREE_RETURN_IF_ERROR(iree_hal_task_driver_create(
        IREE_SV("local-task"), &task_params, /*queue_count=*/1, &executor_,
        loader_count_, loaders_, device_allocator_, host_allocator_,
        out_driver));
  }

  logger().debug("CPU driver created");
  return iree_ok_status();
}

bool CPUClientInstance::SetDefaultCompilerFlags(CompilerJob* compiler_job) {
  return compiler_job->SetFlag("--iree-hal-target-backends=llvm-cpu");
}

}  // namespace iree::pjrt::cpu
