// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree_pjrt/rocm/client.h"

namespace iree::pjrt::rocm {

ROCMClientInstance::ROCMClientInstance(std::unique_ptr<Platform> platform)
    : ClientInstance(std::move(platform)) {
  // Seems that it must match how registered. Action at a distance not
  // great.
  // TODO: Get this when constructing the client so it is guaranteed to
  // match.
  cached_platform_name_ = "iree_rocm";
}

ROCMClientInstance::~ROCMClientInstance() {}

iree_status_t ROCMClientInstance::CreateDriver(iree_hal_driver_t** out_driver) {
  iree_string_view_t driver_name = iree_make_cstring_view("hip");

  // Device params.
  iree_hal_hip_device_params_t default_params;
  iree_hal_hip_device_params_initialize(&default_params);

  // Driver params.
  iree_hal_hip_driver_options_t driver_options;
  iree_hal_hip_driver_options_initialize(&driver_options);
  driver_options.default_device_index = 0;

  IREE_RETURN_IF_ERROR(iree_hal_hip_driver_create(driver_name, &driver_options,
                                                  &default_params,
                                                  host_allocator_, out_driver));
  logger().debug("HIP driver created");

  // Get available devices and filter into visible_devices_.
  iree_host_size_t available_devices_count = 0;
  iree_hal_device_info_t* available_devices = nullptr;
  IREE_RETURN_IF_ERROR(iree_hal_driver_query_available_devices(
      *out_driver, host_allocator_, &available_devices_count,
      &available_devices));
  for (iree_host_size_t i = 0; i < available_devices_count; ++i) {
    iree_hal_device_info_t* info = &available_devices[i];
    logger().debug("Enumerated available AMDGPU device:" +
                   std::string(info->path.data, info->path.size) + " " +
                   std::string(info->name.data, info->name.size));
  }
  return iree_ok_status();
}

bool ROCMClientInstance::SetDefaultCompilerFlags(CompilerJob* compiler_job) {
  return compiler_job->SetFlag("--iree-hal-target-device=hip");
}

}  // namespace iree::pjrt::rocm
