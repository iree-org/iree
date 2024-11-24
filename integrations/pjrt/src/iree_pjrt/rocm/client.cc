// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree_pjrt/rocm/client.h"

#include "iree/hal/drivers/hip/api.h"
#include "iree/hal/drivers/hip/hip_device.h"
#include "iree/hal/drivers/hip/registration/driver_module.h"

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

  IREE_RETURN_IF_ERROR(iree_hal_hip_driver_create(driver_name, &driver_options,
                                                  &default_params,
                                                  host_allocator_, out_driver));
  logger().debug("ROCM driver created");

  // retrieve the target name of current available device
  iree_host_size_t device_info_count;
  iree_hal_device_info_t* device_infos;
  IREE_RETURN_IF_ERROR(iree_hal_driver_query_available_devices(
      *out_driver, host_allocator_, &device_info_count, &device_infos));

  // TODO: here we just use the target name of the first available device,
  // but ideally we should find the device which will run the program
  if (device_info_count > 0) {
    hipDeviceProp_tR0000 props;
    IREE_RETURN_IF_ERROR(iree_hal_hip_get_device_properties(
        *out_driver, device_infos->device_id, &props));

    // `gcnArchName` comes back like gfx90a:sramecc+:xnack- for a fully
    // specified target. However the IREE target-chip flag only expects the
    // prefix. refer to
    // https://github.com/iree-org/iree-turbine/blob/965247e/iree/turbine/runtime/device.py#L495
    std::string_view target = props.gcnArchName;
    if (auto pos = target.find(':'); pos != target.npos) {
      target = target.substr(0, pos);
    }

    hip_target_ = target;
    logger().debug("HIP target detected: " + hip_target_);
  }

  return iree_ok_status();
}

bool ROCMClientInstance::SetDefaultCompilerFlags(CompilerJob* compiler_job) {
  std::vector<std::string> flags = {
      "--iree-hal-target-backends=rocm",
  };

  if (!hip_target_.empty()) {
    flags.push_back("--iree-hip-target=" + hip_target_);
  }

  for (auto flag : flags) {
    if (!compiler_job->SetFlag(flag.c_str())) return false;
  }
  return true;
}

}  // namespace iree::pjrt::rocm
