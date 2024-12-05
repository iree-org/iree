// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree_pjrt/vulkan/client.h"

#include "iree/hal/drivers/vulkan/registration/driver_module.h"

namespace iree::pjrt::vulkan {

VulkanClientInstance::VulkanClientInstance(std::unique_ptr<Platform> platform)
    : ClientInstance(std::move(platform)) {
  // Seems that it must match how registered. Action at a distance not
  // great.
  // TODO: Get this when constructing the client so it is guaranteed to
  // match.
  cached_platform_name_ = "iree_vulkan";
  IREE_CHECK_OK(iree_hal_vulkan_driver_module_register(driver_registry_));
}

VulkanClientInstance::~VulkanClientInstance() {}

iree_status_t VulkanClientInstance::CreateDriver(
    iree_hal_driver_t** out_driver) {
  iree_string_view_t driver_name = iree_make_cstring_view("vulkan");
  IREE_RETURN_IF_ERROR(iree_hal_driver_registry_try_create(
      driver_registry_, driver_name, host_allocator_, out_driver));
  logger().debug("Vulkan driver created");
  return iree_ok_status();
}

bool VulkanClientInstance::SetDefaultCompilerFlags(CompilerJob* compiler_job) {
  return compiler_job->SetFlag("--iree-hal-target-backends=vulkan");
}

}  // namespace iree::pjrt::vulkan
