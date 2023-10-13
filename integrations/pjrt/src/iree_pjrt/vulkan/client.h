// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_PJRT_PLUGIN_PJRT_VULKAN_CLIENT_H_
#define IREE_PJRT_PLUGIN_PJRT_VULKAN_CLIENT_H_

#include "iree/hal/drivers/vulkan/api.h"
#include "iree_pjrt/common/api_impl.h"

namespace iree::pjrt::vulkan {

class VulkanClientInstance final : public ClientInstance {
 public:
  VulkanClientInstance(std::unique_ptr<Platform> platform);
  ~VulkanClientInstance();
  iree_status_t CreateDriver(iree_hal_driver_t** out_driver) override;
  bool SetDefaultCompilerFlags(CompilerJob* compiler_job) override;

 private:
};

}  // namespace iree::pjrt::vulkan

#endif  // IREE_PJRT_PLUGIN_PJRT_VULKAN_CLIENT_H_
