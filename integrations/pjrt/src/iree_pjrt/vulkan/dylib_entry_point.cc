// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree_pjrt/common/dylib_platform.h"
#include "iree_pjrt/vulkan/client.h"

// Provides the shared library exports.
#include "iree_pjrt/common/dylib_entry_point.cc.inc"

namespace iree::pjrt {
namespace {

// Declared but not implemented by the include file.
void InitializeAPI(PJRT_Api* api) {
  BindApi<DylibPlatform, vulkan::VulkanClientInstance>(api);
}

}  // namespace
}  // namespace iree::pjrt
