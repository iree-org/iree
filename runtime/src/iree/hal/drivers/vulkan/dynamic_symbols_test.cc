// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/dynamic_symbols.h"

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace hal {
namespace vulkan {
namespace {

VkApplicationInfo GetApplicationInfo() {
  VkApplicationInfo app_info;
  app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app_info.pNext = nullptr;
  app_info.pApplicationName = "IREE-ML-TEST";
  app_info.applicationVersion = 0;
  app_info.pEngineName = "IREE";
  app_info.engineVersion = 0;
  app_info.apiVersion = VK_API_VERSION_1_0;
  return app_info;
}

VkInstanceCreateInfo GetInstanceCreateInfo(VkApplicationInfo* app_info) {
  VkInstanceCreateInfo create_info;
  create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  create_info.pNext = nullptr;
  create_info.flags = 0;
  create_info.pApplicationInfo = app_info;
  create_info.enabledLayerCount = 0;
  create_info.ppEnabledLayerNames = nullptr;
  create_info.enabledExtensionCount = 0;
  create_info.ppEnabledExtensionNames = nullptr;
  return create_info;
}

TEST(DynamicSymbolsTest, CreateFromSystemLoader) {
  iree::ref_ptr<iree::hal::vulkan::DynamicSymbols> syms;
  IREE_ASSERT_OK(DynamicSymbols::CreateFromSystemLoader(&syms));

  // Create and destroy a VkInstance using the symbols. This is mainly testing
  // that the symbols were loaded successfully and are actually able to be used.
  VkApplicationInfo app_info = GetApplicationInfo();
  VkInstanceCreateInfo create_info = GetInstanceCreateInfo(&app_info);
  VkInstance instance = VK_NULL_HANDLE;
  ASSERT_EQ(VK_SUCCESS, syms->vkCreateInstance(
                            &create_info, /*pAllocator=*/nullptr, &instance));

  IREE_ASSERT_OK(syms->LoadFromInstance(instance));

  syms->vkDestroyInstance(instance, /*pAllocator=*/nullptr);
}

}  // namespace
}  // namespace vulkan
}  // namespace hal
}  // namespace iree
