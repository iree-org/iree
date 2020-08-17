// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "iree/hal/vulkan/dynamic_symbols.h"

#include "iree/hal/vulkan/status_util.h"
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
  auto status_or_syms = DynamicSymbols::CreateFromSystemLoader();
  IREE_ASSERT_OK(status_or_syms);
  ref_ptr<DynamicSymbols> syms = std::move(status_or_syms.value());

  // Create and destroy a VkInstance using the symbols. This is mainly testing
  // that the symbols were loaded successfully and are actually able to be used.
  VkApplicationInfo app_info = GetApplicationInfo();
  VkInstanceCreateInfo create_info = GetInstanceCreateInfo(&app_info);
  VkInstance instance = VK_NULL_HANDLE;
  VK_CHECK_OK(
      syms->vkCreateInstance(&create_info, /*pAllocator=*/nullptr, &instance));

  IREE_ASSERT_OK(syms->LoadFromInstance(instance));

  syms->vkDestroyInstance(instance, /*pAllocator=*/nullptr);
}

}  // namespace
}  // namespace vulkan
}  // namespace hal
}  // namespace iree
