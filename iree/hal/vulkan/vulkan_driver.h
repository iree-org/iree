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

#ifndef IREE_HAL_VULKAN_VULKAN_DRIVER_H_
#define IREE_HAL_VULKAN_VULKAN_DRIVER_H_

#include <vulkan/vulkan.h>

#include <memory>
#include <vector>

#include "iree/hal/driver.h"
#include "iree/hal/vulkan/debug_reporter.h"
#include "iree/hal/vulkan/dynamic_symbols.h"
#include "iree/hal/vulkan/extensibility_util.h"
#include "iree/hal/vulkan/vulkan_device.h"

namespace iree {
namespace hal {
namespace vulkan {

class VulkanDriver final : public Driver {
 public:
  struct Options {
    // Vulkan version that will be requested.
    // Driver creation will fail if the required version is not available.
    uint32_t api_version = VK_API_VERSION_1_0;

    // Extensibility descriptions for instances and devices.
    // Device descriptions will be used for all devices created by the driver.
    ExtensibilitySpec instance_extensibility;
    ExtensibilitySpec device_extensibility;
  };

  // Creates a VulkanDriver that manages its own VkInstance.
  static StatusOr<ref_ptr<VulkanDriver>> Create(Options options,
                                                ref_ptr<DynamicSymbols> syms);

  // Creates a VulkanDriver that shares an externally managed VkInstance.
  //
  // |options| are checked for compatibility.
  //
  // |syms| must at least have |vkGetInstanceProcAddr| set. Other symbols will
  // be loaded as needed from |instance|.
  //
  // |instance| must remain valid for the life of the returned VulkanDriver.
  static StatusOr<ref_ptr<VulkanDriver>> CreateUsingInstance(
      Options options, ref_ptr<DynamicSymbols> syms, VkInstance instance);

  ~VulkanDriver() override;

  const ref_ptr<DynamicSymbols>& syms() const { return syms_; }

  VkInstance instance() const { return instance_; }

  StatusOr<std::vector<DeviceInfo>> EnumerateAvailableDevices() override;

  StatusOr<ref_ptr<Device>> CreateDefaultDevice() override;

  StatusOr<ref_ptr<Device>> CreateDevice(
      const DeviceInfo& device_info) override;

  // Creates a device that wraps an externally managed VkDevice.
  //
  // The device will schedule commands against the provided queues.
  StatusOr<ref_ptr<Device>> WrapDevice(VkPhysicalDevice physical_device,
                                       VkDevice logical_device,
                                       const QueueSet& compute_queue_set,
                                       const QueueSet& transfer_queue_set);

 private:
  VulkanDriver(ref_ptr<DynamicSymbols> syms, VkInstance instance,
               bool owns_instance,
               std::unique_ptr<DebugReporter> debug_reporter,
               ExtensibilitySpec device_extensibility_spec);

  ref_ptr<DynamicSymbols> syms_;
  VkInstance instance_;
  bool owns_instance_;
  std::unique_ptr<DebugReporter> debug_reporter_;
  ExtensibilitySpec device_extensibility_spec_;
};

}  // namespace vulkan
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_VULKAN_VULKAN_DRIVER_H_
