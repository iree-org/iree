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

#include "hal/driver.h"
#include "hal/vulkan/debug_reporter.h"
#include "hal/vulkan/dynamic_symbols.h"
#include "hal/vulkan/extensibility_util.h"

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

  static StatusOr<std::shared_ptr<VulkanDriver>> Create(
      Options options, ref_ptr<DynamicSymbols> syms);

  // TODO(benvanik): method to wrap an existing instance/device (interop).

  // Private constructor.
  struct CtorKey {
   private:
    friend class VulkanDriver;
    CtorKey() = default;
  };
  VulkanDriver(CtorKey ctor_key, ref_ptr<DynamicSymbols> syms,
               VkInstance instance,
               std::unique_ptr<DebugReporter> debug_reporter,
               ExtensibilitySpec device_extensibility_spec);
  ~VulkanDriver() override;

  const ref_ptr<DynamicSymbols>& syms() const { return syms_; }

  VkInstance instance() const { return instance_; }

  StatusOr<std::vector<DeviceInfo>> EnumerateAvailableDevices() override;

  StatusOr<std::shared_ptr<Device>> CreateDefaultDevice() override;

  StatusOr<std::shared_ptr<Device>> CreateDevice(
      const DeviceInfo& device_info) override;

 private:
  ref_ptr<DynamicSymbols> syms_;
  VkInstance instance_;
  std::unique_ptr<DebugReporter> debug_reporter_;
  ExtensibilitySpec device_extensibility_spec_;
};

}  // namespace vulkan
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_VULKAN_VULKAN_DRIVER_H_
