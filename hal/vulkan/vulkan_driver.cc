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

#include "hal/vulkan/vulkan_driver.h"

#include <memory>

#include "absl/container/inlined_vector.h"
#include "base/memory.h"
#include "base/status.h"
#include "base/tracing.h"
#include "hal/device_info.h"
#include "hal/vulkan/extensibility_util.h"
#include "hal/vulkan/status_util.h"
#include "hal/vulkan/vulkan_device.h"

namespace iree {
namespace hal {
namespace vulkan {

namespace {

// Returns a VkApplicationInfo struct populated with the default app info.
// We may allow hosting applications to override this via weak-linkage if it's
// useful, otherwise this is enough to create the application.
VkApplicationInfo GetDefaultApplicationInfo() {
  VkApplicationInfo info;
  info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  info.pNext = nullptr;
  info.pApplicationName = "IREE-ML";
  info.applicationVersion = 0;
  info.pEngineName = "IREE";
  info.engineVersion = 0;
  info.apiVersion = VK_API_VERSION_1_0;
  return info;
}

// Populates device information from the given Vulkan physical device handle.
StatusOr<DeviceInfo> PopulateDeviceInfo(VkPhysicalDevice physical_device,
                                        const ref_ptr<DynamicSymbols>& syms) {
  VkPhysicalDeviceFeatures physical_device_features;
  syms->vkGetPhysicalDeviceFeatures(physical_device, &physical_device_features);
  // TODO(benvanik): check and optionally require these features:
  // - physical_device_features.robustBufferAccess
  // - physical_device_features.shaderInt16
  // - physical_device_features.shaderInt64
  // - physical_device_features.shaderFloat64

  VkPhysicalDeviceProperties physical_device_properties;
  syms->vkGetPhysicalDeviceProperties(physical_device,
                                      &physical_device_properties);
  // TODO(benvanik): check and optionally require reasonable limits.

  // TODO(benvanik): more clever/sanitized device naming.
  std::string name = std::string(physical_device_properties.deviceName);

  DeviceFeatureBitfield supported_features = DeviceFeature::kNone;
  // TODO(benvanik): implement debugging/profiling features.
  // TODO(benvanik): use props to determine if we have timing info.
  // supported_features |= DeviceFeature::kDebugging;
  // supported_features |= DeviceFeature::kCoverage;
  // supported_features |= DeviceFeature::kProfiling;
  return DeviceInfo(std::move(name), supported_features, physical_device);
}

}  // namespace

// static
StatusOr<std::shared_ptr<VulkanDriver>> VulkanDriver::Create(
    Options options, ref_ptr<DynamicSymbols> syms) {
  IREE_TRACE_SCOPE0("VulkanDriver::Create");

  // Find the layers and extensions we need (or want) that are also available
  // on the instance. This will fail when required ones are not present.
  ASSIGN_OR_RETURN(
      auto enabled_layer_names,
      MatchAvailableInstanceLayers(options.instance_extensibility, *syms));
  ASSIGN_OR_RETURN(
      auto enabled_extension_names,
      MatchAvailableInstanceExtensions(options.instance_extensibility, *syms));
  auto instance_extensions =
      PopulateEnabledInstanceExtensions(enabled_extension_names);

  // Create the instance this driver will use for all requests.
  VkApplicationInfo app_info = GetDefaultApplicationInfo();
  app_info.apiVersion = options.api_version;
  VkInstanceCreateInfo create_info;
  create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  create_info.pNext = nullptr;
  create_info.flags = 0;
  create_info.pApplicationInfo = &app_info;
  create_info.enabledLayerCount = enabled_layer_names.size();
  create_info.ppEnabledLayerNames = enabled_layer_names.data();
  create_info.enabledExtensionCount = enabled_extension_names.size();
  create_info.ppEnabledExtensionNames = enabled_extension_names.data();

  // If we have the debug_utils extension then we can chain a one-shot messenger
  // callback that we can use to log out the instance creation errors. Once we
  // have the real instance we can then register a real messenger.
  union {
    VkDebugUtilsMessengerCreateInfoEXT debug_utils_create_info;
    VkDebugReportCallbackCreateInfoEXT debug_report_create_info;
  };
  if (instance_extensions.debug_utils) {
    create_info.pNext = &debug_utils_create_info;
    DebugReporter::PopulateStaticCreateInfo(&debug_utils_create_info);
  } else if (instance_extensions.debug_report) {
    create_info.pNext = &debug_report_create_info;
    DebugReporter::PopulateStaticCreateInfo(&debug_report_create_info);
  }

  // Some ICDs appear to leak in here, out of our control.
  // Warning: leak checks remain disabled if an error is returned.
  IREE_DISABLE_LEAK_CHECKS();
  VkInstance instance = VK_NULL_HANDLE;
  VK_RETURN_IF_ERROR(
      syms->vkCreateInstance(&create_info, /*pAllocator=*/nullptr, &instance))
      << "Unable to create Vulkan instance";
  IREE_ENABLE_LEAK_CHECKS();

  // TODO(benvanik): enable validation layers if needed.

  // Now that the instance has been created we can fetch all of the instance
  // symbols.
  RETURN_IF_ERROR(syms->LoadFromInstance(instance));

  // The real debug messenger (not just the static one used above) can now be
  // created as we've loaded all the required symbols.
  // TODO(benvanik): strip in release builds.
  std::unique_ptr<DebugReporter> debug_reporter;
  if (instance_extensions.debug_utils) {
    ASSIGN_OR_RETURN(debug_reporter, DebugReporter::CreateDebugUtilsMessenger(
                                         instance, syms,
                                         /*allocation_callbacks=*/nullptr));
  } else if (instance_extensions.debug_report) {
    ASSIGN_OR_RETURN(debug_reporter,
                     DebugReporter::CreateDebugReportCallback(
                         instance, syms, /*allocation_callbacks=*/nullptr));
  }

  return std::make_shared<VulkanDriver>(
      CtorKey{}, std::move(syms), instance, std::move(debug_reporter),
      std::move(options.device_extensibility));
}

VulkanDriver::VulkanDriver(CtorKey ctor_key, ref_ptr<DynamicSymbols> syms,
                           VkInstance instance,
                           std::unique_ptr<DebugReporter> debug_reporter,
                           ExtensibilitySpec device_extensibility_spec)
    : Driver("vulkan"),
      syms_(std::move(syms)),
      instance_(instance),
      debug_reporter_(std::move(debug_reporter)),
      device_extensibility_spec_(std::move(device_extensibility_spec)) {}

VulkanDriver::~VulkanDriver() {
  IREE_TRACE_SCOPE0("VulkanDriver::dtor");
  debug_reporter_.reset();
  syms()->vkDestroyInstance(instance_, /*pAllocator=*/nullptr);
}

StatusOr<std::vector<DeviceInfo>> VulkanDriver::EnumerateAvailableDevices() {
  IREE_TRACE_SCOPE0("VulkanDriver::EnumerateAvailableDevices");

  // Query all available devices (at this moment, note that this may change!).
  uint32_t physical_device_count = 0;
  VK_RETURN_IF_ERROR(syms()->vkEnumeratePhysicalDevices(
      instance_, &physical_device_count, nullptr));
  absl::InlinedVector<VkPhysicalDevice, 2> physical_devices(
      physical_device_count);
  VK_RETURN_IF_ERROR(syms()->vkEnumeratePhysicalDevices(
      instance_, &physical_device_count, physical_devices.data()));

  // Convert to our HAL structure.
  std::vector<DeviceInfo> device_infos;
  device_infos.reserve(physical_device_count);
  for (auto physical_device : physical_devices) {
    // TODO(benvanik): if we fail should we just ignore the device in the list?
    ASSIGN_OR_RETURN(auto device_info,
                     PopulateDeviceInfo(physical_device, syms()));
    device_infos.push_back(std::move(device_info));
  }
  return device_infos;
}

StatusOr<std::shared_ptr<Device>> VulkanDriver::CreateDefaultDevice() {
  IREE_TRACE_SCOPE0("VulkanDriver::CreateDefaultDevice");

  // Query available devices.
  ASSIGN_OR_RETURN(auto available_devices, EnumerateAvailableDevices());
  if (available_devices.empty()) {
    return NotFoundErrorBuilder(IREE_LOC) << "No devices are available";
  }

  // Just create the first one we find.
  return CreateDevice(available_devices.front());
}

StatusOr<std::shared_ptr<Device>> VulkanDriver::CreateDevice(
    const DeviceInfo& device_info) {
  IREE_TRACE_SCOPE0("VulkanDriver::CreateDevice");

  auto physical_device =
      static_cast<VkPhysicalDevice>(device_info.driver_handle());

  // Attempt to create the device.
  // This may fail if the device was enumerated but is in exclusive use,
  // disabled by the system, or permission is denied.
  ASSIGN_OR_RETURN(auto device,
                   VulkanDevice::Create(device_info, physical_device,
                                        device_extensibility_spec_, syms()));

  return device;
}

}  // namespace vulkan
}  // namespace hal
}  // namespace iree
