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

#include "iree/hal/vulkan/vulkan_driver.h"

#include <memory>

#include "absl/container/inlined_vector.h"
#include "absl/flags/flag.h"
#include "iree/base/memory.h"
#include "iree/base/status.h"
#include "iree/base/tracing.h"
#include "iree/hal/device_info.h"
#include "iree/hal/vulkan/extensibility_util.h"
#include "iree/hal/vulkan/status_util.h"

ABSL_FLAG(bool, vulkan_renderdoc, false, "Enables RenderDoc API integration.");
ABSL_FLAG(int, vulkan_default_index, 0, "Index of the default Vulkan device.");

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
  return DeviceInfo("vulkan", std::move(name), supported_features,
                    reinterpret_cast<DriverDeviceID>(physical_device));
}

}  // namespace

// static
StatusOr<ref_ptr<VulkanDriver>> VulkanDriver::Create(
    Options options, ref_ptr<DynamicSymbols> syms) {
  IREE_TRACE_SCOPE0("VulkanDriver::Create");

  // Load and connect to RenderDoc before instance creation.
  // Note: RenderDoc assumes that only a single VkDevice is used:
  //   https://renderdoc.org/docs/behind_scenes/vulkan_support.html#current-support
  std::unique_ptr<RenderDocCaptureManager> renderdoc_capture_manager;
  if (absl::GetFlag(FLAGS_vulkan_renderdoc)) {
    renderdoc_capture_manager = std::make_unique<RenderDocCaptureManager>();
    IREE_RETURN_IF_ERROR(renderdoc_capture_manager->Connect());
  }

  // Find the layers and extensions we need (or want) that are also available
  // on the instance. This will fail when required ones are not present.
  IREE_ASSIGN_OR_RETURN(
      auto enabled_layer_names,
      MatchAvailableInstanceLayers(options.instance_extensibility, *syms));
  IREE_ASSIGN_OR_RETURN(
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
  IREE_RETURN_IF_ERROR(syms->LoadFromInstance(instance));

  // The real debug messenger (not just the static one used above) can now be
  // created as we've loaded all the required symbols.
  // TODO(benvanik): strip in release builds.
  std::unique_ptr<DebugReporter> debug_reporter;
  if (instance_extensions.debug_utils) {
    IREE_ASSIGN_OR_RETURN(debug_reporter,
                          DebugReporter::CreateDebugUtilsMessenger(
                              instance, syms,
                              /*allocation_callbacks=*/nullptr));
  } else if (instance_extensions.debug_report) {
    IREE_ASSIGN_OR_RETURN(
        debug_reporter, DebugReporter::CreateDebugReportCallback(
                            instance, syms, /*allocation_callbacks=*/nullptr));
  }

  return assign_ref(new VulkanDriver(std::move(syms), instance,
                                     /*owns_instance=*/true,
                                     std::move(debug_reporter),
                                     std::move(options.device_extensibility),
                                     std::move(renderdoc_capture_manager)));
}

// static
StatusOr<ref_ptr<VulkanDriver>> VulkanDriver::CreateUsingInstance(
    Options options, ref_ptr<DynamicSymbols> syms, VkInstance instance) {
  IREE_TRACE_SCOPE0("VulkanDriver::CreateUsingInstance");

  if (instance == VK_NULL_HANDLE) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "VkInstance must not be VK_NULL_HANDLE";
  }

  // Find the extensions we need (or want) that are also available on the
  // instance. This will fail when required ones are not present.
  //
  // Since the instance is already created, we can't actually enable any
  // extensions or query if they are really enabled - we just have to trust
  // that the caller already enabled them for us (or we may fail later).
  IREE_ASSIGN_OR_RETURN(
      auto enabled_extension_names,
      MatchAvailableInstanceExtensions(options.instance_extensibility, *syms));
  auto instance_extensions =
      PopulateEnabledInstanceExtensions(enabled_extension_names);

  IREE_RETURN_IF_ERROR(syms->LoadFromInstance(instance));

  // TODO(benvanik): strip in release builds.
  std::unique_ptr<DebugReporter> debug_reporter;
  if (instance_extensions.debug_utils) {
    IREE_ASSIGN_OR_RETURN(debug_reporter,
                          DebugReporter::CreateDebugUtilsMessenger(
                              instance, syms,
                              /*allocation_callbacks=*/nullptr));
  } else if (instance_extensions.debug_report) {
    IREE_ASSIGN_OR_RETURN(
        debug_reporter, DebugReporter::CreateDebugReportCallback(
                            instance, syms, /*allocation_callbacks=*/nullptr));
  }

  // Note: no RenderDocCaptureManager here since the VkInstance is already
  // created externally. Applications using this function must provide their
  // own RenderDoc / debugger integration as desired.

  return assign_ref(new VulkanDriver(
      std::move(syms), instance, /*owns_instance=*/false,
      std::move(debug_reporter), std::move(options.device_extensibility),
      /*debug_capture_manager=*/nullptr));
}

VulkanDriver::VulkanDriver(
    ref_ptr<DynamicSymbols> syms, VkInstance instance, bool owns_instance,
    std::unique_ptr<DebugReporter> debug_reporter,
    ExtensibilitySpec device_extensibility_spec,
    std::unique_ptr<RenderDocCaptureManager> renderdoc_capture_manager)
    : Driver("vulkan"),
      syms_(std::move(syms)),
      instance_(instance),
      owns_instance_(owns_instance),
      debug_reporter_(std::move(debug_reporter)),
      device_extensibility_spec_(std::move(device_extensibility_spec)),
      renderdoc_capture_manager_(std::move(renderdoc_capture_manager)) {}

VulkanDriver::~VulkanDriver() {
  IREE_TRACE_SCOPE0("VulkanDriver::dtor");
  debug_reporter_.reset();
  if (owns_instance_) {
    syms()->vkDestroyInstance(instance_, /*pAllocator=*/nullptr);
  }
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
    IREE_ASSIGN_OR_RETURN(auto device_info,
                          PopulateDeviceInfo(physical_device, syms()));
    device_infos.push_back(std::move(device_info));
  }
  return device_infos;
}

StatusOr<ref_ptr<Device>> VulkanDriver::CreateDefaultDevice() {
  IREE_TRACE_SCOPE0("VulkanDriver::CreateDefaultDevice");

  // Query available devices.
  IREE_ASSIGN_OR_RETURN(auto available_devices, EnumerateAvailableDevices());
  int default_device_index = absl::GetFlag(FLAGS_vulkan_default_index);
  if (default_device_index < 0 ||
      default_device_index >= available_devices.size()) {
    return NotFoundErrorBuilder(IREE_LOC)
           << "Device index " << default_device_index << " not found "
           << "(of " << available_devices.size() << ")";
  }

  // Just create the first one we find.
  return CreateDevice(available_devices[default_device_index].device_id());
}

StatusOr<ref_ptr<Device>> VulkanDriver::CreateDevice(DriverDeviceID device_id) {
  IREE_TRACE_SCOPE0("VulkanDriver::CreateDevice");

  auto physical_device = reinterpret_cast<VkPhysicalDevice>(device_id);
  IREE_ASSIGN_OR_RETURN(auto device_info,
                        PopulateDeviceInfo(physical_device, syms()));

  // Attempt to create the device.
  // This may fail if the device was enumerated but is in exclusive use,
  // disabled by the system, or permission is denied.
  IREE_ASSIGN_OR_RETURN(
      auto device,
      VulkanDevice::Create(add_ref(this), instance(), device_info,
                           physical_device, device_extensibility_spec_, syms(),
                           renderdoc_capture_manager_.get()));

  LOG(INFO) << "Created Vulkan Device: " << device->info().name();

  return device;
}

StatusOr<ref_ptr<Device>> VulkanDriver::WrapDevice(
    VkPhysicalDevice physical_device, VkDevice logical_device,
    const QueueSet& compute_queue_set, const QueueSet& transfer_queue_set) {
  IREE_TRACE_SCOPE0("VulkanDriver::WrapDevice");

  IREE_ASSIGN_OR_RETURN(auto device_info,
                        PopulateDeviceInfo(physical_device, syms()));

  // Attempt to create the device.
  // This may fail if the VkDevice does not support all necessary features.
  IREE_ASSIGN_OR_RETURN(
      auto device,
      VulkanDevice::Wrap(add_ref(this), device_info, physical_device,
                         logical_device, device_extensibility_spec_,
                         compute_queue_set, transfer_queue_set, syms()));
  return device;
}

}  // namespace vulkan
}  // namespace hal
}  // namespace iree
