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

#include "iree/hal/vulkan/vulkan_device.h"

#include <functional>
#include <utility>

#include "absl/container/inlined_vector.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "iree/base/math.h"
#include "iree/base/status.h"
#include "iree/base/tracing.h"
#include "iree/hal/command_buffer_validation.h"
#include "iree/hal/command_queue.h"
#include "iree/hal/semaphore.h"
#include "iree/hal/vulkan/direct_command_buffer.h"
#include "iree/hal/vulkan/direct_command_queue.h"
#include "iree/hal/vulkan/dynamic_symbols.h"
#include "iree/hal/vulkan/extensibility_util.h"
#include "iree/hal/vulkan/native_descriptor_set.h"
#include "iree/hal/vulkan/native_event.h"
#include "iree/hal/vulkan/native_timeline_semaphore.h"
#include "iree/hal/vulkan/pipeline_cache.h"
#include "iree/hal/vulkan/pipeline_executable_layout.h"
#include "iree/hal/vulkan/status_util.h"
#include "iree/hal/vulkan/vma_allocator.h"

namespace iree {
namespace hal {
namespace vulkan {

namespace {

constexpr uint32_t kInvalidQueueFamilyIndex = -1;

struct QueueFamilyInfo {
  uint32_t dispatch_index = kInvalidQueueFamilyIndex;
  uint32_t dispatch_queue_count = 0;
  uint32_t transfer_index = kInvalidQueueFamilyIndex;
  uint32_t transfer_queue_count = 0;
};

// Finds the first queue in the listing (which is usually the
// driver-preferred) that has all of the |required_queue_flags| and none of
// the |excluded_queue_flags|. Returns kInvalidQueueFamilyIndex if no matching
// queue is found.
uint32_t FindFirstQueueFamilyWithFlags(
    absl::Span<const VkQueueFamilyProperties> queue_family_properties,
    uint32_t required_queue_flags, uint32_t excluded_queue_flags) {
  for (int queue_family_index = 0;
       queue_family_index < queue_family_properties.size();
       ++queue_family_index) {
    const auto& properties = queue_family_properties[queue_family_index];
    if ((properties.queueFlags & required_queue_flags) ==
            required_queue_flags &&
        (properties.queueFlags & excluded_queue_flags) == 0) {
      return queue_family_index;
    }
  }
  return kInvalidQueueFamilyIndex;
}

// Selects queue family indices for compute and transfer queues.
// Note that both queue families may be the same if there is only one family
// available.
StatusOr<QueueFamilyInfo> SelectQueueFamilies(
    VkPhysicalDevice physical_device, const ref_ptr<DynamicSymbols>& syms) {
  // Enumerate queue families available on the device.
  uint32_t queue_family_count = 0;
  syms->vkGetPhysicalDeviceQueueFamilyProperties(physical_device,
                                                 &queue_family_count, nullptr);
  absl::InlinedVector<VkQueueFamilyProperties, 4> queue_family_properties(
      queue_family_count);
  syms->vkGetPhysicalDeviceQueueFamilyProperties(
      physical_device, &queue_family_count, queue_family_properties.data());

  QueueFamilyInfo queue_family_info;

  // Try to find a dedicated compute queue (no graphics caps).
  // Some may support both transfer and compute. If that fails then fallback
  // to any queue that supports compute.
  queue_family_info.dispatch_index = FindFirstQueueFamilyWithFlags(
      queue_family_properties, VK_QUEUE_COMPUTE_BIT, VK_QUEUE_GRAPHICS_BIT);
  if (queue_family_info.dispatch_index == kInvalidQueueFamilyIndex) {
    queue_family_info.dispatch_index = FindFirstQueueFamilyWithFlags(
        queue_family_properties, VK_QUEUE_COMPUTE_BIT, 0);
  }
  if (queue_family_info.dispatch_index == kInvalidQueueFamilyIndex) {
    return NotFoundErrorBuilder(IREE_LOC)
           << "Unable to find any queue family support compute operations";
  }
  queue_family_info.dispatch_queue_count =
      queue_family_properties[queue_family_info.dispatch_index].queueCount;

  // Try to find a dedicated transfer queue (no compute or graphics caps).
  // Not all devices have one, and some have only a queue family for
  // everything and possibly a queue family just for compute/etc. If that
  // fails then fallback to any queue that supports transfer. Finally, if
  // /that/ fails then we just won't create a transfer queue and instead use
  // the compute queue for all operations.
  queue_family_info.transfer_index = FindFirstQueueFamilyWithFlags(
      queue_family_properties, VK_QUEUE_TRANSFER_BIT,
      VK_QUEUE_COMPUTE_BIT | VK_QUEUE_GRAPHICS_BIT);
  if (queue_family_info.transfer_index == kInvalidQueueFamilyIndex) {
    queue_family_info.transfer_index = FindFirstQueueFamilyWithFlags(
        queue_family_properties, VK_QUEUE_TRANSFER_BIT, VK_QUEUE_GRAPHICS_BIT);
  }
  if (queue_family_info.transfer_index == kInvalidQueueFamilyIndex) {
    queue_family_info.transfer_index = FindFirstQueueFamilyWithFlags(
        queue_family_properties, VK_QUEUE_TRANSFER_BIT, 0);
  }
  if (queue_family_info.transfer_index != kInvalidQueueFamilyIndex) {
    queue_family_info.transfer_queue_count =
        queue_family_properties[queue_family_info.transfer_index].queueCount;
  }

  // Ensure that we don't share the dispatch queues with transfer queues if
  // that would put us over the queue count.
  if (queue_family_info.dispatch_index == queue_family_info.transfer_index) {
    queue_family_info.transfer_queue_count = std::min(
        queue_family_properties[queue_family_info.dispatch_index].queueCount -
            queue_family_info.dispatch_queue_count,
        queue_family_info.transfer_queue_count);
  }

  return queue_family_info;
}

// Creates a transient command pool for the given queue family.
// Command buffers allocated from the pool must only be issued on queues
// belonging to the specified family.
StatusOr<ref_ptr<VkCommandPoolHandle>> CreateTransientCommandPool(
    const ref_ptr<VkDeviceHandle>& logical_device,
    uint32_t queue_family_index) {
  VkCommandPoolCreateInfo create_info;
  create_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  create_info.pNext = nullptr;
  create_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
  create_info.queueFamilyIndex = queue_family_index;

  auto command_pool = make_ref<VkCommandPoolHandle>(logical_device);
  VK_RETURN_IF_ERROR(logical_device->syms()->vkCreateCommandPool(
      *logical_device, &create_info, logical_device->allocator(),
      command_pool->mutable_value()));
  return command_pool;
}

// Creates command queues for the given sets of queues.
absl::InlinedVector<std::unique_ptr<CommandQueue>, 4> CreateCommandQueues(
    const DeviceInfo& device_info,
    const ref_ptr<VkDeviceHandle>& logical_device,
    const QueueSet& compute_queue_set, const QueueSet& transfer_queue_set,
    const ref_ptr<DynamicSymbols>& syms) {
  absl::InlinedVector<std::unique_ptr<CommandQueue>, 4> command_queues;

  uint64_t compute_queue_count = CountOnes64(compute_queue_set.queue_indices);
  for (uint32_t i = 0; i < compute_queue_count; ++i) {
    if (!(compute_queue_set.queue_indices & (1 << i))) continue;

    VkQueue queue = VK_NULL_HANDLE;
    syms->vkGetDeviceQueue(*logical_device,
                           compute_queue_set.queue_family_index, i, &queue);
    std::string queue_name = absl::StrCat(device_info.name(), ":d", i);
    command_queues.push_back(absl::make_unique<DirectCommandQueue>(
        std::move(queue_name),
        CommandCategory::kDispatch | CommandCategory::kTransfer, logical_device,
        queue));
  }

  uint64_t transfer_queue_count = CountOnes64(transfer_queue_set.queue_indices);
  for (uint32_t i = 0; i < transfer_queue_count; ++i) {
    if (!(transfer_queue_set.queue_indices & (1 << i))) continue;

    VkQueue queue = VK_NULL_HANDLE;
    syms->vkGetDeviceQueue(*logical_device,
                           transfer_queue_set.queue_family_index, i, &queue);
    std::string queue_name = absl::StrCat(device_info.name(), ":t", i);
    command_queues.push_back(absl::make_unique<DirectCommandQueue>(
        std::move(queue_name), CommandCategory::kTransfer, logical_device,
        queue));
  }

  return command_queues;
}

}  // namespace

// static
StatusOr<ref_ptr<VulkanDevice>> VulkanDevice::Create(
    ref_ptr<Driver> driver, VkInstance instance, const DeviceInfo& device_info,
    VkPhysicalDevice physical_device,
    const ExtensibilitySpec& extensibility_spec,
    const ref_ptr<DynamicSymbols>& syms,
    DebugCaptureManager* debug_capture_manager) {
  IREE_TRACE_SCOPE0("VulkanDevice::Create");

  if (!extensibility_spec.optional_layers.empty() ||
      !extensibility_spec.required_layers.empty()) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "Device layers are deprecated and unsupported by IREE";
  }

  // Find the extensions we need (or want) that are also available
  // on the device. This will fail when required ones are not present.
  ASSIGN_OR_RETURN(auto enabled_extension_names,
                   MatchAvailableDeviceExtensions(physical_device,
                                                  extensibility_spec, *syms));
  auto enabled_device_extensions =
      PopulateEnabledDeviceExtensions(enabled_extension_names);

  // Find queue families we will expose as HAL queues.
  ASSIGN_OR_RETURN(auto queue_family_info,
                   SelectQueueFamilies(physical_device, syms));

  // Limit the number of queues we create (for now).
  // We may want to allow this to grow, but each queue adds overhead and we
  // need to measure to make sure we can effectively use them all.
  queue_family_info.dispatch_queue_count =
      std::min(2u, queue_family_info.dispatch_queue_count);
  queue_family_info.transfer_queue_count =
      std::min(1u, queue_family_info.transfer_queue_count);
  bool has_dedicated_transfer_queues =
      queue_family_info.transfer_queue_count > 0;

  // Setup the queue info we'll be using.
  // Each queue here (created from within a family) will map to a HAL queue.
  //
  // Note that we need to handle the case where we have transfer queues that
  // are of the same queue family as the dispatch queues: Vulkan requires that
  // all queues created from the same family are done in the same
  // VkDeviceQueueCreateInfo struct.
  DVLOG(1) << "Creating " << queue_family_info.dispatch_queue_count
           << " dispatch queue(s) in queue family "
           << queue_family_info.dispatch_index;
  absl::InlinedVector<VkDeviceQueueCreateInfo, 2> queue_create_info;
  absl::InlinedVector<float, 4> dispatch_queue_priorities;
  absl::InlinedVector<float, 4> transfer_queue_priorities;
  queue_create_info.push_back({});
  auto& dispatch_queue_info = queue_create_info.back();
  dispatch_queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  dispatch_queue_info.pNext = nullptr;
  dispatch_queue_info.flags = 0;
  dispatch_queue_info.queueFamilyIndex = queue_family_info.dispatch_index;
  dispatch_queue_info.queueCount = queue_family_info.dispatch_queue_count;
  if (has_dedicated_transfer_queues) {
    if (queue_family_info.dispatch_index == queue_family_info.transfer_index) {
      DVLOG(1) << "Creating " << queue_family_info.transfer_queue_count
               << " dedicated transfer queue(s) in shared queue family "
               << queue_family_info.transfer_index;
      dispatch_queue_info.queueCount += queue_family_info.transfer_queue_count;
    } else {
      DVLOG(1) << "Creating " << queue_family_info.transfer_queue_count
               << " dedicated transfer queue(s) in independent queue family "
               << queue_family_info.transfer_index;
      queue_create_info.push_back({});
      auto& transfer_queue_info = queue_create_info.back();
      transfer_queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
      transfer_queue_info.pNext = nullptr;
      transfer_queue_info.queueFamilyIndex = queue_family_info.transfer_index;
      transfer_queue_info.queueCount = queue_family_info.transfer_queue_count;
      transfer_queue_info.flags = 0;
      transfer_queue_priorities.resize(transfer_queue_info.queueCount);
      transfer_queue_info.pQueuePriorities = transfer_queue_priorities.data();
    }
  }
  dispatch_queue_priorities.resize(dispatch_queue_info.queueCount);
  dispatch_queue_info.pQueuePriorities = dispatch_queue_priorities.data();

  // Create device and its queues.
  VkDeviceCreateInfo device_create_info = {};
  device_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  device_create_info.enabledLayerCount = 0;
  device_create_info.ppEnabledLayerNames = nullptr;
  device_create_info.enabledExtensionCount = enabled_extension_names.size();
  device_create_info.ppEnabledExtensionNames = enabled_extension_names.data();
  device_create_info.queueCreateInfoCount = queue_create_info.size();
  device_create_info.pQueueCreateInfos = queue_create_info.data();

  VkPhysicalDeviceTimelineSemaphoreFeatures semaphore_features;
  std::memset(&semaphore_features, 0, sizeof(semaphore_features));
  semaphore_features.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES;
  semaphore_features.timelineSemaphore = VK_TRUE;
  VkPhysicalDeviceFeatures2 features2;
  std::memset(&features2, 0, sizeof(features2));
  features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
  features2.pNext = &semaphore_features;
  VkPhysicalDeviceFeatures features;
  std::memset(&features, 0, sizeof(features));

  device_create_info.pNext = &features2;
  device_create_info.pEnabledFeatures = nullptr;

  auto logical_device =
      make_ref<VkDeviceHandle>(syms, enabled_device_extensions,
                               /*owns_device=*/true, /*allocator=*/nullptr);
  // The Vulkan loader can leak here, depending on which features are enabled.
  // This is out of our control, so disable leak checks.
  IREE_DISABLE_LEAK_CHECKS();
  VK_RETURN_IF_ERROR(syms->vkCreateDevice(physical_device, &device_create_info,
                                          logical_device->allocator(),
                                          logical_device->mutable_value()));
  RETURN_IF_ERROR(logical_device->syms()->LoadFromDevice(
      instance, logical_device->value()));
  IREE_ENABLE_LEAK_CHECKS();

  // Create the device memory allocator.
  // TODO(benvanik): allow other types to be plugged in.
  ASSIGN_OR_RETURN(auto allocator,
                   VmaAllocator::Create(physical_device, logical_device));

  // Create command pools for each queue family. If we don't have a transfer
  // queue then we'll ignore that one and just use the dispatch pool.
  // If we wanted to expose the pools through the HAL to allow the VM to more
  // effectively manage them (pool per fiber, etc) we could, however I doubt
  // the overhead of locking the pool will be even a blip.
  ASSIGN_OR_RETURN(auto dispatch_command_pool,
                   CreateTransientCommandPool(
                       logical_device, queue_family_info.dispatch_index));
  ref_ptr<VkCommandPoolHandle> transfer_command_pool;
  if (has_dedicated_transfer_queues) {
    ASSIGN_OR_RETURN(transfer_command_pool,
                     CreateTransientCommandPool(
                         logical_device, queue_family_info.transfer_index));
  }

  // Select queue indices and create command queues with them.
  QueueSet compute_queue_set = {};
  compute_queue_set.queue_family_index = queue_family_info.dispatch_index;
  for (uint32_t i = 0; i < queue_family_info.dispatch_queue_count; ++i) {
    compute_queue_set.queue_indices |= 1 << i;
  }
  QueueSet transfer_queue_set = {};
  transfer_queue_set.queue_family_index = queue_family_info.transfer_index;
  uint32_t base_queue_index = 0;
  if (queue_family_info.dispatch_index == queue_family_info.transfer_index) {
    // Sharing a family, so transfer queues follow compute queues.
    base_queue_index = queue_family_info.dispatch_index;
  }
  for (uint32_t i = 0; i < queue_family_info.transfer_queue_count; ++i) {
    transfer_queue_set.queue_indices |= 1 << (i + base_queue_index);
  }
  auto command_queues = CreateCommandQueues(
      device_info, logical_device, compute_queue_set, transfer_queue_set, syms);

  return assign_ref(new VulkanDevice(
      std::move(driver), device_info, physical_device,
      std::move(logical_device), std::move(allocator),
      std::move(command_queues), std::move(dispatch_command_pool),
      std::move(transfer_command_pool), debug_capture_manager));
}

// static
StatusOr<ref_ptr<VulkanDevice>> VulkanDevice::Wrap(
    ref_ptr<Driver> driver, const DeviceInfo& device_info,
    VkPhysicalDevice physical_device, VkDevice logical_device,
    const ExtensibilitySpec& extensibility_spec,
    const QueueSet& compute_queue_set, const QueueSet& transfer_queue_set,
    const ref_ptr<DynamicSymbols>& syms) {
  IREE_TRACE_SCOPE0("VulkanDevice::Wrap");

  uint64_t compute_queue_count = CountOnes64(compute_queue_set.queue_indices);
  uint64_t transfer_queue_count = CountOnes64(transfer_queue_set.queue_indices);

  if (compute_queue_count == 0) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "At least one compute queue is required";
  }

  // Find the extensions we need (or want) that are also available on the
  // device. This will fail when required ones are not present.
  //
  // Since the device is already created, we can't actually enable any
  // extensions or query if they are really enabled - we just have to trust
  // that the caller already enabled them for us (or we may fail later).
  ASSIGN_OR_RETURN(auto enabled_extension_names,
                   MatchAvailableDeviceExtensions(physical_device,
                                                  extensibility_spec, *syms));
  auto enabled_device_extensions =
      PopulateEnabledDeviceExtensions(enabled_extension_names);

  // Wrap the provided VkDevice with a VkDeviceHandle for use within the HAL.
  auto device_handle =
      make_ref<VkDeviceHandle>(syms, enabled_device_extensions,
                               /*owns_device=*/false, /*allocator=*/nullptr);
  *device_handle->mutable_value() = logical_device;

  // Create the device memory allocator.
  // TODO(benvanik): allow other types to be plugged in.
  ASSIGN_OR_RETURN(auto allocator,
                   VmaAllocator::Create(physical_device, device_handle));

  bool has_dedicated_transfer_queues = transfer_queue_count > 0;

  // Create command pools for each queue family. If we don't have a transfer
  // queue then we'll ignore that one and just use the dispatch pool.
  // If we wanted to expose the pools through the HAL to allow the VM to more
  // effectively manage them (pool per fiber, etc) we could, however I doubt
  // the overhead of locking the pool will be even a blip.
  ASSIGN_OR_RETURN(auto dispatch_command_pool,
                   CreateTransientCommandPool(
                       device_handle, compute_queue_set.queue_family_index));
  ref_ptr<VkCommandPoolHandle> transfer_command_pool;
  if (has_dedicated_transfer_queues) {
    ASSIGN_OR_RETURN(transfer_command_pool,
                     CreateTransientCommandPool(
                         device_handle, transfer_queue_set.queue_family_index));
  }

  auto command_queues = CreateCommandQueues(
      device_info, device_handle, compute_queue_set, transfer_queue_set, syms);

  return assign_ref(new VulkanDevice(
      std::move(driver), device_info, physical_device, std::move(device_handle),
      std::move(allocator), std::move(command_queues),
      std::move(dispatch_command_pool), std::move(transfer_command_pool),
      /*debug_capture_manager=*/nullptr));
}

VulkanDevice::VulkanDevice(
    ref_ptr<Driver> driver, const DeviceInfo& device_info,
    VkPhysicalDevice physical_device, ref_ptr<VkDeviceHandle> logical_device,
    std::unique_ptr<Allocator> allocator,
    absl::InlinedVector<std::unique_ptr<CommandQueue>, 4> command_queues,
    ref_ptr<VkCommandPoolHandle> dispatch_command_pool,
    ref_ptr<VkCommandPoolHandle> transfer_command_pool,
    DebugCaptureManager* debug_capture_manager)
    : Device(device_info),
      driver_(std::move(driver)),
      physical_device_(physical_device),
      logical_device_(std::move(logical_device)),
      allocator_(std::move(allocator)),
      command_queues_(std::move(command_queues)),
      descriptor_pool_cache_(
          make_ref<DescriptorPoolCache>(add_ref(logical_device_))),
      dispatch_command_pool_(std::move(dispatch_command_pool)),
      transfer_command_pool_(std::move(transfer_command_pool)),
      debug_capture_manager_(debug_capture_manager) {
  // Populate the queue lists based on queue capabilities.
  for (auto& command_queue : command_queues_) {
    if (command_queue->can_dispatch()) {
      dispatch_queues_.push_back(command_queue.get());
      if (transfer_command_pool_ == VK_NULL_HANDLE) {
        transfer_queues_.push_back(command_queue.get());
      }
    } else {
      transfer_queues_.push_back(command_queue.get());
    }
  }

  if (debug_capture_manager_ && debug_capture_manager_->is_connected()) {
    // Record a capture covering the duration of this VkDevice's lifetime.
    debug_capture_manager_->StartCapture();
  }
}

VulkanDevice::~VulkanDevice() {
  IREE_TRACE_SCOPE0("VulkanDevice::dtor");
  if (debug_capture_manager_ && debug_capture_manager_->is_capturing()) {
    debug_capture_manager_->StopCapture();
  }

  // Drop all command queues. These may wait until idle.
  command_queues_.clear();
  dispatch_queues_.clear();
  transfer_queues_.clear();

  // Drop command pools now that we know there are no more outstanding command
  // buffers.
  dispatch_command_pool_.reset();
  transfer_command_pool_.reset();

  // Now that no commands are outstanding we can release all descriptor sets.
  descriptor_pool_cache_.reset();

  // Finally, destroy the device.
  logical_device_.reset();
}

std::string VulkanDevice::DebugString() const {
  return absl::StrCat(Device::DebugString(),                                 //
                      "\n[VulkanDevice]",                                    //
                      "\n  Command Queues: ", command_queues_.size(),        //
                      "\n    - Dispatch Queues: ", dispatch_queues_.size(),  //
                      "\n    - Transfer Queues: ", transfer_queues_.size());
}

ref_ptr<ExecutableCache> VulkanDevice::CreateExecutableCache() {
  IREE_TRACE_SCOPE0("VulkanDevice::CreateExecutableCache");
  return make_ref<PipelineCache>(add_ref(logical_device_));
}

StatusOr<ref_ptr<DescriptorSetLayout>> VulkanDevice::CreateDescriptorSetLayout(
    DescriptorSetLayout::UsageType usage_type,
    absl::Span<const DescriptorSetLayout::Binding> bindings) {
  IREE_TRACE_SCOPE0("VulkanDevice::CreateDescriptorSetLayout");

  absl::InlinedVector<VkDescriptorSetLayoutBinding, 4> native_bindings(
      bindings.size());
  for (int i = 0; i < bindings.size(); ++i) {
    auto& native_binding = native_bindings[i];
    native_binding.binding = bindings[i].binding;
    native_binding.descriptorType =
        static_cast<VkDescriptorType>(bindings[i].type);
    native_binding.descriptorCount = 1;
    native_binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    native_binding.pImmutableSamplers = nullptr;
  }

  VkDescriptorSetLayoutCreateInfo create_info;
  create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  create_info.pNext = nullptr;
  create_info.flags = 0;
  if (usage_type == DescriptorSetLayout::UsageType::kPushOnly &&
      logical_device_->enabled_extensions().push_descriptors) {
    // Note that we can *only* use push descriptor sets if we set this create
    // flag. If push descriptors aren't supported we emulate them with normal
    // descriptors so it's fine to have kPushOnly without support.
    create_info.flags |=
        VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
  }
  create_info.bindingCount = native_bindings.size();
  create_info.pBindings = native_bindings.data();

  // Create and insert into the cache.
  VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
  VK_RETURN_IF_ERROR(syms()->vkCreateDescriptorSetLayout(
      *logical_device_, &create_info, logical_device_->allocator(),
      &descriptor_set_layout));

  return make_ref<NativeDescriptorSetLayout>(add_ref(logical_device_),
                                             descriptor_set_layout);
}

StatusOr<ref_ptr<ExecutableLayout>> VulkanDevice::CreateExecutableLayout(
    absl::Span<DescriptorSetLayout* const> set_layouts, size_t push_constants) {
  IREE_TRACE_SCOPE0("VulkanDevice::CreateExecutableLayout");

  absl::InlinedVector<ref_ptr<NativeDescriptorSetLayout>, 2> typed_set_layouts(
      set_layouts.size());
  absl::InlinedVector<VkDescriptorSetLayout, 2> set_layout_handles(
      set_layouts.size());
  for (int i = 0; i < set_layouts.size(); ++i) {
    typed_set_layouts[i] =
        add_ref(static_cast<NativeDescriptorSetLayout*>(set_layouts[i]));
    set_layout_handles[i] = typed_set_layouts[i]->handle();
  }

  absl::InlinedVector<VkPushConstantRange, 1> push_constant_ranges;
  if (push_constants > 0) {
    push_constant_ranges.push_back(VkPushConstantRange{
        VK_SHADER_STAGE_COMPUTE_BIT, 0,
        static_cast<uint32_t>(sizeof(uint32_t) * push_constants)});
  }

  VkPipelineLayoutCreateInfo create_info;
  create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  create_info.pNext = nullptr;
  create_info.flags = 0;
  create_info.setLayoutCount = set_layout_handles.size();
  create_info.pSetLayouts = set_layout_handles.data();
  create_info.pushConstantRangeCount = push_constant_ranges.size();
  create_info.pPushConstantRanges = push_constant_ranges.data();

  // Create and insert into the cache.
  VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
  VK_RETURN_IF_ERROR(syms()->vkCreatePipelineLayout(
      *logical_device_, &create_info, logical_device_->allocator(),
      &pipeline_layout));

  return make_ref<PipelineExecutableLayout>(
      add_ref(logical_device_), pipeline_layout, std::move(typed_set_layouts));
}

StatusOr<ref_ptr<DescriptorSet>> VulkanDevice::CreateDescriptorSet(
    DescriptorSetLayout* set_layout,
    absl::Span<const DescriptorSet::Binding> bindings) {
  IREE_TRACE_SCOPE0("VulkanDevice::CreateDescriptorSet");
  return UnimplementedErrorBuilder(IREE_LOC)
         << "CreateDescriptorSet not yet implemented (needs timeline)";
}

StatusOr<ref_ptr<CommandBuffer>> VulkanDevice::CreateCommandBuffer(
    CommandBufferModeBitfield mode,
    CommandCategoryBitfield command_categories) {
  IREE_TRACE_SCOPE0("VulkanDevice::CreateCommandBuffer");

  // Select the command pool to used based on the types of commands used.
  // Note that we may not have a dedicated transfer command pool if there are
  // no dedicated transfer queues.
  ref_ptr<VkCommandPoolHandle> command_pool;
  if (transfer_command_pool_ &&
      !AllBitsSet(command_categories, CommandCategory::kDispatch)) {
    command_pool = add_ref(transfer_command_pool_);
  } else {
    command_pool = add_ref(dispatch_command_pool_);
  }

  VkCommandBufferAllocateInfo allocate_info;
  allocate_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocate_info.pNext = nullptr;
  allocate_info.commandPool = *command_pool;
  allocate_info.commandBufferCount = 1;
  allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;

  VkCommandBuffer command_buffer = VK_NULL_HANDLE;
  {
    absl::MutexLock lock(command_pool->mutex());
    VK_RETURN_IF_ERROR(syms()->vkAllocateCommandBuffers(
        *logical_device_, &allocate_info, &command_buffer));
  }

  // TODO(b/140026716): conditionally enable validation.
  auto impl = make_ref<DirectCommandBuffer>(
      mode, command_categories, add_ref(descriptor_pool_cache_),
      add_ref(command_pool), command_buffer);
  return WrapCommandBufferWithValidation(allocator(), std::move(impl));
}

StatusOr<ref_ptr<Event>> VulkanDevice::CreateEvent() {
  IREE_TRACE_SCOPE0("VulkanDevice::CreateEvent");

  // TODO(b/138729892): pool events.
  VkEventCreateInfo create_info;
  create_info.sType = VK_STRUCTURE_TYPE_EVENT_CREATE_INFO;
  create_info.pNext = nullptr;
  create_info.flags = 0;
  VkEvent event_handle = VK_NULL_HANDLE;
  VK_RETURN_IF_ERROR(syms()->vkCreateEvent(*logical_device_, &create_info,
                                           logical_device_->allocator(),
                                           &event_handle));

  return make_ref<NativeEvent>(add_ref(logical_device_), event_handle);
}

StatusOr<ref_ptr<Semaphore>> VulkanDevice::CreateSemaphore(
    uint64_t initial_value) {
  IREE_TRACE_SCOPE0("VulkanDevice::CreateSemaphore");
  return NativeTimelineSemaphore::Create(add_ref(logical_device_),
                                         initial_value);
}

Status VulkanDevice::WaitAllSemaphores(
    absl::Span<const SemaphoreValue> semaphores, absl::Time deadline) {
  IREE_TRACE_SCOPE0("VulkanDevice::WaitAllSemaphores");
  return WaitSemaphores(semaphores, deadline, /*wait_flags=*/0);
}

StatusOr<int> VulkanDevice::WaitAnySemaphore(
    absl::Span<const SemaphoreValue> semaphores, absl::Time deadline) {
  IREE_TRACE_SCOPE0("VulkanDevice::WaitAnySemaphore");
  return WaitSemaphores(semaphores, deadline,
                        /*wait_flags=*/VK_SEMAPHORE_WAIT_ANY_BIT);
}

Status VulkanDevice::WaitSemaphores(absl::Span<const SemaphoreValue> semaphores,
                                    absl::Time deadline,
                                    VkSemaphoreWaitFlags wait_flags) {
  IREE_TRACE_SCOPE0("VulkanDevice::WaitSemaphores");

  absl::InlinedVector<VkSemaphore, 4> semaphore_handles(semaphores.size());
  absl::InlinedVector<uint64_t, 4> semaphore_values(semaphores.size());
  for (int i = 0; i < semaphores.size(); ++i) {
    semaphore_handles[i] =
        static_cast<NativeTimelineSemaphore*>(semaphores[i].semaphore)
            ->handle();
    semaphore_values[i] = semaphores[i].value;
  }

  VkSemaphoreWaitInfo wait_info;
  wait_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
  wait_info.pNext = nullptr;
  wait_info.flags = wait_flags;
  wait_info.semaphoreCount = semaphore_handles.size();
  wait_info.pSemaphores = semaphore_handles.data();
  wait_info.pValues = semaphore_values.data();

  uint64_t timeout_nanos;
  if (deadline == absl::InfiniteFuture()) {
    timeout_nanos = UINT64_MAX;
  } else if (deadline == absl::InfinitePast()) {
    timeout_nanos = 0;
  } else {
    auto relative_nanos = absl::ToInt64Nanoseconds(deadline - absl::Now());
    timeout_nanos = relative_nanos < 0 ? 0 : relative_nanos;
  }

  // NOTE: this may fail with a timeout (VK_TIMEOUT) or in the case of a
  // device loss event may return either VK_SUCCESS *or* VK_ERROR_DEVICE_LOST.
  // We may want to explicitly query for device loss after a successful wait
  // to ensure we consistently return errors.
  VkResult result =
      syms()->vkWaitSemaphores(*logical_device_, &wait_info, timeout_nanos);
  if (result == VK_ERROR_DEVICE_LOST) {
    // Nothing we do now matters.
    return VkResultToStatus(result);
  }

  // TODO(benvanik): notify the resource timeline that it should check for the
  // semaphores we waited on (including those already expired above).

  return OkStatus();
}

Status VulkanDevice::WaitIdle(absl::Time deadline) {
  if (deadline == absl::InfiniteFuture()) {
    // Fast path for using vkDeviceWaitIdle, which is usually cheaper (as it
    // requires fewer calls into the driver).
    IREE_TRACE_SCOPE0("VulkanDevice::WaitIdle#vkDeviceWaitIdle");
    VK_RETURN_IF_ERROR(syms()->vkDeviceWaitIdle(*logical_device_));
    return OkStatus();
  }

  IREE_TRACE_SCOPE0("VulkanDevice::WaitIdle#Semaphores");
  for (auto& command_queue : command_queues_) {
    RETURN_IF_ERROR(command_queue->WaitIdle(deadline));
  }
  return OkStatus();
}

}  // namespace vulkan
}  // namespace hal
}  // namespace iree
