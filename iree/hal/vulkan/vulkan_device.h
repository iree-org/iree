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

#ifndef IREE_HAL_VULKAN_VULKAN_DEVICE_H_
#define IREE_HAL_VULKAN_VULKAN_DEVICE_H_

#include <vulkan/vulkan.h>

#include <functional>
#include <memory>

#include "absl/container/inlined_vector.h"
#include "absl/types/span.h"
#include "iree/base/memory.h"
#include "iree/hal/allocator.h"
#include "iree/hal/debug_capture_manager.h"
#include "iree/hal/device.h"
#include "iree/hal/driver.h"
#include "iree/hal/semaphore.h"
#include "iree/hal/vulkan/descriptor_pool_cache.h"
#include "iree/hal/vulkan/dynamic_symbols.h"
#include "iree/hal/vulkan/emulated_timeline_semaphore.h"
#include "iree/hal/vulkan/extensibility_util.h"
#include "iree/hal/vulkan/handle_util.h"

namespace iree {
namespace hal {
namespace vulkan {

// A set of queues within a specific queue family on a VkDevice.
struct QueueSet {
  // The index of a particular queue family on a VkPhysicalDevice, as described
  // by vkGetPhysicalDeviceQueueFamilyProperties.
  uint32_t queue_family_index;

  // Bitfield of queue indices within the queue family at |queue_family_index|.
  uint64_t queue_indices;
};

class VulkanDevice final : public Device {
 public:
  // Creates a device that manages its own VkDevice.
  static StatusOr<ref_ptr<VulkanDevice>> Create(
      ref_ptr<Driver> driver, VkInstance instance,
      const DeviceInfo& device_info, VkPhysicalDevice physical_device,
      const ExtensibilitySpec& extensibility_spec,
      const ref_ptr<DynamicSymbols>& syms,
      DebugCaptureManager* debug_capture_manager);

  // Creates a device that wraps an externally managed VkDevice.
  static StatusOr<ref_ptr<VulkanDevice>> Wrap(
      ref_ptr<Driver> driver, const DeviceInfo& device_info,
      VkPhysicalDevice physical_device, VkDevice logical_device,
      const ExtensibilitySpec& extensibility_spec,
      const QueueSet& compute_queue_set, const QueueSet& transfer_queue_set,
      const ref_ptr<DynamicSymbols>& syms);

  ~VulkanDevice() override;

  std::string DebugString() const override;

  const ref_ptr<DynamicSymbols>& syms() const {
    return logical_device_->syms();
  }

  Allocator* allocator() const override { return allocator_.get(); }

  absl::Span<CommandQueue*> dispatch_queues() const override {
    return absl::MakeSpan(dispatch_queues_);
  }

  absl::Span<CommandQueue*> transfer_queues() const override {
    return absl::MakeSpan(transfer_queues_);
  }

  ref_ptr<ExecutableCache> CreateExecutableCache() override;

  StatusOr<ref_ptr<DescriptorSetLayout>> CreateDescriptorSetLayout(
      DescriptorSetLayout::UsageType usage_type,
      absl::Span<const DescriptorSetLayout::Binding> bindings) override;

  StatusOr<ref_ptr<ExecutableLayout>> CreateExecutableLayout(
      absl::Span<DescriptorSetLayout* const> set_layouts,
      size_t push_constants) override;

  StatusOr<ref_ptr<DescriptorSet>> CreateDescriptorSet(
      DescriptorSetLayout* set_layout,
      absl::Span<const DescriptorSet::Binding> bindings) override;

  StatusOr<ref_ptr<CommandBuffer>> CreateCommandBuffer(
      CommandBufferModeBitfield mode,
      CommandCategoryBitfield command_categories) override;

  StatusOr<ref_ptr<Event>> CreateEvent() override;

  StatusOr<ref_ptr<Semaphore>> CreateSemaphore(uint64_t initial_value) override;
  Status WaitAllSemaphores(absl::Span<const SemaphoreValue> semaphores,
                           absl::Time deadline) override;
  StatusOr<int> WaitAnySemaphore(absl::Span<const SemaphoreValue> semaphores,
                                 absl::Time deadline) override;

  Status WaitIdle(absl::Time deadline) override;

 private:
  VulkanDevice(
      ref_ptr<Driver> driver, const DeviceInfo& device_info,
      VkPhysicalDevice physical_device, ref_ptr<VkDeviceHandle> logical_device,
      std::unique_ptr<Allocator> allocator,
      absl::InlinedVector<std::unique_ptr<CommandQueue>, 4> command_queues,
      ref_ptr<VkCommandPoolHandle> dispatch_command_pool,
      ref_ptr<VkCommandPoolHandle> transfer_command_pool,
      ref_ptr<TimePointSemaphorePool> semaphore_pool,
      ref_ptr<TimePointFencePool> fence_pool,
      DebugCaptureManager* debug_capture_manager);

  Status WaitSemaphores(absl::Span<const SemaphoreValue> semaphores,
                        absl::Time deadline, VkSemaphoreWaitFlags wait_flags);

  bool emulating_timeline_semaphores() const {
    return semaphore_pool_ != nullptr;
  }

  ref_ptr<Driver> driver_;
  VkPhysicalDevice physical_device_;
  ref_ptr<VkDeviceHandle> logical_device_;

  std::unique_ptr<Allocator> allocator_;

  mutable absl::InlinedVector<std::unique_ptr<CommandQueue>, 4> command_queues_;
  mutable absl::InlinedVector<CommandQueue*, 4> dispatch_queues_;
  mutable absl::InlinedVector<CommandQueue*, 4> transfer_queues_;

  ref_ptr<DescriptorPoolCache> descriptor_pool_cache_;

  ref_ptr<VkCommandPoolHandle> dispatch_command_pool_;
  ref_ptr<VkCommandPoolHandle> transfer_command_pool_;

  // Fields used for emulated timeline semaphores.
  ref_ptr<TimePointSemaphorePool> semaphore_pool_;
  ref_ptr<TimePointFencePool> fence_pool_;

  DebugCaptureManager* debug_capture_manager_ = nullptr;
};

}  // namespace vulkan
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_VULKAN_VULKAN_DEVICE_H_
