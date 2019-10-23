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
#include "base/memory.h"
#include "hal/allocator.h"
#include "hal/device.h"
#include "hal/vulkan/descriptor_pool_cache.h"
#include "hal/vulkan/dynamic_symbols.h"
#include "hal/vulkan/extensibility_util.h"
#include "hal/vulkan/handle_util.h"
#include "hal/vulkan/legacy_fence.h"

namespace iree {
namespace hal {
namespace vulkan {

class VulkanDevice final : public Device {
 public:
  static StatusOr<std::shared_ptr<VulkanDevice>> Create(
      const DeviceInfo& device_info, VkPhysicalDevice physical_device,
      const ExtensibilitySpec& extensibility_spec,
      const ref_ptr<DynamicSymbols>& syms);

  // Private constructor.
  struct CtorKey {
   private:
    friend class VulkanDevice;
    CtorKey() = default;
  };
  VulkanDevice(
      CtorKey ctor_key, const DeviceInfo& device_info,
      VkPhysicalDevice physical_device, ref_ptr<VkDeviceHandle> logical_device,
      std::unique_ptr<Allocator> allocator,
      absl::InlinedVector<std::unique_ptr<CommandQueue>, 4> command_queues,
      ref_ptr<VkCommandPoolHandle> dispatch_command_pool,
      ref_ptr<VkCommandPoolHandle> transfer_command_pool,
      ref_ptr<LegacyFencePool> legacy_fence_pool);
  ~VulkanDevice() override;

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

  std::shared_ptr<ExecutableCache> CreateExecutableCache() override;

  StatusOr<ref_ptr<CommandBuffer>> CreateCommandBuffer(
      CommandBufferModeBitfield mode,
      CommandCategoryBitfield command_categories) override;

  StatusOr<ref_ptr<Event>> CreateEvent() override;

  StatusOr<ref_ptr<BinarySemaphore>> CreateBinarySemaphore(
      bool initial_value) override;
  StatusOr<ref_ptr<TimelineSemaphore>> CreateTimelineSemaphore(
      uint64_t initial_value) override;

  StatusOr<ref_ptr<Fence>> CreateFence(uint64_t initial_value) override;
  Status WaitAllFences(absl::Span<const FenceValue> fences,
                       absl::Time deadline) override;
  StatusOr<int> WaitAnyFence(absl::Span<const FenceValue> fences,
                             absl::Time deadline) override;

  Status WaitIdle(absl::Time deadline) override;

 private:
  VkPhysicalDevice physical_device_;
  ref_ptr<VkDeviceHandle> logical_device_;

  std::unique_ptr<Allocator> allocator_;

  mutable absl::InlinedVector<std::unique_ptr<CommandQueue>, 4> command_queues_;
  mutable absl::InlinedVector<CommandQueue*, 4> dispatch_queues_;
  mutable absl::InlinedVector<CommandQueue*, 4> transfer_queues_;

  ref_ptr<DescriptorPoolCache> descriptor_pool_cache_;

  ref_ptr<VkCommandPoolHandle> dispatch_command_pool_;
  ref_ptr<VkCommandPoolHandle> transfer_command_pool_;

  // TODO(b/140141417): implement timeline semaphore fences and conditionally
  // compile the legacy fence pool out.
  ref_ptr<LegacyFencePool> legacy_fence_pool_;
};

}  // namespace vulkan
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_VULKAN_VULKAN_DEVICE_H_
