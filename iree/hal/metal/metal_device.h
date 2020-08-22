// Copyright 2020 Google LLC
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

#ifndef IREE_HAL_METAL_METAL_DEVICE_H_
#define IREE_HAL_METAL_METAL_DEVICE_H_

#import <Metal/Metal.h>

#include "absl/types/span.h"
#include "iree/base/memory.h"
#include "iree/hal/allocator.h"
#include "iree/hal/device.h"
#include "iree/hal/driver.h"
#include "iree/hal/semaphore.h"

namespace iree {
namespace hal {
namespace metal {

// A device implementation for Metal that directly wraps a MTLDevice.
class MetalDevice final : public Device {
 public:
  // Creates a device that retains the underlying Metal GPU device.
  // The DriverDeviceID in |device_info| is expected to be an id<MTLDevice>.
  static StatusOr<ref_ptr<MetalDevice>> Create(ref_ptr<Driver> driver,
                                               const DeviceInfo& device_info);

  ~MetalDevice() override;

  std::string DebugString() const override;

  Allocator* allocator() const override { return nullptr; }

  absl::Span<CommandQueue*> dispatch_queues() const override {
    return absl::MakeSpan(&common_queue_, 1);
  }

  absl::Span<CommandQueue*> transfer_queues() const override {
    return absl::MakeSpan(&common_queue_, 1);
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
                           Time deadline_ns) override;
  StatusOr<int> WaitAnySemaphore(absl::Span<const SemaphoreValue> semaphores,
                                 Time deadline_ns) override;

  Status WaitIdle(Time deadline_ns) override;

 private:
  MetalDevice(ref_ptr<Driver> driver, const DeviceInfo& device_info);

  ref_ptr<Driver> driver_;
  id<MTLDevice> metal_handle_;

  // Metal does not have clear graphics/dispatch/transfer queue distinction like
  // Vulkan; one just use the same newCommandQueue() API call on MTLDevice to
  // get command queues. Command encoders differ for different categories of
  // commands though. We expose one queue here for everything. This can be
  // changed later if more queues prove to be useful.

  std::unique_ptr<CommandQueue> command_queue_;
  mutable CommandQueue* common_queue_ = nullptr;
};

}  // namespace metal
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_METAL_METAL_DEVICE_H_
