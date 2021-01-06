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

#include <memory>

#include "absl/types/span.h"
#include "iree/base/memory.h"
#include "iree/hal/allocator.h"
#include "iree/hal/debug_capture_manager.h"
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
  // The iree_hal_device_id_t in |device_info| is expected to be an
  // id<MTLDevice>.
  static StatusOr<ref_ptr<MetalDevice>> Create(
      ref_ptr<Driver> driver, const DeviceInfo& device_info,
      DebugCaptureManager* debug_capture_manager);

  ~MetalDevice() override;

  std::string DebugString() const override;

  Allocator* allocator() const override { return allocator_.get(); }

  absl::Span<CommandQueue*> dispatch_queues() const override {
    return absl::MakeSpan(&common_queue_, 1);
  }

  absl::Span<CommandQueue*> transfer_queues() const override {
    return absl::MakeSpan(&common_queue_, 1);
  }

  ref_ptr<ExecutableCache> CreateExecutableCache() override;

  StatusOr<ref_ptr<DescriptorSetLayout>> CreateDescriptorSetLayout(
      iree_hal_descriptor_set_layout_usage_type_t usage_type,
      absl::Span<const iree_hal_descriptor_set_layout_binding_t> bindings)
      override;

  StatusOr<ref_ptr<ExecutableLayout>> CreateExecutableLayout(
      absl::Span<DescriptorSetLayout* const> set_layouts,
      size_t push_constants) override;

  StatusOr<ref_ptr<DescriptorSet>> CreateDescriptorSet(
      DescriptorSetLayout* set_layout,
      absl::Span<const iree_hal_descriptor_set_binding_t> bindings) override;

  StatusOr<ref_ptr<CommandBuffer>> CreateCommandBuffer(
      iree_hal_command_buffer_mode_t mode,
      iree_hal_command_category_t command_categories) override;

  StatusOr<ref_ptr<Event>> CreateEvent() override;

  StatusOr<ref_ptr<Semaphore>> CreateSemaphore(uint64_t initial_value) override;
  Status WaitAllSemaphores(absl::Span<const SemaphoreValue> semaphores,
                           Time deadline_ns) override;
  StatusOr<int> WaitAnySemaphore(absl::Span<const SemaphoreValue> semaphores,
                                 Time deadline_ns) override;

  Status WaitIdle(Time deadline_ns) override;

 private:
  MetalDevice(ref_ptr<Driver> driver, const DeviceInfo& device_info,
              DebugCaptureManager* debug_capture_manager);

  ref_ptr<Driver> driver_;
  id<MTLDevice> metal_handle_;

  std::unique_ptr<Allocator> allocator_;

  // Metal does not have clear graphics/dispatch/transfer queue distinction like
  // Vulkan; one just use the same newCommandQueue() API call on MTLDevice to
  // get command queues. Command encoders differ for different categories of
  // commands though. We expose one queue here for everything. This can be
  // changed later if more queues prove to be useful.

  std::unique_ptr<CommandQueue> command_queue_;
  mutable CommandQueue* common_queue_ = nullptr;

  // A dispatch queue and associated event listener for running Objective-C
  // blocks. This is typically used to wake up threads waiting on some HAL
  // semaphore.
  dispatch_queue_t wait_notifier_;
  MTLSharedEventListener* event_listener_;

  DebugCaptureManager* debug_capture_manager_ = nullptr;
};

}  // namespace metal
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_METAL_METAL_DEVICE_H_
