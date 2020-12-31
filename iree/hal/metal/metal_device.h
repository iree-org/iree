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
#include "iree/hal/cc/device.h"
#include "iree/hal/cc/driver.h"

namespace iree {
namespace hal {
namespace metal {

// A device implementation for Metal that directly wraps a MTLDevice.
class MetalDevice final : public Device {
 public:
  // Creates a device that retains the underlying Metal GPU device.
  // The iree_hal_device_id_t in |device_info| is expected to be an
  // id<MTLDevice>.
  static StatusOr<ref_ptr<MetalDevice>> Create(ref_ptr<Driver> driver,
                                               const DeviceInfo& device_info);

  ~MetalDevice() override;

  Allocator* allocator() const override { return allocator_.get(); }

  Status CreateExecutableCache(
      iree_string_view_t identifier,
      iree_hal_executable_cache_t** out_executable_cache) override;

  Status CreateDescriptorSetLayout(
      iree_hal_descriptor_set_layout_usage_type_t usage_type,
      absl::Span<const iree_hal_descriptor_set_layout_binding_t> bindings,
      iree_hal_descriptor_set_layout_t** out_descriptor_set_layout) override;

  Status CreateExecutableLayout(
      absl::Span<iree_hal_descriptor_set_layout_t*> set_layouts,
      size_t push_constants,
      iree_hal_executable_layout_t** out_executable_layout) override;

  Status CreateDescriptorSet(
      iree_hal_descriptor_set_layout_t* set_layout,
      absl::Span<const iree_hal_descriptor_set_binding_t> bindings,
      iree_hal_descriptor_set_t** out_descriptor_set) override;

  Status CreateCommandBuffer(
      iree_hal_command_buffer_mode_t mode,
      iree_hal_command_category_t command_categories,
      iree_hal_command_buffer_t** out_command_buffer) override;

  Status CreateEvent(iree_hal_event_t** out_event) override;

  Status CreateSemaphore(uint64_t initial_value,
                         iree_hal_semaphore_t** out_semaphore) override;
  Status WaitAllSemaphores(const iree_hal_semaphore_list_t* semaphore_list,
                           iree_time_t deadline_ns) override;
  StatusOr<int> WaitAnySemaphore(
      const iree_hal_semaphore_list_t* semaphore_list,
      iree_time_t deadline_ns) override;

  Status WaitIdle(iree_time_t deadline_ns) override;

 private:
  MetalDevice(ref_ptr<Driver> driver, const DeviceInfo& device_info);

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
};

}  // namespace metal
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_METAL_METAL_DEVICE_H_
