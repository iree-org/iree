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

#ifndef IREE_HAL_HOST_HOST_LOCAL_DEVICE_H_
#define IREE_HAL_HOST_HOST_LOCAL_DEVICE_H_

#include "absl/types/span.h"
#include "iree/base/memory.h"
#include "iree/hal/device.h"
#include "iree/hal/host/host_local_allocator.h"
#include "iree/hal/host/scheduling_model.h"

namespace iree {
namespace hal {

// A host-local device that uses host-local memory and in-process execution.
// This implements the boilerplate needed for any device that runs on the CPU
// using the other Host* types. The scheduling model used to distribute work
// across local CPU resources is provided by the SchedulingModel interface.
class HostLocalDevice : public Device {
 public:
  ~HostLocalDevice() override;

  Allocator* allocator() const override { return &allocator_; }

  absl::Span<CommandQueue*> dispatch_queues() const override {
    return scheduling_model_->dispatch_queues();
  }

  absl::Span<CommandQueue*> transfer_queues() const override {
    return scheduling_model_->transfer_queues();
  }

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

 protected:
  explicit HostLocalDevice(DeviceInfo device_info,
                           std::unique_ptr<SchedulingModel> scheduling_model);

 private:
  std::unique_ptr<SchedulingModel> scheduling_model_;
  mutable HostLocalAllocator allocator_;
};

}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_HOST_HOST_LOCAL_DEVICE_H_
