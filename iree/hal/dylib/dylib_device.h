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

#ifndef IREE_HAL_DYLIB_DYLIB_DEVICE_H_
#define IREE_HAL_DYLIB_DYLIB_DEVICE_H_

#include "absl/container/inlined_vector.h"
#include "absl/types/span.h"
#include "iree/base/memory.h"
#include "iree/hal/device.h"
#include "iree/hal/host/host_local_allocator.h"

namespace iree {
namespace hal {
namespace dylib {

class DyLibDevice final : public Device {
 public:
  static StatusOr<ref_ptr<DyLibDevice>> CreateDyLibDevice(
      DeviceInfo device_info);

  explicit DyLibDevice(DeviceInfo device_info);
  ~DyLibDevice() override;

  std::string DebugString() const override;

  Allocator* allocator() const override { return &allocator_; }

  absl::Span<CommandQueue*> dispatch_queues() const override {
    return RawPtrSpan(absl::MakeSpan(command_queues_));
  }

  absl::Span<CommandQueue*> transfer_queues() const override {
    return RawPtrSpan(absl::MakeSpan(command_queues_));
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
  mutable HostLocalAllocator allocator_;
  mutable absl::InlinedVector<std::unique_ptr<CommandQueue>, 1> command_queues_;
};

}  // namespace dylib
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_DYLIB_DYLIB_DEVICE_H_
