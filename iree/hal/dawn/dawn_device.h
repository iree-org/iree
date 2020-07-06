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

#ifndef IREE_HAL_DAWN_DAWN_DEVICE_H_
#define IREE_HAL_DAWN_DAWN_DEVICE_H_

#include "absl/container/inlined_vector.h"
#include "absl/types/span.h"
#include "iree/base/memory.h"
#include "iree/hal/device.h"
#include "iree/hal/host/host_local_allocator.h"
#include "third_party/dawn/src/include/dawn/webgpu_cpp.h"
#include "third_party/dawn/src/include/dawn_native/DawnNative.h"

namespace iree {
namespace hal {
namespace dawn {

class DawnDevice final : public Device {
 public:
  explicit DawnDevice(const DeviceInfo& device_info,
                      ::wgpu::Device backend_device);
  ~DawnDevice() override;

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
  mutable host::HostLocalAllocator allocator_;
  mutable absl::InlinedVector<std::unique_ptr<CommandQueue>, 1> command_queues_;

  ::wgpu::Device backend_device_;
};

}  // namespace dawn
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_DAWN_DAWN_DEVICE_H_
