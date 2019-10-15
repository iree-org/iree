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
#include "third_party/dawn/src/include/dawn/dawncpp.h"
#include "third_party/dawn/src/include/dawn_native/DawnNative.h"
#include "iree/base/memory.h"
#include "iree/hal/device.h"
#include "iree/hal/host/host_local_allocator.h"

namespace iree {
namespace hal {
namespace dawn {

class DawnDevice final : public Device {
 public:
  explicit DawnDevice(const DeviceInfo& device_info,
                      ::dawn::Device backend_device);
  ~DawnDevice() override;

  Allocator* allocator() const override { return &allocator_; }

  absl::Span<CommandQueue*> dispatch_queues() const override {
    return RawPtrSpan(absl::MakeSpan(command_queues_));
  }

  absl::Span<CommandQueue*> transfer_queues() const override {
    return RawPtrSpan(absl::MakeSpan(command_queues_));
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
  mutable HostLocalAllocator allocator_;
  mutable absl::InlinedVector<std::unique_ptr<CommandQueue>, 1> command_queues_;

  ::dawn::Device backend_device_;
};

}  // namespace dawn
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_DAWN_DAWN_DEVICE_H_
