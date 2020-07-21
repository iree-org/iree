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

#ifndef IREE_HAL_HOST_SERIAL_SERIAL_SCHEDULING_MODEL_H_
#define IREE_HAL_HOST_SERIAL_SERIAL_SCHEDULING_MODEL_H_

#include "absl/container/inlined_vector.h"
#include "iree/base/memory.h"
#include "iree/hal/host/scheduling_model.h"

namespace iree {
namespace hal {
namespace host {

// Performs host-local scheduling by way of a simple serial queue.
// Submissions and commands are processed in-order one at a time on a single
// core. This is a reference implementation that has no dependencies beyond
// std::thread and allows us to quickly bring up new platforms and more easily
// debug/profile as we won't have OS fibers/other weird constructs involved.
class SerialSchedulingModel final : public SchedulingModel {
 public:
  SerialSchedulingModel();
  ~SerialSchedulingModel() override;

  absl::Span<CommandQueue*> dispatch_queues() const override {
    return RawPtrSpan(absl::MakeSpan(command_queues_));
  }

  absl::Span<CommandQueue*> transfer_queues() const override {
    return RawPtrSpan(absl::MakeSpan(command_queues_));
  }

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
  mutable absl::InlinedVector<std::unique_ptr<CommandQueue>, 4> command_queues_;
};

}  // namespace host
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_HOST_SERIAL_SERIAL_SCHEDULING_MODEL_H_
