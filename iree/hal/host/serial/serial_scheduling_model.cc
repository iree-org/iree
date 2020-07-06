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

#include "iree/hal/host/serial/serial_scheduling_model.h"

#include "iree/base/tracing.h"
#include "iree/hal/host/condvar_semaphore.h"
#include "iree/hal/host/inproc_command_buffer.h"
#include "iree/hal/host/nop_event.h"
#include "iree/hal/host/serial/async_command_queue.h"
#include "iree/hal/host/serial/serial_command_processor.h"
#include "iree/hal/host/serial/serial_submission_queue.h"

namespace iree {
namespace hal {
namespace host {
namespace {

// A CommandQueue that performs no synchronization (semaphores/fences) and just
// directly executes command buffers inline.
//
// This is meant to be wrapped by SyncCommandQueue or AsyncCommandQueue that
// themselves perform the synchronization/threading/etc. As such we ignore
// all semaphores in the provided batches under the assumption that if Submit is
// being called then all dependencies are valid. The wrapping queue is also
// responsible for signaling the fence as well as propagating errors in a way
// that is dependent on how it is performing its synchronization.
class UnsynchronizedCommandQueue final : public CommandQueue {
 public:
  UnsynchronizedCommandQueue(std::string name,
                             CommandCategoryBitfield supported_categories)
      : CommandQueue(std::move(name), supported_categories) {}
  ~UnsynchronizedCommandQueue() override = default;

  Status Submit(absl::Span<const SubmissionBatch> batches) override {
    IREE_TRACE_SCOPE0("UnsynchronizedCommandQueue::Submit");

    // Process command buffers and propagate errors asynchronously through the
    // fence. This ensures that even if we are running synchronously we still
    // get consistent failure behavior with drivers that are purely async.
    for (auto& batch : batches) {
      DCHECK(batch.wait_semaphores.empty() && batch.signal_semaphores.empty())
          << "Semaphores must be handled by the wrapping queue";
      RETURN_IF_ERROR(ProcessCommandBuffers(batch.command_buffers));
    }

    return OkStatus();
  }

  Status WaitIdle(absl::Time deadline) override {
    // No-op.
    return OkStatus();
  }

 private:
  // Processes each command buffer in-turn with a fresh processor.
  // This ensures we don't have any state that can carry across buffers.
  Status ProcessCommandBuffers(
      absl::Span<CommandBuffer* const> command_buffers) {
    IREE_TRACE_SCOPE0("UnsynchronizedCommandQueue::ProcessCommandBuffers");
    for (auto* command_buffer : command_buffers) {
      auto* inproc_command_buffer =
          static_cast<InProcCommandBuffer*>(command_buffer->impl());
      SerialCommandProcessor command_processor(supported_categories());
      RETURN_IF_ERROR(inproc_command_buffer->Process(&command_processor));
    }
    return OkStatus();
  }
};

}  // namespace

SerialSchedulingModel::SerialSchedulingModel() {
  // We currently only expose a single command queue.
  auto command_queue = absl::make_unique<UnsynchronizedCommandQueue>(
      "cpu0", CommandCategory::kTransfer | CommandCategory::kDispatch);

  // Wrap in the simple async command queue.
  auto async_command_queue =
      absl::make_unique<AsyncCommandQueue>(std::move(command_queue));
  command_queues_.push_back(std::move(async_command_queue));
}

SerialSchedulingModel::~SerialSchedulingModel() = default;

StatusOr<ref_ptr<CommandBuffer>> SerialSchedulingModel::CreateCommandBuffer(
    CommandBufferModeBitfield mode,
    CommandCategoryBitfield command_categories) {
  return make_ref<InProcCommandBuffer>(mode, command_categories);
}

StatusOr<ref_ptr<Event>> SerialSchedulingModel::CreateEvent() {
  return make_ref<NopEvent>();
}

StatusOr<ref_ptr<Semaphore>> SerialSchedulingModel::CreateSemaphore(
    uint64_t initial_value) {
  return make_ref<CondVarSemaphore>(initial_value);
}

Status SerialSchedulingModel::WaitAllSemaphores(
    absl::Span<const SemaphoreValue> semaphores, absl::Time deadline) {
  return CondVarSemaphore::WaitForSemaphores(semaphores, /*wait_all=*/true,
                                             deadline);
}

StatusOr<int> SerialSchedulingModel::WaitAnySemaphore(
    absl::Span<const SemaphoreValue> semaphores, absl::Time deadline) {
  return CondVarSemaphore::WaitForSemaphores(semaphores, /*wait_all=*/false,
                                             deadline);
}

Status SerialSchedulingModel::WaitIdle(absl::Time deadline) {
  for (auto& command_queue : command_queues_) {
    RETURN_IF_ERROR(command_queue->WaitIdle(deadline));
  }
  return OkStatus();
}

}  // namespace host
}  // namespace hal
}  // namespace iree
