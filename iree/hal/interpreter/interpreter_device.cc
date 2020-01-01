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

#include "iree/hal/interpreter/interpreter_device.h"

#include <utility>

#include "absl/memory/memory.h"
#include "iree/base/status.h"
#include "iree/base/tracing.h"
#include "iree/hal/command_buffer_validation.h"
#include "iree/hal/command_queue.h"
#include "iree/hal/fence.h"
#include "iree/hal/host/async_command_queue.h"
#include "iree/hal/host/host_event.h"
#include "iree/hal/host/host_submission_queue.h"
#include "iree/hal/host/inproc_command_buffer.h"
#include "iree/hal/interpreter/bytecode_cache.h"
#include "iree/hal/interpreter/interpreter_command_processor.h"

namespace iree {
namespace hal {

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
  UnsynchronizedCommandQueue(Allocator* allocator, std::string name,
                             CommandCategoryBitfield supported_categories)
      : CommandQueue(std::move(name), supported_categories),
        allocator_(allocator) {}
  ~UnsynchronizedCommandQueue() override = default;

  Status Submit(absl::Span<const SubmissionBatch> batches,
                FenceValue fence) override {
    IREE_TRACE_SCOPE0("UnsynchronizedCommandQueue::Submit");
    DCHECK_EQ(nullptr, fence.first)
        << "Fences must be handled by the wrapping queue";

    // Process command buffers and propagate errors asynchronously through the
    // fence. This ensures that even if we are running synchronously we still
    // get consistent failure behavior with drivers that are purely async.
    for (auto& batch : batches) {
      DCHECK(batch.wait_semaphores.empty() && batch.signal_semaphores.empty())
          << "Semaphores must be handled by the wrapping queue";
      RETURN_IF_ERROR(ProcessCommandBuffers(batch.command_buffers));
    }

    // NOTE: fence is ignored here.
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
      InterpreterCommandProcessor command_processor(
          allocator_, command_buffer->mode(), supported_categories());
      RETURN_IF_ERROR(inproc_command_buffer->Process(&command_processor));
    }
    return OkStatus();
  }

  Allocator* const allocator_;
};

}  // namespace

InterpreterDevice::InterpreterDevice(DeviceInfo device_info)
    : Device(std::move(device_info)) {
  // We currently only expose a single command queue.
  auto command_queue = absl::make_unique<UnsynchronizedCommandQueue>(
      &allocator_, "cpu0",
      CommandCategory::kTransfer | CommandCategory::kDispatch);

  // TODO(benvanik): allow injection of the wrapper type to support
  // SyncCommandQueue without always linking in both.
  auto async_command_queue =
      absl::make_unique<AsyncCommandQueue>(std::move(command_queue));
  command_queues_.push_back(std::move(async_command_queue));
}

InterpreterDevice::~InterpreterDevice() = default;

ref_ptr<ExecutableCache> InterpreterDevice::CreateExecutableCache() {
  return make_ref<BytecodeCache>(&allocator_);
}

StatusOr<ref_ptr<CommandBuffer>> InterpreterDevice::CreateCommandBuffer(
    CommandBufferModeBitfield mode,
    CommandCategoryBitfield command_categories) {
  // TODO(b/140026716): conditionally enable validation.
  auto impl =
      make_ref<InProcCommandBuffer>(&allocator_, mode, command_categories);
  return WrapCommandBufferWithValidation(std::move(impl));
}

StatusOr<ref_ptr<Event>> InterpreterDevice::CreateEvent() {
  return make_ref<HostEvent>();
}

StatusOr<ref_ptr<BinarySemaphore>> InterpreterDevice::CreateBinarySemaphore(
    bool initial_value) {
  IREE_TRACE_SCOPE0("InterpreterDevice::CreateBinarySemaphore");
  return make_ref<HostBinarySemaphore>(initial_value);
}

StatusOr<ref_ptr<TimelineSemaphore>> InterpreterDevice::CreateTimelineSemaphore(
    uint64_t initial_value) {
  IREE_TRACE_SCOPE0("InterpreterDevice::CreateTimelineSemaphore");

  // TODO(b/140141417): implement timeline semaphores.
  return UnimplementedErrorBuilder(IREE_LOC)
         << "Timeline semaphores not yet implemented";
}

StatusOr<ref_ptr<Fence>> InterpreterDevice::CreateFence(
    uint64_t initial_value) {
  IREE_TRACE_SCOPE0("InterpreterDevice::CreateFence");
  return make_ref<HostFence>(initial_value);
}

Status InterpreterDevice::WaitAllFences(absl::Span<const FenceValue> fences,
                                        absl::Time deadline) {
  IREE_TRACE_SCOPE0("InterpreterDevice::WaitAllFences");
  return HostFence::WaitForFences(fences, /*wait_all=*/true, deadline);
}

StatusOr<int> InterpreterDevice::WaitAnyFence(
    absl::Span<const FenceValue> fences, absl::Time deadline) {
  IREE_TRACE_SCOPE0("InterpreterDevice::WaitAnyFence");
  return HostFence::WaitForFences(fences, /*wait_all=*/false, deadline);
}

Status InterpreterDevice::WaitIdle(absl::Time deadline) {
  for (auto& command_queue : command_queues_) {
    RETURN_IF_ERROR(command_queue->WaitIdle(deadline));
  }
  return OkStatus();
}

}  // namespace hal
}  // namespace iree
