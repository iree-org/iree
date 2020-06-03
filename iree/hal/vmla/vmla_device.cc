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

#include "iree/hal/vmla/vmla_device.h"

#include <utility>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "iree/base/status.h"
#include "iree/base/tracing.h"
#include "iree/hal/command_buffer_validation.h"
#include "iree/hal/command_queue.h"
#include "iree/hal/host/async_command_queue.h"
#include "iree/hal/host/host_descriptor_set.h"
#include "iree/hal/host/host_event.h"
#include "iree/hal/host/host_executable_layout.h"
#include "iree/hal/host/host_semaphore.h"
#include "iree/hal/host/inproc_command_buffer.h"
#include "iree/hal/host/serial_submission_queue.h"
#include "iree/hal/vmla/vmla_cache.h"
#include "iree/hal/vmla/vmla_command_processor.h"

namespace iree {
namespace hal {
namespace vmla {

namespace {

// A CommandQueue that performs no synchronization and just
// directly executes command buffers inline.
//
// This is meant to be wrapped by SyncCommandQueue or AsyncCommandQueue that
// themselves perform the synchronization/threading/etc. As such we ignore
// all semaphores in the provided batches under the assumption that if Submit is
// being called then all dependencies are valid. The wrapping queue is also
// responsible for signaling the semaphore as well as propagating errors in a
// way that is dependent on how it is performing its synchronization.
class UnsynchronizedCommandQueue final : public CommandQueue {
 public:
  UnsynchronizedCommandQueue(Allocator* allocator, std::string name,
                             CommandCategoryBitfield supported_categories)
      : CommandQueue(std::move(name), supported_categories),
        allocator_(allocator),
        command_processor_(allocator_, supported_categories_) {}
  ~UnsynchronizedCommandQueue() override = default;

  Status Submit(absl::Span<const SubmissionBatch> batches) override {
    IREE_TRACE_SCOPE0("UnsynchronizedCommandQueue::Submit");

    // Process command buffers and propagate errors asynchronously through the
    // semaphore. This ensures that even if we are running synchronously we
    // still get consistent failure behavior with drivers that are purely async.
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
      RETURN_IF_ERROR(inproc_command_buffer->Process(&command_processor_));
    }
    return OkStatus();
  }

  Allocator* const allocator_;
  VMLACommandProcessor command_processor_;
};

}  // namespace

VMLADevice::VMLADevice(DeviceInfo device_info, iree_vm_instance_t* instance,
                       iree_vm_module_t* vmla_module)
    : Device(std::move(device_info)),
      instance_(instance),
      vmla_module_(vmla_module) {
  iree_vm_instance_retain(instance_);
  iree_vm_module_retain(vmla_module_);

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

VMLADevice::~VMLADevice() {
  iree_vm_module_release(vmla_module_);
  iree_vm_instance_release(instance_);
}

std::string VMLADevice::DebugString() const {
  return absl::StrCat(Device::DebugString(),  //
                      "\n[VMLADevice]",       //
                      "\n  Command Queues: ", command_queues_.size());
}

ref_ptr<ExecutableCache> VMLADevice::CreateExecutableCache() {
  IREE_TRACE_SCOPE0("VMLADevice::CreateExecutableCache");
  return make_ref<VMLACache>(instance_, vmla_module_);
}

StatusOr<ref_ptr<DescriptorSetLayout>> VMLADevice::CreateDescriptorSetLayout(
    DescriptorSetLayout::UsageType usage_type,
    absl::Span<const DescriptorSetLayout::Binding> bindings) {
  IREE_TRACE_SCOPE0("VMLADevice::CreateDescriptorSetLayout");
  return make_ref<HostDescriptorSetLayout>(usage_type, bindings);
}

StatusOr<ref_ptr<ExecutableLayout>> VMLADevice::CreateExecutableLayout(
    absl::Span<DescriptorSetLayout* const> set_layouts, size_t push_constants) {
  IREE_TRACE_SCOPE0("VMLADevice::CreateExecutableLayout");
  return make_ref<HostExecutableLayout>(set_layouts, push_constants);
}

StatusOr<ref_ptr<DescriptorSet>> VMLADevice::CreateDescriptorSet(
    DescriptorSetLayout* set_layout,
    absl::Span<const DescriptorSet::Binding> bindings) {
  IREE_TRACE_SCOPE0("VMLADevice::CreateDescriptorSet");
  return make_ref<HostDescriptorSet>(set_layout, bindings);
}

StatusOr<ref_ptr<CommandBuffer>> VMLADevice::CreateCommandBuffer(
    CommandBufferModeBitfield mode,
    CommandCategoryBitfield command_categories) {
  // TODO(b/140026716): conditionally enable validation.
  auto impl =
      make_ref<InProcCommandBuffer>(&allocator_, mode, command_categories);
  return WrapCommandBufferWithValidation(std::move(impl));
}

StatusOr<ref_ptr<Event>> VMLADevice::CreateEvent() {
  return make_ref<HostEvent>();
}

StatusOr<ref_ptr<Semaphore>> VMLADevice::CreateSemaphore(
    uint64_t initial_value) {
  IREE_TRACE_SCOPE0("VMLADevice::CreateSemaphore");
  return make_ref<HostSemaphore>(initial_value);
}

Status VMLADevice::WaitAllSemaphores(
    absl::Span<const SemaphoreValue> semaphores, absl::Time deadline) {
  IREE_TRACE_SCOPE0("VMLADevice::WaitAllSemaphores");
  return HostSemaphore::WaitForSemaphores(semaphores, /*wait_all=*/true,
                                          deadline);
}

StatusOr<int> VMLADevice::WaitAnySemaphore(
    absl::Span<const SemaphoreValue> semaphores, absl::Time deadline) {
  IREE_TRACE_SCOPE0("VMLADevice::WaitAnySemaphore");
  return HostSemaphore::WaitForSemaphores(semaphores, /*wait_all=*/false,
                                          deadline);
}

Status VMLADevice::WaitIdle(absl::Time deadline) {
  for (auto& command_queue : command_queues_) {
    RETURN_IF_ERROR(command_queue->WaitIdle(deadline));
  }
  return OkStatus();
}

}  // namespace vmla
}  // namespace hal
}  // namespace iree
