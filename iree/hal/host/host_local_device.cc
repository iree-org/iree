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

#include "iree/hal/host/host_local_device.h"

#include <utility>

#include "absl/memory/memory.h"
#include "iree/base/status.h"
#include "iree/base/tracing.h"
#include "iree/hal/command_buffer_validation.h"
#include "iree/hal/host/host_descriptor_set.h"
#include "iree/hal/host/host_executable_layout.h"

namespace iree {
namespace hal {

HostLocalDevice::HostLocalDevice(
    DeviceInfo device_info, std::unique_ptr<SchedulingModel> scheduling_model)
    : Device(std::move(device_info)),
      scheduling_model_(std::move(scheduling_model)) {}

HostLocalDevice::~HostLocalDevice() = default;

StatusOr<ref_ptr<DescriptorSetLayout>>
HostLocalDevice::CreateDescriptorSetLayout(
    DescriptorSetLayout::UsageType usage_type,
    absl::Span<const DescriptorSetLayout::Binding> bindings) {
  IREE_TRACE_SCOPE0("HostLocalDevice::CreateDescriptorSetLayout");
  return make_ref<HostDescriptorSetLayout>(usage_type, bindings);
}

StatusOr<ref_ptr<ExecutableLayout>> HostLocalDevice::CreateExecutableLayout(
    absl::Span<DescriptorSetLayout* const> set_layouts, size_t push_constants) {
  IREE_TRACE_SCOPE0("HostLocalDevice::CreateExecutableLayout");
  return make_ref<HostExecutableLayout>(set_layouts, push_constants);
}

StatusOr<ref_ptr<DescriptorSet>> HostLocalDevice::CreateDescriptorSet(
    DescriptorSetLayout* set_layout,
    absl::Span<const DescriptorSet::Binding> bindings) {
  IREE_TRACE_SCOPE0("HostLocalDevice::CreateDescriptorSet");
  return make_ref<HostDescriptorSet>(set_layout, bindings);
}

StatusOr<ref_ptr<CommandBuffer>> HostLocalDevice::CreateCommandBuffer(
    CommandBufferModeBitfield mode,
    CommandCategoryBitfield command_categories) {
  IREE_TRACE_SCOPE0("HostLocalDevice::CreateCommandBuffer");
  // TODO(b/140026716): conditionally enable validation.
  ASSIGN_OR_RETURN(auto impl, scheduling_model_->CreateCommandBuffer(
                                  mode, command_categories));
  return WrapCommandBufferWithValidation(allocator(), std::move(impl));
}

StatusOr<ref_ptr<Event>> HostLocalDevice::CreateEvent() {
  IREE_TRACE_SCOPE0("HostLocalDevice::CreateEvent");
  return scheduling_model_->CreateEvent();
}

StatusOr<ref_ptr<Semaphore>> HostLocalDevice::CreateSemaphore(
    uint64_t initial_value) {
  IREE_TRACE_SCOPE0("HostLocalDevice::CreateSemaphore");
  return scheduling_model_->CreateSemaphore(initial_value);
}

Status HostLocalDevice::WaitAllSemaphores(
    absl::Span<const SemaphoreValue> semaphores, absl::Time deadline) {
  IREE_TRACE_SCOPE0("HostLocalDevice::WaitAllSemaphores");
  return scheduling_model_->WaitAllSemaphores(semaphores, deadline);
}

StatusOr<int> HostLocalDevice::WaitAnySemaphore(
    absl::Span<const SemaphoreValue> semaphores, absl::Time deadline) {
  IREE_TRACE_SCOPE0("HostLocalDevice::WaitAnySemaphore");
  return scheduling_model_->WaitAnySemaphore(semaphores, deadline);
}

Status HostLocalDevice::WaitIdle(absl::Time deadline) {
  IREE_TRACE_SCOPE0("HostLocalDevice::WaitIdle");
  return scheduling_model_->WaitIdle(deadline);
}

}  // namespace hal
}  // namespace iree
