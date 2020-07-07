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

#include "iree/hal/host/serial/serial_command_processor.h"

#include "iree/base/source_location.h"
#include "iree/base/status.h"
#include "iree/base/tracing.h"
#include "iree/hal/host/host_descriptor_set.h"
#include "iree/hal/host/host_executable_layout.h"

namespace iree {
namespace hal {
namespace host {

SerialCommandProcessor::SerialCommandProcessor(
    CommandCategoryBitfield command_categories)
    : CommandBuffer(CommandBufferMode::kOneShot, command_categories) {}

SerialCommandProcessor::~SerialCommandProcessor() = default;

Status SerialCommandProcessor::Begin() {
  IREE_TRACE_SCOPE0("SerialCommandProcessor::Begin");
  is_recording_ = true;
  return OkStatus();
}

Status SerialCommandProcessor::End() {
  IREE_TRACE_SCOPE0("SerialCommandProcessor::End");
  is_recording_ = false;
  return OkStatus();
}

Status SerialCommandProcessor::ExecutionBarrier(
    ExecutionStageBitfield source_stage_mask,
    ExecutionStageBitfield target_stage_mask,
    absl::Span<const MemoryBarrier> memory_barriers,
    absl::Span<const BufferBarrier> buffer_barriers) {
  IREE_TRACE_SCOPE0("SerialCommandProcessor::ExecutionBarrier");
  // No-op.
  return OkStatus();
}

Status SerialCommandProcessor::SignalEvent(
    Event* event, ExecutionStageBitfield source_stage_mask) {
  IREE_TRACE_SCOPE0("SerialCommandProcessor::SignalEvent");
  // No-op.
  return OkStatus();
}

Status SerialCommandProcessor::ResetEvent(
    Event* event, ExecutionStageBitfield source_stage_mask) {
  IREE_TRACE_SCOPE0("SerialCommandProcessor::ResetEvent");
  // No-op.
  return OkStatus();
}

Status SerialCommandProcessor::WaitEvents(
    absl::Span<Event*> events, ExecutionStageBitfield source_stage_mask,
    ExecutionStageBitfield target_stage_mask,
    absl::Span<const MemoryBarrier> memory_barriers,
    absl::Span<const BufferBarrier> buffer_barriers) {
  IREE_TRACE_SCOPE0("SerialCommandProcessor::WaitEvents");
  // No-op.
  return OkStatus();
}

Status SerialCommandProcessor::FillBuffer(Buffer* target_buffer,
                                          device_size_t target_offset,
                                          device_size_t length,
                                          const void* pattern,
                                          size_t pattern_length) {
  IREE_TRACE_SCOPE0("SerialCommandProcessor::FillBuffer");
  return target_buffer->Fill(target_offset, length, pattern, pattern_length);
}

Status SerialCommandProcessor::DiscardBuffer(Buffer* buffer) {
  IREE_TRACE_SCOPE0("SerialCommandProcessor::DiscardBuffer");
  // No-op as we don't support lazily allocated buffers.
  return OkStatus();
}

Status SerialCommandProcessor::UpdateBuffer(const void* source_buffer,
                                            device_size_t source_offset,
                                            Buffer* target_buffer,
                                            device_size_t target_offset,
                                            device_size_t length) {
  IREE_TRACE_SCOPE0("SerialCommandProcessor::UpdateBuffer");
  return target_buffer->WriteData(
      target_offset, static_cast<const uint8_t*>(source_buffer) + source_offset,
      length);
}

Status SerialCommandProcessor::CopyBuffer(Buffer* source_buffer,
                                          device_size_t source_offset,
                                          Buffer* target_buffer,
                                          device_size_t target_offset,
                                          device_size_t length) {
  IREE_TRACE_SCOPE0("SerialCommandProcessor::CopyBuffer");
  return target_buffer->CopyData(target_offset, source_buffer, source_offset,
                                 length);
}

Status SerialCommandProcessor::PushConstants(
    ExecutableLayout* executable_layout, size_t offset,
    absl::Span<const uint32_t> values) {
  IREE_TRACE_SCOPE0("SerialCommandProcessor::PushConstants");
  if (offset + values.size() > push_constants_.values.size()) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "Push constants out of range";
  }
  for (int i = 0; i < values.size(); ++i) {
    push_constants_.values[offset + i] = values[i];
  }
  return OkStatus();
}

Status SerialCommandProcessor::PushDescriptorSet(
    ExecutableLayout* executable_layout, int32_t set,
    absl::Span<const DescriptorSet::Binding> bindings) {
  IREE_TRACE_SCOPE0("SerialCommandProcessor::PushDescriptorSet");
  if (!AnyBitSet(command_categories() & CommandCategory::kDispatch)) {
    return FailedPreconditionErrorBuilder(IREE_LOC)
           << "Command processor does not support dispatch operations";
  }

  auto* host_executable_layout =
      static_cast<HostExecutableLayout*>(executable_layout);
  descriptor_sets_.resize(host_executable_layout->set_count());
  if (set < 0 || set >= descriptor_sets_.size()) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "Set " << set << " out of range (" << descriptor_sets_.size()
           << ")";
  }

  auto& set_bindings = descriptor_sets_[set];
  set_bindings = {bindings.begin(), bindings.end()};

  return OkStatus();
}

Status SerialCommandProcessor::BindDescriptorSet(
    ExecutableLayout* executable_layout, int32_t set,
    DescriptorSet* descriptor_set,
    absl::Span<const device_size_t> dynamic_offsets) {
  IREE_TRACE_SCOPE0("SerialCommandProcessor::BindDescriptorSet");
  if (!AnyBitSet(command_categories() & CommandCategory::kDispatch)) {
    return FailedPreconditionErrorBuilder(IREE_LOC)
           << "Command processor does not support dispatch operations";
  }

  auto* host_executable_layout =
      static_cast<HostExecutableLayout*>(executable_layout);
  descriptor_sets_.resize(host_executable_layout->set_count());
  if (set < 0 || descriptor_sets_.size() >= set) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "Set " << set << " out of range (" << descriptor_sets_.size()
           << ")";
  }

  auto* host_descriptor_set = static_cast<HostDescriptorSet*>(descriptor_set);
  auto* set_bindings = &descriptor_sets_[set];
  *set_bindings = {host_descriptor_set->bindings().begin(),
                   host_descriptor_set->bindings().end()};
  if (!dynamic_offsets.empty()) {
    auto dynamic_binding_map =
        host_executable_layout->GetDynamicBindingMap(set);
    if (dynamic_offsets.size() != dynamic_binding_map.size()) {
      return InvalidArgumentErrorBuilder(IREE_LOC)
             << "Dynamic offset count mismatch (provided "
             << dynamic_offsets.size() << " but expected "
             << dynamic_binding_map.size() << ")";
    }
    for (int i = 0; i < dynamic_binding_map.size(); ++i) {
      (*set_bindings)[dynamic_binding_map[i]].offset += dynamic_offsets[i];
    }
  }

  return OkStatus();
}

Status SerialCommandProcessor::Dispatch(Executable* executable,
                                        int32_t entry_point,
                                        std::array<uint32_t, 3> workgroups) {
  IREE_TRACE_SCOPE0("SerialCommandProcessor::Dispatch");
  return DispatchGrid(executable, entry_point, workgroups);
}

Status SerialCommandProcessor::DispatchIndirect(
    Executable* executable, int32_t entry_point, Buffer* workgroups_buffer,
    device_size_t workgroups_offset) {
  IREE_TRACE_SCOPE0("SerialCommandProcessor::DispatchIndirect");

  std::array<uint32_t, 3> workgroup_count;
  RETURN_IF_ERROR(workgroups_buffer->ReadData(
      workgroups_offset, workgroup_count.data(), sizeof(uint32_t) * 3));

  return DispatchGrid(executable, entry_point, workgroup_count);
}

Status SerialCommandProcessor::DispatchGrid(
    Executable* executable, int32_t entry_point,
    std::array<uint32_t, 3> workgroup_count) {
  HostExecutable::DispatchParams params;
  params.entry_point = entry_point;
  params.workgroup_count = workgroup_count;
  params.push_constants = &push_constants_;

  absl::InlinedVector<absl::Span<const DescriptorSet::Binding>, 2>
      descriptor_sets(descriptor_sets_.size());
  for (int i = 0; i < descriptor_sets_.size(); ++i) {
    descriptor_sets[i] = absl::MakeConstSpan(descriptor_sets_[i]);
  }
  params.set_bindings = descriptor_sets;

  auto* host_executable = reinterpret_cast<HostExecutable*>(executable);
  ASSIGN_OR_RETURN(auto dispatch_state,
                   host_executable->PrepareDispatch(params));
  for (uint32_t z = 0; z < params.workgroup_count[2]; ++z) {
    for (uint32_t y = 0; y < params.workgroup_count[1]; ++y) {
      for (uint32_t x = 0; x < params.workgroup_count[0]; ++x) {
        RETURN_IF_ERROR(
            host_executable->DispatchTile(dispatch_state.get(), {x, y, z}));
      }
    }
  }
  return OkStatus();
}

}  // namespace host
}  // namespace hal
}  // namespace iree
