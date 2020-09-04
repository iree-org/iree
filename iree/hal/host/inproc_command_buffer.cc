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

#include "iree/hal/host/inproc_command_buffer.h"

#include "iree/base/tracing.h"

namespace iree {
namespace hal {
namespace host {

InProcCommandBuffer::InProcCommandBuffer(
    CommandBufferModeBitfield mode, CommandCategoryBitfield command_categories)
    : CommandBuffer(mode, command_categories) {}

InProcCommandBuffer::~InProcCommandBuffer() { Reset(); }

Status InProcCommandBuffer::Begin() {
  IREE_TRACE_SCOPE0("InProcCommandBuffer::Begin");
  is_recording_ = true;
  Reset();
  return OkStatus();
}

Status InProcCommandBuffer::End() {
  IREE_TRACE_SCOPE0("InProcCommandBuffer::End");
  is_recording_ = false;
  return OkStatus();
}

Status InProcCommandBuffer::ExecutionBarrier(
    ExecutionStageBitfield source_stage_mask,
    ExecutionStageBitfield target_stage_mask,
    absl::Span<const MemoryBarrier> memory_barriers,
    absl::Span<const BufferBarrier> buffer_barriers) {
  IREE_TRACE_SCOPE0("InProcCommandBuffer::ExecutionBarrier");
  IREE_ASSIGN_OR_RETURN(auto* cmd, AppendCmd<ExecutionBarrierCmd>());
  cmd->source_stage_mask = source_stage_mask;
  cmd->target_stage_mask = target_stage_mask;
  cmd->memory_barriers = AppendStructSpan(memory_barriers);
  cmd->buffer_barriers = AppendStructSpan(buffer_barriers);
  return OkStatus();
}

Status InProcCommandBuffer::SignalEvent(
    Event* event, ExecutionStageBitfield source_stage_mask) {
  IREE_TRACE_SCOPE0("InProcCommandBuffer::SignalEvent");
  IREE_ASSIGN_OR_RETURN(auto* cmd, AppendCmd<SignalEventCmd>());
  cmd->event = event;
  cmd->source_stage_mask = source_stage_mask;
  return OkStatus();
}

Status InProcCommandBuffer::ResetEvent(
    Event* event, ExecutionStageBitfield source_stage_mask) {
  IREE_TRACE_SCOPE0("InProcCommandBuffer::ResetEvent");
  IREE_ASSIGN_OR_RETURN(auto* cmd, AppendCmd<ResetEventCmd>());
  cmd->event = event;
  cmd->source_stage_mask = source_stage_mask;
  return OkStatus();
}

Status InProcCommandBuffer::WaitEvents(
    absl::Span<Event*> events, ExecutionStageBitfield source_stage_mask,
    ExecutionStageBitfield target_stage_mask,
    absl::Span<const MemoryBarrier> memory_barriers,
    absl::Span<const BufferBarrier> buffer_barriers) {
  IREE_TRACE_SCOPE0("InProcCommandBuffer::WaitEvents");
  IREE_ASSIGN_OR_RETURN(auto* cmd, AppendCmd<WaitEventsCmd>());
  cmd->events = AppendStructSpan(events);
  cmd->source_stage_mask = source_stage_mask;
  cmd->target_stage_mask = target_stage_mask;
  cmd->memory_barriers = AppendStructSpan(memory_barriers);
  cmd->buffer_barriers = AppendStructSpan(buffer_barriers);
  return OkStatus();
}

Status InProcCommandBuffer::FillBuffer(Buffer* target_buffer,
                                       device_size_t target_offset,
                                       device_size_t length,
                                       const void* pattern,
                                       size_t pattern_length) {
  IREE_TRACE_SCOPE0("InProcCommandBuffer::FillBuffer");
  IREE_ASSIGN_OR_RETURN(auto* cmd, AppendCmd<FillBufferCmd>());
  cmd->target_buffer = target_buffer;
  cmd->target_offset = target_offset;
  cmd->length = length;
  std::memcpy(cmd->pattern, pattern, pattern_length);
  cmd->pattern_length = pattern_length;
  return OkStatus();
}

Status InProcCommandBuffer::DiscardBuffer(Buffer* buffer) {
  IREE_TRACE_SCOPE0("InProcCommandBuffer::DiscardBuffer");
  IREE_ASSIGN_OR_RETURN(auto* cmd, AppendCmd<DiscardBufferCmd>());
  cmd->buffer = buffer;
  return OkStatus();
}

Status InProcCommandBuffer::UpdateBuffer(const void* source_buffer,
                                         device_size_t source_offset,
                                         Buffer* target_buffer,
                                         device_size_t target_offset,
                                         device_size_t length) {
  IREE_TRACE_SCOPE0("InProcCommandBuffer::UpdateBuffer");
  IREE_ASSIGN_OR_RETURN(auto* cmd, AppendCmd<UpdateBufferCmd>());
  cmd->source_buffer = AppendCmdData(source_buffer, source_offset, length);
  cmd->target_buffer = target_buffer;
  cmd->target_offset = target_offset;
  cmd->length = length;
  return OkStatus();
}

Status InProcCommandBuffer::CopyBuffer(Buffer* source_buffer,
                                       device_size_t source_offset,
                                       Buffer* target_buffer,
                                       device_size_t target_offset,
                                       device_size_t length) {
  IREE_TRACE_SCOPE0("InProcCommandBuffer::CopyBuffer");
  IREE_ASSIGN_OR_RETURN(auto* cmd, AppendCmd<CopyBufferCmd>());
  cmd->source_buffer = source_buffer;
  cmd->source_offset = source_offset;
  cmd->target_buffer = target_buffer;
  cmd->target_offset = target_offset;
  cmd->length = length;
  return OkStatus();
}

Status InProcCommandBuffer::PushConstants(ExecutableLayout* executable_layout,
                                          size_t offset,
                                          absl::Span<const uint32_t> values) {
  IREE_TRACE_SCOPE0("InProcCommandBuffer::PushConstants");
  IREE_ASSIGN_OR_RETURN(auto* cmd, AppendCmd<PushConstantsCmd>());
  cmd->executable_layout = executable_layout;
  cmd->offset = offset;
  cmd->values = AppendStructSpan(values);
  return OkStatus();
}

Status InProcCommandBuffer::PushDescriptorSet(
    ExecutableLayout* executable_layout, int32_t set,
    absl::Span<const DescriptorSet::Binding> bindings) {
  IREE_TRACE_SCOPE0("InProcCommandBuffer::PushDescriptorSet");
  IREE_ASSIGN_OR_RETURN(auto* cmd, AppendCmd<PushDescriptorSetCmd>());
  cmd->executable_layout = executable_layout;
  cmd->set = set;
  cmd->bindings = AppendStructSpan(bindings);
  return OkStatus();
}

Status InProcCommandBuffer::BindDescriptorSet(
    ExecutableLayout* executable_layout, int32_t set,
    DescriptorSet* descriptor_set,
    absl::Span<const device_size_t> dynamic_offsets) {
  IREE_TRACE_SCOPE0("InProcCommandBuffer::BindDescriptorSet");
  IREE_ASSIGN_OR_RETURN(auto* cmd, AppendCmd<BindDescriptorSetCmd>());
  cmd->executable_layout = executable_layout;
  cmd->set = set;
  cmd->descriptor_set = descriptor_set;
  cmd->dynamic_offsets = AppendStructSpan(dynamic_offsets);
  return OkStatus();
}

Status InProcCommandBuffer::Dispatch(Executable* executable,
                                     int32_t entry_point,
                                     std::array<uint32_t, 3> workgroups) {
  IREE_TRACE_SCOPE0("InProcCommandBuffer::Dispatch");
  IREE_ASSIGN_OR_RETURN(auto* cmd, AppendCmd<DispatchCmd>());
  cmd->executable = executable;
  cmd->entry_point = entry_point;
  cmd->workgroups = workgroups;
  return OkStatus();
}

Status InProcCommandBuffer::DispatchIndirect(Executable* executable,
                                             int32_t entry_point,
                                             Buffer* workgroups_buffer,
                                             device_size_t workgroups_offset) {
  IREE_TRACE_SCOPE0("InProcCommandBuffer::DispatchIndirect");
  IREE_ASSIGN_OR_RETURN(auto* cmd, AppendCmd<DispatchIndirectCmd>());
  cmd->executable = executable;
  cmd->entry_point = entry_point;
  cmd->workgroups_buffer = workgroups_buffer;
  cmd->workgroups_offset = workgroups_offset;
  return OkStatus();
}

void InProcCommandBuffer::Reset() {
  auto* cmd_list = &current_cmd_list_;
  cmd_list->head = cmd_list->tail = nullptr;
  cmd_list->arena.Reset();
}

InProcCommandBuffer::CmdHeader* InProcCommandBuffer::AppendCmdHeader(
    CmdType type, size_t cmd_size) {
  auto* cmd_list = &current_cmd_list_;
  auto* cmd_header = reinterpret_cast<CmdHeader*>(
      cmd_list->arena.AllocateBytes(sizeof(CmdHeader) + cmd_size));
  cmd_header->next = nullptr;
  cmd_header->type = type;
  if (!cmd_list->head) {
    cmd_list->head = cmd_header;
  } else if (cmd_list->tail) {
    cmd_list->tail->next = cmd_header;
  }
  cmd_list->tail = cmd_header;
  return cmd_header;
}

void* InProcCommandBuffer::AppendCmdData(const void* source_buffer,
                                         device_size_t source_offset,
                                         device_size_t source_length) {
  auto* cmd_list = &current_cmd_list_;

  uint8_t* allocated_bytes = cmd_list->arena.AllocateBytes(source_length);
  std::memcpy(allocated_bytes,
              static_cast<const uint8_t*>(source_buffer) + source_offset,
              source_length);
  return allocated_bytes;
}

Status InProcCommandBuffer::Process(CommandBuffer* command_processor) const {
  IREE_TRACE_SCOPE0("InProcCommandBuffer::Process");

  IREE_RETURN_IF_ERROR(command_processor->Begin());

  // Process each command in the order they were recorded.
  auto* cmd_list = &current_cmd_list_;
  for (CmdHeader* cmd_header = cmd_list->head; cmd_header != nullptr;
       cmd_header = cmd_header->next) {
    auto command_status = ProcessCmd(cmd_header, command_processor);
    if (!command_status.ok()) {
      LOG(ERROR) << "DeviceQueue failure while executing command; permanently "
                    "failing all future commands: "
                 << command_status;
      return command_status;
    }
  }

  IREE_RETURN_IF_ERROR(command_processor->End());

  return OkStatus();
}

Status InProcCommandBuffer::ProcessCmd(CmdHeader* cmd_header,
                                       CommandBuffer* command_processor) const {
  switch (cmd_header->type) {
    case CmdType::kExecutionBarrier: {
      auto* cmd = reinterpret_cast<ExecutionBarrierCmd*>(cmd_header + 1);
      return command_processor->ExecutionBarrier(
          cmd->source_stage_mask, cmd->target_stage_mask, cmd->memory_barriers,
          cmd->buffer_barriers);
    }
    case CmdType::kSignalEvent: {
      auto* cmd = reinterpret_cast<SignalEventCmd*>(cmd_header + 1);
      return command_processor->SignalEvent(cmd->event, cmd->source_stage_mask);
    }
    case CmdType::kResetEvent: {
      auto* cmd = reinterpret_cast<ResetEventCmd*>(cmd_header + 1);
      return command_processor->ResetEvent(cmd->event, cmd->source_stage_mask);
    }
    case CmdType::kWaitEvents: {
      auto* cmd = reinterpret_cast<WaitEventsCmd*>(cmd_header + 1);
      return command_processor->WaitEvents(
          cmd->events, cmd->source_stage_mask, cmd->target_stage_mask,
          cmd->memory_barriers, cmd->buffer_barriers);
    }
    case CmdType::kFillBuffer: {
      auto* cmd = reinterpret_cast<FillBufferCmd*>(cmd_header + 1);
      return command_processor->FillBuffer(cmd->target_buffer,
                                           cmd->target_offset, cmd->length,
                                           cmd->pattern, cmd->pattern_length);
    }
    case CmdType::kDiscardBuffer: {
      auto* cmd = reinterpret_cast<DiscardBufferCmd*>(cmd_header + 1);
      return command_processor->DiscardBuffer(cmd->buffer);
    }
    case CmdType::kUpdateBuffer: {
      auto* cmd = reinterpret_cast<UpdateBufferCmd*>(cmd_header + 1);
      return command_processor->UpdateBuffer(cmd->source_buffer, 0,
                                             cmd->target_buffer,
                                             cmd->target_offset, cmd->length);
    }
    case CmdType::kCopyBuffer: {
      auto* cmd = reinterpret_cast<CopyBufferCmd*>(cmd_header + 1);
      return command_processor->CopyBuffer(
          cmd->source_buffer, cmd->source_offset, cmd->target_buffer,
          cmd->target_offset, cmd->length);
    }
    case CmdType::kPushConstants: {
      auto* cmd = reinterpret_cast<PushConstantsCmd*>(cmd_header + 1);
      return command_processor->PushConstants(cmd->executable_layout,
                                              cmd->offset, cmd->values);
    }
    case CmdType::kPushDescriptorSet: {
      auto* cmd = reinterpret_cast<PushDescriptorSetCmd*>(cmd_header + 1);
      return command_processor->PushDescriptorSet(cmd->executable_layout,
                                                  cmd->set, cmd->bindings);
    }
    case CmdType::kBindDescriptorSet: {
      auto* cmd = reinterpret_cast<BindDescriptorSetCmd*>(cmd_header + 1);
      return command_processor->BindDescriptorSet(cmd->executable_layout,
                                                  cmd->set, cmd->descriptor_set,
                                                  cmd->dynamic_offsets);
    }
    case CmdType::kDispatch: {
      auto* cmd = reinterpret_cast<DispatchCmd*>(cmd_header + 1);
      return command_processor->Dispatch(cmd->executable, cmd->entry_point,
                                         cmd->workgroups);
    }
    case CmdType::kDispatchIndirect: {
      auto* cmd = reinterpret_cast<DispatchIndirectCmd*>(cmd_header + 1);
      return command_processor->DispatchIndirect(
          cmd->executable, cmd->entry_point, cmd->workgroups_buffer,
          cmd->workgroups_offset);
    }
    default:
      return DataLossErrorBuilder(IREE_LOC)
             << "Unrecognized command type "
             << static_cast<int>(cmd_header->type) << "; corrupt buffer?";
  }
}

}  // namespace host
}  // namespace hal
}  // namespace iree
