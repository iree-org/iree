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

#include "hal/host/host_local_command_processor.h"

#include "base/source_location.h"
#include "base/status.h"
#include "base/tracing.h"

namespace iree {
namespace hal {

HostLocalCommandProcessor::HostLocalCommandProcessor(
    Allocator* allocator, CommandBufferModeBitfield mode,
    CommandCategoryBitfield command_categories)
    : CommandBuffer(allocator, mode, command_categories) {}

HostLocalCommandProcessor::~HostLocalCommandProcessor() = default;

Status HostLocalCommandProcessor::Begin() {
  IREE_TRACE_SCOPE0("HostLocalCommandProcessor::Begin");
  is_recording_ = true;
  return OkStatus();
}

Status HostLocalCommandProcessor::End() {
  IREE_TRACE_SCOPE0("HostLocalCommandProcessor::End");
  is_recording_ = false;
  return OkStatus();
}

Status HostLocalCommandProcessor::ExecutionBarrier(
    ExecutionStageBitfield source_stage_mask,
    ExecutionStageBitfield target_stage_mask,
    absl::Span<const MemoryBarrier> memory_barriers,
    absl::Span<const BufferBarrier> buffer_barriers) {
  IREE_TRACE_SCOPE0("HostLocalCommandProcessor::ExecutionBarrier");
  // No-op.
  return OkStatus();
}

Status HostLocalCommandProcessor::SignalEvent(
    Event* event, ExecutionStageBitfield source_stage_mask) {
  IREE_TRACE_SCOPE0("HostLocalCommandProcessor::SignalEvent");
  // No-op.
  return OkStatus();
}

Status HostLocalCommandProcessor::ResetEvent(
    Event* event, ExecutionStageBitfield source_stage_mask) {
  IREE_TRACE_SCOPE0("HostLocalCommandProcessor::ResetEvent");
  // No-op.
  return OkStatus();
}

Status HostLocalCommandProcessor::WaitEvents(
    absl::Span<Event*> events, ExecutionStageBitfield source_stage_mask,
    ExecutionStageBitfield target_stage_mask,
    absl::Span<const MemoryBarrier> memory_barriers,
    absl::Span<const BufferBarrier> buffer_barriers) {
  IREE_TRACE_SCOPE0("HostLocalCommandProcessor::WaitEvents");
  // No-op.
  return OkStatus();
}

Status HostLocalCommandProcessor::FillBuffer(Buffer* target_buffer,
                                             device_size_t target_offset,
                                             device_size_t length,
                                             const void* pattern,
                                             size_t pattern_length) {
  IREE_TRACE_SCOPE0("HostLocalCommandProcessor::FillBuffer");
  return target_buffer->Fill(target_offset, length, pattern, pattern_length);
}

Status HostLocalCommandProcessor::DiscardBuffer(Buffer* buffer) {
  IREE_TRACE_SCOPE0("HostLocalCommandProcessor::DiscardBuffer");
  // No-op as we don't support lazily allocated buffers.
  return OkStatus();
}

Status HostLocalCommandProcessor::UpdateBuffer(const void* source_buffer,
                                               device_size_t source_offset,
                                               Buffer* target_buffer,
                                               device_size_t target_offset,
                                               device_size_t length) {
  IREE_TRACE_SCOPE0("HostLocalCommandProcessor::UpdateBuffer");
  return target_buffer->WriteData(
      target_offset, static_cast<const uint8_t*>(source_buffer) + source_offset,
      length);
}

Status HostLocalCommandProcessor::CopyBuffer(Buffer* source_buffer,
                                             device_size_t source_offset,
                                             Buffer* target_buffer,
                                             device_size_t target_offset,
                                             device_size_t length) {
  IREE_TRACE_SCOPE0("HostLocalCommandProcessor::CopyBuffer");
  return target_buffer->CopyData(target_offset, source_buffer, source_offset,
                                 length);
}

Status HostLocalCommandProcessor::Dispatch(
    const DispatchRequest& dispatch_request) {
  return FailedPreconditionErrorBuilder(IREE_LOC)
         << "Command processor does not support dispatch operations";
}

}  // namespace hal
}  // namespace iree
