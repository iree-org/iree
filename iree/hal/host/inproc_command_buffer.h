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

#ifndef IREE_HAL_HOST_INPROC_COMMAND_BUFFER_H_
#define IREE_HAL_HOST_INPROC_COMMAND_BUFFER_H_

#include "iree/base/arena.h"
#include "iree/base/intrusive_list.h"
#include "iree/base/status.h"
#include "iree/hal/command_buffer.h"

namespace iree {
namespace hal {
namespace host {

// In-process command buffer with support for recording and playback.
// Commands are recorded into heap-allocated arenas with pointers to used
// resources (Buffer*, etc). To replay a command buffer against a real
// implementation use Process to call each command method as it was originally
// recorded.
//
// Thread-compatible (as with CommandBuffer itself).
class InProcCommandBuffer final : public CommandBuffer {
 public:
  InProcCommandBuffer(CommandBufferModeBitfield mode,
                      CommandCategoryBitfield command_categories);
  ~InProcCommandBuffer() override;

  bool is_recording() const override { return is_recording_; }

  Status Begin() override;
  Status End() override;

  Status ExecutionBarrier(
      ExecutionStageBitfield source_stage_mask,
      ExecutionStageBitfield target_stage_mask,
      absl::Span<const MemoryBarrier> memory_barriers,
      absl::Span<const BufferBarrier> buffer_barriers) override;

  Status SignalEvent(Event* event,
                     ExecutionStageBitfield source_stage_mask) override;

  Status ResetEvent(Event* event,
                    ExecutionStageBitfield source_stage_mask) override;

  Status WaitEvents(absl::Span<Event*> events,
                    ExecutionStageBitfield source_stage_mask,
                    ExecutionStageBitfield target_stage_mask,
                    absl::Span<const MemoryBarrier> memory_barriers,
                    absl::Span<const BufferBarrier> buffer_barriers) override;

  Status FillBuffer(Buffer* target_buffer, device_size_t target_offset,
                    device_size_t length, const void* pattern,
                    size_t pattern_length) override;

  Status DiscardBuffer(Buffer* buffer) override;

  Status UpdateBuffer(const void* source_buffer, device_size_t source_offset,
                      Buffer* target_buffer, device_size_t target_offset,
                      device_size_t length) override;

  Status CopyBuffer(Buffer* source_buffer, device_size_t source_offset,
                    Buffer* target_buffer, device_size_t target_offset,
                    device_size_t length) override;

  Status PushConstants(ExecutableLayout* executable_layout, size_t offset,
                       absl::Span<const uint32_t> values) override;

  Status PushDescriptorSet(
      ExecutableLayout* executable_layout, int32_t set,
      absl::Span<const DescriptorSet::Binding> bindings) override;

  Status BindDescriptorSet(
      ExecutableLayout* executable_layout, int32_t set,
      DescriptorSet* descriptor_set,
      absl::Span<const device_size_t> dynamic_offsets) override;

  Status Dispatch(Executable* executable, int32_t entry_point,
                  std::array<uint32_t, 3> workgroups) override;

  Status DispatchIndirect(Executable* executable, int32_t entry_point,
                          Buffer* workgroups_buffer,
                          device_size_t workgroups_offset) override;

  // Processes all commands in the buffer using the given |command_processor|.
  // The commands are issued in the order they were recorded.
  Status Process(CommandBuffer* command_processor) const;

 private:
  // Type of Cmd, used by CmdHeader to identify the command payload.
  enum class CmdType {
    kExecutionBarrier,
    kSignalEvent,
    kResetEvent,
    kWaitEvents,
    kFillBuffer,
    kDiscardBuffer,
    kUpdateBuffer,
    kCopyBuffer,
    kPushConstants,
    kPushDescriptorSet,
    kBindDescriptorSet,
    kDispatch,
    kDispatchIndirect,
  };

  // Prefix for commands encoded into the CmdList.
  // This is used to identify the type of a command as well as connect commands
  // in the list sequence. Command data immediately follows the header in
  // memory.
  struct CmdHeader {
    // Optional next command in the list.
    CmdHeader* next;
    // Type of the command.
    CmdType type;
  };

  // A lightweight linked list of commands and an arena that stores them.
  // CmdLists are designed to be reused so that the arena allocations are
  // amortized across multiple uses.
  //
  // Note that this and the CmdHeader/Cmd types include raw pointers and as
  // such are *not* portable across processes. It'd be possible, though, to
  // extend this for cross-process use if a shared-memory Buffer was also
  // implemented. For YAGNI we avoid that here.
  struct CmdList : public IntrusiveLinkBase<void> {
    static constexpr size_t kArenaBlockSize = 64 * 1024;

    Arena arena{kArenaBlockSize};
    CmdHeader* head = nullptr;
    CmdHeader* tail = nullptr;
  };

  // Defines an execution barrier.
  struct ExecutionBarrierCmd {
    static constexpr CmdType kType = CmdType::kExecutionBarrier;
    ExecutionStageBitfield source_stage_mask;
    ExecutionStageBitfield target_stage_mask;
    absl::Span<const MemoryBarrier> memory_barriers;
    absl::Span<const BufferBarrier> buffer_barriers;
  };

  // Signals an event.
  struct SignalEventCmd {
    static constexpr CmdType kType = CmdType::kSignalEvent;
    Event* event;
    ExecutionStageBitfield source_stage_mask;
  };

  // Resets an event.
  struct ResetEventCmd {
    static constexpr CmdType kType = CmdType::kResetEvent;
    Event* event;
    ExecutionStageBitfield source_stage_mask;
  };

  // Waits for one or more events.
  struct WaitEventsCmd {
    static constexpr CmdType kType = CmdType::kWaitEvents;
    absl::Span<Event*> events;
    ExecutionStageBitfield source_stage_mask;
    ExecutionStageBitfield target_stage_mask;
    absl::Span<const MemoryBarrier> memory_barriers;
    absl::Span<const BufferBarrier> buffer_barriers;
  };

  // Fills the target buffer with the given repeating value.
  struct FillBufferCmd {
    static constexpr CmdType kType = CmdType::kFillBuffer;
    Buffer* target_buffer;
    device_size_t target_offset;
    device_size_t length;
    uint8_t pattern[4];
    size_t pattern_length;
  };

  // Hints to the device queue that the given buffer will not be used again.
  struct DiscardBufferCmd {
    static constexpr CmdType kType = CmdType::kDiscardBuffer;
    Buffer* buffer;
  };

  // Writes a range of the given target buffer from the embedded memory.
  // The source buffer contents immediately follow the command in the arena.
  struct UpdateBufferCmd {
    static constexpr CmdType kType = CmdType::kUpdateBuffer;
    const void* source_buffer;
    Buffer* target_buffer;
    device_size_t target_offset;
    device_size_t length;
  };

  // Copies a range of one buffer to another.
  struct CopyBufferCmd {
    static constexpr CmdType kType = CmdType::kCopyBuffer;
    Buffer* source_buffer;
    device_size_t source_offset;
    Buffer* target_buffer;
    device_size_t target_offset;
    device_size_t length;
  };

  // Pushes inline constant values.
  struct PushConstantsCmd {
    static constexpr CmdType kType = CmdType::kPushConstants;
    ExecutableLayout* executable_layout;
    size_t offset;
    absl::Span<const uint32_t> values;
  };

  // Pushes an inline descriptor set update.
  struct PushDescriptorSetCmd {
    static constexpr CmdType kType = CmdType::kPushDescriptorSet;
    ExecutableLayout* executable_layout;
    int32_t set;
    absl::Span<const DescriptorSet::Binding> bindings;
  };

  // Binds a descriptor set.
  struct BindDescriptorSetCmd {
    static constexpr CmdType kType = CmdType::kBindDescriptorSet;
    ExecutableLayout* executable_layout;
    int32_t set;
    DescriptorSet* descriptor_set;
    absl::Span<const device_size_t> dynamic_offsets;
  };

  // Dispatches an execution request.
  struct DispatchCmd {
    static constexpr CmdType kType = CmdType::kDispatch;
    Executable* executable;
    int32_t entry_point;
    std::array<uint32_t, 3> workgroups;
  };

  // Dispatches an execution request with indirect workgroup counts.
  struct DispatchIndirectCmd {
    static constexpr CmdType kType = CmdType::kDispatchIndirect;
    Executable* executable;
    int32_t entry_point;
    Buffer* workgroups_buffer;
    device_size_t workgroups_offset;
  };

  // Resets the command list.
  void Reset();

  // Allocates a command and appends it to the current command list.
  // The caller must populate the fields in the returned pointer.
  template <typename T>
  StatusOr<T*> AppendCmd() {
    return reinterpret_cast<T*>(AppendCmdHeader(T::kType, sizeof(T)) + 1);
  }

  // Appends a command with the given |type| and payload |cmd_size| prefixed
  // with a CmdHeader. Returns a pointer to the CmdHeader that is followed
  // immediately by |cmd_size| zero bytes.
  CmdHeader* AppendCmdHeader(CmdType type, size_t cmd_size);

  // Appends a byte buffer to the command buffer and returns a pointer to the
  // copied data within the command buffer arena.
  void* AppendCmdData(const void* source_buffer, device_size_t source_offset,
                      device_size_t source_length);

  // Appends a span of POD structs to the current CmdList and returns a span
  // pointing into the CmdList arena.
  template <typename T>
  absl::Span<T> AppendStructSpan(absl::Span<T> value) {
    static_assert(std::is_standard_layout<T>::value,
                  "Struct must be a POD type");
    void* data_ptr = AppendCmdData(value.data(), 0, value.size() * sizeof(T));
    return absl::MakeSpan(static_cast<T*>(data_ptr), value.size());
  }

  // Processes a single command.
  Status ProcessCmd(CmdHeader* cmd_header,
                    CommandBuffer* command_processor) const;

  bool is_recording_ = false;

  // NOTE: not synchronized. Expected to be used from a single thread.
  CmdList current_cmd_list_;
};

}  // namespace host
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_HOST_INPROC_COMMAND_BUFFER_H_
