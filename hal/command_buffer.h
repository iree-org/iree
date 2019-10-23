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

#ifndef IREE_HAL_COMMAND_BUFFER_H_
#define IREE_HAL_COMMAND_BUFFER_H_

#include <cstdint>

#include "base/bitfield.h"
#include "base/shape.h"
#include "base/status.h"
#include "hal/allocator.h"
#include "hal/buffer.h"
#include "hal/buffer_view.h"
#include "hal/event.h"
#include "hal/executable.h"
#include "hal/resource.h"

namespace iree {
namespace hal {

// A bitfield specifying the mode of operation for a command buffer.
enum class CommandBufferMode : uint32_t {
  // Command buffer will be submitted once and never used again.
  // This may enable in-place patching of command buffers that reduce overhead
  // when it's known that command buffers will not be reused.
  kOneShot = 1 << 0,
};
IREE_BITFIELD(CommandBufferMode);
using CommandBufferModeBitfield = CommandBufferMode;
std::string CommandBufferModeString(CommandBufferModeBitfield mode);

// A bitfield specifying the category of commands in a command queue.
enum class CommandCategory : uint32_t {
  // Command is considered a transfer operation (memcpy, etc).
  kTransfer = 1 << 0,
  // Command is considered a dispatch operation (dispatch/execute).
  kDispatch = 1 << 1,
};
IREE_BITFIELD(CommandCategory);
using CommandCategoryBitfield = CommandCategory;
std::string CommandCategoryString(CommandCategoryBitfield categories);

// Bitfield specifying which execution stage a brarrier should start/end at.
//
// Maps to VkPipelineStageFlagBits.
enum class ExecutionStage : uint32_t {
  // Top of the pipeline when commands are initially issued by the device.
  kCommandIssue = 1 << 0,
  // Stage of the pipeline when dispatch parameter data is consumed.
  kCommandProcess = 1 << 1,
  // Stage where dispatch commands execute.
  kDispatch = 1 << 2,
  // Stage where transfer (copy/clear/fill/etc) commands execute.
  kTransfer = 1 << 3,
  // Final stage in the pipeline when commands are retired on the device.
  kCommandRetire = 1 << 4,
  // Pseudo-stage for read/writes by the host. Not executed on device.
  kHost = 1 << 5,
};
IREE_BITFIELD(ExecutionStage);
using ExecutionStageBitfield = ExecutionStage;

// Bitfield specifying which scopes will access memory and how.
//
// Maps to VkAccessFlagBits.
enum class AccessScope : uint32_t {
  // Read access to indirect command data as part of an indirect dispatch.
  kIndirectCommandRead = 1 << 0,
  // Constant uniform buffer reads by the device.
  kConstantRead = 1 << 1,
  // Storage buffer reads by dispatch commands.
  kDispatchRead = 1 << 2,
  // Storage buffer writes by dispatch commands.
  kDispatchWrite = 1 << 3,
  // Source of a transfer operation.
  kTransferRead = 1 << 4,
  // Target of a transfer operation.
  kTransferWrite = 1 << 5,
  // Read operation by the host through mapped memory.
  kHostRead = 1 << 6,
  // Write operation by the host through mapped memory.
  kHostWrite = 1 << 7,
  // External/non-specific read.
  kMemoryRead = 1 << 8,
  // External/non-specific write.
  kMemoryWrite = 1 << 9,
};
IREE_BITFIELD(AccessScope);
using AccessScopeBitfield = AccessScope;

// Defines a global memory barrier.
// These are cheaper to encode than buffer-specific barriers but may cause
// stalls and bubbles in device pipelines if applied too broadly. Prefer them
// over equivalently large sets of buffer-specific barriers (such as when
// completely changing execution contexts).
//
// Maps to VkMemoryBarrier.
struct MemoryBarrier {
  // All access scopes prior-to the barrier (inclusive).
  AccessScopeBitfield source_scope;
  // All access scopes following the barrier (inclusive).
  AccessScopeBitfield target_scope;
};

// Defines a memory barrier that applies to a range of a specific buffer.
// Use of these (vs. global memory barriers) provides fine-grained execution
// ordering to device command processors and allows for more aggressive
// reordering.
//
// Maps to VkBufferMemoryBarrier.
struct BufferBarrier {
  // All access scopes prior-to the barrier (inclusive).
  AccessScopeBitfield source_scope;
  // All access scopes following the barrier (inclusive).
  AccessScopeBitfield target_scope;
  // Buffer the barrier is restricted to.
  // The barrier will apply to the entire physical device allocation.
  Buffer* buffer = nullptr;
  // Relative offset/length within |buffer| (which may itself be mapped into the
  // device allocation at an offset).
  device_size_t offset = 0;
  device_size_t length = kWholeBuffer;
};

// Represents a binding to a buffer with a set of attributes.
// This may be used by drivers to validate alignment.
struct BufferBinding {
  // Access rights of the buffer contents by the executable.
  MemoryAccessBitfield access = MemoryAccess::kAll;

  // The buffer this binding references.
  // The buffer is not retained by the binding and must be kept alive externally
  // for the duration it is in use by the queue.
  Buffer* buffer = nullptr;

  // Shape of the buffer contents.
  Shape shape;

  // Size of each element within the buffer, in bytes.
  int8_t element_size = 0;

  BufferBinding() = default;
  BufferBinding(MemoryAccessBitfield access, Buffer* buffer)
      : access(access), buffer(buffer) {}
  BufferBinding(MemoryAccessBitfield access, Buffer* buffer, Shape shape,
                int8_t element_size)
      : access(access),
        buffer(buffer),
        shape(shape),
        element_size(element_size) {}
  BufferBinding(MemoryAccessBitfield access, const BufferView& buffer_view)
      : access(access),
        buffer(buffer_view.buffer.get()),
        shape(buffer_view.shape),
        element_size(buffer_view.element_size) {}
};

// Wraps parameters for a Dispatch request.
struct DispatchRequest {
  // Executable prepared for use on the device.
  // The executable must remain alive until all in-flight dispatch requests
  // that use it have completed.
  Executable* executable = nullptr;

  // Executable entry point ordinal.
  int entry_point = 0;

  // TODO(benvanik): predication.

  // Static workload parameters defining the X, Y, and Z workgroup counts.
  std::array<int32_t, 3> workload;

  // An optional buffer containing the dynamic workload to dispatch.
  // The contents need not be available at the time of recording but must be
  // made visible prior to execution of the dispatch command.
  //
  // Buffer contents are expected to be 3 int32 values defining the X, Y, and Z
  // workgroup counts.
  //
  // The buffer must have been allocated with BufferUsage::kDispatch and be
  // of MemoryType::kDeviceVisible.
  Buffer* workload_buffer = nullptr;

  // A list of buffers that contain the execution inputs/outputs.
  // Order is dependent on executable arg layout.
  //
  // Buffers must have been allocated with BufferUsage::kDispatch and be
  // of MemoryType::kDeviceVisible.
  absl::Span<const BufferBinding> bindings;

  // TODO(benvanik): push-constant equivalent (uniforms, etc).
};

// Asynchronous command buffer recording interface.
// Commands are recorded by the implementation for later submission to command
// queues.
//
// Buffers and synchronization objects referenced must remain valid and not be
// modified or read while there are commands in-flight. The usual flow is to
// populate input buffers, Dispatch using those buffers, wait on a Fence until
// the buffers are guaranteed to no longer be in use, and then reuse or release
// the buffers.
//
// Errors that can be recognized when operations are enqueued will be returned
// immediately, such as invalid argument errors. Errors that can only be
// determined at execution time will be returned on fences. Once a failure
// occurs the device queue will enter an error state that invalidates all
// operations on the device queue (as ordering is not strict and any may still
// be in-flight). In this case the user of the device queue should treat all
// in-flight operations as cancelled and fully reset themselves. Other device
// queues that may be waiting on events from the device queue will also enter
// error states. Only once a user has acknowledged and cleared the error state
// with a Reset the queue will become usable, and otherwise all operations will
// return errors.
//
// Command buffers are thread-compatible. Use multiple command buffers if trying
// to record commands from multiple threads. Command buffers must not be mutated
// between when they have are submitted for execution on a queue and when the
// fence fires indicating the completion of their execution.
class CommandBuffer : public Resource {
 public:
  virtual CommandBuffer* impl() { return this; }

  // Device allocator that commands encoded into the buffer share compatibility
  // with.
  Allocator* allocator() const { return allocator_; }

  // Command buffer operation mode.
  CommandBufferModeBitfield mode() const { return mode_; }

  // Command categories that may be recorded into the buffer.
  CommandCategoryBitfield command_categories() const {
    return command_categories_;
  }

  // True if the command buffer is between a Begin/End recording block.
  virtual bool is_recording() const = 0;

  // Resets and begins recording into the command buffer, clearing all
  // previously recorded contents.
  // The command buffer must not be in-flight.
  virtual Status Begin() = 0;

  // Ends recording into the command buffer.
  // This must be called prior to submitting the command buffer for execution.
  virtual Status End() = 0;

  // TODO(benvanik): annotations for debugging and tracing:
  //  enter/exit
  //  stack frame manipulation
  //  explicit timers? or profiling buffer?

  // TODO(b/138719910): cross-queue and external acquire/release.
  // virtual Status AcquireBuffer() = 0;
  // virtual Status ReleaseBuffer() = 0;

  // Defines a memory dependency between commands recorded before and after the
  // barrier. One or more memory or buffer barriers can be specified to indicate
  // between which stages or buffers the dependencies exist.
  virtual Status ExecutionBarrier(
      ExecutionStageBitfield source_stage_mask,
      ExecutionStageBitfield target_stage_mask,
      absl::Span<const MemoryBarrier> memory_barriers,
      absl::Span<const BufferBarrier> buffer_barriers) = 0;

  // Sets an event to the signaled state.
  // |source_stage_mask| specifies when the event is signaled.
  //
  // Events are only valid within a single command buffer. Events can only be
  // used on non-transfer queues.
  virtual Status SignalEvent(Event* event,
                             ExecutionStageBitfield source_stage_mask) = 0;

  // Resets an event to the non-signaled state.
  // |source_stage_mask| specifies when the event is unsignaled.
  //
  // Events are only valid within a single command buffer. Events can only be
  // used on non-transfer queues.
  virtual Status ResetEvent(Event* event,
                            ExecutionStageBitfield source_stage_mask) = 0;

  // Waits for one or more events to be signaled and defines a memory dependency
  // between the synchronization scope of the signal operations and the commands
  // following the wait.
  //
  // |source_stage_mask| must include ExecutionStage::kHost for Event::Signal to
  // be visibile.
  //
  // Events are only valid within a single command buffer. Events remain
  // signaled even after waiting and must be reset to be reused. Events can only
  // be used on non-transfer queues.
  virtual Status WaitEvents(
      absl::Span<Event*> events, ExecutionStageBitfield source_stage_mask,
      ExecutionStageBitfield target_stage_mask,
      absl::Span<const MemoryBarrier> memory_barriers,
      absl::Span<const BufferBarrier> buffer_barriers) = 0;

  // Fills the target buffer with the given repeating value.
  // Expects that value_length is one of 1, 2, or 4 and that the offset and
  // length are aligned to the natural alignment of the value.
  // The target buffer must be compatible with the devices owned by this
  // device queue and be allocated with BufferUsage::kTransfer.
  virtual Status FillBuffer(Buffer* target_buffer, device_size_t target_offset,
                            device_size_t length, const void* pattern,
                            size_t pattern_length) = 0;

  // Hints to the device queue that the given buffer will not be used again.
  // After encoding a discard the buffer contents will be considered undefined.
  // This is because the discard may be used to elide write backs to host memory
  // or aggressively reuse the allocation for other purposes.
  //
  // For buffers allocated with MemoryType::kTransient this may allow
  // the device queue to reclaim the memory used by the buffer earlier than
  // otherwise possible.
  virtual Status DiscardBuffer(Buffer* buffer) = 0;

  // Updates a range of the given target buffer from the source host memory.
  // The source host memory is copied immediately into the command buffer and
  // occupies command buffer space. It is strongly recommended that large buffer
  // updates are performed via CopyBuffer where there is the possibility of a
  // zero-copy path.
  // The |source_buffer| may be releaed by the caller immediately after this
  // call returns.
  // The |target_buffer| must be compatible with the devices owned by this
  // device queue and be allocated with BufferUsage::kTransfer.
  virtual Status UpdateBuffer(const void* source_buffer,
                              device_size_t source_offset,
                              Buffer* target_buffer,
                              device_size_t target_offset,
                              device_size_t length) = 0;

  // Copies a range of one buffer to another.
  // Both buffers must be compatible with the devices owned by this device
  // queue and be allocated with BufferUsage::kTransfer. Though the source and
  // target buffer may be the same the ranges must not overlap (as with memcpy).
  //
  // This can be used to perform device->host, host->device, and device->device
  // copies.
  virtual Status CopyBuffer(Buffer* source_buffer, device_size_t source_offset,
                            Buffer* target_buffer, device_size_t target_offset,
                            device_size_t length) = 0;

  // Dispatches an execution request.
  // The request may execute overlapped with any other transfer operation or
  // dispatch made within the same barrier-defined sequence.
  //
  // The executable specified must be registered for use with the device driver
  // owning this queue. It must not be unregistered until all requests that use
  // it have completed.
  //
  // Fails if the queue does not support dispatch operations (as indicated by
  // can_dispatch).
  virtual Status Dispatch(const DispatchRequest& dispatch_request) = 0;

 protected:
  CommandBuffer(Allocator* allocator, CommandBufferModeBitfield mode,
                CommandCategoryBitfield command_categories)
      : allocator_(allocator),
        mode_(mode),
        command_categories_(command_categories) {}

 private:
  Allocator* const allocator_;
  const CommandBufferModeBitfield mode_;
  const CommandCategoryBitfield command_categories_;
};

}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_COMMAND_BUFFER_H_
