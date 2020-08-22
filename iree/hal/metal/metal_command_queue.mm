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

#include "iree/hal/metal/metal_command_queue.h"

#include "iree/base/status.h"
#include "iree/base/tracing.h"
#include "iree/hal/metal/apple_time_util.h"
#include "iree/hal/metal/metal_command_buffer.h"
#include "iree/hal/metal/metal_shared_event.h"

namespace iree {
namespace hal {
namespace metal {

MetalCommandQueue::MetalCommandQueue(std::string name, CommandCategoryBitfield supported_categories,
                                     id<MTLCommandQueue> queue)
    : CommandQueue(std::move(name), supported_categories), metal_handle_([queue retain]) {}

MetalCommandQueue::~MetalCommandQueue() { [metal_handle_ release]; }

Status MetalCommandQueue::Submit(absl::Span<const SubmissionBatch> batches) {
  IREE_TRACE_SCOPE0("MetalCommandQueue::Submit");
  for (const auto& batch : batches) {
    @autoreleasepool {
      // Wait for semaphores blocking this batch.
      if (!batch.wait_semaphores.empty()) {
        id<MTLCommandBuffer> wait_buffer = [metal_handle_ commandBufferWithUnretainedReferences];
        for (const auto& semaphore : batch.wait_semaphores) {
          auto* event = static_cast<MetalSharedEvent*>(semaphore.semaphore);
          [wait_buffer encodeWaitForEvent:event->handle() value:semaphore.value];
        }
        [wait_buffer commit];
      }

      // Commit command buffers to the queue.
      for (const auto* command_buffer : batch.command_buffers) {
        const auto* cmdbuf = static_cast<const MetalCommandBuffer*>(command_buffer);
        [cmdbuf->handle() commit];
      }

      // Signal semaphores advanced by this batch.
      if (!batch.signal_semaphores.empty()) {
        id<MTLCommandBuffer> signal_buffer = [metal_handle_ commandBufferWithUnretainedReferences];
        for (const auto& semaphore : batch.signal_semaphores) {
          auto* event = static_cast<MetalSharedEvent*>(semaphore.semaphore);
          [signal_buffer encodeSignalEvent:event->handle() value:semaphore.value];
        }
        [signal_buffer commit];
      }
    }
  }
  return OkStatus();
}

Status MetalCommandQueue::WaitIdle(Time deadline_ns) {
  IREE_TRACE_SCOPE0("MetalCommandQueue::WaitIdle");

  dispatch_time_t timeout = DeadlineToDispatchTime(deadline_ns);

  // Submit an empty command buffer and wait it to complete. That will indiate all previous work
  // are done.
  @autoreleasepool {
    id<MTLCommandBuffer> comand_buffer = [metal_handle_ commandBufferWithUnretainedReferences];
    __block dispatch_semaphore_t work_done = dispatch_semaphore_create(0);
    [comand_buffer addCompletedHandler:^(id<MTLCommandBuffer>) {
      dispatch_semaphore_signal(work_done);
    }];
    [comand_buffer commit];
    long timed_out = dispatch_semaphore_wait(work_done, timeout);
    dispatch_release(work_done);
    if (timed_out) {
      return DeadlineExceededErrorBuilder(IREE_LOC)
             << "Deadline exceeded waiting for dispatch_semaphore_t";
    }
    return OkStatus();
  }
}

}  // namespace metal
}  // namespace hal
}  // namespace iree
