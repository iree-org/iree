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

#ifndef IREE_HAL_COMMAND_QUEUE_H_
#define IREE_HAL_COMMAND_QUEUE_H_

#include <cstdint>
#include <string>

#include "absl/types/span.h"
#include "iree/base/bitfield.h"
#include "iree/base/status.h"
#include "iree/base/time.h"
#include "iree/hal/command_buffer.h"
#include "iree/hal/semaphore.h"

namespace iree {
namespace hal {

// A batch of command buffers with synchronization information for submission.
struct SubmissionBatch {
  // A set of semaphores that must have their payload values meet or exceed the
  // specified values prior to any command buffer within this batch executing.
  absl::Span<const SemaphoreValue> wait_semaphores;

  // Command buffers that will execute in this batch.
  // The command buffers will begin execution in order but may complete out of
  // order.
  absl::Span<CommandBuffer* const> command_buffers;

  // Semaphores to signal after execution of all command buffers complete.
  // Semaphore playloads will be set to the maximum of the specified payload or
  // their current payload.
  absl::Span<const SemaphoreValue> signal_semaphores;
};

// Asynchronous command execution queue.
//
// CommandQueues may capture device status at Semaphore barriers, including
// information about device state such as thermal throttling. This information
// is a snapshot of the state at the time the semaphore was signaled and not
// necessarily live at the time of the application query.
//
// Command queues are thread-safe and submissions may occur from multiple
// threads.
class CommandQueue {
 public:
  virtual ~CommandQueue() = default;

  // Name of the queue used for logging purposes.
  // Try to keep at 4 characters total for prettier logging.
  const std::string& name() const { return name_; }

  // Capabilities of the command queue.
  CommandCategoryBitfield supported_categories() const {
    return supported_categories_;
  }

  // Whether this queue may be used for transfer commands.
  bool can_transfer() const {
    return AllBitsSet(supported_categories_, CommandCategory::kTransfer);
  }

  // Whether this queue may be used for dispatch commands.
  bool can_dispatch() const {
    return AllBitsSet(supported_categories_, CommandCategory::kDispatch);
  }

  // Submits one or more command batches for execution on the queue.
  virtual Status Submit(absl::Span<const SubmissionBatch> batches) = 0;
  inline Status Submit(const SubmissionBatch& batch) {
    return Submit(absl::MakeConstSpan(&batch, 1));
  }

  // Blocks until all outstanding requests have been completed.
  // This is equivalent to having waited on all outstanding semaphores.
  // Implicitly calls Flush to ensure delayed requests are scheduled.
  //
  // If the command queue has encountered an error during submission at any
  // point it will be returned here (repeatedly).
  virtual Status WaitIdle(Time deadline_ns) = 0;
  inline Status WaitIdle(Duration timeout_ns) {
    return WaitIdle(RelativeTimeoutToDeadlineNanos(timeout_ns));
  }
  inline Status WaitIdle() { return WaitIdle(InfiniteFuture()); }

 protected:
  CommandQueue(std::string name, CommandCategoryBitfield supported_categories)
      : name_(std::move(name)), supported_categories_(supported_categories) {}

  const std::string name_;
  const CommandCategoryBitfield supported_categories_;
};

}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_COMMAND_QUEUE_H_
