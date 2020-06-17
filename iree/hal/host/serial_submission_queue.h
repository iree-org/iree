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

#ifndef IREE_HAL_HOST_SERIAL_SUBMISSION_QUEUE_H_
#define IREE_HAL_HOST_SERIAL_SUBMISSION_QUEUE_H_

#include "absl/base/thread_annotations.h"
#include "absl/container/inlined_vector.h"
#include "absl/synchronization/mutex.h"
#include "iree/base/intrusive_list.h"
#include "iree/base/status.h"
#include "iree/hal/command_queue.h"
#include "iree/hal/host/host_semaphore.h"

namespace iree {
namespace hal {

// A queue managing CommandQueue submissions that uses host-local
// synchronization primitives. Evaluates submission order by respecting the
// wait and signal semaphores defined per batch and notifies semaphores upon
// submission completion.
//
// Note that it's possible for HAL users to deadlock themselves; we don't try to
// avoid that as in device backends it may not be possible and we want to have
// some kind of warning in the host implementation that TSAN can catch.
//
// Thread-compatible. Const methods may be called from any thread.
class SerialSubmissionQueue final {
 public:
  using ExecuteFn =
      std::function<Status(absl::Span<CommandBuffer* const> command_buffers)>;

  SerialSubmissionQueue();
  ~SerialSubmissionQueue();

  // Returns true if the queue is currently empty.
  bool empty() const { return list_.empty(); }
  // Returns true if SignalShutdown has been called.
  bool has_shutdown() const { return has_shutdown_; }
  // The sticky error status, if an error has occurred.
  Status permanent_error() const { return permanent_error_; }

  // Enqueues a new submission.
  // No work will be performed until Process is called.
  Status Enqueue(absl::Span<const SubmissionBatch> batches);

  // Processes all ready batches using the provided |execute_fn|.
  // The function may be called several times if new batches become ready due to
  // prior batches in the sequence completing during processing.
  //
  // Returns any errors returned by |execute_fn| (which will be the same as
  // permanent_error()). When an error occurs all in-flight submissions are
  // aborted, the permanent_error() is set, and the queue is shutdown.
  Status ProcessBatches(ExecuteFn execute_fn);

  // Marks the queue as having shutdown. All pending submissions will be allowed
  // to complete but future enqueues will fail.
  void SignalShutdown();

 private:
  // A submitted command buffer batch and its synchronization information.
  struct PendingBatch {
    absl::InlinedVector<SemaphoreValue, 4> wait_semaphores;
    absl::InlinedVector<CommandBuffer*, 4> command_buffers;
    absl::InlinedVector<SemaphoreValue, 4> signal_semaphores;
  };
  struct Submission : public IntrusiveLinkBase<void> {
    absl::InlinedVector<PendingBatch, 4> pending_batches;
  };

  // Returns true if all wait semaphores in the |batch| are signaled.
  // If one or more of the wait semaphores have failed then returns a status
  // from one of them arbitrarily.
  StatusOr<bool> CheckBatchReady(const PendingBatch& batch) const;

  // Processes a batch by resetting semaphores, dispatching the command buffers
  // to the specified |execute_fn|, and signaling semaphores.
  //
  // Preconditions: CheckBatchReady(batch) == true
  Status ProcessBatch(const PendingBatch& batch, const ExecuteFn& execute_fn);

  // Completes a submission. Assumes that all batches have had their semaphores
  // signaled and that any remaining here will need to be signaled for failure.
  void CompleteSubmission(Submission* submission, Status status);

  // Fails all pending submissions with the given status.
  // Errors that occur during this process are silently ignored.
  void FailAllPending(Status status);

  // True to exit the thread after all submissions complete.
  bool has_shutdown_ = false;

  // A sticky error that is set on the first failed submit. All future
  // submissions will be skipped except for semaphores, which will receive this
  // error.
  Status permanent_error_;

  // Pending submissions in submission order.
  // Note that we may evaluate batches within the list out of order.
  IntrusiveList<std::unique_ptr<Submission>> list_;
};

}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_HOST_SERIAL_SUBMISSION_QUEUE_H_
