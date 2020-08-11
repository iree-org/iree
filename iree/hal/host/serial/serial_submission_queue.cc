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

#include "iree/hal/host/serial/serial_submission_queue.h"

#include <atomic>
#include <cstdint>

#include "absl/synchronization/mutex.h"
#include "iree/base/status.h"
#include "iree/base/tracing.h"

namespace iree {
namespace hal {
namespace host {

SerialSubmissionQueue::SerialSubmissionQueue() = default;

SerialSubmissionQueue::~SerialSubmissionQueue() = default;

StatusOr<bool> SerialSubmissionQueue::CheckBatchReady(
    const PendingBatch& batch) const {
  for (auto& wait_point : batch.wait_semaphores) {
    auto* semaphore = reinterpret_cast<CondVarSemaphore*>(wait_point.semaphore);
    IREE_ASSIGN_OR_RETURN(uint64_t value, semaphore->Query());
    if (value < wait_point.value) {
      return false;
    }
  }
  return true;
}

Status SerialSubmissionQueue::Enqueue(
    absl::Span<const SubmissionBatch> batches) {
  IREE_TRACE_SCOPE0("SerialSubmissionQueue::Enqueue");

  if (has_shutdown_) {
    return FailedPreconditionErrorBuilder(IREE_LOC)
           << "Cannot enqueue new submissions; queue is exiting";
  } else if (!permanent_error_.ok()) {
    return permanent_error_;
  }

  // Add to list in submission order.
  auto submission = absl::make_unique<Submission>();
  submission->pending_batches.resize(batches.size());
  for (int i = 0; i < batches.size(); ++i) {
    submission->pending_batches[i] = PendingBatch{
        {batches[i].wait_semaphores.begin(), batches[i].wait_semaphores.end()},
        {batches[i].command_buffers.begin(), batches[i].command_buffers.end()},
        {batches[i].signal_semaphores.begin(),
         batches[i].signal_semaphores.end()},
    };
  }
  list_.push_back(std::move(submission));

  return OkStatus();
}

Status SerialSubmissionQueue::ProcessBatches(ExecuteFn execute_fn) {
  IREE_TRACE_SCOPE0("SerialSubmissionQueue::ProcessBatches");

  if (!permanent_error_.ok()) {
    // Sticky failure state.
    return permanent_error_;
  }

  // Repeated try to run things until we quiesce or are blocked.
  while (permanent_error_.ok() && !list_.empty()) {
    // NOTE: to support re-entrancy where |execute_fn| may modify the submission
    // list we need to always start from the beginning. If we wanted we could
    // track a list of ready submissions however that's a lot of bookkeeping and
    // the list is usually short.
    auto* submission = list_.front();
    for (int i = 0; i < submission->pending_batches.size(); ++i) {
      auto& batch = submission->pending_batches[i];
      auto wait_status_or = CheckBatchReady(batch);
      if (!wait_status_or.ok()) {
        // Batch dependencies failed; set the permanent error flag and abort
        // so we don't try to process anything else.
        permanent_error_ = std::move(wait_status_or).status();
        CompleteSubmission(submission, permanent_error_);
        FailAllPending(permanent_error_);
        return permanent_error_;
      } else if (wait_status_or.ok() && !wait_status_or.value()) {
        // To preserve submission order we bail if we encounter a batch that
        // is not ready and wait for something to become ready before pumping
        // again.
        // Note that if we were properly threading here we would potentially
        // be evaluating this while previous batches were still processing
        // but for now we do everything serially.
        return OkStatus();
      }

      // Batch can run! Process now and remove it from the list so we don't
      // try to run it again.
      auto batch_status = ProcessBatch(batch, execute_fn);
      if (!batch_status.ok()) {
        // Batch failed; set the permanent error flag and abort so we don't
        // try to process anything else.
        permanent_error_ = Status(batch_status);
        CompleteSubmission(submission, batch_status);
        FailAllPending(permanent_error_);
        return permanent_error_;
      }
      submission->pending_batches.erase(submission->pending_batches.begin() +
                                        i);

      // Batch succeeded. Since we want to preserve submission order we'll
      // break out of the loop and try from the first submission again.
      if (submission->pending_batches.empty()) {
        // All work for this submission completed successfully. Signal the
        // semaphore and remove the submission from the list.
        CompleteSubmission(submission, OkStatus());
        break;
      }
    }
  }

  return OkStatus();
}

Status SerialSubmissionQueue::ProcessBatch(const PendingBatch& batch,
                                           const ExecuteFn& execute_fn) {
  IREE_TRACE_SCOPE0("SerialSubmissionQueue::ProcessBatch");

  // NOTE: the precondition is that the batch is ready to execute so we don't
  // need to check the wait semaphores here.

  // Let the caller handle execution of the command buffers.
  IREE_RETURN_IF_ERROR(execute_fn(batch.command_buffers));

  // Signal all semaphores to allow them to unblock waiters.
  for (auto& signal_point : batch.signal_semaphores) {
    auto* semaphore =
        reinterpret_cast<CondVarSemaphore*>(signal_point.semaphore);
    IREE_RETURN_IF_ERROR(semaphore->Signal(signal_point.value));
  }

  return OkStatus();
}

void SerialSubmissionQueue::CompleteSubmission(Submission* submission,
                                               Status status) {
  IREE_TRACE_SCOPE0("SerialSubmissionQueue::CompleteSubmission");

  if (status.ok() && !submission->pending_batches.empty()) {
    // Completed with work remaining? Cause a failure.
    status = FailedPreconditionErrorBuilder(IREE_LOC)
             << "Submission ended prior to completion of all batches";
  }
  if (!status.ok()) {
    // Fail all pending batch semaphores that we would have signaled.
    for (auto& batch : submission->pending_batches) {
      for (auto& signal_point : batch.signal_semaphores) {
        auto* semaphore =
            reinterpret_cast<CondVarSemaphore*>(signal_point.semaphore);
        semaphore->Fail(status);
      }
    }
    submission->pending_batches.clear();
  }

  list_.take(submission).reset();
}

void SerialSubmissionQueue::FailAllPending(Status status) {
  IREE_TRACE_SCOPE0("SerialSubmissionQueue::FailAllPending");
  while (!list_.empty()) {
    CompleteSubmission(list_.front(), status);
  }
}

void SerialSubmissionQueue::SignalShutdown() {
  IREE_TRACE_SCOPE0("SerialSubmissionQueue::SignalShutdown");
  has_shutdown_ = true;
}

}  // namespace host
}  // namespace hal
}  // namespace iree
