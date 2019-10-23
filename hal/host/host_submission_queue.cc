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

#include "hal/host/host_submission_queue.h"

#include <atomic>
#include <cstdint>

#include "absl/synchronization/mutex.h"
#include "base/status.h"
#include "base/tracing.h"

namespace iree {
namespace hal {

HostBinarySemaphore::HostBinarySemaphore(bool initial_value) {
  State state = {0};
  state.signaled = initial_value ? 1 : 0;
  state_ = state;
}

bool HostBinarySemaphore::is_signaled() const {
  return state_.load(std::memory_order_acquire).signaled == 1;
}

Status HostBinarySemaphore::BeginSignaling() {
  State old_state = state_.load(std::memory_order_acquire);
  if (old_state.signal_pending != 0) {
    return FailedPreconditionErrorBuilder(IREE_LOC)
           << "A signal operation on a binary semaphore is already pending";
  }
  State new_state = old_state;
  new_state.signal_pending = 1;
  state_.compare_exchange_strong(old_state, new_state);
  return OkStatus();
}

Status HostBinarySemaphore::EndSignaling() {
  State old_state = state_.load(std::memory_order_acquire);
  DCHECK_EQ(old_state.signal_pending, 1)
      << "A signal operation on a binary semaphore was not pending";
  if (old_state.signaled != 0) {
    return FailedPreconditionErrorBuilder(IREE_LOC)
           << "A binary semaphore cannot be signaled multiple times";
  }
  State new_state = old_state;
  new_state.signal_pending = 0;
  new_state.signaled = 1;
  state_.compare_exchange_strong(old_state, new_state);
  return OkStatus();
}

Status HostBinarySemaphore::BeginWaiting() {
  State old_state = state_.load(std::memory_order_acquire);
  if (old_state.wait_pending != 0) {
    return FailedPreconditionErrorBuilder(IREE_LOC)
           << "A wait operation on a binary semaphore is already pending";
  }
  State new_state = old_state;
  new_state.wait_pending = 1;
  state_.compare_exchange_strong(old_state, new_state);
  return OkStatus();
}

Status HostBinarySemaphore::EndWaiting() {
  State old_state = state_.load(std::memory_order_acquire);
  DCHECK_EQ(old_state.wait_pending, 1)
      << "A wait operation on a binary semaphore was not pending";
  if (old_state.signaled != 1) {
    return FailedPreconditionErrorBuilder(IREE_LOC)
           << "A binary semaphore cannot be reset multiple times";
  }
  State new_state = old_state;
  new_state.wait_pending = 0;
  new_state.signaled = 0;
  state_.compare_exchange_strong(old_state, new_state);
  return OkStatus();
}

HostSubmissionQueue::HostSubmissionQueue() = default;

HostSubmissionQueue::~HostSubmissionQueue() = default;

bool HostSubmissionQueue::IsBatchReady(const PendingBatch& batch) const {
  for (auto& wait_point : batch.wait_semaphores) {
    if (wait_point.index() == 0) {
      auto* binary_semaphore =
          reinterpret_cast<HostBinarySemaphore*>(absl::get<0>(wait_point));
      if (!binary_semaphore->is_signaled()) {
        return false;
      }
    } else {
      // TODO(b/140141417): implement timeline semaphores.
      return false;
    }
  }
  return true;
}

Status HostSubmissionQueue::Enqueue(absl::Span<const SubmissionBatch> batches,
                                    FenceValue fence) {
  IREE_TRACE_SCOPE0("HostSubmissionQueue::Enqueue");

  if (has_shutdown_) {
    return FailedPreconditionErrorBuilder(IREE_LOC)
           << "Cannot enqueue new submissions; queue is exiting";
  } else if (!permanent_error_.ok()) {
    return permanent_error_;
  }

  // Verify waiting/signaling behavior on semaphores and prepare them all.
  // We need to track this to ensure that we are modeling the Vulkan behavior
  // and are consistent across HAL implementations.
  for (auto& batch : batches) {
    for (auto& semaphore_value : batch.wait_semaphores) {
      if (semaphore_value.index() == 0) {
        auto* binary_semaphore = reinterpret_cast<HostBinarySemaphore*>(
            absl::get<0>(semaphore_value));
        RETURN_IF_ERROR(binary_semaphore->BeginWaiting());
      } else {
        // TODO(b/140141417): implement timeline semaphores.
        return UnimplementedErrorBuilder(IREE_LOC) << "Timeline semaphores NYI";
      }
    }
    for (auto& semaphore_value : batch.signal_semaphores) {
      if (semaphore_value.index() == 0) {
        auto* binary_semaphore = reinterpret_cast<HostBinarySemaphore*>(
            absl::get<0>(semaphore_value));
        RETURN_IF_ERROR(binary_semaphore->BeginSignaling());
      } else {
        // TODO(b/140141417): implement timeline semaphores.
        return UnimplementedErrorBuilder(IREE_LOC) << "Timeline semaphores NYI";
      }
    }
  }

  // Add to list - order does not matter as Process evaluates semaphores.
  auto submission = absl::make_unique<Submission>();
  submission->fence = std::move(fence);
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

Status HostSubmissionQueue::ProcessBatches(ExecuteFn execute_fn) {
  IREE_TRACE_SCOPE0("HostSubmissionQueue::ProcessBatches");

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
    bool restart_iteration = false;
    for (auto* submission : list_) {
      for (int i = 0; i < submission->pending_batches.size(); ++i) {
        auto& batch = submission->pending_batches[i];
        if (!IsBatchReady(batch)) {
          // Try the next batch in the submission until we find one that is
          // ready. If none are ready we'll return to the caller.
          continue;
        }

        // Batch can run! Process now and remove it from the list so we don't
        // try to run it again.
        auto batch_status = ProcessBatch(batch, execute_fn);
        submission->pending_batches.erase(submission->pending_batches.begin() +
                                          i);
        if (batch_status.ok()) {
          // Batch succeeded. Since we want to preserve submission order we'll
          // break out of the loop and try from the first submission again.
          if (submission->pending_batches.empty()) {
            // All work for this submission completed successfully. Signal the
            // fence and remove the submission from the list.
            RETURN_IF_ERROR(CompleteSubmission(submission, OkStatus()));
            list_.take(submission).reset();
          }
        } else {
          // Batch failed; set the permanent error flag and abort so we don't
          // try to process anything else.
          permanent_error_ = batch_status;
          RETURN_IF_ERROR(CompleteSubmission(submission, batch_status));
          list_.take(submission).reset();
        }
        restart_iteration = true;
        break;
      }
      if (restart_iteration) break;
    }
  }

  if (!permanent_error_.ok()) {
    // If the sticky error got set while processing we need to abort all
    // remaining submissions (simulating a device loss).
    FailAllPending(permanent_error_);
    return permanent_error_;
  }

  return OkStatus();
}

Status HostSubmissionQueue::ProcessBatch(const PendingBatch& batch,
                                         const ExecuteFn& execute_fn) {
  IREE_TRACE_SCOPE0("HostSubmissionQueue::ProcessBatch");

  // Complete the waits on all semaphores and reset them.
  for (auto& semaphore_value : batch.wait_semaphores) {
    if (semaphore_value.index() == 0) {
      auto* binary_semaphore =
          reinterpret_cast<HostBinarySemaphore*>(absl::get<0>(semaphore_value));
      RETURN_IF_ERROR(binary_semaphore->EndWaiting());
    } else {
      // TODO(b/140141417): implement timeline semaphores.
      return UnimplementedErrorBuilder(IREE_LOC) << "Timeline semaphores NYI";
    }
  }

  // Let the caller handle execution of the command buffers.
  RETURN_IF_ERROR(execute_fn(batch.command_buffers));

  // Signal all semaphores to allow them to unblock waiters.
  for (auto& semaphore_value : batch.signal_semaphores) {
    if (semaphore_value.index() == 0) {
      auto* binary_semaphore =
          reinterpret_cast<HostBinarySemaphore*>(absl::get<0>(semaphore_value));
      RETURN_IF_ERROR(binary_semaphore->EndSignaling());
    } else {
      // TODO(b/140141417): implement timeline semaphores.
      return UnimplementedErrorBuilder(IREE_LOC) << "Timeline semaphores NYI";
    }
  }

  return OkStatus();
}

Status HostSubmissionQueue::CompleteSubmission(Submission* submission,
                                               Status status) {
  IREE_TRACE_SCOPE0("HostSubmissionQueue::CompleteSubmission");

  // It's safe to drop any remaining batches - their semaphores will never be
  // signaled but that's fine as we should be the only thing relying on them.
  submission->pending_batches.clear();

  // Signal the fence.
  auto* fence = static_cast<HostFence*>(submission->fence.first);
  if (status.ok()) {
    RETURN_IF_ERROR(fence->Signal(submission->fence.second));
  } else {
    RETURN_IF_ERROR(fence->Fail(std::move(status)));
  }

  return OkStatus();
}

void HostSubmissionQueue::FailAllPending(Status status) {
  IREE_TRACE_SCOPE0("HostSubmissionQueue::FailAllPending");
  while (!list_.empty()) {
    auto submission = list_.take(list_.front());
    CompleteSubmission(submission.get(), status).IgnoreError();
    submission.reset();
  }
}

void HostSubmissionQueue::SignalShutdown() {
  IREE_TRACE_SCOPE0("HostSubmissionQueue::SignalShutdown");
  has_shutdown_ = true;
}

}  // namespace hal
}  // namespace iree
