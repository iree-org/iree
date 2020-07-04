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

#include "iree/hal/host/serial/async_command_queue.h"

#include "absl/base/thread_annotations.h"
#include "iree/base/status.h"
#include "iree/base/tracing.h"

namespace iree {
namespace hal {
namespace host {

AsyncCommandQueue::AsyncCommandQueue(std::unique_ptr<CommandQueue> target_queue)
    : CommandQueue(target_queue->name(), target_queue->supported_categories()),
      target_queue_(std::move(target_queue)) {
  IREE_TRACE_SCOPE0("AsyncCommandQueue::ctor");
  thread_ = std::thread([this]() { ThreadMain(); });
}

AsyncCommandQueue::~AsyncCommandQueue() {
  IREE_TRACE_SCOPE0("AsyncCommandQueue::dtor");
  {
    // Signal to thread that we want to stop. Note that the thread may have
    // already been stopped and that's ok (as we'll Join right away).
    // The thread will finish processing any queued submissions.
    absl::MutexLock lock(&submission_mutex_);
    submission_queue_.SignalShutdown();
  }
  thread_.join();

  // Ensure we shut down OK.
  {
    absl::MutexLock lock(&submission_mutex_);
    CHECK(submission_queue_.empty())
        << "Dirty shutdown of async queue (unexpected thread exit?)";
  }
}

void AsyncCommandQueue::ThreadMain() {
  IREE_TRACE_SET_THREAD_NAME(target_queue_->name().c_str());

  bool is_exiting = false;
  while (!is_exiting) {
    // Block until we are either requested to exit or there are pending
    // submissions.
    submission_mutex_.Lock();
    submission_mutex_.Await(absl::Condition(
        +[](SerialSubmissionQueue* queue) {
          return queue->has_shutdown() || !queue->empty();
        },
        &submission_queue_));
    if (!submission_queue_.empty()) {
      // Run all ready submissions (this may be called many times).
      submission_mutex_.AssertHeld();
      submission_queue_
          .ProcessBatches(
              [this](absl::Span<CommandBuffer* const> command_buffers)
                  ABSL_EXCLUSIVE_LOCKS_REQUIRED(submission_mutex_) {
                    // Release the lock while we perform the processing so that
                    // other threads can submit more work.
                    submission_mutex_.AssertHeld();
                    submission_mutex_.Unlock();

                    // Relay the command buffers to the target queue.
                    // Since we are taking care of all synchronization they
                    // don't need any waiters or semaphores.
                    auto status =
                        target_queue_->Submit({{}, command_buffers, {}});

                    // Take back the lock so we can manipulate the queue safely.
                    submission_mutex_.Lock();
                    submission_mutex_.AssertHeld();

                    return status;
                  })
          .IgnoreError();
      submission_mutex_.AssertHeld();
    }
    if (submission_queue_.has_shutdown()) {
      // Exit when there are no more submissions to process and an exit was
      // requested (or we errored out).
      is_exiting = true;
    }
    submission_mutex_.Unlock();
  }
}

Status AsyncCommandQueue::Submit(absl::Span<const SubmissionBatch> batches) {
  IREE_TRACE_SCOPE0("AsyncCommandQueue::Submit");
  absl::MutexLock lock(&submission_mutex_);
  return submission_queue_.Enqueue(batches);
}

Status AsyncCommandQueue::WaitIdle(absl::Time deadline) {
  IREE_TRACE_SCOPE0("AsyncCommandQueue::WaitIdle");

  // Wait until the deadline, the thread exits, or there are no more pending
  // submissions.
  absl::MutexLock lock(&submission_mutex_);
  if (!submission_mutex_.AwaitWithDeadline(
          absl::Condition(
              +[](SerialSubmissionQueue* queue) {
                return queue->empty() || !queue->permanent_error().ok();
              },
              &submission_queue_),
          deadline)) {
    return DeadlineExceededErrorBuilder(IREE_LOC)
           << "Deadline exceeded waiting for submission thread to go idle";
  }
  return submission_queue_.permanent_error();
}

}  // namespace host
}  // namespace hal
}  // namespace iree
