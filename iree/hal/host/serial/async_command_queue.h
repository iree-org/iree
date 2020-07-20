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

#ifndef IREE_HAL_HOST_SERIAL_ASYNC_COMMAND_QUEUE_H_
#define IREE_HAL_HOST_SERIAL_ASYNC_COMMAND_QUEUE_H_

#include <memory>
#include <thread>  // NOLINT

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "iree/hal/command_queue.h"
#include "iree/hal/host/serial/serial_submission_queue.h"
#include "iree/hal/semaphore.h"

namespace iree {
namespace hal {
namespace host {

// Asynchronous command queue wrapper.
// This creates a single thread to perform all CommandQueue operations. Any
// submitted CommandBuffer is dispatched in FIFO order on the queue thread
// against the provided |target_queue|.
//
// Target queues will receive submissions containing only command buffers as
// all semaphore synchronization is handled by the wrapper. Semaphores will also
// be omitted and code should safely handle nullptr.
//
// AsyncCommandQueue (as with CommandQueue) is thread-safe. Multiple threads
// may submit command buffers concurrently, though the order of execution in
// such a case depends entirely on the synchronization primitives provided.
class AsyncCommandQueue final : public CommandQueue {
 public:
  explicit AsyncCommandQueue(std::unique_ptr<CommandQueue> target_queue);
  ~AsyncCommandQueue() override;

  Status Submit(absl::Span<const SubmissionBatch> batches) override;

  Status WaitIdle(Time deadline_ns) override;

 private:
  // Thread entry point for the async worker thread.
  // Waits for submissions to be queued up and processes them eagerly.
  void ThreadMain();

  // CommandQueue that the async queue relays submissions into.
  std::unique_ptr<CommandQueue> target_queue_;

  // Thread that runs the ThreadMain() function and processes submissions.
  std::thread thread_;

  // Queue that manages submission ordering.
  mutable absl::Mutex submission_mutex_;
  SerialSubmissionQueue submission_queue_ ABSL_GUARDED_BY(submission_mutex_);
};

}  // namespace host
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_HOST_SERIAL_ASYNC_COMMAND_QUEUE_H_
