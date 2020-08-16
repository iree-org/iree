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

#ifndef IREE_HAL_VULKAN_SERIALIZING_COMMAND_QUEUE_H_
#define IREE_HAL_VULKAN_SERIALIZING_COMMAND_QUEUE_H_

#include <vulkan/vulkan.h>

#include <memory>
#include <string>

#include "absl/base/thread_annotations.h"
#include "absl/container/inlined_vector.h"
#include "absl/synchronization/mutex.h"
#include "iree/base/intrusive_list.h"
#include "iree/base/ref_ptr.h"
#include "iree/base/status.h"
#include "iree/base/time.h"
#include "iree/hal/command_buffer.h"
#include "iree/hal/command_queue.h"
#include "iree/hal/vulkan/dynamic_symbols.h"
#include "iree/hal/vulkan/handle_util.h"
#include "iree/hal/vulkan/timepoint_util.h"

namespace iree {
namespace hal {
namespace vulkan {

// A command queue that potentially defers and serializes command buffer
// submission to the GPU.
//
// This command queue is designed to be used together with emulated timeline
// semaphores. Timeline semaphores can follow wait-before-signal submission
// order but binary `VkSemaphore` cannot. So when emulating timeline semaphores
// with binary `VkSemaphore`s and `VkFence`s, we need to make sure no
// wait-before-signal submission order occur for binary `VkSemaphore`s. The way
// to enforce that is to defer the submission until we can be certain that the
// `VkSemaphore`s emulating time points in the timeline are all *submitted* to
// the GPU.
class SerializingCommandQueue final : public CommandQueue {
 public:
  SerializingCommandQueue(std::string name,
                          CommandCategoryBitfield supported_categories,
                          const ref_ptr<VkDeviceHandle>& logical_device,
                          const ref_ptr<TimePointFencePool>& fence_pool,
                          VkQueue queue);
  ~SerializingCommandQueue() override;

  const ref_ptr<DynamicSymbols>& syms() const {
    return logical_device_->syms();
  }

  Status Submit(absl::Span<const SubmissionBatch> batches) override;

  Status WaitIdle(Time deadline_ns) override;

  // Releases all deferred submissions ready to submit to the GPU.
  Status AdvanceQueueSubmission();

  // Aborts all deferred submissions and waits for submitted work to complete.
  void AbortQueueSubmission();

 private:
  struct PendingBatch {
    absl::InlinedVector<SemaphoreValue, 4> wait_semaphores;
    absl::InlinedVector<CommandBuffer*, 4> command_buffers;
    absl::InlinedVector<SemaphoreValue, 4> signal_semaphores;
  };
  // A submission batch together with the fence to singal its status.
  struct FencedSubmission : public IntrusiveLinkBase<void> {
    PendingBatch batch;
    ref_ptr<TimePointFence> fence;
  };

  // Processes deferred submissions in this queue and returns whether there are
  // new workload submitted to the GPU if no errors happen.
  StatusOr<bool> ProcessDeferredSubmissions()
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  ref_ptr<VkDeviceHandle> logical_device_;

  ref_ptr<TimePointFencePool> fence_pool_;

  mutable absl::Mutex mutex_;

  // A list of fences that are submitted to GPU.
  absl::InlinedVector<ref_ptr<TimePointFence>, 4> pending_fences_
      ABSL_GUARDED_BY(mutex_);
  // A list of deferred submissions that haven't been submitted to GPU.
  IntrusiveList<std::unique_ptr<FencedSubmission>> deferred_submissions_
      ABSL_GUARDED_BY(mutex_);

  // VkQueue needs to be externally synchronized.
  VkQueue queue_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace vulkan
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_VULKAN_SERIALIZING_COMMAND_QUEUE_H_
