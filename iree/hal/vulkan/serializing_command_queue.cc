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

#include "iree/hal/vulkan/serializing_command_queue.h"

#include <memory>

#include "absl/types/span.h"
#include "iree/base/api.h"
#include "iree/base/memory.h"
#include "iree/base/tracing.h"
#include "iree/hal/command_buffer.h"
#include "iree/hal/command_queue.h"
#include "iree/hal/semaphore.h"
#include "iree/hal/vulkan/direct_command_buffer.h"
#include "iree/hal/vulkan/emulated_timeline_semaphore.h"
#include "iree/hal/vulkan/status_util.h"

namespace iree {
namespace hal {
namespace vulkan {

namespace {

// Tries to prepare all necessary binary `VKSemaphore`s for emulating the time
// points as specified in the given submission |batch_wait_semaphores| and
// |batch_signal_semaphores|, then returns true if possible so that the
// batch is ready to be submitted to GPU.
// |wait_semaphores| and |signal_semaphores| will be filled with the binary
// `VkSemaphores` on success.
StatusOr<bool> TryToPrepareSemaphores(
    const absl::InlinedVector<SemaphoreValue, 4>& batch_wait_semaphores,
    const absl::InlinedVector<SemaphoreValue, 4>& batch_signal_semaphores,
    const ref_ptr<TimePointFence>& batch_fence,
    absl::InlinedVector<VkSemaphore, 4>* wait_semaphores,
    absl::InlinedVector<VkSemaphore, 4>* signal_semaphores) {
  IREE_TRACE_SCOPE0("TryToPrepareSemaphores");

  wait_semaphores->clear();
  for (const auto& timeline_semaphore : batch_wait_semaphores) {
    // Query first to progress this timeline semaphore to the furthest.
    IREE_ASSIGN_OR_RETURN(auto signaled_value,
                          timeline_semaphore.semaphore->Query());

    // If it's already signaled to a value greater than we require here,
    // we can just ignore this semaphore now.
    if (signaled_value >= timeline_semaphore.value) continue;

    // SerializingCommandQueue only works with EmulatedTimelineSemaphore.
    auto* emulated_semaphore =
        static_cast<EmulatedTimelineSemaphore*>(timeline_semaphore.semaphore);

    // Otherwise try to get a binary semaphore for this time point so that
    // we can wait on.
    VkSemaphore binary_semaphore = emulated_semaphore->GetWaitSemaphore(
        timeline_semaphore.value, batch_fence);

    if (binary_semaphore == VK_NULL_HANDLE) {
      // We cannot wait on this time point yet: there are no previous semaphores
      // submitted to the GPU  that can signal a value greater than what's
      // desired here.

      // Cancel the wait so others may make progress.
      for (VkSemaphore semaphore : *wait_semaphores) {
        IREE_RETURN_IF_ERROR(
            emulated_semaphore->CancelWaitSemaphore(semaphore));
      }

      // This batch cannot be submitted to GPU yet.
      return false;
    }

    wait_semaphores->push_back(binary_semaphore);
  }

  // We've collected all necessary binary semaphores for each timeline we need
  // to wait on. Now prepare binary semaphores for signaling.
  signal_semaphores->clear();
  for (const auto& timeline_semaphore : batch_signal_semaphores) {
    // SerializingCommandQueue only works with EmulatedTimelineSemaphore.
    auto* emulated_semaphore =
        static_cast<EmulatedTimelineSemaphore*>(timeline_semaphore.semaphore);

    IREE_ASSIGN_OR_RETURN(auto binary_semaphore,
                          emulated_semaphore->GetSignalSemaphore(
                              timeline_semaphore.value, batch_fence));
    signal_semaphores->push_back(binary_semaphore);
  }

  // Good to submit!
  return true;
}

// Prepares `VkSubmitInfo` to submit the given list of |command_buffers| that
// waiting on |wait_semaphores| and signalling |signal_semaphores|. Necessary
// structures are allocated from |arena| and the result `VkSubmitInfo` is
// written to |submit_info|.
void PrepareSubmitInfo(
    const absl::InlinedVector<VkSemaphore, 4>& wait_semaphores,
    absl::Span<CommandBuffer* const> command_buffers,
    const absl::InlinedVector<VkSemaphore, 4>& signal_semaphores,
    VkSubmitInfo* submit_info, Arena* arena) {
  IREE_TRACE_SCOPE0("PrepareSubmitInfo");

  // TODO(benvanik): see if we can go to finer-grained stages.
  // For example, if this was just queue ownership transfers then we can use
  // the pseudo-stage of VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT.
  VkPipelineStageFlags dst_stage_mask =
      VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

  auto wait_semaphore_handles =
      arena->AllocateSpan<VkSemaphore>(wait_semaphores.size());
  auto wait_dst_stage_masks =
      arena->AllocateSpan<VkPipelineStageFlags>(wait_semaphores.size());
  for (int i = 0, e = wait_semaphores.size(); i < e; ++i) {
    wait_semaphore_handles[i] = wait_semaphores[i];
    wait_dst_stage_masks[i] = dst_stage_mask;
  }

  auto signal_semaphore_handles =
      arena->AllocateSpan<VkSemaphore>(signal_semaphores.size());
  for (int i = 0, e = signal_semaphores.size(); i < e; ++i) {
    signal_semaphore_handles[i] = signal_semaphores[i];
  }

  auto command_buffer_handles =
      arena->AllocateSpan<VkCommandBuffer>(command_buffers.size());
  for (int i = 0, e = command_buffers.size(); i < e; ++i) {
    const auto& command_buffer = command_buffers[i];
    auto* direct_command_buffer =
        static_cast<DirectCommandBuffer*>(command_buffer->impl());
    command_buffer_handles[i] = direct_command_buffer->handle();
  }

  submit_info->sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info->pNext = nullptr;
  submit_info->waitSemaphoreCount = wait_semaphore_handles.size();
  submit_info->pWaitSemaphores = wait_semaphore_handles.data();
  submit_info->pWaitDstStageMask = wait_dst_stage_masks.data();
  submit_info->commandBufferCount = command_buffer_handles.size();
  submit_info->pCommandBuffers = command_buffer_handles.data();
  submit_info->signalSemaphoreCount = signal_semaphore_handles.size();
  submit_info->pSignalSemaphores = signal_semaphore_handles.data();
}

}  // namespace

SerializingCommandQueue::SerializingCommandQueue(
    std::string name, CommandCategoryBitfield supported_categories,
    const ref_ptr<VkDeviceHandle>& logical_device,
    const ref_ptr<TimePointFencePool>& fence_pool, VkQueue queue)
    : CommandQueue(std::move(name), supported_categories),
      logical_device_(add_ref(logical_device)),
      fence_pool_(add_ref(fence_pool)),
      queue_(queue) {}

SerializingCommandQueue::~SerializingCommandQueue() {
  IREE_TRACE_SCOPE0("SerializingCommandQueue::dtor");
  absl::MutexLock lock(&mutex_);
  syms()->vkQueueWaitIdle(queue_);
}

Status SerializingCommandQueue::Submit(
    absl::Span<const SubmissionBatch> batches) {
  IREE_TRACE_SCOPE0("SerializingCommandQueue::Submit");

  absl::MutexLock lock(&mutex_);
  for (int i = 0; i < batches.size(); ++i) {
    // Grab a fence for this submission first. This will be used to check the
    // progress of emulated timeline semaphores later.
    IREE_ASSIGN_OR_RETURN(auto fence, fence_pool_->Acquire());
    auto submission = std::make_unique<FencedSubmission>();
    submission->batch = PendingBatch{
        {batches[i].wait_semaphores.begin(), batches[i].wait_semaphores.end()},
        {batches[i].command_buffers.begin(), batches[i].command_buffers.end()},
        {batches[i].signal_semaphores.begin(),
         batches[i].signal_semaphores.end()}};
    submission->fence = std::move(fence);
    deferred_submissions_.push_back(std::move(submission));
  }

  return ProcessDeferredSubmissions().status();
}

StatusOr<bool> SerializingCommandQueue::ProcessDeferredSubmissions() {
  IREE_TRACE_SCOPE0("SerializingCommandQueue::ProcessDeferredSubmissions");

  // Prepare `VkSubmitInfo`s for all submissions we are able to submit.

  // Note that we must keep all arrays referenced alive until submission
  // completes and since there are a bunch of them we use an arena.
  Arena arena(4 * 1024);

  absl::InlinedVector<VkSubmitInfo, 4> submit_infos;
  absl::InlinedVector<VkFence, 4> submit_fences;

  absl::InlinedVector<VkSemaphore, 4> wait_semaphores;
  absl::InlinedVector<VkSemaphore, 4> signal_semaphores;

  // A list of submissions that still needs to be deferred.
  IntrusiveList<std::unique_ptr<FencedSubmission>> remaining_submissions;

  // We need to return all remaining submissions back to the queue to avoid
  // dropping work.
  auto submission_cleanup = MakeCleanup([this, &remaining_submissions]() {
// Disable thread-safety-analysis as it doesn't understand this lambda.
//   - This entire function is ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_)
//   - This Cleanup object is destroyed when it drops out of scope
//   - The mutex is always held when executing this function
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wthread-safety-analysis"
#endif
    while (!remaining_submissions.empty()) {
      deferred_submissions_.push_back(
          remaining_submissions.take(remaining_submissions.front()));
    }
#ifdef __clang__
#pragma clang diagnostic pop
#endif
  });

  while (!deferred_submissions_.empty()) {
    wait_semaphores.clear();
    signal_semaphores.clear();

    FencedSubmission* submission = deferred_submissions_.front();
    const PendingBatch& batch = submission->batch;
    ref_ptr<TimePointFence>& fence = submission->fence;

    IREE_ASSIGN_OR_RETURN(
        bool ready_to_submit,
        TryToPrepareSemaphores(batch.wait_semaphores, batch.signal_semaphores,
                               fence, &wait_semaphores, &signal_semaphores));

    if (ready_to_submit) {
      submit_infos.emplace_back();
      PrepareSubmitInfo(wait_semaphores, batch.command_buffers,
                        signal_semaphores, &submit_infos.back(), &arena);

      submit_fences.push_back(fence->value());
      pending_fences_.emplace_back(std::move(fence));
      deferred_submissions_.pop_front();
    } else {
      // We need to defer the submission until later.
      remaining_submissions.push_back(deferred_submissions_.take(submission));
    }
  }

  if (submit_infos.empty()) return false;

  auto infos = arena.AllocateSpan<VkSubmitInfo>(submit_infos.size());
  for (int i = 0, e = submit_infos.size(); i < e; ++i) {
    infos[i] = submit_infos[i];
  }

  // Note: We might be able to batch the submission but it involves non-trivial
  // fence handling. We can handle that if really needed.
  for (int i = 0, e = submit_infos.size(); i < e; ++i) {
    VK_RETURN_IF_ERROR(syms()->vkQueueSubmit(
        queue_, /*submitCount=*/1, &submit_infos[i], submit_fences[i]));
  }

  return true;
}

Status SerializingCommandQueue::WaitIdle(Time deadline_ns) {
  absl::MutexLock lock(&mutex_);

  if (deadline_ns == InfiniteFuture()) {
    IREE_TRACE_SCOPE0("SerializingCommandQueue::WaitIdle#vkQueueWaitIdle");
    // Fast path for using vkQueueWaitIdle, which is usually cheaper (as it
    // requires fewer calls into the driver).

    // Complete all pending work on the queue.
    VK_RETURN_IF_ERROR(syms()->vkQueueWaitIdle(queue_));
    pending_fences_.clear();

    // Submit and complete all deferred work.
    while (!deferred_submissions_.empty()) {
      IREE_ASSIGN_OR_RETURN(bool work_submitted, ProcessDeferredSubmissions());
      if (work_submitted) {
        VK_RETURN_IF_ERROR(syms()->vkQueueWaitIdle(queue_));
        pending_fences_.clear();
      }
    }

    return OkStatus();
  }

  IREE_TRACE_SCOPE0("SerializingCommandQueue::WaitIdle#Fence");

  // Keep trying to submit more workload to the GPU until reaching the deadline.
  do {
    IREE_RETURN_IF_ERROR(ProcessDeferredSubmissions().status());

    uint64_t timeout_ns;
    if (deadline_ns == InfiniteFuture()) {
      timeout_ns = UINT64_MAX;
    } else if (deadline_ns == InfinitePast()) {
      timeout_ns = 0;
    } else {
      // Convert to relative time in nanoseconds.
      // The implementation may not wait with this granularity (like, by
      // 10000x).
      Duration relative_ns = deadline_ns - Now();
      if (relative_ns < ZeroDuration()) {
        return DeadlineExceededErrorBuilder(IREE_LOC)
               << "Deadline exceeded waiting for idle";
      }
      timeout_ns = static_cast<uint64_t>(relative_ns);
    }

    if (pending_fences_.empty()) continue;

    std::vector<VkFence> fences;
    fences.reserve(pending_fences_.size());
    for (const auto& fence : pending_fences_) fences.push_back(fence->value());

    VkResult result =
        syms()->vkWaitForFences(*logical_device_, fences.size(), fences.data(),
                                /*waitAll=*/VK_TRUE, timeout_ns);

    switch (result) {
      case VK_SUCCESS:
        pending_fences_.clear();
        break;
      case VK_TIMEOUT:
        return DeadlineExceededErrorBuilder(IREE_LOC)
               << "Deadline exceeded waiting for idle";
      default:
        return VkResultToStatus(result, IREE_LOC);
    }
    // As long as there is submitted or deferred work still pending.
  } while (!pending_fences_.empty() || !deferred_submissions_.empty());

  return OkStatus();
}

Status SerializingCommandQueue::AdvanceQueueSubmission() {
  absl::MutexLock lock(&mutex_);
  // The returned value just indicates whether there were newly ready
  // submissions gotten submitted to the GPU. Other callers might be
  // interested in that information but for this API we just want to advance
  // queue submisison if possible. So we ignore it here.
  IREE_ASSIGN_OR_RETURN(std::ignore, ProcessDeferredSubmissions());
  return OkStatus();
}

void SerializingCommandQueue::AbortQueueSubmission() {
  absl::MutexLock lock(&mutex_);

  // We have fences in deferred_submissions_ but they are not submitted to GPU
  // yet so we don't need to reset.
  deferred_submissions_.clear();

  std::vector<VkFence> fences;
  fences.reserve(pending_fences_.size());
  for (const auto& fence : pending_fences_) fences.push_back(fence->value());

  syms()->vkWaitForFences(*logical_device_, fences.size(), fences.data(),
                          /*waitAll=*/VK_TRUE, /*timeout=*/UINT64_MAX);
  // Clear the list. Fences will be automatically returned back to the queue
  // after refcount reaches 0.
  pending_fences_.clear();
}

}  // namespace vulkan
}  // namespace hal
}  // namespace iree
