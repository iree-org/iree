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
#include "iree/base/tracing.h"
#include "iree/hal/api.h"
#include "iree/hal/vulkan/direct_command_buffer.h"
#include "iree/hal/vulkan/emulated_semaphore.h"
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
iree_status_t TryToPrepareSemaphores(
    const absl::InlinedVector<SemaphoreValue, 4>& batch_wait_semaphores,
    const absl::InlinedVector<SemaphoreValue, 4>& batch_signal_semaphores,
    const ref_ptr<TimePointFence>& batch_fence,
    absl::InlinedVector<VkSemaphore, 4>* wait_semaphores,
    absl::InlinedVector<VkSemaphore, 4>* signal_semaphores,
    bool* out_ready_to_submit) {
  IREE_TRACE_SCOPE0("TryToPrepareSemaphores");
  *out_ready_to_submit = false;

  wait_semaphores->clear();
  for (const auto& timeline_semaphore : batch_wait_semaphores) {
    // Query first to progress this timeline semaphore to the furthest.
    uint64_t signaled_value = 0;
    IREE_RETURN_IF_ERROR(
        iree_hal_semaphore_query(timeline_semaphore.first, &signaled_value));

    // If it's already signaled to a value greater than we require here,
    // we can just ignore this semaphore now.
    if (signaled_value >= timeline_semaphore.second) {
      continue;
    }

    // Otherwise try to get a binary semaphore for this time point so that
    // we can wait on.
    // TODO(antiagainst): if this fails we need to cancel.
    VkSemaphore wait_semaphore = VK_NULL_HANDLE;
    IREE_RETURN_IF_ERROR(iree_hal_vulkan_emulated_semaphore_acquire_wait_handle(
        timeline_semaphore.first, timeline_semaphore.second, batch_fence,
        &wait_semaphore));
    wait_semaphores->push_back(wait_semaphore);

    if (wait_semaphore == VK_NULL_HANDLE) {
      // We cannot wait on this time point yet: there are no previous semaphores
      // submitted to the GPU that can signal a value greater than what's
      // desired here.

      // Cancel the wait so others may make progress.
      // TODO(antiagainst): if any of these fail we need to cancel.
      for (iree_host_size_t i = 0; i < batch_wait_semaphores.size(); ++i) {
        if (!wait_semaphores->at(i)) break;
        IREE_RETURN_IF_ERROR(
            iree_hal_vulkan_emulated_semaphore_cancel_wait_handle(
                batch_wait_semaphores[i].first, wait_semaphores->at(i)));
      }

      // This batch cannot be submitted to GPU yet.
      return iree_ok_status();
    }
  }

  // We've collected all necessary binary semaphores for each timeline we need
  // to wait on. Now prepare binary semaphores for signaling.
  signal_semaphores->clear();
  for (const auto& timeline_semaphore : batch_signal_semaphores) {
    // SerializingCommandQueue only works with EmulatedTimelineSemaphore.
    VkSemaphore signal_semaphore = VK_NULL_HANDLE;
    IREE_RETURN_IF_ERROR(
        iree_hal_vulkan_emulated_semaphore_acquire_signal_handle(
            timeline_semaphore.first, timeline_semaphore.second, batch_fence,
            &signal_semaphore));
    signal_semaphores->push_back(signal_semaphore);
  }

  // Good to submit!
  *out_ready_to_submit = true;
  return iree_ok_status();
}

// Prepares `VkSubmitInfo` to submit the given list of |command_buffers| that
// waiting on |wait_semaphores| and signalling |signal_semaphores|. Necessary
// structures are allocated from |arena| and the result `VkSubmitInfo` is
// written to |submit_info|.
void PrepareSubmitInfo(absl::Span<const VkSemaphore> wait_semaphore_handles,
                       absl::Span<const VkCommandBuffer> command_buffer_handles,
                       absl::Span<const VkSemaphore> signal_semaphore_handles,
                       VkSubmitInfo* submit_info, Arena* arena) {
  // TODO(benvanik): see if we can go to finer-grained stages.
  // For example, if this was just queue ownership transfers then we can use
  // the pseudo-stage of VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT.
  auto wait_dst_stage_masks =
      arena->AllocateSpan<VkPipelineStageFlags>(wait_semaphore_handles.size());
  for (size_t i = 0, e = wait_semaphore_handles.size(); i < e; ++i) {
    wait_dst_stage_masks[i] =
        VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
  }

  // NOTE: this code does some very weird things - the handles we take in as
  // args are mutated in-place after this function is called so we can't
  // reference them here. If we were going to preserve this code post-Vulkan 1.2
  // then we'd really want to rework all of this to properly use the arena from
  // the start instead of all this InlinedVector tomfoolery.
  auto wait_semaphores =
      arena->AllocateSpan<VkSemaphore>(wait_semaphore_handles.size());
  for (size_t i = 0, e = wait_semaphore_handles.size(); i < e; ++i) {
    wait_semaphores[i] = wait_semaphore_handles[i];
  }
  auto command_buffers =
      arena->AllocateSpan<VkCommandBuffer>(command_buffer_handles.size());
  for (size_t i = 0, e = command_buffer_handles.size(); i < e; ++i) {
    command_buffers[i] = command_buffer_handles[i];
  }
  auto signal_semaphores =
      arena->AllocateSpan<VkSemaphore>(signal_semaphore_handles.size());
  for (size_t i = 0, e = signal_semaphore_handles.size(); i < e; ++i) {
    signal_semaphores[i] = signal_semaphore_handles[i];
  }

  submit_info->sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info->pNext = nullptr;
  submit_info->waitSemaphoreCount =
      static_cast<uint32_t>(wait_semaphores.size());
  submit_info->pWaitSemaphores = wait_semaphores.data();
  submit_info->pWaitDstStageMask = wait_dst_stage_masks.data();
  submit_info->commandBufferCount =
      static_cast<uint32_t>(command_buffers.size());
  submit_info->pCommandBuffers = command_buffers.data();
  submit_info->signalSemaphoreCount =
      static_cast<uint32_t>(signal_semaphores.size());
  submit_info->pSignalSemaphores = signal_semaphores.data();
}

}  // namespace

SerializingCommandQueue::SerializingCommandQueue(
    VkDeviceHandle* logical_device, std::string name,
    iree_hal_command_category_t supported_categories, VkQueue queue,
    TimePointFencePool* fence_pool)
    : CommandQueue(logical_device, std::move(name), supported_categories,
                   queue),
      fence_pool_(fence_pool) {}

SerializingCommandQueue::~SerializingCommandQueue() = default;

iree_status_t SerializingCommandQueue::Submit(
    iree_host_size_t batch_count, const iree_hal_submission_batch_t* batches) {
  IREE_TRACE_SCOPE0("SerializingCommandQueue::Submit");

  IntrusiveList<std::unique_ptr<FencedSubmission>> new_submissions;
  for (iree_host_size_t i = 0; i < batch_count; ++i) {
    const iree_hal_submission_batch_t* batch = &batches[i];

    // Grab a fence for this submission first. This will be used to check the
    // progress of emulated timeline semaphores later.
    auto submission = std::make_unique<FencedSubmission>();
    IREE_RETURN_IF_ERROR(fence_pool_->Acquire(&submission->fence));

    submission->wait_semaphores.resize(batch->wait_semaphores.count);
    for (iree_host_size_t j = 0; j < batch->wait_semaphores.count; ++j) {
      submission->wait_semaphores[j] = {
          batch->wait_semaphores.semaphores[j],
          batch->wait_semaphores.payload_values[j]};
    }

    submission->command_buffers.resize(batch->command_buffer_count);
    for (iree_host_size_t j = 0; j < batch->command_buffer_count; ++j) {
      submission->command_buffers[j] =
          iree_hal_vulkan_direct_command_buffer_handle(
              batch->command_buffers[j]);
    }

    submission->signal_semaphores.resize(batch->signal_semaphores.count);
    for (iree_host_size_t j = 0; j < batch->signal_semaphores.count; ++j) {
      submission->signal_semaphores[j] = {
          batch->signal_semaphores.semaphores[j],
          batch->signal_semaphores.payload_values[j]};
    }

    new_submissions.push_back(std::move(submission));
  }

  iree_slim_mutex_lock(&queue_mutex_);
  deferred_submissions_.merge_from(&new_submissions);
  iree_status_t status = ProcessDeferredSubmissions();
  iree_slim_mutex_unlock(&queue_mutex_);
  return status;
}

iree_status_t SerializingCommandQueue::ProcessDeferredSubmissions(
    bool* out_work_submitted) {
  IREE_TRACE_SCOPE0("SerializingCommandQueue::ProcessDeferredSubmissions");

  // Try to process the submissions and if we hit a stopping point during the
  // process where we need to yield we take the remaining submissions and
  // re-enqueue them.
  IntrusiveList<std::unique_ptr<FencedSubmission>> remaining_submissions;
  iree_status_t status =
      TryProcessDeferredSubmissions(remaining_submissions, out_work_submitted);
  while (!remaining_submissions.empty()) {
    deferred_submissions_.push_back(
        remaining_submissions.take(remaining_submissions.front()));
  }

  return status;
}

iree_status_t SerializingCommandQueue::TryProcessDeferredSubmissions(
    IntrusiveList<std::unique_ptr<FencedSubmission>>& remaining_submissions,
    bool* out_work_submitted) {
  if (out_work_submitted) *out_work_submitted = false;

  Arena arena(4 * 1024);
  absl::InlinedVector<VkSubmitInfo, 4> submit_infos;
  absl::InlinedVector<VkFence, 4> submit_fences;
  while (!deferred_submissions_.empty()) {
    FencedSubmission* submission = deferred_submissions_.front();
    ref_ptr<TimePointFence>& fence = submission->fence;

    absl::InlinedVector<VkSemaphore, 4> wait_semaphores;
    absl::InlinedVector<VkSemaphore, 4> signal_semaphores;
    bool ready_to_submit = false;
    IREE_RETURN_IF_ERROR(TryToPrepareSemaphores(
        submission->wait_semaphores, submission->signal_semaphores, fence,
        &wait_semaphores, &signal_semaphores, &ready_to_submit));
    if (ready_to_submit) {
      submit_infos.emplace_back();
      PrepareSubmitInfo(wait_semaphores, submission->command_buffers,
                        signal_semaphores, &submit_infos.back(), &arena);

      submit_fences.push_back(fence->value());
      pending_fences_.emplace_back(std::move(fence));
      deferred_submissions_.pop_front();
    } else {
      // We need to defer the submission until later.
      remaining_submissions.push_back(deferred_submissions_.take(submission));
    }
  }
  if (submit_infos.empty()) {
    if (out_work_submitted) *out_work_submitted = false;
    return iree_ok_status();
  }

  // Note: We might be able to batch the submission but it involves non-trivial
  // fence handling. We can handle that if really needed.
  for (size_t i = 0, e = submit_infos.size(); i < e; ++i) {
    VK_RETURN_IF_ERROR(
        syms()->vkQueueSubmit(queue_, /*submitCount=*/1, &submit_infos[i],
                              submit_fences[i]),
        "vkQueueSubmit");
  }

  if (out_work_submitted) *out_work_submitted = true;
  return iree_ok_status();
}

iree_status_t SerializingCommandQueue::WaitIdle(iree_time_t deadline_ns) {
  iree_status_t status = iree_ok_status();

  if (deadline_ns == IREE_TIME_INFINITE_FUTURE) {
    IREE_TRACE_SCOPE0("SerializingCommandQueue::WaitIdle#vkQueueWaitIdle");
    // Fast path for using vkQueueWaitIdle, which is usually cheaper (as it
    // requires fewer calls into the driver).

    iree_slim_mutex_lock(&queue_mutex_);

    // Complete all pending work on the queue.
    status =
        VK_RESULT_TO_STATUS(syms()->vkQueueWaitIdle(queue_), "vkQueueWaitIdle");
    if (!iree_status_is_ok(status)) {
      iree_slim_mutex_unlock(&queue_mutex_);
      return status;
    }
    pending_fences_.clear();

    // Submit and complete all deferred work.
    while (!deferred_submissions_.empty()) {
      bool work_submitted = false;
      status = ProcessDeferredSubmissions(&work_submitted);
      if (!iree_status_is_ok(status)) break;
      if (work_submitted) {
        status = VK_RESULT_TO_STATUS(syms()->vkQueueWaitIdle(queue_),
                                     "vkQueueWaitIdle");
        if (!iree_status_is_ok(status)) break;
        pending_fences_.clear();
      }
    }

    iree_slim_mutex_unlock(&queue_mutex_);
    return status;
  }

  IREE_TRACE_SCOPE0("SerializingCommandQueue::WaitIdle#Fence");

  // Keep trying to submit more workload to the GPU until reaching the deadline.
  iree_slim_mutex_lock(&queue_mutex_);
  do {
    status = ProcessDeferredSubmissions();
    bool has_deferred_submissions = !deferred_submissions_.empty();
    absl::InlinedVector<VkFence, 8> fence_handles(pending_fences_.size());
    for (size_t i = 0; i < pending_fences_.size(); ++i) {
      fence_handles[i] = pending_fences_[i]->value();
    }
    if (!iree_status_is_ok(status)) {
      break;  // unable to process submissions
    } else if (!has_deferred_submissions && fence_handles.empty()) {
      break;  // no more work - idle achieved
    }

    uint64_t timeout_ns;
    if (deadline_ns == IREE_TIME_INFINITE_FUTURE) {
      timeout_ns = UINT64_MAX;
    } else if (deadline_ns == IREE_TIME_INFINITE_PAST) {
      timeout_ns = 0;
    } else {
      // Convert to relative time in nanoseconds.
      // The implementation may not wait with this granularity (like by 10000x).
      iree_time_t now_ns = iree_time_now();
      if (deadline_ns < now_ns) {
        return iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED);
      }
      timeout_ns = (uint64_t)(deadline_ns - now_ns);
    }
    VkResult result = syms()->vkWaitForFences(
        *logical_device_, static_cast<uint32_t>(fence_handles.size()),
        fence_handles.data(),
        /*waitAll=*/VK_TRUE, timeout_ns);

    switch (result) {
      case VK_SUCCESS:
        pending_fences_.clear();
        break;
      case VK_TIMEOUT:
        status = iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED);
        break;
      default:
        status = VK_RESULT_TO_STATUS(result, "vkWaitForFences");
        break;
    }
    // As long as there is submitted or deferred work still pending.
  } while (iree_status_is_ok(status));
  iree_slim_mutex_unlock(&queue_mutex_);
  return status;
}

iree_status_t SerializingCommandQueue::AdvanceQueueSubmission() {
  // The returned value just indicates whether there were newly ready
  // submissions gotten submitted to the GPU. Other callers might be
  // interested in that information but for this API we just want to advance
  // queue submisison if possible. So we ignore it here.
  iree_slim_mutex_lock(&queue_mutex_);
  iree_status_t status = ProcessDeferredSubmissions();
  iree_slim_mutex_unlock(&queue_mutex_);
  return status;
}

void SerializingCommandQueue::AbortQueueSubmission() {
  iree_slim_mutex_lock(&queue_mutex_);

  // We have fences in deferred_submissions_ but they are not submitted to GPU
  // yet so we don't need to reset.
  deferred_submissions_.clear();

  absl::InlinedVector<VkFence, 8> fence_handles(pending_fences_.size());
  for (size_t i = 0; i < pending_fences_.size(); ++i) {
    fence_handles[i] = pending_fences_[i]->value();
  }

  syms()->vkWaitForFences(*logical_device_,
                          static_cast<uint32_t>(fence_handles.size()),
                          fence_handles.data(),
                          /*waitAll=*/VK_TRUE, /*timeout=*/UINT64_MAX);

  // Clear the list. Fences will be automatically returned back to the queue
  // after refcount reaches 0.
  pending_fences_.clear();

  iree_slim_mutex_unlock(&queue_mutex_);
}

void SerializingCommandQueue::SignalFences(absl::Span<VkFence> fences) {
  const auto span_contains = [fences](VkFence fence) {
    for (VkFence f : fences) {
      if (f == fence) return true;
    }
    return false;
  };

  iree_slim_mutex_lock(&queue_mutex_);
  auto it = pending_fences_.begin();
  while (it != pending_fences_.end()) {
    if (span_contains((*it)->value())) {
      it = pending_fences_.erase(it);
    } else {
      ++it;
    }
  }
  iree_slim_mutex_unlock(&queue_mutex_);
}

}  // namespace vulkan
}  // namespace hal
}  // namespace iree
