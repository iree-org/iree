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

#include "iree/hal/vulkan/direct_command_queue.h"

#include <cstdint>

#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "iree/base/memory.h"
#include "iree/base/source_location.h"
#include "iree/base/status.h"
#include "iree/base/tracing.h"
#include "iree/hal/vulkan/direct_command_buffer.h"
#include "iree/hal/vulkan/legacy_fence.h"
#include "iree/hal/vulkan/native_binary_semaphore.h"
#include "iree/hal/vulkan/status_util.h"

namespace iree {
namespace hal {
namespace vulkan {

DirectCommandQueue::DirectCommandQueue(
    std::string name, CommandCategoryBitfield supported_categories,
    const ref_ptr<VkDeviceHandle>& logical_device, VkQueue queue)
    : CommandQueue(std::move(name), supported_categories),
      logical_device_(add_ref(logical_device)),
      queue_(queue) {}

DirectCommandQueue::~DirectCommandQueue() {
  IREE_TRACE_SCOPE0("DirectCommandQueue::dtor");
  absl::MutexLock lock(&queue_mutex_);
  syms()->vkQueueWaitIdle(queue_);
}

Status DirectCommandQueue::TranslateBatchInfo(const SubmissionBatch& batch,
                                              VkSubmitInfo* submit_info,
                                              Arena* arena) {
  // TODO(benvanik): see if we can go to finer-grained stages.
  // For example, if this was just queue ownership transfers then we can use
  // the pseudo-stage of VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT.
  VkPipelineStageFlags dst_stage_mask =
      VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

  auto wait_semaphore_handles =
      arena->AllocateSpan<VkSemaphore>(batch.wait_semaphores.size());
  auto wait_dst_stage_masks =
      arena->AllocateSpan<VkPipelineStageFlags>(batch.wait_semaphores.size());
  for (int i = 0; i < batch.wait_semaphores.size(); ++i) {
    const auto& semaphore_value = batch.wait_semaphores[i];
    if (semaphore_value.index() == 0) {
      const auto& binary_semaphore =
          static_cast<NativeBinarySemaphore*>(absl::get<0>(semaphore_value));
      wait_semaphore_handles[i] = binary_semaphore->handle();
    } else {
      // TODO(b/140141417): implement timeline semaphores.
      return UnimplementedErrorBuilder(IREE_LOC)
             << "Timeline semaphores not yet implemented";
    }
    wait_dst_stage_masks[i] = dst_stage_mask;
  }

  auto signal_semaphore_handles =
      arena->AllocateSpan<VkSemaphore>(batch.signal_semaphores.size());
  for (int i = 0; i < batch.signal_semaphores.size(); ++i) {
    const auto& semaphore_value = batch.signal_semaphores[i];
    if (semaphore_value.index() == 0) {
      const auto& binary_semaphore =
          static_cast<NativeBinarySemaphore*>(absl::get<0>(semaphore_value));
      signal_semaphore_handles[i] = binary_semaphore->handle();
    } else {
      // TODO(b/140141417): implement timeline semaphores.
      return UnimplementedErrorBuilder(IREE_LOC)
             << "Timeline semaphores not yet implemented";
    }
  }

  auto command_buffer_handles =
      arena->AllocateSpan<VkCommandBuffer>(batch.command_buffers.size());
  for (int i = 0; i < batch.command_buffers.size(); ++i) {
    const auto& command_buffer = batch.command_buffers[i];
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

  return OkStatus();
}

Status DirectCommandQueue::Submit(absl::Span<const SubmissionBatch> batches,
                                  FenceValue fence) {
  IREE_TRACE_SCOPE0("DirectCommandQueue::Submit");

  // Map the submission batches to VkSubmitInfos.
  // Note that we must keep all arrays referenced alive until submission
  // completes and since there are a bunch of them we use an arena.
  Arena arena(4 * 1024);
  auto submit_infos = arena.AllocateSpan<VkSubmitInfo>(batches.size());
  for (int i = 0; i < batches.size(); ++i) {
    RETURN_IF_ERROR(TranslateBatchInfo(batches[i], &submit_infos[i], &arena));
  }

  // TODO(b/140141417): implement timeline semaphore fences and switch here.
  auto legacy_fence = reinterpret_cast<LegacyFence*>(fence.first);
  ASSIGN_OR_RETURN(VkFence fence_handle,
                   legacy_fence->AcquireSignalFence(fence.second));

  {
    absl::MutexLock lock(&queue_mutex_);
    VK_RETURN_IF_ERROR(syms()->vkQueueSubmit(
        queue_, submit_infos.size(), submit_infos.data(), fence_handle));
  }

  return OkStatus();
}

Status DirectCommandQueue::WaitIdle(absl::Time deadline) {
  if (deadline == absl::InfiniteFuture()) {
    // Fast path for using vkQueueWaitIdle, which is usually cheaper (as it
    // requires fewer calls into the driver).
    IREE_TRACE_SCOPE0("DirectCommandQueue::WaitIdle#vkQueueWaitIdle");
    absl::MutexLock lock(&queue_mutex_);
    VK_RETURN_IF_ERROR(syms()->vkQueueWaitIdle(queue_));
    return OkStatus();
  }

  IREE_TRACE_SCOPE0("DirectCommandQueue::WaitIdle#Fence");

  // Create a new fence just for this wait. This keeps us thread-safe as the
  // behavior of wait+reset is racey.
  VkFenceCreateInfo create_info;
  create_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  create_info.pNext = nullptr;
  create_info.flags = 0;
  VkFence fence = VK_NULL_HANDLE;
  VK_RETURN_IF_ERROR(syms()->vkCreateFence(
      *logical_device_, &create_info, logical_device_->allocator(), &fence));
  auto fence_cleanup = MakeCleanup([this, fence]() {
    syms()->vkDestroyFence(*logical_device_, fence,
                           logical_device_->allocator());
  });

  uint64_t timeout;
  if (deadline == absl::InfinitePast()) {
    // Do not wait.
    timeout = 0;
  } else if (deadline == absl::InfiniteFuture()) {
    // Wait forever.
    timeout = UINT64_MAX;
  } else {
    // Convert to relative time in nanoseconds.
    // The implementation may not wait with this granularity (like, by 10000x).
    absl::Time now = absl::Now();
    if (deadline < now) {
      return DeadlineExceededErrorBuilder(IREE_LOC) << "Deadline in the past";
    }
    timeout = static_cast<uint64_t>(absl::ToInt64Nanoseconds(deadline - now));
  }

  {
    absl::MutexLock lock(&queue_mutex_);
    VK_RETURN_IF_ERROR(syms()->vkQueueSubmit(queue_, 0, nullptr, fence));
  }

  VkResult result =
      syms()->vkWaitForFences(*logical_device_, 1, &fence, VK_TRUE, timeout);
  switch (result) {
    case VK_SUCCESS:
      return OkStatus();
    case VK_TIMEOUT:
      return DeadlineExceededErrorBuilder(IREE_LOC)
             << "Deadline exceeded waiting for idle";
    default:
      return VkResultToStatus(result);
  }
}

}  // namespace vulkan
}  // namespace hal
}  // namespace iree
