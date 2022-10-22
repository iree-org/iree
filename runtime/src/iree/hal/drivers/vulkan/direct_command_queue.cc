// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/direct_command_queue.h"

#include <cstdint>

#include "iree/base/tracing.h"
#include "iree/hal/drivers/vulkan/direct_command_buffer.h"
#include "iree/hal/drivers/vulkan/dynamic_symbols.h"
#include "iree/hal/drivers/vulkan/native_semaphore.h"
#include "iree/hal/drivers/vulkan/status_util.h"
#include "iree/hal/drivers/vulkan/tracing.h"
#include "iree/hal/drivers/vulkan/util/ref_ptr.h"

namespace iree {
namespace hal {
namespace vulkan {

DirectCommandQueue::DirectCommandQueue(
    VkDeviceHandle* logical_device,
    iree_hal_command_category_t supported_categories, VkQueue queue,
    bool use_rgp)
    : CommandQueue(logical_device, supported_categories, queue) {
  this->enable_rgp = use_rgp;
}

DirectCommandQueue::~DirectCommandQueue() = default;

iree_status_t DirectCommandQueue::TranslateBatchInfo(
    const iree_hal_submission_batch_t* batch, VkSubmitInfo* submit_info,
    VkTimelineSemaphoreSubmitInfo* timeline_submit_info, Arena* arena) {
  // TODO(benvanik): see if we can go to finer-grained stages.
  // For example, if this was just queue ownership transfers then we can use
  // the pseudo-stage of VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT.
  VkPipelineStageFlags dst_stage_mask =
      VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

  auto wait_semaphore_handles =
      arena->AllocateSpan<VkSemaphore>(batch->wait_semaphores.count);
  auto wait_semaphore_values =
      arena->AllocateSpan<uint64_t>(batch->wait_semaphores.count);
  auto wait_dst_stage_masks =
      arena->AllocateSpan<VkPipelineStageFlags>(batch->wait_semaphores.count);
  for (iree_host_size_t i = 0; i < batch->wait_semaphores.count; ++i) {
    wait_semaphore_handles[i] = iree_hal_vulkan_native_semaphore_handle(
        batch->wait_semaphores.semaphores[i]);
    wait_semaphore_values[i] = batch->wait_semaphores.payload_values[i];
    wait_dst_stage_masks[i] = dst_stage_mask;
  }

  auto signal_semaphore_handles =
      arena->AllocateSpan<VkSemaphore>(batch->signal_semaphores.count);
  auto signal_semaphore_values =
      arena->AllocateSpan<uint64_t>(batch->signal_semaphores.count);
  for (iree_host_size_t i = 0; i < batch->signal_semaphores.count; ++i) {
    signal_semaphore_handles[i] = iree_hal_vulkan_native_semaphore_handle(
        batch->signal_semaphores.semaphores[i]);
    signal_semaphore_values[i] = batch->signal_semaphores.payload_values[i];
  }

  auto command_buffer_handles =
      arena->AllocateSpan<VkCommandBuffer>(batch->command_buffer_count);
  for (iree_host_size_t i = 0; i < batch->command_buffer_count; ++i) {
    command_buffer_handles[i] =
        iree_hal_vulkan_direct_command_buffer_handle(batch->command_buffers[i]);
  }

  submit_info->sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info->pNext = timeline_submit_info;
  submit_info->waitSemaphoreCount =
      static_cast<uint32_t>(wait_semaphore_handles.size());
  submit_info->pWaitSemaphores = wait_semaphore_handles.data();
  submit_info->pWaitDstStageMask = wait_dst_stage_masks.data();
  submit_info->commandBufferCount =
      static_cast<uint32_t>(command_buffer_handles.size());
  submit_info->pCommandBuffers = command_buffer_handles.data();
  submit_info->signalSemaphoreCount =
      static_cast<uint32_t>(signal_semaphore_handles.size());
  submit_info->pSignalSemaphores = signal_semaphore_handles.data();

  timeline_submit_info->sType =
      VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
  timeline_submit_info->pNext = nullptr;
  timeline_submit_info->waitSemaphoreValueCount =
      static_cast<uint32_t>(wait_semaphore_values.size());
  timeline_submit_info->pWaitSemaphoreValues = wait_semaphore_values.data();
  timeline_submit_info->signalSemaphoreValueCount =
      static_cast<uint32_t>(signal_semaphore_values.size());
  timeline_submit_info->pSignalSemaphoreValues = signal_semaphore_values.data();

  return iree_ok_status();
}

iree_status_t DirectCommandQueue::Submit(
    iree_host_size_t batch_count, const iree_hal_submission_batch_t* batches) {
  IREE_TRACE_SCOPE0("DirectCommandQueue::Submit");

  // Map the submission batches to VkSubmitInfos.
  // Note that we must keep all arrays referenced alive until submission
  // completes and since there are a bunch of them we use an arena.
  Arena arena(4 * 1024);
  auto submit_infos = arena.AllocateSpan<VkSubmitInfo>(batch_count);
  auto timeline_submit_infos =
      arena.AllocateSpan<VkTimelineSemaphoreSubmitInfo>(batch_count);
  for (int i = 0; i < batch_count; ++i) {
    IREE_RETURN_IF_ERROR(TranslateBatchInfo(&batches[i], &submit_infos[i],
                                            &timeline_submit_infos[i], &arena));
  }

  iree_status_t status;
  if (enable_rgp) {
    VkDebugUtilsLabelEXT vkDebugUtilLabelEnd = {};
    vkDebugUtilLabelEnd.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
    vkDebugUtilLabelEnd.pNext = nullptr;
    vkDebugUtilLabelEnd.pLabelName = "AmdFrameEnd";

    VkDebugUtilsLabelEXT vkDebugUtilLabelBegin = {};
    vkDebugUtilLabelBegin.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
    vkDebugUtilLabelBegin.pNext = nullptr;
    vkDebugUtilLabelBegin.pLabelName = "AmdFrameBegin";

    iree_slim_mutex_lock(&queue_mutex_);

    syms()->vkQueueInsertDebugUtilsLabelEXT(queue_, &vkDebugUtilLabelBegin);

    status = VK_RESULT_TO_STATUS(
        syms()->vkQueueSubmit(queue_,
                              static_cast<uint32_t>(submit_infos.size()),
                              submit_infos.data(), VK_NULL_HANDLE),
        "vkQueueSubmit");

    syms()->vkQueueWaitIdle(queue_);
    syms()->vkQueueInsertDebugUtilsLabelEXT(queue_, &vkDebugUtilLabelEnd);

    iree_slim_mutex_unlock(&queue_mutex_);
    IREE_RETURN_IF_ERROR(status);
  } else {
    status = VK_RESULT_TO_STATUS(
        syms()->vkQueueSubmit(queue_,
                              static_cast<uint32_t>(submit_infos.size()),
                              submit_infos.data(), VK_NULL_HANDLE),
        "vkQueueSubmit");
  }

  return iree_ok_status();
}

iree_status_t DirectCommandQueue::WaitIdle(iree_timeout_t timeout) {
  iree_time_t deadline_ns = iree_timeout_as_deadline_ns(timeout);
  if (deadline_ns == IREE_TIME_INFINITE_FUTURE) {
    // Fast path for using vkQueueWaitIdle, which is usually cheaper (as it
    // requires fewer calls into the driver).
    IREE_TRACE_SCOPE0("DirectCommandQueue::WaitIdle#vkQueueWaitIdle");
    iree_slim_mutex_lock(&queue_mutex_);
    iree_status_t status =
        VK_RESULT_TO_STATUS(syms()->vkQueueWaitIdle(queue_), "vkQueueWaitIdle");
    iree_slim_mutex_unlock(&queue_mutex_);
    iree_hal_vulkan_tracing_context_collect(tracing_context(), VK_NULL_HANDLE);
    return status;
  }

  IREE_TRACE_SCOPE0("DirectCommandQueue::WaitIdle#Fence");

  // Create a new fence just for this wait. This keeps us thread-safe as the
  // behavior of wait+reset is racey.
  VkFenceCreateInfo create_info;
  create_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  create_info.pNext = nullptr;
  create_info.flags = 0;
  VkFence fence = VK_NULL_HANDLE;
  VK_RETURN_IF_ERROR(
      syms()->vkCreateFence(*logical_device_, &create_info,
                            logical_device_->allocator(), &fence),
      "vkCreateFence");

  uint64_t timeout_ns;
  if (deadline_ns == IREE_TIME_INFINITE_PAST) {
    // Do not wait.
    timeout_ns = 0;
  } else if (deadline_ns == IREE_TIME_INFINITE_FUTURE) {
    // Wait forever.
    timeout_ns = UINT64_MAX;
  } else {
    // Convert to relative time in nanoseconds.
    // The implementation may not wait with this granularity (like by 10000x).
    iree_time_t now_ns = iree_time_now();
    if (deadline_ns < now_ns) {
      return iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED);
    }
    timeout_ns = (uint64_t)(deadline_ns - now_ns);
  }

  iree_slim_mutex_lock(&queue_mutex_);
  iree_status_t status = VK_RESULT_TO_STATUS(
      syms()->vkQueueSubmit(queue_, 0, nullptr, fence), "vkQueueSubmit");
  iree_slim_mutex_unlock(&queue_mutex_);

  if (iree_status_is_ok(status)) {
    VkResult result = syms()->vkWaitForFences(*logical_device_, 1, &fence,
                                              VK_TRUE, timeout_ns);
    switch (result) {
      case VK_SUCCESS:
        status = iree_ok_status();
        break;
      case VK_TIMEOUT:
        status = iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED);
        break;
      default:
        status = VK_RESULT_TO_STATUS(result, "vkWaitForFences");
        break;
    }
  }

  syms()->vkDestroyFence(*logical_device_, fence, logical_device_->allocator());

  iree_hal_vulkan_tracing_context_collect(tracing_context(), VK_NULL_HANDLE);

  return status;
}

}  // namespace vulkan
}  // namespace hal
}  // namespace iree
