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

#ifndef IREE_HAL_VULKAN_ENUMLATED_SEMAPHORE_H_
#define IREE_HAL_VULKAN_ENUMLATED_SEMAPHORE_H_

#include "iree/hal/api.h"
#include "iree/hal/vulkan/command_queue.h"
#include "iree/hal/vulkan/handle_util.h"
#include "iree/hal/vulkan/timepoint_util.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates a timeline semaphore emulated via `VkFence`s and binary
// `VkSemaphore`s.
//
// Vulkan provides several explicit synchronization primitives: fences,
// (binary/timeline) semaphores, events, pipeline barriers, and render passes.
// See "6. Synchronization and Cache Control" of the Vulkan specification
// for the details.
//
// Render passes are for graphics pipelines so IREE does not care about them.
// Pipeline barriers synchronize control within a command buffer at a single
// point. Fences, (binary/timeline) semaphores, and events are synchronization
// primitives that have separate signal and wait operations. Events are more
// fine-grained compared to fences and semaphores given that they can be
// signaled or waited within a command buffer while fences and semaphores are
// at queue submissions. Each of them have its usage requirements:
//
// * Fences must be signaled on GPU and waited on CPU. Fences must be reset
//   before reuse.
// * Binary semaphores must be signaled on GPU and waited on GPU. They do not
//   support wait-before-signal submission order. More importantly, binary
//   semaphore wait also unsignals the semaphore. So binary semaphore signals
//   and waits should occur in discrete 1:1 pairs.
// * Timeline semaphores can be signaled on CPU or GPU and waited on CPU or GPU.
//   They support wait-before-signal submission order. Timeline semaphores do
//   not need to be reset.
//
// It's clear that timeline semaphore is more flexible than fences and binary
// semaphores: it unifies GPU and CPU synchronization with a single primitive.
// But it's not always available: it requires the VK_KHR_timeline_semaphore
// or Vulkan 1.2. When it's not available, it can be emulated via `VkFence`s
// and binary `VkSemaphore`s. The emulation need to provide the functionality of
// timeline semaphores and also not violate the usage requirements of `VkFence`s
// and binary `VkSemaphore`s.
//
// The basic idea is to create a timeline object with time points to emulate the
// timeline semaphore, which consists of a monotonically increasing 64-bit
// integer value. Each time point represents a specific signaled/waited integer
// value of the timeline semaphore; each time point can associate with binary
// `VkSemaphore`s and/or `VkFence`s for emulating the synchronization.
//
// Concretely, for each of the possible signal -> wait scenarios timeline
// semaphore supports:
//
// ### GPU -> GPU (via `vkQueueSubmit`)
//
// Each `vkQueueSubmit` can attach a `VkTimelineSemaphoreSubmitInfo` to describe
// the timeline semaphore values signaled and waited. Each of the signaled value
// will be a time point and emulated by a binary `VkSemaphore`. We submit the
// binary `VkSemahpore`s to the GPU under the hood. For the waited values, the
// situation is more complicated because of the differences between binary and
// timeline semaphores:
//
// * Binary semaphore signal-wait relationship is strictly 1:1, unlike timeline
//   semaphore where we can have 1:N cases. This means for a specific binary
//   `VkSemaphore` used to emulate a signaled time point, we can have at most
//   one subsequent `vkQueueSubmit` waits on it. We need other mechanisms for
//   additional waits. A simple way is to involve the CPU and don't sumbit
//   the additional work to queue until the desired value is already signaled
//   past. This requires `VkFence`s for letting the CPU know the status of
//   GPU progress, but `VkFence` is needed anyway because of GPU -> CPU
//   synchronization.
// * Binary semaphores does not support wait-before-signal submission order.
//   This means we need to put the submission into a self-managed queue if the
//   binary semaphores used to emulate the time points waited by the submission
//   are not submitted to GPU yet.
//
// ### GPU -> CPU (via `vkWaitSemaphores`)
//
// Without timeline semaphore, we need to use fences to let CPU wait on GPU
// progress. So this direction can be emulated by `vkWaitFences`. It means we
// need to associate a `VkFence` with the given waited timeline semaphores.
// Because we don't know whether a particular `vkQueueSubmit` with timeline
// semaphores will be later waited on by CPU beforehand, we need to bundle each
// of them with a `VkFence` just in case they will be waited on later.
//
// ### CPU -> GPU (via `vkSignalSemaphore`)
//
// This direction can be handled by bumping the signaled timeline value and
// scan the self-managed queue to submit more work to GPU if possible.
//
// ### CPU -> CPU (via `vkWaitSemaphores`)
//
// This is similar to CPU -> GPU direction; we just need to enable other threads
// on CPU side and let them progress.
//
// The implementation is inspired by the Vulkan-ExtensionLayer project:
// https://github.com/KhronosGroup/Vulkan-ExtensionLayer. We don't handle all
// the aspects of the full spec though given that IREE only uses a subset of
// synchronization primitives. So this should not be treated as a full
// emulation of the Vulkan spec and thus does not substitute
// Vulkan-ExtensionLayer.
iree_status_t iree_hal_vulkan_emulated_semaphore_create(
    iree::hal::vulkan::VkDeviceHandle* logical_device,
    iree::hal::vulkan::TimePointSemaphorePool* semaphore_pool,
    iree_host_size_t command_queue_count,
    iree::hal::vulkan::CommandQueue** command_queues, uint64_t initial_value,
    iree_hal_semaphore_t** out_semaphore);

// Acquires a binary semaphore for waiting on the timeline to advance to the
// given |value|. The semaphore returned won't be waited by anyone else.
// |wait_fence| is the fence associated with the queue submission that waiting
// on this semaphore.
//
// Returns VK_NULL_HANDLE if there are no available semaphores for the given
// |value|.
iree_status_t iree_hal_vulkan_emulated_semaphore_acquire_wait_handle(
    iree_hal_semaphore_t* semaphore, uint64_t value,
    const iree::ref_ptr<iree::hal::vulkan::TimePointFence>& wait_fence,
    VkSemaphore* out_handle);

// Cancels the waiting attempt on the given binary |semaphore|. This allows
// the |semaphore| to be waited by others.
iree_status_t iree_hal_vulkan_emulated_semaphore_cancel_wait_handle(
    iree_hal_semaphore_t* semaphore, VkSemaphore handle);

// Acquires a binary semaphore for signaling the timeline to the given |value|.
// |value| must be smaller than the current timeline value. |signal_fence| is
// the fence associated with the queue submission that signals this semaphore.
iree_status_t iree_hal_vulkan_emulated_semaphore_acquire_signal_handle(
    iree_hal_semaphore_t* semaphore, uint64_t value,
    const iree::ref_ptr<iree::hal::vulkan::TimePointFence>& signal_fence,
    VkSemaphore* out_handle);

// Performs a multi-wait on one or more semaphores.
// By default this is an all-wait but |wait_flags| may contain
// VK_SEMAPHORE_WAIT_ANY_BIT to change to an any-wait.
//
// Returns IREE_STATUS_DEADLINE_EXCEEDED if the wait does not complete before
// |deadline_ns| elapses.
iree_status_t iree_hal_vulkan_emulated_semaphore_multi_wait(
    iree::hal::vulkan::VkDeviceHandle* logical_device,
    const iree_hal_semaphore_list_t* semaphore_list, iree_timeout_t timeout,
    VkSemaphoreWaitFlags wait_flags);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_VULKAN_ENUMLATED_SEMAPHORE_H_
