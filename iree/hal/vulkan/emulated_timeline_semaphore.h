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

#ifndef IREE_HAL_VULKAN_ENUMLATED_TIMELINE_SEMAPHORE_H_
#define IREE_HAL_VULKAN_ENUMLATED_TIMELINE_SEMAPHORE_H_

#include <vulkan/vulkan.h>

#include <atomic>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "iree/base/intrusive_list.h"
#include "iree/base/ref_ptr.h"
#include "iree/base/status.h"
#include "iree/hal/semaphore.h"
#include "iree/hal/vulkan/handle_util.h"
#include "iree/hal/vulkan/timepoint_util.h"

namespace iree {
namespace hal {
namespace vulkan {

// A timeline semaphore emulated via `VkFence`s and binary `VkSemaphore`s.
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
class EmulatedTimelineSemaphore final : public Semaphore {
 public:
  // Creates a timeline semaphore with the given |initial_value|.
  static StatusOr<ref_ptr<Semaphore>> Create(
      ref_ptr<VkDeviceHandle> logical_device,
      std::function<Status(Semaphore*)> on_signal,
      std::function<void(Semaphore*)> on_failure,
      ref_ptr<TimePointSemaphorePool> semaphore_pool, uint64_t initial_value);

  EmulatedTimelineSemaphore(ref_ptr<VkDeviceHandle> logical_device,
                            std::function<Status(Semaphore*)> on_signal,
                            std::function<void(Semaphore*)> on_failure,
                            ref_ptr<TimePointSemaphorePool> semaphore_pool,
                            uint64_t initialValue);

  ~EmulatedTimelineSemaphore() override;

  StatusOr<uint64_t> Query() override;

  Status Signal(uint64_t value) override;

  Status Wait(uint64_t value, Time deadline_ns) override;

  void Fail(Status status) override;

  // Gets a binary semaphore for waiting on the timeline to advance to the given
  // |value|. The semaphore returned won't be waited by anyone else. Returns
  // VK_NULL_HANDLE if no available semaphores for the given |value|.
  // |wait_fence| is the fence associated with the queue submission that waiting
  // on this semaphore.
  VkSemaphore GetWaitSemaphore(uint64_t value,
                               const ref_ptr<TimePointFence>& wait_fence);

  // Cancels the waiting attempt on the given binary |semaphore|. This allows
  // the |semaphore| to be waited by others.
  Status CancelWaitSemaphore(VkSemaphore semaphore);

  // Gets a binary semaphore for signaling the timeline to the given |value|.
  // |value| must be smaller than the current timeline value. |signal_fence| is
  // the fence associated with the queue submission that signals this semaphore.
  StatusOr<VkSemaphore> GetSignalSemaphore(
      uint64_t value, const ref_ptr<TimePointFence>& signal_fence);

 private:
  // Tries to advance the timeline to the given |to_upper_value| without
  // blocking and returns whether the |to_upper_value| is reached.
  StatusOr<bool> TryToAdvanceTimeline(uint64_t to_upper_value)
      ABSL_LOCKS_EXCLUDED(mutex_);

  std::atomic<uint64_t> signaled_value_;

  ref_ptr<VkDeviceHandle> logical_device_;

  // Callback to inform that this timeline semaphore has signaled a new value.
  std::function<Status(Semaphore*)> on_signal_;

  // Callback to inform that this timeline semaphore has encountered a failure.
  std::function<void(Semaphore*)> on_failure_;

  ref_ptr<TimePointSemaphorePool> semaphore_pool_;

  mutable absl::Mutex mutex_;

  // A list of outstanding semaphores used to emulate time points.
  //
  // The life time of each semaphore is in one of the following state:
  //
  // * Unused state: value = UINT64_MAX, signal/wait fence = nullptr. This is
  //   the state of the semaphore when it's initially acquired from the pool and
  //   not put in the queue for emulating a time point yet.
  // * Pending state: signaled value < value < UINT64_MAX, signal fence =
  //   <some-fence>, wait fence == nullptr. This is the state of the semaphore
  //   when it's put into the GPU queue for emulating a time point.
  // * Pending and waiting state: signaled value < value < UINT64_MAX, signal
  //   fence = <some-fence>, wait fence == <some-fence>. This is the state of
  //   the semaphore when it's put into the GPU queue for emulating a time
  //   point and there is another queue submission waiting on it in GPU.
  // * Signaled and not ever waited state: value <= signaled value, singal/wait
  //   fence = nullptr. This is the state of the semaphore when we know it's
  //   already signaled on GPU and there is no waiters for it.
  // * Signaled and waiting state: value <= signaled value, signal fence =
  //   nullptr, wait fence = <some-fence>. This is the state of the semaphore
  //   when we know it's already signaled on GPU and there is still one queue
  //   submission on GPU is waiting for it.
  IntrusiveList<TimePointSemaphore> outstanding_semaphores_
      ABSL_GUARDED_BY(mutex_);

  // NOTE: We only need to access this status (and thus take the lock) when we
  // want to either signal failure or query the status in the case of the
  // semaphore being set to UINT64_MAX.
  Status status_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace vulkan
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_VULKAN_ENUMLATED_TIMELINE_SEMAPHORE_H_
