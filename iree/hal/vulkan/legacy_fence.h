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

// TODO(b/140141417): share the pool (and possibly most of the fence impl) with
// the timeline semaphores fallback.

#ifndef IREE_HAL_VULKAN_LEGACY_FENCE_H_
#define IREE_HAL_VULKAN_LEGACY_FENCE_H_

#include <vulkan/vulkan.h>

#include <array>
#include <atomic>

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "iree/base/intrusive_list.h"
#include "iree/base/ref_ptr.h"
#include "iree/base/status.h"
#include "iree/hal/fence.h"
#include "iree/hal/vulkan/handle_util.h"

namespace iree {
namespace hal {
namespace vulkan {

// An outstanding legacy fence signal for a particular timeline value.
// Each signal to a new value gets a new VkFence and these are stored in a
// LegacyFence to quickly scan and process signaled fences.
//
// Must be externally synchronized via the LegacyFence mutex.
struct OutstandingFenceSignal : public IntrusiveLinkBase<void> {
  // Allocated fence that is passed to vkQueueSubmit/vkWaitForFences.
  // Represents a point in the timeline of value.
  VkFence fence = VK_NULL_HANDLE;

  // Value that the fence payload should be when the fence is signaled.
  // Note that since fences may resolve out of order we still need to check that
  // we are only ever advancing the timeline and not just setting this value.
  uint64_t value = UINT64_MAX;

  // True when the fence has been submitted and is pending on the device.
  bool is_pending = false;
};

// A pool of VkFences that can be used by LegacyFence to simulate individual
// payload value signaling. Note that we prefer a pool instead of a ringbuffer
// as we want to allow out-of-order completion.
class LegacyFencePool final : public RefObject<LegacyFencePool> {
 public:
  static constexpr int kMaxInFlightFenceCount = 64;

  // Allocates a new fence pool and all fences.
  static StatusOr<ref_ptr<LegacyFencePool>> Create(
      ref_ptr<VkDeviceHandle> logical_device);

  ~LegacyFencePool();

  const ref_ptr<VkDeviceHandle>& logical_device() const {
    return logical_device_;
  }
  const ref_ptr<DynamicSymbols>& syms() const {
    return logical_device_->syms();
  }

  // Acquires a fence from the pool for use by the caller.
  // The fence is guaranteed to not be in-flight and will have been reset to an
  // unsignaled state.
  //
  // Returns RESOURCE_EXHAUSTED if the pool has no more available fences.
  // Callers are expected to handle this by waiting on previous fences or for
  // complete device idle. Yes, that's as bad as it sounds, and if we start
  // seeing that we should bump up the max count.
  StatusOr<OutstandingFenceSignal*> Acquire();

  // Releases one or more fences back to the pool.
  // The fences must either be signaled or not be in-flight.
  void ReleaseResolved(IntrusiveList<OutstandingFenceSignal>* fence_signals);

  // Releases one or more unresolved fences back to the pool.
  // These may be in any state and will be assumed as untouchable.
  void ReleaseUnresolved(IntrusiveList<OutstandingFenceSignal>* fence_signals);

 private:
  explicit LegacyFencePool(ref_ptr<VkDeviceHandle> logical_device);

  Status PreallocateFences() ABSL_LOCKS_EXCLUDED(mutex_);

  ref_ptr<VkDeviceHandle> logical_device_;

  absl::Mutex mutex_;
  std::array<OutstandingFenceSignal, kMaxInFlightFenceCount> storage_
      ABSL_GUARDED_BY(mutex_);
  IntrusiveList<OutstandingFenceSignal> unused_fences_ ABSL_GUARDED_BY(mutex_);
  IntrusiveList<OutstandingFenceSignal> unresolved_fences_
      ABSL_GUARDED_BY(mutex_);
};

// A fence implemented using a pool of native VkFences.
// This is supported unconditionally on all versions of Vulkan. When timeline
// semaphores are available we prefer using those instead and this is only
// present as a fallback. We keep this implementation separate so that it can be
// compiled out when the target is known to have the extension.
//
// Simulation of timeline semaphore-based fences is done via a pool of native
// VkFences that each represent a single signaled value. This means that worst
// case we are using one fence per submit however that's no different than if
// we did anything else. Though we can't cancel previously-queued fences when
// increasing values are signaled we can be clever when querying and releasing
// by always walking in reverse relying on the monotonically increasing values.
//
// Valid usage patterns we need to handle:
// 1. fence signaled and waited on (common case)
// 2. fence waited on before beginning signaling
// 3. fence signaled and never waited on
//
// Case 1 is fairly straightforward: we acquire a VkFence, pass that to the
// queue submit, and then vkWaitForFences/query it for completion.
//
// Case 2 requires that we reserve a fence during the wait so that we can pass
// it to vkWaitForFences and track it such that we can reuse it during a future
// signal operation. Since we don't know during signaling if the specific value
// we waited on will ever have its own dedicated signal operation we need to be
// conservative and try to coalesce for correctness. This means that if a wait
// for a value of 1 is performed and we get a signal for a value of 2 we need to
// combine the two. If a signal for a value of 1 is later performed it then
// becomes a no-op. This could lead to some additional latency however that's a
// risk (or benefit!) of using timelines. Rule of thumb: don't do out of order
// signaling.
//
// Case 3 is like case 2 where we need to reserve a fence to wait on, however
// since we don't know if it will ever be signaled we need to take care to
// properly release the VkFence back to the pool for reuse: we don't want to
// return it while there are still waiters for its original event. For this
// reason we track the waiters on a given fence during their wait operation and
// if a fence is released with waiters active we put them in a special
// unresolved until the waiters continue on.
class LegacyFence final : public Fence {
 public:
  // Waits for one or more (or all) fences to reach or exceed the given values.
  static Status WaitForFences(VkDeviceHandle* logical_device,
                              absl::Span<const FenceValue> fences,
                              bool wait_all, absl::Time deadline);

  LegacyFence(ref_ptr<LegacyFencePool> fence_pool, uint64_t initial_value);
  ~LegacyFence() override;

  Status status() const override;

  StatusOr<uint64_t> QueryValue() override;

  // Acquires a new fence for signaling a specific value.
  StatusOr<VkFence> AcquireSignalFence(uint64_t value);

 private:
  // Acquires a new fence for waiting on a specific value.
  // Returns VK_NULL_HANDLE if the fence already resolved and the sticky error
  // if the fence is in an error state.
  StatusOr<VkFence> AcquireWaitFence(uint64_t value);

  // Runs down the outstanding fences list and resolves to the latest signaled
  // value. Will early exit if the value moves beyond |upper_value|.
  Status TryResolveOutstandingFences(uint64_t upper_value)
      ABSL_LOCKS_EXCLUDED(mutex_);
  Status TryResolveOutstandingFencesLocked(uint64_t upper_value)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  ref_ptr<LegacyFencePool> fence_pool_;

  // The current highest value of the fence as verified during a wait or query.
  // Kept outside of |mutex_| so that queries do not require a lock.
  std::atomic<uint64_t> value_;

  mutable absl::Mutex mutex_;

  // Sticky status failure value set on first failure.
  Status status_ ABSL_GUARDED_BY(mutex_);

  // Outstanding VkFences representing signal values.
  // Expected to be sorted in ascending order by value.
  IntrusiveList<OutstandingFenceSignal> outstanding_signals_
      ABSL_GUARDED_BY(mutex_);
};

}  // namespace vulkan
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_VULKAN_LEGACY_FENCE_H_
