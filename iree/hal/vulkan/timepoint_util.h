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

#ifndef IREE_HAL_VULKAN_TIMEPOINT_UTIL_H_
#define IREE_HAL_VULKAN_TIMEPOINT_UTIL_H_

#include <vulkan/vulkan.h>

#include <atomic>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "iree/base/intrusive_list.h"
#include "iree/base/ref_ptr.h"
#include "iree/base/status.h"
#include "iree/hal/vulkan/handle_util.h"

namespace iree {
namespace hal {
namespace vulkan {

class TimePointFencePool;
class TimePointSemaphorePool;

// A fence used for tracking progress of timeline semaphores.
//
// Each queue submission gets a new `VkFence` associated with it so that we can
// later query the `VkFence` on CPU to know what time points were signaled for
// timeline semaphores.
//
// Ref-counting allows the fence to be associated with multiple time points from
// different timelines without worrying about ownership complexity.
//
// This is expected to used together with `TimePointFencePool` and must be
// externally synchronized via `TimePointFencePool`'s mutex.
class TimePointFence final : public RefObject<TimePointFence>,
                             public IntrusiveLinkBase<void> {
 public:
  TimePointFence(TimePointFencePool* pool, VkFence fence)
      : pool_(pool), fence_(fence) {}

  TimePointFence(TimePointFence&& that) = delete;
  TimePointFence& operator=(TimePointFence&&) = delete;

  TimePointFence(const TimePointFence&) = delete;
  TimePointFence& operator=(const TimePointFence&) = delete;

  // Returns this fence to the pool on destruction.
  static void Delete(TimePointFence* ptr);

  VkFence value() const noexcept { return fence_; }
  operator VkFence() const noexcept { return fence_; }

  // Gets the status of this fence object. This might issue an Vulkan API call
  // under the hood.
  VkResult GetStatus();

  // Returns the pool from which this fence comes.
  TimePointFencePool* pool() const { return pool_; }

 private:
  // The pool from which this fence comes.
  TimePointFencePool* pool_;

  // Allocated fence that associated with a bunch of time point(s) of
  // timeline(s). This is passed to queue submission so that we can track the
  // timeline(s) progress on CPU and schedule work.
  VkFence fence_;

  // The fence's status.
  absl::Mutex status_mutex_;
  VkResult status_ ABSL_GUARDED_BY(status_mutex_) = VK_NOT_READY;
};

// A semaphore used for emulating a specific time point of timeline semaphores.
//
// Each signaled time point in a timeline semaphore is emulated with a new
// binary `VkSemaphore` associated with queue submission. These time point
// semaphores are stored in `EmulatedTimelineSemaphore` to quickly scan and
// process signaled values.
//
// This is expected to used together with `TimePointSemaphorePool` and
// `EmulatedTimelineSemaphore` and must be externally synchronized via their
// mutexes.
struct TimePointSemaphore final : public IntrusiveLinkBase<void> {
  // Allocated binary semaphore that represents a time point in the timeline.
  // This is passed to queue submission.
  VkSemaphore semaphore = VK_NULL_HANDLE;

  // Value of the timeline should be at when the binary semaphore is signaled.
  uint64_t value = UINT64_MAX;

  // The fence associated with the queue submission signaling this semaphore.
  // nullptr means this binary semaphore has not been submitted to GPU.
  ref_ptr<TimePointFence> signal_fence = nullptr;

  // The fence associated with the queue submission waiting this semaphore.
  // nullptr means this binary semaphore has not been waited by any queue
  // submission.
  ref_ptr<TimePointFence> wait_fence = nullptr;
};

// A pool of `VkFence`s that can be used by `EmulatedTimelineSemaphore` to track
// timeline progress on CPU. Each `VkFence` can be used to query the status of
// all the semaphores in the same submission to a `VkQueue`.
class TimePointFencePool final : public RefObject<TimePointFencePool> {
 public:
  static constexpr int kMaxInFlightFenceCount = 128;

  // Creates a new pool and pre-allocates `kMaxInFlightFenceCount` fences.
  static StatusOr<ref_ptr<TimePointFencePool>> Create(
      ref_ptr<VkDeviceHandle> logical_device);

  ~TimePointFencePool();

  // Acquires a fence from the pool for use by the caller. The fence is
  // guaranteed to be in unsignaled state and not in-flight on GPU.
  //
  // Returns RESOURCE_EXHAUSTED if the pool has no more available fences.
  // Callers are expected to handle this by waiting on previous fences or for
  // complete device idle. Yes, that's as bad as it sounds, and if we start
  // seeing that we should bump up the max count.
  StatusOr<ref_ptr<TimePointFence>> Acquire();

  // Releases one fence back to the pool. The fence must either be signaled or
  // not be in flight on GPU.
  void ReleaseResolved(TimePointFence* fence);

  const ref_ptr<VkDeviceHandle>& logical_device() const {
    return logical_device_;
  }

 private:
  explicit TimePointFencePool(ref_ptr<VkDeviceHandle> logical_device);

  const ref_ptr<DynamicSymbols>& syms() const;

  Status PreallocateFences() ABSL_LOCKS_EXCLUDED(mutex_);

  ref_ptr<VkDeviceHandle> logical_device_;

  absl::Mutex mutex_;

  // Track via unique_ptr, since IntrusiveList doesn't manage memory itself.
  IntrusiveList<std::unique_ptr<TimePointFence>> free_fences_
      ABSL_GUARDED_BY(mutex_);
};

// A pool of `VkSemaphore`s that can be used by `EmulatedTimelineSemaphore` to
// simulate individual timeline value signaling.
class TimePointSemaphorePool final : public RefObject<TimePointSemaphorePool> {
 public:
  static constexpr int kMaxInFlightSemaphoreCount = 64;

  // Creates a new pool and pre-allocates `kMaxInFlightSemaphoreCount` binary
  // semaphores.
  static StatusOr<ref_ptr<TimePointSemaphorePool>> Create(
      ref_ptr<VkDeviceHandle> logical_device);

  ~TimePointSemaphorePool();

  // Acquires a binary semaphore from the pool for use by the caller. The
  // semaphore is guaranteed to be in unsignaled state and not in-flight on GPU.
  //
  // Returns RESOURCE_EXHAUSTED if the pool has no more available semaphores.
  // Callers are expected to handle this by waiting on previous fences or for
  // complete device idle. Yes, that's as bad as it sounds, and if we start
  // seeing that we should bump up the max count.
  StatusOr<TimePointSemaphore*> Acquire();

  // Releases one or more semaphores back to the pool. The binary semaphore must
  // be unsignaled and not in flight on GPU.
  void ReleaseResolved(IntrusiveList<TimePointSemaphore>* semaphores);

  // Releases one or more semaphores back to the pool. These may be in any state
  // and will be assumed as untouchable; the pool will unconditionally recycle
  // them.
  void ReleaseUnresolved(IntrusiveList<TimePointSemaphore>* semaphores);

 private:
  explicit TimePointSemaphorePool(ref_ptr<VkDeviceHandle> logical_device);

  const ref_ptr<DynamicSymbols>& syms() const;

  Status PreallocateSemaphores() ABSL_LOCKS_EXCLUDED(mutex_);

  ref_ptr<VkDeviceHandle> logical_device_;

  absl::Mutex mutex_;

  std::array<TimePointSemaphore, kMaxInFlightSemaphoreCount> storage_
      ABSL_GUARDED_BY(mutex_);
  IntrusiveList<TimePointSemaphore> free_semaphores_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace vulkan
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_VULKAN_TIMEPOINT_UTIL_H_
