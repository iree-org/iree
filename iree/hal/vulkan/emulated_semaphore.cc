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

#include "iree/hal/vulkan/emulated_semaphore.h"

#include <inttypes.h>
#include <stdint.h>

#include "absl/base/thread_annotations.h"
#include "absl/container/inlined_vector.h"
#include "absl/synchronization/mutex.h"
#include "iree/base/status.h"
#include "iree/base/tracing.h"
#include "iree/hal/vulkan/dynamic_symbols.h"
#include "iree/hal/vulkan/serializing_command_queue.h"
#include "iree/hal/vulkan/status_util.h"
#include "iree/hal/vulkan/util/intrusive_list.h"
#include "iree/hal/vulkan/util/ref_ptr.h"

namespace iree {
namespace hal {
namespace vulkan {

class EmulatedTimelineSemaphore final {
 public:
  EmulatedTimelineSemaphore(VkDeviceHandle* logical_device,
                            TimePointSemaphorePool* semaphore_pool,
                            iree_host_size_t command_queue_count,
                            iree::hal::vulkan::CommandQueue** command_queues,
                            uint64_t initial_value);

  ~EmulatedTimelineSemaphore();

  iree_status_t Query(uint64_t* out_value);

  iree_status_t Signal(uint64_t value);

  iree_status_t Wait(uint64_t value, iree_time_t deadline_ns);

  void Fail(iree_status_t status);

  // Gets a binary semaphore for waiting on the timeline to advance to the given
  // |value|. The semaphore returned won't be waited by anyone else. Returns
  // VK_NULL_HANDLE if no available semaphores for the given |value|.
  // |wait_fence| is the fence associated with the queue submission that waiting
  // on this semaphore.
  VkSemaphore GetWaitSemaphore(uint64_t value,
                               const ref_ptr<TimePointFence>& wait_fence);

  // Cancels the waiting attempt on the given binary |semaphore|. This allows
  // the |semaphore| to be waited by others.
  iree_status_t CancelWaitSemaphore(VkSemaphore semaphore);

  // Gets a binary semaphore for signaling the timeline to the given |value|.
  // |value| must be smaller than the current timeline value. |signal_fence| is
  // the fence associated with the queue submission that signals this semaphore.
  iree_status_t GetSignalSemaphore(uint64_t value,
                                   const ref_ptr<TimePointFence>& signal_fence,
                                   VkSemaphore* out_handle);

 private:
  // Tries to advance the timeline to the given |to_upper_value| without
  // blocking and returns whether the |to_upper_value| is reached.
  iree_status_t TryToAdvanceTimeline(uint64_t to_upper_value,
                                     bool* out_reached_upper_value)
      ABSL_LOCKS_EXCLUDED(mutex_);
  // Similar to the above, but also returns the fences that are known to have
  // already signaled via |signaled_fences|.
  iree_status_t TryToAdvanceTimeline(
      uint64_t to_upper_value, bool* out_reached_upper_value,
      absl::InlinedVector<VkFence, 4>* out_signaled_fences)
      ABSL_LOCKS_EXCLUDED(mutex_);

  std::atomic<uint64_t> signaled_value_;

  VkDeviceHandle* logical_device_;
  TimePointSemaphorePool* semaphore_pool_;

  iree_host_size_t command_queue_count_;
  CommandQueue** command_queues_;

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
  iree_status_t status_ ABSL_GUARDED_BY(mutex_) = iree_ok_status();
};

EmulatedTimelineSemaphore::EmulatedTimelineSemaphore(
    VkDeviceHandle* logical_device, TimePointSemaphorePool* semaphore_pool,
    iree_host_size_t command_queue_count, CommandQueue** command_queues,
    uint64_t initial_value)
    : signaled_value_(initial_value),
      logical_device_(logical_device),
      semaphore_pool_(semaphore_pool),
      command_queue_count_(command_queue_count),
      command_queues_(command_queues) {}

EmulatedTimelineSemaphore::~EmulatedTimelineSemaphore() {
  IREE_TRACE_SCOPE0("EmulatedTimelineSemaphore::dtor");
  IREE_CHECK_OK(
      TryToAdvanceTimeline(UINT64_MAX, /*out_reached_upper_value=*/NULL));
  absl::MutexLock lock(&mutex_);
  IREE_CHECK(outstanding_semaphores_.empty())
      << "Destroying an emulated timeline semaphore without first waiting on "
         "outstanding signals";
  iree_status_free(status_);
}

iree_status_t EmulatedTimelineSemaphore::Query(uint64_t* out_value) {
  IREE_TRACE_SCOPE0("EmulatedTimelineSemaphore::Query");
  IREE_DVLOG(2) << "EmulatedTimelineSemaphore::Query";
  IREE_RETURN_IF_ERROR(
      TryToAdvanceTimeline(UINT64_MAX, /*out_reached_upper_value=*/NULL));
  uint64_t value = signaled_value_.load();
  IREE_DVLOG(2) << "Current timeline value: " << value;
  if (value == UINT64_MAX) {
    absl::MutexLock lock(&mutex_);
    return iree_status_clone(status_);
  }
  *out_value = value;
  return iree_ok_status();
}

iree_status_t EmulatedTimelineSemaphore::Signal(uint64_t value) {
  IREE_TRACE_SCOPE0("EmulatedTimelineSemaphore::Signal");
  IREE_DVLOG(2) << "EmulatedTimelineSemaphore::Signal";
  auto signaled_value = signaled_value_.exchange(value);
  IREE_DVLOG(2) << "Previous value: " << signaled_value
                << "; new value: " << value;
  // Make sure the previous signaled value is smaller than the new value.
  IREE_CHECK(signaled_value < value)
      << "Attempting to signal a timeline value out of order; trying " << value
      << " but " << signaled_value << " already signaled";

  // Inform the device to make progress given we have a new value signaled now.
  for (iree_host_size_t i = 0; i < command_queue_count_; ++i) {
    IREE_RETURN_IF_ERROR(((SerializingCommandQueue*)command_queues_[i])
                             ->AdvanceQueueSubmission());
  }

  return iree_ok_status();
}

iree_status_t EmulatedTimelineSemaphore::Wait(uint64_t value,
                                              iree_time_t deadline_ns) {
  IREE_TRACE_SCOPE0("EmulatedTimelineSemaphore::Wait");
  IREE_DVLOG(2) << "EmulatedTimelineSemaphore::Wait";

  VkFence fence = VK_NULL_HANDLE;
  do {
    IREE_TRACE_SCOPE0("EmulatedTimelineSemaphore::Wait#loop");
    // First try to advance the timeline without blocking to see whether we've
    // already reached the desired value.
    bool reached_desired_value = false;
    IREE_RETURN_IF_ERROR(TryToAdvanceTimeline(value, &reached_desired_value));
    if (reached_desired_value) return iree_ok_status();

    // We must wait now. Find the first emulated time point that has a value >=
    // the desired value so we can wait on its associated signal fence to make
    // sure the timeline is advanced to the desired value.
    absl::MutexLock lock(&mutex_);
    auto semaphore = outstanding_semaphores_.begin();
    for (; semaphore != outstanding_semaphores_.end(); ++semaphore) {
      if ((*semaphore)->value >= value) break;
    }
    if (semaphore != outstanding_semaphores_.end()) {
      if (!(*semaphore)->signal_fence) {
        return iree_make_status(IREE_STATUS_INTERNAL,
                                "timeline should have a signal fence for the "
                                "first time point beyond the signaled value");
      }
      IREE_DVLOG(2) << "Found timepoint semaphore " << *semaphore
                    << " (value: " << (*semaphore)->value
                    << ") to wait for desired timeline value: " << value;
      fence = (*semaphore)->signal_fence->value();
      // Found; we can break the loop and proceed to waiting now.
      break;
    }
    // TODO(antiagainst): figure out a better way instead of the busy loop here.
  } while (iree_time_now() < deadline_ns);

  if (fence == VK_NULL_HANDLE) {
    // NOTE: not an error; it may be expected that the semaphore is not ready.
    return iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED);
  }

  uint64_t timeout_ns =
      static_cast<uint64_t>(iree_absolute_deadline_to_timeout_ns(deadline_ns));
  VK_RETURN_IF_ERROR(logical_device_->syms()->vkWaitForFences(
                         *logical_device_, /*fenceCount=*/1, &fence,
                         /*waitAll=*/true, timeout_ns),
                     "vkWaitForFences");

  return TryToAdvanceTimeline(value, /*out_reached_upper_value=*/NULL);
}

void EmulatedTimelineSemaphore::Fail(iree_status_t status) {
  IREE_TRACE_SCOPE0("EmulatedTimelineSemaphore::Fail");
  absl::MutexLock lock(&mutex_);
  if (status_) return;
  status_ = status;
  signaled_value_.store(UINT64_MAX);
}

VkSemaphore EmulatedTimelineSemaphore::GetWaitSemaphore(
    uint64_t value, const ref_ptr<TimePointFence>& wait_fence) {
  IREE_TRACE_SCOPE0("EmulatedTimelineSemaphore::GetWaitSemaphore");
  IREE_DVLOG(2) << "EmulatedTimelineSemaphore::GetWaitSemaphore";

  absl::MutexLock lock(&mutex_);

  VkSemaphore semaphore = VK_NULL_HANDLE;
  for (TimePointSemaphore* point : outstanding_semaphores_) {
    if (point->value > value && point->wait_fence) {
      point->wait_fence = add_ref(wait_fence);
      semaphore = point->semaphore;
      break;
    }
  }

  IREE_DVLOG(2) << "Binary VkSemaphore to wait on for timeline value (" << value
                << ") and wait fence (" << wait_fence.get()
                << "): " << semaphore;

  return semaphore;
}

iree_status_t EmulatedTimelineSemaphore::CancelWaitSemaphore(
    VkSemaphore semaphore) {
  IREE_TRACE_SCOPE0("EmulatedTimelineSemaphore::CancelWaitSemaphore");
  IREE_DVLOG(2) << "EmulatedTimelineSemaphore::CancelWaitSemaphore";

  absl::MutexLock lock(&mutex_);
  for (TimePointSemaphore* point : outstanding_semaphores_) {
    if (point->semaphore != semaphore) continue;

    if (!point->wait_fence) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "time point wasn't waited before");
    }
    point->wait_fence = nullptr;
    IREE_DVLOG(2) << "Cancelled waiting on binary VkSemaphore: " << semaphore;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "no time point for the given semaphore");
}

iree_status_t EmulatedTimelineSemaphore::GetSignalSemaphore(
    uint64_t value, const ref_ptr<TimePointFence>& signal_fence,
    VkSemaphore* out_handle) {
  IREE_TRACE_SCOPE0("EmulatedTimelineSemaphore::GetSignalSemaphore");
  IREE_DVLOG(2) << "EmulatedTimelineSemaphore::GetSignalSemaphore";

  if (signaled_value_.load() >= value) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "timeline semaphore already signaled past %" PRIu64,
                            value);
  }

  absl::MutexLock lock(&mutex_);

  auto insertion_point = outstanding_semaphores_.begin();
  while (insertion_point != outstanding_semaphores_.end()) {
    if ((*insertion_point)->value > value) break;
  }

  TimePointSemaphore* semaphore = NULL;
  IREE_RETURN_IF_ERROR(semaphore_pool_->Acquire(&semaphore));
  semaphore->value = value;
  semaphore->signal_fence = add_ref(signal_fence);
  if (semaphore->wait_fence) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "newly acquired time point semaphore should not have waiters");
  }
  outstanding_semaphores_.insert(insertion_point, semaphore);
  IREE_DVLOG(2) << "Timepoint semaphore to signal for timeline value (" << value
                << ") and wait fence (" << signal_fence.get()
                << "): " << semaphore
                << " (binary VkSemaphore: " << semaphore->semaphore << ")";

  *out_handle = semaphore->semaphore;
  return iree_ok_status();
}

iree_status_t EmulatedTimelineSemaphore::TryToAdvanceTimeline(
    uint64_t to_upper_value, bool* out_reached_upper_value) {
  absl::InlinedVector<VkFence, 4> signaled_fences;
  iree_status_t status = TryToAdvanceTimeline(
      to_upper_value, out_reached_upper_value, &signaled_fences);
  // Inform the queue that some fences are known to have signaled. This should
  // happen here instead of inside the other TryToAdvanceTimeline to avoid
  // potential mutex deadlock, given here we are not holding a mutex anymore.
  if (!signaled_fences.empty()) {
    for (iree_host_size_t i = 0; i < command_queue_count_; ++i) {
      ((SerializingCommandQueue*)command_queues_[i])
          ->SignalFences(absl::MakeSpan(signaled_fences));
    }
  }
  return status;
}

iree_status_t EmulatedTimelineSemaphore::TryToAdvanceTimeline(
    uint64_t to_upper_value, bool* out_reached_upper_value,
    absl::InlinedVector<VkFence, 4>* out_signaled_fences) {
  IREE_TRACE_SCOPE0("EmulatedTimelineSemaphore::TryToAdvanceTimeline");
  IREE_DVLOG(3) << "EmulatedTimelineSemaphore::TryToAdvanceTimeline";
  if (out_reached_upper_value) *out_reached_upper_value = false;

  uint64_t past_value = signaled_value_.load();
  IREE_DVLOG(3) << "Current timeline value: " << past_value
                << "; desired timeline value: " << to_upper_value;

  // Fast path for when already signaled past the desired value.
  if (past_value >= to_upper_value) {
    if (out_reached_upper_value) *out_reached_upper_value = true;
    return iree_ok_status();
  }

  // We hold the lock during the entire resolve process so that we can resolve
  // to the furthest possible value.
  absl::MutexLock lock(&mutex_);

  IREE_DVLOG(3) << "# outstanding semaphores: "
                << outstanding_semaphores_.size();

  // The timeline has not signaled past the desired value and there is no
  // binary semaphore pending on GPU yet: certainly the timeline cannot
  // advance to the desired value.
  if (outstanding_semaphores_.empty()) return iree_ok_status();

  IntrusiveList<TimePointSemaphore> resolved_semaphores;

  auto clear_signal_fence =
      [&out_signaled_fences](ref_ptr<TimePointFence>& fence) {
        if (fence) {
          if (out_signaled_fences)
            out_signaled_fences->push_back(fence->value());
          fence.reset();
        }
      };

  bool keep_resolving = true;
  bool reached_desired_value = false;
  while (keep_resolving && !outstanding_semaphores_.empty()) {
    auto* semaphore = outstanding_semaphores_.front();
    IREE_DVLOG(3) << "Looking at timepoint semaphore " << semaphore << "..";
    IREE_DVLOG(3) << "  value: " << semaphore->value;
    IREE_DVLOG(3) << "  VkSemaphore: " << semaphore->semaphore;
    IREE_DVLOG(3) << "  signal fence: " << semaphore->signal_fence.get();
    IREE_DVLOG(3) << "  wait fence: " << semaphore->wait_fence.get();

    // If the current semaphore is for a value beyond our upper limit, then
    // early exit so that we don't spend time dealing with signals we don't yet
    // care about. This can prevent live lock where one thread is signaling
    // fences as fast/faster than another thread can consume them.
    if (semaphore->value > to_upper_value) {
      keep_resolving = false;
      reached_desired_value = true;
      break;
    }

    // If the current semaphore is for a value not greater than the past
    // signaled value, then we know it was signaled previously. But there might
    // be a waiter on it on GPU.
    if (semaphore->value <= past_value) {
      if (semaphore->signal_fence) {
        return iree_make_status(IREE_STATUS_INTERNAL,
                                "timeline should already signaled past this "
                                "time point and cleared the signal fence");
      }

      // If ther is no waiters, we can recycle this semaphore now. If there
      // exists one waiter, then query its status and recycle on success. We
      // only handle success status here. Others will be handled when the fence
      // is checked for other semaphores' signaling status for the same queue
      // submission.
      if (!semaphore->wait_fence ||
          semaphore->wait_fence->GetStatus() == VK_SUCCESS) {
        clear_signal_fence(semaphore->signal_fence);
        semaphore->wait_fence = nullptr;
        outstanding_semaphores_.erase(semaphore);
        resolved_semaphores.push_back(semaphore);
        IREE_DVLOG(3) << "Resolved and recycling semaphore " << semaphore;
      }

      continue;
    }

    // This semaphore represents a value gerater than the known previously
    // signaled value. We don't know its status so we need to really query now.

    if (!semaphore->signal_fence) {
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "status of this time point in the timeline "
                              "should still be pending with a singal fence");
    }
    VkResult signal_status = semaphore->signal_fence->GetStatus();

    switch (signal_status) {
      case VK_SUCCESS:
        IREE_DVLOG(3) << "..semaphore signaled";
        signaled_value_.store(semaphore->value);
        clear_signal_fence(semaphore->signal_fence);
        // If no waiters, we can recycle this semaphore now.
        if (!semaphore->wait_fence) {
          semaphore->wait_fence = nullptr;
          outstanding_semaphores_.erase(semaphore);
          resolved_semaphores.push_back(semaphore);
          IREE_DVLOG(3) << "Resolved and recycling semaphore " << semaphore;
        }
        break;
      case VK_NOT_READY:
        // The fence has not been signaled yet so this is the furthest time
        // point we can go in this timeline.
        keep_resolving = false;
        IREE_DVLOG(3) << "..semaphore not yet signaled";
        break;
      default:
        // Fence indicates an error (device lost, out of memory, etc).
        // Propagate this back to our status (and thus any waiters).
        // Since we only take the first error we find we skip all remaining
        // fences.
        keep_resolving = false;
        clear_signal_fence(semaphore->signal_fence);
        status_ = VK_RESULT_TO_STATUS(signal_status, "signal status");
        signaled_value_.store(UINT64_MAX);
        break;
    }
  }

  IREE_DVLOG(3) << "Releasing " << resolved_semaphores.size()
                << " resolved semaphores; " << outstanding_semaphores_.size()
                << " still outstanding";
  semaphore_pool_->ReleaseResolved(&resolved_semaphores);
  if (!iree_status_is_ok(status_)) {
    for (iree_host_size_t i = 0; i < command_queue_count_; ++i) {
      ((SerializingCommandQueue*)command_queues_[i])->AbortQueueSubmission();
    }
    semaphore_pool_->ReleaseUnresolved(&outstanding_semaphores_);
    return status_;
  }

  if (out_reached_upper_value) *out_reached_upper_value = reached_desired_value;
  return iree_ok_status();
}

}  // namespace vulkan
}  // namespace hal
}  // namespace iree

using namespace iree::hal::vulkan;

// Wrap the C++ type above so that we have a somewhat normal C interface.
// Porting the above to C is ideal but since this is just a fallback layer I'm
// not sure it's worth it (given that we may require Vulkan 1.2 with timeline
// semaphores built in at some point soon).
typedef struct {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  EmulatedTimelineSemaphore* handle;
} iree_hal_vulkan_emulated_semaphore_t;

extern const iree_hal_semaphore_vtable_t
    iree_hal_vulkan_emulated_semaphore_vtable;

static EmulatedTimelineSemaphore* iree_hal_vulkan_emulated_semaphore_cast(
    iree_hal_semaphore_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_vulkan_emulated_semaphore_vtable);
  return ((iree_hal_vulkan_emulated_semaphore_t*)base_value)->handle;
}

iree_status_t iree_hal_vulkan_emulated_semaphore_create(
    iree::hal::vulkan::VkDeviceHandle* logical_device,
    iree::hal::vulkan::TimePointSemaphorePool* semaphore_pool,
    iree_host_size_t command_queue_count,
    iree::hal::vulkan::CommandQueue** command_queues, uint64_t initial_value,
    iree_hal_semaphore_t** out_semaphore) {
  iree_hal_vulkan_emulated_semaphore_t* semaphore = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(logical_device->host_allocator(),
                                             sizeof(*semaphore),
                                             (void**)&semaphore));
  iree_hal_resource_initialize(&iree_hal_vulkan_emulated_semaphore_vtable,
                               &semaphore->resource);
  semaphore->host_allocator = logical_device->host_allocator();
  semaphore->handle = new EmulatedTimelineSemaphore(
      logical_device, semaphore_pool, command_queue_count, command_queues,
      initial_value);

  *out_semaphore = (iree_hal_semaphore_t*)semaphore;
  return iree_ok_status();
}

static void iree_hal_vulkan_emulated_semaphore_destroy(
    iree_hal_semaphore_t* base_semaphore) {
  iree_hal_vulkan_emulated_semaphore_t* semaphore =
      (iree_hal_vulkan_emulated_semaphore_t*)base_semaphore;
  iree_allocator_t host_allocator = semaphore->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  delete semaphore->handle;
  iree_allocator_free(host_allocator, semaphore);

  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_hal_vulkan_emulated_semaphore_acquire_wait_handle(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    const iree::ref_ptr<iree::hal::vulkan::TimePointFence>& wait_fence,
    VkSemaphore* out_handle) {
  EmulatedTimelineSemaphore* semaphore =
      iree_hal_vulkan_emulated_semaphore_cast(base_semaphore);
  *out_handle = semaphore->GetWaitSemaphore(value, wait_fence);
  return iree_ok_status();
}

iree_status_t iree_hal_vulkan_emulated_semaphore_cancel_wait_handle(
    iree_hal_semaphore_t* base_semaphore, VkSemaphore handle) {
  EmulatedTimelineSemaphore* semaphore =
      iree_hal_vulkan_emulated_semaphore_cast(base_semaphore);
  return semaphore->CancelWaitSemaphore(handle);
}

iree_status_t iree_hal_vulkan_emulated_semaphore_acquire_signal_handle(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    const iree::ref_ptr<iree::hal::vulkan::TimePointFence>& signal_fence,
    VkSemaphore* out_handle) {
  EmulatedTimelineSemaphore* semaphore =
      iree_hal_vulkan_emulated_semaphore_cast(base_semaphore);
  return semaphore->GetSignalSemaphore(value, signal_fence, out_handle);
}

static iree_status_t iree_hal_vulkan_emulated_semaphore_query(
    iree_hal_semaphore_t* base_semaphore, uint64_t* out_value) {
  EmulatedTimelineSemaphore* semaphore =
      iree_hal_vulkan_emulated_semaphore_cast(base_semaphore);
  return semaphore->Query(out_value);
}

static iree_status_t iree_hal_vulkan_emulated_semaphore_signal(
    iree_hal_semaphore_t* base_semaphore, uint64_t new_value) {
  EmulatedTimelineSemaphore* semaphore =
      iree_hal_vulkan_emulated_semaphore_cast(base_semaphore);
  return semaphore->Signal(new_value);
}

static void iree_hal_vulkan_emulated_semaphore_fail(
    iree_hal_semaphore_t* base_semaphore, iree_status_t status) {
  EmulatedTimelineSemaphore* semaphore =
      iree_hal_vulkan_emulated_semaphore_cast(base_semaphore);
  semaphore->Fail(status);
}

static iree_status_t iree_hal_vulkan_emulated_semaphore_wait_with_deadline(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    iree_time_t deadline_ns) {
  EmulatedTimelineSemaphore* semaphore =
      iree_hal_vulkan_emulated_semaphore_cast(base_semaphore);
  return semaphore->Wait(value, deadline_ns);
}

static iree_status_t iree_hal_vulkan_emulated_semaphore_wait_with_timeout(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    iree_duration_t timeout_ns) {
  return iree_hal_vulkan_emulated_semaphore_wait_with_deadline(
      base_semaphore, value, iree_relative_timeout_to_deadline_ns(timeout_ns));
}

iree_status_t iree_hal_vulkan_emulated_semaphore_multi_wait(
    iree::hal::vulkan::VkDeviceHandle* logical_device,
    const iree_hal_semaphore_list_t* semaphore_list, iree_time_t deadline_ns,
    VkSemaphoreWaitFlags wait_flags) {
  // TODO(antiagainst): We actually should get the fences associated with the
  // emulated timeline semaphores so that we can wait them in a bunch. This
  // implementation is problematic if we wait to wait any and we have the
  // first semaphore taking extra long time but the following ones signal
  // quickly.
  for (iree_host_size_t i = 0; i < semaphore_list->count; ++i) {
    IREE_RETURN_IF_ERROR(iree_hal_vulkan_emulated_semaphore_wait_with_deadline(
        semaphore_list->semaphores[i], semaphore_list->payload_values[i],
        deadline_ns));
    if (wait_flags & VK_SEMAPHORE_WAIT_ANY_BIT) return iree_ok_status();
  }
  return iree_ok_status();
}

const iree_hal_semaphore_vtable_t iree_hal_vulkan_emulated_semaphore_vtable = {
    /*.destroy=*/iree_hal_vulkan_emulated_semaphore_destroy,
    /*.query=*/iree_hal_vulkan_emulated_semaphore_query,
    /*.signal=*/iree_hal_vulkan_emulated_semaphore_signal,
    /*.fail=*/iree_hal_vulkan_emulated_semaphore_fail,
    /*.wait_with_deadline=*/
    iree_hal_vulkan_emulated_semaphore_wait_with_deadline,
    /*.wait_with_timeout=*/
    iree_hal_vulkan_emulated_semaphore_wait_with_timeout,
};
