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

#include "iree/hal/vulkan/emulated_timeline_semaphore.h"

#include "absl/container/inlined_vector.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/utility/utility.h"
#include "iree/base/tracing.h"
#include "iree/hal/vulkan/dynamic_symbols.h"
#include "iree/hal/vulkan/status_util.h"

namespace iree {
namespace hal {
namespace vulkan {

// static
StatusOr<ref_ptr<Semaphore>> EmulatedTimelineSemaphore::Create(
    ref_ptr<VkDeviceHandle> logical_device,
    std::function<Status(Semaphore*)> on_signal,
    std::function<void(Semaphore*)> on_failure,
    ref_ptr<TimePointSemaphorePool> semaphore_pool, uint64_t initial_value) {
  IREE_TRACE_SCOPE0("EmulatedTimelineSemaphore::Create");
  return make_ref<EmulatedTimelineSemaphore>(
      std::move(logical_device), std::move(on_signal), std::move(on_failure),
      std::move(semaphore_pool), initial_value);
}

EmulatedTimelineSemaphore::EmulatedTimelineSemaphore(
    ref_ptr<VkDeviceHandle> logical_device,
    std::function<Status(Semaphore*)> on_signal,
    std::function<void(Semaphore*)> on_failure,
    ref_ptr<TimePointSemaphorePool> semaphore_pool, uint64_t initial_value)
    : signaled_value_(initial_value),
      logical_device_(std::move(logical_device)),
      on_signal_(std::move(on_signal)),
      on_failure_(std::move(on_failure)),
      semaphore_pool_(std::move(semaphore_pool)) {}

EmulatedTimelineSemaphore::~EmulatedTimelineSemaphore() {
  IREE_TRACE_SCOPE0("EmulatedTimelineSemaphore::dtor");
  CHECK_OK(TryToAdvanceTimeline(UINT64_MAX).status());
  absl::MutexLock lock(&mutex_);
  CHECK(outstanding_semaphores_.empty())
      << "Destroying an emulated timeline semaphore without first waiting on "
         "outstanding signals";
}

StatusOr<uint64_t> EmulatedTimelineSemaphore::Query() {
  RETURN_IF_ERROR(TryToAdvanceTimeline(UINT64_MAX).status());
  uint64_t value = signaled_value_.load();
  if (value == UINT64_MAX) {
    absl::MutexLock lock(&mutex_);
    return status_;
  }
  return value;
}

Status EmulatedTimelineSemaphore::Signal(uint64_t value) {
  IREE_TRACE_SCOPE0("EmulatedTimelineSemaphore::Signal");
  auto signaled_value = signaled_value_.exchange(value);
  // Make sure the previous signaled value is smaller than the new value.
  CHECK(signaled_value < value)
      << "Attempting to signal a timeline value out of order; trying " << value
      << " but " << signaled_value << " already signaled";

  // Inform the device to make progress given we have a new value signaled now.
  RETURN_IF_ERROR(on_signal_(this));

  return OkStatus();
}

Status EmulatedTimelineSemaphore::Wait(uint64_t value, absl::Time deadline) {
  IREE_TRACE_SCOPE0("EmulatedTimelineSemaphore::Wait");

  VkFence fence = VK_NULL_HANDLE;
  do {
    IREE_TRACE_SCOPE0("EmulatedTimelineSemaphore::Wait#loop");
    // First try to advance the timeline without blocking to see whether we've
    // already reached the desired value.
    ASSIGN_OR_RETURN(bool reached_desired_value, TryToAdvanceTimeline(value));
    if (reached_desired_value) return OkStatus();

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
        return InternalErrorBuilder(IREE_LOC)
               << "Timeline should have a signal fence for the first time "
                  "point beyond the signaled value";
      }
      fence = (*semaphore)->signal_fence->value();
      // Found; we can break the loop and proceed to waiting now.
      break;
    }
    // TODO(antiagainst): figure out a better way instead of the busy loop here.
  } while (absl::Now() < deadline);

  if (fence == VK_NULL_HANDLE) {
    return DeadlineExceededErrorBuilder(IREE_LOC)
           << "Deadline reached when waiting timeline semaphore";
  }

  uint64_t timeout_nanos;
  if (deadline == absl::InfiniteFuture()) {
    timeout_nanos = UINT64_MAX;
  } else if (deadline == absl::InfinitePast()) {
    timeout_nanos = 0;
  } else {
    auto relative_nanos = absl::ToInt64Nanoseconds(deadline - absl::Now());
    timeout_nanos = relative_nanos < 0 ? 0 : relative_nanos;
  }

  VK_RETURN_IF_ERROR(logical_device_->syms()->vkWaitForFences(
      *logical_device_, /*fenceCount=*/1, &fence, /*waitAll=*/true,
      timeout_nanos));

  RETURN_IF_ERROR(TryToAdvanceTimeline(value).status());
  return OkStatus();
}

void EmulatedTimelineSemaphore::Fail(Status status) {
  IREE_TRACE_SCOPE0("EmulatedTimelineSemaphore::Fail");
  absl::MutexLock lock(&mutex_);
  status_ = std::move(status);
  signaled_value_.store(UINT64_MAX);
}

VkSemaphore EmulatedTimelineSemaphore::GetWaitSemaphore(
    uint64_t value, const ref_ptr<TimePointFence>& wait_fence) {
  IREE_TRACE_SCOPE0("EmulatedTimelineSemaphore::GetWaitSemaphore");
  absl::MutexLock lock(&mutex_);

  VkSemaphore semaphore = VK_NULL_HANDLE;
  for (TimePointSemaphore* point : outstanding_semaphores_) {
    if (point->value > value && point->wait_fence) {
      point->wait_fence = add_ref(wait_fence);
      semaphore = point->semaphore;
      break;
    }
  }

  return semaphore;
}

Status EmulatedTimelineSemaphore::CancelWaitSemaphore(VkSemaphore semaphore) {
  IREE_TRACE_SCOPE0("EmulatedTimelineSemaphore::CancelWaitSemaphore");
  absl::MutexLock lock(&mutex_);
  for (TimePointSemaphore* point : outstanding_semaphores_) {
    if (point->semaphore != semaphore) continue;

    if (!point->wait_fence) {
      return InvalidArgumentErrorBuilder(IREE_LOC)
             << "Time point wasn't waited before";
    }
    point->wait_fence = nullptr;
    return OkStatus();
  }
  return InvalidArgumentErrorBuilder(IREE_LOC)
         << "No time point for the given semaphore";
}

StatusOr<VkSemaphore> EmulatedTimelineSemaphore::GetSignalSemaphore(
    uint64_t value, const ref_ptr<TimePointFence>& signal_fence) {
  IREE_TRACE_SCOPE0("EmulatedTimelineSemaphore::GetSignalSemaphore");

  if (signaled_value_.load() >= value) {
    return FailedPreconditionErrorBuilder(IREE_LOC)
           << "Timeline semaphore already signaled past " << value;
  }

  absl::MutexLock lock(&mutex_);

  auto insertion_point = outstanding_semaphores_.begin();
  while (insertion_point != outstanding_semaphores_.end()) {
    if ((*insertion_point)->value > value) break;
  }

  ASSIGN_OR_RETURN(TimePointSemaphore * semaphore, semaphore_pool_->Acquire());
  semaphore->value = value;
  semaphore->signal_fence = add_ref(signal_fence);
  if (semaphore->wait_fence) {
    return InternalErrorBuilder(IREE_LOC)
           << "Newly acquired time point semaphore should not have waiters";
  }
  outstanding_semaphores_.insert(insertion_point, semaphore);

  return semaphore->semaphore;
}

StatusOr<bool> EmulatedTimelineSemaphore::TryToAdvanceTimeline(
    uint64_t to_upper_value) {
  IREE_TRACE_SCOPE0("EmulatedTimelineSemaphore::TryToAdvanceTimeline");

  // We hold the lock during the entire resolve process so that we can resolve
  // to the furthest possible value.
  absl::MutexLock lock(&mutex_);

  uint64_t past_value = signaled_value_.load();

  // Fast path for when already signaled past the desired value.
  if (past_value >= to_upper_value) return true;

  // The timeline has not signaled past the desired value and there is no
  // binary semaphore pending on GPU yet: certainly the timeline cannot
  // advance to the desired value.
  if (outstanding_semaphores_.empty()) return false;

  IntrusiveList<TimePointSemaphore> resolved_semaphores;

  bool keep_resolving = true;
  bool reached_desired_value = false;
  while (keep_resolving && !outstanding_semaphores_.empty()) {
    auto* semaphore = outstanding_semaphores_.front();

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
        return InternalErrorBuilder(IREE_LOC)
               << "Timeline should already signaled past this time point and "
                  "cleared the signal fence";
      }

      // If ther is no waiters, we can recycle this semaphore now. If there
      // exists one waiter, then query its status and recycle on success. We
      // only handle success status here. Others will be handled when the fence
      // is checked for other semaphores' signaling status for the same queue
      // submission.
      if (!semaphore->wait_fence ||
          semaphore->wait_fence->GetStatus() == VK_SUCCESS) {
        semaphore->signal_fence = nullptr;
        semaphore->wait_fence = nullptr;
        outstanding_semaphores_.erase(semaphore);
        resolved_semaphores.push_back(semaphore);
      }

      continue;
    }

    // This semaphore represents a value gerater than the known previously
    // signaled value. We don't know its status so we need to really query now.

    if (!semaphore->signal_fence) {
      return InternalErrorBuilder(IREE_LOC)
             << "The status of this time point in the timeline should still be "
                "pending with a singal fence";
    }
    VkResult signal_status = semaphore->signal_fence->GetStatus();

    switch (signal_status) {
      case VK_SUCCESS:
        signaled_value_.store(semaphore->value);
        semaphore->signal_fence = nullptr;
        // If no waiters, we can recycle this semaphore now.
        if (!semaphore->wait_fence) {
          semaphore->signal_fence = nullptr;
          semaphore->wait_fence = nullptr;
          outstanding_semaphores_.erase(semaphore);
          resolved_semaphores.push_back(semaphore);
        }
        break;
      case VK_NOT_READY:
        // The fence has not been signaled yet so this is the furthest time
        // point we can go in this timeline.
        keep_resolving = false;
        break;
      default:
        // Fence indicates an error (device lost, out of memory, etc).
        // Propagate this back to our status (and thus any waiters).
        // Since we only take the first error we find we skip all remaining
        // fences.
        keep_resolving = false;
        semaphore->signal_fence = nullptr;
        status_ = VkResultToStatus(signal_status);
        signaled_value_.store(UINT64_MAX);
        break;
    }
  }

  semaphore_pool_->ReleaseResolved(&resolved_semaphores);
  if (!status_.ok()) {
    on_failure_(this);
    semaphore_pool_->ReleaseUnresolved(&outstanding_semaphores_);
    return status_;
  }

  return reached_desired_value;
}

}  // namespace vulkan
}  // namespace hal
}  // namespace iree
