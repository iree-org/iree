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

#include "iree/hal/vulkan/legacy_fence.h"

#include <cstdint>

#include "absl/container/inlined_vector.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "iree/base/intrusive_list.h"
#include "iree/base/source_location.h"
#include "iree/base/status.h"
#include "iree/base/tracing.h"
#include "iree/hal/vulkan/status_util.h"

namespace iree {
namespace hal {
namespace vulkan {

namespace {

// Inserts the given |fence_signal| into |list| in ascending order.
void InsertOutstandingFenceSignal(OutstandingFenceSignal* fence_signal,
                                  IntrusiveList<OutstandingFenceSignal>* list) {
  for (auto existing_signal : *list) {
    if (existing_signal->value > fence_signal->value) {
      list->insert(existing_signal, fence_signal);
      return;
    }
  }
  list->push_back(fence_signal);
}

}  // namespace

// static
StatusOr<ref_ptr<LegacyFencePool>> LegacyFencePool::Create(
    ref_ptr<VkDeviceHandle> logical_device) {
  IREE_TRACE_SCOPE0("LegacyFencePool::Create");
  ref_ptr<LegacyFencePool> fence_pool(
      new LegacyFencePool(std::move(logical_device)));
  RETURN_IF_ERROR(fence_pool->PreallocateFences());
  return fence_pool;
}

LegacyFencePool::LegacyFencePool(ref_ptr<VkDeviceHandle> logical_device)
    : logical_device_(std::move(logical_device)) {}

LegacyFencePool::~LegacyFencePool() {
  IREE_TRACE_SCOPE0("LegacyFencePool::dtor");

  absl::MutexLock lock(&mutex_);
  for (auto& fence_signal : storage_) {
    syms()->vkDestroyFence(*logical_device_, fence_signal.fence,
                           logical_device_->allocator());
  }
  unused_fences_.clear();
  unresolved_fences_.clear();
}

Status LegacyFencePool::PreallocateFences() {
  IREE_TRACE_SCOPE0("LegacyFencePool::PreallocateFences");

  VkFenceCreateInfo create_info;
  create_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  create_info.pNext = nullptr;
  create_info.flags = 0;

  absl::MutexLock lock(&mutex_);
  for (int i = 0; i < kMaxInFlightFenceCount; ++i) {
    auto* fence_signal = &storage_[i];
    VK_RETURN_IF_ERROR(syms()->vkCreateFence(*logical_device_, &create_info,
                                             logical_device_->allocator(),
                                             &fence_signal->fence));
    unused_fences_.push_back(fence_signal);
  }

  return OkStatus();
}

StatusOr<OutstandingFenceSignal*> LegacyFencePool::Acquire() {
  IREE_TRACE_SCOPE0("LegacyFencePool::Acquire");

  absl::MutexLock lock(&mutex_);
  if (unused_fences_.empty()) {
    return ResourceExhaustedErrorBuilder(IREE_LOC)
           << "Fence pool out of unused fences";
  }

  auto* fence_signal = unused_fences_.front();
  unused_fences_.pop_front();
  return fence_signal;
}

void LegacyFencePool::ReleaseResolved(
    IntrusiveList<OutstandingFenceSignal>* fence_signals) {
  IREE_TRACE_SCOPE0("LegacyFencePool::ReleaseResolved");

  // Get a list of fences we need to reset. Note that not all fences may have
  // been signaled and we can avoid resetting them.
  absl::InlinedVector<VkFence, 8> handles;
  handles.reserve(fence_signals->size());
  for (auto* fence_signal : *fence_signals) {
    if (fence_signal->is_pending) {
      handles.push_back(fence_signal->fence);
    }
  }
  if (!handles.empty()) {
    syms()->vkResetFences(*logical_device_, handles.size(), handles.data());
  }

  absl::MutexLock lock(&mutex_);
  unused_fences_.merge_from(fence_signals);
}

void LegacyFencePool::ReleaseUnresolved(
    IntrusiveList<OutstandingFenceSignal>* fence_signals) {
  IREE_TRACE_SCOPE0("LegacyFencePool::ReleaseUnresolved");

  absl::MutexLock lock(&mutex_);
  while (!fence_signals->empty()) {
    auto* fence_signal = fence_signals->front();
    fence_signals->pop_front();
    if (fence_signal->is_pending) {
      // Fence was submitted and may still have a pending signal on it. We can't
      // reuse it until it has resolved.
      // TODO(benvanik): fix these fences by reallocating? We aren't leaking
      // here (technically) but we will exhaust the pool pretty quickly.
      unresolved_fences_.push_back(fence_signal);
    } else {
      // Fence was never actually submitted so we can reuse it no problem.
      unused_fences_.push_back(fence_signal);
    }
  }
}

// static
Status LegacyFence::WaitForFences(VkDeviceHandle* logical_device,
                                  absl::Span<const FenceValue> fences,
                                  bool wait_all, absl::Time deadline) {
  IREE_TRACE_SCOPE0("LegacyFence::WaitForFences");

  // NOTE: we could pool this state too (probably right on the LegacyFencePool)
  // or be smarter about using stack-allocated storage. The best idea is to use
  // real timeline semaphores, though, so not much effort has been spent on
  // optimizing this.
  absl::InlinedVector<VkFence, 4> handles;
  handles.reserve(fences.size());

  // Loop over the fences and wait for any/all to signal. In wait_all mode we
  // perform the bookkeeping to remove fences that have already been signaled so
  // that we only wait on ones we need to (and possibly avoid making the vk call
  // entirely!).
  while (true) {
    // Grab handles and acquire fences for all fences not yet at the requested
    // timeline value.
    for (const auto& fence_value : fences) {
      auto* fence = reinterpret_cast<LegacyFence*>(fence_value.first);
      // NOTE: this will return the sticky fence error if the fence has failed.
      ASSIGN_OR_RETURN(VkFence handle,
                       fence->AcquireWaitFence(fence_value.second));
      if (handle != VK_NULL_HANDLE) {
        // Fence is unresolved and we need to really wait for it.
        handles.push_back(handle);
      }
    }
    if (handles.empty()) {
      // All fences resolved.
      return OkStatus();
    }

    uint64_t timeout_nanos;
    if (deadline == absl::InfiniteFuture()) {
      timeout_nanos = UINT64_MAX;
    } else if (deadline == absl::InfinitePast()) {
      timeout_nanos = 0;
    } else {
      absl::Duration relative = deadline - absl::Now();
      timeout_nanos = absl::ToInt64Nanoseconds(relative) < 0
                          ? 0
                          : absl::ToInt64Nanoseconds(relative);
    }

    // Wait on the fences we still need.
    // Note that waking does not actually indicate all fences were hit! We need
    // to do another pass above on the next iteration to make sure that we don't
    // need to wait again on another fence.
    VK_RETURN_IF_ERROR(logical_device->syms()->vkWaitForFences(
        *logical_device, handles.size(), handles.data(), wait_all,
        timeout_nanos));
    handles.clear();
  }

  return OkStatus();
}

LegacyFence::LegacyFence(ref_ptr<LegacyFencePool> fence_pool,
                         uint64_t initial_value)
    : fence_pool_(std::move(fence_pool)), value_(initial_value) {}

LegacyFence::~LegacyFence() {
  IREE_TRACE_SCOPE0("LegacyFence::dtor");
  CHECK_OK(TryResolveOutstandingFences(UINT64_MAX));
  absl::MutexLock lock(&mutex_);
  CHECK(outstanding_signals_.empty())
      << "Destroying a fence without first waiting on outstanding signals";
}

Status LegacyFence::status() const {
  if (value_.load() != UINT64_MAX) {
    return OkStatus();
  }
  absl::MutexLock lock(&mutex_);
  return status_;
}

StatusOr<uint64_t> LegacyFence::QueryValue() {
  RETURN_IF_ERROR(TryResolveOutstandingFences(UINT64_MAX));
  return value_.load();
}

StatusOr<VkFence> LegacyFence::AcquireSignalFence(uint64_t value) {
  absl::MutexLock lock(&mutex_);

  // It's an error to signal out of order (as that requires a lot more
  // tracking and magic to get right).
  if (value_.load() >= value) {
    return FailedPreconditionErrorBuilder(IREE_LOC)
           << "Attempting to signal a timeline fence out of order; value="
           << value_ << ", new_value=" << value;
  }

  // Scan to see if there's waiters for this value (or values before it).
  // We may be able to reuse a previously allocated fence in the case that a
  // user is waiting prior to actually submitting the signal operation.
  OutstandingFenceSignal* signal_state = nullptr;
  for (auto* fence_signal : outstanding_signals_) {
    if (fence_signal->value == value) {
      // Fence is going to be signaled at exactly the required value.
      if (fence_signal->is_pending) {
        // Already have signaled to this value - that's a paddlin'.
        return FailedPreconditionErrorBuilder(IREE_LOC)
               << "Duplicate signal of timeline fence for value=" << value;
      }
      signal_state = fence_signal;
      break;
    }
  }
  if (!signal_state) {
    // Allocate a signal state entry and a VkFence to submit with.
    // TODO(benvanik): check for RESOURCE_EXHAUSTED and force a flush.
    ASSIGN_OR_RETURN(signal_state, fence_pool_->Acquire());
    signal_state->value = value;
    InsertOutstandingFenceSignal(signal_state, &outstanding_signals_);
  }

  signal_state->is_pending = true;
  return signal_state->fence;
}

StatusOr<VkFence> LegacyFence::AcquireWaitFence(uint64_t value) {
  // If we've already resolved then we want to avoid doing any kind of wait.
  // Since the value is monotonically increasing we can do a lock-free peek
  // here to see if we need to bother taking a full lock.
  if (value_.load() >= value) {
    return static_cast<VkFence>(VK_NULL_HANDLE);
  }

  absl::MutexLock lock(&mutex_);

  // Try to resolve any outstanding fence signals.
  RETURN_IF_ERROR(TryResolveOutstandingFencesLocked(value));
  if (value_.load() >= value) {
    return static_cast<VkFence>(VK_NULL_HANDLE);
  }

  // Try to find an existing fence we can reuse based on the required value.
  OutstandingFenceSignal* signal_state = nullptr;
  for (auto* fence_signal : outstanding_signals_) {
    if (fence_signal->value >= value) {
      // Fence is going to be signaled at or above the required value.
      signal_state = fence_signal;
      break;  // |outstanding_signals_| is in sorted order.
    }
  }
  if (!signal_state) {
    // Allocate a signal state entry and a VkFence that we will need to signal
    // in the future. We can't yet insert it into the queue but it will go in
    // when the user tries to signal a value >= the required value.
    // TODO(benvanik): check for RESOURCE_EXHAUSTED and force a flush.
    ASSIGN_OR_RETURN(signal_state, fence_pool_->Acquire());
    signal_state->value = value;
    InsertOutstandingFenceSignal(signal_state, &outstanding_signals_);
  }

  return signal_state->fence;
}

Status LegacyFence::TryResolveOutstandingFences(uint64_t upper_value) {
  absl::MutexLock lock(&mutex_);
  return TryResolveOutstandingFencesLocked(upper_value);
}

Status LegacyFence::TryResolveOutstandingFencesLocked(uint64_t upper_value) {
  // Fast-path for when we have no outstanding fences.
  // NOTE: we hold the lock during the entire resolve process so that any waiter
  // will only be woken once we have resolved to the furthest possible value.
  if (outstanding_signals_.empty() || value_ > upper_value) {
    return OkStatus();
  }

  IREE_TRACE_SCOPE0("LegacyFence::TryResolveOutstandingFences");

  IntrusiveList<OutstandingFenceSignal> resolved_fences;
  IntrusiveList<OutstandingFenceSignal> unresolved_fences;
  VkDevice device = *fence_pool_->logical_device();
  const auto& syms = fence_pool_->syms();
  bool keep_resolving = true;
  while (keep_resolving && !outstanding_signals_.empty()) {
    auto* fence_signal = outstanding_signals_.front();
    if (fence_signal->value > upper_value) {
      // Signal is for a value beyond our upper limit - early exit so that we
      // don't spend time dealing with signals we don't yet care about. This can
      // prevent live lock where one thread is signaling fences as fast/faster
      // than another thread can consume them.
      keep_resolving = false;
      break;
    }
    VkResult fence_status = syms->vkGetFenceStatus(device, fence_signal->fence);
    switch (fence_status) {
      case VK_SUCCESS: {
        // Fence has signaled meaning that we have reached this point in the
        // timeline and can advance the value.
        value_.store(fence_signal->value);
        outstanding_signals_.erase(fence_signal);
        resolved_fences.push_back(fence_signal);

        // Run backwards and resolve any non-pending fences as they will never
        // be used.
        for (auto* it = fence_signal; it != nullptr;) {
          auto* prev_fence_signal = it;
          it = outstanding_signals_.previous(it);
          if (!prev_fence_signal->is_pending) {
            outstanding_signals_.erase(prev_fence_signal);
            unresolved_fences.push_back(prev_fence_signal);
          }
        }
        break;
      }
      case VK_NOT_READY:
        if (fence_signal->is_pending) {
          // Fence has not yet been signaled. We stop here and wait for future
          // attempts at resolution.
          keep_resolving = false;
        }
        // Fence is not even pending yet - we may have skipped it. Keep
        // resolving to see if there's a higher value we can use.
        break;
      default:
        // Fence indicates an error (device lost, out of memory, etc).
        // Propagate this back to our status (and thus any waiters).
        // Since we only take the first error we find we skip all remaining
        // fences.
        status_ = VkResultToStatus(fence_status);
        value_.store(UINT64_MAX);
        outstanding_signals_.erase(fence_signal);
        resolved_fences.push_back(fence_signal);
        break;
    }
  }

  // Release resolved fences back to the pool. Note that we can only do this
  // to fences we know have actually completed: unresolved fences after an error
  // may still be in-flight and we don't want to reuse them.
  fence_pool_->ReleaseResolved(&resolved_fences);
  fence_pool_->ReleaseUnresolved(&unresolved_fences);
  if (!status_.ok()) {
    fence_pool_->ReleaseUnresolved(&outstanding_signals_);
  }

  return status_;
}

}  // namespace vulkan
}  // namespace hal
}  // namespace iree
