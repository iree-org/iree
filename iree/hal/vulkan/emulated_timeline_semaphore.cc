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
#include "iree/hal/vulkan/serializing_command_queue.h"
#include "iree/hal/vulkan/status_util.h"

namespace iree {
namespace hal {
namespace vulkan {

// static
void TimePointFence::Delete(TimePointFence* ptr) {
  ptr->getPool()->ReleaseResolved(ptr);
}

VkResult TimePointFence::GetStatus() {
  absl::MutexLock lock(&status_mutex_);
  if (status_ == VK_NOT_READY) {
    const auto& device = getPool()->logical_device();
    status_ = device->syms()->vkGetFenceStatus(*device, fence_);
  }
  return status_;
}

// static
StatusOr<ref_ptr<Semaphore>> EmulatedTimelineSemaphore::Create(
    ref_ptr<VkDeviceHandle> logical_device,
    ref_ptr<TimePointSemaphorePool> semaphore_pool, uint64_t initial_value) {
  IREE_TRACE_SCOPE0("EmulatedTimelineSemaphore::Create");
  return make_ref<EmulatedTimelineSemaphore>(
      std::move(logical_device), std::move(semaphore_pool), initial_value);
}

EmulatedTimelineSemaphore::EmulatedTimelineSemaphore(
    ref_ptr<VkDeviceHandle> logical_device,
    ref_ptr<TimePointSemaphorePool> semaphore_pool, uint64_t initial_value)
    : signaled_value_(initial_value),
      logical_device_(std::move(logical_device)),
      semaphore_pool_(std::move(semaphore_pool)) {}

EmulatedTimelineSemaphore::~EmulatedTimelineSemaphore() {
  IREE_TRACE_SCOPE0("EmulatedTimelineSemaphore::dtor");
  CHECK_OK(TryToAdvanceTimeline(UINT64_MAX));
  absl::MutexLock lock(&mutex_);
  CHECK(outstanding_semaphores_.empty())
      << "Destroying an emulated timeline semaphore without first waiting on "
         "outstanding signals";
}

StatusOr<uint64_t> EmulatedTimelineSemaphore::Query() {
  RETURN_IF_ERROR(TryToAdvanceTimeline(UINT64_MAX));
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

  // Inform all queues to make progress given we have a new value signaled now.
  for (CommandQueue* queue : logical_device_->queues()) {
    RETURN_IF_ERROR(
        static_cast<SerializingCommandQueue*>(queue)->AdvanceQueueSubmission());
  }

  return OkStatus();
}

Status EmulatedTimelineSemaphore::Wait(uint64_t value, absl::Time deadline) {
  IREE_TRACE_SCOPE0("EmulatedTimelineSemaphore::Wait");

  RETURN_IF_ERROR(TryToAdvanceTimeline(value));

  // Collect the fences associated with all the time points that are before the
  // deadline. We need to wait on them to be all signaled so that we can make
  // sure the timeline is advanced to the required value.

  absl::InlinedVector<VkFence, 4> fences;
  {
    absl::MutexLock lock(&mutex_);
    for (auto* semaphore : outstanding_semaphores_) {
      if (semaphore->value <= value) {
        fences.push_back(semaphore->wait_fence->value());
      } else {
        break;
      }
    }
  }

  if (fences.empty()) return OkStatus();

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
      *logical_device_, fences.size(), fences.data(), /*waitAll=*/true,
      timeout_nanos));

  RETURN_IF_ERROR(TryToAdvanceTimeline(value));
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

    if (!point->wait_fence)
      return InternalError("Time point wasn't waited before");
    point->wait_fence = nullptr;
    return OkStatus();
  }
  return InvalidArgumentError("No time point for the given semaphore");
}

StatusOr<VkSemaphore> EmulatedTimelineSemaphore::GetSignalSemaphore(
    uint64_t value, const ref_ptr<TimePointFence>& signal_fence) {
  IREE_TRACE_SCOPE0("EmulatedTimelineSemaphore::GetSignalSemaphore");

  assert(signaled_value_.load() < value);

  absl::MutexLock lock(&mutex_);

  auto insertion_point = outstanding_semaphores_.begin();
  while (insertion_point != outstanding_semaphores_.end()) {
    if ((*insertion_point)->value > value) break;
  }

  ASSIGN_OR_RETURN(TimePointSemaphore * semaphore, semaphore_pool_->Acquire());
  semaphore->value = value;
  semaphore->signal_fence = add_ref(signal_fence);
  assert(!semaphore->wait_fence);
  outstanding_semaphores_.insert(insertion_point, semaphore);

  return OkStatus();
}

Status EmulatedTimelineSemaphore::TryToAdvanceTimeline(
    uint64_t to_upper_value) {
  IREE_TRACE_SCOPE0("EmulatedTimelineSemaphore::TryToAdvanceTimeline");

  // We hold the lock during the entire resolve process so that we can resolve
  // to the furthest possible value.
  absl::MutexLock lock(&mutex_);

  // Fast path for when already signaled past the desired value or when we have
  // no outstanding semaphores.
  if (signaled_value_ >= to_upper_value || outstanding_semaphores_.empty())
    return OkStatus();

  uint64_t past_value = signaled_value_.load();

  IntrusiveList<TimePointSemaphore> resolved_semaphores;
  auto moveResolvedSemaphore = [&](TimePointSemaphore* s) {
    s->signal_fence = nullptr;
    s->wait_fence = nullptr;
    outstanding_semaphores_.erase(s);
    resolved_semaphores.push_back(s);
  };

  bool keep_resolving = true;
  while (keep_resolving && !outstanding_semaphores_.empty()) {
    auto* semaphore = outstanding_semaphores_.front();

    // If the current semaphore is for a value beyond our upper limit, then
    // early exit so that we don't spend time dealing with signals we don't yet
    // care about. This can prevent live lock where one thread is signaling
    // fences as fast/faster than another thread can consume them.
    if (semaphore->value > to_upper_value) {
      keep_resolving = false;
      break;
    }

    // If the current semaphore is for a value not greater than the past
    // signaled value, then we know it was signaled previously. But there might
    // be a waiter on it on GPU.
    if (semaphore->value <= past_value) {
      assert(!semaphore->signal_fence);

      // If ther is no waiters, we can recycle this semaphore now. If there
      // exists one waiter, then query its status and recycle on success. We
      // only handle success status here. Others will be handled when the fence
      // is checked for other semaphores' signaling status for the same queue
      // submission.
      if (!semaphore->wait_fence ||
          semaphore->wait_fence->GetStatus() == VK_SUCCESS) {
        moveResolvedSemaphore(semaphore);
      }

      continue;
    }

    // This semaphore represents a value gerater than the known previously
    // signaled value. We don't know its status so we need to really query now.

    assert(semaphore->signal_fence);
    VkResult signal_status = semaphore->signal_fence->GetStatus();

    switch (signal_status) {
      case VK_SUCCESS:
        signaled_value_.store(semaphore->value);
        semaphore->signal_fence = nullptr;
        // If no waiters, we can recycle this semaphore now.
        if (!semaphore->wait_fence) moveResolvedSemaphore(semaphore);
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
        status_ = VkResultToStatus(signal_status);
        signaled_value_.store(UINT64_MAX);
        semaphore->signal_fence = nullptr;
        // If no waiters, we can recycle this semaphore now.
        if (!semaphore->wait_fence) moveResolvedSemaphore(semaphore);
        break;
    }
  }

  semaphore_pool_->ReleaseResolved(&resolved_semaphores);
  if (!status_.ok()) {
    // TODO: release unresolved semaphores on failure
  }

  return status_;
}

// static
StatusOr<ref_ptr<TimePointFencePool>> TimePointFencePool::Create(
    ref_ptr<VkDeviceHandle> logical_device) {
  IREE_TRACE_SCOPE0("TimePointFencePool::Create");
  ref_ptr<TimePointFencePool> pool(
      new TimePointFencePool(std::move(logical_device)));
  RETURN_IF_ERROR(pool->PreallocateFences());
  return pool;
}

TimePointFencePool::~TimePointFencePool() {
  IREE_TRACE_SCOPE0("TimePointFencePool::dtor");

  absl::MutexLock lock(&mutex_);
  int free_count = 0;
  for (auto* fence : free_fences_) {
    syms()->vkDestroyFence(*logical_device_, fence->value(),
                           logical_device_->allocator());
    ++free_count;
  }
  assert(free_count == kMaxInFlightFenceCount && "not all fences are returned");
  free_fences_.clear();
}

StatusOr<ref_ptr<TimePointFence>> TimePointFencePool::Acquire() {
  IREE_TRACE_SCOPE0("TimePointFencePool::Acquire");

  absl::MutexLock lock(&mutex_);
  if (free_fences_.empty()) {
    return ResourceExhaustedErrorBuilder(IREE_LOC)
           << "Fence pool out of free fences";
  }

  auto* fence = free_fences_.front();
  free_fences_.pop_front();
  return add_ref(fence);
}

void TimePointFencePool::ReleaseResolved(TimePointFence* fence) {
  IREE_TRACE_SCOPE0("TimePointFencePool::ReleaseResolved");
  VkFence f = fence->value();
  syms()->vkResetFences(*logical_device_, 1, &f);
  absl::MutexLock lock(&mutex_);
  free_fences_.push_back(fence);
}

TimePointFencePool::TimePointFencePool(ref_ptr<VkDeviceHandle> logical_device)
    : logical_device_(std::move(logical_device)) {}

const ref_ptr<DynamicSymbols>& TimePointFencePool::syms() const {
  return logical_device_->syms();
}

Status TimePointFencePool::PreallocateFences() {
  IREE_TRACE_SCOPE0("TimePointFencePool::PreallocateFences");

  VkFenceCreateInfo create_info;
  create_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  create_info.pNext = nullptr;
  create_info.flags = 0;

  std::array<TimePointFence*, kMaxInFlightFenceCount> fences;
  {
    absl::MutexLock lock(&mutex_);
    for (int i = 0; i < kMaxInFlightFenceCount; ++i) {
      VkFence fence = VK_NULL_HANDLE;
      VK_RETURN_IF_ERROR(syms()->vkCreateFence(*logical_device_, &create_info,
                                               logical_device_->allocator(),
                                               &fence));
      fences[i] = new TimePointFence(this, fence);
    }
  }

  for (int i = 0; i < kMaxInFlightFenceCount; ++i) {
    // The `TimePointFence`s was created with an initial ref-count of one.
    // Decrease explicitly to zero so that later we can rely on the ref-count
    // reaching zero to auto-release the `TimePointFence` back to the free
    // list. As a nice side effect, this will also initialize the free list
    // with all newly created fences.
    // TODO: Might want to avoid acquiring and releasing the mutex for each
    // fence.
    fences[i]->ReleaseReference();
  }

  return OkStatus();
}

// static
StatusOr<ref_ptr<TimePointSemaphorePool>> TimePointSemaphorePool::Create(
    ref_ptr<VkDeviceHandle> logical_device) {
  IREE_TRACE_SCOPE0("TimePointSemaphorePool::Create");
  ref_ptr<TimePointSemaphorePool> pool(
      new TimePointSemaphorePool(std::move(logical_device)));
  RETURN_IF_ERROR(pool->PreallocateSemaphores());
  return pool;
}

TimePointSemaphorePool::~TimePointSemaphorePool() {
  IREE_TRACE_SCOPE0("TimePointSemaphorePool::dtor");

  absl::MutexLock lock(&mutex_);

  assert(free_semaphores_.size() == kMaxInFlightSemaphoreCount &&
         "not all semaphores are returned");
  free_semaphores_.clear();

  for (auto& semaphore : storage_) {
    syms()->vkDestroySemaphore(*logical_device_, semaphore.semaphore,
                               logical_device_->allocator());
  }
}

StatusOr<TimePointSemaphore*> TimePointSemaphorePool::Acquire() {
  IREE_TRACE_SCOPE0("TimePointSemaphorePool::Acquire");

  absl::MutexLock lock(&mutex_);
  if (free_semaphores_.empty()) {
    return ResourceExhaustedErrorBuilder(IREE_LOC)
           << "Semaphore pool out of free semaphores";
  }

  auto* semaphore = free_semaphores_.front();
  free_semaphores_.pop_front();
  return semaphore;
}

void TimePointSemaphorePool::ReleaseResolved(
    IntrusiveList<TimePointSemaphore>* semaphores) {
  IREE_TRACE_SCOPE0("TimePointSemaphorePool::ReleaseResolved");

  for (auto* semaphore : *semaphores) {
    assert(!semaphore->signal_fence && !semaphore->wait_fence);
    semaphore->value = UINT64_MAX;
  }

  absl::MutexLock lock(&mutex_);
  free_semaphores_.merge_from(semaphores);
}

TimePointSemaphorePool::TimePointSemaphorePool(
    ref_ptr<VkDeviceHandle> logical_device)
    : logical_device_(std::move(logical_device)) {}

const ref_ptr<DynamicSymbols>& TimePointSemaphorePool::syms() const {
  return logical_device_->syms();
}

Status TimePointSemaphorePool::PreallocateSemaphores() {
  IREE_TRACE_SCOPE0("TimePointSemaphorePool::PreallocateSemaphores");

  VkSemaphoreCreateInfo create_info;
  create_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  create_info.pNext = nullptr;
  create_info.flags = 0;

  absl::MutexLock lock(&mutex_);
  for (int i = 0; i < kMaxInFlightSemaphoreCount; ++i) {
    auto* semaphore = &storage_[i];
    VK_RETURN_IF_ERROR(syms()->vkCreateSemaphore(*logical_device_, &create_info,
                                                 logical_device_->allocator(),
                                                 &semaphore->semaphore));
    free_semaphores_.push_back(semaphore);
  }

  return OkStatus();
}

}  // namespace vulkan
}  // namespace hal
}  // namespace iree
