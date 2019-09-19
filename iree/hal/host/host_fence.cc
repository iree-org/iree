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

#include "iree/hal/host/host_fence.h"

#include <atomic>
#include <cstdint>

#include "absl/container/inlined_vector.h"
#include "absl/synchronization/mutex.h"
#include "iree/base/status.h"
#include "iree/base/tracing.h"

namespace iree {
namespace hal {

HostFence::HostFence(uint64_t initial_value) : value_(initial_value) {}

HostFence::~HostFence() = default;

Status HostFence::status() const {
  absl::MutexLock lock(&mutex_);
  return status_;
}

StatusOr<uint64_t> HostFence::QueryValue() {
  return value_.load(std::memory_order_acquire);
}

Status HostFence::Signal(uint64_t value) {
  absl::MutexLock lock(&mutex_);
  if (!status_.ok()) {
    return status_;
  }
  if (value_.exchange(value) >= value) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "Fence values must be monotonically increasing";
  }
  return OkStatus();
}

Status HostFence::Fail(Status status) {
  absl::MutexLock lock(&mutex_);
  status_ = status;
  value_.store(UINT64_MAX, std::memory_order_release);
  return OkStatus();
}

// static
Status HostFence::WaitForFences(absl::Span<const FenceValue> fences,
                                bool wait_all, absl::Time deadline) {
  IREE_TRACE_SCOPE0("HostFence::WaitForFences");

  // Some of the fences may already be signaled; we only need to wait for those
  // that are not yet at the expected value.
  using HostFenceValue = std::pair<HostFence*, uint64_t>;
  absl::InlinedVector<HostFenceValue, 4> waitable_fences;
  waitable_fences.reserve(fences.size());
  for (auto& fence_value : fences) {
    auto* fence = reinterpret_cast<HostFence*>(fence_value.first);
    ASSIGN_OR_RETURN(uint64_t current_value, fence->QueryValue());
    if (current_value == UINT64_MAX) {
      // Fence has failed. Return the error.
      return fence->status();
    } else if (current_value < fence_value.second) {
      // Fence has not yet hit the required value; wait for it.
      waitable_fences.push_back({fence, fence_value.second});
    }
  }

  // TODO(benvanik): maybe sort fences by value in case we are waiting on
  // multiple values from the same fence.

  // Loop over the fences and wait for them to complete.
  // TODO(b/140026716): add WaitHandle support for !wait_all (wait any).
  for (auto& fence_value : waitable_fences) {
    auto* fence = fence_value.first;
    absl::MutexLock lock(&fence->mutex_);
    if (!fence->mutex_.AwaitWithDeadline(
            absl::Condition(
                +[](HostFenceValue* fence_value) {
                  return fence_value->first->value_.load(
                             std::memory_order_acquire) >= fence_value->second;
                },
                &fence_value),
            deadline)) {
      return DeadlineExceededErrorBuilder(IREE_LOC)
             << "Deadline exceeded waiting for fences";
    }
    if (!fence->status_.ok()) {
      return fence->status_;
    }
  }

  return OkStatus();
}

}  // namespace hal
}  // namespace iree
