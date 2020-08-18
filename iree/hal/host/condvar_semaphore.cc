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

#include "iree/hal/host/condvar_semaphore.h"

#include <atomic>
#include <cstdint>

#include "absl/container/inlined_vector.h"
#include "absl/synchronization/mutex.h"
#include "iree/base/status.h"
#include "iree/base/tracing.h"

namespace iree {
namespace hal {
namespace host {

CondVarSemaphore::CondVarSemaphore(uint64_t initial_value)
    : value_(initial_value) {}

CondVarSemaphore::~CondVarSemaphore() = default;

StatusOr<uint64_t> CondVarSemaphore::Query() {
  absl::MutexLock lock(&mutex_);
  if (!status_.ok()) {
    return status_;
  }
  return value_.load(std::memory_order_acquire);
}

Status CondVarSemaphore::Signal(uint64_t value) {
  absl::MutexLock lock(&mutex_);
  if (!status_.ok()) {
    return status_;
  }
  if (value_.exchange(value) >= value) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "Semaphore values must be monotonically increasing";
  }
  return OkStatus();
}

void CondVarSemaphore::Fail(Status status) {
  absl::MutexLock lock(&mutex_);
  status_ = std::move(status);
  value_.store(UINT64_MAX, std::memory_order_release);
}

// static
Status CondVarSemaphore::WaitForSemaphores(
    absl::Span<const SemaphoreValue> semaphores, bool wait_all,
    Time deadline_ns) {
  IREE_TRACE_SCOPE0("CondVarSemaphore::WaitForSemaphores");

  // Some of the semaphores may already be signaled; we only need to wait for
  // those that are not yet at the expected value.
  using CondVarSemaphoreValue = std::pair<CondVarSemaphore*, uint64_t>;
  absl::InlinedVector<CondVarSemaphoreValue, 4> waitable_semaphores;
  waitable_semaphores.reserve(semaphores.size());
  for (auto& semaphore_value : semaphores) {
    auto* semaphore =
        reinterpret_cast<CondVarSemaphore*>(semaphore_value.semaphore);
    IREE_ASSIGN_OR_RETURN(uint64_t current_value, semaphore->Query());
    if (current_value < semaphore_value.value) {
      // Semaphore has not yet hit the required value; wait for it.
      waitable_semaphores.push_back({semaphore, semaphore_value.value});
    }
  }

  // TODO(benvanik): maybe sort semaphores by value in case we are waiting on
  // multiple values from the same semaphore.

  // Loop over the semaphores and wait for them to complete.
  // TODO(b/140026716): add WaitHandle support for !wait_all (wait any).
  for (auto& semaphore_value : waitable_semaphores) {
    auto* semaphore = semaphore_value.first;
    absl::MutexLock lock(&semaphore->mutex_);
    if (!semaphore->mutex_.AwaitWithDeadline(
            absl::Condition(
                +[](CondVarSemaphoreValue* semaphore_value) {
                  return semaphore_value->first->value_.load(
                             std::memory_order_acquire) >=
                         semaphore_value->second;
                },
                &semaphore_value),
            absl::FromUnixNanos(static_cast<int64_t>(deadline_ns)))) {
      return DeadlineExceededErrorBuilder(IREE_LOC)
             << "Deadline exceeded waiting for semaphores";
    }
    if (!semaphore->status_.ok()) {
      return semaphore->status_;
    }
  }

  return OkStatus();
}

Status CondVarSemaphore::Wait(uint64_t value, Time deadline_ns) {
  return WaitForSemaphores({{this, value}}, /*wait_all=*/true, deadline_ns);
}

}  // namespace host
}  // namespace hal
}  // namespace iree
