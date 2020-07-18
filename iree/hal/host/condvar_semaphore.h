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

#ifndef IREE_HAL_HOST_CONDVAR_SEMAPHORE_H_
#define IREE_HAL_HOST_CONDVAR_SEMAPHORE_H_

#include <atomic>
#include <cstdint>

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "iree/base/status.h"
#include "iree/hal/semaphore.h"

namespace iree {
namespace hal {
namespace host {

// Simple host-only semaphore semaphore implemented with a mutex.
// Uses a condition variable to track the current value.
//
// Thread-safe (as instances may be imported and used by others).
class CondVarSemaphore final : public Semaphore {
 public:
  // Waits for one or more (or all) semaphores to reach or exceed the given
  // values.
  static Status WaitForSemaphores(absl::Span<const SemaphoreValue> semaphores,
                                  bool wait_all, Time deadline_ns);

  explicit CondVarSemaphore(uint64_t initial_value);
  ~CondVarSemaphore() override;

  StatusOr<uint64_t> Query() override;

  Status Signal(uint64_t value) override;
  void Fail(Status status) override;
  Status Wait(uint64_t value, Time deadline_ns) override;

 private:
  // The mutex is not required to query the value; this lets us quickly check if
  // a required value has been exceeded. The mutex is only used to update and
  // notify waiters.
  std::atomic<uint64_t> value_{0};

  // We have a full mutex here so that we can perform condvar waits on value
  // changes.
  mutable absl::Mutex mutex_;
  Status status_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace host
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_HOST_CONDVAR_SEMAPHORE_H_
