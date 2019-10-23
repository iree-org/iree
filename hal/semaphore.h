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

#ifndef IREE_HAL_SEMAPHORE_H_
#define IREE_HAL_SEMAPHORE_H_

#include "absl/types/variant.h"
#include "hal/resource.h"

namespace iree {
namespace hal {

// A synchronization primitive used to indicate submission dependencies.
// Semaphores are either of type binary (signaled or unsignaled) or timeline
// (uint64 payload with >= semantics).
class Semaphore : public Resource {
 public:
};

// Binary semaphores have strict ordering requirements and must be carefully
// balanced. Each binary semaphore must only be waited on after a signal
// operation has been issued and each wait requires exactly one signal. They
// are commonly used only when interacting with external handles that may
// cross device or process boundaries.
class BinarySemaphore : public Semaphore {
 public:
};

// Timeline semaphores act as a fence along a per-semaphore timeline where
// signaling is done by setting the payload to a monotonically increasing
// 64-bit integer and waiting is done by blocking until the payload is set
// greater-than or equal-to the specified value. Timeline semaphores may be
// waited on or signaled in any order and can be significantly more
// efficient due to system-level coalescing.
class TimelineSemaphore : public Semaphore {
 public:
  // TODO(benvanik): add value query support.
  // TODO(benvanik): add host-side signal/wait.
};

// A reference to a strongly-typed semaphore and associated information.
// For TimelineSemaphores the provided payload is used to specify either the
// payload to wait for or new payload value.
using SemaphoreValue =
    absl::variant<BinarySemaphore*, std::pair<TimelineSemaphore*, uint64_t>>;

}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_SEMAPHORE_H_
