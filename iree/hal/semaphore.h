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

#include <cstdint>

#include "iree/base/status.h"
#include "iree/base/time.h"
#include "iree/hal/resource.h"

namespace iree {
namespace hal {

class Semaphore;

// A reference to a semaphore and associated payload value.
struct SemaphoreValue {
  Semaphore* semaphore = nullptr;
  uint64_t value = 0;
};

// Synchronization mechanism for host->device, device->host, host->host,
// and device->device notification. Semaphores behave like Vulkan timeline
// semaphores (or D3D12 fences) and contain a monotonically increasing
// uint64_t payload. They may be waited on any number of times even if they
// have already been signaled for a particular value. They may also be waited
// on for a particular value prior to the signal for that value.
//
// A semaphore is updated to its new value after all prior commands have
// completed but the delay between completion and the host being woken varies.
// Some implementations may coalesce semaphores to avoid spurious waking while
// others will immediately synchronize with the host.
//
// One use of semaphores is for resource lifetime management: all resources used
// by a set of submission batches must be considered live until the semaphore
// attached to the submission has signaled.
//
// Another use of semaphores is device->device synchronization for setting up
// the DAG of command buffers across queue submissions. This allows devices to
// perform non-trivial scheduling behavior without the need to wake the host.
//
// Semaphores may be set to a permanently failed state by implementations when
// errors occur during asynchronous execution. Users are expected to propagate
// the failures and possibly reset the entire device that produced the error.
//
// For more information on semaphores see the following docs describing how
// timelines are generally used (specifically in the device->host case):
// https://www.youtube.com/watch?v=SpE--Rf516Y
// https://www.khronos.org/assets/uploads/developers/library/2018-xdc/Vulkan-Timeline-Semaphores-Part-1_Sep18.pdf
// https://docs.microsoft.com/en-us/windows/win32/direct3d12/user-mode-heap-synchronization
class Semaphore : public Resource {
 public:
  // Queries the current payload of the semaphore. As the payload is
  // monotonically increasing it is guaranteed that the value is at least equal
  // to the previous result of a Query call and coherent with any waits for
  // a specified value via Device::WaitAllSemaphores.
  //
  // Returns the status/payload at the time the method is called without
  // blocking and as such is only valid after a semaphore has been signaled. The
  // same failure status will be returned regardless of when in the timeline the
  // error occurred.
  virtual StatusOr<uint64_t> Query() = 0;

  // Signals the semaphore to the given payload value.
  // The call is ignored if the current payload value exceeds |value|.
  virtual Status Signal(uint64_t value) = 0;

  // Signals the semaphore with a failure. The |status| will be returned from
  // Query and Signal for the lifetime of the semaphore.
  virtual void Fail(Status status) = 0;

  // Blocks the caller until the semaphore reaches or exceedes the specified
  // payload value or the |deadline_ns| elapses.
  //
  // Returns success if the wait is successful and the semaphore has met or
  // exceeded the required payload value.
  //
  // Returns DEADLINE_EXCEEDED if the |deadline_ns| elapses without the
  // semaphore reaching the required value.
  virtual Status Wait(uint64_t value, Time deadline_ns) = 0;
  inline Status Wait(uint64_t value, Duration timeout_ns) {
    return Wait(value, RelativeTimeoutToDeadlineNanos(timeout_ns));
  }
};

}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_SEMAPHORE_H_
