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

#ifndef IREE_HAL_FENCE_H_
#define IREE_HAL_FENCE_H_

#include <cstdint>

#include "iree/base/status.h"
#include "iree/hal/resource.h"

namespace iree {
namespace hal {

// Synchronization mechanism for device->host notification.
// Fences behave like timeline semaphores and contain a monotonically increasing
// uint64_t payload. They may be waited on any number of times - even if they
// have already been signaled.
//
// A fence is updated to its new value after all prior commands have completed
// but the delay between completion and the host being woken varies. Some
// implementations may coalesce fences to avoid spurious waking while others
// will immediately synchronize with the host.
//
// The primary use of fences is for resource lifetime management: all resources
// used by a set of submission batches must be considered live until the fence
// attached to the submission has signaled.
//
// Fences may be set to a permanently failed state by implementations when
// errors occur during asynchronous execution. Users are expected to propagate
// the failures and possibly reset the entire device that produced the error.
//
// For more information on fences see the following docs describing how
// timelines are generally used (specifically in the device->host case):
// https://www.youtube.com/watch?v=SpE--Rf516Y
// https://www.khronos.org/assets/uploads/developers/library/2018-xdc/Vulkan-Timeline-Semaphores-Part-1_Sep18.pdf
// https://docs.microsoft.com/en-us/windows/win32/direct3d12/user-mode-heap-synchronization
class Fence : public Resource {
 public:
  // Returns a permanent failure status if the fence is indicating an
  // asynchronous failure.
  //
  // Returns the status at the time the method is called without blocking and as
  // such is only valid after a fence has been signaled. The same failure status
  // will be returned regardless of when in the timeline the error occurred.
  virtual Status status() const = 0;

  // Queries the current payload of the fence. As the payload is monotonically
  // increasing it is guaranteed that the value is at least equal to the
  // previous result of a QueryValue call and coherent with any waits for a
  // specified value via Device::WaitAllFences.
  virtual StatusOr<uint64_t> QueryValue() = 0;
};

// A reference to a fence and associated payload value.
using FenceValue = std::pair<Fence*, uint64_t>;

}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_FENCE_H_
