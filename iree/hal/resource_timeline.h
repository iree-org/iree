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

#ifndef IREE_HAL_RESOURCE_TIMELINE_H_
#define IREE_HAL_RESOURCE_TIMELINE_H_

#include "iree/base/ref_ptr.h"
#include "iree/hal/resource.h"
#include "iree/hal/resource_set.h"
#include "iree/hal/semaphore.h"

namespace iree {
namespace hal {

// Manages a device-specific timeline of semaphores to perform asynchronous
// resource garbage collection.
//
// Submissions are made with a list of Semaphores to signal to a new payload
// value. To keep the Resources required by the submission alive while the
// submission is in-progress the user must build a ResourceSet and attach it to
// the device-specific global resource timeline. Semaphore wait events will
// cause the resource timeline to try to advance time for all of the pending
// submissions. Once a Semaphore has met or exceeded the payload value of a
// previously-added ResourceSet it will be released.
//
// Thread-safe.
class ResourceTimeline : public RefObject<ResourceTimeline> {
 public:
  ResourceTimeline();
  ~ResourceTimeline();

  // Attaches a |resource_set| to the given |semaphore|. The |resource_set|
  // will be retained until the payload |value| specified is either met or
  // exceeded for the given semaphore.
  //
  // If the semaphore fails with an error (such as in the case of device loss)
  // the resource set will be released without ever being queried.
  Status AttachSet(ref_ptr<Semaphore> semaphore, uint64_t value,
                   ref_ptr<ResourceSet> resource_set);

  // Advances the value of a particular |semaphore|.
  // Any resource set that has been attached to a value less-than or equal to
  // the provided payload value will be released.
  Status AdvanceSemaphore(SemaphoreValue semaphore_value);

  // Processes any pending work after a WaitIdle has completed. Semaphores will
  // be checked to ensure they have completed. Note that because new work may
  // have been enqueued after the idle completed this call returning success
  // does not mean that there is no more work remaining.
  Status AwakeFromIdle();
};

}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_RESOURCE_TIMELINE_H_
