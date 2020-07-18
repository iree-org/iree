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

#ifndef IREE_HAL_VULKAN_NATIVE_TIMELINE_SEMAPHORE_H_
#define IREE_HAL_VULKAN_NATIVE_TIMELINE_SEMAPHORE_H_

#include <vulkan/vulkan.h>

#include "absl/synchronization/mutex.h"
#include "iree/hal/semaphore.h"
#include "iree/hal/vulkan/handle_util.h"

namespace iree {
namespace hal {
namespace vulkan {

// A timeline semaphore implemented using the native VkSemaphore type.
// This may require emulation pre-Vulkan 1.2 when timeline semaphores were only
// an extension.
class NativeTimelineSemaphore final : public Semaphore {
 public:
  // Creates a timeline semaphore with the given |initial_value|.
  static StatusOr<ref_ptr<Semaphore>> Create(
      ref_ptr<VkDeviceHandle> logical_device, uint64_t initial_value);

  NativeTimelineSemaphore(ref_ptr<VkDeviceHandle> logical_device,
                          VkSemaphore handle, uint64_t initial_value);
  ~NativeTimelineSemaphore() override;

  VkSemaphore handle() const { return handle_; }

  StatusOr<uint64_t> Query() override;

  Status Signal(uint64_t value) override;
  void Fail(Status status) override;
  Status Wait(uint64_t value, Time deadline_ns) override;

 private:
  ref_ptr<VkDeviceHandle> logical_device_;
  VkSemaphore handle_;

  // NOTE: the Vulkan semaphore is the source of truth. We only need to access
  // this status (and thus take the lock) when we want to either signal failure
  // or query the status in the case of the semaphore being set to UINT64_MAX.
  mutable absl::Mutex status_mutex_;
  Status status_ ABSL_GUARDED_BY(status_mutex_);
};

}  // namespace vulkan
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_VULKAN_NATIVE_TIMELINE_SEMAPHORE_H_
