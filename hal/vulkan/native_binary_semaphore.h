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

#ifndef IREE_HAL_VULKAN_NATIVE_BINARY_SEMAPHORE_H_
#define IREE_HAL_VULKAN_NATIVE_BINARY_SEMAPHORE_H_

#include <vulkan/vulkan.h>

#include "hal/semaphore.h"
#include "hal/vulkan/handle_util.h"

namespace iree {
namespace hal {
namespace vulkan {

// A binary semaphore implemented using the native VkSemaphore type.
// This is supported unconditionally on all versions of Vulkan.
class NativeBinarySemaphore final : public BinarySemaphore {
 public:
  NativeBinarySemaphore(ref_ptr<VkDeviceHandle> logical_device,
                        VkSemaphore handle);
  ~NativeBinarySemaphore() override;

  VkSemaphore handle() const { return handle_; }

 private:
  ref_ptr<VkDeviceHandle> logical_device_;
  VkSemaphore handle_;
};

}  // namespace vulkan
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_VULKAN_NATIVE_BINARY_SEMAPHORE_H_
