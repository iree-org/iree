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

#ifndef IREE_HAL_VULKAN_QUEUES_INFO_H_
#define IREE_HAL_VULKAN_QUEUES_INFO_H_

#include <vulkan/vulkan.h>

namespace iree {
namespace hal {
namespace vulkan {

// A list of queues within a specific queue family on a VkDevice.
struct QueuesInfo {
 public:
  uint32_t queue_family_index;
  uint32_t* queue_indices;
  uint32_t queue_indices_count;
};

}  // namespace vulkan
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_VULKAN_QUEUES_INFO_H_
