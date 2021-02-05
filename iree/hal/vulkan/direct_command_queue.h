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

#ifndef IREE_HAL_VULKAN_DIRECT_COMMAND_QUEUE_H_
#define IREE_HAL_VULKAN_DIRECT_COMMAND_QUEUE_H_

#include "iree/hal/vulkan/command_queue.h"
#include "iree/hal/vulkan/util/arena.h"

namespace iree {
namespace hal {
namespace vulkan {

// Command queue implementation directly maps to VkQueue.
class DirectCommandQueue final : public CommandQueue {
 public:
  DirectCommandQueue(VkDeviceHandle* logical_device, std::string name,
                     iree_hal_command_category_t supported_categories,
                     VkQueue queue);
  ~DirectCommandQueue() override;

  iree_status_t Submit(iree_host_size_t batch_count,
                       const iree_hal_submission_batch_t* batches) override;

  iree_status_t WaitIdle(iree_time_t deadline_ns) override;

 private:
  iree_status_t TranslateBatchInfo(
      const iree_hal_submission_batch_t* batch, VkSubmitInfo* submit_info,
      VkTimelineSemaphoreSubmitInfo* timeline_submit_info, Arena* arena);
};

}  // namespace vulkan
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_VULKAN_DIRECT_COMMAND_QUEUE_H_
