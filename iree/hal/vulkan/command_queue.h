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

#ifndef IREE_HAL_VULKAN_COMMAND_QUEUE_H_
#define IREE_HAL_VULKAN_COMMAND_QUEUE_H_

#include <string>

#include "iree/base/status.h"
#include "iree/base/synchronization.h"
#include "iree/hal/api.h"
#include "iree/hal/vulkan/dynamic_symbols.h"
#include "iree/hal/vulkan/handle_util.h"
#include "iree/hal/vulkan/util/arena.h"

namespace iree {
namespace hal {
namespace vulkan {

class CommandQueue {
 public:
  virtual ~CommandQueue() {
    IREE_TRACE_SCOPE0("CommandQueue::dtor");
    iree_slim_mutex_lock(&queue_mutex_);
    syms()->vkQueueWaitIdle(queue_);
    iree_slim_mutex_unlock(&queue_mutex_);
    iree_slim_mutex_deinitialize(&queue_mutex_);
  }

  const ref_ptr<DynamicSymbols>& syms() const {
    return logical_device_->syms();
  }

  bool can_dispatch() const {
    return iree_all_bits_set(supported_categories_,
                             IREE_HAL_COMMAND_CATEGORY_DISPATCH);
  }
  virtual iree_status_t Submit(iree_host_size_t batch_count,
                               const iree_hal_submission_batch_t* batches) = 0;

  virtual iree_status_t WaitIdle(iree_time_t deadline_ns) = 0;

 protected:
  CommandQueue(VkDeviceHandle* logical_device, std::string name,
               iree_hal_command_category_t supported_categories, VkQueue queue)
      : logical_device_(logical_device),
        name_(std::move(name)),
        supported_categories_(supported_categories),
        queue_(queue) {
    iree_slim_mutex_initialize(&queue_mutex_);
  }

  VkDeviceHandle* logical_device_;
  const std::string name_;
  const iree_hal_command_category_t supported_categories_;

  // VkQueue needs to be externally synchronized.
  iree_slim_mutex_t queue_mutex_;
  VkQueue queue_ IREE_GUARDED_BY(queue_mutex_);
};

}  // namespace vulkan
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_VULKAN_COMMAND_QUEUE_H_
