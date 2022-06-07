// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_VULKAN_DIRECT_COMMAND_QUEUE_H_
#define IREE_HAL_DRIVERS_VULKAN_DIRECT_COMMAND_QUEUE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/vulkan/command_queue.h"
#include "iree/hal/drivers/vulkan/handle_util.h"
#include "iree/hal/drivers/vulkan/util/arena.h"

namespace iree {
namespace hal {
namespace vulkan {

// Command queue implementation directly maps to VkQueue.
class DirectCommandQueue final : public CommandQueue {
 public:
  DirectCommandQueue(VkDeviceHandle* logical_device,
                     iree_hal_command_category_t supported_categories,
                     VkQueue queue);
  ~DirectCommandQueue() override;

  iree_status_t Submit(iree_host_size_t batch_count,
                       const iree_hal_submission_batch_t* batches) override;

  iree_status_t WaitIdle(iree_timeout_t timeout) override;

 private:
  iree_status_t TranslateBatchInfo(
      const iree_hal_submission_batch_t* batch, VkSubmitInfo* submit_info,
      VkTimelineSemaphoreSubmitInfo* timeline_submit_info, Arena* arena);
};

}  // namespace vulkan
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_DRIVERS_VULKAN_DIRECT_COMMAND_QUEUE_H_
