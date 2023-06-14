// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_VULKAN_COMMAND_QUEUE_H_
#define IREE_HAL_DRIVERS_VULKAN_COMMAND_QUEUE_H_

#include <string>

#include "iree/base/internal/synchronization.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/vulkan/dynamic_symbols.h"
#include "iree/hal/drivers/vulkan/handle_util.h"
#include "iree/hal/drivers/vulkan/tracing.h"
#include "iree/hal/drivers/vulkan/util/arena.h"

namespace iree {
namespace hal {
namespace vulkan {

class CommandQueue {
 public:
  virtual ~CommandQueue() {
    IREE_TRACE_SCOPE_NAMED("CommandQueue::dtor");
    iree_slim_mutex_lock(&queue_mutex_);
    syms()->vkQueueWaitIdle(queue_);
    iree_slim_mutex_unlock(&queue_mutex_);
    iree_slim_mutex_deinitialize(&queue_mutex_);
  }

  const ref_ptr<DynamicSymbols>& syms() const {
    return logical_device_->syms();
  }

  VkQueue handle() const { return queue_; }

  iree_hal_vulkan_tracing_context_t* tracing_context() {
    return tracing_context_;
  }
  void set_tracing_context(iree_hal_vulkan_tracing_context_t* tracing_context) {
    tracing_context_ = tracing_context;
  }

  bool can_dispatch() const {
    return iree_all_bits_set(supported_categories_,
                             IREE_HAL_COMMAND_CATEGORY_DISPATCH);
  }
  virtual iree_status_t Submit(iree_host_size_t batch_count,
                               const iree_hal_submission_batch_t* batches) = 0;

  virtual iree_status_t WaitIdle(iree_timeout_t timeout) = 0;

 protected:
  CommandQueue(VkDeviceHandle* logical_device,
               iree_hal_command_category_t supported_categories, VkQueue queue)
      : logical_device_(logical_device),
        supported_categories_(supported_categories),
        queue_(queue) {
    iree_slim_mutex_initialize(&queue_mutex_);
  }

  VkDeviceHandle* logical_device_;
  const iree_hal_command_category_t supported_categories_;

  iree_hal_vulkan_tracing_context_t* tracing_context_ = nullptr;

  // VkQueue needs to be externally synchronized.
  iree_slim_mutex_t queue_mutex_;
  VkQueue queue_ IREE_GUARDED_BY(queue_mutex_);
};

}  // namespace vulkan
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_DRIVERS_VULKAN_COMMAND_QUEUE_H_
