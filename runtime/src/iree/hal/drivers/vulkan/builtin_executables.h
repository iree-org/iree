// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_VULKAN_BUILTIN_EXECUTABLES_H_
#define IREE_HAL_DRIVERS_VULKAN_BUILTIN_EXECUTABLES_H_

#include <vector>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/vulkan/descriptor_set_arena.h"
#include "iree/hal/drivers/vulkan/dynamic_symbols.h"
#include "iree/hal/drivers/vulkan/handle_util.h"
#include "iree/hal/drivers/vulkan/util/ref_ptr.h"

namespace iree {
namespace hal {
namespace vulkan {

// The `maxBoundDescriptorSets` limit is 4 on many devices we support and we
// want to avoid conflicts with what the compiler uses, so we'll expect the
// compiler to have reserved the index 3 for our exclusive use.
#define IREE_HAL_VULKAN_BUILTIN_DESCRIPTOR_SET_COUNT 4
#define IREE_HAL_VULKAN_BUILTIN_DESCRIPTOR_SET 3

#define IREE_HAL_VULKAN_BUILTIN_PUSH_CONSTANT_COUNT 16

class BuiltinExecutables {
 public:
  BuiltinExecutables(VkDeviceHandle* logical_device);
  ~BuiltinExecutables();

  const ref_ptr<DynamicSymbols>& syms() const {
    return logical_device_->syms();
  }

  iree_status_t InitializeExecutables();

  // Fills a buffer without 4 byte offset or length requirements.
  //
  // This only implements the unaligned edges of fills, vkCmdFillBuffer should
  // be used for the aligned interior (if any).
  //
  // |push_constants_to_restore| will be pushed using vkCmdPushConstants over
  // the bytes used by this call.
  iree_status_t FillBufferUnaligned(
      VkCommandBuffer command_buffer, DescriptorSetArena* descriptor_set_arena,
      iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
      iree_device_size_t length, const void* pattern,
      iree_host_size_t pattern_length, const void* push_constants_to_restore);

 private:
  VkDeviceHandle* logical_device_ = NULL;

  iree_hal_descriptor_set_layout_t*
      descriptor_set_layouts_[IREE_HAL_VULKAN_BUILTIN_DESCRIPTOR_SET_COUNT] = {
          NULL};
  iree_hal_pipeline_layout_t* pipeline_layout_ = NULL;
  VkPipeline pipeline_ = VK_NULL_HANDLE;
};

}  // namespace vulkan
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_DRIVERS_VULKAN_BUILTIN_EXECUTABLES_H_
