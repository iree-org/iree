// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_VULKAN_DESCRIPTOR_SET_ARENA_H_
#define IREE_HAL_DRIVERS_VULKAN_DESCRIPTOR_SET_ARENA_H_

#include <stdint.h>

#include <array>
#include <vector>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/vulkan/descriptor_pool_cache.h"
#include "iree/hal/drivers/vulkan/dynamic_symbols.h"
#include "iree/hal/drivers/vulkan/handle_util.h"
#include "iree/hal/drivers/vulkan/native_executable.h"
#include "iree/hal/drivers/vulkan/util/arena.h"
#include "iree/hal/drivers/vulkan/util/ref_ptr.h"

namespace iree {
namespace hal {
namespace vulkan {

// A reusable arena for allocating descriptor sets and batching updates.
class DescriptorSetArena final {
 public:
  explicit DescriptorSetArena(DescriptorPoolCache* descriptor_pool_cache);
  ~DescriptorSetArena();

  // Allocates and binds a descriptor set from the arena.
  // The command buffer will have the descriptor set containing |bindings| bound
  // to it.
  iree_status_t BindDescriptorSet(
      VkCommandBuffer command_buffer,
      iree_hal_pipeline_layout_t* pipeline_layout, uint32_t set,
      iree_host_size_t binding_count,
      const iree_hal_descriptor_set_binding_t* bindings);

  // Flushes all pending writes to descriptor sets allocated from the arena and
  // returns a group that - when dropped - will release the descriptor sets
  // back to the pools they were allocated from.
  DescriptorSetGroup Flush();

 private:
  const DynamicSymbols& syms() const { return *logical_device_->syms(); }

  // Pushes the descriptor set to the command buffer, if supported.
  void PushDescriptorSet(VkCommandBuffer command_buffer,
                         iree_hal_pipeline_layout_t* pipeline_layout,
                         uint32_t set, iree_host_size_t binding_count,
                         const iree_hal_descriptor_set_binding_t* bindings);

  VkDeviceHandle* logical_device_;
  DescriptorPoolCache* descriptor_pool_cache_;

  // Arena used for temporary binding information used during allocation.
  Arena scratch_arena_;

  // A list of pools acquired on demand as different descriptor counts are
  // needed. Allocation granularity is max_descriptor_count=[8, 16, 32, 64].
  std::array<DescriptorPool, 4> descriptor_pool_buckets_;

  // All pools that have been used during allocation.
  std::vector<DescriptorPool> used_descriptor_pools_;
};

}  // namespace vulkan
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_DRIVERS_VULKAN_DESCRIPTOR_SET_ARENA_H_
