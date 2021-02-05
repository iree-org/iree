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

#ifndef IREE_HAL_VULKAN_DESCRIPTOR_SET_ARENA_H_
#define IREE_HAL_VULKAN_DESCRIPTOR_SET_ARENA_H_

#include <array>
#include <vector>

#include "iree/base/status.h"
#include "iree/hal/vulkan/descriptor_pool_cache.h"
#include "iree/hal/vulkan/native_executable.h"
#include "iree/hal/vulkan/util/arena.h"

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
  Status BindDescriptorSet(VkCommandBuffer command_buffer,
                           iree_hal_executable_layout_t* executable_layout,
                           uint32_t set, iree_host_size_t binding_count,
                           const iree_hal_descriptor_set_binding_t* bindings);

  // Flushes all pending writes to descriptor sets allocated from the arena and
  // returns a group that - when dropped - will release the descriptor sets
  // back to the pools they were allocated from.
  StatusOr<DescriptorSetGroup> Flush();

 private:
  const DynamicSymbols& syms() const { return *logical_device_->syms(); }

  // Pushes the descriptor set to the command buffer, if supported.
  Status PushDescriptorSet(VkCommandBuffer command_buffer,
                           iree_hal_executable_layout_t* executable_layout,
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
  absl::InlinedVector<DescriptorPool, 8> used_descriptor_pools_;
};

}  // namespace vulkan
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_VULKAN_DESCRIPTOR_SET_ARENA_H_
