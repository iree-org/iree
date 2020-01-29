// Copyright 2020 Google LLC
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

#include "iree/hal/buffer.h"
#include "iree/hal/resource.h"

#ifndef IREE_HAL_DESCRIPTOR_SET_LAYOUT_H_
#define IREE_HAL_DESCRIPTOR_SET_LAYOUT_H_

namespace iree {
namespace hal {

// Specifies the type of a descriptor in a descriptor set.
enum class DescriptorType : uint32_t {
  kUniformBuffer = 6,
  kStorageBuffer = 7,
  kUniformBufferDynamic = 8,
  kStorageBufferDynamic = 9,
};

// Opaque handle to a descriptor set layout object.
//
// Maps to VkDescriptorSetLayout:
// https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VkDescriptorSetLayout.html
class DescriptorSetLayout : public Resource {
 public:
  // Specifies a descriptor set layout binding.
  struct Binding {
    // The binding number of this entry and corresponds to a resource of the
    // same binding number in the executable interface.
    int32_t binding = 0;
    // Specifies which type of resource descriptors are used for this binding.
    DescriptorType type = DescriptorType::kStorageBuffer;
    // Specifies the memory access performed by the executables.
    MemoryAccessBitfield access = MemoryAccess::kRead | MemoryAccess::kWrite;
  };
};

}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_DESCRIPTOR_SET_LAYOUT_H_
