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

#ifndef IREE_HAL_VULKAN_PIPELINE_EXECUTABLE_LAYOUT_H_
#define IREE_HAL_VULKAN_PIPELINE_EXECUTABLE_LAYOUT_H_

#include <vulkan/vulkan.h>

#include "absl/container/inlined_vector.h"
#include "absl/types/span.h"
#include "iree/hal/descriptor_set_layout.h"
#include "iree/hal/executable_layout.h"
#include "iree/hal/vulkan/handle_util.h"

namespace iree {
namespace hal {
namespace vulkan {

// A DescriptorSetLayout implemented with the native VkDescriptorSetLayout type.
class NativeDescriptorSetLayout final : public DescriptorSetLayout {
 public:
  NativeDescriptorSetLayout(ref_ptr<VkDeviceHandle> logical_device,
                            VkDescriptorSetLayout handle);
  ~NativeDescriptorSetLayout() override;

  VkDescriptorSetLayout handle() const { return handle_; }

 private:
  ref_ptr<VkDeviceHandle> logical_device_;
  VkDescriptorSetLayout handle_;
};

class PipelineExecutableLayout final : public ExecutableLayout {
 public:
  PipelineExecutableLayout(
      ref_ptr<VkDeviceHandle> logical_device, VkPipelineLayout handle,
      absl::InlinedVector<ref_ptr<NativeDescriptorSetLayout>, 2> set_layouts);
  ~PipelineExecutableLayout() override;

  VkPipelineLayout handle() const { return handle_; }

  absl::Span<const ref_ptr<NativeDescriptorSetLayout>> set_layouts() const {
    return set_layouts_;
  }

 private:
  ref_ptr<VkDeviceHandle> logical_device_;
  VkPipelineLayout handle_;
  absl::InlinedVector<ref_ptr<NativeDescriptorSetLayout>, 2> set_layouts_;
};

}  // namespace vulkan
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_VULKAN_PIPELINE_EXECUTABLE_LAYOUT_H_
