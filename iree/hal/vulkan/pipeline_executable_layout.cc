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

#include "iree/hal/vulkan/pipeline_executable_layout.h"

namespace iree {
namespace hal {
namespace vulkan {

NativeDescriptorSetLayout::NativeDescriptorSetLayout(
    ref_ptr<VkDeviceHandle> logical_device, VkDescriptorSetLayout handle)
    : logical_device_(std::move(logical_device)), handle_(handle) {}

NativeDescriptorSetLayout::~NativeDescriptorSetLayout() {
  logical_device_->syms()->vkDestroyDescriptorSetLayout(
      *logical_device_, handle_, logical_device_->allocator());
}

PipelineExecutableLayout::PipelineExecutableLayout(
    ref_ptr<VkDeviceHandle> logical_device, VkPipelineLayout handle,
    absl::InlinedVector<ref_ptr<NativeDescriptorSetLayout>, 2> set_layouts)
    : logical_device_(std::move(logical_device)),
      handle_(handle),
      set_layouts_(std::move(set_layouts)) {}

PipelineExecutableLayout::~PipelineExecutableLayout() {
  logical_device_->syms()->vkDestroyPipelineLayout(
      *logical_device_, handle_, logical_device_->allocator());
}

}  // namespace vulkan
}  // namespace hal
}  // namespace iree
