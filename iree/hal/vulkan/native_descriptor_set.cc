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

#include "iree/hal/vulkan/native_descriptor_set.h"

namespace iree {
namespace hal {
namespace vulkan {

NativeDescriptorSet::NativeDescriptorSet(ref_ptr<VkDeviceHandle> logical_device,
                                         VkDescriptorSet handle)
    : logical_device_(std::move(logical_device)), handle_(handle) {}

NativeDescriptorSet::~NativeDescriptorSet() {
  // TODO(benvanik): return to pool. For now we rely on the descriptor cache to
  // reset entire pools at once via via vkResetDescriptorPool so we don't need
  // to do anything here (the VkDescriptorSet handle will just be invalidated).
  // In the future if we want to have generational collection/defragmentation
  // of the descriptor cache we'll want to allow both pooled and unpooled
  // descriptors and clean them up here appropriately.
}

}  // namespace vulkan
}  // namespace hal
}  // namespace iree
