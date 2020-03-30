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

#include "absl/strings/str_cat.h"
#include "iree/hal/buffer.h"
#include "iree/hal/resource.h"

#ifndef IREE_HAL_DESCRIPTOR_SET_H_
#define IREE_HAL_DESCRIPTOR_SET_H_

namespace iree {
namespace hal {

// Opaque handle to a descriptor set object.
//
// Maps to VkDescriptorSet:
// https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VkDescriptorSet.html
class DescriptorSet : public Resource {
 public:
  // Specifies a descriptor set binding.
  struct Binding {
    // The binding number of this entry and corresponds to a resource of the
    // same binding number in the executable interface.
    int32_t binding = 0;
    // Buffer bound to the binding number.
    // May be nullptr if the binding is not used by the executable.
    Buffer* buffer;
    // Offset, in bytes, into the buffer that the binding starts at.
    // If the descriptor type is dynamic this will be added to the dynamic
    // offset provided during binding.
    device_size_t offset = 0;
    // Length, in bytes, of the buffer that is available to the executable.
    // This can be kWholeBuffer, however note that if the entire buffer
    // contents are larger than supported by the device (~128MiB, usually) this
    // will fail. If the descriptor type is dynamic this will be used for all
    // ranges regardless of offset.
    device_size_t length = kWholeBuffer;

    std::string DebugStringShort() const {
      return absl::StrCat("binding=", binding, ", ", buffer->DebugStringShort(),
                          ", offset=", offset, ", length=", length);
    }
  };
};

struct DescriptorSetBindingFormatter {
  void operator()(std::string* out,
                  const DescriptorSet::Binding& binding) const {
    out->append("<");
    out->append(binding.DebugStringShort());
    out->append(">");
  }
};

}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_DESCRIPTOR_SET_H_
