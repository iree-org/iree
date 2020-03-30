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

#ifndef IREE_HAL_HOST_HOST_EXECUTABLE_LAYOUT_H_
#define IREE_HAL_HOST_HOST_EXECUTABLE_LAYOUT_H_

#include "absl/container/inlined_vector.h"
#include "iree/hal/descriptor_set_layout.h"
#include "iree/hal/executable_layout.h"

namespace iree {
namespace hal {

class HostDescriptorSetLayout final : public DescriptorSetLayout {
 public:
  HostDescriptorSetLayout(
      DescriptorSetLayout::UsageType usage_type,
      absl::Span<const DescriptorSetLayout::Binding> bindings);
  ~HostDescriptorSetLayout() override;

  absl::Span<const DescriptorSetLayout::Binding> bindings() const {
    return absl::MakeConstSpan(bindings_);
  }

 private:
  absl::InlinedVector<DescriptorSetLayout::Binding, 4> bindings_;
};

class HostExecutableLayout final : public ExecutableLayout {
 public:
  HostExecutableLayout(absl::Span<DescriptorSetLayout* const> set_layouts,
                       size_t push_constants);
  ~HostExecutableLayout() override;

  // Returns the total number of descriptor sets in the layout.
  size_t set_count() const { return dynamic_binding_map_.size(); }

  // Returns a map from dynamic offset index to the binding index in |set|.
  absl::Span<const int> GetDynamicBindingMap(int32_t set) const {
    return dynamic_binding_map_[set];
  }

  size_t push_constants() const { return push_constants_; }

 private:
  size_t push_constants_;
  absl::InlinedVector<absl::InlinedVector<int, 4>, 2> dynamic_binding_map_;
};

}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_HOST_HOST_EXECUTABLE_LAYOUT_H_
