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

#include "iree/hal/host/host_executable_layout.h"

#include "iree/base/memory.h"

namespace iree {
namespace hal {

HostDescriptorSetLayout::HostDescriptorSetLayout(
    DescriptorSetLayout::UsageType usage_type,
    absl::Span<const DescriptorSetLayout::Binding> bindings)
    : bindings_(bindings.begin(), bindings.end()) {}

HostDescriptorSetLayout::~HostDescriptorSetLayout() = default;

HostExecutableLayout::HostExecutableLayout(
    absl::Span<DescriptorSetLayout* const> set_layouts, size_t push_constants)
    : push_constants_(push_constants) {
  dynamic_binding_map_.resize(set_layouts.size());
  for (int i = 0; i < set_layouts.size(); ++i) {
    auto* set_layout = static_cast<HostDescriptorSetLayout*>(set_layouts[i]);
    auto& set_binding_map = dynamic_binding_map_[i];
    for (auto& binding : set_layout->bindings()) {
      if (binding.type == DescriptorType::kStorageBufferDynamic ||
          binding.type == DescriptorType::kUniformBufferDynamic) {
        set_binding_map.push_back(binding.binding);
      }
    }
  }
}

HostExecutableLayout::~HostExecutableLayout() = default;

}  // namespace hal
}  // namespace iree
