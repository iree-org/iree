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

#include "iree/hal/metal/metal_pipeline_argument_buffer.h"

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"

namespace iree {
namespace hal {
namespace metal {

MetalArgumentBufferLayout::MetalArgumentBufferLayout(
    iree_hal_descriptor_set_layout_usage_type_t usage_type,
    absl::Span<const iree_hal_descriptor_set_layout_binding_t> bindings)
    : usage_type_(usage_type), bindings_(bindings.begin(), bindings.end()) {}

const iree_hal_descriptor_set_layout_binding_t*
MetalArgumentBufferLayout::GetBindingForIndex(int index) const {
  for (const auto& binding : bindings_) {
    if (binding.binding == index) return &binding;
  }
  return nullptr;
}

std::string MetalArgumentBufferLayout::DebugString() const {
  std::vector<std::string> binding_strings;
  binding_strings.reserve(bindings_.size());
  for (const auto& binding : bindings_) {
    binding_strings.push_back(
        absl::StrCat("[", binding.DebugStringShort(), "]"));
  }
  return absl::StrCat("bindings=[", absl::StrJoin(binding_strings, ", "), "]");
}

MetalPipelineArgumentBufferLayout::MetalPipelineArgumentBufferLayout(
    absl::Span<DescriptorSetLayout* const> set_layouts, size_t push_constants)
    : set_layouts_(set_layouts.size()), push_constants_(push_constants) {
  for (int i = 0; i < set_layouts.size(); ++i) {
    set_layouts_[i] = static_cast<MetalArgumentBufferLayout*>(set_layouts[i]);
    set_layouts_[i]->AddReference();
  }
}

MetalPipelineArgumentBufferLayout::~MetalPipelineArgumentBufferLayout() {
  for (auto* layout : set_layouts_) layout->ReleaseReference();
}

std::string MetalPipelineArgumentBufferLayout::DebugString() const {
  std::vector<std::string> set_strings;
  set_strings.reserve(set_layouts_.size());
  for (int i = 0; i < set_layouts_.size(); ++i) {
    set_strings.push_back(
        absl::StrCat("{set=", i, ", ", set_layouts_[i]->DebugString(), "}"));
  }
  return absl::StrCat("sets={", absl::StrJoin(set_strings, "; "), "}");
}

MetalArgumentBuffer::MetalArgumentBuffer(
    MetalArgumentBufferLayout* layout,
    absl::Span<const iree_hal_descriptor_set_binding_t> resources)
    : layout_(layout), resources_(resources.begin(), resources.end()) {
  layout_->AddReference();
}

MetalArgumentBuffer::~MetalArgumentBuffer() { layout_->ReleaseReference(); }

}  // namespace metal
}  // namespace hal
}  // namespace iree
