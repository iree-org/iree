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

#ifndef IREE_HAL_METAL_METAL_PIPELINE_ARGUMENT_BUFFER_H_
#define IREE_HAL_METAL_METAL_PIPELINE_ARGUMENT_BUFFER_H_

#include <string>

#include "absl/container/inlined_vector.h"
#include "absl/types/span.h"
#include "iree/hal/cc/descriptor_set.h"
#include "iree/hal/cc/descriptor_set_layout.h"
#include "iree/hal/cc/executable_layout.h"

// Metal implementaion classes for resource descriptor related interfaces.
//
// See docs/design_docs/metal_hal_driver.md#resource-descriptors for more
// details.

namespace iree {
namespace hal {
namespace metal {

class MetalArgumentBufferLayout final : public DescriptorSetLayout {
 public:
  MetalArgumentBufferLayout(UsageType usage_type,
                            absl::Span<const Binding> bindings);
  ~MetalArgumentBufferLayout() override = default;

  absl::Span<const Binding> bindings() const { return bindings_; }
  const Binding* GetBindingForIndex(int index) const;

  std::string DebugString() const override;

 private:
  UsageType usage_type_;
  absl::InlinedVector<Binding, 8> bindings_;
};

class MetalPipelineArgumentBufferLayout final : public ExecutableLayout {
 public:
  MetalPipelineArgumentBufferLayout(
      absl::Span<DescriptorSetLayout* const> set_layouts,
      size_t push_constants);
  ~MetalPipelineArgumentBufferLayout() override;

  absl::Span<MetalArgumentBufferLayout* const> set_layouts() const {
    return set_layouts_;
  }

  std::string DebugString() const override;

 private:
  absl::InlinedVector<MetalArgumentBufferLayout*, 2> set_layouts_;
  size_t push_constants_;
};

class MetalArgumentBuffer final : public DescriptorSet {
 public:
  MetalArgumentBuffer(MetalArgumentBufferLayout* layout,
                      absl::Span<const Binding> resources);
  ~MetalArgumentBuffer() override;

 private:
  MetalArgumentBufferLayout* layout_;
  absl::InlinedVector<Binding, 8> resources_;
};

}  // namespace metal
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_METAL_METAL_PIPELINE_ARGUMENT_BUFFER_H_
