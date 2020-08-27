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
#include "iree/hal/descriptor_set.h"
#include "iree/hal/descriptor_set_layout.h"
#include "iree/hal/executable_layout.h"

// Metal implementaion classes for resource descriptor related interfaces.
//
// A descriptor is an opaque handle pointing to a resource that are accessed in
// the compute kernel. IREE HAL is inspired by the Vulkan API; it models several
// concepts related to GPU resource management explicitly:
//
// * hal::DescriptorSetLayout: a schema for describing an array of descriptor
//   bindings. Each descriptor binding specifies the resource type, access mode
//   and other information.
// * hal::DescriptorSet: a concrete set of resources that get bound to a compute
//   pipeline in a batch. It must match the DescriptorSetLayout describing its
//   layout. DescriptorSet can be thought as the "object" from the
//   DescriptorSetLayout "class".
// * hal::ExecutableLayout: a schema for describing all the resource accessed by
//   a compute pipeline. It includes zero or more DescriptorSetLayouts and
//   (optional) push constants.
//
// One can create DescriptorSetLayout, DescriptorSet, and ExecutableLayout
// beforehand to avoid incur overhead during tight computing loops and also
// amortize the cost by sharing these objects. However, this isn't totally
// matching Metal's paradigm.
//
// In the Metal framework, the closet concept to DescriptorSet would be argument
// buffer. There is no direct correspondance to DescriptorSetLayout and
// ExecutableLayout. Rather, the layout is implicitly encoded in Metal shaders
// as MSL structs. The APIs for creating argument buffers does not encourage
// early creation without the pipeline: one typically create it for an
// MTLFunction. Besides, unlike Vulkan where different descriptor sets can have
// the same binding number, in Metal even we have multiple argument buffers, the
// indices for resources are in the same namespace: they are typically assigned
// sequentially. That means we need to remap hal::DescriptorSets with a set
// number greater than zero by applying an offset to each of its bindings.
//
// All of these means it's better to defer the creation of the argument buffer
// until the point of compute pipeline creation and dispatch. Therefore, we have
// the following classes bascially holding various information until they are
// needed.

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
