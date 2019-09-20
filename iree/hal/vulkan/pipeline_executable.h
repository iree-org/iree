// Copyright 2019 Google LLC
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

#ifndef IREE_HAL_VULKAN_PIPELINE_EXECUTABLE_H_
#define IREE_HAL_VULKAN_PIPELINE_EXECUTABLE_H_

#include <vulkan/vulkan.h>

#include <vector>

#include "absl/container/inlined_vector.h"
#include "iree/base/status.h"
#include "iree/hal/executable.h"
#include "iree/hal/executable_cache.h"
#include "iree/hal/executable_spec.h"
#include "iree/hal/vulkan/handle_util.h"
#include "iree/schemas/spirv_executable_def_generated.h"

namespace iree {
namespace hal {
namespace vulkan {

struct PipelineDescriptorSets {
  uint32_t buffer_binding_set;
  VkDescriptorSetLayout buffer_binding_set_layout;
  absl::InlinedVector<uint32_t, 8> buffer_binding_set_map;
};

class PipelineExecutable final : public Executable {
 public:
  static StatusOr<ref_ptr<PipelineExecutable>> Create(
      const ref_ptr<VkDeviceHandle>& logical_device,
      VkPipelineCache pipeline_cache, VkPipelineLayout pipeline_layout,
      PipelineDescriptorSets descriptor_sets,
      ExecutableCachingModeBitfield mode,
      const SpirVExecutableDef& spirv_executable_def);

  // Private constructor.
  struct CtorKey {
   private:
    friend class PipelineExecutable;
    CtorKey() = default;
  };
  PipelineExecutable(CtorKey ctor_key,
                     const ref_ptr<VkDeviceHandle>& logical_device,
                     VkPipelineLayout pipeline_layout,
                     PipelineDescriptorSets descriptor_sets,
                     absl::InlinedVector<VkPipeline, 1> pipelines);
  ~PipelineExecutable() override;

  const ref_ptr<DynamicSymbols>& syms() const {
    return logical_device_->syms();
  }

  bool supports_debugging() const override { return false; }

  VkPipelineLayout pipeline_layout() const { return pipeline_layout_; }
  const PipelineDescriptorSets& descriptor_sets() const {
    return descriptor_sets_;
  }

  bool is_matmul() const { return tag_ == "__matmul__"; }

  StatusOr<VkPipeline> GetPipelineForEntryPoint(int entry_ordinal) const;

 private:
  ref_ptr<VkDeviceHandle> logical_device_;
  VkPipelineLayout pipeline_layout_;
  PipelineDescriptorSets descriptor_sets_;
  std::string tag_;

  // One pipeline per entry point.
  absl::InlinedVector<VkPipeline, 1> pipelines_;
};

}  // namespace vulkan
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_VULKAN_PIPELINE_EXECUTABLE_H_
