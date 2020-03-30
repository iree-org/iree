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
#include "iree/hal/executable_layout.h"
#include "iree/hal/executable_spec.h"
#include "iree/hal/vulkan/handle_util.h"
#include "iree/hal/vulkan/native_descriptor_set.h"
#include "iree/hal/vulkan/pipeline_executable_layout.h"
#include "iree/schemas/spirv_executable_def_generated.h"

namespace iree {
namespace hal {
namespace vulkan {

class PipelineExecutable final : public Executable {
 public:
  static StatusOr<ref_ptr<PipelineExecutable>> Create(
      ref_ptr<VkDeviceHandle> logical_device, VkPipelineCache pipeline_cache,
      PipelineExecutableLayout* executable_layout,
      ExecutableCachingModeBitfield mode,
      const SpirVExecutableDef& spirv_executable_def);

  PipelineExecutable(ref_ptr<VkDeviceHandle> logical_device,
                     absl::InlinedVector<VkPipeline, 1> pipelines);
  ~PipelineExecutable() override;

  const ref_ptr<DynamicSymbols>& syms() const {
    return logical_device_->syms();
  }

  bool supports_debugging() const override { return false; }

  StatusOr<VkPipeline> GetPipelineForEntryPoint(int entry_ordinal) const;

 private:
  ref_ptr<VkDeviceHandle> logical_device_;
  std::string tag_;

  // One pipeline per entry point.
  absl::InlinedVector<VkPipeline, 1> pipelines_;
};

}  // namespace vulkan
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_VULKAN_PIPELINE_EXECUTABLE_H_
