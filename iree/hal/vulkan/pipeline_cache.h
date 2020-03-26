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

#ifndef IREE_HAL_VULKAN_PIPELINE_CACHE_H_
#define IREE_HAL_VULKAN_PIPELINE_CACHE_H_

#include <vulkan/vulkan.h>

#include "absl/container/inlined_vector.h"
#include "iree/hal/executable.h"
#include "iree/hal/executable_cache.h"
#include "iree/hal/vulkan/handle_util.h"
#include "iree/hal/vulkan/pipeline_executable.h"
#include "iree/schemas/spirv_executable_def_generated.h"

namespace iree {
namespace hal {
namespace vulkan {

class PipelineCache final : public ExecutableCache {
 public:
  explicit PipelineCache(ref_ptr<VkDeviceHandle> logical_device);
  ~PipelineCache() override;

  const ref_ptr<DynamicSymbols>& syms() const {
    return logical_device_->syms();
  }

  bool CanPrepareFormat(ExecutableFormat format) const override;

  StatusOr<ref_ptr<Executable>> PrepareExecutable(
      ExecutableLayout* executable_layout, ExecutableCachingModeBitfield mode,
      const ExecutableSpec& spec) override;

 private:
  ref_ptr<VkDeviceHandle> logical_device_;
};

}  // namespace vulkan
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_VULKAN_PIPELINE_CACHE_H_
