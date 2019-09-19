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

#include "absl/base/thread_annotations.h"
#include "absl/container/inlined_vector.h"
#include "absl/synchronization/mutex.h"
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
  explicit PipelineCache(const ref_ptr<VkDeviceHandle>& logical_device);
  ~PipelineCache() override;

  const ref_ptr<DynamicSymbols>& syms() const {
    return logical_device_->syms();
  }

  bool CanPrepareFormat(ExecutableFormat format) const override;

  StatusOr<ref_ptr<Executable>> PrepareExecutable(
      ExecutableCachingModeBitfield mode, const ExecutableSpec& spec) override;

 private:
  struct CachedDescriptorSetLayout {
    absl::InlinedVector<VkDescriptorSetLayoutBinding, 4> bindings;
    VkDescriptorSetLayout descriptor_set_layout;
  };
  struct CachedPipelineLayout {
    absl::InlinedVector<VkDescriptorSetLayout, 4> descriptor_set_layouts;
    absl::InlinedVector<VkPushConstantRange, 1> push_constant_ranges;
    VkPipelineLayout pipeline_layout;
    PipelineDescriptorSets descriptor_sets;
  };

  StatusOr<const CachedPipelineLayout*> LookupOrInsertPipelineLayout(
      const VkPipelineLayoutDef& pipeline_layout_def)
      ABSL_LOCKS_EXCLUDED(mutex_);
  StatusOr<VkDescriptorSetLayout> LookupOrInsertDescriptorSetLayout(
      const VkDescriptorSetLayoutDef& descriptor_set_layout_def)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);
  void ClearLayoutCaches() ABSL_LOCKS_EXCLUDED(mutex_);

  ref_ptr<VkDeviceHandle> logical_device_;

  // A "cache" of descriptor set and pipeline layouts for various values.
  // We never evict and just do a simple linear scan on lookup. This is fine for
  // now as we only support a single descriptor type and really we only need to
  // check for binding count. As we go toward more general usage of descriptors
  // (images/etc) we will likely want to change this to a real cache.
  absl::Mutex mutex_;
  std::vector<CachedDescriptorSetLayout> descriptor_set_layout_cache_
      ABSL_GUARDED_BY(mutex_);
  std::vector<CachedPipelineLayout> pipeline_layout_cache_
      ABSL_GUARDED_BY(mutex_);
};

}  // namespace vulkan
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_VULKAN_PIPELINE_CACHE_H_
