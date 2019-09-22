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

#include "iree/hal/vulkan/pipeline_cache.h"

#include "absl/synchronization/mutex.h"
#include "flatbuffers/flatbuffers.h"
#include "iree/base/source_location.h"
#include "iree/base/status.h"
#include "iree/base/tracing.h"
#include "iree/hal/executable_format.h"
#include "iree/hal/vulkan/status_util.h"
#include "iree/schemas/spirv_executable_def_generated.h"

namespace iree {
namespace hal {
namespace vulkan {

PipelineCache::PipelineCache(const ref_ptr<VkDeviceHandle>& logical_device)
    : logical_device_(add_ref(logical_device)) {}

PipelineCache::~PipelineCache() {
  IREE_TRACE_SCOPE0("PipelineCache::dtor");
  ClearLayoutCaches();
}

bool PipelineCache::CanPrepareFormat(ExecutableFormat format) const {
  return format == kExecutableFormatSpirV;
}

StatusOr<ref_ptr<Executable>> PipelineCache::PrepareExecutable(
    ExecutableCachingModeBitfield mode, const ExecutableSpec& spec) {
  IREE_TRACE_SCOPE0("PipelineCache::PrepareExecutable");
  if (!CanPrepareFormat(spec.format)) {
    return UnimplementedErrorBuilder(IREE_LOC)
           << "Unsupported 4CC format: 0x" << std::hex << spec.format;
  }
  if (spec.executable_data.size() <= 4 ||
      !SpirVExecutableDefBufferHasIdentifier(spec.executable_data.data())) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "Supplied executable data does not contain a SpirVExecutableDef";
  }

  // Get the SPIR-V executable def flatbuffer.
  const auto& spirv_executable_def =
      *::flatbuffers::GetRoot<SpirVExecutableDef>(spec.executable_data.data());

  // Create (or reuse) a pipeline layout.
  if (!spirv_executable_def.pipeline_layout()) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "Missing pipeline layout def";
  }
  ASSIGN_OR_RETURN(
      auto pipeline_layout_entry,
      LookupOrInsertPipelineLayout(*spirv_executable_def.pipeline_layout()));

  // Create the executable (which may itself own many pipelines).
  ASSIGN_OR_RETURN(auto executable, PipelineExecutable::Create(
                                        logical_device_,
                                        /*pipeline_cache=*/VK_NULL_HANDLE,
                                        pipeline_layout_entry->pipeline_layout,
                                        pipeline_layout_entry->descriptor_sets,
                                        mode, spirv_executable_def));
  return executable;
}

StatusOr<const PipelineCache::CachedPipelineLayout*>
PipelineCache::LookupOrInsertPipelineLayout(
    const VkPipelineLayoutDef& pipeline_layout_def) {
  IREE_TRACE_SCOPE0("PipelineCache::LookupOrInsertPipelineLayout");
  absl::MutexLock lock(&mutex_);

  // Build a list of the required descriptor set layouts and push constants.
  // If we were being fast about this we would just hash the def and directly
  // look up the pipeline layout.
  PipelineDescriptorSets descriptor_sets;
  descriptor_sets.buffer_binding_set = pipeline_layout_def.buffer_binding_set();
  descriptor_sets.buffer_binding_set_layout = VK_NULL_HANDLE;
  absl::InlinedVector<VkDescriptorSetLayout, 4> descriptor_set_layouts;
  if (pipeline_layout_def.descriptor_set_layouts()) {
    const auto& layout_defs = *pipeline_layout_def.descriptor_set_layouts();
    descriptor_set_layouts.resize(layout_defs.size());
    for (int i = 0; i < descriptor_set_layouts.size(); ++i) {
      if (!layout_defs[i]) {
        return InvalidArgumentErrorBuilder(IREE_LOC) << "Missing layout def";
      }
      ASSIGN_OR_RETURN(descriptor_set_layouts[i],
                       LookupOrInsertDescriptorSetLayout(*layout_defs[i]));
      if (i == pipeline_layout_def.buffer_binding_set()) {
        descriptor_sets.buffer_binding_set_layout = descriptor_set_layouts[i];
        descriptor_sets.buffer_binding_set_map.resize(
            layout_defs[i]->bindings()->size());
        for (int j = 0; j < layout_defs[i]->bindings()->size(); ++j) {
          descriptor_sets.buffer_binding_set_map[j] =
              layout_defs[i]->bindings()->Get(j)->binding();
        }
      }
    }
  }

  absl::InlinedVector<VkPushConstantRange, 1> push_constant_ranges;
  if (pipeline_layout_def.push_constant_ranges()) {
    const auto& range_defs = *pipeline_layout_def.push_constant_ranges();
    push_constant_ranges.resize(range_defs.size());
    for (int i = 0; i < push_constant_ranges.size(); ++i) {
      if (!range_defs[i]) {
        return InvalidArgumentErrorBuilder(IREE_LOC)
               << "Missing push constant range def";
      }
      push_constant_ranges[i].stageFlags = range_defs[i]->stage_flags();
      push_constant_ranges[i].offset = range_defs[i]->offset();
      push_constant_ranges[i].size = range_defs[i]->size();
    }
  }

  // Scan for an existing pipeline layout that matches the descriptor sets.
  for (auto& entry : pipeline_layout_cache_) {
    if (entry.descriptor_set_layouts.size() != descriptor_set_layouts.size() ||
        entry.push_constant_ranges.size() != push_constant_ranges.size()) {
      continue;
    }
    if (std::memcmp(
            descriptor_set_layouts.data(), entry.descriptor_set_layouts.data(),
            descriptor_set_layouts.size() * sizeof(VkDescriptorSetLayout)) ==
            0 &&
        std::memcmp(
            push_constant_ranges.data(), entry.push_constant_ranges.data(),
            push_constant_ranges.size() * sizeof(VkPushConstantRange)) == 0) {
      return &entry;
    }
  }

  VkPipelineLayoutCreateInfo create_info;
  create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  create_info.pNext = nullptr;
  create_info.flags = 0;
  create_info.setLayoutCount = descriptor_set_layouts.size();
  create_info.pSetLayouts = descriptor_set_layouts.data();
  create_info.pushConstantRangeCount = push_constant_ranges.size();
  create_info.pPushConstantRanges = push_constant_ranges.data();

  // Create and insert into the cache.
  VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
  VK_RETURN_IF_ERROR(syms()->vkCreatePipelineLayout(
      *logical_device_, &create_info, logical_device_->allocator(),
      &pipeline_layout));
  pipeline_layout_cache_.push_back({std::move(descriptor_set_layouts),
                                    std::move(push_constant_ranges),
                                    pipeline_layout, descriptor_sets});
  return &pipeline_layout_cache_.back();
}

StatusOr<VkDescriptorSetLayout>
PipelineCache::LookupOrInsertDescriptorSetLayout(
    const VkDescriptorSetLayoutDef& descriptor_set_layout_def) {
  // Build a list of bindings in the set.
  // If we were being fast we would hash the bindings and directly lookup
  // without doing this allocation.
  absl::InlinedVector<VkDescriptorSetLayoutBinding, 4> bindings;
  if (descriptor_set_layout_def.bindings()) {
    const auto& binding_defs = *descriptor_set_layout_def.bindings();
    bindings.resize(binding_defs.size());
    for (int i = 0; i < binding_defs.size(); ++i) {
      bindings[i].binding = binding_defs[i]->binding();
      bindings[i].descriptorType =
          static_cast<VkDescriptorType>(binding_defs[i]->descriptor_type());
      bindings[i].descriptorCount = binding_defs[i]->descriptor_count();
      bindings[i].stageFlags = binding_defs[i]->stage_flags();
      bindings[i].pImmutableSamplers = nullptr;
    }
  }

  // Scan for an existing descriptor set layout that matches the bindings.
  for (auto& entry : descriptor_set_layout_cache_) {
    if (entry.bindings.size() != bindings.size()) continue;
    if (std::memcmp(bindings.data(), entry.bindings.data(),
                    bindings.size() * sizeof(VkDescriptorSetLayoutBinding)) ==
        0) {
      return entry.descriptor_set_layout;
    }
  }

  VkDescriptorSetLayoutCreateInfo create_info;
  create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  create_info.pNext = nullptr;
  create_info.flags = 0;
  if (logical_device_->enabled_extensions().push_descriptors) {
    // Note that we can *only* use push descriptor sets if we set this create
    // flag. That's fine, though, as the command buffer recording logic always
    // prefers the extension if available.
    create_info.flags |=
        VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
  }
  create_info.bindingCount = bindings.size();
  create_info.pBindings = bindings.data();

  // Create and insert into the cache.
  VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
  VK_RETURN_IF_ERROR(syms()->vkCreateDescriptorSetLayout(
      *logical_device_, &create_info, logical_device_->allocator(),
      &descriptor_set_layout));
  descriptor_set_layout_cache_.push_back(
      {std::move(bindings), descriptor_set_layout});
  return descriptor_set_layout;
}

void PipelineCache::ClearLayoutCaches() {
  absl::MutexLock lock(&mutex_);
  for (auto& entry : pipeline_layout_cache_) {
    syms()->vkDestroyPipelineLayout(*logical_device_, entry.pipeline_layout,
                                    logical_device_->allocator());
  }
  pipeline_layout_cache_.clear();
  for (auto& entry : descriptor_set_layout_cache_) {
    syms()->vkDestroyDescriptorSetLayout(*logical_device_,
                                         entry.descriptor_set_layout,
                                         logical_device_->allocator());
  }
  descriptor_set_layout_cache_.clear();
}

}  // namespace vulkan
}  // namespace hal
}  // namespace iree
