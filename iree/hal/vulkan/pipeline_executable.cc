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

#include "iree/hal/vulkan/pipeline_executable.h"

#include "absl/container/inlined_vector.h"
#include "iree/base/memory.h"
#include "iree/base/status.h"
#include "iree/base/tracing.h"
#include "iree/hal/vulkan/status_util.h"

namespace iree {
namespace hal {
namespace vulkan {

namespace {

// Generates the baked specialization constant data based on the flatbuffer.
// We only support uint32_t right now so this is easy.
// Note that the returned vectors are referenced by pointers in |out_info| and
// must remain valid until the info is no longer in use.
std::pair<std::vector<VkSpecializationMapEntry>, std::vector<uint8_t>>
PopulateSpecializationInfo(const VkSpecializationInfoDef* info_def) {
  int entry_count =
      info_def && info_def->map_entries() ? info_def->map_entries()->size() : 0;
  if (!entry_count) {
    return {};
  }

  std::vector<VkSpecializationMapEntry> entries;
  entries.reserve(entry_count);
  std::vector<uint8_t> data;
  data.resize(entry_count * sizeof(uint32_t));

  uint32_t offset = 0;
  for (const auto* entry_def : *info_def->map_entries()) {
    if (!entry_def) continue;
    entries.push_back({});
    auto& entry = entries.back();
    entry.constantID = entry_def->constant_id();
    entry.offset = offset;
    entry.size = sizeof(uint32_t);
    uint32_t value = entry_def->uint32_value();
    std::memcpy(data.data() + offset, &value, sizeof(value));
    offset += entry.size;
  }

  return {std::move(entries), std::move(data)};
}

class VkShaderModuleHandle : public RefObject<VkShaderModuleHandle> {
 public:
  explicit VkShaderModuleHandle(const ref_ptr<VkDeviceHandle>& logical_device)
      : logical_device_(add_ref(logical_device)) {}
  ~VkShaderModuleHandle() { reset(); }

  VkShaderModuleHandle(const VkShaderModuleHandle&) = delete;
  VkShaderModuleHandle& operator=(const VkShaderModuleHandle&) = delete;
  VkShaderModuleHandle(VkShaderModuleHandle&& other) noexcept
      : logical_device_(std::move(other.logical_device_)),
        value_(absl::exchange(other.value_,
                              static_cast<VkShaderModule>(VK_NULL_HANDLE))) {}
  VkShaderModuleHandle& operator=(VkShaderModuleHandle&& other) {
    std::swap(logical_device_, other.logical_device_);
    std::swap(value_, other.value_);
    return *this;
  }

  void reset() {
    if (value_ == VK_NULL_HANDLE) return;
    logical_device_->syms()->vkDestroyShaderModule(
        *logical_device_, value_, logical_device_->allocator());
    value_ = VK_NULL_HANDLE;
  }

  VkShaderModule value() const noexcept { return value_; }
  VkShaderModule* mutable_value() noexcept { return &value_; }
  operator VkShaderModule() const noexcept { return value_; }

 private:
  ref_ptr<VkDeviceHandle> logical_device_;
  VkShaderModule value_ = VK_NULL_HANDLE;
};

}  // namespace

// static
StatusOr<ref_ptr<PipelineExecutable>> PipelineExecutable::Create(
    ref_ptr<VkDeviceHandle> logical_device, VkPipelineCache pipeline_cache,
    PipelineExecutableLayout* executable_layout,
    ExecutableCachingModeBitfield mode,
    const SpirVExecutableDef& spirv_executable_def) {
  IREE_TRACE_SCOPE0("PipelineExecutable::Create");
  const auto& syms = logical_device->syms();
  if (!spirv_executable_def.entry_points() ||
      spirv_executable_def.entry_points()->size() == 0) {
    return InvalidArgumentErrorBuilder(IREE_LOC) << "No entry points defined";
  }
  if (!spirv_executable_def.code()) {
    return InvalidArgumentErrorBuilder(IREE_LOC) << "No SPIR-V code present";
  }
  const auto& code = *spirv_executable_def.code();

  // Create the shader module.
  VkShaderModuleCreateInfo shader_module_create_info;
  shader_module_create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  shader_module_create_info.pNext = nullptr;
  shader_module_create_info.flags = 0;
  shader_module_create_info.codeSize = code.size() * sizeof(uint32_t);
  shader_module_create_info.pCode = code.data();
  VkShaderModuleHandle shader_module(add_ref(logical_device));
  VK_RETURN_IF_ERROR(syms->vkCreateShaderModule(
      *logical_device, &shader_module_create_info, logical_device->allocator(),
      shader_module.mutable_value()));

  // Specialization info is currently constant against all entry points.
  std::vector<VkSpecializationMapEntry> spec_entries;
  std::vector<uint8_t> spec_data;
  std::tie(spec_entries, spec_data) =
      PopulateSpecializationInfo(spirv_executable_def.specialization_info());
  VkSpecializationInfo specialization_info;
  specialization_info.mapEntryCount = spec_entries.size();
  specialization_info.pMapEntries = spec_entries.data();
  specialization_info.dataSize = spec_data.size();
  specialization_info.pData = spec_data.data();

  // Create pipelines for each entry point.
  const auto& entry_points = *spirv_executable_def.entry_points();
  absl::InlinedVector<VkComputePipelineCreateInfo, 1> pipeline_create_infos;
  pipeline_create_infos.resize(entry_points.size());
  for (int entry_ordinal = 0; entry_ordinal < entry_points.size();
       ++entry_ordinal) {
    auto& pipeline_create_info = pipeline_create_infos[entry_ordinal];
    pipeline_create_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_create_info.pNext = nullptr;
    pipeline_create_info.flags = 0;
    if (!AllBitsSet(mode, ExecutableCachingMode::kAllowOptimization)) {
      pipeline_create_info.flags |= VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT;
    }
    if (entry_ordinal == 0) {
      pipeline_create_info.flags |= VK_PIPELINE_CREATE_ALLOW_DERIVATIVES_BIT;
    } else {
      pipeline_create_info.flags |= VK_PIPELINE_CREATE_DERIVATIVE_BIT;
    }
    pipeline_create_info.layout = executable_layout->handle();
    pipeline_create_info.basePipelineHandle = VK_NULL_HANDLE;
    pipeline_create_info.basePipelineIndex = 0;
    auto& stage_create_info = pipeline_create_info.stage;
    stage_create_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage_create_info.pNext = nullptr;
    stage_create_info.flags = 0;
    stage_create_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage_create_info.module = shader_module;
    stage_create_info.pName = entry_points[entry_ordinal]->c_str();
    stage_create_info.pSpecializationInfo = &specialization_info;
  }
  absl::InlinedVector<VkPipeline, 1> pipelines;
  pipelines.resize(entry_points.size());

  // Some ICDs appear to leak in here, out of our control.
  // Warning: leak checks remain disabled if an error is returned.
  IREE_DISABLE_LEAK_CHECKS();
  VK_RETURN_IF_ERROR(syms->vkCreateComputePipelines(
      *logical_device, pipeline_cache, pipeline_create_infos.size(),
      pipeline_create_infos.data(), logical_device->allocator(),
      pipelines.data()));
  IREE_ENABLE_LEAK_CHECKS();

  auto executable = make_ref<PipelineExecutable>(std::move(logical_device),
                                                 std::move(pipelines));
  executable->tag_ =
      spirv_executable_def.tag() ? spirv_executable_def.tag()->str() : "";
  return executable;
}

PipelineExecutable::PipelineExecutable(
    ref_ptr<VkDeviceHandle> logical_device,
    absl::InlinedVector<VkPipeline, 1> pipelines)
    : logical_device_(std::move(logical_device)),
      pipelines_(std::move(pipelines)) {}

PipelineExecutable::~PipelineExecutable() {
  IREE_TRACE_SCOPE0("PipelineExecutable::dtor");
  for (auto pipeline : pipelines_) {
    syms()->vkDestroyPipeline(*logical_device_, pipeline,
                              logical_device_->allocator());
  }
  pipelines_.clear();
}

StatusOr<VkPipeline> PipelineExecutable::GetPipelineForEntryPoint(
    int entry_ordinal) const {
  if (entry_ordinal < 0 || entry_ordinal >= pipelines_.size()) {
    return OutOfRangeErrorBuilder(IREE_LOC) << "Invalid entry point ordinal";
  }
  return pipelines_[entry_ordinal];
}

}  // namespace vulkan
}  // namespace hal
}  // namespace iree
