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

// NOTE: include order matters:
#include "flatcc/reflection/flatbuffers_common_reader.h"
#include "iree/schemas/spirv_executable_def_reader.h"
#include "iree/schemas/spirv_executable_def_verifier.h"

// NOTE: starting to port this to C.

// Verifies the structure of the flatbuffer so that we can avoid doing so during
// runtime. There are still some conditions we must be aware of (such as omitted
// names on functions with internal linkage), however we shouldn't need to
// bounds check anything within the flatbuffer after this succeeds.
static iree_status_t iree_hal_spirv_executable_flatbuffer_verify(
    iree_const_byte_span_t flatbuffer_data) {
  if (!flatbuffer_data.data || flatbuffer_data.data_length < 16) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "flatbuffer data is not present or less than 16 bytes (%zu total)",
        flatbuffer_data.data_length);
  }

  // Run flatcc generated verification. This ensures all pointers are in-bounds
  // and that we can safely walk the file, but not that the actual contents of
  // the flatbuffer meet our expectations.
  int verify_ret = iree_SpirVExecutableDef_verify_as_root(
      flatbuffer_data.data, flatbuffer_data.data_length);
  if (verify_ret != flatcc_verify_ok) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "flatbuffer verification failed: %s",
                            flatcc_verify_error_string(verify_ret));
  }

  iree_SpirVExecutableDef_table_t executable_def =
      iree_SpirVExecutableDef_as_root(flatbuffer_data.data);

  flatbuffers_string_vec_t entry_points_vec =
      iree_SpirVExecutableDef_entry_points_get(executable_def);
  size_t entry_point_count = flatbuffers_string_vec_len(entry_points_vec);
  for (size_t i = 0; i < entry_point_count; ++i) {
    if (!flatbuffers_string_len(
            flatbuffers_string_vec_at(entry_points_vec, i))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "executable entry point %zu has no name", i);
    }
  }

  if (flatbuffers_uint32_vec_len(
          iree_SpirVExecutableDef_code_get(executable_def)) < 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "executable SPIR-V code is missing/empty");
  }

  // TODO(benvanik): pull PopulateSpecializationInfo from history and update.
  // For now the compiler isn't generating them, and we don't use them.
  if (iree_SpirVExecutableDef_specialization_info_is_present(executable_def)) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "executable uses SPIR-V specialization constants; "
                            "they need to be revived");
  }

  return iree_ok_status();
}

namespace iree {
namespace hal {
namespace vulkan {

namespace {

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
    ExecutableCachingModeBitfield mode, const ExecutableSpec& spec) {
  IREE_TRACE_SCOPE0("PipelineExecutable::Create");
  const auto& syms = logical_device->syms();

  // Verify and fetch the executable flatbuffer wrapper.
  iree_const_byte_span_t executable_data = iree_make_const_byte_span(
      spec.executable_data.data(), spec.executable_data.size());
  IREE_RETURN_IF_ERROR(
      iree_hal_spirv_executable_flatbuffer_verify(executable_data));
  iree_SpirVExecutableDef_table_t executable_def =
      iree_SpirVExecutableDef_as_root(executable_data.data);

  // Create the shader module.
  VkShaderModuleCreateInfo shader_module_create_info;
  shader_module_create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  shader_module_create_info.pNext = nullptr;
  shader_module_create_info.flags = 0;
  flatbuffers_uint32_vec_t code_vec =
      iree_SpirVExecutableDef_code_get(executable_def);
  shader_module_create_info.codeSize =
      flatbuffers_uint32_vec_len(code_vec) * sizeof(uint32_t);
  shader_module_create_info.pCode = code_vec;
  VkShaderModuleHandle shader_module(add_ref(logical_device));
  VK_RETURN_IF_ERROR(syms->vkCreateShaderModule(
      *logical_device, &shader_module_create_info, logical_device->allocator(),
      shader_module.mutable_value()));

  // Create pipelines for each entry point.
  flatbuffers_string_vec_t entry_points_vec =
      iree_SpirVExecutableDef_entry_points_get(executable_def);
  absl::InlinedVector<VkComputePipelineCreateInfo, 1> pipeline_create_infos;
  pipeline_create_infos.resize(flatbuffers_string_vec_len(entry_points_vec));
  for (size_t entry_ordinal = 0;
       entry_ordinal < flatbuffers_string_vec_len(entry_points_vec);
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
    stage_create_info.pName =
        flatbuffers_string_vec_at(entry_points_vec, entry_ordinal);
    stage_create_info.pSpecializationInfo = NULL;
  }
  absl::InlinedVector<VkPipeline, 1> pipelines;
  pipelines.resize(flatbuffers_string_vec_len(entry_points_vec));

  // Some ICDs appear to leak in here, out of our control.
  // Warning: leak checks remain disabled if an error is returned.
  IREE_DISABLE_LEAK_CHECKS();
  VK_RETURN_IF_ERROR(syms->vkCreateComputePipelines(
      *logical_device, pipeline_cache,
      static_cast<uint32_t>(pipeline_create_infos.size()),
      pipeline_create_infos.data(), logical_device->allocator(),
      pipelines.data()));
  IREE_ENABLE_LEAK_CHECKS();

  return make_ref<PipelineExecutable>(std::move(logical_device),
                                      std::move(pipelines));
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
