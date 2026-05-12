// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/executable.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "iree/base/api.h"
#include "iree/base/internal/flatcc/building.h"
#include "iree/hal/utils/executable_header.h"
#include "iree/schemas/vulkan_executable_def_builder.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::vulkan {
namespace {

static constexpr uint32_t kBdaDispatchRootLength =
    sizeof(iree_hal_vulkan_bda_dispatch_root_v1_t);

struct BdaBindingRequirementOptions {
  // Minimum alignment value emitted into the binding metadata.
  uint32_t minimum_alignment = 1;

  // Minimum length value emitted into the binding metadata.
  uint64_t minimum_length = 0;
};

struct WrappedVulkanExecutableOptions {
  // Dispatch ABI recorded on the exported pipeline.
  iree_hal_vulkan_DispatchAbi_enum_t dispatch_abi =
      iree_hal_vulkan_DispatchAbi_DESCRIPTOR;

  // Whether to attach BDA dispatch metadata to the pipeline.
  bool include_bda_dispatch_layout = false;

  // ABI version recorded in the BDA dispatch metadata.
  uint32_t bda_abi_version = 1;

  // Push-constant offset recorded for the hidden BDA root.
  uint32_t root_push_constant_offset = 0;

  // Push-constant byte length recorded for the hidden BDA root.
  uint32_t root_push_constant_length = kBdaDispatchRootLength;

  // Push-constant offset recorded for HAL inline constants.
  uint32_t constant_push_constant_offset = kBdaDispatchRootLength;

  // Number of HAL inline constants recorded in the BDA layout.
  uint32_t constant_count = 0;

  // Binding table entry type recorded in the BDA layout.
  iree_hal_vulkan_BdaBindingTableEntryType_enum_t binding_table_entry_type =
      iree_hal_vulkan_BdaBindingTableEntryType_ADDRESS64;

  // Number of HAL bindings recorded in the BDA layout.
  uint32_t binding_count = 3;

  // Optional per-binding BDA requirements.
  std::vector<BdaBindingRequirementOptions> binding_requirements;

  // Whether to create a push-constant range covering the BDA root by default.
  bool include_bda_push_constant_range = false;

  // Stage flags used by the generated push-constant range.
  uint32_t push_constant_stage_flags = VK_SHADER_STAGE_COMPUTE_BIT;

  // Push-constant range offset used by the generated pipeline layout.
  uint32_t push_constant_offset = 0;

  // Push-constant range byte length used by the generated pipeline layout.
  uint32_t push_constant_size = kBdaDispatchRootLength;

  // Whether the generated pipeline layout references descriptor set layout 0.
  bool include_descriptor_set_layout_ordinal = false;

  // Whether the executable contains an empty descriptor set layout at ordinal
  // 0.
  bool include_descriptor_set_layout = false;
};

static WrappedVulkanExecutableOptions MakeWrappedBdaExecutableOptions() {
  WrappedVulkanExecutableOptions options;
  options.dispatch_abi = iree_hal_vulkan_DispatchAbi_BDA_V1;
  options.include_bda_dispatch_layout = true;
  options.include_bda_push_constant_range = true;
  return options;
}

static iree_status_t MakeWrappedVulkanExecutable(
    const WrappedVulkanExecutableOptions& options,
    std::vector<uint8_t>* out_executable_data) {
  IREE_ASSERT_ARGUMENT(out_executable_data);
  out_executable_data->clear();

  flatbuffers_builder_t builder;
  if (IREE_UNLIKELY(flatcc_builder_init(&builder) != 0)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "failed to initialize flatbuffer builder");
  }

  iree_status_t status = iree_ok_status();
  if (IREE_UNLIKELY(flatbuffers_failed(
          iree_hal_vulkan_ExecutableDef_start_as_root(&builder)))) {
    status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "failed to start Vulkan executable flatbuffer");
  }

  flatbuffers_string_ref_t entry_point_ref = 0;
  iree_hal_vulkan_BdaDispatchLayoutDef_ref_t bda_dispatch_layout_ref = 0;
  iree_hal_vulkan_BdaBindingDef_vec_ref_t bda_bindings_ref = 0;
  if (iree_status_is_ok(status)) {
    entry_point_ref = flatbuffers_string_create_str(&builder, "main");
    if (!entry_point_ref) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "failed to create entry point string");
    }
  }
  if (iree_status_is_ok(status) && !options.binding_requirements.empty()) {
    std::vector<iree_hal_vulkan_BdaBindingDef_ref_t> binding_requirement_refs;
    binding_requirement_refs.reserve(options.binding_requirements.size());
    for (const auto& binding_requirement : options.binding_requirements) {
      iree_hal_vulkan_BdaBindingDef_ref_t binding_requirement_ref =
          iree_hal_vulkan_BdaBindingDef_create(
              &builder, binding_requirement.minimum_alignment,
              binding_requirement.minimum_length);
      if (!binding_requirement_ref) {
        status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                  "failed to create BDA binding requirement");
        break;
      }
      binding_requirement_refs.push_back(binding_requirement_ref);
    }
    if (iree_status_is_ok(status)) {
      bda_bindings_ref = iree_hal_vulkan_BdaBindingDef_vec_create(
          &builder, binding_requirement_refs.data(),
          binding_requirement_refs.size());
      if (!bda_bindings_ref) {
        status =
            iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                             "failed to create BDA binding requirement vector");
      }
    }
  }
  if (iree_status_is_ok(status) && options.include_bda_dispatch_layout) {
    if (flatbuffers_failed(
            iree_hal_vulkan_BdaDispatchLayoutDef_start(&builder)) ||
        flatbuffers_failed(iree_hal_vulkan_BdaDispatchLayoutDef_abi_version_add(
            &builder, options.bda_abi_version)) ||
        flatbuffers_failed(
            iree_hal_vulkan_BdaDispatchLayoutDef_root_push_constant_offset_add(
                &builder, options.root_push_constant_offset)) ||
        flatbuffers_failed(
            iree_hal_vulkan_BdaDispatchLayoutDef_root_push_constant_length_add(
                &builder, options.root_push_constant_length)) ||
        flatbuffers_failed(
            iree_hal_vulkan_BdaDispatchLayoutDef_constant_push_constant_offset_add(
                &builder, options.constant_push_constant_offset)) ||
        flatbuffers_failed(
            iree_hal_vulkan_BdaDispatchLayoutDef_constant_count_add(
                &builder, options.constant_count)) ||
        flatbuffers_failed(
            iree_hal_vulkan_BdaDispatchLayoutDef_binding_count_add(
                &builder, options.binding_count)) ||
        flatbuffers_failed(
            iree_hal_vulkan_BdaDispatchLayoutDef_binding_table_entry_type_add(
                &builder, options.binding_table_entry_type)) ||
        (bda_bindings_ref &&
         flatbuffers_failed(iree_hal_vulkan_BdaDispatchLayoutDef_bindings_add(
             &builder, bda_bindings_ref))) ||
        !(bda_dispatch_layout_ref =
              iree_hal_vulkan_BdaDispatchLayoutDef_end(&builder))) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "failed to create BDA dispatch layout");
    }
  }

  iree_hal_vulkan_PipelineDef_ref_t pipeline_ref = 0;
  iree_hal_vulkan_PipelineDef_vec_ref_t pipelines_ref = 0;
  if (iree_status_is_ok(status)) {
    if (flatbuffers_failed(iree_hal_vulkan_PipelineDef_start(&builder)) ||
        flatbuffers_failed(iree_hal_vulkan_PipelineDef_entry_point_add(
            &builder, entry_point_ref))) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "failed to start pipeline definition");
    }
    if (iree_status_is_ok(status) &&
        options.dispatch_abi != iree_hal_vulkan_DispatchAbi_DESCRIPTOR &&
        flatbuffers_failed(iree_hal_vulkan_PipelineDef_dispatch_abi_add(
            &builder, options.dispatch_abi))) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "failed to add pipeline dispatch ABI");
    }
    if (iree_status_is_ok(status) && options.include_bda_dispatch_layout &&
        flatbuffers_failed(iree_hal_vulkan_PipelineDef_bda_dispatch_layout_add(
            &builder, bda_dispatch_layout_ref))) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "failed to add BDA pipeline definition");
    }
  }
  if (iree_status_is_ok(status)) {
    pipeline_ref = iree_hal_vulkan_PipelineDef_end(&builder);
    pipelines_ref =
        iree_hal_vulkan_PipelineDef_vec_create(&builder, &pipeline_ref, 1);
    if (!pipeline_ref || !pipelines_ref) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "failed to create pipeline definitions");
    }
  }

  iree_hal_vulkan_PipelineLayoutDef_ref_t pipeline_layout_ref = 0;
  iree_hal_vulkan_PipelineLayoutDef_vec_ref_t pipeline_layouts_ref = 0;
  if (iree_status_is_ok(status)) {
    iree_hal_vulkan_PushConstantRange_vec_ref_t push_constant_ranges_ref = 0;
    if (options.include_bda_push_constant_range) {
      iree_hal_vulkan_PushConstantRange push_constant_range;
      push_constant_range.stage_flags = options.push_constant_stage_flags;
      push_constant_range.offset = options.push_constant_offset;
      push_constant_range.size = options.push_constant_size;
      push_constant_ranges_ref = iree_hal_vulkan_PushConstantRange_vec_create(
          &builder, &push_constant_range, 1);
      if (!push_constant_ranges_ref) {
        status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                  "failed to create push constant ranges");
      }
    }
    flatbuffers_uint32_vec_ref_t descriptor_set_layout_ordinals_ref = 0;
    if (iree_status_is_ok(status) &&
        options.include_descriptor_set_layout_ordinal) {
      const uint32_t descriptor_set_layout_ordinal = 0;
      descriptor_set_layout_ordinals_ref = flatbuffers_uint32_vec_create(
          &builder, &descriptor_set_layout_ordinal, 1);
      if (!descriptor_set_layout_ordinals_ref) {
        status =
            iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                             "failed to create descriptor set layout ordinals");
      }
    }
    if (iree_status_is_ok(status)) {
      if (flatbuffers_failed(
              iree_hal_vulkan_PipelineLayoutDef_start(&builder)) ||
          (descriptor_set_layout_ordinals_ref &&
           flatbuffers_failed(
               iree_hal_vulkan_PipelineLayoutDef_descriptor_set_layout_ordinals_add(
                   &builder, descriptor_set_layout_ordinals_ref))) ||
          (push_constant_ranges_ref &&
           flatbuffers_failed(
               iree_hal_vulkan_PipelineLayoutDef_push_constant_ranges_add(
                   &builder, push_constant_ranges_ref))) ||
          !(pipeline_layout_ref =
                iree_hal_vulkan_PipelineLayoutDef_end(&builder))) {
        status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                  "failed to create pipeline layout");
      }
    }
    if (iree_status_is_ok(status)) {
      pipeline_layouts_ref = iree_hal_vulkan_PipelineLayoutDef_vec_create(
          &builder, &pipeline_layout_ref, 1);
      if (!pipeline_layout_ref || !pipeline_layouts_ref) {
        status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                  "failed to create pipeline layout");
      }
    }
  }

  iree_hal_vulkan_DescriptorSetLayoutDef_vec_ref_t descriptor_set_layouts_ref =
      0;
  if (iree_status_is_ok(status) && options.include_descriptor_set_layout) {
    iree_hal_vulkan_DescriptorSetLayoutDef_ref_t descriptor_set_layout_ref = 0;
    if (flatbuffers_failed(
            iree_hal_vulkan_DescriptorSetLayoutDef_start(&builder)) ||
        !(descriptor_set_layout_ref =
              iree_hal_vulkan_DescriptorSetLayoutDef_end(&builder))) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "failed to create descriptor set layout");
    }
    if (iree_status_is_ok(status)) {
      descriptor_set_layouts_ref =
          iree_hal_vulkan_DescriptorSetLayoutDef_vec_create(
              &builder, &descriptor_set_layout_ref, 1);
    }
    if (!descriptor_set_layout_ref || !descriptor_set_layouts_ref) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "failed to create descriptor set layouts");
    }
  }

  iree_hal_vulkan_ShaderModuleDef_ref_t shader_module_ref = 0;
  iree_hal_vulkan_ShaderModuleDef_vec_ref_t shader_modules_ref = 0;
  if (iree_status_is_ok(status)) {
    const uint32_t spirv_words[5] = {0x07230203u, 0, 0, 0, 0};
    flatbuffers_uint32_vec_ref_t spirv_code_ref = flatbuffers_uint32_vec_create(
        &builder, spirv_words, IREE_ARRAYSIZE(spirv_words));
    shader_module_ref =
        iree_hal_vulkan_ShaderModuleDef_create(&builder, spirv_code_ref);
    shader_modules_ref = iree_hal_vulkan_ShaderModuleDef_vec_create(
        &builder, &shader_module_ref, 1);
    if (!spirv_code_ref || !shader_module_ref || !shader_modules_ref) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "failed to create shader module definitions");
    }
  }

  if (iree_status_is_ok(status) &&
      IREE_UNLIKELY(
          flatbuffers_failed(iree_hal_vulkan_ExecutableDef_pipelines_add(
              &builder, pipelines_ref)) ||
          (descriptor_set_layouts_ref &&
           flatbuffers_failed(
               iree_hal_vulkan_ExecutableDef_descriptor_set_layouts_add(
                   &builder, descriptor_set_layouts_ref))) ||
          flatbuffers_failed(iree_hal_vulkan_ExecutableDef_pipeline_layouts_add(
              &builder, pipeline_layouts_ref)) ||
          flatbuffers_failed(iree_hal_vulkan_ExecutableDef_shader_modules_add(
              &builder, shader_modules_ref)))) {
    status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "failed to populate Vulkan executable");
  }
  if (iree_status_is_ok(status) &&
      IREE_UNLIKELY(!iree_hal_vulkan_ExecutableDef_end_as_root(&builder))) {
    status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "failed to finish Vulkan executable flatbuffer");
  }

  size_t flatbuffer_size = 0;
  void* flatbuffer_data = NULL;
  if (iree_status_is_ok(status)) {
    flatbuffer_data =
        flatcc_builder_finalize_aligned_buffer(&builder, &flatbuffer_size);
    if (!flatbuffer_data || flatbuffer_size == 0) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "failed to finalize Vulkan executable");
    }
  }

  if (iree_status_is_ok(status)) {
    iree_flatbuffer_file_header_t header = {};
    memcpy(&header.magic, iree_hal_vulkan_ExecutableDef_file_identifier,
           sizeof(header.magic));
    header.version = 0;
    header.content_size = flatbuffer_size;

    out_executable_data->resize(sizeof(header) + flatbuffer_size);
    memcpy(out_executable_data->data(), &header, sizeof(header));
    memcpy(out_executable_data->data() + sizeof(header), flatbuffer_data,
           flatbuffer_size);
  }

  flatcc_builder_aligned_free(flatbuffer_data);
  flatcc_builder_clear(&builder);
  return status;
}

static iree_status_t MakeWrappedVulkanExecutable(
    iree_hal_vulkan_DispatchAbi_enum_t dispatch_abi,
    std::vector<uint8_t>* out_executable_data) {
  WrappedVulkanExecutableOptions options;
  if (dispatch_abi == iree_hal_vulkan_DispatchAbi_BDA_V1) {
    options = MakeWrappedBdaExecutableOptions();
  } else {
    options.dispatch_abi = dispatch_abi;
  }
  return MakeWrappedVulkanExecutable(options, out_executable_data);
}

static iree_status_t CreateWrappedVulkanExecutableForValidation(
    const WrappedVulkanExecutableOptions& options) {
  std::vector<uint8_t> executable_data;
  IREE_RETURN_IF_ERROR(MakeWrappedVulkanExecutable(options, &executable_data));

  iree_hal_executable_params_t executable_params = {0};
  executable_params.executable_format =
      options.dispatch_abi == iree_hal_vulkan_DispatchAbi_BDA_V1
          ? IREE_SV("vulkan-spirv-bda-v1")
          : IREE_SV("vulkan-spirv-fb");
  executable_params.executable_data =
      iree_make_const_byte_span(executable_data.data(), executable_data.size());

  const iree_hal_vulkan_features_t enabled_features =
      options.dispatch_abi == iree_hal_vulkan_DispatchAbi_BDA_V1
          ? IREE_HAL_VULKAN_FEATURE_ENABLE_BUFFER_DEVICE_ADDRESSES
          : IREE_HAL_VULKAN_FEATURE_NONE;
  const iree_hal_vulkan_dispatch_abis_t enabled_dispatch_abis =
      options.dispatch_abi == iree_hal_vulkan_DispatchAbi_BDA_V1
          ? IREE_HAL_VULKAN_DISPATCH_ABI_BDA
          : IREE_HAL_VULKAN_DISPATCH_ABI_DESCRIPTOR;

  iree_hal_vulkan_device_syms_t syms = {0};
  iree_hal_vulkan_physical_device_snapshot_t physical_device = {0};
  // Keep generic pipeline-layout validation from preempting BDA metadata
  // checks.
  physical_device.properties2.properties.limits.maxPushConstantsSize = 65536;
  iree_hal_executable_t* executable = nullptr;
  iree_status_t status = iree_hal_vulkan_executable_create(
      &syms, reinterpret_cast<VkDevice>(uintptr_t{1}), &physical_device,
      enabled_features, IREE_HAL_VULKAN_DEVICE_EXTENSION_NONE, VK_NULL_HANDLE,
      enabled_dispatch_abis, &executable_params, iree_allocator_system(),
      &executable);
  iree_hal_executable_release(executable);
  return status;
}

static std::string InferExecutableFormat(iree_const_byte_span_t executable_data,
                                         iree_host_size_t* out_inferred_size) {
  char executable_format[64] = {};
  IREE_CHECK_OK(iree_hal_vulkan_executable_infer_format(
      executable_data, sizeof(executable_format), executable_format,
      out_inferred_size));
  return std::string(executable_format);
}

TEST(ExecutableTest, InfersDescriptorFlatbufferFormat) {
  std::vector<uint8_t> executable_data;
  IREE_ASSERT_OK(MakeWrappedVulkanExecutable(
      iree_hal_vulkan_DispatchAbi_DESCRIPTOR, &executable_data));

  iree_host_size_t inferred_size = 0;
  EXPECT_EQ(
      InferExecutableFormat(iree_make_const_byte_span(executable_data.data(),
                                                      executable_data.size()),
                            &inferred_size),
      "vulkan-spirv-fb");
  EXPECT_EQ(inferred_size, executable_data.size());
}

TEST(ExecutableTest, InfersBdaFlatbufferFormat) {
  std::vector<uint8_t> executable_data;
  IREE_ASSERT_OK(MakeWrappedVulkanExecutable(iree_hal_vulkan_DispatchAbi_BDA_V1,
                                             &executable_data));

  iree_host_size_t inferred_size = 0;
  EXPECT_EQ(
      InferExecutableFormat(iree_make_const_byte_span(executable_data.data(),
                                                      executable_data.size()),
                            &inferred_size),
      "vulkan-spirv-bda-v1");
  EXPECT_EQ(inferred_size, executable_data.size());
}

TEST(ExecutableTest, BdaFlatbufferFormatRequiresBdaFeatureAndAbi) {
  const iree_hal_vulkan_features_t bda_features =
      IREE_HAL_VULKAN_FEATURE_ENABLE_BUFFER_DEVICE_ADDRESSES;

  EXPECT_TRUE(iree_hal_vulkan_executable_format_supported(
      bda_features, IREE_HAL_VULKAN_DISPATCH_ABI_BDA,
      IREE_SV("vulkan-spirv-bda-v1")));
  EXPECT_FALSE(iree_hal_vulkan_executable_format_supported(
      IREE_HAL_VULKAN_FEATURE_NONE, IREE_HAL_VULKAN_DISPATCH_ABI_BDA,
      IREE_SV("vulkan-spirv-bda-v1")));
  EXPECT_FALSE(iree_hal_vulkan_executable_format_supported(
      bda_features, IREE_HAL_VULKAN_DISPATCH_ABI_DESCRIPTOR,
      IREE_SV("vulkan-spirv-bda-v1")));
}

TEST(ExecutableTest, RejectsBdaPipelineWithoutBdaLayout) {
  WrappedVulkanExecutableOptions options = MakeWrappedBdaExecutableOptions();
  options.include_bda_dispatch_layout = false;

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        CreateWrappedVulkanExecutableForValidation(options));
}

TEST(ExecutableTest, RejectsDescriptorPipelineWithBdaLayout) {
  WrappedVulkanExecutableOptions options;
  options.include_bda_dispatch_layout = true;

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        CreateWrappedVulkanExecutableForValidation(options));
}

TEST(ExecutableTest, RejectsBdaLayoutWithUnsupportedAbiVersion) {
  WrappedVulkanExecutableOptions options = MakeWrappedBdaExecutableOptions();
  options.bda_abi_version = 2;

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        CreateWrappedVulkanExecutableForValidation(options));
}

TEST(ExecutableTest, RejectsBdaLayoutWithWrongRootLength) {
  WrappedVulkanExecutableOptions options = MakeWrappedBdaExecutableOptions();
  options.root_push_constant_length =
      sizeof(iree_hal_vulkan_bda_dispatch_root_v1_t) - sizeof(uint32_t);

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        CreateWrappedVulkanExecutableForValidation(options));
}

TEST(ExecutableTest, RejectsBdaLayoutWithUnalignedRootOffset) {
  WrappedVulkanExecutableOptions options = MakeWrappedBdaExecutableOptions();
  options.root_push_constant_offset = 2;

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        CreateWrappedVulkanExecutableForValidation(options));
}

TEST(ExecutableTest, RejectsBdaLayoutWithRootOutsidePipelineLayout) {
  WrappedVulkanExecutableOptions options = MakeWrappedBdaExecutableOptions();
  options.push_constant_size =
      sizeof(iree_hal_vulkan_bda_dispatch_root_v1_t) - sizeof(uint32_t);

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        CreateWrappedVulkanExecutableForValidation(options));
}

TEST(ExecutableTest, RejectsBdaLayoutWithTooManyConstants) {
  WrappedVulkanExecutableOptions options = MakeWrappedBdaExecutableOptions();
  options.constant_count = UINT16_MAX + 1u;

  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        CreateWrappedVulkanExecutableForValidation(options));
}

TEST(ExecutableTest, RejectsBdaLayoutWithUnalignedConstantOffset) {
  WrappedVulkanExecutableOptions options = MakeWrappedBdaExecutableOptions();
  options.constant_push_constant_offset = 34;
  options.constant_count = 1;
  options.push_constant_size =
      sizeof(iree_hal_vulkan_bda_dispatch_root_v1_t) + sizeof(uint32_t);

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        CreateWrappedVulkanExecutableForValidation(options));
}

TEST(ExecutableTest, RejectsBdaLayoutWithConstantsOutsidePipelineLayout) {
  WrappedVulkanExecutableOptions options = MakeWrappedBdaExecutableOptions();
  options.constant_count = 1;

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        CreateWrappedVulkanExecutableForValidation(options));
}

TEST(ExecutableTest, RejectsBdaLayoutWithConstantsOverlappingRoot) {
  WrappedVulkanExecutableOptions options = MakeWrappedBdaExecutableOptions();
  options.constant_push_constant_offset = sizeof(uint32_t);
  options.constant_count = 1;

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        CreateWrappedVulkanExecutableForValidation(options));
}

TEST(ExecutableTest, RejectsUnsupportedBdaBindingTableEntryType) {
  WrappedVulkanExecutableOptions options = MakeWrappedBdaExecutableOptions();
  options.binding_table_entry_type =
      iree_hal_vulkan_BdaBindingTableEntryType_ADDRESS64_LENGTH64;

  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED,
                        CreateWrappedVulkanExecutableForValidation(options));
}

TEST(ExecutableTest, RejectsBdaLayoutWithTooManyBindings) {
  WrappedVulkanExecutableOptions options = MakeWrappedBdaExecutableOptions();
  options.binding_count = UINT16_MAX + 1u;

  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        CreateWrappedVulkanExecutableForValidation(options));
}

TEST(ExecutableTest, RejectsBdaBindingRequirementCountMismatch) {
  WrappedVulkanExecutableOptions options = MakeWrappedBdaExecutableOptions();
  options.binding_count = 3;
  options.binding_requirements = {{1, 0}, {1, 0}};

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        CreateWrappedVulkanExecutableForValidation(options));
}

TEST(ExecutableTest, RejectsBdaBindingRequirementWithZeroAlignment) {
  WrappedVulkanExecutableOptions options = MakeWrappedBdaExecutableOptions();
  options.binding_requirements = {{1, 0}, {0, 0}, {1, 0}};

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        CreateWrappedVulkanExecutableForValidation(options));
}

TEST(ExecutableTest, RejectsBdaBindingRequirementWithNonPowerOfTwoAlignment) {
  WrappedVulkanExecutableOptions options = MakeWrappedBdaExecutableOptions();
  options.binding_requirements = {{1, 0}, {3, 0}, {1, 0}};

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        CreateWrappedVulkanExecutableForValidation(options));
}

TEST(ExecutableTest, RejectsBdaPipelineWithDescriptorSetLayoutOrdinals) {
  WrappedVulkanExecutableOptions options = MakeWrappedBdaExecutableOptions();
  options.include_descriptor_set_layout_ordinal = true;
  options.include_descriptor_set_layout = true;

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        CreateWrappedVulkanExecutableForValidation(options));
}

}  // namespace
}  // namespace iree::hal::vulkan
