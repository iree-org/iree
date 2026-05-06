// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/executable.h"

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

static iree_status_t MakeWrappedVulkanExecutable(
    iree_hal_vulkan_DispatchAbi_enum_t dispatch_abi,
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
  if (iree_status_is_ok(status)) {
    entry_point_ref = flatbuffers_string_create_str(&builder, "main");
    if (!entry_point_ref) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "failed to create entry point string");
    }
  }
  if (iree_status_is_ok(status) &&
      dispatch_abi == iree_hal_vulkan_DispatchAbi_BDA_V1) {
    if (flatbuffers_failed(
            iree_hal_vulkan_BdaDispatchLayoutDef_start(&builder)) ||
        flatbuffers_failed(
            iree_hal_vulkan_BdaDispatchLayoutDef_binding_count_add(&builder,
                                                                   3)) ||
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
        dispatch_abi == iree_hal_vulkan_DispatchAbi_BDA_V1 &&
        (flatbuffers_failed(iree_hal_vulkan_PipelineDef_dispatch_abi_add(
             &builder, dispatch_abi)) ||
         flatbuffers_failed(iree_hal_vulkan_PipelineDef_bda_dispatch_layout_add(
             &builder, bda_dispatch_layout_ref)))) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "failed to populate BDA pipeline definition");
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
    if (dispatch_abi == iree_hal_vulkan_DispatchAbi_BDA_V1) {
      iree_hal_vulkan_PushConstantRange push_constant_range;
      push_constant_range.stage_flags = VK_SHADER_STAGE_COMPUTE_BIT;
      push_constant_range.offset = 0;
      push_constant_range.size = 32;
      push_constant_ranges_ref = iree_hal_vulkan_PushConstantRange_vec_create(
          &builder, &push_constant_range, 1);
      if (!push_constant_ranges_ref) {
        status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                  "failed to create push constant ranges");
      }
    }
    if (iree_status_is_ok(status)) {
      if (flatbuffers_failed(
              iree_hal_vulkan_PipelineLayoutDef_start(&builder)) ||
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

}  // namespace
}  // namespace iree::hal::vulkan
