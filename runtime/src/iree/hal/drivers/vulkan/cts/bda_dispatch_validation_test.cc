// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Vulkan BDA-specific validation coverage. These cases assert failures at the
// HAL boundary before malformed pointer tables can reach the device.

#include <cstdint>
#include <cstring>
#include <vector>

#include "iree/base/internal/flatcc/building.h"
#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/utils/executable_header.h"
#include "iree/schemas/vulkan_executable_def_builder.h"

namespace iree::hal::cts {

using iree::testing::status::StatusIs;

static constexpr uint32_t kRequiredBindingAlignment = 2;
static constexpr uint64_t kRequiredBindingLength = 17;
static constexpr uint32_t kBdaDispatchRootLength = 32;
// VK_SHADER_STAGE_COMPUTE_BIT without depending on private Vulkan headers.
static constexpr uint32_t kComputeShaderStageFlag = 0x00000020u;

// Descriptor-free BDA-environment no-op shader. The shader does not need to
// consume bindings because these tests fail while publishing the host-validated
// BDA table.
static const uint32_t kBdaNoopSpirv[] = {
    0x07230203u,
    0x00010600u,
    0u,
    5u,
    0u,
    // Declares OpCapability Shader.
    0x00020011u,
    1u,
    // Declares OpCapability PhysicalStorageBufferAddresses.
    0x00020011u,
    5347u,
    // Declares OpMemoryModel PhysicalStorageBuffer64 GLSL450.
    0x0003000eu,
    5348u,
    1u,
    // Declares OpEntryPoint GLCompute %main "main".
    0x0005000fu,
    5u,
    3u,
    0x6e69616du,
    0u,
    // Declares OpExecutionMode %main LocalSize 1 1 1.
    0x00060010u,
    3u,
    17u,
    1u,
    1u,
    1u,
    // Declares OpTypeVoid %void.
    0x00020013u,
    1u,
    // Declares OpTypeFunction %fn %void.
    0x00030021u,
    2u,
    1u,
    // Defines %main as an empty compute function.
    0x00050036u,
    1u,
    3u,
    0u,
    2u,
    0x000200f8u,
    4u,
    0x000100fdu,
    0x00010038u,
};

static iree_status_t MakeBdaRequirementExecutableData(
    std::vector<uint8_t>* out_executable_data) {
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
  if (iree_status_is_ok(status)) {
    entry_point_ref = flatbuffers_string_create_str(&builder, "main");
    if (!entry_point_ref) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "failed to create entry point string");
    }
  }

  iree_hal_vulkan_BdaBindingDef_vec_ref_t bda_bindings_ref = 0;
  if (iree_status_is_ok(status)) {
    iree_hal_vulkan_BdaBindingDef_ref_t binding_requirements[2] = {
        iree_hal_vulkan_BdaBindingDef_create(&builder, /*minimum_alignment=*/1,
                                             kRequiredBindingLength),
        iree_hal_vulkan_BdaBindingDef_create(
            &builder, kRequiredBindingAlignment, /*minimum_length=*/0),
    };
    bda_bindings_ref = iree_hal_vulkan_BdaBindingDef_vec_create(
        &builder, binding_requirements, IREE_ARRAYSIZE(binding_requirements));
    if (!binding_requirements[0] || !binding_requirements[1] ||
        !bda_bindings_ref) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "failed to create BDA binding requirements");
    }
  }

  iree_hal_vulkan_BdaDispatchLayoutDef_ref_t bda_dispatch_layout_ref = 0;
  if (iree_status_is_ok(status)) {
    if (flatbuffers_failed(
            iree_hal_vulkan_BdaDispatchLayoutDef_start(&builder)) ||
        flatbuffers_failed(iree_hal_vulkan_BdaDispatchLayoutDef_abi_version_add(
            &builder, 1)) ||
        flatbuffers_failed(
            iree_hal_vulkan_BdaDispatchLayoutDef_root_push_constant_offset_add(
                &builder, 0)) ||
        flatbuffers_failed(
            iree_hal_vulkan_BdaDispatchLayoutDef_root_push_constant_length_add(
                &builder, kBdaDispatchRootLength)) ||
        flatbuffers_failed(
            iree_hal_vulkan_BdaDispatchLayoutDef_constant_push_constant_offset_add(
                &builder, kBdaDispatchRootLength)) ||
        flatbuffers_failed(
            iree_hal_vulkan_BdaDispatchLayoutDef_binding_count_add(&builder,
                                                                   2)) ||
        flatbuffers_failed(
            iree_hal_vulkan_BdaDispatchLayoutDef_binding_table_entry_type_add(
                &builder,
                iree_hal_vulkan_BdaBindingTableEntryType_ADDRESS64)) ||
        flatbuffers_failed(iree_hal_vulkan_BdaDispatchLayoutDef_bindings_add(
            &builder, bda_bindings_ref)) ||
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
            &builder, entry_point_ref)) ||
        flatbuffers_failed(iree_hal_vulkan_PipelineDef_dispatch_abi_add(
            &builder, iree_hal_vulkan_DispatchAbi_BDA_V1)) ||
        flatbuffers_failed(iree_hal_vulkan_PipelineDef_bda_dispatch_layout_add(
            &builder, bda_dispatch_layout_ref)) ||
        !(pipeline_ref = iree_hal_vulkan_PipelineDef_end(&builder)) ||
        !(pipelines_ref = iree_hal_vulkan_PipelineDef_vec_create(
              &builder, &pipeline_ref, 1))) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "failed to create pipeline definition");
    }
  }

  iree_hal_vulkan_PipelineLayoutDef_ref_t pipeline_layout_ref = 0;
  iree_hal_vulkan_PipelineLayoutDef_vec_ref_t pipeline_layouts_ref = 0;
  if (iree_status_is_ok(status)) {
    iree_hal_vulkan_PushConstantRange push_constant_range = {
        /*.stage_flags=*/kComputeShaderStageFlag,
        /*.offset=*/0,
        /*.size=*/kBdaDispatchRootLength,
    };
    iree_hal_vulkan_PushConstantRange_vec_ref_t push_constant_ranges_ref =
        iree_hal_vulkan_PushConstantRange_vec_create(&builder,
                                                     &push_constant_range, 1);
    if (!push_constant_ranges_ref ||
        flatbuffers_failed(iree_hal_vulkan_PipelineLayoutDef_start(&builder)) ||
        flatbuffers_failed(
            iree_hal_vulkan_PipelineLayoutDef_push_constant_ranges_add(
                &builder, push_constant_ranges_ref)) ||
        !(pipeline_layout_ref =
              iree_hal_vulkan_PipelineLayoutDef_end(&builder)) ||
        !(pipeline_layouts_ref = iree_hal_vulkan_PipelineLayoutDef_vec_create(
              &builder, &pipeline_layout_ref, 1))) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "failed to create pipeline layout");
    }
  }

  iree_hal_vulkan_ShaderModuleDef_ref_t shader_module_ref = 0;
  iree_hal_vulkan_ShaderModuleDef_vec_ref_t shader_modules_ref = 0;
  if (iree_status_is_ok(status)) {
    flatbuffers_uint32_vec_ref_t spirv_code_ref = flatbuffers_uint32_vec_create(
        &builder, kBdaNoopSpirv, IREE_ARRAYSIZE(kBdaNoopSpirv));
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
      (flatbuffers_failed(iree_hal_vulkan_ExecutableDef_pipelines_add(
           &builder, pipelines_ref)) ||
       flatbuffers_failed(iree_hal_vulkan_ExecutableDef_pipeline_layouts_add(
           &builder, pipeline_layouts_ref)) ||
       flatbuffers_failed(iree_hal_vulkan_ExecutableDef_shader_modules_add(
           &builder, shader_modules_ref)) ||
       !iree_hal_vulkan_ExecutableDef_end_as_root(&builder))) {
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

class BdaDispatchValidationTest : public CtsTestBase<> {
 protected:
  void SetUp() override {
    CtsTestBase::SetUp();
    if (HasFatalFailure() || IsSkipped()) return;

    IREE_ASSERT_OK(iree_hal_executable_cache_create(
        device_, iree_make_cstring_view("default"), &executable_cache_));

    iree_hal_executable_params_t executable_params;
    iree_hal_executable_params_initialize(&executable_params);
    executable_params.caching_mode =
        IREE_HAL_EXECUTABLE_CACHING_MODE_ALIAS_PROVIDED_DATA;
    executable_params.executable_format =
        iree_make_cstring_view(executable_format());
    executable_params.executable_data = executable_data(iree_make_cstring_view(
        "command_buffer_dispatch_constants_bindings_test.bin"));
    IREE_ASSERT_OK(iree_hal_executable_cache_prepare_executable(
        executable_cache_, &executable_params, &executable_));

    std::vector<uint8_t> requirement_executable_data;
    IREE_ASSERT_OK(
        MakeBdaRequirementExecutableData(&requirement_executable_data));
    iree_hal_executable_params_t requirement_executable_params;
    iree_hal_executable_params_initialize(&requirement_executable_params);
    requirement_executable_params.executable_format =
        iree_make_cstring_view("vulkan-spirv-bda-v1");
    requirement_executable_params.executable_data = iree_make_const_byte_span(
        requirement_executable_data.data(), requirement_executable_data.size());
    IREE_ASSERT_OK(iree_hal_executable_cache_prepare_executable(
        executable_cache_, &requirement_executable_params,
        &requirement_executable_));
  }

  void TearDown() override {
    iree_hal_executable_release(requirement_executable_);
    requirement_executable_ = nullptr;
    iree_hal_executable_release(executable_);
    executable_ = nullptr;
    iree_hal_executable_cache_release(executable_cache_);
    executable_cache_ = nullptr;
    CtsTestBase::TearDown();
  }

  iree_const_byte_span_t constants() const {
    return iree_make_const_byte_span(constant_data_, sizeof(constant_data_));
  }

  iree_status_t CreateInputOutputBuffers(
      iree_hal_buffer_t** out_input_buffer,
      iree_hal_buffer_t** out_output_buffer) {
    *out_input_buffer = nullptr;
    *out_output_buffer = nullptr;
    const uint32_t input_data[4] = {1, 2, 3, 4};
    IREE_RETURN_IF_ERROR(CreateDeviceBufferWithData(
        input_data, sizeof(input_data), out_input_buffer));
    iree_status_t status =
        CreateZeroedDeviceBuffer(sizeof(input_data), out_output_buffer);
    if (!iree_status_is_ok(status)) {
      iree_hal_buffer_release(*out_input_buffer);
      *out_input_buffer = nullptr;
    }
    return status;
  }

  static constexpr uint32_t constant_data_[2] = {3, 10};

  iree_hal_executable_cache_t* executable_cache_ = nullptr;
  iree_hal_executable_t* executable_ = nullptr;
  iree_hal_executable_t* requirement_executable_ = nullptr;
};

TEST_P(BdaDispatchValidationTest, QueueDispatchRejectsBindingCountMismatch) {
  iree_hal_buffer_t* input_buffer = nullptr;
  iree_hal_buffer_t* output_buffer = nullptr;
  IREE_ASSERT_OK(CreateInputOutputBuffers(&input_buffer, &output_buffer));

  iree_hal_buffer_ref_t binding_refs[1] = {
      iree_hal_make_buffer_ref(input_buffer, /*offset=*/0,
                               iree_hal_buffer_byte_length(input_buffer)),
  };
  const iree_hal_buffer_ref_list_t bindings = {
      /*.count=*/IREE_ARRAYSIZE(binding_refs),
      /*.values=*/binding_refs,
  };

  EXPECT_THAT(
      Status(iree_hal_device_queue_dispatch(
          device_, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
          iree_hal_semaphore_list_empty(), executable_, /*export_ordinal=*/0,
          iree_hal_make_static_dispatch_config(1, 1, 1), constants(), bindings,
          IREE_HAL_DISPATCH_FLAG_NONE)),
      StatusIs(StatusCode::kInvalidArgument));

  iree_hal_buffer_release(output_buffer);
  iree_hal_buffer_release(input_buffer);
}

TEST_P(BdaDispatchValidationTest,
       CommandBufferDispatchRejectsBindingCountMismatch) {
  iree_hal_buffer_t* input_buffer = nullptr;
  iree_hal_buffer_t* output_buffer = nullptr;
  IREE_ASSERT_OK(CreateInputOutputBuffers(&input_buffer, &output_buffer));

  iree_hal_command_buffer_t* command_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, &command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));

  iree_hal_buffer_ref_t binding_refs[1] = {
      iree_hal_make_buffer_ref(input_buffer, /*offset=*/0,
                               iree_hal_buffer_byte_length(input_buffer)),
  };
  const iree_hal_buffer_ref_list_t bindings = {
      /*.count=*/IREE_ARRAYSIZE(binding_refs),
      /*.values=*/binding_refs,
  };

  EXPECT_THAT(Status(iree_hal_command_buffer_dispatch(
                  command_buffer, executable_, /*entry_point=*/0,
                  iree_hal_make_static_dispatch_config(1, 1, 1), constants(),
                  bindings, IREE_HAL_DISPATCH_FLAG_NONE)),
              StatusIs(StatusCode::kInvalidArgument));

  iree_hal_command_buffer_release(command_buffer);
  iree_hal_buffer_release(output_buffer);
  iree_hal_buffer_release(input_buffer);
}

TEST_P(BdaDispatchValidationTest, QueueDispatchRejectsEmptyBindingRange) {
  iree_hal_buffer_t* input_buffer = nullptr;
  iree_hal_buffer_t* output_buffer = nullptr;
  IREE_ASSERT_OK(CreateInputOutputBuffers(&input_buffer, &output_buffer));

  iree_hal_buffer_ref_t binding_refs[2] = {
      iree_hal_make_buffer_ref(input_buffer, /*offset=*/0,
                               iree_hal_buffer_byte_length(input_buffer)),
      iree_hal_make_buffer_ref(output_buffer, /*offset=*/0, /*length=*/0),
  };
  const iree_hal_buffer_ref_list_t bindings = {
      /*.count=*/IREE_ARRAYSIZE(binding_refs),
      /*.values=*/binding_refs,
  };

  EXPECT_THAT(
      Status(iree_hal_device_queue_dispatch(
          device_, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
          iree_hal_semaphore_list_empty(), executable_, /*export_ordinal=*/0,
          iree_hal_make_static_dispatch_config(1, 1, 1), constants(), bindings,
          IREE_HAL_DISPATCH_FLAG_NONE)),
      StatusIs(StatusCode::kInvalidArgument));

  iree_hal_buffer_release(output_buffer);
  iree_hal_buffer_release(input_buffer);
}

TEST_P(BdaDispatchValidationTest,
       CommandBufferExecuteRejectsEmptyBindingRange) {
  iree_hal_buffer_t* input_buffer = nullptr;
  iree_hal_buffer_t* output_buffer = nullptr;
  IREE_ASSERT_OK(CreateInputOutputBuffers(&input_buffer, &output_buffer));

  iree_hal_command_buffer_t* command_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, &command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));

  iree_hal_buffer_ref_t binding_refs[2] = {
      iree_hal_make_buffer_ref(input_buffer, /*offset=*/0,
                               iree_hal_buffer_byte_length(input_buffer)),
      iree_hal_make_buffer_ref(output_buffer, /*offset=*/0, /*length=*/0),
  };
  const iree_hal_buffer_ref_list_t bindings = {
      /*.count=*/IREE_ARRAYSIZE(binding_refs),
      /*.values=*/binding_refs,
  };
  IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
      command_buffer, executable_, /*entry_point=*/0,
      iree_hal_make_static_dispatch_config(1, 1, 1), constants(), bindings,
      IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  EXPECT_THAT(
      Status(iree_hal_device_queue_execute(
          device_, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
          iree_hal_semaphore_list_empty(), command_buffer,
          iree_hal_buffer_binding_table_empty(), IREE_HAL_EXECUTE_FLAG_NONE)),
      StatusIs(StatusCode::kInvalidArgument));

  iree_hal_command_buffer_release(command_buffer);
  iree_hal_buffer_release(output_buffer);
  iree_hal_buffer_release(input_buffer);
}

TEST_P(BdaDispatchValidationTest, QueueDispatchRejectsMinimumBindingLength) {
  iree_hal_buffer_t* input_buffer = nullptr;
  iree_hal_buffer_t* output_buffer = nullptr;
  IREE_ASSERT_OK(
      CreateZeroedDeviceBuffer(kRequiredBindingLength - 1, &input_buffer));
  IREE_ASSERT_OK(CreateZeroedDeviceBuffer(64, &output_buffer));

  iree_hal_buffer_ref_t binding_refs[2] = {
      iree_hal_make_buffer_ref(input_buffer, /*offset=*/0,
                               iree_hal_buffer_byte_length(input_buffer)),
      iree_hal_make_buffer_ref(output_buffer, /*offset=*/0,
                               iree_hal_buffer_byte_length(output_buffer)),
  };
  const iree_hal_buffer_ref_list_t bindings = {
      /*.count=*/IREE_ARRAYSIZE(binding_refs),
      /*.values=*/binding_refs,
  };

  EXPECT_THAT(
      Status(iree_hal_device_queue_dispatch(
          device_, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
          iree_hal_semaphore_list_empty(), requirement_executable_,
          /*export_ordinal=*/0, iree_hal_make_static_dispatch_config(1, 1, 1),
          iree_const_byte_span_empty(), bindings, IREE_HAL_DISPATCH_FLAG_NONE)),
      StatusIs(StatusCode::kOutOfRange));

  iree_hal_buffer_release(output_buffer);
  iree_hal_buffer_release(input_buffer);
}

TEST_P(BdaDispatchValidationTest,
       CommandBufferExecuteRejectsMinimumBindingLength) {
  iree_hal_buffer_t* input_buffer = nullptr;
  iree_hal_buffer_t* output_buffer = nullptr;
  IREE_ASSERT_OK(
      CreateZeroedDeviceBuffer(kRequiredBindingLength - 1, &input_buffer));
  IREE_ASSERT_OK(CreateZeroedDeviceBuffer(64, &output_buffer));

  iree_hal_command_buffer_t* command_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, &command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));

  iree_hal_buffer_ref_t binding_refs[2] = {
      iree_hal_make_buffer_ref(input_buffer, /*offset=*/0,
                               iree_hal_buffer_byte_length(input_buffer)),
      iree_hal_make_buffer_ref(output_buffer, /*offset=*/0,
                               iree_hal_buffer_byte_length(output_buffer)),
  };
  const iree_hal_buffer_ref_list_t bindings = {
      /*.count=*/IREE_ARRAYSIZE(binding_refs),
      /*.values=*/binding_refs,
  };
  IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
      command_buffer, requirement_executable_, /*entry_point=*/0,
      iree_hal_make_static_dispatch_config(1, 1, 1),
      iree_const_byte_span_empty(), bindings, IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  EXPECT_THAT(
      Status(iree_hal_device_queue_execute(
          device_, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
          iree_hal_semaphore_list_empty(), command_buffer,
          iree_hal_buffer_binding_table_empty(), IREE_HAL_EXECUTE_FLAG_NONE)),
      StatusIs(StatusCode::kOutOfRange));

  iree_hal_command_buffer_release(command_buffer);
  iree_hal_buffer_release(output_buffer);
  iree_hal_buffer_release(input_buffer);
}

TEST_P(BdaDispatchValidationTest, QueueDispatchRejectsMinimumBindingAlignment) {
  iree_hal_buffer_t* input_buffer = nullptr;
  iree_hal_buffer_t* output_buffer = nullptr;
  IREE_ASSERT_OK(
      CreateZeroedDeviceBuffer(kRequiredBindingLength, &input_buffer));
  IREE_ASSERT_OK(CreateZeroedDeviceBuffer(64, &output_buffer));

  const iree_device_size_t output_offset = 1;
  iree_hal_buffer_ref_t binding_refs[2] = {
      iree_hal_make_buffer_ref(input_buffer, /*offset=*/0,
                               iree_hal_buffer_byte_length(input_buffer)),
      iree_hal_make_buffer_ref(
          output_buffer, output_offset,
          iree_hal_buffer_byte_length(output_buffer) - output_offset),
  };
  const iree_hal_buffer_ref_list_t bindings = {
      /*.count=*/IREE_ARRAYSIZE(binding_refs),
      /*.values=*/binding_refs,
  };

  EXPECT_THAT(
      Status(iree_hal_device_queue_dispatch(
          device_, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
          iree_hal_semaphore_list_empty(), requirement_executable_,
          /*export_ordinal=*/0, iree_hal_make_static_dispatch_config(1, 1, 1),
          iree_const_byte_span_empty(), bindings, IREE_HAL_DISPATCH_FLAG_NONE)),
      StatusIs(StatusCode::kInvalidArgument));

  iree_hal_buffer_release(output_buffer);
  iree_hal_buffer_release(input_buffer);
}

TEST_P(BdaDispatchValidationTest,
       CommandBufferExecuteRejectsMinimumBindingAlignment) {
  iree_hal_buffer_t* input_buffer = nullptr;
  iree_hal_buffer_t* output_buffer = nullptr;
  IREE_ASSERT_OK(
      CreateZeroedDeviceBuffer(kRequiredBindingLength, &input_buffer));
  IREE_ASSERT_OK(CreateZeroedDeviceBuffer(64, &output_buffer));

  const iree_device_size_t output_offset = 1;

  iree_hal_command_buffer_t* command_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, &command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));

  iree_hal_buffer_ref_t binding_refs[2] = {
      iree_hal_make_buffer_ref(input_buffer, /*offset=*/0,
                               iree_hal_buffer_byte_length(input_buffer)),
      iree_hal_make_buffer_ref(
          output_buffer, output_offset,
          iree_hal_buffer_byte_length(output_buffer) - output_offset),
  };
  const iree_hal_buffer_ref_list_t bindings = {
      /*.count=*/IREE_ARRAYSIZE(binding_refs),
      /*.values=*/binding_refs,
  };
  IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
      command_buffer, requirement_executable_, /*entry_point=*/0,
      iree_hal_make_static_dispatch_config(1, 1, 1),
      iree_const_byte_span_empty(), bindings, IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  EXPECT_THAT(
      Status(iree_hal_device_queue_execute(
          device_, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
          iree_hal_semaphore_list_empty(), command_buffer,
          iree_hal_buffer_binding_table_empty(), IREE_HAL_EXECUTE_FLAG_NONE)),
      StatusIs(StatusCode::kInvalidArgument));

  iree_hal_command_buffer_release(command_buffer);
  iree_hal_buffer_release(output_buffer);
  iree_hal_buffer_release(input_buffer);
}

CTS_REGISTER_EXECUTABLE_TEST_SUITE(BdaDispatchValidationTest);

}  // namespace iree::hal::cts
