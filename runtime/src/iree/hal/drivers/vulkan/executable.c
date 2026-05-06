// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/executable.h"

#include <string.h>

#include "iree/base/internal/atomics.h"
#include "iree/base/internal/flatcc/parsing.h"
#include "iree/hal/drivers/vulkan/spirv.h"
#include "iree/hal/utils/executable_debug_info.h"
#include "iree/hal/utils/executable_header.h"
#include "iree/schemas/vulkan_executable_def_reader.h"
#include "iree/schemas/vulkan_executable_def_verifier.h"

//===----------------------------------------------------------------------===//
// Executable Format
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_vulkan_executable_infer_format(
    iree_const_byte_span_t executable_data,
    iree_host_size_t executable_format_capacity, char* executable_format,
    iree_host_size_t* out_inferred_size) {
  IREE_ASSERT_ARGUMENT(executable_format);
  IREE_ASSERT_ARGUMENT(out_inferred_size);
  *out_inferred_size = 0;

  const bool unsafe_infer_size = executable_data.data_length == 0;
  iree_const_byte_span_t flatbuffer_data = iree_const_byte_span_empty();
  IREE_RETURN_IF_ERROR(iree_hal_read_executable_flatbuffer_header(
      executable_data, unsafe_infer_size,
      iree_hal_vulkan_ExecutableDef_file_identifier, &flatbuffer_data));

  const int verify_result = iree_hal_vulkan_ExecutableDef_verify_as_root(
      flatbuffer_data.data, flatbuffer_data.data_length);
  if (verify_result != flatcc_verify_ok) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "flatbuffer verification failed: %s",
                            flatcc_verify_error_string(verify_result));
  }

  iree_string_view_t format = IREE_SV("vulkan-spirv-fb");
  iree_hal_vulkan_ExecutableDef_table_t executable_def =
      iree_hal_vulkan_ExecutableDef_as_root(flatbuffer_data.data);
  iree_hal_vulkan_PipelineDef_vec_t pipelines_vec =
      iree_hal_vulkan_ExecutableDef_pipelines_get(executable_def);
  const iree_host_size_t pipeline_count =
      iree_hal_vulkan_PipelineDef_vec_len(pipelines_vec);
  for (iree_host_size_t i = 0; i < pipeline_count; ++i) {
    iree_hal_vulkan_PipelineDef_table_t pipeline_def =
        iree_hal_vulkan_PipelineDef_vec_at(pipelines_vec, i);
    if (iree_hal_vulkan_PipelineDef_dispatch_abi_get(pipeline_def) ==
        iree_hal_vulkan_DispatchAbi_BDA_V1) {
      format = IREE_SV("vulkan-spirv-bda-v1");
      break;
    }
  }

  if (format.size >= executable_format_capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "executable format buffer too small");
  }
  memcpy(executable_format, format.data, format.size);
  executable_format[format.size] = 0;

  *out_inferred_size =
      sizeof(iree_flatbuffer_file_header_t) + flatbuffer_data.data_length;
  return iree_ok_status();
}

bool iree_hal_vulkan_executable_format_supported(
    iree_hal_vulkan_features_t enabled_features,
    iree_hal_vulkan_dispatch_abis_t enabled_dispatch_abis,
    iree_string_view_t executable_format) {
  if (iree_string_view_equal(executable_format, IREE_SV("vulkan-spirv-fb"))) {
    return iree_all_bits_set(enabled_dispatch_abis,
                             IREE_HAL_VULKAN_DISPATCH_ABI_DESCRIPTOR);
  }
  if (iree_string_view_equal(executable_format,
                             IREE_SV("vulkan-spirv-fb-ptr"))) {
    return iree_all_bits_set(enabled_dispatch_abis,
                             IREE_HAL_VULKAN_DISPATCH_ABI_DESCRIPTOR) &&
           iree_all_bits_set(
               enabled_features,
               IREE_HAL_VULKAN_FEATURE_ENABLE_BUFFER_DEVICE_ADDRESSES);
  }
  if (iree_string_view_equal(executable_format,
                             IREE_SV("vulkan-spirv-bda-raw"))) {
    return iree_all_bits_set(enabled_dispatch_abis,
                             IREE_HAL_VULKAN_DISPATCH_ABI_BDA) &&
           iree_all_bits_set(
               enabled_features,
               IREE_HAL_VULKAN_FEATURE_ENABLE_BUFFER_DEVICE_ADDRESSES);
  }
  if (iree_string_view_equal(executable_format,
                             IREE_SV("vulkan-spirv-bda-v1"))) {
    return iree_all_bits_set(enabled_dispatch_abis,
                             IREE_HAL_VULKAN_DISPATCH_ABI_BDA) &&
           iree_all_bits_set(
               enabled_features,
               IREE_HAL_VULKAN_FEATURE_ENABLE_BUFFER_DEVICE_ADDRESSES);
  }
  return false;
}

static iree_hal_vulkan_dispatch_abis_t
iree_hal_vulkan_executable_dispatch_abi_for_format(
    iree_string_view_t executable_format) {
  if (iree_string_view_equal(executable_format, IREE_SV("vulkan-spirv-fb")) ||
      iree_string_view_equal(executable_format,
                             IREE_SV("vulkan-spirv-fb-ptr"))) {
    return IREE_HAL_VULKAN_DISPATCH_ABI_DESCRIPTOR;
  }
  if (iree_string_view_equal(executable_format,
                             IREE_SV("vulkan-spirv-bda-raw")) ||
      iree_string_view_equal(executable_format,
                             IREE_SV("vulkan-spirv-bda-v1"))) {
    return IREE_HAL_VULKAN_DISPATCH_ABI_BDA;
  }
  return IREE_HAL_VULKAN_DISPATCH_ABI_NONE;
}

//===----------------------------------------------------------------------===//
// FlatBuffer Verification
//===----------------------------------------------------------------------===//

static bool iree_hal_vulkan_stage_flags_include_compute(
    VkShaderStageFlags stage_flags) {
  return stage_flags == VK_SHADER_STAGE_ALL ||
         iree_any_bit_set(stage_flags, VK_SHADER_STAGE_COMPUTE_BIT);
}

static iree_status_t iree_hal_vulkan_verify_descriptor_set_layout_def(
    iree_hal_vulkan_DescriptorSetLayoutDef_table_t descriptor_set_layout_def,
    iree_host_size_t descriptor_set_layout_ordinal) {
  if (!descriptor_set_layout_def) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "descriptor_set_layouts[%" PRIhsz "] is NULL",
                            descriptor_set_layout_ordinal);
  }

  iree_hal_vulkan_DescriptorSetLayoutBindingDef_vec_t bindings_vec =
      iree_hal_vulkan_DescriptorSetLayoutDef_bindings_get(
          descriptor_set_layout_def);
  const iree_host_size_t binding_count =
      iree_hal_vulkan_DescriptorSetLayoutBindingDef_vec_len(bindings_vec);
  for (iree_host_size_t i = 0; i < binding_count; ++i) {
    iree_hal_vulkan_DescriptorSetLayoutBindingDef_table_t binding_def =
        iree_hal_vulkan_DescriptorSetLayoutBindingDef_vec_at(bindings_vec, i);
    if (!binding_def) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "descriptor_set_layouts[%" PRIhsz
                              "] bindings[%" PRIhsz "] is NULL",
                              descriptor_set_layout_ordinal, i);
    }

    const uint32_t binding =
        iree_hal_vulkan_DescriptorSetLayoutBindingDef_binding_get(binding_def);
    for (iree_host_size_t j = 0; j < i; ++j) {
      iree_hal_vulkan_DescriptorSetLayoutBindingDef_table_t existing_def =
          iree_hal_vulkan_DescriptorSetLayoutBindingDef_vec_at(bindings_vec, j);
      if (iree_hal_vulkan_DescriptorSetLayoutBindingDef_binding_get(
              existing_def) == binding) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "descriptor_set_layouts[%" PRIhsz
                                "] has duplicate binding ordinal %u",
                                descriptor_set_layout_ordinal, binding);
      }
    }

    const uint32_t descriptor_count =
        iree_hal_vulkan_DescriptorSetLayoutBindingDef_descriptor_count_get(
            binding_def);
    if (descriptor_count == 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "descriptor_set_layouts[%" PRIhsz
                              "] bindings[%" PRIhsz "] has zero descriptors",
                              descriptor_set_layout_ordinal, i);
    }

    switch (iree_hal_vulkan_DescriptorSetLayoutBindingDef_descriptor_type_get(
        binding_def)) {
      case iree_hal_vulkan_VkDescriptorType_SAMPLER:
      case iree_hal_vulkan_VkDescriptorType_UNIFORM_BUFFER:
      case iree_hal_vulkan_VkDescriptorType_STORAGE_BUFFER:
        break;
      default:
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "descriptor_set_layouts[%" PRIhsz
                                "] bindings[%" PRIhsz
                                "] has an unsupported descriptor type",
                                descriptor_set_layout_ordinal, i);
    }

    const VkShaderStageFlags stage_flags =
        iree_hal_vulkan_DescriptorSetLayoutBindingDef_stage_flags_get(
            binding_def);
    if (!iree_hal_vulkan_stage_flags_include_compute(stage_flags)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "descriptor_set_layouts[%" PRIhsz
                              "] bindings[%" PRIhsz
                              "] stage flags 0x%08x do not include compute",
                              descriptor_set_layout_ordinal, i, stage_flags);
    }
  }

  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_count_descriptor_set_layout_def(
    iree_hal_vulkan_DescriptorSetLayoutDef_table_t descriptor_set_layout_def,
    uint64_t* inout_sampler_count, uint64_t* inout_uniform_buffer_count,
    uint64_t* inout_storage_buffer_count, uint64_t* inout_binding_count) {
  iree_hal_vulkan_DescriptorSetLayoutBindingDef_vec_t bindings_vec =
      iree_hal_vulkan_DescriptorSetLayoutDef_bindings_get(
          descriptor_set_layout_def);
  const iree_host_size_t binding_count =
      iree_hal_vulkan_DescriptorSetLayoutBindingDef_vec_len(bindings_vec);
  for (iree_host_size_t i = 0; i < binding_count; ++i) {
    iree_hal_vulkan_DescriptorSetLayoutBindingDef_table_t binding_def =
        iree_hal_vulkan_DescriptorSetLayoutBindingDef_vec_at(bindings_vec, i);
    const uint32_t descriptor_count =
        iree_hal_vulkan_DescriptorSetLayoutBindingDef_descriptor_count_get(
            binding_def);
    if (inout_binding_count) *inout_binding_count += descriptor_count;
    switch (iree_hal_vulkan_DescriptorSetLayoutBindingDef_descriptor_type_get(
        binding_def)) {
      case iree_hal_vulkan_VkDescriptorType_SAMPLER:
        if (inout_sampler_count) *inout_sampler_count += descriptor_count;
        break;
      case iree_hal_vulkan_VkDescriptorType_UNIFORM_BUFFER:
        if (inout_uniform_buffer_count) {
          *inout_uniform_buffer_count += descriptor_count;
        }
        break;
      case iree_hal_vulkan_VkDescriptorType_STORAGE_BUFFER:
        if (inout_storage_buffer_count) {
          *inout_storage_buffer_count += descriptor_count;
        }
        break;
      default:
        break;
    }
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_calculate_pipeline_layout_counts(
    iree_hal_vulkan_DescriptorSetLayoutDef_vec_t descriptor_set_layouts_vec,
    iree_hal_vulkan_PipelineLayoutDef_table_t pipeline_layout_def,
    uint16_t* out_constant_count, uint16_t* out_binding_count) {
  *out_constant_count = 0;
  *out_binding_count = 0;

  uint64_t constant_byte_count = 0;
  iree_hal_vulkan_PushConstantRange_vec_t push_constant_ranges_vec =
      iree_hal_vulkan_PipelineLayoutDef_push_constant_ranges_get(
          pipeline_layout_def);
  const iree_host_size_t push_constant_range_count =
      iree_hal_vulkan_PushConstantRange_vec_len(push_constant_ranges_vec);
  for (iree_host_size_t i = 0; i < push_constant_range_count; ++i) {
    const iree_hal_vulkan_PushConstantRange_t* push_constant_range =
        iree_hal_vulkan_PushConstantRange_vec_at(push_constant_ranges_vec, i);
    constant_byte_count =
        iree_max(constant_byte_count, (uint64_t)push_constant_range->offset +
                                          push_constant_range->size);
  }
  const uint64_t constant_count = constant_byte_count / sizeof(uint32_t);
  if (constant_count > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "pipeline layout declares %" PRIu64
                            " constants, exceeding the HAL limit %u",
                            constant_count, UINT16_MAX);
  }

  uint64_t binding_count = 0;
  flatbuffers_uint32_vec_t descriptor_set_layout_ordinals_vec =
      iree_hal_vulkan_PipelineLayoutDef_descriptor_set_layout_ordinals_get(
          pipeline_layout_def);
  const iree_host_size_t descriptor_set_layout_ordinal_count =
      flatbuffers_uint32_vec_len(descriptor_set_layout_ordinals_vec);
  for (iree_host_size_t i = 0; i < descriptor_set_layout_ordinal_count; ++i) {
    const uint32_t descriptor_set_layout_ordinal =
        flatbuffers_uint32_vec_at(descriptor_set_layout_ordinals_vec, i);
    iree_hal_vulkan_DescriptorSetLayoutDef_table_t descriptor_set_layout_def =
        iree_hal_vulkan_DescriptorSetLayoutDef_vec_at(
            descriptor_set_layouts_vec, descriptor_set_layout_ordinal);
    IREE_RETURN_IF_ERROR(iree_hal_vulkan_count_descriptor_set_layout_def(
        descriptor_set_layout_def, /*inout_sampler_count=*/NULL,
        /*inout_uniform_buffer_count=*/NULL,
        /*inout_storage_buffer_count=*/NULL, &binding_count));
  }
  if (binding_count > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "pipeline layout declares %" PRIu64
                            " bindings, exceeding the HAL limit %u",
                            binding_count, UINT16_MAX);
  }

  *out_constant_count = (uint16_t)constant_count;
  *out_binding_count = (uint16_t)binding_count;
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_verify_pipeline_layout_def(
    const iree_hal_vulkan_physical_device_snapshot_t* physical_device,
    iree_hal_vulkan_DescriptorSetLayoutDef_vec_t descriptor_set_layouts_vec,
    iree_hal_vulkan_PipelineLayoutDef_table_t pipeline_layout_def,
    iree_host_size_t pipeline_layout_ordinal) {
  if (!pipeline_layout_def) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pipeline_layouts[%" PRIhsz "] is NULL",
                            pipeline_layout_ordinal);
  }

  const VkPhysicalDeviceLimits* limits =
      &physical_device->properties2.properties.limits;
  uint64_t sampler_count = 0;
  uint64_t uniform_buffer_count = 0;
  uint64_t storage_buffer_count = 0;
  uint64_t binding_count = 0;
  flatbuffers_uint32_vec_t descriptor_set_layout_ordinals_vec =
      iree_hal_vulkan_PipelineLayoutDef_descriptor_set_layout_ordinals_get(
          pipeline_layout_def);
  const iree_host_size_t descriptor_set_layout_ordinal_count =
      flatbuffers_uint32_vec_len(descriptor_set_layout_ordinals_vec);
  for (iree_host_size_t i = 0; i < descriptor_set_layout_ordinal_count; ++i) {
    const uint32_t descriptor_set_layout_ordinal =
        flatbuffers_uint32_vec_at(descriptor_set_layout_ordinals_vec, i);
    if (descriptor_set_layout_ordinal >=
        iree_hal_vulkan_DescriptorSetLayoutDef_vec_len(
            descriptor_set_layouts_vec)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "pipeline_layouts[%" PRIhsz
          "] references descriptor set layout ordinal %u out of range",
          pipeline_layout_ordinal, descriptor_set_layout_ordinal);
    }
    iree_hal_vulkan_DescriptorSetLayoutDef_table_t descriptor_set_layout_def =
        iree_hal_vulkan_DescriptorSetLayoutDef_vec_at(
            descriptor_set_layouts_vec, descriptor_set_layout_ordinal);
    IREE_RETURN_IF_ERROR(iree_hal_vulkan_count_descriptor_set_layout_def(
        descriptor_set_layout_def, &sampler_count, &uniform_buffer_count,
        &storage_buffer_count, &binding_count));
  }

  if (sampler_count > limits->maxPerStageDescriptorSamplers) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pipeline_layouts[%" PRIhsz "] declares %" PRIu64
                            " samplers, exceeding device "
                            "limit %u",
                            pipeline_layout_ordinal, sampler_count,
                            limits->maxPerStageDescriptorSamplers);
  }
  if (uniform_buffer_count > limits->maxPerStageDescriptorUniformBuffers) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pipeline_layouts[%" PRIhsz "] declares %" PRIu64
                            " uniform buffers, exceeding device limit %u",
                            pipeline_layout_ordinal, uniform_buffer_count,
                            limits->maxPerStageDescriptorUniformBuffers);
  }
  if (storage_buffer_count > limits->maxPerStageDescriptorStorageBuffers) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pipeline_layouts[%" PRIhsz "] declares %" PRIu64
                            " storage buffers, exceeding device limit %u",
                            pipeline_layout_ordinal, storage_buffer_count,
                            limits->maxPerStageDescriptorStorageBuffers);
  }

  iree_hal_vulkan_PushConstantRange_vec_t push_constant_ranges_vec =
      iree_hal_vulkan_PipelineLayoutDef_push_constant_ranges_get(
          pipeline_layout_def);
  const iree_host_size_t push_constant_range_count =
      iree_hal_vulkan_PushConstantRange_vec_len(push_constant_ranges_vec);
  for (iree_host_size_t i = 0; i < push_constant_range_count; ++i) {
    const iree_hal_vulkan_PushConstantRange_t* push_constant_range =
        iree_hal_vulkan_PushConstantRange_vec_at(push_constant_ranges_vec, i);
    if (!iree_hal_vulkan_stage_flags_include_compute(
            push_constant_range->stage_flags)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "pipeline_layouts[%" PRIhsz "] push_constant_ranges[%" PRIhsz
          "] stage flags 0x%08x do not include compute",
          pipeline_layout_ordinal, i, push_constant_range->stage_flags);
    }
    if ((push_constant_range->offset % sizeof(uint32_t)) != 0 ||
        (push_constant_range->size % sizeof(uint32_t)) != 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "pipeline_layouts[%" PRIhsz
                              "] push_constant_ranges[%" PRIhsz
                              "] must be 4-byte aligned",
                              pipeline_layout_ordinal, i);
    }
    const uint64_t range_end =
        (uint64_t)push_constant_range->offset + push_constant_range->size;
    if (range_end > limits->maxPushConstantsSize) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "pipeline_layouts[%" PRIhsz "] push_constant_ranges[%" PRIhsz
          "] end offset %" PRIu64 " exceeds device limit %u",
          pipeline_layout_ordinal, i, range_end, limits->maxPushConstantsSize);
    }
  }

  uint16_t constant_count = 0;
  uint16_t hal_binding_count = 0;
  return iree_hal_vulkan_calculate_pipeline_layout_counts(
      descriptor_set_layouts_vec, pipeline_layout_def, &constant_count,
      &hal_binding_count);
}

static iree_status_t iree_hal_vulkan_verify_shader_module_def(
    iree_hal_vulkan_ShaderModuleDef_table_t shader_module_def,
    iree_host_size_t shader_module_ordinal) {
  if (!shader_module_def) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "shader_modules[%" PRIhsz "] is NULL",
                            shader_module_ordinal);
  }
  flatbuffers_uint32_vec_t spirv_code_vec =
      iree_hal_vulkan_ShaderModuleDef_spirv_code_get(shader_module_def);
  const iree_host_size_t spirv_word_count =
      flatbuffers_uint32_vec_len(spirv_code_vec);
  if (spirv_word_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "shader_modules[%" PRIhsz "] SPIR-V is empty",
                            shader_module_ordinal);
  }
  if (flatbuffers_uint32_vec_at(spirv_code_vec, 0) != 0x07230203u) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "shader_modules[%" PRIhsz
                            "] does not start with SPIR-V magic",
                            shader_module_ordinal);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_verify_required_subgroup_size(
    const iree_hal_vulkan_physical_device_snapshot_t* physical_device,
    iree_hal_vulkan_features_t enabled_features, uint32_t subgroup_size,
    iree_host_size_t pipeline_ordinal) {
  if (subgroup_size == 0) return iree_ok_status();

  if (!iree_all_bits_set(
          enabled_features,
          IREE_HAL_VULKAN_FEATURE_ENABLE_SUBGROUP_SIZE_CONTROL)) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "pipelines[%" PRIhsz
        "] requires subgroup size %u but subgroupSizeControl is not enabled",
        pipeline_ordinal, subgroup_size);
  }

  const VkPhysicalDeviceSubgroupSizeControlProperties* properties =
      &physical_device->subgroup_size_control_properties;
  if (subgroup_size < properties->minSubgroupSize ||
      subgroup_size > properties->maxSubgroupSize) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "pipelines[%" PRIhsz
        "] requires subgroup size %u outside device range [%u, %u]",
        pipeline_ordinal, subgroup_size, properties->minSubgroupSize,
        properties->maxSubgroupSize);
  }
  if (!iree_any_bit_set(properties->requiredSubgroupSizeStages,
                        VK_SHADER_STAGE_COMPUTE_BIT)) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "pipelines[%" PRIhsz
        "] requires subgroup size %u but compute pipelines do not support "
        "required subgroup sizes on this device",
        pipeline_ordinal, subgroup_size);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_pipeline_def_dispatch_abi(
    iree_hal_vulkan_PipelineDef_table_t pipeline_def,
    iree_hal_vulkan_dispatch_abis_t* out_dispatch_abi) {
  switch (iree_hal_vulkan_PipelineDef_dispatch_abi_get(pipeline_def)) {
    case iree_hal_vulkan_DispatchAbi_DESCRIPTOR:
      *out_dispatch_abi = IREE_HAL_VULKAN_DISPATCH_ABI_DESCRIPTOR;
      return iree_ok_status();
    case iree_hal_vulkan_DispatchAbi_BDA_V1:
      *out_dispatch_abi = IREE_HAL_VULKAN_DISPATCH_ABI_BDA;
      return iree_ok_status();
    default:
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "pipeline declares unknown dispatch ABI %u",
          (uint32_t)iree_hal_vulkan_PipelineDef_dispatch_abi_get(pipeline_def));
  }
}

static bool iree_hal_vulkan_push_constant_layout_covers_range(
    iree_hal_vulkan_PipelineLayoutDef_table_t pipeline_layout_def,
    uint32_t required_offset, uint32_t required_length) {
  if (required_length == 0) return true;
  const uint64_t required_end =
      (uint64_t)required_offset + (uint64_t)required_length;
  iree_hal_vulkan_PushConstantRange_vec_t push_constant_ranges_vec =
      iree_hal_vulkan_PipelineLayoutDef_push_constant_ranges_get(
          pipeline_layout_def);
  const iree_host_size_t push_constant_range_count =
      iree_hal_vulkan_PushConstantRange_vec_len(push_constant_ranges_vec);
  for (iree_host_size_t i = 0; i < push_constant_range_count; ++i) {
    const iree_hal_vulkan_PushConstantRange_t* push_constant_range =
        iree_hal_vulkan_PushConstantRange_vec_at(push_constant_ranges_vec, i);
    const uint64_t range_end =
        (uint64_t)push_constant_range->offset + push_constant_range->size;
    if (iree_hal_vulkan_stage_flags_include_compute(
            push_constant_range->stage_flags) &&
        required_offset >= push_constant_range->offset &&
        required_end <= range_end) {
      return true;
    }
  }
  return false;
}

static iree_status_t iree_hal_vulkan_verify_bda_dispatch_layout_def(
    iree_hal_vulkan_PipelineLayoutDef_table_t pipeline_layout_def,
    iree_hal_vulkan_BdaDispatchLayoutDef_table_t bda_dispatch_layout_def,
    iree_host_size_t pipeline_ordinal) {
  if (!bda_dispatch_layout_def) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pipelines[%" PRIhsz
                            "] BDA dispatch ABI requires a layout",
                            pipeline_ordinal);
  }

  const uint32_t abi_version =
      iree_hal_vulkan_BdaDispatchLayoutDef_abi_version_get(
          bda_dispatch_layout_def);
  if (abi_version != 1) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pipelines[%" PRIhsz
                            "] BDA dispatch layout version %u is unsupported",
                            pipeline_ordinal, abi_version);
  }

  const uint32_t root_offset =
      iree_hal_vulkan_BdaDispatchLayoutDef_root_push_constant_offset_get(
          bda_dispatch_layout_def);
  const uint32_t root_length =
      iree_hal_vulkan_BdaDispatchLayoutDef_root_push_constant_length_get(
          bda_dispatch_layout_def);
  if (root_length != sizeof(iree_hal_vulkan_bda_dispatch_root_v1_t)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pipelines[%" PRIhsz
                            "] BDA root push constant length %u does not match "
                            "ABI v1 length %" PRIhsz,
                            pipeline_ordinal, root_length,
                            sizeof(iree_hal_vulkan_bda_dispatch_root_v1_t));
  }
  if ((root_offset % sizeof(uint32_t)) != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pipelines[%" PRIhsz
        "] BDA root push constant offset must be 4-byte aligned",
        pipeline_ordinal);
  }
  if (!iree_hal_vulkan_push_constant_layout_covers_range(
          pipeline_layout_def, root_offset, root_length)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pipelines[%" PRIhsz
        "] BDA root push constants are outside the pipeline layout",
        pipeline_ordinal);
  }

  const uint32_t constant_offset =
      iree_hal_vulkan_BdaDispatchLayoutDef_constant_push_constant_offset_get(
          bda_dispatch_layout_def);
  const uint32_t constant_count =
      iree_hal_vulkan_BdaDispatchLayoutDef_constant_count_get(
          bda_dispatch_layout_def);
  if (constant_count > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "pipelines[%" PRIhsz
                            "] BDA layout declares %u constants, exceeding "
                            "the HAL limit %u",
                            pipeline_ordinal, constant_count, UINT16_MAX);
  }
  const uint64_t constant_length = (uint64_t)constant_count * sizeof(uint32_t);
  if ((constant_offset % sizeof(uint32_t)) != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pipelines[%" PRIhsz
        "] BDA constant push constant offset must be 4-byte aligned",
        pipeline_ordinal);
  }
  if (constant_length != 0 &&
      !iree_hal_vulkan_push_constant_layout_covers_range(
          pipeline_layout_def, constant_offset, (uint32_t)constant_length)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pipelines[%" PRIhsz
        "] BDA inline constants are outside the pipeline layout",
        pipeline_ordinal);
  }
  const uint64_t root_end = (uint64_t)root_offset + root_length;
  const uint64_t constant_end = (uint64_t)constant_offset + constant_length;
  if (constant_length != 0 && root_offset < constant_end &&
      constant_offset < root_end) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pipelines[%" PRIhsz
        "] BDA inline constants overlap the hidden root push constants",
        pipeline_ordinal);
  }

  switch (iree_hal_vulkan_BdaDispatchLayoutDef_binding_table_entry_type_get(
      bda_dispatch_layout_def)) {
    case iree_hal_vulkan_BdaBindingTableEntryType_ADDRESS64:
      break;
    case iree_hal_vulkan_BdaBindingTableEntryType_ADDRESS64_LENGTH64:
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "pipelines[%" PRIhsz
          "] BDA checked address64_length64 binding tables require an "
          "explicit sanitizer/debug mode",
          pipeline_ordinal);
    default:
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "pipelines[%" PRIhsz
          "] BDA layout declares an unknown binding table entry type",
          pipeline_ordinal);
  }

  const uint32_t binding_count =
      iree_hal_vulkan_BdaDispatchLayoutDef_binding_count_get(
          bda_dispatch_layout_def);
  if (binding_count > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "pipelines[%" PRIhsz
                            "] BDA layout declares %u bindings, exceeding "
                            "the HAL limit %u",
                            pipeline_ordinal, binding_count, UINT16_MAX);
  }
  iree_hal_vulkan_BdaBindingDef_vec_t bindings_vec =
      iree_hal_vulkan_BdaDispatchLayoutDef_bindings_get(
          bda_dispatch_layout_def);
  const iree_host_size_t binding_requirement_count =
      iree_hal_vulkan_BdaBindingDef_vec_len(bindings_vec);
  if (binding_requirement_count != 0 &&
      binding_requirement_count != binding_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pipelines[%" PRIhsz "] BDA layout has %" PRIhsz
                            " binding requirements but declares %u bindings",
                            pipeline_ordinal, binding_requirement_count,
                            binding_count);
  }
  for (iree_host_size_t i = 0; i < binding_requirement_count; ++i) {
    iree_hal_vulkan_BdaBindingDef_table_t binding_def =
        iree_hal_vulkan_BdaBindingDef_vec_at(bindings_vec, i);
    if (!binding_def) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "pipelines[%" PRIhsz
                              "] BDA binding requirement[%" PRIhsz "] is NULL",
                              pipeline_ordinal, i);
    }
    const uint32_t minimum_alignment =
        iree_hal_vulkan_BdaBindingDef_minimum_alignment_get(binding_def);
    if (minimum_alignment == 0 ||
        !iree_device_size_is_power_of_two(minimum_alignment)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "pipelines[%" PRIhsz "] BDA binding requirement[%" PRIhsz
          "] minimum alignment %u is not a non-zero power of two",
          pipeline_ordinal, i, minimum_alignment);
    }
  }

  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_verify_pipeline_def(
    const iree_hal_vulkan_physical_device_snapshot_t* physical_device,
    iree_hal_vulkan_features_t enabled_features,
    iree_hal_vulkan_ShaderModuleDef_vec_t shader_modules_vec,
    iree_hal_vulkan_PipelineLayoutDef_vec_t pipeline_layouts_vec,
    iree_hal_vulkan_dispatch_abis_t executable_dispatch_abi,
    iree_hal_vulkan_PipelineDef_table_t pipeline_def,
    iree_host_size_t pipeline_ordinal) {
  if (!pipeline_def) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pipelines[%" PRIhsz "] is NULL", pipeline_ordinal);
  }

  const uint32_t shader_module_ordinal =
      iree_hal_vulkan_PipelineDef_shader_module_ordinal_get(pipeline_def);
  if (shader_module_ordinal >=
      iree_hal_vulkan_ShaderModuleDef_vec_len(shader_modules_vec)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pipelines[%" PRIhsz
                            "] shader module ordinal %u out of range",
                            pipeline_ordinal, shader_module_ordinal);
  }

  flatbuffers_string_t entry_point =
      iree_hal_vulkan_PipelineDef_entry_point_get(pipeline_def);
  if (flatbuffers_string_len(entry_point) == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pipelines[%" PRIhsz "] entry point is empty",
                            pipeline_ordinal);
  }

  const uint32_t pipeline_layout_ordinal =
      iree_hal_vulkan_PipelineDef_pipeline_layout_ordinal_get(pipeline_def);
  if (pipeline_layout_ordinal >=
      iree_hal_vulkan_PipelineLayoutDef_vec_len(pipeline_layouts_vec)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pipelines[%" PRIhsz
                            "] pipeline layout ordinal %u out of range",
                            pipeline_ordinal, pipeline_layout_ordinal);
  }
  iree_hal_vulkan_PipelineLayoutDef_table_t pipeline_layout_def =
      iree_hal_vulkan_PipelineLayoutDef_vec_at(pipeline_layouts_vec,
                                               pipeline_layout_ordinal);

  IREE_RETURN_IF_ERROR(iree_hal_vulkan_verify_required_subgroup_size(
      physical_device, enabled_features,
      iree_hal_vulkan_PipelineDef_subgroup_size_get(pipeline_def),
      pipeline_ordinal));
  iree_hal_vulkan_dispatch_abis_t pipeline_dispatch_abi =
      IREE_HAL_VULKAN_DISPATCH_ABI_NONE;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_pipeline_def_dispatch_abi(
      pipeline_def, &pipeline_dispatch_abi));
  if (pipeline_dispatch_abi != executable_dispatch_abi) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pipelines[%" PRIhsz
                            "] dispatch ABI does not match executable format",
                            pipeline_ordinal);
  }
  if (pipeline_dispatch_abi == IREE_HAL_VULKAN_DISPATCH_ABI_DESCRIPTOR) {
    if (iree_hal_vulkan_PipelineDef_bda_dispatch_layout_get(pipeline_def)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "pipelines[%" PRIhsz
          "] descriptor dispatch ABI must not declare a BDA layout",
          pipeline_ordinal);
    }
  } else if (pipeline_dispatch_abi == IREE_HAL_VULKAN_DISPATCH_ABI_BDA) {
    flatbuffers_uint32_vec_t descriptor_set_layout_ordinals_vec =
        iree_hal_vulkan_PipelineLayoutDef_descriptor_set_layout_ordinals_get(
            pipeline_layout_def);
    if (flatbuffers_uint32_vec_len(descriptor_set_layout_ordinals_vec) != 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "pipelines[%" PRIhsz
          "] BDA dispatch ABI must use a descriptor-free pipeline layout",
          pipeline_ordinal);
    }
    IREE_RETURN_IF_ERROR(iree_hal_vulkan_verify_bda_dispatch_layout_def(
        pipeline_layout_def,
        iree_hal_vulkan_PipelineDef_bda_dispatch_layout_get(pipeline_def),
        pipeline_ordinal));
  }
  return iree_hal_debug_verify_export_def(
      iree_hal_vulkan_PipelineDef_debug_info_get(pipeline_def));
}

static iree_status_t iree_hal_vulkan_executable_flatbuffer_verify(
    const iree_hal_vulkan_physical_device_snapshot_t* physical_device,
    iree_hal_vulkan_features_t enabled_features,
    iree_hal_vulkan_dispatch_abis_t executable_dispatch_abi,
    iree_const_byte_span_t flatbuffer_data) {
  const int verify_result = iree_hal_vulkan_ExecutableDef_verify_as_root(
      flatbuffer_data.data, flatbuffer_data.data_length);
  if (verify_result != flatcc_verify_ok) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "flatbuffer verification failed: %s",
                            flatcc_verify_error_string(verify_result));
  }

  iree_hal_vulkan_ExecutableDef_table_t executable_def =
      iree_hal_vulkan_ExecutableDef_as_root(flatbuffer_data.data);

  iree_hal_vulkan_DescriptorSetLayoutDef_vec_t descriptor_set_layouts_vec =
      iree_hal_vulkan_ExecutableDef_descriptor_set_layouts_get(executable_def);
  const iree_host_size_t descriptor_set_layout_count =
      iree_hal_vulkan_DescriptorSetLayoutDef_vec_len(
          descriptor_set_layouts_vec);
  for (iree_host_size_t i = 0; i < descriptor_set_layout_count; ++i) {
    IREE_RETURN_IF_ERROR(iree_hal_vulkan_verify_descriptor_set_layout_def(
        iree_hal_vulkan_DescriptorSetLayoutDef_vec_at(
            descriptor_set_layouts_vec, i),
        i));
  }

  iree_hal_vulkan_PipelineLayoutDef_vec_t pipeline_layouts_vec =
      iree_hal_vulkan_ExecutableDef_pipeline_layouts_get(executable_def);
  const iree_host_size_t pipeline_layout_count =
      iree_hal_vulkan_PipelineLayoutDef_vec_len(pipeline_layouts_vec);
  for (iree_host_size_t i = 0; i < pipeline_layout_count; ++i) {
    IREE_RETURN_IF_ERROR(iree_hal_vulkan_verify_pipeline_layout_def(
        physical_device, descriptor_set_layouts_vec,
        iree_hal_vulkan_PipelineLayoutDef_vec_at(pipeline_layouts_vec, i), i));
  }

  iree_hal_vulkan_ShaderModuleDef_vec_t shader_modules_vec =
      iree_hal_vulkan_ExecutableDef_shader_modules_get(executable_def);
  const iree_host_size_t shader_module_count =
      iree_hal_vulkan_ShaderModuleDef_vec_len(shader_modules_vec);
  for (iree_host_size_t i = 0; i < shader_module_count; ++i) {
    IREE_RETURN_IF_ERROR(iree_hal_vulkan_verify_shader_module_def(
        iree_hal_vulkan_ShaderModuleDef_vec_at(shader_modules_vec, i), i));
  }

  iree_hal_vulkan_PipelineDef_vec_t pipelines_vec =
      iree_hal_vulkan_ExecutableDef_pipelines_get(executable_def);
  const iree_host_size_t pipeline_count =
      iree_hal_vulkan_PipelineDef_vec_len(pipelines_vec);
  if (pipeline_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "executable declares no pipelines");
  }
  for (iree_host_size_t i = 0; i < pipeline_count; ++i) {
    IREE_RETURN_IF_ERROR(iree_hal_vulkan_verify_pipeline_def(
        physical_device, enabled_features, shader_modules_vec,
        pipeline_layouts_vec, executable_dispatch_abi,
        iree_hal_vulkan_PipelineDef_vec_at(pipelines_vec, i), i));
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Vulkan Object Creation
//===----------------------------------------------------------------------===//

static void iree_hal_vulkan_destroy_shader_modules(
    const iree_hal_vulkan_device_syms_t* syms, VkDevice logical_device,
    iree_host_size_t shader_module_count, VkShaderModule* shader_modules,
    iree_allocator_t host_allocator) {
  for (iree_host_size_t i = 0; i < shader_module_count; ++i) {
    if (shader_modules[i]) {
      iree_vkDestroyShaderModule(IREE_VULKAN_DEVICE(syms), logical_device,
                                 shader_modules[i], /*pAllocator=*/NULL);
    }
  }
  iree_allocator_free(host_allocator, shader_modules);
}

static iree_status_t iree_hal_vulkan_create_shader_modules(
    const iree_hal_vulkan_device_syms_t* syms, VkDevice logical_device,
    iree_hal_vulkan_ShaderModuleDef_vec_t shader_modules_vec,
    iree_allocator_t host_allocator, iree_host_size_t* out_shader_module_count,
    VkShaderModule** out_shader_modules) {
  *out_shader_module_count = 0;
  *out_shader_modules = NULL;

  const iree_host_size_t shader_module_count =
      iree_hal_vulkan_ShaderModuleDef_vec_len(shader_modules_vec);
  VkShaderModule* shader_modules = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      host_allocator, shader_module_count, sizeof(shader_modules[0]),
      (void**)&shader_modules));

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < shader_module_count; ++i) {
    iree_hal_vulkan_ShaderModuleDef_table_t shader_module_def =
        iree_hal_vulkan_ShaderModuleDef_vec_at(shader_modules_vec, i);
    flatbuffers_uint32_vec_t spirv_code_vec =
        iree_hal_vulkan_ShaderModuleDef_spirv_code_get(shader_module_def);
    VkShaderModuleCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize =
            flatbuffers_uint32_vec_len(spirv_code_vec) * sizeof(uint32_t),
        .pCode = (const uint32_t*)spirv_code_vec,
    };
    status = iree_vkCreateShaderModule(IREE_VULKAN_DEVICE(syms), logical_device,
                                       &create_info, /*pAllocator=*/NULL,
                                       &shader_modules[i]);
    if (!iree_status_is_ok(status)) {
      status = iree_status_annotate_f(status, "shader_modules[%" PRIhsz "]", i);
    }
  }

  if (iree_status_is_ok(status)) {
    *out_shader_module_count = shader_module_count;
    *out_shader_modules = shader_modules;
  } else {
    iree_hal_vulkan_destroy_shader_modules(syms, logical_device,
                                           shader_module_count, shader_modules,
                                           host_allocator);
  }
  return status;
}

static iree_status_t iree_hal_vulkan_create_descriptor_set_layouts(
    const iree_hal_vulkan_device_syms_t* syms, VkDevice logical_device,
    iree_hal_vulkan_DescriptorSetLayoutDef_vec_t descriptor_set_layouts_vec,
    iree_host_size_t descriptor_set_layout_count,
    VkDescriptorSetLayout* descriptor_set_layouts,
    iree_allocator_t host_allocator) {
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < descriptor_set_layout_count; ++i) {
    iree_hal_vulkan_DescriptorSetLayoutDef_table_t descriptor_set_layout_def =
        iree_hal_vulkan_DescriptorSetLayoutDef_vec_at(
            descriptor_set_layouts_vec, i);
    iree_hal_vulkan_DescriptorSetLayoutBindingDef_vec_t bindings_vec =
        iree_hal_vulkan_DescriptorSetLayoutDef_bindings_get(
            descriptor_set_layout_def);
    const iree_host_size_t binding_count =
        iree_hal_vulkan_DescriptorSetLayoutBindingDef_vec_len(bindings_vec);

    VkDescriptorSetLayoutBinding* bindings = NULL;
    if (binding_count > 0) {
      status =
          iree_allocator_malloc_array(host_allocator, binding_count,
                                      sizeof(bindings[0]), (void**)&bindings);
    }
    if (iree_status_is_ok(status)) {
      for (iree_host_size_t j = 0; j < binding_count; ++j) {
        iree_hal_vulkan_DescriptorSetLayoutBindingDef_table_t binding_def =
            iree_hal_vulkan_DescriptorSetLayoutBindingDef_vec_at(bindings_vec,
                                                                 j);
        bindings[j] = (VkDescriptorSetLayoutBinding){
            .binding =
                iree_hal_vulkan_DescriptorSetLayoutBindingDef_binding_get(
                    binding_def),
            .descriptorType = (VkDescriptorType)
                iree_hal_vulkan_DescriptorSetLayoutBindingDef_descriptor_type_get(
                    binding_def),
            .descriptorCount =
                iree_hal_vulkan_DescriptorSetLayoutBindingDef_descriptor_count_get(
                    binding_def),
            .stageFlags =
                iree_hal_vulkan_DescriptorSetLayoutBindingDef_stage_flags_get(
                    binding_def),
        };
      }
      VkDescriptorSetLayoutCreateInfo create_info = {
          .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
          .bindingCount = (uint32_t)binding_count,
          .pBindings = bindings,
      };
      status = iree_vkCreateDescriptorSetLayout(
          IREE_VULKAN_DEVICE(syms), logical_device, &create_info,
          /*pAllocator=*/NULL, &descriptor_set_layouts[i]);
    }

    iree_allocator_free(host_allocator, bindings);
    if (!iree_status_is_ok(status)) {
      status = iree_status_annotate_f(status,
                                      "descriptor_set_layouts[%" PRIhsz "]", i);
    }
  }
  return status;
}

static iree_status_t iree_hal_vulkan_create_pipeline_layouts(
    const iree_hal_vulkan_device_syms_t* syms, VkDevice logical_device,
    iree_hal_vulkan_PipelineLayoutDef_vec_t pipeline_layouts_vec,
    const VkDescriptorSetLayout* descriptor_set_layouts,
    iree_host_size_t pipeline_layout_count, VkPipelineLayout* pipeline_layouts,
    iree_allocator_t host_allocator) {
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < pipeline_layout_count; ++i) {
    iree_hal_vulkan_PipelineLayoutDef_table_t pipeline_layout_def =
        iree_hal_vulkan_PipelineLayoutDef_vec_at(pipeline_layouts_vec, i);

    flatbuffers_uint32_vec_t descriptor_set_layout_ordinals_vec =
        iree_hal_vulkan_PipelineLayoutDef_descriptor_set_layout_ordinals_get(
            pipeline_layout_def);
    const iree_host_size_t descriptor_set_layout_ordinal_count =
        flatbuffers_uint32_vec_len(descriptor_set_layout_ordinals_vec);
    VkDescriptorSetLayout* selected_descriptor_set_layouts = NULL;
    if (descriptor_set_layout_ordinal_count > 0) {
      status = iree_allocator_malloc_array(
          host_allocator, descriptor_set_layout_ordinal_count,
          sizeof(selected_descriptor_set_layouts[0]),
          (void**)&selected_descriptor_set_layouts);
    }
    if (iree_status_is_ok(status)) {
      for (iree_host_size_t j = 0; j < descriptor_set_layout_ordinal_count;
           ++j) {
        const uint32_t descriptor_set_layout_ordinal =
            flatbuffers_uint32_vec_at(descriptor_set_layout_ordinals_vec, j);
        selected_descriptor_set_layouts[j] =
            descriptor_set_layouts[descriptor_set_layout_ordinal];
      }

      iree_hal_vulkan_PushConstantRange_vec_t push_constant_ranges_vec =
          iree_hal_vulkan_PipelineLayoutDef_push_constant_ranges_get(
              pipeline_layout_def);
      const iree_host_size_t push_constant_range_count =
          iree_hal_vulkan_PushConstantRange_vec_len(push_constant_ranges_vec);
      static_assert(sizeof(iree_hal_vulkan_PushConstantRange_t) ==
                        sizeof(VkPushConstantRange),
                    "VKE1 PushConstantRange must match VkPushConstantRange");
      const VkPushConstantRange* push_constant_ranges =
          push_constant_range_count
              ? (const VkPushConstantRange*)
                    iree_hal_vulkan_PushConstantRange_vec_at(
                        push_constant_ranges_vec, 0)
              : NULL;
      VkPipelineLayoutCreateInfo create_info = {
          .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
          .setLayoutCount = (uint32_t)descriptor_set_layout_ordinal_count,
          .pSetLayouts = selected_descriptor_set_layouts,
          .pushConstantRangeCount = (uint32_t)push_constant_range_count,
          .pPushConstantRanges = push_constant_ranges,
      };
      status = iree_vkCreatePipelineLayout(
          IREE_VULKAN_DEVICE(syms), logical_device, &create_info,
          /*pAllocator=*/NULL, &pipeline_layouts[i]);
    }

    iree_allocator_free(host_allocator, selected_descriptor_set_layouts);
    if (!iree_status_is_ok(status)) {
      status =
          iree_status_annotate_f(status, "pipeline_layouts[%" PRIhsz "]", i);
    }
  }
  return status;
}

static iree_status_t iree_hal_vulkan_create_specialization_info(
    const iree_hal_executable_params_t* executable_params,
    iree_allocator_t host_allocator, VkSpecializationInfo* out_info,
    VkSpecializationMapEntry** out_map_entries) {
  memset(out_info, 0, sizeof(*out_info));
  *out_map_entries = NULL;
  if (executable_params->constant_count == 0) return iree_ok_status();
  if (!executable_params->constants) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "executable declares %" PRIhsz
                            " specialization constants but no value storage",
                            executable_params->constant_count);
  }
  if (executable_params->constant_count > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "executable declares %" PRIhsz
                            " specialization constants, exceeding Vulkan "
                            "limit %u",
                            executable_params->constant_count, UINT32_MAX);
  }
  if (executable_params->constant_count >
      IREE_HOST_SIZE_MAX / sizeof(uint32_t)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "executable specialization constant data size "
                            "overflows");
  }
  const iree_host_size_t max_constant_offset_count =
      (iree_host_size_t)UINT32_MAX / sizeof(uint32_t) + 1;
  if (executable_params->constant_count > max_constant_offset_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "executable declares %" PRIhsz
                            " specialization constants, exceeding Vulkan "
                            "constant offset limit %" PRIhsz,
                            executable_params->constant_count,
                            max_constant_offset_count);
  }

  VkSpecializationMapEntry* map_entries = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      host_allocator, executable_params->constant_count, sizeof(map_entries[0]),
      (void**)&map_entries));
  for (iree_host_size_t i = 0; i < executable_params->constant_count; ++i) {
    map_entries[i] = (VkSpecializationMapEntry){
        .constantID = (uint32_t)i,
        .offset = (uint32_t)(i * sizeof(uint32_t)),
        .size = sizeof(uint32_t),
    };
  }

  *out_info = (VkSpecializationInfo){
      .mapEntryCount = (uint32_t)executable_params->constant_count,
      .pMapEntries = map_entries,
      .dataSize = executable_params->constant_count * sizeof(uint32_t),
      .pData = executable_params->constants,
  };
  *out_map_entries = map_entries;
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_initialize_pipeline_descriptor_metadata(
    iree_hal_vulkan_PipelineLayoutDef_table_t pipeline_layout_def,
    iree_hal_vulkan_DescriptorSetLayoutDef_vec_t descriptor_set_layouts_vec,
    const VkDescriptorSetLayout* descriptor_set_layouts,
    iree_allocator_t host_allocator, iree_hal_vulkan_pipeline_t* out_pipeline) {
  flatbuffers_uint32_vec_t descriptor_set_layout_ordinals_vec =
      iree_hal_vulkan_PipelineLayoutDef_descriptor_set_layout_ordinals_get(
          pipeline_layout_def);
  const iree_host_size_t descriptor_set_layout_count =
      flatbuffers_uint32_vec_len(descriptor_set_layout_ordinals_vec);
  if (descriptor_set_layout_count > UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "pipeline layout declares %" PRIhsz
        " descriptor set layouts, exceeding Vulkan limit %u",
        descriptor_set_layout_count, UINT32_MAX);
  }

  out_pipeline->descriptor_set_layout_count = descriptor_set_layout_count;
  if (descriptor_set_layout_count > 0) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
        host_allocator, descriptor_set_layout_count,
        sizeof(out_pipeline->descriptor_set_layouts[0]),
        (void**)&out_pipeline->descriptor_set_layouts));
  }

  uint64_t descriptor_binding_count = 0;
  for (iree_host_size_t i = 0; i < descriptor_set_layout_count; ++i) {
    const uint32_t descriptor_set_layout_ordinal =
        flatbuffers_uint32_vec_at(descriptor_set_layout_ordinals_vec, i);
    out_pipeline->descriptor_set_layouts[i] =
        descriptor_set_layouts[descriptor_set_layout_ordinal];

    iree_hal_vulkan_DescriptorSetLayoutDef_table_t descriptor_set_layout_def =
        iree_hal_vulkan_DescriptorSetLayoutDef_vec_at(
            descriptor_set_layouts_vec, descriptor_set_layout_ordinal);
    IREE_RETURN_IF_ERROR(iree_hal_vulkan_count_descriptor_set_layout_def(
        descriptor_set_layout_def, /*inout_sampler_count=*/NULL,
        /*inout_uniform_buffer_count=*/NULL,
        /*inout_storage_buffer_count=*/NULL, &descriptor_binding_count));
  }
  if (descriptor_binding_count > IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "pipeline layout declares %" PRIu64
                            " descriptor bindings, exceeding host size limit",
                            descriptor_binding_count);
  }

  out_pipeline->descriptor_binding_count =
      (iree_host_size_t)descriptor_binding_count;
  if (descriptor_binding_count > 0) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
        host_allocator, (iree_host_size_t)descriptor_binding_count,
        sizeof(out_pipeline->descriptor_bindings[0]),
        (void**)&out_pipeline->descriptor_bindings));
  }

  iree_host_size_t binding_index = 0;
  for (iree_host_size_t i = 0; i < descriptor_set_layout_count; ++i) {
    const uint32_t descriptor_set_layout_ordinal =
        flatbuffers_uint32_vec_at(descriptor_set_layout_ordinals_vec, i);
    iree_hal_vulkan_DescriptorSetLayoutDef_table_t descriptor_set_layout_def =
        iree_hal_vulkan_DescriptorSetLayoutDef_vec_at(
            descriptor_set_layouts_vec, descriptor_set_layout_ordinal);
    iree_hal_vulkan_DescriptorSetLayoutBindingDef_vec_t bindings_vec =
        iree_hal_vulkan_DescriptorSetLayoutDef_bindings_get(
            descriptor_set_layout_def);
    const iree_host_size_t set_binding_count =
        iree_hal_vulkan_DescriptorSetLayoutBindingDef_vec_len(bindings_vec);
    for (iree_host_size_t j = 0; j < set_binding_count; ++j) {
      iree_hal_vulkan_DescriptorSetLayoutBindingDef_table_t binding_def =
          iree_hal_vulkan_DescriptorSetLayoutBindingDef_vec_at(bindings_vec, j);
      const uint32_t descriptor_count =
          iree_hal_vulkan_DescriptorSetLayoutBindingDef_descriptor_count_get(
              binding_def);
      for (uint32_t array_element = 0; array_element < descriptor_count;
           ++array_element) {
        out_pipeline->descriptor_bindings
            [binding_index++] = (iree_hal_vulkan_descriptor_binding_t){
            .set_ordinal = (uint32_t)i,
            .binding =
                iree_hal_vulkan_DescriptorSetLayoutBindingDef_binding_get(
                    binding_def),
            .array_element = array_element,
            .descriptor_type = (VkDescriptorType)
                iree_hal_vulkan_DescriptorSetLayoutBindingDef_descriptor_type_get(
                    binding_def),
        };
      }
    }
  }

  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_initialize_pipeline_bda_metadata(
    iree_hal_vulkan_BdaDispatchLayoutDef_table_t bda_dispatch_layout_def,
    iree_allocator_t host_allocator, iree_hal_vulkan_pipeline_t* out_pipeline) {
  const uint32_t constant_count =
      iree_hal_vulkan_BdaDispatchLayoutDef_constant_count_get(
          bda_dispatch_layout_def);
  const uint32_t binding_count =
      iree_hal_vulkan_BdaDispatchLayoutDef_binding_count_get(
          bda_dispatch_layout_def);
  out_pipeline->constant_count = (uint16_t)constant_count;
  out_pipeline->binding_count = (uint16_t)binding_count;
  out_pipeline->bda.root_push_constant_offset =
      iree_hal_vulkan_BdaDispatchLayoutDef_root_push_constant_offset_get(
          bda_dispatch_layout_def);
  out_pipeline->bda.root_push_constant_length =
      iree_hal_vulkan_BdaDispatchLayoutDef_root_push_constant_length_get(
          bda_dispatch_layout_def);
  out_pipeline->bda.constant_push_constant_offset =
      iree_hal_vulkan_BdaDispatchLayoutDef_constant_push_constant_offset_get(
          bda_dispatch_layout_def);
  out_pipeline->bda.binding_count_known = true;

  switch (iree_hal_vulkan_BdaDispatchLayoutDef_binding_table_entry_type_get(
      bda_dispatch_layout_def)) {
    case iree_hal_vulkan_BdaBindingTableEntryType_ADDRESS64:
      out_pipeline->bda.binding_table_entry_length = sizeof(uint64_t);
      break;
    default:
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "BDA pipeline metadata has unsupported binding table entry type");
  }

  iree_hal_vulkan_BdaBindingDef_vec_t bindings_vec =
      iree_hal_vulkan_BdaDispatchLayoutDef_bindings_get(
          bda_dispatch_layout_def);
  const iree_host_size_t binding_requirement_count =
      iree_hal_vulkan_BdaBindingDef_vec_len(bindings_vec);
  bool has_binding_requirements = false;
  for (iree_host_size_t i = 0; i < binding_requirement_count; ++i) {
    iree_hal_vulkan_BdaBindingDef_table_t binding_def =
        iree_hal_vulkan_BdaBindingDef_vec_at(bindings_vec, i);
    if (iree_hal_vulkan_BdaBindingDef_minimum_alignment_get(binding_def) != 1 ||
        iree_hal_vulkan_BdaBindingDef_minimum_length_get(binding_def) != 0) {
      has_binding_requirements = true;
      break;
    }
  }
  if (!has_binding_requirements) return iree_ok_status();

  out_pipeline->bda.binding_requirement_count = binding_requirement_count;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      host_allocator, binding_requirement_count,
      sizeof(out_pipeline->bda.binding_requirements[0]),
      (void**)&out_pipeline->bda.binding_requirements));
  for (iree_host_size_t i = 0; i < binding_requirement_count; ++i) {
    iree_hal_vulkan_BdaBindingDef_table_t binding_def =
        iree_hal_vulkan_BdaBindingDef_vec_at(bindings_vec, i);
    out_pipeline->bda.binding_requirements[i] =
        (iree_hal_vulkan_bda_binding_requirement_t){
            .minimum_alignment =
                iree_hal_vulkan_BdaBindingDef_minimum_alignment_get(
                    binding_def),
            .minimum_length =
                iree_hal_vulkan_BdaBindingDef_minimum_length_get(binding_def),
        };
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_parse_flatbuffer_spirv_workgroup_size(
    flatbuffers_uint32_vec_t spirv_code_vec, iree_string_view_t entry_point,
    uint32_t out_workgroup_size[3]) {
  return iree_hal_vulkan_spirv_parse_compute_workgroup_size(
      (const uint32_t*)spirv_code_vec,
      flatbuffers_uint32_vec_len(spirv_code_vec), entry_point,
      /*out_entry_point_found=*/NULL, out_workgroup_size);
}

typedef enum iree_hal_vulkan_bda_spirv_verification_flag_bits_e {
  IREE_HAL_VULKAN_BDA_SPIRV_VERIFICATION_FLAG_NONE = 0u,
  IREE_HAL_VULKAN_BDA_SPIRV_VERIFICATION_FLAG_REQUIRE_PUSH_CONSTANT_ROOT = 0x1u,
} iree_hal_vulkan_bda_spirv_verification_flag_bits_t;
typedef uint32_t iree_hal_vulkan_bda_spirv_verification_flags_t;

static iree_status_t iree_hal_vulkan_verify_bda_spirv_module(
    const uint32_t* spirv_words, iree_host_size_t spirv_word_count,
    iree_hal_vulkan_bda_spirv_verification_flags_t verification_flags) {
  iree_hal_vulkan_spirv_module_analysis_t analysis;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_spirv_analyze_module(
      spirv_words, spirv_word_count, &analysis));
  if (!analysis.has_physical_storage_buffer_addresses_capability) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Vulkan BDA executable must declare PhysicalStorageBufferAddresses");
  }

  if (!analysis.uses_physical_storage_buffer64_glsl450) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Vulkan BDA executable must use PhysicalStorageBuffer64 GLSL450");
  }
  if (analysis.has_descriptor_binding_decorations) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Vulkan BDA executable must not declare descriptor set or binding "
        "decorations");
  }
  if (analysis.has_descriptor_storage_class_variables) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Vulkan BDA executable must not declare descriptor-backed variables");
  }

  if (iree_all_bits_set(
          verification_flags,
          IREE_HAL_VULKAN_BDA_SPIRV_VERIFICATION_FLAG_REQUIRE_PUSH_CONSTANT_ROOT)) {
    IREE_RETURN_IF_ERROR(
        iree_hal_vulkan_spirv_verify_bda_root_push_constant_layout(
            spirv_words, spirv_word_count));
  }

  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_verify_bda_spirv(
    const uint32_t* spirv_words, iree_host_size_t spirv_word_count,
    iree_string_view_t entry_point,
    iree_hal_vulkan_bda_spirv_verification_flags_t verification_flags,
    uint32_t out_workgroup_size[3]) {
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_verify_bda_spirv_module(
      spirv_words, spirv_word_count, verification_flags));

  bool entry_point_found = false;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_spirv_parse_compute_workgroup_size(
      spirv_words, spirv_word_count, entry_point, &entry_point_found,
      out_workgroup_size));
  if (!entry_point_found) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Vulkan BDA executable has no compute entry point '%.*s'",
        (int)entry_point.size, entry_point.data);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_create_compute_pipeline(
    const iree_hal_vulkan_device_syms_t* syms, VkDevice logical_device,
    VkPipelineCache pipeline_cache,
    const iree_hal_executable_params_t* executable_params,
    const VkSpecializationInfo* specialization_info,
    iree_hal_vulkan_PipelineLayoutDef_vec_t pipeline_layouts_vec,
    iree_hal_vulkan_DescriptorSetLayoutDef_vec_t descriptor_set_layouts_vec,
    iree_hal_vulkan_ShaderModuleDef_vec_t shader_modules_vec,
    const VkPipelineLayout* pipeline_layouts,
    const VkDescriptorSetLayout* descriptor_set_layouts,
    const VkShaderModule* shader_modules, iree_allocator_t host_allocator,
    iree_hal_vulkan_dispatch_abis_t dispatch_abi,
    iree_hal_vulkan_PipelineDef_table_t pipeline_def,
    iree_hal_vulkan_pipeline_t* out_pipeline) {
  const uint32_t pipeline_layout_ordinal =
      iree_hal_vulkan_PipelineDef_pipeline_layout_ordinal_get(pipeline_def);
  iree_hal_vulkan_PipelineLayoutDef_table_t pipeline_layout_def =
      iree_hal_vulkan_PipelineLayoutDef_vec_at(pipeline_layouts_vec,
                                               pipeline_layout_ordinal);

  const uint32_t shader_module_ordinal =
      iree_hal_vulkan_PipelineDef_shader_module_ordinal_get(pipeline_def);
  const uint32_t subgroup_size =
      iree_hal_vulkan_PipelineDef_subgroup_size_get(pipeline_def);
  iree_hal_vulkan_ShaderModuleDef_table_t shader_module_def =
      iree_hal_vulkan_ShaderModuleDef_vec_at(shader_modules_vec,
                                             shader_module_ordinal);
  flatbuffers_uint32_vec_t spirv_code_vec =
      iree_hal_vulkan_ShaderModuleDef_spirv_code_get(shader_module_def);
  flatbuffers_string_t entry_point =
      iree_hal_vulkan_PipelineDef_entry_point_get(pipeline_def);
  iree_string_view_t entry_point_view =
      iree_make_string_view(entry_point, flatbuffers_string_len(entry_point));
  out_pipeline->layout = pipeline_layouts[pipeline_layout_ordinal];
  out_pipeline->dispatch_abi = dispatch_abi;
  out_pipeline->subgroup_size = subgroup_size;
  switch (dispatch_abi) {
    case IREE_HAL_VULKAN_DISPATCH_ABI_DESCRIPTOR: {
      IREE_RETURN_IF_ERROR(iree_hal_vulkan_calculate_pipeline_layout_counts(
          descriptor_set_layouts_vec, pipeline_layout_def,
          &out_pipeline->constant_count, &out_pipeline->binding_count));
      IREE_RETURN_IF_ERROR(
          iree_hal_vulkan_initialize_pipeline_descriptor_metadata(
              pipeline_layout_def, descriptor_set_layouts_vec,
              descriptor_set_layouts, host_allocator, out_pipeline));
      break;
    }
    case IREE_HAL_VULKAN_DISPATCH_ABI_BDA: {
      IREE_RETURN_IF_ERROR(iree_hal_vulkan_initialize_pipeline_bda_metadata(
          iree_hal_vulkan_PipelineDef_bda_dispatch_layout_get(pipeline_def),
          host_allocator, out_pipeline));
      IREE_RETURN_IF_ERROR(iree_hal_vulkan_verify_bda_spirv(
          (const uint32_t*)spirv_code_vec,
          flatbuffers_uint32_vec_len(spirv_code_vec), entry_point_view,
          IREE_HAL_VULKAN_BDA_SPIRV_VERIFICATION_FLAG_NONE,
          out_pipeline->workgroup_size));
      break;
    }
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Vulkan pipeline has invalid dispatch ABI 0x%08x",
                              dispatch_abi);
  }

  VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subgroup_size_info = {
      .sType =
          VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO,
      .requiredSubgroupSize = subgroup_size,
  };
  if (dispatch_abi == IREE_HAL_VULKAN_DISPATCH_ABI_DESCRIPTOR) {
    IREE_RETURN_IF_ERROR(iree_hal_vulkan_parse_flatbuffer_spirv_workgroup_size(
        spirv_code_vec, entry_point_view, out_pipeline->workgroup_size));
  }
  VkPipelineShaderStageCreateInfo stage_create_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .pNext = subgroup_size ? &subgroup_size_info : NULL,
      .stage = VK_SHADER_STAGE_COMPUTE_BIT,
      .module = shader_modules[shader_module_ordinal],
      .pName = entry_point,
      .pSpecializationInfo = specialization_info,
  };
  VkComputePipelineCreateInfo create_info = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = stage_create_info,
      .layout = out_pipeline->layout,
  };
  if (!iree_all_bits_set(executable_params->caching_mode,
                         IREE_HAL_EXECUTABLE_CACHING_MODE_ALLOW_OPTIMIZATION)) {
    create_info.flags |= VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT;
  }
  return iree_vkCreateComputePipelines(
      IREE_VULKAN_DEVICE(syms), logical_device, pipeline_cache,
      /*createInfoCount=*/1, &create_info, /*pAllocator=*/NULL,
      &out_pipeline->handle);
}

//===----------------------------------------------------------------------===//
// iree_hal_vulkan_executable_t
//===----------------------------------------------------------------------===//

static iree_atomic_int64_t iree_hal_vulkan_executable_next_profile_id =
    IREE_ATOMIC_VAR_INIT(1);

typedef struct iree_hal_vulkan_executable_t {
  // HAL resource header.
  iree_hal_resource_t resource;

  // Host allocator used for executable lifetime.
  iree_allocator_t host_allocator;

  // Borrowed logical-device dispatch table.
  const iree_hal_vulkan_device_syms_t* syms;

  // Borrowed logical-device handle.
  VkDevice logical_device;

  // Process-local nonzero executable identifier used by profiling sessions.
  uint64_t profile_id;

  // Number of descriptor set layouts in |descriptor_set_layouts|.
  iree_host_size_t descriptor_set_layout_count;

  // Vulkan descriptor set layout handles owned by the executable.
  VkDescriptorSetLayout* descriptor_set_layouts;

  // Number of pipeline layouts in |pipeline_layouts|.
  iree_host_size_t pipeline_layout_count;

  // Vulkan pipeline layout handles owned by the executable.
  VkPipelineLayout* pipeline_layouts;

  // Number of prepared compute pipelines in |pipelines|.
  iree_host_size_t pipeline_count;

  // Prepared compute pipelines and export metadata.
  iree_hal_vulkan_pipeline_t* pipelines;

  // Storage backing all exported name string views.
  char* name_storage;
} iree_hal_vulkan_executable_t;

static const iree_hal_executable_vtable_t iree_hal_vulkan_executable_vtable;

static iree_hal_vulkan_executable_t* iree_hal_vulkan_executable_cast(
    iree_hal_executable_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_vulkan_executable_vtable);
  return (iree_hal_vulkan_executable_t*)base_value;
}

bool iree_hal_vulkan_executable_isa(iree_hal_executable_t* executable) {
  return iree_hal_resource_is((const iree_hal_resource_t*)executable,
                              &iree_hal_vulkan_executable_vtable);
}

uint64_t iree_hal_vulkan_executable_profile_id(
    iree_hal_executable_t* executable) {
  iree_hal_vulkan_executable_t* vulkan_executable =
      iree_hal_vulkan_executable_cast(executable);
  return vulkan_executable->profile_id;
}

static iree_status_t iree_hal_vulkan_allocate_executable(
    const iree_hal_vulkan_device_syms_t* syms, VkDevice logical_device,
    iree_host_size_t descriptor_set_layout_count,
    iree_host_size_t pipeline_layout_count, iree_host_size_t pipeline_count,
    iree_host_size_t name_storage_size, iree_allocator_t host_allocator,
    iree_hal_vulkan_executable_t** out_executable) {
  *out_executable = NULL;

  iree_hal_vulkan_executable_t* executable = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, sizeof(*executable), (void**)&executable));
  memset(executable, 0, sizeof(*executable));
  iree_hal_resource_initialize(&iree_hal_vulkan_executable_vtable,
                               &executable->resource);
  executable->host_allocator = host_allocator;
  executable->syms = syms;
  executable->logical_device = logical_device;
  executable->profile_id = (uint64_t)iree_atomic_fetch_add(
      &iree_hal_vulkan_executable_next_profile_id, 1,
      iree_memory_order_relaxed);
  executable->descriptor_set_layout_count = descriptor_set_layout_count;
  executable->pipeline_layout_count = pipeline_layout_count;
  executable->pipeline_count = pipeline_count;

  iree_status_t status = iree_ok_status();
  if (iree_status_is_ok(status) && descriptor_set_layout_count > 0) {
    status = iree_allocator_malloc_array(
        host_allocator, descriptor_set_layout_count,
        sizeof(executable->descriptor_set_layouts[0]),
        (void**)&executable->descriptor_set_layouts);
    if (iree_status_is_ok(status)) {
      memset(executable->descriptor_set_layouts, 0,
             descriptor_set_layout_count *
                 sizeof(executable->descriptor_set_layouts[0]));
    }
  }
  if (iree_status_is_ok(status) && pipeline_layout_count > 0) {
    status =
        iree_allocator_malloc_array(host_allocator, pipeline_layout_count,
                                    sizeof(executable->pipeline_layouts[0]),
                                    (void**)&executable->pipeline_layouts);
    if (iree_status_is_ok(status)) {
      memset(executable->pipeline_layouts, 0,
             pipeline_layout_count * sizeof(executable->pipeline_layouts[0]));
    }
  }
  if (iree_status_is_ok(status) && pipeline_count > 0) {
    status = iree_allocator_malloc_array(host_allocator, pipeline_count,
                                         sizeof(executable->pipelines[0]),
                                         (void**)&executable->pipelines);
    if (iree_status_is_ok(status)) {
      memset(executable->pipelines, 0,
             pipeline_count * sizeof(executable->pipelines[0]));
    }
  }
  if (iree_status_is_ok(status) && name_storage_size > 0) {
    status = iree_allocator_malloc(host_allocator, name_storage_size,
                                   (void**)&executable->name_storage);
  }

  if (iree_status_is_ok(status)) {
    *out_executable = executable;
  } else {
    iree_hal_executable_destroy((iree_hal_executable_t*)executable);
  }
  return status;
}

static iree_status_t iree_hal_vulkan_create_raw_bda_executable(
    const iree_hal_vulkan_device_syms_t* syms, VkDevice logical_device,
    VkPipelineCache pipeline_cache,
    const iree_hal_executable_params_t* executable_params,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable) {
  *out_executable = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  if (executable_params->executable_data.data_length % sizeof(uint32_t) != 0) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                             "raw Vulkan BDA executable byte length must be a "
                             "multiple of 4"));
  }

  const iree_host_size_t spirv_word_count =
      executable_params->executable_data.data_length / sizeof(uint32_t);
  const uint32_t* spirv_words =
      (const uint32_t*)executable_params->executable_data.data;
  uint32_t* aligned_spirv_words = NULL;
  if (executable_params->executable_data.data_length != 0 &&
      !iree_host_ptr_has_alignment(spirv_words, sizeof(uint32_t))) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_allocator_malloc(
                host_allocator, executable_params->executable_data.data_length,
                (void**)&aligned_spirv_words));
    memcpy(aligned_spirv_words, executable_params->executable_data.data,
           executable_params->executable_data.data_length);
    spirv_words = aligned_spirv_words;
  }

  iree_host_size_t entry_point_count = 0;
  iree_host_size_t name_storage_size = 0;
  iree_status_t status = iree_hal_vulkan_spirv_count_compute_entry_points(
      spirv_words, spirv_word_count, &entry_point_count, &name_storage_size);
  if (iree_status_is_ok(status) && entry_point_count == 0) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "raw Vulkan BDA executable has no compute entry "
                              "points");
  }

  iree_hal_vulkan_spirv_compute_entry_point_t* entry_points = NULL;
  if (iree_status_is_ok(status)) {
    iree_host_size_t entry_point_table_size = 0;
    if (!iree_host_size_checked_mul(entry_point_count, sizeof(entry_points[0]),
                                    &entry_point_table_size)) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "raw Vulkan BDA entry point table overflows");
    } else {
      status = iree_allocator_malloc(host_allocator, entry_point_table_size,
                                     (void**)&entry_points);
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_spirv_parse_compute_entry_points(
        spirv_words, spirv_word_count, entry_point_count, entry_points);
  }
  iree_hal_vulkan_bda_spirv_verification_flags_t verification_flags =
      IREE_HAL_VULKAN_BDA_SPIRV_VERIFICATION_FLAG_REQUIRE_PUSH_CONSTANT_ROOT;
  if (iree_all_bits_set(
          executable_params->caching_mode,
          IREE_HAL_EXECUTABLE_CACHING_MODE_DISABLE_VERIFICATION)) {
    verification_flags = IREE_HAL_VULKAN_BDA_SPIRV_VERIFICATION_FLAG_NONE;
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_verify_bda_spirv_module(
        spirv_words, spirv_word_count, verification_flags);
  }

  iree_hal_vulkan_executable_t* executable = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_allocate_executable(
        syms, logical_device, /*descriptor_set_layout_count=*/0,
        /*pipeline_layout_count=*/1, entry_point_count, name_storage_size,
        host_allocator, &executable);
  }

  VkShaderModule shader_module = VK_NULL_HANDLE;
  if (iree_status_is_ok(status)) {
    VkShaderModuleCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = executable_params->executable_data.data_length,
        .pCode = spirv_words,
    };
    status = iree_vkCreateShaderModule(IREE_VULKAN_DEVICE(syms), logical_device,
                                       &create_info,
                                       /*pAllocator=*/NULL, &shader_module);
  }

  if (iree_status_is_ok(status)) {
    VkPushConstantRange root_range = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = sizeof(iree_hal_vulkan_bda_dispatch_root_v1_t),
    };
    VkPipelineLayoutCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &root_range,
    };
    status = iree_vkCreatePipelineLayout(
        IREE_VULKAN_DEVICE(syms), logical_device, &create_info,
        /*pAllocator=*/NULL, &executable->pipeline_layouts[0]);
  }

  VkSpecializationInfo specialization_info;
  VkSpecializationMapEntry* specialization_map_entries = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_create_specialization_info(
        executable_params, host_allocator, &specialization_info,
        &specialization_map_entries);
  }

  char* name_storage = executable ? executable->name_storage : NULL;
  iree_host_size_t name_storage_offset = 0;
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < entry_point_count; ++i) {
    iree_hal_vulkan_pipeline_t* pipeline = &executable->pipelines[i];
    const iree_string_view_t entry_point = entry_points[i].name;
    memcpy(name_storage + name_storage_offset, entry_point.data,
           entry_point.size);
    name_storage[name_storage_offset + entry_point.size] = 0;
    pipeline->name = iree_make_string_view(name_storage + name_storage_offset,
                                           entry_point.size);
    name_storage_offset += entry_point.size + /*NUL=*/1;

    pipeline->handle = VK_NULL_HANDLE;
    pipeline->dispatch_abi = IREE_HAL_VULKAN_DISPATCH_ABI_BDA;
    pipeline->layout = executable->pipeline_layouts[0];
    pipeline->bda.root_push_constant_offset = 0;
    pipeline->bda.root_push_constant_length =
        sizeof(iree_hal_vulkan_bda_dispatch_root_v1_t);
    pipeline->bda.constant_push_constant_offset =
        sizeof(iree_hal_vulkan_bda_dispatch_root_v1_t);
    pipeline->bda.binding_table_entry_length = sizeof(uint64_t);
    pipeline->bda.binding_count_known = false;
    memcpy(pipeline->workgroup_size, entry_points[i].workgroup_size,
           sizeof(pipeline->workgroup_size));

    VkPipelineShaderStageCreateInfo stage_create_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = shader_module,
        .pName = pipeline->name.data,
        .pSpecializationInfo = &specialization_info,
    };
    VkComputePipelineCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = stage_create_info,
        .layout = pipeline->layout,
    };
    if (!iree_all_bits_set(
            executable_params->caching_mode,
            IREE_HAL_EXECUTABLE_CACHING_MODE_ALLOW_OPTIMIZATION)) {
      create_info.flags |= VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT;
    }
    status = iree_vkCreateComputePipelines(
        IREE_VULKAN_DEVICE(syms), logical_device, pipeline_cache,
        /*createInfoCount=*/1, &create_info, /*pAllocator=*/NULL,
        &pipeline->handle);
    if (!iree_status_is_ok(status)) {
      status =
          iree_status_annotate_f(status, "raw BDA entry point '%.*s'",
                                 (int)pipeline->name.size, pipeline->name.data);
    }
  }

  iree_allocator_free(host_allocator, specialization_map_entries);
  if (shader_module) {
    iree_vkDestroyShaderModule(IREE_VULKAN_DEVICE(syms), logical_device,
                               shader_module, /*pAllocator=*/NULL);
  }
  iree_allocator_free(host_allocator, entry_points);
  iree_allocator_free(host_allocator, aligned_spirv_words);

  if (iree_status_is_ok(status)) {
    *out_executable = (iree_hal_executable_t*)executable;
  } else if (executable) {
    iree_hal_executable_destroy((iree_hal_executable_t*)executable);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_vulkan_calculate_name_storage_size(
    iree_hal_vulkan_PipelineDef_vec_t pipelines_vec,
    iree_host_size_t* out_name_storage_size) {
  *out_name_storage_size = 0;
  const iree_host_size_t pipeline_count =
      iree_hal_vulkan_PipelineDef_vec_len(pipelines_vec);
  for (iree_host_size_t i = 0; i < pipeline_count; ++i) {
    iree_hal_vulkan_PipelineDef_table_t pipeline_def =
        iree_hal_vulkan_PipelineDef_vec_at(pipelines_vec, i);
    flatbuffers_string_t entry_point =
        iree_hal_vulkan_PipelineDef_entry_point_get(pipeline_def);
    iree_host_size_t entry_point_size = flatbuffers_string_len(entry_point);
    if (!iree_host_size_checked_add(entry_point_size, /*NUL=*/1,
                                    &entry_point_size) ||
        !iree_host_size_checked_add(*out_name_storage_size, entry_point_size,
                                    out_name_storage_size)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "executable export name storage overflows");
    }
  }
  return iree_ok_status();
}

iree_status_t iree_hal_vulkan_executable_create(
    const iree_hal_vulkan_device_syms_t* syms, VkDevice logical_device,
    const iree_hal_vulkan_physical_device_snapshot_t* physical_device,
    iree_hal_vulkan_features_t enabled_features, VkPipelineCache pipeline_cache,
    iree_hal_vulkan_dispatch_abis_t enabled_dispatch_abis,
    const iree_hal_executable_params_t* executable_params,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable) {
  IREE_ASSERT_ARGUMENT(syms);
  IREE_ASSERT_ARGUMENT(logical_device);
  IREE_ASSERT_ARGUMENT(physical_device);
  IREE_ASSERT_ARGUMENT(executable_params);
  IREE_ASSERT_ARGUMENT(out_executable);
  *out_executable = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  if (!iree_hal_vulkan_executable_format_supported(
          enabled_features, enabled_dispatch_abis,
          executable_params->executable_format)) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(IREE_STATUS_NOT_FOUND,
                             "unsupported Vulkan executable format '%.*s'",
                             (int)executable_params->executable_format.size,
                             executable_params->executable_format.data));
  }
  const iree_hal_vulkan_dispatch_abis_t dispatch_abi =
      iree_hal_vulkan_executable_dispatch_abi_for_format(
          executable_params->executable_format);
  if (dispatch_abi == IREE_HAL_VULKAN_DISPATCH_ABI_BDA &&
      iree_string_view_equal(executable_params->executable_format,
                             IREE_SV("vulkan-spirv-bda-raw"))) {
    IREE_TRACE_ZONE_END(z0);
    return iree_hal_vulkan_create_raw_bda_executable(
        syms, logical_device, pipeline_cache, executable_params, host_allocator,
        out_executable);
  }

  iree_const_byte_span_t executable_flatbuffer = iree_const_byte_span_empty();
  iree_status_t status = iree_hal_read_executable_flatbuffer_header(
      executable_params->executable_data, /*unsafe_infer_size=*/false,
      iree_hal_vulkan_ExecutableDef_file_identifier, &executable_flatbuffer);
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_executable_flatbuffer_verify(
        physical_device, enabled_features, dispatch_abi, executable_flatbuffer);
  }

  iree_hal_vulkan_ExecutableDef_table_t executable_def = 0;
  iree_hal_vulkan_DescriptorSetLayoutDef_vec_t descriptor_set_layouts_vec = 0;
  iree_hal_vulkan_PipelineLayoutDef_vec_t pipeline_layouts_vec = 0;
  iree_hal_vulkan_ShaderModuleDef_vec_t shader_modules_vec = 0;
  iree_hal_vulkan_PipelineDef_vec_t pipelines_vec = 0;
  iree_host_size_t descriptor_set_layout_count = 0;
  iree_host_size_t pipeline_layout_count = 0;
  iree_host_size_t pipeline_count = 0;
  iree_host_size_t name_storage_size = 0;
  if (iree_status_is_ok(status)) {
    executable_def =
        iree_hal_vulkan_ExecutableDef_as_root(executable_flatbuffer.data);
    descriptor_set_layouts_vec =
        iree_hal_vulkan_ExecutableDef_descriptor_set_layouts_get(
            executable_def);
    pipeline_layouts_vec =
        iree_hal_vulkan_ExecutableDef_pipeline_layouts_get(executable_def);
    shader_modules_vec =
        iree_hal_vulkan_ExecutableDef_shader_modules_get(executable_def);
    pipelines_vec = iree_hal_vulkan_ExecutableDef_pipelines_get(executable_def);
    descriptor_set_layout_count =
        iree_hal_vulkan_DescriptorSetLayoutDef_vec_len(
            descriptor_set_layouts_vec);
    pipeline_layout_count =
        iree_hal_vulkan_PipelineLayoutDef_vec_len(pipeline_layouts_vec);
    pipeline_count = iree_hal_vulkan_PipelineDef_vec_len(pipelines_vec);
    status = iree_hal_vulkan_calculate_name_storage_size(pipelines_vec,
                                                         &name_storage_size);
  }

  iree_hal_vulkan_executable_t* executable = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_allocate_executable(
        syms, logical_device, descriptor_set_layout_count,
        pipeline_layout_count, pipeline_count, name_storage_size,
        host_allocator, &executable);
  }

  if (iree_status_is_ok(status)) {
    iree_hal_debug_publish_source_files(
        iree_hal_vulkan_ExecutableDef_source_files_get(executable_def));
    status = iree_hal_vulkan_create_descriptor_set_layouts(
        syms, logical_device, descriptor_set_layouts_vec,
        descriptor_set_layout_count, executable->descriptor_set_layouts,
        host_allocator);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_create_pipeline_layouts(
        syms, logical_device, pipeline_layouts_vec,
        executable->descriptor_set_layouts, pipeline_layout_count,
        executable->pipeline_layouts, host_allocator);
  }

  iree_host_size_t shader_module_count = 0;
  VkShaderModule* shader_modules = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_create_shader_modules(
        syms, logical_device, shader_modules_vec, host_allocator,
        &shader_module_count, &shader_modules);
  }

  VkSpecializationInfo specialization_info;
  VkSpecializationMapEntry* specialization_map_entries = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_create_specialization_info(
        executable_params, host_allocator, &specialization_info,
        &specialization_map_entries);
  }

  char* name_storage = executable ? executable->name_storage : NULL;
  iree_host_size_t name_storage_offset = 0;
  for (iree_host_size_t i = 0; iree_status_is_ok(status) && i < pipeline_count;
       ++i) {
    iree_hal_vulkan_PipelineDef_table_t pipeline_def =
        iree_hal_vulkan_PipelineDef_vec_at(pipelines_vec, i);
    iree_hal_vulkan_pipeline_t* pipeline = &executable->pipelines[i];
    flatbuffers_string_t entry_point =
        iree_hal_vulkan_PipelineDef_entry_point_get(pipeline_def);
    const iree_host_size_t entry_point_size =
        flatbuffers_string_len(entry_point);
    memcpy(name_storage + name_storage_offset, entry_point, entry_point_size);
    name_storage[name_storage_offset + entry_point_size] = 0;
    pipeline->name = iree_make_string_view(name_storage + name_storage_offset,
                                           entry_point_size);
    name_storage_offset += entry_point_size + /*NUL=*/1;

    status = iree_hal_vulkan_create_compute_pipeline(
        syms, logical_device, pipeline_cache, executable_params,
        &specialization_info, pipeline_layouts_vec, descriptor_set_layouts_vec,
        shader_modules_vec, executable->pipeline_layouts,
        executable->descriptor_set_layouts, shader_modules, host_allocator,
        dispatch_abi, pipeline_def, pipeline);
    if (!iree_status_is_ok(status)) {
      status = iree_status_annotate_f(status, "pipelines[%" PRIhsz "]", i);
    }
  }

  iree_allocator_free(host_allocator, specialization_map_entries);
  iree_hal_vulkan_destroy_shader_modules(syms, logical_device,
                                         shader_module_count, shader_modules,
                                         host_allocator);

  if (iree_status_is_ok(status)) {
    *out_executable = (iree_hal_executable_t*)executable;
  } else if (executable) {
    iree_hal_executable_destroy((iree_hal_executable_t*)executable);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_vulkan_executable_destroy(
    iree_hal_executable_t* base_executable) {
  iree_hal_vulkan_executable_t* executable =
      iree_hal_vulkan_executable_cast(base_executable);
  iree_allocator_t host_allocator = executable->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  for (iree_host_size_t i = 0; i < executable->pipeline_count; ++i) {
    if (executable->pipelines[i].handle) {
      iree_vkDestroyPipeline(IREE_VULKAN_DEVICE(executable->syms),
                             executable->logical_device,
                             executable->pipelines[i].handle,
                             /*pAllocator=*/NULL);
    }
    iree_allocator_free(host_allocator,
                        executable->pipelines[i].descriptor_bindings);
    iree_allocator_free(host_allocator,
                        executable->pipelines[i].descriptor_set_layouts);
    iree_allocator_free(host_allocator,
                        executable->pipelines[i].bda.binding_requirements);
  }
  for (iree_host_size_t i = 0; i < executable->pipeline_layout_count; ++i) {
    if (executable->pipeline_layouts[i]) {
      iree_vkDestroyPipelineLayout(IREE_VULKAN_DEVICE(executable->syms),
                                   executable->logical_device,
                                   executable->pipeline_layouts[i],
                                   /*pAllocator=*/NULL);
    }
  }
  for (iree_host_size_t i = 0; i < executable->descriptor_set_layout_count;
       ++i) {
    if (executable->descriptor_set_layouts[i]) {
      iree_vkDestroyDescriptorSetLayout(
          IREE_VULKAN_DEVICE(executable->syms), executable->logical_device,
          executable->descriptor_set_layouts[i], /*pAllocator=*/NULL);
    }
  }
  iree_allocator_free(host_allocator, executable->name_storage);
  iree_allocator_free(host_allocator, executable->pipelines);
  iree_allocator_free(host_allocator, executable->pipeline_layouts);
  iree_allocator_free(host_allocator, executable->descriptor_set_layouts);
  iree_allocator_free(host_allocator, executable);

  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_hal_vulkan_executable_lookup_pipeline(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    const iree_hal_vulkan_pipeline_t** out_pipeline) {
  IREE_ASSERT_ARGUMENT(out_pipeline);
  *out_pipeline = NULL;
  iree_hal_vulkan_executable_t* executable =
      iree_hal_vulkan_executable_cast(base_executable);
  if (export_ordinal >= executable->pipeline_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "export ordinal %" PRIu32
                            " out of range; executable has %" PRIhsz " exports",
                            export_ordinal, executable->pipeline_count);
  }
  *out_pipeline = &executable->pipelines[export_ordinal];
  return iree_ok_status();
}

static iree_host_size_t iree_hal_vulkan_executable_export_count(
    iree_hal_executable_t* base_executable) {
  iree_hal_vulkan_executable_t* executable =
      iree_hal_vulkan_executable_cast(base_executable);
  return executable->pipeline_count;
}

static iree_status_t iree_hal_vulkan_executable_export_info(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    iree_hal_executable_export_info_t* out_info) {
  memset(out_info, 0, sizeof(*out_info));
  const iree_hal_vulkan_pipeline_t* pipeline = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_executable_lookup_pipeline(
      base_executable, export_ordinal, &pipeline));
  out_info->name = pipeline->name;
  out_info->constant_count = pipeline->constant_count;
  out_info->binding_count = pipeline->binding_count;
  memcpy(out_info->workgroup_size, pipeline->workgroup_size,
         sizeof(out_info->workgroup_size));
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_executable_export_parameters(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    iree_host_size_t capacity,
    iree_hal_executable_export_parameter_t* out_parameters) {
  IREE_ASSERT_ARGUMENT(out_parameters || capacity == 0);
  const iree_hal_vulkan_pipeline_t* pipeline = NULL;
  return iree_hal_vulkan_executable_lookup_pipeline(base_executable,
                                                    export_ordinal, &pipeline);
}

static iree_status_t iree_hal_vulkan_executable_lookup_export_by_name(
    iree_hal_executable_t* base_executable, iree_string_view_t name,
    iree_hal_executable_export_ordinal_t* out_export_ordinal) {
  iree_hal_vulkan_executable_t* executable =
      iree_hal_vulkan_executable_cast(base_executable);
  for (iree_host_size_t i = 0; i < executable->pipeline_count; ++i) {
    if (iree_string_view_equal(executable->pipelines[i].name, name)) {
      *out_export_ordinal = (iree_hal_executable_export_ordinal_t)i;
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "export '%.*s' not found in executable",
                          (int)name.size, name.data);
}

static const iree_hal_executable_vtable_t iree_hal_vulkan_executable_vtable = {
    .destroy = iree_hal_vulkan_executable_destroy,
    .export_count = iree_hal_vulkan_executable_export_count,
    .export_info = iree_hal_vulkan_executable_export_info,
    .export_parameters = iree_hal_vulkan_executable_export_parameters,
    .lookup_export_by_name = iree_hal_vulkan_executable_lookup_export_by_name,
};
