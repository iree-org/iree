// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/native_executable.h"

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "iree/base/api.h"
#include "iree/hal/drivers/vulkan/dynamic_symbols.h"
#include "iree/hal/drivers/vulkan/handle_util.h"
#include "iree/hal/drivers/vulkan/pipeline_layout.h"
#include "iree/hal/drivers/vulkan/status_util.h"
#include "iree/hal/drivers/vulkan/util/ref_ptr.h"
#include "iree/hal/utils/executable_debug_info.h"

// flatcc schemas:
#include "iree/base/internal/flatcc/parsing.h"
#include "iree/schemas/executable_debug_info_reader.h"
#include "iree/schemas/executable_debug_info_verifier.h"
#include "iree/schemas/vulkan_executable_def_reader.h"
#include "iree/schemas/vulkan_executable_def_verifier.h"

using namespace iree::hal::vulkan;

//===----------------------------------------------------------------------===//
// FlatBuffer Verification
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_vulkan_shader_module_flatbuffer_verify(
    const iree_hal_vulkan_device_properties_t* device_properties,
    iree_hal_vulkan_ShaderModuleDef_table_t shader_module_def) {
  if (!shader_module_def) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "shader module is NULL");
  }
  flatbuffers_uint32_vec_t spirv_code_vec =
      iree_hal_vulkan_ShaderModuleDef_spirv_code_get(shader_module_def);
  if (!spirv_code_vec || flatbuffers_uint32_vec_len(spirv_code_vec) == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "shader module spirv_code is empty");
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_pipeline_layout_flatbuffer_verify(
    const iree_hal_vulkan_device_properties_t* device_properties,
    iree_hal_vulkan_DescriptorSetLayoutDef_vec_t descriptor_set_layouts_vec,
    iree_hal_vulkan_PipelineLayoutDef_table_t pipeline_layout_def) {
  if (!pipeline_layout_def) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pipeline layout is NULL");
  }

  // Basic descriptor set verification based on device limits. We don't know all
  // of the ways this can fail here but can provide better error messages when
  // limits are exceeded instead of relying on the optional validation layers
  // during execution.
  flatbuffers_uint32_vec_t descriptor_set_layout_ordinals_vec =
      iree_hal_vulkan_PipelineLayoutDef_descriptor_set_layout_ordinals_get(
          pipeline_layout_def);
  if (!descriptor_set_layout_ordinals_vec ||
      flatbuffers_uint32_vec_len(descriptor_set_layout_ordinals_vec) == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pipeline layout is has no descriptor sets");
  }
  uint32_t uniform_descriptor_count = 0;
  uint32_t storage_descriptor_count = 0;
  for (iree_host_size_t i = 0;
       i < flatbuffers_uint32_vec_len(descriptor_set_layout_ordinals_vec);
       ++i) {
    uint32_t descriptor_set_layout_ordinal =
        flatbuffers_uint32_vec_at(descriptor_set_layout_ordinals_vec, i);
    if (descriptor_set_layout_ordinal >
        iree_hal_vulkan_DescriptorSetLayoutDef_vec_len(
            descriptor_set_layouts_vec)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "pipeline layout references an invalid descriptor set ordinal %u",
          descriptor_set_layout_ordinal);
    }
    iree_hal_vulkan_DescriptorSetLayoutDef_table_t set_layout_def =
        iree_hal_vulkan_DescriptorSetLayoutDef_vec_at(
            descriptor_set_layouts_vec, descriptor_set_layout_ordinal);
    iree_hal_vulkan_DescriptorSetLayoutBindingDef_vec_t bindings_vec =
        iree_hal_vulkan_DescriptorSetLayoutDef_bindings_get(set_layout_def);
    for (iree_host_size_t j = 0;
         j <
         iree_hal_vulkan_DescriptorSetLayoutBindingDef_vec_len(bindings_vec);
         ++j) {
      iree_hal_vulkan_DescriptorSetLayoutBindingDef_table_t binding_def =
          iree_hal_vulkan_DescriptorSetLayoutBindingDef_vec_at(bindings_vec, j);
      uint32_t descriptor_count =
          iree_hal_vulkan_DescriptorSetLayoutBindingDef_descriptor_count_get(
              binding_def);
      iree_hal_vulkan_VkDescriptorType_enum_t type =
          iree_hal_vulkan_DescriptorSetLayoutBindingDef_descriptor_type_get(
              binding_def);
      uint32_t stage_flags =
          iree_hal_vulkan_DescriptorSetLayoutBindingDef_stage_flags_get(
              binding_def);
      switch (type) {
        case iree_hal_vulkan_VkDescriptorType_UNIFORM_BUFFER:
          uniform_descriptor_count += descriptor_count;
          break;
        case iree_hal_vulkan_VkDescriptorType_STORAGE_BUFFER:
          storage_descriptor_count += descriptor_count;
          break;
        default:
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "pipeline layout set[%" PRIhsz
                                  "] binding[%" PRIhsz
                                  "] has an unsupported descriptor_type",
                                  i, j);
      }
      // For now we limit to just COMPUTE. If we support other pipeline types in
      // the future we can expand these.
      if (stage_flags != VK_SHADER_STAGE_COMPUTE_BIT &&
          stage_flags != VK_SHADER_STAGE_ALL) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "pipeline layout set[%" PRIhsz "] binding[%" PRIhsz
            "] has invalid stage flags; must be VK_SHADER_STAGE_COMPUTE_BIT",
            i, j);
      }
    }
  }
  if (uniform_descriptor_count + storage_descriptor_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pipeline layout has no declared descriptor "
                            "bindings and must have at least one");
  } else if (uniform_descriptor_count >
             device_properties->limits
                 .max_per_stage_descriptor_uniform_buffers) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pipeline layout exceeds device maximum uniform "
        "buffer limit %u by using %u uniform descriptors",
        device_properties->limits.max_per_stage_descriptor_uniform_buffers,
        uniform_descriptor_count);
  } else if (storage_descriptor_count >
             device_properties->limits
                 .max_per_stage_descriptor_storage_buffers) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pipeline layout exceeds device maximum storage "
        "buffer limit %u by using %u storage descriptors",
        device_properties->limits.max_per_stage_descriptor_storage_buffers,
        storage_descriptor_count);
  }

  iree_hal_vulkan_PushConstantRange_vec_t push_constant_ranges_vec =
      iree_hal_vulkan_PipelineLayoutDef_push_constant_ranges_get(
          pipeline_layout_def);
  for (iree_host_size_t i = 0;
       i < iree_hal_vulkan_PushConstantRange_vec_len(push_constant_ranges_vec);
       ++i) {
    const iree_hal_vulkan_PushConstantRange_t* push_constant_range =
        iree_hal_vulkan_PushConstantRange_vec_at(push_constant_ranges_vec, i);

    // For now we limit to just COMPUTE. If we support other pipeline types in
    // the future we can expand these.
    if (push_constant_range->stage_flags != VK_SHADER_STAGE_COMPUTE_BIT &&
        push_constant_range->stage_flags != VK_SHADER_STAGE_ALL) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "pipeline layout push_constant_ranges[%" PRIhsz
                              "] has invalid stage flags; "
                              "must be VK_SHADER_STAGE_COMPUTE_BIT",
                              i);
    }

    // Ensure the push constant range is in-bounds. This is additional early
    // verification that is otherwise (probably) performed by the driver.
    uint32_t range_end =
        push_constant_range->offset + push_constant_range->size;
    if (range_end > device_properties->limits.max_push_constants_size) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "pipeline layout push_constant_ranges[%" PRIhsz
          "] (offset=%u, size=%u) "
          "exceeds device limit %u",
          i, push_constant_range->offset, push_constant_range->size,
          device_properties->limits.max_push_constants_size);
    }
  }

  return iree_ok_status();
}

// Verifies the structure of the FlatBuffer so that we can avoid doing so during
// runtime. There are still some conditions we must be aware of (such as omitted
// names on functions with internal linkage), however we shouldn't need to
// bounds check anything within the FlatBuffer after this succeeds.
static iree_status_t iree_hal_vulkan_executable_flatbuffer_verify(
    const iree_hal_vulkan_device_properties_t* device_properties,
    iree_const_byte_span_t flatbuffer_data) {
  if (!flatbuffer_data.data || flatbuffer_data.data_length < 16) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "FlatBuffer data is not present or less than 16 bytes (%" PRIhsz
        " total)",
        flatbuffer_data.data_length);
  }

  // Run flatcc generated verification. This ensures all pointers are in-bounds
  // and that we can safely walk the file, but not that the actual contents of
  // the FlatBuffer meet our expectations.
  int verify_ret = iree_hal_vulkan_ExecutableDef_verify_as_root(
      flatbuffer_data.data, flatbuffer_data.data_length);
  if (verify_ret != flatcc_verify_ok) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "FlatBuffer verification failed: %s",
                            flatcc_verify_error_string(verify_ret));
  }

  iree_hal_vulkan_ExecutableDef_table_t executable_def =
      iree_hal_vulkan_ExecutableDef_as_root(flatbuffer_data.data);

  iree_hal_vulkan_DescriptorSetLayoutDef_vec_t descriptor_set_layouts_vec =
      iree_hal_vulkan_ExecutableDef_descriptor_set_layouts_get(executable_def);
  iree_hal_vulkan_PipelineLayoutDef_vec_t pipeline_layouts_vec =
      iree_hal_vulkan_ExecutableDef_pipeline_layouts_get(executable_def);
  for (iree_host_size_t i = 0;
       i < iree_hal_vulkan_PipelineLayoutDef_vec_len(pipeline_layouts_vec);
       ++i) {
    iree_hal_vulkan_PipelineLayoutDef_table_t pipeline_layout_def =
        iree_hal_vulkan_PipelineLayoutDef_vec_at(pipeline_layouts_vec, i);
    IREE_RETURN_IF_ERROR(
        iree_hal_vulkan_pipeline_layout_flatbuffer_verify(
            device_properties, descriptor_set_layouts_vec, pipeline_layout_def),
        "pipeline_layouts[%" PRIhsz "]", i);
  }

  iree_hal_vulkan_ShaderModuleDef_vec_t shader_modules_vec =
      iree_hal_vulkan_ExecutableDef_shader_modules_get(executable_def);
  for (iree_host_size_t i = 0;
       i < iree_hal_vulkan_ShaderModuleDef_vec_len(shader_modules_vec); ++i) {
    iree_hal_vulkan_ShaderModuleDef_table_t shader_module_def =
        iree_hal_vulkan_ShaderModuleDef_vec_at(shader_modules_vec, i);
    IREE_RETURN_IF_ERROR(iree_hal_vulkan_shader_module_flatbuffer_verify(
                             device_properties, shader_module_def),
                         "shader_modules[%" PRIhsz "]", i);
  }

  iree_hal_vulkan_PipelineDef_vec_t pipelines_vec =
      iree_hal_vulkan_ExecutableDef_pipelines_get(executable_def);
  for (size_t i = 0; i < iree_hal_vulkan_PipelineDef_vec_len(pipelines_vec);
       ++i) {
    iree_hal_vulkan_PipelineDef_table_t export_def =
        iree_hal_vulkan_PipelineDef_vec_at(pipelines_vec, i);
    if (!export_def) continue;

    uint32_t shader_module_ordinal =
        iree_hal_vulkan_PipelineDef_shader_module_ordinal_get(export_def);
    if (shader_module_ordinal >=
        iree_hal_vulkan_ShaderModuleDef_vec_len(shader_modules_vec)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "pipelines[%" PRIhsz "] shader_module_ordinal is out of bounds", i);
    }

    if (flatbuffers_string_len(
            iree_hal_vulkan_PipelineDef_entry_point_get(export_def)) == 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "pipelines[%" PRIhsz "] name is empty", i);
    }

    uint32_t pipeline_layout_ordinal =
        iree_hal_vulkan_PipelineDef_pipeline_layout_ordinal_get(export_def);
    if (pipeline_layout_ordinal >=
        iree_hal_vulkan_PipelineLayoutDef_vec_len(pipeline_layouts_vec)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "pipelines[%" PRIhsz "] pipeline_layout_ordinal is out of bounds", i);
    }

    IREE_RETURN_IF_ERROR(iree_hal_debug_verify_export_def(
        iree_hal_vulkan_PipelineDef_debug_info_get(export_def)));
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Descriptor Set Layouts
//===----------------------------------------------------------------------===//

static void iree_hal_vulkan_release_descriptor_set_layouts(
    VkDeviceHandle* logical_device,
    iree_host_size_t descriptor_set_layout_count,
    iree_hal_vulkan_descriptor_set_layout_t** descriptor_set_layouts) {
  for (iree_host_size_t i = 0; i < descriptor_set_layout_count; ++i) {
    iree_hal_vulkan_descriptor_set_layout_release(descriptor_set_layouts[i]);
  }
  iree_allocator_free(logical_device->host_allocator(), descriptor_set_layouts);
}

// Creates a descriptor set layout based on the flatbuffer definition.
static iree_status_t iree_hal_vulkan_create_descriptor_set_layout(
    VkDeviceHandle* logical_device,
    iree_hal_vulkan_DescriptorSetLayoutDef_table_t descriptor_set_layout_def,
    iree_hal_vulkan_descriptor_set_layout_t** out_descriptor_set_layout) {
  IREE_ASSERT_ARGUMENT(logical_device);
  IREE_ASSERT_ARGUMENT(descriptor_set_layout_def);
  IREE_ASSERT_ARGUMENT(out_descriptor_set_layout);
  *out_descriptor_set_layout = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_vulkan_DescriptorSetLayoutBindingDef_vec_t bindings_vec =
      iree_hal_vulkan_DescriptorSetLayoutDef_bindings_get(
          descriptor_set_layout_def);
  iree_host_size_t binding_count =
      iree_hal_vulkan_DescriptorSetLayoutBindingDef_vec_len(bindings_vec);

  VkDescriptorSetLayoutBinding* bindings = NULL;
  if (binding_count > 0) {
    // TODO(benvanik): avoid this allocation if possible (inline_array).
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(
        logical_device->host_allocator(),
        binding_count * sizeof(VkDescriptorSetLayoutBinding),
        (void**)&bindings));
    for (iree_host_size_t i = 0; i < binding_count; ++i) {
      iree_hal_vulkan_DescriptorSetLayoutBindingDef_table_t binding_def =
          iree_hal_vulkan_DescriptorSetLayoutBindingDef_vec_at(bindings_vec, i);
      VkDescriptorSetLayoutBinding* binding = &bindings[i];
      binding->binding =
          iree_hal_vulkan_DescriptorSetLayoutBindingDef_binding_get(
              binding_def);
      binding->descriptorType = static_cast<VkDescriptorType>(
          iree_hal_vulkan_DescriptorSetLayoutBindingDef_descriptor_type_get(
              binding_def));
      binding->descriptorCount =
          iree_hal_vulkan_DescriptorSetLayoutBindingDef_descriptor_count_get(
              binding_def);
      binding->stageFlags = static_cast<VkShaderStageFlags>(
          iree_hal_vulkan_DescriptorSetLayoutBindingDef_stage_flags_get(
              binding_def));
      binding->pImmutableSamplers = NULL;
    }
  }

  VkDescriptorSetLayoutCreateFlags flags = 0;
  iree_hal_vulkan_descriptor_set_layout_t* descriptor_set_layout = NULL;
  iree_status_t status = iree_hal_vulkan_descriptor_set_layout_create(
      logical_device, flags, binding_count, bindings, &descriptor_set_layout);

  iree_allocator_free(logical_device->host_allocator(), bindings);

  if (iree_status_is_ok(status)) {
    *out_descriptor_set_layout = descriptor_set_layout;
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Creates all descriptor set layouts specified and returns a temporary heap
// array with them in the same order. Callers must use
// iree_hal_vulkan_release_descriptor_set_layouts when done with the array to
// release the resources.
static iree_status_t iree_hal_vulkan_create_descriptor_set_layouts(
    VkDeviceHandle* logical_device,
    iree_hal_vulkan_DescriptorSetLayoutDef_vec_t descriptor_set_layouts_vec,
    iree_host_size_t* out_descriptor_set_layout_count,
    iree_hal_vulkan_descriptor_set_layout_t*** out_descriptor_set_layouts) {
  IREE_ASSERT_ARGUMENT(logical_device);
  IREE_ASSERT_ARGUMENT(descriptor_set_layouts_vec);
  IREE_ASSERT_ARGUMENT(out_descriptor_set_layout_count);
  IREE_ASSERT_ARGUMENT(out_descriptor_set_layouts);
  *out_descriptor_set_layout_count = 0;
  *out_descriptor_set_layouts = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_host_size_t descriptor_set_layout_count =
      iree_hal_vulkan_DescriptorSetLayoutDef_vec_len(
          descriptor_set_layouts_vec);
  iree_hal_vulkan_descriptor_set_layout_t** descriptor_set_layouts = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(
              logical_device->host_allocator(),
              descriptor_set_layout_count * sizeof(descriptor_set_layouts[0]),
              (void**)&descriptor_set_layouts));

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < descriptor_set_layout_count; ++i) {
    iree_hal_vulkan_DescriptorSetLayoutDef_table_t descriptor_set_layout_def =
        iree_hal_vulkan_DescriptorSetLayoutDef_vec_at(
            descriptor_set_layouts_vec, i);
    status = iree_hal_vulkan_create_descriptor_set_layout(
        logical_device, descriptor_set_layout_def, &descriptor_set_layouts[i]);
    if (!iree_status_is_ok(status)) {
      status = iree_status_annotate_f(status,
                                      "descriptor_set_layouts[%" PRIhsz "]", i);
      break;
    }
  }

  if (iree_status_is_ok(status)) {
    *out_descriptor_set_layout_count = descriptor_set_layout_count;
    *out_descriptor_set_layouts = descriptor_set_layouts;
  } else {
    iree_hal_vulkan_release_descriptor_set_layouts(
        logical_device, descriptor_set_layout_count, descriptor_set_layouts);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Pipeline Layouts
//===----------------------------------------------------------------------===//

// Creates a pipeline layout from the flatbuffer definition using the descriptor
// set layouts provided.
static iree_status_t iree_hal_vulkan_create_pipeline_layout(
    VkDeviceHandle* logical_device,
    iree_host_size_t descriptor_set_layout_count,
    iree_hal_vulkan_descriptor_set_layout_t** descriptor_set_layouts,
    iree_hal_vulkan_PipelineLayoutDef_table_t pipeline_layout_def,
    iree_hal_vulkan_pipeline_layout_t** out_pipeline_layout) {
  IREE_ASSERT_ARGUMENT(logical_device);
  IREE_ASSERT_ARGUMENT(pipeline_layout_def);
  IREE_ASSERT_ARGUMENT(descriptor_set_layouts);
  IREE_ASSERT_ARGUMENT(out_pipeline_layout);
  *out_pipeline_layout = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_vulkan_PushConstantRange_vec_t push_constant_ranges =
      iree_hal_vulkan_PipelineLayoutDef_push_constant_ranges_get(
          pipeline_layout_def);
  iree_host_size_t push_constant_range_count =
      iree_hal_vulkan_PushConstantRange_vec_len(push_constant_ranges);
  const VkPushConstantRange* push_constant_range_ptr = NULL;
  if (push_constant_range_count > 0) {
    static_assert(sizeof(iree_hal_vulkan_PushConstantRange) ==
                      sizeof(VkPushConstantRange),
                  "expecting to overlay VkPushConstantRange");
    push_constant_range_ptr =
        (const VkPushConstantRange*)iree_hal_vulkan_PushConstantRange_vec_at(
            push_constant_ranges, 0);
  }

  flatbuffers_uint32_vec_t descriptor_set_layout_ordinals_vec =
      iree_hal_vulkan_PipelineLayoutDef_descriptor_set_layout_ordinals_get(
          pipeline_layout_def);
  iree_host_size_t selected_set_layouts_count =
      flatbuffers_uint32_vec_len(descriptor_set_layout_ordinals_vec);
  iree_hal_vulkan_descriptor_set_layout_t** selected_set_layouts =
      (iree_hal_vulkan_descriptor_set_layout_t**)iree_alloca(
          selected_set_layouts_count *
          sizeof(iree_hal_vulkan_descriptor_set_layout_t*));
  for (iree_host_size_t i = 0; i < selected_set_layouts_count; ++i) {
    uint32_t ordinal =
        flatbuffers_uint32_vec_at(descriptor_set_layout_ordinals_vec, i);
    selected_set_layouts[i] = descriptor_set_layouts[ordinal];
  }

  iree_status_t status = iree_hal_vulkan_pipeline_layout_create(
      logical_device, push_constant_range_count, push_constant_range_ptr,
      selected_set_layouts_count, selected_set_layouts, out_pipeline_layout);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_vulkan_release_pipeline_layouts(
    VkDeviceHandle* logical_device, iree_host_size_t pipeline_layout_count,
    iree_hal_vulkan_pipeline_layout_t** pipeline_layouts) {
  IREE_TRACE_ZONE_BEGIN(z0);
  for (iree_host_size_t i = 0; i < pipeline_layout_count; ++i) {
    iree_hal_vulkan_pipeline_layout_release(pipeline_layouts[i]);
  }
  iree_allocator_free(logical_device->host_allocator(), pipeline_layouts);
  IREE_TRACE_ZONE_END(z0);
}

// Creates all pipeline layouts specified and returns a temporary heap array
// with them in the same order. Callers must use
// iree_hal_vulkan_release_pipeline_layouts when done with the array to
// release the resources.
static iree_status_t iree_hal_vulkan_create_pipeline_layouts(
    VkDeviceHandle* logical_device,
    iree_hal_vulkan_DescriptorSetLayoutDef_vec_t descriptor_set_layouts_vec,
    iree_hal_vulkan_PipelineLayoutDef_vec_t pipeline_layouts_vec,
    iree_host_size_t* out_pipeline_layout_count,
    iree_hal_vulkan_pipeline_layout_t*** out_pipeline_layouts) {
  IREE_ASSERT_ARGUMENT(logical_device);
  IREE_ASSERT_ARGUMENT(descriptor_set_layouts_vec);
  IREE_ASSERT_ARGUMENT(pipeline_layouts_vec);
  IREE_ASSERT_ARGUMENT(out_pipeline_layout_count);
  IREE_ASSERT_ARGUMENT(out_pipeline_layouts);
  *out_pipeline_layout_count = 0;
  *out_pipeline_layouts = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Create a temporary descriptor set layout list to retain the layouts while
  // creating pipeline layouts. The created pipeline layouts will retain the
  // descriptor set layouts for as long as they are live even once we free the
  // list below.
  iree_host_size_t descriptor_set_layout_count = 0;
  iree_hal_vulkan_descriptor_set_layout_t** descriptor_set_layouts = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_vulkan_create_descriptor_set_layouts(
              logical_device, descriptor_set_layouts_vec,
              &descriptor_set_layout_count, &descriptor_set_layouts));

  iree_host_size_t pipeline_layout_count =
      iree_hal_vulkan_PipelineLayoutDef_vec_len(pipeline_layouts_vec);
  iree_hal_vulkan_pipeline_layout_t** pipeline_layouts = NULL;
  iree_status_t status =
      iree_allocator_malloc(logical_device->host_allocator(),
                            pipeline_layout_count * sizeof(pipeline_layouts[0]),
                            (void**)&pipeline_layouts);

  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < pipeline_layout_count; ++i) {
      iree_hal_vulkan_PipelineLayoutDef_table_t pipeline_layout_def =
          iree_hal_vulkan_PipelineLayoutDef_vec_at(pipeline_layouts_vec, i);
      status = iree_hal_vulkan_create_pipeline_layout(
          logical_device, descriptor_set_layout_count, descriptor_set_layouts,
          pipeline_layout_def, &pipeline_layouts[i]);
      if (!iree_status_is_ok(status)) {
        status =
            iree_status_annotate_f(status, "pipeline_layouts[%" PRIhsz "]", i);
        break;
      }
    }
  }

  // Release temporary descriptor set layouts; pipeline layouts retain them as
  // needed.
  iree_hal_vulkan_release_descriptor_set_layouts(
      logical_device, descriptor_set_layout_count, descriptor_set_layouts);

  if (iree_status_is_ok(status)) {
    *out_pipeline_layout_count = pipeline_layout_count;
    *out_pipeline_layouts = pipeline_layouts;
  } else {
    iree_hal_vulkan_release_pipeline_layouts(
        logical_device, pipeline_layout_count, pipeline_layouts);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Shader Modules
//===----------------------------------------------------------------------===//

static void iree_hal_vulkan_release_shader_modules(
    VkDeviceHandle* logical_device, iree_host_size_t shader_module_count,
    VkShaderModule* shader_modules) {
  IREE_TRACE_ZONE_BEGIN(z0);
  for (iree_host_size_t i = 0; i < shader_module_count; ++i) {
    if (shader_modules[i] != VK_NULL_HANDLE) {
      logical_device->syms()->vkDestroyShaderModule(
          *logical_device, shader_modules[i], logical_device->allocator());
    }
  }
  iree_allocator_free(logical_device->host_allocator(), shader_modules);
  IREE_TRACE_ZONE_END(z0);
}

// Creates a VkShaderModule from the given flatbuffer definition.
// This usually spends quite a bit of blocking time in the driver.
static iree_status_t iree_hal_vulkan_create_shader_module(
    VkDeviceHandle* logical_device,
    iree_hal_vulkan_ShaderModuleDef_table_t shader_module_def,
    VkShaderModule* out_shader_module) {
  IREE_ASSERT_ARGUMENT(logical_device);
  IREE_ASSERT_ARGUMENT(shader_module_def);
  IREE_ASSERT_ARGUMENT(out_shader_module);
  *out_shader_module = VK_NULL_HANDLE;
  IREE_TRACE_ZONE_BEGIN(z0);

  VkShaderModuleCreateInfo create_info;
  create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  create_info.pNext = NULL;
  create_info.flags = 0;

  flatbuffers_uint32_vec_t spirv_code_vec =
      iree_hal_vulkan_ShaderModuleDef_spirv_code_get(shader_module_def);
  create_info.codeSize =
      flatbuffers_uint32_vec_len(spirv_code_vec) * sizeof(uint32_t);
  create_info.pCode = (const uint32_t*)spirv_code_vec;

  VkShaderModule shader_module = VK_NULL_HANDLE;
  iree_status_t status =
      VK_RESULT_TO_STATUS(logical_device->syms()->vkCreateShaderModule(
                              *logical_device, &create_info,
                              logical_device->allocator(), &shader_module),
                          "vkCreateShaderModule");

  if (iree_status_is_ok(status)) {
    *out_shader_module = shader_module;
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Creates all shader modules specified and returns a temporary heap array with
// them in the same order. Callers must use
// iree_hal_vulkan_release_shader_modules when done with the array to release
// the resources.
static iree_status_t iree_hal_vulkan_create_shader_modules(
    VkDeviceHandle* logical_device,
    iree_hal_vulkan_ShaderModuleDef_vec_t shader_modules_vec,
    iree_host_size_t* out_shader_module_count,
    VkShaderModule** out_shader_modules) {
  IREE_ASSERT_ARGUMENT(logical_device);
  IREE_ASSERT_ARGUMENT(shader_modules_vec);
  IREE_ASSERT_ARGUMENT(out_shader_module_count);
  IREE_ASSERT_ARGUMENT(out_shader_modules);
  *out_shader_module_count = 0;
  *out_shader_modules = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_host_size_t shader_module_count =
      iree_hal_vulkan_ShaderModuleDef_vec_len(shader_modules_vec);
  VkShaderModule* shader_modules = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(logical_device->host_allocator(),
                                shader_module_count * sizeof(shader_modules[0]),
                                (void**)&shader_modules));

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < shader_module_count; ++i) {
    iree_hal_vulkan_ShaderModuleDef_table_t shader_module_def =
        iree_hal_vulkan_ShaderModuleDef_vec_at(shader_modules_vec, i);
    status = iree_hal_vulkan_create_shader_module(
        logical_device, shader_module_def, &shader_modules[i]);
    if (!iree_status_is_ok(status)) {
      status = iree_status_annotate_f(status, "shader_modules[%" PRIhsz "]", i);
      break;
    }
  }

  if (iree_status_is_ok(status)) {
    *out_shader_module_count = shader_module_count;
    *out_shader_modules = shader_modules;
  } else {
    iree_hal_vulkan_release_shader_modules(logical_device, shader_module_count,
                                           shader_modules);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Pipelines
//===----------------------------------------------------------------------===//

// Creates a pipeline from the set of available pipeline layouts and shader
// modules and stores it into |out_pipeline|.
//
// NOTE: vkCreateComputePipelines takes multiple pipelines but doesn't speed up
// creation on any known driver; we process one at a time so that we can get
// better error messages and multithread the pipeline creation ourselves.
static iree_status_t iree_hal_vulkan_create_pipeline(
    VkDeviceHandle* logical_device, VkPipelineCache pipeline_cache,
    const iree_hal_executable_params_t* executable_params,
    const VkSpecializationInfo* specialization_info,
    iree_hal_vulkan_pipeline_layout_t** pipeline_layouts,
    VkShaderModule* shader_modules,
    iree_hal_vulkan_PipelineDef_table_t pipeline_def,
    iree_hal_vulkan_pipeline_t* out_pipeline) {
  IREE_ASSERT_ARGUMENT(logical_device);
  IREE_ASSERT_ARGUMENT(pipeline_layouts);
  IREE_ASSERT_ARGUMENT(shader_modules);
  IREE_ASSERT_ARGUMENT(out_pipeline);
  IREE_TRACE_ZONE_BEGIN(z0);

  flatbuffers_string_t entry_point =
      iree_hal_vulkan_PipelineDef_entry_point_get(pipeline_def);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, entry_point);

  uint32_t shader_module_ordinal =
      iree_hal_vulkan_PipelineDef_shader_module_ordinal_get(pipeline_def);
  VkShaderModule shader_module = shader_modules[shader_module_ordinal];
  uint32_t pipeline_layout_ordinal =
      iree_hal_vulkan_PipelineDef_pipeline_layout_ordinal_get(pipeline_def);
  iree_hal_vulkan_pipeline_layout_t* pipeline_layout =
      pipeline_layouts[pipeline_layout_ordinal];

  VkComputePipelineCreateInfo create_info;
  memset(&create_info, 0, sizeof(create_info));
  create_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  create_info.pNext = NULL;
  create_info.flags = 0;
  if (!iree_all_bits_set(executable_params->caching_mode,
                         IREE_HAL_EXECUTABLE_CACHING_MODE_ALLOW_OPTIMIZATION)) {
    create_info.flags |= VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT;
  }
  create_info.layout = iree_hal_vulkan_pipeline_layout_handle(pipeline_layout);
  create_info.basePipelineHandle = VK_NULL_HANDLE;
  create_info.basePipelineIndex = 0;

  VkPipelineShaderStageCreateInfo* stage_create_info = &create_info.stage;
  stage_create_info->sType =
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stage_create_info->pNext = NULL;
  stage_create_info->flags = 0;
  stage_create_info->stage = VK_SHADER_STAGE_COMPUTE_BIT;
  stage_create_info->module = shader_module;
  stage_create_info->pName = entry_point;
  stage_create_info->pSpecializationInfo = specialization_info;

  // If subgroup size is not 0, request the said subgroup size via
  // VK_EXT_subgroup_size_control (promoted to core since v1.3).
  VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subgroup_size_info;
  memset(&subgroup_size_info, 0, sizeof(subgroup_size_info));
  if (iree_hal_vulkan_PipelineDef_subgroup_size_is_present(pipeline_def)) {
    if (uint32_t subgroup_size =
            iree_hal_vulkan_PipelineDef_subgroup_size(pipeline_def)) {
      subgroup_size_info.sType =
          VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO;
      subgroup_size_info.pNext = NULL;
      subgroup_size_info.requiredSubgroupSize = subgroup_size;
      stage_create_info->pNext = &subgroup_size_info;
    }
  }

  // Create the pipeline. This may fail if the shader module or pipeline are
  // invalid or the pipeline layout does not match expectations.
  iree_status_t status = VK_RESULT_TO_STATUS(
      logical_device->syms()->vkCreateComputePipelines(
          *logical_device, pipeline_cache, 1, &create_info,
          logical_device->allocator(), &out_pipeline->handle),
      "vkCreateComputePipelines");

  // Retain the pipeline layout for as long as the pipeline is live.
  out_pipeline->layout = pipeline_layout;
  iree_hal_vulkan_pipeline_layout_retain(out_pipeline->layout);

  // Set pipeline name for tooling.
  if (iree_status_is_ok(status)) {
    if (PFN_vkSetDebugUtilsObjectNameEXT set_name =
            logical_device->syms()->vkSetDebugUtilsObjectNameEXT) {
      VkDebugUtilsObjectNameInfoEXT name_info = {};
      name_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
      name_info.pNext = NULL;
      name_info.objectHandle = (uint64_t)out_pipeline->handle;
      name_info.objectType = VK_OBJECT_TYPE_PIPELINE;
      name_info.pObjectName =
          iree_hal_vulkan_PipelineDef_entry_point_get(pipeline_def);
      set_name(*logical_device, &name_info);
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_vulkan_destroy_pipeline(
    VkDeviceHandle* logical_device, iree_hal_vulkan_pipeline_t* pipeline) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (pipeline->handle != VK_NULL_HANDLE) {
    logical_device->syms()->vkDestroyPipeline(*logical_device, pipeline->handle,
                                              logical_device->allocator());
  }
  iree_hal_vulkan_pipeline_layout_release(pipeline->layout);
  IREE_TRACE_ZONE_END(z0);
}

// Creates all pipelines in the flatbuffer and stores them directly into
// the caller-allocated |pipelines| array. Upon failure the caller is
// responsible for releasing partially initialized pipelines.
//
// NOTE: this function is designed as a top-level flatbuffer->VkPipeline[] entry
// point for future multi-threaded pipeline creation. Today we do everything
// serially but could farm out to an iree_loop_t.
static iree_status_t iree_hal_vulkan_create_pipelines(
    VkDeviceHandle* logical_device, VkPipelineCache pipeline_cache,
    const iree_hal_executable_params_t* executable_params,
    iree_hal_vulkan_DescriptorSetLayoutDef_vec_t descriptor_set_layouts_vec,
    iree_hal_vulkan_PipelineLayoutDef_vec_t pipeline_layouts_vec,
    iree_hal_vulkan_ShaderModuleDef_vec_t shader_modules_vec,
    iree_hal_vulkan_PipelineDef_vec_t pipelines_vec,
    iree_hal_vulkan_pipeline_t* pipelines) {
  IREE_ASSERT_ARGUMENT(logical_device);
  IREE_ASSERT_ARGUMENT(descriptor_set_layouts_vec);
  IREE_ASSERT_ARGUMENT(pipeline_layouts_vec);
  IREE_ASSERT_ARGUMENT(shader_modules_vec);
  IREE_ASSERT_ARGUMENT(pipelines_vec);
  IREE_ASSERT_ARGUMENT(pipelines);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Create a temporary pipeline layout list to retain the layouts while
  // creating pipelines. The created pipelines will retain the pipeline layouts
  // (and transitively the descriptor set layouts) for as long as they are live
  // even once we free the list below. This is usually a much smaller set than
  // the total number of pipelines (~5-10 for 1000 pipelines) so we split this
  // from the pipeline creation.
  iree_host_size_t pipeline_layout_count = 0;
  iree_hal_vulkan_pipeline_layout_t** pipeline_layouts = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_vulkan_create_pipeline_layouts(
              logical_device, descriptor_set_layouts_vec, pipeline_layouts_vec,
              &pipeline_layout_count, &pipeline_layouts));

  // Create all shader modules used by pipelines into a temporary array.
  // The shader modules are only required during pipeline creation and are then
  // discarded.
  iree_host_size_t shader_module_count = 0;
  VkShaderModule* shader_modules = NULL;
  iree_status_t status = iree_hal_vulkan_create_shader_modules(
      logical_device, shader_modules_vec, &shader_module_count,
      &shader_modules);

  // Prepare specialization entries used across all pipelines.
  VkSpecializationMapEntry* specialization_map_entries = NULL;
  VkSpecializationInfo specialization_info;
  memset(&specialization_info, 0, sizeof(specialization_info));
  if (iree_status_is_ok(status) && executable_params->constant_count) {
    status = iree_allocator_malloc(logical_device->host_allocator(),
                                   executable_params->constant_count *
                                       sizeof(specialization_map_entries[0]),
                                   (void**)&specialization_map_entries);
  }
  if (iree_status_is_ok(status)) {
    specialization_info.mapEntryCount = executable_params->constant_count;
    specialization_info.pMapEntries = specialization_map_entries;
    specialization_info.dataSize =
        executable_params->constant_count * sizeof(uint32_t);
    specialization_info.pData = executable_params->constants;
    for (iree_host_size_t i = 0; i < executable_params->constant_count; ++i) {
      specialization_map_entries[i].constantID = i;
      specialization_map_entries[i].offset = i * sizeof(uint32_t);
      specialization_map_entries[i].size = sizeof(uint32_t);
    }
  }

  // Create pipelines in-place in the output storage using the temporary
  // pipeline layouts array. The pipeline layouts will be retained as needed.
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0;
         i < iree_hal_vulkan_PipelineDef_vec_len(pipelines_vec); ++i) {
      iree_hal_vulkan_PipelineDef_table_t pipeline_def =
          iree_hal_vulkan_PipelineDef_vec_at(pipelines_vec, i);
      status = iree_hal_vulkan_create_pipeline(
          logical_device, pipeline_cache, executable_params,
          &specialization_info, pipeline_layouts, shader_modules, pipeline_def,
          &pipelines[i]);
      if (!iree_status_is_ok(status)) {
        status = iree_status_annotate_f(status, "pipelines[%" PRIhsz "]", i);
        break;
      }
    }
  }

  iree_allocator_free(logical_device->host_allocator(),
                      specialization_map_entries);
  iree_hal_vulkan_release_shader_modules(logical_device, shader_module_count,
                                         shader_modules);
  iree_hal_vulkan_release_pipeline_layouts(
      logical_device, pipeline_layout_count, pipeline_layouts);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// iree_hal_vulkan_native_executable_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_vulkan_native_executable_t {
  iree_hal_resource_t resource;
  VkDeviceHandle* logical_device;
  iree_host_size_t pipeline_count;
  iree_hal_vulkan_pipeline_t pipelines[];
} iree_hal_vulkan_native_executable_t;

namespace {
extern const iree_hal_executable_vtable_t
    iree_hal_vulkan_native_executable_vtable;
}  // namespace

static iree_hal_vulkan_native_executable_t*
iree_hal_vulkan_native_executable_cast(iree_hal_executable_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_vulkan_native_executable_vtable);
  return (iree_hal_vulkan_native_executable_t*)base_value;
}

iree_status_t iree_hal_vulkan_native_executable_create(
    iree::hal::vulkan::VkDeviceHandle* logical_device,
    VkPipelineCache pipeline_cache,
    const iree_hal_executable_params_t* executable_params,
    iree_hal_executable_t** out_executable) {
  IREE_ASSERT_ARGUMENT(logical_device);
  IREE_ASSERT_ARGUMENT(executable_params);
  IREE_ASSERT_ARGUMENT(out_executable);
  *out_executable = NULL;
  iree_allocator_t host_allocator = logical_device->host_allocator();
  IREE_TRACE_ZONE_BEGIN(z0);

  // Verify and fetch the executable FlatBuffer wrapper.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_vulkan_executable_flatbuffer_verify(
              &logical_device->supported_properties(),
              executable_params->executable_data));
  iree_hal_vulkan_ExecutableDef_table_t executable_def =
      iree_hal_vulkan_ExecutableDef_as_root(
          executable_params->executable_data.data);

  iree_hal_vulkan_PipelineDef_vec_t pipelines_vec =
      iree_hal_vulkan_ExecutableDef_pipelines_get(executable_def);
  iree_host_size_t pipeline_count =
      iree_hal_vulkan_PipelineDef_vec_len(pipelines_vec);

  // Calculate the total number of characters across all entry point names. This
  // is only required when tracing so that we can store copies of the names as
  // the flatbuffer storing the strings may be released while the executable is
  // still live.
  iree_host_size_t total_export_info_length = 0;
  IREE_TRACE({
    for (iree_host_size_t i = 0; i < pipeline_count; ++i) {
      iree_hal_vulkan_PipelineDef_table_t pipeline_def =
          iree_hal_vulkan_PipelineDef_vec_at(pipelines_vec, i);
      total_export_info_length += iree_hal_debug_calculate_export_info_size(
          iree_hal_vulkan_PipelineDef_debug_info_get(pipeline_def));
    }
  });

  iree_hal_vulkan_native_executable_t* executable = NULL;
  const iree_host_size_t total_size =
      sizeof(*executable) + pipeline_count * sizeof(executable->pipelines[0]) +
      total_export_info_length;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator, total_size, (void**)&executable));
  iree_hal_resource_initialize(&iree_hal_vulkan_native_executable_vtable,
                               &executable->resource);
  executable->logical_device = logical_device;
  executable->pipeline_count = pipeline_count;
  memset(executable->pipelines, 0,
         pipeline_count * sizeof(executable->pipelines[0]));

  // Publish any embedded source files to the tracing infrastructure.
  iree_hal_debug_publish_source_files(
      iree_hal_vulkan_ExecutableDef_source_files_get(executable_def));

  // Create one pipeline per exported function.
  iree_hal_vulkan_DescriptorSetLayoutDef_vec_t descriptor_set_layouts_vec =
      iree_hal_vulkan_ExecutableDef_descriptor_set_layouts_get(executable_def);
  iree_hal_vulkan_PipelineLayoutDef_vec_t pipeline_layouts_vec =
      iree_hal_vulkan_ExecutableDef_pipeline_layouts_get(executable_def);
  iree_hal_vulkan_ShaderModuleDef_vec_t shader_modules_vec =
      iree_hal_vulkan_ExecutableDef_shader_modules_get(executable_def);
  iree_status_t status = iree_hal_vulkan_create_pipelines(
      logical_device, pipeline_cache, executable_params,
      descriptor_set_layouts_vec, pipeline_layouts_vec, shader_modules_vec,
      pipelines_vec, executable->pipelines);

  // Populate tracing info for each pipeline.
  if (iree_status_is_ok(status)) {
    IREE_TRACE({
      iree_hal_debug_export_info_t* export_infos =
          (iree_hal_debug_export_info_t*)((uint8_t*)executable->pipelines +
                                          pipeline_count *
                                              sizeof(executable->pipelines[0]));
      for (iree_host_size_t i = 0; i < pipeline_count; ++i) {
        iree_hal_vulkan_PipelineDef_table_t pipeline_def =
            iree_hal_vulkan_PipelineDef_vec_at(pipelines_vec, i);
        iree_hal_vulkan_pipeline_t* pipeline = &executable->pipelines[i];
        iree_hal_debug_copy_export_info(
            iree_hal_vulkan_PipelineDef_debug_info_get(pipeline_def),
            &export_infos[i]);
        pipeline->source_location.file_name = export_infos[i].source_filename;
        pipeline->source_location.line = export_infos[i].source_line;
        pipeline->source_location.func_name = export_infos[i].function_name;
      }
    });
  }

  if (iree_status_is_ok(status)) {
    *out_executable = (iree_hal_executable_t*)executable;
  } else {
    iree_hal_executable_destroy((iree_hal_executable_t*)executable);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_vulkan_native_executable_destroy(
    iree_hal_executable_t* base_executable) {
  iree_hal_vulkan_native_executable_t* executable =
      iree_hal_vulkan_native_executable_cast(base_executable);
  iree_allocator_t host_allocator =
      executable->logical_device->host_allocator();
  IREE_TRACE_ZONE_BEGIN(z0);

  for (iree_host_size_t i = 0; i < executable->pipeline_count; ++i) {
    iree_hal_vulkan_destroy_pipeline(executable->logical_device,
                                     &executable->pipelines[i]);
  }
  iree_allocator_free(host_allocator, executable);

  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_hal_vulkan_native_executable_lookup_pipeline(
    iree_hal_executable_t* base_executable, uint32_t entry_ordinal,
    const iree_hal_vulkan_pipeline_t** out_pipeline) {
  iree_hal_vulkan_native_executable_t* executable =
      iree_hal_vulkan_native_executable_cast(base_executable);
  if (entry_ordinal >= executable->pipeline_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "invalid entry point ordinal %u", entry_ordinal);
  }
  *out_pipeline = &executable->pipelines[entry_ordinal];
  return iree_ok_status();
}

namespace {
const iree_hal_executable_vtable_t iree_hal_vulkan_native_executable_vtable = {
    /*.destroy=*/iree_hal_vulkan_native_executable_destroy,
};
}  // namespace
