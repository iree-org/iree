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
#include "iree/base/tracing.h"
#include "iree/hal/drivers/vulkan/dynamic_symbol_tables.h"
#include "iree/hal/drivers/vulkan/dynamic_symbols.h"
#include "iree/hal/drivers/vulkan/handle_util.h"
#include "iree/hal/drivers/vulkan/native_pipeline_layout.h"
#include "iree/hal/drivers/vulkan/status_util.h"
#include "iree/hal/drivers/vulkan/util/ref_ptr.h"

// flatcc schemas:
#include "iree/base/internal/flatcc/parsing.h"
#include "iree/schemas/spirv_executable_def_reader.h"
#include "iree/schemas/spirv_executable_def_verifier.h"

using namespace iree::hal::vulkan;

typedef struct iree_hal_vulkan_entry_point_t {
  VkPipeline pipeline;
  iree_string_view_t name;

  // Optional debug information.
  IREE_TRACE(iree_string_view_t source_filename;)
  IREE_TRACE(uint32_t source_line;)
} iree_hal_vulkan_entry_point_t;

static iree_status_t iree_hal_vulkan_create_shader_module(
    VkDeviceHandle* logical_device, iree_const_byte_span_t code,
    VkShaderModule* out_shader_module) {
  IREE_TRACE_SCOPE();

  VkShaderModuleCreateInfo create_info;
  create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  create_info.pNext = NULL;
  create_info.flags = 0;
  create_info.codeSize = code.data_length;
  create_info.pCode = (const uint32_t*)code.data;
  VK_RETURN_IF_ERROR(logical_device->syms()->vkCreateShaderModule(
                         *logical_device, &create_info,
                         logical_device->allocator(), out_shader_module),
                     "vkCreateShaderModule");

  return iree_ok_status();
}

static void iree_hal_vulkan_destroy_shader_module(
    VkDeviceHandle* logical_device, VkShaderModule handle) {
  if (handle == VK_NULL_HANDLE) return;
  logical_device->syms()->vkDestroyShaderModule(*logical_device, handle,
                                                logical_device->allocator());
}

static iree_status_t iree_hal_vulkan_create_pipelines(
    VkDeviceHandle* logical_device, VkPipelineCache pipeline_cache,
    const iree_hal_executable_params_t* executable_params,
    iree_hal_spirv_ExecutableDef_table_t executable_def,
    VkShaderModule shader_module, iree_host_size_t pipeline_count,
    iree_hal_vulkan_entry_point_t* out_entry_points) {
  IREE_TRACE_SCOPE();
  uint8_t* scratch_memory = NULL;
  size_t create_info_size =
      pipeline_count * sizeof(VkComputePipelineCreateInfo);
  size_t spec_map_size =
      executable_params->constant_count * sizeof(VkSpecializationMapEntry);
  size_t subgroup_control_size =
      pipeline_count *
      sizeof(VkPipelineShaderStageRequiredSubgroupSizeCreateInfo);
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      logical_device->host_allocator(),
      create_info_size + spec_map_size + subgroup_control_size,
      (void**)&scratch_memory));
  VkComputePipelineCreateInfo* create_infos =
      (VkComputePipelineCreateInfo*)scratch_memory;
  VkSpecializationMapEntry* spec_map_entries =
      (VkSpecializationMapEntry*)(scratch_memory + create_info_size);
  VkPipelineShaderStageRequiredSubgroupSizeCreateInfo* subgroup_control_entries =
      (VkPipelineShaderStageRequiredSubgroupSizeCreateInfo*)(scratch_memory +
                                                             create_info_size +
                                                             spec_map_size);

  VkSpecializationInfo spec_info;
  memset(&spec_info, 0, sizeof(spec_info));
  spec_info.mapEntryCount = executable_params->constant_count;
  spec_info.pMapEntries = spec_map_entries;
  spec_info.dataSize = executable_params->constant_count * sizeof(uint32_t);
  spec_info.pData = executable_params->constants;
  for (iree_host_size_t i = 0; i < executable_params->constant_count; ++i) {
    spec_map_entries[i].constantID = i;
    spec_map_entries[i].offset = i * sizeof(uint32_t);
    spec_map_entries[i].size = sizeof(uint32_t);
  }

  flatbuffers_string_vec_t entry_points_vec =
      iree_hal_spirv_ExecutableDef_entry_points_get(executable_def);
  flatbuffers_uint32_vec_t subgroup_sizes_vec =
      iree_hal_spirv_ExecutableDef_subgroup_sizes_get(executable_def);
  for (iree_host_size_t entry_ordinal = 0; entry_ordinal < pipeline_count;
       ++entry_ordinal) {
    VkComputePipelineCreateInfo* create_info = &create_infos[entry_ordinal];
    create_info->sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    create_info->pNext = NULL;
    create_info->flags = 0;
    if (!iree_all_bits_set(
            executable_params->caching_mode,
            IREE_HAL_EXECUTABLE_CACHING_MODE_ALLOW_OPTIMIZATION)) {
      create_info->flags |= VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT;
    }
    if (entry_ordinal == 0) {
      create_info->flags |= VK_PIPELINE_CREATE_ALLOW_DERIVATIVES_BIT;
    } else {
      create_info->flags |= VK_PIPELINE_CREATE_DERIVATIVE_BIT;
    }
    create_info->layout = iree_hal_vulkan_native_pipeline_layout_handle(
        executable_params->pipeline_layouts[entry_ordinal]);
    create_info->basePipelineHandle = VK_NULL_HANDLE;
    create_info->basePipelineIndex = 0;

    VkPipelineShaderStageCreateInfo* stage_create_info = &create_info->stage;
    stage_create_info->sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage_create_info->flags = 0;
    stage_create_info->stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage_create_info->module = shader_module;
    stage_create_info->pName =
        flatbuffers_string_vec_at(entry_points_vec, entry_ordinal);
    stage_create_info->pSpecializationInfo = &spec_info;

    // If subgroup size is not 0, request the said subgroup size via
    // VK_EXT_subgroup_size_control (promoted to core since v1.3).
    stage_create_info->pNext = NULL;
    if (subgroup_sizes_vec) {
      if (uint32_t subgroup_size =
              flatbuffers_uint32_vec_at(subgroup_sizes_vec, entry_ordinal)) {
        subgroup_control_entries[entry_ordinal].sType =
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO;
        subgroup_control_entries[entry_ordinal].pNext = NULL;
        subgroup_control_entries[entry_ordinal].requiredSubgroupSize =
            subgroup_size;
        stage_create_info->pNext = &subgroup_control_entries[entry_ordinal];
      }
    }
  }

  VkPipeline* pipelines =
      (VkPipeline*)iree_alloca(pipeline_count * sizeof(VkPipeline));
  iree_status_t status = VK_RESULT_TO_STATUS(
      logical_device->syms()->vkCreateComputePipelines(
          *logical_device, pipeline_cache, (uint32_t)pipeline_count,
          create_infos, logical_device->allocator(), pipelines),
      "vkCreateComputePipelines");
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < pipeline_count; ++i) {
      out_entry_points[i].pipeline = pipelines[i];

      // Set pipeline name for tooling.
      if (PFN_vkSetDebugUtilsObjectNameEXT set_name =
              logical_device->syms()->vkSetDebugUtilsObjectNameEXT) {
        VkDebugUtilsObjectNameInfoEXT name_info = {};
        name_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
        name_info.pNext = NULL;
        name_info.objectHandle = (uint64_t)pipelines[i];
        name_info.objectType = VK_OBJECT_TYPE_PIPELINE;
        name_info.pObjectName = flatbuffers_string_vec_at(entry_points_vec, i);
        set_name(*logical_device, &name_info);
      }
    }
  }

  iree_allocator_free(logical_device->host_allocator(), scratch_memory);
  return status;
}

static void iree_hal_vulkan_destroy_pipeline(VkDeviceHandle* logical_device,
                                             VkPipeline handle) {
  IREE_TRACE_SCOPE();
  if (handle == VK_NULL_HANDLE) return;
  logical_device->syms()->vkDestroyPipeline(*logical_device, handle,
                                            logical_device->allocator());
}

// Verifies the structure of the FlatBuffer so that we can avoid doing so during
// runtime. There are still some conditions we must be aware of (such as omitted
// names on functions with internal linkage), however we shouldn't need to
// bounds check anything within the FlatBuffer after this succeeds.
static iree_status_t iree_hal_spirv_executable_flatbuffer_verify(
    iree_const_byte_span_t flatbuffer_data,
    iree_host_size_t expected_entry_point_count) {
  if (!flatbuffer_data.data || flatbuffer_data.data_length < 16) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "FlatBuffer data is not present or less than 16 bytes (%zu total)",
        flatbuffer_data.data_length);
  }

  // Run flatcc generated verification. This ensures all pointers are in-bounds
  // and that we can safely walk the file, but not that the actual contents of
  // the FlatBuffer meet our expectations.
  int verify_ret = iree_hal_spirv_ExecutableDef_verify_as_root(
      flatbuffer_data.data, flatbuffer_data.data_length);
  if (verify_ret != flatcc_verify_ok) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "FlatBuffer verification failed: %s",
                            flatcc_verify_error_string(verify_ret));
  }

  iree_hal_spirv_ExecutableDef_table_t executable_def =
      iree_hal_spirv_ExecutableDef_as_root(flatbuffer_data.data);

  flatbuffers_string_vec_t entry_points_vec =
      iree_hal_spirv_ExecutableDef_entry_points_get(executable_def);
  size_t entry_point_count = flatbuffers_string_vec_len(entry_points_vec);
  if (entry_point_count != expected_entry_point_count) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "executable provides %zu entry points but caller "
                            "provided %zu; must match",
                            entry_point_count, expected_entry_point_count);
  }

  for (size_t i = 0; i < entry_point_count; ++i) {
    if (!flatbuffers_string_len(
            flatbuffers_string_vec_at(entry_points_vec, i))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "executable entry point %zu has no name", i);
    }
  }

  flatbuffers_uint32_vec_t subgroup_sizes_vec =
      iree_hal_spirv_ExecutableDef_subgroup_sizes_get(executable_def);
  if (subgroup_sizes_vec) {
    size_t subgroup_sizes_count = flatbuffers_vec_len(subgroup_sizes_vec);
    if (subgroup_sizes_count != expected_entry_point_count) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "executable has %zu entry points but %zu subgroup sizes are defined",
          expected_entry_point_count, subgroup_sizes_count);
    }
  }

  if (flatbuffers_uint32_vec_len(
          iree_hal_spirv_ExecutableDef_code_get(executable_def)) == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "executable SPIR-V code is missing/empty");
  }

  return iree_ok_status();
}

typedef struct iree_hal_vulkan_native_executable_t {
  iree_hal_resource_t resource;
  VkDeviceHandle* logical_device;
  iree_host_size_t entry_point_count;
  iree_hal_vulkan_entry_point_t entry_points[];
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
  IREE_TRACE_ZONE_BEGIN(z0);

  // Verify and fetch the executable FlatBuffer wrapper.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_spirv_executable_flatbuffer_verify(
              executable_params->executable_data,
              executable_params->pipeline_layout_count));
  iree_hal_spirv_ExecutableDef_table_t executable_def =
      iree_hal_spirv_ExecutableDef_as_root(
          executable_params->executable_data.data);

  // Create the shader module.
  flatbuffers_uint32_vec_t code_vec =
      iree_hal_spirv_ExecutableDef_code_get(executable_def);
  VkShaderModule shader_module = VK_NULL_HANDLE;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_vulkan_create_shader_module(
              logical_device,
              iree_make_const_byte_span(
                  code_vec,
                  flatbuffers_uint32_vec_len(code_vec) * sizeof(uint32_t)),
              &shader_module));

  // Create pipelines for each entry point.
  flatbuffers_string_vec_t entry_points_vec =
      iree_hal_spirv_ExecutableDef_entry_points_get(executable_def);
  iree_host_size_t entry_point_count =
      flatbuffers_string_vec_len(entry_points_vec);

  iree_hal_vulkan_native_executable_t* executable = NULL;
  iree_host_size_t total_size =
      sizeof(*executable) +
      entry_point_count * sizeof(*executable->entry_points);
  iree_status_t status = iree_allocator_malloc(logical_device->host_allocator(),
                                               total_size, (void**)&executable);
  if (iree_status_is_ok(status)) {
    iree_hal_resource_initialize(&iree_hal_vulkan_native_executable_vtable,
                                 &executable->resource);
    executable->logical_device = logical_device;
    executable->entry_point_count = entry_point_count;
    memset(executable->entry_points, 0,
           entry_point_count * sizeof(*executable->entry_points));
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_create_pipelines(
        logical_device, pipeline_cache, executable_params, executable_def,
        shader_module, executable->entry_point_count, executable->entry_points);
  }
  iree_hal_vulkan_destroy_shader_module(logical_device, shader_module);

  if (iree_status_is_ok(status)) {
    flatbuffers_string_vec_t entry_points_vec =
        iree_hal_spirv_ExecutableDef_entry_points_get(executable_def);
    for (iree_host_size_t i = 0; i < entry_point_count; ++i) {
      flatbuffers_string_t name =
          flatbuffers_string_vec_at(entry_points_vec, i);
      executable->entry_points[i].name =
          iree_make_string_view(name, flatbuffers_string_len(name));
      IREE_TRACE_ZONE_APPEND_TEXT(z0, name);
    }
  }

  IREE_TRACE({
    if (iree_status_is_ok(status) &&
        iree_hal_spirv_ExecutableDef_source_locations_is_present(
            executable_def)) {
      iree_hal_spirv_FileLineLocDef_vec_t source_locs_vec =
          iree_hal_spirv_ExecutableDef_source_locations_get(executable_def);
      for (iree_host_size_t i = 0; i < entry_point_count; ++i) {
        iree_hal_spirv_FileLineLocDef_table_t source_loc =
            iree_hal_spirv_FileLineLocDef_vec_at(source_locs_vec, i);
        flatbuffers_string_t filename =
            iree_hal_spirv_FileLineLocDef_filename_get(source_loc);
        uint32_t line = iree_hal_spirv_FileLineLocDef_line_get(source_loc);
        executable->entry_points[i].source_filename =
            iree_make_string_view(filename, flatbuffers_string_len(filename));
        executable->entry_points[i].source_line = line;
      }
    }
  });

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

  for (iree_host_size_t i = 0; i < executable->entry_point_count; ++i) {
    iree_hal_vulkan_destroy_pipeline(executable->logical_device,
                                     executable->entry_points[i].pipeline);
  }
  iree_allocator_free(host_allocator, executable);

  IREE_TRACE_ZONE_END(z0);
}

void iree_hal_vulkan_native_executable_entry_point_source_location(
    iree_hal_executable_t* base_executable, iree_host_size_t entry_ordinal,
    iree_hal_vulkan_source_location_t* out_source_location) {
  iree_hal_vulkan_native_executable_t* executable =
      iree_hal_vulkan_native_executable_cast(base_executable);
  memset(out_source_location, 0, sizeof(*out_source_location));
  if (entry_ordinal >= executable->entry_point_count) {
    return;
  }
  iree_hal_vulkan_entry_point_t entry_point =
      executable->entry_points[entry_ordinal];

  out_source_location->func_name = entry_point.name;

  out_source_location->file_name = out_source_location->func_name;
  out_source_location->line = 0;
  IREE_TRACE({
    out_source_location->file_name = entry_point.source_filename;
    out_source_location->line = entry_point.source_line;
  });
}

iree_status_t iree_hal_vulkan_native_executable_pipeline_for_entry_point(
    iree_hal_executable_t* base_executable, iree_host_size_t entry_ordinal,
    VkPipeline* out_pipeline_handle) {
  iree_hal_vulkan_native_executable_t* executable =
      iree_hal_vulkan_native_executable_cast(base_executable);
  if (entry_ordinal >= executable->entry_point_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "invalid entry point ordinal %zu", entry_ordinal);
  }
  *out_pipeline_handle = executable->entry_points[entry_ordinal].pipeline;
  return iree_ok_status();
}

namespace {
const iree_hal_executable_vtable_t iree_hal_vulkan_native_executable_vtable = {
    /*.destroy=*/iree_hal_vulkan_native_executable_destroy,
};
}  // namespace
