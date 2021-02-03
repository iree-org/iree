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

#include "iree/hal/vulkan/native_executable.h"

#include "iree/base/tracing.h"
#include "iree/hal/vulkan/handle_util.h"
#include "iree/hal/vulkan/native_executable_layout.h"
#include "iree/hal/vulkan/status_util.h"

// flatcc schemas:
#include "iree/base/flatcc.h"
#include "iree/schemas/spirv_executable_def_reader.h"
#include "iree/schemas/spirv_executable_def_verifier.h"

using namespace iree::hal::vulkan;

static iree_status_t iree_hal_vulkan_create_shader_module(
    VkDeviceHandle* logical_device, iree_const_byte_span_t code,
    VkShaderModule* out_shader_module) {
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
    iree_hal_executable_caching_mode_t caching_mode,
    iree_SpirVExecutableDef_table_t executable_def,
    VkShaderModule shader_module, iree_host_size_t executable_layout_count,
    iree_hal_executable_layout_t* const* executable_layouts,
    iree_host_size_t pipeline_count, VkPipeline* out_pipelines) {
  VkComputePipelineCreateInfo* create_infos = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      logical_device->host_allocator(),
      pipeline_count * sizeof(VkComputePipelineCreateInfo),
      (void**)&create_infos));

  flatbuffers_string_vec_t entry_points_vec =
      iree_SpirVExecutableDef_entry_points_get(executable_def);
  for (iree_host_size_t entry_ordinal = 0; entry_ordinal < pipeline_count;
       ++entry_ordinal) {
    VkComputePipelineCreateInfo* create_info = &create_infos[entry_ordinal];
    create_info->sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    create_info->pNext = NULL;
    create_info->flags = 0;
    if (!iree_all_bits_set(
            caching_mode,
            IREE_HAL_EXECUTABLE_CACHING_MODE_ALLOW_OPTIMIZATION)) {
      create_info->flags |= VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT;
    }
    if (entry_ordinal == 0) {
      create_info->flags |= VK_PIPELINE_CREATE_ALLOW_DERIVATIVES_BIT;
    } else {
      create_info->flags |= VK_PIPELINE_CREATE_DERIVATIVE_BIT;
    }
    create_info->layout = iree_hal_vulkan_native_executable_layout_handle(
        executable_layouts[entry_ordinal]);
    create_info->basePipelineHandle = VK_NULL_HANDLE;
    create_info->basePipelineIndex = 0;
    VkPipelineShaderStageCreateInfo* stage_create_info = &create_info->stage;
    stage_create_info->sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage_create_info->pNext = NULL;
    stage_create_info->flags = 0;
    stage_create_info->stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage_create_info->module = shader_module;
    stage_create_info->pName =
        flatbuffers_string_vec_at(entry_points_vec, entry_ordinal);
    stage_create_info->pSpecializationInfo = NULL;
  }

  iree_status_t status = VK_RESULT_TO_STATUS(
      logical_device->syms()->vkCreateComputePipelines(
          *logical_device, pipeline_cache, (uint32_t)pipeline_count,
          create_infos, logical_device->allocator(), out_pipelines),
      "vkCreateComputePipelines");

  iree_allocator_free(logical_device->host_allocator(), create_infos);
  return status;
}

static void iree_hal_vulkan_destroy_pipeline(VkDeviceHandle* logical_device,
                                             VkPipeline handle) {
  if (handle == VK_NULL_HANDLE) return;
  logical_device->syms()->vkDestroyPipeline(*logical_device, handle,
                                            logical_device->allocator());
}

// Verifies the structure of the flatbuffer so that we can avoid doing so during
// runtime. There are still some conditions we must be aware of (such as omitted
// names on functions with internal linkage), however we shouldn't need to
// bounds check anything within the flatbuffer after this succeeds.
static iree_status_t iree_hal_spirv_executable_flatbuffer_verify(
    iree_const_byte_span_t flatbuffer_data,
    iree_host_size_t expected_entry_point_count) {
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

typedef struct {
  iree_hal_resource_t resource;
  VkDeviceHandle* logical_device;
  iree_host_size_t pipeline_count;
  VkPipeline pipelines[];
} iree_hal_vulkan_native_executable_t;

extern const iree_hal_executable_vtable_t
    iree_hal_vulkan_native_executable_vtable;

static iree_hal_vulkan_native_executable_t*
iree_hal_vulkan_native_executable_cast(iree_hal_executable_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_vulkan_native_executable_vtable);
  return (iree_hal_vulkan_native_executable_t*)base_value;
}

iree_status_t iree_hal_vulkan_native_executable_create(
    iree::hal::vulkan::VkDeviceHandle* logical_device,
    VkPipelineCache pipeline_cache,
    const iree_hal_executable_spec_t* executable_spec,
    iree_hal_executable_t** out_executable) {
  IREE_ASSERT_ARGUMENT(logical_device);
  IREE_ASSERT_ARGUMENT(executable_spec);
  IREE_ASSERT_ARGUMENT(out_executable);
  *out_executable = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Verify and fetch the executable flatbuffer wrapper.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_spirv_executable_flatbuffer_verify(
              executable_spec->executable_data,
              executable_spec->executable_layout_count));
  iree_SpirVExecutableDef_table_t executable_def =
      iree_SpirVExecutableDef_as_root(executable_spec->executable_data.data);

  // Create the shader module.
  flatbuffers_uint32_vec_t code_vec =
      iree_SpirVExecutableDef_code_get(executable_def);
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
      iree_SpirVExecutableDef_entry_points_get(executable_def);
  iree_host_size_t pipeline_count =
      flatbuffers_string_vec_len(entry_points_vec);

  iree_hal_vulkan_native_executable_t* executable = NULL;
  iree_host_size_t total_size =
      sizeof(*executable) + pipeline_count * sizeof(*executable->pipelines);
  iree_status_t status = iree_allocator_malloc(logical_device->host_allocator(),
                                               total_size, (void**)&executable);
  if (iree_status_is_ok(status)) {
    iree_hal_resource_initialize(&iree_hal_vulkan_native_executable_vtable,
                                 &executable->resource);
    executable->logical_device = logical_device;
    executable->pipeline_count = pipeline_count;
    memset(executable->pipelines, 0,
           pipeline_count * sizeof(*executable->pipelines));
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_create_pipelines(
        logical_device, pipeline_cache, executable_spec->caching_mode,
        executable_def, shader_module, executable_spec->executable_layout_count,
        executable_spec->executable_layouts, executable->pipeline_count,
        executable->pipelines);
  }
  iree_hal_vulkan_destroy_shader_module(logical_device, shader_module);

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
                                     executable->pipelines[i]);
  }
  iree_allocator_free(host_allocator, executable);

  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_hal_vulkan_native_executable_pipeline_for_entry_point(
    iree_hal_executable_t* base_executable, iree_host_size_t entry_ordinal,
    VkPipeline* out_pipeline_handle) {
  iree_hal_vulkan_native_executable_t* executable =
      iree_hal_vulkan_native_executable_cast(base_executable);
  if (entry_ordinal >= executable->pipeline_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "invalid entry point ordinal %zu", entry_ordinal);
  }
  *out_pipeline_handle = executable->pipelines[entry_ordinal];
  return iree_ok_status();
}

const iree_hal_executable_vtable_t iree_hal_vulkan_native_executable_vtable = {
    /*.destroy=*/iree_hal_vulkan_native_executable_destroy,
};
