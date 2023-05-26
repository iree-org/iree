// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/webgpu/executable.h"

#include <stddef.h>

#include "iree/base/api.h"
#include "iree/base/internal/inline_array.h"
#include "iree/base/tracing.h"

// flatcc schemas:
#include "iree/base/internal/flatcc/parsing.h"
#include "iree/schemas/wgsl_executable_def_reader.h"
#include "iree/schemas/wgsl_executable_def_verifier.h"

typedef struct iree_hal_webgpu_executable_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  iree_host_size_t entry_point_count;
  iree_hal_webgpu_entry_point_t entry_points[];
} iree_hal_webgpu_executable_t;

extern const iree_hal_executable_vtable_t iree_hal_webgpu_executable_vtable;

static iree_hal_webgpu_executable_t* iree_hal_webgpu_executable_cast(
    iree_hal_executable_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_webgpu_executable_vtable);
  return (iree_hal_webgpu_executable_t*)base_value;
}

// Verifies the structure of the flatbuffer.
static iree_status_t iree_hal_webgpu_executable_flatbuffer_verify(
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
  int verify_ret = iree_hal_wgsl_ExecutableDef_verify_as_root(
      flatbuffer_data.data, flatbuffer_data.data_length);
  if (verify_ret != flatcc_verify_ok) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "flatbuffer verification failed: %s",
                            flatcc_verify_error_string(verify_ret));
  }

  iree_hal_wgsl_ExecutableDef_table_t executable_def =
      iree_hal_wgsl_ExecutableDef_as_root(flatbuffer_data.data);

  iree_hal_wgsl_ShaderModuleDef_vec_t shader_modules_vec =
      iree_hal_wgsl_ExecutableDef_shader_modules_get(executable_def);
  size_t shader_module_count =
      iree_hal_wgsl_ShaderModuleDef_vec_len(shader_modules_vec);
  for (size_t i = 0; i < shader_module_count; ++i) {
    iree_hal_wgsl_ShaderModuleDef_table_t shader_module_def =
        iree_hal_wgsl_ShaderModuleDef_vec_at(shader_modules_vec, i);
    if (flatbuffers_string_len(
            iree_hal_wgsl_ShaderModuleDef_code_get(shader_module_def)) == 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "shader module %zu WGSL code is missing/empty",
                              i);
    }
  }

  flatbuffers_uint32_vec_t entry_points_vec =
      iree_hal_wgsl_ExecutableDef_entry_points_get(executable_def);
  size_t entry_point_count = flatbuffers_uint32_vec_len(entry_points_vec);
  if (entry_point_count != expected_entry_point_count) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "executable provides %zu entry points but caller "
                            "provided %zu; must match",
                            entry_point_count, expected_entry_point_count);
  }

  for (size_t i = 0; i < entry_point_count; ++i) {
    uint32_t module_ordinal = flatbuffers_uint32_vec_at(entry_points_vec, i);
    if (module_ordinal >= shader_module_count) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "executable entry point %zu references an invalid shader module %d",
          i, module_ordinal);
    }
  }

  return iree_ok_status();
}

static iree_status_t iree_hal_webgpu_create_wgsl_shader_module(
    WGPUDevice device, iree_hal_wgsl_ShaderModuleDef_table_t shader_module_def,
    WGPUShaderModule* out_shader_module) {
  IREE_ASSERT_ARGUMENT(shader_module_def);
  IREE_ASSERT_ARGUMENT(out_shader_module);
  *out_shader_module = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  const char* code = iree_hal_wgsl_ShaderModuleDef_code_get(shader_module_def);

  const WGPUShaderModuleWGSLDescriptor descriptor = {
    .chain =
        {
            .next = NULL,
            .sType = WGPUSType_ShaderModuleWGSLDescriptor,
        },
#if defined(IREE_PLATFORM_EMSCRIPTEN)
    // Emscripten uses this older name.
    .source = code,
#else
    // Spec uses this name: https://www.w3.org/TR/webgpu/#shader-module-creation
    .code = code,
#endif
  };
  const WGPUShaderModuleDescriptor module_descriptor = {
      .nextInChain = &descriptor.chain,
      .label = NULL,
  };
  *out_shader_module = wgpuDeviceCreateShaderModule(device, &module_descriptor);
  iree_status_t status = iree_ok_status();
  if (!*out_shader_module) {
    // TODO(benvanik): see if we can get more detailed error info.
    status = iree_make_status(IREE_STATUS_INTERNAL,
                              "wgpuDeviceCreateShaderModule failed");
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Enough room for `d` + max uint32 characters + NUL.
#define IREE_HAL_WEBGPU_MAX_ENTRY_NAME_LENGTH (1 + /*uint32*/ 10 + /*NUL*/ 1)

// Makes a canonical entry point name based on its entry ordinal.
// |buffer| must have at least
// Example: ordinal 3 => 'd3'
static void iree_hal_webgpu_make_entry_name(uint32_t entry_ordinal,
                                            char* buffer) {
  // Inlined base 10 unsigned itoa-like.
  // Generates the string in reverse and then flips it around.
  // It's not worth pulling in snprintf for this.
  buffer[0] = 'd';
  ++buffer;
  uint32_t n = entry_ordinal;
  int length = 0;
  do {
    buffer[length++] = '0' + (n % 10);
  } while ((n /= 10) > 0);
  buffer[length] = '\0';
  for (int i = 0, j = length - 1; i < j; ++i, --j) {
    char c = buffer[i];
    buffer[i] = buffer[j];
    buffer[j] = c;
  }
}

// TODO(benvanik): switch to async compilation using
// wgpuDeviceCreateComputePipelineAsync. We pack all pipelines into a single
// executable (usually) and can batch compilation of all of them and only
// join at the end. Technically we could extend the join point until first use
// but it's harder to reason about lifetime that way. Today we just compile
// them all synchronously.
static iree_status_t iree_hal_webgpu_create_pipeline(
    WGPUDevice device, WGPUShaderModule shader_module, uint32_t entry_ordinal,
    iree_hal_pipeline_layout_t* pipeline_layout,
    iree_hal_webgpu_entry_point_t* out_entry_point) {
  IREE_ASSERT_ARGUMENT(shader_module);
  IREE_ASSERT_ARGUMENT(pipeline_layout);
  IREE_ASSERT_ARGUMENT(out_entry_point);
  IREE_TRACE_ZONE_BEGIN(z0);

  char entry_name[IREE_HAL_WEBGPU_MAX_ENTRY_NAME_LENGTH] = {0};
  iree_hal_webgpu_make_entry_name(entry_ordinal, entry_name);

  const WGPUComputePipelineDescriptor pipeline_descriptor = {
      .nextInChain = NULL,
      .label = WGPU_DEBUG_LABEL(entry_name),
      .layout = iree_hal_webgpu_pipeline_layout_handle(pipeline_layout),
      .compute =
          {
              .nextInChain = NULL,
              .module = shader_module,
              .entryPoint = entry_name,
          },
  };

  WGPUComputePipeline pipeline =
      wgpuDeviceCreateComputePipeline(device, &pipeline_descriptor);
  iree_status_t status = iree_ok_status();
  if (!pipeline) {
    status = iree_make_status(IREE_STATUS_INTERNAL,
                              "wgpuDeviceCreateComputePipeline "
                              "failed for entry point '%s'",
                              entry_name);
  }

  if (iree_status_is_ok(status)) {
    out_entry_point->pipeline = pipeline;
    out_entry_point->layout = pipeline_layout;
    iree_hal_pipeline_layout_retain(pipeline_layout);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_webgpu_executable_create(
    WGPUDevice device, const iree_hal_executable_params_t* executable_params,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(executable_params);
  IREE_ASSERT_ARGUMENT(out_executable);
  *out_executable = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Verify support up-front - the code below assumes
  if (!iree_string_view_equal(executable_params->executable_format,
                              iree_make_cstring_view("webgpu-wgsl-fb"))) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "executable format '%.*s' not available in this build",
        (int)executable_params->executable_format.size,
        executable_params->executable_format.data);
  }

  // Verify and fetch the executable flatbuffer wrapper.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_webgpu_executable_flatbuffer_verify(
              executable_params->executable_data,
              executable_params->pipeline_layout_count));
  iree_hal_wgsl_ExecutableDef_table_t executable_def =
      iree_hal_wgsl_ExecutableDef_as_root(
          executable_params->executable_data.data);

  // Create shader modules. This will be cheap on some implementations like
  // Metal that need pipeline information in order to be JIT'ed from WGSL while
  // on others it can be more expensive.
  iree_hal_wgsl_ShaderModuleDef_vec_t shader_modules_vec =
      iree_hal_wgsl_ExecutableDef_shader_modules_get(executable_def);
  size_t shader_module_count =
      iree_hal_wgsl_ShaderModuleDef_vec_len(shader_modules_vec);
  iree_inline_array(WGPUShaderModule, shader_modules, shader_module_count,
                    host_allocator);
  memset(iree_inline_array_data(shader_modules), 0,
         sizeof(WGPUShaderModule) * shader_module_count);
  iree_status_t status = iree_ok_status();
  for (size_t i = 0; i < shader_module_count; ++i) {
    status = iree_hal_webgpu_create_wgsl_shader_module(
        device, iree_hal_wgsl_ShaderModuleDef_vec_at(shader_modules_vec, i),
        iree_inline_array_at(shader_modules, i));
    if (!iree_status_is_ok(status)) break;
  }

  // Allocate the executable with storage for the pipeline handles.
  iree_hal_webgpu_executable_t* executable = NULL;
  if (iree_status_is_ok(status)) {
    iree_host_size_t total_size =
        sizeof(*executable) + executable_params->pipeline_layout_count *
                                  sizeof(iree_hal_webgpu_entry_point_t);
    status =
        iree_allocator_malloc(host_allocator, total_size, (void**)&executable);
  }

  if (iree_status_is_ok(status)) {
    iree_hal_resource_initialize(&iree_hal_webgpu_executable_vtable,
                                 &executable->resource);
    executable->host_allocator = host_allocator;
    executable->entry_point_count = executable_params->pipeline_layout_count;

    // Create one pipeline per entry point.
    flatbuffers_uint32_vec_t entry_points_vec =
        iree_hal_wgsl_ExecutableDef_entry_points_get(executable_def);
    for (iree_host_size_t i = 0; i < executable->entry_point_count; i++) {
      uint32_t module_ordinal = flatbuffers_uint32_vec_at(entry_points_vec, i);
      status = iree_hal_webgpu_create_pipeline(
          device, *iree_inline_array_at(shader_modules, module_ordinal), i,
          executable_params->pipeline_layouts[i], &executable->entry_points[i]);
      if (!iree_status_is_ok(status)) break;
    }
  }

  for (size_t i = 0; i < shader_module_count; ++i) {
    iree_wgpuShaderModuleDrop(*iree_inline_array_at(shader_modules, i));
  }
  iree_inline_array_deinitialize(shader_modules);

  if (iree_status_is_ok(status)) {
    *out_executable = (iree_hal_executable_t*)executable;
  } else {
    iree_hal_executable_destroy((iree_hal_executable_t*)executable);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_webgpu_executable_destroy(
    iree_hal_executable_t* base_executable) {
  iree_hal_webgpu_executable_t* executable =
      iree_hal_webgpu_executable_cast(base_executable);
  iree_allocator_t host_allocator = executable->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  for (iree_host_size_t i = 0; i < executable->entry_point_count; i++) {
    iree_hal_webgpu_entry_point_t* entry_point = &executable->entry_points[i];
    iree_hal_pipeline_layout_release(entry_point->layout);
    iree_wgpuComputePipelineDrop(entry_point->pipeline);
  }
  iree_allocator_free(host_allocator, executable);

  IREE_TRACE_ZONE_END(z0);
}

const iree_hal_webgpu_entry_point_t*
iree_hal_webgpu_executable_lookup_entry_point(
    iree_hal_executable_t* base_executable, uint32_t ordinal) {
  iree_hal_webgpu_executable_t* executable =
      iree_hal_webgpu_executable_cast(base_executable);
  IREE_ASSERT_LT(ordinal, executable->entry_point_count);
  return &executable->entry_points[ordinal];
}

const iree_hal_executable_vtable_t iree_hal_webgpu_executable_vtable = {
    .destroy = iree_hal_webgpu_executable_destroy,
};
