// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/webgpu/builtins.h"

#include "experimental/webgpu/shaders/builtin_shaders.h"
#include "iree/base/api.h"
#include "iree/base/tracing.h"

static const char* iree_hal_webgpu_builtins_find_code(const char* file_name) {
  const iree_file_toc_t* files = iree_hal_wgsl_builtin_shaders_create();
  for (size_t i = 0; i < iree_hal_wgsl_builtin_shaders_size(); ++i) {
    if (strcmp(file_name, files[i].name) == 0) {
      return files[i].data;
    }
  }
  IREE_ASSERT_TRUE(false, "builtin wgsl file not found");
  return NULL;
}

static iree_status_t iree_hal_webgpu_builtins_initialize_fill_buffer(
    WGPUDevice device, iree_hal_webgpu_staging_buffer_t* staging_buffer,
    iree_hal_webgpu_builtin_fill_buffer_t* out_fill_buffer) {
  const WGPUBindGroupLayoutEntry buffer_binding = {
      .nextInChain = NULL,
      .binding = 0,
      .visibility = WGPUShaderStage_Compute,
      .buffer =
          {
              .nextInChain = NULL,
              .type = WGPUBufferBindingType_Storage,
              .hasDynamicOffset = false,
              .minBindingSize = 0,  // variable
          },
  };

  const WGPUBindGroupLayoutDescriptor buffer_group_layout_descriptor = {
      .nextInChain = NULL,
      .label = WGPU_DEBUG_LABEL("_builtin_fill_buffer_buffer"),
      .entryCount = 1,
      .entries = &buffer_binding,
  };
  WGPUBindGroupLayout buffer_group_layout =
      wgpuDeviceCreateBindGroupLayout(device, &buffer_group_layout_descriptor);
  if (!buffer_group_layout) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "failed to create fill_buffer builtin bind group layout");
  }

  const WGPUBindGroupLayout group_layouts[] = {
      staging_buffer->bind_group_layout,
      buffer_group_layout,
  };
  const WGPUPipelineLayoutDescriptor pipeline_layout_descriptor = {
      .nextInChain = NULL,
      .label = WGPU_DEBUG_LABEL("_builtin_fill_buffer_layout"),
      .bindGroupLayoutCount = (uint32_t)IREE_ARRAYSIZE(group_layouts),
      .bindGroupLayouts = group_layouts,
  };
  WGPUPipelineLayout pipeline_layout =
      wgpuDeviceCreatePipelineLayout(device, &pipeline_layout_descriptor);
  iree_wgpuBindGroupLayoutDrop(buffer_group_layout);
  if (!pipeline_layout) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "failed to create fill_buffer builtin pipeline layout");
  }

  const char* code = iree_hal_webgpu_builtins_find_code("fill_buffer.wgsl");
  const WGPUShaderModuleWGSLDescriptor wgsl_descriptor = {
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
      .nextInChain = &wgsl_descriptor.chain,
      .label = WGPU_DEBUG_LABEL("_builtin_fill_buffer_wgsl"),
  };
  WGPUShaderModule module =
      wgpuDeviceCreateShaderModule(device, &module_descriptor);
  if (!module) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "failed to create fill_buffer builtin shader module");
  }

  const WGPUComputePipelineDescriptor pipeline_descriptor = {
      .nextInChain = NULL,
      .label = WGPU_DEBUG_LABEL("_builtin_fill_buffer"),
      .layout = pipeline_layout,
      .compute =
          {
              .nextInChain = NULL,
              .module = module,
              .entryPoint = "main",
          },
  };
  WGPUComputePipeline pipeline =
      wgpuDeviceCreateComputePipeline(device, &pipeline_descriptor);
  if (!pipeline) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "failed to create fill_buffer builtin pipeline");
  }
  out_fill_buffer->pipeline = pipeline;
  out_fill_buffer->buffer_group_layout = buffer_group_layout;
  return iree_ok_status();
}

iree_status_t iree_hal_webgpu_builtins_initialize(
    WGPUDevice device, iree_hal_webgpu_staging_buffer_t* staging_buffer,
    iree_hal_webgpu_builtins_t* out_builtins) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(staging_buffer);
  IREE_ASSERT_ARGUMENT(out_builtins);
  IREE_TRACE_ZONE_BEGIN(z0);
  memset(out_builtins, 0, sizeof(*out_builtins));

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_webgpu_builtins_initialize_fill_buffer(
              device, staging_buffer, &out_builtins->fill_buffer));

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

void iree_hal_webgpu_builtins_deinitialize(
    iree_hal_webgpu_builtins_t* builtins) {
  IREE_ASSERT_ARGUMENT(builtins);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_wgpuBindGroupLayoutDrop(builtins->fill_buffer.buffer_group_layout);
  iree_wgpuComputePipelineDrop(builtins->fill_buffer.pipeline);

  memset(builtins, 0, sizeof(*builtins));
  IREE_TRACE_ZONE_END(z0);
}
