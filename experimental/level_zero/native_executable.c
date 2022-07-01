// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/level_zero/native_executable.h"

#include <stddef.h>

#include "experimental/level_zero/dynamic_symbols.h"
#include "experimental/level_zero/pipeline_layout.h"
#include "experimental/level_zero/status_util.h"
#include "iree/base/api.h"
#include "iree/base/tracing.h"

// flatcc schemas:
#include "iree/base/internal/flatcc/parsing.h"
#include "iree/schemas/level_zero_executable_def_reader.h"
#include "iree/schemas/level_zero_executable_def_verifier.h"

typedef struct iree_hal_level_zero_native_executable_function_t {
  ze_kernel_handle_t level_zero_function;
  uint32_t block_size_x;
  uint32_t block_size_y;
  uint32_t block_size_z;
} iree_hal_level_zero_native_executable_function_t;

typedef struct iree_hal_level_zero_native_executable_t {
  iree_hal_resource_t resource;
  iree_hal_level_zero_context_wrapper_t* context;
  iree_hal_pipeline_layout_t** pipeline_layouts;
  iree_host_size_t entry_count;
  ze_module_handle_t module;
  iree_hal_level_zero_native_executable_function_t entry_functions[];
} iree_hal_level_zero_native_executable_t;

static const iree_hal_executable_vtable_t
    iree_hal_level_zero_native_executable_vtable;

static iree_hal_level_zero_native_executable_t*
iree_hal_level_zero_native_executable_cast(iree_hal_executable_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value,
                       &iree_hal_level_zero_native_executable_vtable);
  return (iree_hal_level_zero_native_executable_t*)base_value;
}

iree_status_t iree_hal_level_zero_native_executable_create(
    iree_hal_level_zero_context_wrapper_t* context,
    const iree_hal_executable_params_t* executable_params,
    ze_device_handle_t level_zero_device,
    iree_hal_executable_t** out_executable) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(executable_params);
  IREE_ASSERT_ARGUMENT(out_executable);
  *out_executable = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_level_zero_native_executable_t* executable = NULL;

  iree_LEVEL_ZEROExecutableDef_table_t executable_def =
      iree_LEVEL_ZEROExecutableDef_as_root(
          executable_params->executable_data.data);

  // Create the kernel module.
  flatbuffers_uint32_vec_t level_zero_image =
      iree_LEVEL_ZEROExecutableDef_level_zero_image_get(executable_def);
  flatbuffers_string_vec_t entry_points_vec =
      iree_LEVEL_ZEROExecutableDef_entry_points_get(executable_def);
  iree_LEVEL_ZEROBlockSizeDef_vec_t block_sizes_vec =
      iree_LEVEL_ZEROExecutableDef_block_sizes_get(executable_def);
  iree_host_size_t entry_count = flatbuffers_string_vec_len(entry_points_vec);
  iree_host_size_t total_size =
      sizeof(*executable) +
      entry_count * sizeof(iree_hal_level_zero_native_executable_function_t) +
      entry_count * sizeof(iree_hal_pipeline_layout_t*);
  iree_status_t status = iree_allocator_malloc(context->host_allocator,
                                               total_size, (void**)&executable);
  executable->pipeline_layouts =
      (void*)((char*)executable + sizeof(*executable) +
              entry_count *
                  sizeof(iree_hal_level_zero_native_executable_function_t));
  ze_module_handle_t module = NULL;
  ze_module_build_log_handle_t build_log;
  if (iree_status_is_ok(status)) {
    ze_module_desc_t module_desc = {};
    module_desc.format = ZE_MODULE_FORMAT_IL_SPIRV;
    iree_const_byte_span_t code = iree_make_const_byte_span(
        level_zero_image,
        flatbuffers_uint32_vec_len(level_zero_image) * sizeof(uint32_t));
    module_desc.pInputModule = (const uint8_t*)(code.data);
    module_desc.inputSize = code.data_length;
    module_desc.pBuildFlags = "";
    status = LEVEL_ZERO_RESULT_TO_STATUS(
        context->syms,
        zeModuleCreate(context->level_zero_context, level_zero_device,
                       &module_desc, &module, &build_log),
        "zeModuleCreate");
  }
  for (iree_host_size_t i = 0; i < entry_count; i++) {
    if (iree_status_is_ok(status)) {
      const char* entry_name = flatbuffers_string_vec_at(entry_points_vec, i);
      ze_kernel_handle_t function = NULL;
      ze_kernel_desc_t kernel_desc = {};
      // kernel_desc.pKernelName = "simple_mul_dispatch_0";
      kernel_desc.pKernelName = entry_name;
      LEVEL_ZERO_RETURN_IF_ERROR(
          context->syms, zeKernelCreate(module, &kernel_desc, &function),
          "zeKernelCreate");
      executable->entry_functions[i].level_zero_function = function;
      executable->entry_functions[i].block_size_x = block_sizes_vec[i].x;
      executable->entry_functions[i].block_size_y = block_sizes_vec[i].y;
      executable->entry_functions[i].block_size_z = block_sizes_vec[i].z;
      executable->pipeline_layouts[i] = executable_params->pipeline_layouts[i];
      iree_hal_pipeline_layout_retain(executable_params->pipeline_layouts[i]);
    }
  }
  if (iree_status_is_ok(status)) {
    iree_hal_resource_initialize(&iree_hal_level_zero_native_executable_vtable,
                                 &executable->resource);
    executable->module = module;
    executable->context = context;
    *out_executable = (iree_hal_executable_t*)executable;
  } else {
    // print log
    size_t szLog = 0;
    status = LEVEL_ZERO_RESULT_TO_STATUS(
        context->syms, zeModuleBuildLogGetString(build_log, &szLog, NULL),
        "zeModuleBuildLogGetString");
    char* stringLog = (char*)malloc(szLog);
    status = LEVEL_ZERO_RESULT_TO_STATUS(
        context->syms, zeModuleBuildLogGetString(build_log, &szLog, stringLog),
        "zeModuleBuildLogGetString");
    status = LEVEL_ZERO_RESULT_TO_STATUS(context->syms,
                                         zeModuleBuildLogDestroy(build_log),
                                         "zeModuleBuildLogDestroy");
    iree_hal_executable_destroy((iree_hal_executable_t*)executable);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

ze_kernel_handle_t iree_hal_level_zero_native_executable_for_entry_point(
    iree_hal_executable_t* base_executable, int32_t entry_point) {
  iree_hal_level_zero_native_executable_t* executable =
      iree_hal_level_zero_native_executable_cast(base_executable);
  return executable->entry_functions[entry_point].level_zero_function;
}

iree_status_t iree_hal_level_zero_native_executable_block_size(
    iree_hal_executable_t* base_executable, int32_t entry_point, uint32_t* x,
    uint32_t* y, uint32_t* z) {
  iree_hal_level_zero_native_executable_t* executable =
      iree_hal_level_zero_native_executable_cast(base_executable);
  *x = executable->entry_functions[entry_point].block_size_x;
  *y = executable->entry_functions[entry_point].block_size_y;
  *z = executable->entry_functions[entry_point].block_size_z;
  return iree_ok_status();
}

iree_hal_pipeline_layout_t* iree_hal_level_zero_executable_get_layout(
    iree_hal_executable_t* base_executable, int32_t entry_point) {
  iree_hal_level_zero_native_executable_t* executable =
      iree_hal_level_zero_native_executable_cast(base_executable);
  return executable->pipeline_layouts[entry_point];
}

static void iree_hal_level_zero_native_executable_destroy(
    iree_hal_executable_t* base_executable) {
  iree_hal_level_zero_native_executable_t* executable =
      iree_hal_level_zero_native_executable_cast(base_executable);
  iree_allocator_t host_allocator = executable->context->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  for (iree_host_size_t i = 0; i < executable->entry_count; ++i) {
    iree_hal_pipeline_layout_release(executable->pipeline_layouts[i]);
  }
  iree_allocator_free(host_allocator, executable);

  IREE_TRACE_ZONE_END(z0);
}

static const iree_hal_executable_vtable_t
    iree_hal_level_zero_native_executable_vtable = {
        .destroy = iree_hal_level_zero_native_executable_destroy,
};
