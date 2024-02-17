// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/rocm/native_executable.h"

#include <stddef.h>

#include "experimental/rocm/dynamic_symbols.h"
#include "experimental/rocm/pipeline_layout.h"
#include "experimental/rocm/status_util.h"
#include "iree/base/api.h"

// flatcc schemas:
#include "iree/base/internal/flatcc/parsing.h"
#include "iree/schemas/rocm_executable_def_reader.h"
#include "iree/schemas/rocm_executable_def_verifier.h"

typedef struct iree_hal_rocm_native_executable_t {
  iree_hal_resource_t resource;
  iree_hal_rocm_context_wrapper_t* context;
  iree_hal_pipeline_layout_t** pipeline_layouts;
  iree_host_size_t entry_count;
  hipModule_t module;
  iree_host_size_t entry_point_count;
  iree_hal_rocm_kernel_params_t entry_points[];
} iree_hal_rocm_native_executable_t;

static const iree_hal_executable_vtable_t
    iree_hal_rocm_native_executable_vtable;

static iree_hal_rocm_native_executable_t* iree_hal_rocm_native_executable_cast(
    iree_hal_executable_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_rocm_native_executable_vtable);
  return (iree_hal_rocm_native_executable_t*)base_value;
}

iree_status_t iree_hal_rocm_native_executable_create(
    iree_hal_rocm_context_wrapper_t* context,
    const iree_hal_executable_params_t* executable_params,
    iree_hal_executable_t** out_executable) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(executable_params);
  IREE_ASSERT_ARGUMENT(out_executable);
  *out_executable = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_rocm_native_executable_t* executable = NULL;

  // TODO: Verify the flat buffer.
  iree_hal_rocm_ExecutableDef_table_t executable_def =
      iree_hal_rocm_ExecutableDef_as_root(
          executable_params->executable_data.data);

  // Create the kernel module.
  flatbuffers_string_t hsaco_image =
      iree_hal_rocm_ExecutableDef_hsaco_image_get(executable_def);
  flatbuffers_string_vec_t entry_points_vec =
      iree_hal_rocm_ExecutableDef_entry_points_get(executable_def);
  iree_hal_rocm_BlockSizeDef_vec_t block_sizes_vec =
      iree_hal_rocm_ExecutableDef_block_sizes_get(executable_def);
  flatbuffers_uint32_vec_t shared_memory_sizes =
      iree_hal_rocm_ExecutableDef_shared_memory_sizes_get(executable_def);
  iree_host_size_t entry_count = flatbuffers_string_vec_len(entry_points_vec);

  // Calculate the total number of characters across all entry point names. This
  // is only required when tracing so that we can store copies of the names as
  // the flatbuffer storing the strings may be released while the executable is
  // still live.
  iree_host_size_t total_entry_point_name_chars = 0;
  IREE_TRACE({
    for (iree_host_size_t i = 0; i < entry_count; i++) {
      const char* entry_name = flatbuffers_string_vec_at(entry_points_vec, i);
      total_entry_point_name_chars += flatbuffers_string_len(entry_name);
    }
  });

  iree_host_size_t total_size =
      sizeof(*executable) + entry_count * sizeof(executable->entry_points[0]) +
      total_entry_point_name_chars;
  iree_status_t status = iree_allocator_malloc(context->host_allocator,
                                               total_size, (void**)&executable);
  IREE_TRACE(char* string_table_buffer =
                 (char*)((char*)executable + sizeof(*executable) +
                         entry_count * sizeof(executable->entry_points[0])));
  executable->context = context;
  if (iree_status_is_ok(status)) {
    iree_hal_resource_initialize(&iree_hal_rocm_native_executable_vtable,
                                 &executable->resource);
    executable->context = context;
    status = ROCM_RESULT_TO_STATUS(
        context->syms,
        hipModuleLoadDataEx(&executable->module, hsaco_image, 0, NULL, NULL),
        "hipModuleLoadDataEx");
  }

  // Query allowed max shared memory.
  int32_t max_shared_mem = 0;
  if (iree_status_is_ok(status)) {
    status = ROCM_RESULT_TO_STATUS(
        context->syms,
        hipDeviceGetAttribute(&max_shared_mem,
                              hipDeviceAttributeMaxSharedMemoryPerBlock,
                              context->rocm_device),
        "hipDeviceGetAttribute");
  }

  if (iree_status_is_ok(status)) {
    executable->entry_count = entry_count;
    for (iree_host_size_t i = 0; i < entry_count; i++) {
      if (iree_status_is_ok(status)) {
        hipFunction_t function = NULL;
        const char* entry_name = flatbuffers_string_vec_at(entry_points_vec, i);
        status = ROCM_RESULT_TO_STATUS(
            context->syms,
            hipModuleGetFunction(&function, executable->module, entry_name),
            "hipModuleGetFunction");
        if (!iree_status_is_ok(status)) break;
        if (!function) {
          status = iree_make_status(IREE_STATUS_NOT_FOUND,
                                    "exported module function %s not found",
                                    entry_name);
          break;
        }
        if (shared_memory_sizes[i] > max_shared_mem) {
          status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                    "function '%s' requested shared memory "
                                    "size of %d larger than allowed size of %d",
                                    entry_name, shared_memory_sizes[i],
                                    max_shared_mem);
        } else if (shared_memory_sizes[i] != 0) {
          status = ROCM_RESULT_TO_STATUS(
              context->syms,
              hipFuncSetAttribute(
                  function,
                  (hipFuncAttribute)
                      HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                  shared_memory_sizes[i]),
              "hipFuncSetAttribute");
        }
        // Package required parameters for kernel launches for each entry point.
        iree_hal_rocm_kernel_params_t* params = &executable->entry_points[i];
        params->layout = executable_params->pipeline_layouts[i];
        iree_hal_pipeline_layout_retain(params->layout);
        params->function = function;
        params->block_size[0] = block_sizes_vec[i].x;
        params->block_size[1] = block_sizes_vec[i].y;
        params->block_size[2] = block_sizes_vec[i].z;
        params->shared_memory_size = shared_memory_sizes[i];
        // Stash the entry point name in the string table for use when tracing.
        IREE_TRACE({
          iree_host_size_t entry_name_length =
              flatbuffers_string_len(entry_name);
          memcpy(string_table_buffer, entry_name, entry_name_length);
          params->function_name =
              iree_make_string_view(string_table_buffer, entry_name_length);
          string_table_buffer += entry_name_length;
        });
      }
    }
  }

  if (iree_status_is_ok(status)) {
    *out_executable = (iree_hal_executable_t*)executable;
  } else {
    if (executable) {
      iree_hal_executable_destroy((iree_hal_executable_t*)executable);
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

hipFunction_t iree_hal_rocm_native_executable_for_entry_point(
    iree_hal_executable_t* base_executable, int32_t entry_point) {
  iree_hal_rocm_native_executable_t* executable =
      iree_hal_rocm_native_executable_cast(base_executable);
  return executable->entry_points[entry_point].function;
}

static void iree_hal_rocm_native_executable_destroy(
    iree_hal_executable_t* base_executable) {
  iree_hal_rocm_native_executable_t* executable =
      iree_hal_rocm_native_executable_cast(base_executable);
  iree_allocator_t host_allocator = executable->context->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  if (executable->module) {
    iree_status_t status = ROCM_RESULT_TO_STATUS(
        executable->context->syms, hipModuleUnload(executable->module),
        "hipModuleUnload");
    if (!iree_status_is_ok(status)) {
      fprintf(stderr, "Failed unloading ROCm module: ");
      iree_status_fprint(stderr, status);
      iree_status_free(status);
    }
  }

  if (executable->pipeline_layouts) {
    for (iree_host_size_t i = 0; i < executable->entry_count; ++i) {
      if (executable->pipeline_layouts[i]) {
        iree_hal_pipeline_layout_release(executable->pipeline_layouts[i]);
      }
    }
  }

  iree_allocator_free(host_allocator, executable);

  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_hal_rocm_native_executable_entry_point_kernel_params(
    iree_hal_executable_t* base_executable, int32_t entry_point,
    iree_hal_rocm_kernel_params_t* out_params) {
  iree_hal_rocm_native_executable_t* executable =
      iree_hal_rocm_native_executable_cast(base_executable);
  if (entry_point >= executable->entry_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "invalid entry point ordinal %d", entry_point);
  }
  memcpy(out_params, &executable->entry_points[entry_point],
         sizeof(*out_params));
  return iree_ok_status();
}

static const iree_hal_executable_vtable_t
    iree_hal_rocm_native_executable_vtable = {
        .destroy = iree_hal_rocm_native_executable_destroy,
};
