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

  iree_host_size_t total_size =
      sizeof(*executable) + entry_count * sizeof(executable->entry_points[0]);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(context->host_allocator, total_size,
                                (void**)&executable));

  iree_hal_resource_initialize(&iree_hal_rocm_native_executable_vtable,
                               &executable->resource);

  executable->context = context;
  executable->entry_point_count = entry_count;
  iree_status_t status = ROCM_RESULT_TO_STATUS(
      context->syms,
      hipModuleLoadDataEx(&executable->module, hsaco_image, 0, NULL, NULL),
      "hipModuleLoadDataEx");
  if (!iree_status_is_ok(status)) {
    status = iree_status_annotate(
        status,
        IREE_SV("mismatched target chip? missing/wrong bitcode directory?"));
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
        flatbuffers_string_t entry_name =
            flatbuffers_string_vec_at(entry_points_vec, i);
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
        params->name = iree_make_string_view(
            entry_name, flatbuffers_string_len(entry_name));
        params->block_size[0] = block_sizes_vec[i].x;
        params->block_size[1] = block_sizes_vec[i].y;
        params->block_size[2] = block_sizes_vec[i].z;
        params->shared_memory_size = shared_memory_sizes[i];
      }
    }

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION
    if (iree_status_is_ok(status)) {
      if (iree_hal_rocm_ExecutableDef_source_locations_is_present(
              executable_def)) {
        iree_hal_rocm_FileLineLocDef_vec_t source_locations_vec =
            iree_hal_rocm_ExecutableDef_source_locations_get(executable_def);
        for (iree_host_size_t i = 0; i < entry_count; ++i) {
          executable->entry_points[i].source_location =
              iree_hal_rocm_FileLineLocDef_vec_at(source_locations_vec, i);
        }
      }
      if (iree_hal_rocm_ExecutableDef_stage_locations_is_present(
              executable_def)) {
        iree_hal_rocm_StageLocationsDef_vec_t stage_locations_vec =
            iree_hal_rocm_ExecutableDef_stage_locations_get(executable_def);
        for (iree_host_size_t i = 0; i < entry_count; ++i) {
          iree_hal_rocm_StageLocationsDef_table_t stage_locations =
              iree_hal_rocm_StageLocationsDef_vec_at(stage_locations_vec, i);
          executable->entry_points[i].stage_locations =
              iree_hal_rocm_StageLocationsDef_locations_get(stage_locations);
        }
      }

      // Publish any embedded source files to the tracing infrastructure.
      if (iree_hal_rocm_ExecutableDef_source_files_is_present(executable_def)) {
        iree_hal_rocm_SourceFileDef_vec_t source_files_vec =
            iree_hal_rocm_ExecutableDef_source_files_get(executable_def);
        for (iree_host_size_t i = 0;
             i < iree_hal_rocm_SourceFileDef_vec_len(source_files_vec); ++i) {
          iree_hal_rocm_SourceFileDef_table_t source_file =
              iree_hal_rocm_SourceFileDef_vec_at(source_files_vec, i);
          flatbuffers_string_t path =
              iree_hal_rocm_SourceFileDef_path_get(source_file);
          flatbuffers_uint8_vec_t content =
              iree_hal_rocm_SourceFileDef_content_get(source_file);
          IREE_TRACE_PUBLISH_SOURCE_FILE(path, flatbuffers_string_len(path),
                                         content,
                                         flatbuffers_uint8_vec_len(content));
        }
      }
    }
#endif  // IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION
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

void iree_hal_rocm_native_executable_entry_point_source_location(
    iree_hal_executable_t* base_executable, iree_host_size_t entry_ordinal,
    iree_hal_rocm_source_location_t* out_source_location) {
  iree_hal_rocm_native_executable_t* executable =
      iree_hal_rocm_native_executable_cast(base_executable);
  memset(out_source_location, 0, sizeof(*out_source_location));
  if (entry_ordinal >= executable->entry_point_count) {
    return;
  }
  const iree_hal_rocm_kernel_params_t* entry_point =
      &executable->entry_points[entry_ordinal];

  out_source_location->func_name = entry_point->name;

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION
  iree_hal_rocm_FileLineLocDef_table_t source_location =
      entry_point->source_location;
  if (entry_point->stage_locations) {
    for (size_t i = 0; i < iree_hal_rocm_StageLocationDef_vec_len(
                               entry_point->stage_locations);
         ++i) {
      iree_hal_rocm_StageLocationDef_table_t stage_location =
          iree_hal_rocm_StageLocationDef_vec_at(entry_point->stage_locations,
                                                i);
      // TODO(benvanik): a way to select what location is chosen. For now we
      // just pick the first one.
      source_location =
          iree_hal_rocm_StageLocationDef_location_get(stage_location);
      break;
    }
  }
  if (source_location) {
    flatbuffers_string_t filename =
        iree_hal_rocm_FileLineLocDef_filename_get(source_location);
    out_source_location->file_name =
        iree_make_string_view(filename, flatbuffers_string_len(filename));
    out_source_location->line =
        iree_hal_rocm_FileLineLocDef_line_get(source_location);
  } else {
    out_source_location->file_name = out_source_location->func_name;
    out_source_location->line = 0;
  }
#else
  out_source_location->file_name = out_source_location->func_name;
  out_source_location->line = 0;
#endif  // IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION
}

static const iree_hal_executable_vtable_t
    iree_hal_rocm_native_executable_vtable = {
        .destroy = iree_hal_rocm_native_executable_destroy,
};
