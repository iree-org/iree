// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/cuda2/native_executable.h"

#include <stddef.h>

#include "experimental/cuda2/cuda_dynamic_symbols.h"
#include "experimental/cuda2/cuda_status_util.h"
#include "experimental/cuda2/pipeline_layout.h"
#include "iree/base/api.h"

// flatcc schemas:
#include "iree/base/internal/flatcc/parsing.h"
#include "iree/schemas/cuda_executable_def_reader.h"
#include "iree/schemas/cuda_executable_def_verifier.h"

typedef struct iree_hal_cuda2_native_executable_t {
  // Abstract resource used for injecting reference counting and vtable;
  // must be at offset 0.
  iree_hal_resource_t resource;

  iree_allocator_t host_allocator;

  const iree_hal_cuda2_dynamic_symbols_t* symbols;

  // The loaded CUDA module.
  CUmodule cu_module;

  iree_host_size_t entry_point_count;
  // The list of entry point data pointers, pointing to trailing inline
  // allocation after the end of this struct.
  iree_hal_cuda2_kernel_info_t entry_points[];
} iree_hal_cuda2_native_executable_t;
// + Additional inline allocation for holding entry point information.

static const iree_hal_executable_vtable_t
    iree_hal_cuda2_native_executable_vtable;

static iree_hal_cuda2_native_executable_t*
iree_hal_cuda2_native_executable_cast(iree_hal_executable_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_cuda2_native_executable_vtable);
  return (iree_hal_cuda2_native_executable_t*)base_value;
}

// Verifies the structure of the flatbuffer so that we can avoid doing so during
// runtime.
//
// There are still some conditions we must be aware of (such as omitted names on
// functions with internal linkage), however we shouldn't need to bounds check
// anything within the flatbuffer after this succeeds.
static iree_status_t iree_hal_cuda2_native_executable_flatbuffer_verify(
    iree_const_byte_span_t flatbuffer_data) {
  if (!flatbuffer_data.data) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "flatbuffer data is not present");
  }

  // Run flatcc generated verification. This ensures all pointers are in-bounds
  // and that we can safely walk the file, but not that the actual contents of
  // the flatbuffer meet our expectations.
  int verify_ret = iree_hal_cuda_ExecutableDef_verify_as_root(
      flatbuffer_data.data, flatbuffer_data.data_length);
  if (verify_ret != flatcc_verify_ok) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "flatbuffer verification failed: %s",
                            flatcc_verify_error_string(verify_ret));
  }

  iree_hal_cuda_ExecutableDef_table_t executable_def =
      iree_hal_cuda_ExecutableDef_as_root(flatbuffer_data.data);

  flatbuffers_string_vec_t entry_points_vec =
      iree_hal_cuda_ExecutableDef_entry_points_get(executable_def);
  size_t entry_point_count = flatbuffers_string_vec_len(entry_points_vec);
  for (size_t i = 0; i < entry_point_count; ++i) {
    if (flatbuffers_string_len(
            flatbuffers_string_vec_at(entry_points_vec, i)) == 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "executable entry point %zu has no name", i);
    }
  }

  iree_hal_cuda_BlockSizeDef_vec_t block_sizes_vec =
      iree_hal_cuda_ExecutableDef_block_sizes_get(executable_def);
  size_t block_size_count = iree_hal_cuda_BlockSizeDef_vec_len(block_sizes_vec);
  if (block_size_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "no block sizes present");
  }

  if (entry_point_count != block_size_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "entry points (%zu) and block sizes (%zu) count mismatch",
        entry_point_count, block_size_count);
  }

  flatbuffers_string_t ptx_image =
      iree_hal_cuda_ExecutableDef_ptx_image_get(executable_def);
  if (flatbuffers_string_len(ptx_image) == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "no PTX image present");
  }

  return iree_ok_status();
}

iree_status_t iree_hal_cuda2_native_executable_create(
    const iree_hal_cuda2_dynamic_symbols_t* symbols, CUdevice device,
    const iree_hal_executable_params_t* executable_params,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable) {
  IREE_ASSERT_ARGUMENT(executable_params);
  IREE_ASSERT_ARGUMENT(out_executable);
  IREE_TRACE_ZONE_BEGIN(z0);

  *out_executable = NULL;
  iree_hal_cuda2_native_executable_t* executable = NULL;

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_cuda2_native_executable_flatbuffer_verify(
              executable_params->executable_data));

  iree_hal_cuda_ExecutableDef_table_t executable_def =
      iree_hal_cuda_ExecutableDef_as_root(
          executable_params->executable_data.data);

  flatbuffers_string_t ptx_image =
      iree_hal_cuda_ExecutableDef_ptx_image_get(executable_def);
  flatbuffers_uint32_vec_t shared_memory_sizes =
      iree_hal_cuda_ExecutableDef_shared_memory_size_get(executable_def);
  flatbuffers_string_vec_t entry_points_vec =
      iree_hal_cuda_ExecutableDef_entry_points_get(executable_def);
  iree_hal_cuda_BlockSizeDef_vec_t block_sizes_vec =
      iree_hal_cuda_ExecutableDef_block_sizes_get(executable_def);
  iree_host_size_t entry_point_count =
      flatbuffers_string_vec_len(entry_points_vec);

  // Calculate the total number of characters across all entry point names. This
  // is only required when tracing so that we can store copies of the names as
  // the flatbuffer storing the strings may be released while the executable is
  // still live.
  iree_host_size_t total_entry_point_name_chars = 0;
  IREE_TRACE({
    for (iree_host_size_t i = 0; i < entry_point_count; i++) {
      const char* entry_name = flatbuffers_string_vec_at(entry_points_vec, i);
      total_entry_point_name_chars += flatbuffers_string_len(entry_name);
    }
  });

  // Allocate storage for the kernel module.
  iree_host_size_t total_size =
      sizeof(*executable) +
      entry_point_count * sizeof(executable->entry_points[0]) +
      total_entry_point_name_chars;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator, total_size, (void**)&executable));
  IREE_TRACE(
      char* string_table_buffer =
          (char*)((char*)executable + sizeof(*executable) +
                  entry_point_count * sizeof(executable->entry_points[0])));

  iree_hal_resource_initialize(&iree_hal_cuda2_native_executable_vtable,
                               &executable->resource);

  // Load the PTX image - this will fail if the device cannot handle the
  // contents. We could check this prior to creating
  CUmodule module = NULL;

  iree_status_t status = IREE_CURESULT_TO_STATUS(
      symbols, cuModuleLoadDataEx(&module, ptx_image, 0, NULL, NULL),
      "cuModuleLoadDataEx");

  // Query max optin shared memory per block - we'll use it to compare with
  // kernel usages.
  int32_t max_shared_memory = 0;
  if (iree_status_is_ok(status)) {
    status = IREE_CURESULT_TO_STATUS(
        symbols,
        cuDeviceGetAttribute(
            &max_shared_memory,
            CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, device),
        "cuDeviceGetAttribute");
  }

  if (iree_status_is_ok(status)) {
    executable->host_allocator = host_allocator;
    executable->symbols = symbols;
    executable->cu_module = module;
    executable->entry_point_count = entry_point_count;
    for (iree_host_size_t i = 0; i < entry_point_count; i++) {
      // Lookup the function in the module; this should always succeed but we
      // cannot trust that the input was generated by our compiler.
      CUfunction function = NULL;
      const char* entry_name = flatbuffers_string_vec_at(entry_points_vec, i);
      status = IREE_CURESULT_TO_STATUS(
          symbols,
          cuModuleGetFunction(&function, executable->cu_module, entry_name),
          "cuModuleGetFunction");
      if (!iree_status_is_ok(status)) break;
      if (!function) {
        status = iree_make_status(IREE_STATUS_NOT_FOUND,
                                  "exported module function '%s' not found",
                                  entry_name);
        break;
      }

      if (shared_memory_sizes[i] > max_shared_memory) {
        status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                  "requested shared memory size of %d bytes "
                                  "larger than allowed size of %d bytes",
                                  shared_memory_sizes[i], max_shared_memory);
      } else {
        status = IREE_CURESULT_TO_STATUS(
            symbols,
            cuFuncSetAttribute(function,
                               CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                               shared_memory_sizes[i]),
            "cuFuncSetAttribute");
      }
      if (!iree_status_is_ok(status)) break;

      // Package required parameters for kernel launches for each entry point.
      iree_hal_cuda2_kernel_info_t* info = &executable->entry_points[i];
      info->layout = executable_params->pipeline_layouts[i];
      iree_hal_pipeline_layout_retain(info->layout);
      info->function = function;
      info->block_size[0] = block_sizes_vec[i].x;
      info->block_size[1] = block_sizes_vec[i].y;
      info->block_size[2] = block_sizes_vec[i].z;
      info->shared_memory_size = shared_memory_sizes[i];

      // Stash the entry point name in the string table for use when tracing.
      IREE_TRACE({
        iree_host_size_t entry_name_length = flatbuffers_string_len(entry_name);
        memcpy(string_table_buffer, entry_name, entry_name_length);
        info->function_name =
            iree_make_string_view(string_table_buffer, entry_name_length);
        string_table_buffer += entry_name_length;
      });

      IREE_TRACE({
        if (iree_hal_cuda_ExecutableDef_source_locations_is_present(
                executable_def)) {
          iree_hal_cuda_FileLineLocDef_vec_t source_locs_vec =
              iree_hal_cuda_ExecutableDef_source_locations_get(executable_def);
          iree_hal_cuda_FileLineLocDef_table_t source_loc =
              iree_hal_cuda_FileLineLocDef_vec_at(source_locs_vec, i);
          flatbuffers_string_t filename =
              iree_hal_cuda_FileLineLocDef_filename_get(source_loc);
          uint32_t line = iree_hal_cuda_FileLineLocDef_line_get(source_loc);
          info->source_filename =
              iree_make_string_view(filename, flatbuffers_string_len(filename));
          info->source_line = line;
        }
      });
    }
  }

  if (iree_status_is_ok(status)) {
    *out_executable = (iree_hal_executable_t*)executable;
  } else {
    iree_hal_executable_destroy((iree_hal_executable_t*)executable);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_cuda2_native_executable_destroy(
    iree_hal_executable_t* base_executable) {
  iree_hal_cuda2_native_executable_t* executable =
      iree_hal_cuda2_native_executable_cast(base_executable);
  iree_allocator_t host_allocator = executable->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  for (iree_host_size_t i = 0; i < executable->entry_point_count; ++i) {
    iree_hal_pipeline_layout_release(executable->entry_points[i].layout);
  }
  if (executable->cu_module) {
    IREE_CUDA_IGNORE_ERROR(executable->symbols,
                           cuModuleUnload(executable->cu_module));
  }
  iree_allocator_free(host_allocator, executable);

  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_hal_cuda2_native_executable_entry_point_kernel_info(
    iree_hal_executable_t* base_executable, int32_t entry_point,
    iree_hal_cuda2_kernel_info_t* out_info) {
  iree_hal_cuda2_native_executable_t* executable =
      iree_hal_cuda2_native_executable_cast(base_executable);
  if (entry_point >= executable->entry_point_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "entry point ordinal %d out of range; executable "
                            "only contains %" PRIhsz " entry points",
                            entry_point, executable->entry_point_count);
  }
  memcpy(out_info, &executable->entry_points[entry_point], sizeof(*out_info));
  return iree_ok_status();
}

static const iree_hal_executable_vtable_t
    iree_hal_cuda2_native_executable_vtable = {
        .destroy = iree_hal_cuda2_native_executable_destroy,
};
