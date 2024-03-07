// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/hip/native_executable.h"

#include <stddef.h>

#include "iree/base/api.h"
#include "iree/hal/drivers/hip/dynamic_symbols.h"
#include "iree/hal/drivers/hip/status_util.h"

// flatcc schemas:
#include "iree/base/internal/flatcc/parsing.h"
// Using the existing ROCM schema fow now.
#include "iree/schemas/rocm_executable_def_reader.h"
#include "iree/schemas/rocm_executable_def_verifier.h"

typedef struct iree_hal_hip_native_executable_t {
  // Abstract resource used for injecting reference counting and vtable;
  // must be at offset 0.
  iree_hal_resource_t resource;

  iree_allocator_t host_allocator;

  const iree_hal_hip_dynamic_symbols_t* symbols;

  // The loaded HIP module.
  hipModule_t hip_module;

  iree_host_size_t entry_point_count;
  // The list of entry point data pointers, pointing to trailing inline
  // allocation after the end of this struct.
  iree_hal_hip_kernel_info_t entry_points[];
} iree_hal_hip_native_executable_t;
// + Additional inline allocation for holding entry point information.

static const iree_hal_executable_vtable_t iree_hal_hip_native_executable_vtable;

static iree_hal_hip_native_executable_t* iree_hal_hip_native_executable_cast(
    iree_hal_executable_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_hip_native_executable_vtable);
  return (iree_hal_hip_native_executable_t*)base_value;
}

// Verifies the structure of the flatbuffer so that we can avoid doing so during
// runtime.
//
// There are still some conditions we must be aware of (such as omitted names on
// functions with internal linkage), however we shouldn't need to bounds check
// anything within the flatbuffer after this succeeds.
static iree_status_t iree_hal_hip_native_executable_flatbuffer_verify(
    iree_const_byte_span_t flatbuffer_data) {
  if (!flatbuffer_data.data) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "flatbuffer data is not present");
  }

  // Run flatcc generated verification. This ensures all pointers are in-bounds
  // and that we can safely walk the file, but not that the actual contents of
  // the flatbuffer meet our expectations.
  int verify_ret = iree_hal_rocm_ExecutableDef_verify_as_root(
      flatbuffer_data.data, flatbuffer_data.data_length);
  if (verify_ret != flatcc_verify_ok) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "flatbuffer verification failed: %s",
                            flatcc_verify_error_string(verify_ret));
  }

  iree_hal_rocm_ExecutableDef_table_t executable_def =
      iree_hal_rocm_ExecutableDef_as_root(flatbuffer_data.data);

  flatbuffers_string_vec_t entry_points_vec =
      iree_hal_rocm_ExecutableDef_entry_points_get(executable_def);
  size_t entry_point_count = flatbuffers_string_vec_len(entry_points_vec);
  if (entry_point_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "no entry points present");
  }
  for (size_t i = 0; i < entry_point_count; ++i) {
    if (flatbuffers_string_len(
            flatbuffers_string_vec_at(entry_points_vec, i)) == 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "executable entry point %zu has no name", i);
    }
  }

  iree_hal_rocm_BlockSizeDef_vec_t block_sizes_vec =
      iree_hal_rocm_ExecutableDef_block_sizes_get(executable_def);
  size_t block_size_count = iree_hal_rocm_BlockSizeDef_vec_len(block_sizes_vec);
  if (entry_point_count != block_size_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "entry points (%zu) and block sizes (%zu) count mismatch",
        entry_point_count, block_size_count);
  }

  flatbuffers_uint32_vec_t shared_memory_sizes_vec =
      iree_hal_rocm_ExecutableDef_shared_memory_sizes_get(executable_def);
  size_t shared_memory_sizes_count =
      flatbuffers_string_vec_len(shared_memory_sizes_vec);
  if (entry_point_count != shared_memory_sizes_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "entry points (%zu) and shared memory sizes (%zu) count mismatch",
        entry_point_count, shared_memory_sizes_count);
  }

  flatbuffers_string_t hsaco_image =
      iree_hal_rocm_ExecutableDef_hsaco_image_get(executable_def);
  if (flatbuffers_string_len(hsaco_image) == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "no HSACO image present");
  }

  return iree_ok_status();
}

iree_status_t iree_hal_hip_native_executable_create(
    const iree_hal_hip_dynamic_symbols_t* symbols, hipDevice_t device,
    const iree_hal_executable_params_t* executable_params,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable) {
  IREE_ASSERT_ARGUMENT(symbols);
  IREE_ASSERT_ARGUMENT(executable_params);
  IREE_ASSERT_ARGUMENT(out_executable);
  IREE_TRACE_ZONE_BEGIN(z0);

  *out_executable = NULL;
  iree_hal_hip_native_executable_t* executable = NULL;

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_hip_native_executable_flatbuffer_verify(
              executable_params->executable_data));

  iree_hal_rocm_ExecutableDef_table_t executable_def =
      iree_hal_rocm_ExecutableDef_as_root(
          executable_params->executable_data.data);

  flatbuffers_string_vec_t entry_points_vec =
      iree_hal_rocm_ExecutableDef_entry_points_get(executable_def);
  iree_hal_rocm_BlockSizeDef_vec_t block_sizes_vec =
      iree_hal_rocm_ExecutableDef_block_sizes_get(executable_def);
  flatbuffers_uint32_vec_t shared_memory_sizes_vec =
      iree_hal_rocm_ExecutableDef_shared_memory_sizes_get(executable_def);
  flatbuffers_string_t hsaco_image =
      iree_hal_rocm_ExecutableDef_hsaco_image_get(executable_def);
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

  iree_hal_resource_initialize(&iree_hal_hip_native_executable_vtable,
                               &executable->resource);

  // Load the HSACO image - this will fail if the device cannot handle the
  // contents. We could check this prior to creating
  hipModule_t module = NULL;

  iree_status_t status = IREE_HIP_RESULT_TO_STATUS(
      symbols, hipModuleLoadDataEx(&module, hsaco_image, 0, NULL, NULL),
      "hipModuleLoadDataEx");
  if (!iree_status_is_ok(status)) {
    status = iree_status_annotate(
        status,
        IREE_SV("mismatched target chip? missing/wrong bitcode directory?"));
  }

  // Query max optin shared memory per block - we'll use it to compare with
  // kernel usages.
  uint32_t max_shared_memory = 0;
  if (iree_status_is_ok(status)) {
    status = IREE_HIP_RESULT_TO_STATUS(
        symbols,
        hipDeviceGetAttribute(&max_shared_memory,
                              hipDeviceAttributeMaxSharedMemoryPerBlock,
                              device),
        "hipDeviceGetAttribute");
  }

  if (iree_status_is_ok(status)) {
    executable->host_allocator = host_allocator;
    executable->symbols = symbols;
    executable->hip_module = module;
    executable->entry_point_count = entry_point_count;
    for (iree_host_size_t i = 0; i < entry_point_count; i++) {
      // Lookup the function in the module; this should always succeed but we
      // cannot trust that the input was generated by our compiler.
      hipFunction_t function = NULL;
      const char* entry_name = flatbuffers_string_vec_at(entry_points_vec, i);
      status = IREE_HIP_RESULT_TO_STATUS(
          symbols,
          hipModuleGetFunction(&function, executable->hip_module, entry_name),
          "hipModuleGetFunction");
      if (!iree_status_is_ok(status)) break;
      if (!function) {
        status = iree_make_status(IREE_STATUS_NOT_FOUND,
                                  "exported module function '%s' not found",
                                  entry_name);
        break;
      }

      if (shared_memory_sizes_vec[i] > max_shared_memory) {
        status = iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "function '%s' requested shared memory size of %u bytes larger "
            "than allowed size of %u bytes",
            entry_name, shared_memory_sizes_vec[i], max_shared_memory);
      } else {
        status = IREE_HIP_RESULT_TO_STATUS(
            symbols,
            hipFuncSetAttribute(
                function,
                (hipFuncAttribute)
                    HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                shared_memory_sizes_vec[i]),
            "hipFuncSetAttribute");
      }
      if (!iree_status_is_ok(status)) break;

      // Package required parameters for kernel launches for each entry point.
      iree_hal_hip_kernel_info_t* kernel_info = &executable->entry_points[i];
      kernel_info->layout = executable_params->pipeline_layouts[i];
      iree_hal_pipeline_layout_retain(kernel_info->layout);
      kernel_info->function = function;
      kernel_info->block_size[0] = block_sizes_vec[i].x;
      kernel_info->block_size[1] = block_sizes_vec[i].y;
      kernel_info->block_size[2] = block_sizes_vec[i].z;
      kernel_info->shared_memory_size = shared_memory_sizes_vec[i];

      // Stash the entry point name in the string table for use when tracing.
      IREE_TRACE({
        iree_host_size_t entry_name_length = flatbuffers_string_len(entry_name);
        memcpy(string_table_buffer, entry_name, entry_name_length);
        kernel_info->function_name =
            iree_make_string_view(string_table_buffer, entry_name_length);
        string_table_buffer += entry_name_length;
      });

      IREE_TRACE({
        if (iree_hal_rocm_ExecutableDef_source_locations_is_present(
                executable_def)) {
          iree_hal_rocm_FileLineLocDef_vec_t source_locs_vec =
              iree_hal_rocm_ExecutableDef_source_locations_get(executable_def);
          iree_hal_rocm_FileLineLocDef_table_t source_loc =
              iree_hal_rocm_FileLineLocDef_vec_at(source_locs_vec, i);
          flatbuffers_string_t filename =
              iree_hal_rocm_FileLineLocDef_filename_get(source_loc);
          uint32_t line = iree_hal_rocm_FileLineLocDef_line_get(source_loc);
          kernel_info->source_filename =
              iree_make_string_view(filename, flatbuffers_string_len(filename));
          kernel_info->source_line = line;
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

static void iree_hal_hip_native_executable_destroy(
    iree_hal_executable_t* base_executable) {
  iree_hal_hip_native_executable_t* executable =
      iree_hal_hip_native_executable_cast(base_executable);
  iree_allocator_t host_allocator = executable->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  for (iree_host_size_t i = 0; i < executable->entry_point_count; ++i) {
    iree_hal_pipeline_layout_release(executable->entry_points[i].layout);
  }
  if (executable->hip_module) {
    IREE_HIP_IGNORE_ERROR(executable->symbols,
                          hipModuleUnload(executable->hip_module));
  }
  iree_allocator_free(host_allocator, executable);

  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_hal_hip_native_executable_entry_point_kernel_info(
    iree_hal_executable_t* base_executable, int32_t entry_point,
    iree_hal_hip_kernel_info_t* out_info) {
  iree_hal_hip_native_executable_t* executable =
      iree_hal_hip_native_executable_cast(base_executable);
  if (entry_point >= executable->entry_point_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "entry point ordinal %d out of range; executable "
                            "only contains %ld entry points",
                            entry_point, executable->entry_point_count);
  }
  memcpy(out_info, &executable->entry_points[entry_point], sizeof(*out_info));
  return iree_ok_status();
}

static const iree_hal_executable_vtable_t
    iree_hal_hip_native_executable_vtable = {
        .destroy = iree_hal_hip_native_executable_destroy,
};
