// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/hip/native_executable.h"

#include <stddef.h>

#include "iree/base/api.h"
#include "iree/base/internal/math.h"
#include "iree/hal/drivers/hip/dynamic_symbols.h"
#include "iree/hal/drivers/hip/native_executable_hipf.h"
#include "iree/hal/drivers/hip/status_util.h"
#include "iree/hal/utils/executable_debug_info.h"
#include "iree/hal/utils/executable_header.h"

// flatcc schemas:
#include "iree/base/internal/flatcc/parsing.h"
#include "iree/schemas/executable_debug_info_reader.h"
#include "iree/schemas/executable_debug_info_verifier.h"
#include "iree/schemas/hip_executable_def_reader.h"
#include "iree/schemas/hip_executable_def_verifier.h"

typedef struct iree_hal_hip_native_executable_per_device_data_t {
  // Loaded HIP modules.
  iree_host_size_t module_count;
  hipModule_t* modules;

  // Exported kernels referencing the loaded modules.
  iree_host_size_t export_count;
  iree_hal_hip_kernel_params_t exports[];
} iree_hal_hip_native_executable_per_device_data_t;

typedef struct iree_hal_hip_native_executable_t {
  // Abstract resource used for injecting reference counting and vtable;
  // must be at offset 0.
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;

  const iree_hal_hip_dynamic_symbols_t* symbols;

  // Kernel info from fat binary (for FPIH format only)
  iree_host_size_t kernel_count;
  iree_hal_hip_kernel_info_t* kernels;

  iree_host_size_t num_devices;
  iree_hal_hip_native_executable_per_device_data_t* per_device_data[];
} iree_hal_hip_native_executable_t;

static const iree_hal_executable_vtable_t iree_hal_hip_native_executable_vtable;

static iree_hal_hip_native_executable_t* iree_hal_hip_native_executable_cast(
    iree_hal_executable_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_hip_native_executable_vtable);
  return (iree_hal_hip_native_executable_t*)base_value;
}

typedef struct iree_hal_hip_limits_t {
  int max_block_dims[3];
  int max_block_shared_memory_size;
} iree_hal_hip_limits_t;
static iree_status_t iree_hal_hip_query_limits(
    const iree_hal_hip_dynamic_symbols_t* symbols, hipDevice_t device,
    iree_hal_hip_limits_t* out_limits) {
  memset(out_limits, 0, sizeof(*out_limits));

  IREE_HIP_RETURN_IF_ERROR(
      symbols,
      hipDeviceGetAttribute(&out_limits->max_block_dims[0],
                            hipDeviceAttributeMaxBlockDimX, device),
      "hipDeviceGetAttribute");
  IREE_HIP_RETURN_IF_ERROR(
      symbols,
      hipDeviceGetAttribute(&out_limits->max_block_dims[1],
                            hipDeviceAttributeMaxBlockDimY, device),
      "hipDeviceGetAttribute");
  IREE_HIP_RETURN_IF_ERROR(
      symbols,
      hipDeviceGetAttribute(&out_limits->max_block_dims[2],
                            hipDeviceAttributeMaxBlockDimZ, device),
      "hipDeviceGetAttribute");

  IREE_HIP_RETURN_IF_ERROR(
      symbols,
      hipDeviceGetAttribute(&out_limits->max_block_shared_memory_size,
                            hipDeviceAttributeMaxSharedMemoryPerBlock, device),
      "hipDeviceGetAttribute");

  return iree_ok_status();
}

iree_status_t iree_hal_hip_native_executable_infer_format(
    iree_const_byte_span_t executable_data,
    iree_host_size_t executable_format_capacity, char* executable_format,
    iree_host_size_t* out_inferred_size) {
  // Read the size prefix (with unsafe inference if size is unknown).
  const bool unsafe_infer_size = (executable_data.data_length == 0);
  iree_const_byte_span_t contained_data = iree_const_byte_span_empty();

  iree_status_t native_hip = iree_hal_hip_read_native_header(
      executable_data, unsafe_infer_size, &contained_data);
  if (iree_status_is_ok(native_hip)) {
    // Successfully read as native HIP executable.
    iree_string_view_t format = IREE_SV("FPIH");
    if (format.size >= executable_format_capacity) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "executable format buffer too small");
    }
    memcpy(executable_format, format.data, format.size + /*NUL*/ 1);
    *out_inferred_size = contained_data.data_length;
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_hal_read_executable_flatbuffer_header(
      executable_data, unsafe_infer_size,
      iree_hal_hip_ExecutableDef_file_identifier, &contained_data));

  // Verify the flatbuffer structure.
  if (!iree_hal_hip_ExecutableDef_verify_as_root(contained_data.data,
                                                 contained_data.data_length)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "failed to verify executable flatbuffer structure");
  }

  // Write the format string.
  iree_string_view_t format = IREE_SV("HSACO");
  if (format.size >= executable_format_capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "executable format buffer too small");
  }
  memcpy(executable_format, format.data, format.size + /*NUL*/ 1);

  // Return the total size (header + flatbuffer).
  *out_inferred_size =
      sizeof(iree_flatbuffer_file_header_t) + contained_data.data_length;
  return iree_ok_status();
}

// Verifies the structure of the flatbuffer so that we can avoid doing so during
// runtime.
//
// There are still some conditions we must be aware of (such as omitted names on
// functions with internal linkage), however we shouldn't need to bounds check
// anything within the flatbuffer after this succeeds.
static iree_status_t iree_hal_hip_native_executable_flatbuffer_verify(
    iree_const_byte_span_t flatbuffer_data,
    const iree_hal_hip_limits_t* limits) {
  IREE_ASSERT(flatbuffer_data.data && flatbuffer_data.data_length >= 16);

  // Run flatcc generated verification. This ensures all pointers are in-bounds
  // and that we can safely walk the file, but not that the actual contents of
  // the flatbuffer meet our expectations.
  int verify_ret = iree_hal_hip_ExecutableDef_verify_as_root(
      flatbuffer_data.data, flatbuffer_data.data_length);
  if (verify_ret != flatcc_verify_ok) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "flatbuffer verification failed: %s",
                            flatcc_verify_error_string(verify_ret));
  }

  iree_hal_hip_ExecutableDef_table_t executable_def =
      iree_hal_hip_ExecutableDef_as_root(flatbuffer_data.data);

  iree_hal_hip_ModuleDef_vec_t modules_vec =
      iree_hal_hip_ExecutableDef_modules_get(executable_def);
  iree_host_size_t module_count = iree_hal_hip_ModuleDef_vec_len(modules_vec);
  for (iree_host_size_t i = 0; i < module_count; ++i) {
    iree_hal_hip_ModuleDef_table_t module_def =
        iree_hal_hip_ModuleDef_vec_at(modules_vec, i);
    if (!module_def) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "modules[%" PRIhsz "] is NULL", i);
    }
    if (flatbuffers_string_len(
            iree_hal_hip_ModuleDef_hsaco_image_get(module_def)) == 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "modules[%" PRIhsz "] contents are empty", i);
    }
  }

  iree_hal_hip_ExportDef_vec_t exports_vec =
      iree_hal_hip_ExecutableDef_exports_get(executable_def);
  for (iree_host_size_t i = 0; i < iree_hal_hip_ExportDef_vec_len(exports_vec);
       ++i) {
    iree_hal_hip_ExportDef_table_t export_def =
        iree_hal_hip_ExportDef_vec_at(exports_vec, i);
    if (!export_def) continue;

    uint32_t module_ordinal =
        iree_hal_hip_ExportDef_module_ordinal_get(export_def);
    if (module_ordinal >= module_count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "exports[%" PRIhsz
                              "] module_ordinal %u is out of bounds %" PRIhsz,
                              i, module_ordinal, module_count);
    }

    if (flatbuffers_string_len(
            iree_hal_hip_ExportDef_kernel_name_get(export_def)) == 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "exports[%" PRIhsz "] name is empty", i);
    }

    if (iree_hal_hip_ExportDef_block_dims_is_present(export_def)) {
      const iree_hal_hip_BlockDims_t* block_dims =
          iree_hal_hip_ExportDef_block_dims_get(export_def);
      if (block_dims->x > limits->max_block_dims[0] ||
          block_dims->y > limits->max_block_dims[1] ||
          block_dims->z > limits->max_block_dims[2]) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "exports[%" PRIhsz
            "] block dims %ux%ux%u exceeds device maximum %ux%ux%u",
            i, block_dims->x, block_dims->y, block_dims->z,
            limits->max_block_dims[0], limits->max_block_dims[1],
            limits->max_block_dims[2]);
      }
    } else {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "exports[%" PRIhsz "] blocks dims are missing",
                              i);
    }

    uint32_t constant_count =
        iree_hal_hip_ExportDef_constant_count_get(export_def);
    if (constant_count > IREE_HAL_HIP_MAX_DISPATCH_CONSTANT_COUNT) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "exports[%" PRIhsz "] constant_count %u exceeds maximum of %u", i,
          constant_count, IREE_HAL_HIP_MAX_DISPATCH_CONSTANT_COUNT);
    }

    iree_hal_hip_BindingBits_vec_t binding_flags_vec =
        iree_hal_hip_ExportDef_binding_flags_get(export_def);
    if (iree_hal_hip_BindingBits_vec_len(binding_flags_vec) >
        IREE_HAL_HIP_MAX_DISPATCH_BINDING_COUNT) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "exports[%" PRIhsz "] binding_flags count %zu exceeds maximum of %u",
          i, iree_hal_hip_BindingBits_vec_len(binding_flags_vec),
          IREE_HAL_HIP_MAX_DISPATCH_BINDING_COUNT);
    }

    IREE_RETURN_IF_ERROR(iree_hal_debug_verify_export_def(
        iree_hal_hip_ExportDef_debug_info_get(export_def)));
  }

  return iree_ok_status();
}

// Verifies a function against the device limits so that we can avoid doing so
// during runtime.
static iree_status_t iree_hal_hip_function_attributes_verify(
    iree_host_size_t id, const iree_hal_hip_dynamic_symbols_t* symbols,
    hipFunction_t function, const iree_hal_hip_limits_t* limits) {
  int block_shared_memory_size;
  IREE_RETURN_IF_ERROR(IREE_HIP_CALL_TO_STATUS(
      symbols,
      hipFuncGetAttribute(
          &block_shared_memory_size,
          (hipFuncAttribute)HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, function),
      "hipFuncGetAttribute"));
  if (block_shared_memory_size > limits->max_block_shared_memory_size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "exports[%" PRIhsz
                            "] requires %uB of shared memory and "
                            "exceeds the device maximum of %uB per block",
                            id, block_shared_memory_size,
                            limits->max_block_shared_memory_size);
  }

  return iree_ok_status();
}

// Creates a native executable from an FPIH (Fat Binary) format.
static iree_status_t iree_hal_hip_native_executable_create_fpih(
    const iree_hal_hip_dynamic_symbols_t* symbols,
    iree_hal_hip_device_topology_t topology,
    const iree_hal_executable_params_t* executable_params,
    const iree_hal_hip_limits_t* limits, iree_allocator_t host_allocator,
    iree_hal_executable_t** out_executable) {
  IREE_TRACE_ZONE_BEGIN(z0);
  hipDeviceProp_tR0000 props;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, IREE_HIP_CALL_TO_STATUS(
              symbols,
              hipGetDeviceProperties(&props, topology.devices[0].hip_device)));

  // Construct target triple (e.g., "hip-amdgcn-amd-amdhsa--gfx942")
  // For now, use the device name directly as the target
  // TODO: map device name to proper LLVM triple
  char target_triple_str[512];
  snprintf(target_triple_str, sizeof(target_triple_str),
           "hipv4-amdgcn-amd-amdhsa--%s", props.gcnArchName);
  char* col = strstr(target_triple_str, ":");
  if (col) {
    *col = '\0';
  }

  iree_string_view_t target_triple = iree_make_cstring_view(target_triple_str);

  // Parse the fat binary to extract kernel information for this device
  iree_hal_hip_fat_binary_info_t fat_binary_info;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_hip_parse_fat_binary_kernels(
              executable_params->executable_data, target_triple, host_allocator,
              &fat_binary_info));

  iree_host_size_t export_count = fat_binary_info.kernel_count;

  // Allocate storage for the executable and its associated data structures.
  iree_hal_hip_native_executable_t* executable = NULL;
  iree_host_size_t native_executable_device_info_size =
      sizeof(*executable->per_device_data[0]) +
      1 * sizeof(executable->per_device_data[0]->modules[0]) +  // Single module
      export_count * sizeof(executable->per_device_data[0]->exports[0]);
  native_executable_device_info_size =
      iree_host_align(native_executable_device_info_size, iree_max_align_t);
  const iree_host_size_t total_size =
      sizeof(*executable) +
      topology.count * sizeof(executable->per_device_data[0]) +
      topology.count * native_executable_device_info_size;

  iree_status_t status =
      iree_allocator_malloc(host_allocator, total_size, (void**)&executable);
  if (!iree_status_is_ok(status)) {
    iree_hal_hip_free_kernel_info(host_allocator, fat_binary_info.kernel_count,
                                   fat_binary_info.kernels);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  iree_hal_resource_initialize(&iree_hal_hip_native_executable_vtable,
                               &executable->resource);
  executable->host_allocator = host_allocator;
  executable->symbols = symbols;
  executable->kernel_count = fat_binary_info.kernel_count;
  executable->kernels = fat_binary_info.kernels;
  executable->num_devices = topology.count;
  const iree_host_size_t per_device_data_size =
      topology.count * sizeof(executable->per_device_data[0]);
  const uint8_t* per_device_data_location =
      (uint8_t*)executable + sizeof(*executable);

  for (iree_host_size_t i = 0; i < topology.count; ++i) {
    const iree_host_size_t native_executable_device_info_size_offset =
        (i * native_executable_device_info_size);

    executable->per_device_data[i] =
        (iree_hal_hip_native_executable_per_device_data_t*)(per_device_data_location +
                                                            per_device_data_size +
                                                            native_executable_device_info_size_offset);
  }

  // Load the module for each device
  status = iree_ok_status();
  for (iree_host_size_t j = 0; j < topology.count && iree_status_is_ok(status);
       ++j) {
    status = IREE_HIP_CALL_TO_STATUS(
        symbols, hipCtxPushCurrent(topology.devices[j].hip_context));
    if (!iree_status_is_ok(status)) break;

    iree_hal_hip_native_executable_per_device_data_t* per_device_data =
        executable->per_device_data[j];

    per_device_data->module_count = 1;
    per_device_data->modules =
        (hipModule_t*)((uint8_t*)per_device_data + sizeof(*per_device_data) +
                       (export_count * sizeof(per_device_data->exports[0])));
    per_device_data->export_count = export_count;

    // Load the fat binary module
    char error_log[8192] = {0};
    hipJitOption jit_options[] = {
        hipJitOptionErrorLogBuffer,
        hipJitOptionErrorLogBufferSizeBytes,
    };
    void* jit_option_values[] = {
        (void*)error_log,
        (void*)(uint32_t)sizeof(error_log),
    };
    hipModule_t module = NULL;
    status = IREE_HIP_CALL_TO_STATUS(
        symbols,
        hipModuleLoadDataEx(&module, fat_binary_info.bundle_data,
                            IREE_ARRAYSIZE(jit_options), jit_options,
                            jit_option_values),
        "hipModuleLoadDataEx");
    if (!iree_status_is_ok(status)) {
      status = iree_status_annotate(
          status,
          IREE_SV("mismatched target chip? missing/wrong bitcode directory?"));
      if (strlen(error_log) > 0) {
        status =
            iree_status_annotate(status, iree_make_cstring_view(error_log));
      }
      IREE_IGNORE_ERROR(
          IREE_HIP_CALL_TO_STATUS(symbols, hipCtxPopCurrent(NULL)));
      break;
    }

    per_device_data->modules[0] = module;

    // Get function handles for each kernel
    for (iree_host_size_t i = 0; i < export_count; ++i) {
      iree_string_view_t kernel_name = fat_binary_info.kernels[i].name;

      // Convert kernel name to null-terminated string
      char kernel_name_cstr[1024];
      iree_host_size_t name_len =
          kernel_name.size < 1023 ? kernel_name.size : 1023;
      memcpy(kernel_name_cstr, kernel_name.data, name_len);
      kernel_name_cstr[name_len] = '\0';

      hipFunction_t function = NULL;
      status = IREE_HIP_CALL_TO_STATUS(
          symbols, hipModuleGetFunction(&function, module, kernel_name_cstr),
          "hipModuleGetFunction");
      if (!iree_status_is_ok(status)) {
        IREE_IGNORE_ERROR(
            IREE_HIP_CALL_TO_STATUS(symbols, hipCtxPopCurrent(NULL)));
        break;
      }
      if (!function) {
        status = iree_make_status(IREE_STATUS_NOT_FOUND,
                                  "kernel `%.*s` not found in fat binary",
                                  (int)kernel_name.size, kernel_name.data);
        IREE_IGNORE_ERROR(
            IREE_HIP_CALL_TO_STATUS(symbols, hipCtxPopCurrent(NULL)));
        break;
      }

      status =
          iree_hal_hip_function_attributes_verify(i, symbols, function, limits);
      if (!iree_status_is_ok(status)) {
        IREE_IGNORE_ERROR(
            IREE_HIP_CALL_TO_STATUS(symbols, hipCtxPopCurrent(NULL)));
        break;
      }

      // Store kernel info from parsed fat binary
      iree_hal_hip_kernel_params_t* kernel_info = &per_device_data->exports[i];
      kernel_info->function = function;
      kernel_info->function_name = kernel_name;
      // Copy block dimensions from parsed kernel info
      kernel_info->block_dims[0] = fat_binary_info.kernels[i].block_dims[0];
      kernel_info->block_dims[1] = fat_binary_info.kernels[i].block_dims[1];
      kernel_info->block_dims[2] = fat_binary_info.kernels[i].block_dims[2];
      // Copy binding and constant counts
      kernel_info->constant_count = fat_binary_info.kernels[i].constant_count;
      kernel_info->binding_count = fat_binary_info.kernels[i].binding_count;

      // Convert parsed kernel parameters to executable export parameter format
      iree_hal_hip_kernel_info_t* kernel_meta = &fat_binary_info.kernels[i];
      if (kernel_meta->parameters != NULL) {
        // Allocate parameter array using the actual parameter count from metadata
        kernel_info->parameter_count = kernel_meta->parameter_count;
        if (kernel_info->parameter_count > 0) {
          status = iree_allocator_malloc(
              host_allocator,
              kernel_info->parameter_count *
                  sizeof(iree_hal_hip_kernel_export_parameter_t),
              (void**)&kernel_info->parameters);
          if (!iree_status_is_ok(status)) {
            IREE_IGNORE_ERROR(
                IREE_HIP_CALL_TO_STATUS(symbols, hipCtxPopCurrent(NULL)));
            break;
          }

          // Fill in parameters from parsed kernel metadata
          for (iree_host_size_t p = 0; p < kernel_info->parameter_count; ++p) {
            iree_hal_hip_kernel_param_t* src_param =
                &kernel_meta->parameters[p];
            iree_hal_hip_kernel_export_parameter_t* dst_param =
                &kernel_info->parameters[p];

            // Determine parameter type based on parsed value_kind
            if (src_param->type == 1) {
              // Pointer/buffer parameter
              dst_param->export.type =
                  IREE_HAL_EXECUTABLE_EXPORT_PARAMETER_TYPE_BINDING;
              // Use the actual kernel ABI offset from metadata.
              // This is critical for native kernels where we need to pack
              // arguments at the correct offsets in the kernarg buffer.
              dst_param->export.offset = (uint16_t)src_param->offset;
              dst_param->export.size = 8;  // Pointer size
              dst_param->buffer_offset = src_param->offset;
            } else {
              // Value/constant parameter
              dst_param->export.type =
                  IREE_HAL_EXECUTABLE_EXPORT_PARAMETER_TYPE_CONSTANT;
              // Use the actual kernel ABI offset from metadata.
              dst_param->export.offset = (uint16_t)src_param->offset;
              dst_param->export.size = src_param->size;  // Don't truncate to uint8_t
              dst_param->buffer_offset = src_param->offset;
            }

            dst_param->export.flags =
                IREE_HAL_EXECUTABLE_EXPORT_PARAMETER_FLAG_NONE;
            dst_param->export.name = iree_string_view_empty();
          }
        }
      } else {
        // No parsed parameters available
        kernel_info->parameter_count = 0;
        kernel_info->parameters = NULL;
      }

      IREE_TRACE({
        kernel_info->debug_info.function_name = kernel_name;
        kernel_info->debug_info.source_filename = iree_string_view_empty();
        kernel_info->debug_info.source_line = 0;
      });
    }

    status = IREE_HIP_CALL_TO_STATUS(symbols, hipCtxPopCurrent(NULL));
  }

  // Clean up temporary parameter arrays (kernel info ownership transferred to executable)
  for (iree_host_size_t i = 0; i < fat_binary_info.kernel_count; ++i) {
    if (fat_binary_info.kernels[i].parameters) {
      iree_allocator_free(host_allocator,
                          fat_binary_info.kernels[i].parameters);
    }
  }
  // Note: kernel info (including allocated names) is now owned by the executable
  // and will be freed in iree_hal_hip_native_executable_destroy

  if (iree_status_is_ok(status)) {
    *out_executable = (iree_hal_executable_t*)executable;
  } else {
    iree_hal_executable_destroy((iree_hal_executable_t*)executable);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Creates a native executable from an HSACO (flatbuffer) format.
static iree_status_t iree_hal_hip_native_executable_create_flatbuffer(
    const iree_hal_hip_dynamic_symbols_t* symbols,
    iree_hal_hip_device_topology_t topology,
    const iree_hal_executable_params_t* executable_params,
    const iree_hal_hip_limits_t* limits, iree_allocator_t host_allocator,
    iree_hal_executable_t** out_executable) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Read and strip the flatbuffer header prefix.
  iree_const_byte_span_t executable_flatbuffer = iree_const_byte_span_empty();
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hal_read_executable_flatbuffer_header(
          executable_params->executable_data, /*unsafe_infer_size=*/false,
          iree_hal_hip_ExecutableDef_file_identifier, &executable_flatbuffer));

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_hip_native_executable_flatbuffer_verify(
              executable_flatbuffer, limits));

  iree_hal_hip_ExecutableDef_table_t executable_def =
      iree_hal_hip_ExecutableDef_as_root(executable_flatbuffer.data);

  iree_hal_hip_ModuleDef_vec_t modules_vec =
      iree_hal_hip_ExecutableDef_modules_get(executable_def);
  iree_host_size_t module_count = iree_hal_hip_ModuleDef_vec_len(modules_vec);
  iree_hal_hip_ExportDef_vec_t exports_vec =
      iree_hal_hip_ExecutableDef_exports_get(executable_def);
  iree_host_size_t export_count = iree_hal_hip_ExportDef_vec_len(exports_vec);

  // Calculate the total number of characters across all entry point names. This
  // is only required when tracing so that we can store copies of the names as
  // the flatbuffer storing the strings may be released while the executable is
  // still live.
  iree_host_size_t total_export_info_length = 0;
  IREE_TRACE({
    for (iree_host_size_t i = 0; i < export_count; ++i) {
      iree_hal_hip_ExportDef_table_t export_def =
          iree_hal_hip_ExportDef_vec_at(exports_vec, i);
      total_export_info_length += iree_hal_debug_calculate_export_info_size(
          iree_hal_hip_ExportDef_debug_info_get(export_def));
    }
  });

  // Allocate storage for the executable and its associated data structures.
  iree_hal_hip_native_executable_t* executable = NULL;
  iree_host_size_t native_executable_device_info_size =
      sizeof(*executable->per_device_data[0]) +
      module_count * sizeof(executable->per_device_data[0]->modules[0]) +
      export_count * sizeof(executable->per_device_data[0]->exports[0]) +
      total_export_info_length;
  native_executable_device_info_size =
      iree_host_align(native_executable_device_info_size, iree_max_align_t);
  const iree_host_size_t total_size =
      sizeof(*executable) +
      topology.count * sizeof(executable->per_device_data[0]) +
      topology.count * native_executable_device_info_size;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator, total_size, (void**)&executable));
  iree_hal_resource_initialize(&iree_hal_hip_native_executable_vtable,
                               &executable->resource);
  executable->host_allocator = host_allocator;
  executable->symbols = symbols;
  executable->kernel_count = 0;
  executable->kernels = NULL;
  executable->num_devices = topology.count;
  const iree_host_size_t per_device_data_size =
      topology.count * sizeof(executable->per_device_data[0]);
  const uint8_t* per_device_data_location =
      (uint8_t*)executable + sizeof(*executable);

  for (iree_host_size_t i = 0; i < topology.count; ++i) {
    const iree_host_size_t native_executable_device_info_size_offset =
        (i * native_executable_device_info_size);

    executable->per_device_data[i] =
        (iree_hal_hip_native_executable_per_device_data_t*)(per_device_data_location +
                                                            per_device_data_size +
                                                            native_executable_device_info_size_offset);
  }

  // Publish any embedded source files to the tracing infrastructure.
  iree_hal_debug_publish_source_files(
      iree_hal_hip_ExecutableDef_source_files_get(executable_def));

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t j = 0; j < topology.count && iree_status_is_ok(status);
       ++j) {
    IREE_RETURN_IF_ERROR(IREE_HIP_CALL_TO_STATUS(
        symbols, hipCtxPushCurrent(topology.devices[j].hip_context)));

    if (!iree_status_is_ok(status)) {
      IREE_RETURN_IF_ERROR(
          IREE_HIP_CALL_TO_STATUS(symbols, hipCtxPopCurrent(NULL)));
      break;
    }
    iree_hal_hip_native_executable_per_device_data_t* per_device_data =
        executable->per_device_data[j];

    per_device_data->module_count = module_count;
    per_device_data->modules =
        (hipModule_t*)((uint8_t*)per_device_data + sizeof(*per_device_data) +
                       (export_count * sizeof(per_device_data->exports[0])));
    per_device_data->export_count = export_count;
    IREE_TRACE(uint8_t* export_info_ptr =
                   ((uint8_t*)per_device_data->modules +
                    module_count * sizeof(per_device_data->modules[0])));

    // Load each module first so that exports can reference them.
    for (iree_host_size_t i = 0; i < module_count; ++i) {
      iree_hal_hip_ModuleDef_table_t module_def =
          iree_hal_hip_ModuleDef_vec_at(modules_vec, i);

      // WARNING: HIP doesn't take an expected length here so we can't bound it.
      // It's likely that users could craft inputs that read beyond the extents
      // of the embedded binary.
      flatbuffers_string_t hsaco_image =
          iree_hal_hip_ModuleDef_hsaco_image_get(module_def);

      // TODO: pass hipJitOption values to get log info and other info back.
      // We pass the error buffer today but could use the info log to diagnose
      // performance warnings.
      char error_log[8192] = {0};
      hipJitOption jit_options[] = {
          hipJitOptionErrorLogBuffer,
          hipJitOptionErrorLogBufferSizeBytes,
      };
      void* jit_option_values[] = {
          (void*)error_log,
          (void*)(uint32_t)sizeof(error_log),
      };
      hipModule_t module = NULL;
      status = IREE_HIP_CALL_TO_STATUS(
          symbols,
          hipModuleLoadDataEx(&module, hsaco_image, IREE_ARRAYSIZE(jit_options),
                              jit_options, jit_option_values),
          "hipModuleLoadDataEx");
      if (!iree_status_is_ok(status)) {
        status = iree_status_annotate(
            status,
            IREE_SV(
                "mismatched target chip? missing/wrong bitcode directory?"));
        if (strlen(error_log) > 0) {
          status =
              iree_status_annotate(status, iree_make_cstring_view(error_log));
        }
        break;
      }

      per_device_data->modules[i] = module;

      if (!iree_status_is_ok(status)) {
        break;
      }
      for (iree_host_size_t i = 0; i < export_count; ++i) {
        iree_hal_hip_ExportDef_table_t export_def =
            iree_hal_hip_ExportDef_vec_at(exports_vec, i);

        // Lookup the function in the module; this should always succeed but
        // we cannot trust that the input was generated by our compiler.
        uint32_t module_ordinal =
            iree_hal_hip_ExportDef_module_ordinal_get(export_def);
        hipModule_t module = per_device_data->modules[module_ordinal];
        flatbuffers_string_t kernel_name =
            iree_hal_hip_ExportDef_kernel_name_get(export_def);
        hipFunction_t function = NULL;
        status = IREE_HIP_CALL_TO_STATUS(
            symbols, hipModuleGetFunction(&function, module, kernel_name),
            "hipModuleGetFunction");
        if (!iree_status_is_ok(status)) break;
        if (!function) {
          status = iree_make_status(IREE_STATUS_NOT_FOUND,
                                    "exports[%" PRIhsz
                                    "] kernel `%s` not found in modules[%u]",
                                    i, kernel_name, module_ordinal);
          break;
        }

        status = iree_hal_hip_function_attributes_verify(i, symbols, function,
                                                         limits);
        if (!iree_status_is_ok(status)) break;

        // Package required parameters for kernel launches for each entry
        // point.
        iree_hal_hip_kernel_params_t* kernel_info =
            &per_device_data->exports[i];
        kernel_info->function = function;
        const iree_hal_hip_BlockDims_t* block_dims =
            iree_hal_hip_ExportDef_block_dims_get(export_def);
        kernel_info->block_dims[0] = block_dims->x;
        kernel_info->block_dims[1] = block_dims->y;
        kernel_info->block_dims[2] = block_dims->z;
        kernel_info->constant_count =
            iree_hal_hip_ExportDef_constant_count_get(export_def);
        iree_hal_hip_BindingBits_vec_t binding_flags_vec =
            iree_hal_hip_ExportDef_binding_flags_get(export_def);
        kernel_info->binding_count =
            iree_hal_hip_BindingBits_vec_len(binding_flags_vec);

        if (iree_hal_hip_ExportDef_block_shared_memory_size_get(export_def) !=
            0) {
          status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                    "exports[%" PRIhsz
                                    "] for kernel `%s` specified non-zero "
                                    "deprecated field block_shared_memory_size "
                                    "in modules[%u]. Verify matching compiler "
                                    "and runtime versions.",
                                    i, kernel_name, module_ordinal);
          break;
        }

        IREE_TRACE({
          iree_hal_debug_export_info_t* export_info =
              (iree_hal_debug_export_info_t*)export_info_ptr;
          export_info_ptr += iree_hal_debug_copy_export_info(
              iree_hal_hip_ExportDef_debug_info_get(export_def), export_info);
          kernel_info->debug_info.function_name = export_info->function_name;
          kernel_info->debug_info.source_filename =
              export_info->source_filename;
          kernel_info->debug_info.source_line = export_info->source_line;
        });
      }
    }
    IREE_RETURN_IF_ERROR(
        IREE_HIP_CALL_TO_STATUS(symbols, hipCtxPopCurrent(NULL)));
  }

  if (iree_status_is_ok(status)) {
    *out_executable = (iree_hal_executable_t*)executable;
  } else {
    iree_hal_executable_destroy((iree_hal_executable_t*)executable);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_hip_native_executable_create(
    const iree_hal_hip_dynamic_symbols_t* symbols,
    iree_hal_hip_device_topology_t topology,
    const iree_hal_executable_params_t* executable_params,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable) {
  IREE_ASSERT_ARGUMENT(symbols);
  IREE_ASSERT_ARGUMENT(executable_params);
  IREE_ASSERT_ARGUMENT(out_executable);
  if (topology.count < 1) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "at least one device is required but none were provided");
  }
  IREE_TRACE_ZONE_BEGIN(z0);

  *out_executable = NULL;

  // TODO: move to the executable cache to avoid repeated queries.
  iree_hal_hip_limits_t limits = {0};
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_hip_query_limits(symbols, topology.devices[0].hip_device,
                                    &limits));

  // Dispatch to the appropriate loader based on executable format
  iree_string_view_t executable_format = executable_params->executable_format;

  if (iree_string_view_equal(executable_format, IREE_SV("FPIH"))) {
    // FPIH (Fat Binary) format
    IREE_TRACE_ZONE_END(z0);
    return iree_hal_hip_native_executable_create_fpih(
        symbols, topology, executable_params, &limits, host_allocator,
        out_executable);
  }

  // HSACO (flatbuffer) format
  IREE_TRACE_ZONE_END(z0);
  return iree_hal_hip_native_executable_create_flatbuffer(
      symbols, topology, executable_params, &limits, host_allocator,
      out_executable);
}

static void iree_hal_hip_native_executable_destroy(
    iree_hal_executable_t* base_executable) {
  iree_hal_hip_native_executable_t* executable =
      iree_hal_hip_native_executable_cast(base_executable);
  iree_allocator_t host_allocator = executable->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  for (iree_host_size_t i = 0; i < executable->num_devices; ++i) {
    const iree_hal_hip_native_executable_per_device_data_t* data =
        executable->per_device_data[i];

    // Free allocated parameter arrays for each export
    for (iree_host_size_t j = 0; j < data->export_count; ++j) {
      if (data->exports[j].parameters) {
        iree_allocator_free(host_allocator, data->exports[j].parameters);
      }
    }

    // Unload HIP modules
    for (iree_host_size_t j = 0; j < data->module_count; ++j) {
      if (data->modules[j]) {
        IREE_HIP_IGNORE_ERROR(executable->symbols,
                              hipModuleUnload(data->modules[j]));
      }
    }
  }

  // Free kernel info (including allocated names) from fat binary
  iree_hal_hip_free_kernel_info(host_allocator, executable->kernel_count,
                                 executable->kernels);

  iree_allocator_free(host_allocator, executable);

  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_hal_hip_native_executable_lookup_kernel_params(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_export_ordinal_t ordinal,
    iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_hip_kernel_params_t** out_params) {
  *out_params = NULL;
  iree_hal_hip_native_executable_t* executable =
      iree_hal_hip_native_executable_cast(base_executable);
  int device_ordinal = 0;
  if (queue_affinity) {
    device_ordinal = iree_math_count_trailing_zeros_u64(queue_affinity);
  }
  if (device_ordinal > executable->num_devices) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "affinity for non-existent queue was provided.");
  }

  const iree_hal_hip_native_executable_per_device_data_t* data =
      executable->per_device_data[device_ordinal];
  if (ordinal >= data->export_count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "export ordinal %d out of range; executable contains %" PRIhsz
        " exports",
        ordinal, data->export_count);
  }
  *out_params = &data->exports[ordinal];
  return iree_ok_status();
}

static iree_host_size_t iree_hal_hip_native_executable_export_count(
    iree_hal_executable_t* base_executable) {
  iree_hal_hip_native_executable_t* executable =
      iree_hal_hip_native_executable_cast(base_executable);
  iree_hal_hip_native_executable_t* exe =
      (iree_hal_hip_native_executable_t*)executable;

  return exe->per_device_data[0]->export_count;
}

static iree_status_t iree_hal_hip_native_executable_export_info(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    iree_hal_executable_export_info_t* out_info) {
  iree_hal_hip_native_executable_t* executable =
      iree_hal_hip_native_executable_cast(base_executable);
  if (export_ordinal >=
      iree_hal_hip_native_executable_export_count(base_executable)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "export ordinal %d out of range; executable "
        "contains %" PRIhsz " exports",
        export_ordinal,
        iree_hal_hip_native_executable_export_count(base_executable));
  }
  iree_hal_hip_native_executable_t* exe =
      (iree_hal_hip_native_executable_t*)executable;
  const iree_hal_hip_kernel_params_t* export =
      &exe->per_device_data[0]->exports[export_ordinal];
  out_info->binding_count = export->binding_count;
  out_info->name = export->function_name;
  out_info->constant_count = export->constant_count;
  out_info->workgroup_size[0] = export->block_dims[0];
  out_info->workgroup_size[1] = export->block_dims[1];
  out_info->workgroup_size[2] = export->block_dims[2];
  out_info->parameter_count = export->parameter_count;
  return iree_ok_status();
}

static iree_status_t iree_hal_hip_native_executable_export_parameters(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    iree_host_size_t capacity,
    iree_hal_executable_export_parameter_t* out_parameters) {
  iree_hal_hip_native_executable_t* executable =
      iree_hal_hip_native_executable_cast(base_executable);

  if (export_ordinal >=
      iree_hal_hip_native_executable_export_count(base_executable)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "export ordinal %d out of range; executable contains %" PRIhsz
        " exports",
        export_ordinal,
        iree_hal_hip_native_executable_export_count(base_executable));
  }

  // Get the export parameters from the first device (they're the same for all)
  const iree_hal_hip_kernel_params_t* export_params =
      &executable->per_device_data[0]->exports[export_ordinal];

  if (!export_params->parameters) {
    // No parameter information available
    return iree_ok_status();
  }

  // Copy parameters up to capacity
  iree_host_size_t copy_count = export_params->parameter_count < capacity
                                    ? export_params->parameter_count
                                    : capacity;
  for (iree_host_size_t i = 0; i < copy_count; ++i) {
    out_parameters[i] = export_params->parameters[i].export;
  }

  return iree_ok_status();
}

static iree_status_t iree_hal_hip_native_executable_lookup_export_by_name(
    iree_hal_executable_t* base_executable, iree_string_view_t name,
    iree_hal_executable_export_ordinal_t* out_export_ordinal) {
  iree_hal_hip_native_executable_t* executable =
      iree_hal_hip_native_executable_cast(base_executable);
  (void)executable;
  // TODO(hip): lookup the export ordinal by name.
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "reflection not implemented");
}

static iree_status_t iree_hal_hip_native_executable_lookup_global(
    iree_hal_executable_t* base_executable, iree_string_view_t name,
    iree_hal_queue_affinity_t queue_affinity, uint64_t* out_device_address,
    iree_device_size_t* out_size) {
  iree_hal_hip_native_executable_t* executable =
      iree_hal_hip_native_executable_cast(base_executable);

  *out_device_address = 0;
  if (out_size) *out_size = 0;

  int device_ordinal = 0;
  if (queue_affinity) {
    device_ordinal = iree_math_count_trailing_zeros_u64(queue_affinity);
  }
  if (device_ordinal >= (int)executable->num_devices) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "affinity for non-existent device was provided.");
  }

  const iree_hal_hip_native_executable_per_device_data_t* data =
      executable->per_device_data[device_ordinal];

  // Create a null-terminated copy of the name.
  char name_cstr[1024];
  if (name.size >= sizeof(name_cstr)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "global name too long: %.*s", (int)name.size,
                            name.data);
  }
  memcpy(name_cstr, name.data, name.size);
  name_cstr[name.size] = '\0';

  // Try to find the global in each module.
  for (iree_host_size_t i = 0; i < data->module_count; ++i) {
    hipModule_t module = data->modules[i];

    hipDeviceptr_t device_ptr = 0;
    size_t size = 0;
    hipError_t result =
        executable->symbols->hipModuleGetGlobal(&device_ptr, &size, module,
                                                 name_cstr);
    if (result == hipSuccess) {
      *out_device_address = (uint64_t)device_ptr;
      if (out_size) *out_size = size;
      return iree_ok_status();
    }
    // If not found in this module, continue to the next one.
    // hipErrorNotFound means the symbol wasn't in this module.
  }

  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "global variable '%.*s' not found in executable",
                          (int)name.size, name.data);
}

static const iree_hal_executable_vtable_t
    iree_hal_hip_native_executable_vtable = {
        .destroy = iree_hal_hip_native_executable_destroy,
        .export_count = iree_hal_hip_native_executable_export_count,
        .export_info = iree_hal_hip_native_executable_export_info,
        .export_parameters = iree_hal_hip_native_executable_export_parameters,
        .lookup_export_by_name =
            iree_hal_hip_native_executable_lookup_export_by_name,
        .lookup_global = iree_hal_hip_native_executable_lookup_global,
};
