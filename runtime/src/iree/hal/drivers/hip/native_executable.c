// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/hip/native_executable.h"

#include <stddef.h>

#include "iree/base/api.h"
#include "iree/base/internal/math.h"
#include "iree/hal/drivers/hip/context_util.h"
#include "iree/hal/drivers/hip/dynamic_symbols.h"
#include "iree/hal/drivers/hip/hip_buffer.h"
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
  // HIP context owning all modules in this per-device table.
  hipCtx_t hip_context;

  // Number of loaded HIP modules.
  iree_host_size_t module_count;
  // Loaded HIP modules.
  hipModule_t* modules;

  // Number of exported kernels referencing the loaded modules.
  iree_host_size_t export_count;
  // Exported kernels referencing the loaded modules.
  iree_hal_hip_kernel_params_t exports[];
} iree_hal_hip_native_executable_per_device_data_t;

typedef struct iree_hal_hip_native_executable_t {
  // Abstract resource used for injecting reference counting and vtable;
  // must be at offset 0.
  iree_hal_resource_t resource;
  // Host allocator used for executable lifetime.
  iree_allocator_t host_allocator;

  // Borrowed HAL device used for buffer placement metadata.
  iree_hal_device_t* device;
  // Borrowed HIP dynamic symbols used for module and global lookup.
  const iree_hal_hip_dynamic_symbols_t* symbols;

  // Number of HIP devices this executable was loaded onto.
  iree_host_size_t num_devices;
  // Per-device module and export tables.
  iree_hal_hip_native_executable_per_device_data_t* per_device_data[];
} iree_hal_hip_native_executable_t;

static const iree_hal_executable_vtable_t iree_hal_hip_native_executable_vtable;

static iree_hal_hip_native_executable_t* iree_hal_hip_native_executable_cast(
    iree_hal_executable_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_hip_native_executable_vtable);
  return (iree_hal_hip_native_executable_t*)base_value;
}

typedef struct iree_hal_hip_limits_t {
  uint32_t max_block_dims[3];
  uint32_t max_block_shared_memory_size;
} iree_hal_hip_limits_t;
static iree_status_t iree_hal_hip_query_limits(
    const iree_hal_hip_dynamic_symbols_t* symbols, hipDevice_t device,
    iree_hal_hip_limits_t* out_limits) {
  memset(out_limits, 0, sizeof(*out_limits));

  IREE_HIP_RETURN_IF_ERROR(
      symbols,
      hipDeviceGetAttribute((int32_t*)&out_limits->max_block_dims[0],
                            hipDeviceAttributeMaxBlockDimX, device),
      "hipDeviceGetAttribute");
  IREE_HIP_RETURN_IF_ERROR(
      symbols,
      hipDeviceGetAttribute((int32_t*)&out_limits->max_block_dims[1],
                            hipDeviceAttributeMaxBlockDimY, device),
      "hipDeviceGetAttribute");
  IREE_HIP_RETURN_IF_ERROR(
      symbols,
      hipDeviceGetAttribute((int32_t*)&out_limits->max_block_dims[2],
                            hipDeviceAttributeMaxBlockDimZ, device),
      "hipDeviceGetAttribute");

  IREE_HIP_RETURN_IF_ERROR(
      symbols,
      hipDeviceGetAttribute((int32_t*)&out_limits->max_block_shared_memory_size,
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
  iree_const_byte_span_t flatbuffer_data = iree_const_byte_span_empty();
  IREE_RETURN_IF_ERROR(iree_hal_read_executable_flatbuffer_header(
      executable_data, unsafe_infer_size,
      iree_hal_hip_ExecutableDef_file_identifier, &flatbuffer_data));

  // Verify the flatbuffer structure.
  if (!iree_hal_hip_ExecutableDef_verify_as_root(flatbuffer_data.data,
                                                 flatbuffer_data.data_length)) {
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
      sizeof(iree_flatbuffer_file_header_t) + flatbuffer_data.data_length;
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

iree_status_t iree_hal_hip_native_executable_create(
    iree_hal_device_t* device, const iree_hal_hip_dynamic_symbols_t* symbols,
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

  // Read and strip the flatbuffer header prefix.
  iree_const_byte_span_t executable_flatbuffer = iree_const_byte_span_empty();
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hal_read_executable_flatbuffer_header(
          executable_params->executable_data, /*unsafe_infer_size=*/false,
          iree_hal_hip_ExecutableDef_file_identifier, &executable_flatbuffer));

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_hip_native_executable_flatbuffer_verify(
              executable_flatbuffer, &limits));

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
  executable->device = device;
  executable->symbols = symbols;
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

    per_device_data->hip_context = topology.devices[j].hip_context;
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
                                                         &limits);
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

static void iree_hal_hip_native_executable_destroy(
    iree_hal_executable_t* base_executable) {
  iree_hal_hip_native_executable_t* executable =
      iree_hal_hip_native_executable_cast(base_executable);
  iree_allocator_t host_allocator = executable->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  for (iree_host_size_t i = 0; i < executable->num_devices; ++i) {
    const iree_hal_hip_native_executable_per_device_data_t* data =
        executable->per_device_data[i];
    for (iree_host_size_t j = 0; j < data->module_count; ++j) {
      if (data->modules[j]) {
        IREE_HIP_IGNORE_ERROR(executable->symbols,
                              hipModuleUnload(data->modules[j]));
      }
    }
  }

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
  // TODO(hip): return the total number of exports in the executable.
  (void)executable;
  return 0;
}

static iree_status_t iree_hal_hip_native_executable_export_info(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    iree_hal_executable_export_info_t* out_info) {
  iree_hal_hip_native_executable_t* executable =
      iree_hal_hip_native_executable_cast(base_executable);
  (void)executable;
  // TODO(hip): return export information from kernel metadata.
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "reflection not implemented");
}

static iree_status_t iree_hal_hip_native_executable_export_parameters(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    iree_host_size_t capacity,
    iree_hal_executable_export_parameter_t* out_parameters) {
  iree_hal_hip_native_executable_t* executable =
      iree_hal_hip_native_executable_cast(base_executable);
  (void)executable;
  // TODO(hip): return export parameter information from kernel metadata.
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "parameter reflection not implemented");
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

#define IREE_HAL_HIP_MAX_STACK_GLOBAL_NAME_LENGTH ((iree_host_size_t)(4 * 1024))

static bool iree_hal_hip_native_executable_is_global_not_found(
    hipError_t result) {
  return result == hipErrorNotFound ||
         result == hipErrorSharedObjectSymbolNotFound;
}

static iree_status_t iree_hal_hip_native_executable_select_global_device(
    const iree_hal_hip_native_executable_t* executable,
    iree_hal_queue_affinity_t queue_affinity,
    iree_host_size_t* out_device_ordinal,
    iree_hal_queue_affinity_t* out_queue_affinity) {
  *out_device_ordinal = 0;
  *out_queue_affinity = 0;

  iree_hal_queue_affinity_t available_affinity =
      executable->num_devices == IREE_HAL_MAX_QUEUES
          ? IREE_HAL_QUEUE_AFFINITY_ANY
          : (((iree_hal_queue_affinity_t)1) << executable->num_devices) - 1;
  iree_hal_queue_affinity_t selected_affinity = queue_affinity;
  if (iree_hal_queue_affinity_is_empty(selected_affinity) ||
      iree_hal_queue_affinity_is_any(selected_affinity)) {
    selected_affinity = available_affinity;
  } else {
    iree_hal_queue_affinity_and_into(selected_affinity, available_affinity);
  }
  if (iree_hal_queue_affinity_is_empty(selected_affinity)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "global lookup queue affinity 0x%" PRIx64
                            " does not select a loaded HIP device",
                            queue_affinity);
  }

  iree_host_size_t device_ordinal =
      iree_hal_queue_affinity_find_first_set(selected_affinity);
  *out_device_ordinal = device_ordinal;
  *out_queue_affinity = ((iree_hal_queue_affinity_t)1) << device_ordinal;
  return iree_ok_status();
}

static void iree_hal_hip_native_executable_global_buffer_release(
    void* user_data, iree_hal_buffer_t* buffer) {
  (void)buffer;
  iree_hal_executable_release((iree_hal_executable_t*)user_data);
}

static iree_status_t iree_hal_hip_native_executable_lookup_global_by_name(
    iree_hal_executable_t* base_executable, iree_string_view_t name,
    iree_hal_queue_affinity_t queue_affinity, iree_hal_buffer_t** out_buffer) {
  iree_hal_hip_native_executable_t* executable =
      iree_hal_hip_native_executable_cast(base_executable);
  *out_buffer = NULL;

  if (iree_string_view_is_empty(name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "executable global name is empty");
  }
  if (name.size > IREE_HAL_HIP_MAX_STACK_GLOBAL_NAME_LENGTH) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "executable global name `%.*s` exceeds maximum length %" PRIhsz,
        (int)name.size, name.data, IREE_HAL_HIP_MAX_STACK_GLOBAL_NAME_LENGTH);
  }

  iree_host_size_t device_ordinal = 0;
  iree_hal_queue_affinity_t selected_queue_affinity = 0;
  IREE_RETURN_IF_ERROR(iree_hal_hip_native_executable_select_global_device(
      executable, queue_affinity, &device_ordinal, &selected_queue_affinity));

  const iree_hal_hip_native_executable_per_device_data_t* per_device_data =
      executable->per_device_data[device_ordinal];
  IREE_RETURN_IF_ERROR(iree_hal_hip_set_context(executable->symbols,
                                                per_device_data->hip_context),
                       "setting HIP context for executable global lookup");

  char* global_name = (char*)iree_alloca(name.size + 1);
  memcpy(global_name, name.data, name.size);
  global_name[name.size] = 0;

  hipError_t terminal_result = hipErrorNotFound;
  hipDeviceptr_t global_device_ptr = 0;
  size_t global_size = 0;
  for (iree_host_size_t module_ordinal = 0;
       module_ordinal < per_device_data->module_count; ++module_ordinal) {
    terminal_result = executable->symbols->hipModuleGetGlobal(
        &global_device_ptr, &global_size,
        per_device_data->modules[module_ordinal], global_name);
    if (terminal_result == hipSuccess) break;
    if (!iree_hal_hip_native_executable_is_global_not_found(terminal_result)) {
      return IREE_HIP_RESULT_TO_STATUS(executable->symbols, terminal_result);
    }
  }
  if (terminal_result != hipSuccess) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "executable global `%.*s` not found",
                            (int)name.size, name.data);
  }

  iree_hal_buffer_placement_t placement = {
      .device = executable->device,
      .queue_affinity = selected_queue_affinity,
      .flags = IREE_HAL_BUFFER_PLACEMENT_FLAG_NONE,
  };
  iree_hal_buffer_release_callback_t release_callback = {
      .fn = iree_hal_hip_native_executable_global_buffer_release,
      .user_data = base_executable,
  };
  iree_hal_executable_retain(base_executable);
  iree_status_t status = iree_hal_hip_buffer_wrap(
      placement, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
      IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
      IREE_HAL_BUFFER_USAGE_DEFAULT, global_size, /*byte_offset=*/0,
      global_size, IREE_HAL_HIP_BUFFER_TYPE_EXTERNAL, global_device_ptr,
      /*host_ptr=*/NULL, release_callback, executable->host_allocator,
      out_buffer);
  if (!iree_status_is_ok(status)) {
    iree_hal_executable_release(base_executable);
  }
  return status;
}

static const iree_hal_executable_vtable_t
    iree_hal_hip_native_executable_vtable = {
        .destroy = iree_hal_hip_native_executable_destroy,
        .export_count = iree_hal_hip_native_executable_export_count,
        .export_info = iree_hal_hip_native_executable_export_info,
        .export_parameters = iree_hal_hip_native_executable_export_parameters,
        .lookup_export_by_name =
            iree_hal_hip_native_executable_lookup_export_by_name,
        .lookup_global_by_name =
            iree_hal_hip_native_executable_lookup_global_by_name,
};
