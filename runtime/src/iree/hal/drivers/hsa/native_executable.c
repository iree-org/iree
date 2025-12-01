// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/hsa/native_executable.h"

#include <stddef.h>

#include "iree/base/api.h"
#include "iree/base/internal/math.h"
#include "iree/hal/drivers/hsa/dynamic_symbols.h"
#include "iree/hal/drivers/hsa/native_executable_hsaf.h"
#include "iree/hal/drivers/hsa/status_util.h"
#include "iree/hal/utils/executable_debug_info.h"
#include "iree/hal/utils/executable_header.h"

// flatcc schemas:
#include "iree/base/internal/flatcc/parsing.h"
#include "iree/schemas/executable_debug_info_reader.h"
#include "iree/schemas/executable_debug_info_verifier.h"
#include "iree/schemas/hip_executable_def_reader.h"
#include "iree/schemas/hip_executable_def_verifier.h"

typedef struct iree_hal_hsa_native_executable_per_device_data_t {
  // Loaded HSA executables.
  iree_host_size_t executable_count;
  hsa_executable_t* executables;

  // Exported kernels referencing the loaded executables.
  iree_host_size_t export_count;
  iree_hal_hsa_kernel_params_t exports[];
} iree_hal_hsa_native_executable_per_device_data_t;

typedef struct iree_hal_hsa_native_executable_t {
  // Abstract resource used for injecting reference counting and vtable;
  // must be at offset 0.
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;

  const iree_hal_hsa_dynamic_symbols_t* symbols;

  iree_host_size_t num_devices;
  iree_hal_hsa_native_executable_per_device_data_t* per_device_data[];
} iree_hal_hsa_native_executable_t;

static const iree_hal_executable_vtable_t iree_hal_hsa_native_executable_vtable;

static iree_hal_hsa_native_executable_t* iree_hal_hsa_native_executable_cast(
    iree_hal_executable_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_hsa_native_executable_vtable);
  return (iree_hal_hsa_native_executable_t*)base_value;
}

iree_status_t iree_hal_hsa_native_executable_infer_format(
    iree_const_byte_span_t executable_data,
    iree_host_size_t executable_format_capacity, char* executable_format,
    iree_host_size_t* out_inferred_size) {
  // Read the size prefix (with unsafe inference if size is unknown).
  const bool unsafe_infer_size = (executable_data.data_length == 0);
  iree_const_byte_span_t contained_data = iree_const_byte_span_empty();

  iree_status_t native_hsa = iree_hal_hsa_read_native_header(
      executable_data, unsafe_infer_size, &contained_data);
  if (iree_status_is_ok(native_hsa)) {
    // Successfully read as native HSA executable (fat binary format).
    iree_string_view_t format = IREE_SV("FPIH");  // Use same format as HIP
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

iree_status_t iree_hal_hsa_native_executable_create(
    const iree_hal_hsa_dynamic_symbols_t* symbols,
    iree_hal_hsa_device_topology_t topology,
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

  // Read and strip the flatbuffer header prefix.
  iree_const_byte_span_t executable_flatbuffer = iree_const_byte_span_empty();
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hal_read_executable_flatbuffer_header(
          executable_params->executable_data, /*unsafe_infer_size=*/false,
          iree_hal_hip_ExecutableDef_file_identifier, &executable_flatbuffer));

  // Verify the flatbuffer structure.
  if (!iree_hal_hip_ExecutableDef_verify_as_root(executable_flatbuffer.data,
                                                 executable_flatbuffer.data_length)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "failed to verify executable flatbuffer structure");
  }

  iree_hal_hip_ExecutableDef_table_t executable_def =
      iree_hal_hip_ExecutableDef_as_root(executable_flatbuffer.data);

  iree_hal_hip_ModuleDef_vec_t modules_vec =
      iree_hal_hip_ExecutableDef_modules_get(executable_def);
  iree_host_size_t module_count = iree_hal_hip_ModuleDef_vec_len(modules_vec);
  iree_hal_hip_ExportDef_vec_t exports_vec =
      iree_hal_hip_ExecutableDef_exports_get(executable_def);
  iree_host_size_t export_count = iree_hal_hip_ExportDef_vec_len(exports_vec);

  // Calculate the total number of characters across all entry point names.
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
  iree_hal_hsa_native_executable_t* executable = NULL;
  iree_host_size_t native_executable_device_info_size =
      sizeof(*executable->per_device_data[0]) +
      module_count * sizeof(executable->per_device_data[0]->executables[0]) +
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
  iree_hal_resource_initialize(&iree_hal_hsa_native_executable_vtable,
                               &executable->resource);
  executable->host_allocator = host_allocator;
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
        (iree_hal_hsa_native_executable_per_device_data_t*)(per_device_data_location +
                                                            per_device_data_size +
                                                            native_executable_device_info_size_offset);
  }

  // Publish any embedded source files to the tracing infrastructure.
  iree_hal_debug_publish_source_files(
      iree_hal_hip_ExecutableDef_source_files_get(executable_def));

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t j = 0; j < topology.count && iree_status_is_ok(status);
       ++j) {
    iree_hal_hsa_native_executable_per_device_data_t* per_device_data =
        executable->per_device_data[j];
    hsa_agent_t agent = topology.devices[j].agent;

    per_device_data->executable_count = module_count;
    per_device_data->executables =
        (hsa_executable_t*)((uint8_t*)per_device_data + sizeof(*per_device_data) +
                            (export_count * sizeof(per_device_data->exports[0])));
    per_device_data->export_count = export_count;
    IREE_TRACE(uint8_t* export_info_ptr =
                   ((uint8_t*)per_device_data->executables +
                    module_count * sizeof(per_device_data->executables[0])));

    // Load each module first so that exports can reference them.
    for (iree_host_size_t i = 0; i < module_count && iree_status_is_ok(status); ++i) {
      iree_hal_hip_ModuleDef_table_t module_def =
          iree_hal_hip_ModuleDef_vec_at(modules_vec, i);

      flatbuffers_string_t hsaco_image =
          iree_hal_hip_ModuleDef_hsaco_image_get(module_def);
      size_t hsaco_size = flatbuffers_string_len(hsaco_image);

      // Create code object reader.
      hsa_code_object_reader_t code_reader;
      status = IREE_HSA_CALL_TO_STATUS(
          symbols,
          hsa_code_object_reader_create_from_memory(hsaco_image, hsaco_size,
                                                    &code_reader),
          "hsa_code_object_reader_create_from_memory");
      if (!iree_status_is_ok(status)) break;

      // Create executable.
      hsa_executable_t hsa_executable;
      status = IREE_HSA_CALL_TO_STATUS(
          symbols,
          hsa_executable_create_alt(HSA_PROFILE_FULL,
                                    HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
                                    NULL, &hsa_executable),
          "hsa_executable_create_alt");
      if (!iree_status_is_ok(status)) {
        symbols->hsa_code_object_reader_destroy(code_reader);
        break;
      }

      // Load code object.
      status = IREE_HSA_CALL_TO_STATUS(
          symbols,
          hsa_executable_load_agent_code_object(hsa_executable, agent,
                                                code_reader, NULL, NULL),
          "hsa_executable_load_agent_code_object");
      symbols->hsa_code_object_reader_destroy(code_reader);
      if (!iree_status_is_ok(status)) {
        symbols->hsa_executable_destroy(hsa_executable);
        break;
      }

      // Freeze executable.
      status = IREE_HSA_CALL_TO_STATUS(
          symbols, hsa_executable_freeze(hsa_executable, NULL),
          "hsa_executable_freeze");
      if (!iree_status_is_ok(status)) {
        symbols->hsa_executable_destroy(hsa_executable);
        break;
      }

      per_device_data->executables[i] = hsa_executable;
    }

    // Look up kernel symbols and populate export info.
    for (iree_host_size_t i = 0; i < export_count && iree_status_is_ok(status); ++i) {
      iree_hal_hip_ExportDef_table_t export_def =
          iree_hal_hip_ExportDef_vec_at(exports_vec, i);

      uint32_t module_ordinal =
          iree_hal_hip_ExportDef_module_ordinal_get(export_def);
      hsa_executable_t hsa_executable = per_device_data->executables[module_ordinal];
      flatbuffers_string_t kernel_name =
          iree_hal_hip_ExportDef_kernel_name_get(export_def);

      // Look up kernel symbol.
      hsa_executable_symbol_t kernel_symbol;
      status = IREE_HSA_CALL_TO_STATUS(
          symbols,
          hsa_executable_get_symbol_by_name(hsa_executable, kernel_name,
                                            &agent, &kernel_symbol),
          "hsa_executable_get_symbol_by_name");
      if (!iree_status_is_ok(status)) {
        status = iree_status_annotate_f(
            status, "failed to find kernel '%s' in executable", kernel_name);
        break;
      }

      // Get kernel object (entry point address).
      uint64_t kernel_object = 0;
      status = IREE_HSA_CALL_TO_STATUS(
          symbols,
          hsa_executable_symbol_get_info(
              kernel_symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
              &kernel_object),
          "hsa_executable_symbol_get_info(KERNEL_OBJECT)");
      if (!iree_status_is_ok(status)) break;

      // Get kernarg segment size.
      uint32_t kernarg_segment_size = 0;
      status = IREE_HSA_CALL_TO_STATUS(
          symbols,
          hsa_executable_symbol_get_info(
              kernel_symbol,
              HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE,
              &kernarg_segment_size),
          "hsa_executable_symbol_get_info(KERNARG_SEGMENT_SIZE)");
      if (!iree_status_is_ok(status)) break;

      // Get group segment size.
      uint32_t group_segment_size = 0;
      status = IREE_HSA_CALL_TO_STATUS(
          symbols,
          hsa_executable_symbol_get_info(
              kernel_symbol,
              HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE,
              &group_segment_size),
          "hsa_executable_symbol_get_info(GROUP_SEGMENT_SIZE)");
      if (!iree_status_is_ok(status)) break;

      // Get private segment size.
      uint32_t private_segment_size = 0;
      status = IREE_HSA_CALL_TO_STATUS(
          symbols,
          hsa_executable_symbol_get_info(
              kernel_symbol,
              HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE,
              &private_segment_size),
          "hsa_executable_symbol_get_info(PRIVATE_SEGMENT_SIZE)");
      if (!iree_status_is_ok(status)) break;

      // Package required parameters for kernel launches.
      iree_hal_hsa_kernel_params_t* kernel_info = &per_device_data->exports[i];
      kernel_info->kernel_object = kernel_object;
      kernel_info->kernarg_segment_size = kernarg_segment_size;
      kernel_info->group_segment_size = group_segment_size;
      kernel_info->private_segment_size = private_segment_size;

      const iree_hal_hip_BlockDims_t* block_dims =
          iree_hal_hip_ExportDef_block_dims_get(export_def);
      if (block_dims) {
        kernel_info->block_dims[0] = block_dims->x;
        kernel_info->block_dims[1] = block_dims->y;
        kernel_info->block_dims[2] = block_dims->z;
      } else {
        kernel_info->block_dims[0] = 1;
        kernel_info->block_dims[1] = 1;
        kernel_info->block_dims[2] = 1;
      }

      kernel_info->constant_count =
          iree_hal_hip_ExportDef_constant_count_get(export_def);
      iree_hal_hip_BindingBits_vec_t binding_flags_vec =
          iree_hal_hip_ExportDef_binding_flags_get(export_def);
      kernel_info->binding_count =
          iree_hal_hip_BindingBits_vec_len(binding_flags_vec);

      IREE_TRACE({
        iree_hal_debug_export_info_t* export_info =
            (iree_hal_debug_export_info_t*)export_info_ptr;
        export_info_ptr += iree_hal_debug_copy_export_info(
            iree_hal_hip_ExportDef_debug_info_get(export_def), export_info);
        kernel_info->debug_info.function_name = export_info->function_name;
        kernel_info->debug_info.source_filename = export_info->source_filename;
        kernel_info->debug_info.source_line = export_info->source_line;
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

static void iree_hal_hsa_native_executable_destroy(
    iree_hal_executable_t* base_executable) {
  iree_hal_hsa_native_executable_t* executable =
      iree_hal_hsa_native_executable_cast(base_executable);
  iree_allocator_t host_allocator = executable->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  for (iree_host_size_t i = 0; i < executable->num_devices; ++i) {
    const iree_hal_hsa_native_executable_per_device_data_t* data =
        executable->per_device_data[i];
    for (iree_host_size_t j = 0; j < data->executable_count; ++j) {
      if (data->executables[j].handle) {
        IREE_HSA_IGNORE_ERROR(executable->symbols,
                              hsa_executable_destroy(data->executables[j]));
      }
    }
  }

  iree_allocator_free(host_allocator, executable);

  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_hal_hsa_native_executable_lookup_kernel_params(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_export_ordinal_t ordinal,
    iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_hsa_kernel_params_t** out_params) {
  *out_params = NULL;
  iree_hal_hsa_native_executable_t* executable =
      iree_hal_hsa_native_executable_cast(base_executable);
  int device_ordinal = 0;
  if (queue_affinity) {
    device_ordinal = iree_math_count_trailing_zeros_u64(queue_affinity);
  }
  if (device_ordinal > (int)executable->num_devices) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "affinity for non-existent queue was provided.");
  }

  const iree_hal_hsa_native_executable_per_device_data_t* data =
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

static iree_host_size_t iree_hal_hsa_native_executable_export_count(
    iree_hal_executable_t* base_executable) {
  iree_hal_hsa_native_executable_t* executable =
      iree_hal_hsa_native_executable_cast(base_executable);
  (void)executable;
  return 0;
}

static iree_status_t iree_hal_hsa_native_executable_export_info(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    iree_hal_executable_export_info_t* out_info) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "reflection not implemented");
}

static iree_status_t iree_hal_hsa_native_executable_export_parameters(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    iree_host_size_t capacity,
    iree_hal_executable_export_parameter_t* out_parameters) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "parameter reflection not implemented");
}

static iree_status_t iree_hal_hsa_native_executable_lookup_export_by_name(
    iree_hal_executable_t* base_executable, iree_string_view_t name,
    iree_hal_executable_export_ordinal_t* out_export_ordinal) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "reflection not implemented");
}

static const iree_hal_executable_vtable_t
    iree_hal_hsa_native_executable_vtable = {
        .destroy = iree_hal_hsa_native_executable_destroy,
        .export_count = iree_hal_hsa_native_executable_export_count,
        .export_info = iree_hal_hsa_native_executable_export_info,
        .export_parameters = iree_hal_hsa_native_executable_export_parameters,
        .lookup_export_by_name =
            iree_hal_hsa_native_executable_lookup_export_by_name,
};

