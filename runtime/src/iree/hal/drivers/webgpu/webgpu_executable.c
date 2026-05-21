// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/webgpu/webgpu_executable.h"

#include "iree/base/internal/flatcc/parsing.h"
#include "iree/hal/drivers/webgpu/webgpu_imports.h"
#include "iree/hal/utils/executable_header.h"
#include "iree/schemas/webgpu_executable_def_reader.h"
#include "iree/schemas/webgpu_executable_def_verifier.h"

//===----------------------------------------------------------------------===//
// Per-function entry
//===----------------------------------------------------------------------===//

typedef struct iree_hal_webgpu_executable_entry_t {
  // Bridge handle for the compute pipeline.
  iree_hal_webgpu_handle_t pipeline_handle;
  // Bridge handle for bind group layout 0 of the compute pipeline.
  iree_hal_webgpu_handle_t bind_group_layout_handle;
  // Function name used by the command buffer and reflection APIs.
  iree_string_view_t name;
  // Static workgroup size for the function.
  uint32_t workgroup_size[3];
  // Number of resource bindings declared by the function.
  uint16_t binding_count;
  // Number of push constants declared by the executable.
  uint16_t constant_count;
} iree_hal_webgpu_executable_entry_t;

//===----------------------------------------------------------------------===//
// iree_hal_webgpu_executable_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_webgpu_executable_t {
  // Abstract resource header for HAL lifetime management.
  iree_hal_resource_t resource;
  // Host allocator used to allocate the executable and entry storage.
  iree_allocator_t host_allocator;
  // Number of functions in the executable.
  iree_host_size_t function_count;
  // Per-function entry table with |function_count| entries.
  iree_hal_webgpu_executable_entry_t entries[];
} iree_hal_webgpu_executable_t;

static const iree_hal_executable_vtable_t iree_hal_webgpu_executable_vtable;

static iree_hal_webgpu_executable_t* iree_hal_webgpu_executable_cast(
    iree_hal_executable_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_webgpu_executable_vtable);
  return (iree_hal_webgpu_executable_t*)base_value;
}

static iree_status_t iree_hal_webgpu_executable_calculate_name_storage_size(
    iree_hal_webgpu_ExportDef_vec_t exports_vec, iree_host_size_t export_count,
    iree_host_size_t* out_name_storage_size) {
  iree_host_size_t name_storage_size = 0;
  for (iree_host_size_t i = 0; i < export_count; ++i) {
    iree_hal_webgpu_ExportDef_table_t export_def =
        iree_hal_webgpu_ExportDef_vec_at(exports_vec, i);
    flatbuffers_string_t entry_point =
        iree_hal_webgpu_ExportDef_entry_point_get(export_def);
    iree_host_size_t entry_point_length =
        entry_point ? flatbuffers_string_len(entry_point) : 0;
    if (entry_point_length == 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "exports[%" PRIhsz "] has no entry point name",
                              i);
    }
    if (!iree_host_size_checked_add(name_storage_size, entry_point_length + 1,
                                    &name_storage_size)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "WebGPU executable export name storage exceeds "
                              "host size limits");
    }
  }
  *out_name_storage_size = name_storage_size;
  return iree_ok_status();
}

static iree_status_t iree_hal_webgpu_executable_initialize_export(
    iree_hal_webgpu_handle_t device_handle,
    iree_hal_webgpu_ShaderModuleDef_vec_t shader_modules_vec,
    iree_host_size_t shader_module_count, iree_host_size_t export_ordinal,
    iree_hal_webgpu_ExportDef_table_t export_def, char** inout_name_storage,
    iree_hal_webgpu_executable_entry_t* out_entry) {
  uint32_t shader_module_ordinal =
      iree_hal_webgpu_ExportDef_shader_module_ordinal_get(export_def);
  if (shader_module_ordinal >= shader_module_count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "exports[%" PRIhsz "] references shader module %u but only %" PRIhsz
        " shader modules are present",
        export_ordinal, shader_module_ordinal, shader_module_count);
  }

  iree_hal_webgpu_ShaderModuleDef_table_t shader_module_def =
      iree_hal_webgpu_ShaderModuleDef_vec_at(shader_modules_vec,
                                             shader_module_ordinal);
  flatbuffers_string_t wgsl_source =
      iree_hal_webgpu_ShaderModuleDef_wgsl_source_get(shader_module_def);
  iree_host_size_t wgsl_source_length =
      wgsl_source ? flatbuffers_string_len(wgsl_source) : 0;
  if (wgsl_source_length == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "shader_modules[%u] has no WGSL source",
                            shader_module_ordinal);
  }

  flatbuffers_string_t entry_point =
      iree_hal_webgpu_ExportDef_entry_point_get(export_def);
  iree_host_size_t entry_point_length =
      entry_point ? flatbuffers_string_len(entry_point) : 0;
  if (entry_point_length == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "exports[%" PRIhsz "] has no entry point name",
                            export_ordinal);
  }
  memcpy(*inout_name_storage, entry_point, entry_point_length);
  (*inout_name_storage)[entry_point_length] = '\0';
  out_entry->name =
      iree_make_string_view(*inout_name_storage, entry_point_length);
  *inout_name_storage += entry_point_length + 1;

  uint32_t constant_count =
      iree_hal_webgpu_ExportDef_constant_count_get(export_def);
  if (constant_count > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "exports[%" PRIhsz
                            "] declares %u constants, exceeding the HAL "
                            "reflection limit of %u",
                            export_ordinal, constant_count, UINT16_MAX);
  }
  out_entry->constant_count = (uint16_t)constant_count;

  iree_hal_webgpu_BindingBits_vec_t binding_flags_vec =
      iree_hal_webgpu_ExportDef_binding_flags_get(export_def);
  iree_host_size_t binding_count =
      iree_hal_webgpu_BindingBits_vec_len(binding_flags_vec);
  if (binding_count > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "exports[%" PRIhsz "] declares %" PRIhsz
                            " bindings, exceeding the HAL reflection limit "
                            "of %u",
                            export_ordinal, binding_count, UINT16_MAX);
  }
  out_entry->binding_count = (uint16_t)binding_count;

  const iree_hal_webgpu_WorkgroupSize_t* workgroup_size =
      iree_hal_webgpu_ExportDef_workgroup_size_get(export_def);
  if (!workgroup_size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "exports[%" PRIhsz "] has no static workgroup size",
                            export_ordinal);
  }
  out_entry->workgroup_size[0] = workgroup_size->x;
  out_entry->workgroup_size[1] = workgroup_size->y;
  out_entry->workgroup_size[2] = workgroup_size->z;

  out_entry->pipeline_handle =
      iree_hal_webgpu_import_device_create_compute_pipeline(
          device_handle, /*layout_handle=*/0, (uint32_t)(uintptr_t)wgsl_source,
          (uint32_t)wgsl_source_length,
          (uint32_t)(uintptr_t)out_entry->name.data,
          (uint32_t)out_entry->name.size);
  if (out_entry->pipeline_handle == 0) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "failed to create WebGPU compute pipeline for entry point '%.*s'",
        (int)out_entry->name.size, out_entry->name.data);
  }

  out_entry->bind_group_layout_handle =
      iree_hal_webgpu_import_pipeline_get_bind_group_layout(
          out_entry->pipeline_handle, /*index=*/0);
  if (out_entry->bind_group_layout_handle == 0) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "failed to query WebGPU bind group layout for entry point '%.*s'",
        (int)out_entry->name.size, out_entry->name.data);
  }

  return iree_ok_status();
}

iree_status_t iree_hal_webgpu_executable_create(
    iree_hal_webgpu_handle_t device_handle,
    const iree_hal_executable_params_t* executable_params,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable) {
  IREE_ASSERT_ARGUMENT(executable_params);
  IREE_ASSERT_ARGUMENT(out_executable);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_executable = NULL;

  if (!iree_string_view_equal(executable_params->executable_format,
                              IREE_SV("webgpu-wgsl-fb"))) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INCOMPATIBLE,
                            "unsupported WebGPU executable format '%.*s'",
                            (int)executable_params->executable_format.size,
                            executable_params->executable_format.data);
  }

  iree_const_byte_span_t flatbuffer_data = iree_const_byte_span_empty();
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_read_executable_flatbuffer_header(
              executable_params->executable_data,
              /*unsafe_infer_size=*/false,
              iree_hal_webgpu_ExecutableDef_file_identifier, &flatbuffer_data));

  int verify_ret = iree_hal_webgpu_ExecutableDef_verify_as_root(
      flatbuffer_data.data, flatbuffer_data.data_length);
  if (verify_ret != flatcc_verify_ok) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "WebGPU executable flatbuffer verification failed: "
                            "%s",
                            flatcc_verify_error_string(verify_ret));
  }

  iree_hal_webgpu_ExecutableDef_table_t executable_def =
      iree_hal_webgpu_ExecutableDef_as_root(flatbuffer_data.data);
  iree_hal_webgpu_ShaderModuleDef_vec_t shader_modules_vec =
      iree_hal_webgpu_ExecutableDef_shader_modules_get(executable_def);
  iree_hal_webgpu_ExportDef_vec_t exports_vec =
      iree_hal_webgpu_ExecutableDef_exports_get(executable_def);
  iree_host_size_t shader_module_count =
      iree_hal_webgpu_ShaderModuleDef_vec_len(shader_modules_vec);
  iree_host_size_t export_count =
      iree_hal_webgpu_ExportDef_vec_len(exports_vec);
  if (export_count == 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "WebGPU executable has no exports");
  }

  iree_host_size_t name_storage_size = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_webgpu_executable_calculate_name_storage_size(
              exports_vec, export_count, &name_storage_size));

  iree_host_size_t entry_storage_size = 0;
  iree_host_size_t total_size = 0;
  if (!iree_host_size_checked_mul(export_count,
                                  sizeof(iree_hal_webgpu_executable_entry_t),
                                  &entry_storage_size) ||
      !iree_host_size_checked_add(sizeof(iree_hal_webgpu_executable_t),
                                  entry_storage_size, &total_size) ||
      !iree_host_size_checked_add(total_size, name_storage_size, &total_size)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "WebGPU executable metadata storage exceeds host "
                            "size limits");
  }
  iree_hal_webgpu_executable_t* executable = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator, total_size, (void**)&executable));
  memset(executable, 0, total_size);
  iree_hal_resource_initialize(&iree_hal_webgpu_executable_vtable,
                               &executable->resource);
  executable->host_allocator = host_allocator;
  executable->function_count = export_count;
  char* name_storage =
      (char*)executable + sizeof(*executable) +
      export_count * sizeof(iree_hal_webgpu_executable_entry_t);

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < export_count && iree_status_is_ok(status);
       ++i) {
    status = iree_hal_webgpu_executable_initialize_export(
        device_handle, shader_modules_vec, shader_module_count, i,
        iree_hal_webgpu_ExportDef_vec_at(exports_vec, i), &name_storage,
        &executable->entries[i]);
  }

  if (iree_status_is_ok(status)) {
    *out_executable = (iree_hal_executable_t*)executable;
  } else {
    iree_hal_executable_destroy((iree_hal_executable_t*)executable);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_hal_webgpu_handle_t iree_hal_webgpu_executable_pipeline_handle(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_function_t function) {
  iree_hal_webgpu_executable_t* executable =
      iree_hal_webgpu_executable_cast(base_executable);
  IREE_ASSERT(iree_hal_executable_function_is_index_in_range(
      function, executable->function_count));
  const uint32_t function_index = iree_hal_executable_function_index(function);
  return executable->entries[function_index].pipeline_handle;
}

iree_hal_webgpu_handle_t iree_hal_webgpu_executable_bind_group_layout_handle(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_function_t function) {
  iree_hal_webgpu_executable_t* executable =
      iree_hal_webgpu_executable_cast(base_executable);
  IREE_ASSERT(iree_hal_executable_function_is_index_in_range(
      function, executable->function_count));
  const uint32_t function_index = iree_hal_executable_function_index(function);
  return executable->entries[function_index].bind_group_layout_handle;
}

static void iree_hal_webgpu_executable_destroy(
    iree_hal_executable_t* base_executable) {
  iree_hal_webgpu_executable_t* executable =
      iree_hal_webgpu_executable_cast(base_executable);
  iree_allocator_t host_allocator = executable->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  for (iree_host_size_t i = 0; i < executable->function_count; ++i) {
    iree_hal_webgpu_executable_entry_t* entry = &executable->entries[i];
    iree_hal_webgpu_import_handle_release(entry->pipeline_handle);
    iree_hal_webgpu_import_handle_release(entry->bind_group_layout_handle);
  }

  iree_allocator_free(host_allocator, executable);
  IREE_TRACE_ZONE_END(z0);
}

static iree_host_size_t iree_hal_webgpu_executable_function_count(
    iree_hal_executable_t* base_executable) {
  iree_hal_webgpu_executable_t* executable =
      iree_hal_webgpu_executable_cast(base_executable);
  return executable->function_count;
}

static iree_status_t iree_hal_webgpu_executable_function_info(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_function_t function,
    iree_hal_executable_function_info_t* out_info) {
  iree_hal_webgpu_executable_t* executable =
      iree_hal_webgpu_executable_cast(base_executable);
  if (!iree_hal_executable_function_is_index_in_range(
          function, executable->function_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "function id %" PRIu64
                            " out of range (count=%" PRIhsz ")",
                            function.value, executable->function_count);
  }
  const uint32_t function_index = iree_hal_executable_function_index(function);
  const iree_hal_webgpu_executable_entry_t* entry =
      &executable->entries[function_index];
  memset(out_info, 0, sizeof(*out_info));
  out_info->name = entry->name;
  out_info->binding_count = entry->binding_count;
  out_info->constant_count = entry->constant_count;
  out_info->workgroup_size[0] = entry->workgroup_size[0];
  out_info->workgroup_size[1] = entry->workgroup_size[1];
  out_info->workgroup_size[2] = entry->workgroup_size[2];
  return iree_ok_status();
}

static iree_status_t iree_hal_webgpu_executable_function_parameters(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_function_t function, iree_host_size_t capacity,
    iree_hal_executable_function_parameter_t* out_parameters) {
  (void)base_executable;
  (void)function;
  (void)capacity;
  (void)out_parameters;
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "WebGPU executables do not support parameter "
                          "reflection; WGSL shader metadata does not carry "
                          "per-parameter name/description information");
}

static iree_status_t iree_hal_webgpu_executable_lookup_function_by_name(
    iree_hal_executable_t* base_executable, iree_string_view_t name,
    iree_hal_executable_function_t* out_function) {
  iree_hal_webgpu_executable_t* executable =
      iree_hal_webgpu_executable_cast(base_executable);
  for (iree_host_size_t i = 0; i < executable->function_count; ++i) {
    if (iree_string_view_equal(executable->entries[i].name, name)) {
      *out_function = iree_hal_executable_function_from_index((uint32_t)i);
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "no function named '%.*s' in executable",
                          (int)name.size, name.data);
}

static const iree_hal_executable_vtable_t iree_hal_webgpu_executable_vtable = {
    .destroy = iree_hal_webgpu_executable_destroy,
    .function_count = iree_hal_webgpu_executable_function_count,
    .function_info = iree_hal_webgpu_executable_function_info,
    .function_parameters = iree_hal_webgpu_executable_function_parameters,
    .lookup_function_by_name =
        iree_hal_webgpu_executable_lookup_function_by_name,
};
