// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/webgpu/webgpu_executable.h"

#include "iree/hal/drivers/webgpu/webgpu_imports.h"

//===----------------------------------------------------------------------===//
// Per-export entry
//===----------------------------------------------------------------------===//

// Stores the bridge handles and metadata for a single compute shader export.
// Pipeline, pipeline layout, and bind group layout handles are created during
// executable initialization and released during destruction.
typedef struct iree_hal_webgpu_executable_entry_t {
  iree_hal_webgpu_handle_t pipeline_handle;
  iree_hal_webgpu_handle_t pipeline_layout_handle;
  iree_hal_webgpu_handle_t bind_group_layout_handle;
  iree_string_view_t name;
  uint32_t workgroup_size[3];
  uint16_t binding_count;
  uint16_t constant_count;
} iree_hal_webgpu_executable_entry_t;

//===----------------------------------------------------------------------===//
// iree_hal_webgpu_executable_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_webgpu_executable_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  iree_host_size_t export_count;
  iree_hal_webgpu_executable_entry_t entries[];
} iree_hal_webgpu_executable_t;

static const iree_hal_executable_vtable_t iree_hal_webgpu_executable_vtable;

static iree_hal_webgpu_executable_t* iree_hal_webgpu_executable_cast(
    iree_hal_executable_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_webgpu_executable_vtable);
  return (iree_hal_webgpu_executable_t*)base_value;
}

iree_status_t iree_hal_webgpu_executable_create(
    iree_hal_webgpu_handle_t device_handle,
    const iree_hal_executable_params_t* executable_params,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable) {
  IREE_ASSERT_ARGUMENT(executable_params);
  IREE_ASSERT_ARGUMENT(out_executable);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_executable = NULL;

  // The executable binary format is defined by the compiler's WebGPU target
  // backend. It contains WGSL shader sources and per-export metadata (binding
  // counts, workgroup sizes, entry point names). The compiler target backend
  // that produces this format does not exist yet — the format definition and
  // parser will be implemented alongside the compiler backend. Until then,
  // executable creation fails here.
  //
  // When the format is defined, this function will:
  //   1. Validate the binary header (magic, version, bounds).
  //   2. Parse per-export definitions.
  //   3. For each export, create a bind group layout, pipeline layout, and
  //      compute pipeline via the bridge.
  //   4. Store the handles in the entries[] flexible array.
  iree_status_t status = iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "WebGPU executable format parsing requires the compiler's WebGPU target "
      "backend (IREE SPIR-V -> WGSL via Tint), which has not been "
      "implemented; the executable binary in '%.*s' format cannot be loaded",
      (int)executable_params->executable_format.size,
      executable_params->executable_format.data);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_hal_webgpu_handle_t iree_hal_webgpu_executable_pipeline_handle(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_export_ordinal_t export_ordinal) {
  iree_hal_webgpu_executable_t* executable =
      iree_hal_webgpu_executable_cast(base_executable);
  IREE_ASSERT((iree_host_size_t)export_ordinal < executable->export_count);
  return executable->entries[export_ordinal].pipeline_handle;
}

iree_hal_webgpu_handle_t iree_hal_webgpu_executable_pipeline_layout_handle(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_export_ordinal_t export_ordinal) {
  iree_hal_webgpu_executable_t* executable =
      iree_hal_webgpu_executable_cast(base_executable);
  IREE_ASSERT((iree_host_size_t)export_ordinal < executable->export_count);
  return executable->entries[export_ordinal].pipeline_layout_handle;
}

iree_hal_webgpu_handle_t iree_hal_webgpu_executable_bind_group_layout_handle(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_export_ordinal_t export_ordinal) {
  iree_hal_webgpu_executable_t* executable =
      iree_hal_webgpu_executable_cast(base_executable);
  IREE_ASSERT((iree_host_size_t)export_ordinal < executable->export_count);
  return executable->entries[export_ordinal].bind_group_layout_handle;
}

static void iree_hal_webgpu_executable_destroy(
    iree_hal_executable_t* base_executable) {
  iree_hal_webgpu_executable_t* executable =
      iree_hal_webgpu_executable_cast(base_executable);
  iree_allocator_t host_allocator = executable->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  for (iree_host_size_t i = 0; i < executable->export_count; ++i) {
    iree_hal_webgpu_executable_entry_t* entry = &executable->entries[i];
    iree_hal_webgpu_import_handle_release(entry->pipeline_handle);
    iree_hal_webgpu_import_handle_release(entry->pipeline_layout_handle);
    iree_hal_webgpu_import_handle_release(entry->bind_group_layout_handle);
  }

  iree_allocator_free(host_allocator, executable);
  IREE_TRACE_ZONE_END(z0);
}

static iree_host_size_t iree_hal_webgpu_executable_export_count(
    iree_hal_executable_t* base_executable) {
  iree_hal_webgpu_executable_t* executable =
      iree_hal_webgpu_executable_cast(base_executable);
  return executable->export_count;
}

static iree_status_t iree_hal_webgpu_executable_export_info(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    iree_hal_executable_export_info_t* out_info) {
  iree_hal_webgpu_executable_t* executable =
      iree_hal_webgpu_executable_cast(base_executable);
  if ((iree_host_size_t)export_ordinal >= executable->export_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "export ordinal %u out of range (count=%" PRIhsz
                            ")",
                            export_ordinal, executable->export_count);
  }
  const iree_hal_webgpu_executable_entry_t* entry =
      &executable->entries[export_ordinal];
  memset(out_info, 0, sizeof(*out_info));
  out_info->name = entry->name;
  out_info->binding_count = entry->binding_count;
  out_info->constant_count = entry->constant_count;
  out_info->workgroup_size[0] = entry->workgroup_size[0];
  out_info->workgroup_size[1] = entry->workgroup_size[1];
  out_info->workgroup_size[2] = entry->workgroup_size[2];
  return iree_ok_status();
}

static iree_status_t iree_hal_webgpu_executable_export_parameters(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    iree_host_size_t capacity,
    iree_hal_executable_export_parameter_t* out_parameters) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "WebGPU executables do not support parameter "
                          "reflection; WGSL shader metadata does not carry "
                          "per-parameter name/description information");
}

static iree_status_t iree_hal_webgpu_executable_lookup_export_by_name(
    iree_hal_executable_t* base_executable, iree_string_view_t name,
    iree_hal_executable_export_ordinal_t* out_export_ordinal) {
  iree_hal_webgpu_executable_t* executable =
      iree_hal_webgpu_executable_cast(base_executable);
  for (iree_host_size_t i = 0; i < executable->export_count; ++i) {
    if (iree_string_view_equal(executable->entries[i].name, name)) {
      *out_export_ordinal = (iree_hal_executable_export_ordinal_t)i;
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "no export named '%.*s' in executable",
                          (int)name.size, name.data);
}

static const iree_hal_executable_vtable_t iree_hal_webgpu_executable_vtable = {
    .destroy = iree_hal_webgpu_executable_destroy,
    .export_count = iree_hal_webgpu_executable_export_count,
    .export_info = iree_hal_webgpu_executable_export_info,
    .export_parameters = iree_hal_webgpu_executable_export_parameters,
    .lookup_export_by_name = iree_hal_webgpu_executable_lookup_export_by_name,
};
