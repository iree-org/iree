// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/bytecode/assembler/archive.h"

#include <string.h>

#include "iree/base/internal/flatcc/building.h"
#include "iree/schemas/bytecode_module_def_builder.h"
#include "iree/vm/bytecode/isa/isa.h"

static iree_status_t iree_vm_bytecode_assembler_clone_flatbuffer(
    void* flatbuffer_data, size_t flatbuffer_size,
    iree_allocator_t host_allocator, iree_byte_span_t* out_archive) {
  if (!flatbuffer_data || flatbuffer_size == 0) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "failed to finalize module flatbuffer");
  }
  if (flatbuffer_size > IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "module flatbuffer too large");
  }
  void* archive_data = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, (iree_host_size_t)flatbuffer_size, &archive_data));
  memcpy(archive_data, flatbuffer_data, flatbuffer_size);
  *out_archive =
      iree_make_byte_span(archive_data, (iree_host_size_t)flatbuffer_size);
  return iree_ok_status();
}

iree_status_t iree_vm_bytecode_assembler_build_archive(
    iree_vm_bytecode_assembler_module_t* state, iree_byte_span_t* out_archive) {
  iree_string_view_t bytecode =
      iree_string_builder_view(&state->bytecode_builder);
  if (bytecode.size > INT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "module bytecode is too large");
  }

  flatcc_builder_t builder;
  if (IREE_UNLIKELY(flatcc_builder_init(&builder) != 0)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "failed to initialize flatbuffer builder");
  }

  void* flatbuffer_data = NULL;
  size_t flatbuffer_size = 0;
  iree_status_t status = iree_ok_status();

  iree_vm_FunctionDescriptor_t* descriptors = NULL;
  iree_vm_FunctionSignatureDef_ref_t* signature_refs = NULL;
  iree_vm_TypeDef_ref_t* type_refs = NULL;
  iree_vm_ImportFunctionDef_ref_t* import_refs = NULL;
  iree_vm_ExportFunctionDef_ref_t* export_refs = NULL;
  iree_vm_RodataSegmentDef_ref_t* rodata_refs = NULL;

  flatbuffers_string_ref_t module_name_ref = 0;
  flatbuffers_uint8_vec_ref_t bytecode_data_ref = 0;
  iree_vm_TypeDef_vec_ref_t types_ref = 0;
  iree_vm_ImportFunctionDef_vec_ref_t imported_functions_ref = 0;
  iree_vm_FunctionDescriptor_vec_ref_t function_descriptors_ref = 0;
  iree_vm_FunctionSignatureDef_vec_ref_t function_signatures_ref = 0;
  iree_vm_ExportFunctionDef_vec_ref_t exported_functions_ref = 0;
  iree_vm_RodataSegmentDef_vec_ref_t rodata_segments_ref = 0;
  iree_vm_ModuleStateDef_ref_t module_state_ref = 0;

  if (iree_status_is_ok(status)) {
    module_name_ref = flatbuffers_string_create(
        &builder, state->module_name.data, state->module_name.size);
    bytecode_data_ref = flatbuffers_uint8_vec_create(
        &builder, (const uint8_t*)bytecode.data, bytecode.size);
    if (!module_name_ref || !bytecode_data_ref) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "failed to create module flatbuffer data");
    }
  }

  if (iree_status_is_ok(status) && state->type_count > 0) {
    status = iree_allocator_malloc(state->host_allocator,
                                   state->type_count * sizeof(type_refs[0]),
                                   (void**)&type_refs);
  }
  if (iree_status_is_ok(status) && state->type_count > 0) {
    for (iree_host_size_t i = 0; i < state->type_count; ++i) {
      const iree_vm_bytecode_assembler_type_t* type = &state->types[i];
      flatbuffers_string_ref_t full_name_ref = flatbuffers_string_create(
          &builder, type->full_name.data, type->full_name.size);
      if (!full_name_ref) {
        status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                  "failed to create type name");
        break;
      }
      iree_vm_TypeDef_start(&builder);
      iree_vm_TypeDef_full_name_add(&builder, full_name_ref);
      type_refs[i] = iree_vm_TypeDef_end(&builder);
      if (!type_refs[i]) {
        status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                  "failed to create type definition");
        break;
      }
    }
    if (iree_status_is_ok(status)) {
      types_ref =
          iree_vm_TypeDef_vec_create(&builder, type_refs, state->type_count);
      if (!types_ref) {
        status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                  "failed to create type table");
      }
    }
  }

  if (iree_status_is_ok(status) && state->import_count > 0) {
    status = iree_allocator_malloc(state->host_allocator,
                                   state->import_count * sizeof(import_refs[0]),
                                   (void**)&import_refs);
  }
  if (iree_status_is_ok(status) && state->import_count > 0) {
    for (iree_host_size_t i = 0; i < state->import_count; ++i) {
      const iree_vm_bytecode_assembler_import_t* import = &state->imports[i];
      flatbuffers_string_ref_t full_name_ref = flatbuffers_string_create(
          &builder, import->full_name.data, import->full_name.size);
      flatbuffers_string_ref_t cconv_ref =
          flatbuffers_string_create(&builder, import->calling_convention.data,
                                    import->calling_convention.size);
      if (!full_name_ref || !cconv_ref) {
        status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                  "failed to create import strings");
        break;
      }
      iree_vm_FunctionSignatureDef_start(&builder);
      iree_vm_FunctionSignatureDef_calling_convention_add(&builder, cconv_ref);
      iree_vm_FunctionSignatureDef_ref_t signature_ref =
          iree_vm_FunctionSignatureDef_end(&builder);
      if (!signature_ref) {
        status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                  "failed to create import signature");
        break;
      }
      iree_vm_ImportFunctionDef_start(&builder);
      iree_vm_ImportFunctionDef_full_name_add(&builder, full_name_ref);
      iree_vm_ImportFunctionDef_signature_add(&builder, signature_ref);
      iree_vm_ImportFunctionDef_flags_add(
          &builder, import->optional ? iree_vm_ImportFlagBits_OPTIONAL
                                     : iree_vm_ImportFlagBits_REQUIRED);
      import_refs[i] = iree_vm_ImportFunctionDef_end(&builder);
      if (!import_refs[i]) {
        status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                  "failed to create import definition");
        break;
      }
    }
    if (iree_status_is_ok(status)) {
      imported_functions_ref = iree_vm_ImportFunctionDef_vec_create(
          &builder, import_refs, state->import_count);
      if (!imported_functions_ref) {
        status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                  "failed to create import table");
      }
    }
  }

  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(
        state->host_allocator, state->function_count * sizeof(descriptors[0]),
        (void**)&descriptors);
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_allocator_malloc(state->host_allocator,
                              state->function_count * sizeof(signature_refs[0]),
                              (void**)&signature_refs);
  }
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < state->function_count; ++i) {
      const iree_vm_bytecode_assembler_function_t* function =
          &state->functions[i];
      iree_vm_FunctionDescriptor_assign(
          &descriptors[i], (int32_t)function->bytecode_offset,
          (int32_t)function->bytecode_length, function->requirements,
          /*reserved=*/0, (int16_t)function->block_count,
          (int16_t)function->i32_register_count,
          (int16_t)function->ref_register_count);
      flatbuffers_string_ref_t cconv_ref = flatbuffers_string_create(
          &builder, function->cconv.data, function->cconv.size);
      if (!cconv_ref) {
        status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                  "failed to create calling convention string");
        break;
      }
      iree_vm_FunctionSignatureDef_start(&builder);
      iree_vm_FunctionSignatureDef_calling_convention_add(&builder, cconv_ref);
      signature_refs[i] = iree_vm_FunctionSignatureDef_end(&builder);
      if (!signature_refs[i]) {
        status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                  "failed to create function signature");
        break;
      }
    }
  }

  if (iree_status_is_ok(status)) {
    function_descriptors_ref = iree_vm_FunctionDescriptor_vec_create(
        &builder, descriptors, state->function_count);
    function_signatures_ref = iree_vm_FunctionSignatureDef_vec_create(
        &builder, signature_refs, state->function_count);
    if (!function_descriptors_ref || !function_signatures_ref) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "failed to create function flatbuffer tables");
    }
  }

  if (iree_status_is_ok(status) && state->export_count > 0) {
    status = iree_allocator_malloc(state->host_allocator,
                                   state->export_count * sizeof(export_refs[0]),
                                   (void**)&export_refs);
  }
  if (iree_status_is_ok(status) && state->export_count > 0) {
    for (iree_host_size_t i = 0; i < state->export_count; ++i) {
      const iree_vm_bytecode_assembler_export_t* export_def =
          &state->exports[i];
      const iree_host_size_t function_ordinal =
          iree_vm_bytecode_assembler_find_function_ordinal(
              state, export_def->function_name);
      if (function_ordinal > INT32_MAX) {
        status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                  "export function ordinal is too large");
        break;
      }
      flatbuffers_string_ref_t export_name_ref = flatbuffers_string_create(
          &builder, export_def->name.data, export_def->name.size);
      if (!export_name_ref) {
        status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                  "failed to create export name");
        break;
      }
      iree_vm_ExportFunctionDef_start(&builder);
      iree_vm_ExportFunctionDef_local_name_add(&builder, export_name_ref);
      iree_vm_ExportFunctionDef_internal_ordinal_add(&builder,
                                                     (int32_t)function_ordinal);
      export_refs[i] = iree_vm_ExportFunctionDef_end(&builder);
      if (!export_refs[i]) {
        status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                  "failed to create export");
        break;
      }
    }
    if (iree_status_is_ok(status)) {
      exported_functions_ref = iree_vm_ExportFunctionDef_vec_create(
          &builder, export_refs, state->export_count);
      if (!exported_functions_ref) {
        status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                  "failed to create export flatbuffer table");
      }
    }
  }

  if (iree_status_is_ok(status) && state->rodata_segment_count > 0) {
    status = iree_allocator_malloc(
        state->host_allocator,
        state->rodata_segment_count * sizeof(rodata_refs[0]),
        (void**)&rodata_refs);
  }
  if (iree_status_is_ok(status) && state->rodata_segment_count > 0) {
    for (iree_host_size_t i = 0; i < state->rodata_segment_count; ++i) {
      const iree_vm_bytecode_assembler_rodata_t* rodata =
          &state->rodata_segments[i];
      flatbuffers_uint8_vec_ref_t data_ref = flatbuffers_uint8_vec_create(
          &builder, rodata->data, rodata->data_length);
      if (!data_ref) {
        status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                  "failed to create rodata segment bytes");
        break;
      }
      iree_vm_RodataSegmentDef_start(&builder);
      iree_vm_RodataSegmentDef_embedded_data_add(&builder, data_ref);
      rodata_refs[i] = iree_vm_RodataSegmentDef_end(&builder);
      if (!rodata_refs[i]) {
        status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                  "failed to create rodata segment");
        break;
      }
    }
    if (iree_status_is_ok(status)) {
      rodata_segments_ref = iree_vm_RodataSegmentDef_vec_create(
          &builder, rodata_refs, state->rodata_segment_count);
      if (!rodata_segments_ref) {
        status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                  "failed to create rodata segment table");
      }
    }
  }

  if (iree_status_is_ok(status) &&
      (state->global_byte_capacity != 0 || state->global_ref_count != 0)) {
    if (state->global_byte_capacity > INT32_MAX ||
        state->global_ref_count > INT32_MAX) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "module state is too large");
    } else {
      iree_vm_ModuleStateDef_start(&builder);
      iree_vm_ModuleStateDef_global_bytes_capacity_add(
          &builder, (int32_t)state->global_byte_capacity);
      iree_vm_ModuleStateDef_global_ref_count_add(
          &builder, (int32_t)state->global_ref_count);
      module_state_ref = iree_vm_ModuleStateDef_end(&builder);
      if (!module_state_ref) {
        status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                  "failed to create module state table");
      }
    }
  }

  if (iree_status_is_ok(status)) {
    const uint32_t bytecode_version =
        (IREE_VM_BYTECODE_VERSION_MAJOR << 16) | IREE_VM_BYTECODE_VERSION_MINOR;
    if (flatbuffers_failed(
            iree_vm_BytecodeModuleDef_start_as_root_with_size(&builder)) ||
        flatbuffers_failed(
            iree_vm_BytecodeModuleDef_name_add(&builder, module_name_ref)) ||
        flatbuffers_failed(iree_vm_BytecodeModuleDef_version_add(
            &builder, state->module_version)) ||
        flatbuffers_failed(iree_vm_BytecodeModuleDef_requirements_add(
            &builder, state->module_requirements)) ||
        (types_ref && flatbuffers_failed(iree_vm_BytecodeModuleDef_types_add(
                          &builder, types_ref))) ||
        (imported_functions_ref &&
         flatbuffers_failed(iree_vm_BytecodeModuleDef_imported_functions_add(
             &builder, imported_functions_ref))) ||
        (exported_functions_ref &&
         flatbuffers_failed(iree_vm_BytecodeModuleDef_exported_functions_add(
             &builder, exported_functions_ref))) ||
        flatbuffers_failed(iree_vm_BytecodeModuleDef_function_signatures_add(
            &builder, function_signatures_ref)) ||
        (rodata_segments_ref &&
         flatbuffers_failed(iree_vm_BytecodeModuleDef_rodata_segments_add(
             &builder, rodata_segments_ref))) ||
        (module_state_ref &&
         flatbuffers_failed(iree_vm_BytecodeModuleDef_module_state_add(
             &builder, module_state_ref))) ||
        flatbuffers_failed(iree_vm_BytecodeModuleDef_function_descriptors_add(
            &builder, function_descriptors_ref)) ||
        flatbuffers_failed(iree_vm_BytecodeModuleDef_bytecode_version_add(
            &builder, bytecode_version)) ||
        flatbuffers_failed(iree_vm_BytecodeModuleDef_bytecode_data_add(
            &builder, bytecode_data_ref)) ||
        !iree_vm_BytecodeModuleDef_end_as_root(&builder)) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "failed to create module flatbuffer root");
    }
  }

  if (iree_status_is_ok(status)) {
    flatbuffer_data =
        flatcc_builder_finalize_aligned_buffer(&builder, &flatbuffer_size);
    status = iree_vm_bytecode_assembler_clone_flatbuffer(
        flatbuffer_data, flatbuffer_size, state->host_allocator, out_archive);
  }

  flatcc_builder_aligned_free(flatbuffer_data);
  flatcc_builder_clear(&builder);
  iree_allocator_free(state->host_allocator, rodata_refs);
  iree_allocator_free(state->host_allocator, export_refs);
  iree_allocator_free(state->host_allocator, import_refs);
  iree_allocator_free(state->host_allocator, type_refs);
  iree_allocator_free(state->host_allocator, signature_refs);
  iree_allocator_free(state->host_allocator, descriptors);
  return status;
}
