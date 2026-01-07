// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/bytecode/verifier.h"

#include "iree/base/internal/math.h"
#include "iree/vm/bytecode/utils/block_list.h"
#include "iree/vm/bytecode/utils/features.h"

//===----------------------------------------------------------------------===//
// Module metadata verification
//===----------------------------------------------------------------------===//

iree_status_t iree_vm_bytecode_module_flatbuffer_verify(
    iree_const_byte_span_t archive_contents,
    iree_const_byte_span_t flatbuffer_contents,
    iree_host_size_t archive_rodata_offset) {
  // Run flatcc generated verification. This ensures all pointers are in-bounds
  // and that we can safely walk the file, but not that the actual contents of
  // the FlatBuffer meet our expectations.
  int verify_ret = iree_vm_BytecodeModuleDef_verify_as_root(
      flatbuffer_contents.data, flatbuffer_contents.data_length);
  if (verify_ret != flatcc_verify_ok) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "FlatBuffer verification failed: %s",
                            flatcc_verify_error_string(verify_ret));
  }

  iree_vm_BytecodeModuleDef_table_t module_def =
      iree_vm_BytecodeModuleDef_as_root(flatbuffer_contents.data);

  const iree_vm_FeatureBits_enum_t available_features =
      iree_vm_bytecode_available_features();
  const iree_vm_FeatureBits_enum_t required_features =
      iree_vm_BytecodeModuleDef_requirements(module_def);
  IREE_RETURN_IF_ERROR(iree_vm_check_feature_mismatch(
      __FILE__, __LINE__, required_features, available_features));

  flatbuffers_string_t name = iree_vm_BytecodeModuleDef_name(module_def);
  if (!flatbuffers_string_len(name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "module missing name field");
  }

  iree_vm_TypeDef_vec_t types = iree_vm_BytecodeModuleDef_types(module_def);
  for (size_t i = 0; i < iree_vm_TypeDef_vec_len(types); ++i) {
    iree_vm_TypeDef_table_t type_def = iree_vm_TypeDef_vec_at(types, i);
    if (!type_def) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "types[%zu] missing body", i);
    }
    flatbuffers_string_t full_name = iree_vm_TypeDef_full_name(type_def);
    if (flatbuffers_string_len(full_name) <= 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "types[%zu] missing name", i);
    }
  }

  iree_vm_RodataSegmentDef_vec_t rodata_segments =
      iree_vm_BytecodeModuleDef_rodata_segments(module_def);
  for (size_t i = 0; i < iree_vm_RodataSegmentDef_vec_len(rodata_segments);
       ++i) {
    iree_vm_RodataSegmentDef_table_t segment =
        iree_vm_RodataSegmentDef_vec_at(rodata_segments, i);
    if (iree_vm_RodataSegmentDef_embedded_data_is_present(segment)) {
      continue;  // embedded data is verified by FlatBuffers
    }
    uint64_t segment_offset =
        iree_vm_RodataSegmentDef_external_data_offset(segment);
    uint64_t segment_length =
        iree_vm_RodataSegmentDef_external_data_length(segment);
    uint64_t segment_end =
        archive_rodata_offset + segment_offset + segment_length;
    if (segment_end > archive_contents.data_length) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "rodata[%zu] external reference out of range", i);
    }
  }

  iree_vm_ModuleDependencyDef_vec_t dependencies =
      iree_vm_BytecodeModuleDef_dependencies(module_def);
  for (size_t i = 0; i < iree_vm_ModuleDependencyDef_vec_len(dependencies);
       ++i) {
    iree_vm_ModuleDependencyDef_table_t dependency_def =
        iree_vm_ModuleDependencyDef_vec_at(dependencies, i);
    flatbuffers_string_t module_name =
        iree_vm_ModuleDependencyDef_name(dependency_def);
    if (flatbuffers_string_len(module_name) == 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "dependencies[%zu] has no module name", i);
    }
  }

  iree_vm_ImportFunctionDef_vec_t imported_functions =
      iree_vm_BytecodeModuleDef_imported_functions(module_def);
  iree_vm_ExportFunctionDef_vec_t exported_functions =
      iree_vm_BytecodeModuleDef_exported_functions(module_def);
  iree_vm_FunctionSignatureDef_vec_t function_signatures =
      iree_vm_BytecodeModuleDef_function_signatures(module_def);
  iree_vm_FunctionDescriptor_vec_t function_descriptors =
      iree_vm_BytecodeModuleDef_function_descriptors(module_def);

  for (size_t i = 0;
       i < iree_vm_FunctionSignatureDef_vec_len(function_signatures); ++i) {
    iree_vm_FunctionSignatureDef_table_t function_signature =
        iree_vm_FunctionSignatureDef_vec_at(function_signatures, i);
    if (!function_signature) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "function_signatures[%zu] missing body", i);
    }
  }

  for (size_t i = 0; i < iree_vm_ImportFunctionDef_vec_len(imported_functions);
       ++i) {
    iree_vm_ImportFunctionDef_table_t import_def =
        iree_vm_ImportFunctionDef_vec_at(imported_functions, i);
    if (!import_def) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "imports[%zu] missing body", i);
    }
    flatbuffers_string_t full_name =
        iree_vm_ImportFunctionDef_full_name(import_def);
    if (!flatbuffers_string_len(full_name)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "imports[%zu] missing full_name", i);
    }
  }

  if (iree_vm_FunctionSignatureDef_vec_len(function_signatures) !=
      iree_vm_FunctionDescriptor_vec_len(function_descriptors)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "function signature and descriptor table length mismatch (%zu vs %zu)",
        iree_vm_FunctionSignatureDef_vec_len(function_signatures),
        iree_vm_FunctionDescriptor_vec_len(function_descriptors));
  }

  for (size_t i = 0; i < iree_vm_ExportFunctionDef_vec_len(exported_functions);
       ++i) {
    iree_vm_ExportFunctionDef_table_t export_def =
        iree_vm_ExportFunctionDef_vec_at(exported_functions, i);
    if (!export_def) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "exports[%zu] missing body", i);
    }
    flatbuffers_string_t local_name =
        iree_vm_ExportFunctionDef_local_name(export_def);
    if (!flatbuffers_string_len(local_name)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "exports[%zu] missing local_name", i);
    }
    iree_host_size_t internal_ordinal =
        iree_vm_ExportFunctionDef_internal_ordinal(export_def);
    if (internal_ordinal >=
        iree_vm_FunctionDescriptor_vec_len(function_descriptors)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "exports[%zu] internal_ordinal out of bounds (0 < %" PRIhsz " < %zu)",
          i, internal_ordinal,
          iree_vm_FunctionDescriptor_vec_len(function_descriptors));
    }
  }

  // Verify that we can properly handle the bytecode embedded in the module.
  // We require that major versions match and allow loading of older minor
  // versions (we keep changes backwards-compatible).
  const uint32_t bytecode_version =
      iree_vm_BytecodeModuleDef_bytecode_version(module_def);
  const uint32_t bytecode_version_major = bytecode_version >> 16;
  const uint32_t bytecode_version_minor = bytecode_version & 0xFFFF;
  if ((bytecode_version_major != IREE_VM_BYTECODE_VERSION_MAJOR) ||
      (bytecode_version_minor > IREE_VM_BYTECODE_VERSION_MINOR)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "bytecode version mismatch; runtime supports %d.%d, module has %d.%d",
        IREE_VM_BYTECODE_VERSION_MAJOR, IREE_VM_BYTECODE_VERSION_MINOR,
        bytecode_version_major, bytecode_version_minor);
  }

  flatbuffers_uint8_vec_t bytecode_data =
      iree_vm_BytecodeModuleDef_bytecode_data(module_def);
  for (size_t i = 0;
       i < iree_vm_FunctionDescriptor_vec_len(function_descriptors); ++i) {
    iree_vm_FunctionDescriptor_struct_t function_descriptor =
        iree_vm_FunctionDescriptor_vec_at(function_descriptors, i);
    if (function_descriptor->block_count == 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "functions[%zu] descriptor block count is 0; "
          "functions must have at least 1 block, expected %d",
          i, function_descriptor->block_count);
    }
    if (function_descriptor->bytecode_length == 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "functions[%zu] descriptor bytecode reports 0 "
                              "length; functions must have at least one block",
                              i);
    }
    if (function_descriptor->bytecode_offset < 0 ||
        function_descriptor->bytecode_offset +
                function_descriptor->bytecode_length >
            flatbuffers_uint8_vec_len(bytecode_data)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "functions[%zu] descriptor bytecode span out of "
                              "range (0 < %d < %zu)",
                              i, function_descriptor->bytecode_offset,
                              flatbuffers_uint8_vec_len(bytecode_data));
    }
    if (function_descriptor->i32_register_count >
            IREE_VM_ISA_I32_REGISTER_COUNT ||
        function_descriptor->ref_register_count >
            IREE_VM_ISA_REF_REGISTER_COUNT) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "functions[%zu] descriptor register count out of range", i);
    }
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Function verification
//===----------------------------------------------------------------------===//

// State used during verification of a function.
typedef struct iree_vm_bytecode_verify_state_t {
  // Within a block (encountered a block marker and not yet a terminator).
  uint32_t in_block : 1;

  // Maximum valid register ordinals.
  uint32_t i32_register_count;
  uint32_t ref_register_count;

  // Parsed argument and result cconv fragments.
  iree_string_view_t cconv_arguments;
  iree_string_view_t cconv_results;

  // All block branch points.
  iree_vm_bytecode_block_list_t block_list;

  // Quick lookups of flatbuffer properties.
  const iree_vm_ImportFunctionDef_vec_t imported_functions;
  const iree_vm_ExportFunctionDef_vec_t exported_functions;
  const iree_vm_FunctionSignatureDef_vec_t function_signatures;
  const iree_vm_FunctionDescriptor_vec_t function_descriptors;
  iree_host_size_t rodata_storage_size;
  iree_host_size_t rodata_ref_count;
  iree_host_size_t rwdata_storage_size;
  iree_host_size_t global_ref_count;
} iree_vm_bytecode_verify_state_t;

// Parses the cconv fragments from the given |signature_def|.
static iree_status_t iree_vm_bytecode_function_get_cconv_fragments(
    iree_vm_FunctionSignatureDef_table_t signature_def,
    iree_string_view_t* out_arguments, iree_string_view_t* out_results);

// Verifies that the function has storage for the declared arguments.
static iree_status_t iree_vm_bytecode_function_verify_arguments(
    const iree_vm_bytecode_verify_state_t* verify_state);

// Verifies a single operation at |pc| in the function |bytecode_data|.
// Returns an error if the op is invalid and otherwise sets |out_next_pc| to the
// program counter immediately following the op (which may be the end of data!).
static iree_status_t iree_vm_bytecode_function_verify_bytecode_op(
    iree_vm_bytecode_module_t* module,
    iree_vm_bytecode_verify_state_t* verify_state,
    iree_vm_FunctionSignatureDef_table_t function_signature,
    iree_vm_FunctionDescriptor_struct_t function_descriptor,
    iree_const_byte_span_t bytecode_data, iree_vm_source_offset_t pc,
    iree_vm_source_offset_t max_pc, iree_vm_source_offset_t* out_next_pc);

// NOTE: by the time this is called we have the module created and can assume
// all information on it has been verified. The only thing this verifies is
// function bytecode and capabilities!
iree_status_t iree_vm_bytecode_function_verify(
    iree_vm_bytecode_module_t* module, uint16_t function_ordinal,
    iree_allocator_t scratch_allocator) {
  if (function_ordinal >= module->function_descriptor_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "invalid function ordinal");
  }
  iree_vm_FunctionSignatureDef_table_t function_signature_def =
      iree_vm_FunctionSignatureDef_vec_at(
          iree_vm_BytecodeModuleDef_function_signatures(module->def),
          function_ordinal);
  const iree_vm_FunctionDescriptor_t* function_descriptor =
      &module->function_descriptor_table[function_ordinal];

  const iree_vm_FeatureBits_enum_t available_features =
      iree_vm_bytecode_available_features();
  const iree_vm_FeatureBits_enum_t required_features =
      function_descriptor->requirements;
  IREE_RETURN_IF_ERROR(iree_vm_check_feature_mismatch(
      __FILE__, __LINE__, required_features, available_features));

  if (function_descriptor->block_count == 0) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "no blocks defined; functions must have at least one block");
  }

  // State used during verification.
  iree_vm_bytecode_verify_state_t verify_state = {
      .in_block = 0,
      .imported_functions =
          iree_vm_BytecodeModuleDef_imported_functions(module->def),
      .exported_functions =
          iree_vm_BytecodeModuleDef_exported_functions(module->def),
      .function_signatures =
          iree_vm_BytecodeModuleDef_function_signatures(module->def),
      .function_descriptors =
          iree_vm_BytecodeModuleDef_function_descriptors(module->def),
      .rodata_storage_size = 0,
      .rodata_ref_count = 0,
      .rwdata_storage_size = 0,
      .global_ref_count = 0,
  };

  // NOTE: these must be consistent with iree_vm_bytecode_module_layout_state.
  verify_state.rodata_storage_size = 0;
  verify_state.rodata_ref_count = iree_vm_RodataSegmentDef_vec_len(
      iree_vm_BytecodeModuleDef_rodata_segments(module->def));
  iree_vm_ModuleStateDef_table_t module_state_def =
      iree_vm_BytecodeModuleDef_module_state(module->def);
  if (module_state_def) {
    verify_state.rwdata_storage_size =
        iree_vm_ModuleStateDef_global_bytes_capacity(module_state_def);
    verify_state.global_ref_count =
        iree_vm_ModuleStateDef_global_ref_count(module_state_def);
  }

  // Ensure the register storage (rounded to the nearest power of 2) won't
  // exceed the maximum allowed registers.
  verify_state.i32_register_count = iree_math_round_up_to_pow2_u32(
      iree_max(1, function_descriptor->i32_register_count));
  verify_state.ref_register_count = iree_math_round_up_to_pow2_u32(
      iree_max(1, function_descriptor->ref_register_count));
  if (IREE_UNLIKELY(verify_state.i32_register_count >
                    IREE_VM_ISA_I32_REGISTER_MASK) ||
      IREE_UNLIKELY(verify_state.ref_register_count >
                    IREE_VM_ISA_REF_REGISTER_MASK)) {
    // Register count overflow. A valid compiler should never produce files that
    // hit this.
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "register count overflow");
  }

  // Grab the cconv fragments declaring the arguments/results of the function.
  verify_state.cconv_arguments = iree_string_view_empty();
  verify_state.cconv_results = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_function_get_cconv_fragments(
      function_signature_def, &verify_state.cconv_arguments,
      &verify_state.cconv_results));

  // Verify there is storage for passed arguments.
  IREE_RETURN_IF_ERROR(
      iree_vm_bytecode_function_verify_arguments(&verify_state));

  // NOTE: module verification ensures the function descriptor has a valid
  // bytecode range so we can assume that's true here.
  IREE_ASSERT(function_descriptor->bytecode_length > 0);
  IREE_ASSERT(function_descriptor->bytecode_offset >= 0);
  IREE_ASSERT(function_descriptor->bytecode_offset +
                  function_descriptor->bytecode_length <=
              module->bytecode_data.data_length);
  iree_const_byte_span_t bytecode_data = iree_make_const_byte_span(
      module->bytecode_data.data + function_descriptor->bytecode_offset,
      function_descriptor->bytecode_length);
  const iree_vm_source_offset_t max_pc =
      (iree_vm_source_offset_t)function_descriptor->bytecode_length;

  // Reserve the block list. As we walk the bytecode we'll declare/define blocks
  // and then afterward verify all were found.
  IREE_ASSERT(function_descriptor->block_count > 0);
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_block_list_initialize(
      function_descriptor->block_count, scratch_allocator,
      &verify_state.block_list));

  // Perform bytecode verification by performing a single-pass walk of all
  // function bytecode.
  iree_status_t status = iree_ok_status();
  const iree_vm_source_offset_t end_pc =
      (iree_vm_source_offset_t)bytecode_data.data_length - 1;
  for (iree_vm_source_offset_t pc = 0; pc < end_pc;) {
    iree_vm_source_offset_t start_pc = pc;
    status = iree_vm_bytecode_function_verify_bytecode_op(
        module, &verify_state, function_signature_def, function_descriptor,
        bytecode_data, start_pc, max_pc, &pc);
    if (!iree_status_is_ok(status)) {
#if IREE_STATUS_MODE
      // To get a useful source location we have to ask the main module; the
      // base function table may only contain public symbols and not any
      // internal ones.
      iree_string_view_t module_name = iree_vm_module_name(&module->interface);
      iree_vm_function_t function = {0};
      iree_status_ignore(iree_vm_module_lookup_function_by_ordinal(
          &module->interface, IREE_VM_FUNCTION_LINKAGE_INTERNAL,
          function_ordinal, &function));
      iree_string_view_t function_name = iree_vm_function_name(&function);
      if (!iree_string_view_is_empty(function_name)) {
        status = iree_status_annotate_f(status, "at %.*s.%.*s+%08X",
                                        (int)module_name.size, module_name.data,
                                        (int)function_name.size,
                                        function_name.data, (uint32_t)start_pc);
      } else {
        status = iree_status_annotate_f(status, "at %.*s@%u+%08X",
                                        (int)module_name.size, module_name.data,
                                        function_ordinal, (uint32_t)start_pc);
      }
#endif  // IREE_STATUS_MODE
      break;
    }
  }

  // Ensure there was a terminator.
  if (iree_status_is_ok(status) && verify_state.in_block) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "function missing terminator in the last block");
  }

  // Verify all blocks are defined and have proper markers.
  if (iree_status_is_ok(status)) {
    status = iree_vm_bytecode_block_list_verify(&verify_state.block_list,
                                                bytecode_data);
  }

  iree_vm_bytecode_block_list_deinitialize(&verify_state.block_list,
                                           scratch_allocator);

  return status;
}

//===----------------------------------------------------------------------===//
// Utilities matching the tablegen op encoding scheme
//===----------------------------------------------------------------------===//
// These utilities match the VM_Enc* statements in VMBase.td 1:1, allowing us
// to have the inverse of the encoding which make things easier to read.
//
// Each macro will increment the pc by the number of bytes read and as such must
// be called in the same order the values are encoded.

// Bails if the |pc| exceeds the |max_pc|.
#define IREE_VM_VERIFY_PC_RANGE(pc, max_pc)                                  \
  if (IREE_UNLIKELY((pc) > (max_pc))) {                                      \
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,                        \
                            "bytecode data overrun trying to parsing op at " \
                            "%08X (%u) of %u available bytes",               \
                            (uint32_t)(pc), (uint32_t)(pc),                  \
                            (uint32_t)(max_pc));                             \
  }

// Bails if the function doesn't have the given |required_features| declared.
#define IREE_VM_VERIFY_REQUIREMENT(required_features)                         \
  if (IREE_UNLIKELY(!iree_all_bits_set(function_descriptor->requirements,     \
                                       (required_features)))) {               \
    return iree_vm_check_feature_mismatch(__FILE__, __LINE__,                 \
                                          (required_features),                \
                                          function_descriptor->requirements); \
  }

// Bails if the register ordinal for the given register type is out of bounds.
#define IREE_VM_VERIFY_REG_ORDINAL_X32(ordinal, category)                      \
  if (IREE_UNLIKELY(((ordinal) & IREE_VM_ISA_REF_REGISTER_TYPE_BIT) != 0)) {   \
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,                      \
                            category                                           \
                            " register required but ref register %u provided", \
                            (ordinal));                                        \
  } else if (IREE_UNLIKELY((ordinal) >= verify_state->i32_register_count)) {   \
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,                          \
                            category " register ordinal %u out of range %u",   \
                            (ordinal), verify_state->i32_register_count);      \
  }
#define IREE_VM_VERIFY_REG_ORDINAL_X64(ordinal, category)                      \
  if (IREE_UNLIKELY(((ordinal) & IREE_VM_ISA_REF_REGISTER_TYPE_BIT) != 0)) {   \
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,                      \
                            category                                           \
                            " register required but ref register %u provided", \
                            (ordinal));                                        \
  } else if (IREE_UNLIKELY((ordinal & 1) != 0)) {                              \
    return iree_make_status(                                                   \
        IREE_STATUS_INVALID_ARGUMENT,                                          \
        category " register ordinal %u not 8-byte aligned", (ordinal));        \
  } else if (IREE_UNLIKELY((ordinal) + 1 >=                                    \
                           verify_state->i32_register_count)) {                \
    return iree_make_status(                                                   \
        IREE_STATUS_OUT_OF_RANGE,                                              \
        category " register ordinal %u:%u out of range %u", (ordinal),         \
        (ordinal) + 1, verify_state->i32_register_count);                      \
  }
#define IREE_VM_VERIFY_REG_I32(ordinal) \
  IREE_VM_VERIFY_REG_ORDINAL_X32(ordinal, "i32");
#define IREE_VM_VERIFY_REG_I64(ordinal) \
  IREE_VM_VERIFY_REG_ORDINAL_X64(ordinal, "i64");
#define IREE_VM_VERIFY_REG_F32(ordinal) \
  IREE_VM_VERIFY_REG_ORDINAL_X32(ordinal, "f32");
#define IREE_VM_VERIFY_REG_F64(ordinal) \
  IREE_VM_VERIFY_REG_ORDINAL_X64(ordinal, "f64");
#define IREE_VM_VERIFY_REG_REF(ordinal)                                      \
  if (IREE_UNLIKELY(((ordinal) & IREE_VM_ISA_REF_REGISTER_TYPE_BIT) == 0)) { \
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,                    \
                            "ref register required but non-ref %u provided", \
                            (ordinal));                                      \
  } else if (IREE_UNLIKELY(((ordinal) & IREE_VM_ISA_REF_REGISTER_MASK) >=    \
                           verify_state->ref_register_count)) {              \
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,                        \
                            "ref register ordinal %u out of range %u",       \
                            (ordinal), verify_state->ref_register_count);    \
  }
#define IREE_VM_VERIFY_REG_ANY(ordinal)                                       \
  if (((ordinal) & IREE_VM_ISA_REF_REGISTER_TYPE_BIT) != 0) {                 \
    int32_t ref_ordinal = (ordinal) & IREE_VM_ISA_REF_REGISTER_MASK;          \
    if (IREE_UNLIKELY(ref_ordinal >= verify_state->ref_register_count)) {     \
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,                       \
                              "ref register ordinal %u out of range %u",      \
                              ref_ordinal, verify_state->ref_register_count); \
    }                                                                         \
  } else if (IREE_UNLIKELY((ordinal) >= verify_state->i32_register_count)) {  \
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,                         \
                            "i32 register ordinal %u out of range %u",        \
                            (ordinal), verify_state->i32_register_count);     \
  }

//===----------------------------------------------------------------------===//
// ISA decoding (verifier policy)
//===----------------------------------------------------------------------===//

#define IREE_VM_ISA_BYTECODE_DATA bytecode_data
#define IREE_VM_ISA_PC pc
#define IREE_VM_ISA_REQUIRE(bytes) IREE_VM_VERIFY_PC_RANGE(pc + (bytes), max_pc)

// Verifier wants strict bounds checks on type IDs.
#define IREE_VM_ISA_VERIFY_TYPE_ID(type_id)                       \
  do {                                                            \
    if (IREE_UNLIKELY((type_id) >= module->type_count)) {         \
      return iree_make_status(                                    \
          IREE_STATUS_OUT_OF_RANGE,                               \
          "type id ordinal out of range: %u (table=%" PRIhsz ")", \
          (uint32_t)(type_id), module->type_count);               \
    }                                                             \
  } while (0)

// Type lookups in the verifier use pointers into the module type table.
#define IREE_VM_ISA_TYPE_T const iree_vm_type_def_t*
#define IREE_VM_ISA_LOOKUP_TYPE(type_id, out_type) \
  do {                                             \
    (out_type) = &module->type_table[(type_id)];   \
  } while (0)

// Register validation uses the verifier's current register-count limits.
#define IREE_VM_ISA_VERIFY_REG_I32(ordinal) \
  do {                                      \
    IREE_VM_VERIFY_REG_I32(ordinal);        \
  } while (0)
#define IREE_VM_ISA_VERIFY_REG_I64(ordinal) \
  do {                                      \
    IREE_VM_VERIFY_REG_I64(ordinal);        \
  } while (0)
#define IREE_VM_ISA_VERIFY_REG_F32(ordinal) \
  do {                                      \
    IREE_VM_VERIFY_REG_F32(ordinal);        \
  } while (0)
#define IREE_VM_ISA_VERIFY_REG_F64(ordinal) \
  do {                                      \
    IREE_VM_VERIFY_REG_F64(ordinal);        \
  } while (0)
#define IREE_VM_ISA_VERIFY_REG_ANY(ordinal) \
  do {                                      \
    IREE_VM_VERIFY_REG_ANY(ordinal);        \
  } while (0)

// Ref register validation differs based on whether the op supports MOVE.
#define IREE_VM_ISA_VERIFY_REG_REF_ALLOW_MOVE(ordinal) \
  do {                                                 \
    IREE_VM_VERIFY_REG_REF(ordinal);                   \
  } while (0)
#define IREE_VM_ISA_VERIFY_REG_REF_NO_MOVE(ordinal)                        \
  do {                                                                     \
    IREE_VM_VERIFY_REG_REF(ordinal);                                       \
    if (IREE_UNLIKELY((ordinal) & IREE_VM_ISA_REF_REGISTER_MOVE_BIT)) {    \
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,                \
                              "ref register has MOVE bit but op does not " \
                              "support MOVE");                             \
    }                                                                      \
  } while (0)

#include "iree/vm/bytecode/utils/isa_decoder.inl"

//===----------------------------------------------------------------------===//
// Verifier-specific helpers for decoded aggregates
//===----------------------------------------------------------------------===//

#define IREE_VM_ISA_VERIFY_BRANCH_TARGET(name)             \
  IREE_VM_ISA_DECODE_BRANCH_TARGET_PC(name);               \
  iree_vm_bytecode_block_t* name##_block = NULL;           \
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_block_list_insert( \
      &verify_state->block_list, (name), &name##_block));  \
  (void)(name##_block)

#define IREE_VM_ISA_VERIFY_BRANCH_OPERANDS(name)        \
  IREE_VM_ISA_DECODE_BRANCH_OPERANDS(name);             \
  for (uint16_t __i = 0; __i < (name)->size; ++__i) {   \
    IREE_VM_VERIFY_REG_ANY((name)->pairs[__i].src_reg); \
    IREE_VM_VERIFY_REG_ANY((name)->pairs[__i].dst_reg); \
  }

#define IREE_VM_ISA_VERIFY_VARIADIC_OPERANDS(name) \
  IREE_VM_ISA_DECODE_VARIADIC_OPERANDS(name)
#define IREE_VM_ISA_VERIFY_VARIADIC_RESULTS(name) \
  IREE_VM_ISA_DECODE_VARIADIC_RESULTS(name)

#define IREE_VM_ISA_VERIFY_VARIADIC_OPERANDS_I32(name) \
  IREE_VM_ISA_VERIFY_VARIADIC_OPERANDS(name);          \
  for (uint16_t __i = 0; __i < (name)->size; ++__i) {  \
    IREE_VM_VERIFY_REG_I32((name)->registers[__i]);    \
  }
#define IREE_VM_ISA_VERIFY_VARIADIC_OPERANDS_I64(name) \
  IREE_VM_ISA_VERIFY_VARIADIC_OPERANDS(name);          \
  for (uint16_t __i = 0; __i < (name)->size; ++__i) {  \
    IREE_VM_VERIFY_REG_I64((name)->registers[__i]);    \
  }
#define IREE_VM_ISA_VERIFY_VARIADIC_OPERANDS_F32(name) \
  IREE_VM_ISA_VERIFY_VARIADIC_OPERANDS(name);          \
  for (uint16_t __i = 0; __i < (name)->size; ++__i) {  \
    IREE_VM_VERIFY_REG_F32((name)->registers[__i]);    \
  }
#define IREE_VM_ISA_VERIFY_VARIADIC_OPERANDS_F64(name) \
  IREE_VM_ISA_VERIFY_VARIADIC_OPERANDS(name);          \
  for (uint16_t __i = 0; __i < (name)->size; ++__i) {  \
    IREE_VM_VERIFY_REG_F64((name)->registers[__i]);    \
  }
#define IREE_VM_ISA_VERIFY_VARIADIC_OPERANDS_REF(name, type_def) \
  IREE_VM_ISA_VERIFY_VARIADIC_OPERANDS(name);                    \
  for (uint16_t __i = 0; __i < (name)->size; ++__i) {            \
    IREE_VM_VERIFY_REG_REF((name)->registers[__i]);              \
  }                                                              \
  (void)(type_def)
#define IREE_VM_ISA_VERIFY_VARIADIC_OPERANDS_ANY(name) \
  IREE_VM_ISA_VERIFY_VARIADIC_OPERANDS(name);          \
  for (uint16_t __i = 0; __i < (name)->size; ++__i) {  \
    IREE_VM_VERIFY_REG_ANY((name)->registers[__i]);    \
  }

#define IREE_VM_ISA_VERIFY_VARIADIC_RESULTS_ANY(name) \
  IREE_VM_ISA_VERIFY_VARIADIC_OPERANDS_ANY(name)

#define IREE_VM_ISA_VERIFY_IMPORT_ORDINAL(name)                               \
  if (IREE_UNLIKELY((name) >= iree_vm_ImportFunctionDef_vec_len(              \
                                  verify_state->imported_functions))) {       \
    return iree_make_status(                                                  \
        IREE_STATUS_INVALID_ARGUMENT, "import ordinal %u out of range %zu",   \
        name,                                                                 \
        iree_vm_ImportFunctionDef_vec_len(verify_state->imported_functions)); \
  }
#define IREE_VM_ISA_VERIFY_FUNCTION_ORDINAL(name)                           \
  if (IREE_UNLIKELY((name)) >= iree_vm_FunctionDescriptor_vec_len(          \
                                   verify_state->function_descriptors)) {   \
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,                   \
                            "function ordinal %u out of range %zu", (name), \
                            iree_vm_FunctionDescriptor_vec_len(             \
                                verify_state->function_descriptors));       \
  }
#define IREE_VM_ISA_VERIFY_RWDATA_OFFSET(name, access_length)               \
  if (IREE_UNLIKELY(((name) + (access_length)) >                            \
                    verify_state->rwdata_storage_size)) {                   \
    return iree_make_status(                                                \
        IREE_STATUS_OUT_OF_RANGE,                                           \
        "global byte_offset out of range: %d (rwdata=%" PRIhsz ")", (name), \
        verify_state->rwdata_storage_size);                                 \
  }
#define IREE_VM_ISA_VERIFY_GLOBAL_REF_ORDINAL(name)                        \
  if (IREE_UNLIKELY((name) >= verify_state->global_ref_count)) {           \
    return iree_make_status(                                               \
        IREE_STATUS_OUT_OF_RANGE,                                          \
        "global ref ordinal out of range: %d (table=%" PRIhsz ")", (name), \
        verify_state->global_ref_count);                                   \
  }
#define IREE_VM_ISA_VERIFY_RODATA_ORDINAL(name)                            \
  if (IREE_UNLIKELY((name) >= verify_state->rodata_ref_count)) {           \
    return iree_make_status(                                               \
        IREE_STATUS_OUT_OF_RANGE,                                          \
        "rodata ref ordinal out of range: %d (table=%" PRIhsz ")", (name), \
        verify_state->rodata_ref_count);                                   \
  }
#define IREE_VM_ISA_VERIFY_OP_CORE_UNARY_I32(op_name) \
  IREE_VM_ISA_VERIFY_OP(CORE, op_name, {              \
    IREE_VM_ISA_DECODE_OPERAND_I32(operand);          \
    IREE_VM_ISA_DECODE_RESULT_I32(result);            \
  });

#define IREE_VM_ISA_VERIFY_OP_CORE_UNARY_I64(op_name) \
  IREE_VM_ISA_VERIFY_OP(CORE, op_name, {              \
    IREE_VM_ISA_DECODE_OPERAND_I64(operand);          \
    IREE_VM_ISA_DECODE_RESULT_I64(result);            \
  });

#define IREE_VM_ISA_VERIFY_OP_CORE_BINARY_I32(op_name) \
  IREE_VM_ISA_VERIFY_OP(CORE, op_name, {               \
    IREE_VM_ISA_DECODE_OPERAND_I32(lhs);               \
    IREE_VM_ISA_DECODE_OPERAND_I32(rhs);               \
    IREE_VM_ISA_DECODE_RESULT_I32(result);             \
  });

#define IREE_VM_ISA_VERIFY_OP_CORE_BINARY_I64(op_name) \
  IREE_VM_ISA_VERIFY_OP(CORE, op_name, {               \
    IREE_VM_ISA_DECODE_OPERAND_I64(lhs);               \
    IREE_VM_ISA_DECODE_OPERAND_I64(rhs);               \
    IREE_VM_ISA_DECODE_RESULT_I64(result);             \
  });

#define IREE_VM_ISA_VERIFY_OP_CORE_TERNARY_I32(op_name) \
  IREE_VM_ISA_VERIFY_OP(CORE, op_name, {                \
    IREE_VM_ISA_DECODE_OPERAND_I32(a);                  \
    IREE_VM_ISA_DECODE_OPERAND_I32(b);                  \
    IREE_VM_ISA_DECODE_OPERAND_I32(c);                  \
    IREE_VM_ISA_DECODE_RESULT_I32(result);              \
  });

#define IREE_VM_ISA_VERIFY_OP_CORE_TERNARY_I64(op_name) \
  IREE_VM_ISA_VERIFY_OP(CORE, op_name, {                \
    IREE_VM_ISA_DECODE_OPERAND_I64(a);                  \
    IREE_VM_ISA_DECODE_OPERAND_I64(b);                  \
    IREE_VM_ISA_DECODE_OPERAND_I64(c);                  \
    IREE_VM_ISA_DECODE_RESULT_I64(result);              \
  });

#define IREE_VM_ISA_VERIFY_OP_EXT_F32_UNARY_F32(op_name) \
  IREE_VM_ISA_VERIFY_OP(EXT_F32, op_name, {              \
    IREE_VM_ISA_DECODE_OPERAND_F32(operand);             \
    IREE_VM_ISA_DECODE_RESULT_F32(result);               \
  });

#define IREE_VM_ISA_VERIFY_OP_EXT_F32_BINARY_F32(op_name) \
  IREE_VM_ISA_VERIFY_OP(EXT_F32, op_name, {               \
    IREE_VM_ISA_DECODE_OPERAND_F32(lhs);                  \
    IREE_VM_ISA_DECODE_OPERAND_F32(rhs);                  \
    IREE_VM_ISA_DECODE_RESULT_F32(result);                \
  });

#define IREE_VM_ISA_VERIFY_OP_EXT_F32_TERNARY_F32(op_name) \
  IREE_VM_ISA_VERIFY_OP(EXT_F32, op_name, {                \
    IREE_VM_ISA_DECODE_OPERAND_F32(a);                     \
    IREE_VM_ISA_DECODE_OPERAND_F32(b);                     \
    IREE_VM_ISA_DECODE_OPERAND_F32(c);                     \
    IREE_VM_ISA_DECODE_RESULT_F32(result);                 \
  });

#define IREE_VM_ISA_VERIFY_OP_EXT_F64_UNARY_F64(op_name) \
  IREE_VM_ISA_VERIFY_OP(EXT_F64, op_name, {              \
    IREE_VM_ISA_DECODE_OPERAND_F64(operand);             \
    IREE_VM_ISA_DECODE_RESULT_F64(result);               \
  });

#define IREE_VM_ISA_VERIFY_OP_EXT_F64_BINARY_F64(op_name) \
  IREE_VM_ISA_VERIFY_OP(EXT_F64, op_name, {               \
    IREE_VM_ISA_DECODE_OPERAND_F64(lhs);                  \
    IREE_VM_ISA_DECODE_OPERAND_F64(rhs);                  \
    IREE_VM_ISA_DECODE_RESULT_F64(result);                \
  });

#define IREE_VM_ISA_VERIFY_OP_EXT_F64_TERNARY_F64(op_name) \
  IREE_VM_ISA_VERIFY_OP(EXT_F64, op_name, {                \
    IREE_VM_ISA_DECODE_OPERAND_F64(a);                     \
    IREE_VM_ISA_DECODE_OPERAND_F64(b);                     \
    IREE_VM_ISA_DECODE_OPERAND_F64(c);                     \
    IREE_VM_ISA_DECODE_RESULT_F64(result);                 \
  });

//===----------------------------------------------------------------------===//
// Call verification
//===----------------------------------------------------------------------===//

static iree_status_t iree_vm_bytecode_function_get_cconv_fragments(
    iree_vm_FunctionSignatureDef_table_t signature_def,
    iree_string_view_t* out_arguments, iree_string_view_t* out_results) {
  flatbuffers_string_t cconv_str =
      iree_vm_FunctionSignatureDef_calling_convention(signature_def);
  iree_vm_function_signature_t signature = {
      .calling_convention =
          iree_make_string_view(cconv_str, flatbuffers_string_len(cconv_str)),
  };
  return iree_vm_function_call_get_cconv_fragments(&signature, out_arguments,
                                                   out_results);
}

static iree_status_t iree_vm_bytecode_function_count_cconv_regs(
    iree_string_view_t cconv_fragment, iree_host_size_t* out_i32_count,
    iree_host_size_t* out_ref_count) {
  iree_host_size_t i32_count = 0;
  iree_host_size_t ref_count = 0;
  for (iree_host_size_t i = 0; i < cconv_fragment.size; ++i) {
    switch (cconv_fragment.data[i]) {
      default:
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "unsupported cconv fragment char '%c'",
                                cconv_fragment.data[i]);
      case IREE_VM_CCONV_TYPE_VOID:
        break;
      case IREE_VM_CCONV_TYPE_I32:
      case IREE_VM_CCONV_TYPE_F32:
        ++i32_count;
        break;
      case IREE_VM_CCONV_TYPE_I64:
      case IREE_VM_CCONV_TYPE_F64:
        if ((i32_count % 2) != 0) {
          // Unaligned; pad an i32 register to get to i64 alignment.
          ++i32_count;
        }
        i32_count += 2;
        break;
      case IREE_VM_CCONV_TYPE_REF:
        ++ref_count;
        break;
      case IREE_VM_CCONV_TYPE_SPAN_START:
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "internal functions cannot accept variadic arguments");
    }
  }
  *out_i32_count = i32_count;
  *out_ref_count = ref_count;
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_function_verify_arguments(
    const iree_vm_bytecode_verify_state_t* verify_state) {
  iree_host_size_t args_i32 = 0;
  iree_host_size_t args_ref = 0;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_function_count_cconv_regs(
      verify_state->cconv_arguments, &args_i32, &args_ref));
  if (verify_state->i32_register_count < args_i32 ||
      verify_state->ref_register_count < args_ref) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "insufficient register storage for function arguments/results");
  }
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_function_verify_cconv_register(
    const iree_vm_bytecode_verify_state_t* verify_state, char cconv_type,
    const iree_vm_register_list_t* IREE_RESTRICT reg_list, int reg_i) {
  if (reg_i >= reg_list->size) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "register list underflow (have %u, trying to access %u)",
        reg_list->size, reg_i);
  }
  switch (cconv_type) {
    case IREE_VM_CCONV_TYPE_I32:
    case IREE_VM_CCONV_TYPE_F32: {
      IREE_VM_VERIFY_REG_ORDINAL_X32(reg_list->registers[reg_i], "i32/f32");
    } break;
    case IREE_VM_CCONV_TYPE_I64:
    case IREE_VM_CCONV_TYPE_F64: {
      IREE_VM_VERIFY_REG_ORDINAL_X64(reg_list->registers[reg_i], "i64/f64");
    } break;
    case IREE_VM_CCONV_TYPE_REF: {
      IREE_VM_VERIFY_REG_REF(reg_list->registers[reg_i]);
    } break;
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported cconv fragment char '%c'",
                              cconv_type);
  }
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_function_verify_cconv_registers(
    const iree_vm_bytecode_verify_state_t* verify_state,
    iree_string_view_t cconv_fragment,
    const iree_vm_register_list_t* IREE_RESTRICT segment_size_list,
    const iree_vm_register_list_t* IREE_RESTRICT reg_list) {
  for (uint16_t i = 0, seg_i = 0, reg_i = 0; i < cconv_fragment.size;
       ++i, ++seg_i) {
    switch (cconv_fragment.data[i]) {
      case IREE_VM_CCONV_TYPE_VOID:
        break;
      case IREE_VM_CCONV_TYPE_I32:
      case IREE_VM_CCONV_TYPE_F32:
      case IREE_VM_CCONV_TYPE_I64:
      case IREE_VM_CCONV_TYPE_F64:
      case IREE_VM_CCONV_TYPE_REF: {
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_function_verify_cconv_register(
            verify_state, cconv_fragment.data[i], reg_list, reg_i++));
      } break;
      case IREE_VM_CCONV_TYPE_SPAN_START: {
        if (!segment_size_list) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "function is variadic but no segment size list provided");
        } else if (seg_i >= segment_size_list->size) {
          return iree_make_status(
              IREE_STATUS_OUT_OF_RANGE,
              "segment size list underflow (have %u, trying to access %u)",
              segment_size_list->size, seg_i);
        }
        uint16_t span_count = segment_size_list->registers[seg_i];
        if (!span_count) {
          // No items; skip the span.
          do {
            ++i;
          } while (i < cconv_fragment.size &&
                   cconv_fragment.data[i] != IREE_VM_CCONV_TYPE_SPAN_END);
          continue;
        }
        uint16_t span_start_i = i + 1;
        for (uint16_t j = 0; j < span_count; ++j) {
          for (i = span_start_i;
               i < cconv_fragment.size &&
               cconv_fragment.data[i] != IREE_VM_CCONV_TYPE_SPAN_END;
               ++i) {
            // TODO(benvanik): share with switch above.
            switch (cconv_fragment.data[i]) {
              case IREE_VM_CCONV_TYPE_VOID:
                break;
              case IREE_VM_CCONV_TYPE_I32:
              case IREE_VM_CCONV_TYPE_F32:
              case IREE_VM_CCONV_TYPE_I64:
              case IREE_VM_CCONV_TYPE_F64:
              case IREE_VM_CCONV_TYPE_REF: {
                IREE_RETURN_IF_ERROR(
                    iree_vm_bytecode_function_verify_cconv_register(
                        verify_state, cconv_fragment.data[i], reg_list,
                        reg_i++));
              } break;
              default:
                return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                        "unsupported cconv fragment char '%c'",
                                        cconv_fragment.data[i]);
            }
          }
        }
      } break;
      default:
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "unsupported cconv fragment char '%c'",
                                cconv_fragment.data[i]);
    }
  }
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_function_verify_call(
    const iree_vm_bytecode_verify_state_t* verify_state,
    iree_vm_FunctionSignatureDef_table_t signature_def,
    const iree_vm_register_list_t* IREE_RESTRICT segment_size_list,
    const iree_vm_register_list_t* IREE_RESTRICT src_reg_list,
    const iree_vm_register_list_t* IREE_RESTRICT dst_reg_list) {
  iree_string_view_t arguments = iree_string_view_empty();
  iree_string_view_t results = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_function_get_cconv_fragments(
      signature_def, &arguments, &results));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_function_verify_cconv_registers(
      verify_state, arguments, segment_size_list, src_reg_list));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_function_verify_cconv_registers(
      verify_state, results, /*segment_sizes=*/NULL, dst_reg_list));
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Bytecode verification
//===----------------------------------------------------------------------===//

#define IREE_VM_ISA_VERIFY_OP(ext, op_name, body) \
  case IREE_VM_OP_##ext##_##op_name: {            \
    body;                                         \
  } break;

#define BEGIN_VERIFY_PREFIX(op_name, ext)    \
  case IREE_VM_OP_CORE_##op_name: {          \
    IREE_VM_VERIFY_PC_RANGE(pc + 1, max_pc); \
    IREE_VM_VERIFY_REQUIREMENT(ext);         \
    switch (bytecode_data[pc++]) {
#define END_VERIFY_PREFIX(op_name, ext)                               \
  default:                                                            \
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,             \
                            "unhandled ext " #op_name " opcode %02X", \
                            bytecode_data[pc - 1]);                   \
    }                                                                 \
    break;                                                            \
    }
#define UNHANDLED_VERIFY_PREFIX(op_name, ext)                                 \
  case IREE_VM_OP_CORE_##op_name: {                                           \
    return iree_vm_check_feature_mismatch(__FILE__, __LINE__, ext,            \
                                          function_descriptor->requirements); \
  }

static iree_status_t iree_vm_bytecode_function_verify_bytecode_op(
    iree_vm_bytecode_module_t* module,
    iree_vm_bytecode_verify_state_t* verify_state,
    iree_vm_FunctionSignatureDef_table_t function_signature,
    iree_vm_FunctionDescriptor_struct_t function_descriptor,
    iree_const_byte_span_t function_bytecode, iree_vm_source_offset_t start_pc,
    iree_vm_source_offset_t max_pc, iree_vm_source_offset_t* out_next_pc) {
  *out_next_pc = 0;
  iree_vm_source_offset_t pc = start_pc;
  const uint8_t* bytecode_data = function_bytecode.data;

  // NOTE: we keep this as simple as possible so that we can one day auto
  // generate it from tblgen, which has all the encodings in a similar form
  // such that we could do string substitution to get the verifier macros.

  // All ops except for Block must be inside of a block. We hoist that check
  // that here before switching out.
  IREE_VM_VERIFY_PC_RANGE(pc + 1, max_pc);
  if (verify_state->in_block == 0) {
    // If not in a block then the next opcode must be a block.
    if (bytecode_data[pc] != IREE_VM_OP_CORE_Block) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "op at pc %08" PRIX64 " is not in a block", pc);
    }
  } else {
    // If in a block then the next opcode must not be a block.
    if (bytecode_data[pc] == IREE_VM_OP_CORE_Block) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "op at pc %08" PRIX64 " is a block while still in a block", pc);
    }
  }

  // Get primary opcode. All ops have at least 1 byte.
  switch (bytecode_data[pc++]) {
    //===------------------------------------------------------------------===//
    // Globals
    //===------------------------------------------------------------------===//

    IREE_VM_ISA_VERIFY_OP(CORE, GlobalLoadI32, {
      IREE_VM_ISA_DECODE_GLOBAL_ATTR(byte_offset);
      IREE_VM_ISA_VERIFY_RWDATA_OFFSET(byte_offset, 4);
      IREE_VM_ISA_DECODE_RESULT_I32(value);
    });

    IREE_VM_ISA_VERIFY_OP(CORE, GlobalStoreI32, {
      IREE_VM_ISA_DECODE_GLOBAL_ATTR(byte_offset);
      IREE_VM_ISA_VERIFY_RWDATA_OFFSET(byte_offset, 4);
      IREE_VM_ISA_DECODE_OPERAND_I32(value);
    });

    IREE_VM_ISA_VERIFY_OP(CORE, GlobalLoadIndirectI32, {
      IREE_VM_ISA_DECODE_OPERAND_I32(byte_offset);
      // NOTE: we have to verify the offset at runtime.
      IREE_VM_ISA_DECODE_RESULT_I32(value);
    });

    IREE_VM_ISA_VERIFY_OP(CORE, GlobalStoreIndirectI32, {
      IREE_VM_ISA_DECODE_OPERAND_I32(byte_offset);
      // NOTE: we have to verify the offset at runtime.
      IREE_VM_ISA_DECODE_OPERAND_I32(value);
    });

    IREE_VM_ISA_VERIFY_OP(CORE, GlobalLoadI64, {
      IREE_VM_ISA_DECODE_GLOBAL_ATTR(byte_offset);
      IREE_VM_ISA_VERIFY_RWDATA_OFFSET(byte_offset, 8);
      IREE_VM_ISA_DECODE_RESULT_I64(value);
    });

    IREE_VM_ISA_VERIFY_OP(CORE, GlobalStoreI64, {
      IREE_VM_ISA_DECODE_GLOBAL_ATTR(byte_offset);
      IREE_VM_ISA_VERIFY_RWDATA_OFFSET(byte_offset, 8);
      IREE_VM_ISA_DECODE_OPERAND_I64(value);
    });

    IREE_VM_ISA_VERIFY_OP(CORE, GlobalLoadIndirectI64, {
      IREE_VM_ISA_DECODE_OPERAND_I32(byte_offset);
      // NOTE: we have to verify the offset at runtime.
      IREE_VM_ISA_DECODE_RESULT_I64(value);
    });

    IREE_VM_ISA_VERIFY_OP(CORE, GlobalStoreIndirectI64, {
      IREE_VM_ISA_DECODE_OPERAND_I32(byte_offset);
      // NOTE: we have to verify the offset at runtime.
      IREE_VM_ISA_DECODE_OPERAND_I64(value);
    });

    IREE_VM_ISA_VERIFY_OP(CORE, GlobalLoadRef, {
      IREE_VM_ISA_DECODE_GLOBAL_ATTR(global);
      IREE_VM_ISA_VERIFY_GLOBAL_REF_ORDINAL(global);
      IREE_VM_ISA_DECODE_TYPE_OF(type_def);
      IREE_VM_ISA_DECODE_RESULT_REF_MOVE(value);
    });

    IREE_VM_ISA_VERIFY_OP(CORE, GlobalStoreRef, {
      IREE_VM_ISA_DECODE_GLOBAL_ATTR(global);
      IREE_VM_ISA_VERIFY_GLOBAL_REF_ORDINAL(global);
      IREE_VM_ISA_DECODE_TYPE_OF(type_def);
      IREE_VM_ISA_DECODE_OPERAND_REF_MOVE(value);
    });

    IREE_VM_ISA_VERIFY_OP(CORE, GlobalLoadIndirectRef, {
      IREE_VM_ISA_DECODE_OPERAND_I32(global);
      // NOTE: we have to verify the ordinal at runtime.
      IREE_VM_ISA_DECODE_TYPE_OF(type_def);
      IREE_VM_ISA_DECODE_RESULT_REF_MOVE(value);
    });

    IREE_VM_ISA_VERIFY_OP(CORE, GlobalStoreIndirectRef, {
      IREE_VM_ISA_DECODE_OPERAND_I32(global);
      // NOTE: we have to verify the ordinal at runtime.
      IREE_VM_ISA_DECODE_TYPE_OF(type_def);
      IREE_VM_ISA_DECODE_OPERAND_REF_MOVE(value);
    });

    //===------------------------------------------------------------------===//
    // Constants
    //===------------------------------------------------------------------===//

    IREE_VM_ISA_VERIFY_OP(CORE, ConstI32, {
      IREE_VM_ISA_DECODE_ATTR_I32(value);
      IREE_VM_ISA_DECODE_RESULT_I32(result);
    });

    IREE_VM_ISA_VERIFY_OP(CORE, ConstI32Zero,
                          { IREE_VM_ISA_DECODE_RESULT_I32(result); });

    IREE_VM_ISA_VERIFY_OP(CORE, ConstI64, {
      IREE_VM_ISA_DECODE_ATTR_I64(value);
      IREE_VM_ISA_DECODE_RESULT_I64(result);
    });

    IREE_VM_ISA_VERIFY_OP(CORE, ConstI64Zero,
                          { IREE_VM_ISA_DECODE_RESULT_I64(result); });

    IREE_VM_ISA_VERIFY_OP(CORE, ConstRefZero,
                          { IREE_VM_ISA_DECODE_RESULT_REF_MOVE(result); });

    IREE_VM_ISA_VERIFY_OP(CORE, DiscardRefs,
                          { IREE_VM_ISA_VERIFY_VARIADIC_OPERANDS_ANY(refs); });

    IREE_VM_ISA_VERIFY_OP(CORE, AssignRef, {
      IREE_VM_ISA_DECODE_OPERAND_REF_MOVE(source);
      IREE_VM_ISA_DECODE_RESULT_REF_MOVE(result);
    });

    IREE_VM_ISA_VERIFY_OP(CORE, ConstRefRodata, {
      IREE_VM_ISA_DECODE_RODATA_ATTR(rodata);
      IREE_VM_ISA_VERIFY_RODATA_ORDINAL(rodata);
      IREE_VM_ISA_DECODE_RESULT_REF_MOVE(value);
    });

    //===------------------------------------------------------------------===//
    // Buffers
    //===------------------------------------------------------------------===//

    IREE_VM_ISA_VERIFY_OP(CORE, BufferAlloc, {
      IREE_VM_ISA_DECODE_OPERAND_I64(length);
      IREE_VM_ISA_DECODE_OPERAND_I32(alignment);
      IREE_VM_ISA_DECODE_RESULT_REF_MOVE(result);
    });

    IREE_VM_ISA_VERIFY_OP(CORE, BufferClone, {
      IREE_VM_ISA_DECODE_OPERAND_REF(source);  // Source buffer - no MOVE.
      IREE_VM_ISA_DECODE_OPERAND_I64(offset);
      IREE_VM_ISA_DECODE_OPERAND_I64(length);
      IREE_VM_ISA_DECODE_OPERAND_I32(alignment);
      IREE_VM_ISA_DECODE_RESULT_REF_MOVE(result);
    });

    IREE_VM_ISA_VERIFY_OP(CORE, BufferLength, {
      IREE_VM_ISA_DECODE_OPERAND_REF(buffer);
      IREE_VM_ISA_DECODE_RESULT_I64(result);
    });

    IREE_VM_ISA_VERIFY_OP(CORE, BufferCopy, {
      IREE_VM_ISA_DECODE_OPERAND_REF(source_buffer);
      IREE_VM_ISA_DECODE_OPERAND_I64(source_offset);
      IREE_VM_ISA_DECODE_OPERAND_REF(target_buffer);
      IREE_VM_ISA_DECODE_OPERAND_I64(target_offset);
      IREE_VM_ISA_DECODE_OPERAND_I64(length);
    });

    IREE_VM_ISA_VERIFY_OP(CORE, BufferCompare, {
      IREE_VM_ISA_DECODE_OPERAND_REF(lhs_buffer);
      IREE_VM_ISA_DECODE_OPERAND_I64(lhs_offset);
      IREE_VM_ISA_DECODE_OPERAND_REF(rhs_buffer);
      IREE_VM_ISA_DECODE_OPERAND_I64(rhs_offset);
      IREE_VM_ISA_DECODE_OPERAND_I64(length);
      IREE_VM_ISA_DECODE_RESULT_I32(result);
    });

    IREE_VM_ISA_VERIFY_OP(CORE, BufferFillI8, {
      IREE_VM_ISA_DECODE_OPERAND_REF(target_buffer);
      IREE_VM_ISA_DECODE_OPERAND_I64(target_offset);
      IREE_VM_ISA_DECODE_OPERAND_I64(length);
      IREE_VM_ISA_DECODE_OPERAND_I32(value);
    });
    IREE_VM_ISA_VERIFY_OP(CORE, BufferFillI16, {
      IREE_VM_ISA_DECODE_OPERAND_REF(target_buffer);
      IREE_VM_ISA_DECODE_OPERAND_I64(target_offset);
      IREE_VM_ISA_DECODE_OPERAND_I64(length);
      IREE_VM_ISA_DECODE_OPERAND_I32(value);
    });
    IREE_VM_ISA_VERIFY_OP(CORE, BufferFillI32, {
      IREE_VM_ISA_DECODE_OPERAND_REF(target_buffer);
      IREE_VM_ISA_DECODE_OPERAND_I64(target_offset);
      IREE_VM_ISA_DECODE_OPERAND_I64(length);
      IREE_VM_ISA_DECODE_OPERAND_I32(value);
    });
    IREE_VM_ISA_VERIFY_OP(CORE, BufferFillI64, {
      IREE_VM_ISA_DECODE_OPERAND_REF(target_buffer);
      IREE_VM_ISA_DECODE_OPERAND_I64(target_offset);
      IREE_VM_ISA_DECODE_OPERAND_I64(length);
      IREE_VM_ISA_DECODE_OPERAND_I64(value);
    });

    IREE_VM_ISA_VERIFY_OP(CORE, BufferLoadI8U, {
      IREE_VM_ISA_DECODE_OPERAND_REF(source_buffer);
      IREE_VM_ISA_DECODE_OPERAND_I64(source_offset);
      IREE_VM_ISA_DECODE_RESULT_I32(result);
    });
    IREE_VM_ISA_VERIFY_OP(CORE, BufferLoadI8S, {
      IREE_VM_ISA_DECODE_OPERAND_REF(source_buffer);
      IREE_VM_ISA_DECODE_OPERAND_I64(source_offset);
      IREE_VM_ISA_DECODE_RESULT_I32(result);
    });
    IREE_VM_ISA_VERIFY_OP(CORE, BufferLoadI16U, {
      IREE_VM_ISA_DECODE_OPERAND_REF(source_buffer);
      IREE_VM_ISA_DECODE_OPERAND_I64(source_offset);
      IREE_VM_ISA_DECODE_RESULT_I32(result);
    });
    IREE_VM_ISA_VERIFY_OP(CORE, BufferLoadI16S, {
      IREE_VM_ISA_DECODE_OPERAND_REF(source_buffer);
      IREE_VM_ISA_DECODE_OPERAND_I64(source_offset);
      IREE_VM_ISA_DECODE_RESULT_I32(result);
    });
    IREE_VM_ISA_VERIFY_OP(CORE, BufferLoadI32, {
      IREE_VM_ISA_DECODE_OPERAND_REF(source_buffer);
      IREE_VM_ISA_DECODE_OPERAND_I64(source_offset);
      IREE_VM_ISA_DECODE_RESULT_I32(result);
    });
    IREE_VM_ISA_VERIFY_OP(CORE, BufferLoadI64, {
      IREE_VM_ISA_DECODE_OPERAND_REF(source_buffer);
      IREE_VM_ISA_DECODE_OPERAND_I64(source_offset);
      IREE_VM_ISA_DECODE_RESULT_I64(result);
    });

    IREE_VM_ISA_VERIFY_OP(CORE, BufferStoreI8, {
      IREE_VM_ISA_DECODE_OPERAND_REF(target_buffer);
      IREE_VM_ISA_DECODE_OPERAND_I64(target_offset);
      IREE_VM_ISA_DECODE_OPERAND_I32(value);
    });
    IREE_VM_ISA_VERIFY_OP(CORE, BufferStoreI16, {
      IREE_VM_ISA_DECODE_OPERAND_REF(target_buffer);
      IREE_VM_ISA_DECODE_OPERAND_I64(target_offset);
      IREE_VM_ISA_DECODE_OPERAND_I32(value);
    });
    IREE_VM_ISA_VERIFY_OP(CORE, BufferStoreI32, {
      IREE_VM_ISA_DECODE_OPERAND_REF(target_buffer);
      IREE_VM_ISA_DECODE_OPERAND_I64(target_offset);
      IREE_VM_ISA_DECODE_OPERAND_I32(value);
    });
    IREE_VM_ISA_VERIFY_OP(CORE, BufferStoreI64, {
      IREE_VM_ISA_DECODE_OPERAND_REF(target_buffer);
      IREE_VM_ISA_DECODE_OPERAND_I64(target_offset);
      IREE_VM_ISA_DECODE_OPERAND_I64(value);
    });
    IREE_VM_ISA_VERIFY_OP(CORE, BufferHash, {
      IREE_VM_ISA_DECODE_OPERAND_REF(source_buffer);
      IREE_VM_ISA_DECODE_OPERAND_I64(source_offset);
      IREE_VM_ISA_DECODE_OPERAND_I64(length);
      IREE_VM_ISA_DECODE_RESULT_I64(result);
    });

    //===------------------------------------------------------------------===//
    // Lists
    //===------------------------------------------------------------------===//

    IREE_VM_ISA_VERIFY_OP(CORE, ListAlloc, {
      IREE_VM_ISA_DECODE_TYPE_OF(element_type);
      IREE_VM_ISA_DECODE_OPERAND_I32(initial_capacity);
      IREE_VM_ISA_DECODE_RESULT_REF_MOVE(result);
    });

    IREE_VM_ISA_VERIFY_OP(CORE, ListReserve, {
      IREE_VM_ISA_DECODE_OPERAND_REF(list);
      IREE_VM_ISA_DECODE_OPERAND_I32(minimum_capacity);
    });

    IREE_VM_ISA_VERIFY_OP(CORE, ListSize, {
      IREE_VM_ISA_DECODE_OPERAND_REF(list);
      IREE_VM_ISA_DECODE_RESULT_I32(result);
    });

    IREE_VM_ISA_VERIFY_OP(CORE, ListResize, {
      IREE_VM_ISA_DECODE_OPERAND_REF(list);
      IREE_VM_ISA_DECODE_OPERAND_I32(new_size);
    });

    IREE_VM_ISA_VERIFY_OP(CORE, ListGetI32, {
      IREE_VM_ISA_DECODE_OPERAND_REF(list);
      IREE_VM_ISA_DECODE_OPERAND_I32(index);
      IREE_VM_ISA_DECODE_RESULT_I32(result);
    });

    IREE_VM_ISA_VERIFY_OP(CORE, ListSetI32, {
      IREE_VM_ISA_DECODE_OPERAND_REF(list);
      IREE_VM_ISA_DECODE_OPERAND_I32(index);
      IREE_VM_ISA_DECODE_OPERAND_I32(raw_value);
    });

    IREE_VM_ISA_VERIFY_OP(CORE, ListGetI64, {
      IREE_VM_ISA_DECODE_OPERAND_REF(list);
      IREE_VM_ISA_DECODE_OPERAND_I32(index);
      IREE_VM_ISA_DECODE_RESULT_I64(result);
    });

    IREE_VM_ISA_VERIFY_OP(CORE, ListSetI64, {
      IREE_VM_ISA_DECODE_OPERAND_REF(list);
      IREE_VM_ISA_DECODE_OPERAND_I32(index);
      IREE_VM_ISA_DECODE_OPERAND_I64(value);
    });

    IREE_VM_ISA_VERIFY_OP(CORE, ListGetRef, {
      IREE_VM_ISA_DECODE_OPERAND_REF(list);  // List operand - no MOVE.
      IREE_VM_ISA_DECODE_OPERAND_I32(index);
      IREE_VM_ISA_DECODE_TYPE_OF(type_def);
      IREE_VM_ISA_DECODE_RESULT_REF_MOVE(result);
    });

    IREE_VM_ISA_VERIFY_OP(CORE, ListSetRef, {
      IREE_VM_ISA_DECODE_OPERAND_REF(list);  // List operand - no MOVE.
      IREE_VM_ISA_DECODE_OPERAND_I32(index);
      IREE_VM_ISA_DECODE_OPERAND_REF_MOVE(value);
    });

    //===------------------------------------------------------------------===//
    // Conditional assignment
    //===------------------------------------------------------------------===//

    IREE_VM_ISA_VERIFY_OP(CORE, SelectI32, {
      IREE_VM_ISA_DECODE_OPERAND_I32(condition);
      IREE_VM_ISA_DECODE_OPERAND_I32(true_value);
      IREE_VM_ISA_DECODE_OPERAND_I32(false_value);
      IREE_VM_ISA_DECODE_RESULT_I32(result);
    });

    IREE_VM_ISA_VERIFY_OP(CORE, SelectI64, {
      IREE_VM_ISA_DECODE_OPERAND_I32(condition);
      IREE_VM_ISA_DECODE_OPERAND_I64(true_value);
      IREE_VM_ISA_DECODE_OPERAND_I64(false_value);
      IREE_VM_ISA_DECODE_RESULT_I64(result);
    });

    IREE_VM_ISA_VERIFY_OP(CORE, SelectRef, {
      IREE_VM_ISA_DECODE_OPERAND_I32(condition);
      // TODO(benvanik): remove the type_id and use either LHS/RHS (if both are
      // null then output is always null so no need to know the type).
      IREE_VM_ISA_DECODE_TYPE_OF(true_value_type_def);
      IREE_VM_ISA_DECODE_OPERAND_REF_MOVE(true_value);
      IREE_VM_ISA_DECODE_OPERAND_REF_MOVE(false_value);
      IREE_VM_ISA_DECODE_RESULT_REF_MOVE(result);
    });

    IREE_VM_ISA_VERIFY_OP(CORE, SwitchI32, {
      IREE_VM_ISA_DECODE_OPERAND_I32(index);
      IREE_VM_ISA_DECODE_OPERAND_I32(default_value);
      IREE_VM_ISA_VERIFY_VARIADIC_OPERANDS_I32(values);
      IREE_VM_ISA_DECODE_RESULT_I32(result);
    });

    IREE_VM_ISA_VERIFY_OP(CORE, SwitchI64, {
      IREE_VM_ISA_DECODE_OPERAND_I32(index);
      IREE_VM_ISA_DECODE_OPERAND_I64(default_value);
      IREE_VM_ISA_VERIFY_VARIADIC_OPERANDS_I64(values);
      IREE_VM_ISA_DECODE_RESULT_I64(result);
    });

    IREE_VM_ISA_VERIFY_OP(CORE, SwitchRef, {
      IREE_VM_ISA_DECODE_OPERAND_I32(index);
      IREE_VM_ISA_DECODE_TYPE_OF(type_def);
      IREE_VM_ISA_DECODE_OPERAND_REF_MOVE(default_value);
      IREE_VM_ISA_VERIFY_VARIADIC_OPERANDS_REF(
          values,
          type_def);  // TODO: add Move variant.
      IREE_VM_ISA_DECODE_RESULT_REF_MOVE(result);
    });

    //===------------------------------------------------------------------===//
    // Native integer arithmetic
    //===------------------------------------------------------------------===//

    IREE_VM_ISA_VERIFY_OP_CORE_BINARY_I32(AddI32);
    IREE_VM_ISA_VERIFY_OP_CORE_BINARY_I32(SubI32);
    IREE_VM_ISA_VERIFY_OP_CORE_BINARY_I32(MulI32);
    IREE_VM_ISA_VERIFY_OP_CORE_BINARY_I32(DivI32S);
    IREE_VM_ISA_VERIFY_OP_CORE_BINARY_I32(DivI32U);
    IREE_VM_ISA_VERIFY_OP_CORE_BINARY_I32(RemI32S);
    IREE_VM_ISA_VERIFY_OP_CORE_BINARY_I32(RemI32U);
    IREE_VM_ISA_VERIFY_OP_CORE_TERNARY_I32(FMAI32);
    IREE_VM_ISA_VERIFY_OP_CORE_UNARY_I32(AbsI32);
    IREE_VM_ISA_VERIFY_OP_CORE_BINARY_I32(MinI32S);
    IREE_VM_ISA_VERIFY_OP_CORE_BINARY_I32(MinI32U);
    IREE_VM_ISA_VERIFY_OP_CORE_BINARY_I32(MaxI32S);
    IREE_VM_ISA_VERIFY_OP_CORE_BINARY_I32(MaxI32U);
    IREE_VM_ISA_VERIFY_OP_CORE_UNARY_I32(NotI32);
    IREE_VM_ISA_VERIFY_OP_CORE_BINARY_I32(AndI32);
    IREE_VM_ISA_VERIFY_OP_CORE_BINARY_I32(OrI32);
    IREE_VM_ISA_VERIFY_OP_CORE_BINARY_I32(XorI32);
    IREE_VM_ISA_VERIFY_OP_CORE_UNARY_I32(CtlzI32);

    IREE_VM_ISA_VERIFY_OP_CORE_BINARY_I64(AddI64);
    IREE_VM_ISA_VERIFY_OP_CORE_BINARY_I64(SubI64);
    IREE_VM_ISA_VERIFY_OP_CORE_BINARY_I64(MulI64);
    IREE_VM_ISA_VERIFY_OP_CORE_BINARY_I64(DivI64S);
    IREE_VM_ISA_VERIFY_OP_CORE_BINARY_I64(DivI64U);
    IREE_VM_ISA_VERIFY_OP_CORE_BINARY_I64(RemI64S);
    IREE_VM_ISA_VERIFY_OP_CORE_BINARY_I64(RemI64U);
    IREE_VM_ISA_VERIFY_OP_CORE_TERNARY_I64(FMAI64);
    IREE_VM_ISA_VERIFY_OP_CORE_UNARY_I64(AbsI64);
    IREE_VM_ISA_VERIFY_OP_CORE_BINARY_I64(MinI64S);
    IREE_VM_ISA_VERIFY_OP_CORE_BINARY_I64(MinI64U);
    IREE_VM_ISA_VERIFY_OP_CORE_BINARY_I64(MaxI64S);
    IREE_VM_ISA_VERIFY_OP_CORE_BINARY_I64(MaxI64U);
    IREE_VM_ISA_VERIFY_OP_CORE_UNARY_I64(NotI64);
    IREE_VM_ISA_VERIFY_OP_CORE_BINARY_I64(AndI64);
    IREE_VM_ISA_VERIFY_OP_CORE_BINARY_I64(OrI64);
    IREE_VM_ISA_VERIFY_OP_CORE_BINARY_I64(XorI64);
    IREE_VM_ISA_VERIFY_OP_CORE_UNARY_I64(CtlzI64);

    //===------------------------------------------------------------------===//
    // Casting and type conversion/emulation
    //===------------------------------------------------------------------===//

    // NOTE: these all operate on 32-bit registers.
    IREE_VM_ISA_VERIFY_OP_CORE_UNARY_I32(TruncI32I8);
    IREE_VM_ISA_VERIFY_OP_CORE_UNARY_I32(TruncI32I16);
    IREE_VM_ISA_VERIFY_OP_CORE_UNARY_I32(ExtI8I32S);
    IREE_VM_ISA_VERIFY_OP_CORE_UNARY_I32(ExtI8I32U);
    IREE_VM_ISA_VERIFY_OP_CORE_UNARY_I32(ExtI16I32S);
    IREE_VM_ISA_VERIFY_OP_CORE_UNARY_I32(ExtI16I32U);

    // NOTE: 64-bit ones are actually changing register widths.
    IREE_VM_ISA_VERIFY_OP(CORE, TruncI64I32, {
      IREE_VM_ISA_DECODE_OPERAND_I64(operand);
      IREE_VM_ISA_DECODE_RESULT_I32(result);
    });
    IREE_VM_ISA_VERIFY_OP(CORE, ExtI32I64S, {
      IREE_VM_ISA_DECODE_OPERAND_I32(operand);
      IREE_VM_ISA_DECODE_RESULT_I64(result);
    });
    IREE_VM_ISA_VERIFY_OP(CORE, ExtI32I64U, {
      IREE_VM_ISA_DECODE_OPERAND_I32(operand);
      IREE_VM_ISA_DECODE_RESULT_I64(result);
    });

    IREE_VM_ISA_VERIFY_OP(CORE, CastAnyRef, {
      IREE_VM_ISA_DECODE_OPERAND_REF_MOVE(operand);
      IREE_VM_ISA_DECODE_TYPE_OF(result_type);
      IREE_VM_ISA_DECODE_RESULT_REF_MOVE(result);
    });

    //===------------------------------------------------------------------===//
    // Native bitwise shifts and rotates
    //===------------------------------------------------------------------===//

#define IREE_VM_ISA_VERIFY_OP_CORE_SHIFT_I32(op_name) \
  IREE_VM_ISA_VERIFY_OP(CORE, op_name, {              \
    IREE_VM_ISA_DECODE_OPERAND_I32(operand);          \
    IREE_VM_ISA_DECODE_OPERAND_I32(amount);           \
    IREE_VM_ISA_DECODE_RESULT_I32(result);            \
  });

    IREE_VM_ISA_VERIFY_OP_CORE_SHIFT_I32(ShlI32);
    IREE_VM_ISA_VERIFY_OP_CORE_SHIFT_I32(ShrI32S);
    IREE_VM_ISA_VERIFY_OP_CORE_SHIFT_I32(ShrI32U);

#define IREE_VM_ISA_VERIFY_OP_CORE_SHIFT_I64(op_name) \
  IREE_VM_ISA_VERIFY_OP(CORE, op_name, {              \
    IREE_VM_ISA_DECODE_OPERAND_I64(operand);          \
    IREE_VM_ISA_DECODE_OPERAND_I32(amount);           \
    IREE_VM_ISA_DECODE_RESULT_I64(result);            \
  });

    IREE_VM_ISA_VERIFY_OP_CORE_SHIFT_I64(ShlI64);
    IREE_VM_ISA_VERIFY_OP_CORE_SHIFT_I64(ShrI64S);
    IREE_VM_ISA_VERIFY_OP_CORE_SHIFT_I64(ShrI64U);

    //===------------------------------------------------------------------===//
    // Comparison ops
    //===------------------------------------------------------------------===//

    IREE_VM_ISA_VERIFY_OP_CORE_BINARY_I32(CmpEQI32);
    IREE_VM_ISA_VERIFY_OP_CORE_BINARY_I32(CmpNEI32);
    IREE_VM_ISA_VERIFY_OP_CORE_BINARY_I32(CmpLTI32S);
    IREE_VM_ISA_VERIFY_OP_CORE_BINARY_I32(CmpLTI32U);
    IREE_VM_ISA_VERIFY_OP_CORE_UNARY_I32(CmpNZI32);

#define IREE_VM_ISA_VERIFY_OP_CORE_CMP_I64(op_name) \
  IREE_VM_ISA_VERIFY_OP(CORE, op_name, {            \
    IREE_VM_ISA_DECODE_OPERAND_I64(lhs);            \
    IREE_VM_ISA_DECODE_OPERAND_I64(rhs);            \
    IREE_VM_ISA_DECODE_RESULT_I32(result);          \
  });

    IREE_VM_ISA_VERIFY_OP_CORE_CMP_I64(CmpEQI64);
    IREE_VM_ISA_VERIFY_OP_CORE_CMP_I64(CmpNEI64);
    IREE_VM_ISA_VERIFY_OP_CORE_CMP_I64(CmpLTI64S);
    IREE_VM_ISA_VERIFY_OP_CORE_CMP_I64(CmpLTI64U);
    IREE_VM_ISA_VERIFY_OP(CORE, CmpNZI64, {
      IREE_VM_ISA_DECODE_OPERAND_I64(operand);
      IREE_VM_ISA_DECODE_RESULT_I32(result);
    });

    IREE_VM_ISA_VERIFY_OP(CORE, CmpEQRef, {
      IREE_VM_ISA_DECODE_OPERAND_REF(lhs);
      IREE_VM_ISA_DECODE_OPERAND_REF(rhs);
      IREE_VM_ISA_DECODE_RESULT_I32(result);
    });
    IREE_VM_ISA_VERIFY_OP(CORE, CmpNERef, {
      IREE_VM_ISA_DECODE_OPERAND_REF(lhs);
      IREE_VM_ISA_DECODE_OPERAND_REF(rhs);
      IREE_VM_ISA_DECODE_RESULT_I32(result);
    });
    IREE_VM_ISA_VERIFY_OP(CORE, CmpNZRef, {
      IREE_VM_ISA_DECODE_OPERAND_REF(operand);
      IREE_VM_ISA_DECODE_RESULT_I32(result);
    });

    //===------------------------------------------------------------------===//
    // Control flow
    //===------------------------------------------------------------------===//

    IREE_VM_ISA_VERIFY_OP(CORE, Block, {
      // Define the new block in the block list. It may already be declared from
      // a prior branch.
      iree_vm_bytecode_block_t* block = NULL;
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_block_list_insert(
          &verify_state->block_list, pc - 1, &block));
      block->defined = 1;
      verify_state->in_block = 1;
    });

    IREE_VM_ISA_VERIFY_OP(CORE, Branch, {
      IREE_VM_ISA_VERIFY_BRANCH_TARGET(dest_pc);
      IREE_VM_ISA_VERIFY_BRANCH_OPERANDS(operands);
      verify_state->in_block = 0;  // terminator
    });

    IREE_VM_ISA_VERIFY_OP(CORE, CondBranch, {
      IREE_VM_ISA_DECODE_OPERAND_I32(condition);
      IREE_VM_ISA_VERIFY_BRANCH_TARGET(true_dest_pc);
      IREE_VM_ISA_VERIFY_BRANCH_OPERANDS(true_operands);
      IREE_VM_ISA_VERIFY_BRANCH_TARGET(false_dest_pc);
      IREE_VM_ISA_VERIFY_BRANCH_OPERANDS(false_operands);
      verify_state->in_block = 0;  // terminator
    });

    IREE_VM_ISA_VERIFY_OP(CORE, BranchTable, {
      IREE_VM_ISA_DECODE_OPERAND_I32(index);
      IREE_VM_ISA_VERIFY_BRANCH_TARGET(default_dest_pc);
      IREE_VM_ISA_VERIFY_BRANCH_OPERANDS(default_operands);
      IREE_VM_ISA_DECODE_CONST_I16(table_size);
      for (uint16_t i = 0; i < table_size; ++i) {
        IREE_VM_ISA_VERIFY_BRANCH_TARGET(case_dest_pc);
        IREE_VM_ISA_VERIFY_BRANCH_OPERANDS(case_operands);
      }
      verify_state->in_block = 0;  // terminator
    });

    IREE_VM_ISA_VERIFY_OP(CORE, Call, {
      IREE_VM_ISA_DECODE_FUNC_ATTR(callee_ordinal);
      IREE_VM_ISA_VERIFY_VARIADIC_OPERANDS_ANY(operands);
      IREE_VM_ISA_VERIFY_VARIADIC_RESULTS_ANY(results);
      if (iree_vm_isa_function_ordinal_is_import(callee_ordinal)) {
        callee_ordinal = iree_vm_isa_function_ordinal_as_import(callee_ordinal);
        IREE_VM_ISA_VERIFY_IMPORT_ORDINAL(callee_ordinal);
        iree_vm_ImportFunctionDef_table_t import_def =
            iree_vm_ImportFunctionDef_vec_at(verify_state->imported_functions,
                                             callee_ordinal);
        IREE_RETURN_IF_ERROR(
            iree_vm_bytecode_function_verify_call(
                verify_state, iree_vm_ImportFunctionDef_signature(import_def),
                /*segment_sizes=*/NULL, operands, results),
            "call to import '%s'",
            iree_vm_ImportFunctionDef_full_name(import_def));
      } else {
        IREE_VM_ISA_VERIFY_FUNCTION_ORDINAL(callee_ordinal);
        IREE_RETURN_IF_ERROR(
            iree_vm_bytecode_function_verify_call(
                verify_state,
                iree_vm_FunctionSignatureDef_vec_at(
                    verify_state->function_signatures, callee_ordinal),
                /*segment_sizes=*/NULL, operands, results),
            "call to internal function %d", callee_ordinal);
      }
    });

    IREE_VM_ISA_VERIFY_OP(CORE, CallVariadic, {
      IREE_VM_ISA_DECODE_FUNC_ATTR(callee_ordinal);
      IREE_VM_ISA_DECODE_VARIADIC_OPERANDS(segment_sizes);
      IREE_VM_ISA_VERIFY_VARIADIC_OPERANDS_ANY(operands);
      IREE_VM_ISA_VERIFY_VARIADIC_RESULTS_ANY(results);
      if (IREE_UNLIKELY(
              !iree_vm_isa_function_ordinal_is_import(callee_ordinal))) {
        // Variadic calls are currently only supported for import functions.
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "variadic calls only supported for internal callees");
      }
      callee_ordinal = iree_vm_isa_function_ordinal_as_import(callee_ordinal);
      IREE_VM_ISA_VERIFY_IMPORT_ORDINAL(callee_ordinal);
      iree_vm_ImportFunctionDef_table_t import_def =
          iree_vm_ImportFunctionDef_vec_at(verify_state->imported_functions,
                                           callee_ordinal);
      IREE_RETURN_IF_ERROR(
          iree_vm_bytecode_function_verify_call(
              verify_state, iree_vm_ImportFunctionDef_signature(import_def),
              segment_sizes, operands, results),
          "variadic call to import '%s'",
          iree_vm_ImportFunctionDef_full_name(import_def));
    });

    IREE_VM_ISA_VERIFY_OP(CORE, Return, {
      IREE_VM_ISA_VERIFY_VARIADIC_OPERANDS_ANY(operands);
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_function_verify_cconv_registers(
          verify_state, verify_state->cconv_results, /*segment_sizes=*/NULL,
          operands));
      verify_state->in_block = 0;  // terminator
    });

    IREE_VM_ISA_VERIFY_OP(CORE, Fail, {
      IREE_VM_ISA_DECODE_OPERAND_I32(status);
      IREE_VM_ISA_DECODE_STRING_ATTR(message);
      verify_state->in_block = 0;  // terminator
    });

    IREE_VM_ISA_VERIFY_OP(CORE, ImportResolved, {
      IREE_VM_ISA_DECODE_FUNC_ATTR(import_ordinal);
      if (IREE_UNLIKELY(
              !iree_vm_isa_function_ordinal_is_import(import_ordinal))) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "function ordinal %u is not an import ordinal",
                                import_ordinal);
      }
      import_ordinal = iree_vm_isa_function_ordinal_as_import(import_ordinal);
      IREE_VM_ISA_VERIFY_IMPORT_ORDINAL(import_ordinal);
      IREE_VM_ISA_DECODE_RESULT_I32(result);
    });

    //===------------------------------------------------------------------===//
    // Async/fiber ops
    //===------------------------------------------------------------------===//

    IREE_VM_ISA_VERIFY_OP(CORE, Yield, {
      IREE_VM_ISA_VERIFY_BRANCH_TARGET(dest_pc);
      IREE_VM_ISA_VERIFY_BRANCH_OPERANDS(operands);
      verify_state->in_block = 0;  // terminator
    });

    //===------------------------------------------------------------------===//
    // Debugging
    //===------------------------------------------------------------------===//

    IREE_VM_ISA_VERIFY_OP(CORE, Trace, {
      IREE_VM_ISA_DECODE_STRING_ATTR(event_name);
      IREE_VM_ISA_VERIFY_VARIADIC_OPERANDS_ANY(operands);
    });

    IREE_VM_ISA_VERIFY_OP(CORE, Print, {
      IREE_VM_ISA_DECODE_STRING_ATTR(event_name);
      IREE_VM_ISA_VERIFY_VARIADIC_OPERANDS_ANY(operands);
    });

    IREE_VM_ISA_VERIFY_OP(CORE, Break, {
      IREE_VM_ISA_VERIFY_BRANCH_TARGET(dest_pc);
      IREE_VM_ISA_VERIFY_BRANCH_OPERANDS(operands);
      verify_state->in_block = 0;  // terminator
    });

    IREE_VM_ISA_VERIFY_OP(CORE, CondBreak, {
      IREE_VM_ISA_DECODE_OPERAND_I32(condition);
      IREE_VM_ISA_VERIFY_BRANCH_TARGET(dest);
      IREE_VM_ISA_VERIFY_BRANCH_OPERANDS(operands);
      verify_state->in_block = 0;  // terminator
    });

    //===------------------------------------------------------------------===//
    // Extension trampolines
    //===------------------------------------------------------------------===//

#if IREE_VM_EXT_F32_ENABLE
    BEGIN_VERIFY_PREFIX(PrefixExtF32, iree_vm_FeatureBits_EXT_F32)

    //===----------------------------------------------------------------===//
    // ExtF32: Globals
    //===----------------------------------------------------------------===//

    IREE_VM_ISA_VERIFY_OP(EXT_F32, GlobalLoadF32, {
      IREE_VM_ISA_DECODE_GLOBAL_ATTR(byte_offset);
      IREE_VM_ISA_VERIFY_RWDATA_OFFSET(byte_offset, 4);
      IREE_VM_ISA_DECODE_RESULT_F32(value);
    });

    IREE_VM_ISA_VERIFY_OP(EXT_F32, GlobalStoreF32, {
      IREE_VM_ISA_DECODE_GLOBAL_ATTR(byte_offset);
      IREE_VM_ISA_VERIFY_RWDATA_OFFSET(byte_offset, 4);
      IREE_VM_ISA_DECODE_OPERAND_F32(value);
    });

    IREE_VM_ISA_VERIFY_OP(EXT_F32, GlobalLoadIndirectF32, {
      IREE_VM_ISA_DECODE_OPERAND_I32(byte_offset);
      // NOTE: we have to verify the offset at runtime.
      IREE_VM_ISA_DECODE_RESULT_F32(value);
    });

    IREE_VM_ISA_VERIFY_OP(EXT_F32, GlobalStoreIndirectF32, {
      IREE_VM_ISA_DECODE_OPERAND_I32(byte_offset);
      // NOTE: we have to verify the offset at runtime.
      IREE_VM_ISA_DECODE_OPERAND_F32(value);
    });

    //===----------------------------------------------------------------===//
    // ExtF32: Constants
    //===----------------------------------------------------------------===//

    IREE_VM_ISA_VERIFY_OP(EXT_F32, ConstF32, {
      IREE_VM_ISA_DECODE_ATTR_F32(value);
      IREE_VM_ISA_DECODE_RESULT_F32(result);
    });

    IREE_VM_ISA_VERIFY_OP(EXT_F32, ConstF32Zero,
                          { IREE_VM_ISA_DECODE_RESULT_F32(result); });

    //===----------------------------------------------------------------===//
    // ExtF32: Lists
    //===----------------------------------------------------------------===//

    IREE_VM_ISA_VERIFY_OP(EXT_F32, ListGetF32, {
      IREE_VM_ISA_DECODE_OPERAND_REF(list);
      IREE_VM_ISA_DECODE_OPERAND_I32(index);
      IREE_VM_ISA_DECODE_RESULT_F32(result);
    });

    IREE_VM_ISA_VERIFY_OP(EXT_F32, ListSetF32, {
      IREE_VM_ISA_DECODE_OPERAND_REF(list);
      IREE_VM_ISA_DECODE_OPERAND_I32(index);
      IREE_VM_ISA_DECODE_OPERAND_F32(value);
    });

    //===----------------------------------------------------------------===//
    // ExtF32: Conditional assignment
    //===----------------------------------------------------------------===//

    IREE_VM_ISA_VERIFY_OP(EXT_F32, SelectF32, {
      IREE_VM_ISA_DECODE_OPERAND_I32(condition);
      IREE_VM_ISA_DECODE_OPERAND_F32(true_value);
      IREE_VM_ISA_DECODE_OPERAND_F32(false_value);
      IREE_VM_ISA_DECODE_RESULT_F32(result);
    });

    IREE_VM_ISA_VERIFY_OP(EXT_F32, SwitchF32, {
      IREE_VM_ISA_DECODE_OPERAND_I32(index);
      IREE_VM_ISA_DECODE_OPERAND_F32(default_value);
      IREE_VM_ISA_VERIFY_VARIADIC_OPERANDS_F32(values);
      IREE_VM_ISA_DECODE_RESULT_F32(result);
    });

    //===----------------------------------------------------------------===//
    // ExtF32: Native floating-point arithmetic
    //===----------------------------------------------------------------===//

    IREE_VM_ISA_VERIFY_OP_EXT_F32_BINARY_F32(AddF32);
    IREE_VM_ISA_VERIFY_OP_EXT_F32_BINARY_F32(SubF32);
    IREE_VM_ISA_VERIFY_OP_EXT_F32_BINARY_F32(MulF32);
    IREE_VM_ISA_VERIFY_OP_EXT_F32_BINARY_F32(DivF32);
    IREE_VM_ISA_VERIFY_OP_EXT_F32_BINARY_F32(RemF32);
    IREE_VM_ISA_VERIFY_OP_EXT_F32_TERNARY_F32(FMAF32);
    IREE_VM_ISA_VERIFY_OP_EXT_F32_UNARY_F32(AbsF32);
    IREE_VM_ISA_VERIFY_OP_EXT_F32_UNARY_F32(NegF32);
    IREE_VM_ISA_VERIFY_OP_EXT_F32_UNARY_F32(CeilF32);
    IREE_VM_ISA_VERIFY_OP_EXT_F32_UNARY_F32(FloorF32);
    IREE_VM_ISA_VERIFY_OP_EXT_F32_UNARY_F32(RoundF32);
    IREE_VM_ISA_VERIFY_OP_EXT_F32_UNARY_F32(RoundF32Even);
    IREE_VM_ISA_VERIFY_OP_EXT_F32_BINARY_F32(MinF32);
    IREE_VM_ISA_VERIFY_OP_EXT_F32_BINARY_F32(MaxF32);

    IREE_VM_ISA_VERIFY_OP_EXT_F32_UNARY_F32(AtanF32);
    IREE_VM_ISA_VERIFY_OP_EXT_F32_BINARY_F32(Atan2F32);
    IREE_VM_ISA_VERIFY_OP_EXT_F32_UNARY_F32(CosF32);
    IREE_VM_ISA_VERIFY_OP_EXT_F32_UNARY_F32(SinF32);
    IREE_VM_ISA_VERIFY_OP_EXT_F32_UNARY_F32(ExpF32);
    IREE_VM_ISA_VERIFY_OP_EXT_F32_UNARY_F32(Exp2F32);
    IREE_VM_ISA_VERIFY_OP_EXT_F32_UNARY_F32(ExpM1F32);
    IREE_VM_ISA_VERIFY_OP_EXT_F32_UNARY_F32(LogF32);
    IREE_VM_ISA_VERIFY_OP_EXT_F32_UNARY_F32(Log10F32);
    IREE_VM_ISA_VERIFY_OP_EXT_F32_UNARY_F32(Log1pF32);
    IREE_VM_ISA_VERIFY_OP_EXT_F32_UNARY_F32(Log2F32);
    IREE_VM_ISA_VERIFY_OP_EXT_F32_BINARY_F32(PowF32);
    IREE_VM_ISA_VERIFY_OP_EXT_F32_UNARY_F32(RsqrtF32);
    IREE_VM_ISA_VERIFY_OP_EXT_F32_UNARY_F32(SqrtF32);
    IREE_VM_ISA_VERIFY_OP_EXT_F32_UNARY_F32(TanhF32);
    IREE_VM_ISA_VERIFY_OP_EXT_F32_UNARY_F32(ErfF32);

    //===----------------------------------------------------------------===//
    // ExtF32: Casting and type conversion/emulation
    //===----------------------------------------------------------------===//

    IREE_VM_ISA_VERIFY_OP(EXT_F32, CastSI32F32, {
      IREE_VM_ISA_DECODE_OPERAND_I32(operand);
      IREE_VM_ISA_DECODE_RESULT_F32(result);
    });
    IREE_VM_ISA_VERIFY_OP(EXT_F32, CastSI64F32, {
      IREE_VM_ISA_DECODE_OPERAND_I64(operand);
      IREE_VM_ISA_DECODE_RESULT_F32(result);
    });
    IREE_VM_ISA_VERIFY_OP(EXT_F32, CastUI32F32, {
      IREE_VM_ISA_DECODE_OPERAND_I32(operand);
      IREE_VM_ISA_DECODE_RESULT_F32(result);
    });
    IREE_VM_ISA_VERIFY_OP(EXT_F32, CastUI64F32, {
      IREE_VM_ISA_DECODE_OPERAND_I64(operand);
      IREE_VM_ISA_DECODE_RESULT_F32(result);
    });
    IREE_VM_ISA_VERIFY_OP(EXT_F32, CastF32SI32, {
      IREE_VM_ISA_DECODE_OPERAND_F32(operand);
      IREE_VM_ISA_DECODE_RESULT_I32(result);
    });
    IREE_VM_ISA_VERIFY_OP(EXT_F32, CastF32SI64, {
      IREE_VM_ISA_DECODE_OPERAND_F32(operand);
      IREE_VM_ISA_DECODE_RESULT_I64(result);
    });
    IREE_VM_ISA_VERIFY_OP(EXT_F32, CastF32UI32, {
      IREE_VM_ISA_DECODE_OPERAND_F32(operand);
      IREE_VM_ISA_DECODE_RESULT_I32(result);
    });
    IREE_VM_ISA_VERIFY_OP(EXT_F32, CastF32UI64, {
      IREE_VM_ISA_DECODE_OPERAND_F32(operand);
      IREE_VM_ISA_DECODE_RESULT_I64(result);
    });
    IREE_VM_ISA_VERIFY_OP(EXT_F32, BitcastI32F32, {
      IREE_VM_ISA_DECODE_OPERAND_I32(operand);
      IREE_VM_ISA_DECODE_RESULT_F32(result);
    });
    IREE_VM_ISA_VERIFY_OP(EXT_F32, BitcastF32I32, {
      IREE_VM_ISA_DECODE_OPERAND_F32(operand);
      IREE_VM_ISA_DECODE_RESULT_I32(result);
    });

    //===----------------------------------------------------------------===//
    // ExtF32: Comparison ops
    //===----------------------------------------------------------------===//

#define IREE_VM_ISA_VERIFY_OP_EXT_F32_CMP_F32(op_name) \
  IREE_VM_ISA_VERIFY_OP(EXT_F32, op_name, {            \
    IREE_VM_ISA_DECODE_OPERAND_F32(lhs);               \
    IREE_VM_ISA_DECODE_OPERAND_F32(rhs);               \
    IREE_VM_ISA_DECODE_RESULT_I32(result);             \
  });

    IREE_VM_ISA_VERIFY_OP_EXT_F32_CMP_F32(CmpEQF32O);
    IREE_VM_ISA_VERIFY_OP_EXT_F32_CMP_F32(CmpEQF32U);
    IREE_VM_ISA_VERIFY_OP_EXT_F32_CMP_F32(CmpNEF32O);
    IREE_VM_ISA_VERIFY_OP_EXT_F32_CMP_F32(CmpNEF32U);
    IREE_VM_ISA_VERIFY_OP_EXT_F32_CMP_F32(CmpLTF32O);
    IREE_VM_ISA_VERIFY_OP_EXT_F32_CMP_F32(CmpLTF32U);
    IREE_VM_ISA_VERIFY_OP_EXT_F32_CMP_F32(CmpLTEF32O);
    IREE_VM_ISA_VERIFY_OP_EXT_F32_CMP_F32(CmpLTEF32U);
    IREE_VM_ISA_VERIFY_OP(EXT_F32, CmpNaNF32, {
      IREE_VM_ISA_DECODE_OPERAND_F32(operand);
      IREE_VM_ISA_DECODE_RESULT_I32(result);
    });

    //===----------------------------------------------------------------===//
    // ExtF32: Buffers
    //===----------------------------------------------------------------===//

    IREE_VM_ISA_VERIFY_OP(EXT_F32, BufferFillF32, {
      IREE_VM_ISA_DECODE_OPERAND_REF(target_buffer);
      IREE_VM_ISA_DECODE_OPERAND_I64(target_offset);
      IREE_VM_ISA_DECODE_OPERAND_I64(length);
      IREE_VM_ISA_DECODE_OPERAND_F32(value);
    });

    IREE_VM_ISA_VERIFY_OP(EXT_F32, BufferLoadF32, {
      IREE_VM_ISA_DECODE_OPERAND_REF(source_buffer);
      IREE_VM_ISA_DECODE_OPERAND_I64(source_offset);
      IREE_VM_ISA_DECODE_RESULT_F32(result);
    });

    IREE_VM_ISA_VERIFY_OP(EXT_F32, BufferStoreF32, {
      IREE_VM_ISA_DECODE_OPERAND_REF(target_buffer);
      IREE_VM_ISA_DECODE_OPERAND_I64(target_offset);
      IREE_VM_ISA_DECODE_OPERAND_F32(value);
    });

    END_VERIFY_PREFIX(PrefixExtF32, iree_vm_FeatureBits_EXT_F32);
#else
    UNHANDLED_VERIFY_PREFIX(PrefixExtF32, iree_vm_FeatureBits_EXT_F32);
#endif  // IREE_VM_EXT_F32_ENABLE

#if IREE_VM_EXT_F64_ENABLE
    BEGIN_VERIFY_PREFIX(PrefixExtF64, iree_vm_FeatureBits_EXT_F64)

    //===----------------------------------------------------------------===//
    // ExtF64: Globals
    //===----------------------------------------------------------------===//

    IREE_VM_ISA_VERIFY_OP(EXT_F64, GlobalLoadF64, {
      IREE_VM_ISA_DECODE_GLOBAL_ATTR(byte_offset);
      IREE_VM_ISA_VERIFY_RWDATA_OFFSET(byte_offset, 4);
      IREE_VM_ISA_DECODE_RESULT_F64(value);
    });

    IREE_VM_ISA_VERIFY_OP(EXT_F64, GlobalStoreF64, {
      IREE_VM_ISA_DECODE_GLOBAL_ATTR(byte_offset);
      IREE_VM_ISA_VERIFY_RWDATA_OFFSET(byte_offset, 4);
      IREE_VM_ISA_DECODE_OPERAND_F64(value);
    });

    IREE_VM_ISA_VERIFY_OP(EXT_F64, GlobalLoadIndirectF64, {
      IREE_VM_ISA_DECODE_OPERAND_I32(byte_offset);
      // NOTE: we have to verify the offset at runtime.
      IREE_VM_ISA_DECODE_RESULT_F64(value);
    });

    IREE_VM_ISA_VERIFY_OP(EXT_F64, GlobalStoreIndirectF64, {
      IREE_VM_ISA_DECODE_OPERAND_I32(byte_offset);
      // NOTE: we have to verify the offset at runtime.
      IREE_VM_ISA_DECODE_OPERAND_F64(value);
    });

    //===----------------------------------------------------------------===//
    // ExtF64: Constants
    //===----------------------------------------------------------------===//

    IREE_VM_ISA_VERIFY_OP(EXT_F64, ConstF64, {
      IREE_VM_ISA_DECODE_ATTR_F64(value);
      IREE_VM_ISA_DECODE_RESULT_F64(result);
    });

    IREE_VM_ISA_VERIFY_OP(EXT_F64, ConstF64Zero,
                          { IREE_VM_ISA_DECODE_RESULT_F64(result); });

    //===----------------------------------------------------------------===//
    // ExtF64: Lists
    //===----------------------------------------------------------------===//

    IREE_VM_ISA_VERIFY_OP(EXT_F64, ListGetF64, {
      IREE_VM_ISA_DECODE_OPERAND_REF(list);
      IREE_VM_ISA_DECODE_OPERAND_I32(index);
      IREE_VM_ISA_DECODE_RESULT_F64(result);
    });

    IREE_VM_ISA_VERIFY_OP(EXT_F64, ListSetF64, {
      IREE_VM_ISA_DECODE_OPERAND_REF(list);
      IREE_VM_ISA_DECODE_OPERAND_I32(index);
      IREE_VM_ISA_DECODE_OPERAND_F64(value);
    });

    //===----------------------------------------------------------------===//
    // ExtF64: Conditional assignment
    //===----------------------------------------------------------------===//

    IREE_VM_ISA_VERIFY_OP(EXT_F64, SelectF64, {
      IREE_VM_ISA_DECODE_OPERAND_I32(condition);
      IREE_VM_ISA_DECODE_OPERAND_F64(true_value);
      IREE_VM_ISA_DECODE_OPERAND_F64(false_value);
      IREE_VM_ISA_DECODE_RESULT_F64(result);
    });

    IREE_VM_ISA_VERIFY_OP(EXT_F64, SwitchF64, {
      IREE_VM_ISA_DECODE_OPERAND_I32(index);
      IREE_VM_ISA_DECODE_OPERAND_F64(default_value);
      IREE_VM_ISA_VERIFY_VARIADIC_OPERANDS_F64(values);
      IREE_VM_ISA_DECODE_RESULT_F64(result);
    });

    //===----------------------------------------------------------------===//
    // ExtF64: Native floating-point arithmetic
    //===----------------------------------------------------------------===//

    IREE_VM_ISA_VERIFY_OP_EXT_F64_BINARY_F64(AddF64);
    IREE_VM_ISA_VERIFY_OP_EXT_F64_BINARY_F64(SubF64);
    IREE_VM_ISA_VERIFY_OP_EXT_F64_BINARY_F64(MulF64);
    IREE_VM_ISA_VERIFY_OP_EXT_F64_BINARY_F64(DivF64);
    IREE_VM_ISA_VERIFY_OP_EXT_F64_BINARY_F64(RemF64);
    IREE_VM_ISA_VERIFY_OP_EXT_F64_TERNARY_F64(FMAF64);
    IREE_VM_ISA_VERIFY_OP_EXT_F64_UNARY_F64(AbsF64);
    IREE_VM_ISA_VERIFY_OP_EXT_F64_UNARY_F64(NegF64);
    IREE_VM_ISA_VERIFY_OP_EXT_F64_UNARY_F64(CeilF64);
    IREE_VM_ISA_VERIFY_OP_EXT_F64_UNARY_F64(FloorF64);
    IREE_VM_ISA_VERIFY_OP_EXT_F64_UNARY_F64(RoundF64);
    IREE_VM_ISA_VERIFY_OP_EXT_F64_UNARY_F64(RoundF64Even);
    IREE_VM_ISA_VERIFY_OP_EXT_F64_BINARY_F64(MinF64);
    IREE_VM_ISA_VERIFY_OP_EXT_F64_BINARY_F64(MaxF64);

    IREE_VM_ISA_VERIFY_OP_EXT_F64_UNARY_F64(AtanF64);
    IREE_VM_ISA_VERIFY_OP_EXT_F64_BINARY_F64(Atan2F64);
    IREE_VM_ISA_VERIFY_OP_EXT_F64_UNARY_F64(CosF64);
    IREE_VM_ISA_VERIFY_OP_EXT_F64_UNARY_F64(SinF64);
    IREE_VM_ISA_VERIFY_OP_EXT_F64_UNARY_F64(ExpF64);
    IREE_VM_ISA_VERIFY_OP_EXT_F64_UNARY_F64(Exp2F64);
    IREE_VM_ISA_VERIFY_OP_EXT_F64_UNARY_F64(ExpM1F64);
    IREE_VM_ISA_VERIFY_OP_EXT_F64_UNARY_F64(LogF64);
    IREE_VM_ISA_VERIFY_OP_EXT_F64_UNARY_F64(Log10F64);
    IREE_VM_ISA_VERIFY_OP_EXT_F64_UNARY_F64(Log1pF64);
    IREE_VM_ISA_VERIFY_OP_EXT_F64_UNARY_F64(Log2F64);
    IREE_VM_ISA_VERIFY_OP_EXT_F64_BINARY_F64(PowF64);
    IREE_VM_ISA_VERIFY_OP_EXT_F64_UNARY_F64(RsqrtF64);
    IREE_VM_ISA_VERIFY_OP_EXT_F64_UNARY_F64(SqrtF64);
    IREE_VM_ISA_VERIFY_OP_EXT_F64_UNARY_F64(TanhF64);
    IREE_VM_ISA_VERIFY_OP_EXT_F64_UNARY_F64(ErfF64);

    //===----------------------------------------------------------------===//
    // ExtF64: Casting and type conversion/emulation
    //===----------------------------------------------------------------===//

    IREE_VM_ISA_VERIFY_OP(EXT_F64, TruncF64F32, {
      IREE_VM_ISA_DECODE_OPERAND_F64(operand);
      IREE_VM_ISA_DECODE_RESULT_F32(result);
    });
    IREE_VM_ISA_VERIFY_OP(EXT_F64, ExtF32F64, {
      IREE_VM_ISA_DECODE_OPERAND_F32(operand);
      IREE_VM_ISA_DECODE_RESULT_F64(result);
    });
    IREE_VM_ISA_VERIFY_OP(EXT_F64, CastSI32F64, {
      IREE_VM_ISA_DECODE_OPERAND_I32(operand);
      IREE_VM_ISA_DECODE_RESULT_F64(result);
    });
    IREE_VM_ISA_VERIFY_OP(EXT_F64, CastUI32F64, {
      IREE_VM_ISA_DECODE_OPERAND_I32(operand);
      IREE_VM_ISA_DECODE_RESULT_F64(result);
    });
    IREE_VM_ISA_VERIFY_OP(EXT_F64, CastF64SI32, {
      IREE_VM_ISA_DECODE_OPERAND_F64(operand);
      IREE_VM_ISA_DECODE_RESULT_I32(result);
    });
    IREE_VM_ISA_VERIFY_OP(EXT_F64, CastF64UI32, {
      IREE_VM_ISA_DECODE_OPERAND_F64(operand);
      IREE_VM_ISA_DECODE_RESULT_I32(result);
    });
    IREE_VM_ISA_VERIFY_OP(EXT_F64, CastSI64F64, {
      IREE_VM_ISA_DECODE_OPERAND_I64(operand);
      IREE_VM_ISA_DECODE_RESULT_F64(result);
    });
    IREE_VM_ISA_VERIFY_OP(EXT_F64, CastUI64F64, {
      IREE_VM_ISA_DECODE_OPERAND_I64(operand);
      IREE_VM_ISA_DECODE_RESULT_F64(result);
    });
    IREE_VM_ISA_VERIFY_OP(EXT_F64, CastF64SI64, {
      IREE_VM_ISA_DECODE_OPERAND_F64(operand);
      IREE_VM_ISA_DECODE_RESULT_I64(result);
    });
    IREE_VM_ISA_VERIFY_OP(EXT_F64, CastF64UI64, {
      IREE_VM_ISA_DECODE_OPERAND_F64(operand);
      IREE_VM_ISA_DECODE_RESULT_I64(result);
    });
    IREE_VM_ISA_VERIFY_OP(EXT_F64, BitcastI64F64, {
      IREE_VM_ISA_DECODE_OPERAND_I64(operand);
      IREE_VM_ISA_DECODE_RESULT_F64(result);
    });
    IREE_VM_ISA_VERIFY_OP(EXT_F64, BitcastF64I64, {
      IREE_VM_ISA_DECODE_OPERAND_F64(operand);
      IREE_VM_ISA_DECODE_RESULT_I64(result);
    });

    //===----------------------------------------------------------------===//
    // ExtF64: Comparison ops
    //===----------------------------------------------------------------===//

#define IREE_VM_ISA_VERIFY_OP_EXT_F64_CMP_F64(op_name) \
  IREE_VM_ISA_VERIFY_OP(EXT_F64, op_name, {            \
    IREE_VM_ISA_DECODE_OPERAND_F64(lhs);               \
    IREE_VM_ISA_DECODE_OPERAND_F64(rhs);               \
    IREE_VM_ISA_DECODE_RESULT_I32(result);             \
  });

    IREE_VM_ISA_VERIFY_OP_EXT_F64_CMP_F64(CmpEQF64O);
    IREE_VM_ISA_VERIFY_OP_EXT_F64_CMP_F64(CmpEQF64U);
    IREE_VM_ISA_VERIFY_OP_EXT_F64_CMP_F64(CmpNEF64O);
    IREE_VM_ISA_VERIFY_OP_EXT_F64_CMP_F64(CmpNEF64U);
    IREE_VM_ISA_VERIFY_OP_EXT_F64_CMP_F64(CmpLTF64O);
    IREE_VM_ISA_VERIFY_OP_EXT_F64_CMP_F64(CmpLTF64U);
    IREE_VM_ISA_VERIFY_OP_EXT_F64_CMP_F64(CmpLTEF64O);
    IREE_VM_ISA_VERIFY_OP_EXT_F64_CMP_F64(CmpLTEF64U);
    IREE_VM_ISA_VERIFY_OP(EXT_F64, CmpNaNF64, {
      IREE_VM_ISA_DECODE_OPERAND_F64(operand);
      IREE_VM_ISA_DECODE_RESULT_I32(result);
    });

    //===----------------------------------------------------------------===//
    // ExtF64: Buffers
    //===----------------------------------------------------------------===//

    IREE_VM_ISA_VERIFY_OP(EXT_F64, BufferFillF64, {
      IREE_VM_ISA_DECODE_OPERAND_REF(target_buffer);
      IREE_VM_ISA_DECODE_OPERAND_I64(target_offset);
      IREE_VM_ISA_DECODE_OPERAND_I64(length);
      IREE_VM_ISA_DECODE_OPERAND_F64(value);
    });

    IREE_VM_ISA_VERIFY_OP(EXT_F64, BufferLoadF64, {
      IREE_VM_ISA_DECODE_OPERAND_REF(source_buffer);
      IREE_VM_ISA_DECODE_OPERAND_I64(source_offset);
      IREE_VM_ISA_DECODE_RESULT_F64(result);
    });

    IREE_VM_ISA_VERIFY_OP(EXT_F64, BufferStoreF64, {
      IREE_VM_ISA_DECODE_OPERAND_REF(target_buffer);
      IREE_VM_ISA_DECODE_OPERAND_I64(target_offset);
      IREE_VM_ISA_DECODE_OPERAND_F64(value);
    });

    END_VERIFY_PREFIX(PrefixExtF64, iree_vm_FeatureBits_EXT_F64);
#else
    UNHANDLED_VERIFY_PREFIX(PrefixExtF64, iree_vm_FeatureBits_EXT_F64);
#endif  // IREE_VM_EXT_F64_ENABLE

    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unrecognized opcode %u", bytecode_data[pc - 1]);
  }

  *out_next_pc = pc;
  return iree_ok_status();
}
