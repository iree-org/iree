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
    if (function_descriptor->i32_register_count > IREE_I32_REGISTER_COUNT ||
        function_descriptor->ref_register_count > IREE_REF_REGISTER_COUNT) {
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
    iree_const_byte_span_t bytecode_data, uint32_t pc, uint32_t max_pc,
    uint32_t* out_next_pc);

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
      VMMAX(1, function_descriptor->i32_register_count));
  verify_state.ref_register_count = iree_math_round_up_to_pow2_u32(
      VMMAX(1, function_descriptor->ref_register_count));
  if (IREE_UNLIKELY(verify_state.i32_register_count > IREE_I32_REGISTER_MASK) ||
      IREE_UNLIKELY(verify_state.ref_register_count > IREE_REF_REGISTER_MASK)) {
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
  const uint32_t max_pc = (uint32_t)function_descriptor->bytecode_length;

  // Reserve the block list. As we walk the bytecode we'll declare/define blocks
  // and then afterward verify all were found.
  IREE_ASSERT(function_descriptor->block_count > 0);
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_block_list_initialize(
      function_descriptor->block_count, scratch_allocator,
      &verify_state.block_list));

  // Perform bytecode verification by performing a single-pass walk of all
  // function bytecode.
  iree_status_t status = iree_ok_status();
  for (uint32_t pc = 0; pc < bytecode_data.data_length - 1;) {
    uint32_t start_pc = pc;
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
                                        function_name.data, start_pc);
      } else {
        status = iree_status_annotate_f(status, "at %.*s@%u+%08X",
                                        (int)module_name.size, module_name.data,
                                        function_ordinal, start_pc);
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
#define IREE_VM_VERIFY_REG_ORDINAL(name)                            \
  IREE_VM_VERIFY_PC_RANGE(pc + IREE_REGISTER_ORDINAL_SIZE, max_pc); \
  const uint32_t name = OP_I16(0);
#define IREE_VM_VERIFY_REG_ORDINAL_X32(ordinal, category)                      \
  if (IREE_UNLIKELY(((ordinal)&IREE_REF_REGISTER_TYPE_BIT) != 0)) {            \
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
  if (IREE_UNLIKELY(((ordinal)&IREE_REF_REGISTER_TYPE_BIT) != 0)) {            \
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
  if (IREE_UNLIKELY(((ordinal)&IREE_REF_REGISTER_TYPE_BIT) == 0)) {          \
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,                    \
                            "ref register required but non-ref %u provided", \
                            (ordinal));                                      \
  } else if (IREE_UNLIKELY(((ordinal)&IREE_REF_REGISTER_MASK) >=             \
                           verify_state->ref_register_count)) {              \
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,                        \
                            "ref register ordinal %u out of range %u",       \
                            (ordinal), verify_state->ref_register_count);    \
  }
#define IREE_VM_VERIFY_REG_ANY(ordinal)                             \
  if (IREE_UNLIKELY(((ordinal)&IREE_REF_REGISTER_TYPE_BIT) == 0)) { \
  } else {                                                          \
  }

#define VM_VerifyConstI8(name)             \
  IREE_VM_VERIFY_PC_RANGE(pc + 1, max_pc); \
  uint8_t name = OP_I8(0);                 \
  (void)(name);                            \
  ++pc;
#define VM_VerifyConstI16(name)            \
  IREE_VM_VERIFY_PC_RANGE(pc + 2, max_pc); \
  uint32_t name = OP_I16(0);               \
  (void)(name);                            \
  pc += 2;
#define VM_VerifyConstI32(name)            \
  IREE_VM_VERIFY_PC_RANGE(pc + 4, max_pc); \
  uint32_t name = OP_I32(0);               \
  (void)(name);                            \
  pc += 4;
#define VM_VerifyConstI64(name)            \
  IREE_VM_VERIFY_PC_RANGE(pc + 8, max_pc); \
  uint64_t name = OP_I64(0);               \
  (void)(name);                            \
  pc += 8;
#define VM_VerifyConstF32(name)            \
  IREE_VM_VERIFY_PC_RANGE(pc + 4, max_pc); \
  float name = OP_F32(0);                  \
  (void)(name);                            \
  pc += 4;
#define VM_VerifyConstF64(name)            \
  IREE_VM_VERIFY_PC_RANGE(pc + 8, max_pc); \
  double name = OP_F64(0);                 \
  (void)(name);                            \
  pc += 8;

#define VM_VerifyFuncAttr(name) VM_VerifyConstI32(name)
#define VM_IsImportOrdinal(name) (((name)&0x80000000u) != 0)
#define VM_UnmaskImportOrdinal(name) name &= ~0x80000000u
#define VM_VerifyImportOrdinal(name)                                          \
  if (IREE_UNLIKELY((name) >= iree_vm_ImportFunctionDef_vec_len(              \
                                  verify_state->imported_functions))) {       \
    return iree_make_status(                                                  \
        IREE_STATUS_INVALID_ARGUMENT, "import ordinal %u out of range %zu",   \
        name,                                                                 \
        iree_vm_ImportFunctionDef_vec_len(verify_state->imported_functions)); \
  }
#define VM_VerifyFunctionOrdinal(name)                                      \
  if (IREE_UNLIKELY((name)) >= iree_vm_FunctionDescriptor_vec_len(          \
                                   verify_state->function_descriptors)) {   \
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,                   \
                            "function ordinal %u out of range %zu", (name), \
                            iree_vm_FunctionDescriptor_vec_len(             \
                                verify_state->function_descriptors));       \
  }
#define VM_VerifyGlobalAttr(name) VM_VerifyConstI32(name)
#define VM_VerifyRwdataOffset(name, access_length)                          \
  if (IREE_UNLIKELY(((name) + (access_length)) >                            \
                    verify_state->rwdata_storage_size)) {                   \
    return iree_make_status(                                                \
        IREE_STATUS_OUT_OF_RANGE,                                           \
        "global byte_offset out of range: %d (rwdata=%" PRIhsz ")", (name), \
        verify_state->rwdata_storage_size);                                 \
  }
#define VM_VerifyGlobalRefOrdinal(name)                                    \
  if (IREE_UNLIKELY((name) >= verify_state->global_ref_count)) {           \
    return iree_make_status(                                               \
        IREE_STATUS_OUT_OF_RANGE,                                          \
        "global ref ordinal out of range: %d (table=%" PRIhsz ")", (name), \
        verify_state->global_ref_count);                                   \
  }
#define VM_VerifyRodataAttr(name) VM_VerifyConstI32(name)
#define VM_VerifyRodataOrdinal(name)                                       \
  if (IREE_UNLIKELY((name) >= verify_state->rodata_ref_count)) {           \
    return iree_make_status(                                               \
        IREE_STATUS_OUT_OF_RANGE,                                          \
        "rodata ref ordinal out of range: %d (table=%" PRIhsz ")", (name), \
        verify_state->rodata_ref_count);                                   \
  }
#define VM_VerifyType(name)                                                    \
  IREE_VM_VERIFY_PC_RANGE(pc + 4, max_pc);                                     \
  uint32_t name##_id = OP_I32(0);                                              \
  if (IREE_UNLIKELY(name##_id >= module->type_count)) {                        \
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,                          \
                            "type id ordinal out of range: %d (table=%" PRIhsz \
                            ")",                                               \
                            name##_id, module->type_count);                    \
  }                                                                            \
  const iree_vm_type_def_t* name = &module->type_table[name##_id];             \
  (void)(name);                                                                \
  pc += 4;
#define VM_VerifyTypeOf(name) VM_VerifyType(name)
#define VM_VerifyIntAttr32(name) VM_VerifyConstI32(name)
#define VM_VerifyIntAttr64(name) VM_VerifyConstI64(name)
#define VM_VerifyFloatAttr32(name) VM_VerifyConstF32(name)
#define VM_VerifyFloatAttr64(name) VM_VerifyConstF64(name)
#define VM_VerifyStrAttr(name, out_str)                      \
  IREE_VM_VERIFY_PC_RANGE(pc + 2, max_pc);                   \
  (out_str)->size = (iree_host_size_t)OP_I16(0);             \
  IREE_VM_VERIFY_PC_RANGE(pc + 2 + (out_str)->size, max_pc); \
  (out_str)->data = (const char*)&bytecode_data[pc + 2];     \
  pc += 2 + (out_str)->size;

#define VM_VerifyBranchTarget(name)                        \
  VM_VerifyConstI32(name##_pc);                            \
  iree_vm_bytecode_block_t* name = NULL;                   \
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_block_list_insert( \
      &verify_state->block_list, name##_pc, &name));
#define VM_VerifyBranchOperands(name)                                         \
  VM_AlignPC(pc, IREE_REGISTER_ORDINAL_SIZE);                                 \
  IREE_VM_VERIFY_PC_RANGE(pc + IREE_REGISTER_ORDINAL_SIZE, max_pc);           \
  const iree_vm_register_remap_list_t* name =                                 \
      (const iree_vm_register_remap_list_t*)&bytecode_data[pc];               \
  pc += IREE_REGISTER_ORDINAL_SIZE;                                           \
  IREE_VM_VERIFY_PC_RANGE(pc + (name)->size * 2 * IREE_REGISTER_ORDINAL_SIZE, \
                          max_pc);                                            \
  pc += (name)->size * 2 * IREE_REGISTER_ORDINAL_SIZE;                        \
  for (uint16_t i = 0; i < name->size; ++i) {                                 \
    IREE_VM_VERIFY_REG_ANY(name->pairs[i].src_reg);                           \
    IREE_VM_VERIFY_REG_ANY(name->pairs[i].dst_reg);                           \
  }

#define VM_VerifyOperandRegI32(name)          \
  IREE_VM_VERIFY_REG_ORDINAL(name##_ordinal); \
  IREE_VM_VERIFY_REG_I32(name##_ordinal);     \
  pc += IREE_REGISTER_ORDINAL_SIZE;
#define VM_VerifyOperandRegI64(name)          \
  IREE_VM_VERIFY_REG_ORDINAL(name##_ordinal); \
  IREE_VM_VERIFY_REG_I64(name##_ordinal);     \
  pc += IREE_REGISTER_ORDINAL_SIZE;
#define VM_VerifyOperandRegI64HostSize(name) VM_VerifyOperandRegI64(name)
#define VM_VerifyOperandRegF32(name)          \
  IREE_VM_VERIFY_REG_ORDINAL(name##_ordinal); \
  IREE_VM_VERIFY_REG_F32(name##_ordinal);     \
  pc += IREE_REGISTER_ORDINAL_SIZE;
#define VM_VerifyOperandRegF64(name)          \
  IREE_VM_VERIFY_REG_ORDINAL(name##_ordinal); \
  IREE_VM_VERIFY_REG_F32(name##_ordinal);     \
  pc += IREE_REGISTER_ORDINAL_SIZE;
#define VM_VerifyOperandRegRef(name)          \
  IREE_VM_VERIFY_REG_ORDINAL(name##_ordinal); \
  IREE_VM_VERIFY_REG_REF(name##_ordinal);     \
  pc += IREE_REGISTER_ORDINAL_SIZE;
#define VM_VerifyVariadicOperands(name)                                   \
  VM_AlignPC(pc, IREE_REGISTER_ORDINAL_SIZE);                             \
  IREE_VM_VERIFY_PC_RANGE(pc + IREE_REGISTER_ORDINAL_SIZE, max_pc);       \
  const iree_vm_register_list_t* name =                                   \
      (const iree_vm_register_list_t*)&bytecode_data[pc];                 \
  pc += IREE_REGISTER_ORDINAL_SIZE;                                       \
  IREE_VM_VERIFY_PC_RANGE(pc + (name)->size * IREE_REGISTER_ORDINAL_SIZE, \
                          max_pc);                                        \
  pc += (name)->size * IREE_REGISTER_ORDINAL_SIZE;
#define VM_VerifyVariadicOperandsI32(name)            \
  VM_VerifyVariadicOperands(name);                    \
  for (uint16_t __i = 0; __i < (name)->size; ++__i) { \
    IREE_VM_VERIFY_REG_I32((name)->registers[__i]);   \
  }
#define VM_VerifyVariadicOperandsI64(name)            \
  VM_VerifyVariadicOperands(name);                    \
  for (uint16_t __i = 0; __i < (name)->size; ++__i) { \
    IREE_VM_VERIFY_REG_I64((name)->registers[__i]);   \
  }
#define VM_VerifyVariadicOperandsF32(name)            \
  VM_VerifyVariadicOperands(name);                    \
  for (uint16_t __i = 0; __i < (name)->size; ++__i) { \
    IREE_VM_VERIFY_REG_F32((name)->registers[__i]);   \
  }
#define VM_VerifyVariadicOperandsF64(name)            \
  VM_VerifyVariadicOperands(name);                    \
  for (uint16_t __i = 0; __i < (name)->size; ++__i) { \
    IREE_VM_VERIFY_REG_F64((name)->registers[__i]);   \
  }
#define VM_VerifyVariadicOperandsRef(name, type_def)  \
  VM_VerifyVariadicOperands(name);                    \
  for (uint16_t __i = 0; __i < (name)->size; ++__i) { \
    IREE_VM_VERIFY_REG_REF((name)->registers[__i]);   \
  }
#define VM_VerifyVariadicOperandsAny(name)            \
  VM_VerifyVariadicOperands(name);                    \
  for (uint16_t __i = 0; __i < (name)->size; ++__i) { \
    IREE_VM_VERIFY_REG_ANY((name)->registers[__i]);   \
  }
#define VM_VerifyResultRegI32(name)           \
  IREE_VM_VERIFY_REG_ORDINAL(name##_ordinal); \
  IREE_VM_VERIFY_REG_I32(name##_ordinal);     \
  pc += IREE_REGISTER_ORDINAL_SIZE;
#define VM_VerifyResultRegI64(name)           \
  IREE_VM_VERIFY_REG_ORDINAL(name##_ordinal); \
  IREE_VM_VERIFY_REG_I64(name##_ordinal);     \
  pc += IREE_REGISTER_ORDINAL_SIZE;
#define VM_VerifyResultRegF32(name)           \
  IREE_VM_VERIFY_REG_ORDINAL(name##_ordinal); \
  IREE_VM_VERIFY_REG_F32(name##_ordinal);     \
  pc += IREE_REGISTER_ORDINAL_SIZE;
#define VM_VerifyResultRegF64(name)           \
  IREE_VM_VERIFY_REG_ORDINAL(name##_ordinal); \
  IREE_VM_VERIFY_REG_F64(name##_ordinal);     \
  pc += IREE_REGISTER_ORDINAL_SIZE;
#define VM_VerifyResultRegRef(name)           \
  IREE_VM_VERIFY_REG_ORDINAL(name##_ordinal); \
  IREE_VM_VERIFY_REG_REF(name##_ordinal);     \
  pc += IREE_REGISTER_ORDINAL_SIZE;
#define VM_VerifyVariadicResultsAny(name) VM_VerifyVariadicOperandsAny(name)

#define VERIFY_OP_CORE_UNARY_I32(op_name) \
  VERIFY_OP(CORE, op_name, {              \
    VM_VerifyOperandRegI32(operand);      \
    VM_VerifyResultRegI32(result);        \
  });

#define VERIFY_OP_CORE_UNARY_I64(op_name) \
  VERIFY_OP(CORE, op_name, {              \
    VM_VerifyOperandRegI64(operand);      \
    VM_VerifyResultRegI64(result);        \
  });

#define VERIFY_OP_CORE_BINARY_I32(op_name) \
  VERIFY_OP(CORE, op_name, {               \
    VM_VerifyOperandRegI32(lhs);           \
    VM_VerifyOperandRegI32(rhs);           \
    VM_VerifyResultRegI32(result);         \
  });

#define VERIFY_OP_CORE_BINARY_I64(op_name) \
  VERIFY_OP(CORE, op_name, {               \
    VM_VerifyOperandRegI64(lhs);           \
    VM_VerifyOperandRegI64(rhs);           \
    VM_VerifyResultRegI64(result);         \
  });

#define VERIFY_OP_CORE_TERNARY_I32(op_name) \
  VERIFY_OP(CORE, op_name, {                \
    VM_VerifyOperandRegI32(a);              \
    VM_VerifyOperandRegI32(b);              \
    VM_VerifyOperandRegI32(c);              \
    VM_VerifyResultRegI32(result);          \
  });

#define VERIFY_OP_CORE_TERNARY_I64(op_name) \
  VERIFY_OP(CORE, op_name, {                \
    VM_VerifyOperandRegI64(a);              \
    VM_VerifyOperandRegI64(b);              \
    VM_VerifyOperandRegI64(c);              \
    VM_VerifyResultRegI64(result);          \
  });

#define VERIFY_OP_EXT_F32_UNARY_F32(op_name) \
  VERIFY_OP(EXT_F32, op_name, {              \
    VM_VerifyOperandRegF32(operand);         \
    VM_VerifyResultRegF32(result);           \
  });

#define VERIFY_OP_EXT_F32_BINARY_F32(op_name) \
  VERIFY_OP(EXT_F32, op_name, {               \
    VM_VerifyOperandRegF32(lhs);              \
    VM_VerifyOperandRegF32(rhs);              \
    VM_VerifyResultRegF32(result);            \
  });

#define VERIFY_OP_EXT_F32_TERNARY_F32(op_name) \
  VERIFY_OP(EXT_F32, op_name, {                \
    VM_VerifyOperandRegF32(a);                 \
    VM_VerifyOperandRegF32(b);                 \
    VM_VerifyOperandRegF32(c);                 \
    VM_VerifyResultRegF32(result);             \
  });

#define VERIFY_OP_EXT_F64_UNARY_F64(op_name) \
  VERIFY_OP(EXT_F64, op_name, {              \
    VM_VerifyOperandRegF64(operand);         \
    VM_VerifyResultRegF64(result);           \
  });

#define VERIFY_OP_EXT_F64_BINARY_F64(op_name) \
  VERIFY_OP(EXT_F64, op_name, {               \
    VM_VerifyOperandRegF64(lhs);              \
    VM_VerifyOperandRegF64(rhs);              \
    VM_VerifyResultRegF64(result);            \
  });

#define VERIFY_OP_EXT_F64_TERNARY_F64(op_name) \
  VERIFY_OP(EXT_F64, op_name, {                \
    VM_VerifyOperandRegF64(a);                 \
    VM_VerifyOperandRegF64(b);                 \
    VM_VerifyOperandRegF64(c);                 \
    VM_VerifyResultRegF64(result);             \
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

#define VERIFY_OP(ext, op_name, body)  \
  case IREE_VM_OP_##ext##_##op_name: { \
    body;                              \
  } break;

#define BEGIN_VERIFY_PREFIX(op_name, ext)    \
  case IREE_VM_OP_CORE_##op_name: {          \
    IREE_VM_VERIFY_PC_RANGE(pc + 1, max_pc); \
    IREE_VM_VERIFY_REQUIREMENT(ext);         \
    switch (bytecode_data[pc++]) {
#define END_VERIFY_PREFIX()                               \
  default:                                                \
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, \
                            "unhandled ext opcode");      \
    }                                                     \
    break;                                                \
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
    iree_const_byte_span_t function_bytecode, uint32_t start_pc,
    uint32_t max_pc, uint32_t* out_next_pc) {
  *out_next_pc = 0;
  uint32_t pc = start_pc;
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
                              "op at pc %08X is not in a block", pc);
    }
  } else {
    // If in a block then the next opcode must not be a block.
    if (bytecode_data[pc] == IREE_VM_OP_CORE_Block) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "op at pc %08X is a block while still in a block",
                              pc);
    }
  }

  // Get primary opcode. All ops have at least 1 byte.
  switch (bytecode_data[pc++]) {
    //===------------------------------------------------------------------===//
    // Globals
    //===------------------------------------------------------------------===//

    VERIFY_OP(CORE, GlobalLoadI32, {
      VM_VerifyGlobalAttr(byte_offset);
      VM_VerifyRwdataOffset(byte_offset, 4);
      VM_VerifyResultRegI32(value);
    });

    VERIFY_OP(CORE, GlobalStoreI32, {
      VM_VerifyGlobalAttr(byte_offset);
      VM_VerifyRwdataOffset(byte_offset, 4);
      VM_VerifyOperandRegI32(value);
    });

    VERIFY_OP(CORE, GlobalLoadIndirectI32, {
      VM_VerifyOperandRegI32(byte_offset);
      // NOTE: we have to verify the offset at runtime.
      VM_VerifyResultRegI32(value);
    });

    VERIFY_OP(CORE, GlobalStoreIndirectI32, {
      VM_VerifyOperandRegI32(byte_offset);
      // NOTE: we have to verify the offset at runtime.
      VM_VerifyOperandRegI32(value);
    });

    VERIFY_OP(CORE, GlobalLoadI64, {
      VM_VerifyGlobalAttr(byte_offset);
      VM_VerifyRwdataOffset(byte_offset, 8);
      VM_VerifyResultRegI64(value);
    });

    VERIFY_OP(CORE, GlobalStoreI64, {
      VM_VerifyGlobalAttr(byte_offset);
      VM_VerifyRwdataOffset(byte_offset, 8);
      VM_VerifyOperandRegI64(value);
    });

    VERIFY_OP(CORE, GlobalLoadIndirectI64, {
      VM_VerifyOperandRegI32(byte_offset);
      // NOTE: we have to verify the offset at runtime.
      VM_VerifyResultRegI64(value);
    });

    VERIFY_OP(CORE, GlobalStoreIndirectI64, {
      VM_VerifyOperandRegI32(byte_offset);
      // NOTE: we have to verify the offset at runtime.
      VM_VerifyOperandRegI64(value);
    });

    VERIFY_OP(CORE, GlobalLoadRef, {
      VM_VerifyGlobalAttr(global);
      VM_VerifyGlobalRefOrdinal(global);
      VM_VerifyTypeOf(type_def);
      VM_VerifyResultRegRef(value);
    });

    VERIFY_OP(CORE, GlobalStoreRef, {
      VM_VerifyGlobalAttr(global);
      VM_VerifyGlobalRefOrdinal(global);
      VM_VerifyTypeOf(type_def);
      VM_VerifyOperandRegRef(value);
    });

    VERIFY_OP(CORE, GlobalLoadIndirectRef, {
      VM_VerifyOperandRegI32(global);
      // NOTE: we have to verify the ordinal at runtime.
      VM_VerifyTypeOf(type_def);
      VM_VerifyResultRegRef(value);
    });

    VERIFY_OP(CORE, GlobalStoreIndirectRef, {
      VM_VerifyOperandRegI32(global);
      // NOTE: we have to verify the ordinal at runtime.
      VM_VerifyTypeOf(type_def);
      VM_VerifyOperandRegRef(value);
    });

    //===------------------------------------------------------------------===//
    // Constants
    //===------------------------------------------------------------------===//

    VERIFY_OP(CORE, ConstI32, {
      VM_VerifyIntAttr32(value);
      VM_VerifyResultRegI32(result);
    });

    VERIFY_OP(CORE, ConstI32Zero, { VM_VerifyResultRegI32(result); });

    VERIFY_OP(CORE, ConstI64, {
      VM_VerifyIntAttr64(value);
      VM_VerifyResultRegI64(result);
    });

    VERIFY_OP(CORE, ConstI64Zero, { VM_VerifyResultRegI64(result); });

    VERIFY_OP(CORE, ConstRefZero, { VM_VerifyResultRegRef(result); });

    VERIFY_OP(CORE, ConstRefRodata, {
      VM_VerifyRodataAttr(rodata);
      VM_VerifyRodataOrdinal(rodata);
      VM_VerifyResultRegRef(value);
    });

    //===------------------------------------------------------------------===//
    // Buffers
    //===------------------------------------------------------------------===//

    VERIFY_OP(CORE, BufferAlloc, {
      VM_VerifyOperandRegI64HostSize(length);
      VM_VerifyOperandRegI32(alignment);
      VM_VerifyResultRegRef(result);
    });

    VERIFY_OP(CORE, BufferClone, {
      VM_VerifyOperandRegRef(source);
      VM_VerifyOperandRegI64HostSize(offset);
      VM_VerifyOperandRegI64HostSize(length);
      VM_VerifyOperandRegI32(alignment);
      VM_VerifyResultRegRef(result);
    });

    VERIFY_OP(CORE, BufferLength, {
      VM_VerifyOperandRegRef(buffer);
      VM_VerifyResultRegI64(result);
    });

    VERIFY_OP(CORE, BufferCopy, {
      VM_VerifyOperandRegRef(source_buffer);
      VM_VerifyOperandRegI64HostSize(source_offset);
      VM_VerifyOperandRegRef(target_buffer);
      VM_VerifyOperandRegI64HostSize(target_offset);
      VM_VerifyOperandRegI64HostSize(length);
    });

    VERIFY_OP(CORE, BufferCompare, {
      VM_VerifyOperandRegRef(lhs_buffer);
      VM_VerifyOperandRegI64HostSize(lhs_offset);
      VM_VerifyOperandRegRef(rhs_buffer);
      VM_VerifyOperandRegI64HostSize(rhs_offset);
      VM_VerifyOperandRegI64HostSize(length);
      VM_VerifyResultRegI32(result);
    });

    VERIFY_OP(CORE, BufferFillI8, {
      VM_VerifyOperandRegRef(target_buffer);
      VM_VerifyOperandRegI64HostSize(target_offset);
      VM_VerifyOperandRegI64HostSize(length);
      VM_VerifyOperandRegI32(value);
    });
    VERIFY_OP(CORE, BufferFillI16, {
      VM_VerifyOperandRegRef(target_buffer);
      VM_VerifyOperandRegI64HostSize(target_offset);
      VM_VerifyOperandRegI64HostSize(length);
      VM_VerifyOperandRegI32(value);
    });
    VERIFY_OP(CORE, BufferFillI32, {
      VM_VerifyOperandRegRef(target_buffer);
      VM_VerifyOperandRegI64HostSize(target_offset);
      VM_VerifyOperandRegI64HostSize(length);
      VM_VerifyOperandRegI32(value);
    });
    VERIFY_OP(CORE, BufferFillI64, {
      VM_VerifyOperandRegRef(target_buffer);
      VM_VerifyOperandRegI64HostSize(target_offset);
      VM_VerifyOperandRegI64HostSize(length);
      VM_VerifyOperandRegI64(value);
    });

    VERIFY_OP(CORE, BufferLoadI8U, {
      VM_VerifyOperandRegRef(source_buffer);
      VM_VerifyOperandRegI64HostSize(source_offset);
      VM_VerifyResultRegI32(result);
    });
    VERIFY_OP(CORE, BufferLoadI8S, {
      VM_VerifyOperandRegRef(source_buffer);
      VM_VerifyOperandRegI64HostSize(source_offset);
      VM_VerifyResultRegI32(result);
    });
    VERIFY_OP(CORE, BufferLoadI16U, {
      VM_VerifyOperandRegRef(source_buffer);
      VM_VerifyOperandRegI64HostSize(source_offset);
      VM_VerifyResultRegI32(result);
    });
    VERIFY_OP(CORE, BufferLoadI16S, {
      VM_VerifyOperandRegRef(source_buffer);
      VM_VerifyOperandRegI64HostSize(source_offset);
      VM_VerifyResultRegI32(result);
    });
    VERIFY_OP(CORE, BufferLoadI32, {
      VM_VerifyOperandRegRef(source_buffer);
      VM_VerifyOperandRegI64HostSize(source_offset);
      VM_VerifyResultRegI32(result);
    });
    VERIFY_OP(CORE, BufferLoadI64, {
      VM_VerifyOperandRegRef(source_buffer);
      VM_VerifyOperandRegI64HostSize(source_offset);
      VM_VerifyResultRegI64(result);
    });

    VERIFY_OP(CORE, BufferStoreI8, {
      VM_VerifyOperandRegRef(target_buffer);
      VM_VerifyOperandRegI64HostSize(target_offset);
      VM_VerifyOperandRegI32(value);
    });
    VERIFY_OP(CORE, BufferStoreI16, {
      VM_VerifyOperandRegRef(target_buffer);
      VM_VerifyOperandRegI64HostSize(target_offset);
      VM_VerifyOperandRegI32(value);
    });
    VERIFY_OP(CORE, BufferStoreI32, {
      VM_VerifyOperandRegRef(target_buffer);
      VM_VerifyOperandRegI64HostSize(target_offset);
      VM_VerifyOperandRegI32(value);
    });
    VERIFY_OP(CORE, BufferStoreI64, {
      VM_VerifyOperandRegRef(target_buffer);
      VM_VerifyOperandRegI64HostSize(target_offset);
      VM_VerifyOperandRegI64(value);
    });
    VERIFY_OP(CORE, BufferHash, {
      VM_VerifyOperandRegRef(source_buffer);
      VM_VerifyOperandRegI64HostSize(source_offset);
      VM_VerifyOperandRegI64HostSize(length);
      VM_VerifyResultRegI64(result);
    });

    //===------------------------------------------------------------------===//
    // Lists
    //===------------------------------------------------------------------===//

    VERIFY_OP(CORE, ListAlloc, {
      VM_VerifyTypeOf(element_type);
      VM_VerifyOperandRegI32(initial_capacity);
      VM_VerifyResultRegRef(result);
    });

    VERIFY_OP(CORE, ListReserve, {
      VM_VerifyOperandRegRef(list);
      VM_VerifyOperandRegI32(minimum_capacity);
    });

    VERIFY_OP(CORE, ListSize, {
      VM_VerifyOperandRegRef(list);
      VM_VerifyResultRegI32(result);
    });

    VERIFY_OP(CORE, ListResize, {
      VM_VerifyOperandRegRef(list);
      VM_VerifyOperandRegI32(new_size);
    });

    VERIFY_OP(CORE, ListGetI32, {
      VM_VerifyOperandRegRef(list);
      VM_VerifyOperandRegI32(index);
      VM_VerifyResultRegI32(result);
    });

    VERIFY_OP(CORE, ListSetI32, {
      VM_VerifyOperandRegRef(list);
      VM_VerifyOperandRegI32(index);
      VM_VerifyOperandRegI32(raw_value);
    });

    VERIFY_OP(CORE, ListGetI64, {
      VM_VerifyOperandRegRef(list);
      VM_VerifyOperandRegI32(index);
      VM_VerifyResultRegI64(result);
    });

    VERIFY_OP(CORE, ListSetI64, {
      VM_VerifyOperandRegRef(list);
      VM_VerifyOperandRegI32(index);
      VM_VerifyOperandRegI64(value);
    });

    VERIFY_OP(CORE, ListGetRef, {
      VM_VerifyOperandRegRef(list);
      VM_VerifyOperandRegI32(index);
      VM_VerifyTypeOf(type_def);
      VM_VerifyResultRegRef(result);
    });

    VERIFY_OP(CORE, ListSetRef, {
      VM_VerifyOperandRegRef(list);
      VM_VerifyOperandRegI32(index);
      VM_VerifyOperandRegRef(value);
    });

    //===------------------------------------------------------------------===//
    // Conditional assignment
    //===------------------------------------------------------------------===//

    VERIFY_OP(CORE, SelectI32, {
      VM_VerifyOperandRegI32(condition);
      VM_VerifyOperandRegI32(true_value);
      VM_VerifyOperandRegI32(false_value);
      VM_VerifyResultRegI32(result);
    });

    VERIFY_OP(CORE, SelectI64, {
      VM_VerifyOperandRegI32(condition);
      VM_VerifyOperandRegI64(true_value);
      VM_VerifyOperandRegI64(false_value);
      VM_VerifyResultRegI64(result);
    });

    VERIFY_OP(CORE, SelectRef, {
      VM_VerifyOperandRegI32(condition);
      // TODO(benvanik): remove the type_id and use either LHS/RHS (if both are
      // null then output is always null so no need to know the type).
      VM_VerifyTypeOf(true_value_type_def);
      VM_VerifyOperandRegRef(true_value);
      VM_VerifyOperandRegRef(false_value);
      VM_VerifyResultRegRef(result);
    });

    VERIFY_OP(CORE, SwitchI32, {
      VM_VerifyOperandRegI32(index);
      VM_VerifyOperandRegI32(default_value);
      VM_VerifyVariadicOperandsI32(values);
      VM_VerifyResultRegI32(result);
    });

    VERIFY_OP(CORE, SwitchI64, {
      VM_VerifyOperandRegI32(index);
      VM_VerifyOperandRegI64(default_value);
      VM_VerifyVariadicOperandsI64(values);
      VM_VerifyResultRegI64(result);
    });

    VERIFY_OP(CORE, SwitchRef, {
      VM_VerifyOperandRegI32(index);
      VM_VerifyTypeOf(type_def);
      VM_VerifyOperandRegRef(default_value);
      VM_VerifyVariadicOperandsRef(values, type_def);
      VM_VerifyResultRegRef(result);
    });

    //===------------------------------------------------------------------===//
    // Native integer arithmetic
    //===------------------------------------------------------------------===//

    VERIFY_OP_CORE_BINARY_I32(AddI32);
    VERIFY_OP_CORE_BINARY_I32(SubI32);
    VERIFY_OP_CORE_BINARY_I32(MulI32);
    VERIFY_OP_CORE_BINARY_I32(DivI32S);
    VERIFY_OP_CORE_BINARY_I32(DivI32U);
    VERIFY_OP_CORE_BINARY_I32(RemI32S);
    VERIFY_OP_CORE_BINARY_I32(RemI32U);
    VERIFY_OP_CORE_TERNARY_I32(FMAI32);
    VERIFY_OP_CORE_UNARY_I32(AbsI32);
    VERIFY_OP_CORE_BINARY_I32(MinI32S);
    VERIFY_OP_CORE_BINARY_I32(MinI32U);
    VERIFY_OP_CORE_BINARY_I32(MaxI32S);
    VERIFY_OP_CORE_BINARY_I32(MaxI32U);
    VERIFY_OP_CORE_UNARY_I32(NotI32);
    VERIFY_OP_CORE_BINARY_I32(AndI32);
    VERIFY_OP_CORE_BINARY_I32(OrI32);
    VERIFY_OP_CORE_BINARY_I32(XorI32);
    VERIFY_OP_CORE_UNARY_I32(CtlzI32);

    VERIFY_OP_CORE_BINARY_I64(AddI64);
    VERIFY_OP_CORE_BINARY_I64(SubI64);
    VERIFY_OP_CORE_BINARY_I64(MulI64);
    VERIFY_OP_CORE_BINARY_I64(DivI64S);
    VERIFY_OP_CORE_BINARY_I64(DivI64U);
    VERIFY_OP_CORE_BINARY_I64(RemI64S);
    VERIFY_OP_CORE_BINARY_I64(RemI64U);
    VERIFY_OP_CORE_TERNARY_I64(FMAI64);
    VERIFY_OP_CORE_UNARY_I64(AbsI64);
    VERIFY_OP_CORE_BINARY_I64(MinI64S);
    VERIFY_OP_CORE_BINARY_I64(MinI64U);
    VERIFY_OP_CORE_BINARY_I64(MaxI64S);
    VERIFY_OP_CORE_BINARY_I64(MaxI64U);
    VERIFY_OP_CORE_UNARY_I64(NotI64);
    VERIFY_OP_CORE_BINARY_I64(AndI64);
    VERIFY_OP_CORE_BINARY_I64(OrI64);
    VERIFY_OP_CORE_BINARY_I64(XorI64);
    VERIFY_OP_CORE_UNARY_I64(CtlzI64);

    //===------------------------------------------------------------------===//
    // Casting and type conversion/emulation
    //===------------------------------------------------------------------===//

    // NOTE: these all operate on 32-bit registers.
    VERIFY_OP_CORE_UNARY_I32(TruncI32I8);
    VERIFY_OP_CORE_UNARY_I32(TruncI32I16);
    VERIFY_OP_CORE_UNARY_I32(ExtI8I32S);
    VERIFY_OP_CORE_UNARY_I32(ExtI8I32U);
    VERIFY_OP_CORE_UNARY_I32(ExtI16I32S);
    VERIFY_OP_CORE_UNARY_I32(ExtI16I32U);

    // NOTE: 64-bit ones are actually changing register widths.
    VERIFY_OP(CORE, TruncI64I32, {
      VM_VerifyOperandRegI64(operand);
      VM_VerifyResultRegI32(result);
    });
    VERIFY_OP(CORE, ExtI32I64S, {
      VM_VerifyOperandRegI32(operand);
      VM_VerifyResultRegI64(result);
    });
    VERIFY_OP(CORE, ExtI32I64U, {
      VM_VerifyOperandRegI32(operand);
      VM_VerifyResultRegI64(result);
    });

    VERIFY_OP(CORE, CastAnyRef, {
      VM_VerifyOperandRegRef(operand);
      VM_VerifyTypeOf(result);
      VM_VerifyResultRegRef(result);
    });

    //===------------------------------------------------------------------===//
    // Native bitwise shifts and rotates
    //===------------------------------------------------------------------===//

#define VERIFY_OP_CORE_SHIFT_I32(op_name) \
  VERIFY_OP(CORE, op_name, {              \
    VM_VerifyOperandRegI32(operand);      \
    VM_VerifyOperandRegI32(amount);       \
    VM_VerifyResultRegI32(result);        \
  });

    VERIFY_OP_CORE_SHIFT_I32(ShlI32);
    VERIFY_OP_CORE_SHIFT_I32(ShrI32S);
    VERIFY_OP_CORE_SHIFT_I32(ShrI32U);

#define VERIFY_OP_CORE_SHIFT_I64(op_name) \
  VERIFY_OP(CORE, op_name, {              \
    VM_VerifyOperandRegI64(operand);      \
    VM_VerifyOperandRegI32(amount);       \
    VM_VerifyResultRegI64(result);        \
  });

    VERIFY_OP_CORE_SHIFT_I64(ShlI64);
    VERIFY_OP_CORE_SHIFT_I64(ShrI64S);
    VERIFY_OP_CORE_SHIFT_I64(ShrI64U);

    //===------------------------------------------------------------------===//
    // Comparison ops
    //===------------------------------------------------------------------===//

    VERIFY_OP_CORE_BINARY_I32(CmpEQI32);
    VERIFY_OP_CORE_BINARY_I32(CmpNEI32);
    VERIFY_OP_CORE_BINARY_I32(CmpLTI32S);
    VERIFY_OP_CORE_BINARY_I32(CmpLTI32U);
    VERIFY_OP_CORE_UNARY_I32(CmpNZI32);

#define VERIFY_OP_CORE_CMP_I64(op_name) \
  VERIFY_OP(CORE, op_name, {            \
    VM_VerifyOperandRegI64(lhs);        \
    VM_VerifyOperandRegI64(rhs);        \
    VM_VerifyResultRegI32(result);      \
  });

    VERIFY_OP_CORE_CMP_I64(CmpEQI64);
    VERIFY_OP_CORE_CMP_I64(CmpNEI64);
    VERIFY_OP_CORE_CMP_I64(CmpLTI64S);
    VERIFY_OP_CORE_CMP_I64(CmpLTI64U);
    VERIFY_OP(CORE, CmpNZI64, {
      VM_VerifyOperandRegI64(operand);
      VM_VerifyResultRegI32(result);
    });

    VERIFY_OP(CORE, CmpEQRef, {
      VM_VerifyOperandRegRef(lhs);
      VM_VerifyOperandRegRef(rhs);
      VM_VerifyResultRegI32(result);
    });
    VERIFY_OP(CORE, CmpNERef, {
      VM_VerifyOperandRegRef(lhs);
      VM_VerifyOperandRegRef(rhs);
      VM_VerifyResultRegI32(result);
    });
    VERIFY_OP(CORE, CmpNZRef, {
      VM_VerifyOperandRegRef(operand);
      VM_VerifyResultRegI32(result);
    });

    //===------------------------------------------------------------------===//
    // Control flow
    //===------------------------------------------------------------------===//

    VERIFY_OP(CORE, Block, {
      // Define the new block in the block list. It may already be declared from
      // a prior branch.
      iree_vm_bytecode_block_t* block = NULL;
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_block_list_insert(
          &verify_state->block_list, pc - 1, &block));
      block->defined = 1;
      verify_state->in_block = 1;
    });

    VERIFY_OP(CORE, Branch, {
      VM_VerifyBranchTarget(dest_pc);
      VM_VerifyBranchOperands(operands);
      verify_state->in_block = 0;  // terminator
    });

    VERIFY_OP(CORE, CondBranch, {
      VM_VerifyOperandRegI32(condition);
      VM_VerifyBranchTarget(true_dest_pc);
      VM_VerifyBranchOperands(true_operands);
      VM_VerifyBranchTarget(false_dest_pc);
      VM_VerifyBranchOperands(false_operands);
      verify_state->in_block = 0;  // terminator
    });

    VERIFY_OP(CORE, BranchTable, {
      VM_VerifyOperandRegI32(index);
      VM_VerifyBranchTarget(default_dest_pc);
      VM_VerifyBranchOperands(default_operands);
      VM_VerifyConstI16(table_size);
      for (uint16_t i = 0; i < table_size; ++i) {
        VM_VerifyBranchTarget(case_dest_pc);
        VM_VerifyBranchOperands(case_operands);
      }
      verify_state->in_block = 0;  // terminator
    });

    VERIFY_OP(CORE, Call, {
      VM_VerifyFuncAttr(callee_ordinal);
      VM_VerifyVariadicOperandsAny(operands);
      VM_VerifyVariadicResultsAny(results);
      if (VM_IsImportOrdinal(callee_ordinal)) {
        VM_UnmaskImportOrdinal(callee_ordinal);
        VM_VerifyImportOrdinal(callee_ordinal);
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
        VM_VerifyFunctionOrdinal(callee_ordinal);
        IREE_RETURN_IF_ERROR(
            iree_vm_bytecode_function_verify_call(
                verify_state,
                iree_vm_FunctionSignatureDef_vec_at(
                    verify_state->function_signatures, callee_ordinal),
                /*segment_sizes=*/NULL, operands, results),
            "call to internal function %d", callee_ordinal);
      }
    });

    VERIFY_OP(CORE, CallVariadic, {
      VM_VerifyFuncAttr(callee_ordinal);
      VM_VerifyVariadicOperands(segment_sizes);
      VM_VerifyVariadicOperandsAny(operands);
      VM_VerifyVariadicResultsAny(results);
      if (IREE_UNLIKELY(!VM_IsImportOrdinal(callee_ordinal))) {
        // Variadic calls are currently only supported for import functions.
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "variadic calls only supported for internal callees");
      }
      VM_UnmaskImportOrdinal(callee_ordinal);
      VM_VerifyImportOrdinal(callee_ordinal);
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

    VERIFY_OP(CORE, Return, {
      VM_VerifyVariadicOperandsAny(operands);
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_function_verify_cconv_registers(
          verify_state, verify_state->cconv_results, /*segment_sizes=*/NULL,
          operands));
      verify_state->in_block = 0;  // terminator
    });

    VERIFY_OP(CORE, Fail, {
      VM_VerifyOperandRegI32(status);
      iree_string_view_t message;
      VM_VerifyStrAttr(message, &message);
      verify_state->in_block = 0;  // terminator
    });

    VERIFY_OP(CORE, ImportResolved, {
      VM_VerifyFuncAttr(import_ordinal);
      if (IREE_UNLIKELY(!VM_IsImportOrdinal(import_ordinal))) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "function ordinal %u is not an import ordinal",
                                import_ordinal);
      }
      VM_UnmaskImportOrdinal(import_ordinal);
      VM_VerifyImportOrdinal(import_ordinal);
      VM_VerifyResultRegI32(result);
    });

    //===------------------------------------------------------------------===//
    // Async/fiber ops
    //===------------------------------------------------------------------===//

    VERIFY_OP(CORE, Yield, {
      VM_VerifyBranchTarget(dest_pc);
      VM_VerifyBranchOperands(operands);
      verify_state->in_block = 0;  // terminator
    });

    //===------------------------------------------------------------------===//
    // Debugging
    //===------------------------------------------------------------------===//

    VERIFY_OP(CORE, Trace, {
      iree_string_view_t event_name;
      VM_VerifyStrAttr(event_name, &event_name);
      VM_VerifyVariadicOperandsAny(operands);
    });

    VERIFY_OP(CORE, Print, {
      iree_string_view_t event_name;
      VM_VerifyStrAttr(event_name, &event_name);
      VM_VerifyVariadicOperandsAny(operands);
    });

    VERIFY_OP(CORE, Break, {
      VM_VerifyBranchTarget(dest_pc);
      VM_VerifyBranchOperands(operands);
      verify_state->in_block = 0;  // terminator
    });

    VERIFY_OP(CORE, CondBreak, {
      VM_VerifyOperandRegI32(condition);
      VM_VerifyBranchTarget(dest);
      VM_VerifyBranchOperands(operands);
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

    VERIFY_OP(EXT_F32, GlobalLoadF32, {
      VM_VerifyGlobalAttr(byte_offset);
      VM_VerifyRwdataOffset(byte_offset, 4);
      VM_VerifyResultRegF32(value);
    });

    VERIFY_OP(EXT_F32, GlobalStoreF32, {
      VM_VerifyGlobalAttr(byte_offset);
      VM_VerifyRwdataOffset(byte_offset, 4);
      VM_VerifyOperandRegF32(value);
    });

    VERIFY_OP(EXT_F32, GlobalLoadIndirectF32, {
      VM_VerifyOperandRegI32(byte_offset);
      // NOTE: we have to verify the offset at runtime.
      VM_VerifyResultRegF32(value);
    });

    VERIFY_OP(EXT_F32, GlobalStoreIndirectF32, {
      VM_VerifyOperandRegI32(byte_offset);
      // NOTE: we have to verify the offset at runtime.
      VM_VerifyOperandRegF32(value);
    });

    //===----------------------------------------------------------------===//
    // ExtF32: Constants
    //===----------------------------------------------------------------===//

    VERIFY_OP(EXT_F32, ConstF32, {
      VM_VerifyFloatAttr32(value);
      VM_VerifyResultRegF32(result);
    });

    VERIFY_OP(EXT_F32, ConstF32Zero, { VM_VerifyResultRegF32(result); });

    //===----------------------------------------------------------------===//
    // ExtF32: Lists
    //===----------------------------------------------------------------===//

    VERIFY_OP(EXT_F32, ListGetF32, {
      VM_VerifyOperandRegRef(list);
      VM_VerifyOperandRegI32(index);
      VM_VerifyResultRegF32(result);
    });

    VERIFY_OP(EXT_F32, ListSetF32, {
      VM_VerifyOperandRegRef(list);
      VM_VerifyOperandRegI32(index);
      VM_VerifyOperandRegF32(value);
    });

    //===----------------------------------------------------------------===//
    // ExtF32: Conditional assignment
    //===----------------------------------------------------------------===//

    VERIFY_OP(EXT_F32, SelectF32, {
      VM_VerifyOperandRegI32(condition);
      VM_VerifyOperandRegF32(true_value);
      VM_VerifyOperandRegF32(false_value);
      VM_VerifyResultRegF32(result);
    });

    VERIFY_OP(EXT_F32, SwitchF32, {
      VM_VerifyOperandRegI32(index);
      VM_VerifyOperandRegF32(default_value);
      VM_VerifyVariadicOperandsF32(values);
      VM_VerifyResultRegF32(result);
    });

    //===----------------------------------------------------------------===//
    // ExtF32: Native floating-point arithmetic
    //===----------------------------------------------------------------===//

    VERIFY_OP_EXT_F32_BINARY_F32(AddF32);
    VERIFY_OP_EXT_F32_BINARY_F32(SubF32);
    VERIFY_OP_EXT_F32_BINARY_F32(MulF32);
    VERIFY_OP_EXT_F32_BINARY_F32(DivF32);
    VERIFY_OP_EXT_F32_BINARY_F32(RemF32);
    VERIFY_OP_EXT_F32_TERNARY_F32(FMAF32);
    VERIFY_OP_EXT_F32_UNARY_F32(AbsF32);
    VERIFY_OP_EXT_F32_UNARY_F32(NegF32);
    VERIFY_OP_EXT_F32_UNARY_F32(CeilF32);
    VERIFY_OP_EXT_F32_UNARY_F32(FloorF32);
    VERIFY_OP_EXT_F32_UNARY_F32(RoundF32);
    VERIFY_OP_EXT_F32_UNARY_F32(RoundF32Even);
    VERIFY_OP_EXT_F32_BINARY_F32(MinF32);
    VERIFY_OP_EXT_F32_BINARY_F32(MaxF32);

    VERIFY_OP_EXT_F32_UNARY_F32(AtanF32);
    VERIFY_OP_EXT_F32_BINARY_F32(Atan2F32);
    VERIFY_OP_EXT_F32_UNARY_F32(CosF32);
    VERIFY_OP_EXT_F32_UNARY_F32(SinF32);
    VERIFY_OP_EXT_F32_UNARY_F32(ExpF32);
    VERIFY_OP_EXT_F32_UNARY_F32(Exp2F32);
    VERIFY_OP_EXT_F32_UNARY_F32(ExpM1F32);
    VERIFY_OP_EXT_F32_UNARY_F32(LogF32);
    VERIFY_OP_EXT_F32_UNARY_F32(Log10F32);
    VERIFY_OP_EXT_F32_UNARY_F32(Log1pF32);
    VERIFY_OP_EXT_F32_UNARY_F32(Log2F32);
    VERIFY_OP_EXT_F32_BINARY_F32(PowF32);
    VERIFY_OP_EXT_F32_UNARY_F32(RsqrtF32);
    VERIFY_OP_EXT_F32_UNARY_F32(SqrtF32);
    VERIFY_OP_EXT_F32_UNARY_F32(TanhF32);
    VERIFY_OP_EXT_F32_UNARY_F32(ErfF32);

    //===----------------------------------------------------------------===//
    // ExtF32: Casting and type conversion/emulation
    //===----------------------------------------------------------------===//

    VERIFY_OP(EXT_F32, CastSI32F32, {
      VM_VerifyOperandRegI32(operand);
      VM_VerifyResultRegF32(result);
    });
    VERIFY_OP(EXT_F32, CastUI32F32, {
      VM_VerifyOperandRegI32(operand);
      VM_VerifyResultRegF32(result);
    });
    VERIFY_OP(EXT_F32, CastF32SI32, {
      VM_VerifyOperandRegF32(operand);
      VM_VerifyResultRegI32(result);
    });
    VERIFY_OP(EXT_F32, CastF32UI32, {
      VM_VerifyOperandRegF32(operand);
      VM_VerifyResultRegI32(result);
    });
    VERIFY_OP(EXT_F32, BitcastI32F32, {
      VM_VerifyOperandRegI32(operand);
      VM_VerifyResultRegF32(result);
    });
    VERIFY_OP(EXT_F32, BitcastF32I32, {
      VM_VerifyOperandRegF32(operand);
      VM_VerifyResultRegI32(result);
    });

    //===----------------------------------------------------------------===//
    // ExtF32: Comparison ops
    //===----------------------------------------------------------------===//

#define VERIFY_OP_EXT_F32_CMP_F32(op_name) \
  VERIFY_OP(EXT_F32, op_name, {            \
    VM_VerifyOperandRegF32(lhs);           \
    VM_VerifyOperandRegF32(rhs);           \
    VM_VerifyResultRegI32(result);         \
  });

    VERIFY_OP_EXT_F32_CMP_F32(CmpEQF32O);
    VERIFY_OP_EXT_F32_CMP_F32(CmpEQF32U);
    VERIFY_OP_EXT_F32_CMP_F32(CmpNEF32O);
    VERIFY_OP_EXT_F32_CMP_F32(CmpNEF32U);
    VERIFY_OP_EXT_F32_CMP_F32(CmpLTF32O);
    VERIFY_OP_EXT_F32_CMP_F32(CmpLTF32U);
    VERIFY_OP_EXT_F32_CMP_F32(CmpLTEF32O);
    VERIFY_OP_EXT_F32_CMP_F32(CmpLTEF32U);
    VERIFY_OP(EXT_F32, CmpNaNF32, {
      VM_VerifyOperandRegF32(operand);
      VM_VerifyResultRegI32(result);
    });

    //===----------------------------------------------------------------===//
    // ExtF32: Buffers
    //===----------------------------------------------------------------===//

    VERIFY_OP(EXT_F32, BufferFillF32, {
      VM_VerifyOperandRegRef(target_buffer);
      VM_VerifyOperandRegI64HostSize(target_offset);
      VM_VerifyOperandRegI64HostSize(length);
      VM_VerifyOperandRegF32(value);
    });

    VERIFY_OP(EXT_F32, BufferLoadF32, {
      VM_VerifyOperandRegRef(source_buffer);
      VM_VerifyOperandRegI64HostSize(source_offset);
      VM_VerifyResultRegF32(result);
    });

    VERIFY_OP(EXT_F32, BufferStoreF32, {
      VM_VerifyOperandRegRef(target_buffer);
      VM_VerifyOperandRegI64HostSize(target_offset);
      VM_VerifyOperandRegF32(value);
    });

    END_VERIFY_PREFIX();
#else
    UNHANDLED_VERIFY_PREFIX(PrefixExtF32, iree_vm_FeatureBits_EXT_F32);
#endif  // IREE_VM_EXT_F32_ENABLE

    VERIFY_OP(CORE, PrefixExtF64, {
      IREE_VM_VERIFY_REQUIREMENT(iree_vm_FeatureBits_EXT_F64);
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "EXT_64 not yet implemented");
    });

    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unrecognized opcode %u", bytecode_data[pc - 1]);
  }

  *out_next_pc = pc;
  return iree_ok_status();
}
