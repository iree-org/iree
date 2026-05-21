// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_BYTECODE_ASSEMBLER_MODULE_H_
#define IREE_VM_BYTECODE_ASSEMBLER_MODULE_H_

#include <stdbool.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/vm/bytecode/isa/isa.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Basic block label defined in a function body.
typedef struct iree_vm_bytecode_assembler_label_t {
  // Label name without the leading caret.
  iree_string_view_t name;
  // Bytecode PC where the label's block begins.
  uint32_t pc;
} iree_vm_bytecode_assembler_label_t;

// Branch target patch site in a function body.
typedef struct iree_vm_bytecode_assembler_fixup_t {
  // Target label name without the leading caret.
  iree_string_view_t label_name;
  // Byte offset in the bytecode stream where the little-endian PC is patched.
  uint32_t bytecode_offset;
} iree_vm_bytecode_assembler_fixup_t;

// Module global declaration used for symbolic global references.
typedef struct iree_vm_bytecode_assembler_global_t {
  // Global symbol name without the leading '@'.
  iree_string_view_t name;
  // Size in bytes of the primitive storage slot, or 0 for ref storage.
  uint32_t storage_size;
  // Assigned byte offset for primitive storage, or ref ordinal for ref storage.
  uint32_t ordinal;
} iree_vm_bytecode_assembler_global_t;

// Global symbol patch site in the module bytecode.
typedef struct iree_vm_bytecode_assembler_global_fixup_t {
  // Global symbol name without the leading '@'.
  iree_string_view_t name;
  // Byte offset in the bytecode stream where the little-endian ordinal lives.
  uint32_t bytecode_offset;
  // Expected primitive storage size, or 0 when the fixup targets ref storage.
  uint32_t storage_size;
} iree_vm_bytecode_assembler_global_fixup_t;

// Module type declaration used by type-of bytecode operands.
typedef struct iree_vm_bytecode_assembler_type_t {
  // Fully-qualified VM type name as stored in the bytecode module type table.
  iree_string_view_t full_name;
} iree_vm_bytecode_assembler_type_t;

// Runtime import declaration used by call and import-exists operands.
typedef struct iree_vm_bytecode_assembler_import_t {
  // Fully-qualified import name without the leading '@'.
  iree_string_view_t full_name;
  // Raw VM calling convention string used by the import signature.
  iree_string_view_t calling_convention;
  // Whether the import may remain unresolved at module load time.
  bool optional;
} iree_vm_bytecode_assembler_import_t;

// Embedded read-only data segment.
typedef struct iree_vm_bytecode_assembler_rodata_t {
  // Segment bytes owned by the assembler module.
  uint8_t* data;
  // Number of bytes in |data|.
  iree_host_size_t data_length;
} iree_vm_bytecode_assembler_rodata_t;

// Export declaration mapping a public symbol to an internal function symbol.
typedef struct iree_vm_bytecode_assembler_export_t {
  // Exported function name without the leading '@'.
  iree_string_view_t name;
  // Internal function symbol referenced by the export.
  iree_string_view_t function_name;
} iree_vm_bytecode_assembler_export_t;

// Finalized function descriptor collected by the parser.
typedef struct iree_vm_bytecode_assembler_function_t {
  // Internal function name without the leading '@'.
  iree_string_view_t name;
  // Calling convention string owned by the assembler state.
  iree_string_view_t cconv;
  // Byte offset of the function body within the module bytecode data.
  uint32_t bytecode_offset;
  // Byte length of the function body.
  uint32_t bytecode_length;
  // Feature bits required by bytecode instructions in this function.
  iree_vm_FeatureBits_enum_t requirements;
  // Number of basic blocks in the function body.
  uint16_t block_count;
  // Minimum i32 register file size required by the function body.
  uint16_t i32_register_count;
  // Minimum ref register file size required by the function body.
  uint16_t ref_register_count;
} iree_vm_bytecode_assembler_function_t;

// Function symbol patch site in the module bytecode.
typedef struct iree_vm_bytecode_assembler_function_fixup_t {
  // Internal function name without the leading '@'.
  iree_string_view_t name;
  // Byte offset in the bytecode stream where the little-endian ordinal lives.
  uint32_t bytecode_offset;
} iree_vm_bytecode_assembler_function_fixup_t;

// Mutable in-memory assembly module.
typedef struct iree_vm_bytecode_assembler_module_t {
  // Allocator used for transient parser storage and the returned archive.
  iree_allocator_t host_allocator;

  // Whether a module directive has been parsed.
  bool has_module;
  // Module name without the leading '@'.
  iree_string_view_t module_name;
  // Module version stored in the bytecode module FlatBuffer.
  uint32_t module_version;
  // Feature bits required by any function in this module.
  iree_vm_FeatureBits_enum_t module_requirements;

  // Whether subsequent lines are part of the function body.
  bool in_function;
  // Whether the current function body has emitted a terminator.
  bool has_terminator;
  // Feature bits required by the current function body.
  iree_vm_FeatureBits_enum_t function_requirements;
  // Internal function name without the leading '@'.
  iree_string_view_t function_name;
  // Number of ABI arguments in the function signature.
  iree_host_size_t argument_count;
  // Number of ABI results in the function signature.
  iree_host_size_t result_count;
  // Number of basic blocks in the function body.
  iree_host_size_t block_count;
  // Minimum i32 register file size required by the function body.
  uint16_t i32_register_count;
  // Minimum ref register file size required by the function body.
  uint16_t ref_register_count;
  // Byte offset of the current function body within |bytecode_builder|.
  uint32_t function_bytecode_offset;

  // Primitive global byte storage required by the module state.
  uint32_t global_byte_capacity;
  // Ref global storage required by the module state.
  uint32_t global_ref_count;

  // Global definitions declared at module scope.
  iree_vm_bytecode_assembler_global_t* globals;
  // Number of entries in |globals|.
  iree_host_size_t global_count;
  // Allocated capacity of |globals|.
  iree_host_size_t global_capacity;
  // Global symbol patch sites in the module bytecode.
  iree_vm_bytecode_assembler_global_fixup_t* global_fixups;
  // Number of entries in |global_fixups|.
  iree_host_size_t global_fixup_count;
  // Allocated capacity of |global_fixups|.
  iree_host_size_t global_fixup_capacity;

  // Type definitions declared at module scope.
  iree_vm_bytecode_assembler_type_t* types;
  // Number of entries in |types|.
  iree_host_size_t type_count;
  // Allocated capacity of |types|.
  iree_host_size_t type_capacity;

  // Import definitions declared at module scope.
  iree_vm_bytecode_assembler_import_t* imports;
  // Number of entries in |imports|.
  iree_host_size_t import_count;
  // Allocated capacity of |imports|.
  iree_host_size_t import_capacity;

  // Read-only data segments declared at module scope.
  iree_vm_bytecode_assembler_rodata_t* rodata_segments;
  // Number of entries in |rodata_segments|.
  iree_host_size_t rodata_segment_count;
  // Allocated capacity of |rodata_segments|.
  iree_host_size_t rodata_segment_capacity;

  // Export definitions declared at module scope.
  iree_vm_bytecode_assembler_export_t* exports;
  // Number of entries in |exports|.
  iree_host_size_t export_count;
  // Allocated capacity of |exports|.
  iree_host_size_t export_capacity;

  // Function definitions declared at module scope.
  iree_vm_bytecode_assembler_function_t* functions;
  // Number of entries in |functions|.
  iree_host_size_t function_count;
  // Allocated capacity of |functions|.
  iree_host_size_t function_capacity;
  // Function symbol patch sites in the module bytecode.
  iree_vm_bytecode_assembler_function_fixup_t* function_fixups;
  // Number of entries in |function_fixups|.
  iree_host_size_t function_fixup_count;
  // Allocated capacity of |function_fixups|.
  iree_host_size_t function_fixup_capacity;

  // Labels defined in the current function body.
  iree_vm_bytecode_assembler_label_t* labels;
  // Number of entries in |labels|.
  iree_host_size_t label_count;
  // Allocated capacity of |labels|.
  iree_host_size_t label_capacity;
  // Branch target patch sites in the current function body.
  iree_vm_bytecode_assembler_fixup_t* fixups;
  // Number of entries in |fixups|.
  iree_host_size_t fixup_count;
  // Allocated capacity of |fixups|.
  iree_host_size_t fixup_capacity;

  // Calling convention fragment for function arguments.
  iree_string_builder_t argument_cconv_builder;
  // Calling convention fragment for function results.
  iree_string_builder_t result_cconv_builder;
  // Encoded bytecode for all supported functions.
  iree_string_builder_t bytecode_builder;
  // Calling convention string for the current function.
  iree_string_builder_t cconv_builder;
} iree_vm_bytecode_assembler_module_t;

// Initializes an empty in-memory assembly module.
void iree_vm_bytecode_assembler_module_initialize(
    iree_allocator_t host_allocator,
    iree_vm_bytecode_assembler_module_t* out_module);

// Releases all transient storage owned by |module|.
void iree_vm_bytecode_assembler_module_deinitialize(
    iree_vm_bytecode_assembler_module_t* module);

// Clones |value| into module-owned storage.
iree_status_t iree_vm_bytecode_assembler_clone_string(
    iree_vm_bytecode_assembler_module_t* module, iree_string_view_t value,
    iree_string_view_t* out_value);

// Finds an already-finalized function ordinal by internal symbol name.
iree_host_size_t iree_vm_bytecode_assembler_find_function_ordinal(
    const iree_vm_bytecode_assembler_module_t* module, iree_string_view_t name);

// Reserves storage for finalized functions.
iree_status_t iree_vm_bytecode_assembler_reserve_functions(
    iree_vm_bytecode_assembler_module_t* module,
    iree_host_size_t minimum_capacity);

// Appends a block label to the current function label table.
iree_status_t iree_vm_bytecode_assembler_append_label(
    iree_vm_bytecode_assembler_module_t* module, iree_string_view_t name,
    uint32_t pc);

// Appends a branch target fixup to the current function fixup table.
iree_status_t iree_vm_bytecode_assembler_append_fixup(
    iree_vm_bytecode_assembler_module_t* module, iree_string_view_t label_name,
    uint32_t bytecode_offset);

// Appends a global declaration to the module symbol table.
iree_status_t iree_vm_bytecode_assembler_append_global(
    iree_vm_bytecode_assembler_module_t* module, iree_string_view_t name,
    uint32_t storage_size);

// Appends a global symbol fixup to the module fixup table.
iree_status_t iree_vm_bytecode_assembler_append_global_fixup(
    iree_vm_bytecode_assembler_module_t* module, iree_string_view_t name,
    uint32_t bytecode_offset, uint32_t storage_size);

// Finds a type ordinal by fully-qualified name.
iree_host_size_t iree_vm_bytecode_assembler_find_type_ordinal(
    const iree_vm_bytecode_assembler_module_t* module,
    iree_string_view_t full_name);

// Appends a type definition to the module type table.
iree_status_t iree_vm_bytecode_assembler_append_type(
    iree_vm_bytecode_assembler_module_t* module, iree_string_view_t full_name,
    iree_host_size_t* out_ordinal);

// Finds or appends a type definition and returns its ordinal.
iree_status_t iree_vm_bytecode_assembler_lookup_or_append_type(
    iree_vm_bytecode_assembler_module_t* module, iree_string_view_t full_name,
    iree_host_size_t* out_ordinal);

// Finds an import ordinal by fully-qualified name.
iree_host_size_t iree_vm_bytecode_assembler_find_import_ordinal(
    const iree_vm_bytecode_assembler_module_t* module,
    iree_string_view_t full_name);

// Appends an imported function definition.
iree_status_t iree_vm_bytecode_assembler_append_import(
    iree_vm_bytecode_assembler_module_t* module, iree_string_view_t full_name,
    iree_string_view_t calling_convention, bool optional);

// Appends an embedded read-only data segment.
iree_status_t iree_vm_bytecode_assembler_append_rodata_segment(
    iree_vm_bytecode_assembler_module_t* module, iree_host_size_t ordinal,
    iree_const_byte_span_t data);

// Appends an export declaration to the module export table.
iree_status_t iree_vm_bytecode_assembler_append_export(
    iree_vm_bytecode_assembler_module_t* module, iree_string_view_t export_name,
    iree_string_view_t function_name);

// Appends a function symbol fixup to the module bytecode.
iree_status_t iree_vm_bytecode_assembler_append_function_fixup(
    iree_vm_bytecode_assembler_module_t* module, iree_string_view_t name,
    uint32_t bytecode_offset);

// Assigns storage ordinals to declared globals and computes module state size.
iree_status_t iree_vm_bytecode_assembler_assign_global_ordinals(
    iree_vm_bytecode_assembler_module_t* module);

// Resolves global symbol fixups in the encoded bytecode buffer.
iree_status_t iree_vm_bytecode_assembler_resolve_global_fixups(
    iree_vm_bytecode_assembler_module_t* module);

// Resolves function symbol fixups in the encoded bytecode buffer.
iree_status_t iree_vm_bytecode_assembler_resolve_function_fixups(
    iree_vm_bytecode_assembler_module_t* module);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_VM_BYTECODE_ASSEMBLER_MODULE_H_
