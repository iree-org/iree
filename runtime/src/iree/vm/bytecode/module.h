// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_BYTECODE_MODULE_H_
#define IREE_VM_BYTECODE_MODULE_H_

#include "iree/base/api.h"
#include "iree/vm/api.h"
#include "iree/vm/bytecode/isa/isa.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Flags controlling bytecode module creation behavior.
typedef enum iree_vm_bytecode_module_flags_e {
  IREE_VM_BYTECODE_MODULE_FLAG_NONE = 0,
  // Allows unregistered ref types to be resolved as placeholder types.
  // The module can be used for reflection/disassembly but NOT execution.
  // Types will be registered with the instance as placeholders.
  IREE_VM_BYTECODE_MODULE_FLAG_ALLOW_PLACEHOLDER_TYPES = 1u << 0,
} iree_vm_bytecode_module_flags_t;

// Controls how bytecode disassembly is formatted.
typedef enum iree_vm_bytecode_disassembly_format_e {
  IREE_VM_BYTECODE_DISASSEMBLY_FORMAT_DEFAULT = 0,
  // Includes the input register values inline in the op text.
  // Example: `%i0 <= ShrI32U %i2(5), %i3(6)`
  IREE_VM_BYTECODE_DISASSEMBLY_FORMAT_INLINE_VALUES = 1u << 0,
  // Includes bytecode offsets before each disassembled op.
  // Example: `[00000001]    %i0 = vm.const.i32 9`.
  IREE_VM_BYTECODE_DISASSEMBLY_FORMAT_BYTECODE_OFFSETS = 1u << 1,
} iree_vm_bytecode_disassembly_format_t;

// Creates a VM module from an in-memory ModuleDef FlatBuffer archive.
// If a |archive_allocator| is provided then it will be used to free the
// |archive_contents| when the module is destroyed and otherwise the ownership
// of the memory remains with the caller.
IREE_API_EXPORT iree_status_t iree_vm_bytecode_module_create(
    iree_vm_instance_t* instance, iree_vm_bytecode_module_flags_t flags,
    iree_const_byte_span_t archive_contents, iree_allocator_t archive_allocator,
    iree_allocator_t allocator, iree_vm_module_t** out_module);

// Returns the feature bits required by a bytecode module archive.
//
// This only validates the archive enough to read module metadata. Use
// iree_vm_bytecode_module_create to fully verify and load executable modules.
IREE_API_EXPORT iree_status_t iree_vm_bytecode_module_query_required_features(
    iree_const_byte_span_t archive_contents,
    iree_vm_FeatureBits_enum_t* out_required_features);

// Disassembles an entire function's bytecode.
// Output is assembly-like by default:
// ^bb0:
//   %i0 = vm.const.i32 9  // 0x00000009
//
// IREE_VM_BYTECODE_DISASSEMBLY_FORMAT_BYTECODE_OFFSETS can be used to include
// bytecode offsets before each line:
// [00000000]^bb0:
// [00000001]  %i0 = vm.const.i32 9  // 0x00000009
IREE_API_EXPORT iree_status_t iree_vm_bytecode_module_disassemble_function(
    iree_vm_module_t* module, uint16_t function_ordinal,
    iree_vm_bytecode_disassembly_format_t format,
    iree_string_builder_t* string_builder);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_VM_BYTECODE_MODULE_H_
