// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/bytecode/disassembler.h"

#include <inttypes.h>

#include "iree/vm/ops.h"

// ISA decoding policy for the disassembler.
// NOTE: the disassembler is a debugging tool and assumes verified bytecode.
#define IREE_VM_ISA_BYTECODE_DATA bytecode_data
#define IREE_VM_ISA_PC pc
#define IREE_VM_ISA_REQUIRE(bytes) ((void)0)
#define IREE_VM_ISA_LOOKUP_TYPE(type_id, out_type)             \
  do {                                                         \
    (out_type) = iree_vm_map_type(module, (int32_t)(type_id)); \
  } while (0)
#include "iree/vm/bytecode/utils/isa_decoder.inl"

#define IREE_VM_ISA_BEGIN_DISASM_PREFIX(op_name, ext) \
  case IREE_VM_OP_CORE_##op_name: {                   \
    switch (bytecode_data[pc++]) {
#define IREE_VM_ISA_END_DISASM_PREFIX()                \
  default:                                             \
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED, \
                            "unhandled ext opcode");   \
    }                                                  \
    break;                                             \
    }
#define IREE_VM_ISA_UNHANDLED_DISASM_PREFIX(op_name, ext)          \
  case IREE_VM_OP_CORE_##op_name: {                                \
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,             \
                            "unhandled dispatch extension " #ext); \
  }

#define IREE_VM_ISA_EMIT_OP(ext, op_name) case IREE_VM_OP_##ext##_##op_name:

// Emits register name, extracting move bit from raw register value for refs.
#define IREE_VM_ISA_EMIT_REG_NAME(reg)                   \
  if ((reg) & IREE_VM_ISA_REF_REGISTER_TYPE_BIT) {       \
    IREE_VM_ISA_EMIT_REF_REG_NAME_MOVE(                  \
        reg, (reg) & IREE_VM_ISA_REF_REGISTER_MOVE_BIT); \
  } else {                                               \
    IREE_VM_ISA_EMIT_I32_REG_NAME(reg);                  \
  }
#define IREE_VM_ISA_EMIT_I32_REG_NAME(reg)                \
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format( \
      b, "%%i%u", ((reg) & IREE_VM_ISA_I32_REGISTER_MASK)));
#define IREE_VM_ISA_EMIT_I64_REG_NAME(reg)                    \
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(     \
      b, "%%i%u:%u", ((reg) & IREE_VM_ISA_I32_REGISTER_MASK), \
      ((reg) & IREE_VM_ISA_I32_REGISTER_MASK) + 1));
#define IREE_VM_ISA_EMIT_F32_REG_NAME(reg) IREE_VM_ISA_EMIT_I32_REG_NAME(reg)
#define IREE_VM_ISA_EMIT_F64_REG_NAME(reg) IREE_VM_ISA_EMIT_I64_REG_NAME(reg)
// Emits %r0 for ref registers (no move info available).
#define IREE_VM_ISA_EMIT_REF_REG_NAME(reg)                \
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format( \
      b, "%%r%u", ((reg) & IREE_VM_ISA_REF_REGISTER_MASK)));
// Emits %r0 for retain or %R0 for move (uppercase indicates move semantics).
#define IREE_VM_ISA_EMIT_REF_REG_NAME_MOVE(reg, is_move)  \
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format( \
      b, (is_move) ? "%%R%u" : "%%r%u",                   \
      ((reg) & IREE_VM_ISA_REF_REGISTER_MASK)));

#define IREE_VM_ISA_EMIT_REG_VALUE(regs, reg)                                 \
  if ((reg) & IREE_VM_ISA_REF_REGISTER_TYPE_BIT) {                            \
    iree_vm_ref_t* ref = &(regs)->ref[(reg) & IREE_VM_ISA_REF_REGISTER_MASK]; \
    if (iree_vm_ref_is_null(ref)) {                                           \
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, "null"));    \
    } else {                                                                  \
      iree_string_view_t type_name = iree_vm_ref_type_name(ref->type);        \
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(                 \
          b, "!%.*s/%p", (int)type_name.size, type_name.data, ref->ptr));     \
    }                                                                         \
  } else {                                                                    \
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(                   \
        b, "%u", ((regs)->i32[(reg) & IREE_VM_ISA_I32_REGISTER_MASK])));      \
  }

static iree_status_t iree_vm_bytecode_disassembler_emit_type_name(
    iree_vm_type_def_t type_def, iree_string_builder_t* b) {
  if (iree_vm_type_def_is_value(type_def)) {
    const char* type_name;
    switch (iree_vm_type_def_as_value(type_def)) {
      case IREE_VM_VALUE_TYPE_I8:
        type_name = "i8";
        break;
      case IREE_VM_VALUE_TYPE_I16:
        type_name = "i16";
        break;
      case IREE_VM_VALUE_TYPE_I32:
        type_name = "i32";
        break;
      case IREE_VM_VALUE_TYPE_I64:
        type_name = "i64";
        break;
      case IREE_VM_VALUE_TYPE_F32:
        type_name = "f32";
        break;
      case IREE_VM_VALUE_TYPE_F64:
        type_name = "f64";
        break;
      default:
        type_name = "unknown";
        break;
    }
    return iree_string_builder_append_cstring(b, type_name);
  } else if (iree_vm_type_def_is_ref(type_def)) {
    iree_string_view_t type_name =
        iree_vm_ref_type_name(iree_vm_type_def_as_ref(type_def));
    return iree_string_builder_append_format(b, "%.*s", (int)type_name.size,
                                             type_name.data);
  } else {
    return iree_string_builder_append_cstring(b, "*");
  }
}
#define IREE_VM_ISA_EMIT_TYPE_NAME(type_def) \
  IREE_RETURN_IF_ERROR(                      \
      iree_vm_bytecode_disassembler_emit_type_name(type_def, b))

static iree_status_t iree_vm_bytecode_disassembler_emit_operand_list(
    const iree_vm_registers_t* regs, const iree_vm_register_list_t* list,
    iree_vm_bytecode_disassembly_format_t format, iree_string_builder_t* b) {
  bool include_values =
      regs && (format & IREE_VM_BYTECODE_DISASSEMBLY_FORMAT_INLINE_VALUES);
  for (uint16_t i = 0; i < list->size; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
    }
    uint16_t reg = list->registers[i];
    IREE_VM_ISA_EMIT_REG_NAME(reg);
    if (include_values) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, "("));
      IREE_VM_ISA_EMIT_REG_VALUE(regs, reg);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ")"));
    }
  }
  return iree_ok_status();
}
#define IREE_VM_ISA_EMIT_OPERAND_REG_LIST(reg_list)                     \
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_disassembler_emit_operand_list( \
      regs, reg_list, format, b))
static iree_status_t iree_vm_bytecode_disassembler_emit_result_list(
    const iree_vm_register_list_t* list,
    iree_vm_bytecode_disassembly_format_t format, iree_string_builder_t* b) {
  for (uint16_t i = 0; i < list->size; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
    }
    uint16_t reg = list->registers[i];
    IREE_VM_ISA_EMIT_REG_NAME(reg);
  }
  return iree_ok_status();
}
#define IREE_VM_ISA_EMIT_RESULT_REG_LIST(reg_list) \
  IREE_RETURN_IF_ERROR(                            \
      iree_vm_bytecode_disassembler_emit_result_list(reg_list, format, b))
static iree_status_t iree_vm_bytecode_disassembler_emit_remap_list(
    const iree_vm_registers_t* regs,
    const iree_vm_register_remap_list_t* remap_list,
    iree_vm_bytecode_disassembly_format_t format, iree_string_builder_t* b) {
  bool include_values =
      regs && (format & IREE_VM_BYTECODE_DISASSEMBLY_FORMAT_INLINE_VALUES);
  for (uint16_t i = 0; i < remap_list->size; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
    }
    IREE_VM_ISA_EMIT_REG_NAME(remap_list->pairs[i].src_reg);
    if (include_values) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, "("));
      IREE_VM_ISA_EMIT_REG_VALUE(regs, remap_list->pairs[i].src_reg);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ")"));
    }
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, "->"));
    IREE_VM_ISA_EMIT_REG_NAME(remap_list->pairs[i].dst_reg);
  }
  return iree_ok_status();
}
#define IREE_VM_ISA_EMIT_REMAP_LIST(remap_list)                       \
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_disassembler_emit_remap_list( \
      regs, remap_list, format, b))

#define IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I32(expr)                              \
  if (regs && (format & IREE_VM_BYTECODE_DISASSEMBLY_FORMAT_INLINE_VALUES)) {  \
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(b, "(%" PRId32 ")", \
                                                           (int32_t)(expr)));  \
  }
#define IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I64(expr)                             \
  if (regs && (format & IREE_VM_BYTECODE_DISASSEMBLY_FORMAT_INLINE_VALUES)) { \
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(                   \
        b, "(%" PRId64 ")", *(int64_t*)&(expr)));                             \
  }
#define IREE_VM_ISA_EMIT_OPTIONAL_VALUE_F32(expr)                             \
  if (regs && (format & IREE_VM_BYTECODE_DISASSEMBLY_FORMAT_INLINE_VALUES)) { \
    IREE_RETURN_IF_ERROR(                                                     \
        iree_string_builder_append_format(b, "(%f)", *(float*)&(expr)));      \
  }
#define IREE_VM_ISA_EMIT_OPTIONAL_VALUE_F64(expr)                             \
  if (regs && (format & IREE_VM_BYTECODE_DISASSEMBLY_FORMAT_INLINE_VALUES)) { \
    IREE_RETURN_IF_ERROR(                                                     \
        iree_string_builder_append_format(b, "(%f)", *(double*)&(expr)));     \
  }
#define IREE_VM_ISA_EMIT_OPTIONAL_VALUE_REF(expr)                             \
  if (regs && (format & IREE_VM_BYTECODE_DISASSEMBLY_FORMAT_INLINE_VALUES)) { \
    iree_vm_ref_t* ref = (expr);                                              \
    if (iree_vm_ref_is_null(ref)) {                                           \
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, "(null)"));  \
    } else {                                                                  \
      iree_string_view_t type_name = iree_vm_ref_type_name(ref->type);        \
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(                 \
          b, "(!%.*s/%p)", (int)type_name.size, type_name.data, ref->ptr));   \
    }                                                                         \
  }

#define IREE_VM_ISA_EMIT_OP_CORE_UNARY_I32(op_name, op_mnemonic)      \
  IREE_VM_ISA_EMIT_OP(CORE, op_name) {                                \
    IREE_VM_ISA_DECODE_OPERAND_I32(operand_reg);                      \
    IREE_VM_ISA_DECODE_RESULT_I32(result_reg);                        \
    IREE_VM_ISA_EMIT_I32_REG_NAME(result_reg);                        \
    IREE_RETURN_IF_ERROR(                                             \
        iree_string_builder_append_format(b, " = %s ", op_mnemonic)); \
    IREE_VM_ISA_EMIT_I32_REG_NAME(operand_reg);                       \
    IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I32(regs->i32[operand_reg]);      \
    break;                                                            \
  }

#define IREE_VM_ISA_EMIT_OP_CORE_BINARY_I32(op_name, op_mnemonic)      \
  IREE_VM_ISA_EMIT_OP(CORE, op_name) {                                 \
    IREE_VM_ISA_DECODE_OPERAND_I32(lhs_reg);                           \
    IREE_VM_ISA_DECODE_OPERAND_I32(rhs_reg);                           \
    IREE_VM_ISA_DECODE_RESULT_I32(result_reg);                         \
    IREE_VM_ISA_EMIT_I32_REG_NAME(result_reg);                         \
    IREE_RETURN_IF_ERROR(                                              \
        iree_string_builder_append_format(b, " = %s ", op_mnemonic));  \
    IREE_VM_ISA_EMIT_I32_REG_NAME(lhs_reg);                            \
    IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I32(regs->i32[lhs_reg]);           \
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", ")); \
    IREE_VM_ISA_EMIT_I32_REG_NAME(rhs_reg);                            \
    IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I32(regs->i32[rhs_reg]);           \
    break;                                                             \
  }

#define IREE_VM_ISA_EMIT_OP_CORE_TERNARY_I32(op_name, op_mnemonic)     \
  IREE_VM_ISA_EMIT_OP(CORE, op_name) {                                 \
    IREE_VM_ISA_DECODE_OPERAND_I32(a_reg);                             \
    IREE_VM_ISA_DECODE_OPERAND_I32(b_reg);                             \
    IREE_VM_ISA_DECODE_OPERAND_I32(c_reg);                             \
    IREE_VM_ISA_DECODE_RESULT_I32(result_reg);                         \
    IREE_VM_ISA_EMIT_I32_REG_NAME(result_reg);                         \
    IREE_RETURN_IF_ERROR(                                              \
        iree_string_builder_append_format(b, " = %s ", op_mnemonic));  \
    IREE_VM_ISA_EMIT_I32_REG_NAME(a_reg);                              \
    IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I32(regs->i32[a_reg]);             \
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", ")); \
    IREE_VM_ISA_EMIT_I32_REG_NAME(b_reg);                              \
    IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I32(regs->i32[b_reg]);             \
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", ")); \
    IREE_VM_ISA_EMIT_I32_REG_NAME(c_reg);                              \
    IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I32(regs->i32[c_reg]);             \
    break;                                                             \
  }

#define IREE_VM_ISA_EMIT_OP_CORE_UNARY_I64(op_name, op_mnemonic)      \
  IREE_VM_ISA_EMIT_OP(CORE, op_name) {                                \
    IREE_VM_ISA_DECODE_OPERAND_I64(operand_reg);                      \
    IREE_VM_ISA_DECODE_RESULT_I64(result_reg);                        \
    IREE_VM_ISA_EMIT_I64_REG_NAME(result_reg);                        \
    IREE_RETURN_IF_ERROR(                                             \
        iree_string_builder_append_format(b, " = %s ", op_mnemonic)); \
    IREE_VM_ISA_EMIT_I64_REG_NAME(operand_reg);                       \
    IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I64(regs->i32[operand_reg]);      \
    break;                                                            \
  }

#define IREE_VM_ISA_EMIT_OP_CORE_BINARY_I64(op_name, op_mnemonic)      \
  IREE_VM_ISA_EMIT_OP(CORE, op_name) {                                 \
    IREE_VM_ISA_DECODE_OPERAND_I64(lhs_reg);                           \
    IREE_VM_ISA_DECODE_OPERAND_I64(rhs_reg);                           \
    IREE_VM_ISA_DECODE_RESULT_I64(result_reg);                         \
    IREE_VM_ISA_EMIT_I64_REG_NAME(result_reg);                         \
    IREE_RETURN_IF_ERROR(                                              \
        iree_string_builder_append_format(b, " = %s ", op_mnemonic));  \
    IREE_VM_ISA_EMIT_I64_REG_NAME(lhs_reg);                            \
    IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I64(regs->i32[lhs_reg]);           \
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", ")); \
    IREE_VM_ISA_EMIT_I64_REG_NAME(rhs_reg);                            \
    IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I64(regs->i32[rhs_reg]);           \
    break;                                                             \
  }

#define IREE_VM_ISA_EMIT_OP_CORE_TERNARY_I64(op_name, op_mnemonic)     \
  IREE_VM_ISA_EMIT_OP(CORE, op_name) {                                 \
    IREE_VM_ISA_DECODE_OPERAND_I64(a_reg);                             \
    IREE_VM_ISA_DECODE_OPERAND_I64(b_reg);                             \
    IREE_VM_ISA_DECODE_OPERAND_I64(c_reg);                             \
    IREE_VM_ISA_DECODE_RESULT_I64(result_reg);                         \
    IREE_VM_ISA_EMIT_I64_REG_NAME(result_reg);                         \
    IREE_RETURN_IF_ERROR(                                              \
        iree_string_builder_append_format(b, " = %s ", op_mnemonic));  \
    IREE_VM_ISA_EMIT_I64_REG_NAME(a_reg);                              \
    IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I64(regs->i32[a_reg]);             \
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", ")); \
    IREE_VM_ISA_EMIT_I64_REG_NAME(b_reg);                              \
    IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I64(regs->i32[b_reg]);             \
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", ")); \
    IREE_VM_ISA_EMIT_I64_REG_NAME(c_reg);                              \
    IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I64(regs->i32[c_reg]);             \
    break;                                                             \
  }

#define IREE_VM_ISA_EMIT_OP_EXT_F32_UNARY_F32(op_name, op_mnemonic)   \
  IREE_VM_ISA_EMIT_OP(EXT_F32, op_name) {                             \
    IREE_VM_ISA_DECODE_OPERAND_F32(operand_reg);                      \
    IREE_VM_ISA_DECODE_RESULT_F32(result_reg);                        \
    IREE_VM_ISA_EMIT_F32_REG_NAME(result_reg);                        \
    IREE_RETURN_IF_ERROR(                                             \
        iree_string_builder_append_format(b, " = %s ", op_mnemonic)); \
    IREE_VM_ISA_EMIT_F32_REG_NAME(operand_reg);                       \
    IREE_VM_ISA_EMIT_OPTIONAL_VALUE_F32(regs->i32[operand_reg]);      \
    break;                                                            \
  }

#define IREE_VM_ISA_EMIT_OP_EXT_F32_BINARY_F32(op_name, op_mnemonic)   \
  IREE_VM_ISA_EMIT_OP(EXT_F32, op_name) {                              \
    IREE_VM_ISA_DECODE_OPERAND_F32(lhs_reg);                           \
    IREE_VM_ISA_DECODE_OPERAND_F32(rhs_reg);                           \
    IREE_VM_ISA_DECODE_RESULT_F32(result_reg);                         \
    IREE_VM_ISA_EMIT_F32_REG_NAME(result_reg);                         \
    IREE_RETURN_IF_ERROR(                                              \
        iree_string_builder_append_format(b, " = %s ", op_mnemonic));  \
    IREE_VM_ISA_EMIT_F32_REG_NAME(lhs_reg);                            \
    IREE_VM_ISA_EMIT_OPTIONAL_VALUE_F32(regs->i32[lhs_reg]);           \
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", ")); \
    IREE_VM_ISA_EMIT_F32_REG_NAME(rhs_reg);                            \
    IREE_VM_ISA_EMIT_OPTIONAL_VALUE_F32(regs->i32[rhs_reg]);           \
    break;                                                             \
  }

#define IREE_VM_ISA_EMIT_OP_EXT_F32_TERNARY_F32(op_name, op_mnemonic)  \
  IREE_VM_ISA_EMIT_OP(EXT_F32, op_name) {                              \
    IREE_VM_ISA_DECODE_OPERAND_F32(a_reg);                             \
    IREE_VM_ISA_DECODE_OPERAND_F32(b_reg);                             \
    IREE_VM_ISA_DECODE_OPERAND_F32(c_reg);                             \
    IREE_VM_ISA_DECODE_RESULT_F32(result_reg);                         \
    IREE_VM_ISA_EMIT_F32_REG_NAME(result_reg);                         \
    IREE_RETURN_IF_ERROR(                                              \
        iree_string_builder_append_format(b, " = %s ", op_mnemonic));  \
    IREE_VM_ISA_EMIT_F32_REG_NAME(a_reg);                              \
    IREE_VM_ISA_EMIT_OPTIONAL_VALUE_F32(regs->i32[a_reg]);             \
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", ")); \
    IREE_VM_ISA_EMIT_F32_REG_NAME(b_reg);                              \
    IREE_VM_ISA_EMIT_OPTIONAL_VALUE_F32(regs->i32[b_reg]);             \
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", ")); \
    IREE_VM_ISA_EMIT_F32_REG_NAME(c_reg);                              \
    IREE_VM_ISA_EMIT_OPTIONAL_VALUE_F32(regs->i32[c_reg]);             \
    break;                                                             \
  }

#define IREE_VM_ISA_EMIT_OP_EXT_F64_UNARY_F64(op_name, op_mnemonic)   \
  IREE_VM_ISA_EMIT_OP(EXT_F64, op_name) {                             \
    IREE_VM_ISA_DECODE_OPERAND_F64(operand_reg);                      \
    IREE_VM_ISA_DECODE_RESULT_F64(result_reg);                        \
    IREE_VM_ISA_EMIT_F64_REG_NAME(result_reg);                        \
    IREE_RETURN_IF_ERROR(                                             \
        iree_string_builder_append_format(b, " = %s ", op_mnemonic)); \
    IREE_VM_ISA_EMIT_F64_REG_NAME(operand_reg);                       \
    IREE_VM_ISA_EMIT_OPTIONAL_VALUE_F64(regs->i32[operand_reg]);      \
    break;                                                            \
  }

#define IREE_VM_ISA_EMIT_OP_EXT_F64_BINARY_F64(op_name, op_mnemonic)   \
  IREE_VM_ISA_EMIT_OP(EXT_F64, op_name) {                              \
    IREE_VM_ISA_DECODE_OPERAND_F64(lhs_reg);                           \
    IREE_VM_ISA_DECODE_OPERAND_F64(rhs_reg);                           \
    IREE_VM_ISA_DECODE_RESULT_F64(result_reg);                         \
    IREE_VM_ISA_EMIT_F64_REG_NAME(result_reg);                         \
    IREE_RETURN_IF_ERROR(                                              \
        iree_string_builder_append_format(b, " = %s ", op_mnemonic));  \
    IREE_VM_ISA_EMIT_F64_REG_NAME(lhs_reg);                            \
    IREE_VM_ISA_EMIT_OPTIONAL_VALUE_F64(regs->i32[lhs_reg]);           \
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", ")); \
    IREE_VM_ISA_EMIT_F64_REG_NAME(rhs_reg);                            \
    IREE_VM_ISA_EMIT_OPTIONAL_VALUE_F64(regs->i32[rhs_reg]);           \
    break;                                                             \
  }

#define IREE_VM_ISA_EMIT_OP_EXT_F64_TERNARY_F64(op_name, op_mnemonic)  \
  IREE_VM_ISA_EMIT_OP(EXT_F64, op_name) {                              \
    IREE_VM_ISA_DECODE_OPERAND_F64(a_reg);                             \
    IREE_VM_ISA_DECODE_OPERAND_F64(b_reg);                             \
    IREE_VM_ISA_DECODE_OPERAND_F64(c_reg);                             \
    IREE_VM_ISA_DECODE_RESULT_F64(result_reg);                         \
    IREE_VM_ISA_EMIT_F64_REG_NAME(result_reg);                         \
    IREE_RETURN_IF_ERROR(                                              \
        iree_string_builder_append_format(b, " = %s ", op_mnemonic));  \
    IREE_VM_ISA_EMIT_F64_REG_NAME(a_reg);                              \
    IREE_VM_ISA_EMIT_OPTIONAL_VALUE_F64(regs->i32[a_reg]);             \
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", ")); \
    IREE_VM_ISA_EMIT_F64_REG_NAME(b_reg);                              \
    IREE_VM_ISA_EMIT_OPTIONAL_VALUE_F64(regs->i32[b_reg]);             \
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", ")); \
    IREE_VM_ISA_EMIT_F64_REG_NAME(c_reg);                              \
    IREE_VM_ISA_EMIT_OPTIONAL_VALUE_F64(regs->i32[c_reg]);             \
    break;                                                             \
  }

// Prints the function name (`foo` for `@foo` in MLIR) or an equivalent.
// The |function_ordinal| is either an import or internal function reference
// within |module|. Resolve status of an import will only be available if
// |module_state| is not NULL.
static iree_status_t iree_vm_bytecode_disassembler_print_function_name(
    iree_vm_bytecode_module_t* module,
    iree_vm_bytecode_module_state_t* module_state, uint32_t function_ordinal,
    iree_string_builder_t* b) {
  const bool is_import =
      iree_vm_isa_function_ordinal_is_import(function_ordinal);
  if (!is_import) {
    iree_vm_function_t function = {
        .module = &module->interface,
        .linkage = IREE_VM_FUNCTION_LINKAGE_INTERNAL,
        .ordinal = function_ordinal,
    };
    iree_string_view_t module_name = iree_vm_module_name(function.module);
    iree_string_view_t func_name = iree_vm_function_name(&function);
    if (iree_string_view_is_empty(func_name)) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          b, "%.*s:%u", (int)module_name.size, module_name.data,
          function.ordinal));
    } else {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          b, "%.*s.%.*s", (int)module_name.size, module_name.data,
          (int)func_name.size, func_name.data));
    }
    return iree_ok_status();
  }

  iree_vm_function_t import = {
      .module = &module->interface,
      .linkage = IREE_VM_FUNCTION_LINKAGE_IMPORT_OPTIONAL,
      .ordinal = iree_vm_isa_function_ordinal_as_import(function_ordinal),
  };
  iree_string_view_t import_name = iree_vm_function_name(&import);
  if (iree_string_view_is_empty(import_name)) {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_format(b, "import:%u", (int)import.ordinal));
  } else {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        b, "%.*s", (int)import_name.size, import_name.data));
  }

  return iree_ok_status();
}

// Internal implementation that also returns the next PC.
static iree_status_t iree_vm_bytecode_disassemble_op_impl(
    iree_vm_bytecode_module_t* module,
    iree_vm_bytecode_module_state_t* module_state, uint16_t function_ordinal,
    iree_vm_source_offset_t pc, const iree_vm_registers_t* regs,
    iree_vm_bytecode_disassembly_format_t format, iree_string_builder_t* b,
    iree_vm_source_offset_t* out_next_pc) {
  const uint8_t* IREE_RESTRICT bytecode_data =
      module->bytecode_data.data +
      module->function_descriptor_table[function_ordinal].bytecode_offset;
  iree_vm_source_offset_t start_pc = pc;
  (void)start_pc;  // May be unused.

  switch (bytecode_data[pc++]) {
    //===------------------------------------------------------------------===//
    // Globals
    //===------------------------------------------------------------------===//

    IREE_VM_ISA_EMIT_OP(CORE, GlobalLoadI32) {
      IREE_VM_ISA_DECODE_GLOBAL_ATTR(byte_offset);
      IREE_VM_ISA_DECODE_RESULT_I32(value_reg);
      IREE_VM_ISA_EMIT_I32_REG_NAME(value_reg);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          b, " = vm.global.load.i32 .rwdata[%u]", byte_offset));
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I32(
          vm_global_load_i32(module_state->rwdata_storage.data, byte_offset));
      break;
    }

    IREE_VM_ISA_EMIT_OP(CORE, GlobalStoreI32) {
      IREE_VM_ISA_DECODE_GLOBAL_ATTR(byte_offset);
      IREE_VM_ISA_DECODE_OPERAND_I32(value_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_format(b, "vm.global.store.i32 "));
      IREE_VM_ISA_EMIT_I32_REG_NAME(value_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I32(regs->i32[value_reg]);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_format(b, ", .rwdata[%u]", byte_offset));
      break;
    }

    IREE_VM_ISA_EMIT_OP(CORE, GlobalLoadIndirectI32) {
      IREE_VM_ISA_DECODE_OPERAND_I32(byte_offset_reg);
      IREE_VM_ISA_DECODE_RESULT_I32(value_reg);
      IREE_VM_ISA_EMIT_I32_REG_NAME(value_reg);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
          b, " = vm.global.load.indirect.i32 .rwdata["));
      IREE_VM_ISA_EMIT_I32_REG_NAME(byte_offset_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I32(regs->i32[byte_offset_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, "]"));
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I32(vm_global_load_i32(
          module_state->rwdata_storage.data, regs->i32[byte_offset_reg]));
      break;
    }

    IREE_VM_ISA_EMIT_OP(CORE, GlobalStoreIndirectI32) {
      IREE_VM_ISA_DECODE_OPERAND_I32(byte_offset_reg);
      IREE_VM_ISA_DECODE_OPERAND_I32(value_reg);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
          b, "vm.global.store.indirect.i32 "));
      IREE_VM_ISA_EMIT_I32_REG_NAME(value_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I32(regs->i32[value_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", .rwdata["));
      IREE_VM_ISA_EMIT_I32_REG_NAME(byte_offset_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I32(regs->i32[byte_offset_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, "]"));
      break;
    }

    IREE_VM_ISA_EMIT_OP(CORE, GlobalLoadI64) {
      IREE_VM_ISA_DECODE_GLOBAL_ATTR(byte_offset);
      IREE_VM_ISA_DECODE_RESULT_I64(value_reg);
      IREE_VM_ISA_EMIT_I32_REG_NAME(value_reg);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          b, " = vm.global.load.i64 .rwdata[%u]", byte_offset));
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I64(
          module_state->rwdata_storage.data[byte_offset]);
      break;
    }

    IREE_VM_ISA_EMIT_OP(CORE, GlobalStoreI64) {
      IREE_VM_ISA_DECODE_GLOBAL_ATTR(byte_offset);
      IREE_VM_ISA_DECODE_OPERAND_I64(value_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_format(b, "vm.global.store.i64 "));
      IREE_VM_ISA_EMIT_I64_REG_NAME(value_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I64(regs->i32[value_reg]);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_format(b, ", .rwdata[%u]", byte_offset));
      break;
    }

    IREE_VM_ISA_EMIT_OP(CORE, GlobalLoadIndirectI64) {
      IREE_VM_ISA_DECODE_OPERAND_I32(byte_offset_reg);
      IREE_VM_ISA_DECODE_RESULT_I64(value_reg);
      IREE_VM_ISA_EMIT_I64_REG_NAME(value_reg);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
          b, " = vm.global.load.indirect.i64 .rwdata["));
      IREE_VM_ISA_EMIT_I32_REG_NAME(byte_offset_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I32(regs->i32[byte_offset_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, "]"));
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I64(
          module_state->rwdata_storage.data[regs->i32[byte_offset_reg]]);
      break;
    }

    IREE_VM_ISA_EMIT_OP(CORE, GlobalStoreIndirectI64) {
      IREE_VM_ISA_DECODE_OPERAND_I32(byte_offset_reg);
      IREE_VM_ISA_DECODE_OPERAND_I64(value_reg);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
          b, "vm.global.store.indirect.i64 "));
      IREE_VM_ISA_EMIT_I64_REG_NAME(value_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I64(regs->i32[value_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", .rwdata["));
      IREE_VM_ISA_EMIT_I32_REG_NAME(byte_offset_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I32(regs->i32[byte_offset_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, "]"));
      break;
    }

    IREE_VM_ISA_EMIT_OP(CORE, GlobalLoadRef) {
      IREE_VM_ISA_DECODE_GLOBAL_ATTR(global);
      IREE_VM_ISA_DECODE_TYPE_OF(type_def);
      IREE_VM_ISA_DECODE_RESULT_REF_MOVE(value_reg);
      IREE_VM_ISA_EMIT_REF_REG_NAME_MOVE(value_reg, value_reg_is_move);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          b, " = vm.global.load.ref .refs[%u]", global));
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_REF(
          &module_state->global_ref_table[global]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, " : !"));
      IREE_VM_ISA_EMIT_TYPE_NAME(type_def);
      break;
    }

    IREE_VM_ISA_EMIT_OP(CORE, GlobalStoreRef) {
      IREE_VM_ISA_DECODE_GLOBAL_ATTR(global);
      IREE_VM_ISA_DECODE_TYPE_OF(type_def);
      IREE_VM_ISA_DECODE_OPERAND_REF_MOVE(value_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, "vm.global.store.ref "));
      IREE_VM_ISA_EMIT_REF_REG_NAME_MOVE(value_reg, value_reg_is_move);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_REF(&regs->ref[value_reg]);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_format(b, ", .refs[%u] : !", global));
      IREE_VM_ISA_EMIT_TYPE_NAME(type_def);
      break;
    }

    IREE_VM_ISA_EMIT_OP(CORE, GlobalLoadIndirectRef) {
      IREE_VM_ISA_DECODE_OPERAND_I32(global_reg);
      IREE_VM_ISA_DECODE_TYPE_OF(type_def);
      IREE_VM_ISA_DECODE_RESULT_REF_MOVE(value_reg);
      IREE_VM_ISA_EMIT_REF_REG_NAME_MOVE(value_reg, value_reg_is_move);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
          b, " = vm.global.load.indirect.ref .refs["));
      IREE_VM_ISA_EMIT_I32_REG_NAME(global_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I32(regs->i32[global_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, "]"));
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_REF(
          &module_state->global_ref_table[regs->i32[global_reg]]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, " : !"));
      IREE_VM_ISA_EMIT_TYPE_NAME(type_def);
      break;
    }

    IREE_VM_ISA_EMIT_OP(CORE, GlobalStoreIndirectRef) {
      IREE_VM_ISA_DECODE_OPERAND_I32(global_reg);
      IREE_VM_ISA_DECODE_TYPE_OF(type_def);
      IREE_VM_ISA_DECODE_OPERAND_REF_MOVE(value_reg);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          b, "vm.global.store.indirect.ref "));
      IREE_VM_ISA_EMIT_REF_REG_NAME_MOVE(value_reg, value_reg_is_move);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_REF(&regs->ref[value_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(b, ", .refs["));
      IREE_VM_ISA_EMIT_I32_REG_NAME(global_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I32(regs->i32[global_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(b, "] : !"));
      IREE_VM_ISA_EMIT_TYPE_NAME(type_def);
      break;
    }

    //===------------------------------------------------------------------===//
    // Constants
    //===------------------------------------------------------------------===//

    IREE_VM_ISA_EMIT_OP(CORE, ConstI32) {
      IREE_VM_ISA_DECODE_ATTR_I32(value);
      IREE_VM_ISA_DECODE_RESULT_I32(result_reg);
      IREE_VM_ISA_EMIT_I32_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          b, " = vm.const.i32 %d  // 0x%08X", value, value));
      break;
    }

    IREE_VM_ISA_EMIT_OP(CORE, ConstI32Zero) {
      IREE_VM_ISA_DECODE_RESULT_I32(result_reg);
      IREE_VM_ISA_EMIT_I32_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.const.i32.zero"));
      break;
    }

    IREE_VM_ISA_EMIT_OP(CORE, ConstI64) {
      IREE_VM_ISA_DECODE_ATTR_I64(value);
      IREE_VM_ISA_DECODE_RESULT_I64(result_reg);
      IREE_VM_ISA_EMIT_I64_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          b, " = vm.const.i64 %" PRId64 "  // 0x%016" PRIX64 "", value, value));
      break;
    }

    IREE_VM_ISA_EMIT_OP(CORE, ConstI64Zero) {
      IREE_VM_ISA_DECODE_RESULT_I64(result_reg);
      IREE_VM_ISA_EMIT_I64_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.const.i64.zero"));
      break;
    }

    IREE_VM_ISA_EMIT_OP(CORE, ConstRefZero) {
      IREE_VM_ISA_DECODE_RESULT_REF(result_reg);
      IREE_VM_ISA_EMIT_REF_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.const.ref.zero"));
      break;
    }

    IREE_VM_ISA_EMIT_OP(CORE, DiscardRefs) {
      IREE_VM_ISA_DECODE_VARIADIC_OPERANDS(reg_list);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, "vm.discard.refs "));
      IREE_VM_ISA_EMIT_OPERAND_REG_LIST(reg_list);
      break;
    }

    IREE_VM_ISA_EMIT_OP(CORE, AssignRef) {
      IREE_VM_ISA_DECODE_OPERAND_REF_MOVE(source_reg);
      IREE_VM_ISA_DECODE_RESULT_REF_MOVE(result_reg);
      IREE_VM_ISA_EMIT_REF_REG_NAME_MOVE(result_reg, result_reg_is_move);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.assign.ref "));
      IREE_VM_ISA_EMIT_REF_REG_NAME_MOVE(source_reg, source_reg_is_move);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_REF(&regs->ref[source_reg]);
      break;
    }

    IREE_VM_ISA_EMIT_OP(CORE, ConstRefRodata) {
      IREE_VM_ISA_DECODE_RODATA_ATTR(rodata_ordinal);
      IREE_VM_ISA_DECODE_RESULT_REF_MOVE(value_reg);
      iree_vm_buffer_t* buffer = &module->rodata_ref_table[rodata_ordinal];
      IREE_VM_ISA_EMIT_REF_REG_NAME_MOVE(value_reg, value_reg_is_move);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          b, " = vm.const.ref.rodata %u  // %p %" PRIhsz "b", rodata_ordinal,
          buffer->data.data, buffer->data.data_length));
      break;
    }

    //===------------------------------------------------------------------===//
    // Buffers
    //===------------------------------------------------------------------===//

    IREE_VM_ISA_EMIT_OP(CORE, BufferAlloc) {
      IREE_VM_ISA_DECODE_OPERAND_I64(length_reg);
      IREE_VM_ISA_DECODE_OPERAND_I32(alignment_reg);
      IREE_VM_ISA_DECODE_RESULT_REF_MOVE(result_reg);
      IREE_VM_ISA_EMIT_REF_REG_NAME_MOVE(result_reg, result_reg_is_move);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.buffer.alloc "));
      IREE_VM_ISA_EMIT_I64_REG_NAME(length_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I64(regs->i32[length_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      IREE_VM_ISA_EMIT_I32_REG_NAME(alignment_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I32(regs->i32[alignment_reg]);
      break;
    }

    IREE_VM_ISA_EMIT_OP(CORE, BufferClone) {
      IREE_VM_ISA_DECODE_OPERAND_REF(source_reg);
      IREE_VM_ISA_DECODE_OPERAND_I64(offset_reg);
      IREE_VM_ISA_DECODE_OPERAND_I64(length_reg);
      IREE_VM_ISA_DECODE_OPERAND_I32(alignment_reg);
      IREE_VM_ISA_DECODE_RESULT_REF_MOVE(result_reg);
      IREE_VM_ISA_EMIT_REF_REG_NAME_MOVE(result_reg, result_reg_is_move);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.buffer.clone "));
      IREE_VM_ISA_EMIT_REF_REG_NAME(source_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_REF(&regs->ref[source_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      IREE_VM_ISA_EMIT_I64_REG_NAME(offset_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I64(regs->i32[offset_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      IREE_VM_ISA_EMIT_I64_REG_NAME(length_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I64(regs->i32[length_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      IREE_VM_ISA_EMIT_I32_REG_NAME(alignment_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I32(regs->i32[alignment_reg]);
      break;
    }

    IREE_VM_ISA_EMIT_OP(CORE, BufferLength) {
      IREE_VM_ISA_DECODE_OPERAND_REF(buffer_reg);
      IREE_VM_ISA_DECODE_RESULT_I64(result_reg);
      IREE_VM_ISA_EMIT_I64_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.buffer.length "));
      IREE_VM_ISA_EMIT_REF_REG_NAME(buffer_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_REF(&regs->ref[buffer_reg]);
      break;
    }

    IREE_VM_ISA_EMIT_OP(CORE, BufferCopy) {
      IREE_VM_ISA_DECODE_OPERAND_REF(source_buffer_reg);
      IREE_VM_ISA_DECODE_OPERAND_I64(source_offset_reg);
      IREE_VM_ISA_DECODE_OPERAND_REF(target_buffer_reg);
      IREE_VM_ISA_DECODE_OPERAND_I64(target_offset_reg);
      IREE_VM_ISA_DECODE_OPERAND_I64(length_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, "vm.buffer.copy "));
      IREE_VM_ISA_EMIT_REF_REG_NAME(source_buffer_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_REF(&regs->ref[source_buffer_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      IREE_VM_ISA_EMIT_I64_REG_NAME(source_offset_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I64(regs->i32[source_offset_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      IREE_VM_ISA_EMIT_REF_REG_NAME(target_buffer_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_REF(&regs->ref[target_buffer_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      IREE_VM_ISA_EMIT_I64_REG_NAME(target_offset_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I64(regs->i32[target_offset_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      IREE_VM_ISA_EMIT_I64_REG_NAME(length_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I64(regs->i32[length_reg]);
      break;
    }

    IREE_VM_ISA_EMIT_OP(CORE, BufferCompare) {
      IREE_VM_ISA_DECODE_OPERAND_REF(lhs_buffer_reg);
      IREE_VM_ISA_DECODE_OPERAND_I64(lhs_offset_reg);
      IREE_VM_ISA_DECODE_OPERAND_REF(rhs_buffer_reg);
      IREE_VM_ISA_DECODE_OPERAND_I64(rhs_offset_reg);
      IREE_VM_ISA_DECODE_OPERAND_I64(length_reg);
      IREE_VM_ISA_DECODE_RESULT_I32(result_reg);
      IREE_VM_ISA_EMIT_I32_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.buffer.compare "));
      IREE_VM_ISA_EMIT_REF_REG_NAME(lhs_buffer_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_REF(&regs->ref[lhs_buffer_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      IREE_VM_ISA_EMIT_I64_REG_NAME(lhs_offset_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I64(regs->i32[lhs_offset_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      IREE_VM_ISA_EMIT_REF_REG_NAME(rhs_buffer_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_REF(&regs->ref[rhs_buffer_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      IREE_VM_ISA_EMIT_I64_REG_NAME(rhs_offset_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I64(regs->i32[rhs_offset_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      IREE_VM_ISA_EMIT_I64_REG_NAME(length_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I64(regs->i32[length_reg]);
      break;
    }

    IREE_VM_ISA_EMIT_OP(CORE, BufferFillI8) {
      IREE_VM_ISA_DECODE_OPERAND_REF(buffer_reg);
      IREE_VM_ISA_DECODE_OPERAND_I64(offset_reg);
      IREE_VM_ISA_DECODE_OPERAND_I64(length_reg);
      IREE_VM_ISA_DECODE_OPERAND_I32(value_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, "vm.buffer.fill.i8 "));
      IREE_VM_ISA_EMIT_REF_REG_NAME(buffer_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_REF(&regs->ref[buffer_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      IREE_VM_ISA_EMIT_I64_REG_NAME(offset_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I64(regs->i32[offset_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      IREE_VM_ISA_EMIT_I64_REG_NAME(length_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I64(regs->i32[length_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      IREE_VM_ISA_EMIT_I32_REG_NAME(value_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I32((uint8_t)regs->i32[value_reg]);
      break;
    }
    IREE_VM_ISA_EMIT_OP(CORE, BufferFillI16) {
      IREE_VM_ISA_DECODE_OPERAND_REF(buffer_reg);
      IREE_VM_ISA_DECODE_OPERAND_I64(offset_reg);
      IREE_VM_ISA_DECODE_OPERAND_I64(length_reg);
      IREE_VM_ISA_DECODE_OPERAND_I32(value_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, "vm.buffer.fill.i16 "));
      IREE_VM_ISA_EMIT_REF_REG_NAME(buffer_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_REF(&regs->ref[buffer_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      IREE_VM_ISA_EMIT_I64_REG_NAME(offset_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I64(regs->i32[offset_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      IREE_VM_ISA_EMIT_I64_REG_NAME(length_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I64(regs->i32[length_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      IREE_VM_ISA_EMIT_I32_REG_NAME(value_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I32((uint16_t)regs->i32[value_reg]);
      break;
    }
    IREE_VM_ISA_EMIT_OP(CORE, BufferFillI32) {
      IREE_VM_ISA_DECODE_OPERAND_REF(buffer_reg);
      IREE_VM_ISA_DECODE_OPERAND_I64(offset_reg);
      IREE_VM_ISA_DECODE_OPERAND_I64(length_reg);
      IREE_VM_ISA_DECODE_OPERAND_I32(value_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, "vm.buffer.fill.i32 "));
      IREE_VM_ISA_EMIT_REF_REG_NAME(buffer_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_REF(&regs->ref[buffer_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      IREE_VM_ISA_EMIT_I64_REG_NAME(offset_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I64(regs->i32[offset_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      IREE_VM_ISA_EMIT_I64_REG_NAME(length_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I64(regs->i32[length_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      IREE_VM_ISA_EMIT_I32_REG_NAME(value_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I32(regs->i32[value_reg]);
      break;
    }

    IREE_VM_ISA_EMIT_OP(CORE, BufferFillI64) {
      IREE_VM_ISA_DECODE_OPERAND_REF(buffer_reg);
      IREE_VM_ISA_DECODE_OPERAND_I64(offset_reg);
      IREE_VM_ISA_DECODE_OPERAND_I64(length_reg);
      IREE_VM_ISA_DECODE_OPERAND_I64(value_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, "vm.buffer.fill.i64 "));
      IREE_VM_ISA_EMIT_REF_REG_NAME(buffer_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_REF(&regs->ref[buffer_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      IREE_VM_ISA_EMIT_I64_REG_NAME(offset_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I64(regs->i32[offset_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      IREE_VM_ISA_EMIT_I64_REG_NAME(length_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I64(regs->i32[length_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      IREE_VM_ISA_EMIT_I64_REG_NAME(value_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I64(regs->i32[value_reg]);
      break;
    }

    IREE_VM_ISA_EMIT_OP(CORE, BufferLoadI8U) {
      IREE_VM_ISA_DECODE_OPERAND_REF(buffer_reg);
      IREE_VM_ISA_DECODE_OPERAND_I64(offset_reg);
      IREE_VM_ISA_DECODE_RESULT_I32(result_reg);
      IREE_VM_ISA_EMIT_I32_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.buffer.load.i8.u "));
      IREE_VM_ISA_EMIT_REF_REG_NAME(buffer_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_REF(&regs->ref[buffer_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      IREE_VM_ISA_EMIT_I64_REG_NAME(offset_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I64(regs->i32[offset_reg]);
      break;
    }
    IREE_VM_ISA_EMIT_OP(CORE, BufferLoadI8S) {
      IREE_VM_ISA_DECODE_OPERAND_REF(buffer_reg);
      IREE_VM_ISA_DECODE_OPERAND_I64(offset_reg);
      IREE_VM_ISA_DECODE_RESULT_I32(result_reg);
      IREE_VM_ISA_EMIT_I32_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.buffer.load.i8.s "));
      IREE_VM_ISA_EMIT_REF_REG_NAME(buffer_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_REF(&regs->ref[buffer_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      IREE_VM_ISA_EMIT_I64_REG_NAME(offset_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I64(regs->i32[offset_reg]);
      break;
    }
    IREE_VM_ISA_EMIT_OP(CORE, BufferLoadI16U) {
      IREE_VM_ISA_DECODE_OPERAND_REF(buffer_reg);
      IREE_VM_ISA_DECODE_OPERAND_I64(offset_reg);
      IREE_VM_ISA_DECODE_RESULT_I32(result_reg);
      IREE_VM_ISA_EMIT_I32_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.buffer.load.i16.u "));
      IREE_VM_ISA_EMIT_REF_REG_NAME(buffer_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_REF(&regs->ref[buffer_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      IREE_VM_ISA_EMIT_I64_REG_NAME(offset_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I64(regs->i32[offset_reg]);
      break;
    }
    IREE_VM_ISA_EMIT_OP(CORE, BufferLoadI16S) {
      IREE_VM_ISA_DECODE_OPERAND_REF(buffer_reg);
      IREE_VM_ISA_DECODE_OPERAND_I64(offset_reg);
      IREE_VM_ISA_DECODE_RESULT_I32(result_reg);
      IREE_VM_ISA_EMIT_I32_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.buffer.load.i16.s "));
      IREE_VM_ISA_EMIT_REF_REG_NAME(buffer_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_REF(&regs->ref[buffer_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      IREE_VM_ISA_EMIT_I64_REG_NAME(offset_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I64(regs->i32[offset_reg]);
      break;
    }
    IREE_VM_ISA_EMIT_OP(CORE, BufferLoadI32) {
      IREE_VM_ISA_DECODE_OPERAND_REF(buffer_reg);
      IREE_VM_ISA_DECODE_OPERAND_I64(offset_reg);
      IREE_VM_ISA_DECODE_RESULT_I32(result_reg);
      IREE_VM_ISA_EMIT_I32_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.buffer.load.i32 "));
      IREE_VM_ISA_EMIT_REF_REG_NAME(buffer_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_REF(&regs->ref[buffer_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      IREE_VM_ISA_EMIT_I64_REG_NAME(offset_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I64(regs->i32[offset_reg]);
      break;
    }
    IREE_VM_ISA_EMIT_OP(CORE, BufferLoadI64) {
      IREE_VM_ISA_DECODE_OPERAND_REF(buffer_reg);
      IREE_VM_ISA_DECODE_OPERAND_I64(offset_reg);
      IREE_VM_ISA_DECODE_RESULT_I64(result_reg);
      IREE_VM_ISA_EMIT_I64_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.buffer.load.i64 "));
      IREE_VM_ISA_EMIT_REF_REG_NAME(buffer_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_REF(&regs->ref[buffer_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      IREE_VM_ISA_EMIT_I64_REG_NAME(offset_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I64(regs->i32[offset_reg]);
      break;
    }

    IREE_VM_ISA_EMIT_OP(CORE, BufferStoreI8) {
      IREE_VM_ISA_DECODE_OPERAND_REF(buffer_reg);
      IREE_VM_ISA_DECODE_OPERAND_I64(offset_reg);
      IREE_VM_ISA_DECODE_OPERAND_I32(value_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, "vm.buffer.store.i8 "));
      IREE_VM_ISA_EMIT_I32_REG_NAME(value_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I32((uint8_t)regs->i32[value_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      IREE_VM_ISA_EMIT_REF_REG_NAME(buffer_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_REF(&regs->ref[buffer_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      IREE_VM_ISA_EMIT_I64_REG_NAME(offset_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I64(regs->i32[offset_reg]);
      break;
    }
    IREE_VM_ISA_EMIT_OP(CORE, BufferStoreI16) {
      IREE_VM_ISA_DECODE_OPERAND_REF(buffer_reg);
      IREE_VM_ISA_DECODE_OPERAND_I64(offset_reg);
      IREE_VM_ISA_DECODE_OPERAND_I32(value_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, "vm.buffer.store.i16 "));
      IREE_VM_ISA_EMIT_I32_REG_NAME(value_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I32((uint16_t)regs->i32[value_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      IREE_VM_ISA_EMIT_REF_REG_NAME(buffer_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_REF(&regs->ref[buffer_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      IREE_VM_ISA_EMIT_I64_REG_NAME(offset_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I64(regs->i32[offset_reg]);
      break;
    }
    IREE_VM_ISA_EMIT_OP(CORE, BufferStoreI32) {
      IREE_VM_ISA_DECODE_OPERAND_REF(buffer_reg);
      IREE_VM_ISA_DECODE_OPERAND_I64(offset_reg);
      IREE_VM_ISA_DECODE_OPERAND_I32(value_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, "vm.buffer.store.i32 "));
      IREE_VM_ISA_EMIT_I32_REG_NAME(value_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I32(regs->i32[value_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      IREE_VM_ISA_EMIT_REF_REG_NAME(buffer_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_REF(&regs->ref[buffer_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      IREE_VM_ISA_EMIT_I64_REG_NAME(offset_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I64(regs->i32[offset_reg]);
      break;
    }
    IREE_VM_ISA_EMIT_OP(CORE, BufferStoreI64) {
      IREE_VM_ISA_DECODE_OPERAND_REF(buffer_reg);
      IREE_VM_ISA_DECODE_OPERAND_I64(offset_reg);
      IREE_VM_ISA_DECODE_OPERAND_I64(value_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, "vm.buffer.store.i64 "));
      IREE_VM_ISA_EMIT_I64_REG_NAME(value_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I64(regs->i32[value_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      IREE_VM_ISA_EMIT_REF_REG_NAME(buffer_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_REF(&regs->ref[buffer_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      IREE_VM_ISA_EMIT_I64_REG_NAME(offset_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I64(regs->i32[offset_reg]);
      break;
    }

    IREE_VM_ISA_EMIT_OP(CORE, BufferHash) {
      IREE_VM_ISA_DECODE_OPERAND_REF(buffer_reg);
      IREE_VM_ISA_DECODE_OPERAND_I64(offset_reg);
      IREE_VM_ISA_DECODE_OPERAND_I64(length_reg);
      IREE_VM_ISA_DECODE_RESULT_I64(result_reg);
      IREE_VM_ISA_EMIT_I32_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.buffer.hash "));
      IREE_VM_ISA_EMIT_REF_REG_NAME(buffer_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_REF(&regs->ref[buffer_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      IREE_VM_ISA_EMIT_I64_REG_NAME(offset_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I64(regs->i32[offset_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      IREE_VM_ISA_EMIT_I64_REG_NAME(length_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I64(regs->i32[length_reg]);
      break;
    }

    //===------------------------------------------------------------------===//
    // Lists
    //===------------------------------------------------------------------===//

    IREE_VM_ISA_EMIT_OP(CORE, ListAlloc) {
      IREE_VM_ISA_DECODE_TYPE_OF(element_type_def);
      IREE_VM_ISA_DECODE_OPERAND_I32(initial_capacity_reg);
      IREE_VM_ISA_DECODE_RESULT_REF_MOVE(result_reg);
      IREE_VM_ISA_EMIT_REF_REG_NAME_MOVE(result_reg, result_reg_is_move);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.list.alloc "));
      IREE_VM_ISA_EMIT_I32_REG_NAME(initial_capacity_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I32(regs->i32[initial_capacity_reg]);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " : !vm.list<"));
      IREE_VM_ISA_EMIT_TYPE_NAME(element_type_def);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ">"));
      break;
    }

    IREE_VM_ISA_EMIT_OP(CORE, ListReserve) {
      IREE_VM_ISA_DECODE_OPERAND_REF(list_reg);
      IREE_VM_ISA_DECODE_OPERAND_I32(minimum_capacity_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, "vm.list.reserve "));
      IREE_VM_ISA_EMIT_REF_REG_NAME(list_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_REF(&regs->ref[list_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      IREE_VM_ISA_EMIT_I32_REG_NAME(minimum_capacity_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I32(regs->i32[minimum_capacity_reg]);
      break;
    }

    IREE_VM_ISA_EMIT_OP(CORE, ListSize) {
      IREE_VM_ISA_DECODE_OPERAND_REF(list_reg);
      IREE_VM_ISA_DECODE_RESULT_I32(result_reg);
      IREE_VM_ISA_EMIT_I32_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.list.size "));
      IREE_VM_ISA_EMIT_REF_REG_NAME(list_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_REF(&regs->ref[list_reg]);
      break;
    }

    IREE_VM_ISA_EMIT_OP(CORE, ListResize) {
      IREE_VM_ISA_DECODE_OPERAND_REF(list_reg);
      IREE_VM_ISA_DECODE_OPERAND_I32(new_size_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, "vm.list.resize "));
      IREE_VM_ISA_EMIT_REF_REG_NAME(list_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_REF(&regs->ref[list_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      IREE_VM_ISA_EMIT_I32_REG_NAME(new_size_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I32(regs->i32[new_size_reg]);
      break;
    }

    IREE_VM_ISA_EMIT_OP(CORE, ListGetI32) {
      IREE_VM_ISA_DECODE_OPERAND_REF(list_reg);
      IREE_VM_ISA_DECODE_OPERAND_I32(index_reg);
      IREE_VM_ISA_DECODE_RESULT_I32(result_reg);
      IREE_VM_ISA_EMIT_I32_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.list.get.i32 "));
      IREE_VM_ISA_EMIT_REF_REG_NAME(list_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_REF(&regs->ref[list_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      IREE_VM_ISA_EMIT_I32_REG_NAME(index_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I32(regs->i32[index_reg]);
      break;
    }

    IREE_VM_ISA_EMIT_OP(CORE, ListSetI32) {
      IREE_VM_ISA_DECODE_OPERAND_REF(list_reg);
      IREE_VM_ISA_DECODE_OPERAND_I32(index_reg);
      IREE_VM_ISA_DECODE_OPERAND_I32(raw_value_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, "vm.list.set.i32 "));
      IREE_VM_ISA_EMIT_REF_REG_NAME(list_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_REF(&regs->ref[list_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      IREE_VM_ISA_EMIT_I32_REG_NAME(index_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I32(regs->i32[index_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      IREE_VM_ISA_EMIT_I32_REG_NAME(raw_value_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I32(regs->i32[raw_value_reg]);
      break;
    }

    IREE_VM_ISA_EMIT_OP(CORE, ListGetI64) {
      IREE_VM_ISA_DECODE_OPERAND_REF(list_reg);
      IREE_VM_ISA_DECODE_OPERAND_I32(index_reg);
      IREE_VM_ISA_DECODE_RESULT_I64(result_reg);
      IREE_VM_ISA_EMIT_I64_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.list.get.i64 "));
      IREE_VM_ISA_EMIT_REF_REG_NAME(list_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_REF(&regs->ref[list_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      IREE_VM_ISA_EMIT_I32_REG_NAME(index_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I32(regs->i32[index_reg]);
      break;
    }

    IREE_VM_ISA_EMIT_OP(CORE, ListSetI64) {
      IREE_VM_ISA_DECODE_OPERAND_REF(list_reg);
      IREE_VM_ISA_DECODE_OPERAND_I32(index_reg);
      IREE_VM_ISA_DECODE_OPERAND_I64(value_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, "vm.list.set.i64 "));
      IREE_VM_ISA_EMIT_REF_REG_NAME(list_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_REF(&regs->ref[list_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      IREE_VM_ISA_EMIT_I32_REG_NAME(index_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I32(regs->i32[index_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      IREE_VM_ISA_EMIT_I64_REG_NAME(value_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I64(regs->i32[value_reg]);
      break;
    }

    IREE_VM_ISA_EMIT_OP(CORE, ListGetRef) {
      IREE_VM_ISA_DECODE_OPERAND_REF(list_reg);
      IREE_VM_ISA_DECODE_OPERAND_I32(index_reg);
      IREE_VM_ISA_DECODE_TYPE_OF(type_def);
      IREE_VM_ISA_DECODE_RESULT_REF_MOVE(result_reg);
      IREE_VM_ISA_EMIT_REF_REG_NAME_MOVE(result_reg, result_reg_is_move);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.list.get.ref "));
      IREE_VM_ISA_EMIT_REF_REG_NAME(list_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_REF(&regs->ref[list_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      IREE_VM_ISA_EMIT_I32_REG_NAME(index_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I32(regs->i32[index_reg]);
      IREE_VM_ISA_EMIT_TYPE_NAME(type_def);
      break;
    }

    IREE_VM_ISA_EMIT_OP(CORE, ListSetRef) {
      IREE_VM_ISA_DECODE_OPERAND_REF(list_reg);
      IREE_VM_ISA_DECODE_OPERAND_I32(index_reg);
      IREE_VM_ISA_DECODE_OPERAND_REF_MOVE(value_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, "vm.list.set.ref "));
      IREE_VM_ISA_EMIT_REF_REG_NAME(list_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_REF(&regs->ref[list_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      IREE_VM_ISA_EMIT_I32_REG_NAME(index_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I32(regs->i32[index_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      IREE_VM_ISA_EMIT_REF_REG_NAME_MOVE(value_reg, value_reg_is_move);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_REF(&regs->ref[value_reg]);
      break;
    }

    //===------------------------------------------------------------------===//
    // Conditional assignment
    //===------------------------------------------------------------------===//

    IREE_VM_ISA_EMIT_OP(CORE, SelectI32) {
      IREE_VM_ISA_DECODE_OPERAND_I32(condition_reg);
      IREE_VM_ISA_DECODE_OPERAND_I32(true_value_reg);
      IREE_VM_ISA_DECODE_OPERAND_I32(false_value_reg);
      IREE_VM_ISA_DECODE_RESULT_I32(result_reg);
      IREE_VM_ISA_EMIT_I32_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.select.i32 "));
      IREE_VM_ISA_EMIT_I32_REG_NAME(condition_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I32(regs->i32[condition_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, " ? "));
      IREE_VM_ISA_EMIT_I32_REG_NAME(true_value_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I32(regs->i32[true_value_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, " : "));
      IREE_VM_ISA_EMIT_I32_REG_NAME(false_value_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I32(regs->i32[false_value_reg]);
      break;
    }

    IREE_VM_ISA_EMIT_OP(CORE, SelectI64) {
      IREE_VM_ISA_DECODE_OPERAND_I32(condition_reg);
      IREE_VM_ISA_DECODE_OPERAND_I64(true_value_reg);
      IREE_VM_ISA_DECODE_OPERAND_I64(false_value_reg);
      IREE_VM_ISA_DECODE_RESULT_I64(result_reg);
      IREE_VM_ISA_EMIT_I64_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.select.i64 "));
      IREE_VM_ISA_EMIT_I32_REG_NAME(condition_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I32(regs->i32[condition_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, " ? "));
      IREE_VM_ISA_EMIT_I64_REG_NAME(true_value_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I64(regs->i32[true_value_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, " : "));
      IREE_VM_ISA_EMIT_I64_REG_NAME(false_value_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I64(regs->i32[false_value_reg]);
      break;
    }

    IREE_VM_ISA_EMIT_OP(CORE, SelectRef) {
      IREE_VM_ISA_DECODE_OPERAND_I32(condition_reg);
      IREE_VM_ISA_DECODE_TYPE_OF(type_def);
      IREE_VM_ISA_DECODE_OPERAND_REF_MOVE(true_value_reg);
      IREE_VM_ISA_DECODE_OPERAND_REF_MOVE(false_value_reg);
      IREE_VM_ISA_DECODE_RESULT_REF_MOVE(result_reg);
      IREE_VM_ISA_EMIT_REF_REG_NAME_MOVE(result_reg, result_reg_is_move);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.select.ref "));
      IREE_VM_ISA_EMIT_I32_REG_NAME(condition_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I32(regs->i32[condition_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, " ? "));
      IREE_VM_ISA_EMIT_REF_REG_NAME_MOVE(true_value_reg,
                                         true_value_reg_is_move);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_REF(&regs->ref[true_value_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, " : "));
      IREE_VM_ISA_EMIT_REF_REG_NAME_MOVE(false_value_reg,
                                         false_value_reg_is_move);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_REF(&regs->ref[false_value_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, " -> !"));
      IREE_VM_ISA_EMIT_TYPE_NAME(type_def);
      break;
    }

    IREE_VM_ISA_EMIT_OP(CORE, SwitchI32) {
      IREE_VM_ISA_DECODE_OPERAND_I32(index_reg);
      IREE_VM_ISA_DECODE_OPERAND_I32(default_value_reg);
      IREE_VM_ISA_DECODE_VARIADIC_OPERANDS(value_reg_list);
      IREE_VM_ISA_DECODE_RESULT_I32(result_reg);
      IREE_VM_ISA_EMIT_I32_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.switch.i32 "));
      IREE_VM_ISA_EMIT_I32_REG_NAME(index_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I32(regs->i32[index_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, "["));
      IREE_VM_ISA_EMIT_OPERAND_REG_LIST(value_reg_list);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, "] else "));
      IREE_VM_ISA_EMIT_I32_REG_NAME(default_value_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I32(regs->i32[default_value_reg]);
      break;
    }

    IREE_VM_ISA_EMIT_OP(CORE, SwitchI64) {
      IREE_VM_ISA_DECODE_OPERAND_I32(index_reg);
      IREE_VM_ISA_DECODE_OPERAND_I64(default_value_reg);
      IREE_VM_ISA_DECODE_VARIADIC_OPERANDS(value_reg_list);
      IREE_VM_ISA_DECODE_RESULT_I64(result_reg);
      IREE_VM_ISA_EMIT_I64_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.switch.i64 "));
      IREE_VM_ISA_EMIT_I32_REG_NAME(index_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I32(regs->i32[index_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, "["));
      IREE_VM_ISA_EMIT_OPERAND_REG_LIST(value_reg_list);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, "] else "));
      IREE_VM_ISA_EMIT_I64_REG_NAME(default_value_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I64(regs->i32[default_value_reg]);
      break;
    }

    IREE_VM_ISA_EMIT_OP(CORE, SwitchRef) {
      IREE_VM_ISA_DECODE_OPERAND_I32(index_reg);
      IREE_VM_ISA_DECODE_OPERAND_REF_MOVE(default_value_reg);
      IREE_VM_ISA_DECODE_VARIADIC_OPERANDS(value_reg_list);
      IREE_VM_ISA_DECODE_RESULT_REF_MOVE(result_reg);
      IREE_VM_ISA_EMIT_REF_REG_NAME_MOVE(result_reg, result_reg_is_move);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.switch.ref "));
      IREE_VM_ISA_EMIT_I32_REG_NAME(index_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I32(regs->i32[index_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, "["));
      IREE_VM_ISA_EMIT_OPERAND_REG_LIST(value_reg_list);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, "] else "));
      IREE_VM_ISA_EMIT_REF_REG_NAME_MOVE(default_value_reg,
                                         default_value_reg_is_move);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_REF(&regs->ref[default_value_reg]);
      break;
    }

    //===------------------------------------------------------------------===//
    // Native integer arithmetic
    //===------------------------------------------------------------------===//

    IREE_VM_ISA_EMIT_OP_CORE_BINARY_I32(AddI32, "vm.add.i32");
    IREE_VM_ISA_EMIT_OP_CORE_BINARY_I32(SubI32, "vm.sub.i32");
    IREE_VM_ISA_EMIT_OP_CORE_BINARY_I32(MulI32, "vm.mul.i32");
    IREE_VM_ISA_EMIT_OP_CORE_BINARY_I32(DivI32S, "vm.div.i32.s");
    IREE_VM_ISA_EMIT_OP_CORE_BINARY_I32(DivI32U, "vm.div.i32.u");
    IREE_VM_ISA_EMIT_OP_CORE_BINARY_I32(RemI32S, "vm.rem.i32.s");
    IREE_VM_ISA_EMIT_OP_CORE_BINARY_I32(RemI32U, "vm.rem.i32.u");
    IREE_VM_ISA_EMIT_OP_CORE_TERNARY_I32(FMAI32, "vm.fma.i32");
    IREE_VM_ISA_EMIT_OP_CORE_UNARY_I32(AbsI32, "vm.abs.i32");
    IREE_VM_ISA_EMIT_OP_CORE_BINARY_I32(MinI32S, "vm.min.i32.s");
    IREE_VM_ISA_EMIT_OP_CORE_BINARY_I32(MinI32U, "vm.min.i32.u");
    IREE_VM_ISA_EMIT_OP_CORE_BINARY_I32(MaxI32S, "vm.max.i32.s");
    IREE_VM_ISA_EMIT_OP_CORE_BINARY_I32(MaxI32U, "vm.max.i32.u");
    IREE_VM_ISA_EMIT_OP_CORE_UNARY_I32(NotI32, "vm.not.i32");
    IREE_VM_ISA_EMIT_OP_CORE_BINARY_I32(AndI32, "vm.and.i32");
    IREE_VM_ISA_EMIT_OP_CORE_BINARY_I32(OrI32, "vm.or.i32");
    IREE_VM_ISA_EMIT_OP_CORE_BINARY_I32(XorI32, "vm.xor.i32");
    IREE_VM_ISA_EMIT_OP_CORE_UNARY_I32(CtlzI32, "vm.ctlz.i32");

    IREE_VM_ISA_EMIT_OP_CORE_BINARY_I64(AddI64, "vm.add.i64");
    IREE_VM_ISA_EMIT_OP_CORE_BINARY_I64(SubI64, "vm.sub.i64");
    IREE_VM_ISA_EMIT_OP_CORE_BINARY_I64(MulI64, "vm.mul.i64");
    IREE_VM_ISA_EMIT_OP_CORE_BINARY_I64(DivI64S, "vm.div.i64.s");
    IREE_VM_ISA_EMIT_OP_CORE_BINARY_I64(DivI64U, "vm.div.i64.u");
    IREE_VM_ISA_EMIT_OP_CORE_BINARY_I64(RemI64S, "vm.rem.i64.s");
    IREE_VM_ISA_EMIT_OP_CORE_BINARY_I64(RemI64U, "vm.rem.i64.u");
    IREE_VM_ISA_EMIT_OP_CORE_TERNARY_I64(FMAI64, "vm.fma.i64");
    IREE_VM_ISA_EMIT_OP_CORE_UNARY_I64(AbsI64, "vm.abs.i64");
    IREE_VM_ISA_EMIT_OP_CORE_BINARY_I64(MinI64S, "vm.min.i64.s");
    IREE_VM_ISA_EMIT_OP_CORE_BINARY_I64(MinI64U, "vm.min.i64.u");
    IREE_VM_ISA_EMIT_OP_CORE_BINARY_I64(MaxI64S, "vm.max.i64.s");
    IREE_VM_ISA_EMIT_OP_CORE_BINARY_I64(MaxI64U, "vm.max.i64.u");
    IREE_VM_ISA_EMIT_OP_CORE_UNARY_I64(NotI64, "vm.not.i64");
    IREE_VM_ISA_EMIT_OP_CORE_BINARY_I64(AndI64, "vm.and.i64");
    IREE_VM_ISA_EMIT_OP_CORE_BINARY_I64(OrI64, "vm.or.i64");
    IREE_VM_ISA_EMIT_OP_CORE_BINARY_I64(XorI64, "vm.xor.i64");
    IREE_VM_ISA_EMIT_OP_CORE_UNARY_I64(CtlzI64, "vm.ctlz.i64");

    //===------------------------------------------------------------------===//
    // Casting and type conversion/emulation
    //===------------------------------------------------------------------===//

    IREE_VM_ISA_EMIT_OP_CORE_UNARY_I32(TruncI32I8, "vm.trunc.i32.i8");
    IREE_VM_ISA_EMIT_OP_CORE_UNARY_I32(TruncI32I16, "vm.trunc.i32.i16");
    IREE_VM_ISA_EMIT_OP_CORE_UNARY_I32(ExtI8I32S, "vm.ext.i8.i32.s");
    IREE_VM_ISA_EMIT_OP_CORE_UNARY_I32(ExtI8I32U, "vm.ext.i8.i32.u");
    IREE_VM_ISA_EMIT_OP_CORE_UNARY_I32(ExtI16I32S, "vm.ext.i16.i32.s");
    IREE_VM_ISA_EMIT_OP_CORE_UNARY_I32(ExtI16I32U, "vm.ext.i16.i32.u");

    IREE_VM_ISA_EMIT_OP(CORE, TruncI64I32) {
      IREE_VM_ISA_DECODE_OPERAND_I64(operand_reg);
      IREE_VM_ISA_DECODE_RESULT_I32(result_reg);
      IREE_VM_ISA_EMIT_I32_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.trunc.i64.i32 "));
      IREE_VM_ISA_EMIT_I64_REG_NAME(operand_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I64(regs->i32[operand_reg]);
      break;
    }
    IREE_VM_ISA_EMIT_OP(CORE, ExtI32I64S) {
      IREE_VM_ISA_DECODE_OPERAND_I32(operand_reg);
      IREE_VM_ISA_DECODE_RESULT_I64(result_reg);
      IREE_VM_ISA_EMIT_I64_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.ext.i32.i64.s "));
      IREE_VM_ISA_EMIT_I32_REG_NAME(operand_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I32(regs->i32[operand_reg]);
      break;
    }
    IREE_VM_ISA_EMIT_OP(CORE, ExtI32I64U) {
      IREE_VM_ISA_DECODE_OPERAND_I32(operand_reg);
      IREE_VM_ISA_DECODE_RESULT_I64(result_reg);
      IREE_VM_ISA_EMIT_I64_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.ext.i32.i64.u "));
      IREE_VM_ISA_EMIT_I32_REG_NAME(operand_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I32(regs->i32[operand_reg]);
      break;
    }

    IREE_VM_ISA_EMIT_OP(CORE, CastAnyRef) {
      IREE_VM_ISA_DECODE_OPERAND_REF_MOVE(operand_reg);
      IREE_VM_ISA_DECODE_TYPE_OF(type_def);
      IREE_VM_ISA_DECODE_RESULT_REF_MOVE(result_reg);
      IREE_VM_ISA_EMIT_REF_REG_NAME_MOVE(result_reg, result_reg_is_move);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.cast.any.ref "));
      IREE_VM_ISA_EMIT_REF_REG_NAME_MOVE(operand_reg, operand_reg_is_move);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_REF(&regs->ref[operand_reg]);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " : !vm.ref<?> -> "));
      IREE_VM_ISA_EMIT_TYPE_NAME(type_def);
      break;
    }

    //===------------------------------------------------------------------===//
    // Native bitwise shifts and rotates
    //===------------------------------------------------------------------===//

#define IREE_VM_ISA_EMIT_OP_CORE_SHIFT_I32(op_name, op_mnemonic)       \
  IREE_VM_ISA_EMIT_OP(CORE, op_name) {                                 \
    IREE_VM_ISA_DECODE_OPERAND_I32(operand_reg);                       \
    IREE_VM_ISA_DECODE_OPERAND_I32(amount_reg);                        \
    IREE_VM_ISA_DECODE_RESULT_I32(result_reg);                         \
    IREE_VM_ISA_EMIT_I32_REG_NAME(result_reg);                         \
    IREE_RETURN_IF_ERROR(                                              \
        iree_string_builder_append_format(b, " = %s ", op_mnemonic));  \
    IREE_VM_ISA_EMIT_I32_REG_NAME(operand_reg);                        \
    IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I32(regs->i32[operand_reg]);       \
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", ")); \
    IREE_VM_ISA_EMIT_I32_REG_NAME(amount_reg);                         \
    IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I32(regs->i32[amount_reg]);        \
    break;                                                             \
  }

    IREE_VM_ISA_EMIT_OP_CORE_SHIFT_I32(ShlI32, "vm.shl.i32");
    IREE_VM_ISA_EMIT_OP_CORE_SHIFT_I32(ShrI32S, "vm.shr.i32.s");
    IREE_VM_ISA_EMIT_OP_CORE_SHIFT_I32(ShrI32U, "vm.shr.i32.u");

#define IREE_VM_ISA_EMIT_OP_CORE_SHIFT_I64(op_name, op_mnemonic)       \
  IREE_VM_ISA_EMIT_OP(CORE, op_name) {                                 \
    IREE_VM_ISA_DECODE_OPERAND_I64(operand_reg);                       \
    IREE_VM_ISA_DECODE_OPERAND_I32(amount_reg);                        \
    IREE_VM_ISA_DECODE_RESULT_I64(result_reg);                         \
    IREE_VM_ISA_EMIT_I64_REG_NAME(result_reg);                         \
    IREE_RETURN_IF_ERROR(                                              \
        iree_string_builder_append_format(b, " = %s ", op_mnemonic));  \
    IREE_VM_ISA_EMIT_I64_REG_NAME(operand_reg);                        \
    IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I64(regs->i32[operand_reg]);       \
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", ")); \
    IREE_VM_ISA_EMIT_I32_REG_NAME(amount_reg);                         \
    IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I32(regs->i32[amount_reg]);        \
    break;                                                             \
  }

    IREE_VM_ISA_EMIT_OP_CORE_SHIFT_I64(ShlI64, "vm.shl.i64");
    IREE_VM_ISA_EMIT_OP_CORE_SHIFT_I64(ShrI64S, "vm.shr.i64.s");
    IREE_VM_ISA_EMIT_OP_CORE_SHIFT_I64(ShrI64U, "vm.shr.i64.u");

    //===------------------------------------------------------------------===//
    // Comparison ops
    //===------------------------------------------------------------------===//

    IREE_VM_ISA_EMIT_OP_CORE_BINARY_I32(CmpEQI32, "vm.cmp.eq.i32");
    IREE_VM_ISA_EMIT_OP_CORE_BINARY_I32(CmpNEI32, "vm.cmp.ne.i32");
    IREE_VM_ISA_EMIT_OP_CORE_BINARY_I32(CmpLTI32S, "vm.cmp.lt.i32.s");
    IREE_VM_ISA_EMIT_OP_CORE_BINARY_I32(CmpLTI32U, "vm.cmp.lt.i32.u");
    IREE_VM_ISA_EMIT_OP_CORE_UNARY_I32(CmpNZI32, "vm.cmp.nz.i32");

#define IREE_VM_ISA_EMIT_OP_CORE_CMP_I64(op_name, op_mnemonic)         \
  IREE_VM_ISA_EMIT_OP(CORE, op_name) {                                 \
    IREE_VM_ISA_DECODE_OPERAND_I64(lhs_reg);                           \
    IREE_VM_ISA_DECODE_OPERAND_I64(rhs_reg);                           \
    IREE_VM_ISA_DECODE_RESULT_I32(result_reg);                         \
    IREE_VM_ISA_EMIT_I32_REG_NAME(result_reg);                         \
    IREE_RETURN_IF_ERROR(                                              \
        iree_string_builder_append_format(b, " = %s ", op_mnemonic));  \
    IREE_VM_ISA_EMIT_I64_REG_NAME(lhs_reg);                            \
    IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I64(regs->i32[lhs_reg]);           \
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", ")); \
    IREE_VM_ISA_EMIT_I64_REG_NAME(rhs_reg);                            \
    IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I64(regs->i32[rhs_reg]);           \
    break;                                                             \
  }

    IREE_VM_ISA_EMIT_OP_CORE_CMP_I64(CmpEQI64, "vm.cmp.eq.i64");
    IREE_VM_ISA_EMIT_OP_CORE_CMP_I64(CmpNEI64, "vm.cmp.ne.i64");
    IREE_VM_ISA_EMIT_OP_CORE_CMP_I64(CmpLTI64S, "vm.cmp.lt.i64.s");
    IREE_VM_ISA_EMIT_OP_CORE_CMP_I64(CmpLTI64U, "vm.cmp.lt.i64.u");
    IREE_VM_ISA_EMIT_OP(CORE, CmpNZI64) {
      IREE_VM_ISA_DECODE_OPERAND_I64(operand_reg);
      IREE_VM_ISA_DECODE_RESULT_I32(result_reg);
      IREE_VM_ISA_EMIT_I32_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.cmp.nz.i64 "));
      IREE_VM_ISA_EMIT_I64_REG_NAME(operand_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I64(regs->i32[operand_reg]);
      break;
    }

    IREE_VM_ISA_EMIT_OP(CORE, CmpEQRef) {
      IREE_VM_ISA_DECODE_OPERAND_REF(lhs_reg);
      IREE_VM_ISA_DECODE_OPERAND_REF(rhs_reg);
      IREE_VM_ISA_DECODE_RESULT_I32(result_reg);
      IREE_VM_ISA_EMIT_I32_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.cmp.eq.ref "));
      IREE_VM_ISA_EMIT_REF_REG_NAME(lhs_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_REF(&regs->ref[lhs_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      IREE_VM_ISA_EMIT_REF_REG_NAME(rhs_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_REF(&regs->ref[rhs_reg]);
      break;
    }
    IREE_VM_ISA_EMIT_OP(CORE, CmpNERef) {
      IREE_VM_ISA_DECODE_OPERAND_REF(lhs_reg);
      IREE_VM_ISA_DECODE_OPERAND_REF(rhs_reg);
      IREE_VM_ISA_DECODE_RESULT_I32(result_reg);
      IREE_VM_ISA_EMIT_I32_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.cmp.ne.ref "));
      IREE_VM_ISA_EMIT_REF_REG_NAME(lhs_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_REF(&regs->ref[lhs_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      IREE_VM_ISA_EMIT_REF_REG_NAME(rhs_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_REF(&regs->ref[rhs_reg]);
      break;
    }
    IREE_VM_ISA_EMIT_OP(CORE, CmpNZRef) {
      IREE_VM_ISA_DECODE_OPERAND_REF(operand_reg);
      IREE_VM_ISA_DECODE_RESULT_I32(result_reg);
      IREE_VM_ISA_EMIT_I32_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.cmp.nz.ref "));
      IREE_VM_ISA_EMIT_REF_REG_NAME(operand_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_REF(&regs->ref[operand_reg]);
      break;
    }

    //===------------------------------------------------------------------===//
    // Control flow
    //===------------------------------------------------------------------===//

    IREE_VM_ISA_EMIT_OP(CORE, Block) {
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_string(b, IREE_SV("<block>")));
      break;
    }

    IREE_VM_ISA_EMIT_OP(CORE, Branch) {
      IREE_VM_ISA_DECODE_BRANCH_TARGET_PC(block_pc);
      IREE_VM_ISA_DECODE_BRANCH_OPERANDS(remap_list);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_format(b, "vm.br ^%08X(", block_pc));
      IREE_VM_ISA_EMIT_REMAP_LIST(remap_list);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ")"));
      break;
    }

    IREE_VM_ISA_EMIT_OP(CORE, CondBranch) {
      IREE_VM_ISA_DECODE_OPERAND_I32(condition_reg);
      IREE_VM_ISA_DECODE_BRANCH_TARGET_PC(true_block_pc);
      IREE_VM_ISA_DECODE_BRANCH_OPERANDS(true_remap_list);
      IREE_VM_ISA_DECODE_BRANCH_TARGET_PC(false_block_pc);
      IREE_VM_ISA_DECODE_BRANCH_OPERANDS(false_remap_list);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, "vm.cond_br "));
      IREE_VM_ISA_EMIT_I32_REG_NAME(condition_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I32(regs->i32[condition_reg]);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_format(b, ", ^%08X(", true_block_pc));
      IREE_VM_ISA_EMIT_REMAP_LIST(true_remap_list);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_format(b, "), ^%08X(", false_block_pc));
      IREE_VM_ISA_EMIT_REMAP_LIST(false_remap_list);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ")"));
      break;
    }

    IREE_VM_ISA_EMIT_OP(CORE, BranchTable) {
      IREE_VM_ISA_DECODE_OPERAND_I32(index_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, "vm.br_table "));
      IREE_VM_ISA_EMIT_I32_REG_NAME(index_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I32(regs->i32[index_reg]);
      IREE_VM_ISA_DECODE_BRANCH_TARGET_PC(default_block_pc);
      IREE_VM_ISA_DECODE_BRANCH_OPERANDS(default_remap_list);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          b, " { default: ^%08X(", default_block_pc));
      IREE_VM_ISA_EMIT_REMAP_LIST(default_remap_list);
      IREE_VM_ISA_DECODE_CONST_I16(table_size);
      for (uint16_t i = 0; i < table_size; ++i) {
        IREE_VM_ISA_DECODE_BRANCH_TARGET_PC(case_block_pc);
        IREE_VM_ISA_DECODE_BRANCH_OPERANDS(case_remap_list);
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            b, "), %u: ^%08X(", i, case_block_pc));
        IREE_VM_ISA_EMIT_REMAP_LIST(case_remap_list);
      }
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ") }"));
      break;
    }

    IREE_VM_ISA_EMIT_OP(CORE, Call) {
      IREE_VM_ISA_DECODE_FUNC_ATTR(function_ordinal);
      IREE_VM_ISA_DECODE_VARIADIC_OPERANDS(src_reg_list);
      IREE_VM_ISA_DECODE_VARIADIC_RESULTS(dst_reg_list);
      if (dst_reg_list->size > 0) {
        IREE_VM_ISA_EMIT_RESULT_REG_LIST(dst_reg_list);
        IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, " = "));
      }
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, "vm.call @"));
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_disassembler_print_function_name(
          module, module_state, function_ordinal, b));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, "("));
      IREE_VM_ISA_EMIT_OPERAND_REG_LIST(src_reg_list);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ")"));
      break;
    }

    IREE_VM_ISA_EMIT_OP(CORE, CallVariadic) {
      IREE_VM_ISA_DECODE_FUNC_ATTR(function_ordinal);
      // TODO(benvanik): print segment sizes.
      IREE_VM_ISA_DECODE_VARIADIC_OPERANDS(segment_size_list);
      (void)segment_size_list;
      IREE_VM_ISA_DECODE_VARIADIC_OPERANDS(src_reg_list);
      IREE_VM_ISA_DECODE_VARIADIC_RESULTS(dst_reg_list);
      if (dst_reg_list->size > 0) {
        IREE_VM_ISA_EMIT_RESULT_REG_LIST(dst_reg_list);
        IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, " = "));
      }
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, "vm.call.varadic @"));
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_disassembler_print_function_name(
          module, module_state, function_ordinal, b));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, "("));
      IREE_VM_ISA_EMIT_OPERAND_REG_LIST(src_reg_list);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ")"));
      break;
    }

    IREE_VM_ISA_EMIT_OP(CORE, Return) {
      IREE_VM_ISA_DECODE_VARIADIC_OPERANDS(src_reg_list);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, "vm.return "));
      IREE_VM_ISA_EMIT_OPERAND_REG_LIST(src_reg_list);
      break;
    }

    IREE_VM_ISA_EMIT_OP(CORE, Fail) {
      IREE_VM_ISA_DECODE_OPERAND_I32(status_code_reg);
      IREE_VM_ISA_DECODE_STRING_ATTR(message);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, "vm.fail "));
      IREE_VM_ISA_EMIT_I32_REG_NAME(status_code_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I32(regs->i32[status_code_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          b, ", \"%.*s\"", (int)message.size, message.data));
      break;
    }

    IREE_VM_ISA_EMIT_OP(CORE, ImportResolved) {
      IREE_VM_ISA_DECODE_FUNC_ATTR(function_ordinal);
      IREE_VM_ISA_DECODE_RESULT_I32(result_reg);
      IREE_VM_ISA_EMIT_I32_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.import.exists @"));
      int is_import = iree_vm_isa_function_ordinal_is_import(function_ordinal);
      if (IREE_UNLIKELY(!is_import)) {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            b, "{{INVALID ORDINAL %d}}", function_ordinal));
        break;
      }
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_disassembler_print_function_name(
          module, module_state, function_ordinal, b));
      if (module_state) {
        uint32_t import_ordinal =
            iree_vm_isa_function_ordinal_as_import(function_ordinal);
        if (IREE_UNLIKELY(import_ordinal >= module_state->import_count)) {
          IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
              b, "{{OUT OF RANGE ORDINAL %u}}", import_ordinal));
          break;
        }
        const iree_vm_bytecode_import_t* import =
            &module_state->import_table[import_ordinal];
        IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
            b, import->function.module != NULL ? " // (resolved)"
                                               : " // (unresolved)"));
      }
      break;
    }

    //===------------------------------------------------------------------===//
    // Async/fiber ops
    //===------------------------------------------------------------------===//

    IREE_VM_ISA_EMIT_OP(CORE, Yield) {
      IREE_VM_ISA_DECODE_BRANCH_TARGET_PC(block_pc);
      IREE_VM_ISA_DECODE_BRANCH_OPERANDS(remap_list);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_format(b, "vm.yield ^%08X(", block_pc));
      IREE_VM_ISA_EMIT_REMAP_LIST(remap_list);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ")"));
      break;
    }

    //===------------------------------------------------------------------===//
    // Debugging
    //===------------------------------------------------------------------===//

    IREE_VM_ISA_EMIT_OP(CORE, Trace) {
      IREE_VM_ISA_DECODE_STRING_ATTR(event_name);
      IREE_VM_ISA_DECODE_VARIADIC_OPERANDS(src_reg_list);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          b, "vm.trace \"%.*s\"(", (int)event_name.size, event_name.data));
      IREE_VM_ISA_EMIT_OPERAND_REG_LIST(src_reg_list);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ")"));
      break;
    }

    IREE_VM_ISA_EMIT_OP(CORE, Print) {
      IREE_VM_ISA_DECODE_STRING_ATTR(event_name);
      IREE_VM_ISA_DECODE_VARIADIC_OPERANDS(src_reg_list);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          b, "vm.print \"%.*s\"(", (int)event_name.size, event_name.data));
      IREE_VM_ISA_EMIT_OPERAND_REG_LIST(src_reg_list);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ")"));
      break;
    }

    IREE_VM_ISA_EMIT_OP(CORE, Break) {
      IREE_VM_ISA_DECODE_BRANCH_TARGET_PC(block_pc);
      IREE_VM_ISA_DECODE_BRANCH_OPERANDS(remap_list);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_format(b, "vm.break ^%08X(", block_pc));
      IREE_VM_ISA_EMIT_REMAP_LIST(remap_list);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ")"));
      break;
    }

    IREE_VM_ISA_EMIT_OP(CORE, CondBreak) {
      IREE_VM_ISA_DECODE_OPERAND_I32(condition_reg);
      IREE_VM_ISA_DECODE_BRANCH_TARGET_PC(block_pc);
      IREE_VM_ISA_DECODE_BRANCH_OPERANDS(remap_list);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, "vm.cond_break "));
      IREE_VM_ISA_EMIT_I32_REG_NAME(condition_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I32(regs->i32[condition_reg]);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_format(b, ", ^%08X(", block_pc));
      IREE_VM_ISA_EMIT_REMAP_LIST(remap_list);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ")"));
      break;
    }

    //===------------------------------------------------------------------===//
    // Extension trampolines
    //===------------------------------------------------------------------===//

#if IREE_VM_EXT_F32_ENABLE
    IREE_VM_ISA_BEGIN_DISASM_PREFIX(PrefixExtF32, EXT_F32)

    //===----------------------------------------------------------------===//
    // ExtF32: Globals
    //===----------------------------------------------------------------===//

    IREE_VM_ISA_EMIT_OP(EXT_F32, GlobalLoadF32) {
      IREE_VM_ISA_DECODE_GLOBAL_ATTR(byte_offset);
      IREE_VM_ISA_DECODE_RESULT_F32(value_reg);
      IREE_VM_ISA_EMIT_F32_REG_NAME(value_reg);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          b, " = vm.global.load.f32 .rwdata[%u]", byte_offset));
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_F32(
          module_state->rwdata_storage.data[byte_offset]);
      break;
    }

    IREE_VM_ISA_EMIT_OP(EXT_F32, GlobalStoreF32) {
      IREE_VM_ISA_DECODE_GLOBAL_ATTR(byte_offset);
      IREE_VM_ISA_DECODE_OPERAND_F32(value_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_format(b, "vm.global.store.f32 "));
      IREE_VM_ISA_EMIT_F32_REG_NAME(value_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_F32(regs->i32[value_reg]);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_format(b, ", .rwdata[%u]", byte_offset));
      break;
    }

    IREE_VM_ISA_EMIT_OP(EXT_F32, GlobalLoadIndirectF32) {
      IREE_VM_ISA_DECODE_OPERAND_I32(byte_offset_reg);
      IREE_VM_ISA_DECODE_RESULT_I32(value_reg);
      IREE_VM_ISA_EMIT_F32_REG_NAME(value_reg);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
          b, " = vm.global.load.indirect.f32 .rwdata["));
      IREE_VM_ISA_EMIT_I32_REG_NAME(byte_offset_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I32(regs->i32[byte_offset_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, "]"));
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_F32(
          module_state->rwdata_storage.data[regs->i32[byte_offset_reg]]);
      break;
    }

    IREE_VM_ISA_EMIT_OP(EXT_F32, GlobalStoreIndirectF32) {
      IREE_VM_ISA_DECODE_OPERAND_I32(byte_offset_reg);
      IREE_VM_ISA_DECODE_OPERAND_F32(value_reg);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
          b, "vm.global.store.indirect.f32 "));
      IREE_VM_ISA_EMIT_F32_REG_NAME(value_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_F32(regs->i32[value_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", .rwdata["));
      IREE_VM_ISA_EMIT_I32_REG_NAME(byte_offset_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I32(regs->i32[byte_offset_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, "]"));
      break;
    }

    //===----------------------------------------------------------------===//
    // ExtF32: Constants
    //===----------------------------------------------------------------===//

    IREE_VM_ISA_EMIT_OP(EXT_F32, ConstF32) {
      IREE_VM_ISA_DECODE_ATTR_F32(value);
      IREE_VM_ISA_DECODE_RESULT_F32(result_reg);
      IREE_VM_ISA_EMIT_F32_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_format(b, " = vm.const.f32 %f", value));
      break;
    }

    IREE_VM_ISA_EMIT_OP(EXT_F32, ConstF32Zero) {
      IREE_VM_ISA_DECODE_RESULT_F32(result_reg);
      IREE_VM_ISA_EMIT_F32_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.const.f32.zero"));
      break;
    }

    //===----------------------------------------------------------------===//
    // ExtF32: Lists
    //===----------------------------------------------------------------===//

    IREE_VM_ISA_EMIT_OP(EXT_F32, ListGetF32) {
      IREE_VM_ISA_DECODE_OPERAND_REF(list_reg);
      IREE_VM_ISA_DECODE_OPERAND_I32(index_reg);
      IREE_VM_ISA_DECODE_RESULT_F32(result_reg);
      IREE_VM_ISA_EMIT_F32_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.list.get.f32 "));
      IREE_VM_ISA_EMIT_REF_REG_NAME(list_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_REF(&regs->ref[list_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      IREE_VM_ISA_EMIT_I32_REG_NAME(index_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I32(regs->i32[index_reg]);
      break;
    }

    IREE_VM_ISA_EMIT_OP(EXT_F32, ListSetF32) {
      IREE_VM_ISA_DECODE_OPERAND_REF(list_reg);
      IREE_VM_ISA_DECODE_OPERAND_I32(index_reg);
      IREE_VM_ISA_DECODE_OPERAND_F32(raw_value_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, "vm.list.set.f32 "));
      IREE_VM_ISA_EMIT_REF_REG_NAME(list_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_REF(&regs->ref[list_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      IREE_VM_ISA_EMIT_I32_REG_NAME(index_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I32(regs->i32[index_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      IREE_VM_ISA_EMIT_F32_REG_NAME(raw_value_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_F32(regs->i32[raw_value_reg]);
      break;
    }

    //===----------------------------------------------------------------===//
    // ExtF32: Conditional assignment
    //===----------------------------------------------------------------===//

    IREE_VM_ISA_EMIT_OP(EXT_F32, SelectF32) {
      IREE_VM_ISA_DECODE_OPERAND_I32(condition_reg);
      IREE_VM_ISA_DECODE_OPERAND_F32(true_value_reg);
      IREE_VM_ISA_DECODE_OPERAND_F32(false_value_reg);
      IREE_VM_ISA_DECODE_RESULT_F32(result_reg);
      IREE_VM_ISA_EMIT_F32_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.select.f32 "));
      IREE_VM_ISA_EMIT_I32_REG_NAME(condition_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I32(regs->i32[condition_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, " ? "));
      IREE_VM_ISA_EMIT_F32_REG_NAME(true_value_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_F32(regs->i32[true_value_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, " : "));
      IREE_VM_ISA_EMIT_F32_REG_NAME(false_value_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_F32(regs->i32[false_value_reg]);
      break;
    }

    IREE_VM_ISA_EMIT_OP(EXT_F32, SwitchF32) {
      IREE_VM_ISA_DECODE_OPERAND_I32(index_reg);
      IREE_VM_ISA_DECODE_OPERAND_F32(default_value_reg);
      IREE_VM_ISA_DECODE_VARIADIC_OPERANDS(value_reg_list);
      IREE_VM_ISA_DECODE_RESULT_F32(result_reg);
      IREE_VM_ISA_EMIT_F32_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.switch.f32 "));
      IREE_VM_ISA_EMIT_I32_REG_NAME(index_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I32(regs->i32[index_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, "["));
      IREE_VM_ISA_EMIT_OPERAND_REG_LIST(value_reg_list);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, "] else "));
      IREE_VM_ISA_EMIT_F32_REG_NAME(default_value_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_F32(regs->i32[default_value_reg]);
      break;
    }

    //===----------------------------------------------------------------===//
    // ExtF32: Native floating-point arithmetic
    //===----------------------------------------------------------------===//

    IREE_VM_ISA_EMIT_OP_EXT_F32_BINARY_F32(AddF32, "vm.add.f32");
    IREE_VM_ISA_EMIT_OP_EXT_F32_BINARY_F32(SubF32, "vm.sub.f32");
    IREE_VM_ISA_EMIT_OP_EXT_F32_BINARY_F32(MulF32, "vm.mul.f32");
    IREE_VM_ISA_EMIT_OP_EXT_F32_BINARY_F32(DivF32, "vm.div.f32");
    IREE_VM_ISA_EMIT_OP_EXT_F32_BINARY_F32(RemF32, "vm.rem.f32");
    IREE_VM_ISA_EMIT_OP_EXT_F32_TERNARY_F32(FMAF32, "vm.fma.f32");
    IREE_VM_ISA_EMIT_OP_EXT_F32_UNARY_F32(AbsF32, "vm.abs.f32");
    IREE_VM_ISA_EMIT_OP_EXT_F32_UNARY_F32(NegF32, "vm.neg.f32");
    IREE_VM_ISA_EMIT_OP_EXT_F32_UNARY_F32(CeilF32, "vm.ceil.f32");
    IREE_VM_ISA_EMIT_OP_EXT_F32_UNARY_F32(FloorF32, "vm.floor.f32");
    IREE_VM_ISA_EMIT_OP_EXT_F32_UNARY_F32(RoundF32, "vm.round.f32");
    IREE_VM_ISA_EMIT_OP_EXT_F32_UNARY_F32(RoundF32Even, "vm.round.f32.even");
    IREE_VM_ISA_EMIT_OP_EXT_F32_BINARY_F32(MinF32, "vm.min.f32");
    IREE_VM_ISA_EMIT_OP_EXT_F32_BINARY_F32(MaxF32, "vm.max.f32");

    IREE_VM_ISA_EMIT_OP_EXT_F32_UNARY_F32(AtanF32, "vm.atan.f32");
    IREE_VM_ISA_EMIT_OP_EXT_F32_BINARY_F32(Atan2F32, "vm.atan2.f32");
    IREE_VM_ISA_EMIT_OP_EXT_F32_UNARY_F32(CosF32, "vm.cos.f32");
    IREE_VM_ISA_EMIT_OP_EXT_F32_UNARY_F32(SinF32, "vm.sin.f32");
    IREE_VM_ISA_EMIT_OP_EXT_F32_UNARY_F32(ExpF32, "vm.exp.f32");
    IREE_VM_ISA_EMIT_OP_EXT_F32_UNARY_F32(Exp2F32, "vm.exp2.f32");
    IREE_VM_ISA_EMIT_OP_EXT_F32_UNARY_F32(ExpM1F32, "vm.expm1.f32");
    IREE_VM_ISA_EMIT_OP_EXT_F32_UNARY_F32(LogF32, "vm.log.f32");
    IREE_VM_ISA_EMIT_OP_EXT_F32_UNARY_F32(Log10F32, "vm.log10.f32");
    IREE_VM_ISA_EMIT_OP_EXT_F32_UNARY_F32(Log1pF32, "vm.log1p.f32");
    IREE_VM_ISA_EMIT_OP_EXT_F32_UNARY_F32(Log2F32, "vm.log2.f32");
    IREE_VM_ISA_EMIT_OP_EXT_F32_BINARY_F32(PowF32, "vm.pow.f32");
    IREE_VM_ISA_EMIT_OP_EXT_F32_UNARY_F32(RsqrtF32, "vm.rsqrt.f32");
    IREE_VM_ISA_EMIT_OP_EXT_F32_UNARY_F32(SqrtF32, "vm.sqrt.f32");
    IREE_VM_ISA_EMIT_OP_EXT_F32_UNARY_F32(TanhF32, "vm.tanh.f32");
    IREE_VM_ISA_EMIT_OP_EXT_F32_UNARY_F32(ErfF32, "vm.erf.f32");

    //===----------------------------------------------------------------===//
    // ExtF32: Casting and type conversion/emulation
    //===----------------------------------------------------------------===//

    IREE_VM_ISA_EMIT_OP(EXT_F32, CastSI32F32) {
      IREE_VM_ISA_DECODE_OPERAND_I32(operand_reg);
      IREE_VM_ISA_DECODE_RESULT_F32(result_reg);
      IREE_VM_ISA_EMIT_F32_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.cast.si32.f32 "));
      IREE_VM_ISA_EMIT_I32_REG_NAME(operand_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I32(regs->i32[operand_reg]);
      break;
    }
    IREE_VM_ISA_EMIT_OP(EXT_F32, CastSI64F32) {
      IREE_VM_ISA_DECODE_OPERAND_I64(operand_reg);
      IREE_VM_ISA_DECODE_RESULT_F32(result_reg);
      IREE_VM_ISA_EMIT_F32_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.cast.si64.f32 "));
      IREE_VM_ISA_EMIT_I64_REG_NAME(operand_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I64(regs->i32[operand_reg]);
      break;
    }
    IREE_VM_ISA_EMIT_OP(EXT_F32, CastUI32F32) {
      IREE_VM_ISA_DECODE_OPERAND_I32(operand_reg);
      IREE_VM_ISA_DECODE_RESULT_F32(result_reg);
      IREE_VM_ISA_EMIT_F32_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.cast.ui32.f32 "));
      IREE_VM_ISA_EMIT_I32_REG_NAME(operand_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I32(regs->i32[operand_reg]);
      break;
    }
    IREE_VM_ISA_EMIT_OP(EXT_F32, CastF32SI32) {
      IREE_VM_ISA_DECODE_OPERAND_F32(operand_reg);
      IREE_VM_ISA_DECODE_RESULT_I32(result_reg);
      IREE_VM_ISA_EMIT_I32_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.cast.f32.si32 "));
      IREE_VM_ISA_EMIT_F32_REG_NAME(operand_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_F32(regs->i32[operand_reg]);
      break;
    }
    IREE_VM_ISA_EMIT_OP(EXT_F32, CastF32SI64) {
      IREE_VM_ISA_DECODE_OPERAND_F32(operand_reg);
      IREE_VM_ISA_DECODE_RESULT_I64(result_reg);
      IREE_VM_ISA_EMIT_I64_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.cast.f32.si64 "));
      IREE_VM_ISA_EMIT_F32_REG_NAME(operand_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_F32(regs->i32[operand_reg]);
      break;
    }
    IREE_VM_ISA_EMIT_OP(EXT_F32, CastF32UI32) {
      IREE_VM_ISA_DECODE_OPERAND_F32(operand_reg);
      IREE_VM_ISA_DECODE_RESULT_I32(result_reg);
      IREE_VM_ISA_EMIT_I32_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.cast.f32.ui32 "));
      IREE_VM_ISA_EMIT_F32_REG_NAME(operand_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_F32(regs->i32[operand_reg]);
      break;
    }
    IREE_VM_ISA_EMIT_OP(EXT_F32, CastF32UI64) {
      IREE_VM_ISA_DECODE_OPERAND_F32(operand_reg);
      IREE_VM_ISA_DECODE_RESULT_I64(result_reg);
      IREE_VM_ISA_EMIT_I64_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.cast.f32.ui64 "));
      IREE_VM_ISA_EMIT_F32_REG_NAME(operand_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_F32(regs->i32[operand_reg]);
      break;
    }
    IREE_VM_ISA_EMIT_OP(EXT_F32, BitcastI32F32) {
      IREE_VM_ISA_DECODE_OPERAND_I32(operand_reg);
      IREE_VM_ISA_DECODE_RESULT_F32(result_reg);
      IREE_VM_ISA_EMIT_F32_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.bitcast.i32.f32 "));
      IREE_VM_ISA_EMIT_I32_REG_NAME(operand_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I32(regs->i32[operand_reg]);
      break;
    }
    IREE_VM_ISA_EMIT_OP(EXT_F32, BitcastF32I32) {
      IREE_VM_ISA_DECODE_OPERAND_F32(operand_reg);
      IREE_VM_ISA_DECODE_RESULT_I32(result_reg);
      IREE_VM_ISA_EMIT_I32_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.bitcast.f32.i32 "));
      IREE_VM_ISA_EMIT_F32_REG_NAME(operand_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_F32(regs->i32[operand_reg]);
      break;
    }

    //===----------------------------------------------------------------===//
    // ExtF32: Comparison ops
    //===----------------------------------------------------------------===//

#define IREE_VM_ISA_EMIT_OP_EXT_F32_CMP_F32(op_name, op_mnemonic)      \
  IREE_VM_ISA_EMIT_OP(EXT_F32, op_name) {                              \
    IREE_VM_ISA_DECODE_OPERAND_F32(lhs_reg);                           \
    IREE_VM_ISA_DECODE_OPERAND_F32(rhs_reg);                           \
    IREE_VM_ISA_DECODE_RESULT_I32(result_reg);                         \
    IREE_VM_ISA_EMIT_I32_REG_NAME(result_reg);                         \
    IREE_RETURN_IF_ERROR(                                              \
        iree_string_builder_append_format(b, " = %s ", op_mnemonic));  \
    IREE_VM_ISA_EMIT_F32_REG_NAME(lhs_reg);                            \
    IREE_VM_ISA_EMIT_OPTIONAL_VALUE_F32(regs->i32[lhs_reg]);           \
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", ")); \
    IREE_VM_ISA_EMIT_F32_REG_NAME(rhs_reg);                            \
    IREE_VM_ISA_EMIT_OPTIONAL_VALUE_F32(regs->i32[rhs_reg]);           \
    break;                                                             \
  }

    IREE_VM_ISA_EMIT_OP_EXT_F32_CMP_F32(CmpEQF32O, "vm.cmp.eq.f32.o");
    IREE_VM_ISA_EMIT_OP_EXT_F32_CMP_F32(CmpEQF32U, "vm.cmp.eq.f32.u");
    IREE_VM_ISA_EMIT_OP_EXT_F32_CMP_F32(CmpNEF32O, "vm.cmp.ne.f32.o");
    IREE_VM_ISA_EMIT_OP_EXT_F32_CMP_F32(CmpNEF32U, "vm.cmp.ne.f32.u");
    IREE_VM_ISA_EMIT_OP_EXT_F32_CMP_F32(CmpLTF32O, "vm.cmp.lt.f32.o");
    IREE_VM_ISA_EMIT_OP_EXT_F32_CMP_F32(CmpLTF32U, "vm.cmp.lt.f32.u");
    IREE_VM_ISA_EMIT_OP_EXT_F32_CMP_F32(CmpLTEF32O, "vm.cmp.lte.f32.o");
    IREE_VM_ISA_EMIT_OP_EXT_F32_CMP_F32(CmpLTEF32U, "vm.cmp.lte.f32.u");
    IREE_VM_ISA_EMIT_OP(EXT_F32, CmpNaNF32) {
      IREE_VM_ISA_DECODE_OPERAND_F32(operand_reg);
      IREE_VM_ISA_DECODE_RESULT_I32(result_reg);
      IREE_VM_ISA_EMIT_I32_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.cmp.nan.f32 "));
      IREE_VM_ISA_EMIT_F32_REG_NAME(operand_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_F32(regs->i32[operand_reg]);
      break;
    }

    //===----------------------------------------------------------------===//
    // ExtF32: Buffers
    //===----------------------------------------------------------------===//

    IREE_VM_ISA_EMIT_OP(EXT_F32, BufferFillF32) {
      IREE_VM_ISA_DECODE_OPERAND_REF(buffer_reg);
      IREE_VM_ISA_DECODE_OPERAND_I64(offset_reg);
      IREE_VM_ISA_DECODE_OPERAND_I64(length_reg);
      IREE_VM_ISA_DECODE_OPERAND_F32(value_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, "vm.buffer.fill.f32 "));
      IREE_VM_ISA_EMIT_REF_REG_NAME(buffer_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_REF(&regs->ref[buffer_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      IREE_VM_ISA_EMIT_I64_REG_NAME(offset_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I64(regs->i32[offset_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      IREE_VM_ISA_EMIT_I64_REG_NAME(length_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I64(regs->i32[length_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      IREE_VM_ISA_EMIT_F32_REG_NAME(value_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_F32(regs->i32[value_reg]);
      break;
    }

    IREE_VM_ISA_EMIT_OP(EXT_F32, BufferLoadF32) {
      IREE_VM_ISA_DECODE_OPERAND_REF(buffer_reg);
      IREE_VM_ISA_DECODE_OPERAND_I64(offset_reg);
      IREE_VM_ISA_DECODE_RESULT_F32(result_reg);
      IREE_VM_ISA_EMIT_F32_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.buffer.load.f32 "));
      IREE_VM_ISA_EMIT_REF_REG_NAME(buffer_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_REF(&regs->ref[buffer_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      IREE_VM_ISA_EMIT_I64_REG_NAME(offset_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I64(regs->i32[offset_reg]);
      break;
    }

    IREE_VM_ISA_EMIT_OP(EXT_F32, BufferStoreF32) {
      IREE_VM_ISA_DECODE_OPERAND_REF(buffer_reg);
      IREE_VM_ISA_DECODE_OPERAND_I64(offset_reg);
      IREE_VM_ISA_DECODE_OPERAND_F32(value_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, "vm.buffer.store.f32 "));
      IREE_VM_ISA_EMIT_F32_REG_NAME(value_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_F32(regs->i32[value_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      IREE_VM_ISA_EMIT_REF_REG_NAME(buffer_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_REF(&regs->ref[buffer_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      IREE_VM_ISA_EMIT_I64_REG_NAME(offset_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I64(regs->i32[offset_reg]);
      break;
    }

    IREE_VM_ISA_END_DISASM_PREFIX()
#else
    IREE_VM_ISA_UNHANDLED_DISASM_PREFIX(PrefixExtF32, EXT_F32)
#endif  // IREE_VM_EXT_F32_ENABLE

#if IREE_VM_EXT_F64_ENABLE
    IREE_VM_ISA_BEGIN_DISASM_PREFIX(PrefixExtF64, EXT_F64)

    //===----------------------------------------------------------------===//
    // ExtF64: Globals
    //===----------------------------------------------------------------===//

    IREE_VM_ISA_EMIT_OP(EXT_F64, GlobalLoadF64) {
      IREE_VM_ISA_DECODE_GLOBAL_ATTR(byte_offset);
      IREE_VM_ISA_DECODE_RESULT_F64(value_reg);
      IREE_VM_ISA_EMIT_F64_REG_NAME(value_reg);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          b, " = vm.global.load.f64 .rwdata[%u]", byte_offset));
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_F64(
          module_state->rwdata_storage.data[byte_offset]);
      break;
    }

    IREE_VM_ISA_EMIT_OP(EXT_F64, GlobalStoreF64) {
      IREE_VM_ISA_DECODE_GLOBAL_ATTR(byte_offset);
      IREE_VM_ISA_DECODE_OPERAND_F64(value_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_format(b, "vm.global.store.f64 "));
      IREE_VM_ISA_EMIT_F64_REG_NAME(value_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_F64(regs->i32[value_reg]);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_format(b, ", .rwdata[%u]", byte_offset));
      break;
    }

    IREE_VM_ISA_EMIT_OP(EXT_F64, GlobalLoadIndirectF64) {
      IREE_VM_ISA_DECODE_OPERAND_I32(byte_offset_reg);
      IREE_VM_ISA_DECODE_RESULT_I32(value_reg);
      IREE_VM_ISA_EMIT_F64_REG_NAME(value_reg);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
          b, " = vm.global.load.indirect.f64 .rwdata["));
      IREE_VM_ISA_EMIT_I32_REG_NAME(byte_offset_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I32(regs->i32[byte_offset_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, "]"));
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_F64(
          module_state->rwdata_storage.data[regs->i32[byte_offset_reg]]);
      break;
    }

    IREE_VM_ISA_EMIT_OP(EXT_F64, GlobalStoreIndirectF64) {
      IREE_VM_ISA_DECODE_OPERAND_I32(byte_offset_reg);
      IREE_VM_ISA_DECODE_OPERAND_F64(value_reg);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
          b, "vm.global.store.indirect.f64 "));
      IREE_VM_ISA_EMIT_F64_REG_NAME(value_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_F64(regs->i32[value_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", .rwdata["));
      IREE_VM_ISA_EMIT_I32_REG_NAME(byte_offset_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I32(regs->i32[byte_offset_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, "]"));
      break;
    }

    //===----------------------------------------------------------------===//
    // ExtF64: Constants
    //===----------------------------------------------------------------===//

    IREE_VM_ISA_EMIT_OP(EXT_F64, ConstF64) {
      IREE_VM_ISA_DECODE_ATTR_F64(value);
      IREE_VM_ISA_DECODE_RESULT_F64(result_reg);
      IREE_VM_ISA_EMIT_F64_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_format(b, " = vm.const.f64 %f", value));
      break;
    }

    IREE_VM_ISA_EMIT_OP(EXT_F64, ConstF64Zero) {
      IREE_VM_ISA_DECODE_RESULT_F64(result_reg);
      IREE_VM_ISA_EMIT_F64_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.const.f64.zero"));
      break;
    }

    //===----------------------------------------------------------------===//
    // ExtF64: Lists
    //===----------------------------------------------------------------===//

    IREE_VM_ISA_EMIT_OP(EXT_F64, ListGetF64) {
      IREE_VM_ISA_DECODE_OPERAND_REF(list_reg);
      IREE_VM_ISA_DECODE_OPERAND_I32(index_reg);
      IREE_VM_ISA_DECODE_RESULT_F64(result_reg);
      IREE_VM_ISA_EMIT_F64_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.list.get.f64 "));
      IREE_VM_ISA_EMIT_REF_REG_NAME(list_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_REF(&regs->ref[list_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      IREE_VM_ISA_EMIT_I32_REG_NAME(index_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I32(regs->i32[index_reg]);
      break;
    }

    IREE_VM_ISA_EMIT_OP(EXT_F64, ListSetF64) {
      IREE_VM_ISA_DECODE_OPERAND_REF(list_reg);
      IREE_VM_ISA_DECODE_OPERAND_I32(index_reg);
      IREE_VM_ISA_DECODE_OPERAND_F64(raw_value_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, "vm.list.set.f64 "));
      IREE_VM_ISA_EMIT_REF_REG_NAME(list_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_REF(&regs->ref[list_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      IREE_VM_ISA_EMIT_I32_REG_NAME(index_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I32(regs->i32[index_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      IREE_VM_ISA_EMIT_F64_REG_NAME(raw_value_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_F64(regs->i32[raw_value_reg]);
      break;
    }

    //===----------------------------------------------------------------===//
    // ExtF64: Conditional assignment
    //===----------------------------------------------------------------===//

    IREE_VM_ISA_EMIT_OP(EXT_F64, SelectF64) {
      IREE_VM_ISA_DECODE_OPERAND_I32(condition_reg);
      IREE_VM_ISA_DECODE_OPERAND_F64(true_value_reg);
      IREE_VM_ISA_DECODE_OPERAND_F64(false_value_reg);
      IREE_VM_ISA_DECODE_RESULT_F64(result_reg);
      IREE_VM_ISA_EMIT_F64_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.select.f64 "));
      IREE_VM_ISA_EMIT_I32_REG_NAME(condition_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I32(regs->i32[condition_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, " ? "));
      IREE_VM_ISA_EMIT_F64_REG_NAME(true_value_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_F64(regs->i32[true_value_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, " : "));
      IREE_VM_ISA_EMIT_F64_REG_NAME(false_value_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_F64(regs->i32[false_value_reg]);
      break;
    }

    IREE_VM_ISA_EMIT_OP(EXT_F64, SwitchF64) {
      IREE_VM_ISA_DECODE_OPERAND_I32(index_reg);
      IREE_VM_ISA_DECODE_OPERAND_F64(default_value_reg);
      IREE_VM_ISA_DECODE_VARIADIC_OPERANDS(value_reg_list);
      IREE_VM_ISA_DECODE_RESULT_F64(result_reg);
      IREE_VM_ISA_EMIT_F64_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.switch.f64 "));
      IREE_VM_ISA_EMIT_I32_REG_NAME(index_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I32(regs->i32[index_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, "["));
      IREE_VM_ISA_EMIT_OPERAND_REG_LIST(value_reg_list);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, "] else "));
      IREE_VM_ISA_EMIT_F64_REG_NAME(default_value_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_F64(regs->i32[default_value_reg]);
      break;
    }

    //===----------------------------------------------------------------===//
    // ExtF64: Native floating-point arithmetic
    //===----------------------------------------------------------------===//

    IREE_VM_ISA_EMIT_OP_EXT_F64_BINARY_F64(AddF64, "vm.add.f64");
    IREE_VM_ISA_EMIT_OP_EXT_F64_BINARY_F64(SubF64, "vm.sub.f64");
    IREE_VM_ISA_EMIT_OP_EXT_F64_BINARY_F64(MulF64, "vm.mul.f64");
    IREE_VM_ISA_EMIT_OP_EXT_F64_BINARY_F64(DivF64, "vm.div.f64");
    IREE_VM_ISA_EMIT_OP_EXT_F64_BINARY_F64(RemF64, "vm.rem.f64");
    IREE_VM_ISA_EMIT_OP_EXT_F64_TERNARY_F64(FMAF64, "vm.fma.f64");
    IREE_VM_ISA_EMIT_OP_EXT_F64_UNARY_F64(AbsF64, "vm.abs.f64");
    IREE_VM_ISA_EMIT_OP_EXT_F64_UNARY_F64(NegF64, "vm.neg.f64");
    IREE_VM_ISA_EMIT_OP_EXT_F64_UNARY_F64(CeilF64, "vm.ceil.f64");
    IREE_VM_ISA_EMIT_OP_EXT_F64_UNARY_F64(FloorF64, "vm.floor.f64");
    IREE_VM_ISA_EMIT_OP_EXT_F64_UNARY_F64(RoundF64, "vm.round.f64");
    IREE_VM_ISA_EMIT_OP_EXT_F64_UNARY_F64(RoundF64Even, "vm.round.f64.even");
    IREE_VM_ISA_EMIT_OP_EXT_F64_BINARY_F64(MinF64, "vm.min.f64");
    IREE_VM_ISA_EMIT_OP_EXT_F64_BINARY_F64(MaxF64, "vm.max.f64");

    IREE_VM_ISA_EMIT_OP_EXT_F64_UNARY_F64(AtanF64, "vm.atan.f64");
    IREE_VM_ISA_EMIT_OP_EXT_F64_BINARY_F64(Atan2F64, "vm.atan2.f64");
    IREE_VM_ISA_EMIT_OP_EXT_F64_UNARY_F64(CosF64, "vm.cos.f64");
    IREE_VM_ISA_EMIT_OP_EXT_F64_UNARY_F64(SinF64, "vm.sin.f64");
    IREE_VM_ISA_EMIT_OP_EXT_F64_UNARY_F64(ExpF64, "vm.exp.f64");
    IREE_VM_ISA_EMIT_OP_EXT_F64_UNARY_F64(Exp2F64, "vm.exp2.f64");
    IREE_VM_ISA_EMIT_OP_EXT_F64_UNARY_F64(ExpM1F64, "vm.expm1.f64");
    IREE_VM_ISA_EMIT_OP_EXT_F64_UNARY_F64(LogF64, "vm.log.f64");
    IREE_VM_ISA_EMIT_OP_EXT_F64_UNARY_F64(Log10F64, "vm.log10.f64");
    IREE_VM_ISA_EMIT_OP_EXT_F64_UNARY_F64(Log1pF64, "vm.log1p.f64");
    IREE_VM_ISA_EMIT_OP_EXT_F64_UNARY_F64(Log2F64, "vm.log2.f64");
    IREE_VM_ISA_EMIT_OP_EXT_F64_BINARY_F64(PowF64, "vm.pow.f64");
    IREE_VM_ISA_EMIT_OP_EXT_F64_UNARY_F64(RsqrtF64, "vm.rsqrt.f64");
    IREE_VM_ISA_EMIT_OP_EXT_F64_UNARY_F64(SqrtF64, "vm.sqrt.f64");
    IREE_VM_ISA_EMIT_OP_EXT_F64_UNARY_F64(TanhF64, "vm.tanh.f64");
    IREE_VM_ISA_EMIT_OP_EXT_F64_UNARY_F64(ErfF64, "vm.erf.f64");

    //===----------------------------------------------------------------===//
    // ExtF64: Casting and type conversion/emulation
    //===----------------------------------------------------------------===//

    IREE_VM_ISA_EMIT_OP(EXT_F64, TruncF64F32) {
      IREE_VM_ISA_DECODE_OPERAND_F64(operand_reg);
      IREE_VM_ISA_DECODE_RESULT_F32(result_reg);
      IREE_VM_ISA_EMIT_F32_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.trunc.f64.f32 "));
      IREE_VM_ISA_EMIT_F64_REG_NAME(operand_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_F64(regs->i32[operand_reg]);
      break;
    }
    IREE_VM_ISA_EMIT_OP(EXT_F64, ExtF32F64) {
      IREE_VM_ISA_DECODE_OPERAND_F32(operand_reg);
      IREE_VM_ISA_DECODE_RESULT_F64(result_reg);
      IREE_VM_ISA_EMIT_F64_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.ext.f32.f64 "));
      IREE_VM_ISA_EMIT_F32_REG_NAME(operand_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_F32(regs->i32[operand_reg]);
      break;
    }
    IREE_VM_ISA_EMIT_OP(EXT_F64, CastSI32F64) {
      IREE_VM_ISA_DECODE_OPERAND_I32(operand_reg);
      IREE_VM_ISA_DECODE_RESULT_F64(result_reg);
      IREE_VM_ISA_EMIT_F64_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.cast.si32.f64 "));
      IREE_VM_ISA_EMIT_I32_REG_NAME(operand_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I32(regs->i32[operand_reg]);
      break;
    }
    IREE_VM_ISA_EMIT_OP(EXT_F64, CastUI32F64) {
      IREE_VM_ISA_DECODE_OPERAND_I32(operand_reg);
      IREE_VM_ISA_DECODE_RESULT_F64(result_reg);
      IREE_VM_ISA_EMIT_F64_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.cast.ui32.f64 "));
      IREE_VM_ISA_EMIT_I32_REG_NAME(operand_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I32(regs->i32[operand_reg]);
      break;
    }
    IREE_VM_ISA_EMIT_OP(EXT_F64, CastF64SI32) {
      IREE_VM_ISA_DECODE_OPERAND_F64(operand_reg);
      IREE_VM_ISA_DECODE_RESULT_I32(result_reg);
      IREE_VM_ISA_EMIT_I32_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.cast.f64.si32 "));
      IREE_VM_ISA_EMIT_F64_REG_NAME(operand_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_F64(regs->i32[operand_reg]);
      break;
    }
    IREE_VM_ISA_EMIT_OP(EXT_F64, CastF64UI32) {
      IREE_VM_ISA_DECODE_OPERAND_F64(operand_reg);
      IREE_VM_ISA_DECODE_RESULT_I32(result_reg);
      IREE_VM_ISA_EMIT_I32_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.cast.f64.ui32 "));
      IREE_VM_ISA_EMIT_F64_REG_NAME(operand_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_F64(regs->i32[operand_reg]);
      break;
    }
    IREE_VM_ISA_EMIT_OP(EXT_F64, CastSI64F64) {
      IREE_VM_ISA_DECODE_OPERAND_I64(operand_reg);
      IREE_VM_ISA_DECODE_RESULT_F64(result_reg);
      IREE_VM_ISA_EMIT_F64_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.cast.si64.f64 "));
      IREE_VM_ISA_EMIT_I64_REG_NAME(operand_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I64(regs->i32[operand_reg]);
      break;
    }
    IREE_VM_ISA_EMIT_OP(EXT_F64, CastUI64F64) {
      IREE_VM_ISA_DECODE_OPERAND_I64(operand_reg);
      IREE_VM_ISA_DECODE_RESULT_F64(result_reg);
      IREE_VM_ISA_EMIT_F64_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.cast.ui64.f64 "));
      IREE_VM_ISA_EMIT_I64_REG_NAME(operand_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I64(regs->i32[operand_reg]);
      break;
    }
    IREE_VM_ISA_EMIT_OP(EXT_F64, CastF64SI64) {
      IREE_VM_ISA_DECODE_OPERAND_F64(operand_reg);
      IREE_VM_ISA_DECODE_RESULT_I64(result_reg);
      IREE_VM_ISA_EMIT_I64_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.cast.f64.si64 "));
      IREE_VM_ISA_EMIT_F64_REG_NAME(operand_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_F64(regs->i32[operand_reg]);
      break;
    }
    IREE_VM_ISA_EMIT_OP(EXT_F64, CastF64UI64) {
      IREE_VM_ISA_DECODE_OPERAND_F64(operand_reg);
      IREE_VM_ISA_DECODE_RESULT_I64(result_reg);
      IREE_VM_ISA_EMIT_I64_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.cast.f64.ui64 "));
      IREE_VM_ISA_EMIT_F64_REG_NAME(operand_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_F64(regs->i32[operand_reg]);
      break;
    }
    IREE_VM_ISA_EMIT_OP(EXT_F64, BitcastI64F64) {
      IREE_VM_ISA_DECODE_OPERAND_I64(operand_reg);
      IREE_VM_ISA_DECODE_RESULT_F64(result_reg);
      IREE_VM_ISA_EMIT_F64_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.bitcast.i64.f64 "));
      IREE_VM_ISA_EMIT_I32_REG_NAME(operand_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I64(regs->i32[operand_reg]);
      break;
    }
    IREE_VM_ISA_EMIT_OP(EXT_F64, BitcastF64I64) {
      IREE_VM_ISA_DECODE_OPERAND_F64(operand_reg);
      IREE_VM_ISA_DECODE_RESULT_I64(result_reg);
      IREE_VM_ISA_EMIT_I64_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.bitcast.f64.i64 "));
      IREE_VM_ISA_EMIT_F64_REG_NAME(operand_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_F64(regs->i32[operand_reg]);
      break;
    }

    //===----------------------------------------------------------------===//
    // ExtF64: Comparison ops
    //===----------------------------------------------------------------===//

#define IREE_VM_ISA_EMIT_OP_EXT_F64_CMP_F64(op_name, op_mnemonic)      \
  IREE_VM_ISA_EMIT_OP(EXT_F64, op_name) {                              \
    IREE_VM_ISA_DECODE_OPERAND_F64(lhs_reg);                           \
    IREE_VM_ISA_DECODE_OPERAND_F64(rhs_reg);                           \
    IREE_VM_ISA_DECODE_RESULT_I32(result_reg);                         \
    IREE_VM_ISA_EMIT_I32_REG_NAME(result_reg);                         \
    IREE_RETURN_IF_ERROR(                                              \
        iree_string_builder_append_format(b, " = %s ", op_mnemonic));  \
    IREE_VM_ISA_EMIT_F64_REG_NAME(lhs_reg);                            \
    IREE_VM_ISA_EMIT_OPTIONAL_VALUE_F64(regs->i32[lhs_reg]);           \
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", ")); \
    IREE_VM_ISA_EMIT_F64_REG_NAME(rhs_reg);                            \
    IREE_VM_ISA_EMIT_OPTIONAL_VALUE_F64(regs->i32[rhs_reg]);           \
    break;                                                             \
  }

    IREE_VM_ISA_EMIT_OP_EXT_F64_CMP_F64(CmpEQF64O, "vm.cmp.eq.f64.o");
    IREE_VM_ISA_EMIT_OP_EXT_F64_CMP_F64(CmpEQF64U, "vm.cmp.eq.f64.u");
    IREE_VM_ISA_EMIT_OP_EXT_F64_CMP_F64(CmpNEF64O, "vm.cmp.ne.f64.o");
    IREE_VM_ISA_EMIT_OP_EXT_F64_CMP_F64(CmpNEF64U, "vm.cmp.ne.f64.u");
    IREE_VM_ISA_EMIT_OP_EXT_F64_CMP_F64(CmpLTF64O, "vm.cmp.lt.f64.o");
    IREE_VM_ISA_EMIT_OP_EXT_F64_CMP_F64(CmpLTF64U, "vm.cmp.lt.f64.u");
    IREE_VM_ISA_EMIT_OP_EXT_F64_CMP_F64(CmpLTEF64O, "vm.cmp.lte.f64.o");
    IREE_VM_ISA_EMIT_OP_EXT_F64_CMP_F64(CmpLTEF64U, "vm.cmp.lte.f64.u");
    IREE_VM_ISA_EMIT_OP(EXT_F64, CmpNaNF64) {
      IREE_VM_ISA_DECODE_OPERAND_F64(operand_reg);
      IREE_VM_ISA_DECODE_RESULT_I32(result_reg);
      IREE_VM_ISA_EMIT_I32_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.cmp.nan.f64 "));
      IREE_VM_ISA_EMIT_F64_REG_NAME(operand_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_F64(regs->i32[operand_reg]);
      break;
    }

    //===----------------------------------------------------------------===//
    // ExtF64: Buffers
    //===----------------------------------------------------------------===//

    IREE_VM_ISA_EMIT_OP(EXT_F64, BufferFillF64) {
      IREE_VM_ISA_DECODE_OPERAND_REF(buffer_reg);
      IREE_VM_ISA_DECODE_OPERAND_I64(offset_reg);
      IREE_VM_ISA_DECODE_OPERAND_I64(length_reg);
      IREE_VM_ISA_DECODE_OPERAND_F64(value_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, "vm.buffer.fill.f64 "));
      IREE_VM_ISA_EMIT_REF_REG_NAME(buffer_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_REF(&regs->ref[buffer_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      IREE_VM_ISA_EMIT_I64_REG_NAME(offset_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I64(regs->i32[offset_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      IREE_VM_ISA_EMIT_I64_REG_NAME(length_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I64(regs->i32[length_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      IREE_VM_ISA_EMIT_F64_REG_NAME(value_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_F64(regs->i32[value_reg]);
      break;
    }

    IREE_VM_ISA_EMIT_OP(EXT_F64, BufferLoadF64) {
      IREE_VM_ISA_DECODE_OPERAND_REF(buffer_reg);
      IREE_VM_ISA_DECODE_OPERAND_I64(offset_reg);
      IREE_VM_ISA_DECODE_RESULT_F64(result_reg);
      IREE_VM_ISA_EMIT_F64_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.buffer.load.f64 "));
      IREE_VM_ISA_EMIT_REF_REG_NAME(buffer_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_REF(&regs->ref[buffer_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      IREE_VM_ISA_EMIT_I64_REG_NAME(offset_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I64(regs->i32[offset_reg]);
      break;
    }

    IREE_VM_ISA_EMIT_OP(EXT_F64, BufferStoreF64) {
      IREE_VM_ISA_DECODE_OPERAND_REF(buffer_reg);
      IREE_VM_ISA_DECODE_OPERAND_I64(offset_reg);
      IREE_VM_ISA_DECODE_OPERAND_F64(value_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, "vm.buffer.store.f64 "));
      IREE_VM_ISA_EMIT_F64_REG_NAME(value_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_F64(regs->i32[value_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      IREE_VM_ISA_EMIT_REF_REG_NAME(buffer_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_REF(&regs->ref[buffer_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      IREE_VM_ISA_EMIT_I64_REG_NAME(offset_reg);
      IREE_VM_ISA_EMIT_OPTIONAL_VALUE_I64(regs->i32[offset_reg]);
      break;
    }

    IREE_VM_ISA_END_DISASM_PREFIX()
#else
    IREE_VM_ISA_UNHANDLED_DISASM_PREFIX(PrefixExtF64, EXT_F64)
#endif  // IREE_VM_EXT_F64_ENABLE

    default:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "unhandled core opcode");
  }
  if (out_next_pc) *out_next_pc = pc;
  return iree_ok_status();
}

iree_status_t iree_vm_bytecode_disassemble_op(
    iree_vm_bytecode_module_t* module,
    iree_vm_bytecode_module_state_t* module_state, uint16_t function_ordinal,
    iree_vm_source_offset_t pc, const iree_vm_registers_t* regs,
    iree_vm_bytecode_disassembly_format_t format, iree_string_builder_t* b) {
  return iree_vm_bytecode_disassemble_op_impl(module, module_state,
                                              function_ordinal, pc, regs,
                                              format, b, /*out_next_pc=*/NULL);
}

iree_status_t iree_vm_bytecode_trace_disassembly(
    iree_vm_stack_frame_t* frame, iree_vm_source_offset_t pc,
    const iree_vm_registers_t* regs, FILE* file) {
  IREE_ALLOCATOR_INLINE_STORAGE(inline_storage, 2048);
  iree_string_builder_t b;
  iree_string_builder_initialize(
      iree_allocator_inline_arena(&inline_storage.header), &b);

  // TODO(benvanik): ensure frame is in-sync before call or restore original.
  // It's shady to manipulate the frame here but I know we expect the pc to be
  // valid only on entry/exit from a function.
  frame->pc = pc;

#if IREE_VM_EXECUTION_TRACING_SRC_LOC_ENABLE
  iree_vm_source_location_t source_location;
  iree_status_t status = iree_vm_module_resolve_source_location(
      frame->function.module, frame->function, pc, &source_location);
  if (iree_status_is_ok(status)) {
    status = iree_vm_source_location_format(
        &source_location, IREE_VM_SOURCE_LOCATION_FORMAT_FLAG_SINGLE_LINE, &b);
  }
  if (iree_status_is_ok(status)) {
    // Pad out to keep alignment. This is just guesswork based on my machine.
    static const iree_host_size_t pad_to = 80;
    iree_host_size_t col = iree_string_builder_size(&b);
    if (col < pad_to) {
      iree_string_builder_append_format(&b, "%*s ", (int)(pad_to - col), "");
    } else {
      status = iree_string_builder_append_cstring(&b, " ");
    }
  } else {
    // Ignore failures when no source location is available.
    if (iree_status_is_unavailable(status)) {
      status = iree_ok_status();
    } else {
      return status;
    }
  }
#else
  iree_status_t status = iree_ok_status();
#endif  // IREE_VM_EXECUTION_TRACING_ENABLE

  if (iree_status_is_ok(status)) {
    iree_string_view_t module_name =
        iree_vm_module_name(frame->function.module);
    status = iree_string_builder_append_format(
        &b, "[%.*s", (int)module_name.size, module_name.data);
  }
  if (iree_status_is_ok(status)) {
    iree_string_view_t function_name = iree_vm_function_name(&frame->function);
    if (iree_string_view_is_empty(function_name)) {
      status = iree_string_builder_append_format(
          &b, "@%u", (uint32_t)frame->function.ordinal);
    } else {
      status = iree_string_builder_append_format(
          &b, ".%.*s", (int)function_name.size, function_name.data);
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_format(&b, "+%08" PRIX64 "]    ", pc);
  }

  if (iree_status_is_ok(status)) {
    status = iree_vm_bytecode_disassemble_op(
        (iree_vm_bytecode_module_t*)frame->function.module,
        (iree_vm_bytecode_module_state_t*)frame->module_state,
        frame->function.ordinal, pc, regs,
        IREE_VM_BYTECODE_DISASSEMBLY_FORMAT_INLINE_VALUES, &b);
  }

  if (iree_status_is_ok(status)) {
    fprintf(file, "%.*s\n", (int)iree_string_builder_size(&b),
            iree_string_builder_buffer(&b));
  } else {
    fprintf(file, "<<disassembly failed>>\n");
  }

  iree_string_builder_deinitialize(&b);
  return status;
}

iree_status_t iree_vm_bytecode_disassemble_function(
    iree_vm_bytecode_module_t* module,
    iree_vm_bytecode_module_state_t* module_state, uint16_t function_ordinal,
    iree_vm_bytecode_disassembly_format_t format,
    iree_string_builder_t* string_builder) {
  if (function_ordinal >= module->function_descriptor_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "function ordinal %u out of range (0 < %u < %zu)",
                            function_ordinal, function_ordinal,
                            module->function_descriptor_count);
  }

  const iree_vm_FunctionDescriptor_t* descriptor =
      &module->function_descriptor_table[function_ordinal];
  uint32_t bytecode_length = descriptor->bytecode_length;

  // Iterate through bytecode.
  iree_vm_source_offset_t pc = 0;
  while (pc < bytecode_length) {
    // Emit prefix: [module.function+PC]
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        string_builder, "[%08" PRIX64 "]    ", pc));

    // Disassemble the op and get the next PC.
    iree_vm_source_offset_t next_pc = 0;
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_disassemble_op_impl(
        module, module_state, function_ordinal, pc, /*regs=*/NULL, format,
        string_builder, &next_pc));

    // Append newline.
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(string_builder, "\n"));

    pc = next_pc;
  }

  return iree_ok_status();
}
