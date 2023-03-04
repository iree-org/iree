// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/bytecode/disassembler.h"

#include <inttypes.h>

#include "iree/vm/ops.h"

#define BEGIN_DISASM_PREFIX(op_name, ext) \
  case IREE_VM_OP_CORE_##op_name: {       \
    switch (bytecode_data[pc++]) {
#define END_DISASM_PREFIX()                            \
  default:                                             \
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED, \
                            "unhandled ext opcode");   \
    }                                                  \
    break;                                             \
    }
#define UNHANDLED_DISASM_PREFIX(op_name, ext)                      \
  case IREE_VM_OP_CORE_##op_name: {                                \
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,             \
                            "unhandled dispatch extension " #ext); \
  }

#define DISASM_OP(ext, op_name) case IREE_VM_OP_##ext##_##op_name:

#define VM_ParseConstI8(name) \
  OP_I8(0);                   \
  ++pc;
#define VM_ParseConstI32(name) \
  OP_I32(0);                   \
  pc += 4;
#define VM_ParseConstI64(name) \
  OP_I64(0);                   \
  pc += 8;
#define VM_ParseConstF32(name) \
  OP_F32(0);                   \
  pc += 4;
#define VM_ParseConstF64(name) \
  OP_F64(0);                   \
  pc += 8;
#define VM_ParseOpcode(opcode) VM_ParseConstI8(#opcode)
#define VM_ParseFuncAttr(name) VM_ParseConstI32(name)
#define VM_ParseGlobalAttr(name) VM_ParseConstI32(name)
#define VM_ParseRodataAttr(name) VM_ParseConstI32(name)
#define VM_ParseType(name)             \
  iree_vm_map_type(module, OP_I32(0)); \
  pc += 4;
#define VM_ParseTypeOf(name) VM_ParseType(name)
#define VM_ParseIntAttr32(name) VM_ParseConstI32(name)
#define VM_ParseIntAttr64(name) VM_ParseConstI64(name)
#define VM_ParseFloatAttr32(name) VM_ParseConstF32(name)
#define VM_ParseFloatAttr64(name) VM_ParseConstF64(name)
#define VM_ParseStrAttr(name, out_str)                   \
  (out_str)->size = (iree_host_size_t)OP_I16(0);         \
  (out_str)->data = (const char*)&bytecode_data[pc + 2]; \
  pc += 2 + (out_str)->size;
#define VM_ParseBranchTarget(block_name) VM_ParseConstI32(name)
#define VM_ParseBranchOperands(operands_name) \
  VM_DecBranchOperandsImpl(bytecode_data, &pc)
#define VM_ParseOperandRegI32(name) \
  OP_I16(0);                        \
  pc += IREE_REGISTER_ORDINAL_SIZE;
#define VM_ParseOperandRegI64(name) \
  OP_I16(0);                        \
  pc += IREE_REGISTER_ORDINAL_SIZE;
#define VM_ParseOperandRegF32(name) \
  OP_I16(0);                        \
  pc += IREE_REGISTER_ORDINAL_SIZE;
#define VM_ParseOperandRegF64(name) \
  OP_I16(0);                        \
  pc += IREE_REGISTER_ORDINAL_SIZE;
#define VM_ParseOperandRegRef(name, out_is_move)                    \
  OP_I16(0) & IREE_REF_REGISTER_MASK;                               \
  *(out_is_move) = 0; /*= OP_I16(0) & IREE_REF_REGISTER_MOVE_BIT;*/ \
  pc += IREE_REGISTER_ORDINAL_SIZE;
#define VM_ParseVariadicOperands(name) \
  VM_DecVariadicOperandsImpl(bytecode_data, &pc)
#define VM_ParseResultRegI32(name) \
  OP_I16(0);                       \
  pc += IREE_REGISTER_ORDINAL_SIZE;
#define VM_ParseResultRegI64(name) \
  OP_I16(0);                       \
  pc += IREE_REGISTER_ORDINAL_SIZE;
#define VM_ParseResultRegF32(name) \
  OP_I16(0);                       \
  pc += IREE_REGISTER_ORDINAL_SIZE;
#define VM_ParseResultRegF64(name) \
  OP_I16(0);                       \
  pc += IREE_REGISTER_ORDINAL_SIZE;
#define VM_ParseResultRegRef(name, out_is_move)                     \
  OP_I16(0) & IREE_REF_REGISTER_MASK;                               \
  *(out_is_move) = 0; /*= OP_I16(0) & IREE_REF_REGISTER_MOVE_BIT;*/ \
  pc += IREE_REGISTER_ORDINAL_SIZE;
#define VM_ParseVariadicResults(name) VM_ParseVariadicOperands(name)

#define EMIT_REG_NAME(reg)                \
  if ((reg)&IREE_REF_REGISTER_TYPE_BIT) { \
    EMIT_REF_REG_NAME(reg);               \
  } else {                                \
    EMIT_I32_REG_NAME(reg);               \
  }
#define EMIT_I32_REG_NAME(reg)                            \
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format( \
      b, "%%i%u", ((reg)&IREE_I32_REGISTER_MASK)));
#define EMIT_I64_REG_NAME(reg)                            \
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format( \
      b, "%%i%u:%u", ((reg)&IREE_I32_REGISTER_MASK),      \
      ((reg)&IREE_I32_REGISTER_MASK) + 1));
#define EMIT_F32_REG_NAME(reg) EMIT_I32_REG_NAME(reg)
#define EMIT_REF_REG_NAME(reg)                            \
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format( \
      b, "%%r%u", ((reg)&IREE_REF_REGISTER_MASK)));

#define EMIT_REG_VALUE(regs, reg)                                           \
  if ((reg)&IREE_REF_REGISTER_TYPE_BIT) {                                   \
    iree_vm_ref_t* ref = &(regs)->ref[(reg)&IREE_REF_REGISTER_MASK];        \
    if (iree_vm_ref_is_null(ref)) {                                         \
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, "null"));  \
    } else {                                                                \
      iree_string_view_t type_name = iree_vm_ref_type_name(ref->type);      \
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(               \
          b, "!%.*s/0x%p", (int)type_name.size, type_name.data, ref->ptr)); \
    }                                                                       \
  } else {                                                                  \
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(                 \
        b, "%u", ((regs)->i32[(reg)&IREE_I32_REGISTER_MASK])));             \
  }

static iree_status_t iree_vm_bytecode_disassembler_emit_type_name(
    const iree_vm_type_def_t* type_def, iree_string_builder_t* b) {
  if (iree_vm_type_def_is_value(type_def)) {
    const char* type_name;
    switch (type_def->value_type) {
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
    iree_string_view_t type_name = iree_vm_ref_type_name(type_def->ref_type);
    return iree_string_builder_append_format(b, "%.*s", (int)type_name.size,
                                             type_name.data);
  } else {
    return iree_string_builder_append_cstring(b, "*");
  }
}
#define EMIT_TYPE_NAME(type_def) \
  iree_vm_bytecode_disassembler_emit_type_name(type_def, b);

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
    EMIT_REG_NAME(reg);
    if (include_values) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, "("));
      EMIT_REG_VALUE(regs, reg);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ")"));
    }
  }
  return iree_ok_status();
}
#define EMIT_OPERAND_REG_LIST(reg_list) \
  iree_vm_bytecode_disassembler_emit_operand_list(regs, reg_list, format, b)
static iree_status_t iree_vm_bytecode_disassembler_emit_result_list(
    const iree_vm_register_list_t* list,
    iree_vm_bytecode_disassembly_format_t format, iree_string_builder_t* b) {
  for (uint16_t i = 0; i < list->size; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
    }
    uint16_t reg = list->registers[i];
    EMIT_REG_NAME(reg);
  }
  return iree_ok_status();
}
#define EMIT_RESULT_REG_LIST(reg_list) \
  iree_vm_bytecode_disassembler_emit_result_list(reg_list, format, b)
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
    EMIT_REG_NAME(remap_list->pairs[i].src_reg);
    if (include_values) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, "("));
      EMIT_REG_VALUE(regs, remap_list->pairs[i].src_reg);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ")"));
    }
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, "->"));
    EMIT_REG_NAME(remap_list->pairs[i].dst_reg);
  }
  return iree_ok_status();
}
#define EMIT_REMAP_LIST(remap_list) \
  iree_vm_bytecode_disassembler_emit_remap_list(regs, remap_list, format, b)

#define EMIT_OPTIONAL_VALUE_I32(expr)                                          \
  if (regs && (format & IREE_VM_BYTECODE_DISASSEMBLY_FORMAT_INLINE_VALUES)) {  \
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(b, "(%" PRId32 ")", \
                                                           (int32_t)(expr)));  \
  }
#define EMIT_OPTIONAL_VALUE_I64(expr)                                         \
  if (regs && (format & IREE_VM_BYTECODE_DISASSEMBLY_FORMAT_INLINE_VALUES)) { \
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(                   \
        b, "(%" PRId64 ")", *(int64_t*)&(expr)));                             \
  }
#define EMIT_OPTIONAL_VALUE_F32(expr)                                         \
  if (regs && (format & IREE_VM_BYTECODE_DISASSEMBLY_FORMAT_INLINE_VALUES)) { \
    IREE_RETURN_IF_ERROR(                                                     \
        iree_string_builder_append_format(b, "(%f)", *(float*)&(expr)));      \
  }
#define EMIT_OPTIONAL_VALUE_F64(expr)                                         \
  if (regs && (format & IREE_VM_BYTECODE_DISASSEMBLY_FORMAT_INLINE_VALUES)) { \
    IREE_RETURN_IF_ERROR(                                                     \
        iree_string_builder_append_format(b, "(%f)", *(double*)&(expr)));     \
  }
#define EMIT_OPTIONAL_VALUE_REF(expr)                                         \
  if (regs && (format & IREE_VM_BYTECODE_DISASSEMBLY_FORMAT_INLINE_VALUES)) { \
    iree_vm_ref_t* ref = (expr);                                              \
    if (iree_vm_ref_is_null(ref)) {                                           \
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, "(null)"));  \
    } else {                                                                  \
      iree_string_view_t type_name = iree_vm_ref_type_name(ref->type);        \
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(                 \
          b, "(!%.*s/0x%p)", (int)type_name.size, type_name.data, ref->ptr)); \
    }                                                                         \
  }

#define DISASM_OP_CORE_UNARY_I32(op_name, op_mnemonic)                \
  DISASM_OP(CORE, op_name) {                                          \
    uint16_t operand_reg = VM_ParseOperandRegI32("operand");          \
    uint16_t result_reg = VM_ParseResultRegI32("result");             \
    EMIT_I32_REG_NAME(result_reg);                                    \
    IREE_RETURN_IF_ERROR(                                             \
        iree_string_builder_append_format(b, " = %s ", op_mnemonic)); \
    EMIT_I32_REG_NAME(operand_reg);                                   \
    EMIT_OPTIONAL_VALUE_I32(regs->i32[operand_reg]);                  \
    break;                                                            \
  }

#define DISASM_OP_CORE_BINARY_I32(op_name, op_mnemonic)                \
  DISASM_OP(CORE, op_name) {                                           \
    uint16_t lhs_reg = VM_ParseOperandRegI32("lhs");                   \
    uint16_t rhs_reg = VM_ParseOperandRegI32("rhs");                   \
    uint16_t result_reg = VM_ParseResultRegI32("result");              \
    EMIT_I32_REG_NAME(result_reg);                                     \
    IREE_RETURN_IF_ERROR(                                              \
        iree_string_builder_append_format(b, " = %s ", op_mnemonic));  \
    EMIT_I32_REG_NAME(lhs_reg);                                        \
    EMIT_OPTIONAL_VALUE_I32(regs->i32[lhs_reg]);                       \
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", ")); \
    EMIT_I32_REG_NAME(rhs_reg);                                        \
    EMIT_OPTIONAL_VALUE_I32(regs->i32[rhs_reg]);                       \
    break;                                                             \
  }

#define DISASM_OP_CORE_TERNARY_I32(op_name, op_mnemonic)               \
  DISASM_OP(CORE, op_name) {                                           \
    uint16_t a_reg = VM_ParseOperandRegI32("a");                       \
    uint16_t b_reg = VM_ParseOperandRegI32("b");                       \
    uint16_t c_reg = VM_ParseOperandRegI32("c");                       \
    uint16_t result_reg = VM_ParseResultRegI32("result");              \
    EMIT_I32_REG_NAME(result_reg);                                     \
    IREE_RETURN_IF_ERROR(                                              \
        iree_string_builder_append_format(b, " = %s ", op_mnemonic));  \
    EMIT_I32_REG_NAME(a_reg);                                          \
    EMIT_OPTIONAL_VALUE_I32(regs->i32[a_reg]);                         \
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", ")); \
    EMIT_I32_REG_NAME(b_reg);                                          \
    EMIT_OPTIONAL_VALUE_I32(regs->i32[b_reg]);                         \
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", ")); \
    EMIT_I32_REG_NAME(c_reg);                                          \
    EMIT_OPTIONAL_VALUE_I32(regs->i32[c_reg]);                         \
    break;                                                             \
  }

#define DISASM_OP_CORE_UNARY_I64(op_name, op_mnemonic)                \
  DISASM_OP(CORE, op_name) {                                          \
    uint16_t operand_reg = VM_ParseOperandRegI64("operand");          \
    uint16_t result_reg = VM_ParseResultRegI64("result");             \
    EMIT_I64_REG_NAME(result_reg);                                    \
    IREE_RETURN_IF_ERROR(                                             \
        iree_string_builder_append_format(b, " = %s ", op_mnemonic)); \
    EMIT_I64_REG_NAME(operand_reg);                                   \
    EMIT_OPTIONAL_VALUE_I64(regs->i32[operand_reg]);                  \
    break;                                                            \
  }

#define DISASM_OP_CORE_BINARY_I64(op_name, op_mnemonic)                \
  DISASM_OP(CORE, op_name) {                                           \
    uint16_t lhs_reg = VM_ParseOperandRegI64("lhs");                   \
    uint16_t rhs_reg = VM_ParseOperandRegI64("rhs");                   \
    uint16_t result_reg = VM_ParseResultRegI64("result");              \
    EMIT_I64_REG_NAME(result_reg);                                     \
    IREE_RETURN_IF_ERROR(                                              \
        iree_string_builder_append_format(b, " = %s ", op_mnemonic));  \
    EMIT_I64_REG_NAME(lhs_reg);                                        \
    EMIT_OPTIONAL_VALUE_I64(regs->i32[lhs_reg]);                       \
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", ")); \
    EMIT_I64_REG_NAME(rhs_reg);                                        \
    EMIT_OPTIONAL_VALUE_I64(regs->i32[rhs_reg]);                       \
    break;                                                             \
  }

#define DISASM_OP_CORE_TERNARY_I64(op_name, op_mnemonic)               \
  DISASM_OP(CORE, op_name) {                                           \
    uint16_t a_reg = VM_ParseOperandRegI64("a");                       \
    uint16_t b_reg = VM_ParseOperandRegI64("b");                       \
    uint16_t c_reg = VM_ParseOperandRegI64("c");                       \
    uint16_t result_reg = VM_ParseResultRegI64("result");              \
    EMIT_I64_REG_NAME(result_reg);                                     \
    IREE_RETURN_IF_ERROR(                                              \
        iree_string_builder_append_format(b, " = %s ", op_mnemonic));  \
    EMIT_I64_REG_NAME(a_reg);                                          \
    EMIT_OPTIONAL_VALUE_I64(regs->i32[a_reg]);                         \
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", ")); \
    EMIT_I64_REG_NAME(b_reg);                                          \
    EMIT_OPTIONAL_VALUE_I64(regs->i32[b_reg]);                         \
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", ")); \
    EMIT_I64_REG_NAME(c_reg);                                          \
    EMIT_OPTIONAL_VALUE_I64(regs->i32[c_reg]);                         \
    break;                                                             \
  }

#define DISASM_OP_EXT_F32_UNARY_F32(op_name, op_mnemonic)             \
  DISASM_OP(EXT_F32, op_name) {                                       \
    uint16_t operand_reg = VM_ParseOperandRegF32("operand");          \
    uint16_t result_reg = VM_ParseResultRegF32("result");             \
    EMIT_F32_REG_NAME(result_reg);                                    \
    IREE_RETURN_IF_ERROR(                                             \
        iree_string_builder_append_format(b, " = %s ", op_mnemonic)); \
    EMIT_F32_REG_NAME(operand_reg);                                   \
    EMIT_OPTIONAL_VALUE_F32(regs->i32[operand_reg]);                  \
    break;                                                            \
  }

#define DISASM_OP_EXT_F32_BINARY_F32(op_name, op_mnemonic)             \
  DISASM_OP(EXT_F32, op_name) {                                        \
    uint16_t lhs_reg = VM_ParseOperandRegF32("lhs");                   \
    uint16_t rhs_reg = VM_ParseOperandRegF32("rhs");                   \
    uint16_t result_reg = VM_ParseResultRegF32("result");              \
    EMIT_F32_REG_NAME(result_reg);                                     \
    IREE_RETURN_IF_ERROR(                                              \
        iree_string_builder_append_format(b, " = %s ", op_mnemonic));  \
    EMIT_F32_REG_NAME(lhs_reg);                                        \
    EMIT_OPTIONAL_VALUE_F32(regs->i32[lhs_reg]);                       \
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", ")); \
    EMIT_F32_REG_NAME(rhs_reg);                                        \
    EMIT_OPTIONAL_VALUE_F32(regs->i32[rhs_reg]);                       \
    break;                                                             \
  }

#define DISASM_OP_EXT_F32_TERNARY_F32(op_name, op_mnemonic)            \
  DISASM_OP(EXT_F32, op_name) {                                        \
    uint16_t a_reg = VM_ParseOperandRegF32("a");                       \
    uint16_t b_reg = VM_ParseOperandRegF32("b");                       \
    uint16_t c_reg = VM_ParseOperandRegF32("c");                       \
    uint16_t result_reg = VM_ParseResultRegF32("result");              \
    EMIT_F32_REG_NAME(result_reg);                                     \
    IREE_RETURN_IF_ERROR(                                              \
        iree_string_builder_append_format(b, " = %s ", op_mnemonic));  \
    EMIT_F32_REG_NAME(a_reg);                                          \
    EMIT_OPTIONAL_VALUE_F32(regs->i32[a_reg]);                         \
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", ")); \
    EMIT_F32_REG_NAME(b_reg);                                          \
    EMIT_OPTIONAL_VALUE_F32(regs->i32[b_reg]);                         \
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", ")); \
    EMIT_F32_REG_NAME(c_reg);                                          \
    EMIT_OPTIONAL_VALUE_F32(regs->i32[c_reg]);                         \
    break;                                                             \
  }

iree_status_t iree_vm_bytecode_disassemble_op(
    iree_vm_bytecode_module_t* module,
    iree_vm_bytecode_module_state_t* module_state, uint16_t function_ordinal,
    iree_vm_source_offset_t pc, const iree_vm_registers_t* regs,
    iree_vm_bytecode_disassembly_format_t format, iree_string_builder_t* b) {
  const uint8_t* IREE_RESTRICT bytecode_data =
      module->bytecode_data.data +
      module->function_descriptor_table[function_ordinal].bytecode_offset;

  switch (bytecode_data[pc++]) {
    //===------------------------------------------------------------------===//
    // Globals
    //===------------------------------------------------------------------===//

    DISASM_OP(CORE, GlobalLoadI32) {
      uint32_t byte_offset = VM_ParseGlobalAttr("global");
      uint16_t value_reg = VM_ParseResultRegI32("value");
      EMIT_I32_REG_NAME(value_reg);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          b, " = vm.global.load.i32 .rwdata[%u]", byte_offset));
      EMIT_OPTIONAL_VALUE_I32(
          vm_global_load_i32(module_state->rwdata_storage.data, byte_offset));
      break;
    }

    DISASM_OP(CORE, GlobalStoreI32) {
      uint32_t byte_offset = VM_ParseGlobalAttr("global");
      uint16_t value_reg = VM_ParseOperandRegI32("value");
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_format(b, "vm.global.store.i32 "));
      EMIT_I32_REG_NAME(value_reg);
      EMIT_OPTIONAL_VALUE_I32(regs->i32[value_reg]);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_format(b, ", .rwdata[%u]", byte_offset));
      break;
    }

    DISASM_OP(CORE, GlobalLoadIndirectI32) {
      uint16_t byte_offset_reg = VM_ParseOperandRegI32("global");
      uint16_t value_reg = VM_ParseResultRegI32("value");
      EMIT_I32_REG_NAME(value_reg);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
          b, " = vm.global.load.indirect.i32 .rwdata["));
      EMIT_I32_REG_NAME(byte_offset_reg);
      EMIT_OPTIONAL_VALUE_I32(regs->i32[byte_offset_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, "]"));
      EMIT_OPTIONAL_VALUE_I32(vm_global_load_i32(
          module_state->rwdata_storage.data, regs->i32[byte_offset_reg]));
      break;
    }

    DISASM_OP(CORE, GlobalStoreIndirectI32) {
      uint16_t byte_offset_reg = VM_ParseOperandRegI32("global");
      uint16_t value_reg = VM_ParseOperandRegI32("value");
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
          b, "vm.global.store.indirect.i32 "));
      EMIT_I32_REG_NAME(value_reg);
      EMIT_OPTIONAL_VALUE_I32(regs->i32[value_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", .rwdata["));
      EMIT_I32_REG_NAME(byte_offset_reg);
      EMIT_OPTIONAL_VALUE_I32(regs->i32[byte_offset_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, "]"));
      break;
    }

    DISASM_OP(CORE, GlobalLoadI64) {
      uint32_t byte_offset = VM_ParseGlobalAttr("global");
      uint16_t value_reg = VM_ParseResultRegI64("value");
      EMIT_I32_REG_NAME(value_reg);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          b, " = vm.global.load.i64 .rwdata[%u]", byte_offset));
      EMIT_OPTIONAL_VALUE_I64(module_state->rwdata_storage.data[byte_offset]);
      break;
    }

    DISASM_OP(CORE, GlobalStoreI64) {
      uint32_t byte_offset = VM_ParseGlobalAttr("global");
      uint16_t value_reg = VM_ParseOperandRegI64("value");
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_format(b, "vm.global.store.i64 "));
      EMIT_I64_REG_NAME(value_reg);
      EMIT_OPTIONAL_VALUE_I64(regs->i32[value_reg]);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_format(b, ", .rwdata[%u]", byte_offset));
      break;
    }

    DISASM_OP(CORE, GlobalLoadIndirectI64) {
      uint16_t byte_offset_reg = VM_ParseOperandRegI32("global");
      uint16_t value_reg = VM_ParseResultRegI64("value");
      EMIT_I64_REG_NAME(value_reg);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
          b, " = vm.global.load.indirect.i64 .rwdata["));
      EMIT_I32_REG_NAME(byte_offset_reg);
      EMIT_OPTIONAL_VALUE_I32(regs->i32[byte_offset_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, "]"));
      EMIT_OPTIONAL_VALUE_I64(
          module_state->rwdata_storage.data[regs->i32[byte_offset_reg]]);
      break;
    }

    DISASM_OP(CORE, GlobalStoreIndirectI64) {
      uint16_t byte_offset_reg = VM_ParseOperandRegI32("global");
      uint16_t value_reg = VM_ParseOperandRegI64("value");
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
          b, "vm.global.store.indirect.i64 "));
      EMIT_I64_REG_NAME(value_reg);
      EMIT_OPTIONAL_VALUE_I64(regs->i32[value_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", .rwdata["));
      EMIT_I32_REG_NAME(byte_offset_reg);
      EMIT_OPTIONAL_VALUE_I32(regs->i32[byte_offset_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, "]"));
      break;
    }

    DISASM_OP(CORE, GlobalLoadRef) {
      uint32_t global = VM_ParseGlobalAttr("global");
      const iree_vm_type_def_t* type_def = VM_ParseTypeOf("value");
      bool result_is_move;
      uint16_t result_reg = VM_ParseResultRegRef("value", &result_is_move);
      EMIT_REF_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          b, " = vm.global.load.ref .refs[%u]", global));
      EMIT_OPTIONAL_VALUE_REF(&module_state->global_ref_table[global]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, " : !"));
      EMIT_TYPE_NAME(type_def);
      break;
    }

    DISASM_OP(CORE, GlobalStoreRef) {
      uint32_t global = VM_ParseGlobalAttr("global");
      const iree_vm_type_def_t* type_def = VM_ParseTypeOf("value");
      bool value_is_move;
      uint16_t value_reg = VM_ParseOperandRegRef("value", &value_is_move);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, "vm.global.store.ref "));
      EMIT_REF_REG_NAME(value_reg);
      EMIT_OPTIONAL_VALUE_REF(&regs->ref[value_reg]);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_format(b, ", .refs[%u] : !", global));
      EMIT_TYPE_NAME(type_def);
      break;
    }

    DISASM_OP(CORE, GlobalLoadIndirectRef) {
      uint16_t global_reg = VM_ParseOperandRegI32("global");
      const iree_vm_type_def_t* type_def = VM_ParseTypeOf("value");
      bool result_is_move;
      uint16_t result_reg = VM_ParseResultRegRef("value", &result_is_move);
      EMIT_REF_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
          b, " = vm.global.load.indirect.ref .refs["));
      EMIT_I32_REG_NAME(global_reg);
      EMIT_OPTIONAL_VALUE_I32(regs->i32[global_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, "]"));
      EMIT_OPTIONAL_VALUE_REF(
          &module_state->global_ref_table[regs->i32[global_reg]]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, " : !"));
      EMIT_TYPE_NAME(type_def);
      break;
    }

    DISASM_OP(CORE, GlobalStoreIndirectRef) {
      uint16_t global_reg = VM_ParseOperandRegI32("global");
      const iree_vm_type_def_t* type_def = VM_ParseTypeOf("value");
      bool value_is_move;
      uint16_t value_reg = VM_ParseOperandRegRef("value", &value_is_move);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          b, "vm.global.store.indirect.ref "));
      EMIT_REF_REG_NAME(value_reg);
      EMIT_OPTIONAL_VALUE_REF(&regs->ref[value_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(b, ", .refs["));
      EMIT_I32_REG_NAME(global_reg);
      EMIT_OPTIONAL_VALUE_I32(regs->i32[global_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(b, "] : !"));
      EMIT_TYPE_NAME(type_def);
      break;
    }

    //===------------------------------------------------------------------===//
    // Constants
    //===------------------------------------------------------------------===//

    DISASM_OP(CORE, ConstI32) {
      int32_t value = VM_ParseIntAttr32("value");
      uint16_t result_reg = VM_ParseResultRegI32("result");
      EMIT_I32_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          b, " = vm.const.i32 %d  // 0x%08X", value, value));
      break;
    }

    DISASM_OP(CORE, ConstI32Zero) {
      uint16_t result_reg = VM_ParseResultRegI32("result");
      EMIT_I32_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.const.i32.zero"));
      break;
    }

    DISASM_OP(CORE, ConstI64) {
      int64_t value = VM_ParseIntAttr64("value");
      uint16_t result_reg = VM_ParseResultRegI64("result");
      EMIT_I64_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          b, " = vm.const.i64 %" PRId64 "  // 0x%016" PRIX64 "", value, value));
      break;
    }

    DISASM_OP(CORE, ConstI64Zero) {
      uint16_t result_reg = VM_ParseResultRegI64("result");
      EMIT_I64_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.const.i64.zero"));
      break;
    }

    DISASM_OP(CORE, ConstRefZero) {
      bool result_is_move;
      uint16_t result_reg = VM_ParseResultRegRef("result", &result_is_move);
      EMIT_REF_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.const.ref.zero"));
      break;
    }

    DISASM_OP(CORE, ConstRefRodata) {
      uint32_t rodata_ordinal = VM_ParseRodataAttr("rodata");
      bool result_is_move;
      uint16_t result_reg = VM_ParseResultRegRef("value", &result_is_move);
      iree_vm_buffer_t* buffer =
          &module_state->rodata_ref_table[rodata_ordinal];
      EMIT_REF_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          b, " = vm.const.ref.rodata %u  // 0x%p %" PRIhsz "b", rodata_ordinal,
          buffer->data.data, buffer->data.data_length));
      break;
    }

    //===------------------------------------------------------------------===//
    // Buffers
    //===------------------------------------------------------------------===//

    DISASM_OP(CORE, BufferAlloc) {
      uint16_t length_reg = VM_ParseOperandRegI64("length");
      bool result_is_move;
      uint16_t result_reg = VM_ParseResultRegRef("result", &result_is_move);
      EMIT_REF_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.buffer.alloc "));
      EMIT_I64_REG_NAME(length_reg);
      EMIT_OPTIONAL_VALUE_I64(regs->i32[length_reg]);
      break;
    }

    DISASM_OP(CORE, BufferClone) {
      bool source_is_move;
      uint16_t source_reg = VM_ParseOperandRegRef("source", &source_is_move);
      uint16_t offset_reg = VM_ParseOperandRegI64("offset");
      uint16_t length_reg = VM_ParseOperandRegI64("length");
      bool result_is_move;
      uint16_t result_reg = VM_ParseResultRegRef("result", &result_is_move);
      EMIT_REF_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.buffer.clone "));
      EMIT_REF_REG_NAME(source_reg);
      EMIT_OPTIONAL_VALUE_REF(&regs->ref[source_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      EMIT_I64_REG_NAME(offset_reg);
      EMIT_OPTIONAL_VALUE_I64(regs->i32[offset_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      EMIT_I64_REG_NAME(length_reg);
      EMIT_OPTIONAL_VALUE_I64(regs->i32[length_reg]);
      break;
    }

    DISASM_OP(CORE, BufferLength) {
      bool buffer_is_move;
      uint16_t buffer_reg = VM_ParseOperandRegRef("buffer", &buffer_is_move);
      uint16_t result_reg = VM_ParseResultRegI64("result");
      EMIT_I64_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.buffer.length "));
      EMIT_REF_REG_NAME(buffer_reg);
      EMIT_OPTIONAL_VALUE_REF(&regs->ref[buffer_reg]);
      break;
    }

    DISASM_OP(CORE, BufferCopy) {
      bool source_buffer_is_move;
      uint16_t source_buffer_reg =
          VM_ParseOperandRegRef("source_buffer", &source_buffer_is_move);
      uint16_t source_offset_reg = VM_ParseOperandRegI64("source_offset");
      bool target_buffer_is_move;
      uint16_t target_buffer_reg =
          VM_ParseOperandRegRef("target_buffer", &target_buffer_is_move);
      uint16_t target_offset_reg = VM_ParseOperandRegI64("target_offset");
      uint16_t length_reg = VM_ParseOperandRegI64("length");
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, "vm.buffer.copy "));
      EMIT_REF_REG_NAME(source_buffer_reg);
      EMIT_OPTIONAL_VALUE_REF(&regs->ref[source_buffer_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      EMIT_I64_REG_NAME(source_offset_reg);
      EMIT_OPTIONAL_VALUE_I64(regs->i32[source_offset_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      EMIT_REF_REG_NAME(target_buffer_reg);
      EMIT_OPTIONAL_VALUE_REF(&regs->ref[target_buffer_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      EMIT_I64_REG_NAME(target_offset_reg);
      EMIT_OPTIONAL_VALUE_I64(regs->i32[target_offset_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      EMIT_I64_REG_NAME(length_reg);
      EMIT_OPTIONAL_VALUE_I64(regs->i32[length_reg]);
      break;
    }

    DISASM_OP(CORE, BufferCompare) {
      bool lhs_buffer_is_move;
      uint16_t lhs_buffer_reg =
          VM_ParseOperandRegRef("lhs_buffer", &lhs_buffer_is_move);
      uint16_t lhs_offset_reg = VM_ParseOperandRegI64("lhs_offset");
      bool rhs_buffer_is_move;
      uint16_t rhs_buffer_reg =
          VM_ParseOperandRegRef("rhs_buffer", &rhs_buffer_is_move);
      uint16_t rhs_offset_reg = VM_ParseOperandRegI64("rhs_offset");
      uint16_t length_reg = VM_ParseOperandRegI64("length");
      uint16_t result_reg = VM_ParseResultRegI32("result");
      EMIT_I32_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.buffer.compare "));
      EMIT_REF_REG_NAME(lhs_buffer_reg);
      EMIT_OPTIONAL_VALUE_REF(&regs->ref[lhs_buffer_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      EMIT_I64_REG_NAME(lhs_offset_reg);
      EMIT_OPTIONAL_VALUE_I64(regs->i32[lhs_offset_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      EMIT_REF_REG_NAME(rhs_buffer_reg);
      EMIT_OPTIONAL_VALUE_REF(&regs->ref[rhs_buffer_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      EMIT_I64_REG_NAME(rhs_offset_reg);
      EMIT_OPTIONAL_VALUE_I64(regs->i32[rhs_offset_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      EMIT_I64_REG_NAME(length_reg);
      EMIT_OPTIONAL_VALUE_I64(regs->i32[length_reg]);
      break;
    }

    DISASM_OP(CORE, BufferFillI8) {
      bool buffer_is_move;
      uint16_t buffer_reg =
          VM_ParseOperandRegRef("target_buffer", &buffer_is_move);
      uint16_t offset_reg = VM_ParseOperandRegI64("target_offset");
      uint16_t length_reg = VM_ParseOperandRegI64("length");
      uint16_t value_reg = VM_ParseOperandRegI32("value");
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, "vm.buffer.fill.i8 "));
      EMIT_REF_REG_NAME(buffer_reg);
      EMIT_OPTIONAL_VALUE_REF(&regs->ref[buffer_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      EMIT_I64_REG_NAME(offset_reg);
      EMIT_OPTIONAL_VALUE_I64(regs->i32[offset_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      EMIT_I64_REG_NAME(length_reg);
      EMIT_OPTIONAL_VALUE_I64(regs->i32[length_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      EMIT_I32_REG_NAME(value_reg);
      EMIT_OPTIONAL_VALUE_I32((uint8_t)regs->i32[value_reg]);
      break;
    }
    DISASM_OP(CORE, BufferFillI16) {
      bool buffer_is_move;
      uint16_t buffer_reg =
          VM_ParseOperandRegRef("target_buffer", &buffer_is_move);
      uint16_t offset_reg = VM_ParseOperandRegI64("target_offset");
      uint16_t length_reg = VM_ParseOperandRegI64("length");
      uint16_t value_reg = VM_ParseOperandRegI32("value");
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, "vm.buffer.fill.i16 "));
      EMIT_REF_REG_NAME(buffer_reg);
      EMIT_OPTIONAL_VALUE_REF(&regs->ref[buffer_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      EMIT_I64_REG_NAME(offset_reg);
      EMIT_OPTIONAL_VALUE_I64(regs->i32[offset_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      EMIT_I64_REG_NAME(length_reg);
      EMIT_OPTIONAL_VALUE_I64(regs->i32[length_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      EMIT_I32_REG_NAME(value_reg);
      EMIT_OPTIONAL_VALUE_I32((uint16_t)regs->i32[value_reg]);
      break;
    }
    DISASM_OP(CORE, BufferFillI32) {
      bool buffer_is_move;
      uint16_t buffer_reg =
          VM_ParseOperandRegRef("target_buffer", &buffer_is_move);
      uint16_t offset_reg = VM_ParseOperandRegI64("target_offset");
      uint16_t length_reg = VM_ParseOperandRegI64("length");
      uint16_t value_reg = VM_ParseOperandRegI32("value");
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, "vm.buffer.fill.i32 "));
      EMIT_REF_REG_NAME(buffer_reg);
      EMIT_OPTIONAL_VALUE_REF(&regs->ref[buffer_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      EMIT_I64_REG_NAME(offset_reg);
      EMIT_OPTIONAL_VALUE_I64(regs->i32[offset_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      EMIT_I64_REG_NAME(length_reg);
      EMIT_OPTIONAL_VALUE_I64(regs->i32[length_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      EMIT_I32_REG_NAME(value_reg);
      EMIT_OPTIONAL_VALUE_I32(regs->i32[value_reg]);
      break;
    }

    DISASM_OP(CORE, BufferFillI64) {
      bool buffer_is_move;
      uint16_t buffer_reg =
          VM_ParseOperandRegRef("target_buffer", &buffer_is_move);
      uint16_t offset_reg = VM_ParseOperandRegI64("target_offset");
      uint16_t length_reg = VM_ParseOperandRegI64("length");
      uint16_t value_reg = VM_ParseOperandRegI64("value");
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, "vm.buffer.fill.i64 "));
      EMIT_REF_REG_NAME(buffer_reg);
      EMIT_OPTIONAL_VALUE_REF(&regs->ref[buffer_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      EMIT_I64_REG_NAME(offset_reg);
      EMIT_OPTIONAL_VALUE_I64(regs->i32[offset_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      EMIT_I64_REG_NAME(length_reg);
      EMIT_OPTIONAL_VALUE_I64(regs->i32[length_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      EMIT_I64_REG_NAME(value_reg);
      EMIT_OPTIONAL_VALUE_I64(regs->i32[value_reg]);
      break;
    }

    DISASM_OP(CORE, BufferLoadI8U) {
      bool buffer_is_move;
      uint16_t buffer_reg =
          VM_ParseOperandRegRef("source_buffer", &buffer_is_move);
      uint16_t offset_reg = VM_ParseOperandRegI64("source_offset");
      uint16_t result_reg = VM_ParseResultRegI32("result");
      EMIT_I32_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.buffer.load.i8.u "));
      EMIT_REF_REG_NAME(buffer_reg);
      EMIT_OPTIONAL_VALUE_REF(&regs->ref[buffer_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      EMIT_I64_REG_NAME(offset_reg);
      EMIT_OPTIONAL_VALUE_I64(regs->i32[offset_reg]);
      break;
    }
    DISASM_OP(CORE, BufferLoadI8S) {
      bool buffer_is_move;
      uint16_t buffer_reg =
          VM_ParseOperandRegRef("source_buffer", &buffer_is_move);
      uint16_t offset_reg = VM_ParseOperandRegI64("source_offset");
      uint16_t result_reg = VM_ParseResultRegI32("result");
      EMIT_I32_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.buffer.load.i8.s "));
      EMIT_REF_REG_NAME(buffer_reg);
      EMIT_OPTIONAL_VALUE_REF(&regs->ref[buffer_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      EMIT_I64_REG_NAME(offset_reg);
      EMIT_OPTIONAL_VALUE_I64(regs->i32[offset_reg]);
      break;
    }
    DISASM_OP(CORE, BufferLoadI16U) {
      bool buffer_is_move;
      uint16_t buffer_reg =
          VM_ParseOperandRegRef("source_buffer", &buffer_is_move);
      uint16_t offset_reg = VM_ParseOperandRegI64("source_offset");
      uint16_t result_reg = VM_ParseResultRegI32("result");
      EMIT_I32_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.buffer.load.i16.u "));
      EMIT_REF_REG_NAME(buffer_reg);
      EMIT_OPTIONAL_VALUE_REF(&regs->ref[buffer_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      EMIT_I64_REG_NAME(offset_reg);
      EMIT_OPTIONAL_VALUE_I64(regs->i32[offset_reg]);
      break;
    }
    DISASM_OP(CORE, BufferLoadI16S) {
      bool buffer_is_move;
      uint16_t buffer_reg =
          VM_ParseOperandRegRef("source_buffer", &buffer_is_move);
      uint16_t offset_reg = VM_ParseOperandRegI64("source_offset");
      uint16_t result_reg = VM_ParseResultRegI32("result");
      EMIT_I32_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.buffer.load.i16.s "));
      EMIT_REF_REG_NAME(buffer_reg);
      EMIT_OPTIONAL_VALUE_REF(&regs->ref[buffer_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      EMIT_I64_REG_NAME(offset_reg);
      EMIT_OPTIONAL_VALUE_I64(regs->i32[offset_reg]);
      break;
    }
    DISASM_OP(CORE, BufferLoadI32) {
      bool buffer_is_move;
      uint16_t buffer_reg =
          VM_ParseOperandRegRef("source_buffer", &buffer_is_move);
      uint16_t offset_reg = VM_ParseOperandRegI64("source_offset");
      uint16_t result_reg = VM_ParseResultRegI32("result");
      EMIT_I32_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.buffer.load.i32 "));
      EMIT_REF_REG_NAME(buffer_reg);
      EMIT_OPTIONAL_VALUE_REF(&regs->ref[buffer_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      EMIT_I64_REG_NAME(offset_reg);
      EMIT_OPTIONAL_VALUE_I64(regs->i32[offset_reg]);
      break;
    }
    DISASM_OP(CORE, BufferLoadI64) {
      bool buffer_is_move;
      uint16_t buffer_reg =
          VM_ParseOperandRegRef("source_buffer", &buffer_is_move);
      uint16_t offset_reg = VM_ParseOperandRegI64("source_offset");
      uint16_t result_reg = VM_ParseResultRegI64("result");
      EMIT_I64_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.buffer.load.i64 "));
      EMIT_REF_REG_NAME(buffer_reg);
      EMIT_OPTIONAL_VALUE_REF(&regs->ref[buffer_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      EMIT_I64_REG_NAME(offset_reg);
      EMIT_OPTIONAL_VALUE_I64(regs->i32[offset_reg]);
      break;
    }

    DISASM_OP(CORE, BufferStoreI8) {
      bool buffer_is_move;
      uint16_t buffer_reg =
          VM_ParseOperandRegRef("target_buffer", &buffer_is_move);
      uint16_t offset_reg = VM_ParseOperandRegI64("target_offset");
      uint16_t value_reg = VM_ParseOperandRegI32("value");
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, "vm.buffer.store.i8 "));
      EMIT_I32_REG_NAME(value_reg);
      EMIT_OPTIONAL_VALUE_I32((uint8_t)regs->i32[value_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      EMIT_REF_REG_NAME(buffer_reg);
      EMIT_OPTIONAL_VALUE_REF(&regs->ref[buffer_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      EMIT_I64_REG_NAME(offset_reg);
      EMIT_OPTIONAL_VALUE_I64(regs->i32[offset_reg]);
      break;
    }
    DISASM_OP(CORE, BufferStoreI16) {
      bool buffer_is_move;
      uint16_t buffer_reg =
          VM_ParseOperandRegRef("target_buffer", &buffer_is_move);
      uint16_t offset_reg = VM_ParseOperandRegI64("target_offset");
      uint16_t value_reg = VM_ParseOperandRegI32("value");
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, "vm.buffer.store.i16 "));
      EMIT_I32_REG_NAME(value_reg);
      EMIT_OPTIONAL_VALUE_I32((uint16_t)regs->i32[value_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      EMIT_REF_REG_NAME(buffer_reg);
      EMIT_OPTIONAL_VALUE_REF(&regs->ref[buffer_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      EMIT_I64_REG_NAME(offset_reg);
      EMIT_OPTIONAL_VALUE_I64(regs->i32[offset_reg]);
      break;
    }
    DISASM_OP(CORE, BufferStoreI32) {
      bool buffer_is_move;
      uint16_t buffer_reg =
          VM_ParseOperandRegRef("target_buffer", &buffer_is_move);
      uint16_t offset_reg = VM_ParseOperandRegI64("target_offset");
      uint16_t value_reg = VM_ParseOperandRegI32("value");
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, "vm.buffer.store.i32 "));
      EMIT_I32_REG_NAME(value_reg);
      EMIT_OPTIONAL_VALUE_I32(regs->i32[value_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      EMIT_REF_REG_NAME(buffer_reg);
      EMIT_OPTIONAL_VALUE_REF(&regs->ref[buffer_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      EMIT_I64_REG_NAME(offset_reg);
      EMIT_OPTIONAL_VALUE_I64(regs->i32[offset_reg]);
      break;
    }
    DISASM_OP(CORE, BufferStoreI64) {
      bool buffer_is_move;
      uint16_t buffer_reg =
          VM_ParseOperandRegRef("target_buffer", &buffer_is_move);
      uint16_t offset_reg = VM_ParseOperandRegI64("target_offset");
      uint16_t value_reg = VM_ParseOperandRegI64("value");
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, "vm.buffer.store.i64 "));
      EMIT_I64_REG_NAME(value_reg);
      EMIT_OPTIONAL_VALUE_I64(regs->i32[value_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      EMIT_REF_REG_NAME(buffer_reg);
      EMIT_OPTIONAL_VALUE_REF(&regs->ref[buffer_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      EMIT_I64_REG_NAME(offset_reg);
      EMIT_OPTIONAL_VALUE_I64(regs->i32[offset_reg]);
      break;
    }

    //===------------------------------------------------------------------===//
    // Lists
    //===------------------------------------------------------------------===//

    DISASM_OP(CORE, ListAlloc) {
      const iree_vm_type_def_t* element_type_def =
          VM_ParseTypeOf("element_type");
      uint16_t initial_capacity_reg = VM_ParseOperandRegI32("initial_capacity");
      bool result_is_move;
      uint16_t result_reg = VM_ParseResultRegRef("result", &result_is_move);
      EMIT_REF_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.list.alloc "));
      EMIT_I32_REG_NAME(initial_capacity_reg);
      EMIT_OPTIONAL_VALUE_I32(regs->i32[initial_capacity_reg]);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " : !vm.list<"));
      EMIT_TYPE_NAME(element_type_def);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ">"));
      break;
    }

    DISASM_OP(CORE, ListReserve) {
      bool list_is_move;
      uint16_t list_reg = VM_ParseOperandRegRef("list", &list_is_move);
      uint16_t minimum_capacity_reg = VM_ParseOperandRegI32("minimum_capacity");
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, "vm.list.reserve "));
      EMIT_REF_REG_NAME(list_reg);
      EMIT_OPTIONAL_VALUE_REF(&regs->ref[list_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      EMIT_I32_REG_NAME(minimum_capacity_reg);
      EMIT_OPTIONAL_VALUE_I32(regs->i32[minimum_capacity_reg]);
      break;
    }

    DISASM_OP(CORE, ListSize) {
      bool list_is_move;
      uint16_t list_reg = VM_ParseOperandRegRef("list", &list_is_move);
      uint16_t result_reg = VM_ParseResultRegI32("result");
      EMIT_I32_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.list.size "));
      EMIT_REF_REG_NAME(list_reg);
      EMIT_OPTIONAL_VALUE_REF(&regs->ref[list_reg]);
      break;
    }

    DISASM_OP(CORE, ListResize) {
      bool list_is_move;
      uint16_t list_reg = VM_ParseOperandRegRef("list", &list_is_move);
      uint16_t new_size_reg = VM_ParseOperandRegI32("new_size");
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, "vm.list.resize "));
      EMIT_REF_REG_NAME(list_reg);
      EMIT_OPTIONAL_VALUE_REF(&regs->ref[list_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      EMIT_I32_REG_NAME(new_size_reg);
      EMIT_OPTIONAL_VALUE_I32(regs->i32[new_size_reg]);
      break;
    }

    DISASM_OP(CORE, ListGetI32) {
      bool list_is_move;
      uint16_t list_reg = VM_ParseOperandRegRef("list", &list_is_move);
      uint16_t index_reg = VM_ParseOperandRegI32("index");
      uint16_t result_reg = VM_ParseResultRegI32("result");
      EMIT_I32_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.list.get.i32 "));
      EMIT_REF_REG_NAME(list_reg);
      EMIT_OPTIONAL_VALUE_REF(&regs->ref[list_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      EMIT_I32_REG_NAME(index_reg);
      EMIT_OPTIONAL_VALUE_I32(regs->i32[index_reg]);
      break;
    }

    DISASM_OP(CORE, ListSetI32) {
      bool list_is_move;
      uint16_t list_reg = VM_ParseOperandRegRef("list", &list_is_move);
      uint16_t index_reg = VM_ParseOperandRegI32("index");
      uint16_t raw_value_reg = VM_ParseOperandRegI32("raw_value");
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, "vm.list.set.i32 "));
      EMIT_REF_REG_NAME(list_reg);
      EMIT_OPTIONAL_VALUE_REF(&regs->ref[list_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      EMIT_I32_REG_NAME(index_reg);
      EMIT_OPTIONAL_VALUE_I32(regs->i32[index_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      EMIT_I32_REG_NAME(raw_value_reg);
      EMIT_OPTIONAL_VALUE_I32(regs->i32[raw_value_reg]);
      break;
    }

    DISASM_OP(CORE, ListGetI64) {
      bool list_is_move;
      uint16_t list_reg = VM_ParseOperandRegRef("list", &list_is_move);
      uint16_t index_reg = VM_ParseOperandRegI32("index");
      uint16_t result_reg = VM_ParseResultRegI64("result");
      EMIT_I64_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.list.get.i64 "));
      EMIT_REF_REG_NAME(list_reg);
      EMIT_OPTIONAL_VALUE_REF(&regs->ref[list_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      EMIT_I32_REG_NAME(index_reg);
      EMIT_OPTIONAL_VALUE_I32(regs->i32[index_reg]);
      break;
    }

    DISASM_OP(CORE, ListSetI64) {
      bool list_is_move;
      uint16_t list_reg = VM_ParseOperandRegRef("list", &list_is_move);
      uint16_t index_reg = VM_ParseOperandRegI32("index");
      uint16_t value_reg = VM_ParseOperandRegI64("value");
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, "vm.list.set.i64 "));
      EMIT_REF_REG_NAME(list_reg);
      EMIT_OPTIONAL_VALUE_REF(&regs->ref[list_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      EMIT_I32_REG_NAME(index_reg);
      EMIT_OPTIONAL_VALUE_I32(regs->i32[index_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      EMIT_I64_REG_NAME(value_reg);
      EMIT_OPTIONAL_VALUE_I64(regs->i32[value_reg]);
      break;
    }

    DISASM_OP(CORE, ListGetRef) {
      bool list_is_move;
      uint16_t list_reg = VM_ParseOperandRegRef("list", &list_is_move);
      uint16_t index_reg = VM_ParseOperandRegI32("index");
      const iree_vm_type_def_t* type_def = VM_ParseTypeOf("result");
      bool result_is_move;
      uint16_t result_reg = VM_ParseResultRegRef("result", &result_is_move);
      EMIT_REF_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.list.get.ref "));
      EMIT_REF_REG_NAME(list_reg);
      EMIT_OPTIONAL_VALUE_REF(&regs->ref[list_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      EMIT_I32_REG_NAME(index_reg);
      EMIT_OPTIONAL_VALUE_I32(regs->i32[index_reg]);
      EMIT_TYPE_NAME(type_def);
      break;
    }

    DISASM_OP(CORE, ListSetRef) {
      bool list_is_move;
      uint16_t list_reg = VM_ParseOperandRegRef("list", &list_is_move);
      uint16_t index_reg = VM_ParseOperandRegI32("index");
      bool operand_is_move;
      uint16_t operand_reg = VM_ParseOperandRegRef("value", &operand_is_move);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, "vm.list.set.ref "));
      EMIT_REF_REG_NAME(list_reg);
      EMIT_OPTIONAL_VALUE_REF(&regs->ref[list_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      EMIT_I32_REG_NAME(index_reg);
      EMIT_OPTIONAL_VALUE_I32(regs->i32[index_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      EMIT_REF_REG_NAME(operand_reg);
      EMIT_OPTIONAL_VALUE_REF(&regs->ref[operand_reg]);
      break;
    }

    //===------------------------------------------------------------------===//
    // Conditional assignment
    //===------------------------------------------------------------------===//

    DISASM_OP(CORE, SelectI32) {
      uint16_t condition_reg = VM_ParseOperandRegI32("condition");
      uint16_t true_value_reg = VM_ParseOperandRegI32("true_value");
      uint16_t false_value_reg = VM_ParseOperandRegI32("false_value");
      uint16_t result_reg = VM_ParseResultRegI32("result");
      EMIT_I32_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.select.i32 "));
      EMIT_I32_REG_NAME(condition_reg);
      EMIT_OPTIONAL_VALUE_I32(regs->i32[condition_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, " ? "));
      EMIT_I32_REG_NAME(true_value_reg);
      EMIT_OPTIONAL_VALUE_I32(regs->i32[true_value_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, " : "));
      EMIT_I32_REG_NAME(false_value_reg);
      EMIT_OPTIONAL_VALUE_I32(regs->i32[false_value_reg]);
      break;
    }

    DISASM_OP(CORE, SelectI64) {
      uint16_t condition_reg = VM_ParseOperandRegI32("condition");
      uint16_t true_value_reg = VM_ParseOperandRegI64("true_value");
      uint16_t false_value_reg = VM_ParseOperandRegI64("false_value");
      uint16_t result_reg = VM_ParseResultRegI64("result");
      EMIT_I64_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.select.i64 "));
      EMIT_I32_REG_NAME(condition_reg);
      EMIT_OPTIONAL_VALUE_I32(regs->i32[condition_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, " ? "));
      EMIT_I64_REG_NAME(true_value_reg);
      EMIT_OPTIONAL_VALUE_I64(regs->i32[true_value_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, " : "));
      EMIT_I64_REG_NAME(false_value_reg);
      EMIT_OPTIONAL_VALUE_I64(regs->i32[false_value_reg]);
      break;
    }

    DISASM_OP(CORE, SelectRef) {
      uint16_t condition_reg = VM_ParseOperandRegI32("condition");
      const iree_vm_type_def_t* type_def = VM_ParseTypeOf("true_value");
      bool true_value_is_move;
      uint16_t true_value_reg =
          VM_ParseOperandRegRef("true_value", &true_value_is_move);
      bool false_value_is_move;
      uint16_t false_value_reg =
          VM_ParseOperandRegRef("false_value", &false_value_is_move);
      bool result_is_move;
      uint16_t result_reg = VM_ParseResultRegRef("result", &result_is_move);
      EMIT_REF_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.select.ref "));
      EMIT_I32_REG_NAME(condition_reg);
      EMIT_OPTIONAL_VALUE_I32(regs->i32[condition_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, " ? "));
      EMIT_REF_REG_NAME(true_value_reg);
      EMIT_OPTIONAL_VALUE_REF(&regs->ref[true_value_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, " : "));
      EMIT_REF_REG_NAME(false_value_reg);
      EMIT_OPTIONAL_VALUE_REF(&regs->ref[false_value_reg]);
      EMIT_TYPE_NAME(type_def);
      break;
    }

    DISASM_OP(CORE, SwitchI32) {
      uint16_t index_reg = VM_ParseOperandRegI32("index");
      int32_t default_value = VM_ParseIntAttr32("default_value");
      const iree_vm_register_list_t* value_reg_list =
          VM_ParseVariadicOperands("values");
      uint16_t result_reg = VM_ParseResultRegI32("result");
      EMIT_I32_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.switch.i32 "));
      EMIT_I32_REG_NAME(index_reg);
      EMIT_OPTIONAL_VALUE_I32(regs->i32[index_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, "["));
      EMIT_OPERAND_REG_LIST(value_reg_list);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_format(b, "] else %u", default_value));
      break;
    }

    DISASM_OP(CORE, SwitchI64) {
      uint16_t index_reg = VM_ParseOperandRegI32("index");
      int64_t default_value = VM_ParseIntAttr64("default_value");
      const iree_vm_register_list_t* value_reg_list =
          VM_ParseVariadicOperands("values");
      uint16_t result_reg = VM_ParseResultRegI64("result");
      EMIT_I64_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.switch.i64 "));
      EMIT_I32_REG_NAME(index_reg);
      EMIT_OPTIONAL_VALUE_I32(regs->i32[index_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, "["));
      EMIT_OPERAND_REG_LIST(value_reg_list);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          b, "] else %" PRId64, default_value));
      break;
    }

    DISASM_OP(CORE, SwitchRef) {
      uint16_t index_reg = VM_ParseOperandRegI32("index");
      bool default_is_move;
      uint16_t default_value_reg =
          VM_ParseOperandRegRef("default_value", &default_is_move);
      const iree_vm_register_list_t* value_reg_list =
          VM_ParseVariadicOperands("values");
      bool result_is_move;
      uint16_t result_reg = VM_ParseResultRegRef("result", &result_is_move);
      EMIT_REF_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.switch.ref "));
      EMIT_I32_REG_NAME(index_reg);
      EMIT_OPTIONAL_VALUE_I32(regs->i32[index_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, "["));
      EMIT_OPERAND_REG_LIST(value_reg_list);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, "] else "));
      EMIT_REF_REG_NAME(default_value_reg);
      EMIT_OPTIONAL_VALUE_REF(&regs->ref[default_value_reg]);
      break;
    }

    //===------------------------------------------------------------------===//
    // Native integer arithmetic
    //===------------------------------------------------------------------===//

    DISASM_OP_CORE_BINARY_I32(AddI32, "vm.add.i32");
    DISASM_OP_CORE_BINARY_I32(SubI32, "vm.sub.i32");
    DISASM_OP_CORE_BINARY_I32(MulI32, "vm.mul.i32");
    DISASM_OP_CORE_BINARY_I32(DivI32S, "vm.div.i32.s");
    DISASM_OP_CORE_BINARY_I32(DivI32U, "vm.div.i32.u");
    DISASM_OP_CORE_BINARY_I32(RemI32S, "vm.rem.i32.s");
    DISASM_OP_CORE_BINARY_I32(RemI32U, "vm.rem.i32.u");
    DISASM_OP_CORE_TERNARY_I32(FMAI32, "vm.fma.i32");
    DISASM_OP_CORE_UNARY_I32(AbsI32, "vm.abs.i32");
    DISASM_OP_CORE_BINARY_I32(MinI32S, "vm.min.i32.s");
    DISASM_OP_CORE_BINARY_I32(MinI32U, "vm.min.i32.u");
    DISASM_OP_CORE_BINARY_I32(MaxI32S, "vm.max.i32.s");
    DISASM_OP_CORE_BINARY_I32(MaxI32U, "vm.max.i32.u");
    DISASM_OP_CORE_UNARY_I32(NotI32, "vm.not.i32");
    DISASM_OP_CORE_BINARY_I32(AndI32, "vm.and.i32");
    DISASM_OP_CORE_BINARY_I32(OrI32, "vm.or.i32");
    DISASM_OP_CORE_BINARY_I32(XorI32, "vm.xor.i32");
    DISASM_OP_CORE_UNARY_I32(CtlzI32, "vm.ctlz.i32");

    DISASM_OP_CORE_BINARY_I64(AddI64, "vm.add.i64");
    DISASM_OP_CORE_BINARY_I64(SubI64, "vm.sub.i64");
    DISASM_OP_CORE_BINARY_I64(MulI64, "vm.mul.i64");
    DISASM_OP_CORE_BINARY_I64(DivI64S, "vm.div.i64.s");
    DISASM_OP_CORE_BINARY_I64(DivI64U, "vm.div.i64.u");
    DISASM_OP_CORE_BINARY_I64(RemI64S, "vm.rem.i64.s");
    DISASM_OP_CORE_BINARY_I64(RemI64U, "vm.rem.i64.u");
    DISASM_OP_CORE_TERNARY_I64(FMAI64, "vm.fma.i64");
    DISASM_OP_CORE_UNARY_I64(AbsI64, "vm.abs.i64");
    DISASM_OP_CORE_BINARY_I64(MinI64S, "vm.min.i64.s");
    DISASM_OP_CORE_BINARY_I64(MinI64U, "vm.min.i64.u");
    DISASM_OP_CORE_BINARY_I64(MaxI64S, "vm.max.i64.s");
    DISASM_OP_CORE_BINARY_I64(MaxI64U, "vm.max.i64.u");
    DISASM_OP_CORE_UNARY_I64(NotI64, "vm.not.i64");
    DISASM_OP_CORE_BINARY_I64(AndI64, "vm.and.i64");
    DISASM_OP_CORE_BINARY_I64(OrI64, "vm.or.i64");
    DISASM_OP_CORE_BINARY_I64(XorI64, "vm.xor.i64");
    DISASM_OP_CORE_UNARY_I64(CtlzI64, "vm.ctlz.i64");

    //===------------------------------------------------------------------===//
    // Casting and type conversion/emulation
    //===------------------------------------------------------------------===//

    DISASM_OP_CORE_UNARY_I32(TruncI32I8, "vm.trunc.i32.i8");
    DISASM_OP_CORE_UNARY_I32(TruncI32I16, "vm.trunc.i32.i16");
    DISASM_OP_CORE_UNARY_I32(ExtI8I32S, "vm.ext.i8.i32.s");
    DISASM_OP_CORE_UNARY_I32(ExtI8I32U, "vm.ext.i8.i32.u");
    DISASM_OP_CORE_UNARY_I32(ExtI16I32S, "vm.ext.i16.i32.s");
    DISASM_OP_CORE_UNARY_I32(ExtI16I32U, "vm.ext.i16.i32.u");

    DISASM_OP(CORE, TruncI64I32) {
      uint16_t operand_reg = VM_ParseOperandRegI64("operand");
      uint16_t result_reg = VM_ParseResultRegI32("result");
      EMIT_I32_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.trunc.i64.i32 "));
      EMIT_I64_REG_NAME(operand_reg);
      EMIT_OPTIONAL_VALUE_I64(regs->i32[operand_reg]);
      break;
    }
    DISASM_OP(CORE, ExtI32I64S) {
      uint16_t operand_reg = VM_ParseOperandRegI32("operand");
      uint16_t result_reg = VM_ParseResultRegI64("result");
      EMIT_I64_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.ext.i32.i64.s "));
      EMIT_I32_REG_NAME(operand_reg);
      EMIT_OPTIONAL_VALUE_I32(regs->i32[operand_reg]);
      break;
    }
    DISASM_OP(CORE, ExtI32I64U) {
      uint16_t operand_reg = VM_ParseOperandRegI32("operand");
      uint16_t result_reg = VM_ParseResultRegI64("result");
      EMIT_I64_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.ext.i32.i64.u "));
      EMIT_I32_REG_NAME(operand_reg);
      EMIT_OPTIONAL_VALUE_I32(regs->i32[operand_reg]);
      break;
    }

    //===------------------------------------------------------------------===//
    // Native bitwise shifts and rotates
    //===------------------------------------------------------------------===//

#define DISASM_OP_CORE_SHIFT_I32(op_name, op_mnemonic)                 \
  DISASM_OP(CORE, op_name) {                                           \
    uint16_t operand_reg = VM_ParseOperandRegI32("operand");           \
    uint16_t amount_reg = VM_ParseOperandRegI32("amount");             \
    uint16_t result_reg = VM_ParseResultRegI32("result");              \
    EMIT_I32_REG_NAME(result_reg);                                     \
    IREE_RETURN_IF_ERROR(                                              \
        iree_string_builder_append_format(b, " = %s ", op_mnemonic));  \
    EMIT_I32_REG_NAME(operand_reg);                                    \
    EMIT_OPTIONAL_VALUE_I32(regs->i32[operand_reg]);                   \
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", ")); \
    EMIT_I32_REG_NAME(amount_reg);                                     \
    EMIT_OPTIONAL_VALUE_I32(regs->i32[amount_reg]);                    \
    break;                                                             \
  }

    DISASM_OP_CORE_SHIFT_I32(ShlI32, "vm.shl.i32");
    DISASM_OP_CORE_SHIFT_I32(ShrI32S, "vm.shr.i32.s");
    DISASM_OP_CORE_SHIFT_I32(ShrI32U, "vm.shr.i32.u");

#define DISASM_OP_CORE_SHIFT_I64(op_name, op_mnemonic)                 \
  DISASM_OP(CORE, op_name) {                                           \
    uint16_t operand_reg = VM_ParseOperandRegI64("operand");           \
    uint16_t amount_reg = VM_ParseOperandRegI32("amount");             \
    uint16_t result_reg = VM_ParseResultRegI64("result");              \
    EMIT_I64_REG_NAME(result_reg);                                     \
    IREE_RETURN_IF_ERROR(                                              \
        iree_string_builder_append_format(b, " = %s ", op_mnemonic));  \
    EMIT_I64_REG_NAME(operand_reg);                                    \
    EMIT_OPTIONAL_VALUE_I64(regs->i32[operand_reg]);                   \
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", ")); \
    EMIT_I32_REG_NAME(amount_reg);                                     \
    EMIT_OPTIONAL_VALUE_I32(regs->i32[amount_reg]);                    \
    break;                                                             \
  }

    DISASM_OP_CORE_SHIFT_I64(ShlI64, "vm.shl.i64");
    DISASM_OP_CORE_SHIFT_I64(ShrI64S, "vm.shr.i64.s");
    DISASM_OP_CORE_SHIFT_I64(ShrI64U, "vm.shr.i64.u");

    //===------------------------------------------------------------------===//
    // Comparison ops
    //===------------------------------------------------------------------===//

    DISASM_OP_CORE_BINARY_I32(CmpEQI32, "vm.cmp.eq.i32");
    DISASM_OP_CORE_BINARY_I32(CmpNEI32, "vm.cmp.ne.i32");
    DISASM_OP_CORE_BINARY_I32(CmpLTI32S, "vm.cmp.lt.i32.s");
    DISASM_OP_CORE_BINARY_I32(CmpLTI32U, "vm.cmp.lt.i32.u");
    DISASM_OP_CORE_UNARY_I32(CmpNZI32, "vm.cmp.nz.i32");

#define DISASM_OP_CORE_CMP_I64(op_name, op_mnemonic)                   \
  DISASM_OP(CORE, op_name) {                                           \
    uint16_t lhs_reg = VM_ParseOperandRegI64("lhs");                   \
    uint16_t rhs_reg = VM_ParseOperandRegI64("rhs");                   \
    uint16_t result_reg = VM_ParseResultRegI32("result");              \
    EMIT_I32_REG_NAME(result_reg);                                     \
    IREE_RETURN_IF_ERROR(                                              \
        iree_string_builder_append_format(b, " = %s ", op_mnemonic));  \
    EMIT_I64_REG_NAME(lhs_reg);                                        \
    EMIT_OPTIONAL_VALUE_I64(regs->i32[lhs_reg]);                       \
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", ")); \
    EMIT_I64_REG_NAME(rhs_reg);                                        \
    EMIT_OPTIONAL_VALUE_I64(regs->i32[rhs_reg]);                       \
    break;                                                             \
  }

    DISASM_OP_CORE_CMP_I64(CmpEQI64, "vm.cmp.eq.i64");
    DISASM_OP_CORE_CMP_I64(CmpNEI64, "vm.cmp.ne.i64");
    DISASM_OP_CORE_CMP_I64(CmpLTI64S, "vm.cmp.lt.i64.s");
    DISASM_OP_CORE_CMP_I64(CmpLTI64U, "vm.cmp.lt.i64.u");
    DISASM_OP(CORE, CmpNZI64) {
      uint16_t operand_reg = VM_ParseOperandRegI64("operand");
      uint16_t result_reg = VM_ParseResultRegI32("result");
      EMIT_I32_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.cmp.nz.i64 "));
      EMIT_I64_REG_NAME(operand_reg);
      EMIT_OPTIONAL_VALUE_I64(regs->i32[operand_reg]);
      break;
    }

    DISASM_OP(CORE, CmpEQRef) {
      bool lhs_is_move;
      uint16_t lhs_reg = VM_ParseOperandRegRef("lhs", &lhs_is_move);
      bool rhs_is_move;
      uint16_t rhs_reg = VM_ParseOperandRegRef("rhs", &rhs_is_move);
      uint16_t result_reg = VM_ParseResultRegI32("result");
      EMIT_I32_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.cmp.eq.ref "));
      EMIT_REF_REG_NAME(lhs_reg);
      EMIT_OPTIONAL_VALUE_REF(&regs->ref[lhs_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      EMIT_REF_REG_NAME(rhs_reg);
      EMIT_OPTIONAL_VALUE_REF(&regs->ref[rhs_reg]);
      break;
    }
    DISASM_OP(CORE, CmpNERef) {
      bool lhs_is_move;
      uint16_t lhs_reg = VM_ParseOperandRegRef("lhs", &lhs_is_move);
      bool rhs_is_move;
      uint16_t rhs_reg = VM_ParseOperandRegRef("rhs", &rhs_is_move);
      uint16_t result_reg = VM_ParseResultRegI32("result");
      EMIT_I32_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.cmp.ne.ref "));
      EMIT_REF_REG_NAME(lhs_reg);
      EMIT_OPTIONAL_VALUE_REF(&regs->ref[lhs_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      EMIT_REF_REG_NAME(rhs_reg);
      EMIT_OPTIONAL_VALUE_REF(&regs->ref[rhs_reg]);
      break;
    }
    DISASM_OP(CORE, CmpNZRef) {
      bool operand_is_move;
      uint16_t operand_reg = VM_ParseOperandRegRef("operand", &operand_is_move);
      uint16_t result_reg = VM_ParseResultRegI32("result");
      EMIT_I32_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.cmp.nz.ref "));
      EMIT_REF_REG_NAME(operand_reg);
      EMIT_OPTIONAL_VALUE_REF(&regs->ref[operand_reg]);
      break;
    }

    //===------------------------------------------------------------------===//
    // Control flow
    //===------------------------------------------------------------------===//

    DISASM_OP(CORE, Block) {
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_string(b, IREE_SV("<block>")));
      break;
    }

    DISASM_OP(CORE, Branch) {
      int32_t block_pc = VM_ParseBranchTarget("dest");
      const iree_vm_register_remap_list_t* remap_list =
          VM_ParseBranchOperands("operands");
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_format(b, "vm.br ^%08X(", block_pc));
      EMIT_REMAP_LIST(remap_list);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ")"));
      break;
    }

    DISASM_OP(CORE, CondBranch) {
      uint16_t condition_reg = VM_ParseOperandRegI32("condition");
      int32_t true_block_pc = VM_ParseBranchTarget("true_dest");
      const iree_vm_register_remap_list_t* true_remap_list =
          VM_ParseBranchOperands("true_operands");
      int32_t false_block_pc = VM_ParseBranchTarget("false_dest");
      const iree_vm_register_remap_list_t* false_remap_list =
          VM_ParseBranchOperands("false_operands");
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, "vm.cond_br "));
      EMIT_I32_REG_NAME(condition_reg);
      EMIT_OPTIONAL_VALUE_I32(regs->i32[condition_reg]);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_format(b, ", ^%08X(", true_block_pc));
      EMIT_REMAP_LIST(true_remap_list);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_format(b, "), ^%08X(", false_block_pc));
      EMIT_REMAP_LIST(false_remap_list);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ")"));
      break;
    }

    DISASM_OP(CORE, Call) {
      int32_t function_ordinal = VM_ParseFuncAttr("callee");
      const iree_vm_register_list_t* src_reg_list =
          VM_ParseVariadicOperands("operands");
      const iree_vm_register_list_t* dst_reg_list =
          VM_ParseVariadicResults("results");
      if (dst_reg_list->size > 0) {
        EMIT_RESULT_REG_LIST(dst_reg_list);
        IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, " = "));
      }
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, "vm.call @"));
      int is_import = (function_ordinal & 0x80000000u) != 0;
      iree_vm_function_t function;
      if (is_import) {
        const iree_vm_bytecode_import_t* import =
            &module_state->import_table[function_ordinal & 0x7FFFFFFFu];
        function = import->function;
      } else {
        function.module = &module->interface;
        function.linkage = IREE_VM_FUNCTION_LINKAGE_INTERNAL;
        function.ordinal = function_ordinal;
      }
      if (function.module) {
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
      } else {
        IREE_RETURN_IF_ERROR(
            iree_string_builder_append_cstring(b, "{{UNRESOLVED}}"));
      }
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, "("));
      EMIT_OPERAND_REG_LIST(src_reg_list);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ")"));
      break;
    }

    DISASM_OP(CORE, CallVariadic) {
      int32_t function_ordinal = VM_ParseFuncAttr("callee");
      // TODO(benvanik): print segment sizes.
      // const iree_vm_register_list_t* segment_size_list =
      VM_ParseVariadicOperands("segment_sizes");
      const iree_vm_register_list_t* src_reg_list =
          VM_ParseVariadicOperands("operands");
      const iree_vm_register_list_t* dst_reg_list =
          VM_ParseVariadicResults("results");
      if (dst_reg_list->size > 0) {
        EMIT_RESULT_REG_LIST(dst_reg_list);
        IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, " = "));
      }
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, "vm.call.varadic @"));
      int is_import = (function_ordinal & 0x80000000u) != 0;
      iree_vm_function_t function;
      if (is_import) {
        const iree_vm_bytecode_import_t* import =
            &module_state->import_table[function_ordinal & 0x7FFFFFFFu];
        function = import->function;
      } else {
        function.module = &module->interface;
        function.linkage = IREE_VM_FUNCTION_LINKAGE_INTERNAL;
        function.ordinal = function_ordinal;
      }
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
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, "("));
      EMIT_OPERAND_REG_LIST(src_reg_list);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ")"));
      break;
    }

    DISASM_OP(CORE, Return) {
      const iree_vm_register_list_t* src_reg_list =
          VM_ParseVariadicOperands("operands");
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, "vm.return "));
      EMIT_OPERAND_REG_LIST(src_reg_list);
      break;
    }

    DISASM_OP(CORE, Fail) {
      uint16_t status_code_reg = VM_ParseOperandRegI32("status");
      iree_string_view_t message;
      VM_ParseStrAttr("message", &message);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, "vm.fail "));
      EMIT_I32_REG_NAME(status_code_reg);
      EMIT_OPTIONAL_VALUE_I32(regs->i32[status_code_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          b, ", \"%.*s\"", (int)message.size, message.data));
      break;
    }

    DISASM_OP(CORE, ImportResolved) {
      int32_t function_ordinal = VM_ParseFuncAttr("import");
      uint16_t result_reg = VM_ParseResultRegI32("result");
      EMIT_I32_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.import.exists @"));
      int is_import = (function_ordinal & 0x80000000u) != 0;
      if (IREE_UNLIKELY(!is_import)) {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            b, "{{INVALID ORDINAL %d}}", function_ordinal));
        break;
      }
      uint32_t import_ordinal = function_ordinal & 0x7FFFFFFFu;
      if (IREE_UNLIKELY(import_ordinal >= module_state->import_count)) {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            b, "{{OUT OF RANGE ORDINAL %u}}", import_ordinal));
        break;
      }
      iree_vm_function_t decl_function;
      IREE_RETURN_IF_ERROR(iree_vm_module_lookup_function_by_ordinal(
          &module->interface, IREE_VM_FUNCTION_LINKAGE_IMPORT_OPTIONAL,
          import_ordinal, &decl_function));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
          b, iree_vm_function_name(&decl_function)));
      const iree_vm_bytecode_import_t* import =
          &module_state->import_table[import_ordinal];
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
          b, import->function.module != NULL ? " // (resolved)"
                                             : " // (unresolved)"));
      break;
    }

    //===------------------------------------------------------------------===//
    // Async/fiber ops
    //===------------------------------------------------------------------===//

    DISASM_OP(CORE, Yield) {
      int32_t block_pc = VM_DecBranchTarget("dest");
      const iree_vm_register_remap_list_t* remap_list =
          VM_ParseBranchOperands("operands");
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_format(b, "vm.yield ^%08X(", block_pc));
      EMIT_REMAP_LIST(remap_list);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ")"));
      break;
    }

    //===------------------------------------------------------------------===//
    // Debugging
    //===------------------------------------------------------------------===//

    DISASM_OP(CORE, Trace) {
      iree_string_view_t event_name;
      VM_ParseStrAttr("event_name", &event_name);
      const iree_vm_register_list_t* src_reg_list =
          VM_ParseVariadicOperands("operands");
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          b, "vm.trace \"%.*s\"(", (int)event_name.size, event_name.data));
      EMIT_OPERAND_REG_LIST(src_reg_list);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ")"));
      break;
    }

    DISASM_OP(CORE, Print) {
      iree_string_view_t event_name;
      VM_ParseStrAttr("event_name", &event_name);
      const iree_vm_register_list_t* src_reg_list =
          VM_ParseVariadicOperands("operands");
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          b, "vm.print \"%.*s\"(", (int)event_name.size, event_name.data));
      EMIT_OPERAND_REG_LIST(src_reg_list);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ")"));
      break;
    }

    DISASM_OP(CORE, Break) {
      int32_t block_pc = VM_DecBranchTarget("dest");
      const iree_vm_register_remap_list_t* remap_list =
          VM_ParseBranchOperands("operands");
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_format(b, "vm.break ^%08X(", block_pc));
      EMIT_REMAP_LIST(remap_list);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ")"));
      break;
    }

    DISASM_OP(CORE, CondBreak) {
      uint16_t condition_reg = VM_ParseOperandRegI32("condition");
      int32_t block_pc = VM_ParseBranchTarget("dest");
      const iree_vm_register_remap_list_t* remap_list =
          VM_ParseBranchOperands("operands");
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, "vm.cond_break "));
      EMIT_I32_REG_NAME(condition_reg);
      EMIT_OPTIONAL_VALUE_I32(regs->i32[condition_reg]);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_format(b, ", ^%08X(", block_pc));
      EMIT_REMAP_LIST(remap_list);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ")"));
      break;
    }

    //===------------------------------------------------------------------===//
    // Extension trampolines
    //===------------------------------------------------------------------===//

#if IREE_VM_EXT_F32_ENABLE
    BEGIN_DISASM_PREFIX(PrefixExtF32, EXT_F32)

    //===----------------------------------------------------------------===//
    // ExtF32: Globals
    //===----------------------------------------------------------------===//

    DISASM_OP(EXT_F32, GlobalLoadF32) {
      uint32_t byte_offset = VM_ParseGlobalAttr("global");
      uint16_t value_reg = VM_ParseResultRegF32("value");
      EMIT_F32_REG_NAME(value_reg);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          b, " = vm.global.load.f32 .rwdata[%u]", byte_offset));
      EMIT_OPTIONAL_VALUE_F32(module_state->rwdata_storage.data[byte_offset]);
      break;
    }

    DISASM_OP(EXT_F32, GlobalStoreF32) {
      uint32_t byte_offset = VM_ParseGlobalAttr("global");
      uint16_t value_reg = VM_ParseOperandRegF32("value");
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_format(b, "vm.global.store.f32 "));
      EMIT_F32_REG_NAME(value_reg);
      EMIT_OPTIONAL_VALUE_F32(regs->i32[value_reg]);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_format(b, ", .rwdata[%u]", byte_offset));
      break;
    }

    DISASM_OP(EXT_F32, GlobalLoadIndirectF32) {
      uint16_t byte_offset_reg = VM_ParseOperandRegI32("global");
      uint16_t value_reg = VM_ParseResultRegI32("value");
      EMIT_F32_REG_NAME(value_reg);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
          b, " = vm.global.load.indirect.f32 .rwdata["));
      EMIT_I32_REG_NAME(byte_offset_reg);
      EMIT_OPTIONAL_VALUE_I32(regs->i32[byte_offset_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, "]"));
      EMIT_OPTIONAL_VALUE_F32(
          module_state->rwdata_storage.data[regs->i32[byte_offset_reg]]);
      break;
    }

    DISASM_OP(EXT_F32, GlobalStoreIndirectF32) {
      uint16_t byte_offset_reg = VM_ParseOperandRegI32("global");
      uint16_t value_reg = VM_ParseOperandRegF32("value");
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
          b, "vm.global.store.indirect.f32 "));
      EMIT_F32_REG_NAME(value_reg);
      EMIT_OPTIONAL_VALUE_F32(regs->i32[value_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", .rwdata["));
      EMIT_I32_REG_NAME(byte_offset_reg);
      EMIT_OPTIONAL_VALUE_I32(regs->i32[byte_offset_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, "]"));
      break;
    }

    //===----------------------------------------------------------------===//
    // ExtF32: Constants
    //===----------------------------------------------------------------===//

    DISASM_OP(EXT_F32, ConstF32) {
      float value = VM_ParseFloatAttr32("value");
      uint16_t result_reg = VM_ParseResultRegF32("result");
      EMIT_F32_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_format(b, " = vm.const.f32 %f", value));
      break;
    }

    DISASM_OP(EXT_F32, ConstF32Zero) {
      uint16_t result_reg = VM_ParseResultRegF32("result");
      EMIT_F32_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.const.f32.zero"));
      break;
    }

    //===----------------------------------------------------------------===//
    // ExtF32: Lists
    //===----------------------------------------------------------------===//

    DISASM_OP(EXT_F32, ListGetF32) {
      bool list_is_move;
      uint16_t list_reg = VM_ParseOperandRegRef("list", &list_is_move);
      uint16_t index_reg = VM_ParseOperandRegI32("index");
      uint16_t result_reg = VM_ParseResultRegF32("result");
      EMIT_F32_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.list.get.f32 "));
      EMIT_REF_REG_NAME(list_reg);
      EMIT_OPTIONAL_VALUE_REF(&regs->ref[list_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      EMIT_I32_REG_NAME(index_reg);
      EMIT_OPTIONAL_VALUE_I32(regs->i32[index_reg]);
      break;
    }

    DISASM_OP(EXT_F32, ListSetF32) {
      bool list_is_move;
      uint16_t list_reg = VM_ParseOperandRegRef("list", &list_is_move);
      uint16_t index_reg = VM_ParseOperandRegI32("index");
      uint16_t raw_value_reg = VM_ParseOperandRegF32("raw_value");
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, "vm.list.set.f32 "));
      EMIT_REF_REG_NAME(list_reg);
      EMIT_OPTIONAL_VALUE_REF(&regs->ref[list_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      EMIT_I32_REG_NAME(index_reg);
      EMIT_OPTIONAL_VALUE_I32(regs->i32[index_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      EMIT_F32_REG_NAME(raw_value_reg);
      EMIT_OPTIONAL_VALUE_F32(regs->i32[raw_value_reg]);
      break;
    }

    //===----------------------------------------------------------------===//
    // ExtF32: Conditional assignment
    //===----------------------------------------------------------------===//

    DISASM_OP(EXT_F32, SelectF32) {
      uint16_t condition_reg = VM_ParseOperandRegI32("condition");
      uint16_t true_value_reg = VM_ParseOperandRegF32("true_value");
      uint16_t false_value_reg = VM_ParseOperandRegF32("false_value");
      uint16_t result_reg = VM_ParseResultRegF32("result");
      EMIT_F32_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.select.f32 "));
      EMIT_I32_REG_NAME(condition_reg);
      EMIT_OPTIONAL_VALUE_I32(regs->i32[condition_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, " ? "));
      EMIT_F32_REG_NAME(true_value_reg);
      EMIT_OPTIONAL_VALUE_F32(regs->i32[true_value_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, " : "));
      EMIT_F32_REG_NAME(false_value_reg);
      EMIT_OPTIONAL_VALUE_F32(regs->i32[false_value_reg]);
      break;
    }

    DISASM_OP(EXT_F32, SwitchF32) {
      uint16_t index_reg = VM_ParseOperandRegI32("index");
      float default_value = VM_ParseFloatAttr32("default_value");
      const iree_vm_register_list_t* value_reg_list =
          VM_ParseVariadicOperands("values");
      uint16_t result_reg = VM_ParseResultRegF32("result");
      EMIT_F32_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.switch.f32 "));
      EMIT_I32_REG_NAME(index_reg);
      EMIT_OPTIONAL_VALUE_I32(regs->i32[index_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, "["));
      EMIT_OPERAND_REG_LIST(value_reg_list);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_format(b, "] else %f", default_value));
      break;
    }

    //===----------------------------------------------------------------===//
    // ExtF32: Native floating-point arithmetic
    //===----------------------------------------------------------------===//

    DISASM_OP_EXT_F32_BINARY_F32(AddF32, "vm.add.f32");
    DISASM_OP_EXT_F32_BINARY_F32(SubF32, "vm.sub.f32");
    DISASM_OP_EXT_F32_BINARY_F32(MulF32, "vm.mul.f32");
    DISASM_OP_EXT_F32_BINARY_F32(DivF32, "vm.div.f32");
    DISASM_OP_EXT_F32_BINARY_F32(RemF32, "vm.rem.f32");
    DISASM_OP_EXT_F32_TERNARY_F32(FMAF32, "vm.fma.f32");
    DISASM_OP_EXT_F32_UNARY_F32(AbsF32, "vm.abs.f32");
    DISASM_OP_EXT_F32_UNARY_F32(NegF32, "vm.neg.f32");
    DISASM_OP_EXT_F32_UNARY_F32(CeilF32, "vm.ceil.f32");
    DISASM_OP_EXT_F32_UNARY_F32(FloorF32, "vm.floor.f32");
    DISASM_OP_EXT_F32_UNARY_F32(RoundF32, "vm.round.f32");
    DISASM_OP_EXT_F32_BINARY_F32(MinF32, "vm.min.f32");
    DISASM_OP_EXT_F32_BINARY_F32(MaxF32, "vm.max.f32");

    DISASM_OP_EXT_F32_UNARY_F32(AtanF32, "vm.atan.f32");
    DISASM_OP_EXT_F32_BINARY_F32(Atan2F32, "vm.atan2.f32");
    DISASM_OP_EXT_F32_UNARY_F32(CosF32, "vm.cos.f32");
    DISASM_OP_EXT_F32_UNARY_F32(SinF32, "vm.sin.f32");
    DISASM_OP_EXT_F32_UNARY_F32(ExpF32, "vm.exp.f32");
    DISASM_OP_EXT_F32_UNARY_F32(Exp2F32, "vm.exp2.f32");
    DISASM_OP_EXT_F32_UNARY_F32(ExpM1F32, "vm.expm1.f32");
    DISASM_OP_EXT_F32_UNARY_F32(LogF32, "vm.log.f32");
    DISASM_OP_EXT_F32_UNARY_F32(Log10F32, "vm.log10.f32");
    DISASM_OP_EXT_F32_UNARY_F32(Log1pF32, "vm.log1p.f32");
    DISASM_OP_EXT_F32_UNARY_F32(Log2F32, "vm.log2.f32");
    DISASM_OP_EXT_F32_BINARY_F32(PowF32, "vm.pow.f32");
    DISASM_OP_EXT_F32_UNARY_F32(RsqrtF32, "vm.rsqrt.f32");
    DISASM_OP_EXT_F32_UNARY_F32(SqrtF32, "vm.sqrt.f32");
    DISASM_OP_EXT_F32_UNARY_F32(TanhF32, "vm.tanh.f32");
    DISASM_OP_EXT_F32_UNARY_F32(ErfF32, "vm.erf.f32");

    //===----------------------------------------------------------------===//
    // ExtF32: Casting and type conversion/emulation
    //===----------------------------------------------------------------===//

    DISASM_OP(EXT_F32, CastSI32F32) {
      uint16_t operand_reg = VM_ParseOperandRegI32("operand");
      uint16_t result_reg = VM_ParseResultRegF32("result");
      EMIT_F32_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.cast.si32.f32 "));
      EMIT_I32_REG_NAME(operand_reg);
      EMIT_OPTIONAL_VALUE_I32(regs->i32[operand_reg]);
      break;
    }
    DISASM_OP(EXT_F32, CastUI32F32) {
      uint16_t operand_reg = VM_ParseOperandRegI32("operand");
      uint16_t result_reg = VM_ParseResultRegF32("result");
      EMIT_F32_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.cast.ui32.f32 "));
      EMIT_I32_REG_NAME(operand_reg);
      EMIT_OPTIONAL_VALUE_I32(regs->i32[operand_reg]);
      break;
    }
    DISASM_OP(EXT_F32, CastF32SI32) {
      uint16_t operand_reg = VM_ParseOperandRegF32("operand");
      uint16_t result_reg = VM_ParseResultRegI32("result");
      EMIT_I32_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.cast.f32.sif32 "));
      EMIT_F32_REG_NAME(operand_reg);
      EMIT_OPTIONAL_VALUE_F32(regs->i32[operand_reg]);
      break;
    }
    DISASM_OP(EXT_F32, CastF32UI32) {
      uint16_t operand_reg = VM_ParseOperandRegF32("operand");
      uint16_t result_reg = VM_ParseResultRegI32("result");
      EMIT_I32_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.cast.f32.uif32 "));
      EMIT_F32_REG_NAME(operand_reg);
      EMIT_OPTIONAL_VALUE_F32(regs->i32[operand_reg]);
      break;
    }
    DISASM_OP(EXT_F32, BitcastI32F32) {
      uint16_t operand_reg = VM_ParseOperandRegI32("operand");
      uint16_t result_reg = VM_ParseResultRegF32("result");
      EMIT_F32_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.bitcast.i32.f32 "));
      EMIT_I32_REG_NAME(operand_reg);
      EMIT_OPTIONAL_VALUE_I32(regs->i32[operand_reg]);
      break;
    }
    DISASM_OP(EXT_F32, BitcastF32I32) {
      uint16_t operand_reg = VM_ParseOperandRegF32("operand");
      uint16_t result_reg = VM_ParseResultRegI32("result");
      EMIT_I32_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.bitcast.f32.i32 "));
      EMIT_F32_REG_NAME(operand_reg);
      EMIT_OPTIONAL_VALUE_F32(regs->i32[operand_reg]);
      break;
    }

    //===----------------------------------------------------------------===//
    // ExtF32: Comparison ops
    //===----------------------------------------------------------------===//

#define DISASM_OP_EXT_F32_CMP_F32(op_name, op_mnemonic)                \
  DISASM_OP(EXT_F32, op_name) {                                        \
    uint16_t lhs_reg = VM_ParseOperandRegF32("lhs");                   \
    uint16_t rhs_reg = VM_ParseOperandRegF32("rhs");                   \
    uint16_t result_reg = VM_ParseResultRegI32("result");              \
    EMIT_I32_REG_NAME(result_reg);                                     \
    IREE_RETURN_IF_ERROR(                                              \
        iree_string_builder_append_format(b, " = %s ", op_mnemonic));  \
    EMIT_F32_REG_NAME(lhs_reg);                                        \
    EMIT_OPTIONAL_VALUE_F32(regs->i32[lhs_reg]);                       \
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", ")); \
    EMIT_F32_REG_NAME(rhs_reg);                                        \
    EMIT_OPTIONAL_VALUE_F32(regs->i32[rhs_reg]);                       \
    break;                                                             \
  }

    DISASM_OP_EXT_F32_CMP_F32(CmpEQF32O, "vm.cmp.eq.f32.o");
    DISASM_OP_EXT_F32_CMP_F32(CmpEQF32U, "vm.cmp.eq.f32.u");
    DISASM_OP_EXT_F32_CMP_F32(CmpNEF32O, "vm.cmp.ne.f32.o");
    DISASM_OP_EXT_F32_CMP_F32(CmpNEF32U, "vm.cmp.ne.f32.u");
    DISASM_OP_EXT_F32_CMP_F32(CmpLTF32O, "vm.cmp.lt.f32.o");
    DISASM_OP_EXT_F32_CMP_F32(CmpLTF32U, "vm.cmp.lt.f32.u");
    DISASM_OP_EXT_F32_CMP_F32(CmpLTEF32O, "vm.cmp.lte.f32.o");
    DISASM_OP_EXT_F32_CMP_F32(CmpLTEF32U, "vm.cmp.lte.f32.u");
    DISASM_OP(EXT_F32, CmpNaNF32) {
      uint16_t operand_reg = VM_ParseOperandRegF32("operand");
      uint16_t result_reg = VM_ParseResultRegI32("result");
      EMIT_I32_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.cmp.nan.f32 "));
      EMIT_F32_REG_NAME(operand_reg);
      EMIT_OPTIONAL_VALUE_F32(regs->i32[operand_reg]);
      break;
    }

    //===----------------------------------------------------------------===//
    // ExtF32: Buffers
    //===----------------------------------------------------------------===//

    DISASM_OP(EXT_F32, BufferFillF32) {
      bool buffer_is_move;
      uint16_t buffer_reg =
          VM_ParseOperandRegRef("target_buffer", &buffer_is_move);
      uint16_t offset_reg = VM_ParseOperandRegI64("target_offset");
      uint16_t length_reg = VM_ParseOperandRegI64("length");
      uint16_t value_reg = VM_ParseOperandRegF32("value");
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, "vm.buffer.fill.f32 "));
      EMIT_REF_REG_NAME(buffer_reg);
      EMIT_OPTIONAL_VALUE_REF(&regs->ref[buffer_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      EMIT_I64_REG_NAME(offset_reg);
      EMIT_OPTIONAL_VALUE_I64(regs->i32[offset_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      EMIT_I64_REG_NAME(length_reg);
      EMIT_OPTIONAL_VALUE_I64(regs->i32[length_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      EMIT_F32_REG_NAME(value_reg);
      EMIT_OPTIONAL_VALUE_F32(regs->i32[value_reg]);
      break;
    }

    DISASM_OP(EXT_F32, BufferLoadF32) {
      bool buffer_is_move;
      uint16_t buffer_reg =
          VM_ParseOperandRegRef("source_buffer", &buffer_is_move);
      uint16_t offset_reg = VM_ParseOperandRegI64("source_offset");
      uint16_t result_reg = VM_ParseResultRegF32("result");
      EMIT_F32_REG_NAME(result_reg);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, " = vm.buffer.load.f32 "));
      EMIT_REF_REG_NAME(buffer_reg);
      EMIT_OPTIONAL_VALUE_REF(&regs->ref[buffer_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      EMIT_I64_REG_NAME(offset_reg);
      EMIT_OPTIONAL_VALUE_I64(regs->i32[offset_reg]);
      break;
    }

    DISASM_OP(EXT_F32, BufferStoreF32) {
      bool buffer_is_move;
      uint16_t buffer_reg =
          VM_ParseOperandRegRef("target_buffer", &buffer_is_move);
      uint16_t offset_reg = VM_ParseOperandRegI64("target_offset");
      uint16_t value_reg = VM_ParseOperandRegF32("value");
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(b, "vm.buffer.store.f32 "));
      EMIT_F32_REG_NAME(value_reg);
      EMIT_OPTIONAL_VALUE_F32(regs->i32[value_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      EMIT_REF_REG_NAME(buffer_reg);
      EMIT_OPTIONAL_VALUE_REF(&regs->ref[buffer_reg]);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(b, ", "));
      EMIT_I64_REG_NAME(offset_reg);
      EMIT_OPTIONAL_VALUE_I64(regs->i32[offset_reg]);
      break;
    }

    END_DISASM_PREFIX()
#else
    UNHANDLED_DISASM_PREFIX(PrefixExtF32, EXT_F32)
#endif  // IREE_VM_EXT_F32_ENABLE
    UNHANDLED_DISASM_PREFIX(PrefixExtF64, EXT_F64)

    default:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "unhandled core opcode");
  }
  return iree_ok_status();
}

iree_status_t iree_vm_bytecode_trace_disassembly(
    iree_vm_stack_frame_t* frame, iree_vm_source_offset_t pc,
    const iree_vm_registers_t* regs, FILE* file) {
  iree_string_builder_t b;
  iree_string_builder_initialize(iree_allocator_system(), &b);

  // TODO(benvanik): ensure frame is in-sync before call or restore original.
  // It's shady to manipulate the frame here but I know we expect the pc to be
  // valid only on entry/exit from a function.
  frame->pc = pc;

#if IREE_VM_EXECUTION_TRACING_SRC_LOC_ENABLE
  iree_vm_source_location_t source_location;
  iree_status_t status = iree_vm_module_resolve_source_location(
      frame->function.module, frame, &source_location);
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
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        &b, "[%.*s", (int)module_name.size, module_name.data));
    iree_string_view_t function_name = iree_vm_function_name(&frame->function);
    if (iree_string_view_is_empty(function_name)) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          &b, "@%u", (uint32_t)frame->function.ordinal));
    } else {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          &b, ".%.*s", (int)function_name.size, function_name.data));
    }
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
  }

  iree_string_builder_deinitialize(&b);
  return status;
}
