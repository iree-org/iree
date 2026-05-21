// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/bytecode/assembler/bytecode.h"

#include <ctype.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>

#include "iree/vm/bytecode/assembler/source.h"

static iree_status_t iree_vm_bytecode_assembler_append_bytes(
    iree_string_builder_t* builder, iree_host_size_t byte_count,
    uint8_t** out_bytes) {
  char* storage = NULL;
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_inline(builder, byte_count, &storage));
  *out_bytes = (uint8_t*)storage;
  return iree_ok_status();
}

iree_status_t iree_vm_bytecode_assembler_emit_u8(
    iree_vm_bytecode_assembler_module_t* state, uint8_t value) {
  uint8_t* bytes = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_append_bytes(
      &state->bytecode_builder, 1, &bytes));
  bytes[0] = value;
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_assembler_emit_u16(
    iree_vm_bytecode_assembler_module_t* state, uint16_t value) {
  uint8_t* bytes = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_append_bytes(
      &state->bytecode_builder, 2, &bytes));
  bytes[0] = (uint8_t)(value & 0xFFu);
  bytes[1] = (uint8_t)((value >> 8) & 0xFFu);
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_assembler_emit_u32(
    iree_vm_bytecode_assembler_module_t* state, uint32_t value) {
  uint8_t* bytes = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_append_bytes(
      &state->bytecode_builder, 4, &bytes));
  bytes[0] = (uint8_t)(value & 0xFFu);
  bytes[1] = (uint8_t)((value >> 8) & 0xFFu);
  bytes[2] = (uint8_t)((value >> 16) & 0xFFu);
  bytes[3] = (uint8_t)((value >> 24) & 0xFFu);
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_assembler_emit_u64(
    iree_vm_bytecode_assembler_module_t* state, uint64_t value) {
  uint8_t* bytes = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_append_bytes(
      &state->bytecode_builder, 8, &bytes));
  bytes[0] = (uint8_t)(value & 0xFFu);
  bytes[1] = (uint8_t)((value >> 8) & 0xFFu);
  bytes[2] = (uint8_t)((value >> 16) & 0xFFu);
  bytes[3] = (uint8_t)((value >> 24) & 0xFFu);
  bytes[4] = (uint8_t)((value >> 32) & 0xFFu);
  bytes[5] = (uint8_t)((value >> 40) & 0xFFu);
  bytes[6] = (uint8_t)((value >> 48) & 0xFFu);
  bytes[7] = (uint8_t)((value >> 56) & 0xFFu);
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_assembler_emit_f32(
    iree_vm_bytecode_assembler_module_t* state, float value) {
  uint32_t bits = 0;
  memcpy(&bits, &value, sizeof(bits));
  return iree_vm_bytecode_assembler_emit_u32(state, bits);
}

static iree_status_t iree_vm_bytecode_assembler_emit_f64(
    iree_vm_bytecode_assembler_module_t* state, double value) {
  uint64_t bits = 0;
  memcpy(&bits, &value, sizeof(bits));
  return iree_vm_bytecode_assembler_emit_u64(state, bits);
}

static iree_status_t iree_vm_bytecode_assembler_emit_opcode(
    iree_vm_bytecode_assembler_module_t* state,
    const iree_vm_isa_instruction_t* instruction) {
  if (instruction->opcode_set == IREE_VM_ISA_OPCODE_SET_CORE) {
    IREE_RETURN_IF_ERROR(
        iree_vm_bytecode_assembler_emit_u8(state, instruction->opcode));
  } else {
    const iree_vm_isa_opcode_set_descriptor_t* opcode_set =
        iree_vm_isa_opcode_set_descriptor(instruction->opcode_set);
    if (!opcode_set || opcode_set->prefix_opcode == 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "instruction has no opcode prefix");
    }
    IREE_RETURN_IF_ERROR(
        iree_vm_bytecode_assembler_emit_u8(state, opcode_set->prefix_opcode));
    IREE_RETURN_IF_ERROR(
        iree_vm_bytecode_assembler_emit_u8(state, instruction->opcode));
  }

  state->function_requirements |= instruction->required_features;
  state->module_requirements |= instruction->required_features;
  return iree_ok_status();
}

iree_status_t iree_vm_bytecode_assembler_align_bytecode(
    iree_vm_bytecode_assembler_module_t* state, iree_host_size_t alignment) {
  const iree_host_size_t current_size =
      iree_string_builder_size(&state->bytecode_builder);
  const iree_host_size_t aligned_size =
      (current_size + (alignment - 1)) & ~(alignment - 1);
  const iree_host_size_t padding = aligned_size - current_size;
  if (padding == 0) return iree_ok_status();
  uint8_t* bytes = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_append_bytes(
      &state->bytecode_builder, padding, &bytes));
  memset(bytes, 0, padding);
  return iree_ok_status();
}

iree_status_t iree_vm_bytecode_assembler_parse_register(
    iree_vm_bytecode_assembler_module_t* state, iree_string_view_t value,
    iree_vm_isa_register_bank_t register_bank, uint16_t* out_register) {
  value = iree_string_view_trim(value);
  if (register_bank == IREE_VM_ISA_REGISTER_BANK_REF) {
    if (!iree_string_view_consume_prefix(&value, IREE_SV("%r")) ||
        iree_vm_bytecode_assembler_string_view_is_empty(value)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "expected ref register");
    }
    uint32_t register_ordinal = 0;
    if (!iree_string_view_atoi_uint32(value, &register_ordinal) ||
        register_ordinal > IREE_VM_ISA_REF_REGISTER_COUNT) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "ref register ordinal out of range: %.*s",
                              (int)value.size, value.data);
    }
    *out_register =
        (uint16_t)(IREE_VM_ISA_REF_REGISTER_TYPE_BIT | register_ordinal);
    if (register_ordinal + 1 > state->ref_register_count) {
      state->ref_register_count = (uint16_t)(register_ordinal + 1);
    }
    return iree_ok_status();
  }

  if (!iree_string_view_consume_prefix(&value, IREE_SV("%i")) ||
      iree_vm_bytecode_assembler_string_view_is_empty(value)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected value register");
  }

  iree_string_view_t low_ordinal_text = value;
  iree_string_view_t high_ordinal_text = iree_string_view_empty();
  iree_string_view_split(value, ':', &low_ordinal_text, &high_ordinal_text);
  if (register_bank == IREE_VM_ISA_REGISTER_BANK_I64 ||
      register_bank == IREE_VM_ISA_REGISTER_BANK_F64) {
    if (iree_vm_bytecode_assembler_string_view_is_empty(high_ordinal_text)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "expected i64 register like %%i0:1");
    }
  } else if (!iree_vm_bytecode_assembler_string_view_is_empty(
                 high_ordinal_text)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unexpected register pair for scalar register");
  }

  uint32_t register_ordinal = 0;
  if (!iree_string_view_atoi_uint32(low_ordinal_text, &register_ordinal) ||
      register_ordinal > IREE_VM_ISA_I32_REGISTER_COUNT) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "i32 register ordinal out of range: %.*s",
                            (int)low_ordinal_text.size, low_ordinal_text.data);
  }

  uint32_t register_span = 1;
  if (register_bank == IREE_VM_ISA_REGISTER_BANK_I64 ||
      register_bank == IREE_VM_ISA_REGISTER_BANK_F64) {
    uint32_t high_ordinal = 0;
    if (!iree_string_view_atoi_uint32(high_ordinal_text, &high_ordinal) ||
        high_ordinal != register_ordinal + 1) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "i64 register must name adjacent i32 slots");
    }
    register_span = 2;
  } else if (register_bank != IREE_VM_ISA_REGISTER_BANK_I32 &&
             register_bank != IREE_VM_ISA_REGISTER_BANK_F32) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "only scalar value registers are supported");
  }

  *out_register = (uint16_t)register_ordinal;
  if (register_ordinal + register_span > state->i32_register_count) {
    state->i32_register_count = (uint16_t)(register_ordinal + register_span);
  }
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_assembler_parse_register_field(
    iree_vm_bytecode_assembler_module_t* state, iree_string_view_t value,
    const iree_vm_isa_field_t* field, uint16_t* out_register) {
  value = iree_string_view_trim(value);
  if (field->register_bank != IREE_VM_ISA_REGISTER_BANK_REF) {
    return iree_vm_bytecode_assembler_parse_register(
        state, value, field->register_bank, out_register);
  }

  bool is_move = false;
  if (iree_string_view_starts_with(value, IREE_SV("%R"))) {
    is_move = true;
    if (!field->allows_move) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "instruction does not allow ref move operands");
    }
    value = iree_string_view_substr(value, 2, IREE_HOST_SIZE_MAX);
    if (iree_vm_bytecode_assembler_string_view_is_empty(value)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "expected ref register ordinal");
    }
    uint32_t register_ordinal = 0;
    if (!iree_string_view_atoi_uint32(value, &register_ordinal) ||
        register_ordinal > IREE_VM_ISA_REF_REGISTER_COUNT) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "ref register ordinal out of range: %.*s",
                              (int)value.size, value.data);
    }
    *out_register =
        (uint16_t)(IREE_VM_ISA_REF_REGISTER_TYPE_BIT |
                   IREE_VM_ISA_REF_REGISTER_MOVE_BIT | register_ordinal);
    if (register_ordinal + 1 > state->ref_register_count) {
      state->ref_register_count = (uint16_t)(register_ordinal + 1);
    }
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_register(
      state, value, IREE_VM_ISA_REGISTER_BANK_REF, out_register));
  if (is_move) *out_register |= IREE_VM_ISA_REF_REGISTER_MOVE_BIT;
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_assembler_parse_any_register(
    iree_vm_bytecode_assembler_module_t* state, iree_string_view_t value,
    uint16_t* out_register) {
  value = iree_string_view_trim(value);
  if (iree_string_view_starts_with(value, IREE_SV("%r")) ||
      iree_string_view_starts_with(value, IREE_SV("%R"))) {
    const iree_vm_isa_field_t field = {
        .register_bank = IREE_VM_ISA_REGISTER_BANK_REF,
        .allows_move = true,
    };
    return iree_vm_bytecode_assembler_parse_register_field(state, value, &field,
                                                           out_register);
  }
  return iree_vm_bytecode_assembler_parse_register(
      state, value,
      iree_string_view_find_char(value, ':', 0) == IREE_STRING_VIEW_NPOS
          ? IREE_VM_ISA_REGISTER_BANK_I32
          : IREE_VM_ISA_REGISTER_BANK_I64,
      out_register);
}

static iree_vm_isa_register_bank_t iree_vm_bytecode_assembler_value_type_bank(
    iree_vm_isa_value_type_t value_type) {
  switch (value_type) {
    case IREE_VM_ISA_VALUE_TYPE_I32:
      return IREE_VM_ISA_REGISTER_BANK_I32;
    case IREE_VM_ISA_VALUE_TYPE_I64:
      return IREE_VM_ISA_REGISTER_BANK_I64;
    case IREE_VM_ISA_VALUE_TYPE_F32:
      return IREE_VM_ISA_REGISTER_BANK_F32;
    case IREE_VM_ISA_VALUE_TYPE_F64:
      return IREE_VM_ISA_REGISTER_BANK_F64;
    case IREE_VM_ISA_VALUE_TYPE_REF:
      return IREE_VM_ISA_REGISTER_BANK_REF;
    default:
      return IREE_VM_ISA_REGISTER_BANK_NONE;
  }
}

static bool iree_vm_bytecode_assembler_is_space(char c) {
  return isspace((unsigned char)c);
}

static iree_host_size_t iree_vm_bytecode_assembler_find_spaced_char(
    iree_string_view_t value, char c) {
  for (iree_host_size_t i = 0; i < value.size; ++i) {
    if (value.data[i] != c) continue;
    const bool has_space_before =
        i == 0 || iree_vm_bytecode_assembler_is_space(value.data[i - 1]);
    const bool has_space_after =
        i + 1 == value.size ||
        iree_vm_bytecode_assembler_is_space(value.data[i + 1]);
    if (has_space_before && has_space_after) return i;
  }
  return IREE_STRING_VIEW_NPOS;
}

static iree_status_t iree_vm_bytecode_assembler_parse_operand_token(
    iree_string_view_t* operands, iree_string_view_t* out_operand) {
  *operands = iree_string_view_trim(*operands);
  if (iree_vm_bytecode_assembler_string_view_is_empty(*operands)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "missing operand");
  }
  iree_string_view_t operand;
  iree_string_view_t remainder;
  iree_vm_bytecode_assembler_split_operand(*operands, &operand, &remainder);
  *out_operand = operand;
  *operands = remainder;
  return iree_ok_status();
}

static iree_string_view_t iree_vm_bytecode_assembler_canonical_type_name(
    iree_string_view_t type_name) {
  return iree_string_view_trim(type_name);
}

static iree_status_t iree_vm_bytecode_assembler_emit_type_ordinal(
    iree_vm_bytecode_assembler_module_t* state, iree_string_view_t type_name) {
  type_name = iree_vm_bytecode_assembler_canonical_type_name(type_name);
  iree_host_size_t ordinal = 0;
  if (iree_string_view_starts_with(type_name, IREE_SV("vm."))) {
    for (iree_host_size_t i = 0; i < state->type_count; ++i) {
      iree_string_view_t full_name = state->types[i].full_name;
      if (full_name.size == type_name.size + 1 && full_name.data[0] == '!' &&
          memcmp(full_name.data + 1, type_name.data, type_name.size) == 0) {
        ordinal = i;
        return iree_vm_bytecode_assembler_emit_u32(state, (uint32_t)ordinal);
      }
    }
  }
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_lookup_or_append_type(
      state, type_name, &ordinal));
  if (ordinal > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "type ordinal is too large");
  }
  return iree_vm_bytecode_assembler_emit_u32(state, (uint32_t)ordinal);
}

static bool iree_vm_bytecode_assembler_split_type_suffix(
    iree_string_view_t value, iree_string_view_t* out_value,
    iree_string_view_t* out_type_name) {
  const iree_host_size_t colon_position =
      iree_vm_bytecode_assembler_find_spaced_char(value, ':');
  if (colon_position == IREE_STRING_VIEW_NPOS) return false;
  *out_value =
      iree_string_view_trim(iree_string_view_substr(value, 0, colon_position));
  *out_type_name = iree_string_view_trim(
      iree_string_view_substr(value, colon_position + 1, IREE_HOST_SIZE_MAX));
  return true;
}

static iree_status_t iree_vm_bytecode_assembler_list_element_type(
    iree_string_view_t list_type_name, iree_string_view_t* out_element_type) {
  list_type_name = iree_string_view_trim(list_type_name);
  if (!iree_string_view_starts_with(list_type_name, IREE_SV("!vm.list<")) ||
      list_type_name.size <= strlen("!vm.list<>") ||
      list_type_name.data[list_type_name.size - 1] != '>') {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected list type annotation");
  }
  *out_element_type = iree_string_view_trim(
      iree_string_view_substr(list_type_name, strlen("!vm.list<"),
                              list_type_name.size - strlen("!vm.list<") - 1));
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_assembler_parse_u32_function_ordinal(
    iree_string_view_t value, uint32_t* out_ordinal) {
  value = iree_string_view_trim(value);
  if (!iree_string_view_atoi_uint32(value, out_ordinal)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected function ordinal");
  }
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_assembler_emit_function_attr(
    iree_vm_bytecode_assembler_module_t* state, iree_string_view_t operand) {
  operand = iree_string_view_trim(operand);
  if (!iree_string_view_consume_prefix_char(&operand, '@') ||
      iree_vm_bytecode_assembler_string_view_is_empty(operand)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected function symbol");
  }

  iree_string_view_t module_prefix = state->module_name;
  iree_string_view_t ordinal_text = iree_string_view_empty();
  if (iree_string_view_split(operand, ':', &module_prefix, &ordinal_text) !=
      -1) {
    if (!iree_string_view_equal(module_prefix, state->module_name)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "internal function ordinal uses wrong module");
    }
    uint32_t ordinal = 0;
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_u32_function_ordinal(
        ordinal_text, &ordinal));
    return iree_vm_bytecode_assembler_emit_u32(state, ordinal);
  }

  const iree_host_size_t import_ordinal =
      iree_vm_bytecode_assembler_find_import_ordinal(state, operand);
  if (import_ordinal != IREE_HOST_SIZE_MAX) {
    if (import_ordinal > IREE_VM_ISA_FUNCTION_ORDINAL_IMPORT_BIT) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "import ordinal is too large");
    }
    return iree_vm_bytecode_assembler_emit_u32(
        state,
        IREE_VM_ISA_FUNCTION_ORDINAL_IMPORT_BIT | (uint32_t)import_ordinal);
  }

  iree_string_view_t local_name = operand;
  iree_string_view_t remainder = iree_string_view_empty();
  if (iree_string_view_split(operand, '.', &module_prefix, &remainder) != -1 &&
      iree_string_view_equal(module_prefix, state->module_name)) {
    local_name = remainder;
  }

  const iree_host_size_t function_ordinal =
      iree_vm_bytecode_assembler_find_function_ordinal(state, local_name);
  const uint32_t patch_offset =
      (uint32_t)iree_string_builder_size(&state->bytecode_builder);
  if (function_ordinal == IREE_HOST_SIZE_MAX) {
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_emit_u32(state, 0));
    return iree_vm_bytecode_assembler_append_function_fixup(state, local_name,
                                                            patch_offset);
  }
  if (function_ordinal > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "function ordinal is too large");
  }
  return iree_vm_bytecode_assembler_emit_u32(state, (uint32_t)function_ordinal);
}

static iree_status_t iree_vm_bytecode_assembler_patch_u16(
    iree_vm_bytecode_assembler_module_t* state, uint32_t bytecode_offset,
    uint16_t value) {
  if (bytecode_offset + 2 >
      iree_string_builder_size(&state->bytecode_builder)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "patch is out of range");
  }
  uint8_t* target = (uint8_t*)state->bytecode_builder.buffer + bytecode_offset;
  target[0] = (uint8_t)(value & 0xFFu);
  target[1] = (uint8_t)((value >> 8) & 0xFFu);
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_assembler_parse_float_text(
    iree_string_view_t value, char* buffer, iree_host_size_t buffer_capacity) {
  value = iree_string_view_trim(value);
  if (value.size == 0 || value.size >= buffer_capacity) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid floating-point immediate");
  }
  memcpy(buffer, value.data, value.size);
  buffer[value.size] = 0;
  errno = 0;
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_assembler_parse_f32(
    iree_string_view_t value, float* out_value) {
  char buffer[64] = {0};
  char* end = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_float_text(
      value, buffer, IREE_ARRAYSIZE(buffer)));
  *out_value = strtof(buffer, &end);
  if (buffer == end || *end != 0 || errno == ERANGE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected f32 immediate");
  }
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_assembler_parse_f64(
    iree_string_view_t value, double* out_value) {
  char buffer[64] = {0};
  char* end = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_float_text(
      value, buffer, IREE_ARRAYSIZE(buffer)));
  *out_value = strtod(buffer, &end);
  if (buffer == end || *end != 0 || errno == ERANGE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected f64 immediate");
  }
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_assembler_parse_register_for_bank(
    iree_vm_bytecode_assembler_module_t* state, iree_string_view_t operand,
    iree_vm_isa_register_bank_t register_bank, uint16_t* out_register) {
  if (register_bank == IREE_VM_ISA_REGISTER_BANK_NONE) {
    return iree_vm_bytecode_assembler_parse_any_register(state, operand,
                                                         out_register);
  }
  if ((register_bank == IREE_VM_ISA_REGISTER_BANK_I64 ||
       register_bank == IREE_VM_ISA_REGISTER_BANK_F64) &&
      iree_string_view_find_char(operand, ':', 0) == IREE_STRING_VIEW_NPOS) {
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_register(
        state, operand, IREE_VM_ISA_REGISTER_BANK_I32, out_register));
    if (*out_register + 2 > state->i32_register_count) {
      state->i32_register_count = (uint16_t)(*out_register + 2);
    }
    return iree_ok_status();
  }
  return iree_vm_bytecode_assembler_parse_register(state, operand,
                                                   register_bank, out_register);
}

static uint32_t iree_vm_bytecode_assembler_register_bank_storage_size(
    iree_vm_isa_register_bank_t register_bank) {
  switch (register_bank) {
    case IREE_VM_ISA_REGISTER_BANK_I32:
    case IREE_VM_ISA_REGISTER_BANK_F32:
      return 4;
    case IREE_VM_ISA_REGISTER_BANK_I64:
    case IREE_VM_ISA_REGISTER_BANK_F64:
      return 8;
    default:
      return 0;
  }
}

static uint32_t iree_vm_bytecode_assembler_global_storage_size(
    const iree_vm_isa_instruction_t* instruction,
    iree_host_size_t global_field_ordinal) {
  for (iree_host_size_t i = global_field_ordinal + 1;
       i < instruction->field_count; ++i) {
    const iree_vm_isa_field_t* field = &instruction->fields[i];
    if (field->kind != IREE_VM_ISA_FIELD_KIND_REGISTER) continue;
    return iree_vm_bytecode_assembler_register_bank_storage_size(
        field->register_bank);
  }
  return 0;
}

static iree_status_t iree_vm_bytecode_assembler_parse_storage_ordinal(
    iree_string_view_t value, iree_string_view_t prefix,
    uint32_t* out_ordinal) {
  value = iree_string_view_trim(value);
  if (!iree_string_view_consume_prefix(&value, prefix)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected storage reference");
  }
  if (value.size == 0 || value.data[value.size - 1] != ']') {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unterminated storage reference");
  }
  value = iree_string_view_substr(value, 0, value.size - 1);
  if (!iree_string_view_atoi_uint32(value, out_ordinal)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid storage ordinal");
  }
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_assembler_emit_global_attr(
    iree_vm_bytecode_assembler_module_t* state, iree_string_view_t operand,
    uint32_t storage_size) {
  operand = iree_string_view_trim(operand);
  uint32_t ordinal = 0;
  if (storage_size == 0) {
    if (iree_string_view_starts_with(operand, IREE_SV(".refs["))) {
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_storage_ordinal(
          operand, IREE_SV(".refs["), &ordinal));
      if (ordinal == UINT32_MAX) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "global ref ordinal is too large");
      }
      if (ordinal + 1 > state->global_ref_count) {
        state->global_ref_count = ordinal + 1;
      }
      return iree_vm_bytecode_assembler_emit_u32(state, ordinal);
    }
  } else if (iree_string_view_starts_with(operand, IREE_SV(".rwdata["))) {
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_storage_ordinal(
        operand, IREE_SV(".rwdata["), &ordinal));
    if (ordinal > UINT32_MAX - storage_size) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "global byte offset is too large");
    }
    if (ordinal + storage_size > state->global_byte_capacity) {
      state->global_byte_capacity = ordinal + storage_size;
    }
    return iree_vm_bytecode_assembler_emit_u32(state, ordinal);
  }

  iree_string_view_t symbol = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(
      iree_vm_bytecode_assembler_parse_symbol(operand, &symbol));
  const uint32_t patch_offset =
      (uint32_t)iree_string_builder_size(&state->bytecode_builder);
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_emit_u32(state, 0));
  return iree_vm_bytecode_assembler_append_global_fixup(
      state, symbol, patch_offset, storage_size);
}

static iree_status_t iree_vm_bytecode_assembler_emit_variadic_register_list(
    iree_vm_bytecode_assembler_module_t* state, iree_string_view_t operands,
    iree_vm_isa_register_bank_t register_bank) {
  operands = iree_string_view_trim(operands);

  uint16_t register_count = 0;
  iree_string_view_t scan = operands;
  while (!iree_vm_bytecode_assembler_string_view_is_empty(scan)) {
    iree_string_view_t operand;
    IREE_RETURN_IF_ERROR(
        iree_vm_bytecode_assembler_parse_operand_token(&scan, &operand));
    if (register_count == UINT16_MAX) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "variadic register list is too large");
    }
    ++register_count;
  }

  IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_align_bytecode(
      state, IREE_REGISTER_ORDINAL_SIZE));
  IREE_RETURN_IF_ERROR(
      iree_vm_bytecode_assembler_emit_u16(state, register_count));

  while (!iree_vm_bytecode_assembler_string_view_is_empty(operands)) {
    iree_string_view_t operand;
    IREE_RETURN_IF_ERROR(
        iree_vm_bytecode_assembler_parse_operand_token(&operands, &operand));
    uint16_t register_ordinal = 0;
    if (register_bank == IREE_VM_ISA_REGISTER_BANK_REF) {
      const iree_vm_isa_field_t field = {
          .register_bank = IREE_VM_ISA_REGISTER_BANK_REF,
          .allows_move = true,
      };
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_register_field(
          state, operand, &field, &register_ordinal));
    } else {
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_register_for_bank(
          state, operand, register_bank, &register_ordinal));
    }
    IREE_RETURN_IF_ERROR(
        iree_vm_bytecode_assembler_emit_u16(state, register_ordinal));
  }
  return iree_ok_status();
}

static bool iree_vm_bytecode_assembler_instruction_has_encoding(
    const iree_vm_isa_instruction_t* instruction, const char* encoding) {
  return iree_vm_bytecode_assembler_string_view_equal_cstring(
      instruction->encoding, encoding);
}

static bool iree_vm_bytecode_assembler_is_select_instruction(
    const iree_vm_isa_instruction_t* instruction) {
  return iree_vm_bytecode_assembler_instruction_has_encoding(instruction,
                                                             "i32_select") ||
         iree_vm_bytecode_assembler_instruction_has_encoding(instruction,
                                                             "i64_select") ||
         iree_vm_bytecode_assembler_instruction_has_encoding(instruction,
                                                             "f32_select") ||
         iree_vm_bytecode_assembler_instruction_has_encoding(instruction,
                                                             "f64_select") ||
         iree_vm_bytecode_assembler_instruction_has_encoding(instruction,
                                                             "select_ref");
}

static bool iree_vm_bytecode_assembler_is_switch_instruction(
    const iree_vm_isa_instruction_t* instruction) {
  return iree_vm_bytecode_assembler_instruction_has_encoding(instruction,
                                                             "i32_switch") ||
         iree_vm_bytecode_assembler_instruction_has_encoding(instruction,
                                                             "i64_switch") ||
         iree_vm_bytecode_assembler_instruction_has_encoding(instruction,
                                                             "f32_switch") ||
         iree_vm_bytecode_assembler_instruction_has_encoding(instruction,
                                                             "f64_switch");
}

static bool iree_vm_bytecode_assembler_is_buffer_store_instruction(
    const iree_vm_isa_instruction_t* instruction) {
  return iree_vm_bytecode_assembler_instruction_has_encoding(
             instruction, "i32_buffer_store") ||
         iree_vm_bytecode_assembler_instruction_has_encoding(
             instruction, "i64_buffer_store") ||
         iree_vm_bytecode_assembler_instruction_has_encoding(
             instruction, "f32_buffer_store") ||
         iree_vm_bytecode_assembler_instruction_has_encoding(
             instruction, "f64_buffer_store");
}

static iree_status_t iree_vm_bytecode_assembler_emit_select_instruction(
    iree_vm_bytecode_assembler_module_t* state,
    const iree_vm_isa_instruction_t* instruction,
    iree_string_view_t result_register_text, iree_string_view_t operands) {
  const bool is_ref_select =
      iree_vm_bytecode_assembler_instruction_has_encoding(instruction,
                                                          "select_ref");
  if ((!is_ref_select && instruction->field_count != 4) ||
      (is_ref_select && instruction->field_count != 5) ||
      instruction->fields[0].kind != IREE_VM_ISA_FIELD_KIND_REGISTER ||
      instruction->fields[is_ref_select ? 2 : 1].kind !=
          IREE_VM_ISA_FIELD_KIND_REGISTER ||
      instruction->fields[is_ref_select ? 3 : 2].kind !=
          IREE_VM_ISA_FIELD_KIND_REGISTER ||
      instruction->fields[is_ref_select ? 4 : 3].kind !=
          IREE_VM_ISA_FIELD_KIND_REGISTER) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "select instruction has unexpected encoding");
  }

  const iree_host_size_t question_position =
      iree_string_view_find_char(operands, '?', 0);
  if (question_position == IREE_STRING_VIEW_NPOS) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "expected select condition like %%i0 ? %%i1 : %%i2");
  }
  iree_string_view_t condition = iree_string_view_trim(
      iree_string_view_substr(operands, 0, question_position));
  operands = iree_string_view_trim(iree_string_view_substr(
      operands, question_position + 1, IREE_HOST_SIZE_MAX));
  const iree_host_size_t colon_position =
      iree_vm_bytecode_assembler_find_spaced_char(operands, ':');
  if (colon_position == IREE_STRING_VIEW_NPOS) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected select false value after ':'");
  }
  iree_string_view_t true_value = iree_string_view_trim(
      iree_string_view_substr(operands, 0, colon_position));
  iree_string_view_t false_value =
      iree_string_view_trim(iree_string_view_substr(
          operands, colon_position + 1, IREE_HOST_SIZE_MAX));
  iree_string_view_t type_name = iree_string_view_empty();
  if (is_ref_select) {
    const iree_host_size_t arrow_position =
        iree_string_view_find_char(false_value, '-', 0);
    if (arrow_position == IREE_STRING_VIEW_NPOS ||
        arrow_position + 1 >= false_value.size ||
        false_value.data[arrow_position + 1] != '>') {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "expected select result type after '->'");
    }
    type_name = iree_string_view_trim(iree_string_view_substr(
        false_value, arrow_position + 2, IREE_HOST_SIZE_MAX));
    false_value = iree_string_view_trim(
        iree_string_view_substr(false_value, 0, arrow_position));
  }

  uint16_t register_ordinal = 0;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_register_field(
      state, condition, &instruction->fields[0], &register_ordinal));
  IREE_RETURN_IF_ERROR(
      iree_vm_bytecode_assembler_emit_u16(state, register_ordinal));
  if (is_ref_select) {
    IREE_RETURN_IF_ERROR(
        iree_vm_bytecode_assembler_emit_type_ordinal(state, type_name));
  }
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_register_field(
      state, true_value, &instruction->fields[is_ref_select ? 2 : 1],
      &register_ordinal));
  IREE_RETURN_IF_ERROR(
      iree_vm_bytecode_assembler_emit_u16(state, register_ordinal));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_register_field(
      state, false_value, &instruction->fields[is_ref_select ? 3 : 2],
      &register_ordinal));
  IREE_RETURN_IF_ERROR(
      iree_vm_bytecode_assembler_emit_u16(state, register_ordinal));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_register_field(
      state, result_register_text, &instruction->fields[is_ref_select ? 4 : 3],
      &register_ordinal));
  return iree_vm_bytecode_assembler_emit_u16(state, register_ordinal);
}

static iree_status_t iree_vm_bytecode_assembler_emit_switch_instruction(
    iree_vm_bytecode_assembler_module_t* state,
    const iree_vm_isa_instruction_t* instruction,
    iree_string_view_t result_register_text, iree_string_view_t operands) {
  if (instruction->field_count != 4 ||
      instruction->fields[0].kind != IREE_VM_ISA_FIELD_KIND_REGISTER ||
      instruction->fields[1].kind != IREE_VM_ISA_FIELD_KIND_REGISTER ||
      instruction->fields[2].kind != IREE_VM_ISA_FIELD_KIND_VARIADIC_OPERANDS ||
      instruction->fields[3].kind != IREE_VM_ISA_FIELD_KIND_REGISTER) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "switch instruction has unexpected encoding");
  }

  const iree_host_size_t list_open_position =
      iree_string_view_find_char(operands, '[', 0);
  if (list_open_position == IREE_STRING_VIEW_NPOS) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected switch value list");
  }
  const iree_host_size_t list_close_position =
      iree_string_view_find_char(operands, ']', list_open_position + 1);
  if (list_close_position == IREE_STRING_VIEW_NPOS) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unterminated switch value list");
  }

  iree_string_view_t index = iree_string_view_trim(
      iree_string_view_substr(operands, 0, list_open_position));
  iree_string_view_t values = iree_string_view_trim(
      iree_string_view_substr(operands, list_open_position + 1,
                              list_close_position - list_open_position - 1));
  iree_string_view_t default_value =
      iree_string_view_trim(iree_string_view_substr(
          operands, list_close_position + 1, IREE_HOST_SIZE_MAX));
  if (!iree_string_view_consume_prefix(&default_value, IREE_SV("else"))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected switch default value after 'else'");
  }
  default_value = iree_string_view_trim(default_value);

  uint16_t register_ordinal = 0;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_register(
      state, index, instruction->fields[0].register_bank, &register_ordinal));
  IREE_RETURN_IF_ERROR(
      iree_vm_bytecode_assembler_emit_u16(state, register_ordinal));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_register(
      state, default_value, instruction->fields[1].register_bank,
      &register_ordinal));
  IREE_RETURN_IF_ERROR(
      iree_vm_bytecode_assembler_emit_u16(state, register_ordinal));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_emit_variadic_register_list(
      state, values,
      iree_vm_bytecode_assembler_value_type_bank(
          instruction->fields[2].value_type)));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_register(
      state, result_register_text, instruction->fields[3].register_bank,
      &register_ordinal));
  return iree_vm_bytecode_assembler_emit_u16(state, register_ordinal);
}

static iree_status_t iree_vm_bytecode_assembler_parse_quoted_string(
    iree_string_view_t value, iree_string_view_t* out_string) {
  value = iree_string_view_trim(value);
  if (!iree_string_view_consume_prefix_char(&value, '"')) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected quoted string");
  }
  const iree_host_size_t closing_quote_position =
      iree_string_view_find_char(value, '"', 0);
  if (closing_quote_position == IREE_STRING_VIEW_NPOS) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unterminated quoted string");
  }
  *out_string = iree_string_view_substr(value, 0, closing_quote_position);
  value = iree_string_view_trim(iree_string_view_substr(
      value, closing_quote_position + 1, IREE_HOST_SIZE_MAX));
  if (!iree_vm_bytecode_assembler_string_view_is_empty(value)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unexpected text after quoted string");
  }
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_assembler_parse_branch_target(
    iree_string_view_t* operands, iree_string_view_t* out_label_name,
    iree_string_view_t* out_remaps) {
  iree_string_view_t scan = iree_string_view_trim(*operands);
  if (!iree_string_view_consume_prefix_char(&scan, '^')) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected branch target");
  }

  iree_host_size_t label_length = 0;
  while (label_length < scan.size &&
         iree_vm_bytecode_assembler_is_symbol_char(scan.data[label_length])) {
    ++label_length;
  }
  if (label_length == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected branch target label");
  }
  *out_label_name = iree_string_view_substr(scan, 0, label_length);
  scan = iree_string_view_trim(
      iree_string_view_substr(scan, label_length, IREE_HOST_SIZE_MAX));

  if (iree_string_view_consume_prefix_char(&scan, '(')) {
    const iree_host_size_t close_position =
        iree_string_view_find_char(scan, ')', 0);
    if (close_position == IREE_STRING_VIEW_NPOS) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unterminated branch operand list");
    }
    *out_remaps =
        iree_string_view_trim(iree_string_view_substr(scan, 0, close_position));
    scan = iree_string_view_trim(
        iree_string_view_substr(scan, close_position + 1, IREE_HOST_SIZE_MAX));
  } else {
    *out_remaps = iree_string_view_empty();
  }

  if (iree_string_view_consume_prefix_char(&scan, ',')) {
    scan = iree_string_view_trim(scan);
  }
  *operands = scan;
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_assembler_emit_branch_remap_list(
    iree_vm_bytecode_assembler_module_t* state, iree_string_view_t remaps) {
  remaps = iree_string_view_trim(remaps);

  uint16_t remap_count = 0;
  iree_string_view_t scan = remaps;
  while (!iree_vm_bytecode_assembler_string_view_is_empty(scan)) {
    iree_string_view_t pair;
    iree_string_view_t remainder;
    iree_string_view_split(scan, ',', &pair, &remainder);
    if (iree_vm_bytecode_assembler_string_view_is_empty(
            iree_string_view_trim(pair))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "empty branch remap");
    }
    if (remap_count == UINT16_MAX) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "branch remap list is too large");
    }
    ++remap_count;
    scan = iree_string_view_trim(remainder);
  }

  IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_align_bytecode(
      state, IREE_REGISTER_ORDINAL_SIZE));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_emit_u16(state, remap_count));

  while (!iree_vm_bytecode_assembler_string_view_is_empty(remaps)) {
    iree_string_view_t pair;
    iree_string_view_t remainder;
    iree_string_view_split(remaps, ',', &pair, &remainder);
    pair = iree_string_view_trim(pair);

    const iree_host_size_t arrow_position =
        iree_string_view_find_char(pair, '-', 0);
    if (arrow_position == IREE_STRING_VIEW_NPOS ||
        arrow_position + 1 >= pair.size ||
        pair.data[arrow_position + 1] != '>') {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "expected branch remap like %%i0->%%i1");
    }
    iree_string_view_t source =
        iree_string_view_substr(pair, 0, arrow_position);
    iree_string_view_t target =
        iree_string_view_substr(pair, arrow_position + 2, IREE_HOST_SIZE_MAX);

    uint16_t source_register = 0;
    uint16_t target_register = 0;
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_any_register(
        state, source, &source_register));
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_any_register(
        state, target, &target_register));
    IREE_RETURN_IF_ERROR(
        iree_vm_bytecode_assembler_emit_u16(state, source_register));
    IREE_RETURN_IF_ERROR(
        iree_vm_bytecode_assembler_emit_u16(state, target_register));

    remaps = iree_string_view_trim(remainder);
  }

  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_assembler_emit_u16_list(
    iree_vm_bytecode_assembler_module_t* state, iree_string_view_t values) {
  values = iree_string_view_trim(values);

  uint16_t value_count = 0;
  iree_string_view_t scan = values;
  while (!iree_vm_bytecode_assembler_string_view_is_empty(scan)) {
    iree_string_view_t value;
    IREE_RETURN_IF_ERROR(
        iree_vm_bytecode_assembler_parse_operand_token(&scan, &value));
    if (value_count == UINT16_MAX) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "u16 list is too large");
    }
    ++value_count;
  }

  IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_align_bytecode(
      state, IREE_REGISTER_ORDINAL_SIZE));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_emit_u16(state, value_count));

  while (!iree_vm_bytecode_assembler_string_view_is_empty(values)) {
    iree_string_view_t value_text;
    IREE_RETURN_IF_ERROR(
        iree_vm_bytecode_assembler_parse_operand_token(&values, &value_text));
    uint32_t value = 0;
    if (!iree_string_view_atoi_uint32(value_text, &value) ||
        value > UINT16_MAX) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "expected u16 list entry");
    }
    IREE_RETURN_IF_ERROR(
        iree_vm_bytecode_assembler_emit_u16(state, (uint16_t)value));
  }
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_assembler_parse_call_arguments(
    iree_string_view_t operands, iree_string_view_t* out_callee,
    iree_string_view_t* out_segment_sizes, iree_string_view_t* out_arguments,
    iree_string_view_t* out_remainder) {
  operands = iree_string_view_trim(operands);
  const iree_host_size_t open_position =
      iree_string_view_find_char(operands, '(', 0);
  if (open_position == IREE_STRING_VIEW_NPOS) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected call argument list");
  }
  iree_string_view_t callee_and_segments = iree_string_view_trim(
      iree_string_view_substr(operands, 0, open_position));
  const iree_host_size_t close_position =
      iree_string_view_find_char(operands, ')', open_position + 1);
  if (close_position == IREE_STRING_VIEW_NPOS) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unterminated call argument list");
  }
  *out_arguments = iree_string_view_trim(iree_string_view_substr(
      operands, open_position + 1, close_position - open_position - 1));
  *out_remainder = iree_string_view_trim(iree_string_view_substr(
      operands, close_position + 1, IREE_HOST_SIZE_MAX));

  iree_string_view_t callee = iree_string_view_empty();
  iree_string_view_t segments = iree_string_view_empty();
  iree_vm_bytecode_assembler_split_first_token(callee_and_segments, &callee,
                                               &segments);
  *out_callee = callee;
  segments = iree_string_view_trim(segments);
  *out_segment_sizes = iree_string_view_empty();
  if (!iree_vm_bytecode_assembler_string_view_is_empty(segments)) {
    if (!iree_string_view_consume_prefix(&segments, IREE_SV("segments["))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "expected call segment size list");
    }
    if (segments.size == 0 || segments.data[segments.size - 1] != ']') {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unterminated call segment size list");
    }
    *out_segment_sizes =
        iree_string_view_substr(segments, 0, segments.size - 1);
  }
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_assembler_parse_call_continuation(
    iree_string_view_t operands, iree_string_view_t* out_label_name,
    iree_string_view_t* out_results) {
  operands = iree_string_view_trim(operands);
  if (!iree_string_view_consume_prefix(&operands, IREE_SV("->"))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected call continuation after '->'");
  }
  operands = iree_string_view_trim(operands);
  if (!iree_string_view_consume_prefix_char(&operands, '^')) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected continuation block label");
  }
  iree_host_size_t label_length = 0;
  while (
      label_length < operands.size &&
      iree_vm_bytecode_assembler_is_symbol_char(operands.data[label_length])) {
    ++label_length;
  }
  if (label_length == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected continuation block label");
  }
  *out_label_name = iree_string_view_substr(operands, 0, label_length);
  operands = iree_string_view_trim(
      iree_string_view_substr(operands, label_length, IREE_HOST_SIZE_MAX));
  if (!iree_string_view_consume_prefix_char(&operands, '(')) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected continuation result list");
  }
  const iree_host_size_t close_position =
      iree_string_view_find_char(operands, ')', 0);
  if (close_position == IREE_STRING_VIEW_NPOS) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unterminated continuation result list");
  }
  *out_results = iree_string_view_trim(
      iree_string_view_substr(operands, 0, close_position));
  operands = iree_string_view_trim(iree_string_view_substr(
      operands, close_position + 1, IREE_HOST_SIZE_MAX));
  if (!iree_vm_bytecode_assembler_string_view_is_empty(operands)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unexpected text after call continuation");
  }
  return iree_ok_status();
}

static bool iree_vm_bytecode_assembler_is_call_instruction(
    const iree_vm_isa_instruction_t* instruction) {
  return iree_vm_bytecode_assembler_instruction_has_encoding(instruction,
                                                             "call") ||
         iree_vm_bytecode_assembler_instruction_has_encoding(instruction,
                                                             "call_variadic") ||
         iree_vm_bytecode_assembler_instruction_has_encoding(
             instruction, "yieldable_call") ||
         iree_vm_bytecode_assembler_instruction_has_encoding(
             instruction, "call_variadic_yieldable");
}

static iree_status_t iree_vm_bytecode_assembler_emit_call_instruction(
    iree_vm_bytecode_assembler_module_t* state,
    const iree_vm_isa_instruction_t* instruction, bool has_result,
    iree_string_view_t result_register_text, iree_string_view_t operands) {
  const bool is_variadic = iree_vm_bytecode_assembler_instruction_has_encoding(
                               instruction, "call_variadic") ||
                           iree_vm_bytecode_assembler_instruction_has_encoding(
                               instruction, "call_variadic_yieldable");
  const bool is_yieldable = iree_vm_bytecode_assembler_instruction_has_encoding(
                                instruction, "yieldable_call") ||
                            iree_vm_bytecode_assembler_instruction_has_encoding(
                                instruction, "call_variadic_yieldable");
  iree_string_view_t callee = iree_string_view_empty();
  iree_string_view_t segment_sizes = iree_string_view_empty();
  iree_string_view_t arguments = iree_string_view_empty();
  iree_string_view_t remainder = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_call_arguments(
      operands, &callee, &segment_sizes, &arguments, &remainder));
  if (is_variadic &&
      iree_vm_bytecode_assembler_string_view_is_empty(segment_sizes)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "variadic calls must list segment sizes");
  }

  IREE_RETURN_IF_ERROR(
      iree_vm_bytecode_assembler_emit_function_attr(state, callee));
  if (is_variadic) {
    IREE_RETURN_IF_ERROR(
        iree_vm_bytecode_assembler_emit_u16_list(state, segment_sizes));
  }
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_emit_variadic_register_list(
      state, arguments, IREE_VM_ISA_REGISTER_BANK_NONE));

  iree_string_view_t results = result_register_text;
  if (is_yieldable) {
    iree_string_view_t label_name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_call_continuation(
        remainder, &label_name, &results));
    const uint32_t patch_offset =
        (uint32_t)iree_string_builder_size(&state->bytecode_builder);
    IREE_RETURN_IF_ERROR(
        iree_vm_bytecode_assembler_emit_u32(state, /*value=*/0));
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_append_fixup(
        state, label_name, patch_offset));
  } else if (!iree_vm_bytecode_assembler_string_view_is_empty(remainder)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unexpected text after call");
  }

  return iree_vm_bytecode_assembler_emit_variadic_register_list(
      state, has_result || is_yieldable ? results : iree_string_view_empty(),
      IREE_VM_ISA_REGISTER_BANK_NONE);
}

static iree_status_t iree_vm_bytecode_assembler_emit_branch_table_instruction(
    iree_vm_bytecode_assembler_module_t* state,
    const iree_vm_isa_instruction_t* instruction, iree_string_view_t operands) {
  if (instruction->field_count != 5 ||
      instruction->fields[0].kind != IREE_VM_ISA_FIELD_KIND_REGISTER ||
      instruction->fields[1].kind != IREE_VM_ISA_FIELD_KIND_BRANCH_TARGET ||
      instruction->fields[2].kind != IREE_VM_ISA_FIELD_KIND_BRANCH_OPERANDS ||
      instruction->fields[3].kind != IREE_VM_ISA_FIELD_KIND_CONST_I16 ||
      instruction->fields[4].kind !=
          IREE_VM_ISA_FIELD_KIND_BRANCH_TABLE_CASES) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "branch table has unexpected encoding");
  }

  const iree_host_size_t body_open =
      iree_string_view_find_char(operands, '{', 0);
  if (body_open == IREE_STRING_VIEW_NPOS) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected branch table body");
  }
  iree_string_view_t index =
      iree_string_view_trim(iree_string_view_substr(operands, 0, body_open));
  iree_string_view_t body = iree_string_view_trim(
      iree_string_view_substr(operands, body_open + 1, IREE_HOST_SIZE_MAX));
  if (body.size == 0 || body.data[body.size - 1] != '}') {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unterminated branch table body");
  }
  body = iree_string_view_trim(iree_string_view_substr(body, 0, body.size - 1));

  uint16_t register_ordinal = 0;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_register_field(
      state, index, &instruction->fields[0], &register_ordinal));
  IREE_RETURN_IF_ERROR(
      iree_vm_bytecode_assembler_emit_u16(state, register_ordinal));

  if (!iree_string_view_consume_prefix(&body, IREE_SV("default:"))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected default branch table case");
  }
  iree_string_view_t label_name = iree_string_view_empty();
  iree_string_view_t remaps = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_branch_target(
      &body, &label_name, &remaps));
  uint32_t patch_offset =
      (uint32_t)iree_string_builder_size(&state->bytecode_builder);
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_emit_u32(state, /*value=*/0));
  IREE_RETURN_IF_ERROR(
      iree_vm_bytecode_assembler_append_fixup(state, label_name, patch_offset));
  IREE_RETURN_IF_ERROR(
      iree_vm_bytecode_assembler_emit_branch_remap_list(state, remaps));

  const uint32_t table_size_offset =
      (uint32_t)iree_string_builder_size(&state->bytecode_builder);
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_emit_u16(state, 0));

  uint16_t table_size = 0;
  while (!iree_vm_bytecode_assembler_string_view_is_empty(
      iree_string_view_trim(body))) {
    iree_string_view_t case_text = iree_string_view_empty();
    iree_string_view_split(body, ':', &case_text, &body);
    case_text = iree_string_view_trim(case_text);
    uint32_t case_ordinal = 0;
    if (!iree_string_view_atoi_uint32(case_text, &case_ordinal) ||
        case_ordinal != table_size) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "branch table cases must be dense from 0");
    }
    if (table_size == UINT16_MAX) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "branch table is too large");
    }
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_branch_target(
        &body, &label_name, &remaps));
    patch_offset = (uint32_t)iree_string_builder_size(&state->bytecode_builder);
    IREE_RETURN_IF_ERROR(
        iree_vm_bytecode_assembler_emit_u32(state, /*value=*/0));
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_append_fixup(
        state, label_name, patch_offset));
    IREE_RETURN_IF_ERROR(
        iree_vm_bytecode_assembler_emit_branch_remap_list(state, remaps));
    ++table_size;
  }
  return iree_vm_bytecode_assembler_patch_u16(state, table_size_offset,
                                              table_size);
}

static iree_status_t iree_vm_bytecode_assembler_emit_instruction(
    iree_vm_bytecode_assembler_module_t* state,
    const iree_vm_isa_instruction_t* instruction, bool has_result,
    iree_string_view_t result_register_text, iree_string_view_t operands) {
  IREE_RETURN_IF_ERROR(
      iree_vm_bytecode_assembler_emit_opcode(state, instruction));

  if (iree_vm_bytecode_assembler_is_call_instruction(instruction)) {
    return iree_vm_bytecode_assembler_emit_call_instruction(
        state, instruction, has_result, result_register_text, operands);
  }

  if (iree_vm_bytecode_assembler_instruction_has_encoding(instruction,
                                                          "branch_table")) {
    if (has_result) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "branch table instruction has a result");
    }
    return iree_vm_bytecode_assembler_emit_branch_table_instruction(
        state, instruction, operands);
  }

  if (iree_vm_bytecode_assembler_is_select_instruction(instruction)) {
    if (!has_result) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "select instruction requires a result register");
    }
    return iree_vm_bytecode_assembler_emit_select_instruction(
        state, instruction, result_register_text, operands);
  }

  if (iree_vm_bytecode_assembler_is_switch_instruction(instruction)) {
    if (!has_result) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "switch instruction requires a result register");
    }
    return iree_vm_bytecode_assembler_emit_switch_instruction(
        state, instruction, result_register_text, operands);
  }

  if (iree_vm_bytecode_assembler_is_buffer_store_instruction(instruction)) {
    if (has_result || instruction->field_count != 3 ||
        instruction->fields[0].kind != IREE_VM_ISA_FIELD_KIND_REGISTER ||
        instruction->fields[1].kind != IREE_VM_ISA_FIELD_KIND_REGISTER ||
        instruction->fields[2].kind != IREE_VM_ISA_FIELD_KIND_REGISTER) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "buffer store instruction has unexpected encoding");
    }
    iree_string_view_t value_operand = iree_string_view_empty();
    iree_string_view_t target_operand = iree_string_view_empty();
    iree_string_view_t offset_operand = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_operand_token(
        &operands, &value_operand));
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_operand_token(
        &operands, &target_operand));
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_operand_token(
        &operands, &offset_operand));
    uint16_t register_ordinal = 0;
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_register_field(
        state, target_operand, &instruction->fields[0], &register_ordinal));
    IREE_RETURN_IF_ERROR(
        iree_vm_bytecode_assembler_emit_u16(state, register_ordinal));
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_register_field(
        state, offset_operand, &instruction->fields[1], &register_ordinal));
    IREE_RETURN_IF_ERROR(
        iree_vm_bytecode_assembler_emit_u16(state, register_ordinal));
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_register_field(
        state, value_operand, &instruction->fields[2], &register_ordinal));
    IREE_RETURN_IF_ERROR(
        iree_vm_bytecode_assembler_emit_u16(state, register_ordinal));
    if (!iree_vm_bytecode_assembler_string_view_is_empty(
            iree_string_view_trim(operands))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "too many operands for instruction");
    }
    return iree_ok_status();
  }

  if ((instruction->field_count == 2 || instruction->field_count == 3) &&
      instruction->fields[0].kind == IREE_VM_ISA_FIELD_KIND_GLOBAL_ATTR &&
      instruction->fields[instruction->field_count - 1].kind ==
          IREE_VM_ISA_FIELD_KIND_REGISTER &&
      instruction->fields[instruction->field_count - 1].access ==
          IREE_VM_ISA_FIELD_ACCESS_READ) {
    if (has_result) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "global store instruction has a result");
    }
    iree_string_view_t value_operand = iree_string_view_empty();
    iree_string_view_t global_operand = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_operand_token(
        &operands, &value_operand));
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_operand_token(
        &operands, &global_operand));
    iree_string_view_t storage_operand = global_operand;
    iree_string_view_t type_name = iree_string_view_empty();
    if (instruction->field_count == 3) {
      if (!iree_vm_bytecode_assembler_split_type_suffix(
              global_operand, &storage_operand, &type_name)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "expected global ref type");
      }
    }
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_emit_global_attr(
        state, storage_operand,
        iree_vm_bytecode_assembler_register_bank_storage_size(
            instruction->fields[instruction->field_count - 1].register_bank)));
    if (instruction->field_count == 3) {
      IREE_RETURN_IF_ERROR(
          iree_vm_bytecode_assembler_emit_type_ordinal(state, type_name));
    }
    uint16_t register_ordinal = 0;
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_register_field(
        state, value_operand,
        &instruction->fields[instruction->field_count - 1], &register_ordinal));
    IREE_RETURN_IF_ERROR(
        iree_vm_bytecode_assembler_emit_u16(state, register_ordinal));
    if (!iree_vm_bytecode_assembler_string_view_is_empty(
            iree_string_view_trim(operands))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "too many operands for instruction");
    }
    return iree_ok_status();
  }

  if (iree_vm_bytecode_assembler_instruction_has_encoding(instruction,
                                                          "list_alloc")) {
    if (!has_result || instruction->field_count != 3 ||
        instruction->fields[0].kind != IREE_VM_ISA_FIELD_KIND_TYPE_OF ||
        instruction->fields[1].kind != IREE_VM_ISA_FIELD_KIND_REGISTER ||
        instruction->fields[2].kind != IREE_VM_ISA_FIELD_KIND_REGISTER) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "list.alloc instruction has unexpected encoding");
    }
    iree_string_view_t operand = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(
        iree_vm_bytecode_assembler_parse_operand_token(&operands, &operand));
    iree_string_view_t capacity_operand = iree_string_view_empty();
    iree_string_view_t list_type_name = iree_string_view_empty();
    if (!iree_vm_bytecode_assembler_split_type_suffix(
            operand, &capacity_operand, &list_type_name)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "expected list element type");
    }
    iree_string_view_t element_type_name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_list_element_type(
        list_type_name, &element_type_name));
    IREE_RETURN_IF_ERROR(
        iree_vm_bytecode_assembler_emit_type_ordinal(state, element_type_name));
    uint16_t register_ordinal = 0;
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_register_field(
        state, capacity_operand, &instruction->fields[1], &register_ordinal));
    IREE_RETURN_IF_ERROR(
        iree_vm_bytecode_assembler_emit_u16(state, register_ordinal));
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_register_field(
        state, result_register_text, &instruction->fields[2],
        &register_ordinal));
    IREE_RETURN_IF_ERROR(
        iree_vm_bytecode_assembler_emit_u16(state, register_ordinal));
    if (!iree_vm_bytecode_assembler_string_view_is_empty(
            iree_string_view_trim(operands))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "too many operands for instruction");
    }
    return iree_ok_status();
  }

  if (iree_vm_bytecode_assembler_instruction_has_encoding(instruction,
                                                          "cast_any_ref")) {
    if (!has_result || instruction->field_count != 3 ||
        instruction->fields[0].kind != IREE_VM_ISA_FIELD_KIND_REGISTER ||
        instruction->fields[1].kind != IREE_VM_ISA_FIELD_KIND_TYPE_OF ||
        instruction->fields[2].kind != IREE_VM_ISA_FIELD_KIND_REGISTER) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "cast.any.ref has unexpected encoding");
    }
    const iree_host_size_t arrow_position =
        iree_string_view_find_char(operands, '-', 0);
    if (arrow_position == IREE_STRING_VIEW_NPOS ||
        arrow_position + 1 >= operands.size ||
        operands.data[arrow_position + 1] != '>') {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "expected cast result type after '->'");
    }
    iree_string_view_t operand_and_source_type = iree_string_view_trim(
        iree_string_view_substr(operands, 0, arrow_position));
    iree_string_view_t result_type =
        iree_string_view_trim(iree_string_view_substr(
            operands, arrow_position + 2, IREE_HOST_SIZE_MAX));
    iree_string_view_t operand_register = iree_string_view_empty();
    iree_string_view_t source_type = iree_string_view_empty();
    if (!iree_vm_bytecode_assembler_split_type_suffix(
            operand_and_source_type, &operand_register, &source_type)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "expected cast source type");
    }
    (void)source_type;
    uint16_t register_ordinal = 0;
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_register_field(
        state, operand_register, &instruction->fields[0], &register_ordinal));
    IREE_RETURN_IF_ERROR(
        iree_vm_bytecode_assembler_emit_u16(state, register_ordinal));
    IREE_RETURN_IF_ERROR(
        iree_vm_bytecode_assembler_emit_type_ordinal(state, result_type));
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_register_field(
        state, result_register_text, &instruction->fields[2],
        &register_ordinal));
    return iree_vm_bytecode_assembler_emit_u16(state, register_ordinal);
  }

  iree_string_view_t pending_branch_remaps = iree_string_view_empty();
  iree_string_view_t pending_type_name = iree_string_view_empty();
  for (iree_host_size_t i = 0; i < instruction->field_count; ++i) {
    const iree_vm_isa_field_t* field = &instruction->fields[i];
    switch (field->kind) {
      case IREE_VM_ISA_FIELD_KIND_ATTRIBUTE: {
        iree_string_view_t operand;
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_operand_token(
            &operands, &operand));
        if (field->value_type == IREE_VM_ISA_VALUE_TYPE_I32) {
          int32_t value = 0;
          if (!iree_string_view_atoi_int32(operand, &value)) {
            return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                    "expected i32 immediate");
          }
          IREE_RETURN_IF_ERROR(
              iree_vm_bytecode_assembler_emit_u32(state, (uint32_t)value));
        } else if (field->value_type == IREE_VM_ISA_VALUE_TYPE_I64) {
          int64_t value = 0;
          if (!iree_string_view_atoi_int64(operand, &value)) {
            return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                    "expected i64 immediate");
          }
          IREE_RETURN_IF_ERROR(
              iree_vm_bytecode_assembler_emit_u64(state, (uint64_t)value));
        } else if (field->value_type == IREE_VM_ISA_VALUE_TYPE_F32) {
          float value = 0.0f;
          IREE_RETURN_IF_ERROR(
              iree_vm_bytecode_assembler_parse_f32(operand, &value));
          IREE_RETURN_IF_ERROR(
              iree_vm_bytecode_assembler_emit_f32(state, value));
        } else if (field->value_type == IREE_VM_ISA_VALUE_TYPE_F64) {
          double value = 0.0;
          IREE_RETURN_IF_ERROR(
              iree_vm_bytecode_assembler_parse_f64(operand, &value));
          IREE_RETURN_IF_ERROR(
              iree_vm_bytecode_assembler_emit_f64(state, value));
        } else {
          return iree_make_status(
              IREE_STATUS_UNIMPLEMENTED,
              "attribute value type is not supported by the assembler yet");
        }
      } break;

      case IREE_VM_ISA_FIELD_KIND_GLOBAL_ATTR: {
        iree_string_view_t operand;
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_operand_token(
            &operands, &operand));
        iree_string_view_t storage_operand = operand;
        if (i + 1 < instruction->field_count &&
            instruction->fields[i + 1].kind == IREE_VM_ISA_FIELD_KIND_TYPE_OF) {
          if (!iree_vm_bytecode_assembler_split_type_suffix(
                  operand, &storage_operand, &pending_type_name)) {
            return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                    "expected global ref type");
          }
        }
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_emit_global_attr(
            state, storage_operand,
            iree_vm_bytecode_assembler_global_storage_size(instruction, i)));
      } break;

      case IREE_VM_ISA_FIELD_KIND_FUNC_ATTR: {
        iree_string_view_t operand;
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_operand_token(
            &operands, &operand));
        IREE_RETURN_IF_ERROR(
            iree_vm_bytecode_assembler_emit_function_attr(state, operand));
      } break;

      case IREE_VM_ISA_FIELD_KIND_RODATA_ATTR: {
        iree_string_view_t operand;
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_operand_token(
            &operands, &operand));
        uint32_t ordinal = 0;
        if (!iree_string_view_atoi_uint32(operand, &ordinal)) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "expected rodata ordinal");
        }
        IREE_RETURN_IF_ERROR(
            iree_vm_bytecode_assembler_emit_u32(state, ordinal));
      } break;

      case IREE_VM_ISA_FIELD_KIND_TYPE_OF:
        if (iree_vm_bytecode_assembler_string_view_is_empty(
                pending_type_name)) {
          iree_string_view_t operand = iree_string_view_trim(operands);
          if (!iree_string_view_consume_prefix(&operand, IREE_SV(":"))) {
            return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                    "expected type annotation");
          }
          pending_type_name = iree_string_view_trim(operand);
          operands = iree_string_view_empty();
        }
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_emit_type_ordinal(
            state, pending_type_name));
        pending_type_name = iree_string_view_empty();
        break;

      case IREE_VM_ISA_FIELD_KIND_CONST_I16: {
        iree_string_view_t operand;
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_operand_token(
            &operands, &operand));
        uint32_t value = 0;
        if (!iree_string_view_atoi_uint32(operand, &value) ||
            value > UINT16_MAX) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "expected u16 immediate");
        }
        IREE_RETURN_IF_ERROR(
            iree_vm_bytecode_assembler_emit_u16(state, (uint16_t)value));
      } break;

      case IREE_VM_ISA_FIELD_KIND_REGISTER: {
        uint16_t register_ordinal = 0;
        if (field->access == IREE_VM_ISA_FIELD_ACCESS_WRITE) {
          if (!has_result) {
            return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                    "instruction requires a result register");
          }
          IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_register_field(
              state, result_register_text, field, &register_ordinal));
          IREE_RETURN_IF_ERROR(
              iree_vm_bytecode_assembler_emit_u16(state, register_ordinal));
        } else if (field->access == IREE_VM_ISA_FIELD_ACCESS_READ) {
          iree_string_view_t operand;
          IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_operand_token(
              &operands, &operand));
          if (i + 1 < instruction->field_count &&
              instruction->fields[i + 1].kind ==
                  IREE_VM_ISA_FIELD_KIND_TYPE_OF) {
            iree_string_view_t register_operand = operand;
            if (iree_vm_bytecode_assembler_split_type_suffix(
                    operand, &register_operand, &pending_type_name)) {
              operand = register_operand;
            }
          }
          IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_register_field(
              state, operand, field, &register_ordinal));
          IREE_RETURN_IF_ERROR(
              iree_vm_bytecode_assembler_emit_u16(state, register_ordinal));
        } else {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "register field has no access mode");
        }
      } break;

      case IREE_VM_ISA_FIELD_KIND_VARIADIC_OPERANDS: {
        if (has_result) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "variadic operand instruction has a result");
        }
        IREE_RETURN_IF_ERROR(
            iree_vm_bytecode_assembler_emit_variadic_register_list(
                state, operands,
                iree_vm_bytecode_assembler_value_type_bank(field->value_type)));
        operands = iree_string_view_empty();
      } break;

      case IREE_VM_ISA_FIELD_KIND_VARIADIC_RESULTS: {
        IREE_RETURN_IF_ERROR(
            iree_vm_bytecode_assembler_emit_variadic_register_list(
                state,
                has_result ? result_register_text : iree_string_view_empty(),
                iree_vm_bytecode_assembler_value_type_bank(field->value_type)));
      } break;

      case IREE_VM_ISA_FIELD_KIND_BRANCH_TARGET: {
        iree_string_view_t label_name = iree_string_view_empty();
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_branch_target(
            &operands, &label_name, &pending_branch_remaps));
        const uint32_t patch_offset =
            (uint32_t)iree_string_builder_size(&state->bytecode_builder);
        IREE_RETURN_IF_ERROR(
            iree_vm_bytecode_assembler_emit_u32(state, /*value=*/0));
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_append_fixup(
            state, label_name, patch_offset));
      } break;

      case IREE_VM_ISA_FIELD_KIND_BRANCH_OPERANDS: {
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_emit_branch_remap_list(
            state, pending_branch_remaps));
        pending_branch_remaps = iree_string_view_empty();
        break;
      }

      case IREE_VM_ISA_FIELD_KIND_STRING_ATTR: {
        iree_string_view_t operand;
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_operand_token(
            &operands, &operand));
        iree_string_view_t value = iree_string_view_empty();
        IREE_RETURN_IF_ERROR(
            iree_vm_bytecode_assembler_parse_quoted_string(operand, &value));
        if (value.size > UINT16_MAX) {
          return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                  "string attribute is too large");
        }
        IREE_RETURN_IF_ERROR(
            iree_vm_bytecode_assembler_emit_u16(state, (uint16_t)value.size));
        uint8_t* bytes = NULL;
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_append_bytes(
            &state->bytecode_builder, value.size, &bytes));
        memcpy(bytes, value.data, value.size);
      } break;

      default:
        return iree_make_status(
            IREE_STATUS_UNIMPLEMENTED,
            "field kind is not supported by the assembler yet");
    }
  }

  if (!iree_vm_bytecode_assembler_string_view_is_empty(
          iree_string_view_trim(operands))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "too many operands for instruction");
  }
  return iree_ok_status();
}

iree_status_t iree_vm_bytecode_assembler_parse_instruction(
    iree_vm_bytecode_assembler_module_t* state, iree_string_view_t line) {
  bool has_result = false;
  iree_string_view_t result_register_text = iree_string_view_empty();
  iree_string_view_t instruction_text = line;

  const iree_host_size_t equals_position =
      iree_string_view_find_char(line, '=', 0);
  if (equals_position != IREE_STRING_VIEW_NPOS) {
    iree_string_view_t lhs = iree_string_view_trim(
        iree_string_view_substr(line, 0, equals_position));
    iree_string_view_t rhs = iree_string_view_trim(
        iree_string_view_substr(line, equals_position + 1, IREE_HOST_SIZE_MAX));
    if (iree_string_view_starts_with_char(lhs, '%')) {
      has_result = true;
      result_register_text = iree_string_view_trim(lhs);
      instruction_text = iree_string_view_trim(rhs);
    }
  }

  iree_string_view_t mnemonic;
  iree_string_view_t operands;
  iree_vm_bytecode_assembler_split_first_token(instruction_text, &mnemonic,
                                               &operands);
  iree_string_view_consume_prefix(&mnemonic, IREE_SV("vm."));
  const iree_vm_isa_instruction_t* instruction =
      iree_vm_isa_lookup_mnemonic(mnemonic);
  if (!instruction) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown VM instruction '%.*s'", (int)mnemonic.size,
                            mnemonic.data);
  }

  IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_emit_instruction(
      state, instruction, has_result, result_register_text, operands));
  if (iree_vm_bytecode_assembler_string_view_equal_cstring(mnemonic,
                                                           "return") ||
      iree_vm_bytecode_assembler_string_view_equal_cstring(mnemonic, "fail")) {
    state->has_terminator = true;
  }
  return iree_ok_status();
}
