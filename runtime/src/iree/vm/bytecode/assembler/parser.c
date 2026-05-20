// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/bytecode/assembler/parser.h"

#include <ctype.h>
#include <inttypes.h>

#include "iree/vm/bytecode/assembler/bytecode.h"
#include "iree/vm/bytecode/assembler/source.h"
#include "iree/vm/bytecode/isa/isa.h"

static bool iree_vm_bytecode_assembler_is_block_label(iree_string_view_t line) {
  line = iree_string_view_trim(line);
  if (!iree_string_view_consume_prefix(&line, IREE_SV("^bb"))) {
    return false;
  }
  iree_host_size_t digit_count = 0;
  while (digit_count < line.size &&
         isdigit((unsigned char)line.data[digit_count])) {
    ++digit_count;
  }
  if (digit_count == 0) return false;
  line = iree_string_view_substr(line, digit_count, IREE_HOST_SIZE_MAX);
  if (!iree_string_view_consume_prefix_char(&line, ':')) return false;
  return iree_vm_bytecode_assembler_string_view_is_empty(
      iree_string_view_trim(line));
}

static iree_status_t iree_vm_bytecode_assembler_parse_block_label(
    iree_string_view_t line, iree_string_view_t* out_name) {
  line = iree_string_view_trim(line);
  if (!iree_string_view_consume_prefix_char(&line, '^')) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected block label");
  }
  const iree_host_size_t colon_pos = iree_string_view_find_char(line, ':', 0);
  if (colon_pos == IREE_STRING_VIEW_NPOS) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unterminated block label");
  }
  iree_string_view_t name = iree_string_view_substr(line, 0, colon_pos);
  if (iree_vm_bytecode_assembler_string_view_is_empty(name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "empty block label");
  }
  for (iree_host_size_t i = 0; i < name.size; ++i) {
    if (!iree_vm_bytecode_assembler_is_symbol_char(name.data[i])) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid block label character '%c'",
                              name.data[i]);
    }
  }
  *out_name = name;
  return iree_ok_status();
}

static iree_string_view_t iree_vm_bytecode_assembler_strip_comments(
    iree_string_view_t line) {
  iree_host_size_t comment_pos = IREE_STRING_VIEW_NPOS;
  bool in_string = false;
  for (iree_host_size_t i = 0; i < line.size; ++i) {
    if (line.data[i] == '"') {
      in_string = !in_string;
      continue;
    }
    if (in_string) continue;
    if (line.data[i] == '#') {
      comment_pos = i;
      break;
    }
    if (line.data[i] == '/' && i + 1 < line.size && line.data[i + 1] == '/') {
      comment_pos = i;
      break;
    }
  }
  if (comment_pos == IREE_STRING_VIEW_NPOS) return line;
  return iree_string_view_substr(line, 0, comment_pos);
}

static iree_string_view_t iree_vm_bytecode_assembler_strip_line_prefix(
    iree_string_view_t line) {
  line = iree_string_view_trim(line);
  if (!iree_string_view_starts_with_char(line, '[')) return line;
  const iree_host_size_t close_pos = iree_string_view_find_char(line, ']', 0);
  if (close_pos == IREE_STRING_VIEW_NPOS) return line;
  return iree_string_view_trim(
      iree_string_view_substr(line, close_pos + 1, IREE_HOST_SIZE_MAX));
}

static iree_string_view_t iree_vm_bytecode_assembler_normalize_line(
    iree_string_view_t line) {
  return iree_vm_bytecode_assembler_strip_line_prefix(
      iree_string_view_trim(iree_vm_bytecode_assembler_strip_comments(line)));
}

static iree_status_t iree_vm_bytecode_assembler_parse_quoted_directive_string(
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

static iree_status_t iree_vm_bytecode_assembler_append_cconv_type(
    iree_string_view_t type_part, iree_string_builder_t* cconv_builder) {
  type_part = iree_string_view_trim(type_part);
  if (iree_vm_bytecode_assembler_string_view_equal_cstring(type_part, "i32")) {
    return iree_string_builder_append_cstring(cconv_builder, "i");
  }
  if (iree_vm_bytecode_assembler_string_view_equal_cstring(type_part, "i64")) {
    return iree_string_builder_append_cstring(cconv_builder, "I");
  }
  if (iree_vm_bytecode_assembler_string_view_equal_cstring(type_part, "f32")) {
    return iree_string_builder_append_cstring(cconv_builder, "f");
  }
  if (iree_vm_bytecode_assembler_string_view_equal_cstring(type_part, "f64")) {
    return iree_string_builder_append_cstring(cconv_builder, "F");
  }
  if (iree_string_view_starts_with_char(type_part, '!')) {
    return iree_string_builder_append_cstring(cconv_builder, "r");
  }
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "only primitive and ref function signatures are supported");
}

static iree_status_t iree_vm_bytecode_assembler_register_bank_from_type(
    iree_string_view_t type_part, iree_vm_isa_register_bank_t* out_bank) {
  if (iree_vm_bytecode_assembler_string_view_equal_cstring(type_part, "i32")) {
    *out_bank = IREE_VM_ISA_REGISTER_BANK_I32;
    return iree_ok_status();
  }
  if (iree_vm_bytecode_assembler_string_view_equal_cstring(type_part, "i64")) {
    *out_bank = IREE_VM_ISA_REGISTER_BANK_I64;
    return iree_ok_status();
  }
  if (iree_vm_bytecode_assembler_string_view_equal_cstring(type_part, "f32")) {
    *out_bank = IREE_VM_ISA_REGISTER_BANK_F32;
    return iree_ok_status();
  }
  if (iree_vm_bytecode_assembler_string_view_equal_cstring(type_part, "f64")) {
    *out_bank = IREE_VM_ISA_REGISTER_BANK_F64;
    return iree_ok_status();
  }
  if (iree_string_view_starts_with_char(type_part, '!')) {
    *out_bank = IREE_VM_ISA_REGISTER_BANK_REF;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "only primitive and ref registers are supported");
}

static iree_status_t iree_vm_bytecode_assembler_parse_typed_register_list(
    iree_vm_bytecode_assembler_module_t* state, iree_string_view_t value,
    bool require_sequential, iree_string_builder_t* cconv_builder,
    iree_host_size_t* out_count) {
  value = iree_string_view_trim(value);
  *out_count = 0;
  if (iree_vm_bytecode_assembler_string_view_is_empty(value)) {
    return iree_ok_status();
  }

  uint32_t expected_i32_register_ordinal = 0;
  uint32_t expected_ref_register_ordinal = 0;
  while (!iree_vm_bytecode_assembler_string_view_is_empty(value)) {
    iree_string_view_t item;
    iree_string_view_t remainder;
    iree_string_view_split(value, ',', &item, &remainder);
    item = iree_string_view_trim(item);

    iree_string_view_t register_part;
    iree_string_view_t type_part;
    const iree_host_size_t type_delimiter =
        iree_vm_bytecode_assembler_find_last_char(item, ':');
    if (type_delimiter == IREE_STRING_VIEW_NPOS) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "expected typed register declaration");
    }
    register_part =
        iree_string_view_trim(iree_string_view_substr(item, 0, type_delimiter));
    type_part = iree_string_view_trim(
        iree_string_view_substr(item, type_delimiter + 1, IREE_HOST_SIZE_MAX));

    uint16_t register_ordinal = 0;
    iree_vm_isa_register_bank_t register_bank = IREE_VM_ISA_REGISTER_BANK_NONE;
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_register_bank_from_type(
        type_part, &register_bank));
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_register(
        state, register_part, register_bank, &register_ordinal));
    if (require_sequential) {
      if (register_bank == IREE_VM_ISA_REGISTER_BANK_REF) {
        if (register_ordinal != (IREE_VM_ISA_REF_REGISTER_TYPE_BIT |
                                 expected_ref_register_ordinal)) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "function ref argument registers must be sequential from %%r0");
        }
      } else if (register_ordinal != expected_i32_register_ordinal) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "function value argument registers must be sequential from %%i0");
      }
    }

    IREE_RETURN_IF_ERROR(
        iree_vm_bytecode_assembler_append_cconv_type(type_part, cconv_builder));

    ++*out_count;
    if (register_bank == IREE_VM_ISA_REGISTER_BANK_REF) {
      ++expected_ref_register_ordinal;
    } else if (register_bank == IREE_VM_ISA_REGISTER_BANK_I64 ||
               register_bank == IREE_VM_ISA_REGISTER_BANK_F64) {
      expected_i32_register_ordinal += 2;
    } else {
      ++expected_i32_register_ordinal;
    }
    value = iree_string_view_trim(remainder);
  }

  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_assembler_parse_result_type_list(
    iree_string_view_t value, iree_string_builder_t* cconv_builder,
    iree_host_size_t* out_count) {
  value = iree_string_view_trim(value);
  *out_count = 0;
  if (iree_vm_bytecode_assembler_string_view_is_empty(value)) {
    return iree_ok_status();
  }

  while (!iree_vm_bytecode_assembler_string_view_is_empty(value)) {
    iree_string_view_t item;
    iree_string_view_t remainder;
    iree_string_view_split(value, ',', &item, &remainder);
    item = iree_string_view_trim(item);

    const iree_host_size_t type_delimiter =
        iree_vm_bytecode_assembler_find_last_char(item, ':');
    iree_string_view_t type_part =
        type_delimiter == IREE_STRING_VIEW_NPOS
            ? item
            : iree_string_view_substr(item, type_delimiter + 1,
                                      IREE_HOST_SIZE_MAX);
    IREE_RETURN_IF_ERROR(
        iree_vm_bytecode_assembler_append_cconv_type(type_part, cconv_builder));
    ++*out_count;
    value = iree_string_view_trim(remainder);
  }

  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_assembler_build_cconv(
    iree_vm_bytecode_assembler_module_t* state) {
  iree_string_builder_reset(&state->cconv_builder);
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(&state->cconv_builder, "0"));
  if (iree_string_builder_size(&state->argument_cconv_builder) == 0) {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(&state->cconv_builder, "v"));
  } else {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
        &state->cconv_builder,
        iree_string_builder_view(&state->argument_cconv_builder)));
  }
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(&state->cconv_builder, "_"));
  if (iree_string_builder_size(&state->result_cconv_builder) == 0) {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(&state->cconv_builder, "v"));
  } else {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
        &state->cconv_builder,
        iree_string_builder_view(&state->result_cconv_builder)));
  }
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_assembler_parse_module_directive(
    iree_vm_bytecode_assembler_module_t* state, iree_string_view_t line) {
  if (state->has_module) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "module already declared");
  }
  iree_string_view_consume_prefix(&line, IREE_SV("vm.module"));
  iree_string_view_t module_symbol;
  iree_string_view_t remainder;
  iree_vm_bytecode_assembler_split_first_token(line, &module_symbol,
                                               &remainder);
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_symbol(
      module_symbol, &state->module_name));
  state->module_version = 0;
  if (!iree_vm_bytecode_assembler_string_view_is_empty(remainder)) {
    if (!iree_string_view_consume_prefix(&remainder, IREE_SV("version"))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "expected optional 'version <uint32>'");
    }
    remainder = iree_string_view_trim(remainder);
    if (!iree_string_view_atoi_uint32(remainder, &state->module_version)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid module version");
    }
  }
  state->has_module = true;
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_assembler_parse_type_directive(
    iree_vm_bytecode_assembler_module_t* state, iree_string_view_t line) {
  if (!state->has_module) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "module directive must appear before types");
  }
  if (state->function_count > 0 || state->in_function) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "type directives must appear before functions");
  }
  iree_string_view_consume_prefix(&line, IREE_SV("vm.type"));
  line = iree_string_view_trim(line);
  if (iree_vm_bytecode_assembler_string_view_is_empty(line)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "expected type name");
  }
  iree_host_size_t ordinal = 0;
  return iree_vm_bytecode_assembler_append_type(state, line, &ordinal);
}

static uint8_t iree_vm_bytecode_assembler_hex_value(char c) {
  if (c >= '0' && c <= '9') return (uint8_t)(c - '0');
  if (c >= 'a' && c <= 'f') return (uint8_t)(10 + c - 'a');
  if (c >= 'A' && c <= 'F') return (uint8_t)(10 + c - 'A');
  return 0xFFu;
}

static iree_status_t iree_vm_bytecode_assembler_parse_rodata_directive(
    iree_vm_bytecode_assembler_module_t* state, iree_string_view_t line) {
  if (!state->has_module) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "module directive must appear before rodata");
  }
  if (state->function_count > 0 || state->in_function) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "rodata directives must appear before functions");
  }

  iree_string_view_consume_prefix(&line, IREE_SV("vm.rodata["));
  iree_string_view_t ordinal_text = iree_string_view_empty();
  iree_string_view_t hex_text = iree_string_view_empty();
  if (iree_string_view_split(line, ']', &ordinal_text, &line) == -1) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unterminated rodata ordinal");
  }
  line = iree_string_view_trim(line);
  if (!iree_string_view_consume_prefix_char(&line, '=')) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected '= <hex bytes>'");
  }
  hex_text = iree_string_view_trim(line);

  uint32_t ordinal = 0;
  if (!iree_string_view_atoi_uint32(ordinal_text, &ordinal)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid rodata ordinal");
  }
  if ((hex_text.size & 1) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "rodata hex data must have an even length");
  }
  const iree_host_size_t data_length = hex_text.size / 2;
  uint8_t* data = NULL;
  if (data_length > 0) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(state->host_allocator,
                                               data_length, (void**)&data));
    for (iree_host_size_t i = 0; i < data_length; ++i) {
      const uint8_t high_nibble =
          iree_vm_bytecode_assembler_hex_value(hex_text.data[i * 2]);
      const uint8_t low_nibble =
          iree_vm_bytecode_assembler_hex_value(hex_text.data[i * 2 + 1]);
      if (high_nibble == 0xFFu || low_nibble == 0xFFu) {
        iree_allocator_free(state->host_allocator, data);
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "invalid rodata hex digit");
      }
      data[i] = (uint8_t)((high_nibble << 4) | low_nibble);
    }
  }

  iree_status_t status = iree_vm_bytecode_assembler_append_rodata_segment(
      state, ordinal, iree_make_const_byte_span(data, data_length));
  iree_allocator_free(state->host_allocator, data);
  return status;
}

static iree_status_t iree_vm_bytecode_assembler_parse_import_directive(
    iree_vm_bytecode_assembler_module_t* state, iree_string_view_t line) {
  if (!state->has_module) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "module directive must appear before imports");
  }
  if (state->function_count > 0 || state->in_function) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "import directives must appear before functions");
  }

  iree_string_view_consume_prefix(&line, IREE_SV("vm.import"));
  iree_string_view_t token = iree_string_view_empty();
  iree_string_view_t remainder = iree_string_view_empty();
  iree_vm_bytecode_assembler_split_first_token(line, &token, &remainder);
  bool optional = false;
  if (iree_string_view_equal(token, IREE_SV("optional"))) {
    optional = true;
    line = remainder;
    iree_vm_bytecode_assembler_split_first_token(line, &token, &remainder);
  }

  iree_string_view_t full_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(
      iree_vm_bytecode_assembler_parse_symbol(token, &full_name));
  iree_vm_bytecode_assembler_split_first_token(remainder, &token, &remainder);
  if (!iree_string_view_equal(token, IREE_SV("cconv"))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected import cconv");
  }
  iree_string_view_t calling_convention = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_quoted_directive_string(
      remainder, &calling_convention));
  return iree_vm_bytecode_assembler_append_import(state, full_name,
                                                  calling_convention, optional);
}

static iree_status_t iree_vm_bytecode_assembler_parse_global_directive(
    iree_vm_bytecode_assembler_module_t* state, iree_string_view_t line,
    uint32_t storage_size, iree_string_view_t type_name) {
  if (!state->has_module) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "module directive must appear before globals");
  }
  if (state->function_count > 0 || state->in_function) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "global directives must appear before functions");
  }

  iree_string_view_t directive;
  iree_vm_bytecode_assembler_split_first_token(line, &directive, &line);
  (void)directive;

  iree_string_view_t token = iree_string_view_empty();
  iree_string_view_t remainder = iree_string_view_empty();
  iree_vm_bytecode_assembler_split_first_token(line, &token, &remainder);
  if (iree_string_view_equal(token, IREE_SV("private"))) {
    line = remainder;
    iree_vm_bytecode_assembler_split_first_token(line, &token, &remainder);
  }
  if (iree_string_view_equal(token, IREE_SV("mutable"))) {
    line = remainder;
    iree_vm_bytecode_assembler_split_first_token(line, &token, &remainder);
  }

  iree_string_view_t symbol = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_symbol(token, &symbol));

  remainder = iree_string_view_trim(remainder);
  if (!iree_vm_bytecode_assembler_string_view_is_empty(remainder)) {
    if (!iree_string_view_consume_prefix_char(&remainder, '=')) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "expected optional '= 0 : <type>' initializer");
    }
    iree_string_view_t initializer = iree_string_view_empty();
    iree_string_view_split(remainder, ':', &initializer, &remainder);
    initializer = iree_string_view_trim(initializer);
    remainder = iree_string_view_trim(remainder);
    int64_t initial_value = 0;
    if (!iree_string_view_atoi_int64(initializer, &initial_value)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid global initializer");
    }
    if (initial_value != 0) {
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "non-zero global initializers must be encoded as __init bytecode");
    }
    if (!iree_string_view_equal(remainder, type_name)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "global initializer type mismatch");
    }
  }

  return iree_vm_bytecode_assembler_append_global(state, symbol, storage_size);
}

static iree_status_t iree_vm_bytecode_assembler_parse_export_directive(
    iree_vm_bytecode_assembler_module_t* state, iree_string_view_t line) {
  iree_string_view_consume_prefix(&line, IREE_SV("vm.export"));
  iree_string_view_t lhs;
  iree_string_view_t rhs;
  iree_string_view_t export_name;
  iree_string_view_t function_name;
  if (iree_string_view_split(line, '=', &lhs, &rhs) == -1) {
    IREE_RETURN_IF_ERROR(
        iree_vm_bytecode_assembler_parse_symbol(line, &export_name));
    function_name = export_name;
  } else {
    IREE_RETURN_IF_ERROR(
        iree_vm_bytecode_assembler_parse_symbol(lhs, &export_name));
    IREE_RETURN_IF_ERROR(
        iree_vm_bytecode_assembler_parse_symbol(rhs, &function_name));
  }
  return iree_vm_bytecode_assembler_append_export(state, export_name,
                                                  function_name);
}

static iree_status_t iree_vm_bytecode_assembler_parse_function_header(
    iree_vm_bytecode_assembler_module_t* state, iree_string_view_t line) {
  if (!state->has_module) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "module directive must appear before functions");
  }

  iree_string_view_consume_prefix(&line, IREE_SV("vm.func"));
  line = iree_string_view_trim(line);
  const iree_host_size_t args_open = iree_string_view_find_char(line, '(', 0);
  if (args_open == IREE_STRING_VIEW_NPOS) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected function argument list");
  }
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_symbol(
      iree_string_view_substr(line, 0, args_open), &state->function_name));
  if (iree_vm_bytecode_assembler_find_function_ordinal(
          state, state->function_name) != IREE_HOST_SIZE_MAX) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT, "duplicate function @%.*s",
        (int)state->function_name.size, state->function_name.data);
  }

  const iree_host_size_t args_close =
      iree_string_view_find_char(line, ')', args_open + 1);
  if (args_close == IREE_STRING_VIEW_NPOS) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unterminated function argument list");
  }
  iree_string_view_t arguments =
      iree_string_view_substr(line, args_open + 1, args_close - args_open - 1);

  iree_string_view_t remainder = iree_string_view_trim(
      iree_string_view_substr(line, args_close + 1, IREE_HOST_SIZE_MAX));
  if (!iree_string_view_consume_prefix(&remainder, IREE_SV("->"))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected '->' result list");
  }
  remainder = iree_string_view_trim(remainder);
  if (!iree_string_view_consume_prefix_char(&remainder, '(')) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected result list");
  }
  const iree_host_size_t results_close =
      iree_string_view_find_char(remainder, ')', 0);
  if (results_close == IREE_STRING_VIEW_NPOS) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unterminated function result list");
  }
  iree_string_view_t results =
      iree_string_view_substr(remainder, 0, results_close);
  remainder = iree_string_view_trim(iree_string_view_substr(
      remainder, results_close + 1, IREE_HOST_SIZE_MAX));
  if (!iree_string_view_consume_prefix_char(&remainder, '{') ||
      !iree_vm_bytecode_assembler_string_view_is_empty(
          iree_string_view_trim(remainder))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected '{' after function signature");
  }

  iree_string_builder_reset(&state->argument_cconv_builder);
  iree_string_builder_reset(&state->result_cconv_builder);
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_typed_register_list(
      state, arguments, /*require_sequential=*/true,
      &state->argument_cconv_builder, &state->argument_count));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_result_type_list(
      results, &state->result_cconv_builder, &state->result_count));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_build_cconv(state));

  state->in_function = true;
  state->has_terminator = false;
  state->function_requirements = 0;
  state->block_count = 1;
  state->i32_register_count = 0;
  state->ref_register_count = 0;
  state->label_count = 0;
  state->fixup_count = 0;
  state->function_bytecode_offset =
      (uint32_t)iree_string_builder_size(&state->bytecode_builder);
  IREE_RETURN_IF_ERROR(
      iree_vm_bytecode_assembler_emit_u8(state, IREE_VM_OP_CORE_Block));
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_assembler_define_block_label(
    iree_vm_bytecode_assembler_module_t* state, iree_string_view_t line) {
  iree_string_view_t name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(
      iree_vm_bytecode_assembler_parse_block_label(line, &name));

  uint32_t pc = (uint32_t)(iree_string_builder_size(&state->bytecode_builder) -
                           state->function_bytecode_offset);
  if (state->label_count == 0 && pc == 1) {
    pc = 0;
  } else {
    IREE_RETURN_IF_ERROR(
        iree_vm_bytecode_assembler_emit_u8(state, IREE_VM_OP_CORE_Block));
    ++state->block_count;
  }
  return iree_vm_bytecode_assembler_append_label(state, name, pc);
}

static iree_status_t iree_vm_bytecode_assembler_resolve_fixups(
    iree_vm_bytecode_assembler_module_t* state) {
  for (iree_host_size_t i = 0; i < state->fixup_count; ++i) {
    const iree_vm_bytecode_assembler_fixup_t* fixup = &state->fixups[i];
    const iree_vm_bytecode_assembler_label_t* label = NULL;
    for (iree_host_size_t j = 0; j < state->label_count; ++j) {
      if (iree_string_view_equal(state->labels[j].name, fixup->label_name)) {
        label = &state->labels[j];
        break;
      }
    }
    if (!label) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT, "unknown block label ^%.*s",
          (int)fixup->label_name.size, fixup->label_name.data);
    }
    if (fixup->bytecode_offset + 4 >
        iree_string_builder_size(&state->bytecode_builder)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "branch fixup is out of range");
    }
    uint8_t* target =
        (uint8_t*)state->bytecode_builder.buffer + fixup->bytecode_offset;
    target[0] = (uint8_t)(label->pc & 0xFFu);
    target[1] = (uint8_t)((label->pc >> 8) & 0xFFu);
    target[2] = (uint8_t)((label->pc >> 16) & 0xFFu);
    target[3] = (uint8_t)((label->pc >> 24) & 0xFFu);
  }
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_assembler_finish_function(
    iree_vm_bytecode_assembler_module_t* state) {
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_resolve_fixups(state));

  const iree_host_size_t bytecode_size =
      iree_string_builder_size(&state->bytecode_builder);
  if (bytecode_size > UINT32_MAX ||
      state->function_bytecode_offset > bytecode_size) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "function bytecode is too large");
  }
  const iree_host_size_t bytecode_length =
      bytecode_size - state->function_bytecode_offset;
  if (bytecode_length > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "function bytecode is too large");
  }
  if (state->i32_register_count > INT16_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "function register count is too large");
  }
  if (state->ref_register_count > INT16_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "function ref register count is too large");
  }
  if (state->block_count > INT16_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "function block count is too large");
  }

  iree_string_view_t cconv = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_clone_string(
      state, iree_string_builder_view(&state->cconv_builder), &cconv));

  IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_reserve_functions(
      state, state->function_count + 1));
  state->functions[state->function_count++] =
      (iree_vm_bytecode_assembler_function_t){
          .name = state->function_name,
          .cconv = cconv,
          .bytecode_offset = state->function_bytecode_offset,
          .bytecode_length = (uint32_t)bytecode_length,
          .requirements = state->function_requirements,
          .block_count = (uint16_t)state->block_count,
          .i32_register_count = state->i32_register_count,
          .ref_register_count = state->ref_register_count,
      };
  IREE_RETURN_IF_ERROR(
      iree_vm_bytecode_assembler_align_bytecode(state, /*alignment=*/8));
  state->label_count = 0;
  state->fixup_count = 0;
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_assembler_parse_line(
    iree_vm_bytecode_assembler_module_t* state, iree_string_view_t line) {
  line = iree_vm_bytecode_assembler_normalize_line(line);
  if (iree_vm_bytecode_assembler_string_view_is_empty(line)) {
    return iree_ok_status();
  }

  if (state->in_function) {
    if (iree_vm_bytecode_assembler_string_view_equal_cstring(line, "}")) {
      if (!state->has_terminator) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "function missing terminator");
      }
      state->in_function = false;
      return iree_vm_bytecode_assembler_finish_function(state);
    }
    if (iree_vm_bytecode_assembler_is_block_label(line)) {
      return iree_vm_bytecode_assembler_define_block_label(state, line);
    }
    return iree_vm_bytecode_assembler_parse_instruction(state, line);
  }

  iree_string_view_t directive;
  iree_string_view_t directive_remainder;
  iree_vm_bytecode_assembler_split_first_token(line, &directive,
                                               &directive_remainder);
  (void)directive_remainder;
  if (iree_string_view_equal(directive, IREE_SV("vm.module"))) {
    return iree_vm_bytecode_assembler_parse_module_directive(state, line);
  }
  if (iree_string_view_equal(directive, IREE_SV("vm.type"))) {
    return iree_vm_bytecode_assembler_parse_type_directive(state, line);
  }
  if (iree_string_view_equal(directive, IREE_SV("vm.import"))) {
    return iree_vm_bytecode_assembler_parse_import_directive(state, line);
  }
  if (iree_string_view_starts_with(directive, IREE_SV("vm.rodata["))) {
    return iree_vm_bytecode_assembler_parse_rodata_directive(state, line);
  }
  if (iree_string_view_equal(directive, IREE_SV("vm.export"))) {
    return iree_vm_bytecode_assembler_parse_export_directive(state, line);
  }
  if (iree_string_view_equal(directive, IREE_SV("vm.global.i32"))) {
    return iree_vm_bytecode_assembler_parse_global_directive(
        state, line, /*storage_size=*/4, IREE_SV("i32"));
  }
  if (iree_string_view_equal(directive, IREE_SV("vm.global.i64"))) {
    return iree_vm_bytecode_assembler_parse_global_directive(
        state, line, /*storage_size=*/8, IREE_SV("i64"));
  }
  if (iree_string_view_equal(directive, IREE_SV("vm.global.f32"))) {
    return iree_vm_bytecode_assembler_parse_global_directive(
        state, line, /*storage_size=*/4, IREE_SV("f32"));
  }
  if (iree_string_view_equal(directive, IREE_SV("vm.global.f64"))) {
    return iree_vm_bytecode_assembler_parse_global_directive(
        state, line, /*storage_size=*/8, IREE_SV("f64"));
  }
  if (iree_string_view_equal(directive, IREE_SV("vm.func"))) {
    return iree_vm_bytecode_assembler_parse_function_header(state, line);
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unexpected top-level directive");
}

iree_status_t iree_vm_bytecode_assembler_parse_source(
    iree_vm_bytecode_assembler_module_t* state, iree_string_view_t source) {
  uint32_t line_number = 1;
  while (!iree_vm_bytecode_assembler_string_view_is_empty(source)) {
    iree_string_view_t line;
    iree_string_view_t remainder;
    iree_string_view_split(source, '\n', &line, &remainder);
    iree_status_t status = iree_vm_bytecode_assembler_parse_line(state, line);
    if (!iree_status_is_ok(status)) {
      return iree_status_annotate_f(status, "at line %" PRIu32, line_number);
    }
    source = remainder;
    ++line_number;
  }

  if (state->in_function) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unterminated function body");
  }
  if (!state->has_module) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "module directive missing");
  }
  if (state->function_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "function directive missing");
  }
  for (iree_host_size_t i = 0; i < state->export_count; ++i) {
    if (iree_vm_bytecode_assembler_find_function_ordinal(
            state, state->exports[i].function_name) == IREE_HOST_SIZE_MAX) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "export references unknown function @%.*s",
                              (int)state->exports[i].function_name.size,
                              state->exports[i].function_name.data);
    }
  }
  return iree_ok_status();
}
