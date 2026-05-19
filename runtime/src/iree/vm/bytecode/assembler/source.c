// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/bytecode/assembler/source.h"

#include <ctype.h>

bool iree_vm_bytecode_assembler_string_view_is_empty(iree_string_view_t value) {
  return value.data == NULL || value.size == 0;
}

bool iree_vm_bytecode_assembler_string_view_equal_cstring(
    iree_string_view_t value, const char* literal) {
  return iree_string_view_equal(value, iree_make_cstring_view(literal));
}

bool iree_vm_bytecode_assembler_is_symbol_char(char c) {
  return isalnum((unsigned char)c) || c == '_' || c == '$' || c == '.';
}

iree_host_size_t iree_vm_bytecode_assembler_find_last_char(
    iree_string_view_t value, char c) {
  for (iree_host_size_t i = value.size; i > 0; --i) {
    if (value.data[i - 1] == c) return i - 1;
  }
  return IREE_STRING_VIEW_NPOS;
}

void iree_vm_bytecode_assembler_split_first_token(
    iree_string_view_t value, iree_string_view_t* out_token,
    iree_string_view_t* out_remainder) {
  value = iree_string_view_trim(value);
  iree_host_size_t token_end =
      iree_string_view_find_first_of(value, IREE_SV(" \t\r\n"), 0);
  if (token_end == IREE_STRING_VIEW_NPOS) {
    *out_token = value;
    *out_remainder = iree_string_view_empty();
    return;
  }
  *out_token = iree_string_view_substr(value, 0, token_end);
  *out_remainder = iree_string_view_trim(
      iree_string_view_substr(value, token_end + 1, IREE_HOST_SIZE_MAX));
}

void iree_vm_bytecode_assembler_split_operand(
    iree_string_view_t value, iree_string_view_t* out_operand,
    iree_string_view_t* out_remainder) {
  value = iree_string_view_trim(value);
  bool in_quote = false;
  for (iree_host_size_t i = 0; i < value.size; ++i) {
    if (value.data[i] == '"') {
      in_quote = !in_quote;
      continue;
    }
    if (value.data[i] == ',' && !in_quote) {
      *out_operand =
          iree_string_view_trim(iree_string_view_substr(value, 0, i));
      *out_remainder = iree_string_view_trim(
          iree_string_view_substr(value, i + 1, IREE_HOST_SIZE_MAX));
      return;
    }
  }
  *out_operand = value;
  *out_remainder = iree_string_view_empty();
}

iree_status_t iree_vm_bytecode_assembler_parse_symbol(
    iree_string_view_t value, iree_string_view_t* out_symbol) {
  value = iree_string_view_trim(value);
  if (!iree_string_view_consume_prefix_char(&value, '@') ||
      iree_vm_bytecode_assembler_string_view_is_empty(value)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected symbol name beginning with '@'");
  }
  for (iree_host_size_t i = 0; i < value.size; ++i) {
    if (!iree_vm_bytecode_assembler_is_symbol_char(value.data[i])) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid symbol character '%c'", value.data[i]);
    }
  }
  *out_symbol = value;
  return iree_ok_status();
}
