// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_BYTECODE_ASSEMBLER_SOURCE_H_
#define IREE_VM_BYTECODE_ASSEMBLER_SOURCE_H_

#include <stdbool.h>

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Returns true when |value| has no bytes or no backing storage.
bool iree_vm_bytecode_assembler_string_view_is_empty(iree_string_view_t value);

// Compares a string view against a nul-terminated static literal.
bool iree_vm_bytecode_assembler_string_view_equal_cstring(
    iree_string_view_t value, const char* literal);

// Returns true when |c| is valid inside an assembly symbol or block label.
bool iree_vm_bytecode_assembler_is_symbol_char(char c);

// Finds the last occurrence of |c| in |value|, or IREE_STRING_VIEW_NPOS.
iree_host_size_t iree_vm_bytecode_assembler_find_last_char(
    iree_string_view_t value, char c);

// Splits the first whitespace-delimited token from |value|.
void iree_vm_bytecode_assembler_split_first_token(
    iree_string_view_t value, iree_string_view_t* out_token,
    iree_string_view_t* out_remainder);

// Splits the first comma-delimited operand from |value|.
//
// Commas inside double-quoted strings are part of the operand.
void iree_vm_bytecode_assembler_split_operand(
    iree_string_view_t value, iree_string_view_t* out_operand,
    iree_string_view_t* out_remainder);

// Parses an assembly symbol beginning with '@' and returns the name without it.
iree_status_t iree_vm_bytecode_assembler_parse_symbol(
    iree_string_view_t value, iree_string_view_t* out_symbol);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_VM_BYTECODE_ASSEMBLER_SOURCE_H_
