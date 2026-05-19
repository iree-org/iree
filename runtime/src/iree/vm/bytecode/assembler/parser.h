// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_BYTECODE_ASSEMBLER_PARSER_H_
#define IREE_VM_BYTECODE_ASSEMBLER_PARSER_H_

#include "iree/vm/bytecode/assembler/module.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Parses a complete textual VM assembly module into |module|.
iree_status_t iree_vm_bytecode_assembler_parse_source(
    iree_vm_bytecode_assembler_module_t* module, iree_string_view_t source);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_VM_BYTECODE_ASSEMBLER_PARSER_H_
