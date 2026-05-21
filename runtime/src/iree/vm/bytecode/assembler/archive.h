// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_BYTECODE_ASSEMBLER_ARCHIVE_H_
#define IREE_VM_BYTECODE_ASSEMBLER_ARCHIVE_H_

#include "iree/vm/bytecode/assembler/module.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Serializes a fully parsed and resolved assembly module into a VMFB archive.
iree_status_t iree_vm_bytecode_assembler_build_archive(
    iree_vm_bytecode_assembler_module_t* module, iree_byte_span_t* out_archive);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_VM_BYTECODE_ASSEMBLER_ARCHIVE_H_
