// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_BYTECODE_ASSEMBLER_ASSEMBLER_H_
#define IREE_VM_BYTECODE_ASSEMBLER_ASSEMBLER_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Assembles a textual VM module into a size-prefixed VM bytecode archive.
//
// The returned |out_archive| owns memory allocated from |host_allocator|. The
// caller must release it with iree_allocator_free(host_allocator,
// out_archive->data).
IREE_API_EXPORT iree_status_t iree_vm_bytecode_assembler_assemble(
    iree_string_view_t source, iree_allocator_t host_allocator,
    iree_byte_span_t* out_archive);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_VM_BYTECODE_ASSEMBLER_ASSEMBLER_H_
