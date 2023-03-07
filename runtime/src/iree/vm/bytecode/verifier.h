// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_BYTECODE_VERIFIER_H_
#define IREE_VM_BYTECODE_VERIFIER_H_

#include "iree/base/api.h"
#include "iree/vm/api.h"
#include "iree/vm/bytecode/module_impl.h"

// Verifies the structure of the FlatBuffer so that we can avoid doing so during
// runtime. There are still some conditions we must be aware of (such as omitted
// names on functions with internal linkage), however we shouldn't need to
// bounds check anything within the FlatBuffer after this succeeds.
iree_status_t iree_vm_bytecode_module_flatbuffer_verify(
    iree_const_byte_span_t archive_contents,
    iree_const_byte_span_t flatbuffer_contents,
    iree_host_size_t archive_rodata_offset);

// Verifies the bytecode contained within the given |function_ordinal|.
// Assumes that all information on |module| has been verified and only function
// information requires verification.
//
// NOTE: verification only checks that the function is well-formed and not that
// it is correct or will execute successfully. The only thing this tries to
// guarantee is that executing the bytecode won't cause a crash.
//
// If verification requires transient allocations for tracking they will be made
// from |scratch_allocator|. No allocation will live outside of the function and
// callers may provide stack-based arenas.
iree_status_t iree_vm_bytecode_function_verify(
    iree_vm_bytecode_module_t* module, uint16_t function_ordinal,
    iree_allocator_t scratch_allocator);

#endif  // IREE_VM_BYTECODE_VERIFIER_H_
