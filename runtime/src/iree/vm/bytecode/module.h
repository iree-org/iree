// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_BYTECODE_MODULE_H_
#define IREE_VM_BYTECODE_MODULE_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/vm/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates a VM module from an in-memory ModuleDef FlatBuffer archive.
// If a |archive_allocator| is provided then it will be used to free the
// |archive_contents| when the module is destroyed and otherwise the ownership
// of the memory remains with the caller.
IREE_API_EXPORT iree_status_t iree_vm_bytecode_module_create(
    iree_vm_instance_t* instance, iree_const_byte_span_t archive_contents,
    iree_allocator_t archive_allocator, iree_allocator_t allocator,
    iree_vm_module_t** out_module);

// Parses the module archive header in |archive_contents|.
// The subrange containing the FlatBuffer data is returned as well as the
// offset where external rodata begins. Note that archives may have
// non-contiguous layouts!
IREE_API_EXPORT iree_status_t iree_vm_bytecode_module_parse_header(
    iree_const_byte_span_t archive_contents,
    iree_const_byte_span_t* out_flatbuffer_contents,
    iree_host_size_t* out_rodata_offset);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_VM_BYTECODE_MODULE_H_
