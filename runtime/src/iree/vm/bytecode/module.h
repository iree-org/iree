// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_BYTECODE_MODULE_H_
#define IREE_VM_BYTECODE_MODULE_H_

#include "iree/base/api.h"
#include "iree/vm/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Flags controlling bytecode module creation behavior.
typedef enum iree_vm_bytecode_module_flags_e {
  IREE_VM_BYTECODE_MODULE_FLAG_NONE = 0,
  // Allows unregistered ref types to be resolved as placeholder types.
  // The module can be used for reflection/disassembly but NOT execution.
  // Types will be registered with the instance as placeholders.
  IREE_VM_BYTECODE_MODULE_FLAG_ALLOW_PLACEHOLDER_TYPES = 1u << 0,
} iree_vm_bytecode_module_flags_t;

// Creates a VM module from an in-memory ModuleDef FlatBuffer archive.
// If a |archive_allocator| is provided then it will be used to free the
// |archive_contents| when the module is destroyed and otherwise the ownership
// of the memory remains with the caller.
IREE_API_EXPORT iree_status_t iree_vm_bytecode_module_create(
    iree_vm_instance_t* instance, iree_vm_bytecode_module_flags_t flags,
    iree_const_byte_span_t archive_contents, iree_allocator_t archive_allocator,
    iree_allocator_t allocator, iree_vm_module_t** out_module);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_VM_BYTECODE_MODULE_H_
