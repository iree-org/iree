// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_DYNAMIC_MODULE_H_
#define IREE_VM_DYNAMIC_MODULE_H_

#include "iree/base/api.h"
#include "iree/vm/api.h"
#include "iree/vm/dynamic/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates a VM module from a dynamically loaded shared library at |path|.
// The exported symbol with |export_name| matching the signature of
// iree_vm_dynamic_module_create_fn_t will be used to create the module.
//
// Optionally key-value parameters may be provided to the module on
// initialization. They strings referenced need only exist during creation and
// modules will clone any they need to retain.
//
// |allocator| will be passed to the module to serve its allocations and must
// remain valid for the module lifetime.
IREE_API_EXPORT iree_status_t iree_vm_dynamic_module_load_from_file(
    iree_vm_instance_t* instance, iree_string_view_t path,
    iree_string_view_t export_name, iree_host_size_t param_count,
    const iree_string_pair_t* params, iree_allocator_t allocator,
    iree_vm_module_t** out_module);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_VM_DYNAMIC_MODULE_H_
