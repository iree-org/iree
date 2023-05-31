// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOOLING_MODULES_REGISTRY_H_
#define IREE_TOOLING_MODULES_REGISTRY_H_

#include "iree/base/api.h"
#include "iree/vm/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Registers all types used by available modules.
// NOTE: this is only required as a separate step in the tooling as we perform
// dynamic module resolution and in order to load modules to find their
// dependencies their dependent types must be registered first. This could be
// fixed in the future to only resolve types when a module is instantiated in
// a context instead of when loaded but is TBD.
iree_status_t iree_tooling_register_all_module_types(
    iree_vm_instance_t* instance);

// Resolves a module dependency to an initialized module.
// Returns OK if the dependency is optional and not found.
iree_status_t iree_tooling_resolve_module_dependency(
    iree_vm_instance_t* instance, const iree_vm_module_dependency_t* dependency,
    iree_allocator_t host_allocator, iree_vm_module_t** out_module);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_TOOLING_MODULES_REGISTRY_H_
