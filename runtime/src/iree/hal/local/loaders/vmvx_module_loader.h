// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_LOCAL_LOADERS_VMVX_MODULE_LOADER_H_
#define IREE_HAL_LOCAL_LOADERS_VMVX_MODULE_LOADER_H_

#include <stdbool.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/local/executable_loader.h"
#include "iree/vm/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates an executable loader that can load compiled IREE VM bytecode modules
// using the VMVX module. |instance| will be used for all loaded contexts.
iree_status_t iree_hal_vmvx_module_loader_create(
    iree_vm_instance_t* instance, iree_allocator_t host_allocator,
    iree_hal_executable_loader_t** out_executable_loader);

// Creates an executable loader that can load compiled IREE VM bytecode modules
// using the VMVX module. Uses an isolated VM instance.
iree_status_t iree_hal_vmvx_module_loader_create_isolated(
    iree_allocator_t host_allocator,
    iree_hal_executable_loader_t** out_executable_loader);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_LOCAL_LOADERS_VMVX_MODULE_LOADER_H_
