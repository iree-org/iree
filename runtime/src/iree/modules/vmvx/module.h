// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_MODULES_VMVX_MODULE_H_
#define IREE_MODULES_VMVX_MODULE_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/vm/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates the VMVX module with a default configuration.
IREE_API_EXPORT iree_status_t iree_vmvx_module_create(
    iree_vm_instance_t* instance, iree_allocator_t host_allocator,
    iree_vm_module_t** out_module);

// Updates the context-local state of the module.
IREE_API_EXPORT void iree_vmvx_module_state_update_workgroup_state(
    iree_vm_module_state_t* module_state, uint32_t processor_id);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_MODULES_VMVX_MODULE_H_
