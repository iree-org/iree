// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOOLING_CONTEXT_UTIL_H_
#define IREE_TOOLING_CONTEXT_UTIL_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/vm/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Module management
//===----------------------------------------------------------------------===//

// On-stack storage for a list of VM modules.
// Contained modules are retained until the list is reset.
typedef struct {
  iree_host_size_t capacity;
  iree_host_size_t count;
  iree_vm_module_t* values[64];
} iree_tooling_module_list_t;

// Initializes |out_list| to empty.
void iree_tooling_module_list_initialize(iree_tooling_module_list_t* out_list);

// Copies |source_list| to |out_list| and retains all modules.
void iree_tooling_module_list_clone(
    const iree_tooling_module_list_t* source_list,
    iree_tooling_module_list_t* out_list);

// Resets |list|, releasing all retained modules.
void iree_tooling_module_list_reset(iree_tooling_module_list_t* list);

// Pushes |module| onto the end of |list| and retains a reference.
iree_status_t iree_tooling_module_list_push_back(
    iree_tooling_module_list_t* list, iree_vm_module_t* module);

// Returns the last module in the module list or NULL if the list is empty.
iree_vm_module_t* iree_tooling_module_list_back(
    const iree_tooling_module_list_t* list);

// Resolves module dependencies required by |user_modules| and produces a
// flattened list of all resolved modules.
//
// |default_device_uri| can be specified to provide a default if a device flag
// is not provided by the user.
// |out_device| will contain the created device if using the full HAL.
// |out_device_allocator| can be used to allocate buffers for use with the
// context and is available in all execution models.
iree_status_t iree_tooling_resolve_modules(
    iree_vm_instance_t* instance, iree_host_size_t user_module_count,
    iree_vm_module_t** user_modules, iree_string_view_t default_device_uri,
    iree_allocator_t host_allocator, iree_tooling_module_list_t* resolved_list,
    iree_hal_device_t** out_device,
    iree_hal_allocator_t** out_device_allocator);

//===----------------------------------------------------------------------===//
// Module loading
//===----------------------------------------------------------------------===//

// Loads modules in the order specified by the --module= flag.
// Appends the modules to the |list|.
iree_status_t iree_tooling_load_modules_from_flags(
    iree_vm_instance_t* instance, iree_allocator_t host_allocator,
    iree_tooling_module_list_t* list);

//===----------------------------------------------------------------------===//
// Context management
//===----------------------------------------------------------------------===//

// Creates a VM instance and registers types.
iree_status_t iree_tooling_create_instance(iree_allocator_t host_allocator,
                                           iree_vm_instance_t** out_instance);

// Creates a new VM context with the provided |user_modules| and dependent
// system modules. The provided user module order is preserved.
// The context is returned frozen.
//
// |default_device_uri| can be specified to provide a default if a device flag
// is not provided by the user.
// |out_device| will contain the created device if using the full HAL.
// |out_device_allocator| can be used to allocate buffers for use with the
// context and is available in all execution models.
iree_status_t iree_tooling_create_context_from_flags(
    iree_vm_instance_t* instance, iree_host_size_t user_module_count,
    iree_vm_module_t** user_modules, iree_string_view_t default_device_uri,
    iree_allocator_t host_allocator, iree_vm_context_t** out_context,
    iree_hal_device_t** out_device,
    iree_hal_allocator_t** out_device_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_TOOLING_CONTEXT_UTIL_H_
