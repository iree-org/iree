// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_MODULES_HAL_TYPES_H_
#define IREE_MODULES_HAL_TYPES_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/vm/api.h"

IREE_VM_DECLARE_TYPE_ADAPTERS(iree_hal_allocator, iree_hal_allocator_t);
IREE_VM_DECLARE_TYPE_ADAPTERS(iree_hal_buffer, iree_hal_buffer_t);
IREE_VM_DECLARE_TYPE_ADAPTERS(iree_hal_buffer_view, iree_hal_buffer_view_t);
IREE_VM_DECLARE_TYPE_ADAPTERS(iree_hal_channel, iree_hal_channel_t);
IREE_VM_DECLARE_TYPE_ADAPTERS(iree_hal_command_buffer,
                              iree_hal_command_buffer_t);
IREE_VM_DECLARE_TYPE_ADAPTERS(iree_hal_descriptor_set_layout,
                              iree_hal_descriptor_set_layout_t);
IREE_VM_DECLARE_TYPE_ADAPTERS(iree_hal_device, iree_hal_device_t);
IREE_VM_DECLARE_TYPE_ADAPTERS(iree_hal_event, iree_hal_event_t);
IREE_VM_DECLARE_TYPE_ADAPTERS(iree_hal_executable, iree_hal_executable_t);
IREE_VM_DECLARE_TYPE_ADAPTERS(iree_hal_executable_cache,
                              iree_hal_executable_cache_t);
IREE_VM_DECLARE_TYPE_ADAPTERS(iree_hal_fence, iree_hal_fence_t);
IREE_VM_DECLARE_TYPE_ADAPTERS(iree_hal_pipeline_layout,
                              iree_hal_pipeline_layout_t);
IREE_VM_DECLARE_TYPE_ADAPTERS(iree_hal_semaphore, iree_hal_semaphore_t);

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Registers the custom types used by the inline HAL module.
IREE_API_EXPORT iree_status_t
iree_hal_module_register_inline_types(iree_vm_instance_t* instance);

// Registers the custom types used by the dynamic HAL executable loader module.
IREE_API_EXPORT iree_status_t
iree_hal_module_register_loader_types(iree_vm_instance_t* instance);

// Registers the custom types used by the full HAL module.
// This should only be called in the hosting executable/library that has the
// IREE VM/HAL compiled in.
IREE_API_EXPORT iree_status_t
iree_hal_module_register_all_types(iree_vm_instance_t* instance);

// Resolves the custom types used by the inline HAL module.
IREE_API_EXPORT iree_status_t
iree_hal_module_resolve_inline_types(iree_vm_instance_t* instance);

// Resolves the custom types used by the dynamic HAL executable loader module.
IREE_API_EXPORT iree_status_t
iree_hal_module_resolve_loader_types(iree_vm_instance_t* instance);

// Resolves all HAL types by looking them up on the instance.
// This should only be called in dynamically-loaded libraries that contain only
// the HAL shims.
IREE_API_EXPORT iree_status_t
iree_hal_module_resolve_all_types(iree_vm_instance_t* instance);

// TODO(benvanik): generate these list helpers:

IREE_API_EXPORT iree_hal_buffer_t* iree_vm_list_get_buffer_assign(
    const iree_vm_list_t* list, iree_host_size_t i);
IREE_API_EXPORT iree_hal_buffer_t* iree_vm_list_get_buffer_retain(
    const iree_vm_list_t* list, iree_host_size_t i);
IREE_API_EXPORT iree_status_t iree_vm_list_set_buffer_retain(
    iree_vm_list_t* list, iree_host_size_t i, iree_hal_buffer_t* value);

IREE_API_EXPORT iree_hal_buffer_view_t* iree_vm_list_get_buffer_view_assign(
    const iree_vm_list_t* list, iree_host_size_t i);
IREE_API_EXPORT iree_hal_buffer_view_t* iree_vm_list_get_buffer_view_retain(
    const iree_vm_list_t* list, iree_host_size_t i);
IREE_API_EXPORT iree_status_t iree_vm_list_set_buffer_view_retain(
    iree_vm_list_t* list, iree_host_size_t i, iree_hal_buffer_view_t* value);

IREE_API_EXPORT iree_hal_fence_t* iree_vm_list_get_fence_assign(
    const iree_vm_list_t* list, iree_host_size_t i);
IREE_API_EXPORT iree_hal_fence_t* iree_vm_list_get_fence_retain(
    const iree_vm_list_t* list, iree_host_size_t i);
IREE_API_EXPORT iree_status_t iree_vm_list_set_fence_retain(
    iree_vm_list_t* list, iree_host_size_t i, iree_hal_fence_t* value);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_MODULES_HAL_TYPES_H_
