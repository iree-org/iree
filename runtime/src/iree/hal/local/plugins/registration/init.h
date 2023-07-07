// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_LOCAL_PLUGINS_REGISTRATION_INIT_H_
#define IREE_HAL_LOCAL_PLUGINS_REGISTRATION_INIT_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/local/executable_plugin_manager.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates a plugin manager and registers all plugins in the
// --executable_plugin= flag list. Fails if any declared plugin cannot be
// loaded or is unsupported in this runtime configuration.  The plugins will be
// appended to the registry; import resolution happens in reverse registration
// order such that the last plugin registered will be scanned first.
iree_status_t iree_hal_executable_plugin_manager_create_from_flags(
    iree_allocator_t host_allocator,
    iree_hal_executable_plugin_manager_t** out_manager);

// Registers all plugins in the --executable_plugin= flag list. Fails if any
// declared plugin cannot be loaded or is unsupported in this runtime
// configuration. The plugins will be appended to the registry; import
// resolution happens in reverse registration order such that the last plugin
// registered will be scanned first.
iree_status_t iree_hal_register_executable_plugins_from_flags(
    iree_hal_executable_plugin_manager_t* manager,
    iree_allocator_t host_allocator);

// Registers a plugin with the given |spec| matching the flag value of
// --executable_plugin=. This allows for binding layers to load plugins
// programmatically using the same style the flag uses.
iree_status_t iree_hal_register_executable_plugin_from_spec(
    iree_hal_executable_plugin_manager_t* manager, iree_string_view_t spec,
    iree_allocator_t host_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_LOCAL_PLUGINS_REGISTRATION_INIT_H_
