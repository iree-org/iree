// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_LOCAL_PLUGINS_STATIC_PLUGIN_H_
#define IREE_HAL_LOCAL_PLUGINS_STATIC_PLUGIN_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/local/executable_plugin_manager.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates a statically-declared executable import plugin available from the
// given |query_fn|. The plugin will be loaded and verified for compatibility.
//
// Optionally key-value parameters may be provided to the plugin on
// initialization. They strings referenced need only exist during creation and
// plugins will clone any they need to retain.
//
// |host_allocator| will be used to create the plugin wrapper and be passed to
// the plugin to service all additional allocation requests it may have. It must
// remain valid for the lifetime of the plugin.
iree_status_t iree_hal_static_executable_plugin_create(
    iree_hal_executable_plugin_query_fn_t query_fn,
    iree_host_size_t param_count, const iree_string_pair_t* params,
    iree_allocator_t host_allocator, iree_hal_executable_plugin_t** out_plugin);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_LOCAL_PLUGINS_STATIC_PLUGIN_H_
