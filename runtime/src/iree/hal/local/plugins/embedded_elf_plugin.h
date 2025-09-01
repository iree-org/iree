// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_LOCAL_PLUGINS_EMBEDDED_ELF_PLUGIN_H_
#define IREE_HAL_LOCAL_PLUGINS_EMBEDDED_ELF_PLUGIN_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/local/executable_plugin_manager.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Loads a standalone embedded ELF executable import plugin from the given
// in-memory |buffer|. The plugin will be loaded and verified for compatibility.
// Only ELFs compiled for portable deployment and declaring the
// IREE_HAL_EXECUTABLE_PLUGIN_FEATURE_STANDALONE feature are supported.
//
// Optionally key-value parameters may be provided to the plugin on
// initialization. They strings referenced need only exist during creation and
// plugins will clone any they need to retain.
//
// |host_allocator| will be used to create the plugin wrapper and be passed to
// the plugin to service all additional allocation requests it may have. It must
// remain valid for the lifetime of the plugin.
iree_status_t iree_hal_embedded_elf_executable_plugin_load_from_memory(
    iree_const_byte_span_t buffer, iree_host_size_t param_count,
    const iree_string_pair_t* params, iree_allocator_t host_allocator,
    iree_hal_executable_plugin_t** out_plugin);

#if IREE_FILE_IO_ENABLE
// Loads a standalone embedded ELF executable import plugin from the given
// file |path|. See iree_hal_embedded_elf_executable_plugin_load_from_memory
// for more information.
iree_status_t iree_hal_embedded_elf_executable_plugin_load_from_file(
    iree_string_view_t path, iree_host_size_t param_count,
    const iree_string_pair_t* params, iree_allocator_t host_allocator,
    iree_hal_executable_plugin_t** out_plugin);
#endif  // IREE_FILE_IO_ENABLE

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_LOCAL_PLUGINS_EMBEDDED_ELF_PLUGIN_H_
