// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_LOCAL_LOADERS_SYSTEM_LIBRARY_LOADER_H_
#define IREE_HAL_LOCAL_LOADERS_SYSTEM_LIBRARY_LOADER_H_

#include <stdbool.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/local/executable_loader.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_executable_plugin_manager_t
    iree_hal_executable_plugin_manager_t;

// Creates an executable loader that can load files from platform-supported
// dynamic libraries (such as .dylib on darwin, .so on linux, .dll on windows).
//
// This uses the legacy "dylib"-style format that will be deleted soon and is
// only a placeholder until the compiler can be switched to output
// iree_hal_executable_library_t-compatible files.
iree_status_t iree_hal_system_library_loader_create(
    iree_hal_executable_plugin_manager_t* plugin_manager,
    iree_allocator_t host_allocator,
    iree_hal_executable_loader_t** out_executable_loader);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_LOCAL_LOADERS_SYSTEM_LIBRARY_LOADER_H_
