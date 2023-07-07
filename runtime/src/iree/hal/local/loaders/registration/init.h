// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_LOCAL_LOADERS_REGISTRATION_INIT_H_
#define IREE_HAL_LOCAL_LOADERS_REGISTRATION_INIT_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/local/executable_loader.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_executable_plugin_manager_t
    iree_hal_executable_plugin_manager_t;

// Queries and creates all linked in executable library loaders and retains them
// in the |out_loaders| list. |out_count| contains the total number of loaders.
// If there is not enough |capacity| to store all of the loaders
// IREE_STATUS_OUT_OF_RANGE is returned and |out_count| is set to the required
// capacity. Loaders are retained upon return and must be released by the
// caller.
//
// Default options are used to create the loaders. If customization is required
// then callers should create the loaders themselves.
//
// Usage:
//  iree_host_size_t count = 0;
//  iree_hal_executable_loader_t* loaders[8] = {NULL};
//  IREE_RETURN_IF_ERROR(iree_hal_create_all_available_executable_loaders(
//      plugin_manager,
//      IREE_ARRAYSIZE(loaders), &count, loaders,
//      host_allocator));
//  ...
//  // use up to count loaders
//  ...
//  for (iree_host_size_t i = 0; i < count; ++i) {
//    iree_hal_executable_loader_release(loaders[i]);
//  }
IREE_API_EXPORT iree_status_t iree_hal_create_all_available_executable_loaders(
    iree_hal_executable_plugin_manager_t* plugin_manager,
    iree_host_size_t capacity, iree_host_size_t* out_count,
    iree_hal_executable_loader_t** loaders, iree_allocator_t host_allocator);

// Creates an executable loader with the given |name|.
// |out_executable_loader| must be released by the caller.
IREE_API_EXPORT iree_status_t iree_hal_create_executable_loader_by_name(
    iree_string_view_t name,
    iree_hal_executable_plugin_manager_t* plugin_manager,
    iree_allocator_t host_allocator,
    iree_hal_executable_loader_t** out_executable_loader);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_LOCAL_LOADERS_REGISTRATION_INIT_H_
