// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_LOCAL_LOADERS_STATIC_LIBRARY_LOADER_H_
#define IREE_HAL_LOCAL_LOADERS_STATIC_LIBRARY_LOADER_H_

#include <stdbool.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/local/executable_library.h"
#include "iree/hal/local/executable_loader.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates a library loader that exposes the provided libraries to the HAL for
// use as executables.
//
// This loader will handle executable formats of 'static'. Version checks will
// ensure that the IREE compiler-produced static library version is one that the
// runtime can support.
//
// The name defined on each library will be used to lookup the executables and
// must match with the names used during compilation exactly. The
// iree_hal_executable_params_t used to reference the executables will contain
// the library name and be used to lookup the library in the list.
//
// Multiple static library loaders can be registered in cases when several
// independent sets of libraries are linked in however duplicate names both
// within and across loaders will result in undefined behavior.
iree_status_t iree_hal_static_library_loader_create(
    iree_host_size_t library_count,
    const iree_hal_executable_library_query_fn_t* library_query_fns,
    iree_hal_executable_import_provider_t import_provider,
    iree_allocator_t host_allocator,
    iree_hal_executable_loader_t** out_executable_loader);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_LOCAL_LOADERS_STATIC_LIBRARY_LOADER_H_
