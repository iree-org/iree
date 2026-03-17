// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Mock HAL executable cache for testing executable upload paths without
// a real compiler backend.
//
// The mock cache accepts a list of supported format strings at creation
// time. When prepare_executable is called with a matching format, it
// creates a mock executable with a configurable number of exports.
// Formats not in the supported list are rejected with INCOMPATIBLE.

#ifndef IREE_HAL_TESTING_MOCK_EXECUTABLE_CACHE_H_
#define IREE_HAL_TESTING_MOCK_EXECUTABLE_CACHE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates a mock executable cache that supports the given format strings.
// |supported_formats| is an array of |supported_format_count| C strings.
// The strings are copied into the cache's own storage.
//
// When prepare_executable is called, the cache checks whether the
// executable_format matches any supported format. If so, it creates a
// mock executable with 1 export. If not, it returns INCOMPATIBLE.
iree_status_t iree_hal_mock_executable_cache_create(
    iree_string_view_t identifier, const iree_string_view_t* supported_formats,
    iree_host_size_t supported_format_count, iree_allocator_t host_allocator,
    iree_hal_executable_cache_t** out_executable_cache);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_TESTING_MOCK_EXECUTABLE_CACHE_H_
