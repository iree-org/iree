// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_NULL_EXECUTABLE_CACHE_H_
#define IREE_HAL_DRIVERS_NULL_EXECUTABLE_CACHE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

//===----------------------------------------------------------------------===//
// iree_hal_null_executable_cache_t
//===----------------------------------------------------------------------===//

// Creates a no-op executable cache that does not cache at all.
// This is useful to isolate pipeline caching behavior and verify compilation
// behavior.
//
// TODO(null): retain any shared resources (like device handles and symbols)
// that are needed to create executables.
iree_status_t iree_hal_null_executable_cache_create(
    iree_string_view_t identifier, iree_allocator_t host_allocator,
    iree_hal_executable_cache_t** out_executable_cache);

#endif  // IREE_HAL_DRIVERS_NULL_EXECUTABLE_CACHE_H_
