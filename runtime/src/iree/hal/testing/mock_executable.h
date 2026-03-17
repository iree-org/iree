// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Mock HAL executable for testing executable upload and dispatch paths
// without a real compiler backend.
//
// The mock executable stores the provided binary data and reports a
// configurable number of exports with default metadata. It does not
// execute anything — dispatch operations that reference it will succeed
// on the server side (the local device's queue_dispatch handles the
// actual execution), but the mock executable itself is inert.

#ifndef IREE_HAL_TESTING_MOCK_EXECUTABLE_H_
#define IREE_HAL_TESTING_MOCK_EXECUTABLE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates a mock executable with the given number of exports.
// Each export has default metadata (binding_count=0, constant_count=0,
// workgroup_size=1x1x1).
iree_status_t iree_hal_mock_executable_create(
    iree_host_size_t export_count, iree_allocator_t host_allocator,
    iree_hal_executable_t** out_executable);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_TESTING_MOCK_EXECUTABLE_H_
