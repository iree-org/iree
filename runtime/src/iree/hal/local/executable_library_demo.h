// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_LOCAL_EXECUTABLE_LIBRARY_DEMO_H_
#define IREE_HAL_LOCAL_EXECUTABLE_LIBRARY_DEMO_H_

#include <stdint.h>

#include "iree/hal/local/executable_library.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Ideally we would have the IREE compiler generate a header like this so that
// it's possible to manually call into executables. For now this is just an
// example for the demo: the real HAL does not require this header as it
// dlsym's the function pointer and packs the push constants itself.

// Push constants used in the 'dispatch_tile_a' entry point.
typedef union {
  uint32_t values[1];
  struct {
    float f0;
  };
} dispatch_tile_a_push_constants_t;

// Returns a simple demo library with the following structure:
//
// Name: 'demo_library'
//
// [0] 'dispatch_tile_a': matmul+div
//       push constants: 1 (dispatch_tile_a_push_constants_t)
//       bindings: 2
//         [0] = R
//         [1] = W
//
// [1] 'dispatch_tile_b': conv2d[512x512]
//       push constants: 0
//       bindings: 0
//
const iree_hal_executable_library_header_t** demo_executable_library_query(
    iree_hal_executable_library_version_t max_version,
    const iree_hal_executable_environment_v0_t* environment);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_LOCAL_EXECUTABLE_LIBRARY_DEMO_H_
