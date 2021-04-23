// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef IREE_HAL_LOCAL_EXECUTABLE_LIBRARY_DEMO_H_
#define IREE_HAL_LOCAL_EXECUTABLE_LIBRARY_DEMO_H_

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
    iree_hal_executable_library_version_t max_version, void* reserved);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_LOCAL_EXECUTABLE_LIBRARY_DEMO_H_
