// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BASE_LOOP_EMSCRIPTEN_H_
#define IREE_BASE_LOOP_EMSCRIPTEN_H_

#include "iree/base/api.h"

#if defined(IREE_PLATFORM_EMSCRIPTEN)

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_loop_emscripten_t
//===----------------------------------------------------------------------===//

// A loop backed by the web browser event loop, built using Emscripten.
// TODO(scotttodd): comment on thread safety (when established)
typedef struct iree_loop_emscripten_t iree_loop_emscripten_t;

// Allocates a loop using |allocator| stored into |out_loop|.
IREE_API_EXPORT iree_status_t iree_loop_emscripten_allocate(
    iree_allocator_t allocator, iree_loop_emscripten_t** out_loop);

// Frees |loop_emscripten|, aborting all pending operations.
IREE_API_EXPORT void iree_loop_emscripten_free(iree_loop_emscripten_t* loop);

IREE_API_EXPORT iree_status_t
iree_loop_emscripten_ctl(void* self, iree_loop_command_t command,
                         const void* params, void** inout_ptr);

// Returns a loop that uses |data|.
static inline iree_loop_t iree_loop_emscripten(iree_loop_emscripten_t* data) {
  iree_loop_t loop = {
      data,
      iree_loop_emscripten_ctl,
  };
  return loop;
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_PLATFORM_EMSCRIPTEN

#endif  // IREE_BASE_LOOP_EMSCRIPTEN_H_
