// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// TODO(scotttodd): rewrite as a JS/C file and test with Promises
//   The C++ test uses IREE_LOOP_COMMAND_DRAIN, which is not implemented here

#include "iree/base/loop_emscripten.h"

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

// Contains the test definitions applied to all loop implementations:
#include "iree/base/loop_test.h"

void AllocateLoop(iree_status_t* out_status, iree_allocator_t allocator,
                  iree_loop_t* out_loop) {
  iree_loop_emscripten_t* loop_emscripten = NULL;
  IREE_CHECK_OK(iree_loop_emscripten_allocate(allocator, &loop_emscripten));

  *out_status = iree_ok_status();
  *out_loop = iree_loop_emscripten(loop_emscripten);
}

void FreeLoop(iree_allocator_t allocator, iree_loop_t loop) {
  iree_loop_emscripten_t* loop_emscripten = (iree_loop_emscripten_t*)loop.self;
  iree_loop_emscripten_free(loop_emscripten);
}
