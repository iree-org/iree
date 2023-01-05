// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/loop_emscripten.h"

#if defined(IREE_PLATFORM_EMSCRIPTEN)

#include <emscripten.h>

#include "iree/base/assert.h"

//===----------------------------------------------------------------------===//
// externs from loop_emscripten.js
//===----------------------------------------------------------------------===//

typedef uint32_t iree_loop_emscripten_scope_t;  // Opaque handle.

extern iree_loop_emscripten_scope_t iree_loop_allocate_scope();
extern void iree_loop_free_scope(iree_loop_emscripten_scope_t scope);

extern iree_status_t iree_loop_command_call(iree_loop_emscripten_scope_t scope,
                                            iree_loop_callback_fn_t callback,
                                            void* user_data, iree_loop_t loop);

//===----------------------------------------------------------------------===//
// iree_loop_emscripten_t
//===----------------------------------------------------------------------===//

typedef struct iree_loop_emscripten_t {
  iree_allocator_t allocator;
  iree_loop_emscripten_scope_t scope;
} iree_loop_emscripten_t;

IREE_API_EXPORT iree_status_t iree_loop_emscripten_allocate(
    iree_allocator_t allocator, iree_loop_emscripten_t** out_loop) {
  IREE_ASSERT_ARGUMENT(out_loop);
  iree_loop_emscripten_t* loop = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, sizeof(*loop), (void**)&loop));
  loop->allocator = allocator;
  loop->scope = iree_loop_allocate_scope();
  *out_loop = loop;
  return iree_ok_status();
}

IREE_API_EXPORT void iree_loop_emscripten_free(iree_loop_emscripten_t* loop) {
  IREE_ASSERT_ARGUMENT(loop);
  iree_allocator_t allocator = loop->allocator;

  iree_loop_free_scope(loop->scope);

  // After all operations are cleared we can release the data structures.
  iree_allocator_free(allocator, loop);
}

static iree_status_t iree_loop_emscripten_run_call(
    iree_loop_emscripten_t* loop_emscripten, iree_loop_call_params_t* params) {
  iree_loop_t loop = iree_loop_emscripten(loop_emscripten);
  return iree_loop_command_call(loop_emscripten->scope, params->callback.fn,
                                params->callback.user_data, loop);
}

// Control function for the Emscripten loop.
IREE_API_EXPORT iree_status_t
iree_loop_emscripten_ctl(void* self, iree_loop_command_t command,
                         const void* params, void** inout_ptr) {
  IREE_ASSERT_ARGUMENT(self);

  iree_loop_emscripten_t* loop_emscripten = (iree_loop_emscripten_t*)self;

  // NOTE: we return immediately to make this all (hopefully) tail calls.
  switch (command) {
    case IREE_LOOP_COMMAND_CALL:
      return iree_loop_emscripten_run_call(loop_emscripten,
                                           (iree_loop_call_params_t*)params);
    case IREE_LOOP_COMMAND_DISPATCH:
    case IREE_LOOP_COMMAND_WAIT_UNTIL:
    case IREE_LOOP_COMMAND_WAIT_ONE:
    case IREE_LOOP_COMMAND_WAIT_ALL:
    case IREE_LOOP_COMMAND_WAIT_ANY:
      // TODO(scotttodd): implement these commands
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "unimplemented loop command");
    case IREE_LOOP_COMMAND_DRAIN:
      return iree_make_status(IREE_STATUS_DEADLINE_EXCEEDED,
                              "unsupported loop command");
    default:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "unimplemented loop command");
  }
}

#endif  // IREE_PLATFORM_EMSCRIPTEN
