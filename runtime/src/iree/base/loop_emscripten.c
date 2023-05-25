// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/loop_emscripten.h"

#if defined(IREE_PLATFORM_EMSCRIPTEN)

#include <emscripten.h>

#include "iree/base/assert.h"
#include "iree/base/internal/wait_handle.h"
#include "iree/base/wait_source.h"

// General implementation notes:
//
// Several loop commands use 'deadline_ns' to implement timeouts. We convert
// those int64 deadlines to int32 timeouts since
//   * int64 support across C <-> JS requires BigInt
//   * we _might_ get microsecond precision on the web, never nanosecond
//   * this will be passed to setTimeout, which takes milliseconds

//===----------------------------------------------------------------------===//
// externs from loop_emscripten.js
//===----------------------------------------------------------------------===//

typedef uint32_t iree_loop_emscripten_scope_t;  // Opaque handle.

extern iree_loop_emscripten_scope_t iree_loop_allocate_scope();
extern void iree_loop_free_scope(iree_loop_emscripten_scope_t scope_handle);

extern iree_status_t iree_loop_command(
    iree_loop_emscripten_scope_t scope_handle, int command,
    iree_loop_callback_fn_t callback, void* user_data, uint32_t timeout_ms,
    int promise_handles_count, int* promise_handles, iree_loop_t loop);

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
  // Note: ignoring params->priority.
  return iree_loop_command(loop_emscripten->scope, IREE_LOOP_COMMAND_CALL,
                           params->callback.fn, params->callback.user_data,
                           /*timeout_ms=*/0, /*promise_handles_count=*/0,
                           /*promise_handles=*/NULL, loop);
}

static iree_status_t iree_loop_emscripten_run_wait_until(
    iree_loop_emscripten_t* loop_emscripten,
    iree_loop_wait_until_params_t* params) {
  iree_loop_t loop = iree_loop_emscripten(loop_emscripten);
  uint32_t timeout_ms =
      iree_absolute_deadline_to_timeout_ms(params->deadline_ns);
  return iree_loop_command(loop_emscripten->scope, IREE_LOOP_COMMAND_WAIT_UNTIL,
                           params->callback.fn, params->callback.user_data,
                           timeout_ms, /*promise_handles_count=*/0,
                           /*promise_handles=*/NULL, loop);
}

static iree_status_t iree_loop_emscripten_get_promise_handle(
    iree_wait_source_t wait_source, int* out_handle) {
  if (iree_wait_source_is_immediate(wait_source)) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "wait immediate not implemented");
  }

  iree_wait_handle_t wait_handle = iree_wait_handle_immediate();
  iree_wait_handle_t* wait_handle_ptr =
      iree_wait_handle_from_source(&wait_source);
  if (wait_handle_ptr) {
    wait_handle = *wait_handle_ptr;
  } else {
    // TODO(scotttodd): iree_wait_source_export (see loop_sync.c)
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "wait handle export/import not implemented");
  }

  if (wait_handle.type != IREE_WAIT_PRIMITIVE_TYPE_JAVASCRIPT_PROMISE) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "only Promise wait primitives are supported");
  }

  *out_handle = wait_handle.value.promise.handle;
  return iree_ok_status();
}

static iree_status_t iree_loop_emscripten_run_wait_one(
    iree_loop_emscripten_t* loop_emscripten,
    iree_loop_wait_one_params_t* params) {
  int promise_handle = 0;
  IREE_RETURN_IF_ERROR(iree_loop_emscripten_get_promise_handle(
      params->wait_source, &promise_handle));
  iree_loop_t loop = iree_loop_emscripten(loop_emscripten);
  uint32_t timeout_ms =
      iree_absolute_deadline_to_timeout_ms(params->deadline_ns);
  return iree_loop_command(loop_emscripten->scope, IREE_LOOP_COMMAND_WAIT_ONE,
                           params->callback.fn, params->callback.user_data,
                           timeout_ms, 1, &promise_handle, loop);
}

static iree_status_t iree_loop_emscripten_run_wait_any(
    iree_loop_emscripten_t* loop_emscripten,
    iree_loop_wait_multi_params_t* params) {
  int* promise_handles = (int*)iree_alloca(sizeof(int) * params->count);

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < params->count; ++i) {
    status = iree_loop_emscripten_get_promise_handle(params->wait_sources[i],
                                                     &promise_handles[i]);
    if (!iree_status_is_ok(status)) break;
  }

  if (iree_status_is_ok(status)) {
    iree_loop_t loop = iree_loop_emscripten(loop_emscripten);
    uint32_t timeout_ms =
        iree_absolute_deadline_to_timeout_ms(params->deadline_ns);
    status =
        iree_loop_command(loop_emscripten->scope, IREE_LOOP_COMMAND_WAIT_ANY,
                          params->callback.fn, params->callback.user_data,
                          timeout_ms, params->count, promise_handles, loop);
  }

  iree_allocator_free(loop_emscripten->allocator, promise_handles);
  return status;
}

static iree_status_t iree_loop_emscripten_run_wait_all(
    iree_loop_emscripten_t* loop_emscripten,
    iree_loop_wait_multi_params_t* params) {
  int* promise_handles = (int*)iree_alloca(sizeof(int) * params->count);

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < params->count; ++i) {
    status = iree_loop_emscripten_get_promise_handle(params->wait_sources[i],
                                                     &promise_handles[i]);
    if (!iree_status_is_ok(status)) break;
  }

  if (iree_status_is_ok(status)) {
    iree_loop_t loop = iree_loop_emscripten(loop_emscripten);
    uint32_t timeout_ms =
        iree_absolute_deadline_to_timeout_ms(params->deadline_ns);
    status =
        iree_loop_command(loop_emscripten->scope, IREE_LOOP_COMMAND_WAIT_ALL,
                          params->callback.fn, params->callback.user_data,
                          timeout_ms, params->count, promise_handles, loop);
  }

  iree_allocator_free(loop_emscripten->allocator, promise_handles);
  return status;
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
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "IREE_LOOP_COMMAND_DISPATCH not implemented");
    case IREE_LOOP_COMMAND_WAIT_UNTIL:
      return iree_loop_emscripten_run_wait_until(
          loop_emscripten, (iree_loop_wait_until_params_t*)params);
    case IREE_LOOP_COMMAND_WAIT_ONE:
      return iree_loop_emscripten_run_wait_one(
          loop_emscripten, (iree_loop_wait_one_params_t*)params);
    case IREE_LOOP_COMMAND_WAIT_ANY:
      return iree_loop_emscripten_run_wait_any(
          loop_emscripten, (iree_loop_wait_multi_params_t*)params);
    case IREE_LOOP_COMMAND_WAIT_ALL:
      return iree_loop_emscripten_run_wait_all(
          loop_emscripten, (iree_loop_wait_multi_params_t*)params);
    case IREE_LOOP_COMMAND_DRAIN:
      return iree_make_status(IREE_STATUS_DEADLINE_EXCEEDED,
                              "unsupported loop command");
    default:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "unimplemented loop command");
  }
}

#endif  // IREE_PLATFORM_EMSCRIPTEN
