// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// clang-format off: must be included before all other headers.
#include "iree/base/internal/wait_handle_impl.h"
// clang-format on

#include "iree/base/api.h"
#include "iree/base/internal/wait_handle.h"

// This implementation for the web platform via Emscripten uses JavaScript
// Promise objects for asynchronous waiting:
// https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Promise
//
// Synchronous wait APIs (e.g. iree_wait_one) are not supported, as using
// Promises requires interacting with the asynchronous browser event loop.
//   * Note: JSPI (https://v8.dev/blog/jspi) and Asyncify
//     (https://emscripten.org/docs/porting/asyncify.html)
//     could probably be used to interop with this, but the real goal of this
//     implementation is browser-native asynchronous behavior.
//
// Wait handles may be asynchronously waited on via the JavaScript API:
//   ```
//   // C
//   int handle = iree_wait_primitive_promise_create(false);
//
//   // JS
//   const promiseWrapper = IreeWaitHandlePromise.getPromiseWrapper(handle);
//   promiseWrapper.promise.then(() => { ... });
//
//   // C
//   iree_wait_primitive_promise_set(handle);
//   ```
#if IREE_WAIT_API == IREE_WAIT_API_PROMISE && defined(IREE_PLATFORM_EMSCRIPTEN)

#include <emscripten.h>

//===----------------------------------------------------------------------===//
// externs from wait_handle_emscripten.js
//===----------------------------------------------------------------------===//

extern int iree_wait_primitive_promise_create(bool initial_state);
extern void iree_wait_primitive_promise_delete(int promise_handle);
extern void iree_wait_primitive_promise_set(int promise_handle);
extern void iree_wait_primitive_promise_reset(int promise_handle);

//===----------------------------------------------------------------------===//
// iree_wait_primitive_* raw calls
//===----------------------------------------------------------------------===//

void iree_wait_handle_close(iree_wait_handle_t* handle) {
  switch (handle->type) {
#if defined(IREE_HAVE_WAIT_TYPE_JAVASCRIPT_PROMISE)
    case IREE_WAIT_PRIMITIVE_TYPE_JAVASCRIPT_PROMISE: {
      iree_wait_primitive_promise_delete(handle->value.promise.handle);
      break;
    }
#endif  // IREE_HAVE_WAIT_TYPE_JAVASCRIPT_PROMISE
    default:
      break;
  }
  iree_wait_handle_deinitialize(handle);
}

iree_status_t iree_wait_one(iree_wait_handle_t* handle,
                            iree_time_t deadline_ns) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "wait_one unimplemented on Emscripten");
}

//===----------------------------------------------------------------------===//
// iree_event_t
//===----------------------------------------------------------------------===//

iree_status_t iree_event_initialize(bool initial_state,
                                    iree_event_t* out_event) {
  memset(out_event, 0, sizeof(*out_event));
  out_event->type = IREE_WAIT_PRIMITIVE_TYPE_JAVASCRIPT_PROMISE;

  int promise_handle = iree_wait_primitive_promise_create(initial_state);
  out_event->value.promise.handle = promise_handle;
  return iree_ok_status();
}

void iree_event_deinitialize(iree_event_t* event) {
  iree_wait_handle_close(event);
}

void iree_event_set(iree_event_t* event) {
  if (!event) return;
  iree_wait_primitive_promise_set(event->value.promise.handle);
}

void iree_event_reset(iree_event_t* event) {
  if (!event) return;
  iree_wait_primitive_promise_reset(event->value.promise.handle);
}

#endif  // IREE_WAIT_API == IREE_WAIT_API_INPROC &&
        // defined(IREE_PLATFORM_EMSCRIPTEN)
