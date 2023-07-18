// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// This is the JavaScript side of wait_handle_emscripten.c
//
// Each `iree_wait_handle_t` tracks a "iree_wait_primitive".
// This Emscripten implementation is backed by Promises and it
//   * creates a "promise wrapper" for each active wait handle, where each
//     wrapper includes a Promise object and supporting data
//   * returns an opaque handle used to reference that wrapper

const LibraryIreeWaitHandleEmscripten = {
  $IreeWaitHandleEmscripten: {
    // Counter for opaque handles, shared across the entire module/process/page.
    // Note: start at 1, leaving 0 as a sentinel for uninitialized.
    _nextPromiseHandle: 1,

    // Dictionary of promiseHandles -> promise wrapper objects:
    // {
    //   promise:    Promise object
    //   isSettled:  Bool
    //   resolve:    Function that resolves |promise|
    //   reject:     Function that rejects  |promise|
    // }
    // TODO(scotttodd): use a class for that ^
    _promiseWrappers: {},

    _createPromiseWrapper: function(promiseHandle, initialState) {
      if (initialState) {
        // Start resolved (settled).
        const resolvedPromise = Promise.resolve();
        const promiseWrapper = {
          promise: resolvedPromise,
          isSettled: true,
        };
        IreeWaitHandleEmscripten._promiseWrappers[promiseHandle] =
            promiseWrapper;
      } else {
        // Start pending, track the resolve and reject functions.
        const pendingPromise = new Promise((resolve, reject) => {
          const promiseWrapper = {
            promise: undefined,
            isSettled: false,
            resolve: resolve,
            reject: reject,
          };
          IreeWaitHandleEmscripten._promiseWrappers[promiseHandle] =
              promiseWrapper;
        });
        IreeWaitHandleEmscripten._promiseWrappers[promiseHandle].promise =
            pendingPromise;
      }
    },

    getPromise: function(promiseHandle) {
      return IreeWaitHandleEmscripten._promiseWrappers[promiseHandle].promise;
    },
  },

  iree_wait_primitive_promise_create: function(initialState) {
    const promiseHandle = IreeWaitHandleEmscripten._nextPromiseHandle++;
    IreeWaitHandleEmscripten._createPromiseWrapper(promiseHandle, initialState);
    return promiseHandle;
  },

  iree_wait_primitive_promise_delete: function(promiseHandle) {
    const promiseWrapper =
        IreeWaitHandleEmscripten._promiseWrappers[promiseHandle];
    if (!promiseWrapper.isSettled && promiseWrapper.reject !== undefined) {
      promiseWrapper.reject();
      promiseWrapper.isSettled = true;
    }
    delete IreeWaitHandleEmscripten._promiseWrappers[promiseHandle];
  },

  iree_wait_primitive_promise_set: function(promiseHandle) {
    const promiseWrapper =
        IreeWaitHandleEmscripten._promiseWrappers[promiseHandle];
    if (promiseWrapper.resolve !== undefined) {
      promiseWrapper.resolve();
      promiseWrapper.isSettled = true;
    }
  },

  iree_wait_primitive_promise_reset: function(promiseHandle) {
    const promiseWrapper =
        IreeWaitHandleEmscripten._promiseWrappers[promiseHandle];

    // No-op if already unsignaled.
    if (!promiseWrapper.isSettled) return;

    // Promises are are permanently resolved and can't be 'reset', so create a
    // new wrapper using the same handle.
    // Since the previous Promise was resolved or rejected already, listeners
    // should have already been notified and we aren't leaving any orphaned.
    // (?) This synchronizes on the browser event loop, so there are no races.
    IreeWaitHandleEmscripten._createPromiseWrapper(promiseHandle, false);
  },
}

autoAddDeps(LibraryIreeWaitHandleEmscripten, '$IreeWaitHandleEmscripten');
mergeInto(LibraryManager.library, LibraryIreeWaitHandleEmscripten);
