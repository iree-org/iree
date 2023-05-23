// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// This is the JavaScript side of loop_emscripten.c
//
// References:
//   * https://emscripten.org/docs/porting/connecting_cpp_and_javascript/Interacting-with-code.html
//   * https://github.com/evanw/emscripten-library-generator
//   * https://github.com/emscripten-core/emscripten/tree/main/src
//   * https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Promise
//   * https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Promise/race

const LibraryIreeLoopEmscripten = {
  $iree_loop_emscripten_support__postset: 'iree_loop_emscripten_support();',
  $iree_loop_emscripten_support: function() {
    const IREE_STATUS_OK = 0;
    const IREE_STATUS_CODE_MASK = 0x1F;
    const IREE_STATUS_ABORTED = 10 & IREE_STATUS_CODE_MASK;
    const IREE_STATUS_OUT_OF_RANGE = 11 & IREE_STATUS_CODE_MASK;

    class LoopCommand {
      abort() {}
    }

    // IREE_LOOP_COMMAND_CALL
    class LoopCommandCall extends LoopCommand {
      constructor(scope, operationId, callback, user_data, loop) {
        super();

        this.callback = callback;
        this.user_data = user_data;
        this.loop = loop;

        this.timeoutId = setTimeout(() => {
          Module['dynCall'](
              'iiii', this.callback,
              [this.user_data, this.loop, IREE_STATUS_OK]);
          // TODO(scotttodd): handle the returned status (sticky failure state?)
          //     at least free the status so it doesn't leak
          delete scope.pendingOperations[operationId];
        }, 0);
      }

      abort() {
        clearTimeout(this.timeoutId);
        Module['dynCall'](
            'iiii', this.callback,
            [this.user_data, this.loop, IREE_STATUS_ABORTED]);
      }
    }

    // IREE_LOOP_COMMAND_WAIT_UNTIL
    class LoopCommandWaitUntil extends LoopCommand {
      constructor(scope, operationId, callback, user_data, timeout_ms, loop) {
        super();

        this.callback = callback;
        this.user_data = user_data;
        this.loop = loop;
        this.abortFn = undefined;

        const abortPromise = new Promise((_, reject) => {
          this.abortFn = reject;
        });
        const timeoutPromise = new Promise((resolve, _) => {
          setTimeout(() => {
            resolve();
          }, timeout_ms);
        });

        Promise.race([abortPromise, timeoutPromise])
            .then(() => {
              Module['dynCall'](
                  'iiii', this.callback,
                  [this.user_data, this.loop, IREE_STATUS_OK]);
              // TODO(scotttodd): handle the returned status (sticky failure
              //     state?) at least free the status so it doesn't leak
              delete scope.pendingOperations[operationId];
            })
            .catch(() => {
              Module['dynCall'](
                  'iiii', this.callback,
                  [this.user_data, this.loop, IREE_STATUS_ABORTED]);
            })
            .finally(() => {
              delete scope.pendingOperations[operationId];
            });
      }

      abort() {
        this.abortFn();
      }
    }

    // IREE_LOOP_COMMAND_WAIT_ONE
    // IREE_LOOP_COMMAND_WAIT_ANY
    // IREE_LOOP_COMMAND_WAIT_ALL
    class LoopCommandWaitPromise extends LoopCommand {
      constructor(
          scope, operationId, callback, user_data, timeout_ms, wait_promise,
          loop) {
        super();

        this.callback = callback;
        this.user_data = user_data;
        this.loop = loop;
        this.abortFn = undefined;

        // We're given a primary Promise to wait on, but we'll race it against
        // two other Promises:
        //   * timeoutPromise: rejected after a timeout
        //   * abortPromise : rejected if abort() is called
        // This way, whichever Promise settles first will decide the eventual
        // state and either issue the callback with IREE_STATUS_OK or
        // IREE_STATUS_ABORTED.
        const abortPromise = new Promise((_, reject) => {
          this.abortFn = reject;
        });
        const timeoutPromise = new Promise((_, reject) => {
          setTimeout(() => {
            reject();
          }, timeout_ms);
        });

        Promise.race([wait_promise, abortPromise, timeoutPromise])
            .then(() => {
              Module['dynCall'](
                  'iiii', this.callback,
                  [this.user_data, this.loop, IREE_STATUS_OK]);
              // TODO(scotttodd): handle the returned status (sticky failure
              //     state?) at least free the status so it doesn't leak
              delete scope.pendingOperations[operationId];
            })
            .catch(() => {
              Module['dynCall'](
                  'iiii', this.callback,
                  [this.user_data, this.loop, IREE_STATUS_ABORTED]);
            })
            .finally(() => {
              delete scope.pendingOperations[operationId];
            });
      }

      abort() {
        this.abortFn();
      }
    }

    class LoopEmscriptenScope {
      constructor() {
        // Note: start at 1, leaving 0 as a sentinel for uninitialized.
        this.nextOperationId = 1;

        // Dictionary of operationIds -> LoopCommands.
        this.pendingOperations = {};
      }

      destroy() {
        for (const id in this.pendingOperations) {
          const operation = this.pendingOperations[id];
          operation.abort();
          delete this.pendingOperations[id];
        }
      }

      command_call(callback, user_data, loop) {
        // TODO(scotttodd): assert not destroyed to avoid reentrant queueing?
        const operationId = this.nextOperationId++;
        this.pendingOperations[operationId] =
            new LoopCommandCall(this, operationId, callback, user_data, loop);
        return IREE_STATUS_OK;
      }

      command_wait_until(callback, user_data, timeout_ms, loop) {
        // TODO(scotttodd): assert not destroyed to avoid reentrant queueing?
        const operationId = this.nextOperationId++;
        this.pendingOperations[operationId] = new LoopCommandWaitUntil(
            this, operationId, callback, user_data, timeout_ms, loop);
        return IREE_STATUS_OK;
      }

      command_wait_one(callback, user_data, timeout_ms, wait_promise, loop) {
        // TODO(scotttodd): assert not destroyed to avoid reentrant queueing?
        const operationId = this.nextOperationId++;
        this.pendingOperations[operationId] = new LoopCommandWaitPromise(
            this, operationId, callback, user_data, timeout_ms, wait_promise,
            loop);
        return IREE_STATUS_OK;
      }

      command_wait_any(callback, user_data, timeout_ms, wait_promises, loop) {
        // TODO(scotttodd): assert not destroyed to avoid reentrant queueing?
        const operationId = this.nextOperationId++;
        const anyPromise = Promise.any(wait_promises);
        this.pendingOperations[operationId] = new LoopCommandWaitPromise(
            this, operationId, callback, user_data, timeout_ms, anyPromise,
            loop);
        return IREE_STATUS_OK;
      }

      command_wait_all(callback, user_data, timeout_ms, wait_promises, loop) {
        // TODO(scotttodd): assert not destroyed to avoid reentrant queueing?
        const operationId = this.nextOperationId++;
        const allPromise = Promise.all(wait_promises);
        this.pendingOperations[operationId] = new LoopCommandWaitPromise(
            this, operationId, callback, user_data, timeout_ms, allPromise,
            loop);
        return IREE_STATUS_OK;
      }
    }

    class LoopEmscripten {
      constructor() {
        // Note: start at 1, leaving 0 as a sentinel for uninitialized.
        this.nextScopeHandle = 1;

        // Dictionary of scopeHandles -> LoopEmscriptenScopes.
        this.scopes = {};
      }

      iree_loop_allocate_scope() {
        const scopeHandle = this.nextScopeHandle++;
        this.scopes[scopeHandle] = new LoopEmscriptenScope();
        return scopeHandle;
      }

      iree_loop_free_scope(scope_handle) {
        if (!(scope_handle in this.scopes)) return;

        const scope = this.scopes[scope_handle];
        scope.destroy();
        delete this.scopes[scope_handle];
      }

      iree_loop_command_call(scope_handle, callback, user_data, loop) {
        if (!(scope_handle in this.scopes)) return IREE_STATUS_OUT_OF_RANGE;

        const scope = this.scopes[scope_handle];
        return scope.command_call(callback, user_data, loop);
      }

      iree_loop_command_wait_until(
          scope_handle, callback, user_data, timeout_ms, loop) {
        if (!(scope_handle in this.scopes)) return IREE_STATUS_OUT_OF_RANGE;

        const scope = this.scopes[scope_handle];
        return scope.command_wait_until(
            callback, user_data, timeout_ms, wait_promise, loop);
      }

      iree_loop_command_wait_one(
          scope_handle, callback, user_data, timeout_ms, promise_handle, loop) {
        if (!(scope_handle in this.scopes)) return IREE_STATUS_OUT_OF_RANGE;

        const scope = this.scopes[scope_handle];
        const wait_promise = IreeWaitHandlePromise.getPromise(promise_handle);
        return scope.command_wait_one(
            callback, user_data, timeout_ms, wait_promise, loop);
      }

      iree_loop_command_wait_any(
          scope_handle, callback, user_data, timeout_ms, promise_handles_count,
          promise_handles, loop) {
        if (!(scope_handle in this.scopes)) return IREE_STATUS_OUT_OF_RANGE;

        const scope = this.scopes[scope_handle];
        const wait_promises = [];
        for (let i = 0; i < promise_handles_count; ++i) {
          const promise_handle = getValue(promise_handles + i * 4);
          wait_promises[i] = IreeWaitHandlePromise.getPromise(promise_handle);
        }
        return scope.command_wait_any(
            callback, user_data, timeout_ms, wait_promises, loop);
      }

      iree_loop_command_wait_all(
          scope_handle, callback, user_data, timeout_ms, promise_handles_count,
          promise_handles, loop) {
        if (!(scope_handle in this.scopes)) return IREE_STATUS_OUT_OF_RANGE;

        const scope = this.scopes[scope_handle];
        const wait_promises = [];
        for (let i = 0; i < promise_handles_count; ++i) {
          const promise_handle = getValue(promise_handles + i * 4);
          wait_promises[i] = IreeWaitHandlePromise.getPromise(promise_handle);
        }
        return scope.command_wait_all(
            callback, user_data, timeout_ms, wait_promises, loop);
      }
    }

    const instance = new LoopEmscripten();
    _iree_loop_allocate_scope =
        instance.iree_loop_allocate_scope.bind(instance);
    _iree_loop_free_scope = instance.iree_loop_free_scope.bind(instance);
    _iree_loop_command_call = instance.iree_loop_command_call.bind(instance);
    _iree_loop_command_wait_until =
        instance.iree_loop_command_wait_until.bind(instance);
    _iree_loop_command_wait_one =
        instance.iree_loop_command_wait_one.bind(instance);
    _iree_loop_command_wait_any =
        instance.iree_loop_command_wait_any.bind(instance);
    _iree_loop_command_wait_all =
        instance.iree_loop_command_wait_all.bind(instance);
  },
  $iree_loop_emscripten_support__deps:
      ['$dynCall', '$IreeWaitHandleEmscripten'],

  iree_loop_allocate_scope: function() {},
  iree_loop_allocate_scope__deps: ['$iree_loop_emscripten_support'],
  iree_loop_free_scope: function() {},
  iree_loop_free_scope__deps: ['$iree_loop_emscripten_support'],
  iree_loop_command_call: function() {},
  iree_loop_command_call__deps: ['$iree_loop_emscripten_support'],
  iree_loop_command_wait_until: function() {},
  iree_loop_command_wait_until__deps: ['$iree_loop_emscripten_support'],
  iree_loop_command_wait_one: function() {},
  iree_loop_command_wait_one__deps: ['$iree_loop_emscripten_support'],
  iree_loop_command_wait_any: function() {},
  iree_loop_command_wait_any__deps: ['$iree_loop_emscripten_support'],
  iree_loop_command_wait_all: function() {},
  iree_loop_command_wait_all__deps: ['$iree_loop_emscripten_support'],
}

mergeInto(LibraryManager.library, LibraryIreeLoopEmscripten);
