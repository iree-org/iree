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
    // iree_status_t objects for common iree_status_code_t values.
    // Keep in sync with status.h.
    const IREE_STATUS_OK = 0;
    const IREE_STATUS_CODE_MASK = 0x1F;
    const IREE_STATUS_INVALID_ARGUMENT = 3 & IREE_STATUS_CODE_MASK;
    const IREE_STATUS_ABORTED = 10 & IREE_STATUS_CODE_MASK;
    const IREE_STATUS_UNIMPLEMENTED = 12 & IREE_STATUS_CODE_MASK;

    // iree_loop_command_e values, keep in sync with loop.h.
    const IREE_LOOP_COMMAND_CALL = 0;
    const IREE_LOOP_COMMAND_DISPATCH = 1;
    const IREE_LOOP_COMMAND_WAIT_UNTIL = 2;
    const IREE_LOOP_COMMAND_WAIT_ONE = 3;
    const IREE_LOOP_COMMAND_WAIT_ANY = 4;
    const IREE_LOOP_COMMAND_WAIT_ALL = 5;

    class LoopCommand {
      abort() {}
    }

    class LoopCommandCall extends LoopCommand {
      constructor(scope, operationId, callback, userData, loop) {
        super();

        this.callback = callback;
        this.userData = userData;
        this.loop = loop;

        this.timeoutId = setTimeout(() => {
          Module['dynCall'](
              'iiii', this.callback,
              [this.userData, this.loop, IREE_STATUS_OK]);
          // TODO(scotttodd): handle the returned status (sticky failure state?)
          //     at least free the status so it doesn't leak
          delete scope.pendingOperations[operationId];
        }, 0);
      }

      abort() {
        clearTimeout(this.timeoutId);
        Module['dynCall'](
            'iiii', this.callback,
            [this.userData, this.loop, IREE_STATUS_ABORTED]);
      }
    }

    class LoopCommandWaitUntil extends LoopCommand {
      constructor(scope, operationId, callback, userData, timeoutMs, loop) {
        super();

        this.callback = callback;
        this.userData = userData;
        this.loop = loop;
        this.abortFn = undefined;

        const abortPromise = new Promise((_, reject) => {
          this.abortFn = reject;
        });
        const timeoutPromise = new Promise((resolve, _) => {
          setTimeout(() => {
            resolve();
          }, timeoutMs);
        });

        Promise.race([abortPromise, timeoutPromise])
            .then(() => {
              Module['dynCall'](
                  'iiii', this.callback,
                  [this.userData, this.loop, IREE_STATUS_OK]);
              // TODO(scotttodd): handle the returned status (sticky failure
              //     state?) at least free the status so it doesn't leak
              delete scope.pendingOperations[operationId];
            })
            .catch(() => {
              Module['dynCall'](
                  'iiii', this.callback,
                  [this.userData, this.loop, IREE_STATUS_ABORTED]);
            })
            .finally(() => {
              delete scope.pendingOperations[operationId];
            });
      }

      abort() {
        this.abortFn();
      }
    }

    class LoopCommandWaitPromise extends LoopCommand {
      constructor(
          scope, operationId, callback, userData, timeoutMs, waitPromise,
          loop) {
        super();

        this.callback = callback;
        this.userData = userData;
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

        let racePromise;
        if (timeoutMs >= 0 && timeoutMs < 2147483647) {
          const timeoutPromise = new Promise((_, reject) => {
            setTimeout(() => {
              reject();
            }, timeoutMs);
          });
          racePromise =
              Promise.race([waitPromise, abortPromise, timeoutPromise]);
        } else {
          // "Infinite" timeout.
          racePromise = Promise.race([waitPromise, abortPromise]);
        }

        racePromise
            .then(() => {
              Module['dynCall'](
                  'iiii', this.callback,
                  [this.userData, this.loop, IREE_STATUS_OK]);
              // TODO(scotttodd): handle the returned status (sticky failure
              //     state?) at least free the status so it doesn't leak
              delete scope.pendingOperations[operationId];
            })
            .catch(() => {
              Module['dynCall'](
                  'iiii', this.callback,
                  [this.userData, this.loop, IREE_STATUS_ABORTED]);
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

      runCommand(command, callback, userData, timeoutMs, waitPromises, loop) {
        // TODO(scotttodd): assert not destroyed to avoid reentrant queueing?
        const operationId = this.nextOperationId++;

        switch (command) {
          case IREE_LOOP_COMMAND_CALL:
            this.pendingOperations[operationId] = new LoopCommandCall(
                this, operationId, callback, userData, loop);
            break;
          case IREE_LOOP_COMMAND_DISPATCH:
            return IREE_STATUS_UNIMPLEMENTED;
          case IREE_LOOP_COMMAND_WAIT_UNTIL:
            this.pendingOperations[operationId] = new LoopCommandWaitUntil(
                this, operationId, callback, userData, timeoutMs, loop);
            break;
          case IREE_LOOP_COMMAND_WAIT_ONE:
            this.pendingOperations[operationId] = new LoopCommandWaitPromise(
                this, operationId, callback, userData, timeoutMs,
                waitPromises[0], loop);
            break;
          case IREE_LOOP_COMMAND_WAIT_ANY:
            const anyPromise = Promise.any(waitPromises);
            this.pendingOperations[operationId] = new LoopCommandWaitPromise(
                this, operationId, callback, userData, timeoutMs, anyPromise,
                loop);
            break;
          case IREE_LOOP_COMMAND_WAIT_ALL:
            const allPromise = Promise.all(waitPromises);
            this.pendingOperations[operationId] = new LoopCommandWaitPromise(
                this, operationId, callback, userData, timeoutMs, allPromise,
                loop);
            break;
          default:
            return IREE_STATUS_UNIMPLEMENTED;
        }

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

      iree_loop_free_scope(scopeHandle) {
        if (!(scopeHandle in this.scopes)) return;

        const scope = this.scopes[scopeHandle];
        scope.destroy();
        delete this.scopes[scopeHandle];
      }

      iree_loop_command(
          scopeHandle, command, callback, userData, timeoutMs,
          promiseHandlesCount, promiseHandles, loop) {
        if (!(scopeHandle in this.scopes)) return IREE_STATUS_INVALID_ARGUMENT;
        const scope = this.scopes[scopeHandle];

        const waitPromises = [];
        for (let i = 0; i < promiseHandlesCount; ++i) {
          const promiseHandle = getValue(promiseHandles + i * 4);
          waitPromises[i] = IreeWaitHandleEmscripten.getPromise(promiseHandle);
        }

        return scope.runCommand(
            command, callback, userData, timeoutMs, waitPromises, loop);
      }
    }

    const instance = new LoopEmscripten();
    _iree_loop_allocate_scope =
        instance.iree_loop_allocate_scope.bind(instance);
    _iree_loop_free_scope = instance.iree_loop_free_scope.bind(instance);
    _iree_loop_command = instance.iree_loop_command.bind(instance);
  },
  $iree_loop_emscripten_support__deps:
      ['$dynCall', '$IreeWaitHandleEmscripten'],

  iree_loop_allocate_scope: function() {},
  iree_loop_allocate_scope__deps: ['$iree_loop_emscripten_support'],
  iree_loop_free_scope: function() {},
  iree_loop_free_scope__deps: ['$iree_loop_emscripten_support'],
  iree_loop_command: function() {},
  iree_loop_command__deps: ['$iree_loop_emscripten_support'],
}

mergeInto(LibraryManager.library, LibraryIreeLoopEmscripten);
