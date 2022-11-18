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

const LibraryLoopEmscripten = {
  $loop_emscripten_support__postset: 'loop_emscripten_support();',
  $loop_emscripten_support: function() {
    class LoopEmscripten {
      constructor() {
        // TODO(scotttodd): store state here
      }

      loop_command_call(callback, user_data, loop) {
        const IREE_STATUS_OK = 0;

        setTimeout(() => {
          const ret =
              Module['dynCall_iiii'](callback, user_data, loop, IREE_STATUS_OK);
          // TODO(scotttodd): handle the returned status (sticky failure state?)
        }, 0);

        return IREE_STATUS_OK;
      }
    }

    const instance = new LoopEmscripten();
    _loop_command_call = instance.loop_command_call.bind(instance);
  },

  loop_command_call: function() {},
  loop_command_call__deps: ['$loop_emscripten_support'],
}

mergeInto(LibraryManager.library, LibraryLoopEmscripten);
