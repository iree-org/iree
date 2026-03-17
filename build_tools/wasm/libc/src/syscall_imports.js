// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// JS implementations of iree_syscall wasm imports.
//
// These functions are provided to the wasm module via the import object when
// instantiating. They implement the minimal OS-level interface that the
// freestanding libc needs: console output, monotonic clock, program exit,
// and assertion failure reporting.
//
// Usage (standalone):
//   import { createImports } from './syscall_imports.js';
//   const imports = { iree_syscall: createImports(context) };
//
// Usage (via bundler — the typical path):
//   The wasm binary bundler inlines this file and wires it up automatically.
//   The entry point calls createWasmImports(context) which delegates here.

// Creates the iree_syscall import implementations.
//
// |context.memory| must be a WebAssembly.Memory instance. The function reads
// context.memory.buffer on each use (not cached) because memory.grow can
// detach the underlying ArrayBuffer.
export function createImports(context) {
  const getMemory = () => context.memory;
  const decoder = new TextDecoder();

  return {
    // write(fd, buffer_ptr, length) -> bytes_written
    write(fd, pointer, length) {
      const bytes = new Uint8Array(getMemory().buffer, pointer, length);
      const text = decoder.decode(bytes);
      if (fd === 2) {
        console.error(text);
      } else {
        // fd 1 (stdout) or any other fd — write to console.log.
        // Avoid process.stdout.write to keep this browser-compatible.
        console.log(text);
      }
      return length;
    },

    // clock_now_ns() -> int64 nanoseconds (monotonic)
    //
    // Returns a BigInt because wasm i64 values are represented as BigInt
    // in the JS-wasm boundary.
    clock_now_ns() {
      // performance.now() returns milliseconds as a float64.
      // Resolution: ~5us when cross-origin isolated, ~100us-1ms otherwise.
      return BigInt(Math.round(performance.now() * 1e6));
    },

    // exit(status)
    //
    // If context.exit is provided, it is called instead of process.exit.
    // This is necessary in Worker threads where process.exit would terminate
    // the entire Node.js process. The Worker entry point sets context.exit
    // to throw a sentinel that unwinds the wasm call stack.
    exit(status) {
      if (context.exit) {
        context.exit(status);
      } else if (typeof process !== 'undefined' && process.exit) {
        process.exit(status);
      }
      // In browser context (or if context.exit returns), throwing stops
      // execution. The host page can catch this to display an error.
      throw new Error(`IREE wasm module exited with status ${status}`);
    },

    // assert_fail(expression_ptr, file_ptr, line, function_ptr)
    assert_fail(expressionPointer, filePointer, line, functionPointer) {
      const memory = getMemory();
      const readString = (pointer) => {
        const bytes = new Uint8Array(memory.buffer, pointer);
        const end = bytes.indexOf(0);
        return decoder.decode(bytes.subarray(0, end === -1 ? 256 : end));
      };
      const expression = readString(expressionPointer);
      const file = readString(filePointer);
      const func = readString(functionPointer);
      console.error(
          `Assertion failed: ${expression}\n  at ${func} (${file}:${line})`);
    },
  };
}
