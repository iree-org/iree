// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Entry point for the bundler end-to-end test.
//
// This file is concatenated into the bundled .mjs by the wasm binary bundler.
// At runtime, createWasmImports(context) and __IREE_WASM_BINARY are already
// defined in the same module scope (injected by the bundler).

import {readFile} from 'node:fs/promises';
import {dirname, join} from 'node:path';
import {fileURLToPath} from 'node:url';

const scriptDirectory = dirname(fileURLToPath(import.meta.url));
const wasmPath = join(scriptDirectory, __IREE_WASM_BINARY);

const wasmBytes = await readFile(wasmPath);
const memory = new WebAssembly.Memory({initial: 2, maximum: 256, shared: true});
const imports = {
  // wasm-ld places imported memory under the 'env' module by default.
  env: {memory},
  // Companion modules provide the application-level imports.
  ...createWasmImports({memory}),
};
const {instance} = await WebAssembly.instantiate(wasmBytes, imports);

const result = instance.exports.run_test();
process.exit(result);
