// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Inline-mode entry point for JS proactor wasm binaries.
//
// Runs the wasm binary on the main thread without Workers or SharedArrayBuffer.
// The JS event loop serves as the completion source: timers fire via
// setTimeout, and completions are buffered in a local JS array until the
// proactor polls.
//
// This is the simplest deployment mode — no COOP/COEP headers required, no
// Worker setup, no SAB allocation. The tradeoff: poll() cannot block, so
// operations that require waiting for async completions (timers, I/O) only
// resolve when the wasm program yields control to the JS event loop.
//
// For CTS tests, this means NOP and lifecycle tests pass, while timer-dependent
// tests that require blocking poll are filtered out via tag filtering.
//
// The bundler resolves relative imports and inlines them into the bundle.
// createWasmImports() is available in module scope from the bundled companions.
// __IREE_WASM_BINARY is defined by the bundler with the .wasm filename.

import {readFileSync} from 'node:fs';
import {dirname, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {WASI} from 'node:wasi';

// Resolve the wasm binary from the bundler-defined filename.
const scriptDirectory = dirname(fileURLToPath(import.meta.url));
const wasmPath = resolve(scriptDirectory, __IREE_WASM_BINARY);
const testArgs = process.argv.slice(2);

// Build preopens for directories the test needs to access. Bazel sets
// TEST_TMPDIR for scratch space and XML_OUTPUT_FILE for gtest XML output.
const preopens = {};
if (process.env.TEST_TMPDIR) {
  preopens[process.env.TEST_TMPDIR] = process.env.TEST_TMPDIR;
}
if (process.env.XML_OUTPUT_FILE) {
  const xmlDirectory = dirname(process.env.XML_OUTPUT_FILE);
  preopens[xmlDirectory] = xmlDirectory;
}

// WASI provides the C runtime environment: stdio, filesystem, clocks.
const wasi = new WASI({
  version: 'preview1',
  args: [__IREE_WASM_BINARY, ...(testArgs || [])],
  env: process.env,
  preopens,
});

// Build the wasm import context for inline mode.
// memory and exports are set after instantiation.
const context = {
  memory: null,
  exports: null,
  inline: true,
};

// Start with WASI imports, merge companion imports on top.
const imports = wasi.getImportObject();
const companionImports = createWasmImports(context);
for (const [moduleName, moduleImports] of Object.entries(companionImports)) {
  if (imports[moduleName]) {
    Object.assign(imports[moduleName], moduleImports);
  } else {
    imports[moduleName] = moduleImports;
  }
}

// Compile and instantiate.
const wasmBytes = readFileSync(wasmPath);
const {instance} = await WebAssembly.instantiate(wasmBytes, imports);

// Wire up exported memory and exports for the companion imports.
context.memory = instance.exports.memory;
context.exports = instance.exports;

// Run the wasm program. WASI.start() calls _start() on the instance.
let exitStatus = 0;
try {
  wasi.start(instance);
} catch (error) {
  // WASI proc_exit throws to unwind the wasm call stack. process.exitCode
  // is already set by the WASI runtime. If it was not set, this is a
  // genuine error (not a proc_exit).
  if (process.exitCode != null) {
    exitStatus = process.exitCode;
    process.exitCode = undefined;
  } else {
    console.error(error);
    exitStatus = 1;
  }
}

process.exit(exitStatus);
