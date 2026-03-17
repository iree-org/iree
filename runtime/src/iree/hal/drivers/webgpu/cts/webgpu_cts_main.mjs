// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CTS entry point for the WebGPU HAL driver under Node.js + dawn.
//
// Inline-mode entry point (no workers, no Atomics.wait). Before wasm starts:
//   1. Import dawn and create GPUAdapter + GPUDevice.
//   2. Create companion imports (handle table, proactor, WebGPU bridge).
//   3. Insert the GPUDevice into the handle table.
//   4. Set context.preConfiguredDevice so the driver uses it directly.
//   5. Instantiate wasm and run the CTS gtest binary.
//
// The bundler appends this file after inlined companion code and the generated
// createWasmImports() function. The bundler also defines __IREE_WASM_BINARY
// with the .wasm filename.

import {readFileSync} from 'node:fs';
import {dirname, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {WASI} from 'node:wasi';

const scriptDirectory = dirname(fileURLToPath(import.meta.url));
const wasmPath = resolve(scriptDirectory, __IREE_WASM_BINARY);

// Obtain a WebGPU device from dawn via the 'webgpu' npm package.
// This ships prebuilt dawn.node native addons for all major platforms.
// Install with: npm install webgpu
let gpu;
try {
  const {create, globals} = await import('webgpu');
  // Populate globalThis with WebGPU constants (GPUBufferUsage, etc.).
  Object.assign(globalThis, globals);
  gpu = create([]);
} catch (error) {
  console.error(
      'Failed to import webgpu package: ' + error.message + '\n' +
      'Install with: npm install webgpu');
  process.exit(1);
}

const adapter = await gpu.requestAdapter();
if (!adapter) {
  console.error('WebGPU adapter not available (no GPU or dawn not configured)');
  process.exit(1);
}

const device = await adapter.requestDevice();
if (!device) {
  console.error('WebGPU device creation failed');
  process.exit(1);
}

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
  args: [__IREE_WASM_BINARY, ...process.argv.slice(2)],
  env: process.env,
  preopens,
});

// Build the wasm import context. Inline mode — no workers, no
// SharedArrayBuffer, no Atomics.wait. Completions are delivered through
// the proactor inline queue via context.complete (set by proactor imports).
const context = {
  memory: null,
  exports: null,
  inline: true,
  gpu,
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

// Insert the pre-created GPUDevice into the WebGPU handle table. The driver's
// create_device_by_id will find this via get_preconfigured_device().
const deviceHandle = context.insertWebGPUHandle(device);
context.preConfiguredDevice = deviceHandle;

// Compile and instantiate.
const wasmBytes = readFileSync(wasmPath);
const {instance} = await WebAssembly.instantiate(wasmBytes, imports);

// Wire up exported memory and exports for companions.
context.memory = instance.exports.memory;
context.exports = instance.exports;

// Run the CTS gtest binary. WASI.start() calls _start() and handles
// proc_exit internally.
let exitStatus = 0;
try {
  wasi.start(instance);
} catch (error) {
  if (process.exitCode != null) {
    exitStatus = process.exitCode;
    process.exitCode = undefined;
  } else {
    console.error(error);
    exitStatus = 1;
  }
}

// Dawn's GPU object keeps the Node.js event loop alive. Destroy the device
// to release it, then exit with the test status.
device.destroy();
process.exit(exitStatus);
