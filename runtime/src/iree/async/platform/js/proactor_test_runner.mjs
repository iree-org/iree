// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CLI test runner for JS proactor wasm binaries.
//
// Runs a bundled wasm binary in a Node.js Worker thread with the full
// iree_proactor import set wired up. The main thread acts as the event host:
// scheduling timers, writing completions to the shared ring, and handling
// cancel requests. The exit code from the wasm program (gtest returns 0 on
// pass, 1 on fail) propagates to this process.
//
// The bundle is produced by the wasm binary bundler, which inlines companion
// JS (proactor_imports.mjs) and the worker entry point
// (proactor_worker_main.mjs) into a single .mjs file. The bundler also defines
// __IREE_WASM_BINARY so the worker can locate the .wasm file.
//
// Usage:
//   node proactor_test_runner.mjs <path/to/bundle.mjs> [test args...]

import {resolve} from 'node:path';
import {Worker} from 'node:worker_threads';

import {ProactorEventHost} from './proactor_event_host.mjs';
import {ProactorRing} from './proactor_ring.mjs';

const bundlePath = process.argv[2];
if (!bundlePath) {
  console.error(
      'Usage: node proactor_test_runner.mjs <bundle.mjs> [test args...]');
  process.exit(1);
}
const testArgs = process.argv.slice(3);

// Ring capacity matches the default token table capacity in the C proactor.
const RING_CAPACITY = 256;

// Shared buffers for the completion ring and cancel protocol.
const ringBuffer = ProactorRing.createSharedBuffer(RING_CAPACITY);
const cancelBuffer = new SharedArrayBuffer(16);

// Start the event host (timer management, cancel listener).
const eventHost =
    new ProactorEventHost(ringBuffer, cancelBuffer, RING_CAPACITY);
eventHost.start();

// Start the worker with the bundled wasm binary.
const worker = new Worker(resolve(bundlePath), {
  workerData: {
    wasmArgs: testArgs,
    ringBuffer,
    cancelBuffer,
    ringCapacity: RING_CAPACITY,
  },
});

worker.on('message', (message) => {
  if (message.type === 'exit') {
    eventHost.stop();
    process.exit(message.status);
  } else {
    eventHost.handleMessage(message);
  }
});

worker.on('error', (error) => {
  console.error('Proactor worker error:', error);
  eventHost.stop();
  process.exit(1);
});

// Safety net: if the worker exits without sending an exit message (crash,
// unhandled exception), propagate a failure exit code.
worker.on('exit', (code) => {
  eventHost.stop();
  process.exit(code || 1);
});
