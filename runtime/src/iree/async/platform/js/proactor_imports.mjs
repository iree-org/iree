// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// JS companion for the iree_proactor wasm import module.
//
// Provides the six functions imported by the C proactor under
// __attribute__((import_module("iree_proactor"))). Used by the wasm binary
// bundler: the bundler inlines this file, wraps it in an IIFE, and calls
// createImports(context) with shared state from the host.
//
// Supports three execution modes determined by context fields:
//
//   Worker mode (context.ringBuffer provided):
//     Wasm runs in a Web Worker. The event host runs on the main thread.
//     ring_drain reads from a SharedArrayBuffer SPSC ring. poll_wait blocks
//     via Atomics.wait. timer_start sends postMessage to the event host.
//     timer_cancel uses the SAB cancel protocol.
//
//   Inline mode (context.inline === true):
//     Wasm runs on the main thread. No Worker, no SharedArrayBuffer.
//     ring_drain reads from a local JS array. poll_wait returns immediately
//     (cannot block the main thread). timer_start calls setTimeout directly.
//     timer_cancel calls clearTimeout. schedule_drain coalesces completions
//     via queueMicrotask.
//
//   Single-thread test mode (neither ringBuffer nor inline):
//     Used by wasm_test_main.mjs for build verification. A minimal local
//     SAB ring is created but no event host writes completions. timer_start
//     is a no-op. timer_cancel always reports cancelled.

// SPSC completion ring implementation. Inlined here (rather than imported)
// because the bundler strips ESM imports from companion files. The event host
// and test runner import the same class from proactor_ring.mjs.
const HEAD = 0;
const TAIL = 1;
const FUTEX = 2;
const HEADER_INT32S = 4;

class ProactorRing {
  constructor(sharedBuffer, capacity) {
    this.capacity = capacity;
    this.control = new Int32Array(sharedBuffer, 0, HEADER_INT32S);
    this.entries =
        new Int32Array(sharedBuffer, HEADER_INT32S * 4, capacity * 2);
  }

  drain(wasmMemory, bufferPointer, maxCount) {
    const wasmView =
        new Int32Array(wasmMemory.buffer, bufferPointer, maxCount * 2);
    let count = 0;
    while (count < maxCount) {
      const head = Atomics.load(this.control, HEAD);
      const tail = Atomics.load(this.control, TAIL);
      if (head === tail) break;
      const index = (tail & (this.capacity - 1)) * 2;
      wasmView[count * 2] = this.entries[index];
      wasmView[count * 2 + 1] = this.entries[index + 1];
      Atomics.store(this.control, TAIL, tail + 1);
      count++;
    }
    return count;
  }

  wait(timeoutMs) {
    const futexValue = Atomics.load(this.control, FUTEX);
    const head = Atomics.load(this.control, HEAD);
    const tail = Atomics.load(this.control, TAIL);
    if (head !== tail) return 0;
    const result = Atomics.wait(
        this.control, FUTEX, futexValue, timeoutMs <= 0 ? 0 : timeoutMs);
    return result === 'timed-out' ? 1 : 0;
  }
}

const CANCEL_REQUEST = 0;
const CANCEL_RESPONSE = 1;

// Creates the iree_proactor import implementations.
//
// Required context fields:
//   memory        — WebAssembly.Memory (set after instantiation)
//
// Worker mode context fields:
//   ringBuffer    — SharedArrayBuffer for the completion ring.
//   ringCapacity  — Ring capacity (power of 2).
//   cancelControl — Int32Array over a SharedArrayBuffer for cancel protocol.
//   postMessage   — function(message) to send to the event host thread.
//
// Inline mode context fields:
//   inline        — true to use inline (main-thread) mode.
//
// When neither ringBuffer nor inline is set, single-thread test mode is used.
export function createImports(context) {
  if (context.inline) {
    return createInlineImports(context);
  }
  return createWorkerImports(context);
}

// Inline mode: wasm runs on the main thread. Completions are buffered in a
// local JS array and drained synchronously during poll(). Timers use setTimeout
// directly. schedule_drain coalesces re-entry via queueMicrotask.
function createInlineImports(context) {
  // Local completion ring: plain JS array of {token, statusCode} objects.
  // No SharedArrayBuffer needed — everything is single-threaded.
  const completionQueue = [];

  // Active timers: token → setTimeout handle.
  const timers = new Map();

  // Coalesced drain scheduling. Multiple calls to scheduleDrain before the
  // microtask fires result in a single drain call.
  let drainPending = false;
  function scheduleDrain() {
    if (drainPending) return;
    drainPending = true;
    queueMicrotask(() => {
      drainPending = false;
      // Re-enter wasm to process completions. The wasm binary exports
      // proactor_drain() as the inline-mode re-entry point. If the export
      // is not available (e.g., CTS tests that don't export it), the drain
      // is deferred until the next poll() call.
      if (context.exports?.proactor_drain) {
        context.exports.proactor_drain();
      }
    });
  }

  // Expose the completion delivery function on context so other companion
  // modules (WebGPU, future network I/O) can push async completions into the
  // same queue. This is the production inline-mode completion bridge.
  context.complete = (token, statusCode) => {
    completionQueue.push({token, statusCode});
    scheduleDrain();
  };

  return {
    // Copies completions from the local queue into wasm linear memory.
    ring_drain(bufferPointer, capacity) {
      if (completionQueue.length === 0) return 0;
      const count = Math.min(completionQueue.length, capacity);
      // Fresh view each call — memory.grow() can detach the buffer.
      const wasmView =
          new Int32Array(context.memory.buffer, bufferPointer, count * 2);
      for (let i = 0; i < count; i++) {
        const entry = completionQueue[i];
        wasmView[i * 2] = entry.token;
        wasmView[i * 2 + 1] = entry.statusCode;
      }
      completionQueue.splice(0, count);
      return count;
    },

    // Never blocks in inline mode. The main thread cannot use Atomics.wait.
    poll_wait(deadlineNs) {
      return 1;  // Timed out immediately.
    },

    // Starts a timer via setTimeout. When the timer fires, the completion is
    // queued locally and a drain is scheduled.
    // deadlineNs is BigInt (wasm i64) from CLOCK_MONOTONIC.
    timer_start(token, deadlineNs) {
      // process.hrtime.bigint() uses CLOCK_MONOTONIC, matching the wasm
      // clock source (WASI clock_time_get with MONOTONIC).
      const nowNs = process.hrtime.bigint();
      const delayNs = deadlineNs - nowNs;
      const delayMs = Math.max(0, Number(delayNs) / 1_000_000);
      const timerId = setTimeout(() => {
        timers.delete(token);
        completionQueue.push({token, statusCode: 0});
        scheduleDrain();
      }, delayMs);
      timers.set(token, timerId);
    },

    // Cancels a timer synchronously. No cross-thread protocol needed.
    timer_cancel(token) {
      const timerId = timers.get(token);
      if (timerId !== undefined) {
        clearTimeout(timerId);
        timers.delete(token);
        return 1;  // Successfully cancelled.
      }
      return 0;  // Already fired or unknown.
    },

    // Schedules a drain to process pending completions.
    wake() {
      scheduleDrain();
    },

    // Schedules a drain. Used when NOPs are enqueued or events are signaled.
    schedule_drain() {
      scheduleDrain();
    },
  };
}

// Worker mode (and single-thread test mode): uses SharedArrayBuffer ring and
// Atomics for cross-thread communication.
function createWorkerImports(context) {
  // Create ring from provided shared buffer or a minimal local ring for
  // single-thread mode where no event host produces completions.
  const ringCapacity = context.ringCapacity || 16;
  let ringBuffer = context.ringBuffer;
  if (!ringBuffer) {
    const bytes = HEADER_INT32S * 4 + ringCapacity * 2 * 4;
    ringBuffer = new SharedArrayBuffer(bytes);
  }
  const ring = new ProactorRing(ringBuffer, ringCapacity);

  // Cancel control: shared Int32Array for synchronous cancel protocol.
  // In single-thread mode, create a dummy buffer (cancel always reports
  // "already fired" since there's no event host to cancel timers).
  const cancelControl =
      context.cancelControl || new Int32Array(new SharedArrayBuffer(8));

  return {
    // Copies completions from the shared ring into wasm linear memory.
    ring_drain(bufferPointer, capacity) {
      return ring.drain(context.memory, bufferPointer, capacity);
    },

    // Blocks until completions are available or deadline expires.
    // deadlineNs is BigInt (wasm i64) from CLOCK_MONOTONIC.
    // Uses process.hrtime.bigint() which shares CLOCK_MONOTONIC's epoch.
    poll_wait(deadlineNs) {
      const nowNs = process.hrtime.bigint();
      const remainingNs = deadlineNs - nowNs;
      const timeoutMs = Number(remainingNs) / 1_000_000;
      return ring.wait(timeoutMs);
    },

    // Posts a timer request to the event host.
    // deadlineNs is BigInt (wasm i64).
    // In single-thread mode (no postMessage), this is a no-op — the timer
    // will never fire, matching native stub behavior.
    timer_start(token, deadlineNs) {
      if (context.postMessage) {
        context.postMessage({type: 'timer_start', token, deadlineNs});
      }
    },

    // Synchronous cancel via the SharedArrayBuffer cancel protocol.
    // In worker mode, blocks until the event host responds.
    // In single-thread mode (no postMessage), returns 1 (always cancelled)
    // to match native stub behavior.
    timer_cancel(token) {
      if (!context.postMessage) {
        return 1;
      }
      Atomics.store(cancelControl, CANCEL_RESPONSE, 0);
      Atomics.store(cancelControl, CANCEL_REQUEST, token + 1);
      Atomics.notify(cancelControl, CANCEL_REQUEST);
      Atomics.wait(cancelControl, CANCEL_RESPONSE, 0);
      return Atomics.load(cancelControl, CANCEL_RESPONSE) === 1 ? 1 : 0;
    },

    // No-op in worker mode. The worker drives poll() directly.
    wake() {},

    // No-op in worker mode. NOPs are in the ready queue and drained during
    // the next poll() call on the same thread.
    schedule_drain() {},
  };
}
