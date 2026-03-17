// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Event host for the JS proactor's worker mode.
//
// The event host runs on the main thread (or an I/O worker). It handles
// asynchronous JS operations on behalf of the wasm proactor worker: timer
// scheduling/cancellation, and (in future phases) WebGPU completions, network
// events, etc. When an operation completes, the event host writes a completion
// entry to the shared ring and the ring's Atomics.notify wakes the worker.
//
// Timer requests arrive from the worker via postMessage. Timer cancellation
// uses a synchronous SharedArrayBuffer protocol: the worker writes a cancel
// request and blocks (Atomics.wait), the event host processes it via
// Atomics.waitAsync and writes the result back.
//
// Cancel control SharedArrayBuffer layout (Int32Array, 4 slots = 16 bytes):
//   [0] request   — token+1 when cancel requested, 0 when idle
//   [1] response  — 0=pending, 1=cancelled, 2=already fired

import {ProactorRing} from './proactor_ring.mjs';

const CANCEL_REQUEST = 0;
const CANCEL_RESPONSE = 1;

export class ProactorEventHost {
  constructor(ringBuffer, cancelBuffer, ringCapacity) {
    this.ring = new ProactorRing(ringBuffer, ringCapacity);
    this.cancelControl = new Int32Array(cancelBuffer);
    this.timers = new Map();
    this.running = false;
  }

  // Starts the cancel listener. Call before starting the worker.
  start() {
    this.running = true;
    this._cancelListenerPromise = this._cancelListener();
  }

  // Stops the event host: cancels all pending timers and shuts down the
  // cancel listener.
  stop() {
    this.running = false;
    // Wake the cancel listener if it's blocked in waitAsync.
    Atomics.notify(this.cancelControl, CANCEL_REQUEST);
    for (const timerId of this.timers.values()) {
      clearTimeout(timerId);
    }
    this.timers.clear();
  }

  // Handles a message from the worker thread.
  handleMessage(message) {
    switch (message.type) {
      case 'timer_start':
        this._startTimer(message.token, message.deadlineNs);
        break;
    }
  }

  _startTimer(token, deadlineNs) {
    // deadlineNs is from CLOCK_MONOTONIC (via WASI clock_time_get).
    // process.hrtime.bigint() uses the same clock, so the delay is correct.
    // performance.now() has a different epoch (process start) and must not
    // be used here — the mismatch would cause delays off by system uptime.
    const nowNs = process.hrtime.bigint();
    const delayNs = deadlineNs - nowNs;
    const delayMs = Math.max(0, Number(delayNs) / 1_000_000);
    const timerId = setTimeout(() => {
      this.timers.delete(token);
      this.ring.write(token, 0);  // status_code = IREE_STATUS_OK
    }, delayMs);
    this.timers.set(token, timerId);
  }

  // Listens for cancel requests from the worker via the SharedArrayBuffer
  // cancel control region. Uses Atomics.waitAsync so the event loop stays
  // responsive for timer callbacks and other async work.
  async _cancelListener() {
    while (this.running) {
      // Block (async) until a cancel request arrives.
      const result = Atomics.waitAsync(this.cancelControl, CANCEL_REQUEST, 0);
      if (result.async) {
        await result.value;
      }
      if (!this.running) break;

      const tokenPlusOne = Atomics.load(this.cancelControl, CANCEL_REQUEST);
      if (tokenPlusOne === 0) continue;
      Atomics.store(this.cancelControl, CANCEL_REQUEST, 0);

      // Try to cancel the timer.
      const token = tokenPlusOne - 1;
      const timerId = this.timers.get(token);
      let cancelled;
      if (timerId !== undefined) {
        clearTimeout(timerId);
        this.timers.delete(token);
        cancelled = 1;  // Successfully cancelled before firing.
      } else {
        cancelled = 2;  // Already fired or unknown token.
      }

      // Write response and wake the worker.
      Atomics.store(this.cancelControl, CANCEL_RESPONSE, cancelled);
      Atomics.notify(this.cancelControl, CANCEL_RESPONSE);
    }
  }
}
