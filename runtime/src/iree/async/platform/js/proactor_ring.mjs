// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// SPSC completion ring on SharedArrayBuffer for the JS proactor.
//
// Connects the event host (producer) to the proactor worker (consumer). The
// event host writes completion entries when async JS operations complete (timer
// fired, WebGPU work done, etc.). The worker reads entries via the ring_drain
// wasm import during proactor poll().
//
// SharedArrayBuffer layout (all Int32):
//   [0] head    — next write position (producer increments after write)
//   [1] tail    — next read position (consumer increments after read)
//   [2] futex   — Atomics.wait/notify target for blocking poll
//   [3] reserved
//   [4..] entries — pairs of (token: int32, status_code: int32), 8 bytes each
//
// Capacity must be a power of 2 for efficient modular indexing. The head and
// tail wrap naturally via unsigned arithmetic (JS bitwise ops are 32-bit).

const HEAD = 0;
const TAIL = 1;
const FUTEX = 2;
const HEADER_INT32S = 4;

export class ProactorRing {
  // Wraps an existing SharedArrayBuffer as a ring with the given capacity.
  constructor(sharedBuffer, capacity) {
    if ((capacity & (capacity - 1)) !== 0) {
      throw new Error(`Ring capacity must be a power of 2, got ${capacity}`);
    }
    this.capacity = capacity;
    // Control region: head, tail, futex, reserved (4 Int32s = 16 bytes).
    this.control = new Int32Array(sharedBuffer, 0, HEADER_INT32S);
    // Entry region: capacity * 2 Int32s (token + status_code per entry).
    this.entries =
        new Int32Array(sharedBuffer, HEADER_INT32S * 4, capacity * 2);
  }

  // Creates a SharedArrayBuffer sized for the given ring capacity.
  static createSharedBuffer(capacity) {
    const bytes = HEADER_INT32S * 4 + capacity * 2 * 4;
    return new SharedArrayBuffer(bytes);
  }

  // Writes a completion entry (producer side, event host thread).
  // Throws if the ring is full — this indicates a capacity mismatch between
  // the ring and the token table and should never happen in correct operation.
  write(token, statusCode) {
    const head = Atomics.load(this.control, HEAD);
    const tail = Atomics.load(this.control, TAIL);
    if ((head - tail) >= this.capacity) {
      throw new Error(
          `Completion ring full (capacity=${this.capacity}). ` +
          `This is a bug: ring capacity must be >= token table capacity.`);
    }
    const index = (head & (this.capacity - 1)) * 2;
    this.entries[index] = token;
    this.entries[index + 1] = statusCode;
    Atomics.store(this.control, HEAD, head + 1);
    // Bump futex to wake any consumer blocked in wait(). The monotonic
    // increment ensures that a write between the consumer's futex load and
    // its Atomics.wait produces a value mismatch, preventing lost wakes.
    Atomics.add(this.control, FUTEX, 1);
    Atomics.notify(this.control, FUTEX);
  }

  // Copies up to |maxCount| entries from the ring into wasm linear memory
  // (consumer side, worker thread). Each entry occupies 8 bytes in the wasm
  // buffer: {int32 token, int32 status_code}. Returns the number of entries
  // copied.
  drain(wasmMemory, bufferPointer, maxCount) {
    // Fresh view each call — memory.grow() can detach the underlying buffer.
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

  // Blocks until the ring has data or |timeoutMs| expires (consumer side).
  // Returns 0 if data is available, 1 if timed out.
  wait(timeoutMs) {
    // Capture the current futex value BEFORE checking the ring. Any write
    // that lands after this load bumps the futex, guaranteeing that the
    // subsequent Atomics.wait sees a value mismatch and returns immediately
    // (no lost wake).
    const futexValue = Atomics.load(this.control, FUTEX);
    // Fast path: ring already has data.
    const head = Atomics.load(this.control, HEAD);
    const tail = Atomics.load(this.control, TAIL);
    if (head !== tail) return 0;
    // Block on the futex. Negative or zero timeout returns immediately.
    const result = Atomics.wait(
        this.control, FUTEX, futexValue, timeoutMs <= 0 ? 0 : timeoutMs);
    return result === 'timed-out' ? 1 : 0;
  }
}
