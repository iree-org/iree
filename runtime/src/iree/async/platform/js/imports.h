// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Wasm import declarations for the JS proactor.
//
// When compiling for wasm32, these are declared as wasm imports using
// __attribute__((import_module, import_name)). The JS host (worker or inline)
// provides the implementations.
//
// When compiling for native (unit tests, host tooling), static inline stubs
// provide safe defaults: ring_drain returns 0, poll_wait returns 1 (timed out),
// timer_start/timer_cancel/wake/schedule_drain are no-ops. This allows the
// proactor's submit/poll logic to be tested natively without JS.

#ifndef IREE_ASYNC_PLATFORM_JS_IMPORTS_H_
#define IREE_ASYNC_PLATFORM_JS_IMPORTS_H_

#include "iree/async/platform/js/token_table.h"
#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#if defined(IREE_PLATFORM_WASM)

// Copies up to |capacity| completion entries from the JS ring into the wasm
// buffer at |buffer|. Each entry is 8 bytes: {uint32_t token, int32_t status}.
// Returns the number of entries copied (0 if ring is empty).
__attribute__((import_module("iree_proactor"),
               import_name("ring_drain"))) extern uint32_t
iree_async_js_import_ring_drain(iree_async_js_completion_entry_t* buffer,
                                uint32_t capacity);

// Blocks until completions are available or timeout expires.
// Worker mode: Atomics.wait on ring write_index.
// Inline mode: returns 1 immediately (never blocks).
// Returns: 0 = data available, 1 = timed out.
__attribute__((import_module("iree_proactor"),
               import_name("poll_wait"))) extern uint32_t
iree_async_js_import_poll_wait(int64_t timeout_ns);

// Starts a timer in JS. When the deadline is reached, JS writes a completion
// entry {token, OK} to the ring and wakes the wasm thread.
__attribute__((import_module("iree_proactor"),
               import_name("timer_start"))) extern void
iree_async_js_import_timer_start(uint32_t token, int64_t deadline_ns);

// Cancels a previously started timer. Returns 1 if the timer was cancelled
// before firing, 0 if it had already fired.
__attribute__((import_module("iree_proactor"),
               import_name("timer_cancel"))) extern uint32_t
iree_async_js_import_timer_cancel(uint32_t token);

// Wakes the poll thread.
// Worker mode: writes wake sentinel to ring + Atomics.notify.
// Inline mode: schedules drain via queueMicrotask.
__attribute__((import_module("iree_proactor"), import_name("wake"))) extern void
iree_async_js_import_wake(void);

// Schedules a proactor_drain call via queueMicrotask (inline mode).
// Coalesced: multiple calls before the microtask fires result in one drain.
__attribute__((import_module("iree_proactor"),
               import_name("schedule_drain"))) extern void
iree_async_js_import_schedule_drain(void);

#else  // !IREE_PLATFORM_WASM — native stubs for unit tests

static inline uint32_t iree_async_js_import_ring_drain(
    iree_async_js_completion_entry_t* buffer, uint32_t capacity) {
  (void)buffer;
  (void)capacity;
  return 0;  // No completions in native tests.
}

static inline uint32_t iree_async_js_import_poll_wait(int64_t timeout_ns) {
  (void)timeout_ns;
  return 1;  // Timed out (never blocks in native tests).
}

static inline void iree_async_js_import_timer_start(uint32_t token,
                                                    int64_t deadline_ns) {
  (void)token;
  (void)deadline_ns;
}

static inline uint32_t iree_async_js_import_timer_cancel(uint32_t token) {
  (void)token;
  return 1;  // Always report cancelled in native tests.
}

static inline void iree_async_js_import_wake(void) {}

static inline void iree_async_js_import_schedule_drain(void) {}

#endif  // IREE_PLATFORM_WASM

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_ASYNC_PLATFORM_JS_IMPORTS_H_
