// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Internal header for JS proactor implementation.
//
// The JS proactor uses the JavaScript event loop as its kernel. Completions
// originate from JS (setTimeout, WebGPU, fetch) and are delivered to C through
// the ring_drain import. The wasm binary is single-threaded
// (IREE_SYNCHRONIZATION_DISABLE_UNSAFE=1), so no atomics are needed for
// internal state.
//
// Two execution modes share this single C implementation:
//   - Worker mode: wasm runs in a Web Worker. poll() blocks via poll_wait
//     import (Atomics.wait). Completions arrive asynchronously.
//   - Inline mode: wasm runs on the main thread. poll() is non-blocking.
//     Completions are delivered via JS calling proactor_drain() directly.
//
// External users should use the public proactor API (iree/async/proactor.h).

#ifndef IREE_ASYNC_PLATFORM_JS_PROACTOR_H_
#define IREE_ASYNC_PLATFORM_JS_PROACTOR_H_

#include "iree/async/platform/js/token_table.h"
#include "iree/async/proactor.h"
#include "iree/async/util/sequence_emulation.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Default token table capacity if max_concurrent_operations is not specified.
#define IREE_ASYNC_JS_DEFAULT_TOKEN_TABLE_CAPACITY 256

// Default completion buffer capacity (entries per ring_drain call).
#define IREE_ASYNC_JS_DEFAULT_COMPLETION_BUFFER_CAPACITY 64

// Proactor-private flags stored in operation->internal_flags.
enum iree_async_js_operation_internal_flags_e {
  IREE_ASYNC_JS_OPERATION_INTERNAL_FLAG_NONE = 0u,
  IREE_ASYNC_JS_OPERATION_INTERNAL_FLAG_CANCELLED = 1u << 0,
};

// JS proactor state. Single-threaded — no synchronization needed.
typedef struct iree_async_proactor_js_t {
  // Must be first for safe casting.
  iree_async_proactor_t base;

  // Token table mapping tokens to in-flight operations.
  iree_async_js_token_table_t token_table;

  // Ready queue for operations that complete immediately (NOPs, expired
  // timers). Intrusive singly-linked list via operation->next. Drained during
  // poll before checking the ring.
  iree_async_operation_t* ready_head;
  iree_async_operation_t* ready_tail;

  // Buffer for ring_drain results. Points into trailing allocation data.
  iree_async_js_completion_entry_t* completion_buffer;
  iree_host_size_t completion_buffer_capacity;

  // Sequence emulator for IREE_ASYNC_OPERATION_TYPE_SEQUENCE support.
  // Embedded (zero allocation). Uses iree_async_proactor_submit_one as the
  // submit function, which re-enters through the vtable submit path.
  iree_async_sequence_emulator_t sequence_emulator;

  // Stored capabilities (masked with options.allowed_capabilities at create).
  iree_async_proactor_capabilities_t capabilities;
} iree_async_proactor_js_t;

static inline iree_async_proactor_js_t* iree_async_proactor_js_cast(
    iree_async_proactor_t* proactor) {
  return (iree_async_proactor_js_t*)proactor;
}

// Creates a JS proactor.
iree_status_t iree_async_proactor_create_js(
    iree_async_proactor_options_t options, iree_allocator_t allocator,
    iree_async_proactor_t** out_proactor);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_ASYNC_PLATFORM_JS_PROACTOR_H_
