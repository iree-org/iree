// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fixed-capacity token table mapping integer tokens to operation pointers.
//
// The JS proactor assigns a unique token to each operation submitted to JS
// (timers, WebGPU operations, fetch, etc.). When JS delivers a completion,
// it returns the token alongside a status code. The token table resolves the
// token back to the operation pointer for callback dispatch.
//
// Token acquisition uses a monotonic counter: next_token++ mod capacity. If
// the target slot is occupied (previous operation at that index hasn't
// completed), acquisition fails with RESOURCE_EXHAUSTED. This is O(1) with
// no scanning and catches token leaks deterministically.
//
// Thread safety: none. The JS proactor's wasm binary is single-threaded
// (IREE_SYNCHRONIZATION_DISABLE_UNSAFE=1). All token table operations run
// on the same thread.

#ifndef IREE_ASYNC_PLATFORM_JS_TOKEN_TABLE_H_
#define IREE_ASYNC_PLATFORM_JS_TOKEN_TABLE_H_

#include <stdint.h>

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_async_operation_t iree_async_operation_t;

//===----------------------------------------------------------------------===//
// Completion entry
//===----------------------------------------------------------------------===//

// Wire format for a completion entry as written by JS and read by C via the
// ring_drain import. 8 bytes total, matching the TypedArray layout in JS.
typedef struct iree_async_js_completion_entry_t {
  // Token identifying the completed operation, as returned by
  // iree_async_js_token_table_acquire().
  uint32_t token;
  // Result status code (iree_status_code_t value). 0 = OK.
  int32_t status_code;
} iree_async_js_completion_entry_t;

//===----------------------------------------------------------------------===//
// Token table
//===----------------------------------------------------------------------===//

// Fixed-capacity table mapping tokens to operation pointers.
typedef struct iree_async_js_token_table_t {
  // Array of capacity pointers. NULL means the slot is free.
  iree_async_operation_t** entries;
  // Maximum number of concurrent operations (fixed at initialization).
  iree_host_size_t capacity;
  // Number of currently occupied slots.
  iree_host_size_t count;
  // Monotonic counter for token allocation. Wraps naturally at UINT32_MAX.
  uint32_t next_token;
  // Allocator used to allocate the entries array. Stored for deallocation.
  iree_allocator_t allocator;
} iree_async_js_token_table_t;

// Initializes a token table with the given maximum capacity.
// |capacity| must be > 0. The entries array is zero-initialized (all slots
// free).
iree_status_t iree_async_js_token_table_initialize(
    iree_host_size_t capacity, iree_allocator_t allocator,
    iree_async_js_token_table_t* out_table);

// Deinitializes the token table and frees the entries array.
// In debug builds, asserts that all slots are free (count == 0).
void iree_async_js_token_table_deinitialize(iree_async_js_token_table_t* table);

// Acquires a token for |operation| and writes it to |out_token|.
// Returns IREE_STATUS_RESOURCE_EXHAUSTED if the target slot is occupied.
iree_status_t iree_async_js_token_table_acquire(
    iree_async_js_token_table_t* table, iree_async_operation_t* operation,
    uint32_t* out_token);

// Returns the operation associated with |token|, or NULL if the slot is free.
iree_async_operation_t* iree_async_js_token_table_lookup(
    const iree_async_js_token_table_t* table, uint32_t token);

// Releases the slot for |token|, making it available for reuse.
// In debug builds, asserts that the slot is occupied.
void iree_async_js_token_table_release(iree_async_js_token_table_t* table,
                                       uint32_t token);

// Returns true if the token table has no allocated tokens.
static inline bool iree_async_js_token_table_is_empty(
    const iree_async_js_token_table_t* table) {
  return table->count == 0;
}

// Returns the number of currently allocated tokens.
static inline iree_host_size_t iree_async_js_token_table_count(
    const iree_async_js_token_table_t* table) {
  return table->count;
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_ASYNC_PLATFORM_JS_TOKEN_TABLE_H_
