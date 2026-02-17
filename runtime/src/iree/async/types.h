// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Header-only type definitions for embedding in other layers.
//
// This file provides types that can be embedded in HAL buffers, pool slabs, and
// other structures without creating a link dependency on iree/async/. Only
// inline helpers and type declarations are included here; no function
// implementations that would require linking.
//
// Usage:
//   #include "iree/async/types.h"
//   // Embed iree_async_buffer_registration_state_t in your buffer struct.
//   // Call the inline _initialize, _cleanup, _add, _remove, _find helpers.
//   // Full proactor registration APIs live in iree/async/proactor.h.

#ifndef IREE_ASYNC_TYPES_H_
#define IREE_ASYNC_TYPES_H_

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_async_proactor_t iree_async_proactor_t;
typedef struct iree_async_region_t iree_async_region_t;

//===----------------------------------------------------------------------===//
// Buffer registration types
//===----------------------------------------------------------------------===//

// Cleanup function for registration entries.
// Called when the owning buffer is destroyed or explicitly unregistered.
// MUST NOT FAIL. If cleanup would leak or corrupt, it must abort.
// Responsible for releasing the region reference and freeing entry memory.
typedef void (*iree_async_registration_cleanup_fn_t)(void* entry,
                                                     void* proactor);

// Buffer recycle callback for provided buffer rings.
// A single registration entry linking a buffer to one proactor.
// Allocated by the proactor during registration. Linked into the buffer's
// registration state list. Entry lifetime is tied to the buffer — when the
// buffer is destroyed, all entries are cleaned up via their cleanup_fn.
typedef struct iree_async_buffer_registration_entry_t {
  // Intrusive singly-linked list (buffer owns the list).
  struct iree_async_buffer_registration_entry_t* next;

  // Owning proactor (for matching during lookup/unregister).
  iree_async_proactor_t* proactor;

  // Cleanup callback (set by proactor, called on buffer destroy).
  iree_async_registration_cleanup_fn_t cleanup_fn;

  // The registered region covering this buffer's memory.
  // Entry holds a reference; cleanup_fn must release it.
  iree_async_region_t* region;
} iree_async_buffer_registration_entry_t;

// List anchor for buffer registration entries.
// Embedded in allocated buffer implementations. Holds the list of all proactors
// this buffer is registered with.
//
// Thread safety: Access must be serialized by caller. Typical patterns:
//   - Single-threaded registration during setup.
//   - Read-only access during steady-state operations.
//   - Cleanup only happens during buffer destroy (single owner).
typedef struct iree_async_buffer_registration_state_t {
  iree_async_buffer_registration_entry_t* head;
} iree_async_buffer_registration_state_t;

//===----------------------------------------------------------------------===//
// Inline helpers
//===----------------------------------------------------------------------===//

// Initializes registration state. Call once when the buffer is created.
static inline void iree_async_buffer_registration_state_initialize(
    iree_async_buffer_registration_state_t* state) {
  state->head = NULL;
}

// Deinitializes registration state. Call from buffer destroy, before freeing
// the buffer's memory. Walks the list and invokes each entry's cleanup_fn.
static inline void iree_async_buffer_registration_state_deinitialize(
    iree_async_buffer_registration_state_t* state) {
  iree_async_buffer_registration_entry_t* entry = state->head;
  while (entry) {
    iree_async_buffer_registration_entry_t* next = entry->next;
    entry->cleanup_fn(entry, entry->proactor);
    entry = next;
  }
  state->head = NULL;
}

// Returns true if no registrations exist.
static inline bool iree_async_buffer_registration_state_is_empty(
    const iree_async_buffer_registration_state_t* state) {
  return state->head == NULL;
}

// Adds an entry to the front of the list.
// Caller must ensure entry is fully initialized before calling.
static inline void iree_async_buffer_registration_state_add(
    iree_async_buffer_registration_state_t* state,
    iree_async_buffer_registration_entry_t* entry) {
  entry->next = state->head;
  state->head = entry;
}

// Removes an entry from the list.
// Called during explicit unregistration (before buffer destroy).
static inline void iree_async_buffer_registration_state_remove(
    iree_async_buffer_registration_state_t* state,
    iree_async_buffer_registration_entry_t* entry) {
  iree_async_buffer_registration_entry_t** current = &state->head;
  while (*current) {
    if (*current == entry) {
      *current = entry->next;
      entry->next = NULL;
      return;
    }
    current = &(*current)->next;
  }
}

// Finds the registration entry for a specific proactor.
// Returns NULL if the buffer is not registered with this proactor.
static inline iree_async_buffer_registration_entry_t*
iree_async_buffer_registration_state_find(
    const iree_async_buffer_registration_state_t* state,
    const iree_async_proactor_t* proactor) {
  iree_async_buffer_registration_entry_t* entry = state->head;
  while (entry) {
    if (entry->proactor == proactor) return entry;
    entry = entry->next;
  }
  return NULL;
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_ASYNC_TYPES_H_
