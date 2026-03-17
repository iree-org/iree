// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Type-erased handle table mapping uint32_t handles to void* objects.
//
// The WebGPU bridge uses handles (small integers) to reference JS-side GPU
// objects across the wasm boundary. Handle 0 is reserved as the null handle.
// Lookup is O(1) array indexing. Allocation is O(1) pop from a free stack
// (or bump the high-water mark). Deallocation is O(1) push to the free stack.
//
// The table grows dynamically when all slots are occupied and no free indices
// are available. After warmup (all GPU objects created), no allocations occur.
//
// On native builds (Dawn), the same handle table maps handles to WGPUBuffer,
// WGPUDevice, etc. pointers. On wasm builds, the C code passes opaque handle
// integers to JS imports — the JS-side HandleTable holds the actual objects.
// Either way, the HAL driver operates on uint32_t handles uniformly.
//
// Thread safety: none. The wasm binary is single-threaded
// (IREE_SYNCHRONIZATION_DISABLE_UNSAFE=1), and native Dawn usage is likewise
// single-threaded per device.

#ifndef IREE_HAL_DRIVERS_WEBGPU_HANDLE_TABLE_H_
#define IREE_HAL_DRIVERS_WEBGPU_HANDLE_TABLE_H_

#include <stdbool.h>
#include <stdint.h>

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Handle type for WebGPU objects. Handle 0 is the null handle.
typedef uint32_t iree_hal_webgpu_handle_t;

// Sentinel value for null/invalid handles.
#define IREE_HAL_WEBGPU_HANDLE_NULL ((iree_hal_webgpu_handle_t)0)

// Handle table mapping integer handles to opaque pointers.
typedef struct iree_hal_webgpu_handle_table_t {
  // Array of [capacity] pointers. Index 0 is always NULL (reserved).
  // NULL in a non-zero index means the slot is on the free stack.
  void** entries;
  // Stack of free indices (LIFO). Indices pushed here become available for
  // reuse on the next insert. Does not include indices above high_water.
  uint32_t* free_stack;
  // Current allocation of the entries and free_stack arrays.
  uint32_t capacity;
  // Number of occupied entries (not counting the reserved index 0).
  uint32_t count;
  // Next index to allocate when the free stack is empty.
  // Starts at 1 (index 0 is reserved). Ranges from 1 to capacity.
  uint32_t high_water;
  // Number of entries in the free stack.
  uint32_t free_count;
  // Allocator used for the entries and free_stack arrays.
  iree_allocator_t allocator;
} iree_hal_webgpu_handle_table_t;

// Initializes a handle table with room for |initial_capacity| entries.
// |initial_capacity| must be >= 1 (index 0 is reserved, so the effective
// capacity is initial_capacity - 1). The table grows automatically when full.
iree_status_t iree_hal_webgpu_handle_table_initialize(
    uint32_t initial_capacity, iree_allocator_t allocator,
    iree_hal_webgpu_handle_table_t* out_table);

// Deinitializes the handle table and frees the backing arrays.
// In debug builds, asserts that all handles have been removed (count == 0).
void iree_hal_webgpu_handle_table_deinitialize(
    iree_hal_webgpu_handle_table_t* table);

// Inserts |object| into the table and writes the assigned handle to
// |out_handle|. |object| must not be NULL.
// Returns IREE_STATUS_RESOURCE_EXHAUSTED if the table cannot grow (allocation
// failure). On success, the returned handle is always >= 1.
iree_status_t iree_hal_webgpu_handle_table_insert(
    iree_hal_webgpu_handle_table_t* table, void* object,
    iree_hal_webgpu_handle_t* out_handle);

// Returns the object for |handle|, or NULL if the handle is
// IREE_HAL_WEBGPU_HANDLE_NULL, out of range, or has been removed.
void* iree_hal_webgpu_handle_table_get(
    const iree_hal_webgpu_handle_table_t* table,
    iree_hal_webgpu_handle_t handle);

// Removes |handle| from the table and returns the object that was stored.
// The slot is returned to the free stack for reuse.
// In debug builds, asserts that the handle is valid and occupied.
void* iree_hal_webgpu_handle_table_remove(iree_hal_webgpu_handle_table_t* table,
                                          iree_hal_webgpu_handle_t handle);

// Returns the number of live entries in the table.
static inline uint32_t iree_hal_webgpu_handle_table_count(
    const iree_hal_webgpu_handle_table_t* table) {
  return table->count;
}

// Returns true if the table has no live entries.
static inline bool iree_hal_webgpu_handle_table_is_empty(
    const iree_hal_webgpu_handle_table_t* table) {
  return table->count == 0;
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_WEBGPU_HANDLE_TABLE_H_
