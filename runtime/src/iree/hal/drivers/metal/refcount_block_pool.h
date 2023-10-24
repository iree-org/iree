// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_METAL_REFCOUNT_BLOCK_POOL_H_
#define IREE_HAL_DRIVERS_METAL_REFCOUNT_BLOCK_POOL_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/base/internal/atomics.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// An arena block pool with reference counting for the Metal driver.
//
// In Metal we use command buffer completion handler to release resource sets
// tracking resources used in command buffers. It's not possible to guarantee
// the completion handler's timing; so we need to use reference counting to make
// sure we don't destroy the block pool prematurely.
typedef struct iree_hal_metal_arena_block_pool_t {
  iree_atomic_ref_count_t ref_count;
  iree_allocator_t host_allocator;
  iree_arena_block_pool_t block_pool;
} iree_hal_metal_arena_block_pool_t;

// Retains the given |pool| by increasing its reference count.
void iree_hal_metal_arena_block_pool_retain(
    iree_hal_metal_arena_block_pool_t* pool);

// Releases the given |pool| by decreasing its reference count.
//
// |pool| will be destroyed when the reference count is 0.
void iree_hal_metal_arena_block_pool_release(
    iree_hal_metal_arena_block_pool_t* pool);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_METAL_REFCOUNT_BLOCK_POOL_H_
