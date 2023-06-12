// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_UTILS_COLLECTIVE_BATCH_H_
#define IREE_HAL_UTILS_COLLECTIVE_BATCH_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/hal/api.h"
#include "iree/hal/utils/resource_set.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Collective batching utility
//===----------------------------------------------------------------------===//

// Recorded collective operation in a batch.
// The specified channel and binding buffers will be retained by the resource
// set for the lifetime of the parent command buffer.
typedef struct {
  iree_hal_channel_t* channel;
  iree_hal_collective_op_t op;
  uint32_t param;
  iree_hal_buffer_binding_t send_binding;
  iree_hal_buffer_binding_t recv_binding;
  iree_device_size_t element_count;
} iree_hal_collective_batch_entry_t;

// Builds batches of collective operations for grouped submission.
// This is to be embedded in command buffer implementations and used to
// incrementally build batches of collective operations that can be submitted to
// implementations as atomic operations. The compiler is _supposed_ to emit
// collectives within a barrier scope though that's not verified by the API
// today.
//
// Referenced resources, such as channels and buffers, are retained on the
// resource set owned by the command buffer. This allows for async submissions
// to backing implementations to remain valid even if the code submitting the
// command buffers may drop their reference while it is in-flight.
typedef struct {
  // Arena used for scratch allocations used during batch construction.
  // This is owned by the parent of the collective batch and the lifetime of the
  // arena contents is controlled by the parent.
  iree_arena_allocator_t* arena;

  // Resource set that submitted channels and buffers will be retained in.
  iree_hal_resource_set_t* resource_set;

  // Growable list of accumulated operations (starts empty).
  // We could use a linked list into arena storage but we don't need to persist
  // the contents beyond a single flush. Instead we slice out some storage as
  // needed and grow by slicing off more and copying over the existing contents.
  // This should stabilized to the maximum batch size pretty fast with minimal
  // command buffer overhead. If we notice people doing counts following the
  // fibonacci sequence we could rework things but in average usage we expect
  // 1-16 entries on average.
  iree_host_size_t capacity;
  iree_host_size_t count;
  iree_hal_collective_batch_entry_t* entries;
} iree_hal_collective_batch_t;

// Initializes |out_batch| for use using |arena| for any transient allocations
// required. All resources used will be inserted into |resource_set|.
IREE_API_EXPORT void iree_hal_collective_batch_initialize(
    iree_arena_allocator_t* arena, iree_hal_resource_set_t* resource_set,
    iree_hal_collective_batch_t* out_batch);

// Deinitializes |batch| and releases any allocated memory.
IREE_API_EXPORT void iree_hal_collective_batch_deinitialize(
    iree_hal_collective_batch_t* batch);

// Returns true if the batch is empty.
IREE_API_EXPORT bool iree_hal_collective_batch_is_empty(
    const iree_hal_collective_batch_t* batch);

// Clears the collective batch and discards batches while reusing the same
// storage. Expects that the arena remains valid.
IREE_API_EXPORT void iree_hal_collective_batch_clear(
    iree_hal_collective_batch_t* batch);

// Appends a collective operation to the batch.
// Referenced resources will be retained.
IREE_API_EXPORT iree_status_t iree_hal_collective_batch_append(
    iree_hal_collective_batch_t* batch, iree_hal_channel_t* channel,
    iree_hal_collective_op_t op, uint32_t param,
    iree_hal_buffer_binding_t send_binding,
    iree_hal_buffer_binding_t recv_binding, iree_device_size_t element_count);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_UTILS_COLLECTIVE_BATCH_H_
