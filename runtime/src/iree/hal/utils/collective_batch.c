// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/utils/collective_batch.h"

#include "iree/base/tracing.h"

//===----------------------------------------------------------------------===//
// Collective batching utility
//===----------------------------------------------------------------------===//

#define IREE_HAL_COLLECTIVE_BATCH_INITIAL_CAPACITY 16

IREE_API_EXPORT void iree_hal_collective_batch_initialize(
    iree_arena_allocator_t* arena, iree_hal_resource_set_t* resource_set,
    iree_hal_collective_batch_t* out_batch) {
  out_batch->arena = arena;
  out_batch->resource_set = resource_set;
  out_batch->capacity = 0;
  out_batch->count = 0;
  out_batch->entries = NULL;
}

IREE_API_EXPORT void iree_hal_collective_batch_deinitialize(
    iree_hal_collective_batch_t* batch) {
  // Since we are just allocating from the arena we don't need to do anything
  // but clear our pointers for debugging clarity.
  batch->capacity = 0;
  batch->count = 0;
  batch->entries = NULL;
}

IREE_API_EXPORT bool iree_hal_collective_batch_is_empty(
    const iree_hal_collective_batch_t* batch) {
  return batch->count == 0;
}

IREE_API_EXPORT void iree_hal_collective_batch_clear(
    iree_hal_collective_batch_t* batch) {
  // Reset the count to zero but keep the arena storage for reuse.
  // We could memset the contents if we wanted to make debugging easier as ASAN
  // won't be able to help us but it'd probably be better to use ASAN hooks to
  // mark the memory as invalid instead.
  batch->count = 0;
}

// Grows the storage of the |batch| by 2x by slicing off new memory from the
// arena and copying over the existing contents.
static iree_status_t iree_hal_collective_batch_grow(
    iree_hal_collective_batch_t* batch) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Calculate new capacity. Note that we start empty.
  iree_host_size_t new_capacity =
      batch->capacity == 0 ? IREE_HAL_COLLECTIVE_BATCH_INITIAL_CAPACITY
                           : batch->capacity * 2;
  IREE_TRACE_ZONE_APPEND_VALUE(z0, new_capacity);

  // Allocate new storage - this may fail if the system (or block pool) is over
  // capacity.
  iree_hal_collective_batch_entry_t* new_entries = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_arena_allocate(batch->arena, new_capacity * sizeof(*batch->entries),
                          (void**)&new_entries));

  // Copy over existing items. We let the old entry list go as it'll eventually
  // be cleaned up when the arena is reset.
  memcpy(new_entries, batch->entries, batch->count * sizeof(*batch->entries));
  batch->capacity = new_capacity;
  batch->entries = new_entries;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_hal_collective_batch_append(
    iree_hal_collective_batch_t* batch, iree_hal_channel_t* channel,
    iree_hal_collective_op_t op, uint32_t param,
    iree_hal_buffer_binding_t send_binding,
    iree_hal_buffer_binding_t recv_binding, iree_device_size_t element_count) {
  // Grow the entry storage if required.
  if (batch->count + 1 > batch->capacity) {
    IREE_RETURN_IF_ERROR(iree_hal_collective_batch_grow(batch));
  }

  // Insert resources into the resource set to keep them live.
  iree_host_size_t resource_count = 0;
  void* resources[3] = {NULL};
  resources[resource_count++] = channel;
  if (send_binding.buffer) {
    resources[resource_count++] = send_binding.buffer;
  }
  if (recv_binding.buffer) {
    resources[resource_count++] = recv_binding.buffer;
  }
  IREE_RETURN_IF_ERROR(iree_hal_resource_set_insert(batch->resource_set,
                                                    resource_count, resources));

  // Append entry to the list.
  batch->entries[batch->count++] = (iree_hal_collective_batch_entry_t){
      .channel = channel,
      .op = op,
      .param = param,
      .send_binding = send_binding,
      .recv_binding = recv_binding,
      .element_count = element_count,
  };

  return iree_ok_status();
}
