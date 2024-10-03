// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/device/support/ringbuffer.h"

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

#define iree_hal_amdgpu_device_ringbuffer_increment(value, capacity) \
  (((value) + 1u) % (capacity))

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_device_ringbuffer_uint64_t
//===----------------------------------------------------------------------===//

// Algorithm 8 in the paper.
void iree_hal_amdgpu_device_ringbuffer_uint64_enqueue(
    iree_hal_amdgpu_device_ringbuffer_uint64_t* IREE_AMDGPU_RESTRICT ringbuffer,
    uint64_t value) {
  // Retry until there is capacity available.
  bool enqueued = false;
  do {
    enqueued =
        iree_hal_amdgpu_device_ringbuffer_uint64_try_enqueue(ringbuffer, value);
    if (!enqueued) {
      // Yield for a few cycles - this isn't strictly required and may not even
      // help but makes me feel better.
      iree_amdgpu_yield();
    }
  } while (!enqueued);
}

// Algorithm 10 in the paper.
bool iree_hal_amdgpu_device_ringbuffer_uint64_try_enqueue(
    iree_hal_amdgpu_device_ringbuffer_uint64_t* IREE_AMDGPU_RESTRICT ringbuffer,
    uint64_t value) {
  // Fail immediately if the capacity has been reached.
  // Callers will retry if they want.
  uint32_t size = iree_amdgpu_scoped_atomic_load(
      &ringbuffer->size, iree_amdgpu_memory_order_seq_cst,
      iree_amdgpu_memory_scope_device);
  if (size == ringbuffer->capacity) {
    return false;  // capacity reached
  }

  // Reserve space for the entry by trying to bump the size. Note that this may
  // fail, in which case we need to keep retrying. Someone else comes in to
  // grab space before us and hits the capacity we need to bail.
  while (!iree_amdgpu_scoped_atomic_compare_exchange_weak(
      &ringbuffer->size, &size, size + 1u, iree_amdgpu_memory_order_seq_cst,
      iree_amdgpu_memory_order_seq_cst, iree_amdgpu_memory_scope_device)) {
    if (size == ringbuffer->capacity) {
      return false;  // race with another producer, capacity hit
    }
  }

  // Try to grab a write index and populate the value.
  // This is guaranteed to succeed in bounded time as we've reserved the space
  // above and know that there will be some empty slot after all writers and
  // readers do their thing. It's still possible to race and take quite a few
  // tries but only in high contention multi-producer/multi-consumer situations
  // (hopefully).
  while (true) {
    // Reserve a write index. Retries may be needed if multiple producers are
    // enqueuing simultaneously.
    uint32_t write_index = iree_amdgpu_scoped_atomic_load(
        &ringbuffer->write_index, iree_amdgpu_memory_order_seq_cst,
        iree_amdgpu_memory_scope_device);
    while (!iree_amdgpu_scoped_atomic_compare_exchange_weak(
        &ringbuffer->write_index, &write_index,
        iree_hal_amdgpu_device_ringbuffer_increment(write_index,
                                                    ringbuffer->capacity),
        iree_amdgpu_memory_order_seq_cst, iree_amdgpu_memory_order_seq_cst,
        iree_amdgpu_memory_scope_device));

    // Retry until the reserved entry is UNOCCUPIED. As soon as it is we'll mark
    // it as TRANSITIONING, assign the value, and then finalize it by marking it
    // as OCCUPIED. Note that as soon as we change from UNOCCUPIED no other
    // simultaneous writer can take the entry and once we change to OCCUPIED a
    // simultaneous reader may immediately take the value.
    iree_hal_amdgpu_device_ringbuffer_entry_state_t state =
        IREE_HAL_AMDGPU_DEVICE_RINGBUFFER_ENTRY_STATE_UNOCCUPIED;
    if (iree_amdgpu_scoped_atomic_compare_exchange_strong(
            &ringbuffer->entries[write_index].state, &state,
            IREE_HAL_AMDGPU_DEVICE_RINGBUFFER_ENTRY_STATE_TRANSITIONING,
            iree_amdgpu_memory_order_seq_cst, iree_amdgpu_memory_order_seq_cst,
            iree_amdgpu_memory_scope_device)) {
      // Store the value. Must happen before we mark the entry as OCCUPIED.
      ringbuffer->entries[write_index].value = value;

      // Mark the entry as OCCUPIED so that readers can acquire it.
      iree_amdgpu_scoped_atomic_store(
          &ringbuffer->entries[write_index].state,
          IREE_HAL_AMDGPU_DEVICE_RINGBUFFER_ENTRY_STATE_OCCUPIED,
          iree_amdgpu_memory_order_seq_cst, iree_amdgpu_memory_scope_device);

      return true;
    }
  }
}

// Algorithm 8 in the paper (just changed to a pop).
uint64_t iree_hal_amdgpu_device_ringbuffer_uint64_dequeue(
    iree_hal_amdgpu_device_ringbuffer_uint64_t* IREE_AMDGPU_RESTRICT
        ringbuffer) {
  // Try dequeuing until there is an entry to return.
  iree_hal_amdgpu_device_ringbuffer_uint64_result_t result;
  do {
    result = iree_hal_amdgpu_device_ringbuffer_uint64_try_dequeue(ringbuffer);
    if (!result.has_value) {
      // Yield for a few cycles - this isn't strictly required and may not even
      // help but makes me feel better.
      iree_amdgpu_yield();
    }
  } while (!result.has_value);
  return result.value;
}

// Algorithm 11 in the paper.
iree_hal_amdgpu_device_ringbuffer_uint64_result_t
iree_hal_amdgpu_device_ringbuffer_uint64_try_dequeue(
    iree_hal_amdgpu_device_ringbuffer_uint64_t* IREE_AMDGPU_RESTRICT
        ringbuffer) {
  iree_hal_amdgpu_device_ringbuffer_uint64_result_t result = {0, false};

  // Fail immediately if the ringbuffer is empty.
  // Callers will retry if they want.
  uint32_t size = iree_amdgpu_scoped_atomic_load(
      &ringbuffer->size, iree_amdgpu_memory_order_seq_cst,
      iree_amdgpu_memory_scope_device);
  if (size == 0) {
    return result;
  }

  // Reserve an entry to dequeue by decrementing the size.
  // Note that multiple consumers may be trying to dequeue at the same time and
  // we have to check if they take the last entry.
  while (!iree_amdgpu_scoped_atomic_compare_exchange_weak(
      &ringbuffer->size, &size, size - 1u, iree_amdgpu_memory_order_seq_cst,
      iree_amdgpu_memory_order_seq_cst, iree_amdgpu_memory_scope_device)) {
    if (size == 0) {
      return result;  // race with another consumer, ringbuffer empty
    }
  }

  // Try to grab a read index and take the value.
  // This is guaranteed to succeed in bounded time as we've reserved the space
  // above and know that there will be some slot with a valid value after all
  // writers and readers do their thing. It's still possible to race and take
  // quite a few tries but only in high contention multi-producer/multi-consumer
  // situations (hopefully).
  while (true) {
    // Reserve a read index. Retries may be needed if multiple consumers are
    // dequeuing simultaneously.
    uint32_t read_index = iree_amdgpu_scoped_atomic_load(
        &ringbuffer->read_index, iree_amdgpu_memory_order_seq_cst,
        iree_amdgpu_memory_scope_device);
    while (!iree_amdgpu_scoped_atomic_compare_exchange_weak(
        &ringbuffer->read_index, &read_index,
        iree_hal_amdgpu_device_ringbuffer_increment(read_index,
                                                    ringbuffer->capacity),
        iree_amdgpu_memory_order_seq_cst, iree_amdgpu_memory_order_seq_cst,
        iree_amdgpu_memory_scope_device));

    // Retry until the entry is OCCUPIED. A writer may have taken the slot and
    // be actively populating it after we take the read index.
    iree_hal_amdgpu_device_ringbuffer_entry_state_t state =
        IREE_HAL_AMDGPU_DEVICE_RINGBUFFER_ENTRY_STATE_OCCUPIED;
    if (iree_amdgpu_scoped_atomic_compare_exchange_strong(
            &ringbuffer->entries[read_index].state, &state,
            IREE_HAL_AMDGPU_DEVICE_RINGBUFFER_ENTRY_STATE_TRANSITIONING,
            iree_amdgpu_memory_order_seq_cst, iree_amdgpu_memory_order_seq_cst,
            iree_amdgpu_memory_scope_device)) {
      // Take value from the entry and restore it to 0. Clearing isn't needed
      // but does make debugging easier.
      result.value = ringbuffer->entries[read_index].value;
      ringbuffer->entries[read_index].value = 0;

      // Mark the entry as UNOCCUPIED so that writers can take it again.
      iree_amdgpu_scoped_atomic_store(
          &ringbuffer->entries[read_index].state,
          IREE_HAL_AMDGPU_DEVICE_RINGBUFFER_ENTRY_STATE_UNOCCUPIED,
          iree_amdgpu_memory_order_seq_cst, iree_amdgpu_memory_scope_device);

      result.has_value = true;
      return result;
    }
  }
}
