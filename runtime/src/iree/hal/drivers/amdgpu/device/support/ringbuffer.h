// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_DEVICE_SUPPORT_RINGBUFFER_H_
#define IREE_HAL_DRIVERS_AMDGPU_DEVICE_SUPPORT_RINGBUFFER_H_

#include "iree/hal/drivers/amdgpu/device/support/common.h"

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_device_ringbuffer_uint64_t
//===----------------------------------------------------------------------===//

// Defines the state of an entry in the ringbuffer.
typedef uint32_t iree_hal_amdgpu_device_ringbuffer_entry_state_t;
enum iree_hal_amdgpu_device_ringbuffer_entry_state_e {
  // Entry is unoccupied (value is invalid).
  IREE_HAL_AMDGPU_DEVICE_RINGBUFFER_ENTRY_STATE_UNOCCUPIED = 0,
  // Entry is transitioning between unoccupied and occupied.
  IREE_HAL_AMDGPU_DEVICE_RINGBUFFER_ENTRY_STATE_TRANSITIONING,
  // Entry is occupied (value is valid).
  IREE_HAL_AMDGPU_DEVICE_RINGBUFFER_ENTRY_STATE_OCCUPIED,
};

// An entry in the ringbuffer representing a single value.
typedef struct IREE_AMDGPU_ALIGNAS(16)
    iree_hal_amdgpu_device_ringbuffer_uint64_entry_t {
  // Current state of the entry used to block writers from conflicting.
  iree_amdgpu_scoped_atomic_uint32_t state;
  // Value stored in the entry.
  uint64_t value;
} iree_hal_amdgpu_device_ringbuffer_uint64_entry_t;

// Multi-producer/multi-consumer lock-free ringbuffer for uint64_t elements.
// Requires that the capacity is a power of two. Callers are responsible for
// allocating storage for the ringbuffer and must zero initialize it. Capacity
// must be a power-of-two.
//
// Note that today only the device is allowed to manipulate the ringbuffer.
// We could allow the host to share the data structure and operations but would
// need to change our atomics to be system scope instead of device scope.
//
// This implementation is derived from
//   "A Lock-Free Inter-Device Ring Buffer" by Keith Jeffery
//   https://www.keithjeffery.org/assets/files/lock_free_ring_buffer.pdf
//   https://github.com/kjeffery/lock_free_ring_buffer/tree/main
// It was ported to bare-metal C, specialized for uint64_t, and simplified as it
// only needs to run on-device and does not support locking. The paper is quite
// concise and worth a read if touching this code.
//
// The basic idea is that writers reserve space and then repeatedly try to
// acquire a write_index and populate the entry at that index. If any other
// writer comes in between when the space was reserved and the write index was
// attempted it'll be retried. In some pathological cases with multiple writers
// and fewer readers it's possible that the write_index needs to completely loop
// through the ring to find a free slot, but it is guaranteed to complete in
// O(capacity) iterations.
//
// Readers are more straightforward and reserve an entry then repeatedly try to
// acquire a read_index of an entry to take. If any other reader comes in
// between when an entry is reserved and when a particular read_index is tried
// the reader will retry. As with writers in certain pathological cases where
// there are far more readers than writers it's possible for the read_index to
// loop over the entire ringbuffer.
//
// Note that each atomic is kept on its own cache line as we expect
// multi-producer/multi-consumer behavior and want to avoid reads and writers
// from conflicting as much as possible. Probably overkill for the frequencies
// involved with our usage.
typedef struct iree_hal_amdgpu_device_ringbuffer_uint64_t {
  // Current number of elements in the ringbuffer.
  IREE_AMDGPU_ALIGNAS(iree_amdgpu_destructive_interference_size)
  iree_amdgpu_scoped_atomic_uint32_t size;
  // Total capacity of the ringbuffer in entries. Must be a power-of-two.
  uint32_t capacity;
  // Current read index pointing at the first occupied entry. Note that racing
  // may cause the entry to have already been taken and the state of the entry
  // must be checked.
  IREE_AMDGPU_ALIGNAS(iree_amdgpu_destructive_interference_size)
  iree_amdgpu_scoped_atomic_uint32_t read_index;
  // Current write index pointing at the first unoccupied entry. Note that
  // racing may cause the entry to have already been populated and the state of
  // the entry must be checked.
  IREE_AMDGPU_ALIGNAS(iree_amdgpu_destructive_interference_size)
  iree_amdgpu_scoped_atomic_uint32_t write_index;
  // Zero-initialized entries containing the values of a slot in the ringbuffer
  // and a state indicating its validity.
  IREE_AMDGPU_ALIGNAS(iree_amdgpu_destructive_interference_size)
  iree_hal_amdgpu_device_ringbuffer_uint64_entry_t
      entries[/*capacity*/];  // tail array
} iree_hal_amdgpu_device_ringbuffer_uint64_t;

// Returns the total size in bytes required to store a ringbuffer with the
// specified capacity.
#define iree_hal_amdgpu_device_ringbuffer_uint64_calculate_size(capacity) \
  sizeof(iree_hal_amdgpu_device_ringbuffer_uint64_t) +                    \
      (capacity) * sizeof(iree_hal_amdgpu_device_ringbuffer_uint64_entry_t)

#if defined(IREE_AMDGPU_TARGET_DEVICE)

// Returns the total number of valid entries in the ringbuffer at the time
// the method is called. Note that the count may change before the call returns.
static inline IREE_AMDGPU_ATTRIBUTE_ALWAYS_INLINE uint32_t
iree_hal_amdgpu_device_ringbuffer_uint64_size(
    const iree_hal_amdgpu_device_ringbuffer_uint64_t* IREE_AMDGPU_RESTRICT
        ringbuffer) {
  return iree_amdgpu_scoped_atomic_load(&ringbuffer->size,
                                        iree_amdgpu_memory_order_seq_cst,
                                        iree_amdgpu_memory_scope_device);
}

// Returns the maximum capacity in entries.
static inline IREE_AMDGPU_ATTRIBUTE_ALWAYS_INLINE uint32_t
iree_hal_amdgpu_device_ringbuffer_uint64_capacity(
    const iree_hal_amdgpu_device_ringbuffer_uint64_t* IREE_AMDGPU_RESTRICT
        ringbuffer) {
  return ringbuffer->capacity;
}

// Returns true if the ringbuffer is empty.
static inline IREE_AMDGPU_ATTRIBUTE_ALWAYS_INLINE bool
iree_hal_amdgpu_device_ringbuffer_uint64_empty(
    const iree_hal_amdgpu_device_ringbuffer_uint64_t* IREE_AMDGPU_RESTRICT
        ringbuffer) {
  return iree_hal_amdgpu_device_ringbuffer_uint64_size(ringbuffer) == 0;
}

// Returns true if the ringbuffer is full.
static inline IREE_AMDGPU_ATTRIBUTE_ALWAYS_INLINE bool
iree_hal_amdgpu_device_ringbuffer_uint64_full(
    const iree_hal_amdgpu_device_ringbuffer_uint64_t* IREE_AMDGPU_RESTRICT
        ringbuffer) {
  return iree_hal_amdgpu_device_ringbuffer_uint64_size(ringbuffer) ==
         ringbuffer->capacity;
}

// Enqueues a new value to the ringbuffer.
void iree_hal_amdgpu_device_ringbuffer_uint64_enqueue(
    iree_hal_amdgpu_device_ringbuffer_uint64_t* IREE_AMDGPU_RESTRICT ringbuffer,
    uint64_t value);

// Tries to enqueue a value to the ringbuffer if there is space available.
// Callers are expected to loop until a true return if they need to ensure the
// value is enqueued.
bool iree_hal_amdgpu_device_ringbuffer_uint64_try_enqueue(
    iree_hal_amdgpu_device_ringbuffer_uint64_t* IREE_AMDGPU_RESTRICT ringbuffer,
    uint64_t value);

// Dequeues a value from the ringbuffer. Spins if the ringbuffer is empty.
uint64_t iree_hal_amdgpu_device_ringbuffer_uint64_dequeue(
    iree_hal_amdgpu_device_ringbuffer_uint64_t* IREE_AMDGPU_RESTRICT
        ringbuffer);

typedef struct {
  uint64_t value;
  bool has_value;
} iree_hal_amdgpu_device_ringbuffer_uint64_result_t;

// Tries to dequeue a value from the ringbuffer if there is one present.
// Callers are expected to loop until has_value==true if they need a non-try.
// Returns a result with has_value=true if the value was successfully dequeued
// and otherwise has_value=false with an undefined value.
static iree_hal_amdgpu_device_ringbuffer_uint64_result_t
iree_hal_amdgpu_device_ringbuffer_uint64_try_dequeue(
    iree_hal_amdgpu_device_ringbuffer_uint64_t* IREE_AMDGPU_RESTRICT
        ringbuffer);

#endif  // IREE_AMDGPU_TARGET_DEVICE

#endif  // IREE_HAL_DRIVERS_AMDGPU_DEVICE_SUPPORT_RINGBUFFER_H_
