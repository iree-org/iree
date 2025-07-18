// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_DEVICE_SEMAPHORE_H_
#define IREE_HAL_DRIVERS_AMDGPU_DEVICE_SEMAPHORE_H_

#include "iree/hal/drivers/amdgpu/device/support/common.h"
#include "iree/hal/drivers/amdgpu/device/support/mutex.h"
#include "iree/hal/drivers/amdgpu/device/support/signal.h"

typedef struct iree_hal_amdgpu_device_semaphore_t
    iree_hal_amdgpu_device_semaphore_t;

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_device_semaphore_t
//===----------------------------------------------------------------------===//

// A semaphore with an intrusive linked list of targets to wake.
// Semaphores are an HSA signal plus tracking of waiters to allow direct wakes.
// For semaphores only ever used on devices we avoid host interrupts and only
// use our own wakes.
//
// Targets are registered with the minimum value that must be reached before
// they can wake and notifications will wake all that are satisfied. The linked
// list uses storage within the waiting targets.
//
// This optimizes for being able to wake multiple waiters when the signal is
// notified and tries to allow the waiters to do the completion state polling.
// This enables external wakes to be handled the same as ones from the wake list
// and for us to not need an exhaustive wake list per semaphore per queue entry.
// Instead, we only need to track what unique semaphores are being waited on
// and ensure each waiter is in the list once with the minimum payload that
// would cause them to wake. Since we may have long pipelines of work on a small
// number of semaphores this optimizes for "wake and dequeue next" instead of
// needing to walk the entire pipeline graph.
//
// When a waiter wants to register for notification they insert themselves into
// the list. The insertion will fail if the last notified value is >= the
// requested minimum value (as no signal will ever be made for it) and callers
// can immediately continue processing. This allows the waiters to treat
// insertions as the polling operation instead of having to check multiple
// times. If the waiter is already in the wake list at a larger value then the
// entry is removed and reinserted in the appropriate order. This is relatively
// rare and handled on a slow path.
//
// Must be allocated in device/host shared memory if it will ever be used on the
// host - and most may be. A outer wrapper iree_hal_semaphore_t owns the memory
// and manages its lifetime and can pool this device-side block for reuse.
//
// Thread-safe. May be accessed from both host and device concurrently.
// Zero initialization compatible.
//
// TODO(benvanik): make doubly-linked? insertion scan from the tail may be best
// as usually we enqueue operations in order (1->2->3).
typedef struct iree_hal_amdgpu_device_semaphore_t {
  // HSA signal in device-visible memory.
  // This may be a ROCR DefaultSignal (busy-wait) or InterruptSignal (event)
  // based on the semaphore type.
  IREE_AMDGPU_DEVICE_PTR iree_amd_signal_t* signal;

  // A pointer back to the owning host iree_hal_amdgpu_*_semaphore_t.
  // Used when asking the host to manipulate the semaphore.
  uint64_t host_semaphore;

  // TODO(benvanik): implement device-side semaphore data for both host and
  // device queue modes.
} iree_hal_amdgpu_device_semaphore_t;

// A list of semaphores and the payload the semaphore is expected to reach or
// be signaled to depending on the operation.
typedef struct iree_hal_amdgpu_device_semaphore_list_t {
  uint16_t count;
  uint16_t reserved0;
  uint32_t reserved1;  // could store wait state tracking
  struct {
    iree_hal_amdgpu_device_semaphore_t* semaphore;
    uint64_t payload;
  } entries[];
} iree_hal_amdgpu_device_semaphore_list_t;

// Returns the total size in bytes of a semaphore list storing |count| entries.
static inline size_t iree_hal_amdgpu_device_semaphore_list_size(
    uint16_t count) {
  iree_hal_amdgpu_device_semaphore_list_t* list = NULL;
  return sizeof(*list) + count * sizeof(list->entries[0]);
}

#if defined(IREE_AMDGPU_TARGET_DEVICE)

// TODO(benvanik): implement device semaphore logic.

#endif  // IREE_AMDGPU_TARGET_DEVICE

#endif  // IREE_HAL_DRIVERS_AMDGPU_DEVICE_SEMAPHORE_H_
