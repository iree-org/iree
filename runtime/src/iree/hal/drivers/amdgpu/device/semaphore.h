// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_DEVICE_SEMAPHORE_H_
#define IREE_HAL_DRIVERS_AMDGPU_DEVICE_SEMAPHORE_H_

#include "iree/hal/drivers/amdgpu/device/support/common.h"
#include "iree/hal/drivers/amdgpu/device/support/mutex.h"
#include "iree/hal/drivers/amdgpu/device/support/signal.h"

typedef struct iree_hal_amdgpu_device_queue_scheduler_t
    iree_hal_amdgpu_device_queue_scheduler_t;
typedef struct iree_hal_amdgpu_device_semaphore_t
    iree_hal_amdgpu_device_semaphore_t;

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_wake_target_t
//===----------------------------------------------------------------------===//

// A target of a wake operation.
// Targets must outlive any wake list they are registered with. Generally this
// is the case as targets have a lifetime matching the HAL device they are
// associated with and any wake list must have a shorter lifetime than the
// device it was created from.
typedef struct iree_hal_amdgpu_wake_target_t {
  // TODO(benvanik): union and use bit 0 for indicating host targets.
  iree_hal_amdgpu_device_queue_scheduler_t* scheduler;
} iree_hal_amdgpu_wake_target_t;

// Wakes the target by signaling its doorbell, enqueuing device agent ticks,
// etc. The target may immediately awake and process before this call returns.
void iree_hal_amdgpu_device_wake_target(iree_hal_amdgpu_wake_target_t target);

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_device_wake_set_t
//===----------------------------------------------------------------------===//

// When set iree_hal_amdgpu_device_wake_set_t will only track self-wakes and
// otherwise wake all targets immediately as they are inserted into the set.
// This may reduce latency when retiring a lot of queue operations but at the
// cost of potentially doing a non-trivial amount of work with locks held.
//
// TODO(benvanik): evaluate if what the right approach is. For now we err on the
// safe side at the potential cost of additional latency.
// #define IREE_HAL_AMDGPU_WAKE_SET_FLUSH_IMMEDIATELY 1

// A uniqued set of wake targets.
// This is intended to be populated over the course of a scheduler tick with
// any targets that need to be woken.
//
// Thread-compatible. Assumed owned and exclusively managed by the scheduler
// that is performing the wakes.
//
// TODO(benvanik): maybe replace with a bitfield? Could have targets registered
// in a 64 element set shared across all schedulers then this just becomes
// bitwise ops. For now it's kept general (and more expensive).
typedef struct iree_hal_amdgpu_device_wake_set_t {
  // The owner of the set used to detect self-wakes.
  iree_hal_amdgpu_wake_target_t self;
  // >0 if the target indicated by self was requested to wake.
  uint32_t self_wake;
#if !defined(IREE_HAL_AMDGPU_WAKE_SET_FLUSH_IMMEDIATELY)
  // Total number of non-self targets in the target list.
  uint32_t target_count;
  // Dense list of wake targets in an arbitrary order.
  iree_hal_amdgpu_wake_target_t targets[62];
#endif  // !IREE_HAL_AMDGPU_WAKE_SET_FLUSH_IMMEDIATELY
} iree_hal_amdgpu_device_wake_set_t;

// Initializes a wake set.
void iree_hal_amdgpu_device_wake_set_initialize(
    iree_hal_amdgpu_wake_target_t self,
    iree_hal_amdgpu_device_wake_set_t* IREE_AMDGPU_RESTRICT out_wake_set);

// Notifies all wake targets in the set and clears the set.
// Returns true if the self target was set to be notified. Note that the self
// target will not be notified via wake posts.
bool iree_hal_amdgpu_device_wake_set_flush(
    iree_hal_amdgpu_device_wake_set_t* IREE_AMDGPU_RESTRICT wake_set);

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_device_wake_pool_t
//===----------------------------------------------------------------------===//

// A linked-list entry in a iree_hal_amdgpu_device_semaphore_t wake list.
// These are stored in the waiters and registered with the wake list in-place.
// Each entry indicates the first (of potentially several) wakes required for
// a target and targets are expected to reregister themselves after being woken.
typedef struct iree_hal_amdgpu_wake_list_entry_t {
  // The wake list this entry is a member of.
  // This is tracked for the owners of the wake list so that from an entry it's
  // possible to get the wake list the entry is in. A bit odd, but prevents the
  // need for additional data structures.
  iree_hal_amdgpu_device_semaphore_t* semaphore;
  // Intrusive linked-list pointer for iree_hal_amdgpu_device_semaphore_t.
  struct iree_hal_amdgpu_wake_list_entry_t* next;
  // Minimum value that must be reached in order to wake the target.
  uint64_t minimum_value;
  // Value last seen on a wake.
  // If this is greater than the minimum_value then the entry is not in a wake
  // list.
  uint64_t last_value;
  // Target to wake.
  iree_hal_amdgpu_wake_target_t target;
} iree_hal_amdgpu_wake_list_entry_t;

// Initializes a wake list entry.
// Zero-initialization is allowed however the target must be set when inserted.
void iree_hal_amdgpu_wake_list_entry_initialize(
    iree_hal_amdgpu_wake_target_t target,
    iree_hal_amdgpu_wake_list_entry_t* IREE_AMDGPU_RESTRICT out_entry);

// A set of semaphore-to-wake-entry mappings used to evaluate whether more work
// can progress.
//
// Note that because the entry addresses are used as part of external linked
// lists we cannot rearrange the slots and will end up with a sparse list over
// time. We could make this better by using a bitmask to indicate which slots
// are valid but for now we rely on the number of active waits being small, the
// frequency at which waits are added and removed being low, and most
// wait-before-signal operations happening in long sequences instead of big
// fan-outs.
//
// This set is fixed size as the set of unique semaphores a single queue should
// be waiting on _should_ be reasonable. If we find programs doing bad things we
// could extend this to use a more complex dynamic data structure or set the
// fixed size higher at the cost of more bitmask munging. Someone has to own the
// backing storage for the wake list entries and the waiters are the best
// option today. An alternative would be to store the wake entries in queue
// entries such that we can scale with the number of pending waits but at the
// point we're needing to scan more than a dozen waits per scheduler tick we'll
// need to redesign things. A more localized option would be to have a free list
// of an excessive size and
//
// Thread-compatible; expected to be accessed only by the waiter.
typedef struct iree_hal_amdgpu_device_wake_pool_t {
  // Intrusive storage for each semaphore wake list entry.
  // For as long as the wait is active the wake list will reference this memory.
  // Contains the minimum value needed for the wait to be satisfied (once).
  iree_hal_amdgpu_wake_list_entry_t slots[64];
} iree_hal_amdgpu_device_wake_pool_t;

// Initializes the slots in |out_wake_pool| routing to the specified |target|.
void iree_hal_amdgpu_device_wake_pool_initialize(
    iree_hal_amdgpu_wake_target_t target,
    iree_hal_amdgpu_device_wake_pool_t* IREE_AMDGPU_RESTRICT out_wake_pool);

// Reserves a pool slot for the given |semaphore|.
// If the slot exists it is returned and otherwise a new slot is found and
// assigned to the wake list. The caller is responsible for returning the slot
// back to the pool if it's not needed.
iree_hal_amdgpu_wake_list_entry_t* iree_hal_amdgpu_device_wake_pool_reserve(
    iree_hal_amdgpu_device_wake_pool_t* IREE_AMDGPU_RESTRICT wake_pool,
    iree_hal_amdgpu_device_semaphore_t* semaphore);

// Releases a previously acquired pool slot.
void iree_hal_amdgpu_device_wake_pool_release(
    iree_hal_amdgpu_device_wake_pool_t* IREE_AMDGPU_RESTRICT wake_pool,
    iree_hal_amdgpu_wake_list_entry_t* entry);

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_device_semaphore_t
//===----------------------------------------------------------------------===//

// A semaphore with an intrusive linked list of targets to wake.
// Semaphores are an HSA signal plus tracking of waiters to allow direct wakes.
// For semaphores only ever used on devices we avoid host interrupts and only
// use our own wakes.
//
// Targets are registered with the minimum value that must be reached before
// they can wake and notifications will wake all that are satisified. The linked
// list uses storage within the waiting targets.
//
// This optimizes for being able to wake multiple waiters when the signal is
// notified and tries to allow the waiters to do the polling as to completion.
// This allows for external wakes to be handled the same as ones from this list
// and for us to not need an exhaustive wake list per semaphore per queue entry.
// Instead, we only need to track what unique semaphores are being waited on
// and ensure each waiter is in the list once with the minimum payload that
// would cause them to wake. Since we may have long pipelines of work on a small
// number of semaphores this optimizes for "wake and dequeue next" instead of
// needing to walk the entire pipeline graph.
//
// When a waiter wants to register for notification they insert themselves into
// the list. The insertion will fail if the last notified value is >= the
// requested minimum value as no signal will ever be made for it and callers can
// immediately continue processing. This allows the waiters to treat insertions
// as the polling operation instead of having to check multiple times.
// If the waiter is already in the wake list at a larger value then the entry is
// removed and reinserted in the appropriate order. This is relatively rare and
// handled on a slow path.
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
  // A pointer back to the owning host iree_hal_amdgpu_*_semaphore_t.
  // Used when asking the host to manipulate the semaphore.
  uint64_t host_semaphore;

  // HSA signal (iree_amd_signal_t) backing the semaphore.
  // Note that this is signaled outside of the lock and may be signaled
  // externally - the last_value we track in the state here is just for ensuring
  // we wake everything.
  iree_hsa_signal_t signal;

  // Mutex used when manipulating the wake list.
  iree_hal_amdgpu_device_mutex_t mutex;

  // Last value signaled. Used for quickly dropping new insertions.
  uint64_t last_value IREE_AMDGPU_GUARDED_BY(mutex);

  // Head of the wake list. May be NULL.
  iree_hal_amdgpu_wake_list_entry_t* wake_list_head
      IREE_AMDGPU_GUARDED_BY(mutex);
  // Tail of the wake list. May be NULL.
  iree_hal_amdgpu_wake_list_entry_t* wake_list_tail
      IREE_AMDGPU_GUARDED_BY(mutex);
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

// Polls the semaphore and tries to insert a wake list entry into the list.
// Callers must populate the entry target and ensure the next pointer is NULL
// on initialization and then always pass the same entry for the same wake list
// until it is signaled. This is the only call that can insert entries into the
// list and iree_hal_amdgpu_device_semaphore_signal is the only call that can
// erase them. Both happen with the wake list lock held.
//
// The entry storage and target must remain live for at least as long as the
// wake list may reference it. This generally happens because the wake list is
// owned by a HAL semaphore and the waiters are schedulers that retain the
// semaphore while they are waiting on it.
//
// Returns true if the entry was (or is already) inserted and otherwise false
// indicating that the entry is already satisfied. In either case the entry will
// be updated with the last signaled value of the wake list.
bool iree_hal_amdgpu_device_semaphore_update_wait(
    iree_hal_amdgpu_device_semaphore_t* IREE_AMDGPU_RESTRICT semaphore,
    iree_hal_amdgpu_wake_list_entry_t* IREE_AMDGPU_RESTRICT entry,
    uint64_t minimum_value);

// Adds any targets that should be woken when the new payload value is reached.
// Targets will be inserted into the wake set.
void iree_hal_amdgpu_device_semaphore_signal(
    iree_hal_amdgpu_device_semaphore_t* IREE_AMDGPU_RESTRICT semaphore,
    uint64_t new_value,
    iree_hal_amdgpu_device_wake_set_t* IREE_AMDGPU_RESTRICT wake_set);

#endif  // IREE_AMDGPU_TARGET_DEVICE

#endif  // IREE_HAL_DRIVERS_AMDGPU_DEVICE_SEMAPHORE_H_
