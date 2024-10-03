// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/device/semaphore.h"

#include "iree/hal/drivers/amdgpu/device/scheduler.h"

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_wake_target_t
//===----------------------------------------------------------------------===//

void iree_hal_amdgpu_device_wake_target(iree_hal_amdgpu_wake_target_t target) {
  // TODO(benvanik): support waking the host directly.
  iree_hal_amdgpu_device_queue_scheduler_enqueue_from_control_queue(
      target.scheduler, IREE_HAL_AMDGPU_DEVICE_QUEUE_TICK_ACTION_WAKE);
}

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_device_wake_set_t
//===----------------------------------------------------------------------===//

void iree_hal_amdgpu_device_wake_set_initialize(
    iree_hal_amdgpu_wake_target_t self,
    iree_hal_amdgpu_device_wake_set_t* IREE_AMDGPU_RESTRICT out_wake_set) {
  out_wake_set->self = self;
  out_wake_set->self_wake = 0;
#if !defined(IREE_HAL_AMDGPU_WAKE_SET_FLUSH_IMMEDIATELY)
  out_wake_set->target_count = 0;
#endif  // !IREE_HAL_AMDGPU_WAKE_SET_FLUSH_IMMEDIATELY
}

// Inserts a wake target into the set if it does not already exist.
// The target must remain live until the wake set has been processed.
static inline void iree_hal_amdgpu_device_wake_set_insert(
    iree_hal_amdgpu_device_wake_set_t* IREE_AMDGPU_RESTRICT wake_set,
    iree_hal_amdgpu_wake_target_t target) {
  if (target.scheduler == wake_set->self.scheduler) {
    // Track self-wakes for the owner (as there's no need to post to self).
    ++wake_set->self_wake;
  } else {
#if !defined(IREE_HAL_AMDGPU_WAKE_SET_FLUSH_IMMEDIATELY)
    // Find target in the list and exit early if already present.
    for (uint32_t i = 0; i < wake_set->target_count; ++i) {
      if (wake_set->targets[i].scheduler == target.scheduler) return;
    }
    wake_set->targets[wake_set->target_count++] = target;
#else
    iree_hal_amdgpu_device_wake_target(target);
#endif  // !IREE_HAL_AMDGPU_WAKE_SET_FLUSH_IMMEDIATELY
  }
}

bool iree_hal_amdgpu_device_wake_set_flush(
    iree_hal_amdgpu_device_wake_set_t* IREE_AMDGPU_RESTRICT wake_set) {
#if !defined(IREE_HAL_AMDGPU_WAKE_SET_FLUSH_IMMEDIATELY)
  for (uint32_t i = 0; i < wake_set->target_count; ++i) {
    iree_hal_amdgpu_device_wake_target(wake_set->targets[i]);
  }
  wake_set->target_count = 0;
#endif  // !IREE_HAL_AMDGPU_WAKE_SET_FLUSH_IMMEDIATELY
  const bool woke_self = wake_set->self_wake > 0;
  wake_set->self_wake = 0;
  return woke_self;
}

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_device_wake_pool_t
//===----------------------------------------------------------------------===//

void iree_hal_amdgpu_wake_list_entry_initialize(
    iree_hal_amdgpu_wake_target_t target,
    iree_hal_amdgpu_wake_list_entry_t* IREE_AMDGPU_RESTRICT out_entry) {
  out_entry->semaphore = NULL;
  out_entry->next = NULL;
  out_entry->minimum_value = 0;
  out_entry->last_value = 0;
  out_entry->target = target;
}

void iree_hal_amdgpu_device_wake_pool_initialize(
    iree_hal_amdgpu_wake_target_t target,
    iree_hal_amdgpu_device_wake_pool_t* IREE_AMDGPU_RESTRICT out_wake_pool) {
  for (uint32_t i = 0; i < IREE_AMDGPU_ARRAYSIZE(out_wake_pool->slots); ++i) {
    // NOTE: we don't rely on zero initialization here (but could).
    out_wake_pool->slots[i].semaphore = NULL;
    out_wake_pool->slots[i].next = NULL;
    out_wake_pool->slots[i].minimum_value = 0;
    out_wake_pool->slots[i].last_value = 0;
    out_wake_pool->slots[i].target = target;
  }
}

iree_hal_amdgpu_wake_list_entry_t* iree_hal_amdgpu_device_wake_pool_reserve(
    iree_hal_amdgpu_device_wake_pool_t* IREE_AMDGPU_RESTRICT wake_pool,
    iree_hal_amdgpu_device_semaphore_t* semaphore) {
  // TODO(benvanik): acceleration data structure for this. Slots are expensive
  // to walk and we're on a single thread.
  uint32_t first_free_slot = -1;
  for (uint32_t i = 0; i < IREE_AMDGPU_ARRAYSIZE(wake_pool->slots); ++i) {
    if (wake_pool->slots[i].semaphore == semaphore) {
      // Found an existing slot for this list - return it.
      return &wake_pool->slots[i];
    } else if (first_free_slot == -1 && wake_pool->slots[i].semaphore == NULL) {
      // Track the first free slot so we can grab it if we fail to find an
      // existing slot for the wake list.
      first_free_slot = i;
    }
  }
  if (IREE_AMDGPU_UNLIKELY(first_free_slot == -1)) {
    // Exhausted - this will be bad.
    return NULL;
  }
  wake_pool->slots[first_free_slot].semaphore = semaphore;
  return &wake_pool->slots[first_free_slot];
}

void iree_hal_amdgpu_device_wake_pool_release(
    iree_hal_amdgpu_device_wake_pool_t* IREE_AMDGPU_RESTRICT wake_pool,
    iree_hal_amdgpu_wake_list_entry_t* entry) {
  // NOTE: we could use the entry pointer as a base from the slots list if we
  // needed the index (for setting a use bitmap, first free slot, etc).
  entry->semaphore = NULL;
}

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_device_semaphore_t
//===----------------------------------------------------------------------===//

// Inserts a wake list entry into the list in minimum value order.
// Requires that the lock be held and the entry not currently be in the list.
static inline void iree_hal_amdgpu_device_semaphore_wake_list_insert_unsafe(
    iree_hal_amdgpu_device_semaphore_t* IREE_AMDGPU_RESTRICT semaphore,
    iree_hal_amdgpu_wake_list_entry_t* IREE_AMDGPU_RESTRICT entry) {
  // If the list is empty then assign to the head/tail.
  entry->next = NULL;
  if (semaphore->wake_list_head == NULL) {
    semaphore->wake_list_head = entry;
    semaphore->wake_list_tail = entry;
    return;
  }

  // Find the insertion point and splice in. Insert immediately prior to the
  // next minimum_value greater than the requested (or the tail).
  iree_hal_amdgpu_wake_list_entry_t* list_prev = NULL;
  iree_hal_amdgpu_wake_list_entry_t* list_cursor = semaphore->wake_list_head;
  while (list_cursor != NULL) {
    if (list_cursor == semaphore->wake_list_tail) {
      list_cursor->next = entry;
      semaphore->wake_list_tail = entry;
      break;
    } else if (list_cursor->minimum_value > entry->minimum_value) {
      if (list_prev != NULL) {
        if (list_prev->next != NULL) {
          entry->next = list_prev->next;
        }
        list_prev->next = entry;
      }
      break;
    }
    list_prev = list_cursor;
    list_cursor = list_cursor->next;
  }
}

// Moves a wake list entry earlier in the list.
// Requires that the lock be held and the entry currently be in the list.
// Requires that the new insertion point is before the existing position (cannot
// re-insert with the same value).
static inline void
iree_hal_amdgpu_device_semaphore_wake_list_reinsert_earlier_unsafe(
    iree_hal_amdgpu_device_semaphore_t* IREE_AMDGPU_RESTRICT semaphore,
    iree_hal_amdgpu_wake_list_entry_t* IREE_AMDGPU_RESTRICT entry) {
  // Insert into the list (again).
  // NOTE: this is unsafe as the entry is already in the list. We rely on the
  // lock being held here to ensure that no one else can be traversing the list
  // at the same time. We know that when the insertion returns the entry's next
  // pointer will point to the remainder of the list after the new position and
  // if we scan forward we should be able to find the entry to unlink.
  iree_hal_amdgpu_wake_list_entry_t* list_next = entry->next;
  iree_hal_amdgpu_device_semaphore_wake_list_insert_unsafe(semaphore, entry);

  // Scan forward from the insertion point to where the entry was. Resuming the
  // scan from insertion lets us remain O(n) instead of O(2n).
  iree_hal_amdgpu_wake_list_entry_t* list_prev = entry;
  iree_hal_amdgpu_wake_list_entry_t* list_cursor = entry->next;
  while (list_cursor != NULL) {
    if (list_cursor == entry) {
      // Found the entry - link previous to the original next.
      if (list_prev != NULL) {
        list_prev->next = list_next;
      }
      break;
    }
    list_prev = list_cursor;
    list_cursor = list_cursor->next;
  }
}

bool iree_hal_amdgpu_device_semaphore_update_wait(
    iree_hal_amdgpu_device_semaphore_t* IREE_AMDGPU_RESTRICT semaphore,
    iree_hal_amdgpu_wake_list_entry_t* IREE_AMDGPU_RESTRICT entry,
    uint64_t minimum_value) {
  bool in_list = false;
  iree_hal_amdgpu_device_mutex_lock(&semaphore->mutex);

  // Read the latest value - note that this may change immediately after we load
  // it. By loading it within the lock with acquire we _should_ always load in
  // order and store into the cached last_value.
  uint64_t latest_value =
      iree_hsa_signal_load(semaphore->signal, iree_amdgpu_memory_order_acquire);
  semaphore->last_value = latest_value;

  // Always populate the last value - it'll allow for more aggressive skipping.
  // If we were fancy we'd do this atomically and allow others to snoop it.
  entry->last_value = latest_value;

  if (entry->semaphore != NULL) {
    // NOTE: could assert list is the semaphore passed.
    // Already in the wake list - may be adding with an older value.
    // Since signaling happens with the lock held it should not be possible for
    // this to get here with the existing entry signaled so we know any one we
    // find will have not yet been reached.
    // If the existing minimum value is less than the one being inserted we can
    // just ignore it.
    if (entry->minimum_value > minimum_value) {
      // New value is less than the old one - need to move in the list so we are
      // woken earlier.
      entry->minimum_value = minimum_value;
      iree_hal_amdgpu_device_semaphore_wake_list_reinsert_earlier_unsafe(
          semaphore, entry);
      in_list = true;
    } else if (semaphore->last_value >= minimum_value) {
      // Requested value has already been reached - it may still be in the list
      // for other later values and we don't want to disturb those. Because we
      // are in the lock there's no way for an entry to be in the list if it
      // has been signaled already going off of the semaphore->last_value set
      // within the lock. We skip here and let the caller know the wake was
      // satisfied.
      in_list = false;
    }
  }
  if (semaphore->last_value >= minimum_value) {
    // Already satisfied, no need to insert. Caller is expected to handle this.
    in_list = false;
  } else {
    // Find the insertion point and splice in. Insert immediately prior to the
    // next minimum_value greater than the requested (or the tail).
    entry->semaphore = semaphore;
    entry->minimum_value = minimum_value;
    iree_hal_amdgpu_device_semaphore_wake_list_insert_unsafe(semaphore, entry);
    in_list = true;
  }

  iree_hal_amdgpu_device_mutex_unlock(&semaphore->mutex);
  return in_list;
}

void iree_hal_amdgpu_device_semaphore_signal(
    iree_hal_amdgpu_device_semaphore_t* IREE_AMDGPU_RESTRICT semaphore,
    uint64_t new_value,
    iree_hal_amdgpu_device_wake_set_t* IREE_AMDGPU_RESTRICT wake_set) {
  // Signal the new value on the HSA signal.
  // If the signal has a kernel event attached this will trigger an interrupt
  // immediately via update_mbox:
  // https://sourcegraph.com/github.com/ROCm/rocMLIR/-/blob/external/llvm-project/amd/device-libs/ockl/src/hsaqs.cl?L69
  //   id = signal->event_id
  //   atomic_store(signal->event_mailbox_ptr, id)
  //   s_sendmsg(1, id)
  // This routes to kfd_signal_event_interrupt/SOC15_INTSRC_SQ_INTERRUPT_MSG:
  // https://github.com/torvalds/linux/blob/master/drivers/gpu/drm/amd/amdkfd/kfd_int_process_v11.c#L351
  iree_hsa_signal_store(semaphore->signal, new_value,
                        iree_amdgpu_memory_order_release);

  iree_hal_amdgpu_device_mutex_lock(&semaphore->mutex);

  // Prevent any new registrations on old values.
  semaphore->last_value = new_value;

  // Walk all wake list entries in order and add any satisfied to the wake set.
  iree_hal_amdgpu_wake_list_entry_t* list_cursor = semaphore->wake_list_head;
  while (list_cursor != NULL) {
    // Break on the first wake entry that is not yet satisfied.
    // The linked list is sorted so we can do this.
    iree_hal_amdgpu_wake_list_entry_t* list_next = list_cursor->next;
    if (list_cursor->minimum_value > new_value) {
      break;
    }

    // Pop from head of list and unlink.
    semaphore->wake_list_head = list_next;
    if (list_next == NULL) {
      semaphore->wake_list_tail = NULL;  // last item
    }
    list_cursor->next = NULL;  // note prev was popped so no need to fix that
    list_cursor->semaphore = NULL;  // now out of the wake list

    // Store the new value in the wake entry so that the waiter can quickly
    // check it on next schedule.
    list_cursor->last_value = new_value;

    // Insert into wake set. This deduplicates if the target has already been
    // requested to wake.
    iree_hal_amdgpu_device_wake_set_insert(wake_set, list_cursor->target);

    list_cursor = list_next;  // continue
  }

  iree_hal_amdgpu_device_mutex_unlock(&semaphore->mutex);
}
