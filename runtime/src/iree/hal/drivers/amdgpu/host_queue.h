// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_H_
#define IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_H_

#include "iree/async/frontier.h"
#include "iree/async/proactor.h"
#include "iree/async/semaphore.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"
#include "iree/hal/drivers/amdgpu/virtual_queue.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_amdgpu_logical_device_t
    iree_hal_amdgpu_logical_device_t;

//===----------------------------------------------------------------------===//
// Constants
//===----------------------------------------------------------------------===//

// Initial value for the per-queue epoch signal. The CP decrements by 1 on
// each submission's last packet completion. The epoch (number of completed
// submissions) is: INITIAL_VALUE - hsa_signal_load(epoch_signal).
// INT64_MAX/2 gives ~4.6e18 decrements before overflow (~146 years at 1
// billion submissions/second).
#define IREE_HAL_AMDGPU_EPOCH_INITIAL_VALUE (INT64_MAX / 2)

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_notification_entry_t
//===----------------------------------------------------------------------===//

// A pending semaphore signal associated with a submission epoch. When the
// epoch is reached (the GPU completes the submission's last packet), the
// proactor drain signals the semaphore to the timeline value with the
// inline frontier.
//
// Entries are stored in a per-queue fixed-size ring buffer, produced by the
// submission path and consumed by the proactor drain.
typedef struct iree_hal_amdgpu_notification_entry_t {
  // Semaphore to signal when the epoch is reached. Not retained — the caller
  // ensures the semaphore outlives the notification (queue teardown waits for
  // all in-flight work and drains before destroying semaphores).
  iree_async_semaphore_t* semaphore;
  // Timeline value to signal the semaphore to.
  uint64_t timeline_value;
  // Submission epoch on this queue. When the queue's epoch advances past
  // this value (current_epoch > submission_epoch), this entry is ready to
  // drain.
  uint64_t submission_epoch;
  // Inline frontier copy (causal context merged into the semaphore on signal).
  // entry_count = 0 means no frontier.
  iree_async_frontier_t frontier;
  iree_async_frontier_entry_t
      frontier_entries[IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY];
} iree_hal_amdgpu_notification_entry_t;

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_host_queue_t
//===----------------------------------------------------------------------===//

// Host-driven queue with per-queue epoch signal and proactor-drained
// notification ring. Embeds iree_hal_amdgpu_virtual_queue_t at offset 0.
//
// The epoch signal is a single hsa_signal_t created at queue initialization.
// Every submission's last AQL packet sets completion_signal to this signal;
// the CP decrements it by 1 on completion. The notification ring maps epochs
// to semaphore signals that the proactor drains when the epoch advances.
//
// Queue operations (execute, fill, copy, etc.) are stubbed — the submission
// path is implemented separately.
typedef struct iree_hal_amdgpu_host_queue_t {
  // Virtual queue vtable at offset 0.
  iree_hal_amdgpu_virtual_queue_t base;

  // Owning logical device (not retained — device owns queue).
  iree_hal_amdgpu_logical_device_t* device;
  // Proactor for async notifications (borrowed from device).
  iree_async_proactor_t* proactor;
  // HSA API handle for signal operations.
  const iree_hal_amdgpu_libhsa_t* libhsa;
  iree_allocator_t host_allocator;

  // Monotonic completion counter. A single hsa_signal_t starts at
  // IREE_HAL_AMDGPU_EPOCH_INITIAL_VALUE and is decremented by 1 on each
  // submission's last AQL packet completion. The current epoch (number of
  // completed submissions) is INITIAL - signal_load(signal). The submission
  // path assigns epochs via next_submission, and the drain reads the signal
  // to determine which epochs have completed. The signal lives in host
  // memory (hsa_amd_signal_create), readable from any GPU via PCIe/XGMI.
  struct {
    // Per-queue hsa_signal_t. Created at init, destroyed at deinit.
    // Set as completion_signal on the last AQL packet of each submission;
    // intermediate packets use the null signal.
    hsa_signal_t signal;
    // Next epoch to assign. Incremented by 1 per submission (whether or not
    // the submission signals any semaphores).
    uint64_t next_submission;
    // Last epoch observed by the drain. Used to skip drain when the signal
    // hasn't advanced since the last check.
    uint64_t last_drained;
  } epoch;

  // Fixed-size FIFO mapping submission epochs to pending semaphore signals.
  // The submission path pushes entries (advance write); the proactor drain
  // pops completed entries (advance read). Only signaling submissions produce
  // entries, so the ring is sparse with respect to epochs. The drain stops
  // at the first entry whose submission_epoch exceeds the current epoch.
  // Capacity is a power of 2, bounded by the AQL ring size.
  struct {
    // Heap-allocated ring buffer. Each entry holds a semaphore pointer,
    // timeline value, submission epoch, and an inline frontier copy.
    iree_hal_amdgpu_notification_entry_t* ring;
    // Producer index (submission path advances).
    uint64_t write;
    // Consumer index (proactor drain advances).
    uint64_t read;
    // Power-of-two ring capacity. Indices are masked by (capacity - 1).
    uint32_t capacity;
  } notification;

  // Proactor bridge. Registers a progress callback that polls the epoch
  // signal each proactor iteration and runs the drain when it advances.
  //
  // v1 (current): hsa_signal_load on each poll iteration.
  // v2 (future): native io_uring HSA_SIGNAL_WAIT SQE delivers a CQE on
  //   signal change, avoiding per-iteration polling. The drain callback is
  //   identical — only the wakeup mechanism changes. Runtime-conditioned
  //   at queue init.
  iree_async_progress_entry_t progress_entry;
} iree_hal_amdgpu_host_queue_t;

// Default notification ring capacity. The caller passes this (or a custom
// power-of-two value) to iree_hal_amdgpu_host_queue_initialize.
#define IREE_HAL_AMDGPU_DEFAULT_NOTIFICATION_CAPACITY 1024

// Initializes a host queue in caller-provided memory.
// The caller must allocate at least sizeof(iree_hal_amdgpu_host_queue_t).
//
// |notification_capacity| is the power-of-two size of the notification ring.
// Returns IREE_STATUS_INVALID_ARGUMENT if not a power of two.
//
// Creates the epoch signal via hsa_amd_signal_create and registers the
// proactor progress callback for drain.
iree_status_t iree_hal_amdgpu_host_queue_initialize(
    iree_hal_amdgpu_logical_device_t* device, iree_async_proactor_t* proactor,
    const iree_hal_amdgpu_libhsa_t* libhsa, uint32_t notification_capacity,
    iree_allocator_t host_allocator, iree_hal_amdgpu_host_queue_t* out_queue);

// Deinitializes the queue. Destroys the epoch signal, frees the notification
// ring, and unregisters the proactor progress callback.
//
// All in-flight work must have completed and been drained before calling.
// The caller must ensure no concurrent access to the queue during deinit.
void iree_hal_amdgpu_host_queue_deinitialize(
    iree_hal_amdgpu_host_queue_t* queue);

// Returns the epoch signal for use as completion_signal in AQL packets.
static inline hsa_signal_t iree_hal_amdgpu_host_queue_epoch_signal(
    const iree_hal_amdgpu_host_queue_t* queue) {
  return queue->epoch.signal;
}

// Advances the submission epoch counter and returns the assigned epoch.
// Called by the submission path after all AQL packets for a submission have
// been written to the hardware queue. The returned epoch is the one to store
// in notification entries and the semaphore's last_signal cache.
static inline uint64_t iree_hal_amdgpu_host_queue_advance_epoch(
    iree_hal_amdgpu_host_queue_t* queue) {
  return queue->epoch.next_submission++;
}

// Pushes a notification entry for a semaphore signal at the current epoch.
// Called by the submission path when queue_execute signals a semaphore.
//
// |submission_epoch| is the epoch returned by advance_epoch for this
// submission. |frontier| may be NULL if no causal context is available.
//
// The caller must ensure the notification ring has capacity (it does as long
// as the AQL ring is not overcommitted).
void iree_hal_amdgpu_host_queue_push_notification(
    iree_hal_amdgpu_host_queue_t* queue, uint64_t submission_epoch,
    iree_async_semaphore_t* semaphore, uint64_t timeline_value,
    const iree_async_frontier_t* frontier);

// Drains all completed notification entries. Reads the current epoch from
// the epoch signal and signals async semaphores for all entries whose
// submission_epoch <= current_epoch.
//
// Called by the proactor progress callback. May also be called manually
// (e.g., during queue teardown to flush remaining entries).
//
// Returns the number of notification entries drained.
iree_host_size_t iree_hal_amdgpu_host_queue_drain(
    iree_hal_amdgpu_host_queue_t* queue);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_H_
