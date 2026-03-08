// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_UTILS_SEMAPHORE_BASE_H_
#define IREE_HAL_UTILS_SEMAPHORE_BASE_H_

#include <stdbool.h>
#include <stdint.h>

#include "iree/async/semaphore.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_semaphore_t
//===----------------------------------------------------------------------===//

// HAL semaphore base type. Embeds iree_async_semaphore_t at offset 0 for
// toll-free bridging: iree_hal_semaphore_t* -> iree_async_semaphore_t*
// -> iree_hal_resource_t* all share offset 0 with compatible {ref_count,
// vtable} layout.
//
// All HAL backends that use this base type get timeline value tracking,
// frontier accumulation, failure tracking, and timepoint dispatch for free.
// Hardware primitives (VkSemaphore, CUevent, hsa_signal_t) are additive
// acceleration on top of this state.
struct iree_hal_semaphore_t {
  iree_async_semaphore_t async;  // must be at offset 0
};

// Initializes a HAL semaphore in-place within caller-provided memory.
// The caller must have allocated at least the size returned by
// iree_async_semaphore_layout() and zeroed the memory.
//
// |vtable| is the driver's HAL semaphore vtable. The .async member at offset 0
// is passed to the async semaphore (toll-free cast).
// |initial_value| is the starting timeline value.
// |frontier_offset| and |frontier_capacity| describe the trailing frontier
// storage (from iree_async_semaphore_layout).
IREE_API_EXPORT void iree_hal_semaphore_initialize(
    const iree_hal_semaphore_vtable_t* vtable, uint64_t initial_value,
    iree_host_size_t frontier_offset, uint8_t frontier_capacity,
    iree_hal_semaphore_t* out_semaphore);

// Deinitializes the HAL semaphore base, releasing internal resources.
// Dispatches all pending timepoints with CANCELLED status.
// Does NOT free the semaphore memory — the driver manages that.
IREE_API_EXPORT void iree_hal_semaphore_deinitialize(
    iree_hal_semaphore_t* semaphore);

//===----------------------------------------------------------------------===//
// Bridge: old HAL timepoint API over async semaphore internals
//===----------------------------------------------------------------------===//
// These types and functions bridge the old HAL timepoint API (used by CUDA,
// local_task, and others) over the async semaphore's timepoint list.
// They are temporary — removed once all drivers migrate to use the async
// semaphore's state directly (via composition helpers).

// Old-style callback for HAL semaphore timepoints.
// Receives the semaphore, current value, and status code.
// Returns a status that is ignored (legacy contract).
typedef iree_status_t(IREE_API_PTR* iree_hal_semaphore_callback_fn_t)(
    void* user_data, iree_hal_semaphore_t* semaphore, uint64_t value,
    iree_status_code_t status_code);

typedef struct iree_hal_semaphore_callback_t {
  iree_hal_semaphore_callback_fn_t fn;
  void* user_data;
} iree_hal_semaphore_callback_t;

// Bridge timepoint storage. Embeds iree_async_semaphore_timepoint_t at offset 0
// so it can be linked into the async semaphore's timepoint list directly.
// Driver-specific timepoint types (iree_hal_cuda_timepoint_t,
// iree_hal_task_timepoint_t) embed this at offset 0 in turn.
typedef struct iree_hal_semaphore_timepoint_t {
  // Async timepoint at offset 0 for list linkage and dispatch.
  iree_async_semaphore_timepoint_t async;

  // Retained semaphore (old contract: timepoints hold a reference).
  // Set to NULL by the bridge adapter callback after releasing.
  iree_hal_semaphore_t* retained_semaphore;

  // Absolute deadline (from timeout conversion).
  iree_time_t deadline_ns;

  // Old-style callback to issue when the timepoint fires.
  iree_hal_semaphore_callback_t callback;
} iree_hal_semaphore_timepoint_t;

// Acquires a timepoint on the semaphore timeline via the bridge.
// Sets up the async timepoint with a bridge adapter callback that translates
// between the new dispatch-under-lock callback and the old HAL callback.
// Retains the semaphore (old contract).
//
// The callback may fire before this function returns if the value is already
// reached or the semaphore has already failed.
IREE_API_EXPORT void iree_hal_semaphore_acquire_timepoint(
    iree_hal_semaphore_t* semaphore, uint64_t minimum_value,
    iree_timeout_t timeout, iree_hal_semaphore_callback_t callback,
    iree_hal_semaphore_timepoint_t* out_timepoint);

// Cancels a bridge timepoint.
// After this returns, the callback will not fire (or has already completed).
// Releases the retained semaphore if the callback hasn't already.
IREE_API_EXPORT void iree_hal_semaphore_cancel_timepoint(
    iree_hal_semaphore_t* semaphore, iree_hal_semaphore_timepoint_t* timepoint);

// Notifies the semaphore that a new value has been reached or a failure
// occurred. Syncs the async semaphore's timeline state and dispatches
// satisfied (or failed) timepoints.
//
// Called by driver signal/fail methods after updating their own state.
// Must not be called from a timepoint callback.
IREE_API_EXPORT void iree_hal_semaphore_notify(
    iree_hal_semaphore_t* semaphore, uint64_t new_value,
    iree_status_code_t new_status_code);

// Polls the semaphore and dispatches any resolved timepoints.
// Queries the current value via the vtable, then calls notify.
IREE_API_EXPORT void iree_hal_semaphore_poll(iree_hal_semaphore_t* semaphore);

//===----------------------------------------------------------------------===//
// Default stubs for async vtable methods
//===----------------------------------------------------------------------===//

// Returns 0 entries (no frontier tracking).
static inline uint8_t iree_hal_semaphore_default_query_frontier(
    iree_async_semaphore_t* semaphore, iree_async_frontier_t* out_frontier,
    uint8_t capacity) {
  (void)semaphore;
  (void)out_frontier;
  (void)capacity;
  return 0;
}

// Async timepoints are not supported by legacy implementations.
static inline iree_status_t iree_hal_semaphore_default_acquire_timepoint(
    iree_async_semaphore_t* semaphore, uint64_t minimum_value,
    iree_async_semaphore_timepoint_t* timepoint) {
  (void)semaphore;
  (void)minimum_value;
  (void)timepoint;
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "async timepoints not supported by this semaphore");
}

// No-op for legacy implementations that don't support async timepoints.
static inline void iree_hal_semaphore_default_cancel_timepoint(
    iree_async_semaphore_t* semaphore,
    iree_async_semaphore_timepoint_t* timepoint) {
  (void)semaphore;
  (void)timepoint;
}

// Primitive export not supported by legacy implementations.
static inline iree_status_t iree_hal_semaphore_default_export_primitive(
    iree_async_semaphore_t* semaphore, uint64_t minimum_value,
    iree_async_primitive_t* out_primitive) {
  (void)semaphore;
  (void)minimum_value;
  (void)out_primitive;
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "primitive export not supported by this semaphore");
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_UTILS_SEMAPHORE_BASE_H_
