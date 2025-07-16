// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_SEMAPHORE_H_
#define IREE_HAL_DRIVERS_AMDGPU_SEMAPHORE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_amdgpu_device_semaphore_t
    iree_hal_amdgpu_device_semaphore_t;

typedef struct iree_hal_amdgpu_internal_semaphore_t
    iree_hal_amdgpu_internal_semaphore_t;
typedef struct iree_hal_amdgpu_system_t iree_hal_amdgpu_system_t;

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

// Options controlling global semaphore behavior.
// Semaphore flags may override these options.
typedef struct iree_hal_amdgpu_semaphore_options_t {
  // Uses HSA_WAIT_STATE_ACTIVE for up to the given duration before switching to
  // HSA_WAIT_STATE_BLOCKED. Above zero this will increase CPU usage in cases
  // where the waits are long and decrease latency in cases where the waits are
  // short. When IREE_DURATION_INFINITE waits will use HSA_WAIT_STATE_ACTIVE.
  iree_duration_t wait_active_for_ns;
} iree_hal_amdgpu_semaphore_options_t;

typedef void(IREE_API_PTR* iree_hal_amdgpu_internal_semaphore_release_fn_t)(
    void* user_data, iree_hal_amdgpu_internal_semaphore_t* semaphore);

// A callback issued when a semaphore is released.
typedef struct {
  // Callback function pointer.
  iree_hal_amdgpu_internal_semaphore_release_fn_t fn;
  // User data passed to the callback function. Unowned.
  void* user_data;
} iree_hal_amdgpu_internal_semaphore_release_callback_t;

// Returns a no-op release callback that implies that no cleanup is required.
static inline iree_hal_amdgpu_internal_semaphore_release_callback_t
iree_hal_amdgpu_internal_semaphore_release_callback_null(void) {
  iree_hal_amdgpu_internal_semaphore_release_callback_t callback = {NULL, NULL};
  return callback;
}

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_internal_semaphore_t
//===----------------------------------------------------------------------===//

// An internally-tracked HAL semaphore.
// These carry additional information used by the implementation to optimize
// wait/wake behavior and allow device-side wait/wake.
typedef struct iree_hal_amdgpu_internal_semaphore_t {
  iree_hal_resource_t resource;  // must be at 0

  // Unowned libhsa handle. Must be retained by the parent pool.
  const iree_hal_amdgpu_libhsa_t* libhsa;

  // Global semaphore options, may be overridden based on flags.
  iree_hal_amdgpu_semaphore_options_t options;

  // Flags controlling semaphore behavior.
  iree_hal_semaphore_flags_t flags;

  // HSA signal handle. Contains the semaphore payload value.
  hsa_signal_t signal;

  // Device-visible semaphore in shared host/device memory.
  // The allocation is owned by the parent semaphore pool.
  IREE_AMDGPU_DEVICE_PTR iree_hal_amdgpu_device_semaphore_t* device_semaphore;

  // Release callback that handles deallocation.
  iree_hal_amdgpu_internal_semaphore_release_callback_t release_callback;
} iree_hal_amdgpu_internal_semaphore_t;

// Initializes an internal semaphore in-place with a 0 ref count.
// The owning pool must increment the ref count to 1 before returning the
// semaphore to users.
iree_status_t iree_hal_amdgpu_internal_semaphore_initialize(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    iree_hal_amdgpu_semaphore_options_t options,
    iree_hal_semaphore_flags_t flags,
    IREE_AMDGPU_DEVICE_PTR iree_hal_amdgpu_device_semaphore_t* device_semaphore,
    iree_hal_amdgpu_internal_semaphore_release_callback_t release_callback,
    iree_hal_amdgpu_internal_semaphore_t* out_semaphore);

// Deinitializes an internal semaphore in-place assuming it has a 0 ref count.
void iree_hal_amdgpu_internal_semaphore_deinitialize(
    iree_hal_amdgpu_internal_semaphore_t* semaphore);

// Returns true if |semaphore| is an iree_hal_amdgpu_internal_semaphore_t.
bool iree_hal_amdgpu_internal_semaphore_isa(iree_hal_semaphore_t* semaphore);

// Resets |semaphore| to |initial_value| as if it had just been allocated.
void iree_hal_amdgpu_internal_semaphore_reset(
    iree_hal_amdgpu_internal_semaphore_t* semaphore, uint64_t initial_value);

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_external_semaphore_t
//===----------------------------------------------------------------------===//

// TODO(benvanik): external imported semaphore wrapper.
typedef uint64_t iree_hal_amdgpu_external_semaphore_t;

//===----------------------------------------------------------------------===//
// Semaphore Operations
//===----------------------------------------------------------------------===//

// Returns a device-side semaphore handle for the provided HAL semaphore.
// Fails if there is no corresponding device-side handle (such as with a
// semaphore from another HAL device). Such semaphores must be imported using
// iree_hal_device_import_semaphore.
iree_status_t iree_hal_amdgpu_semaphore_handle(
    iree_hal_semaphore_t* semaphore,
    IREE_AMDGPU_DEVICE_PTR iree_hal_amdgpu_device_semaphore_t** out_handle);

// Returns the HSA signal for the provided HAL semaphore.
iree_status_t iree_hal_amdgpu_semaphore_hsa_signal(
    iree_hal_semaphore_t* base_semaphore, hsa_signal_t* out_signal);

// Polls |base_semaphore| and returns its current value in |out_current_value|.
// Returns ABORTED if the semaphore is in a failure state.
iree_status_t iree_hal_amdgpu_poll_semaphore(
    iree_hal_semaphore_t* base_semaphore, uint64_t* out_current_value);

// Polls |semaphore_list| and returns either OK or DEADLINE_EXCEEDED if
// satisfied or unsatisfied at the time the method is called.
// Returns ABORTED if any semaphore is in a failure state.
iree_status_t iree_hal_amdgpu_poll_semaphores(
    iree_hal_wait_mode_t wait_mode,
    const iree_hal_semaphore_list_t semaphore_list);

// iree_hal_device_wait_semaphores implementation.
iree_status_t iree_hal_amdgpu_wait_semaphores(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    iree_hal_amdgpu_semaphore_options_t options, iree_hal_wait_mode_t wait_mode,
    const iree_hal_semaphore_list_t semaphore_list, iree_timeout_t timeout);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_SEMAPHORE_H_
