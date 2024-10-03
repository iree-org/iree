// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_SEMAPHORE_H_
#define IREE_HAL_DRIVERS_AMDGPU_SEMAPHORE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"

typedef struct iree_hal_amdgpu_device_semaphore_t
    iree_hal_amdgpu_device_semaphore_t;

typedef struct iree_hal_amdgpu_internal_semaphore_t
    iree_hal_amdgpu_internal_semaphore_t;
typedef struct iree_hal_amdgpu_system_t iree_hal_amdgpu_system_t;

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

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

// DO NOT SUBMIT document iree_hal_amdgpu_internal_semaphore_t
typedef struct iree_hal_amdgpu_internal_semaphore_t {
  iree_hal_resource_t resource;  // must be at 0

  // System this semaphore uses for interfacing with HSA.
  iree_hal_amdgpu_system_t* system;

  // Flags controlling semaphore behavior.
  iree_hal_semaphore_flags_t flags;

  // HSA signal handle. Contains the semaphore payload value.
  hsa_signal_t signal;

  // Device-side semaphore in shared host/device memory.
  // The allocation is owned by the parent semaphore pool.
  IREE_AMDGPU_DEVICE_PTR iree_hal_amdgpu_device_semaphore_t* device_semaphore;

  // Release callback that handles deallocation.
  iree_hal_amdgpu_internal_semaphore_release_callback_t release_callback;
} iree_hal_amdgpu_internal_semaphore_t;

// Initializes an internal semaphore in-place with a 0 ref count.
// The owning pool must increment the ref count to 1 before returning the
// semaphore to users.
iree_status_t iree_hal_amdgpu_internal_semaphore_initialize(
    iree_hal_amdgpu_system_t* system, iree_hal_semaphore_flags_t flags,
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

// DO NOT SUBMIT define external semaphores
typedef uint64_t iree_hal_amdgpu_external_semaphore_t;

// Notifies the external |semaphore| of a new |payload|.
// This is posted from the device scheduler as it signals semaphores.
iree_status_t iree_hal_amdgpu_external_semaphore_notify(
    iree_hal_amdgpu_external_semaphore_t* semaphore, uint64_t payload);

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
    iree_hal_amdgpu_system_t* system, iree_hal_wait_mode_t wait_mode,
    const iree_hal_semaphore_list_t semaphore_list, iree_timeout_t timeout);

#endif  // IREE_HAL_DRIVERS_AMDGPU_SEMAPHORE_H_
