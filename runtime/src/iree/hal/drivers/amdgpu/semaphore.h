// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_SEMAPHORE_H_
#define IREE_HAL_DRIVERS_AMDGPU_SEMAPHORE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/amdgpu/device/semaphore.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"

typedef struct iree_hal_amdgpu_internal_semaphore_t
    iree_hal_amdgpu_internal_semaphore_t;

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

typedef struct iree_hal_amdgpu_internal_semaphore_t {
  iree_hal_resource_t resource;  // must be at 0

  // HSA runtime API.
  const iree_hal_amdgpu_libhsa_t* libhsa;

  // Flags controlling semaphore behavior.
  iree_hal_semaphore_flags_t flags;

  // HSA signal handle.
  hsa_signal_t signal;

  // Device-side semaphore in shared host/device memory.
  // The allocation is owned by the parent semaphore pool.
  iree_hal_amdgpu_device_semaphore_t* device_semaphore;

  // Release callback that handles deallocation.
  iree_hal_amdgpu_internal_semaphore_release_callback_t release_callback;
} iree_hal_amdgpu_internal_semaphore_t;

// Initializes an internal semaphore in-place with a 0 ref count.
// The owning pool must increment the ref count to 1 before returning the
// semaphore to users.
iree_status_t iree_hal_amdgpu_internal_semaphore_initialize(
    const iree_hal_amdgpu_libhsa_t* libhsa, iree_hal_semaphore_flags_t flags,
    iree_hal_amdgpu_device_semaphore_t* device_semaphore,
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

#endif  // IREE_HAL_DRIVERS_AMDGPU_SEMAPHORE_H_
