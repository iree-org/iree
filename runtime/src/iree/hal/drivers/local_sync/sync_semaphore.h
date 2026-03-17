// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_LOCAL_SYNC_SYNC_SEMAPHORE_H_
#define IREE_HAL_DRIVERS_LOCAL_SYNC_SYNC_SEMAPHORE_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"

typedef struct iree_async_proactor_t iree_async_proactor_t;

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_sync_semaphore_t
//===----------------------------------------------------------------------===//

// Creates a semaphore that allows for ordering of operations on the local host.
// Waits use the centralized async semaphore futex-based mechanism. Signals
// dispatch timepoints directly. Suitable for synchronous backends where all
// work is executed inline on the calling thread.
// |proactor| is borrowed from the device's proactor pool and must outlive the
// semaphore.
iree_status_t iree_hal_sync_semaphore_create(
    iree_async_proactor_t* proactor, iree_hal_queue_affinity_t queue_affinity,
    uint64_t initial_value, iree_hal_semaphore_flags_t flags,
    iree_allocator_t host_allocator, iree_hal_semaphore_t** out_semaphore);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_LOCAL_SYNC_SYNC_SEMAPHORE_H_
