// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_SEMAPHORE_H_
#define IREE_HAL_DRIVERS_AMDGPU_SEMAPHORE_H_

#include "iree/async/semaphore.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_semaphore_t
//===----------------------------------------------------------------------===//

// Creates an AMDGPU HAL semaphore backed by an embedded async semaphore.
//
// Signal, query, and wait all delegate to the async semaphore infrastructure.
// The semaphore embeds iree_async_semaphore_t at offset 0 for toll-free
// bridging between HAL and async layers.
iree_status_t iree_hal_amdgpu_semaphore_create(
    iree_async_proactor_t* proactor, uint64_t initial_value,
    iree_allocator_t host_allocator, iree_hal_semaphore_t** out_semaphore);

// Returns true if |semaphore| is an AMDGPU semaphore.
bool iree_hal_amdgpu_semaphore_isa(iree_hal_semaphore_t* semaphore);

#endif  // IREE_HAL_DRIVERS_AMDGPU_SEMAPHORE_H_
