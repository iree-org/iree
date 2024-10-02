// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_NULL_SEMAPHORE_H_
#define IREE_HAL_DRIVERS_NULL_SEMAPHORE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

//===----------------------------------------------------------------------===//
// iree_hal_null_semaphore_t
//===----------------------------------------------------------------------===//

// Creates a {Null} semaphore used for ordering queue operations and
// synchronizing between host/device and device/device.
iree_status_t iree_hal_null_semaphore_create(
    uint64_t initial_value, iree_hal_semaphore_flags_t flags,
    iree_allocator_t host_allocator, iree_hal_semaphore_t** out_semaphore);

#endif  // IREE_HAL_DRIVERS_NULL_SEMAPHORE_H_
