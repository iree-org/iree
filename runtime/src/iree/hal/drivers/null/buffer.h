// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_NULL_BUFFER_H_
#define IREE_HAL_DRIVERS_NULL_BUFFER_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

// Wraps a {Null} allocation in an iree_hal_buffer_t.
iree_status_t iree_hal_null_buffer_wrap(
    iree_hal_allocator_t* allocator, iree_hal_memory_type_t memory_type,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage, iree_device_size_t allocation_size,
    iree_device_size_t byte_offset, iree_device_size_t byte_length,
    iree_hal_buffer_release_callback_t release_callback,
    iree_allocator_t host_allocator, iree_hal_buffer_t** out_buffer);

#endif  // IREE_HAL_DRIVERS_NULL_BUFFER_H_
