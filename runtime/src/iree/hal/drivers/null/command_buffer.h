// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_NULL_COMMAND_BUFFER_H_
#define IREE_HAL_DRIVERS_NULL_COMMAND_BUFFER_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

// Creates {Null} command buffer.
iree_status_t iree_hal_null_command_buffer_create(
    iree_hal_allocator_t* device_allocator, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_allocator_t host_allocator,
    iree_hal_command_buffer_t** out_command_buffer);

// Returns true if |command_buffer| is a {Null} command buffer.
bool iree_hal_null_command_buffer_isa(
    iree_hal_command_buffer_t* command_buffer);

#endif  // IREE_HAL_DRIVERS_NULL_COMMAND_BUFFER_H_
