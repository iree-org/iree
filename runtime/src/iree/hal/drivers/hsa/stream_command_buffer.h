// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_HSA_STREAM_COMMAND_BUFFER_H_
#define IREE_HAL_DRIVERS_HSA_STREAM_COMMAND_BUFFER_H_

#include "iree/base/internal/arena.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/hsa/dynamic_symbols.h"
#include "iree/hal/drivers/hsa/hsa_headers.h"
#include "iree/hal/utils/stream_tracing.h"

typedef struct iree_hal_hsa_per_device_info_t iree_hal_hsa_per_device_info_t;

// Creates command buffer that immediately issues commands against the HSA
// queue.
iree_status_t iree_hal_hsa_stream_command_buffer_create(
    iree_hal_allocator_t* device_allocator,
    const iree_hal_hsa_dynamic_symbols_t* hsa_symbols,
    iree_hal_stream_tracing_context_t* tracing_context,
    iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_hal_hsa_per_device_info_t* device_info,
    iree_arena_block_pool_t* block_pool, iree_allocator_t host_allocator,
    iree_hal_command_buffer_t** out_command_buffer);

// Returns true if |command_buffer| is an HSA stream-based command buffer.
bool iree_hal_hsa_stream_command_buffer_isa(
    iree_hal_command_buffer_t* command_buffer);

#endif  // IREE_HAL_DRIVERS_HSA_STREAM_COMMAND_BUFFER_H_

