// Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.
// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_HSA_PENDING_QUEUE_ACTIONS_H_
#define IREE_EXPERIMENTAL_HSA_PENDING_QUEUE_ACTIONS_H_

#include "experimental/hsa/dynamic_symbols.h"
#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// A data structure to manage pending queue actions
typedef struct iree_hal_hsa_pending_queue_actions_t
    iree_hal_hsa_pending_queue_actions_t;

// Creates a pending actions queue.
iree_status_t iree_hal_hsa_pending_queue_actions_create(
    const iree_hal_hsa_dynamic_symbols_t* symbols,
    iree_arena_block_pool_t* block_pool, iree_allocator_t host_allocator,
    iree_hal_hsa_pending_queue_actions_t** out_actions);

// Destroys the pending |actions| queue.
void iree_hal_hsa_pending_queue_actions_destroy(iree_hal_resource_t* actions);

// Callback to execute user code after action completion but before resource
// releasing.
//
// Data behind |user_data| must remain alive before the action is released.
typedef void(IREE_API_PTR* iree_hal_hsa_pending_action_cleanup_callback_t)(
    void* user_data);

// Enqueues the given list of |command_buffers| that waits on
// |wait_semaphore_list| and signals |signal_semaphore_lsit|.
//
// |cleanup_callback|, if not NULL, will run after the action completes but
// before releasing all retained resources.
iree_status_t iree_hal_hsa_pending_queue_actions_enqueue_execution(
    iree_hal_device_t* device, hsa_queue_t* dispatch_queue,
    iree_hal_hsa_pending_queue_actions_t* actions,
    iree_hal_hsa_pending_action_cleanup_callback_t cleanup_callback,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_host_size_t command_buffer_count,
    iree_hal_command_buffer_t* const* command_buffers);

// Tries to scan the pending actions and release ready ones to the GPU.
iree_status_t iree_hal_hsa_pending_queue_actions_issue(
    iree_hal_hsa_pending_queue_actions_t* actions);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_EXPERIMENTAL_HSA_PENDING_QUEUE_ACTIONS_H_
