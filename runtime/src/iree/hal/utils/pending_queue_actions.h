// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_UTILS_PENDING_QUEUE_ACTIONS_H_
#define IREE_HAL_UTILS_PENDING_QUEUE_ACTIONS_H_

#include <stdbool.h>

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/hal/api.h"
#include "iree/hal/utils/event_pool.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Implementation type for GPU streams.
// It maps to CUevent for CUDA and hipEvent_t for HIP.
typedef struct iree_hal_stream_impl_t* iree_hal_stream_impl_t;
// Implementation type for GPU graph executable.
// It maps to CUgraphExec for CUDA and hipGraphExec_t for HIP.
typedef struct iree_hal_graph_executable_impl_t*
    iree_hal_graph_executable_impl_t;

typedef void(IREE_API_PTR* iree_hal_host_fn_impl_t)(void* user_data);

// Symbol table for GPU stream related implementation APIs.
typedef struct iree_hal_stream_impl_symtable_t {
  iree_status_t(IREE_API_PTR* record_event)(void* user_data,
                                            iree_hal_stream_impl_t stream,
                                            iree_hal_event_impl_t event);
  iree_status_t(IREE_API_PTR* wait_event)(void* user_data,
                                          iree_hal_stream_impl_t stream,
                                          iree_hal_event_impl_t event);
  iree_status_t(IREE_API_PTR* launch_graph)(
      void* user_data, iree_hal_stream_impl_t stream,
      iree_hal_graph_executable_impl_t graph);
  iree_status_t(IREE_API_PTR* launch_host_fn)(void* user_data,
                                              iree_hal_stream_impl_t stream,
                                              iree_hal_host_fn_impl_t host_fn,
                                              void* host_fn_user_data);
} iree_hal_stream_impl_symtable_t;

// Symbol table for GPU command_buffer related implementation APIs.
typedef struct iree_hal_command_buffer_impl_symtable_t {
  bool(IREE_API_PTR* is_graph_command_buffer)(
      iree_hal_command_buffer_t* command_buffer);
  iree_hal_graph_executable_impl_t(IREE_API_PTR* graph_command_buffer_handle)(
      iree_hal_command_buffer_t* command_buffer);
  iree_status_t(IREE_API_PTR* create_stream_command_buffer)(
      iree_hal_device_t* device, iree_hal_command_buffer_mode_t mode,
      iree_hal_command_category_t command_categories,
      iree_host_size_t binding_capacity,
      iree_hal_command_buffer_t** out_command_buffer);
} iree_hal_command_buffer_impl_symtable_t;

// A data structure to manage pending queue actions (kernel launches and async
// allocations).
//
// This is needed in order to satisfy queue action dependencies. IREE uses HAL
// semaphore as the unified mechanism for synchronization directions including
// host to host, host to device, devie to device, and device to host. Plus, it
// allows wait before signal. These flexible capabilities are not all supported
// by CUevent objects. Therefore, we need supporting data structures to
// implement them on top of CUevent objects. Thus this pending queue actions.
//
// This buffers pending queue actions and their associated resources. It
// provides an API to advance the wait list on demand--queue actions are
// released to the GPU when all their wait semaphores are signaled past the
// desired value, or we can have a CUevent already recorded to some CUDA
// stream to wait on.
//
// Thread-safe; multiple threads may enqueue workloads.
typedef struct iree_hal_cuda2_pending_queue_actions_t
    iree_hal_cuda2_pending_queue_actions_t;

// Creates a pending actions queue.
//
// |stream_symbols|, |command_buffer_symbols|, and |symbol_user_data| must
// outlive the created queue.
iree_status_t iree_hal_cuda2_pending_queue_actions_create(
    const iree_hal_stream_impl_symtable_t* stream_symbols,
    const iree_hal_command_buffer_impl_symtable_t* command_buffer_symbols,
    void* stream_symbol_user_data, iree_arena_block_pool_t* block_pool,
    iree_allocator_t host_allocator,
    iree_hal_cuda2_pending_queue_actions_t** out_actions);

// Destroys the pending |actions| queue.
void iree_hal_cuda2_pending_queue_actions_destroy(iree_hal_resource_t* actions);

// Enqueues the given list of |command_buffers| that waits on
// |wait_semaphore_list| and signals |signal_semaphore_lsit|.
iree_status_t iree_hal_cuda2_pending_queue_actions_enqueue_execution(
    iree_hal_device_t* device, iree_hal_stream_impl_t dispatch_stream,
    iree_hal_stream_impl_t callback_stream,
    iree_hal_cuda2_pending_queue_actions_t* actions,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_host_size_t command_buffer_count,
    iree_hal_command_buffer_t* const* command_buffers);

// Tries to scan the pending actions and release ready ones to the GPU.
iree_status_t iree_hal_cuda2_pending_queue_actions_issue(
    iree_hal_cuda2_pending_queue_actions_t* actions);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_UTILS_PENDING_QUEUE_ACTIONS_H_
