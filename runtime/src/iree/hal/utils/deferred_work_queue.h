// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_UTILS_WORK_QUEUE_H_
#define IREE_HAL_UTILS_WORK_QUEUE_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/hal/api.h"
#include "iree/hal/utils/semaphore_base.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_deferred_work_queue_t iree_hal_deferred_work_queue_t;

typedef struct iree_hal_deferred_work_queue_device_interface_vtable_t
    iree_hal_deferred_work_queue_device_interface_vtable_t;

// This interface is used to allow the deferred work queue to interact with
// a specific driver.
// Calls to this vtable may be made from the deferred work queue on
// multile threads simultaneously and so these functions must be thread
// safe.
// Calls to this interface will either come from a thread that has had
// bind_to_thread called on it or as a side-effect from one of the public
// functions on the deferred work queue.
typedef struct iree_hal_deferred_work_queue_device_interface_t {
  const iree_hal_deferred_work_queue_device_interface_vtable_t* _vtable;
} iree_hal_deferred_work_queue_device_interface_t;

typedef struct iree_hal_deferred_work_queue_device_interface_vtable_t {
  void (*destroy)(iree_hal_deferred_work_queue_device_interface_t*);
  // Bind the device work queue to a thread. May be simulatneously
  // bound to multiple threads.
  iree_status_t (*bind_to_thread)(
      iree_hal_deferred_work_queue_device_interface_t*);

  // Create a native device event.
  iree_status_t (*create_native_event)(
      iree_hal_deferred_work_queue_device_interface_t*, void**);

  // Wait on a native device event.
  iree_status_t (*wait_native_event)(
      iree_hal_deferred_work_queue_device_interface_t*, void*);

  // Record a native device event.
  iree_status_t (*record_native_event)(
      iree_hal_deferred_work_queue_device_interface_t*, void*);

  // Synchronize the thread on a native device event.
  iree_status_t (*synchronize_native_event)(
      iree_hal_deferred_work_queue_device_interface_t*, void*);

  // Destroy a native device event.
  iree_status_t (*destroy_native_event)(
      iree_hal_deferred_work_queue_device_interface_t*, void*);

  // Acquire a native device event for the given timepoint.
  iree_status_t (*semaphore_acquire_timepoint_device_signal_native_event)(
      iree_hal_deferred_work_queue_device_interface_t*,
      struct iree_hal_semaphore_t*, uint64_t, void**);

  // Acquire a mixed host/device event for the given timepoint.
  bool (*acquire_host_wait_event)(
      iree_hal_deferred_work_queue_device_interface_t*,
      struct iree_hal_semaphore_t*, uint64_t, void**);

  // Release a mixed host/device event for the given timepoint.
  void (*release_wait_event)(iree_hal_deferred_work_queue_device_interface_t*,
                             void*);

  // Return a device-side event from the given host/device event.
  void* (*native_event_from_wait_event)(
      iree_hal_deferred_work_queue_device_interface_t*, void*);

  // Create a command buffer to be used to record a submitted
  // iree_hal_deferred_command_buffer.
  iree_status_t (*create_command_buffer_for_deferred)(
      iree_hal_deferred_work_queue_device_interface_t*,
      iree_hal_command_buffer_mode_t, iree_hal_command_category_t,
      iree_hal_command_buffer_t**);

  // Submit a command buffer to the device.
  iree_status_t (*submit_command_buffer)(
      iree_hal_deferred_work_queue_device_interface_t*,
      iree_hal_command_buffer_t*);
} iree_hal_deferred_work_queue_device_interface_vtable_t;

iree_status_t iree_hal_deferred_work_queue_create(
    iree_hal_deferred_work_queue_device_interface_t* symbols,
    iree_arena_block_pool_t* block_pool, iree_allocator_t host_allocator,
    iree_hal_deferred_work_queue_t** out_queue);

void iree_hal_deferred_work_queue_destroy(
    iree_hal_deferred_work_queue_t* queue);

typedef void(IREE_API_PTR* iree_hal_deferred_work_queue_cleanup_callback_t)(
    void* user_data);

// Enque command buffer submissions into the work queue to be executed
// once all semaphores have been satisfied.
iree_status_t iree_hal_deferred_work_queue_enque(
    iree_hal_deferred_work_queue_t* deferred_work_queue,
    iree_hal_deferred_work_queue_cleanup_callback_t cleanup_callback,
    void* callback_userdata,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_host_size_t command_buffer_count,
    iree_hal_command_buffer_t* const* command_buffers,
    iree_hal_buffer_binding_table_t const* binding_tables);

// Attempt to advance the work queue by processing using
// the current thread, rather than the worker thread.
iree_status_t iree_hal_deferred_work_queue_issue(
    iree_hal_deferred_work_queue_t* deferred_work_queue);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  //  IREE_HAL_UTILS_WORK_QUEUE_H_
