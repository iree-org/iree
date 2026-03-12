// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/client/semaphore.h"

static const iree_hal_semaphore_vtable_t
    iree_hal_remote_client_semaphore_vtable;

typedef struct iree_hal_remote_client_semaphore_t {
  // Embedded at offset 0 for toll-free bridging with iree_async_semaphore_t.
  iree_async_semaphore_t async;
  iree_allocator_t host_allocator;
} iree_hal_remote_client_semaphore_t;

static iree_hal_remote_client_semaphore_t*
iree_hal_remote_client_semaphore_cast(iree_hal_semaphore_t* base_semaphore) {
  IREE_HAL_ASSERT_TYPE(base_semaphore,
                       &iree_hal_remote_client_semaphore_vtable);
  return (iree_hal_remote_client_semaphore_t*)base_semaphore;
}

iree_status_t iree_hal_remote_client_semaphore_create(
    iree_async_proactor_t* proactor, uint64_t initial_value,
    iree_allocator_t host_allocator, iree_hal_semaphore_t** out_semaphore) {
  IREE_ASSERT_ARGUMENT(out_semaphore);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_semaphore = NULL;

  // Compute allocation layout: struct + trailing frontier storage.
  iree_host_size_t frontier_offset = 0;
  iree_host_size_t total_size = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_async_semaphore_layout(
              sizeof(iree_hal_remote_client_semaphore_t),
              /*frontier_capacity=*/0, &frontier_offset, &total_size));

  iree_hal_remote_client_semaphore_t* semaphore = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator, total_size, (void**)&semaphore));
  memset(semaphore, 0, total_size);

  // Initialize the embedded async semaphore (sets ref_count, vtable, etc.).
  iree_async_semaphore_initialize(
      (const iree_async_semaphore_vtable_t*)&iree_hal_remote_client_semaphore_vtable,
      proactor, initial_value, frontier_offset, /*frontier_capacity=*/0,
      &semaphore->async);

  semaphore->host_allocator = host_allocator;

  *out_semaphore = iree_hal_semaphore_cast(&semaphore->async);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_remote_client_semaphore_destroy(
    iree_async_semaphore_t* base_semaphore) {
  iree_hal_remote_client_semaphore_t* semaphore =
      iree_hal_remote_client_semaphore_cast(
          iree_hal_semaphore_cast(base_semaphore));
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_t host_allocator = semaphore->host_allocator;
  iree_async_semaphore_deinitialize(&semaphore->async);
  iree_allocator_free(host_allocator, semaphore);

  IREE_TRACE_ZONE_END(z0);
}

static uint64_t iree_hal_remote_client_semaphore_query(
    iree_async_semaphore_t* base_semaphore) {
  iree_status_t failure = (iree_status_t)iree_atomic_load(
      &base_semaphore->failure_status, iree_memory_order_acquire);
  if (!iree_status_is_ok(failure)) {
    return iree_hal_status_as_semaphore_failure(failure);
  }
  return (uint64_t)iree_atomic_load(&base_semaphore->timeline_value,
                                    iree_memory_order_acquire);
}

static iree_status_t iree_hal_remote_client_semaphore_signal(
    iree_async_semaphore_t* base_semaphore, uint64_t new_value,
    const iree_async_frontier_t* frontier) {
  // Advance the timeline and merge the frontier. This is the shared helper
  // that handles the CAS loop, monotonicity check, and frontier merge.
  IREE_RETURN_IF_ERROR(iree_async_semaphore_advance_timeline(
      base_semaphore, new_value, frontier));

  // Dispatch any timepoints that are now satisfied.
  iree_async_semaphore_dispatch_timepoints(base_semaphore, new_value);

  return iree_ok_status();
}

static iree_status_t iree_hal_remote_client_semaphore_wait(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    iree_timeout_t timeout, iree_async_wait_flags_t flags) {
  // Delegate to the centralized async semaphore wait which uses a stack-local
  // futex-based notification. Same pattern as task_semaphore_wait.
  iree_async_semaphore_t* async_semaphore =
      (iree_async_semaphore_t*)base_semaphore;
  return iree_async_semaphore_multi_wait(IREE_ASYNC_WAIT_MODE_ALL,
                                         &async_semaphore, &value, 1, timeout,
                                         flags, iree_allocator_system());
}

static iree_status_t iree_hal_remote_client_semaphore_import_timepoint(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    iree_hal_queue_affinity_t queue_affinity,
    iree_hal_external_timepoint_t external_timepoint) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "timepoint import not supported on remote client "
                          "semaphore");
}

static iree_status_t iree_hal_remote_client_semaphore_export_timepoint(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    iree_hal_queue_affinity_t queue_affinity,
    iree_hal_external_timepoint_type_t requested_type,
    iree_hal_external_timepoint_flags_t requested_flags,
    iree_hal_external_timepoint_t* IREE_RESTRICT out_external_timepoint) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "timepoint export not supported on remote client "
                          "semaphore");
}

static const iree_hal_semaphore_vtable_t
    iree_hal_remote_client_semaphore_vtable = {
        .async =
            {
                .destroy = iree_hal_remote_client_semaphore_destroy,
                .query = iree_hal_remote_client_semaphore_query,
                .signal = iree_hal_remote_client_semaphore_signal,
            },
        .wait = iree_hal_remote_client_semaphore_wait,
        .import_timepoint = iree_hal_remote_client_semaphore_import_timepoint,
        .export_timepoint = iree_hal_remote_client_semaphore_export_timepoint,
};
