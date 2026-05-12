// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/local_task/task_semaphore.h"

#include <stdbool.h>
#include <stddef.h>
#include <string.h>

#include "iree/async/semaphore.h"

//===----------------------------------------------------------------------===//
// iree_hal_task_semaphore_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_task_semaphore_t {
  iree_async_semaphore_t async;
  iree_allocator_t host_allocator;
} iree_hal_task_semaphore_t;

static const iree_hal_semaphore_vtable_t iree_hal_task_semaphore_vtable;

static iree_hal_task_semaphore_t* iree_hal_task_semaphore_cast(
    iree_hal_semaphore_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_task_semaphore_vtable);
  return (iree_hal_task_semaphore_t*)base_value;
}

iree_status_t iree_hal_task_semaphore_create(
    iree_async_proactor_t* proactor, uint64_t initial_value,
    iree_allocator_t host_allocator, iree_hal_semaphore_t** out_semaphore) {
  IREE_ASSERT_ARGUMENT(proactor);
  IREE_ASSERT_ARGUMENT(out_semaphore);
  *out_semaphore = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_task_semaphore_t* semaphore = NULL;
  iree_host_size_t frontier_offset = 0, total_size = 0;
  iree_status_t status = iree_async_semaphore_layout(
      sizeof(*semaphore), 0, &frontier_offset, &total_size);
  if (iree_status_is_ok(status)) {
    status =
        iree_allocator_malloc(host_allocator, total_size, (void**)&semaphore);
  }
  if (iree_status_is_ok(status)) {
    iree_async_semaphore_initialize(
        (const iree_async_semaphore_vtable_t*)&iree_hal_task_semaphore_vtable,
        proactor, initial_value, frontier_offset, 0, &semaphore->async);
    semaphore->host_allocator = host_allocator;

    *out_semaphore = iree_hal_semaphore_cast(&semaphore->async);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_task_semaphore_destroy(
    iree_async_semaphore_t* base_semaphore) {
  iree_hal_task_semaphore_t* semaphore =
      iree_hal_task_semaphore_cast(iree_hal_semaphore_cast(base_semaphore));
  iree_allocator_t host_allocator = semaphore->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_async_semaphore_deinitialize(&semaphore->async);
  iree_allocator_free(host_allocator, semaphore);

  IREE_TRACE_ZONE_END(z0);
}

bool iree_hal_task_semaphore_isa(iree_hal_semaphore_t* semaphore) {
  return iree_hal_resource_is((const iree_hal_resource_t*)semaphore,
                              &iree_hal_task_semaphore_vtable);
}

static uint64_t iree_hal_task_semaphore_query(
    iree_async_semaphore_t* base_semaphore) {
  iree_async_semaphore_t* async_sem = (iree_async_semaphore_t*)base_semaphore;

  // Both fields are atomic — fully lock-free query.
  iree_status_t failure = (iree_status_t)iree_atomic_load(
      &async_sem->failure_status, iree_memory_order_acquire);
  if (!iree_status_is_ok(failure)) {
    return iree_hal_status_as_semaphore_failure(failure);
  }
  return (uint64_t)iree_atomic_load(&async_sem->timeline_value,
                                    iree_memory_order_acquire);
}

static iree_status_t iree_hal_task_semaphore_signal(
    iree_async_semaphore_t* base_semaphore, uint64_t new_value,
    const iree_async_frontier_t* frontier) {
  // Advance the timeline (CAS) and merge frontier.
  iree_status_t status = iree_async_semaphore_advance_timeline(
      base_semaphore, new_value, frontier);
  if (!iree_status_is_ok(status)) return status;

  // Dispatch satisfied timepoints.
  iree_async_semaphore_dispatch_timepoints(base_semaphore, new_value);

  return iree_ok_status();
}

static iree_status_t iree_hal_task_semaphore_wait(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    iree_timeout_t timeout, iree_async_wait_flags_t flags) {
  // Delegate to the centralized async semaphore wait which uses a stack-local
  // futex-based notification — no event pool allocation or eventfd overhead.
  return iree_async_semaphore_multi_wait(
      IREE_ASYNC_WAIT_MODE_ALL, (iree_async_semaphore_t**)&base_semaphore,
      &value, 1, timeout, flags, iree_allocator_system());
}

static iree_status_t iree_hal_task_semaphore_import_timepoint(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    iree_hal_queue_affinity_t queue_affinity,
    iree_hal_external_timepoint_t external_timepoint) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "timepoint import is not yet implemented");
}

static iree_status_t iree_hal_task_semaphore_export_timepoint(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    iree_hal_queue_affinity_t queue_affinity,
    iree_hal_external_timepoint_type_t requested_type,
    iree_hal_external_timepoint_flags_t requested_flags,
    iree_hal_external_timepoint_t* IREE_RESTRICT out_external_timepoint) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "timepoint export is not yet implemented");
}

static const iree_hal_semaphore_vtable_t iree_hal_task_semaphore_vtable = {
    .async =
        {
            .destroy = iree_hal_task_semaphore_destroy,
            .query = iree_hal_task_semaphore_query,
            .signal = iree_hal_task_semaphore_signal,
        },
    .wait = iree_hal_task_semaphore_wait,
    .import_timepoint = iree_hal_task_semaphore_import_timepoint,
    .export_timepoint = iree_hal_task_semaphore_export_timepoint,
};
