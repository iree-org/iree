// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "event.h"

#include <stddef.h>

typedef struct iree_hal_nl_task_event_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
} iree_hal_nl_task_event_t;

static const iree_hal_event_vtable_t iree_hal_nl_task_event_vtable;

static iree_hal_nl_task_event_t* iree_hal_nl_task_event_cast(
    iree_hal_event_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_nl_task_event_vtable);
  return (iree_hal_nl_task_event_t*)base_value;
}

iree_status_t iree_hal_nl_task_event_create(
    iree_hal_queue_affinity_t queue_affinity, iree_hal_event_flags_t flags,
    iree_allocator_t host_allocator, iree_hal_event_t** out_event) {
  IREE_ASSERT_ARGUMENT(out_event);
  *out_event = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_nl_task_event_t* event = NULL;
  iree_status_t status =
      iree_allocator_malloc(host_allocator, sizeof(*event), (void**)&event);
  if (iree_status_is_ok(status)) {
    iree_hal_resource_initialize(&iree_hal_nl_task_event_vtable, &event->resource);
    event->host_allocator = host_allocator;
    *out_event = (iree_hal_event_t*)event;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_nl_task_event_destroy(iree_hal_event_t* base_event) {
  iree_hal_nl_task_event_t* event = iree_hal_nl_task_event_cast(base_event);
  iree_allocator_t host_allocator = event->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_free(host_allocator, event);

  IREE_TRACE_ZONE_END(z0);
}

static const iree_hal_event_vtable_t iree_hal_nl_task_event_vtable = {
    .destroy = iree_hal_nl_task_event_destroy,
};
