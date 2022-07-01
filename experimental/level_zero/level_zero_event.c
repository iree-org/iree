// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/level_zero/level_zero_event.h"

#include <stddef.h>

#include "iree/base/api.h"
#include "iree/base/tracing.h"

// Dummy events for now, don't do anything.
typedef struct iree_hal_level_zero_event_t {
  iree_hal_resource_t resource;
  iree_hal_level_zero_context_wrapper_t* context_wrapper;
  ze_event_handle_t handle;
} iree_hal_level_zero_event_t;

static const iree_hal_event_vtable_t iree_hal_level_zero_event_vtable;

static iree_hal_level_zero_event_t* iree_hal_level_zero_event_cast(
    iree_hal_event_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_level_zero_event_vtable);
  return (iree_hal_level_zero_event_t*)base_value;
}

iree_status_t iree_hal_level_zero_event_create(
    iree_hal_level_zero_context_wrapper_t* context_wrapper,
    ze_event_pool_handle_t event_pool,
    iree_hal_event_t** out_event) {
  IREE_ASSERT_ARGUMENT(context_wrapper);
  IREE_ASSERT_ARGUMENT(out_event);
  *out_event = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_level_zero_event_t* event = NULL;
  iree_status_t status = iree_allocator_malloc(context_wrapper->host_allocator,
                                               sizeof(*event), (void**)&event);
  if (iree_status_is_ok(status)) {
    iree_hal_resource_initialize(&iree_hal_level_zero_event_vtable,
                                 &event->resource);
    ze_event_desc_t event_desc = {};
    event_desc.stype = ZE_STRUCTURE_TYPE_EVENT_DESC;
    event_desc.signal = ZE_EVENT_SCOPE_FLAG_HOST;
    event_desc.wait = ZE_EVENT_SCOPE_FLAG_HOST;
    ze_event_handle_t handle;
    LEVEL_ZERO_RETURN_IF_ERROR(
        context_wrapper->syms,
        zeEventCreate(event_pool, &event_desc, &handle), "zeEventCreate");
    event->handle = handle;
    event->context_wrapper = context_wrapper;
    *out_event = (iree_hal_event_t*)event;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_level_zero_event_destroy(iree_hal_event_t* base_event) {
  iree_hal_level_zero_event_t* event =
      iree_hal_level_zero_event_cast(base_event);
  iree_allocator_t host_allocator = event->context_wrapper->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);
  LEVEL_ZERO_IGNORE_ERROR(event->context_wrapper->syms,
                        zeEventDestroy(event->handle));
  iree_allocator_free(host_allocator, event);

  IREE_TRACE_ZONE_END(z0);
}

ze_event_handle_t iree_hal_level_zero_event_handle(
    const iree_hal_event_t* base_event) {
  return ((const iree_hal_level_zero_event_t*)base_event)->handle;
}

static const iree_hal_event_vtable_t iree_hal_level_zero_event_vtable = {
    .destroy = iree_hal_level_zero_event_destroy,
};
