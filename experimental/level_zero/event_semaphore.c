// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/level_zero/event_semaphore.h"

#include <stddef.h>

#include "iree/base/api.h"
#include "iree/base/tracing.h"
#include "iree/hal/utils/semaphore_base.h"

typedef struct iree_hal_level_zero_semaphore_t {
  iree_hal_semaphore_t base;
  iree_hal_level_zero_context_wrapper_t* context;
  iree_atomic_int64_t value;
} iree_hal_level_zero_semaphore_t;

static const iree_hal_semaphore_vtable_t iree_hal_level_zero_semaphore_vtable;

static iree_hal_level_zero_semaphore_t* iree_hal_level_zero_semaphore_cast(
    iree_hal_semaphore_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_level_zero_semaphore_vtable);
  return (iree_hal_level_zero_semaphore_t*)base_value;
}

iree_status_t iree_hal_level_zero_semaphore_create(
    iree_hal_level_zero_context_wrapper_t* context, uint64_t initial_value,
    iree_hal_semaphore_t** out_semaphore) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_semaphore);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_level_zero_semaphore_t* semaphore = NULL;
  iree_status_t status = iree_allocator_malloc(
      context->host_allocator, sizeof(*semaphore), (void**)&semaphore);
  if (iree_status_is_ok(status)) {
    iree_hal_semaphore_initialize(&iree_hal_level_zero_semaphore_vtable,
                                  &semaphore->base);
    iree_atomic_store_int64(&semaphore->value, initial_value,
                            iree_memory_order_release);
    semaphore->context = context;
    *out_semaphore = &semaphore->base;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_level_zero_semaphore_destroy(
    iree_hal_semaphore_t* base_semaphore) {
  iree_hal_level_zero_semaphore_t* semaphore =
      iree_hal_level_zero_semaphore_cast(base_semaphore);
  iree_allocator_t host_allocator = semaphore->context->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_semaphore_deinitialize(&semaphore->base);
  iree_allocator_free(host_allocator, semaphore);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_level_zero_semaphore_query(
    iree_hal_semaphore_t* base_semaphore, uint64_t* out_value) {
  // TODO: Support semaphores completely.
  iree_hal_level_zero_semaphore_t* semaphore =
      iree_hal_level_zero_semaphore_cast(base_semaphore);
  // TODO: Support semaphores completely.
  *out_value =
      iree_atomic_load_int64(&semaphore->value, iree_memory_order_acquire);
  return iree_ok_status();
}

static iree_status_t iree_hal_level_zero_semaphore_signal(
    iree_hal_semaphore_t* base_semaphore, uint64_t new_value) {
  iree_hal_level_zero_semaphore_t* semaphore =
      iree_hal_level_zero_semaphore_cast(base_semaphore);
  // TODO: Support semaphores completely. Return OK currently as everything is
  // synchronized for each submit to allow things to run.
  iree_atomic_store_int64(&semaphore->value, new_value,
                          iree_memory_order_release);
  iree_hal_semaphore_poll(&semaphore->base);
  return iree_ok_status();
}

static void iree_hal_level_zero_semaphore_fail(
    iree_hal_semaphore_t* base_semaphore, iree_status_t status) {
  iree_hal_level_zero_semaphore_t* semaphore =
      iree_hal_level_zero_semaphore_cast(base_semaphore);
  // TODO: save status and mark timepoint as failed.
  iree_status_ignore(status);
  iree_hal_semaphore_poll(&semaphore->base);
}

static iree_status_t iree_hal_level_zero_semaphore_wait(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    iree_timeout_t timeout) {
  iree_hal_level_zero_semaphore_t* semaphore =
      iree_hal_level_zero_semaphore_cast(base_semaphore);
  // TODO: Support semaphores completely. Return OK currently as everything is
  // synchronized for each submit to allow things to run.
  iree_hal_semaphore_poll(&semaphore->base);
  return iree_ok_status();
}

static const iree_hal_semaphore_vtable_t iree_hal_level_zero_semaphore_vtable =
    {
        .destroy = iree_hal_level_zero_semaphore_destroy,
        .query = iree_hal_level_zero_semaphore_query,
        .signal = iree_hal_level_zero_semaphore_signal,
        .fail = iree_hal_level_zero_semaphore_fail,
        .wait = iree_hal_level_zero_semaphore_wait,
};
