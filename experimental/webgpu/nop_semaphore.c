// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/webgpu/nop_semaphore.h"

#include <stddef.h>

#include "iree/base/api.h"
#include "iree/base/tracing.h"

typedef struct iree_hal_webgpu_nop_semaphore_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  iree_atomic_int64_t value;
} iree_hal_webgpu_nop_semaphore_t;

extern const iree_hal_semaphore_vtable_t iree_hal_webgpu_nop_semaphore_vtable;

static iree_hal_webgpu_nop_semaphore_t* iree_hal_webgpu_nop_semaphore_cast(
    iree_hal_semaphore_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_webgpu_nop_semaphore_vtable);
  return (iree_hal_webgpu_nop_semaphore_t*)base_value;
}

iree_status_t iree_hal_webgpu_nop_semaphore_create(
    uint64_t initial_value, iree_allocator_t host_allocator,
    iree_hal_semaphore_t** out_semaphore) {
  IREE_ASSERT_ARGUMENT(out_semaphore);
  *out_semaphore = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_webgpu_nop_semaphore_t* semaphore = NULL;
  iree_status_t status = iree_allocator_malloc(
      host_allocator, sizeof(*semaphore), (void**)&semaphore);
  if (iree_status_is_ok(status)) {
    iree_hal_resource_initialize(&iree_hal_webgpu_nop_semaphore_vtable,
                                 &semaphore->resource);
    semaphore->host_allocator = host_allocator;
    iree_atomic_store_int64(&semaphore->value, initial_value,
                            iree_memory_order_seq_cst);
    *out_semaphore = (iree_hal_semaphore_t*)semaphore;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_webgpu_nop_semaphore_destroy(
    iree_hal_semaphore_t* base_semaphore) {
  iree_hal_webgpu_nop_semaphore_t* semaphore =
      iree_hal_webgpu_nop_semaphore_cast(base_semaphore);
  iree_allocator_t host_allocator = semaphore->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_free(host_allocator, semaphore);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_webgpu_nop_semaphore_query(
    iree_hal_semaphore_t* base_semaphore, uint64_t* out_value) {
  iree_hal_webgpu_nop_semaphore_t* semaphore =
      iree_hal_webgpu_nop_semaphore_cast(base_semaphore);
  *out_value =
      iree_atomic_load_int64(&semaphore->value, iree_memory_order_seq_cst);
  return iree_ok_status();
}

static iree_status_t iree_hal_webgpu_nop_semaphore_signal(
    iree_hal_semaphore_t* base_semaphore, uint64_t new_value) {
  iree_hal_webgpu_nop_semaphore_t* semaphore =
      iree_hal_webgpu_nop_semaphore_cast(base_semaphore);
  iree_atomic_store_int64(&semaphore->value, new_value,
                          iree_memory_order_seq_cst);
  return iree_ok_status();
}

static void iree_hal_webgpu_nop_semaphore_fail(
    iree_hal_semaphore_t* base_semaphore, iree_status_t status) {
  iree_status_ignore(status);
}

static iree_status_t iree_hal_webgpu_nop_semaphore_wait(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    iree_timeout_t timeout) {
  iree_hal_webgpu_nop_semaphore_t* semaphore =
      iree_hal_webgpu_nop_semaphore_cast(base_semaphore);
  uint64_t current_value =
      iree_atomic_load_int64(&semaphore->value, iree_memory_order_seq_cst);
  if (current_value < value) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "expected no-op semaphore to be signaled before wait");
  }
  return iree_ok_status();
}

const iree_hal_semaphore_vtable_t iree_hal_webgpu_nop_semaphore_vtable = {
    .destroy = iree_hal_webgpu_nop_semaphore_destroy,
    .query = iree_hal_webgpu_nop_semaphore_query,
    .signal = iree_hal_webgpu_nop_semaphore_signal,
    .fail = iree_hal_webgpu_nop_semaphore_fail,
    .wait = iree_hal_webgpu_nop_semaphore_wait,
};
