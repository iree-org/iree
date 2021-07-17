// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/rocm/descriptor_set_layout.h"

#include <stddef.h>

#include "iree/base/api.h"
#include "iree/base/tracing.h"

typedef struct iree_hal_rocm_descriptor_set_layout_t {
  iree_hal_resource_t resource;
  iree_hal_rocm_context_wrapper_t *context;
} iree_hal_rocm_descriptor_set_layout_t;

extern const iree_hal_descriptor_set_layout_vtable_t
    iree_hal_rocm_descriptor_set_layout_vtable;

static iree_hal_rocm_descriptor_set_layout_t *
iree_hal_rocm_descriptor_set_layout_cast(
    iree_hal_descriptor_set_layout_t *base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_rocm_descriptor_set_layout_vtable);
  return (iree_hal_rocm_descriptor_set_layout_t *)base_value;
}

iree_status_t iree_hal_rocm_descriptor_set_layout_create(
    iree_hal_rocm_context_wrapper_t *context,
    iree_hal_descriptor_set_layout_usage_type_t usage_type,
    iree_host_size_t binding_count,
    const iree_hal_descriptor_set_layout_binding_t *bindings,
    iree_hal_descriptor_set_layout_t **out_descriptor_set_layout) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(!binding_count || bindings);
  IREE_ASSERT_ARGUMENT(out_descriptor_set_layout);
  *out_descriptor_set_layout = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_rocm_descriptor_set_layout_t *descriptor_set_layout = NULL;
  iree_status_t status = iree_allocator_malloc(context->host_allocator,
                                               sizeof(*descriptor_set_layout),
                                               (void **)&descriptor_set_layout);
  if (iree_status_is_ok(status)) {
    iree_hal_resource_initialize(&iree_hal_rocm_descriptor_set_layout_vtable,
                                 &descriptor_set_layout->resource);
    descriptor_set_layout->context = context;
    *out_descriptor_set_layout =
        (iree_hal_descriptor_set_layout_t *)descriptor_set_layout;
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_rocm_descriptor_set_layout_destroy(
    iree_hal_descriptor_set_layout_t *base_descriptor_set_layout) {
  iree_hal_rocm_descriptor_set_layout_t *descriptor_set_layout =
      iree_hal_rocm_descriptor_set_layout_cast(base_descriptor_set_layout);
  iree_allocator_t host_allocator =
      descriptor_set_layout->context->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_free(host_allocator, descriptor_set_layout);

  IREE_TRACE_ZONE_END(z0);
}

const iree_hal_descriptor_set_layout_vtable_t
    iree_hal_rocm_descriptor_set_layout_vtable = {
        .destroy = iree_hal_rocm_descriptor_set_layout_destroy,
};
