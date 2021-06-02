// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/local/local_descriptor_set.h"

#include <stddef.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/tracing.h"

static const iree_hal_descriptor_set_vtable_t
    iree_hal_local_descriptor_set_vtable;

iree_hal_local_descriptor_set_t* iree_hal_local_descriptor_set_cast(
    iree_hal_descriptor_set_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_local_descriptor_set_vtable);
  return (iree_hal_local_descriptor_set_t*)base_value;
}

iree_status_t iree_hal_local_descriptor_set_create(
    iree_hal_descriptor_set_layout_t* base_layout,
    iree_host_size_t binding_count,
    const iree_hal_descriptor_set_binding_t* bindings,
    iree_hal_descriptor_set_t** out_descriptor_set) {
  IREE_ASSERT_ARGUMENT(base_layout);
  IREE_ASSERT_ARGUMENT(!binding_count || bindings);
  IREE_ASSERT_ARGUMENT(out_descriptor_set);
  *out_descriptor_set = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_local_descriptor_set_layout_t* local_layout =
      iree_hal_local_descriptor_set_layout_cast(base_layout);
  IREE_ASSERT_ARGUMENT(local_layout);

  iree_hal_local_descriptor_set_t* descriptor_set = NULL;
  iree_host_size_t total_size =
      sizeof(*descriptor_set) +
      binding_count * sizeof(*descriptor_set->bindings);
  iree_status_t status = iree_allocator_malloc(
      local_layout->host_allocator, total_size, (void**)&descriptor_set);
  if (iree_status_is_ok(status)) {
    iree_hal_resource_initialize(&iree_hal_local_descriptor_set_vtable,
                                 &descriptor_set->resource);
    descriptor_set->layout = local_layout;
    iree_hal_descriptor_set_layout_retain(base_layout);
    descriptor_set->binding_count = binding_count;
    memcpy(descriptor_set->bindings, bindings,
           binding_count * sizeof(iree_hal_descriptor_set_binding_t));
    for (iree_host_size_t i = 0; i < descriptor_set->binding_count; ++i) {
      iree_hal_buffer_retain(descriptor_set->bindings[i].buffer);
    }
    *out_descriptor_set = (iree_hal_descriptor_set_t*)descriptor_set;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_local_descriptor_set_destroy(
    iree_hal_descriptor_set_t* base_descriptor_set) {
  iree_hal_local_descriptor_set_t* descriptor_set =
      iree_hal_local_descriptor_set_cast(base_descriptor_set);
  iree_allocator_t host_allocator = descriptor_set->layout->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  for (iree_host_size_t i = 0; i < descriptor_set->binding_count; ++i) {
    iree_hal_buffer_release(descriptor_set->bindings[i].buffer);
  }
  iree_hal_descriptor_set_layout_release(
      (iree_hal_descriptor_set_layout_t*)descriptor_set->layout);
  iree_allocator_free(host_allocator, descriptor_set);

  IREE_TRACE_ZONE_END(z0);
}

static const iree_hal_descriptor_set_vtable_t
    iree_hal_local_descriptor_set_vtable = {
        .destroy = iree_hal_local_descriptor_set_destroy,
};
