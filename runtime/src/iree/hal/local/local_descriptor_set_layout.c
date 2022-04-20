// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/local/local_descriptor_set_layout.h"

#include <stddef.h>
#include <string.h>

#include "iree/base/tracing.h"

static const iree_hal_descriptor_set_layout_vtable_t
    iree_hal_local_descriptor_set_layout_vtable;

iree_hal_local_descriptor_set_layout_t*
iree_hal_local_descriptor_set_layout_cast(
    iree_hal_descriptor_set_layout_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value,
                       &iree_hal_local_descriptor_set_layout_vtable);
  return (iree_hal_local_descriptor_set_layout_t*)base_value;
}

iree_status_t iree_hal_local_descriptor_set_layout_create(
    iree_hal_descriptor_set_layout_usage_type_t usage_type,
    iree_host_size_t binding_count,
    const iree_hal_descriptor_set_layout_binding_t* bindings,
    iree_allocator_t host_allocator,
    iree_hal_descriptor_set_layout_t** out_descriptor_set_layout) {
  IREE_ASSERT_ARGUMENT(!binding_count || bindings);
  IREE_ASSERT_ARGUMENT(out_descriptor_set_layout);
  *out_descriptor_set_layout = NULL;
  if (binding_count > IREE_HAL_LOCAL_MAX_DESCRIPTOR_BINDING_COUNT) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT, "binding count %zu over the limit of %d",
        binding_count, IREE_HAL_LOCAL_MAX_DESCRIPTOR_BINDING_COUNT);
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_local_descriptor_set_layout_t* layout = NULL;
  iree_host_size_t total_size =
      sizeof(*layout) + binding_count * sizeof(*layout->bindings);
  iree_status_t status =
      iree_allocator_malloc(host_allocator, total_size, (void**)&layout);
  if (iree_status_is_ok(status)) {
    iree_hal_resource_initialize(&iree_hal_local_descriptor_set_layout_vtable,
                                 &layout->resource);
    layout->host_allocator = host_allocator;
    layout->usage_type = usage_type;
    layout->binding_count = binding_count;
    memcpy(layout->bindings, bindings,
           binding_count * sizeof(iree_hal_descriptor_set_layout_binding_t));
    *out_descriptor_set_layout = (iree_hal_descriptor_set_layout_t*)layout;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_local_descriptor_set_layout_destroy(
    iree_hal_descriptor_set_layout_t* base_layout) {
  iree_hal_local_descriptor_set_layout_t* layout =
      iree_hal_local_descriptor_set_layout_cast(base_layout);
  iree_allocator_t host_allocator = layout->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_free(host_allocator, layout);

  IREE_TRACE_ZONE_END(z0);
}

static const iree_hal_descriptor_set_layout_vtable_t
    iree_hal_local_descriptor_set_layout_vtable = {
        .destroy = iree_hal_local_descriptor_set_layout_destroy,
};
