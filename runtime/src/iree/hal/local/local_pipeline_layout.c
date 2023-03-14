// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/local/local_pipeline_layout.h"

#include <stddef.h>
#include <string.h>

#include "iree/base/tracing.h"

//===----------------------------------------------------------------------===//
// iree_hal_local_descriptor_set_layout_t
//===----------------------------------------------------------------------===//

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
    iree_hal_descriptor_set_layout_flags_t flags,
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
    layout->flags = flags;
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

//===----------------------------------------------------------------------===//
// iree_hal_local_pipeline_layout_t
//===----------------------------------------------------------------------===//

static const iree_hal_pipeline_layout_vtable_t
    iree_hal_local_pipeline_layout_vtable;

iree_hal_local_pipeline_layout_t* iree_hal_local_pipeline_layout_cast(
    iree_hal_pipeline_layout_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_local_pipeline_layout_vtable);
  return (iree_hal_local_pipeline_layout_t*)base_value;
}

iree_status_t iree_hal_local_pipeline_layout_create(
    iree_host_size_t push_constants, iree_host_size_t set_layout_count,
    iree_hal_descriptor_set_layout_t* const* set_layouts,
    iree_allocator_t host_allocator,
    iree_hal_pipeline_layout_t** out_pipeline_layout) {
  IREE_ASSERT_ARGUMENT(!set_layout_count || set_layouts);
  IREE_ASSERT_ARGUMENT(out_pipeline_layout);
  *out_pipeline_layout = NULL;
  if (set_layout_count > IREE_HAL_LOCAL_MAX_DESCRIPTOR_SET_COUNT) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "set layout count %zu over the limit of %d",
                            set_layout_count,
                            IREE_HAL_LOCAL_MAX_DESCRIPTOR_SET_COUNT);
  }
  if (push_constants > IREE_HAL_LOCAL_MAX_PUSH_CONSTANT_COUNT) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "push constant count %zu over the limit of %d",
                            push_constants,
                            IREE_HAL_LOCAL_MAX_PUSH_CONSTANT_COUNT);
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  iree_host_size_t total_size =
      sizeof(iree_hal_local_pipeline_layout_t) +
      set_layout_count * sizeof(iree_hal_descriptor_set_layout_t*);

  iree_hal_local_pipeline_layout_t* layout = NULL;
  iree_status_t status =
      iree_allocator_malloc(host_allocator, total_size, (void**)&layout);
  if (iree_status_is_ok(status)) {
    iree_hal_resource_initialize(&iree_hal_local_pipeline_layout_vtable,
                                 &layout->resource);
    layout->host_allocator = host_allocator;
    layout->push_constants = push_constants;
    layout->used_bindings = 0;
    layout->read_only_bindings = 0;
    layout->set_layout_count = set_layout_count;
    for (iree_host_size_t i = 0; i < set_layout_count; ++i) {
      layout->set_layouts[i] = set_layouts[i];
      iree_hal_descriptor_set_layout_retain(layout->set_layouts[i]);

      iree_hal_local_descriptor_set_layout_t* local_set_layout =
          iree_hal_local_descriptor_set_layout_cast(set_layouts[i]);
      for (iree_host_size_t j = 0; j < local_set_layout->binding_count; ++j) {
        // Track that this binding is used in the sparse set.
        const iree_hal_local_binding_mask_t binding_bit =
            1ull << (i * IREE_HAL_LOCAL_MAX_DESCRIPTOR_BINDING_COUNT + j);
        layout->used_bindings |= binding_bit;

        // Track which bindings are read-only so we can protect memory and
        // verify usage.
        const iree_hal_descriptor_set_layout_binding_t* binding =
            &local_set_layout->bindings[j];
        if (iree_all_bits_set(binding->flags,
                              IREE_HAL_DESCRIPTOR_FLAG_READ_ONLY)) {
          layout->read_only_bindings |= binding_bit;
        }
      }
    }
    *out_pipeline_layout = (iree_hal_pipeline_layout_t*)layout;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_local_pipeline_layout_destroy(
    iree_hal_pipeline_layout_t* base_layout) {
  iree_hal_local_pipeline_layout_t* layout =
      iree_hal_local_pipeline_layout_cast(base_layout);
  iree_allocator_t host_allocator = layout->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  for (iree_host_size_t i = 0; i < layout->set_layout_count; ++i) {
    iree_hal_descriptor_set_layout_release(layout->set_layouts[i]);
  }
  iree_allocator_free(host_allocator, layout);

  IREE_TRACE_ZONE_END(z0);
}

static const iree_hal_pipeline_layout_vtable_t
    iree_hal_local_pipeline_layout_vtable = {
        .destroy = iree_hal_local_pipeline_layout_destroy,
};
