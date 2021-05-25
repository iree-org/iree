// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/cuda/executable_layout.h"

#include "iree/base/tracing.h"
#include "iree/hal/cuda/status_util.h"

typedef struct {
  iree_hal_resource_t resource;
  iree_hal_cuda_context_wrapper_t* context;
  iree_host_size_t set_layout_count;
  iree_hal_descriptor_set_layout_t* set_layouts[];
} iree_hal_cuda_executable_layout_t;

extern const iree_hal_executable_layout_vtable_t
    iree_hal_cuda_executable_layout_vtable;

static iree_hal_cuda_executable_layout_t* iree_hal_cuda_executable_layout_cast(
    iree_hal_executable_layout_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_cuda_executable_layout_vtable);
  return (iree_hal_cuda_executable_layout_t*)base_value;
}

iree_status_t iree_hal_cuda_executable_layout_create(
    iree_hal_cuda_context_wrapper_t* context, iree_host_size_t set_layout_count,
    iree_hal_descriptor_set_layout_t** set_layouts,
    iree_host_size_t push_constant_count,
    iree_hal_executable_layout_t** out_executable_layout) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(!set_layout_count || set_layouts);
  IREE_ASSERT_ARGUMENT(out_executable_layout);
  *out_executable_layout = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  // Currently the executable layout doesn't do anything.
  // TODO: Handle creating the argument layout at that time hadling both push
  // constant and buffers.
  iree_hal_cuda_executable_layout_t* executable_layout = NULL;
  iree_host_size_t total_size =
      sizeof(*executable_layout) +
      set_layout_count * sizeof(*executable_layout->set_layouts);
  iree_status_t status = iree_allocator_malloc(
      context->host_allocator, total_size, (void**)&executable_layout);
  if (iree_status_is_ok(status)) {
    iree_hal_resource_initialize(&iree_hal_cuda_executable_layout_vtable,
                                 &executable_layout->resource);
    executable_layout->context = context;
    executable_layout->set_layout_count = set_layout_count;
    for (iree_host_size_t i = 0; i < set_layout_count; ++i) {
      executable_layout->set_layouts[i] = set_layouts[i];
      iree_hal_descriptor_set_layout_retain(set_layouts[i]);
    }
    *out_executable_layout = (iree_hal_executable_layout_t*)executable_layout;
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_cuda_executable_layout_destroy(
    iree_hal_executable_layout_t* base_executable_layout) {
  iree_hal_cuda_executable_layout_t* executable_layout =
      iree_hal_cuda_executable_layout_cast(base_executable_layout);
  iree_allocator_t host_allocator = executable_layout->context->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  for (iree_host_size_t i = 0; i < executable_layout->set_layout_count; ++i) {
    iree_hal_descriptor_set_layout_release(executable_layout->set_layouts[i]);
  }
  iree_allocator_free(host_allocator, executable_layout);

  IREE_TRACE_ZONE_END(z0);
}

const iree_hal_executable_layout_vtable_t
    iree_hal_cuda_executable_layout_vtable = {
        .destroy = iree_hal_cuda_executable_layout_destroy,
};
