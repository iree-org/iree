// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/cuda2/pipeline_layout.h"

#include <stddef.h>

#include "iree/base/api.h"
#include "iree/base/tracing.h"

//===----------------------------------------------------------------------===//
// iree_hal_cuda2_descriptor_set_layout_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_cuda2_descriptor_set_layout_t {
  // Abstract resource used for injecting reference counting and vtable;
  // must be at offset 0.
  iree_hal_resource_t resource;

  // The host allocator used for creating this descriptor set layout struct.
  iree_allocator_t host_allocator;

  // The total number of bindings in this descriptor set.
  iree_host_size_t binding_count;
} iree_hal_cuda2_descriptor_set_layout_t;

static const iree_hal_descriptor_set_layout_vtable_t
    iree_hal_cuda2_descriptor_set_layout_vtable;

static iree_hal_cuda2_descriptor_set_layout_t*
iree_hal_cuda2_descriptor_set_layout_cast(
    iree_hal_descriptor_set_layout_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value,
                       &iree_hal_cuda2_descriptor_set_layout_vtable);
  return (iree_hal_cuda2_descriptor_set_layout_t*)base_value;
}

static const iree_hal_cuda2_descriptor_set_layout_t*
iree_hal_cuda2_descriptor_set_layout_const_cast(
    const iree_hal_descriptor_set_layout_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value,
                       &iree_hal_cuda2_descriptor_set_layout_vtable);
  return (const iree_hal_cuda2_descriptor_set_layout_t*)base_value;
}

iree_status_t iree_hal_cuda2_descriptor_set_layout_create(
    iree_hal_descriptor_set_layout_flags_t flags,
    iree_host_size_t binding_count,
    const iree_hal_descriptor_set_layout_binding_t* bindings,
    iree_allocator_t host_allocator,
    iree_hal_descriptor_set_layout_t** out_descriptor_set_layout) {
  IREE_ASSERT_ARGUMENT(!binding_count || bindings);
  IREE_ASSERT_ARGUMENT(out_descriptor_set_layout);
  IREE_TRACE_ZONE_BEGIN(z0);

  *out_descriptor_set_layout = NULL;

  iree_hal_cuda2_descriptor_set_layout_t* descriptor_set_layout = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*descriptor_set_layout),
                                (void**)&descriptor_set_layout));

  iree_hal_resource_initialize(&iree_hal_cuda2_descriptor_set_layout_vtable,
                               &descriptor_set_layout->resource);
  descriptor_set_layout->host_allocator = host_allocator;
  descriptor_set_layout->binding_count = binding_count;
  *out_descriptor_set_layout =
      (iree_hal_descriptor_set_layout_t*)descriptor_set_layout;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_host_size_t iree_hal_cuda2_descriptor_set_layout_binding_count(
    const iree_hal_descriptor_set_layout_t* base_descriptor_set_layout) {
  const iree_hal_cuda2_descriptor_set_layout_t* descriptor_set_layout =
      iree_hal_cuda2_descriptor_set_layout_const_cast(
          base_descriptor_set_layout);
  return descriptor_set_layout->binding_count;
}

static void iree_hal_cuda2_descriptor_set_layout_destroy(
    iree_hal_descriptor_set_layout_t* base_descriptor_set_layout) {
  iree_hal_cuda2_descriptor_set_layout_t* descriptor_set_layout =
      iree_hal_cuda2_descriptor_set_layout_cast(base_descriptor_set_layout);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_t host_allocator = descriptor_set_layout->host_allocator;

  iree_allocator_free(host_allocator, descriptor_set_layout);

  IREE_TRACE_ZONE_END(z0);
}

static const iree_hal_descriptor_set_layout_vtable_t
    iree_hal_cuda2_descriptor_set_layout_vtable = {
        .destroy = iree_hal_cuda2_descriptor_set_layout_destroy,
};

//===----------------------------------------------------------------------===//
// iree_hal_cuda2_pipeline_layout_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_cuda2_pipeline_layout_t {
  // Abstract resource used for injecting reference counting and vtable;
  // must be at offset 0.
  iree_hal_resource_t resource;

  // The host allocator used for creating this pipeline layout struct.
  iree_allocator_t host_allocator;

  // The kernel argument index for push constants.
  // Note that push constants are placed after all normal descriptors.
  iree_host_size_t push_constant_base_index;
  iree_host_size_t push_constant_count;

  iree_host_size_t set_layout_count;
  // The list of descriptor set layout pointers, pointing to trailing inline
  // allocation after the end of this struct.
  struct {
    iree_hal_descriptor_set_layout_t* set_layout;
    // Base kernel argument index for this descriptor set.
    iree_host_size_t base_index;
  } set_layouts[];
} iree_hal_cuda2_pipeline_layout_t;
// + Additional inline allocation for holding all descriptor sets.

static const iree_hal_pipeline_layout_vtable_t
    iree_hal_cuda2_pipeline_layout_vtable;

static iree_hal_cuda2_pipeline_layout_t* iree_hal_cuda2_pipeline_layout_cast(
    iree_hal_pipeline_layout_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_cuda2_pipeline_layout_vtable);
  return (iree_hal_cuda2_pipeline_layout_t*)base_value;
}

static const iree_hal_cuda2_pipeline_layout_t*
iree_hal_cuda2_pipeline_layout_const_cast(
    const iree_hal_pipeline_layout_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_cuda2_pipeline_layout_vtable);
  return (const iree_hal_cuda2_pipeline_layout_t*)base_value;
}

iree_status_t iree_hal_cuda2_pipeline_layout_create(
    iree_host_size_t set_layout_count,
    iree_hal_descriptor_set_layout_t* const* set_layouts,
    iree_host_size_t push_constant_count, iree_allocator_t host_allocator,
    iree_hal_pipeline_layout_t** out_pipeline_layout) {
  IREE_ASSERT_ARGUMENT(!set_layout_count || set_layouts);
  IREE_ASSERT_ARGUMENT(out_pipeline_layout);
  IREE_TRACE_ZONE_BEGIN(z0);

  *out_pipeline_layout = NULL;
  if (push_constant_count > IREE_HAL_CUDA_MAX_PUSH_CONSTANT_COUNT) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "push constant count %" PRIhsz " over the limit of %d",
        push_constant_count, IREE_HAL_CUDA_MAX_PUSH_CONSTANT_COUNT);
  }

  // Currently the pipeline layout doesn't do anything.
  // TODO: Handle creating the argument layout at that time hadling both push
  // constant and buffers.
  iree_hal_cuda2_pipeline_layout_t* pipeline_layout = NULL;
  iree_host_size_t total_size =
      sizeof(*pipeline_layout) +
      set_layout_count * sizeof(*pipeline_layout->set_layouts);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, total_size,
                                (void**)&pipeline_layout));

  iree_hal_resource_initialize(&iree_hal_cuda2_pipeline_layout_vtable,
                               &pipeline_layout->resource);
  pipeline_layout->host_allocator = host_allocator;
  pipeline_layout->set_layout_count = set_layout_count;
  iree_host_size_t base_index = 0;
  for (iree_host_size_t i = 0; i < set_layout_count; ++i) {
    pipeline_layout->set_layouts[i].set_layout = set_layouts[i];
    // Copy and retain all descriptor sets so we don't lose them.
    iree_hal_descriptor_set_layout_retain(set_layouts[i]);
    pipeline_layout->set_layouts[i].base_index = base_index;
    base_index +=
        iree_hal_cuda2_descriptor_set_layout_binding_count(set_layouts[i]);
  }
  pipeline_layout->push_constant_base_index = base_index;
  pipeline_layout->push_constant_count = push_constant_count;
  *out_pipeline_layout = (iree_hal_pipeline_layout_t*)pipeline_layout;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_cuda2_pipeline_layout_destroy(
    iree_hal_pipeline_layout_t* base_pipeline_layout) {
  iree_hal_cuda2_pipeline_layout_t* pipeline_layout =
      iree_hal_cuda2_pipeline_layout_cast(base_pipeline_layout);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_t host_allocator = pipeline_layout->host_allocator;

  for (iree_host_size_t i = 0; i < pipeline_layout->set_layout_count; ++i) {
    iree_hal_descriptor_set_layout_release(
        pipeline_layout->set_layouts[i].set_layout);
  }
  iree_allocator_free(host_allocator, pipeline_layout);

  IREE_TRACE_ZONE_END(z0);
}

iree_host_size_t iree_hal_cuda2_pipeline_layout_descriptor_set_count(
    const iree_hal_pipeline_layout_t* base_pipeline_layout) {
  const iree_hal_cuda2_pipeline_layout_t* pipeline_layout =
      iree_hal_cuda2_pipeline_layout_const_cast(base_pipeline_layout);
  return pipeline_layout->set_layout_count;
}

const iree_hal_descriptor_set_layout_t*
iree_hal_cuda2_pipeline_layout_descriptor_set_layout(
    const iree_hal_pipeline_layout_t* base_pipeline_layout, uint32_t set) {
  const iree_hal_cuda2_pipeline_layout_t* pipeline_layout =
      iree_hal_cuda2_pipeline_layout_const_cast(base_pipeline_layout);
  if (set < pipeline_layout->set_layout_count) {
    return pipeline_layout->set_layouts[set].set_layout;
  }
  return NULL;
}

iree_host_size_t iree_hal_cuda2_pipeline_layout_base_binding_index(
    const iree_hal_pipeline_layout_t* base_pipeline_layout, uint32_t set) {
  const iree_hal_cuda2_pipeline_layout_t* pipeline_layout =
      iree_hal_cuda2_pipeline_layout_const_cast(base_pipeline_layout);
  return pipeline_layout->set_layouts[set].base_index;
}

iree_host_size_t iree_hal_cuda2_pipeline_layout_total_binding_count(
    const iree_hal_pipeline_layout_t* base_pipeline_layout) {
  return iree_hal_cuda2_pipeline_layout_push_constant_index(
      base_pipeline_layout);
}

iree_host_size_t iree_hal_cuda2_pipeline_layout_push_constant_index(
    const iree_hal_pipeline_layout_t* base_pipeline_layout) {
  const iree_hal_cuda2_pipeline_layout_t* pipeline_layout =
      iree_hal_cuda2_pipeline_layout_const_cast(base_pipeline_layout);
  return pipeline_layout->push_constant_base_index;
}

iree_host_size_t iree_hal_cuda2_pipeline_layout_push_constant_count(
    const iree_hal_pipeline_layout_t* base_pipeline_layout) {
  const iree_hal_cuda2_pipeline_layout_t* pipeline_layout =
      iree_hal_cuda2_pipeline_layout_const_cast(base_pipeline_layout);
  return pipeline_layout->push_constant_count;
}

static const iree_hal_pipeline_layout_vtable_t
    iree_hal_cuda2_pipeline_layout_vtable = {
        .destroy = iree_hal_cuda2_pipeline_layout_destroy,
};
