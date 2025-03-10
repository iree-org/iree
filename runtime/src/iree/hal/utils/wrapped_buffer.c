// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/utils/wrapped_buffer.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/internal/synchronization.h"
#include "iree/base/tracing.h"

typedef struct iree_hal_wrapped_buffer_t {
  iree_hal_buffer_t base;
  iree_hal_buffer_t* wrapped_buffer;
  iree_hal_buffer_params_t params;
  iree_device_size_t allocation_size;
  iree_allocator_t host_allocator;
} iree_hal_wrapped_buffer_t;

static const iree_hal_buffer_vtable_t iree_hal_wrapped_buffer_vtable;

bool iree_hal_wrapped_buffer_isa(iree_hal_buffer_t* base_value) {
  return iree_hal_resource_is(&base_value->resource,
                              &iree_hal_wrapped_buffer_vtable);
}

static iree_hal_wrapped_buffer_t* iree_hal_wrapped_buffer_cast(
    iree_hal_buffer_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_wrapped_buffer_vtable);
  return (iree_hal_wrapped_buffer_t*)base_value;
}

static const iree_hal_wrapped_buffer_t* iree_hal_wrapped_buffer_const_cast(
    const iree_hal_buffer_t* base_value) {
  IREE_HAL_ASSERT_TYPE(&base_value->resource, &iree_hal_wrapped_buffer_vtable);
  return (const iree_hal_wrapped_buffer_t*)base_value;
}

iree_status_t iree_hal_wrapped_buffer_make_buffer(
    iree_hal_buffer_t* wrapped_buffer, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size, iree_allocator_t host_allocator,
    iree_hal_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = NULL;

  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_wrapped_buffer_t* buffer = NULL;
  iree_status_t status =
      iree_allocator_malloc(host_allocator, sizeof(*buffer), (void**)&buffer);
  if (iree_status_is_ok(status)) {
    iree_hal_resource_initialize(&iree_hal_wrapped_buffer_vtable,
                                 &buffer->base.resource);
    buffer->wrapped_buffer = wrapped_buffer;
    buffer->params.usage = params.usage;
    buffer->params.access = params.access;
    buffer->params.type = params.type;
    buffer->params.queue_affinity = params.queue_affinity;
    buffer->params.min_alignment = params.min_alignment;
    buffer->allocation_size = allocation_size;
    buffer->host_allocator = host_allocator;
    *out_buffer = &buffer->base;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_wrapped_buffer_get_wrapped_buffer(
    iree_hal_buffer_t* base_buffer, iree_hal_buffer_t** out_buffer) {
  iree_hal_wrapped_buffer_t* buffer = iree_hal_wrapped_buffer_cast(base_buffer);
  *out_buffer = buffer->wrapped_buffer;
}

void iree_hal_wrapped_buffer_set_wrapped_buffer(
    iree_hal_buffer_t* base_buffer, iree_hal_buffer_t* wrapped_buffer) {
  iree_hal_wrapped_buffer_t* buffer = iree_hal_wrapped_buffer_cast(base_buffer);
  buffer->wrapped_buffer = wrapped_buffer;
}

iree_device_size_t iree_hal_wrapped_buffer_allocation_size(
    const iree_hal_buffer_t* base_buffer) {
  const iree_hal_wrapped_buffer_t* buffer =
      iree_hal_wrapped_buffer_const_cast(base_buffer);
  return buffer->allocation_size;
}

void iree_hal_wrapped_buffer_get_buffer_params(
    iree_hal_buffer_t* base_buffer, iree_hal_buffer_params_t* out_params) {
  IREE_ASSERT_ARGUMENT(out_params);
  iree_hal_wrapped_buffer_t* buffer = iree_hal_wrapped_buffer_cast(base_buffer);
  out_params->usage = buffer->params.usage;
  out_params->access = buffer->params.access;
  out_params->type = buffer->params.type;
  out_params->queue_affinity = buffer->params.queue_affinity;
  out_params->min_alignment = buffer->params.min_alignment;
}

static void iree_hal_wrapped_buffer_recycle(iree_hal_buffer_t* base_buffer) {
  iree_hal_wrapped_buffer_t* buffer = iree_hal_wrapped_buffer_cast(base_buffer);
  if (buffer->wrapped_buffer != NULL) {
    iree_hal_wrapped_buffer_recycle(buffer->wrapped_buffer);
  }
}

static void iree_hal_wrapped_buffer_destroy(iree_hal_buffer_t* base_buffer) {
  iree_hal_wrapped_buffer_t* buffer = iree_hal_wrapped_buffer_cast(base_buffer);
  if (buffer->wrapped_buffer != NULL) {
    iree_hal_buffer_destroy(buffer->wrapped_buffer);
  }
}

static iree_status_t iree_hal_wrapped_buffer_map_range(
    iree_hal_buffer_t* base_buffer, iree_hal_mapping_mode_t mapping_mode,
    iree_hal_memory_access_t memory_access,
    iree_device_size_t local_byte_offset, iree_device_size_t local_byte_length,
    iree_hal_buffer_mapping_t* mapping) {
  // Nothing to do today.
  return iree_ok_status();
}

static iree_status_t iree_hal_wrapped_buffer_unmap_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length, iree_hal_buffer_mapping_t* mapping) {
  // Nothing to do today.
  return iree_ok_status();
}

static iree_status_t iree_hal_wrapped_buffer_mapping_invalidate_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length) {
  // Nothing to do today.
  return iree_ok_status();
}

static iree_status_t iree_hal_wrapped_buffer_mapping_flush_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length) {
  // Nothing to do today.
  return iree_ok_status();
}

static const iree_hal_buffer_vtable_t iree_hal_wrapped_buffer_vtable = {
    .recycle = iree_hal_wrapped_buffer_recycle,
    .destroy = iree_hal_wrapped_buffer_destroy,
    .map_range = iree_hal_wrapped_buffer_map_range,
    .unmap_range = iree_hal_wrapped_buffer_unmap_range,
    .invalidate_range = iree_hal_wrapped_buffer_mapping_invalidate_range,
    .flush_range = iree_hal_wrapped_buffer_mapping_flush_range,
};
