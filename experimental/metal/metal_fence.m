// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/metal/metal_fence.h"

#import <Metal/Metal.h>

#include "iree/base/api.h"
#include "iree/base/tracing.h"

typedef struct iree_hal_metal_fence_t {
  // Abstract resource used for injecting reference counting and vtable; must be at offset 0.
  iree_hal_resource_t resource;

  id<MTLFence> fence;

  iree_allocator_t host_allocator;
} iree_hal_metal_fence_t;

static const iree_hal_event_vtable_t iree_hal_metal_fence_vtable;

static iree_hal_metal_fence_t* iree_hal_metal_fence_cast(iree_hal_event_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_metal_fence_vtable);
  return (iree_hal_metal_fence_t*)base_value;
}

static const iree_hal_metal_fence_t* iree_hal_metal_fence_const_cast(
    const iree_hal_event_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_metal_fence_vtable);
  return (const iree_hal_metal_fence_t*)base_value;
}

id<MTLFence> iree_hal_metal_fence_handle(const iree_hal_event_t* base_event) {
  const iree_hal_metal_fence_t* fence = iree_hal_metal_fence_const_cast(base_event);
  return fence->fence;
}

iree_status_t iree_hal_metal_fence_create(id<MTLDevice> device, iree_allocator_t host_allocator,
                                          iree_hal_event_t** out_event) {
  IREE_ASSERT_ARGUMENT(out_event);
  IREE_TRACE_ZONE_BEGIN(z0);

  *out_event = NULL;
  iree_hal_metal_fence_t* event = NULL;
  iree_status_t status = iree_allocator_malloc(host_allocator, sizeof(*event), (void**)&event);
  if (iree_status_is_ok(status)) {
    iree_hal_resource_initialize(&iree_hal_metal_fence_vtable, &event->resource);
    event->fence = [device newFence];  // +1
    event->host_allocator = host_allocator;
    *out_event = (iree_hal_event_t*)event;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_metal_fence_recreate(iree_hal_event_t* event) {
  iree_hal_metal_fence_t* fence = iree_hal_metal_fence_cast(event);
  IREE_TRACE_ZONE_BEGIN(z0);

  id<MTLDevice> device = fence->fence.device;
  [fence->fence release];
  fence->fence = [device newFence];

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_metal_fence_destroy(iree_hal_event_t* base_event) {
  iree_hal_metal_fence_t* fence = iree_hal_metal_fence_cast(base_event);
  IREE_TRACE_ZONE_BEGIN(z0);

  [fence->fence release];  // -1
  iree_allocator_free(fence->host_allocator, base_event);

  IREE_TRACE_ZONE_END(z0);
}

static const iree_hal_event_vtable_t iree_hal_metal_fence_vtable = {
    .destroy = iree_hal_metal_fence_destroy,
};
