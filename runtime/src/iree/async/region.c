// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/region.h"

#include "iree/async/proactor.h"

void iree_async_region_destroy(iree_async_region_t* region) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (region->destroy_fn) {
    // Backend-specific teardown handles kernel deregistration, slab release,
    // and region struct free.
    region->destroy_fn(region);
  } else {
    // Legacy path for non-slab regions created by the general registration
    // system (register_buffer / register_dmabuf). The registration entry's
    // cleanup_fn handles backend-specific cleanup before the last reference
    // is released. We just free the struct via the proactor's allocator.
    iree_allocator_free(region->proactor->allocator, region);
  }
  IREE_TRACE_ZONE_END(z0);
}
