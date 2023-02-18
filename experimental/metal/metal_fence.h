// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_METAL_METAL_FENCE_H_
#define IREE_EXPERIMENTAL_METAL_METAL_FENCE_H_

#import <Metal/Metal.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates a Metal fence to implement an IREE event.
//
// An IREE event defines synchronization scopes within the same command buffer.
// It maps to MTLFence in Metal.
iree_status_t iree_hal_metal_fence_create(id<MTLDevice> device,
                                          iree_allocator_t host_allocator,
                                          iree_hal_event_t** out_event);

// Destroys the current Metal fence behind the given |event| and recreate a new
// one. This is meant to support the IREE event reset API.
iree_status_t iree_hal_metal_fence_recreate(iree_hal_event_t* event);

// Returns the underlying Metal fence handle for the given |base_event|.
id<MTLFence> iree_hal_metal_fence_handle(const iree_hal_event_t* base_event);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_EXPERIMENTAL_METAL_METAL_FENCE_H_
