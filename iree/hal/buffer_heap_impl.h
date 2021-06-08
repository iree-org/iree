// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_BUFFER_HEAP_IMPL_H_
#define IREE_HAL_BUFFER_HEAP_IMPL_H_

#include "iree/base/api.h"
#include "iree/hal/buffer.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Private utilities for working with heap buffers
//===----------------------------------------------------------------------===//

// Allocates a new heap buffer from the specified |host_allocator|.
// |out_buffer| must be released by the caller.
iree_status_t iree_hal_heap_buffer_create(
    iree_hal_allocator_t* allocator, iree_hal_memory_type_t memory_type,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage, iree_device_size_t allocation_size,
    iree_allocator_t host_allocator, iree_hal_buffer_t** out_buffer);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_BUFFER_HEAP_IMPL_H_
