// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_BUFFER_CACHING_H_
#define IREE_HAL_BUFFER_CACHING_H_

#include "iree/base/api.h"
#include "iree/hal/buffer.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_caching_buffer_t {
  iree_hal_resource_t resource;
  // iree_hal_buffer_t base;
  iree_hal_buffer_t* delegate_buffer;
  struct iree_hal_caching_buffer_t* next;
} iree_hal_caching_buffer_t;

iree_status_t iree_hal_caching_buffer_wrap(
    iree_hal_allocator_t* base_allocator, iree_hal_memory_type_t memory_type,
    iree_hal_buffer_usage_t allowed_usage, iree_host_size_t allocation_size,
    iree_hal_buffer_t** out_buffer);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_BUFFER_CACHING_H_
