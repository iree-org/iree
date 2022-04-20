// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_LOCAL_LOCAL_EXECUTABLE_LAYOUT_H_
#define IREE_HAL_LOCAL_LOCAL_EXECUTABLE_LAYOUT_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#define IREE_HAL_LOCAL_MAX_DESCRIPTOR_SET_COUNT 2
#define IREE_HAL_LOCAL_MAX_PUSH_CONSTANT_COUNT 64

typedef uint64_t iree_hal_local_binding_mask_t;

#define IREE_HAL_LOCAL_BINDING_MASK_BITS \
  (sizeof(iree_hal_local_binding_mask_t) * 8)

typedef struct iree_hal_local_executable_layout_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  iree_host_size_t push_constants;
  iree_host_size_t dynamic_binding_count;
  iree_hal_local_binding_mask_t used_bindings;
  iree_host_size_t set_layout_count;
  iree_hal_descriptor_set_layout_t* set_layouts[];
} iree_hal_local_executable_layout_t;

iree_status_t iree_hal_local_executable_layout_create(
    iree_host_size_t push_constants, iree_host_size_t set_layout_count,
    iree_hal_descriptor_set_layout_t** set_layouts,
    iree_allocator_t host_allocator,
    iree_hal_executable_layout_t** out_executable_layout);

iree_hal_local_executable_layout_t* iree_hal_local_executable_layout_cast(
    iree_hal_executable_layout_t* base_value);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_LOCAL_LOCAL_EXECUTABLE_LAYOUT_H_
