// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_LOCAL_LOCAL_DESCRIPTOR_SET_LAYOUT_H_
#define IREE_HAL_LOCAL_LOCAL_DESCRIPTOR_SET_LAYOUT_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#define IREE_HAL_LOCAL_MAX_DESCRIPTOR_BINDING_COUNT 32

typedef struct iree_hal_local_descriptor_set_layout_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  iree_hal_descriptor_set_layout_usage_type_t usage_type;
  iree_host_size_t binding_count;
  iree_hal_descriptor_set_layout_binding_t bindings[];
} iree_hal_local_descriptor_set_layout_t;

iree_status_t iree_hal_local_descriptor_set_layout_create(
    iree_hal_descriptor_set_layout_usage_type_t usage_type,
    iree_host_size_t binding_count,
    const iree_hal_descriptor_set_layout_binding_t* bindings,
    iree_allocator_t host_allocator,
    iree_hal_descriptor_set_layout_t** out_descriptor_set_layout);

iree_hal_local_descriptor_set_layout_t*
iree_hal_local_descriptor_set_layout_cast(
    iree_hal_descriptor_set_layout_t* base_value);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_LOCAL_LOCAL_DESCRIPTOR_SET_LAYOUT_H_
