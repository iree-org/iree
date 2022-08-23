// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_LOCAL_LOCAL_PIPELINE_LAYOUT_H_
#define IREE_HAL_LOCAL_LOCAL_PIPELINE_LAYOUT_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_local_descriptor_set_layout_t
//===----------------------------------------------------------------------===//

#define IREE_HAL_LOCAL_MAX_DESCRIPTOR_BINDING_COUNT 32

typedef struct iree_hal_local_descriptor_set_layout_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  iree_hal_descriptor_set_layout_flags_t flags;
  iree_host_size_t binding_count;
  iree_hal_descriptor_set_layout_binding_t bindings[];
} iree_hal_local_descriptor_set_layout_t;

iree_status_t iree_hal_local_descriptor_set_layout_create(
    iree_hal_descriptor_set_layout_flags_t flags,
    iree_host_size_t binding_count,
    const iree_hal_descriptor_set_layout_binding_t* bindings,
    iree_allocator_t host_allocator,
    iree_hal_descriptor_set_layout_t** out_descriptor_set_layout);

iree_hal_local_descriptor_set_layout_t*
iree_hal_local_descriptor_set_layout_cast(
    iree_hal_descriptor_set_layout_t* base_value);

//===----------------------------------------------------------------------===//
// iree_hal_local_pipeline_layout_t
//===----------------------------------------------------------------------===//

#define IREE_HAL_LOCAL_MAX_DESCRIPTOR_SET_COUNT 2
#define IREE_HAL_LOCAL_MAX_PUSH_CONSTANT_COUNT 64

typedef uint64_t iree_hal_local_binding_mask_t;

#define IREE_HAL_LOCAL_BINDING_MASK_BITS \
  (sizeof(iree_hal_local_binding_mask_t) * 8)

typedef struct iree_hal_local_pipeline_layout_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  iree_host_size_t push_constants;
  iree_hal_local_binding_mask_t used_bindings;
  iree_hal_local_binding_mask_t read_only_bindings;
  iree_host_size_t set_layout_count;
  iree_hal_descriptor_set_layout_t* set_layouts[];
} iree_hal_local_pipeline_layout_t;

iree_status_t iree_hal_local_pipeline_layout_create(
    iree_host_size_t push_constants, iree_host_size_t set_layout_count,
    iree_hal_descriptor_set_layout_t* const* set_layouts,
    iree_allocator_t host_allocator,
    iree_hal_pipeline_layout_t** out_pipeline_layout);

iree_hal_local_pipeline_layout_t* iree_hal_local_pipeline_layout_cast(
    iree_hal_pipeline_layout_t* base_value);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_LOCAL_LOCAL_PIPELINE_LAYOUT_H_
