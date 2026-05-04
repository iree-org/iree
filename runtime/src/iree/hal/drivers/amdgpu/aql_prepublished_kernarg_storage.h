// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_AQL_PREPUBLISHED_KERNARG_STORAGE_H_
#define IREE_HAL_DRIVERS_AMDGPU_AQL_PREPUBLISHED_KERNARG_STORAGE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Strategy used to materialize reusable command-buffer kernarg templates.
typedef enum iree_hal_amdgpu_aql_prepublished_kernarg_storage_strategy_e {
  IREE_HAL_AMDGPU_AQL_PREPUBLISHED_KERNARG_STORAGE_STRATEGY_DISABLED = 0,
  // Device-local fine-grained memory that is CPU-visible and host-coherent.
  IREE_HAL_AMDGPU_AQL_PREPUBLISHED_KERNARG_STORAGE_STRATEGY_DEVICE_FINE_HOST_COHERENT =
      1,
} iree_hal_amdgpu_aql_prepublished_kernarg_storage_strategy_t;

// Storage strategy for finalized reusable command-buffer kernarg templates.
typedef struct iree_hal_amdgpu_aql_prepublished_kernarg_storage_t {
  // Selected backing strategy.
  iree_hal_amdgpu_aql_prepublished_kernarg_storage_strategy_t strategy;
  // HAL allocation parameters used for materialized kernarg storage.
  iree_hal_buffer_params_t buffer_params;
} iree_hal_amdgpu_aql_prepublished_kernarg_storage_t;

static inline iree_hal_amdgpu_aql_prepublished_kernarg_storage_t
iree_hal_amdgpu_aql_prepublished_kernarg_storage_disabled(void) {
  iree_hal_amdgpu_aql_prepublished_kernarg_storage_t storage = {
      IREE_HAL_AMDGPU_AQL_PREPUBLISHED_KERNARG_STORAGE_STRATEGY_DISABLED};
  return storage;
}

static inline iree_hal_amdgpu_aql_prepublished_kernarg_storage_t
iree_hal_amdgpu_aql_prepublished_kernarg_storage_device_fine_host_coherent(
    void) {
  iree_hal_amdgpu_aql_prepublished_kernarg_storage_t storage =
      iree_hal_amdgpu_aql_prepublished_kernarg_storage_disabled();
  storage.strategy =
      IREE_HAL_AMDGPU_AQL_PREPUBLISHED_KERNARG_STORAGE_STRATEGY_DEVICE_FINE_HOST_COHERENT;
  storage.buffer_params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL |
                               IREE_HAL_MEMORY_TYPE_HOST_VISIBLE |
                               IREE_HAL_MEMORY_TYPE_HOST_COHERENT;
  storage.buffer_params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  storage.buffer_params.usage = IREE_HAL_BUFFER_USAGE_DISPATCH_UNIFORM_READ |
                                IREE_HAL_BUFFER_USAGE_MAPPING;
  return storage;
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_AQL_PREPUBLISHED_KERNARG_STORAGE_H_
