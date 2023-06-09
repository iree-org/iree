// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_CUDA_MEMORY_POOLS_H_
#define IREE_HAL_DRIVERS_CUDA_MEMORY_POOLS_H_

#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/cuda/api.h"
#include "iree/hal/drivers/cuda/context_wrapper.h"
#include "iree/hal/drivers/cuda/cuda_headers.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Retained CUDA memory pools for various allocation types.
typedef struct iree_hal_cuda_memory_pools_t {
  // CUDA context the pools are attached to.
  iree_hal_cuda_context_wrapper_t* context;
  // Used exclusively for DEVICE_LOCAL allocations.
  CUmemoryPool device_local;
  // Used for any host-visible/host-local memory types.
  CUmemoryPool other;

  IREE_STATISTICS(struct {
    iree_atomic_int64_t device_bytes_allocated;
    iree_atomic_int64_t device_bytes_freed;
    iree_atomic_int64_t host_bytes_allocated;
    iree_atomic_int64_t host_bytes_freed;
  } statistics;)
} iree_hal_cuda_memory_pools_t;

// Initializes |out_pools| by configuring new CUDA memory pools.
iree_status_t iree_hal_cuda_memory_pools_initialize(
    iree_hal_cuda_context_wrapper_t* context,
    const iree_hal_cuda_memory_pooling_params_t* pooling_params,
    iree_hal_cuda_memory_pools_t* IREE_RESTRICT out_pools);

// Deinitializes the |pools| and releases the underlying CUDA resources.
void iree_hal_cuda_memory_pools_deinitialize(
    iree_hal_cuda_memory_pools_t* pools);

// Merges statistics information from |pools| into |statistics|.
void iree_hal_cuda_memory_pools_merge_statistics(
    iree_hal_cuda_memory_pools_t* pools,
    iree_hal_allocator_statistics_t* statistics);

// Trims all memory pools by releasing resources back to the system.
iree_status_t iree_hal_cuda_memory_pools_trim(
    iree_hal_cuda_memory_pools_t* pools,
    const iree_hal_cuda_memory_pooling_params_t* pooling_params);

// Asynchronously allocates a buffer from an appropriate pool.
// The allocation will be stream-ordered on |stream|.
iree_status_t iree_hal_cuda_memory_pools_alloca(
    iree_hal_cuda_memory_pools_t* pools, CUstream stream,
    iree_hal_allocator_pool_t pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer);

// Asynchronously deallocates a buffer from its pool.
// The deallocation will be stream-ordered on |stream|.
iree_status_t iree_hal_cuda_memory_pools_dealloca(
    iree_hal_cuda_memory_pools_t* pools, CUstream stream,
    iree_hal_buffer_t* buffer);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_CUDA_MEMORY_POOLS_H_
