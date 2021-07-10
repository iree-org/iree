// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_CUDA_DIRECT_COMMAND_BUFFER_H_
#define IREE_HAL_CUDA_DIRECT_COMMAND_BUFFER_H_

#include "iree/base/internal/arena.h"
#include "iree/hal/api.h"
#include "iree/hal/cuda/context_wrapper.h"
#include "iree/hal/cuda/cuda_headers.h"
#include "iree/hal/cuda/dynamic_symbols.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// CUDA Kernel Information Structure
typedef struct {
  CUfunction func;
  unsigned int gridDimX;
  unsigned int gridDimY;
  unsigned int gridDimZ;
  unsigned int blockDimX;
  unsigned int blockDimY;
  unsigned int blockDimZ;
  void **kernelParams;
} cuda_launch_params;

// Creates a cuda direct command buffer.
iree_status_t iree_hal_cuda_direct_command_buffer_allocate(
    iree_hal_cuda_context_wrapper_t *context,
    iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_arena_block_pool_t* block_pool,
    iree_hal_queue_affinity_t queue_affinity,
    iree_hal_command_buffer_t **out_command_buffer);

iree_status_t iree_hal_cuda_direct_command_buffer_apply(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_command_buffer_t* target_command_buffer);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_CUDA_DIRECT_COMMAND_BUFFER_H_
