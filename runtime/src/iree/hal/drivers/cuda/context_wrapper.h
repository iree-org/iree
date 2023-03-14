// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_CUDA_CONTEXT_WRAPPER_H_
#define IREE_HAL_DRIVERS_CUDA_CONTEXT_WRAPPER_H_

#include "iree/hal/api.h"
#include "iree/hal/drivers/cuda/cuda_headers.h"
#include "iree/hal/drivers/cuda/dynamic_symbols.h"

// Structure to wrap all objects constant within a context. This makes it
// simpler to pass it to the different objects and saves memory.
typedef struct iree_hal_cuda_context_wrapper_t {
  CUdevice cu_device;
  CUcontext cu_context;
  iree_allocator_t host_allocator;
  iree_hal_cuda_dynamic_symbols_t* syms;
} iree_hal_cuda_context_wrapper_t;

#endif  // IREE_HAL_DRIVERS_CUDA_CONTEXT_WRAPPER_H_
