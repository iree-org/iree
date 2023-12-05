// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_HIP_CONTEXT_WRAPPER_H_
#define IREE_EXPERIMENTAL_HIP_CONTEXT_WRAPPER_H_

#include "experimental/hip/dynamic_symbols.h"
#include "experimental/hip/hip_headers.h"
#include "iree/hal/api.h"

// Structure to wrap all objects constant within a context. This makes it
// simpler to pass it to the different objects and saves memory.
typedef struct iree_hal_hip_context_wrapper_t {
  hipDevice_t hip_device;
  hipCtx_t hip_context;
  iree_allocator_t host_allocator;
  iree_hal_hip_dynamic_symbols_t* syms;
} iree_hal_hip_context_wrapper_t;

#endif  // IREE_EXPERIMENTAL_HIP_CONTEXT_WRAPPER_H_