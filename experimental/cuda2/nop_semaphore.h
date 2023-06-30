// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_CUDA2_NOP_SEMAPHORE_H_
#define EXPERIMENTAL_CUDA2_NOP_SEMAPHORE_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates a HAL semaphore for CUDA that does not perform real synchronization.
// This is expected to work with a command buffer that serializes all commands.
iree_status_t iree_hal_cuda2_semaphore_create(
    uint64_t initial_value, iree_allocator_t host_allocator,
    iree_hal_semaphore_t** out_semaphore);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_CUDA2_NOP_SEMAPHORE_H_
