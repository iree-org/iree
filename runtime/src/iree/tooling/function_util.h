// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOOLING_FUNCTION_UTIL_H_
#define IREE_TOOLING_FUNCTION_UTIL_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/vm/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Appends fences to |list| if the invocation model of |function| requires them
// (has the `iree.abi.model` as `coarse-fences`).
// If no |wait_fence| is provided then the invocation will begin immediately.
// Upon return if |out_signal_fence| is not NULL the caller must wait on the
// returned |out_signal_fence| before accessing the contents of any buffers
// returned from the invocation.
iree_status_t iree_tooling_append_async_fences(
    iree_vm_list_t* list, iree_vm_function_t function,
    iree_hal_device_t* device, iree_hal_fence_t* wait_fence,
    iree_hal_fence_t** out_signal_fence);

// Transfers all buffers in |list| to ones using |target_params|.
// If no |wait_fence| is provided then the transfer will begin immediately.
// If no |signal_fence| is provided then the call will block until the transfer
// completes.
iree_status_t iree_tooling_transfer_variants(
    iree_vm_list_t* list, iree_hal_device_t* target_device,
    iree_hal_allocator_t* target_allocator,
    iree_hal_buffer_params_t target_params, iree_hal_fence_t* wait_fence,
    iree_hal_fence_t* signal_fence);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOOLING_FUNCTION_UTIL_H_
