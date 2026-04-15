// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_HOST_CALL_H_
#define IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_HOST_CALL_H_

#include "iree/hal/drivers/amdgpu/host_queue_submission.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Validates host-call parameters before capture or submission.
iree_status_t iree_hal_amdgpu_host_queue_validate_host_call(
    iree_hal_host_call_t call, const uint64_t args[4],
    iree_hal_host_call_flags_t flags);

// Emits a host-call barrier epoch. Caller must hold submission_mutex.
iree_status_t iree_hal_amdgpu_host_queue_submit_host_call(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_host_call_t call, const uint64_t args[4],
    iree_hal_host_call_flags_t flags, bool* out_ready);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_HOST_CALL_H_
