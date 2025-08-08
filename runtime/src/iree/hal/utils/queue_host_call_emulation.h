// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_UTILS_QUEUE_HOST_CALL_EMULATION_H_
#define IREE_HAL_UTILS_QUEUE_HOST_CALL_EMULATION_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Emulated Host Call
//===----------------------------------------------------------------------===//

#if IREE_THREADING_ENABLE
IREE_API_EXPORT iree_status_t iree_hal_device_queue_emulated_host_call(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_host_call_t call, const uint64_t args[4],
    iree_hal_host_call_flags_t flags);
#endif  // IREE_THREADING_ENABLE

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_UTILS_QUEUE_HOST_CALL_EMULATION_H_
