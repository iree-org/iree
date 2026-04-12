// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_FILE_H_
#define IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_FILE_H_

#include "iree/hal/drivers/amdgpu/virtual_queue.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Implements queue_read for files with a device-accessible storage buffer.
iree_status_t iree_hal_amdgpu_host_queue_read_file(
    iree_hal_amdgpu_virtual_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_file_t* source_file, uint64_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_read_flags_t flags);

// Implements queue_write for files with a device-accessible storage buffer.
iree_status_t iree_hal_amdgpu_host_queue_write_file(
    iree_hal_amdgpu_virtual_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_file_t* target_file, uint64_t target_offset,
    iree_device_size_t length, iree_hal_write_flags_t flags);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_FILE_H_
