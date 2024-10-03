// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// Blits (buffer.h)
//===----------------------------------------------------------------------===//

// DO NOT SUBMIT workgroup size for fills/copies
IREE_HAL_AMDGPU_DEVICE_KERNEL(iree_hal_amdgpu_device_buffer_fill_x1, 1, 1, 1)
IREE_HAL_AMDGPU_DEVICE_KERNEL(iree_hal_amdgpu_device_buffer_fill_x2, 1, 1, 1)
IREE_HAL_AMDGPU_DEVICE_KERNEL(iree_hal_amdgpu_device_buffer_fill_x4, 1, 1, 1)
IREE_HAL_AMDGPU_DEVICE_KERNEL(iree_hal_amdgpu_device_buffer_fill_x8, 1, 1, 1)
IREE_HAL_AMDGPU_DEVICE_KERNEL(iree_hal_amdgpu_device_buffer_copy_x1, 1, 1, 1)
IREE_HAL_AMDGPU_DEVICE_KERNEL(iree_hal_amdgpu_device_buffer_copy_x2, 1, 1, 1)
IREE_HAL_AMDGPU_DEVICE_KERNEL(iree_hal_amdgpu_device_buffer_copy_x4, 1, 1, 1)
IREE_HAL_AMDGPU_DEVICE_KERNEL(iree_hal_amdgpu_device_buffer_copy_x8, 1, 1, 1)
IREE_HAL_AMDGPU_DEVICE_KERNEL(iree_hal_amdgpu_device_buffer_copy_x64, 1, 1, 1)

//===----------------------------------------------------------------------===//
// Command buffers (command_buffer.h)
//===----------------------------------------------------------------------===//

// DO NOT SUBMIT workgroup size for issue
IREE_HAL_AMDGPU_DEVICE_KERNEL(iree_hal_amdgpu_device_cmd_block_issue, 1, 1, 1)
IREE_HAL_AMDGPU_DEVICE_KERNEL(
    iree_hal_amdgpu_device_cmd_dispatch_indirect_update, 1, 1, 1)
IREE_HAL_AMDGPU_DEVICE_KERNEL(iree_hal_amdgpu_device_cmd_branch, 1, 1, 1)
IREE_HAL_AMDGPU_DEVICE_KERNEL(iree_hal_amdgpu_device_cmd_return, 1, 1, 1)

//===----------------------------------------------------------------------===//
// Scheduling (scheduler.h)
//===----------------------------------------------------------------------===//

IREE_HAL_AMDGPU_DEVICE_KERNEL(iree_hal_amdgpu_device_queue_scheduler_initialize,
                              1, 1, 1)
IREE_HAL_AMDGPU_DEVICE_KERNEL(iree_hal_amdgpu_device_queue_scheduler_tick, 1, 1,
                              1)
IREE_HAL_AMDGPU_DEVICE_KERNEL(iree_hal_amdgpu_device_queue_retire_entry, 1, 1,
                              1)

//===----------------------------------------------------------------------===//
// Tracing (tracing.h)
//===----------------------------------------------------------------------===//

IREE_HAL_AMDGPU_DEVICE_KERNEL(iree_hal_amdgpu_device_trace_buffer_initialize,
                              32, 1, 1)
