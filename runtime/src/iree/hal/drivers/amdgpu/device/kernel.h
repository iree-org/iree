// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_DEVICE_KERNEL_H_
#define IREE_HAL_DRIVERS_AMDGPU_DEVICE_KERNEL_H_

#include "iree/hal/drivers/amdgpu/device/support/common.h"

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_device_kernel_args_t
//===----------------------------------------------------------------------===//

// Kernel arguments used for fixed-size kernels.
// This must match what the kernel was compiled to support.
typedef struct iree_hal_amdgpu_device_kernel_args_s {
  // Opaque handle to the kernel object to execute.
  uint64_t kernel_object;
  // hsa_kernel_dispatch_packet_setup_t (grid dimension count).
  uint16_t setup;
  // XYZ dimensions of work-group, in work-items. Must be greater than 0.
  // If the grid has fewer than 3 dimensions the unused must be 1.
  uint16_t workgroup_size[3];
  // Size in bytes of private memory allocation request (per work-item).
  uint32_t private_segment_size;
  // Size in bytes of group memory allocation request (per work-group). Must
  // not be less than the sum of the group memory used by the kernel (and the
  // functions it calls directly or indirectly) and the dynamically allocated
  // group segment variables.
  uint32_t group_segment_size;
  // Allocated source location in host memory. Inaccessible and only here to
  // feed back to the host for trace processing.
  uint64_t trace_src_loc;
} iree_hal_amdgpu_device_kernel_args_t;

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_device_kernels_t
//===----------------------------------------------------------------------===//

// Opaque handles used to launch builtin kernels.
// Stored on the command buffer as they are constant for the lifetime of the
// program and we may have command buffers opt into different DMA modes.
typedef struct iree_hal_amdgpu_device_kernels_s {
  // `iree_hal_amdgpu_device_queue_scheduler_tick` kernel.
  iree_hal_amdgpu_device_kernel_args_t scheduler_tick;
  // `iree_hal_amdgpu_device_queue_retire_entry` kernel.
  iree_hal_amdgpu_device_kernel_args_t retire_entry;
  // `iree_hal_amdgpu_device_cmd_block_issue` kernel.
  iree_hal_amdgpu_device_kernel_args_t cmd_block_issue;
  // `iree_hal_amdgpu_device_cmd_dispatch_indirect_update` kernel.
  iree_hal_amdgpu_device_kernel_args_t cmd_dispatch_indirect_update;
  // `iree_hal_amdgpu_device_cmd_branch` kernel.
  iree_hal_amdgpu_device_kernel_args_t cmd_branch;
  // `iree_hal_amdgpu_device_cmd_return` kernel.
  iree_hal_amdgpu_device_kernel_args_t cmd_return;
  // Kernels used to implement DMA-like operations.
  struct {
    iree_hal_amdgpu_device_kernel_args_t
        fill_x1;  // iree_hal_amdgpu_device_buffer_fill_x1
    iree_hal_amdgpu_device_kernel_args_t
        fill_x2;  // iree_hal_amdgpu_device_buffer_fill_x2
    iree_hal_amdgpu_device_kernel_args_t
        fill_x4;  // iree_hal_amdgpu_device_buffer_fill_x4
    iree_hal_amdgpu_device_kernel_args_t
        fill_x8;  // iree_hal_amdgpu_device_buffer_fill_x8
    iree_hal_amdgpu_device_kernel_args_t
        copy_x1;  // iree_hal_amdgpu_device_buffer_copy_x1
    iree_hal_amdgpu_device_kernel_args_t
        copy_x2;  // iree_hal_amdgpu_device_buffer_copy_x2
    iree_hal_amdgpu_device_kernel_args_t
        copy_x4;  // iree_hal_amdgpu_device_buffer_copy_x4
    iree_hal_amdgpu_device_kernel_args_t
        copy_x8;  // iree_hal_amdgpu_device_buffer_copy_x8
    iree_hal_amdgpu_device_kernel_args_t
        copy_x64;  // iree_hal_amdgpu_device_buffer_copy_x64
  } blit;
} iree_hal_amdgpu_device_kernels_t;

#endif  // IREE_HAL_DRIVERS_AMDGPU_DEVICE_KERNEL_H_
