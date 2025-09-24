// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_DEVICE_KERNELS_H_
#define IREE_HAL_DRIVERS_AMDGPU_DEVICE_KERNELS_H_

#include "iree/hal/drivers/amdgpu/device/support/common.h"
#include "iree/hal/drivers/amdgpu/device/support/kernel_args.h"

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_device_kernels_t
//===----------------------------------------------------------------------===//

// Opaque handles used to launch builtin kernels.
// Stored on the command buffer as they are constant for the lifetime of the
// program and we may have command buffers opt into different DMA modes.
typedef struct iree_hal_amdgpu_device_kernels_t {
#define IREE_HAL_AMDGPU_DEVICE_KERNEL(name, workgroup_size_x,             \
                                      workgroup_size_y, workgroup_size_z) \
  iree_hal_amdgpu_device_kernel_args_t name;
#include "iree/hal/drivers/amdgpu/device/kernel_tables.h"
#undef IREE_HAL_AMDGPU_DEVICE_KERNEL
} iree_hal_amdgpu_device_kernels_t;

#endif  // IREE_HAL_DRIVERS_AMDGPU_DEVICE_KERNELS_H_
