// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// Blits (blit.h)
//===----------------------------------------------------------------------===//

// Conservative metadata defaults used when loading builtin kernel descriptors.
// Transfer packet emission overrides the X dimension from runtime wavefront
// metadata so the same code object table can support wave32 and wave64 devices.
#define IREE_HAL_AMDGPU_BLIT_WORKGROUP_SIZE_X 32
#define IREE_HAL_AMDGPU_BLIT_WORKGROUP_SIZE_Y 1
#define IREE_HAL_AMDGPU_BLIT_WORKGROUP_SIZE_Z 1

IREE_HAL_AMDGPU_DEVICE_KERNEL(iree_hal_amdgpu_device_buffer_fill_x1,
                              IREE_HAL_AMDGPU_BLIT_WORKGROUP_SIZE_X,
                              IREE_HAL_AMDGPU_BLIT_WORKGROUP_SIZE_Y,
                              IREE_HAL_AMDGPU_BLIT_WORKGROUP_SIZE_Z)
IREE_HAL_AMDGPU_DEVICE_KERNEL(iree_hal_amdgpu_device_buffer_fill_x2,
                              IREE_HAL_AMDGPU_BLIT_WORKGROUP_SIZE_X,
                              IREE_HAL_AMDGPU_BLIT_WORKGROUP_SIZE_Y,
                              IREE_HAL_AMDGPU_BLIT_WORKGROUP_SIZE_Z)
IREE_HAL_AMDGPU_DEVICE_KERNEL(iree_hal_amdgpu_device_buffer_fill_x4,
                              IREE_HAL_AMDGPU_BLIT_WORKGROUP_SIZE_X,
                              IREE_HAL_AMDGPU_BLIT_WORKGROUP_SIZE_Y,
                              IREE_HAL_AMDGPU_BLIT_WORKGROUP_SIZE_Z)
IREE_HAL_AMDGPU_DEVICE_KERNEL(iree_hal_amdgpu_device_buffer_fill_x8,
                              IREE_HAL_AMDGPU_BLIT_WORKGROUP_SIZE_X,
                              IREE_HAL_AMDGPU_BLIT_WORKGROUP_SIZE_Y,
                              IREE_HAL_AMDGPU_BLIT_WORKGROUP_SIZE_Z)
IREE_HAL_AMDGPU_DEVICE_KERNEL(iree_hal_amdgpu_device_buffer_fill_block_x16,
                              IREE_HAL_AMDGPU_BLIT_WORKGROUP_SIZE_X,
                              IREE_HAL_AMDGPU_BLIT_WORKGROUP_SIZE_Y,
                              IREE_HAL_AMDGPU_BLIT_WORKGROUP_SIZE_Z)
IREE_HAL_AMDGPU_DEVICE_KERNEL(
    iree_hal_amdgpu_device_buffer_fill_block_unaligned_x16,
    IREE_HAL_AMDGPU_BLIT_WORKGROUP_SIZE_X,
    IREE_HAL_AMDGPU_BLIT_WORKGROUP_SIZE_Y,
    IREE_HAL_AMDGPU_BLIT_WORKGROUP_SIZE_Z)
IREE_HAL_AMDGPU_DEVICE_KERNEL(iree_hal_amdgpu_device_buffer_copy_x1,
                              IREE_HAL_AMDGPU_BLIT_WORKGROUP_SIZE_X,
                              IREE_HAL_AMDGPU_BLIT_WORKGROUP_SIZE_Y,
                              IREE_HAL_AMDGPU_BLIT_WORKGROUP_SIZE_Z)
IREE_HAL_AMDGPU_DEVICE_KERNEL(iree_hal_amdgpu_device_buffer_copy_block_x4,
                              IREE_HAL_AMDGPU_BLIT_WORKGROUP_SIZE_X,
                              IREE_HAL_AMDGPU_BLIT_WORKGROUP_SIZE_Y,
                              IREE_HAL_AMDGPU_BLIT_WORKGROUP_SIZE_Z)
IREE_HAL_AMDGPU_DEVICE_KERNEL(iree_hal_amdgpu_device_buffer_copy_block_x8,
                              IREE_HAL_AMDGPU_BLIT_WORKGROUP_SIZE_X,
                              IREE_HAL_AMDGPU_BLIT_WORKGROUP_SIZE_Y,
                              IREE_HAL_AMDGPU_BLIT_WORKGROUP_SIZE_Z)
IREE_HAL_AMDGPU_DEVICE_KERNEL(iree_hal_amdgpu_device_buffer_copy_block_x16,
                              IREE_HAL_AMDGPU_BLIT_WORKGROUP_SIZE_X,
                              IREE_HAL_AMDGPU_BLIT_WORKGROUP_SIZE_Y,
                              IREE_HAL_AMDGPU_BLIT_WORKGROUP_SIZE_Z)
IREE_HAL_AMDGPU_DEVICE_KERNEL(
    iree_hal_amdgpu_device_buffer_copy_block_unaligned_x16,
    IREE_HAL_AMDGPU_BLIT_WORKGROUP_SIZE_X,
    IREE_HAL_AMDGPU_BLIT_WORKGROUP_SIZE_Y,
    IREE_HAL_AMDGPU_BLIT_WORKGROUP_SIZE_Z)
