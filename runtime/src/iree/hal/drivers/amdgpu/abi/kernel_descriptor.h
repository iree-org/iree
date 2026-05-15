// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_ABI_KERNEL_DESCRIPTOR_H_
#define IREE_HAL_DRIVERS_AMDGPU_ABI_KERNEL_DESCRIPTOR_H_

#include "iree/hal/drivers/amdgpu/abi/common.h"

// AMDHSA code object v3+ kernel descriptor layout. HSA exposes the descriptor
// device address as the executable symbol's kernel object, and the descriptor
// payload contains the PM4 shader resource register values needed for raw
// COMPUTE dispatch.
typedef struct iree_hal_amdgpu_kernel_descriptor_t {
  // Static group segment byte count required by each workgroup.
  uint32_t group_segment_fixed_size;
  // Static private segment byte count required by each workitem.
  uint32_t private_segment_fixed_size;
  // Kernel argument segment byte count.
  uint32_t kernarg_size;
  // Reserved dwords between the segment sizes and entry offset.
  uint8_t reserved0[4];
  // Signed byte offset from the descriptor address to the code entry point.
  int64_t kernel_code_entry_byte_offset;
  // Reserved descriptor payload before COMPUTE_PGM_RSRC3.
  uint8_t reserved1[20];
  // COMPUTE_PGM_RSRC3 payload for gfx10+ and gfx90a+.
  uint32_t compute_pgm_rsrc3;
  // COMPUTE_PGM_RSRC1 payload.
  uint32_t compute_pgm_rsrc1;
  // COMPUTE_PGM_RSRC2 payload.
  uint32_t compute_pgm_rsrc2;
  // Kernel code property bits describing hidden SGPR inputs and wave size.
  uint16_t kernel_code_properties;
  // Kernarg preload descriptor used by architectures that support preloading.
  uint16_t kernarg_preload;
  // Reserved descriptor trailer.
  uint8_t reserved3[4];
} iree_hal_amdgpu_kernel_descriptor_t;

enum {
  IREE_HAL_AMDGPU_KERNEL_DESCRIPTOR_GROUP_SEGMENT_FIXED_SIZE_OFFSET = 0,
  IREE_HAL_AMDGPU_KERNEL_DESCRIPTOR_PRIVATE_SEGMENT_FIXED_SIZE_OFFSET = 4,
  IREE_HAL_AMDGPU_KERNEL_DESCRIPTOR_KERNARG_SIZE_OFFSET = 8,
  IREE_HAL_AMDGPU_KERNEL_DESCRIPTOR_RESERVED0_OFFSET = 12,
  IREE_HAL_AMDGPU_KERNEL_DESCRIPTOR_CODE_ENTRY_BYTE_OFFSET_OFFSET = 16,
  IREE_HAL_AMDGPU_KERNEL_DESCRIPTOR_RESERVED1_OFFSET = 24,
  IREE_HAL_AMDGPU_KERNEL_DESCRIPTOR_COMPUTE_PGM_RSRC3_OFFSET = 44,
  IREE_HAL_AMDGPU_KERNEL_DESCRIPTOR_COMPUTE_PGM_RSRC1_OFFSET = 48,
  IREE_HAL_AMDGPU_KERNEL_DESCRIPTOR_COMPUTE_PGM_RSRC2_OFFSET = 52,
  IREE_HAL_AMDGPU_KERNEL_DESCRIPTOR_KERNEL_CODE_PROPERTIES_OFFSET = 56,
  IREE_HAL_AMDGPU_KERNEL_DESCRIPTOR_KERNARG_PRELOAD_OFFSET = 58,
  IREE_HAL_AMDGPU_KERNEL_DESCRIPTOR_RESERVED3_OFFSET = 60,
  IREE_HAL_AMDGPU_KERNEL_DESCRIPTOR_SIZE = 64,
};

enum {
  IREE_HAL_AMDGPU_KERNEL_DESCRIPTOR_KERNARG_PRELOAD_LENGTH_SHIFT = 0,
  IREE_HAL_AMDGPU_KERNEL_DESCRIPTOR_KERNARG_PRELOAD_LENGTH_MASK = 0x007Fu,
  IREE_HAL_AMDGPU_KERNEL_DESCRIPTOR_KERNARG_PRELOAD_OFFSET_SHIFT = 7,
  IREE_HAL_AMDGPU_KERNEL_DESCRIPTOR_KERNARG_PRELOAD_OFFSET_MASK = 0xFF80u,
};

enum {
  IREE_HAL_AMDGPU_COMPUTE_PGM_RSRC2_ENABLE_PRIVATE_SEGMENT = 1u << 0,
  IREE_HAL_AMDGPU_COMPUTE_PGM_RSRC2_USER_SGPR_COUNT_SHIFT = 1,
  IREE_HAL_AMDGPU_COMPUTE_PGM_RSRC2_USER_SGPR_COUNT_MASK = 0x3Eu,
  IREE_HAL_AMDGPU_COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X = 1u << 7,
  IREE_HAL_AMDGPU_COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Y = 1u << 8,
  IREE_HAL_AMDGPU_COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Z = 1u << 9,
  IREE_HAL_AMDGPU_COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_INFO = 1u << 10,
  IREE_HAL_AMDGPU_COMPUTE_PGM_RSRC2_ENABLE_VGPR_WORKITEM_ID_MASK = 3u << 11,
  IREE_HAL_AMDGPU_COMPUTE_PGM_RSRC2_GRANULATED_LDS_SIZE_SHIFT = 15,
  IREE_HAL_AMDGPU_COMPUTE_PGM_RSRC2_GRANULATED_LDS_SIZE_MASK = 0x1FFu << 15,
};

enum {
  IREE_HAL_AMDGPU_KERNEL_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER =
      1u << 0,
  IREE_HAL_AMDGPU_KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_PTR = 1u << 1,
  IREE_HAL_AMDGPU_KERNEL_CODE_PROPERTY_ENABLE_SGPR_QUEUE_PTR = 1u << 2,
  IREE_HAL_AMDGPU_KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR = 1u
                                                                         << 3,
  IREE_HAL_AMDGPU_KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_ID = 1u << 4,
  IREE_HAL_AMDGPU_KERNEL_CODE_PROPERTY_ENABLE_SGPR_FLAT_SCRATCH_INIT = 1u << 5,
  IREE_HAL_AMDGPU_KERNEL_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_SIZE = 1u
                                                                          << 6,
  IREE_HAL_AMDGPU_KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32 = 1u << 10,
  IREE_HAL_AMDGPU_KERNEL_CODE_PROPERTY_USES_DYNAMIC_STACK = 1u << 11,
};

IREE_AMDGPU_STATIC_ASSERT(sizeof(iree_hal_amdgpu_kernel_descriptor_t) ==
                              IREE_HAL_AMDGPU_KERNEL_DESCRIPTOR_SIZE,
                          "AMDHSA kernel descriptor size must match the ABI");
IREE_AMDGPU_STATIC_ASSERT(
    IREE_AMDGPU_OFFSETOF(iree_hal_amdgpu_kernel_descriptor_t,
                         group_segment_fixed_size) ==
        IREE_HAL_AMDGPU_KERNEL_DESCRIPTOR_GROUP_SEGMENT_FIXED_SIZE_OFFSET,
    "group_segment_fixed_size offset must match the ABI");
IREE_AMDGPU_STATIC_ASSERT(
    IREE_AMDGPU_OFFSETOF(iree_hal_amdgpu_kernel_descriptor_t,
                         private_segment_fixed_size) ==
        IREE_HAL_AMDGPU_KERNEL_DESCRIPTOR_PRIVATE_SEGMENT_FIXED_SIZE_OFFSET,
    "private_segment_fixed_size offset must match the ABI");
IREE_AMDGPU_STATIC_ASSERT(
    IREE_AMDGPU_OFFSETOF(iree_hal_amdgpu_kernel_descriptor_t, kernarg_size) ==
        IREE_HAL_AMDGPU_KERNEL_DESCRIPTOR_KERNARG_SIZE_OFFSET,
    "kernarg_size offset must match the ABI");
IREE_AMDGPU_STATIC_ASSERT(
    IREE_AMDGPU_OFFSETOF(iree_hal_amdgpu_kernel_descriptor_t, reserved0) ==
        IREE_HAL_AMDGPU_KERNEL_DESCRIPTOR_RESERVED0_OFFSET,
    "reserved0 offset must match the ABI");
IREE_AMDGPU_STATIC_ASSERT(
    IREE_AMDGPU_OFFSETOF(iree_hal_amdgpu_kernel_descriptor_t,
                         kernel_code_entry_byte_offset) ==
        IREE_HAL_AMDGPU_KERNEL_DESCRIPTOR_CODE_ENTRY_BYTE_OFFSET_OFFSET,
    "kernel_code_entry_byte_offset offset must match the ABI");
IREE_AMDGPU_STATIC_ASSERT(
    IREE_AMDGPU_OFFSETOF(iree_hal_amdgpu_kernel_descriptor_t, reserved1) ==
        IREE_HAL_AMDGPU_KERNEL_DESCRIPTOR_RESERVED1_OFFSET,
    "reserved1 offset must match the ABI");
IREE_AMDGPU_STATIC_ASSERT(
    IREE_AMDGPU_OFFSETOF(iree_hal_amdgpu_kernel_descriptor_t,
                         compute_pgm_rsrc3) ==
        IREE_HAL_AMDGPU_KERNEL_DESCRIPTOR_COMPUTE_PGM_RSRC3_OFFSET,
    "compute_pgm_rsrc3 offset must match the ABI");
IREE_AMDGPU_STATIC_ASSERT(
    IREE_AMDGPU_OFFSETOF(iree_hal_amdgpu_kernel_descriptor_t,
                         compute_pgm_rsrc1) ==
        IREE_HAL_AMDGPU_KERNEL_DESCRIPTOR_COMPUTE_PGM_RSRC1_OFFSET,
    "compute_pgm_rsrc1 offset must match the ABI");
IREE_AMDGPU_STATIC_ASSERT(
    IREE_AMDGPU_OFFSETOF(iree_hal_amdgpu_kernel_descriptor_t,
                         compute_pgm_rsrc2) ==
        IREE_HAL_AMDGPU_KERNEL_DESCRIPTOR_COMPUTE_PGM_RSRC2_OFFSET,
    "compute_pgm_rsrc2 offset must match the ABI");
IREE_AMDGPU_STATIC_ASSERT(
    IREE_AMDGPU_OFFSETOF(iree_hal_amdgpu_kernel_descriptor_t,
                         kernel_code_properties) ==
        IREE_HAL_AMDGPU_KERNEL_DESCRIPTOR_KERNEL_CODE_PROPERTIES_OFFSET,
    "kernel_code_properties offset must match the ABI");
IREE_AMDGPU_STATIC_ASSERT(
    IREE_AMDGPU_OFFSETOF(iree_hal_amdgpu_kernel_descriptor_t,
                         kernarg_preload) ==
        IREE_HAL_AMDGPU_KERNEL_DESCRIPTOR_KERNARG_PRELOAD_OFFSET,
    "kernarg_preload offset must match the ABI");
IREE_AMDGPU_STATIC_ASSERT(
    IREE_AMDGPU_OFFSETOF(iree_hal_amdgpu_kernel_descriptor_t, reserved3) ==
        IREE_HAL_AMDGPU_KERNEL_DESCRIPTOR_RESERVED3_OFFSET,
    "reserved3 offset must match the ABI");

#endif  // IREE_HAL_DRIVERS_AMDGPU_ABI_KERNEL_DESCRIPTOR_H_
