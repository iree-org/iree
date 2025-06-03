// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_DEVICE_SUPPORT_KERNEL_ARGS_H_
#define IREE_HAL_DRIVERS_AMDGPU_DEVICE_SUPPORT_KERNEL_ARGS_H_

#include "iree/hal/drivers/amdgpu/device/support/common.h"

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_device_kernel_args_t
//===----------------------------------------------------------------------===//

// Kernel arguments used for fixed-size kernels.
// This must match what the kernel was compiled to support.
typedef struct iree_hal_amdgpu_device_kernel_args_t {
  // Opaque handle to the kernel object to execute.
  uint64_t kernel_object;
  // Dispatch setup parameters. Used to configure kernel dispatch parameters
  // such as the number of dimensions in the grid. The parameters are
  // described by hsa_kernel_dispatch_packet_setup_t.
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
  // Size of kernarg segment memory that is required to hold the values of the
  // kernel arguments, in bytes. Must be a multiple of 16.
  uint16_t kernarg_size;
  // Alignment (in bytes) of the buffer used to pass arguments to the kernel,
  // which is the maximum of 16 and the maximum alignment of any of the kernel
  // arguments.
  uint16_t kernarg_alignment;
  // Allocated source location in host memory. Inaccessible and only here to
  // feed back to the host for trace processing.
  uint64_t trace_src_loc;
  // Total number of 4-byte constants used by the dispatch (if a HAL dispatch).
  uint16_t constant_count;
  // Total number of bindings used by the dispatch (if a HAL dispatch).
  uint16_t binding_count;
  uint32_t reserved;
} iree_hal_amdgpu_device_kernel_args_t;
static_assert(
    sizeof(iree_hal_amdgpu_device_kernel_args_t) <= 64,
    "keep hot kernel arg structure in as few cache lines as possible; every "
    "dispatch issued must access this information and it is likely uncached");

#endif  // IREE_HAL_DRIVERS_AMDGPU_DEVICE_SUPPORT_KERNEL_ARGS_H_
