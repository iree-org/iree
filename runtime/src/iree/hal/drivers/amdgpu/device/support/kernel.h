// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Device-side kernel execution geometry helpers built on the AMDGPU dispatch
// packet ABI. For kernel argument and packet layouts see abi/kernel_args.h and
// abi/queue.h (exported below).
//
// These helpers intentionally avoid depending on device queue mutation APIs.
// Builtin kernels and tracing paths can include this header to inspect the
// currently executing dispatch without inheriting the old device-side enqueue
// surface.

#ifndef IREE_HAL_DRIVERS_AMDGPU_DEVICE_SUPPORT_KERNEL_H_
#define IREE_HAL_DRIVERS_AMDGPU_DEVICE_SUPPORT_KERNEL_H_

#include "iree/hal/drivers/amdgpu/abi/kernel_args.h"  // IWYU pragma: export
#include "iree/hal/drivers/amdgpu/abi/queue.h"        // IWYU pragma: export
#include "iree/hal/drivers/amdgpu/device/support/common.h"

//===----------------------------------------------------------------------===//
// OpenCL/HIP Dispatch ABI
//===----------------------------------------------------------------------===//
// These come from llvm-project/amd/device-libs/ockl/src/workitem.cl (the ockl
// functions) and llvm-project/clang/lib/CodeGen/CGBuiltin.cpp (e.g.
// EmitAMDGPUWorkGroupSize). Using either runs a chance of pulling in the
// entire iree_amdgpu_kernel_implicit_args_t struct and we don't want to set
// that. We also don't need it: we aren't requiring OpenCL compatibility and
// have no need for the extra features provided by the implicit args (like
// workgroup offset and device-side enqueue - that's our job).

#if defined(IREE_AMDGPU_TARGET_DEVICE)

// Returns the pointer to the iree_hsa_kernel_dispatch_packet_t being executed.
#define iree_amdgcn_dispatch_ptr()                                 \
  ((const iree_hsa_kernel_dispatch_packet_t* IREE_AMDGPU_RESTRICT) \
       __builtin_amdgcn_dispatch_ptr())

// __ockl_get_global_id(0) / get_global_id_x using OLD_ABI.
static inline IREE_AMDGPU_ATTRIBUTE_ALWAYS_INLINE size_t
iree_hal_amdgpu_device_global_id_x(void) {
  const uint32_t local_id = __builtin_amdgcn_workitem_id_x();
  const uint32_t group_id = __builtin_amdgcn_workgroup_id_x();
  const uint32_t group_size = iree_amdgcn_dispatch_ptr()->workgroup_size[0];
  return group_id * group_size + local_id;
}
static inline IREE_AMDGPU_ATTRIBUTE_ALWAYS_INLINE size_t
iree_hal_amdgpu_device_global_id_y(void) {
  const uint32_t local_id = __builtin_amdgcn_workitem_id_y();
  const uint32_t group_id = __builtin_amdgcn_workgroup_id_y();
  const uint32_t group_size = iree_amdgcn_dispatch_ptr()->workgroup_size[1];
  return group_id * group_size + local_id;
}
static inline IREE_AMDGPU_ATTRIBUTE_ALWAYS_INLINE size_t
iree_hal_amdgpu_device_global_id_z(void) {
  const uint32_t local_id = __builtin_amdgcn_workitem_id_z();
  const uint32_t group_id = __builtin_amdgcn_workgroup_id_z();
  const uint32_t group_size = iree_amdgcn_dispatch_ptr()->workgroup_size[2];
  return group_id * group_size + local_id;
}

// __ockl_get_group_id(0)
#define iree_hal_amdgpu_device_group_id_x() __builtin_amdgcn_workgroup_id_x()
#define iree_hal_amdgpu_device_group_id_y() __builtin_amdgcn_workgroup_id_y()
#define iree_hal_amdgpu_device_group_id_z() __builtin_amdgcn_workgroup_id_z()

// __ockl_get_num_groups(0)
static inline IREE_AMDGPU_ATTRIBUTE_ALWAYS_INLINE size_t
iree_hal_amdgpu_device_group_count_x(void) {
  const uint32_t grid_size = iree_amdgcn_dispatch_ptr()->grid_size[0];
  const uint32_t group_size = iree_amdgcn_dispatch_ptr()->workgroup_size[0];
  const uint32_t group_count = grid_size / group_size;
  return group_count + (grid_size > group_count * group_size);
}
static inline IREE_AMDGPU_ATTRIBUTE_ALWAYS_INLINE size_t
iree_hal_amdgpu_device_group_count_y(void) {
  const uint32_t grid_size = iree_amdgcn_dispatch_ptr()->grid_size[1];
  const uint32_t group_size = iree_amdgcn_dispatch_ptr()->workgroup_size[1];
  const uint32_t group_count = grid_size / group_size;
  return group_count + (grid_size > group_count * group_size);
}
static inline IREE_AMDGPU_ATTRIBUTE_ALWAYS_INLINE size_t
iree_hal_amdgpu_device_group_count_z(void) {
  const uint32_t grid_size = iree_amdgcn_dispatch_ptr()->grid_size[2];
  const uint32_t group_size = iree_amdgcn_dispatch_ptr()->workgroup_size[2];
  const uint32_t group_count = grid_size / group_size;
  return group_count + (grid_size > group_count * group_size);
}

// __ockl_get_local_id(0)
#define iree_hal_amdgpu_device_local_id_x() __builtin_amdgcn_workitem_id_x()
#define iree_hal_amdgpu_device_local_id_y() __builtin_amdgcn_workitem_id_y()
#define iree_hal_amdgpu_device_local_id_z() __builtin_amdgcn_workitem_id_z()

// __ockl_get_local_size(0) / get_local_size_x
static inline IREE_AMDGPU_ATTRIBUTE_ALWAYS_INLINE size_t
iree_hal_amdgpu_device_workgroup_size_x(void) {
  const uint32_t group_id = __builtin_amdgcn_workgroup_id_x();
  const uint32_t group_size = iree_amdgcn_dispatch_ptr()->workgroup_size[0];
  const uint32_t grid_size = iree_amdgcn_dispatch_ptr()->grid_size[0];
  const uint32_t remainder = grid_size - group_id * group_size;
  return IREE_AMDGPU_MIN(remainder, group_size);
}
static inline IREE_AMDGPU_ATTRIBUTE_ALWAYS_INLINE size_t
iree_hal_amdgpu_device_workgroup_size_y(void) {
  const uint32_t group_id = __builtin_amdgcn_workgroup_id_y();
  const uint32_t group_size = iree_amdgcn_dispatch_ptr()->workgroup_size[1];
  const uint32_t grid_size = iree_amdgcn_dispatch_ptr()->grid_size[1];
  const uint32_t remainder = grid_size - group_id * group_size;
  return IREE_AMDGPU_MIN(remainder, group_size);
}
static inline IREE_AMDGPU_ATTRIBUTE_ALWAYS_INLINE size_t
iree_hal_amdgpu_device_workgroup_size_z(void) {
  const uint32_t group_id = __builtin_amdgcn_workgroup_id_z();
  const uint32_t group_size = iree_amdgcn_dispatch_ptr()->workgroup_size[2];
  const uint32_t grid_size = iree_amdgcn_dispatch_ptr()->grid_size[2];
  const uint32_t remainder = grid_size - group_id * group_size;
  return IREE_AMDGPU_MIN(remainder, group_size);
}

static inline IREE_AMDGPU_ATTRIBUTE_ALWAYS_INLINE size_t
iree_hal_amdgpu_device_global_linear_id_1d(void) {
  return iree_hal_amdgpu_device_group_id_x() *
             iree_amdgcn_dispatch_ptr()->workgroup_size[0] +
         iree_hal_amdgpu_device_local_id_x();
}

static inline IREE_AMDGPU_ATTRIBUTE_ALWAYS_INLINE size_t
iree_hal_amdgpu_device_global_linear_id_2d(void) {
  const size_t id_x = iree_hal_amdgpu_device_group_id_x() *
                          iree_amdgcn_dispatch_ptr()->workgroup_size[0] +
                      iree_hal_amdgpu_device_local_id_x();
  const size_t id_y = iree_hal_amdgpu_device_group_id_y() *
                          iree_amdgcn_dispatch_ptr()->workgroup_size[1] +
                      iree_hal_amdgpu_device_local_id_y();
  return id_y * iree_amdgcn_dispatch_ptr()->grid_size[0] + id_x;
}

static inline IREE_AMDGPU_ATTRIBUTE_ALWAYS_INLINE size_t
iree_hal_amdgpu_device_global_linear_id_3d(void) {
  const size_t id_x = iree_hal_amdgpu_device_group_id_x() *
                          iree_amdgcn_dispatch_ptr()->workgroup_size[0] +
                      iree_hal_amdgpu_device_local_id_x();
  const size_t id_y = iree_hal_amdgpu_device_group_id_y() *
                          iree_amdgcn_dispatch_ptr()->workgroup_size[1] +
                      iree_hal_amdgpu_device_local_id_y();
  const size_t id_z = iree_hal_amdgpu_device_group_id_z() *
                          iree_amdgcn_dispatch_ptr()->workgroup_size[2] +
                      iree_hal_amdgpu_device_local_id_z();
  return (id_z * iree_amdgcn_dispatch_ptr()->grid_size[1] + id_y) *
             iree_amdgcn_dispatch_ptr()->grid_size[0] +
         id_x;
}

#endif  // IREE_AMDGPU_TARGET_DEVICE

#endif  // IREE_HAL_DRIVERS_AMDGPU_DEVICE_SUPPORT_KERNEL_H_
