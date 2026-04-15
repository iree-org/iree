// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/device/dispatch.h"

#include "iree/hal/drivers/amdgpu/device/support/kernel.h"

//===----------------------------------------------------------------------===//
// Dispatch packet emission
//===----------------------------------------------------------------------===//

void iree_hal_amdgpu_device_dispatch_emplace_packet(
    const iree_hal_amdgpu_device_kernel_args_t* IREE_AMDGPU_RESTRICT
        kernel_args,
    const uint32_t workgroup_count[3], uint32_t dynamic_workgroup_local_memory,
    iree_hsa_kernel_dispatch_packet_t* IREE_AMDGPU_RESTRICT dispatch_packet,
    void* IREE_AMDGPU_RESTRICT kernarg_ptr) {
  dispatch_packet->setup = kernel_args->setup;
  dispatch_packet->workgroup_size[0] = kernel_args->workgroup_size[0];
  dispatch_packet->workgroup_size[1] = kernel_args->workgroup_size[1];
  dispatch_packet->workgroup_size[2] = kernel_args->workgroup_size[2];
  dispatch_packet->reserved0 = 0;
  dispatch_packet->grid_size[0] =
      workgroup_count[0] * kernel_args->workgroup_size[0];
  dispatch_packet->grid_size[1] =
      workgroup_count[1] * kernel_args->workgroup_size[1];
  dispatch_packet->grid_size[2] =
      workgroup_count[2] * kernel_args->workgroup_size[2];
  dispatch_packet->private_segment_size = kernel_args->private_segment_size;
  dispatch_packet->group_segment_size =
      kernel_args->group_segment_size + dynamic_workgroup_local_memory;
  dispatch_packet->kernel_object = kernel_args->kernel_object;
  dispatch_packet->kernarg_address = kernarg_ptr;
  dispatch_packet->reserved2 = 0;
  dispatch_packet->completion_signal = iree_hsa_signal_null();
}

//===----------------------------------------------------------------------===//
// Dispatch kernarg emission
//===----------------------------------------------------------------------===//

static void iree_hal_amdgpu_device_dispatch_emplace_implicit_args(
    const iree_hal_amdgpu_device_kernel_args_t* IREE_AMDGPU_RESTRICT
        kernel_args,
    const uint32_t workgroup_count[3], uint32_t dynamic_workgroup_local_memory,
    const iree_hal_amdgpu_device_dispatch_kernarg_layout_t* IREE_AMDGPU_RESTRICT
        layout,
    void* IREE_AMDGPU_RESTRICT kernarg_ptr) {
  if (!layout->has_implicit_args) return;

  iree_amdgpu_kernel_implicit_args_t* IREE_AMDGPU_RESTRICT implicit_args =
      (iree_amdgpu_kernel_implicit_args_t*)((uint8_t*)kernarg_ptr +
                                            layout->implicit_args_offset);
  iree_amdgpu_memset(implicit_args, 0, IREE_AMDGPU_KERNEL_IMPLICIT_ARGS_SIZE);
  implicit_args->block_count[0] = workgroup_count[0];
  implicit_args->block_count[1] = workgroup_count[1];
  implicit_args->block_count[2] = workgroup_count[2];
  implicit_args->group_size[0] = kernel_args->workgroup_size[0];
  implicit_args->group_size[1] = kernel_args->workgroup_size[1];
  implicit_args->group_size[2] = kernel_args->workgroup_size[2];
  implicit_args->grid_dims = 3;
  implicit_args->printf_buffer = NULL;
  implicit_args->hostcall_buffer = NULL;
  implicit_args->dynamic_lds_size = dynamic_workgroup_local_memory;
}

void iree_hal_amdgpu_device_dispatch_emplace_hal_kernargs(
    const iree_hal_amdgpu_device_kernel_args_t* IREE_AMDGPU_RESTRICT
        kernel_args,
    const uint32_t workgroup_count[3], uint32_t dynamic_workgroup_local_memory,
    const iree_hal_amdgpu_device_dispatch_kernarg_layout_t* IREE_AMDGPU_RESTRICT
        layout,
    const uint64_t* IREE_AMDGPU_RESTRICT binding_ptrs,
    const uint32_t* IREE_AMDGPU_RESTRICT constants,
    void* IREE_AMDGPU_RESTRICT kernarg_ptr) {
  iree_amdgpu_memset(kernarg_ptr, 0, layout->total_kernarg_size);

  const size_t binding_bytes =
      (size_t)kernel_args->binding_count * sizeof(uint64_t);
  const size_t constant_bytes =
      (size_t)kernel_args->constant_count * sizeof(uint32_t);
  if (binding_bytes > 0) {
    iree_amdgpu_memcpy(kernarg_ptr, binding_ptrs, binding_bytes);
  }
  if (constant_bytes > 0) {
    iree_amdgpu_memcpy((uint8_t*)kernarg_ptr + binding_bytes, constants,
                       constant_bytes);
  }

  iree_hal_amdgpu_device_dispatch_emplace_implicit_args(
      kernel_args, workgroup_count, dynamic_workgroup_local_memory, layout,
      kernarg_ptr);
}

void iree_hal_amdgpu_device_dispatch_emplace_custom_kernargs(
    const iree_hal_amdgpu_device_dispatch_kernarg_layout_t* IREE_AMDGPU_RESTRICT
        layout,
    const void* IREE_AMDGPU_RESTRICT custom_kernarg_ptr,
    void* IREE_AMDGPU_RESTRICT kernarg_ptr) {
  if (layout->total_kernarg_size > 0) {
    iree_amdgpu_memcpy(kernarg_ptr, custom_kernarg_ptr,
                       layout->total_kernarg_size);
  }
}

//===----------------------------------------------------------------------===//
// Indirect dispatch parameter patching
//===----------------------------------------------------------------------===//

void iree_hal_amdgpu_device_dispatch_emplace_indirect_params_patch(
    const iree_hal_amdgpu_device_kernel_args_t* IREE_AMDGPU_RESTRICT
        patch_kernel_args,
    const uint32_t* IREE_AMDGPU_RESTRICT workgroup_count,
    iree_hsa_kernel_dispatch_packet_t* IREE_AMDGPU_RESTRICT dispatch_packet,
    iree_amdgpu_kernel_implicit_args_t* IREE_AMDGPU_RESTRICT implicit_args,
    iree_hsa_kernel_dispatch_packet_t* IREE_AMDGPU_RESTRICT patch_packet,
    void* IREE_AMDGPU_RESTRICT kernarg_ptr) {
  iree_hal_amdgpu_device_dispatch_patch_indirect_params_args_t*
      IREE_AMDGPU_RESTRICT kernargs =
          (iree_hal_amdgpu_device_dispatch_patch_indirect_params_args_t*)
              kernarg_ptr;
  kernargs->workgroup_count = workgroup_count;
  kernargs->dispatch_packet = dispatch_packet;
  kernargs->implicit_args = implicit_args;

  const uint32_t patch_workgroup_count[3] = {1, 1, 1};
  iree_hal_amdgpu_device_dispatch_emplace_packet(
      patch_kernel_args, patch_workgroup_count,
      /*dynamic_workgroup_local_memory=*/0, patch_packet, kernarg_ptr);
}

#if defined(IREE_AMDGPU_TARGET_DEVICE)

IREE_AMDGPU_ATTRIBUTE_KERNEL void
iree_hal_amdgpu_device_dispatch_patch_indirect_params(
    const uint32_t* IREE_AMDGPU_RESTRICT workgroup_count,
    iree_hsa_kernel_dispatch_packet_t* IREE_AMDGPU_RESTRICT dispatch_packet,
    iree_amdgpu_kernel_implicit_args_t* IREE_AMDGPU_RESTRICT implicit_args) {
  dispatch_packet->grid_size[0] =
      workgroup_count[0] * dispatch_packet->workgroup_size[0];
  dispatch_packet->grid_size[1] =
      workgroup_count[1] * dispatch_packet->workgroup_size[1];
  dispatch_packet->grid_size[2] =
      workgroup_count[2] * dispatch_packet->workgroup_size[2];

  if (implicit_args) {
    implicit_args->block_count[0] = workgroup_count[0];
    implicit_args->block_count[1] = workgroup_count[1];
    implicit_args->block_count[2] = workgroup_count[2];
  }

  const uint16_t header =
      (dispatch_packet->header &
       ~(((1u << IREE_HSA_PACKET_HEADER_WIDTH_TYPE) - 1u)
         << IREE_HSA_PACKET_HEADER_TYPE)) |
      (IREE_HSA_PACKET_TYPE_KERNEL_DISPATCH << IREE_HSA_PACKET_HEADER_TYPE);
  const uint32_t header_setup =
      (uint32_t)header | ((uint32_t)dispatch_packet->setup << 16);
  iree_amdgpu_scoped_atomic_store(
      (iree_amdgpu_scoped_atomic_uint32_t*)dispatch_packet, header_setup,
      iree_amdgpu_memory_order_release, iree_amdgpu_memory_scope_system);
}

#endif  // IREE_AMDGPU_TARGET_DEVICE
