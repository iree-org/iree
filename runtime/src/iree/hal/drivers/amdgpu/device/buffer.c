// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/device/buffer.h"

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_device_buffer_ref_t
//===----------------------------------------------------------------------===//

// TODO(benvanik): simplify this for command buffers by pre-baking as much as we
// can during the queue issue - we can at least dereference handles and add in
// the offset for everything such that we only have to deal with the slot offset
// and have less branchy code.
void* iree_hal_amdgpu_device_buffer_ref_resolve(
    iree_hal_amdgpu_device_buffer_ref_t buffer_ref,
    IREE_AMDGPU_ALIGNAS(64)
        const iree_hal_amdgpu_device_buffer_ref_t* IREE_AMDGPU_RESTRICT
            binding_table) {
  if (buffer_ref.type == IREE_HAL_AMDGPU_DEVICE_BUFFER_TYPE_SLOT) {
    const iree_hal_amdgpu_device_buffer_ref_t binding =
        binding_table[buffer_ref.value.slot];
    const uint64_t offset = buffer_ref.offset + binding.offset;
    const uint64_t length = binding.length == UINT64_MAX
                                ? buffer_ref.length - offset
                                : buffer_ref.length;
    buffer_ref = (iree_hal_amdgpu_device_buffer_ref_t){
        .type = binding.type,
        .offset = offset,
        .length = length,
        .value.bits = binding.value.bits,
    };
  }
  if (buffer_ref.type == IREE_HAL_AMDGPU_DEVICE_BUFFER_TYPE_HANDLE) {
    buffer_ref.value.ptr = buffer_ref.value.handle->ptr;
  }
  return buffer_ref.value.ptr
             ? (uint8_t*)buffer_ref.value.ptr + buffer_ref.offset
             : NULL;
}

void* iree_hal_amdgpu_device_workgroup_count_buffer_ref_resolve(
    iree_hal_amdgpu_device_workgroup_count_buffer_ref_t buffer_ref,
    IREE_AMDGPU_ALIGNAS(64)
        const iree_hal_amdgpu_device_buffer_ref_t* IREE_AMDGPU_RESTRICT
            binding_table) {
  if (buffer_ref.type == IREE_HAL_AMDGPU_DEVICE_BUFFER_TYPE_SLOT) {
    const iree_hal_amdgpu_device_buffer_ref_t binding =
        binding_table[buffer_ref.value.slot];
    const uint64_t offset = buffer_ref.offset + binding.offset;
    buffer_ref = (iree_hal_amdgpu_device_workgroup_count_buffer_ref_t){
        .type = binding.type,
        .offset = offset,
        .value.bits = binding.value.bits,
    };
  }
  if (buffer_ref.type == IREE_HAL_AMDGPU_DEVICE_BUFFER_TYPE_HANDLE) {
    buffer_ref.value.ptr = buffer_ref.value.handle->ptr;
  }
  return buffer_ref.value.ptr
             ? (uint8_t*)buffer_ref.value.ptr + buffer_ref.offset
             : NULL;
}

void* iree_hal_amdgpu_device_uint64_buffer_ref_resolve(
    iree_hal_amdgpu_device_uint64_buffer_ref_t buffer_ref,
    IREE_AMDGPU_ALIGNAS(64)
        const iree_hal_amdgpu_device_buffer_ref_t* IREE_AMDGPU_RESTRICT
            binding_table) {
  if (buffer_ref.type == IREE_HAL_AMDGPU_DEVICE_BUFFER_TYPE_SLOT) {
    const iree_hal_amdgpu_device_buffer_ref_t binding =
        binding_table[buffer_ref.value.slot];
    const uint64_t offset = buffer_ref.offset + binding.offset;
    buffer_ref = (iree_hal_amdgpu_device_uint64_buffer_ref_t){
        .type = binding.type,
        .offset = offset,
        .value.bits = binding.value.bits,
    };
  }
  if (buffer_ref.type == IREE_HAL_AMDGPU_DEVICE_BUFFER_TYPE_HANDLE) {
    buffer_ref.value.ptr = buffer_ref.value.handle->ptr;
  }
  return buffer_ref.value.ptr
             ? (uint8_t*)buffer_ref.value.ptr + buffer_ref.offset
             : NULL;
}
