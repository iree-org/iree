// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_DEVICE_BLIT_H_
#define IREE_HAL_DRIVERS_AMDGPU_DEVICE_BLIT_H_

#include "iree/hal/drivers/amdgpu/abi/queue.h"
#include "iree/hal/drivers/amdgpu/device/kernels.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Blit Kernels
//===----------------------------------------------------------------------===//

// Builtin transfer kernel table used when populating blit dispatch packets.
// Queue reservation, packet header commit, completion-signal assignment, and
// doorbell writes are handled by the caller's queue implementation.
typedef struct iree_hal_amdgpu_device_buffer_transfer_context_t {
  // Handles to opaque kernel objects used to dispatch builtin kernels.
  const iree_hal_amdgpu_device_kernels_t* kernels;
} iree_hal_amdgpu_device_buffer_transfer_context_t;

// Kernel arguments for the `iree_hal_amdgpu_device_buffer_fill_*` family.
typedef struct iree_hal_amdgpu_device_buffer_fill_kernargs_t {
  void* target_ptr;
  uint64_t element_length;
  uint64_t pattern;
} iree_hal_amdgpu_device_buffer_fill_kernargs_t;
#define IREE_HAL_AMDGPU_DEVICE_BUFFER_FILL_KERNARG_SIZE \
  sizeof(iree_hal_amdgpu_device_buffer_fill_kernargs_t)
#define IREE_HAL_AMDGPU_DEVICE_BUFFER_FILL_KERNARG_ALIGNMENT \
  IREE_AMDGPU_ALIGNOF(iree_hal_amdgpu_device_buffer_fill_kernargs_t)

// Kernel arguments for the `iree_hal_amdgpu_device_buffer_copy_*` family.
typedef struct iree_hal_amdgpu_device_buffer_copy_kernargs_t {
  const void* source_ptr;
  void* target_ptr;
  uint64_t element_length;
} iree_hal_amdgpu_device_buffer_copy_kernargs_t;
#define IREE_HAL_AMDGPU_DEVICE_BUFFER_COPY_KERNARG_SIZE \
  sizeof(iree_hal_amdgpu_device_buffer_copy_kernargs_t)
#define IREE_HAL_AMDGPU_DEVICE_BUFFER_COPY_KERNARG_ALIGNMENT \
  IREE_AMDGPU_ALIGNOF(iree_hal_amdgpu_device_buffer_copy_kernargs_t)

// Populates a builtin fill dispatch packet and its kernargs in already-reserved
// storage. The caller owns packet header commit, completion signal assignment,
// and queue doorbell signaling.
//
// Returns false if |pattern_length| is unsupported, the target pointer/length
// alignment is incompatible with that pattern width, or |length| cannot be
// represented by the dispatch packet grid dimensions. On failure,
// |dispatch_packet| and |kernarg_ptr| are left unmodified.
bool iree_hal_amdgpu_device_buffer_fill_emplace(
    const iree_hal_amdgpu_device_buffer_transfer_context_t* IREE_AMDGPU_RESTRICT
        context,
    iree_hsa_kernel_dispatch_packet_t* IREE_AMDGPU_RESTRICT dispatch_packet,
    void* target_ptr, uint64_t length, uint64_t pattern, uint8_t pattern_length,
    void* IREE_AMDGPU_RESTRICT kernarg_ptr);

// Populates a builtin copy dispatch packet and its kernargs in already-reserved
// storage. The caller owns packet header commit, completion signal assignment,
// and queue doorbell signaling.
//
// Returns false if |length| cannot be represented by the dispatch packet grid
// dimensions. On failure, |dispatch_packet| and |kernarg_ptr| are left
// unmodified.
bool iree_hal_amdgpu_device_buffer_copy_emplace(
    const iree_hal_amdgpu_device_buffer_transfer_context_t* IREE_AMDGPU_RESTRICT
        context,
    iree_hsa_kernel_dispatch_packet_t* IREE_AMDGPU_RESTRICT dispatch_packet,
    const void* source_ptr, void* target_ptr, uint64_t length,
    void* IREE_AMDGPU_RESTRICT kernarg_ptr);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_DEVICE_BLIT_H_
