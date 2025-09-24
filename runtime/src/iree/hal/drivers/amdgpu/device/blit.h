// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_DEVICE_BLIT_H_
#define IREE_HAL_DRIVERS_AMDGPU_DEVICE_BLIT_H_

#include "iree/hal/drivers/amdgpu/device/kernels.h"
#include "iree/hal/drivers/amdgpu/device/support/common.h"
#include "iree/hal/drivers/amdgpu/device/support/queue.h"
#include "iree/hal/drivers/amdgpu/device/tracing.h"

//===----------------------------------------------------------------------===//
// Blit Kernels
//===----------------------------------------------------------------------===//

// Context used when scheduling transfer commands.
typedef struct iree_hal_amdgpu_device_buffer_transfer_context_t {
  // Target queue that will execute the transfer operation.
  iree_amd_cached_queue_t queue;
  // Handles to opaque kernel objects used to dispatch builtin kernels.
  const iree_hal_amdgpu_device_kernels_t* kernels;
  // Optional trace buffer used when tracing infrastructure is available.
  iree_hal_amdgpu_device_trace_buffer_t* trace_buffer;
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

#if defined(IREE_AMDGPU_TARGET_DEVICE)

// Emplaces a fill dispatch packet in the target queue at the given index.
// The queue doorbell will not be signaled.
//
// NOTE: this only works with blits today. SDMA will require a different
// signature.
iree_hsa_kernel_dispatch_packet_t* IREE_AMDGPU_RESTRICT
iree_hal_amdgpu_device_buffer_fill_emplace_reserve(
    const iree_hal_amdgpu_device_buffer_transfer_context_t* IREE_AMDGPU_RESTRICT
        context,
    void* target_ptr, const uint64_t length, const uint64_t pattern,
    const uint8_t pattern_length, uint64_t* IREE_AMDGPU_RESTRICT kernarg_ptr,
    const uint64_t packet_id);

// Enqueues a fill dispatch packet in the target queue.
// The packet will be acquired at the current write_index and the queue doorbell
// will be signaled.
void iree_hal_amdgpu_device_buffer_fill_enqueue(
    const iree_hal_amdgpu_device_buffer_transfer_context_t* IREE_AMDGPU_RESTRICT
        context,
    void* target_ptr, const uint64_t length, const uint64_t pattern,
    const uint8_t pattern_length, uint64_t* IREE_AMDGPU_RESTRICT kernarg_ptr);

// Emplaces a copy dispatch packet in the target queue at the given index.
// The queue doorbell will not be signaled.
//
// NOTE: this only works with blits today. SDMA will require a different
// signature.
iree_hsa_kernel_dispatch_packet_t* IREE_AMDGPU_RESTRICT
iree_hal_amdgpu_device_buffer_copy_emplace_reserve(
    const iree_hal_amdgpu_device_buffer_transfer_context_t* IREE_AMDGPU_RESTRICT
        context,
    const void* source_ptr, void* target_ptr, const uint64_t length,
    uint64_t* IREE_AMDGPU_RESTRICT kernarg_ptr, const uint64_t packet_id);

// Enqueues a copy dispatch packet in the target queue.
// The packet will be acquired at the current write_index and the queue doorbell
// will be signaled.
void iree_hal_amdgpu_device_buffer_copy_enqueue(
    const iree_hal_amdgpu_device_buffer_transfer_context_t* IREE_AMDGPU_RESTRICT
        context,
    const void* source_ptr, void* target_ptr, const uint64_t length,
    uint64_t* IREE_AMDGPU_RESTRICT kernarg_ptr);

#endif  // IREE_AMDGPU_TARGET_DEVICE

#endif  // IREE_HAL_DRIVERS_AMDGPU_DEVICE_BLIT_H_
