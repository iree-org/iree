// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/device/blit.h"

#include "iree/hal/drivers/amdgpu/device/support/common.h"
#include "iree/hal/drivers/amdgpu/device/support/kernel.h"

//===----------------------------------------------------------------------===//
// Blit kernel utilities
//===----------------------------------------------------------------------===//

// 2 uint64_t values totaling 16 bytes.
typedef uint64_t iree_amdgpu_uint64x2_t __attribute__((vector_size(16)));

#define IREE_HAL_AMDGPU_FILL_BLOCK_ELEMENT_SIZE sizeof(iree_amdgpu_uint64x2_t)
#define IREE_HAL_AMDGPU_FILL_BLOCK_COUNT 8
#define IREE_HAL_AMDGPU_FILL_BLOCK_SIZE \
  (IREE_HAL_AMDGPU_FILL_BLOCK_ELEMENT_SIZE * IREE_HAL_AMDGPU_FILL_BLOCK_COUNT)

#define IREE_HAL_AMDGPU_COPY_BLOCK_ELEMENT_SIZE sizeof(iree_amdgpu_uint64x2_t)
#define IREE_HAL_AMDGPU_COPY_BLOCK_COUNT 8
#define IREE_HAL_AMDGPU_COPY_BLOCK_SIZE \
  (IREE_HAL_AMDGPU_COPY_BLOCK_ELEMENT_SIZE * IREE_HAL_AMDGPU_COPY_BLOCK_COUNT)

#if defined(IREE_AMDGPU_TARGET_DEVICE)

static inline size_t iree_hal_amdgpu_blit_linear_id(void) {
  const size_t id_x = iree_hal_amdgpu_device_group_id_x() *
                          IREE_HAL_AMDGPU_BLIT_WORKGROUP_SIZE_X +
                      iree_hal_amdgpu_device_local_id_x();
  const size_t id_y = iree_hal_amdgpu_device_group_id_y() *
                          IREE_HAL_AMDGPU_BLIT_WORKGROUP_SIZE_Y +
                      iree_hal_amdgpu_device_local_id_y();
  return id_y * iree_amdgcn_dispatch_ptr()->grid_size[0] + id_x;
}

#endif  // IREE_AMDGPU_TARGET_DEVICE

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_device_buffer_fill_*
//===----------------------------------------------------------------------===//

#if defined(IREE_AMDGPU_TARGET_DEVICE)

IREE_AMDGPU_ATTRIBUTE_KERNEL void iree_hal_amdgpu_device_buffer_fill_x1(
    uint8_t* IREE_AMDGPU_RESTRICT target_ptr, const uint64_t element_length,
    const uint8_t pattern) {
  const size_t element_offset = iree_hal_amdgpu_blit_linear_id();
  if (IREE_AMDGPU_LIKELY(element_offset < element_length)) {
    // Each invocation fills one scalar element.
    target_ptr[element_offset] = pattern;
  }
}

IREE_AMDGPU_ATTRIBUTE_KERNEL void iree_hal_amdgpu_device_buffer_fill_x2(
    uint16_t* IREE_AMDGPU_RESTRICT target_ptr, const uint64_t element_length,
    const uint16_t pattern) {
  const size_t element_offset = iree_hal_amdgpu_blit_linear_id();
  if (IREE_AMDGPU_LIKELY(element_offset < element_length)) {
    // Each invocation fills one scalar element.
    target_ptr[element_offset] = pattern;
  }
}

IREE_AMDGPU_ATTRIBUTE_KERNEL void iree_hal_amdgpu_device_buffer_fill_x4(
    uint32_t* IREE_AMDGPU_RESTRICT target_ptr, const uint64_t element_length,
    const uint32_t pattern) {
  const size_t element_offset = iree_hal_amdgpu_blit_linear_id();
  if (IREE_AMDGPU_LIKELY(element_offset < element_length)) {
    // Each invocation fills one scalar element.
    target_ptr[element_offset] = pattern;
  }
}

IREE_AMDGPU_ATTRIBUTE_KERNEL void iree_hal_amdgpu_device_buffer_fill_x8(
    uint64_t* IREE_AMDGPU_RESTRICT target_ptr, const uint64_t element_length,
    const uint64_t pattern) {
  const size_t element_offset = iree_hal_amdgpu_blit_linear_id();
  if (IREE_AMDGPU_LIKELY(element_offset < element_length)) {
    // Each invocation fills one scalar element.
    target_ptr[element_offset] = pattern;
  }
}

// Fills a block of up to IREE_HAL_AMDGPU_FILL_BLOCK_COUNT 16-byte elements with
// a fixed pattern. Requires an alignment of 16-bytes on both the |target_ptr|
// and |length|.
IREE_AMDGPU_ATTRIBUTE_KERNEL void iree_hal_amdgpu_device_buffer_fill_block_x16(
    iree_amdgpu_uint64x2_t* IREE_AMDGPU_RESTRICT target_ptr,
    const uint64_t element_length, const uint64_t pattern) {
  const size_t block_id = iree_hal_amdgpu_blit_linear_id();
  const size_t element_offset = block_id * IREE_HAL_AMDGPU_FILL_BLOCK_COUNT;
  if (IREE_AMDGPU_UNLIKELY(element_offset >= element_length)) return;
  iree_amdgpu_uint64x2_t pattern_x16 = {pattern, pattern};
  const size_t element_count = IREE_AMDGPU_MIN(IREE_HAL_AMDGPU_FILL_BLOCK_COUNT,
                                               element_length - element_offset);
  if (IREE_AMDGPU_LIKELY(element_count == IREE_HAL_AMDGPU_FILL_BLOCK_COUNT)) {
#pragma unroll
    for (size_t i = 0; i < IREE_HAL_AMDGPU_FILL_BLOCK_COUNT; ++i) {
      target_ptr[element_offset + i] = pattern_x16;
    }
  } else {
    for (size_t i = 0; i < element_count; ++i) {
      target_ptr[element_offset + i] = pattern_x16;
    }
  }
}

#endif  // IREE_AMDGPU_TARGET_DEVICE

// Returns the bytes of |pattern| of length |pattern_length| splatted to
// an 8-byte value.
static uint64_t iree_hal_amdgpu_device_extend_pattern_x8(
    const uint64_t pattern, const uint8_t pattern_length) {
  switch (pattern_length) {
    case 1: {
      const uint64_t pattern_x1 = pattern & 0xFFu;
      return (pattern_x1 << 56) | (pattern_x1 << 48) | (pattern_x1 << 40) |
             (pattern_x1 << 32) | (pattern_x1 << 24) | (pattern_x1 << 16) |
             (pattern_x1 << 8) | pattern_x1;
    }
    case 2: {
      const uint64_t pattern_x2 = pattern & 0xFFFFu;
      return (pattern_x2 << 48) | (pattern_x2 << 32) | (pattern_x2 << 16) |
             pattern_x2;
    }
    case 4: {
      const uint64_t pattern_x4 = pattern & 0xFFFFFFFFu;
      return (pattern_x4 << 32) | pattern_x4;
    }
    case 8:
      return pattern;
    default:
      return 0;
  }
}

// Computes a 2D dispatch grid for |block_count| logical blocks without
// exceeding the 32-bit packet grid dimensions. Zero-block launches use a 1x1
// no-op dispatch, one-row launches use [block_count, 1], and larger launches
// choose the smallest X that keeps Y in-range, minimizing overshoot from the
// final partially-filled row.
static bool iree_hal_amdgpu_device_buffer_transfer_calculate_grid_size(
    const uint64_t block_count, uint32_t* out_grid_size_x,
    uint32_t* out_grid_size_y) {
  if (IREE_AMDGPU_UNLIKELY(block_count == 0)) {
    *out_grid_size_x = 1;
    *out_grid_size_y = 1;
    return true;
  }
  if (IREE_AMDGPU_LIKELY(block_count <= UINT32_MAX)) {
    *out_grid_size_x = (uint32_t)block_count;
    *out_grid_size_y = 1;
    return true;
  }
  const uint64_t grid_size_x = 1 + ((block_count - 1) / UINT32_MAX);
  if (IREE_AMDGPU_UNLIKELY(grid_size_x > UINT32_MAX)) {
    return false;
  }
  const uint64_t grid_size_y = 1 + ((block_count - 1) / grid_size_x);
  *out_grid_size_x = (uint32_t)grid_size_x;
  *out_grid_size_y = (uint32_t)grid_size_y;
  return true;
}

static void iree_hal_amdgpu_device_buffer_transfer_emplace_dispatch(
    const iree_hal_amdgpu_device_kernel_args_t* IREE_AMDGPU_RESTRICT
        kernel_args,
    iree_hsa_kernel_dispatch_packet_t* IREE_AMDGPU_RESTRICT dispatch_packet,
    const uint32_t grid_size_x, const uint32_t grid_size_y, void* kernarg_ptr) {
  dispatch_packet->setup = kernel_args->setup;
  dispatch_packet->workgroup_size[0] = kernel_args->workgroup_size[0];
  dispatch_packet->workgroup_size[1] = kernel_args->workgroup_size[1];
  dispatch_packet->workgroup_size[2] = kernel_args->workgroup_size[2];
  dispatch_packet->reserved0 = 0;
  dispatch_packet->grid_size[0] = grid_size_x;
  dispatch_packet->grid_size[1] = grid_size_y;
  dispatch_packet->grid_size[2] = 1;
  dispatch_packet->private_segment_size = kernel_args->private_segment_size;
  dispatch_packet->group_segment_size = kernel_args->group_segment_size;
  dispatch_packet->kernel_object = kernel_args->kernel_object;
  dispatch_packet->kernarg_address = kernarg_ptr;
  dispatch_packet->reserved2 = 0;
  dispatch_packet->completion_signal = iree_hsa_signal_null();
}

bool iree_hal_amdgpu_device_buffer_fill_emplace(
    const iree_hal_amdgpu_device_buffer_transfer_context_t* IREE_AMDGPU_RESTRICT
        context,
    iree_hsa_kernel_dispatch_packet_t* IREE_AMDGPU_RESTRICT dispatch_packet,
    void* target_ptr, uint64_t length, uint64_t pattern, uint8_t pattern_length,
    void* IREE_AMDGPU_RESTRICT kernarg_ptr) {
  // Select the kernel for the fill operation.
  const iree_hal_amdgpu_device_kernel_args_t* IREE_AMDGPU_RESTRICT kernel_args =
      NULL;
  size_t element_size = 1;
  size_t block_size = 1;
  if (IREE_AMDGPU_UNLIKELY(pattern_length != 1 && pattern_length != 2 &&
                           pattern_length != 4 && pattern_length != 8)) {
    return false;
  }
  if (IREE_AMDGPU_UNLIKELY(
          !iree_amdgpu_has_alignment((size_t)target_ptr, pattern_length) ||
          !iree_amdgpu_has_alignment(length, pattern_length))) {
    return false;
  }
  if (iree_amdgpu_has_alignment((size_t)target_ptr,
                                IREE_HAL_AMDGPU_FILL_BLOCK_ELEMENT_SIZE) &&
      iree_amdgpu_has_alignment(length,
                                IREE_HAL_AMDGPU_FILL_BLOCK_ELEMENT_SIZE)) {
    pattern = iree_hal_amdgpu_device_extend_pattern_x8(pattern, pattern_length);
    kernel_args =
        &context->kernels->iree_hal_amdgpu_device_buffer_fill_block_x16;
    element_size = IREE_HAL_AMDGPU_FILL_BLOCK_ELEMENT_SIZE;
    block_size = IREE_HAL_AMDGPU_FILL_BLOCK_COUNT;
  } else {
    switch (pattern_length) {
      case 1:
        kernel_args = &context->kernels->iree_hal_amdgpu_device_buffer_fill_x1;
        break;
      case 2:
        kernel_args = &context->kernels->iree_hal_amdgpu_device_buffer_fill_x2;
        element_size = 2;
        break;
      case 4:
        kernel_args = &context->kernels->iree_hal_amdgpu_device_buffer_fill_x4;
        element_size = 4;
        break;
      case 8:
        kernel_args = &context->kernels->iree_hal_amdgpu_device_buffer_fill_x8;
        element_size = 8;
        break;
      default:
        return false;
    }
    block_size = 1;
  }

  const size_t element_count = length / element_size;
  const size_t block_count = IREE_AMDGPU_CEIL_DIV(element_count, block_size);
  uint32_t grid_size_x = 0;
  uint32_t grid_size_y = 0;
  if (IREE_AMDGPU_UNLIKELY(
          !iree_hal_amdgpu_device_buffer_transfer_calculate_grid_size(
              block_count, &grid_size_x, &grid_size_y))) {
    return false;
  }

  // Update kernargs (same API for all kernels).
  iree_hal_amdgpu_device_buffer_fill_kernargs_t* kernargs =
      (iree_hal_amdgpu_device_buffer_fill_kernargs_t*)kernarg_ptr;
  kernargs->target_ptr = target_ptr;
  kernargs->element_length = element_count;
  kernargs->pattern = pattern;

  iree_hal_amdgpu_device_buffer_transfer_emplace_dispatch(
      kernel_args, dispatch_packet, grid_size_x, grid_size_y, kernarg_ptr);
  return true;
}

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_device_buffer_copy_*
//===----------------------------------------------------------------------===//

#if defined(IREE_AMDGPU_TARGET_DEVICE)

IREE_AMDGPU_ATTRIBUTE_KERNEL void iree_hal_amdgpu_device_buffer_copy_x1(
    const uint8_t* IREE_AMDGPU_RESTRICT source_ptr,
    uint8_t* IREE_AMDGPU_RESTRICT target_ptr, const uint64_t element_length) {
  const size_t element_offset = iree_hal_amdgpu_blit_linear_id();
  if (IREE_AMDGPU_LIKELY(element_offset < element_length)) {
    // Each invocation copies one scalar element.
    target_ptr[element_offset] = source_ptr[element_offset];
  }
}

// Copies a block of up to IREE_HAL_AMDGPU_COPY_BLOCK_COUNT 16-byte elements
// from |source_ptr| to |target_ptr|. Requires an alignment of 16-bytes on all
// of |source_ptr|, |target_ptr|, and |length|.
//
// Dispatched on a 2D grid with up to UINT32_MAX blocks on X.
IREE_AMDGPU_ATTRIBUTE_KERNEL void iree_hal_amdgpu_device_buffer_copy_block_x16(
    const iree_amdgpu_uint64x2_t* IREE_AMDGPU_RESTRICT source_ptr,
    iree_amdgpu_uint64x2_t* IREE_AMDGPU_RESTRICT target_ptr,
    const uint64_t element_length) {
  const size_t block_id = iree_hal_amdgpu_blit_linear_id();
  const size_t element_offset = block_id * IREE_HAL_AMDGPU_COPY_BLOCK_COUNT;
  if (IREE_AMDGPU_UNLIKELY(element_offset >= element_length)) return;
  const size_t element_count = IREE_AMDGPU_MIN(IREE_HAL_AMDGPU_COPY_BLOCK_COUNT,
                                               element_length - element_offset);
  if (IREE_AMDGPU_LIKELY(element_count == IREE_HAL_AMDGPU_COPY_BLOCK_COUNT)) {
#pragma unroll
    for (size_t i = 0; i < IREE_HAL_AMDGPU_COPY_BLOCK_COUNT; ++i) {
      target_ptr[element_offset + i] = source_ptr[element_offset + i];
    }
  } else {
    for (size_t i = 0; i < element_count; ++i) {
      target_ptr[element_offset + i] = source_ptr[element_offset + i];
    }
  }
}

#endif  // IREE_AMDGPU_TARGET_DEVICE

// Copies currently dispatch builtin blit kernels. SDMA emission belongs in a
// queue-specific wrapper because it changes queue ownership and packet
// reservation policy.
bool iree_hal_amdgpu_device_buffer_copy_emplace(
    const iree_hal_amdgpu_device_buffer_transfer_context_t* IREE_AMDGPU_RESTRICT
        context,
    iree_hsa_kernel_dispatch_packet_t* IREE_AMDGPU_RESTRICT dispatch_packet,
    const void* source_ptr, void* target_ptr, uint64_t length,
    void* IREE_AMDGPU_RESTRICT kernarg_ptr) {
  // Select the kernel for the copy operation.
  const iree_hal_amdgpu_device_kernel_args_t* IREE_AMDGPU_RESTRICT kernel_args =
      NULL;
  size_t element_size = 1;
  size_t block_size = 1;
  if (iree_amdgpu_has_alignment((size_t)source_ptr,
                                IREE_HAL_AMDGPU_COPY_BLOCK_ELEMENT_SIZE) &&
      iree_amdgpu_has_alignment((size_t)target_ptr,
                                IREE_HAL_AMDGPU_COPY_BLOCK_ELEMENT_SIZE) &&
      iree_amdgpu_has_alignment(length,
                                IREE_HAL_AMDGPU_COPY_BLOCK_ELEMENT_SIZE)) {
    kernel_args =
        &context->kernels->iree_hal_amdgpu_device_buffer_copy_block_x16;
    element_size = IREE_HAL_AMDGPU_COPY_BLOCK_ELEMENT_SIZE;
    block_size = IREE_HAL_AMDGPU_COPY_BLOCK_COUNT;
  } else {
    kernel_args = &context->kernels->iree_hal_amdgpu_device_buffer_copy_x1;
    element_size = 1;
    block_size = 1;
  }

  const size_t element_count = length / element_size;
  const size_t block_count = IREE_AMDGPU_CEIL_DIV(element_count, block_size);
  uint32_t grid_size_x = 0;
  uint32_t grid_size_y = 0;
  if (IREE_AMDGPU_UNLIKELY(
          !iree_hal_amdgpu_device_buffer_transfer_calculate_grid_size(
              block_count, &grid_size_x, &grid_size_y))) {
    return false;
  }

  // Update kernargs (same API for all kernels).
  iree_hal_amdgpu_device_buffer_copy_kernargs_t* kernargs =
      (iree_hal_amdgpu_device_buffer_copy_kernargs_t*)kernarg_ptr;
  kernargs->source_ptr = source_ptr;
  kernargs->target_ptr = target_ptr;
  kernargs->element_length = element_count;

  iree_hal_amdgpu_device_buffer_transfer_emplace_dispatch(
      kernel_args, dispatch_packet, grid_size_x, grid_size_y, kernarg_ptr);
  return true;
}
