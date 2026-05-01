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
// Unaligned view of a 16-byte vector. The __packed__ attribute on the
// enclosing struct propagates to the |value| member, so a dereference of a
// iree_amdgpu_unaligned_uint64x2_t* generates unaligned loads/stores instead
// of the 16-byte-aligned form the compiler would otherwise assume. Used by the
// unaligned block kernels to vectorize copies/fills when pointers and/or
// length are not 16-byte aligned.
typedef struct IREE_AMDGPU_ATTRIBUTE_PACKED {
  iree_amdgpu_uint64x2_t value;
} iree_amdgpu_unaligned_uint64x2_t;

// 128 bytes is enough to amortize launch overhead over at least one full
// vector store per lane at wave32; anything smaller stays on the scalar byte
// path. The threshold is deliberately independent of the per-block unroll
// count (IREE_HAL_AMDGPU_*_BLOCK_COUNT) so that benchmark-driven tuning of the
// unroll factor does not change the selection boundary between scalar and
// vector paths.
#define IREE_HAL_AMDGPU_BLIT_UNALIGNED_MIN_BYTES 128

#define IREE_HAL_AMDGPU_FILL_BLOCK_ELEMENT_SIZE sizeof(iree_amdgpu_uint64x2_t)
#define IREE_HAL_AMDGPU_FILL_BLOCK_COUNT 4
#define IREE_HAL_AMDGPU_FILL_BLOCK_UNALIGNED_MIN_SIZE \
  IREE_HAL_AMDGPU_BLIT_UNALIGNED_MIN_BYTES

#define IREE_HAL_AMDGPU_FILL_BLOCK_X4_ELEMENT_SIZE sizeof(uint32_t)
#define IREE_HAL_AMDGPU_FILL_BLOCK_X4_COUNT 16

#define IREE_HAL_AMDGPU_COPY_BLOCK_ELEMENT_SIZE sizeof(iree_amdgpu_uint64x2_t)
#define IREE_HAL_AMDGPU_COPY_BLOCK_COUNT 1
#define IREE_HAL_AMDGPU_COPY_BLOCK_UNALIGNED_MIN_SIZE \
  IREE_HAL_AMDGPU_BLIT_UNALIGNED_MIN_BYTES

#define IREE_HAL_AMDGPU_COPY_BLOCK_X8_ELEMENT_SIZE sizeof(uint64_t)
#define IREE_HAL_AMDGPU_COPY_BLOCK_X8_COUNT 8

#define IREE_HAL_AMDGPU_COPY_BLOCK_X4_ELEMENT_SIZE sizeof(uint32_t)
#define IREE_HAL_AMDGPU_COPY_BLOCK_X4_COUNT 16

#define IREE_HAL_AMDGPU_BLIT_WORKGROUPS_PER_COMPUTE_UNIT 4

void iree_hal_amdgpu_device_buffer_transfer_context_initialize(
    const iree_hal_amdgpu_device_kernels_t* kernels,
    uint32_t compute_unit_count, uint32_t wavefront_size,
    iree_hal_amdgpu_device_buffer_transfer_context_t* out_context) {
  // Preconditions (validated by the caller; see physical_device.c):
  //   compute_unit_count > 0
  //   wavefront_size in {32, 64}
  const uint64_t max_workgroup_count =
      (uint64_t)compute_unit_count *
      IREE_HAL_AMDGPU_BLIT_WORKGROUPS_PER_COMPUTE_UNIT;
  *out_context = (iree_hal_amdgpu_device_buffer_transfer_context_t){
      .kernels = kernels,
      .wavefront_size = (uint16_t)wavefront_size,
      .workgroup_size_x = (uint16_t)wavefront_size,
      .max_workgroup_count = max_workgroup_count > UINT32_MAX
                                 ? UINT32_MAX
                                 : (uint32_t)max_workgroup_count,
  };
}

#if defined(IREE_AMDGPU_TARGET_DEVICE)

static inline IREE_AMDGPU_ATTRIBUTE_ALWAYS_INLINE uint64_t
iree_hal_amdgpu_blit_linear_id(void) {
  const uint64_t id_x = iree_hal_amdgpu_device_global_id_x();
  const uint64_t id_y = iree_hal_amdgpu_device_global_id_y();
  return id_y * iree_amdgcn_dispatch_ptr()->grid_size[0] + id_x;
}

static inline IREE_AMDGPU_ATTRIBUTE_ALWAYS_INLINE uint64_t
iree_hal_amdgpu_blit_grid_size(void) {
  const iree_hsa_kernel_dispatch_packet_t* IREE_AMDGPU_RESTRICT dispatch_ptr =
      iree_amdgcn_dispatch_ptr();
  return (uint64_t)dispatch_ptr->grid_size[0] * dispatch_ptr->grid_size[1];
}

static inline IREE_AMDGPU_ATTRIBUTE_ALWAYS_INLINE bool
iree_hal_amdgpu_blit_advance(uint64_t* element_offset,
                             const uint64_t element_stride) {
  if (IREE_AMDGPU_UNLIKELY(*element_offset > UINT64_MAX - element_stride)) {
    return false;
  }
  *element_offset += element_stride;
  return true;
}

// Returns the byte at |byte_offset| of a repeating fill pattern.
// Precondition: |pattern| has been extended to a full 8-byte repetition of
// the original 1/2/4/8-byte pattern (see
// iree_hal_amdgpu_device_extend_pattern_x8). The mask |byte_offset & 7u| works
// for any pattern_length that divides 8 — which includes all valid
// pattern_lengths (1, 2, 4, 8). Callers use this only on tail bytes whose
// offset within the buffer is already a multiple of the original
// pattern_length, so |byte_offset| lines up with the pattern phase.
static inline IREE_AMDGPU_ATTRIBUTE_ALWAYS_INLINE uint8_t
iree_hal_amdgpu_blit_pattern_byte(const uint64_t pattern,
                                  const uint64_t byte_offset) {
  return (uint8_t)(pattern >> ((byte_offset & 7u) * 8u));
}

#endif  // IREE_AMDGPU_TARGET_DEVICE

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_device_buffer_fill_*
//===----------------------------------------------------------------------===//
//
// IREE_HAL_AMDGPU_FILL_BLOCK_COUNT tuning sweep on gfx1100 (RDNA3, wave32,
// 96 CUs, GDDR6). Bandwidth in GB/s; off is buffer target offset alignment,
// pat is fill pattern_length. All rows here use the fill_block_x16 kernel.
// Build: --compilation_mode=opt --copt=-O3 --copt=-march=native
//        --copt=-flto=thin --linkopt=-flto=thin.
//
//                          QueueFill (single)       QueueFillBatch20
//   length   off pat    cnt=1 cnt=2 cnt=4 cnt=8   cnt=1 cnt=2 cnt=4 cnt=8
//   --------------------------------------------------------------------
//    64KiB    0   4      2.8   2.8   2.7   3.0     5.7   6.3   6.5   8.2
//     2MiB    0   2     91.4  99.9 101.4 100.1   207.3 199.3 215.0 212.3
//     2MiB    2   2     91.3 101.6 100.0  98.7   204.5 210.0 210.0 198.9
//     1GiB    0   4    629.7 270.4 652.1 177.3   638.1 273.2 657.9 179.0
//
//   Geomean over bandwidth-relevant rows (64KiB..1GiB across alignments):
//     QueueFill       : cnt=1: 26.8, cnt=2: 24.8, cnt=4: 25.8, cnt=8: 23.8
//     QueueFillBatch20: cnt=4: 55.8, cnt=1: 51.9, cnt=8: 50.9, cnt=2: 47.5
//
// cnt=4 is the tuned value for gfx1100. Unlike copy, fill shows a bathtub:
// cnt=1 and cnt=4 both hit ~640 GB/s at 1GiB (~2/3 of the GDDR6 ceiling)
// while cnt=2 collapses to ~270 GB/s and cnt=8 collapses to ~180 GB/s. The
// cnt=8 cliff is the same VGPR-occupancy story as copy — `#pragma unroll 8`
// over 16-byte writes burns enough extra VGPRs to gate occupancy. The cnt=2
// trough is not fully understood and may be a compiler instruction
// scheduling / VGPR packing artifact on this toolchain version — it is
// reproducible but the mechanism has not been isolated here.
//
// cnt=4 wins the batched benchmark at every bandwidth-relevant size and
// ties cnt=1 at the huge-size ceiling, which is why it is the chosen
// default. The QueueFill (single-op) geomean shows cnt=1 marginally ahead
// of cnt=4 because several single-op 2MiB rows have enough measurement
// variance to push cnt=4 into a temporary 74 GB/s dip that the stable
// batched run does not reproduce; re-running the benchmark with more
// iterations would likely tighten this. Small batched sizes <16KiB prefer
// cnt=8 by ~25%, but absolute throughput there is <1 GB/s and does not
// move real workloads. Sizes <64KiB are omitted here — they are
// measurement-variance-dominated.
//
// CDNA (MI300+, 512 VGPRs per SIMD) may move the cnt=8 cliff further out
// and should be re-swept before changing.

#if defined(IREE_AMDGPU_TARGET_DEVICE)

IREE_AMDGPU_ATTRIBUTE_KERNEL void iree_hal_amdgpu_device_buffer_fill_x1(
    uint8_t* IREE_AMDGPU_RESTRICT target_ptr, const uint64_t element_length,
    const uint8_t pattern) {
  const uint64_t element_stride = iree_hal_amdgpu_blit_grid_size();
  for (uint64_t element_offset = iree_hal_amdgpu_blit_linear_id();
       element_offset < element_length;) {
    target_ptr[element_offset] = pattern;
    if (!iree_hal_amdgpu_blit_advance(&element_offset, element_stride)) break;
  }
}

IREE_AMDGPU_ATTRIBUTE_KERNEL void iree_hal_amdgpu_device_buffer_fill_x2(
    uint16_t* IREE_AMDGPU_RESTRICT target_ptr, const uint64_t element_length,
    const uint16_t pattern) {
  const uint64_t element_stride = iree_hal_amdgpu_blit_grid_size();
  for (uint64_t element_offset = iree_hal_amdgpu_blit_linear_id();
       element_offset < element_length;) {
    target_ptr[element_offset] = pattern;
    if (!iree_hal_amdgpu_blit_advance(&element_offset, element_stride)) break;
  }
}

IREE_AMDGPU_ATTRIBUTE_KERNEL void iree_hal_amdgpu_device_buffer_fill_x4(
    uint32_t* IREE_AMDGPU_RESTRICT target_ptr, const uint64_t element_length,
    const uint32_t pattern) {
  const uint64_t block_stride = iree_hal_amdgpu_blit_grid_size();
  for (uint64_t block_id = iree_hal_amdgpu_blit_linear_id();;) {
    const uint64_t element_offset =
        block_id * IREE_HAL_AMDGPU_FILL_BLOCK_X4_COUNT;
    if (IREE_AMDGPU_UNLIKELY(element_offset >= element_length)) return;
    const uint64_t element_count = IREE_AMDGPU_MIN(
        IREE_HAL_AMDGPU_FILL_BLOCK_X4_COUNT, element_length - element_offset);
    if (IREE_AMDGPU_LIKELY(element_count ==
                           IREE_HAL_AMDGPU_FILL_BLOCK_X4_COUNT)) {
#pragma unroll
      for (size_t i = 0; i < IREE_HAL_AMDGPU_FILL_BLOCK_X4_COUNT; ++i) {
        target_ptr[element_offset + i] = pattern;
      }
    } else {
      for (size_t i = 0; i < element_count; ++i) {
        target_ptr[element_offset + i] = pattern;
      }
    }
    if (!iree_hal_amdgpu_blit_advance(&block_id, block_stride)) return;
  }
}

IREE_AMDGPU_ATTRIBUTE_KERNEL void iree_hal_amdgpu_device_buffer_fill_x8(
    uint64_t* IREE_AMDGPU_RESTRICT target_ptr, const uint64_t element_length,
    const uint64_t pattern) {
  const uint64_t element_stride = iree_hal_amdgpu_blit_grid_size();
  for (uint64_t element_offset = iree_hal_amdgpu_blit_linear_id();
       element_offset < element_length;) {
    target_ptr[element_offset] = pattern;
    if (!iree_hal_amdgpu_blit_advance(&element_offset, element_stride)) break;
  }
}

// Fills blocks of up to IREE_HAL_AMDGPU_FILL_BLOCK_COUNT 16-byte elements with
// a fixed pattern. Requires an alignment of 16-bytes on both |target_ptr| and
// |length|.
IREE_AMDGPU_ATTRIBUTE_KERNEL void iree_hal_amdgpu_device_buffer_fill_block_x16(
    iree_amdgpu_uint64x2_t* IREE_AMDGPU_RESTRICT target_ptr,
    const uint64_t element_length, const uint64_t pattern) {
  const uint64_t block_stride = iree_hal_amdgpu_blit_grid_size();
  const iree_amdgpu_uint64x2_t pattern_x16 = {pattern, pattern};
  for (uint64_t block_id = iree_hal_amdgpu_blit_linear_id();;) {
    const uint64_t element_offset = block_id * IREE_HAL_AMDGPU_FILL_BLOCK_COUNT;
    if (IREE_AMDGPU_UNLIKELY(element_offset >= element_length)) return;
    const uint64_t element_count = IREE_AMDGPU_MIN(
        IREE_HAL_AMDGPU_FILL_BLOCK_COUNT, element_length - element_offset);
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
    if (!iree_hal_amdgpu_blit_advance(&block_id, block_stride)) return;
  }
}

// Fills a byte-granular region using 16-byte vector stores where possible.
// NOTE: |element_length| is a byte length here, not a count of 16-byte
// elements as in fill_block_x16. The kernargs struct field name is shared
// across all fill variants so the host-side emplace path doesn't need a
// second struct; this kernel reinterprets that field as a byte length and
// derives the vector/tail split internally.
IREE_AMDGPU_ATTRIBUTE_KERNEL void
iree_hal_amdgpu_device_buffer_fill_block_unaligned_x16(
    iree_amdgpu_unaligned_uint64x2_t* IREE_AMDGPU_RESTRICT target_ptr,
    const uint64_t element_length, const uint64_t pattern) {
  const uint64_t full_element_count =
      element_length / IREE_HAL_AMDGPU_FILL_BLOCK_ELEMENT_SIZE;
  const uint64_t vector_block_count = IREE_AMDGPU_CEIL_DIV(
      full_element_count, IREE_HAL_AMDGPU_FILL_BLOCK_COUNT);
  const uint64_t tail_offset =
      full_element_count * IREE_HAL_AMDGPU_FILL_BLOCK_ELEMENT_SIZE;
  const uint64_t tail_length = element_length - tail_offset;
  const uint64_t block_stride = iree_hal_amdgpu_blit_grid_size();
  const iree_amdgpu_uint64x2_t pattern_x16 = {pattern, pattern};
  for (uint64_t block_id = iree_hal_amdgpu_blit_linear_id();;) {
    if (IREE_AMDGPU_UNLIKELY(vector_block_count == 0)) {
      if (block_id == 0) {
        uint8_t* tail_ptr = (uint8_t*)target_ptr;
        for (uint64_t i = 0; i < tail_length; ++i) {
          tail_ptr[i] = iree_hal_amdgpu_blit_pattern_byte(pattern, i);
        }
      }
      return;
    }
    if (IREE_AMDGPU_UNLIKELY(block_id >= vector_block_count)) return;
    const uint64_t element_offset = block_id * IREE_HAL_AMDGPU_FILL_BLOCK_COUNT;
    const uint64_t element_count = IREE_AMDGPU_MIN(
        IREE_HAL_AMDGPU_FILL_BLOCK_COUNT, full_element_count - element_offset);
    if (IREE_AMDGPU_LIKELY(element_count == IREE_HAL_AMDGPU_FILL_BLOCK_COUNT)) {
#pragma unroll
      for (size_t i = 0; i < IREE_HAL_AMDGPU_FILL_BLOCK_COUNT; ++i) {
        target_ptr[element_offset + i].value = pattern_x16;
      }
    } else {
      for (size_t i = 0; i < element_count; ++i) {
        target_ptr[element_offset + i].value = pattern_x16;
      }
    }
    if (IREE_AMDGPU_UNLIKELY(tail_length &&
                             block_id + 1 == vector_block_count)) {
      uint8_t* tail_ptr = (uint8_t*)target_ptr + tail_offset;
      for (uint64_t i = 0; i < tail_length; ++i) {
        tail_ptr[i] = iree_hal_amdgpu_blit_pattern_byte(pattern, i);
      }
    }
    if (!iree_hal_amdgpu_blit_advance(&block_id, block_stride)) return;
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

// Returns the number of blocks needed to cover |length| bytes when the vector
// kernels process |elements_per_block| elements of |element_size| bytes each.
// Callers of this helper gate on length >= UNALIGNED_MIN_SIZE (128), so
// full_element_count is guaranteed to be non-zero here.
static uint64_t iree_hal_amdgpu_blit_unaligned_block_count(
    const uint64_t length, const uint64_t element_size,
    const uint64_t elements_per_block) {
  const uint64_t full_element_count = length / element_size;
  return IREE_AMDGPU_CEIL_DIV(full_element_count, elements_per_block);
}

// Computes a bounded 2D dispatch grid for |block_count| logical blocks without
// exceeding the 32-bit packet grid dimensions. Kernels use grid-stride loops,
// so large transfers cap resident work to the context's launch metadata.
// Zero-block launches use a 1x1 no-op dispatch, one-row launches use
// [work_item_count, 1], and larger launches choose the smallest X that keeps Y
// in-range, minimizing overshoot from the final partially-filled row.
static bool iree_hal_amdgpu_blit_calculate_grid_size(
    const iree_hal_amdgpu_device_buffer_transfer_context_t* context,
    const uint64_t block_count, uint32_t* out_grid_size_x,
    uint32_t* out_grid_size_y) {
  const uint64_t max_work_item_count =
      (uint64_t)context->max_workgroup_count * context->workgroup_size_x;
  uint64_t work_item_count = IREE_AMDGPU_MIN(block_count, max_work_item_count);
  if (IREE_AMDGPU_UNLIKELY(work_item_count == 0)) {
    *out_grid_size_x = 1;
    *out_grid_size_y = 1;
    return true;
  }
  if (IREE_AMDGPU_LIKELY(work_item_count <= UINT32_MAX)) {
    *out_grid_size_x = (uint32_t)work_item_count;
    *out_grid_size_y = 1;
    return true;
  }
  const uint64_t grid_size_x = 1 + ((work_item_count - 1) / UINT32_MAX);
  if (IREE_AMDGPU_UNLIKELY(grid_size_x > UINT32_MAX)) {
    return false;
  }
  const uint64_t grid_size_y = 1 + ((work_item_count - 1) / grid_size_x);
  *out_grid_size_x = (uint32_t)grid_size_x;
  *out_grid_size_y = (uint32_t)grid_size_y;
  return true;
}

static void iree_hal_amdgpu_blit_emplace_dispatch(
    const iree_hal_amdgpu_device_buffer_transfer_context_t* IREE_AMDGPU_RESTRICT
        context,
    const iree_hal_amdgpu_device_kernel_args_t* IREE_AMDGPU_RESTRICT
        kernel_args,
    iree_hsa_kernel_dispatch_packet_t* IREE_AMDGPU_RESTRICT dispatch_packet,
    const uint32_t grid_size_x, const uint32_t grid_size_y, void* kernarg_ptr) {
  dispatch_packet->setup = kernel_args->setup;
  dispatch_packet->workgroup_size[0] = context->workgroup_size_x;
  dispatch_packet->workgroup_size[1] = 1;
  dispatch_packet->workgroup_size[2] = 1;
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
  uint64_t element_size = 1;
  uint64_t block_size = 1;
  bool uses_byte_length = false;
  if (IREE_AMDGPU_UNLIKELY(pattern_length != 1 && pattern_length != 2 &&
                           pattern_length != 4 && pattern_length != 8)) {
    return false;
  }
  if (IREE_AMDGPU_UNLIKELY(
          !iree_amdgpu_has_alignment((uintptr_t)target_ptr, pattern_length) ||
          !iree_amdgpu_has_alignment(length, pattern_length))) {
    return false;
  }
  if (iree_amdgpu_has_alignment((uintptr_t)target_ptr,
                                IREE_HAL_AMDGPU_FILL_BLOCK_ELEMENT_SIZE) &&
      iree_amdgpu_has_alignment(length,
                                IREE_HAL_AMDGPU_FILL_BLOCK_ELEMENT_SIZE)) {
    pattern = iree_hal_amdgpu_device_extend_pattern_x8(pattern, pattern_length);
    kernel_args =
        &context->kernels->iree_hal_amdgpu_device_buffer_fill_block_x16;
    element_size = IREE_HAL_AMDGPU_FILL_BLOCK_ELEMENT_SIZE;
    block_size = IREE_HAL_AMDGPU_FILL_BLOCK_COUNT;
  } else if (pattern_length <= 4 &&
             iree_amdgpu_has_alignment(
                 (uintptr_t)target_ptr,
                 IREE_HAL_AMDGPU_FILL_BLOCK_X4_ELEMENT_SIZE) &&
             iree_amdgpu_has_alignment(
                 length, IREE_HAL_AMDGPU_FILL_BLOCK_X4_ELEMENT_SIZE)) {
    pattern = iree_hal_amdgpu_device_extend_pattern_x8(pattern, pattern_length);
    kernel_args = &context->kernels->iree_hal_amdgpu_device_buffer_fill_x4;
    element_size = IREE_HAL_AMDGPU_FILL_BLOCK_X4_ELEMENT_SIZE;
    block_size = IREE_HAL_AMDGPU_FILL_BLOCK_X4_COUNT;
  } else if (length >= IREE_HAL_AMDGPU_FILL_BLOCK_UNALIGNED_MIN_SIZE) {
    pattern = iree_hal_amdgpu_device_extend_pattern_x8(pattern, pattern_length);
    kernel_args = &context->kernels
                       ->iree_hal_amdgpu_device_buffer_fill_block_unaligned_x16;
    element_size = 1;
    block_size = 1;
    uses_byte_length = true;
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

  const uint64_t element_count = length / element_size;
  const uint64_t block_count =
      uses_byte_length ? iree_hal_amdgpu_blit_unaligned_block_count(
                             length, IREE_HAL_AMDGPU_FILL_BLOCK_ELEMENT_SIZE,
                             IREE_HAL_AMDGPU_FILL_BLOCK_COUNT)
                       : IREE_AMDGPU_CEIL_DIV(element_count, block_size);
  uint32_t grid_size_x = 0;
  uint32_t grid_size_y = 0;
  if (IREE_AMDGPU_UNLIKELY(!iree_hal_amdgpu_blit_calculate_grid_size(
          context, block_count, &grid_size_x, &grid_size_y))) {
    return false;
  }

  // Update kernargs (same API for all kernels).
  iree_hal_amdgpu_device_buffer_fill_kernargs_t* kernargs =
      (iree_hal_amdgpu_device_buffer_fill_kernargs_t*)kernarg_ptr;
  kernargs->target_ptr = target_ptr;
  kernargs->element_length = element_count;
  kernargs->pattern = pattern;

  iree_hal_amdgpu_blit_emplace_dispatch(context, kernel_args, dispatch_packet,
                                        grid_size_x, grid_size_y, kernarg_ptr);
  return true;
}

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_device_buffer_copy_*
//===----------------------------------------------------------------------===//
//
// IREE_HAL_AMDGPU_COPY_BLOCK_COUNT tuning sweep on gfx1100 (RDNA3, wave32,
// 96 CUs, GDDR6). Bandwidth in GB/s; src/tgt are buffer offset alignments.
// Build: --compilation_mode=opt --copt=-O3 --copt=-march=native
//        --copt=-flto=thin --linkopt=-flto=thin.
//
//                          QueueCopy (single)       QueueCopyBatch20
//   length   src tgt    cnt=1 cnt=2 cnt=4 cnt=8   cnt=1 cnt=2 cnt=4 cnt=8
//   --------------------------------------------------------------------
//    64KiB    0   0      2.7   2.5   2.7   3.0     5.7   5.4   7.1   8.1
//     2MiB    0   0     91.6  86.1  85.2  86.5   177.7 176.0 146.2 151.7
//     2MiB    1   2     91.7  89.7  82.6  80.6   176.7 161.0 138.4 134.7
//     1GiB    0   0    299.5 161.8 118.8 103.7   303.9 163.0 119.2 104.0
//
//   Geomean over bandwidth-relevant rows (64KiB..1GiB across alignments):
//     QueueCopy       : cnt=1: 28.1, cnt=2: 23.5, cnt=4: 23.0, cnt=8: 22.7
//     QueueCopyBatch20: cnt=1: 50.4, cnt=2: 43.3, cnt=4: 41.3, cnt=8: 41.2
//
// The 3x cliff at 1GiB matches the expected VGPR occupancy drop from the
// `#pragma unroll 8` body: every in-flight 16-byte copy needs ~4 VGPRs, so
// cnt=8 burns ~32 extra VGPRs, cutting max waves-per-SIMD from ~16 to ~5 on
// gfx1100 (256 VGPRs per SIMD) and starving the latency-hiding budget for
// bandwidth-bound transfers.
//
// Small batched transfers (<16KiB) prefer cnt=8 by ~15-20% because the work
// is launch-overhead-bound and each wave doing more reduces dispatch cost,
// but absolute throughput there is <1 GB/s and doesn't move real workloads.
// Sizes <64KiB are omitted here — they are measurement-variance-dominated.
//
// cnt=1 is the tuned value for gfx1100. CDNA (MI300+, 512 VGPRs per SIMD)
// may prefer a larger value and should be re-swept before changing.

#if defined(IREE_AMDGPU_TARGET_DEVICE)

IREE_AMDGPU_ATTRIBUTE_KERNEL void iree_hal_amdgpu_device_buffer_copy_x1(
    const uint8_t* IREE_AMDGPU_RESTRICT source_ptr,
    uint8_t* IREE_AMDGPU_RESTRICT target_ptr, const uint64_t element_length) {
  const uint64_t element_stride = iree_hal_amdgpu_blit_grid_size();
  for (uint64_t element_offset = iree_hal_amdgpu_blit_linear_id();
       element_offset < element_length;) {
    target_ptr[element_offset] = source_ptr[element_offset];
    if (!iree_hal_amdgpu_blit_advance(&element_offset, element_stride)) break;
  }
}

// Copies blocks of up to IREE_HAL_AMDGPU_COPY_BLOCK_X4_COUNT 4-byte elements
// from |source_ptr| to |target_ptr|. Requires an alignment of 4-bytes on all of
// |source_ptr|, |target_ptr|, and |length|.
IREE_AMDGPU_ATTRIBUTE_KERNEL void iree_hal_amdgpu_device_buffer_copy_block_x4(
    const uint32_t* IREE_AMDGPU_RESTRICT source_ptr,
    uint32_t* IREE_AMDGPU_RESTRICT target_ptr, const uint64_t element_length) {
  const uint64_t block_stride = iree_hal_amdgpu_blit_grid_size();
  for (uint64_t block_id = iree_hal_amdgpu_blit_linear_id();;) {
    const uint64_t element_offset =
        block_id * IREE_HAL_AMDGPU_COPY_BLOCK_X4_COUNT;
    if (IREE_AMDGPU_UNLIKELY(element_offset >= element_length)) return;
    const uint64_t element_count = IREE_AMDGPU_MIN(
        IREE_HAL_AMDGPU_COPY_BLOCK_X4_COUNT, element_length - element_offset);
    if (IREE_AMDGPU_LIKELY(element_count ==
                           IREE_HAL_AMDGPU_COPY_BLOCK_X4_COUNT)) {
#pragma unroll
      for (size_t i = 0; i < IREE_HAL_AMDGPU_COPY_BLOCK_X4_COUNT; ++i) {
        target_ptr[element_offset + i] = source_ptr[element_offset + i];
      }
    } else {
      for (size_t i = 0; i < element_count; ++i) {
        target_ptr[element_offset + i] = source_ptr[element_offset + i];
      }
    }
    if (!iree_hal_amdgpu_blit_advance(&block_id, block_stride)) return;
  }
}

// Copies blocks of up to IREE_HAL_AMDGPU_COPY_BLOCK_X8_COUNT 8-byte elements
// from |source_ptr| to |target_ptr|. Requires an alignment of 8-bytes on all of
// |source_ptr|, |target_ptr|, and |length|.
IREE_AMDGPU_ATTRIBUTE_KERNEL void iree_hal_amdgpu_device_buffer_copy_block_x8(
    const uint64_t* IREE_AMDGPU_RESTRICT source_ptr,
    uint64_t* IREE_AMDGPU_RESTRICT target_ptr, const uint64_t element_length) {
  const uint64_t block_stride = iree_hal_amdgpu_blit_grid_size();
  for (uint64_t block_id = iree_hal_amdgpu_blit_linear_id();;) {
    const uint64_t element_offset =
        block_id * IREE_HAL_AMDGPU_COPY_BLOCK_X8_COUNT;
    if (IREE_AMDGPU_UNLIKELY(element_offset >= element_length)) return;
    const uint64_t element_count = IREE_AMDGPU_MIN(
        IREE_HAL_AMDGPU_COPY_BLOCK_X8_COUNT, element_length - element_offset);
    if (IREE_AMDGPU_LIKELY(element_count ==
                           IREE_HAL_AMDGPU_COPY_BLOCK_X8_COUNT)) {
#pragma unroll
      for (size_t i = 0; i < IREE_HAL_AMDGPU_COPY_BLOCK_X8_COUNT; ++i) {
        target_ptr[element_offset + i] = source_ptr[element_offset + i];
      }
    } else {
      for (size_t i = 0; i < element_count; ++i) {
        target_ptr[element_offset + i] = source_ptr[element_offset + i];
      }
    }
    if (!iree_hal_amdgpu_blit_advance(&block_id, block_stride)) return;
  }
}

// Copies blocks of up to IREE_HAL_AMDGPU_COPY_BLOCK_COUNT 16-byte elements
// from |source_ptr| to |target_ptr|. Requires an alignment of 16-bytes on all
// of |source_ptr|, |target_ptr|, and |length|.
IREE_AMDGPU_ATTRIBUTE_KERNEL void iree_hal_amdgpu_device_buffer_copy_block_x16(
    const iree_amdgpu_uint64x2_t* IREE_AMDGPU_RESTRICT source_ptr,
    iree_amdgpu_uint64x2_t* IREE_AMDGPU_RESTRICT target_ptr,
    const uint64_t element_length) {
  const uint64_t block_stride = iree_hal_amdgpu_blit_grid_size();
  for (uint64_t block_id = iree_hal_amdgpu_blit_linear_id();;) {
    const uint64_t element_offset = block_id * IREE_HAL_AMDGPU_COPY_BLOCK_COUNT;
    if (IREE_AMDGPU_UNLIKELY(element_offset >= element_length)) return;
    const uint64_t element_count = IREE_AMDGPU_MIN(
        IREE_HAL_AMDGPU_COPY_BLOCK_COUNT, element_length - element_offset);
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
    if (!iree_hal_amdgpu_blit_advance(&block_id, block_stride)) return;
  }
}

// Copies a byte-granular region using 16-byte vector loads/stores where
// possible. See the note on fill_block_unaligned_x16 regarding the
// |element_length| field sharing its name with the aligned variants despite
// being a byte length here.
IREE_AMDGPU_ATTRIBUTE_KERNEL void
iree_hal_amdgpu_device_buffer_copy_block_unaligned_x16(
    const iree_amdgpu_unaligned_uint64x2_t* IREE_AMDGPU_RESTRICT source_ptr,
    iree_amdgpu_unaligned_uint64x2_t* IREE_AMDGPU_RESTRICT target_ptr,
    const uint64_t element_length) {
  const uint64_t full_element_count =
      element_length / IREE_HAL_AMDGPU_COPY_BLOCK_ELEMENT_SIZE;
  const uint64_t vector_block_count = IREE_AMDGPU_CEIL_DIV(
      full_element_count, IREE_HAL_AMDGPU_COPY_BLOCK_COUNT);
  const uint64_t tail_offset =
      full_element_count * IREE_HAL_AMDGPU_COPY_BLOCK_ELEMENT_SIZE;
  const uint64_t tail_length = element_length - tail_offset;
  const uint64_t block_stride = iree_hal_amdgpu_blit_grid_size();
  for (uint64_t block_id = iree_hal_amdgpu_blit_linear_id();;) {
    if (IREE_AMDGPU_UNLIKELY(vector_block_count == 0)) {
      if (block_id == 0) {
        const uint8_t* source_tail_ptr = (const uint8_t*)source_ptr;
        uint8_t* target_tail_ptr = (uint8_t*)target_ptr;
        for (uint64_t i = 0; i < tail_length; ++i) {
          target_tail_ptr[i] = source_tail_ptr[i];
        }
      }
      return;
    }
    if (IREE_AMDGPU_UNLIKELY(block_id >= vector_block_count)) return;
    const uint64_t element_offset = block_id * IREE_HAL_AMDGPU_COPY_BLOCK_COUNT;
    const uint64_t element_count = IREE_AMDGPU_MIN(
        IREE_HAL_AMDGPU_COPY_BLOCK_COUNT, full_element_count - element_offset);
    if (IREE_AMDGPU_LIKELY(element_count == IREE_HAL_AMDGPU_COPY_BLOCK_COUNT)) {
#pragma unroll
      for (size_t i = 0; i < IREE_HAL_AMDGPU_COPY_BLOCK_COUNT; ++i) {
        target_ptr[element_offset + i].value =
            source_ptr[element_offset + i].value;
      }
    } else {
      for (size_t i = 0; i < element_count; ++i) {
        target_ptr[element_offset + i].value =
            source_ptr[element_offset + i].value;
      }
    }
    if (IREE_AMDGPU_UNLIKELY(tail_length &&
                             block_id + 1 == vector_block_count)) {
      const uint8_t* source_tail_ptr = (const uint8_t*)source_ptr + tail_offset;
      uint8_t* target_tail_ptr = (uint8_t*)target_ptr + tail_offset;
      for (uint64_t i = 0; i < tail_length; ++i) {
        target_tail_ptr[i] = source_tail_ptr[i];
      }
    }
    if (!iree_hal_amdgpu_blit_advance(&block_id, block_stride)) return;
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
  uint64_t element_size = 1;
  uint64_t block_size = 1;
  bool uses_byte_length = false;
  if (iree_amdgpu_has_alignment((uintptr_t)source_ptr,
                                IREE_HAL_AMDGPU_COPY_BLOCK_ELEMENT_SIZE) &&
      iree_amdgpu_has_alignment((uintptr_t)target_ptr,
                                IREE_HAL_AMDGPU_COPY_BLOCK_ELEMENT_SIZE) &&
      iree_amdgpu_has_alignment(length,
                                IREE_HAL_AMDGPU_COPY_BLOCK_ELEMENT_SIZE)) {
    kernel_args =
        &context->kernels->iree_hal_amdgpu_device_buffer_copy_block_x16;
    element_size = IREE_HAL_AMDGPU_COPY_BLOCK_ELEMENT_SIZE;
    block_size = IREE_HAL_AMDGPU_COPY_BLOCK_COUNT;
  } else if (iree_amdgpu_has_alignment(
                 (uintptr_t)source_ptr,
                 IREE_HAL_AMDGPU_COPY_BLOCK_X8_ELEMENT_SIZE) &&
             iree_amdgpu_has_alignment(
                 (uintptr_t)target_ptr,
                 IREE_HAL_AMDGPU_COPY_BLOCK_X8_ELEMENT_SIZE) &&
             iree_amdgpu_has_alignment(
                 length, IREE_HAL_AMDGPU_COPY_BLOCK_X8_ELEMENT_SIZE)) {
    kernel_args =
        &context->kernels->iree_hal_amdgpu_device_buffer_copy_block_x8;
    element_size = IREE_HAL_AMDGPU_COPY_BLOCK_X8_ELEMENT_SIZE;
    block_size = IREE_HAL_AMDGPU_COPY_BLOCK_X8_COUNT;
  } else if (iree_amdgpu_has_alignment(
                 (uintptr_t)source_ptr,
                 IREE_HAL_AMDGPU_COPY_BLOCK_X4_ELEMENT_SIZE) &&
             iree_amdgpu_has_alignment(
                 (uintptr_t)target_ptr,
                 IREE_HAL_AMDGPU_COPY_BLOCK_X4_ELEMENT_SIZE) &&
             iree_amdgpu_has_alignment(
                 length, IREE_HAL_AMDGPU_COPY_BLOCK_X4_ELEMENT_SIZE)) {
    kernel_args =
        &context->kernels->iree_hal_amdgpu_device_buffer_copy_block_x4;
    element_size = IREE_HAL_AMDGPU_COPY_BLOCK_X4_ELEMENT_SIZE;
    block_size = IREE_HAL_AMDGPU_COPY_BLOCK_X4_COUNT;
  } else if (length >= IREE_HAL_AMDGPU_COPY_BLOCK_UNALIGNED_MIN_SIZE) {
    kernel_args = &context->kernels
                       ->iree_hal_amdgpu_device_buffer_copy_block_unaligned_x16;
    element_size = 1;
    block_size = 1;
    uses_byte_length = true;
  } else {
    kernel_args = &context->kernels->iree_hal_amdgpu_device_buffer_copy_x1;
    element_size = 1;
    block_size = 1;
  }

  const uint64_t element_count = length / element_size;
  const uint64_t block_count =
      uses_byte_length ? iree_hal_amdgpu_blit_unaligned_block_count(
                             length, IREE_HAL_AMDGPU_COPY_BLOCK_ELEMENT_SIZE,
                             IREE_HAL_AMDGPU_COPY_BLOCK_COUNT)
                       : IREE_AMDGPU_CEIL_DIV(element_count, block_size);
  uint32_t grid_size_x = 0;
  uint32_t grid_size_y = 0;
  if (IREE_AMDGPU_UNLIKELY(!iree_hal_amdgpu_blit_calculate_grid_size(
          context, block_count, &grid_size_x, &grid_size_y))) {
    return false;
  }

  // Update kernargs (same API for all kernels).
  iree_hal_amdgpu_device_buffer_copy_kernargs_t* kernargs =
      (iree_hal_amdgpu_device_buffer_copy_kernargs_t*)kernarg_ptr;
  kernargs->source_ptr = source_ptr;
  kernargs->target_ptr = target_ptr;
  kernargs->element_length = element_count;

  iree_hal_amdgpu_blit_emplace_dispatch(context, kernel_args, dispatch_packet,
                                        grid_size_x, grid_size_y, kernarg_ptr);
  return true;
}
