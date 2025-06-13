// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/device/blit.h"

//===----------------------------------------------------------------------===//
// Buffer transfer operation utilities
//===----------------------------------------------------------------------===//

// Reserves the next packet in the queue and returns its packet_id.
// If tracing is enabled |out_completion_signal| will be populated with the
// signal that must be attached to the operation.
static uint64_t iree_hal_amdgpu_device_blit_reserve(
    const iree_hal_amdgpu_device_buffer_transfer_context_t* IREE_AMDGPU_RESTRICT
        context,
    iree_hal_amdgpu_trace_execution_zone_type_t zone_type,
    iree_hsa_signal_t* IREE_AMDGPU_RESTRICT out_completion_signal) {
#if IREE_HAL_AMDGPU_HAS_TRACING_FEATURE( \
    IREE_HAL_AMDGPU_TRACING_FEATURE_DEVICE_EXECUTION)
  if (context->trace_buffer) {
    iree_hal_amdgpu_trace_execution_query_id_t execution_query_id =
        iree_hal_amdgpu_device_query_ringbuffer_acquire(
            &context->trace_buffer->query_ringbuffer);
    *out_completion_signal =
        iree_hal_amdgpu_device_trace_execution_zone_dispatch(
            context->trace_buffer, zone_type, 0, execution_query_id);
  } else {
    *out_completion_signal = iree_hsa_signal_null();
  }
#endif  // IREE_HAL_AMDGPU_TRACING_FEATURE_DEVICE_EXECUTION

  // Reserve the next packet in the queue.
  const uint64_t packet_id = iree_hsa_queue_add_write_index(
      &context->queue, 1u, iree_amdgpu_memory_order_relaxed);
  while (packet_id - iree_hsa_queue_load_read_index(
                         &context->queue, iree_amdgpu_memory_order_acquire) >=
         context->queue.size) {
    iree_amdgpu_yield();  // spinning
  }

  return packet_id;
}

// Commits a reserved transfer packet.
// The header will be updated and the target queue doorbell will be signaled.
static void iree_hal_amdgpu_device_blit_commit(
    const iree_hal_amdgpu_device_buffer_transfer_context_t* IREE_AMDGPU_RESTRICT
        context,
    uint64_t packet_id,
    iree_hsa_kernel_dispatch_packet_t* IREE_AMDGPU_RESTRICT packet,
    iree_hsa_signal_t completion_signal) {
  // Chain completion.
  packet->completion_signal = completion_signal;

  // Populate the header and release the packet to the queue.
  uint16_t header = IREE_HSA_PACKET_TYPE_KERNEL_DISPATCH
                    << IREE_HSA_PACKET_HEADER_TYPE;

  // TODO(benvanik): need to pull in barrier/scope overrides from command buffer
  // execution context flags. They should override the barrier bit and the
  // scopes to be on SYSTEM regardless of what we choose here.

  // NOTE: we don't need a barrier bit as the caller is expecting it to run
  // concurrently if needed.
  header |= 0 << IREE_HSA_PACKET_HEADER_BARRIER;

#if IREE_HAL_AMDGPU_HAS_TRACING_FEATURE( \
    IREE_HAL_AMDGPU_TRACING_FEATURE_DEVICE_EXECUTION)
  if (context->trace_buffer) {
    // Force a barrier bit if we are tracing execution. This ensures that we get
    // exclusive timing for the operation.
    header |= 1 << IREE_HSA_PACKET_HEADER_BARRIER;
  }
#endif  // IREE_HAL_AMDGPU_TRACING_FEATURE_DEVICE_EXECUTION

  // TODO(benvanik): scope to agent if the pointer is local, or maybe none in
  // cases where surrounding barriers performed the cache management.
  header |= IREE_HSA_FENCE_SCOPE_SYSTEM
            << IREE_HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE;
  header |= IREE_HSA_FENCE_SCOPE_SYSTEM
            << IREE_HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE;

  const uint32_t header_setup = header | (uint32_t)(packet->setup << 16);
  iree_amdgpu_scoped_atomic_store(
      (iree_amdgpu_scoped_atomic_uint32_t*)packet, header_setup,
      iree_amdgpu_memory_order_release, iree_amdgpu_memory_scope_system);

  // Signal the queue doorbell indicating the packet has been updated.
  iree_hsa_signal_store(context->queue.doorbell_signal, packet_id,
                        iree_amdgpu_memory_order_relaxed);
}

//===----------------------------------------------------------------------===//
// Blit kernel utilities
//===----------------------------------------------------------------------===//

// 2 uint64_t values totaling 16 bytes.
typedef uint32_t iree_amdgpu_uint64x2_t __attribute__((vector_size(16)));

static inline size_t iree_hal_amdgpu_blit_linear_id(void) {
  const size_t id_x = iree_hal_amdgpu_device_group_id_x() *
                          IREE_HAL_AMDGPU_BLIT_WORKGROUP_SIZE_X +
                      iree_hal_amdgpu_device_local_id_x();
  const size_t id_y = iree_hal_amdgpu_device_group_id_y() *
                          IREE_HAL_AMDGPU_BLIT_WORKGROUP_SIZE_Y +
                      iree_hal_amdgpu_device_local_id_y();
  return id_y * iree_amdgcn_dispatch_ptr()->grid_size[0] + id_x;
}

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_device_buffer_fill_*
//===----------------------------------------------------------------------===//

IREE_AMDGPU_ATTRIBUTE_KERNEL void iree_hal_amdgpu_device_buffer_fill_x1(
    uint8_t* IREE_AMDGPU_RESTRICT target_ptr, const uint64_t element_length,
    const uint8_t pattern) {
  const size_t element_offset = iree_hal_amdgpu_blit_linear_id();
  if (IREE_AMDGPU_LIKELY(element_offset < element_length)) {
    // Slowest possible copy; benchmarks required to iterate on better impls.
    target_ptr[element_offset] = pattern;
  }
}

IREE_AMDGPU_ATTRIBUTE_KERNEL void iree_hal_amdgpu_device_buffer_fill_x2(
    uint16_t* IREE_AMDGPU_RESTRICT target_ptr, const uint64_t element_length,
    const uint16_t pattern) {
  const size_t element_offset = iree_hal_amdgpu_blit_linear_id();
  if (IREE_AMDGPU_LIKELY(element_offset < element_length)) {
    // Slowest possible fill; benchmarks required to iterate on better impls.
    target_ptr[element_offset] = pattern;
  }
}

IREE_AMDGPU_ATTRIBUTE_KERNEL void iree_hal_amdgpu_device_buffer_fill_x4(
    uint32_t* IREE_AMDGPU_RESTRICT target_ptr, const uint64_t element_length,
    const uint32_t pattern) {
  const size_t element_offset = iree_hal_amdgpu_blit_linear_id();
  if (IREE_AMDGPU_LIKELY(element_offset < element_length)) {
    // Slowest possible fill; benchmarks required to iterate on better impls.
    target_ptr[element_offset] = pattern;
  }
}

IREE_AMDGPU_ATTRIBUTE_KERNEL void iree_hal_amdgpu_device_buffer_fill_x8(
    uint64_t* IREE_AMDGPU_RESTRICT target_ptr, const uint64_t element_length,
    const uint64_t pattern) {
  const size_t element_offset = iree_hal_amdgpu_blit_linear_id();
  if (IREE_AMDGPU_LIKELY(element_offset < element_length)) {
    // Slowest possible fill; benchmarks required to iterate on better impls.
    target_ptr[element_offset] = pattern;
  }
}

#define IREE_HAL_AMDGPU_FILL_BLOCK_ELEMENT_SIZE sizeof(iree_amdgpu_uint64x2_t)
#define IREE_HAL_AMDGPU_FILL_BLOCK_COUNT 8
#define IREE_HAL_AMDGPU_FILL_BLOCK_SIZE \
  (IREE_HAL_AMDGPU_FILL_BLOCK_ELEMENT_SIZE * IREE_HAL_AMDGPU_FILL_BLOCK_COUNT)

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
  const size_t element_count =
      IREE_AMDGPU_MIN(IREE_HAL_AMDGPU_FILL_BLOCK_COUNT,
                      element_length - element_offset) /
      sizeof(pattern_x16);
  if (IREE_AMDGPU_LIKELY(element_count == IREE_HAL_AMDGPU_FILL_BLOCK_COUNT)) {
#pragma unroll
    for (int i = 0; i < IREE_HAL_AMDGPU_FILL_BLOCK_COUNT; ++i) {
      target_ptr[element_offset + i] = pattern_x16;
    }
  } else {
    for (int i = 0; i < element_count; ++i) {
      target_ptr[element_offset + i] = pattern_x16;
    }
  }
}

// Returns the bytes of |pattern| of length |pattern_length| splatted to
// an 8-byte value.
static uint64_t iree_hal_amdgpu_device_extend_pattern_x8(
    const uint64_t pattern, const uint8_t pattern_length) {
  switch (pattern_length) {
    case 1:
      return ((uint64_t)pattern << 56) | ((uint64_t)pattern << 48) |
             ((uint64_t)pattern << 40) | ((uint64_t)pattern << 32) |
             ((uint64_t)pattern << 24) | ((uint64_t)pattern << 16) |
             ((uint64_t)pattern << 8) | pattern;
    case 2:
      return ((uint64_t)pattern << 48) | ((uint64_t)pattern << 32) |
             ((uint64_t)pattern << 16) | pattern;
    case 4:
      return ((uint64_t)pattern << 32) | pattern;
    case 8:
      return pattern;
    default:
      return 0;
  }
}

iree_hsa_kernel_dispatch_packet_t* IREE_AMDGPU_RESTRICT
iree_hal_amdgpu_device_buffer_fill_emplace_reserve(
    const iree_hal_amdgpu_device_buffer_transfer_context_t* IREE_AMDGPU_RESTRICT
        context,
    void* target_ptr, const uint64_t length, uint64_t pattern,
    const uint8_t pattern_length, uint64_t* IREE_AMDGPU_RESTRICT kernarg_ptr,
    const uint64_t packet_id) {
  IREE_AMDGPU_TRACE_BUFFER_SCOPE(context->trace_buffer);
  IREE_AMDGPU_TRACE_ZONE_BEGIN(z0);

  // Select the kernel for the fill operation.
  const iree_hal_amdgpu_device_kernel_args_t* IREE_AMDGPU_RESTRICT kernel_args =
      NULL;
  size_t element_size = 1;
  size_t block_size = 1;
  if (iree_amdgpu_has_alignment((size_t)target_ptr,
                                IREE_HAL_AMDGPU_FILL_BLOCK_ELEMENT_SIZE) &&
      iree_amdgpu_has_alignment(length,
                                IREE_HAL_AMDGPU_FILL_BLOCK_ELEMENT_SIZE)) {
    IREE_AMDGPU_TRACE_ZONE_APPEND_TEXT_LITERAL(z0, "fill_block_x16");
    pattern = iree_hal_amdgpu_device_extend_pattern_x8(pattern, pattern_length);
    kernel_args =
        &context->kernels->iree_hal_amdgpu_device_buffer_fill_block_x16;
    element_size = IREE_HAL_AMDGPU_FILL_BLOCK_ELEMENT_SIZE;
    block_size = IREE_HAL_AMDGPU_FILL_BLOCK_COUNT;
  } else {
    switch (pattern_length) {
      case 1:
        IREE_AMDGPU_TRACE_ZONE_APPEND_TEXT_LITERAL(z0, "fill_x1");
        kernel_args = &context->kernels->iree_hal_amdgpu_device_buffer_fill_x1;
        break;
      case 2:
        IREE_AMDGPU_TRACE_ZONE_APPEND_TEXT_LITERAL(z0, "fill_x2");
        kernel_args = &context->kernels->iree_hal_amdgpu_device_buffer_fill_x2;
        break;
      case 4:
        IREE_AMDGPU_TRACE_ZONE_APPEND_TEXT_LITERAL(z0, "fill_x4");
        kernel_args = &context->kernels->iree_hal_amdgpu_device_buffer_fill_x4;
        break;
      case 8:
        IREE_AMDGPU_TRACE_ZONE_APPEND_TEXT_LITERAL(z0, "fill_x8");
        kernel_args = &context->kernels->iree_hal_amdgpu_device_buffer_fill_x8;
        break;
    }
    element_size = pattern_length;
    block_size = 1;
  }

  // Update kernargs (same API for all kernels).
  const size_t element_count = length / element_size;
  iree_hal_amdgpu_device_buffer_fill_kernargs_t* kernargs =
      (iree_hal_amdgpu_device_buffer_fill_kernargs_t*)kernarg_ptr;
  kernargs->target_ptr = target_ptr;
  kernargs->element_length = element_count;
  kernargs->pattern = pattern;

  // To support fills with more than UINT_MAX elements (uint32_t grid_size)
  // we divide the problem into chunks as needed. We keep the innermost chunk
  // size small as if we do [X,Y,1] we're likely to overshoot and don't want to
  // have too many wasted invocations.
  const size_t block_count = IREE_AMDGPU_CEIL_DIV(element_count, block_size);
  uint32_t grid_size_x = 1;
  uint32_t grid_size_y = 1;
  if (IREE_AMDGPU_LIKELY(block_count <= 0xFFFFFFFFu)) {
    grid_size_x = (uint32_t)block_count;
  } else {
    grid_size_x = 256;
    grid_size_y = (uint32_t)IREE_AMDGPU_CEIL_DIV(block_count, grid_size_x);
  }

  // Populate the packet.
  const uint64_t queue_mask = context->queue.size - 1;  // power of two
  iree_hsa_kernel_dispatch_packet_t* IREE_AMDGPU_RESTRICT dispatch_packet =
      context->queue.base_address + (packet_id & queue_mask) * 64;
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

  IREE_AMDGPU_TRACE_ZONE_END(z0);
  return dispatch_packet;
}

void iree_hal_amdgpu_device_buffer_fill_enqueue(
    const iree_hal_amdgpu_device_buffer_transfer_context_t* IREE_AMDGPU_RESTRICT
        context,
    void* target_ptr, const uint64_t length, const uint64_t pattern,
    const uint8_t pattern_length, uint64_t* IREE_AMDGPU_RESTRICT kernarg_ptr) {
  IREE_AMDGPU_TRACE_BUFFER_SCOPE(context->trace_buffer);
  IREE_AMDGPU_TRACE_ZONE_BEGIN(z0);
  IREE_AMDGPU_TRACE_ZONE_APPEND_VALUE_I64(z0, length);

  // Reserve and begin populating the operation packet.
  // When tracing is enabled capture the timing signal.
  iree_hsa_signal_t completion_signal = iree_hsa_signal_null();
  const uint64_t packet_id = iree_hal_amdgpu_device_blit_reserve(
      context, IREE_HAL_AMDGPU_TRACE_EXECUTION_ZONE_TYPE_FILL,
      &completion_signal);

  // Emplace the dispatch packet into the queue.
  // Note that until the packet is issued the queue will stall.
  iree_hsa_kernel_dispatch_packet_t* IREE_AMDGPU_RESTRICT packet =
      iree_hal_amdgpu_device_buffer_fill_emplace_reserve(
          context, target_ptr, length, pattern, pattern_length, kernarg_ptr,
          packet_id);

  // Issues the buffer operation packet by configuring its header and signaling
  // the queue doorbell.
  iree_hal_amdgpu_device_blit_commit(context, packet_id, packet,
                                     completion_signal);

  IREE_AMDGPU_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_device_buffer_copy_*
//===----------------------------------------------------------------------===//

IREE_AMDGPU_ATTRIBUTE_KERNEL void iree_hal_amdgpu_device_buffer_copy_x1(
    const uint8_t* IREE_AMDGPU_RESTRICT source_ptr,
    uint8_t* IREE_AMDGPU_RESTRICT target_ptr, const uint64_t element_length) {
  const size_t element_offset = iree_hal_amdgpu_blit_linear_id();
  if (IREE_AMDGPU_LIKELY(element_offset < element_length)) {
    // Slowest possible copy; benchmarks required to iterate on better impls.
    target_ptr[element_offset] = source_ptr[element_offset];
  }
}

#define IREE_HAL_AMDGPU_COPY_BLOCK_ELEMENT_SIZE sizeof(iree_amdgpu_uint64x2_t)
#define IREE_HAL_AMDGPU_COPY_BLOCK_COUNT 8
#define IREE_HAL_AMDGPU_COPY_BLOCK_SIZE \
  (IREE_HAL_AMDGPU_COPY_BLOCK_ELEMENT_SIZE * IREE_HAL_AMDGPU_COPY_BLOCK_COUNT)

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
    for (int i = 0; i < IREE_HAL_AMDGPU_COPY_BLOCK_COUNT; ++i) {
      target_ptr[element_offset + i] = source_ptr[element_offset + i];
    }
  } else {
    for (int i = 0; i < element_count; ++i) {
      target_ptr[element_offset + i] = source_ptr[element_offset + i];
    }
  }
}

// TODO(benvanik): experiment with enqueuing SDMA somehow (may need to take a
// DMA queue as well as the dispatch queue). Note that on some configurations
// (InfinityFabric) blit kernels can be 2x faster than SDMA.
iree_hsa_kernel_dispatch_packet_t* IREE_AMDGPU_RESTRICT
iree_hal_amdgpu_device_buffer_copy_emplace_reserve(
    const iree_hal_amdgpu_device_buffer_transfer_context_t* IREE_AMDGPU_RESTRICT
        context,
    const void* source_ptr, void* target_ptr, const uint64_t length,
    uint64_t* IREE_AMDGPU_RESTRICT kernarg_ptr, const uint64_t packet_id) {
  IREE_AMDGPU_TRACE_BUFFER_SCOPE(context->trace_buffer);
  IREE_AMDGPU_TRACE_ZONE_BEGIN(z0);

  // Select the kernel for the copy operation.
  // TODO(benvanik): switch kernel based on source/target/length alignment.
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
    IREE_AMDGPU_TRACE_ZONE_APPEND_TEXT_LITERAL(z0, "copy_block_x16");
    kernel_args =
        &context->kernels->iree_hal_amdgpu_device_buffer_copy_block_x16;
    element_size = IREE_HAL_AMDGPU_COPY_BLOCK_ELEMENT_SIZE;
    block_size = IREE_HAL_AMDGPU_COPY_BLOCK_COUNT;
  } else {
    IREE_AMDGPU_TRACE_ZONE_APPEND_TEXT_LITERAL(z0, "copy_x1");
    kernel_args = &context->kernels->iree_hal_amdgpu_device_buffer_copy_x1;
    element_size = 1;
    block_size = 1;
  }

  // Update kernargs (same API for all kernels).
  const size_t element_count = length / element_size;
  iree_hal_amdgpu_device_buffer_copy_kernargs_t* kernargs =
      (iree_hal_amdgpu_device_buffer_copy_kernargs_t*)kernarg_ptr;
  kernargs->source_ptr = source_ptr;
  kernargs->target_ptr = target_ptr;
  kernargs->element_length = element_count;

  // To support transfers with more than UINT_MAX elements (uint32_t grid_size)
  // we divide the problem into chunks as needed. We keep the innermost chunk
  // size small as if we do [X,Y,1] we're likely to overshoot and don't want to
  // have too many wasted invocations.
  const size_t block_count = IREE_AMDGPU_CEIL_DIV(element_count, block_size);
  uint32_t grid_size_x = 1;
  uint32_t grid_size_y = 1;
  if (IREE_AMDGPU_LIKELY(block_count <= 0xFFFFFFFFu)) {
    grid_size_x = (uint32_t)block_count;
  } else {
    grid_size_x = 256;
    grid_size_y = (uint32_t)IREE_AMDGPU_CEIL_DIV(block_count, grid_size_x);
  }

  // Populate the packet.
  const uint64_t queue_mask = context->queue.size - 1;  // power of two
  iree_hsa_kernel_dispatch_packet_t* IREE_AMDGPU_RESTRICT dispatch_packet =
      context->queue.base_address + (packet_id & queue_mask) * 64;
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

  IREE_AMDGPU_TRACE_ZONE_END(z0);
  return dispatch_packet;
}

void iree_hal_amdgpu_device_buffer_copy_enqueue(
    const iree_hal_amdgpu_device_buffer_transfer_context_t* IREE_AMDGPU_RESTRICT
        context,
    const void* source_ptr, void* target_ptr, const uint64_t length,
    uint64_t* IREE_AMDGPU_RESTRICT kernarg_ptr) {
  IREE_AMDGPU_TRACE_BUFFER_SCOPE(context->trace_buffer);
  IREE_AMDGPU_TRACE_ZONE_BEGIN(z0);
  IREE_AMDGPU_TRACE_ZONE_APPEND_VALUE_I64(z0, length);

  // Reserve and begin populating the operation packet.
  // When tracing is enabled capture the timing signal.
  iree_hsa_signal_t completion_signal = iree_hsa_signal_null();
  const uint64_t packet_id = iree_hal_amdgpu_device_blit_reserve(
      context, IREE_HAL_AMDGPU_TRACE_EXECUTION_ZONE_TYPE_COPY,
      &completion_signal);

  // Emplace the dispatch packet into the queue.
  // Note that until the packet is issued the queue will stall.
  iree_hsa_kernel_dispatch_packet_t* IREE_AMDGPU_RESTRICT packet =
      iree_hal_amdgpu_device_buffer_copy_emplace_reserve(
          context, source_ptr, target_ptr, length, kernarg_ptr, packet_id);

  // Issues the buffer operation packet by configuring its header and signaling
  // the queue doorbell.
  iree_hal_amdgpu_device_blit_commit(context, packet_id, packet,
                                     completion_signal);

  IREE_AMDGPU_TRACE_ZONE_END(z0);
}
