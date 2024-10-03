// Copyright 2024 The IREE Authors
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
    // DO NOT SUBMIT may not be right
    const uint64_t offset = buffer_ref.offset + binding.offset;
    const uint64_t length = buffer_ref.length == UINT64_MAX
                                ? binding.length - buffer_ref.offset
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
    // DO NOT SUBMIT may not be right
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

//===----------------------------------------------------------------------===//
// Buffer transfer operation utilities
//===----------------------------------------------------------------------===//

// Reserves the next packet in the queue and returns its packet_id.
// If tracing is enabled |out_completion_signal| will be populated with the
// signal that must be attached to the operation.
static uint64_t iree_hal_amdgpu_device_buffer_op_reserve(
    const iree_hal_amdgpu_device_buffer_transfer_state_t* IREE_AMDGPU_RESTRICT
        state,
    iree_hal_amdgpu_trace_execution_zone_type_t zone_type,
    iree_hsa_signal_t* IREE_AMDGPU_RESTRICT out_completion_signal) {
#if IREE_HAL_AMDGPU_HAS_TRACING_FEATURE( \
    IREE_HAL_AMDGPU_TRACING_FEATURE_DEVICE_EXECUTION)
  if (state->trace_buffer) {
    iree_hal_amdgpu_trace_execution_query_id_t execution_query_id =
        iree_hal_amdgpu_device_query_ringbuffer_acquire(
            &state->trace_buffer->query_ringbuffer);
    *out_completion_signal =
        iree_hal_amdgpu_device_trace_execution_zone_dispatch(
            state->trace_buffer, zone_type, 0, execution_query_id);
  }
#endif  // IREE_HAL_AMDGPU_TRACING_FEATURE_DEVICE_EXECUTION

  // Reserve the next packet in the queue.
  const uint64_t packet_id = iree_hsa_queue_add_write_index(
      state->queue, 1u, iree_amdgpu_memory_order_relaxed);
  while (packet_id - iree_hsa_queue_load_read_index(
                         state->queue, iree_amdgpu_memory_order_acquire) >=
         state->queue->size) {
    iree_amdgpu_yield();  // spinning
  }

  return packet_id;
}

// Commits a reserved transfer packet.
// The header will be updated and the target queue doorbell will be signaled.
static void iree_hal_amdgpu_device_buffer_op_commit(
    const iree_hal_amdgpu_device_buffer_transfer_state_t* IREE_AMDGPU_RESTRICT
        state,
    uint64_t packet_id,
    iree_hsa_kernel_dispatch_packet_t* IREE_AMDGPU_RESTRICT packet,
    iree_hsa_signal_t completion_signal) {
  // Chain completion.
  packet->completion_signal = completion_signal;

  // Populate the header and release the packet to the queue.
  uint16_t header = IREE_HSA_PACKET_TYPE_KERNEL_DISPATCH
                    << IREE_HSA_PACKET_HEADER_TYPE;

  // TODO(benvanik): need to pull in barrier/scope overrides from command buffer
  // execution state flags. They should override the barrier bit and the scopes
  // to be on SYSTEM regardless of what we choose here.

  // NOTE: we don't need a barrier bit as the caller is expecting it to run
  // concurrently if needed.
  header |= 0 << IREE_HSA_PACKET_HEADER_BARRIER;

#if IREE_HAL_AMDGPU_HAS_TRACING_FEATURE( \
    IREE_HAL_AMDGPU_TRACING_FEATURE_DEVICE_EXECUTION)
  if (state->trace_buffer) {
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
  iree_hsa_signal_store(state->queue->doorbell_signal, packet_id,
                        iree_amdgpu_memory_order_relaxed);
}

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_device_buffer_fill_*
//===----------------------------------------------------------------------===//

IREE_AMDGPU_ATTRIBUTE_KERNEL void iree_hal_amdgpu_device_buffer_fill_x1(
    void* IREE_AMDGPU_RESTRICT target_ptr, const uint64_t length,
    const uint8_t pattern) {
  // DO NOT SUBMIT fill kernel
  // runtime/src/iree/hal/drivers/metal/builtin/fill_buffer_generic.metal
  // fill_buffer_1byte
}

IREE_AMDGPU_ATTRIBUTE_KERNEL void iree_hal_amdgpu_device_buffer_fill_x2(
    void* IREE_AMDGPU_RESTRICT target_ptr, const uint64_t length,
    const uint16_t pattern) {
  // DO NOT SUBMIT fill kernel
  // runtime/src/iree/hal/drivers/metal/builtin/fill_buffer_generic.metal
  // fill_buffer_2byte
}

IREE_AMDGPU_ATTRIBUTE_KERNEL void iree_hal_amdgpu_device_buffer_fill_x4(
    void* IREE_AMDGPU_RESTRICT target_ptr, const uint64_t length,
    const uint32_t pattern) {
  // DO NOT SUBMIT fill kernel
  // runtime/src/iree/hal/drivers/metal/builtin/fill_buffer_generic.metal
  // fill_buffer_4byte
}

IREE_AMDGPU_ATTRIBUTE_KERNEL void iree_hal_amdgpu_device_buffer_fill_x8(
    void* IREE_AMDGPU_RESTRICT target_ptr, const uint64_t length,
    const uint64_t pattern) {
  // DO NOT SUBMIT fill kernel
  // runtime/src/iree/hal/drivers/metal/builtin/fill_buffer_generic.metal
  // fill_buffer_8byte
}

iree_hsa_kernel_dispatch_packet_t* IREE_AMDGPU_RESTRICT
iree_hal_amdgpu_device_buffer_fill_emplace_reserve(
    const iree_hal_amdgpu_device_buffer_transfer_state_t* IREE_AMDGPU_RESTRICT
        state,
    void* target_ptr, const uint64_t length, const uint64_t pattern,
    const uint8_t pattern_length, uint64_t* IREE_AMDGPU_RESTRICT kernarg_ptr,
    const uint64_t packet_id) {
  IREE_AMDGPU_TRACE_BUFFER_SCOPE(state->trace_buffer);
  IREE_AMDGPU_TRACE_ZONE_BEGIN(z0);

  // Update kernargs (same for all kernels).
  kernarg_ptr[0] = (uint64_t)target_ptr;
  kernarg_ptr[1] = length;
  kernarg_ptr[2] = pattern;

  // Select the kernel for the fill operation.
  const iree_hal_amdgpu_device_kernel_args_t* IREE_AMDGPU_RESTRICT kernel_args =
      NULL;
  uint64_t block_size = 0;
  switch (pattern_length) {
    case 1:
      IREE_AMDGPU_TRACE_ZONE_APPEND_TEXT_LITERAL(z0, "fill_x1");
      kernel_args = &state->kernels->iree_hal_amdgpu_device_buffer_fill_x1;
      block_size = 1;
      break;
    case 2:
      IREE_AMDGPU_TRACE_ZONE_APPEND_TEXT_LITERAL(z0, "fill_x2");
      kernel_args = &state->kernels->iree_hal_amdgpu_device_buffer_fill_x2;
      block_size = 1;
      break;
    case 4:
      IREE_AMDGPU_TRACE_ZONE_APPEND_TEXT_LITERAL(z0, "fill_x4");
      kernel_args = &state->kernels->iree_hal_amdgpu_device_buffer_fill_x4;
      block_size = 1;
      break;
    case 8:
      IREE_AMDGPU_TRACE_ZONE_APPEND_TEXT_LITERAL(z0, "fill_x8");
      kernel_args = &state->kernels->iree_hal_amdgpu_device_buffer_fill_x8;
      block_size = 1;
      break;
  }
  IREE_AMDGPU_TRACE_ZONE_APPEND_VALUE_I64(z0, block_size);

  // Populate the packet.
  const uint64_t queue_mask = state->queue->size - 1;  // power of two
  iree_hsa_kernel_dispatch_packet_t* IREE_AMDGPU_RESTRICT dispatch_packet =
      state->queue->base_address + (packet_id & queue_mask) * 64;
  dispatch_packet->setup = kernel_args->setup;
  dispatch_packet->workgroup_size[0] = kernel_args->workgroup_size[0];
  dispatch_packet->workgroup_size[1] = kernel_args->workgroup_size[1];
  dispatch_packet->workgroup_size[2] = kernel_args->workgroup_size[2];
  dispatch_packet->reserved0 = 0;
  dispatch_packet->grid_size[0] = 0;  // DO NOT SUBMIT block count?
  dispatch_packet->grid_size[1] = 1;
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
    const iree_hal_amdgpu_device_buffer_transfer_state_t* IREE_AMDGPU_RESTRICT
        state,
    void* target_ptr, const uint64_t length, const uint64_t pattern,
    const uint8_t pattern_length, uint64_t* IREE_AMDGPU_RESTRICT kernarg_ptr) {
  IREE_AMDGPU_TRACE_BUFFER_SCOPE(state->trace_buffer);
  IREE_AMDGPU_TRACE_ZONE_BEGIN(z0);
  IREE_AMDGPU_TRACE_ZONE_APPEND_VALUE_I64(z0, length);

  // Reserve and begin populating the operation packet.
  // When tracing is enabled capture the timing signal.
  iree_hsa_signal_t completion_signal = iree_hsa_signal_null();
  const uint64_t packet_id = iree_hal_amdgpu_device_buffer_op_reserve(
      state, IREE_HAL_AMDGPU_TRACE_EXECUTION_ZONE_TYPE_FILL,
      &completion_signal);

  // Emplace the dispatch packet into the queue.
  // Note that until the packet is issued the queue will stall.
  iree_hsa_kernel_dispatch_packet_t* IREE_AMDGPU_RESTRICT packet =
      iree_hal_amdgpu_device_buffer_fill_emplace_reserve(
          state, target_ptr, length, pattern, pattern_length, kernarg_ptr,
          packet_id);

  // Issues the buffer operation packet by configuring its header and signaling
  // the queue doorbell.
  iree_hal_amdgpu_device_buffer_op_commit(state, packet_id, packet,
                                          completion_signal);

  IREE_AMDGPU_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_device_buffer_copy_*
//===----------------------------------------------------------------------===//

IREE_AMDGPU_ATTRIBUTE_KERNEL void iree_hal_amdgpu_device_buffer_copy_x1(
    const uint8_t* IREE_AMDGPU_RESTRICT source_ptr,
    uint8_t* IREE_AMDGPU_RESTRICT target_ptr, const uint64_t length) {
  // DO NOT SUBMIT copy kernel
  // runtime/src/iree/hal/drivers/metal/builtin/copy_buffer_generic.metal
  // copy_buffer_1byte
}

IREE_AMDGPU_ATTRIBUTE_KERNEL void iree_hal_amdgpu_device_buffer_copy_x2(
    const uint16_t* IREE_AMDGPU_RESTRICT source_ptr,
    uint16_t* IREE_AMDGPU_RESTRICT target_ptr, const uint64_t length) {
  // DO NOT SUBMIT copy kernel
}

IREE_AMDGPU_ATTRIBUTE_KERNEL void iree_hal_amdgpu_device_buffer_copy_x4(
    const uint16_t* IREE_AMDGPU_RESTRICT source_ptr,
    uint16_t* IREE_AMDGPU_RESTRICT target_ptr, const uint64_t length) {
  // DO NOT SUBMIT copy kernel
}

IREE_AMDGPU_ATTRIBUTE_KERNEL void iree_hal_amdgpu_device_buffer_copy_x8(
    const uint16_t* IREE_AMDGPU_RESTRICT source_ptr,
    uint16_t* IREE_AMDGPU_RESTRICT target_ptr, const uint64_t length) {
  // DO NOT SUBMIT copy kernel
}

// TODO(benvanik): experiment with best widths for bulk transfers.
IREE_AMDGPU_ATTRIBUTE_KERNEL void iree_hal_amdgpu_device_buffer_copy_x64(
    const uint16_t* IREE_AMDGPU_RESTRICT source_ptr,
    uint16_t* IREE_AMDGPU_RESTRICT target_ptr, const uint64_t length) {
  // DO NOT SUBMIT copy kernel
}

// TODO(benvanik): experiment with enqueuing SDMA somehow (may need to take a
// DMA queue as well as the dispatch queue).
iree_hsa_kernel_dispatch_packet_t* IREE_AMDGPU_RESTRICT
iree_hal_amdgpu_device_buffer_copy_emplace_reserve(
    const iree_hal_amdgpu_device_buffer_transfer_state_t* IREE_AMDGPU_RESTRICT
        state,
    const void* source_ptr, void* target_ptr, const uint64_t length,
    uint64_t* IREE_AMDGPU_RESTRICT kernarg_ptr, const uint64_t packet_id) {
  IREE_AMDGPU_TRACE_BUFFER_SCOPE(state->trace_buffer);
  IREE_AMDGPU_TRACE_ZONE_BEGIN(z0);

  // Update kernargs (same for all kernels).
  kernarg_ptr[0] = (uint64_t)source_ptr;
  kernarg_ptr[1] = (uint64_t)target_ptr;
  kernarg_ptr[2] = length;

  // Select the kernel for the copy operation.
  // TODO(benvanik): switch kernel based on source/target/length alignment.
  const iree_hal_amdgpu_device_kernel_args_t kernel_args =
      state->kernels->iree_hal_amdgpu_device_buffer_copy_x1;
  const uint64_t block_size = 128;
  IREE_AMDGPU_TRACE_ZONE_APPEND_TEXT_LITERAL(z0, "copy_x1");
  IREE_AMDGPU_TRACE_ZONE_APPEND_VALUE_I64(z0, block_size);

  // Populate the packet.
  const uint64_t queue_mask = state->queue->size - 1;  // power of two
  iree_hsa_kernel_dispatch_packet_t* IREE_AMDGPU_RESTRICT dispatch_packet =
      state->queue->base_address + (packet_id & queue_mask) * 64;
  dispatch_packet->setup = kernel_args.setup;
  dispatch_packet->workgroup_size[0] = kernel_args.workgroup_size[0];
  dispatch_packet->workgroup_size[1] = kernel_args.workgroup_size[1];
  dispatch_packet->workgroup_size[2] = kernel_args.workgroup_size[2];
  dispatch_packet->reserved0 = 0;
  dispatch_packet->grid_size[0] = 0;  // DO NOT SUBMIT block count?
  dispatch_packet->grid_size[1] = 1;
  dispatch_packet->grid_size[2] = 1;
  dispatch_packet->private_segment_size = kernel_args.private_segment_size;
  dispatch_packet->group_segment_size = kernel_args.group_segment_size;
  dispatch_packet->kernel_object = kernel_args.kernel_object;
  dispatch_packet->kernarg_address = kernarg_ptr;
  dispatch_packet->reserved2 = 0;

  IREE_AMDGPU_TRACE_ZONE_END(z0);
  return dispatch_packet;
}

void iree_hal_amdgpu_device_buffer_copy_enqueue(
    const iree_hal_amdgpu_device_buffer_transfer_state_t* IREE_AMDGPU_RESTRICT
        state,
    const void* source_ptr, void* target_ptr, const uint64_t length,
    uint64_t* IREE_AMDGPU_RESTRICT kernarg_ptr) {
  IREE_AMDGPU_TRACE_BUFFER_SCOPE(state->trace_buffer);
  IREE_AMDGPU_TRACE_ZONE_BEGIN(z0);
  IREE_AMDGPU_TRACE_ZONE_APPEND_VALUE_I64(z0, length);

  // Reserve and begin populating the operation packet.
  // When tracing is enabled capture the timing signal.
  iree_hsa_signal_t completion_signal = iree_hsa_signal_null();
  const uint64_t packet_id = iree_hal_amdgpu_device_buffer_op_reserve(
      state, IREE_HAL_AMDGPU_TRACE_EXECUTION_ZONE_TYPE_COPY,
      &completion_signal);

  // Emplace the dispatch packet into the queue.
  // Note that until the packet is issued the queue will stall.
  iree_hsa_kernel_dispatch_packet_t* IREE_AMDGPU_RESTRICT packet =
      iree_hal_amdgpu_device_buffer_copy_emplace_reserve(
          state, source_ptr, target_ptr, length, kernarg_ptr, packet_id);

  // Issues the buffer operation packet by configuring its header and signaling
  // the queue doorbell.
  iree_hal_amdgpu_device_buffer_op_commit(state, packet_id, packet,
                                          completion_signal);

  IREE_AMDGPU_TRACE_ZONE_END(z0);
}
