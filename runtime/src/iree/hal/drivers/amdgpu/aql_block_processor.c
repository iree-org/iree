// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/aql_block_processor.h"

#include <string.h>

#include "iree/hal/drivers/amdgpu/buffer.h"
#include "iree/hal/drivers/amdgpu/device/dispatch.h"
#include "iree/hal/drivers/amdgpu/util/aql_emitter.h"

typedef uint32_t iree_hal_amdgpu_aql_block_processor_packet_flags_t;
enum iree_hal_amdgpu_aql_block_processor_packet_flag_bits_t {
  IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_PACKET_FLAG_NONE = 0u,
  // Packet must participate in the command-buffer execution dependency chain.
  IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_PACKET_FLAG_EXECUTION_BARRIER = 1u << 0,
  // Packet carries terminal signal release scope for the block submission.
  IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_PACKET_FLAG_FINAL = 1u << 1,
  // First bit of the two-bit acquire fence scope field in packet flags.
  IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_PACKET_ACQUIRE_SCOPE_SHIFT = 2,
  // Bit mask of the two-bit acquire fence scope field in packet flags.
  IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_PACKET_ACQUIRE_SCOPE_MASK = 0x0Cu,
  // First bit of the two-bit release fence scope field in packet flags.
  IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_PACKET_RELEASE_SCOPE_SHIFT = 4,
  // Bit mask of the two-bit release fence scope field in packet flags.
  IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_PACKET_RELEASE_SCOPE_MASK = 0x30u,
};

// Command flags split across multi-packet replay implementations.
typedef struct iree_hal_amdgpu_aql_block_processor_packet_flag_pair_t {
  // Flags applied to the first AQL packet implementing a recorded command.
  iree_hal_amdgpu_aql_block_processor_packet_flags_t first;
  // Flags applied to the final AQL packet implementing a recorded command.
  iree_hal_amdgpu_aql_block_processor_packet_flags_t final;
} iree_hal_amdgpu_aql_block_processor_packet_flag_pair_t;

typedef struct iree_hal_amdgpu_aql_block_processor_state_t {
  // Packet cursors advanced while invoking the block.
  struct {
    // Number of recorded block AQL packets consumed from the block.
    uint32_t recorded;
    // Number of payload AQL packets emitted into the reserved packet span.
    uint32_t emitted;
  } packets;
  // Queue-owned kernarg cursors advanced while invoking the block.
  struct {
    // Number of queue-owned kernarg blocks consumed from the reserved span.
    uint32_t block;
  } kernargs;
} iree_hal_amdgpu_aql_block_processor_state_t;

static iree_hsa_fence_scope_t iree_hal_amdgpu_aql_block_processor_max_scope(
    iree_hsa_fence_scope_t lhs, iree_hsa_fence_scope_t rhs) {
  return lhs > rhs ? lhs : rhs;
}

static iree_hal_amdgpu_aql_block_processor_packet_flags_t
iree_hal_amdgpu_aql_block_processor_packet_flags_set_fence_scopes(
    iree_hal_amdgpu_aql_block_processor_packet_flags_t flags,
    iree_hsa_fence_scope_t acquire_scope,
    iree_hsa_fence_scope_t release_scope) {
  flags &= ~(IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_PACKET_ACQUIRE_SCOPE_MASK |
             IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_PACKET_RELEASE_SCOPE_MASK);
  flags |= ((uint32_t)acquire_scope & 0x3u)
           << IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_PACKET_ACQUIRE_SCOPE_SHIFT;
  flags |= ((uint32_t)release_scope & 0x3u)
           << IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_PACKET_RELEASE_SCOPE_SHIFT;
  return flags;
}

static iree_hsa_fence_scope_t
iree_hal_amdgpu_aql_block_processor_packet_flags_fence_scope(
    iree_hal_amdgpu_aql_block_processor_packet_flags_t flags, uint32_t mask,
    uint32_t shift) {
  return (iree_hsa_fence_scope_t)((flags & mask) >> shift);
}

static iree_hsa_fence_scope_t
iree_hal_amdgpu_aql_block_processor_packet_flags_acquire_scope(
    iree_hal_amdgpu_aql_block_processor_packet_flags_t flags) {
  return iree_hal_amdgpu_aql_block_processor_packet_flags_fence_scope(
      flags, IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_PACKET_ACQUIRE_SCOPE_MASK,
      IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_PACKET_ACQUIRE_SCOPE_SHIFT);
}

static iree_hsa_fence_scope_t
iree_hal_amdgpu_aql_block_processor_packet_flags_release_scope(
    iree_hal_amdgpu_aql_block_processor_packet_flags_t flags) {
  return iree_hal_amdgpu_aql_block_processor_packet_flags_fence_scope(
      flags, IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_PACKET_RELEASE_SCOPE_MASK,
      IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_PACKET_RELEASE_SCOPE_SHIFT);
}

static bool iree_hal_amdgpu_aql_block_processor_packet_has_barrier(
    const iree_hal_amdgpu_aql_block_processor_t* processor,
    uint32_t packet_index,
    iree_hal_amdgpu_aql_block_processor_packet_flags_t packet_flags) {
  return iree_any_bit_set(
             packet_flags,
             IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_PACKET_FLAG_EXECUTION_BARRIER) ||
         iree_any_bit_set(
             packet_flags,
             IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_PACKET_FLAG_FINAL) ||
         (packet_index == 0 && processor->submission.wait_barrier_count > 0) ||
         (packet_index == 0 && processor->submission.inline_acquire_scope !=
                                   IREE_HSA_FENCE_SCOPE_NONE);
}

static iree_hal_amdgpu_aql_packet_control_t
iree_hal_amdgpu_aql_block_processor_packet_control(
    const iree_hal_amdgpu_aql_block_processor_t* processor,
    uint32_t packet_index, iree_hsa_fence_scope_t minimum_acquire_scope,
    iree_hal_amdgpu_aql_block_processor_packet_flags_t packet_flags) {
  // The replay hot path is intentionally additive: command recording has
  // already encoded execution-barrier scopes in |packet_flags|, and submission
  // policy only overlays wait, queue-kernarg, and terminal-signal visibility.
  // Do not infer memory hazards here by walking command operands.
  const uint32_t logical_packet_index =
      processor->packets.index_base + packet_index;
  const bool has_barrier =
      iree_hal_amdgpu_aql_block_processor_packet_has_barrier(
          processor, logical_packet_index, packet_flags);
  const iree_hsa_fence_scope_t execution_acquire_scope =
      iree_hal_amdgpu_aql_block_processor_packet_flags_acquire_scope(
          packet_flags);
  const iree_hsa_fence_scope_t execution_release_scope =
      iree_hal_amdgpu_aql_block_processor_packet_flags_release_scope(
          packet_flags);
  const iree_hsa_fence_scope_t acquire_scope =
      logical_packet_index == 0
          ? iree_hal_amdgpu_aql_block_processor_max_scope(
                execution_acquire_scope,
                processor->submission.inline_acquire_scope)
          : execution_acquire_scope;
  const iree_hsa_fence_scope_t effective_acquire_scope =
      iree_hal_amdgpu_aql_block_processor_max_scope(acquire_scope,
                                                    minimum_acquire_scope);
  iree_hsa_fence_scope_t release_scope = execution_release_scope;
  if (iree_any_bit_set(packet_flags,
                       IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_PACKET_FLAG_FINAL)) {
    release_scope = iree_hal_amdgpu_aql_block_processor_max_scope(
        release_scope, processor->submission.signal_release_scope);
  }
  return iree_hal_amdgpu_aql_packet_control(
      has_barrier, effective_acquire_scope, release_scope);
}

static iree_hal_amdgpu_aql_block_processor_packet_flags_t
iree_hal_amdgpu_aql_block_processor_command_packet_flags(
    const iree_hal_amdgpu_command_buffer_command_header_t* command) {
  iree_hal_amdgpu_aql_block_processor_packet_flags_t packet_flags =
      IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_PACKET_FLAG_NONE;
  if (iree_any_bit_set(
          command->flags,
          IREE_HAL_AMDGPU_COMMAND_BUFFER_COMMAND_FLAG_HAS_BARRIER)) {
    packet_flags |=
        IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_PACKET_FLAG_EXECUTION_BARRIER;
  }
  return iree_hal_amdgpu_aql_block_processor_packet_flags_set_fence_scopes(
      packet_flags,
      (iree_hsa_fence_scope_t)
          iree_hal_amdgpu_command_buffer_command_flags_acquire_scope(
              command->flags),
      (iree_hsa_fence_scope_t)
          iree_hal_amdgpu_command_buffer_command_flags_release_scope(
              command->flags));
}

static iree_hal_amdgpu_aql_block_processor_packet_flags_t
iree_hal_amdgpu_aql_block_processor_packet_flags_merge(
    iree_hal_amdgpu_aql_block_processor_packet_flags_t lhs,
    iree_hal_amdgpu_aql_block_processor_packet_flags_t rhs) {
  const iree_hsa_fence_scope_t acquire_scope =
      iree_hal_amdgpu_aql_block_processor_max_scope(
          iree_hal_amdgpu_aql_block_processor_packet_flags_acquire_scope(lhs),
          iree_hal_amdgpu_aql_block_processor_packet_flags_acquire_scope(rhs));
  const iree_hsa_fence_scope_t release_scope =
      iree_hal_amdgpu_aql_block_processor_max_scope(
          iree_hal_amdgpu_aql_block_processor_packet_flags_release_scope(lhs),
          iree_hal_amdgpu_aql_block_processor_packet_flags_release_scope(rhs));
  return iree_hal_amdgpu_aql_block_processor_packet_flags_set_fence_scopes(
      lhs | rhs, acquire_scope, release_scope);
}

static iree_hal_amdgpu_aql_block_processor_packet_flags_t
iree_hal_amdgpu_aql_block_processor_execution_barrier_packet_flags(void) {
  return iree_hal_amdgpu_aql_block_processor_packet_flags_set_fence_scopes(
      IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_PACKET_FLAG_EXECUTION_BARRIER,
      IREE_HSA_FENCE_SCOPE_NONE, IREE_HSA_FENCE_SCOPE_NONE);
}

static iree_hal_amdgpu_aql_block_processor_packet_flag_pair_t
iree_hal_amdgpu_aql_block_processor_split_command_packet_flags(
    const iree_hal_amdgpu_command_buffer_command_header_t* command) {
  const iree_hal_amdgpu_aql_block_processor_packet_flags_t command_flags =
      iree_hal_amdgpu_aql_block_processor_command_packet_flags(command);
  const iree_hsa_fence_scope_t acquire_scope =
      iree_hal_amdgpu_aql_block_processor_packet_flags_acquire_scope(
          command_flags);
  const iree_hsa_fence_scope_t release_scope =
      iree_hal_amdgpu_aql_block_processor_packet_flags_release_scope(
          command_flags);
  return (iree_hal_amdgpu_aql_block_processor_packet_flag_pair_t){
      .first =
          iree_hal_amdgpu_aql_block_processor_packet_flags_set_fence_scopes(
              command_flags, acquire_scope, IREE_HSA_FENCE_SCOPE_NONE),
      .final = iree_hal_amdgpu_aql_block_processor_packet_flags_set_fence_scopes(
          command_flags &
              ~IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_PACKET_FLAG_EXECUTION_BARRIER,
          IREE_HSA_FENCE_SCOPE_NONE, release_scope),
  };
}

static bool iree_hal_amdgpu_aql_block_processor_command_uses_queue_kernargs(
    const iree_hal_amdgpu_command_buffer_command_header_t* command) {
  return iree_any_bit_set(
      command->flags,
      IREE_HAL_AMDGPU_COMMAND_BUFFER_COMMAND_FLAG_USES_QUEUE_KERNARGS);
}

static bool iree_hal_amdgpu_aql_block_processor_dispatch_uses_indirect(
    const iree_hal_amdgpu_command_buffer_dispatch_command_t* dispatch_command) {
  return iree_any_bit_set(
      dispatch_command->dispatch_flags,
      IREE_HAL_AMDGPU_COMMAND_BUFFER_DISPATCH_FLAG_INDIRECT_PARAMETERS);
}

static bool iree_hal_amdgpu_aql_block_processor_dispatch_uses_prepublished(
    const iree_hal_amdgpu_command_buffer_dispatch_command_t* dispatch_command) {
  return dispatch_command->kernarg_strategy ==
         IREE_HAL_AMDGPU_COMMAND_BUFFER_KERNARG_STRATEGY_PREPUBLISHED;
}

static bool iree_hal_amdgpu_aql_block_processor_dispatch_uses_queue_kernargs(
    const iree_hal_amdgpu_command_buffer_dispatch_command_t* dispatch_command) {
  return !iree_hal_amdgpu_aql_block_processor_dispatch_uses_prepublished(
      dispatch_command);
}

static uint32_t
iree_hal_amdgpu_aql_block_processor_dispatch_target_kernarg_block_count(
    const iree_hal_amdgpu_command_buffer_dispatch_command_t* dispatch_command) {
  if (iree_hal_amdgpu_aql_block_processor_dispatch_uses_prepublished(
          dispatch_command)) {
    return 0;
  }
  const uint32_t kernarg_length =
      (uint32_t)dispatch_command->kernarg_length_qwords * 8u;
  return iree_max(1u,
                  (uint32_t)iree_host_size_ceil_div(
                      kernarg_length, sizeof(iree_hal_amdgpu_kernarg_block_t)));
}

static uint32_t
iree_hal_amdgpu_aql_block_processor_dispatch_kernarg_block_count(
    const iree_hal_amdgpu_command_buffer_dispatch_command_t* dispatch_command) {
  return iree_hal_amdgpu_aql_block_processor_dispatch_target_kernarg_block_count(
             dispatch_command) +
         (iree_hal_amdgpu_aql_block_processor_dispatch_uses_indirect(
              dispatch_command)
              ? 1u
              : 0u);
}

static iree_status_t iree_hal_amdgpu_aql_block_processor_resolve_buffer_ref_ptr(
    iree_hal_buffer_ref_t buffer_ref, iree_hal_buffer_usage_t required_usage,
    iree_hal_memory_access_t required_access, uint8_t** out_device_ptr) {
  *out_device_ptr = NULL;
  if (IREE_UNLIKELY(!buffer_ref.buffer)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AQL command-buffer dynamic binding resolved to a NULL buffer");
  }
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_usage(
      iree_hal_buffer_allowed_usage(buffer_ref.buffer), required_usage));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_access(
      iree_hal_buffer_allowed_access(buffer_ref.buffer), required_access));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_range(
      buffer_ref.buffer, buffer_ref.offset, buffer_ref.length));
  iree_hal_buffer_t* allocated_buffer =
      iree_hal_buffer_allocated_buffer(buffer_ref.buffer);
  uint8_t* device_ptr =
      (uint8_t*)iree_hal_amdgpu_buffer_device_pointer(allocated_buffer);
  if (IREE_UNLIKELY(!device_ptr)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AQL command-buffer buffer must be backed by an AMDGPU allocation");
  }
  iree_device_size_t device_offset = 0;
  if (IREE_UNLIKELY(!iree_device_size_checked_add(
          iree_hal_buffer_byte_offset(buffer_ref.buffer), buffer_ref.offset,
          &device_offset))) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AQL command-buffer buffer device pointer offset overflows device "
        "size");
  }
  if (IREE_UNLIKELY(device_offset > UINTPTR_MAX)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AQL command-buffer buffer device pointer offset exceeds host pointer "
        "size");
  }
  *out_device_ptr = device_ptr + (uintptr_t)device_offset;
  return iree_ok_status();
}

static iree_status_t
iree_hal_amdgpu_aql_block_processor_resolve_command_buffer_ref(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_amdgpu_command_buffer_binding_kind_t kind, uint32_t ordinal,
    uint64_t offset, uint64_t length, iree_hal_buffer_usage_t required_usage,
    iree_hal_memory_access_t required_access,
    iree_hal_buffer_ref_t* out_buffer_ref, uint8_t** out_device_ptr) {
  memset(out_buffer_ref, 0, sizeof(*out_buffer_ref));
  *out_device_ptr = NULL;
  if (kind == IREE_HAL_AMDGPU_COMMAND_BUFFER_BINDING_KIND_STATIC) {
    iree_hal_buffer_t* buffer =
        iree_hal_amdgpu_aql_command_buffer_static_buffer(command_buffer,
                                                         ordinal);
    if (IREE_UNLIKELY(!buffer)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AQL command-buffer static buffer ordinal %" PRIu32 " is invalid",
          ordinal);
    }
    *out_buffer_ref = iree_hal_make_buffer_ref(buffer, offset, length);
  } else if (kind == IREE_HAL_AMDGPU_COMMAND_BUFFER_BINDING_KIND_DYNAMIC) {
    iree_hal_buffer_ref_t dynamic_ref =
        iree_hal_make_indirect_buffer_ref(ordinal, offset, length);
    IREE_RETURN_IF_ERROR(iree_hal_buffer_binding_table_resolve_ref(
        binding_table, dynamic_ref, out_buffer_ref));
  } else {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AQL command-buffer binding kind %u is invalid",
                            kind);
  }
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_aql_block_processor_resolve_buffer_ref_ptr(
          *out_buffer_ref, required_usage, required_access, out_device_ptr));
  return iree_ok_status();
}

static iree_status_t
iree_hal_amdgpu_aql_block_processor_resolve_static_binding_source_ptr(
    iree_hal_command_buffer_t* command_buffer,
    const iree_hal_amdgpu_command_buffer_binding_source_t* binding_source,
    uint64_t* out_binding_ptr) {
  *out_binding_ptr = 0;
  iree_hal_buffer_t* buffer = iree_hal_amdgpu_aql_command_buffer_static_buffer(
      command_buffer, binding_source->slot);
  if (IREE_UNLIKELY(!buffer)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AQL command-buffer static dispatch binding ordinal %" PRIu32
        " is invalid",
        binding_source->slot);
  }
  iree_hal_buffer_t* allocated_buffer =
      iree_hal_buffer_allocated_buffer(buffer);
  void* device_ptr = iree_hal_amdgpu_buffer_device_pointer(allocated_buffer);
  if (IREE_UNLIKELY(!device_ptr)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AQL command-buffer static dispatch binding has no staged AMDGPU "
        "backing after queue waits completed");
  }
  iree_device_size_t device_offset = 0;
  if (IREE_UNLIKELY(!iree_device_size_checked_add(
          iree_hal_buffer_byte_offset(buffer),
          binding_source->offset_or_pointer, &device_offset))) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AQL command-buffer static dispatch binding pointer offset overflows "
        "device size");
  }
  if (IREE_UNLIKELY(device_offset > UINTPTR_MAX)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AQL command-buffer static dispatch binding pointer offset exceeds "
        "host pointer size");
  }
  *out_binding_ptr =
      (uint64_t)((uintptr_t)device_ptr + (uintptr_t)device_offset);
  return iree_ok_status();
}

static void iree_hal_amdgpu_aql_block_processor_write_dispatch_packet_body(
    const iree_hal_amdgpu_command_buffer_dispatch_command_t* dispatch_command,
    iree_hal_amdgpu_aql_packet_t* packet, uint8_t* kernarg_data,
    iree_hsa_signal_t completion_signal, uint16_t* out_setup) {
  packet->dispatch.setup = dispatch_command->setup;
  packet->dispatch.workgroup_size[0] = dispatch_command->workgroup_size[0];
  packet->dispatch.workgroup_size[1] = dispatch_command->workgroup_size[1];
  packet->dispatch.workgroup_size[2] = dispatch_command->workgroup_size[2];
  packet->dispatch.reserved0 = 0;
  packet->dispatch.grid_size[0] = dispatch_command->grid_size[0];
  packet->dispatch.grid_size[1] = dispatch_command->grid_size[1];
  packet->dispatch.grid_size[2] = dispatch_command->grid_size[2];
  packet->dispatch.private_segment_size =
      dispatch_command->private_segment_size;
  packet->dispatch.group_segment_size = dispatch_command->group_segment_size;
  packet->dispatch.kernel_object = dispatch_command->kernel_object;
  packet->dispatch.kernarg_address = kernarg_data;
  packet->dispatch.reserved2 = 0;
  packet->dispatch.completion_signal = completion_signal;
  *out_setup = packet->dispatch.setup;
}

static inline void iree_hal_amdgpu_aql_block_processor_copy_dispatch_tail(
    const iree_hal_amdgpu_command_buffer_dispatch_command_t* dispatch_command,
    uint8_t* kernarg_data, iree_host_size_t tail_offset) {
  const iree_host_size_t tail_length =
      (iree_host_size_t)dispatch_command->payload.tail_length_qwords * 8u;
  if (tail_length > 0) {
    const uint8_t* tail_payload =
        (const uint8_t*)dispatch_command + dispatch_command->payload_reference;
    memcpy(kernarg_data + tail_offset, tail_payload, tail_length);
  }
}

static iree_status_t
iree_hal_amdgpu_aql_block_processor_replay_dispatch_kernargs(
    const iree_hal_amdgpu_command_buffer_block_header_t* block,
    iree_hal_command_buffer_t* command_buffer, const uint64_t* binding_ptrs,
    const iree_hal_amdgpu_command_buffer_dispatch_command_t* dispatch_command,
    uint8_t* kernarg_data) {
  switch (dispatch_command->kernarg_strategy) {
    case IREE_HAL_AMDGPU_COMMAND_BUFFER_KERNARG_STRATEGY_DYNAMIC_BINDINGS: {
      if (IREE_UNLIKELY(!binding_ptrs)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "AQL command-buffer dispatch has dynamic bindings but no binding "
            "table was provided");
      }
      uint64_t* binding_dst = (uint64_t*)kernarg_data;
      const iree_hal_amdgpu_command_buffer_binding_source_t* binding_sources =
          (const iree_hal_amdgpu_command_buffer_binding_source_t*)((const uint8_t*)
                                                                       block +
                                                                   dispatch_command
                                                                       ->binding_source_offset);
      for (uint16_t i = 0; i < dispatch_command->binding_count; ++i) {
        const iree_hal_amdgpu_command_buffer_binding_source_t* binding_source =
            &binding_sources[i];
        binding_dst[i] = binding_ptrs[binding_source->slot] +
                         binding_source->offset_or_pointer;
      }
      iree_hal_amdgpu_aql_block_processor_copy_dispatch_tail(
          dispatch_command, kernarg_data,
          (iree_host_size_t)dispatch_command->binding_count * sizeof(uint64_t));
      return iree_ok_status();
    }
    case IREE_HAL_AMDGPU_COMMAND_BUFFER_KERNARG_STRATEGY_HAL: {
      uint64_t* binding_dst = (uint64_t*)kernarg_data;
      const iree_hal_amdgpu_command_buffer_binding_source_t* binding_sources =
          (const iree_hal_amdgpu_command_buffer_binding_source_t*)((const uint8_t*)
                                                                       block +
                                                                   dispatch_command
                                                                       ->binding_source_offset);
      for (uint16_t i = 0; i < dispatch_command->binding_count; ++i) {
        const iree_hal_amdgpu_command_buffer_binding_source_t* binding_source =
            &binding_sources[i];
        const uint32_t flags = binding_source->flags;
        if (IREE_LIKELY(
                flags ==
                IREE_HAL_AMDGPU_COMMAND_BUFFER_BINDING_SOURCE_FLAG_NONE)) {
          binding_dst[i] = binding_source->offset_or_pointer;
        } else if (flags ==
                   IREE_HAL_AMDGPU_COMMAND_BUFFER_BINDING_SOURCE_FLAG_DYNAMIC) {
          if (IREE_UNLIKELY(!binding_ptrs)) {
            return iree_make_status(
                IREE_STATUS_INVALID_ARGUMENT,
                "AQL command-buffer dispatch has dynamic bindings but no "
                "binding table was provided");
          }
          binding_dst[i] = binding_ptrs[binding_source->slot] +
                           binding_source->offset_or_pointer;
        } else if (
            flags ==
            IREE_HAL_AMDGPU_COMMAND_BUFFER_BINDING_SOURCE_FLAG_STATIC_BUFFER) {
          IREE_RETURN_IF_ERROR(
              iree_hal_amdgpu_aql_block_processor_resolve_static_binding_source_ptr(
                  command_buffer, binding_source, &binding_dst[i]));
        } else {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "malformed AQL command-buffer dispatch binding source flags %u",
              binding_source->flags);
        }
      }
      iree_hal_amdgpu_aql_block_processor_copy_dispatch_tail(
          dispatch_command, kernarg_data,
          (iree_host_size_t)dispatch_command->binding_count * sizeof(uint64_t));
      return iree_ok_status();
    }
    case IREE_HAL_AMDGPU_COMMAND_BUFFER_KERNARG_STRATEGY_CUSTOM_DIRECT: {
      iree_hal_amdgpu_aql_block_processor_copy_dispatch_tail(
          dispatch_command, kernarg_data, /*tail_offset=*/0);
      return iree_ok_status();
    }
    case IREE_HAL_AMDGPU_COMMAND_BUFFER_KERNARG_STRATEGY_INDIRECT:
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "indirect dispatch arguments are not supported by AMDGPU command "
          "buffers yet");
    case IREE_HAL_AMDGPU_COMMAND_BUFFER_KERNARG_STRATEGY_PATCHED_TEMPLATE: {
      const uint32_t kernarg_length =
          (uint32_t)dispatch_command->kernarg_length_qwords * 8u;
      const uint8_t* kernarg_template =
          iree_hal_amdgpu_aql_command_buffer_rodata(
              command_buffer, dispatch_command->payload_reference,
              kernarg_length);
      if (IREE_UNLIKELY(!kernarg_template)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "AQL command-buffer patched kernarg template range is invalid");
      }
      memcpy(kernarg_data, kernarg_template, kernarg_length);
      uint64_t* binding_dst = (uint64_t*)kernarg_data;
      const uint16_t patch_source_count =
          dispatch_command->payload.patch_source_count;
      if (IREE_UNLIKELY(!binding_ptrs)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "AQL command-buffer dispatch has dynamic bindings but no binding "
            "table was provided");
      }
      const iree_hal_amdgpu_command_buffer_binding_source_t* binding_sources =
          (const iree_hal_amdgpu_command_buffer_binding_source_t*)((const uint8_t*)
                                                                       block +
                                                                   dispatch_command
                                                                       ->binding_source_offset);
      for (uint16_t i = 0; i < patch_source_count; ++i) {
        const iree_hal_amdgpu_command_buffer_binding_source_t* binding_source =
            &binding_sources[i];
        binding_dst[binding_source->target_binding_ordinal] =
            binding_ptrs[binding_source->slot] +
            binding_source->offset_or_pointer;
      }
      return iree_ok_status();
    }
    case IREE_HAL_AMDGPU_COMMAND_BUFFER_KERNARG_STRATEGY_PREPUBLISHED:
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "prepublished command-buffer dispatch should not rewrite kernargs");
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "malformed AQL command-buffer kernarg strategy "
                              "%u",
                              dispatch_command->kernarg_strategy);
  }
}

static iree_status_t
iree_hal_amdgpu_aql_block_processor_replay_dispatch_indirect_params_ptr(
    const iree_hal_amdgpu_aql_block_processor_t* processor,
    const iree_hal_amdgpu_command_buffer_binding_source_t* binding_source,
    const uint32_t** out_workgroup_count_ptr) {
  *out_workgroup_count_ptr = NULL;
  switch (binding_source->flags) {
    case IREE_HAL_AMDGPU_COMMAND_BUFFER_BINDING_SOURCE_FLAG_INDIRECT_PARAMETERS:
      *out_workgroup_count_ptr =
          (const uint32_t*)(uintptr_t)binding_source->offset_or_pointer;
      return iree_ok_status();
    case IREE_HAL_AMDGPU_COMMAND_BUFFER_BINDING_SOURCE_FLAG_DYNAMIC |
        IREE_HAL_AMDGPU_COMMAND_BUFFER_BINDING_SOURCE_FLAG_INDIRECT_PARAMETERS: {
      iree_hal_buffer_ref_t resolved_ref = {0};
      IREE_RETURN_IF_ERROR(iree_hal_buffer_binding_table_resolve_ref(
          processor->bindings.table,
          iree_hal_make_indirect_buffer_ref(binding_source->slot,
                                            binding_source->offset_or_pointer,
                                            sizeof(uint32_t[3])),
          &resolved_ref));
      uint8_t* device_ptr = NULL;
      IREE_RETURN_IF_ERROR(
          iree_hal_amdgpu_aql_block_processor_resolve_buffer_ref_ptr(
              resolved_ref, IREE_HAL_BUFFER_USAGE_DISPATCH_INDIRECT_PARAMETERS,
              IREE_HAL_MEMORY_ACCESS_READ, &device_ptr));
      *out_workgroup_count_ptr = (const uint32_t*)device_ptr;
      return iree_ok_status();
    }
    case IREE_HAL_AMDGPU_COMMAND_BUFFER_BINDING_SOURCE_FLAG_STATIC_BUFFER |
        IREE_HAL_AMDGPU_COMMAND_BUFFER_BINDING_SOURCE_FLAG_INDIRECT_PARAMETERS: {
      uint64_t workgroup_count_ptr = 0;
      IREE_RETURN_IF_ERROR(
          iree_hal_amdgpu_aql_block_processor_resolve_static_binding_source_ptr(
              processor->command_buffer, binding_source, &workgroup_count_ptr));
      *out_workgroup_count_ptr =
          (const uint32_t*)(uintptr_t)workgroup_count_ptr;
      return iree_ok_status();
    }
    default:
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "malformed AQL command-buffer indirect parameter source flags %u",
          binding_source->flags);
  }
}

static iree_amdgpu_kernel_implicit_args_t*
iree_hal_amdgpu_aql_block_processor_dispatch_implicit_args_ptr(
    const iree_hal_amdgpu_command_buffer_dispatch_command_t* dispatch_command,
    uint8_t* kernarg_data) {
  if (dispatch_command->implicit_args_offset_qwords == UINT16_MAX) {
    return NULL;
  }
  return (
      iree_amdgpu_kernel_implicit_args_t*)(kernarg_data +
                                           (iree_host_size_t)dispatch_command
                                                   ->implicit_args_offset_qwords *
                                               8u);
}

static iree_status_t
iree_hal_amdgpu_aql_block_processor_replay_dispatch_packet_body(
    const iree_hal_amdgpu_aql_block_processor_t* processor,
    const iree_hal_amdgpu_command_buffer_block_header_t* block,
    const iree_hal_amdgpu_command_buffer_dispatch_command_t* dispatch_command,
    iree_hal_amdgpu_aql_packet_t* packet, uint8_t* kernarg_data,
    iree_hsa_signal_t completion_signal, uint16_t* out_setup) {
  if (iree_hal_amdgpu_aql_block_processor_dispatch_uses_prepublished(
          dispatch_command)) {
    const uint32_t kernarg_length =
        (uint32_t)dispatch_command->kernarg_length_qwords * 8u;
    kernarg_data = iree_hal_amdgpu_aql_command_buffer_prepublished_kernarg(
        processor->command_buffer, dispatch_command->payload_reference,
        kernarg_length);
    if (IREE_UNLIKELY(!kernarg_data)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AQL command-buffer prepublished kernarg range is invalid");
    }
  } else {
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_aql_block_processor_replay_dispatch_kernargs(
            block, processor->command_buffer, processor->bindings.ptrs,
            dispatch_command, kernarg_data));
  }
  iree_hal_amdgpu_aql_block_processor_write_dispatch_packet_body(
      dispatch_command, packet, kernarg_data, completion_signal, out_setup);
  return iree_ok_status();
}

static iree_status_t
iree_hal_amdgpu_aql_block_processor_replay_indirect_dispatch_packet_bodies(
    const iree_hal_amdgpu_aql_block_processor_t* processor,
    const iree_hal_amdgpu_command_buffer_block_header_t* block,
    const iree_hal_amdgpu_command_buffer_dispatch_command_t* dispatch_command,
    iree_hal_amdgpu_aql_packet_t* patch_packet,
    iree_hal_amdgpu_aql_packet_t* dispatch_packet, uint8_t* patch_kernarg_data,
    uint8_t* dispatch_kernarg_data, iree_hsa_signal_t completion_signal,
    uint16_t dispatch_header, uint16_t* out_patch_setup,
    uint16_t* out_dispatch_setup) {
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_aql_block_processor_replay_dispatch_packet_body(
          processor, block, dispatch_command, dispatch_packet,
          dispatch_kernarg_data, completion_signal, out_dispatch_setup));

  const iree_hal_amdgpu_command_buffer_binding_source_t* binding_sources =
      (const iree_hal_amdgpu_command_buffer_binding_source_t*)((const uint8_t*)
                                                                   block +
                                                               dispatch_command
                                                                   ->binding_source_offset);
  const iree_hal_amdgpu_command_buffer_binding_source_t*
      indirect_params_source =
          &binding_sources[dispatch_command->binding_count];
  const uint32_t* workgroup_count_ptr = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_aql_block_processor_replay_dispatch_indirect_params_ptr(
          processor, indirect_params_source, &workgroup_count_ptr));

  iree_amdgpu_kernel_implicit_args_t* implicit_args =
      iree_hal_amdgpu_aql_block_processor_dispatch_implicit_args_ptr(
          dispatch_command, dispatch_kernarg_data);
  iree_hal_amdgpu_device_dispatch_emplace_indirect_params_patch(
      &processor->transfer_context->kernels
           ->iree_hal_amdgpu_device_dispatch_patch_indirect_params,
      workgroup_count_ptr, &dispatch_packet->dispatch, dispatch_header,
      *out_dispatch_setup, implicit_args, &patch_packet->dispatch,
      patch_kernarg_data);
  *out_patch_setup = patch_packet->dispatch.setup;
  return iree_ok_status();
}

static iree_status_t
iree_hal_amdgpu_aql_block_processor_replay_fill_packet_body(
    const iree_hal_amdgpu_aql_block_processor_t* processor,
    const iree_hal_amdgpu_command_buffer_fill_command_t* fill_command,
    iree_hal_amdgpu_aql_packet_t* packet,
    iree_hal_amdgpu_kernarg_block_t* kernarg_block,
    iree_hsa_signal_t completion_signal, uint16_t* out_setup) {
  iree_hal_buffer_ref_t target_ref = {0};
  uint8_t* target_ptr = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_aql_block_processor_resolve_command_buffer_ref(
          processor->command_buffer, processor->bindings.table,
          fill_command->target_kind, fill_command->target_ordinal,
          fill_command->target_offset, fill_command->length,
          IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET, IREE_HAL_MEMORY_ACCESS_WRITE,
          &target_ref, &target_ptr));
  (void)target_ref;
  if (IREE_UNLIKELY(!iree_hal_amdgpu_device_buffer_fill_emplace(
          processor->transfer_context, &packet->dispatch, target_ptr,
          fill_command->length, fill_command->pattern,
          fill_command->pattern_length, kernarg_block->data))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported command-buffer fill dispatch shape");
  }
  packet->dispatch.completion_signal = completion_signal;
  *out_setup = packet->dispatch.setup;
  return iree_ok_status();
}

static iree_status_t
iree_hal_amdgpu_aql_block_processor_replay_copy_packet_body(
    const iree_hal_amdgpu_aql_block_processor_t* processor,
    const iree_hal_amdgpu_command_buffer_copy_command_t* copy_command,
    iree_hal_amdgpu_aql_packet_t* packet,
    iree_hal_amdgpu_kernarg_block_t* kernarg_block,
    iree_hsa_signal_t completion_signal, uint16_t* out_setup) {
  iree_hal_buffer_ref_t source_ref = {0};
  uint8_t* source_ptr = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_aql_block_processor_resolve_command_buffer_ref(
          processor->command_buffer, processor->bindings.table,
          copy_command->source_kind, copy_command->source_ordinal,
          copy_command->source_offset, copy_command->length,
          IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE, IREE_HAL_MEMORY_ACCESS_READ,
          &source_ref, &source_ptr));
  iree_hal_buffer_ref_t target_ref = {0};
  uint8_t* target_ptr = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_aql_block_processor_resolve_command_buffer_ref(
          processor->command_buffer, processor->bindings.table,
          copy_command->target_kind, copy_command->target_ordinal,
          copy_command->target_offset, copy_command->length,
          IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET, IREE_HAL_MEMORY_ACCESS_WRITE,
          &target_ref, &target_ptr));

  if (IREE_UNLIKELY(
          iree_hal_buffer_test_overlap(source_ref.buffer, source_ref.offset,
                                       source_ref.length, target_ref.buffer,
                                       target_ref.offset, target_ref.length) !=
          IREE_HAL_BUFFER_OVERLAP_DISJOINT)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "source and target ranges must not overlap within the same buffer");
  }
  if (IREE_UNLIKELY(!iree_hal_amdgpu_device_buffer_copy_emplace(
          processor->transfer_context, &packet->dispatch, source_ptr,
          target_ptr, copy_command->length, kernarg_block->data))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported command-buffer copy dispatch shape");
  }
  packet->dispatch.completion_signal = completion_signal;
  *out_setup = packet->dispatch.setup;
  return iree_ok_status();
}

static iree_host_size_t
iree_hal_amdgpu_aql_block_processor_update_kernarg_length(
    uint32_t source_length) {
  const iree_host_size_t source_payload_offset =
      IREE_HAL_AMDGPU_DEVICE_BUFFER_COPY_STAGED_SOURCE_OFFSET;
  return source_payload_offset + (iree_host_size_t)source_length;
}

static uint32_t iree_hal_amdgpu_aql_block_processor_update_kernarg_block_count(
    uint32_t source_length) {
  return (uint32_t)iree_host_size_ceil_div(
      iree_hal_amdgpu_aql_block_processor_update_kernarg_length(source_length),
      sizeof(iree_hal_amdgpu_kernarg_block_t));
}

static iree_status_t
iree_hal_amdgpu_aql_block_processor_resolve_update_packet_operands(
    const iree_hal_amdgpu_aql_block_processor_t* processor,
    const iree_hal_amdgpu_command_buffer_update_command_t* update_command,
    const uint8_t** out_source_bytes, uint8_t** out_target_ptr) {
  *out_source_bytes = NULL;
  *out_target_ptr = NULL;
  iree_hal_buffer_ref_t target_ref = {0};
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_aql_block_processor_resolve_command_buffer_ref(
          processor->command_buffer, processor->bindings.table,
          update_command->target_kind, update_command->target_ordinal,
          update_command->target_offset, update_command->length,
          IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET, IREE_HAL_MEMORY_ACCESS_WRITE,
          &target_ref, out_target_ptr));
  (void)target_ref;
  *out_source_bytes = iree_hal_amdgpu_aql_command_buffer_rodata(
      processor->command_buffer, update_command->rodata_ordinal,
      update_command->length);
  if (IREE_UNLIKELY(!*out_source_bytes)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AQL command-buffer update rodata range is invalid");
  }
  return iree_ok_status();
}

static iree_status_t
iree_hal_amdgpu_aql_block_processor_replay_update_packet_body(
    const iree_hal_amdgpu_aql_block_processor_t* processor,
    const iree_hal_amdgpu_command_buffer_update_command_t* update_command,
    iree_hal_amdgpu_aql_packet_t* packet, uint8_t* kernarg_data,
    iree_host_size_t kernarg_length, iree_hsa_signal_t completion_signal,
    uint16_t* out_setup) {
  const uint8_t* source_bytes = NULL;
  uint8_t* target_ptr = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_aql_block_processor_resolve_update_packet_operands(
          processor, update_command, &source_bytes, &target_ptr));

  const iree_host_size_t source_payload_offset =
      IREE_HAL_AMDGPU_DEVICE_BUFFER_COPY_STAGED_SOURCE_OFFSET;
  const iree_host_size_t required_kernarg_length =
      source_payload_offset + (iree_host_size_t)update_command->length;
  if (IREE_UNLIKELY(required_kernarg_length > kernarg_length)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AQL command-buffer update kernarg range is too small");
  }

  iree_hal_amdgpu_device_buffer_copy_kernargs_t kernargs;
  memset(&kernargs, 0, sizeof(kernargs));
  if (IREE_UNLIKELY(!iree_hal_amdgpu_device_buffer_copy_emplace(
          processor->transfer_context, &packet->dispatch,
          (const void*)(uintptr_t)
              IREE_HAL_AMDGPU_DEVICE_BUFFER_COPY_STAGED_SOURCE_ALIGNMENT,
          target_ptr, update_command->length, &kernargs))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported command-buffer update dispatch shape");
  }

  uint8_t* staged_source_bytes = kernarg_data + source_payload_offset;
  memcpy(kernarg_data, &kernargs, sizeof(kernargs));
  ((iree_hal_amdgpu_device_buffer_copy_kernargs_t*)kernarg_data)->source_ptr =
      staged_source_bytes;
  memcpy(staged_source_bytes, source_bytes, update_command->length);
  packet->dispatch.kernarg_address = kernarg_data;
  packet->dispatch.completion_signal = completion_signal;
  *out_setup = packet->dispatch.setup;
  return iree_ok_status();
}

static uint32_t
iree_hal_amdgpu_aql_block_processor_payload_acquire_packet_count(
    const iree_hal_amdgpu_aql_block_processor_t* processor,
    const iree_hal_amdgpu_command_buffer_block_header_t* block) {
  if (processor->payload.acquire_scope == IREE_HSA_FENCE_SCOPE_NONE) return 0;
  if (block->aql_packet_count == 0) return 0;
  iree_hal_amdgpu_aql_block_processor_packet_flags_t packet_flags =
      IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_PACKET_FLAG_NONE;
  if (block->aql_packet_count == 1 &&
      iree_all_bits_set(
          processor->flags,
          IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_FLAG_FINAL_PAYLOAD_PACKET)) {
    packet_flags |= IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_PACKET_FLAG_FINAL;
  }
  if (iree_hal_amdgpu_aql_block_processor_packet_has_barrier(
          processor, processor->packets.index_base, packet_flags)) {
    return 1;
  }
  return block->initial_barrier_packet_count;
}

static iree_hsa_fence_scope_t
iree_hal_amdgpu_aql_block_processor_payload_acquire_scope(
    const iree_hal_amdgpu_aql_block_processor_t* processor,
    const iree_hal_amdgpu_aql_block_processor_state_t* state,
    uint32_t payload_acquire_packet_count, uint32_t packet_index,
    const iree_hal_amdgpu_command_buffer_command_header_t* command,
    iree_hal_amdgpu_aql_block_processor_packet_flags_t packet_flags) {
  if (processor->payload.acquire_scope == IREE_HSA_FENCE_SCOPE_NONE) {
    return IREE_HSA_FENCE_SCOPE_NONE;
  }
  if (state->packets.recorded >= payload_acquire_packet_count) {
    return IREE_HSA_FENCE_SCOPE_NONE;
  }
  if (iree_hal_amdgpu_aql_block_processor_command_uses_queue_kernargs(
          command)) {
    return processor->payload.acquire_scope;
  }
  if (iree_hal_amdgpu_aql_block_processor_packet_has_barrier(
          processor, processor->packets.index_base + packet_index,
          packet_flags)) {
    return processor->payload.acquire_scope;
  }
  return IREE_HSA_FENCE_SCOPE_NONE;
}

static iree_hal_amdgpu_aql_packet_t* iree_hal_amdgpu_aql_block_processor_packet(
    const iree_hal_amdgpu_aql_block_processor_t* processor,
    uint32_t packet_index) {
  return iree_hal_amdgpu_aql_ring_packet(
      processor->packets.ring, processor->packets.first_id + packet_index);
}

static bool iree_hal_amdgpu_aql_block_processor_is_block_final(
    const iree_hal_amdgpu_command_buffer_block_header_t* block,
    const iree_hal_amdgpu_aql_block_processor_state_t* state,
    uint32_t recorded_packet_count) {
  return state->packets.recorded + recorded_packet_count ==
         block->aql_packet_count;
}

static iree_status_t iree_hal_amdgpu_aql_block_processor_emit_direct_dispatch(
    const iree_hal_amdgpu_aql_block_processor_t* processor,
    const iree_hal_amdgpu_command_buffer_block_header_t* block,
    uint32_t payload_acquire_packet_count,
    const iree_hal_amdgpu_command_buffer_dispatch_command_t* dispatch_command,
    iree_hal_amdgpu_aql_block_processor_state_t* state) {
  const uint32_t dispatch_packet_index = state->packets.emitted;
  iree_hal_amdgpu_aql_packet_t* packet =
      iree_hal_amdgpu_aql_block_processor_packet(processor,
                                                 dispatch_packet_index);
  uint8_t* kernarg_data = NULL;
  if (iree_hal_amdgpu_aql_block_processor_dispatch_uses_queue_kernargs(
          dispatch_command)) {
    kernarg_data = processor->kernargs.blocks[state->kernargs.block].data;
  }
  iree_status_t status =
      iree_hal_amdgpu_aql_block_processor_replay_dispatch_packet_body(
          processor, block, dispatch_command, packet, kernarg_data,
          iree_hsa_signal_null(),
          &processor->packets.setups[dispatch_packet_index]);
  if (iree_status_is_ok(status)) {
    iree_hal_amdgpu_aql_block_processor_packet_flags_t packet_flags =
        iree_hal_amdgpu_aql_block_processor_command_packet_flags(
            &dispatch_command->header);
    if (iree_hal_amdgpu_aql_block_processor_is_block_final(
            block, state, /*recorded_packet_count=*/1) &&
        iree_all_bits_set(
            processor->flags,
            IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_FLAG_FINAL_PAYLOAD_PACKET)) {
      packet_flags |= IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_PACKET_FLAG_FINAL;
    }
    const iree_hsa_fence_scope_t payload_acquire_scope =
        iree_hal_amdgpu_aql_block_processor_payload_acquire_scope(
            processor, state, payload_acquire_packet_count,
            dispatch_packet_index, &dispatch_command->header, packet_flags);
    processor->packets.headers[dispatch_packet_index] =
        iree_hal_amdgpu_aql_make_header(
            IREE_HSA_PACKET_TYPE_KERNEL_DISPATCH,
            iree_hal_amdgpu_aql_block_processor_packet_control(
                processor, dispatch_packet_index, payload_acquire_scope,
                packet_flags));
    ++state->packets.emitted;
    ++state->packets.recorded;
    state->kernargs.block +=
        iree_hal_amdgpu_aql_block_processor_dispatch_kernarg_block_count(
            dispatch_command);
  }
  return status;
}

static iree_status_t iree_hal_amdgpu_aql_block_processor_emit_indirect_dispatch(
    const iree_hal_amdgpu_aql_block_processor_t* processor,
    const iree_hal_amdgpu_command_buffer_block_header_t* block,
    uint32_t payload_acquire_packet_count,
    const iree_hal_amdgpu_command_buffer_dispatch_command_t* dispatch_command,
    iree_hal_amdgpu_aql_block_processor_state_t* state) {
  const uint32_t patch_packet_index = state->packets.emitted++;
  const uint32_t dispatch_packet_index = state->packets.emitted;
  iree_hal_amdgpu_aql_packet_t* patch_packet =
      iree_hal_amdgpu_aql_block_processor_packet(processor, patch_packet_index);
  iree_hal_amdgpu_aql_packet_t* dispatch_packet =
      iree_hal_amdgpu_aql_block_processor_packet(processor,
                                                 dispatch_packet_index);
  const iree_hal_amdgpu_aql_block_processor_packet_flag_pair_t command_flags =
      iree_hal_amdgpu_aql_block_processor_split_command_packet_flags(
          &dispatch_command->header);
  iree_hal_amdgpu_aql_block_processor_packet_flags_t dispatch_packet_flags =
      command_flags.final;
  if (iree_hal_amdgpu_aql_block_processor_is_block_final(
          block, state, /*recorded_packet_count=*/2) &&
      iree_all_bits_set(
          processor->flags,
          IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_FLAG_FINAL_PAYLOAD_PACKET)) {
    dispatch_packet_flags |=
        IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_PACKET_FLAG_FINAL;
  }
  const uint16_t dispatch_header = iree_hal_amdgpu_aql_make_header(
      IREE_HSA_PACKET_TYPE_KERNEL_DISPATCH,
      iree_hal_amdgpu_aql_block_processor_packet_control(
          processor, dispatch_packet_index, IREE_HSA_FENCE_SCOPE_NONE,
          dispatch_packet_flags));
  iree_status_t status =
      iree_hal_amdgpu_aql_block_processor_replay_indirect_dispatch_packet_bodies(
          processor, block, dispatch_command, patch_packet, dispatch_packet,
          processor->kernargs.blocks[state->kernargs.block].data,
          processor->kernargs.blocks[state->kernargs.block + 1].data,
          iree_hsa_signal_null(), dispatch_header,
          &processor->packets.setups[patch_packet_index],
          &processor->packets.setups[dispatch_packet_index]);
  if (iree_status_is_ok(status)) {
    const iree_hal_amdgpu_aql_block_processor_packet_flags_t patch_flags =
        iree_hal_amdgpu_aql_block_processor_packet_flags_merge(
            iree_hal_amdgpu_aql_block_processor_execution_barrier_packet_flags(),
            command_flags.first);
    const iree_hsa_fence_scope_t patch_acquire_scope =
        iree_hal_amdgpu_aql_block_processor_payload_acquire_scope(
            processor, state, payload_acquire_packet_count, patch_packet_index,
            &dispatch_command->header, patch_flags);
    // The patch dispatch publishes the following dispatch packet header, so it
    // must retire before the CP observes that slot.
    processor->packets.headers[patch_packet_index] =
        iree_hal_amdgpu_aql_make_header(
            IREE_HSA_PACKET_TYPE_KERNEL_DISPATCH,
            iree_hal_amdgpu_aql_block_processor_packet_control(
                processor, patch_packet_index, patch_acquire_scope,
                patch_flags));
    // The patch dispatch publishes the target dispatch header after it has
    // updated dynamic workgroup counts on device.
    processor->packets.headers[dispatch_packet_index] =
        IREE_HSA_PACKET_TYPE_INVALID;
    state->packets.emitted = dispatch_packet_index + /*dispatch packet=*/1;
    state->packets.recorded += 2;
    state->kernargs.block +=
        iree_hal_amdgpu_aql_block_processor_dispatch_kernarg_block_count(
            dispatch_command);
  }
  return status;
}

static iree_status_t iree_hal_amdgpu_aql_block_processor_emit_dispatch(
    const iree_hal_amdgpu_aql_block_processor_t* processor,
    const iree_hal_amdgpu_command_buffer_block_header_t* block,
    uint32_t payload_acquire_packet_count,
    const iree_hal_amdgpu_command_buffer_dispatch_command_t* dispatch_command,
    iree_hal_amdgpu_aql_block_processor_state_t* state) {
  if (iree_hal_amdgpu_aql_block_processor_dispatch_uses_indirect(
          dispatch_command)) {
    return iree_hal_amdgpu_aql_block_processor_emit_indirect_dispatch(
        processor, block, payload_acquire_packet_count, dispatch_command,
        state);
  }
  return iree_hal_amdgpu_aql_block_processor_emit_direct_dispatch(
      processor, block, payload_acquire_packet_count, dispatch_command, state);
}

static iree_status_t iree_hal_amdgpu_aql_block_processor_emit_transfer(
    const iree_hal_amdgpu_aql_block_processor_t* processor,
    const iree_hal_amdgpu_command_buffer_block_header_t* block,
    uint32_t payload_acquire_packet_count,
    const iree_hal_amdgpu_command_buffer_command_header_t* command,
    iree_hal_amdgpu_aql_block_processor_state_t* state) {
  const uint32_t packet_index = state->packets.emitted;
  iree_hal_amdgpu_aql_block_processor_packet_flags_t packet_flags =
      iree_hal_amdgpu_aql_block_processor_command_packet_flags(command);
  if (iree_hal_amdgpu_aql_block_processor_is_block_final(
          block, state, /*recorded_packet_count=*/1) &&
      iree_all_bits_set(
          processor->flags,
          IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_FLAG_FINAL_PAYLOAD_PACKET)) {
    packet_flags |= IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_PACKET_FLAG_FINAL;
  }
  iree_hal_amdgpu_aql_packet_t* packet =
      iree_hal_amdgpu_aql_block_processor_packet(processor, packet_index);

  iree_status_t status = iree_ok_status();
  if (command->opcode == IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_FILL) {
    status = iree_hal_amdgpu_aql_block_processor_replay_fill_packet_body(
        processor,
        (const iree_hal_amdgpu_command_buffer_fill_command_t*)command, packet,
        &processor->kernargs.blocks[state->kernargs.block],
        iree_hsa_signal_null(), &processor->packets.setups[packet_index]);
  } else if (command->opcode == IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_COPY) {
    status = iree_hal_amdgpu_aql_block_processor_replay_copy_packet_body(
        processor,
        (const iree_hal_amdgpu_command_buffer_copy_command_t*)command, packet,
        &processor->kernargs.blocks[state->kernargs.block],
        iree_hsa_signal_null(), &processor->packets.setups[packet_index]);
  } else {
    const iree_hal_amdgpu_command_buffer_update_command_t* update_command =
        (const iree_hal_amdgpu_command_buffer_update_command_t*)command;
    const iree_host_size_t kernarg_length =
        iree_hal_amdgpu_aql_block_processor_update_kernarg_length(
            update_command->length);
    const iree_host_size_t kernarg_block_count = iree_host_size_ceil_div(
        kernarg_length, sizeof(iree_hal_amdgpu_kernarg_block_t));
    status = iree_hal_amdgpu_aql_block_processor_replay_update_packet_body(
        processor, update_command, packet,
        processor->kernargs.blocks[state->kernargs.block].data,
        kernarg_block_count * sizeof(iree_hal_amdgpu_kernarg_block_t),
        iree_hsa_signal_null(), &processor->packets.setups[packet_index]);
    if (iree_status_is_ok(status)) {
      state->kernargs.block += (uint32_t)kernarg_block_count - 1u;
    }
  }
  if (iree_status_is_ok(status)) {
    processor->packets.headers[packet_index] = iree_hal_amdgpu_aql_make_header(
        IREE_HSA_PACKET_TYPE_KERNEL_DISPATCH,
        iree_hal_amdgpu_aql_block_processor_packet_control(
            processor, packet_index,
            iree_hal_amdgpu_aql_block_processor_payload_acquire_scope(
                processor, state, payload_acquire_packet_count, packet_index,
                command, packet_flags),
            packet_flags));
    ++state->packets.emitted;
    ++state->packets.recorded;
    ++state->kernargs.block;
  }
  return status;
}

void iree_hal_amdgpu_aql_block_processor_initialize(
    const iree_hal_amdgpu_aql_block_processor_t* params,
    iree_hal_amdgpu_aql_block_processor_t* out_processor) {
  *out_processor = *params;
}

void iree_hal_amdgpu_aql_block_processor_deinitialize(
    iree_hal_amdgpu_aql_block_processor_t* processor) {
  memset(processor, 0, sizeof(*processor));
}

iree_status_t iree_hal_amdgpu_aql_block_processor_invoke(
    const iree_hal_amdgpu_aql_block_processor_t* processor,
    const iree_hal_amdgpu_command_buffer_block_header_t* block,
    iree_hal_amdgpu_aql_block_processor_result_t* out_result) {
  memset(out_result, 0, sizeof(*out_result));
  const uint32_t payload_acquire_packet_count =
      iree_hal_amdgpu_aql_block_processor_payload_acquire_packet_count(
          processor, block);
  const iree_hal_amdgpu_command_buffer_command_header_t* command =
      iree_hal_amdgpu_command_buffer_block_commands_const(block);
  iree_hal_amdgpu_aql_block_processor_state_t state = {0};
  bool reached_terminator = false;
  iree_status_t status = iree_ok_status();
  for (uint16_t i = 0; i < block->command_count && iree_status_is_ok(status) &&
                       !reached_terminator;
       ++i) {
    switch (command->opcode) {
      case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_BARRIER:
        break;
      case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_DISPATCH:
        status = iree_hal_amdgpu_aql_block_processor_emit_dispatch(
            processor, block, payload_acquire_packet_count,
            (const iree_hal_amdgpu_command_buffer_dispatch_command_t*)command,
            &state);
        break;
      case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_FILL:
      case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_COPY:
      case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_UPDATE:
        status = iree_hal_amdgpu_aql_block_processor_emit_transfer(
            processor, block, payload_acquire_packet_count, command, &state);
        break;
      case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_RETURN:
        out_result->terminator =
            IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_TERMINATOR_RETURN;
        reached_terminator = true;
        break;
      case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_BRANCH: {
        const iree_hal_amdgpu_command_buffer_branch_command_t* branch_command =
            (const iree_hal_amdgpu_command_buffer_branch_command_t*)command;
        out_result->terminator =
            IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_TERMINATOR_BRANCH;
        out_result->target_block_ordinal = branch_command->target_block_ordinal;
        reached_terminator = true;
        break;
      }
      case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_PROFILE_MARKER:
      case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_COND_BRANCH:
        status = iree_make_status(
            IREE_STATUS_UNIMPLEMENTED,
            "AQL command-buffer opcode %u replay not yet wired",
            command->opcode);
        break;
      default:
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "malformed AQL command-buffer opcode %u",
                                  command->opcode);
        break;
    }
    if (iree_status_is_ok(status) && !reached_terminator) {
      command = iree_hal_amdgpu_command_buffer_command_next_const(command);
    }
  }
  if (iree_status_is_ok(status) && !reached_terminator) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "AQL command-buffer block %" PRIu32
                              " has no terminator",
                              block->block_ordinal);
  }
  if (iree_status_is_ok(status) &&
      state.packets.recorded != block->aql_packet_count) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AQL command-buffer block %" PRIu32 " consumed %" PRIu32
        " packets but declares %" PRIu32,
        block->block_ordinal, state.packets.recorded, block->aql_packet_count);
  }
  if (iree_status_is_ok(status) &&
      state.packets.emitted != processor->packets.count) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AQL command-buffer block %" PRIu32 " emitted %" PRIu32
        " payload packets but reserved %u",
        block->block_ordinal, state.packets.emitted, processor->packets.count);
  }
  if (iree_status_is_ok(status) &&
      state.kernargs.block != processor->kernargs.count) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AQL command-buffer block %" PRIu32 " emitted %" PRIu32
        " kernarg blocks but reserved %" PRIu32,
        block->block_ordinal, state.kernargs.block, processor->kernargs.count);
  }
  out_result->packets.recorded = state.packets.recorded;
  out_result->packets.emitted = state.packets.emitted;
  out_result->kernargs.consumed = state.kernargs.block;
  return status;
}
