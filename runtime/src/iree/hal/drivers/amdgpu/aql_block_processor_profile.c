// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/aql_block_processor_profile.h"

#include <string.h>

#include "iree/hal/drivers/amdgpu/buffer.h"
#include "iree/hal/drivers/amdgpu/device/blit.h"
#include "iree/hal/drivers/amdgpu/device/dispatch.h"
#include "iree/hal/drivers/amdgpu/host_queue_command_buffer_packet.h"
#include "iree/hal/drivers/amdgpu/host_queue_command_buffer_profile.h"
#include "iree/hal/drivers/amdgpu/host_queue_policy.h"
#include "iree/hal/drivers/amdgpu/profile_counters.h"
#include "iree/hal/drivers/amdgpu/profile_traces.h"
#include "iree/hal/drivers/amdgpu/util/aql_emitter.h"

typedef struct iree_hal_amdgpu_aql_block_processor_profile_state_t {
  // Packet cursors advanced while invoking profiled replay.
  struct {
    // Number of recorded block AQL packets consumed from the block.
    uint32_t recorded;
    // Number of payload AQL packets emitted into the reserved packet span.
    uint32_t emitted;
  } packets;
  // Queue-owned kernarg cursors advanced while invoking profiled replay.
  struct {
    // Number of queue-owned kernarg blocks consumed from the reserved span.
    uint32_t block;
  } kernargs;
  // Profile sidecar cursors advanced while invoking profiled replay.
  struct {
    // Number of dispatch profile events consumed from the reservation.
    uint32_t event;
  } profile;
} iree_hal_amdgpu_aql_block_processor_profile_state_t;

typedef uint32_t iree_hal_amdgpu_aql_block_processor_dispatch_profile_flags_t;
enum iree_hal_amdgpu_aql_block_processor_dispatch_profile_flag_bits_t {
  IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_DISPATCH_PROFILE_FLAG_NONE = 0u,
  // Dispatch has a reserved profiling event and completion signal.
  IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_DISPATCH_PROFILE_FLAG_DISPATCH_PACKET =
      1u << 0,
  // Counter PM4 packets wrap this dispatch event.
  IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_DISPATCH_PROFILE_FLAG_COUNTER_PACKETS =
      1u << 1,
  // ATT trace PM4 packets wrap this dispatch event.
  IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_DISPATCH_PROFILE_FLAG_TRACE_PACKETS =
      1u << 2,
  // Dispatch consumes the final recorded packet in the block.
  IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_DISPATCH_PROFILE_FLAG_BLOCK_FINAL = 1u
                                                                          << 3,
  // Dispatch is the final replayed payload packet in the block.
  IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_DISPATCH_PROFILE_FLAG_FINAL = 1u << 4,
};

typedef struct iree_hal_amdgpu_aql_block_processor_dispatch_profile_t {
  // Flags from
  // iree_hal_amdgpu_aql_block_processor_dispatch_profile_flag_bits_t.
  iree_hal_amdgpu_aql_block_processor_dispatch_profile_flags_t flags;
  // Event-ring position for this dispatch when DISPATCH_PACKET is set.
  uint64_t event_position;
  // Selected dispatch profile sidecar when DISPATCH_PACKET is set.
  const iree_hal_amdgpu_aql_block_processor_profile_dispatch_t* dispatch;
} iree_hal_amdgpu_aql_block_processor_dispatch_profile_t;

// Command flags split across multi-packet profiled replay implementations.
typedef struct iree_hal_amdgpu_aql_block_processor_profile_packet_flag_pair_t {
  // Flags applied to the first AQL packet implementing a recorded command.
  iree_hal_amdgpu_host_queue_command_buffer_packet_flags_t first;
  // Flags applied to the final AQL packet implementing a recorded command.
  iree_hal_amdgpu_host_queue_command_buffer_packet_flags_t final;
} iree_hal_amdgpu_aql_block_processor_profile_packet_flag_pair_t;

static iree_status_t
iree_hal_amdgpu_aql_block_processor_profile_resolve_buffer_ref_ptr(
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
iree_hal_amdgpu_aql_block_processor_profile_resolve_command_buffer_ref(
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
      iree_hal_amdgpu_aql_block_processor_profile_resolve_buffer_ref_ptr(
          *out_buffer_ref, required_usage, required_access, out_device_ptr));
  return iree_ok_status();
}

static iree_status_t
iree_hal_amdgpu_aql_block_processor_profile_resolve_static_binding_source_ptr(
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

static bool iree_hal_amdgpu_aql_block_processor_profile_packet_has_barrier(
    const iree_hal_amdgpu_aql_block_processor_profile_t* processor,
    uint32_t packet_index,
    iree_hal_amdgpu_host_queue_command_buffer_packet_flags_t packet_flags) {
  return iree_any_bit_set(
             packet_flags,
             IREE_HAL_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_PACKET_FLAG_EXECUTION_BARRIER) ||
         iree_any_bit_set(
             packet_flags,
             IREE_HAL_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_PACKET_FLAG_FINAL) ||
         (packet_index == 0 &&
          processor->submission.resolution->barrier_count > 0) ||
         (packet_index == 0 &&
          processor->submission.resolution->inline_acquire_scope !=
              IREE_HSA_FENCE_SCOPE_NONE);
}

static iree_hal_amdgpu_host_queue_command_buffer_packet_flags_t
iree_hal_amdgpu_aql_block_processor_profile_command_packet_flags(
    const iree_hal_amdgpu_command_buffer_command_header_t* command) {
  iree_hal_amdgpu_host_queue_command_buffer_packet_flags_t packet_flags =
      IREE_HAL_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_PACKET_FLAG_NONE;
  if (iree_any_bit_set(
          command->flags,
          IREE_HAL_AMDGPU_COMMAND_BUFFER_COMMAND_FLAG_HAS_BARRIER)) {
    packet_flags |=
        IREE_HAL_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_PACKET_FLAG_EXECUTION_BARRIER;
  }
  return iree_hal_amdgpu_host_queue_command_buffer_packet_flags_set_fence_scopes(
      packet_flags,
      (iree_hsa_fence_scope_t)
          iree_hal_amdgpu_command_buffer_command_flags_acquire_scope(
              command->flags),
      (iree_hsa_fence_scope_t)
          iree_hal_amdgpu_command_buffer_command_flags_release_scope(
              command->flags));
}

static iree_hal_amdgpu_host_queue_command_buffer_packet_flags_t
iree_hal_amdgpu_aql_block_processor_profile_packet_flags_merge(
    iree_hal_amdgpu_host_queue_command_buffer_packet_flags_t lhs,
    iree_hal_amdgpu_host_queue_command_buffer_packet_flags_t rhs) {
  const iree_hsa_fence_scope_t acquire_scope =
      iree_hal_amdgpu_host_queue_max_fence_scope(
          iree_hal_amdgpu_host_queue_command_buffer_packet_flags_acquire_scope(
              lhs),
          iree_hal_amdgpu_host_queue_command_buffer_packet_flags_acquire_scope(
              rhs));
  const iree_hsa_fence_scope_t release_scope =
      iree_hal_amdgpu_host_queue_max_fence_scope(
          iree_hal_amdgpu_host_queue_command_buffer_packet_flags_release_scope(
              lhs),
          iree_hal_amdgpu_host_queue_command_buffer_packet_flags_release_scope(
              rhs));
  return iree_hal_amdgpu_host_queue_command_buffer_packet_flags_set_fence_scopes(
      lhs | rhs, acquire_scope, release_scope);
}

static iree_hal_amdgpu_host_queue_command_buffer_packet_flags_t
iree_hal_amdgpu_aql_block_processor_profile_agent_barrier_packet_flags(void) {
  return iree_hal_amdgpu_host_queue_command_buffer_packet_flags_set_fence_scopes(
      IREE_HAL_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_PACKET_FLAG_EXECUTION_BARRIER,
      IREE_HSA_FENCE_SCOPE_AGENT, IREE_HSA_FENCE_SCOPE_AGENT);
}

static iree_hal_amdgpu_host_queue_command_buffer_packet_flags_t
iree_hal_amdgpu_aql_block_processor_profile_execution_barrier_packet_flags(
    void) {
  return iree_hal_amdgpu_host_queue_command_buffer_packet_flags_set_fence_scopes(
      IREE_HAL_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_PACKET_FLAG_EXECUTION_BARRIER,
      IREE_HSA_FENCE_SCOPE_NONE, IREE_HSA_FENCE_SCOPE_NONE);
}

static iree_hal_amdgpu_aql_block_processor_profile_packet_flag_pair_t
iree_hal_amdgpu_aql_block_processor_profile_split_command_packet_flags(
    const iree_hal_amdgpu_command_buffer_command_header_t* command) {
  const iree_hal_amdgpu_host_queue_command_buffer_packet_flags_t command_flags =
      iree_hal_amdgpu_aql_block_processor_profile_command_packet_flags(command);
  const iree_hsa_fence_scope_t acquire_scope =
      iree_hal_amdgpu_host_queue_command_buffer_packet_flags_acquire_scope(
          command_flags);
  const iree_hsa_fence_scope_t release_scope =
      iree_hal_amdgpu_host_queue_command_buffer_packet_flags_release_scope(
          command_flags);
  return (iree_hal_amdgpu_aql_block_processor_profile_packet_flag_pair_t){
      .first =
          iree_hal_amdgpu_host_queue_command_buffer_packet_flags_set_fence_scopes(
              command_flags, acquire_scope, IREE_HSA_FENCE_SCOPE_NONE),
      .final =
          iree_hal_amdgpu_host_queue_command_buffer_packet_flags_set_fence_scopes(
              command_flags &
                  ~IREE_HAL_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_PACKET_FLAG_EXECUTION_BARRIER,
              IREE_HSA_FENCE_SCOPE_NONE, release_scope),
  };
}

static bool
iree_hal_amdgpu_aql_block_processor_profile_command_uses_queue_kernargs(
    const iree_hal_amdgpu_command_buffer_command_header_t* command) {
  return iree_any_bit_set(
      command->flags,
      IREE_HAL_AMDGPU_COMMAND_BUFFER_COMMAND_FLAG_USES_QUEUE_KERNARGS);
}

static iree_hsa_fence_scope_t
iree_hal_amdgpu_aql_block_processor_profile_payload_acquire_scope(
    const iree_hal_amdgpu_aql_block_processor_profile_t* processor,
    const iree_hal_amdgpu_aql_block_processor_profile_state_t* state,
    uint32_t packet_index,
    const iree_hal_amdgpu_command_buffer_command_header_t* command,
    iree_hal_amdgpu_host_queue_command_buffer_packet_flags_t packet_flags) {
  if (processor->payload.acquire_scope == IREE_HSA_FENCE_SCOPE_NONE) {
    return IREE_HSA_FENCE_SCOPE_NONE;
  }
  if (state->packets.recorded >= processor->payload.acquire_packet_count) {
    return IREE_HSA_FENCE_SCOPE_NONE;
  }
  if (iree_hal_amdgpu_aql_block_processor_profile_command_uses_queue_kernargs(
          command)) {
    return processor->payload.acquire_scope;
  }
  if (iree_hal_amdgpu_aql_block_processor_profile_packet_has_barrier(
          processor, processor->packets.index_base + packet_index,
          packet_flags)) {
    return processor->payload.acquire_scope;
  }
  return IREE_HSA_FENCE_SCOPE_NONE;
}

static bool iree_hal_amdgpu_aql_block_processor_profile_dispatch_uses_indirect(
    const iree_hal_amdgpu_command_buffer_dispatch_command_t* dispatch_command) {
  return iree_any_bit_set(
      dispatch_command->dispatch_flags,
      IREE_HAL_AMDGPU_COMMAND_BUFFER_DISPATCH_FLAG_INDIRECT_PARAMETERS);
}

static bool
iree_hal_amdgpu_aql_block_processor_profile_dispatch_uses_prepublished(
    const iree_hal_amdgpu_command_buffer_dispatch_command_t* dispatch_command) {
  return dispatch_command->kernarg_strategy ==
         IREE_HAL_AMDGPU_COMMAND_BUFFER_KERNARG_STRATEGY_PREPUBLISHED;
}

static bool
iree_hal_amdgpu_aql_block_processor_profile_dispatch_uses_queue_kernargs(
    const iree_hal_amdgpu_command_buffer_dispatch_command_t* dispatch_command) {
  return !iree_hal_amdgpu_aql_block_processor_profile_dispatch_uses_prepublished(
      dispatch_command);
}

static uint32_t
iree_hal_amdgpu_aql_block_processor_profile_dispatch_target_kernarg_block_count(
    const iree_hal_amdgpu_command_buffer_dispatch_command_t* dispatch_command) {
  if (iree_hal_amdgpu_aql_block_processor_profile_dispatch_uses_prepublished(
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
iree_hal_amdgpu_aql_block_processor_profile_dispatch_kernarg_block_count(
    const iree_hal_amdgpu_command_buffer_dispatch_command_t* dispatch_command) {
  return iree_hal_amdgpu_aql_block_processor_profile_dispatch_target_kernarg_block_count(
             dispatch_command) +
         (iree_hal_amdgpu_aql_block_processor_profile_dispatch_uses_indirect(
              dispatch_command)
              ? 1u
              : 0u);
}

static void
iree_hal_amdgpu_aql_block_processor_profile_write_dispatch_packet_body(
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

static iree_status_t
iree_hal_amdgpu_aql_block_processor_profile_replay_dispatch_kernargs(
    const iree_hal_amdgpu_aql_block_processor_profile_t* processor,
    const iree_hal_amdgpu_command_buffer_dispatch_command_t* dispatch_command,
    uint8_t* kernarg_data) {
  switch (dispatch_command->kernarg_strategy) {
    case IREE_HAL_AMDGPU_COMMAND_BUFFER_KERNARG_STRATEGY_DYNAMIC_BINDINGS: {
      if (IREE_UNLIKELY(!processor->bindings.ptrs)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "AQL command-buffer dispatch has dynamic bindings but no binding "
            "table was provided");
      }
      uint64_t* binding_dst = (uint64_t*)kernarg_data;
      const iree_hal_amdgpu_command_buffer_binding_source_t* binding_sources =
          (const iree_hal_amdgpu_command_buffer_binding_source_t*)((const uint8_t*)
                                                                       processor
                                                                           ->block +
                                                                   dispatch_command
                                                                       ->binding_source_offset);
      for (uint16_t i = 0; i < dispatch_command->binding_count; ++i) {
        const iree_hal_amdgpu_command_buffer_binding_source_t* binding_source =
            &binding_sources[i];
        binding_dst[i] = processor->bindings.ptrs[binding_source->slot] +
                         binding_source->offset_or_pointer;
      }
      const iree_host_size_t tail_length =
          (iree_host_size_t)dispatch_command->payload.tail_length_qwords * 8u;
      if (tail_length > 0) {
        const uint8_t* tail_payload = (const uint8_t*)dispatch_command +
                                      dispatch_command->payload_reference;
        memcpy(
            kernarg_data + (iree_host_size_t)dispatch_command->binding_count *
                               sizeof(uint64_t),
            tail_payload, tail_length);
      }
      return iree_ok_status();
    }
    case IREE_HAL_AMDGPU_COMMAND_BUFFER_KERNARG_STRATEGY_HAL: {
      uint64_t* binding_dst = (uint64_t*)kernarg_data;
      const iree_hal_amdgpu_command_buffer_binding_source_t* binding_sources =
          (const iree_hal_amdgpu_command_buffer_binding_source_t*)((const uint8_t*)
                                                                       processor
                                                                           ->block +
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
          if (IREE_UNLIKELY(!processor->bindings.ptrs)) {
            return iree_make_status(
                IREE_STATUS_INVALID_ARGUMENT,
                "AQL command-buffer dispatch has dynamic bindings but no "
                "binding table was provided");
          }
          binding_dst[i] = processor->bindings.ptrs[binding_source->slot] +
                           binding_source->offset_or_pointer;
        } else if (
            flags ==
            IREE_HAL_AMDGPU_COMMAND_BUFFER_BINDING_SOURCE_FLAG_STATIC_BUFFER) {
          IREE_RETURN_IF_ERROR(
              iree_hal_amdgpu_aql_block_processor_profile_resolve_static_binding_source_ptr(
                  processor->command_buffer, binding_source, &binding_dst[i]));
        } else {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "malformed AQL command-buffer dispatch binding source flags %u",
              binding_source->flags);
        }
      }
      const iree_host_size_t tail_length =
          (iree_host_size_t)dispatch_command->payload.tail_length_qwords * 8u;
      if (tail_length > 0) {
        const uint8_t* tail_payload = (const uint8_t*)dispatch_command +
                                      dispatch_command->payload_reference;
        memcpy(
            kernarg_data + (iree_host_size_t)dispatch_command->binding_count *
                               sizeof(uint64_t),
            tail_payload, tail_length);
      }
      return iree_ok_status();
    }
    case IREE_HAL_AMDGPU_COMMAND_BUFFER_KERNARG_STRATEGY_CUSTOM_DIRECT: {
      const iree_host_size_t tail_length =
          (iree_host_size_t)dispatch_command->payload.tail_length_qwords * 8u;
      if (tail_length > 0) {
        const uint8_t* tail_payload = (const uint8_t*)dispatch_command +
                                      dispatch_command->payload_reference;
        memcpy(kernarg_data, tail_payload, tail_length);
      }
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
              processor->command_buffer, dispatch_command->payload_reference,
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
      const uint64_t* binding_ptrs = processor->bindings.ptrs;
      if (IREE_UNLIKELY(!binding_ptrs)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "AQL command-buffer dispatch has dynamic bindings but no binding "
            "table was provided");
      }
      const iree_hal_amdgpu_command_buffer_binding_source_t* binding_sources =
          (const iree_hal_amdgpu_command_buffer_binding_source_t*)((const uint8_t*)
                                                                       processor
                                                                           ->block +
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
iree_hal_amdgpu_aql_block_processor_profile_replay_dispatch_indirect_params_ptr(
    const iree_hal_amdgpu_aql_block_processor_profile_t* processor,
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
          iree_hal_amdgpu_aql_block_processor_profile_resolve_buffer_ref_ptr(
              resolved_ref, IREE_HAL_BUFFER_USAGE_DISPATCH_INDIRECT_PARAMETERS,
              IREE_HAL_MEMORY_ACCESS_READ, &device_ptr));
      *out_workgroup_count_ptr = (const uint32_t*)device_ptr;
      return iree_ok_status();
    }
    case IREE_HAL_AMDGPU_COMMAND_BUFFER_BINDING_SOURCE_FLAG_STATIC_BUFFER |
        IREE_HAL_AMDGPU_COMMAND_BUFFER_BINDING_SOURCE_FLAG_INDIRECT_PARAMETERS: {
      uint64_t workgroup_count_ptr = 0;
      IREE_RETURN_IF_ERROR(
          iree_hal_amdgpu_aql_block_processor_profile_resolve_static_binding_source_ptr(
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
iree_hal_amdgpu_aql_block_processor_profile_dispatch_implicit_args_ptr(
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
iree_hal_amdgpu_aql_block_processor_profile_replay_dispatch_packet_body(
    const iree_hal_amdgpu_aql_block_processor_profile_t* processor,
    const iree_hal_amdgpu_command_buffer_dispatch_command_t* dispatch_command,
    iree_hal_amdgpu_aql_packet_t* packet, uint8_t* kernarg_data,
    iree_hsa_signal_t completion_signal, uint16_t* out_setup) {
  if (iree_hal_amdgpu_aql_block_processor_profile_dispatch_uses_prepublished(
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
        iree_hal_amdgpu_aql_block_processor_profile_replay_dispatch_kernargs(
            processor, dispatch_command, kernarg_data));
  }
  iree_hal_amdgpu_aql_block_processor_profile_write_dispatch_packet_body(
      dispatch_command, packet, kernarg_data, completion_signal, out_setup);
  return iree_ok_status();
}

static iree_status_t
iree_hal_amdgpu_aql_block_processor_profile_replay_indirect_dispatch_packet_bodies(
    const iree_hal_amdgpu_aql_block_processor_profile_t* processor,
    const iree_hal_amdgpu_command_buffer_dispatch_command_t* dispatch_command,
    iree_hal_amdgpu_aql_packet_t* patch_packet,
    iree_hal_amdgpu_aql_packet_t* dispatch_packet, uint8_t* patch_kernarg_data,
    uint8_t* dispatch_kernarg_data, iree_hsa_signal_t completion_signal,
    uint16_t dispatch_header, uint16_t* out_patch_setup,
    uint16_t* out_dispatch_setup) {
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_aql_block_processor_profile_replay_dispatch_packet_body(
          processor, dispatch_command, dispatch_packet, dispatch_kernarg_data,
          completion_signal, out_dispatch_setup));

  const iree_hal_amdgpu_command_buffer_binding_source_t* binding_sources =
      (const iree_hal_amdgpu_command_buffer_binding_source_t*)((const uint8_t*)
                                                                   processor
                                                                       ->block +
                                                               dispatch_command
                                                                   ->binding_source_offset);
  const iree_hal_amdgpu_command_buffer_binding_source_t*
      indirect_params_source =
          &binding_sources[dispatch_command->binding_count];
  const uint32_t* workgroup_count_ptr = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_aql_block_processor_profile_replay_dispatch_indirect_params_ptr(
          processor, indirect_params_source, &workgroup_count_ptr));

  iree_amdgpu_kernel_implicit_args_t* implicit_args =
      iree_hal_amdgpu_aql_block_processor_profile_dispatch_implicit_args_ptr(
          dispatch_command, dispatch_kernarg_data);
  iree_hal_amdgpu_device_dispatch_emplace_indirect_params_patch(
      &processor->queue->transfer_context->kernels
           ->iree_hal_amdgpu_device_dispatch_patch_indirect_params,
      workgroup_count_ptr, &dispatch_packet->dispatch, dispatch_header,
      *out_dispatch_setup, implicit_args, &patch_packet->dispatch,
      patch_kernarg_data);
  *out_patch_setup = patch_packet->dispatch.setup;
  return iree_ok_status();
}

static iree_status_t
iree_hal_amdgpu_aql_block_processor_profile_replay_fill_packet_body(
    const iree_hal_amdgpu_aql_block_processor_profile_t* processor,
    const iree_hal_amdgpu_command_buffer_fill_command_t* fill_command,
    iree_hal_amdgpu_aql_packet_t* packet,
    iree_hal_amdgpu_kernarg_block_t* kernarg_block,
    iree_hsa_signal_t completion_signal, uint16_t* out_setup) {
  iree_hal_buffer_ref_t target_ref = {0};
  uint8_t* target_ptr = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_aql_block_processor_profile_resolve_command_buffer_ref(
          processor->command_buffer, processor->bindings.table,
          fill_command->target_kind, fill_command->target_ordinal,
          fill_command->target_offset, fill_command->length,
          IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET, IREE_HAL_MEMORY_ACCESS_WRITE,
          &target_ref, &target_ptr));
  (void)target_ref;
  if (IREE_UNLIKELY(!iree_hal_amdgpu_device_buffer_fill_emplace(
          processor->queue->transfer_context, &packet->dispatch, target_ptr,
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
iree_hal_amdgpu_aql_block_processor_profile_replay_copy_packet_body(
    const iree_hal_amdgpu_aql_block_processor_profile_t* processor,
    const iree_hal_amdgpu_command_buffer_copy_command_t* copy_command,
    iree_hal_amdgpu_aql_packet_t* packet,
    iree_hal_amdgpu_kernarg_block_t* kernarg_block,
    iree_hsa_signal_t completion_signal, uint16_t* out_setup) {
  iree_hal_buffer_ref_t source_ref = {0};
  uint8_t* source_ptr = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_aql_block_processor_profile_resolve_command_buffer_ref(
          processor->command_buffer, processor->bindings.table,
          copy_command->source_kind, copy_command->source_ordinal,
          copy_command->source_offset, copy_command->length,
          IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE, IREE_HAL_MEMORY_ACCESS_READ,
          &source_ref, &source_ptr));
  iree_hal_buffer_ref_t target_ref = {0};
  uint8_t* target_ptr = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_aql_block_processor_profile_resolve_command_buffer_ref(
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
          processor->queue->transfer_context, &packet->dispatch, source_ptr,
          target_ptr, copy_command->length, kernarg_block->data))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported command-buffer copy dispatch shape");
  }
  packet->dispatch.completion_signal = completion_signal;
  *out_setup = packet->dispatch.setup;
  return iree_ok_status();
}

static iree_host_size_t
iree_hal_amdgpu_aql_block_processor_profile_update_kernarg_length(
    uint32_t source_length) {
  const iree_host_size_t source_payload_offset =
      IREE_HAL_AMDGPU_DEVICE_BUFFER_COPY_STAGED_SOURCE_OFFSET;
  return source_payload_offset + (iree_host_size_t)source_length;
}

static uint32_t
iree_hal_amdgpu_aql_block_processor_profile_update_kernarg_block_count(
    uint32_t source_length) {
  return (uint32_t)iree_host_size_ceil_div(
      iree_hal_amdgpu_aql_block_processor_profile_update_kernarg_length(
          source_length),
      sizeof(iree_hal_amdgpu_kernarg_block_t));
}

static iree_status_t
iree_hal_amdgpu_aql_block_processor_profile_resolve_update_packet_operands(
    const iree_hal_amdgpu_aql_block_processor_profile_t* processor,
    const iree_hal_amdgpu_command_buffer_update_command_t* update_command,
    const uint8_t** out_source_bytes, uint8_t** out_target_ptr) {
  *out_source_bytes = NULL;
  *out_target_ptr = NULL;
  iree_hal_buffer_ref_t target_ref = {0};
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_aql_block_processor_profile_resolve_command_buffer_ref(
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
iree_hal_amdgpu_aql_block_processor_profile_replay_update_packet_body(
    const iree_hal_amdgpu_aql_block_processor_profile_t* processor,
    const iree_hal_amdgpu_command_buffer_update_command_t* update_command,
    iree_hal_amdgpu_aql_packet_t* packet, uint8_t* kernarg_data,
    iree_host_size_t kernarg_length, iree_hsa_signal_t completion_signal,
    uint16_t* out_setup) {
  const uint8_t* source_bytes = NULL;
  uint8_t* target_ptr = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_aql_block_processor_profile_resolve_update_packet_operands(
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
          processor->queue->transfer_context, &packet->dispatch,
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

static iree_hal_amdgpu_aql_packet_t*
iree_hal_amdgpu_aql_block_processor_profile_packet(
    const iree_hal_amdgpu_aql_block_processor_profile_t* processor,
    uint32_t packet_index) {
  return iree_hal_amdgpu_aql_ring_packet(
      &processor->queue->aql_ring,
      processor->packets.first_payload_id + packet_index);
}

static bool iree_hal_amdgpu_aql_block_processor_profile_is_block_final(
    const iree_hal_amdgpu_aql_block_processor_profile_t* processor,
    const iree_hal_amdgpu_aql_block_processor_profile_state_t* state,
    uint32_t recorded_packet_count) {
  return state->packets.recorded + recorded_packet_count ==
         processor->block->aql_packet_count;
}

static iree_hal_amdgpu_aql_packet_control_t
iree_hal_amdgpu_aql_block_processor_profile_packet_control(
    const iree_hal_amdgpu_aql_block_processor_profile_t* processor,
    uint32_t packet_index, iree_hsa_fence_scope_t minimum_acquire_scope,
    iree_hal_amdgpu_host_queue_command_buffer_packet_flags_t packet_flags) {
  return iree_hal_amdgpu_host_queue_command_buffer_packet_control(
      processor->queue, processor->submission.resolution,
      processor->submission.signal_semaphore_list,
      processor->packets.index_base + packet_index, minimum_acquire_scope,
      packet_flags);
}

static iree_hsa_signal_t
iree_hal_amdgpu_aql_block_processor_profile_dispatch_completion_signal(
    iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_amdgpu_profile_dispatch_event_reservation_t profile_events,
    uint32_t profile_event_index) {
  const uint64_t profile_event_position =
      profile_events.first_event_position + profile_event_index;
  return iree_hal_amdgpu_host_queue_profiling_completion_signal(
      queue, profile_event_position);
}

static bool iree_hal_amdgpu_aql_block_processor_dispatch_profile_has(
    iree_hal_amdgpu_aql_block_processor_dispatch_profile_t profile,
    iree_hal_amdgpu_aql_block_processor_dispatch_profile_flags_t flags) {
  return iree_any_bit_set(profile.flags, flags);
}

static iree_hal_amdgpu_host_queue_command_buffer_packet_flags_t
iree_hal_amdgpu_aql_block_processor_profile_packet_flags(
    iree_hal_amdgpu_aql_block_processor_dispatch_profile_t profile) {
  iree_hal_amdgpu_host_queue_command_buffer_packet_flags_t flags =
      IREE_HAL_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_PACKET_FLAG_NONE;
  if (iree_hal_amdgpu_aql_block_processor_dispatch_profile_has(
          profile,
          IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_DISPATCH_PROFILE_FLAG_COUNTER_PACKETS |
              IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_DISPATCH_PROFILE_FLAG_TRACE_PACKETS)) {
    flags = iree_hal_amdgpu_aql_block_processor_profile_packet_flags_merge(
        flags,
        iree_hal_amdgpu_aql_block_processor_profile_agent_barrier_packet_flags());
  }
  if (iree_hal_amdgpu_aql_block_processor_dispatch_profile_has(
          profile,
          IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_DISPATCH_PROFILE_FLAG_FINAL)) {
    flags |= IREE_HAL_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_PACKET_FLAG_FINAL;
  }
  return flags;
}

static iree_hsa_signal_t
iree_hal_amdgpu_aql_block_processor_profile_completion_signal(
    const iree_hal_amdgpu_aql_block_processor_profile_t* processor,
    const iree_hal_amdgpu_aql_block_processor_profile_state_t* state,
    iree_hal_amdgpu_aql_block_processor_dispatch_profile_t profile) {
  if (!iree_hal_amdgpu_aql_block_processor_dispatch_profile_has(
          profile,
          IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_DISPATCH_PROFILE_FLAG_DISPATCH_PACKET)) {
    return iree_hsa_signal_null();
  }
  return iree_hal_amdgpu_aql_block_processor_profile_dispatch_completion_signal(
      processor->queue, processor->profile.dispatch_events,
      state->profile.event);
}

static void iree_hal_amdgpu_aql_block_processor_profile_emit_source(
    const iree_hal_amdgpu_aql_block_processor_profile_t* processor,
    iree_hal_amdgpu_aql_block_processor_dispatch_profile_t profile,
    iree_hal_amdgpu_aql_block_processor_profile_state_t* state) {
  if (!iree_hal_amdgpu_aql_block_processor_dispatch_profile_has(
          profile,
          IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_DISPATCH_PROFILE_FLAG_DISPATCH_PACKET)) {
    return;
  }
  iree_hal_amdgpu_host_queue_record_command_buffer_profile_dispatch_source(
      processor->queue, processor->profile.command_buffer_id, profile.dispatch,
      processor->profile.dispatch_events, processor->profile.harvest_sources,
      &state->profile.event);
}

static const iree_hal_amdgpu_aql_block_processor_profile_dispatch_t*
iree_hal_amdgpu_aql_block_processor_profile_current_dispatch(
    const iree_hal_amdgpu_aql_block_processor_profile_t* processor,
    const iree_hal_amdgpu_aql_block_processor_profile_state_t* state,
    uint32_t dispatch_packet_ordinal) {
  if (state->profile.event >= processor->profile.dispatches.count) {
    return NULL;
  }
  const iree_hal_amdgpu_aql_block_processor_profile_dispatch_t* dispatch =
      &processor->profile.dispatches.values[state->profile.event];
  if (IREE_UNLIKELY(!dispatch->summary)) return NULL;
  if (dispatch->summary->packets.dispatch_ordinal != dispatch_packet_ordinal) {
    return NULL;
  }
  return dispatch;
}

static iree_hal_amdgpu_aql_block_processor_dispatch_profile_t
iree_hal_amdgpu_aql_block_processor_profile_dispatch_profile(
    const iree_hal_amdgpu_aql_block_processor_profile_t* processor,
    const iree_hal_amdgpu_aql_block_processor_profile_state_t* state,
    uint32_t recorded_packet_count, uint32_t dispatch_packet_ordinal) {
  iree_hal_amdgpu_aql_block_processor_dispatch_profile_flags_t flags =
      IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_DISPATCH_PROFILE_FLAG_NONE;
  const iree_hal_amdgpu_aql_block_processor_profile_dispatch_t*
      profile_dispatch =
          iree_hal_amdgpu_aql_block_processor_profile_current_dispatch(
              processor, state, dispatch_packet_ordinal);
  if (iree_any_bit_set(
          processor->flags,
          IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_PROFILE_FLAG_DISPATCH_PACKETS) &&
      profile_dispatch) {
    flags |=
        IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_DISPATCH_PROFILE_FLAG_DISPATCH_PACKET;
  }
  if (iree_any_bit_set(
          flags,
          IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_DISPATCH_PROFILE_FLAG_DISPATCH_PACKET) &&
      processor->profile.counter_set_count != 0) {
    flags |=
        IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_DISPATCH_PROFILE_FLAG_COUNTER_PACKETS;
  }
  if (iree_any_bit_set(
          flags,
          IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_DISPATCH_PROFILE_FLAG_DISPATCH_PACKET) &&
      processor->profile.trace_packet_count != 0) {
    flags |=
        IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_DISPATCH_PROFILE_FLAG_TRACE_PACKETS;
  }
  if (iree_hal_amdgpu_aql_block_processor_profile_is_block_final(
          processor, state, recorded_packet_count)) {
    flags |=
        IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_DISPATCH_PROFILE_FLAG_BLOCK_FINAL;
  }
  if (iree_any_bit_set(
          flags,
          IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_DISPATCH_PROFILE_FLAG_BLOCK_FINAL) &&
      !iree_any_bit_set(
          processor->flags,
          IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_PROFILE_FLAG_DISPATCH_PACKETS |
              IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_PROFILE_FLAG_QUEUE_DEVICE_EVENT)) {
    flags |= IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_DISPATCH_PROFILE_FLAG_FINAL;
  }
  return (iree_hal_amdgpu_aql_block_processor_dispatch_profile_t){
      .flags = flags,
      .event_position =
          iree_any_bit_set(
              flags,
              IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_DISPATCH_PROFILE_FLAG_DISPATCH_PACKET)
              ? processor->profile.dispatch_events.first_event_position +
                    state->profile.event
              : 0,
      .dispatch = profile_dispatch,
  };
}

static void iree_hal_amdgpu_aql_block_processor_profile_emit_counter_starts(
    const iree_hal_amdgpu_aql_block_processor_profile_t* processor,
    uint64_t event_position,
    iree_hal_amdgpu_aql_packet_control_t packet_control,
    iree_hal_amdgpu_aql_block_processor_profile_state_t* state) {
  if (processor->profile.counter_set_count == 0) return;
  iree_hal_amdgpu_host_queue_emplace_profile_counter_start_packets(
      processor->queue, event_position, processor->profile.counter_set_count,
      processor->packets.first_payload_id, state->packets.emitted,
      packet_control, processor->packets.headers, processor->packets.setups);
  state->packets.emitted += processor->profile.counter_set_count;
}

static void iree_hal_amdgpu_aql_block_processor_profile_emit_counter_read_stops(
    const iree_hal_amdgpu_aql_block_processor_profile_t* processor,
    uint64_t event_position,
    iree_hal_amdgpu_aql_block_processor_profile_state_t* state) {
  if (processor->profile.counter_set_count == 0) return;
  iree_hal_amdgpu_host_queue_emplace_profile_counter_read_stop_packets(
      processor->queue, event_position, processor->profile.counter_set_count,
      processor->packets.first_payload_id, state->packets.emitted,
      iree_hal_amdgpu_aql_packet_control_barrier(IREE_HSA_FENCE_SCOPE_AGENT,
                                                 IREE_HSA_FENCE_SCOPE_AGENT),
      processor->packets.headers, processor->packets.setups);
  state->packets.emitted += processor->profile.counter_set_count * 2u;
}

static void iree_hal_amdgpu_aql_block_processor_profile_emit_trace_start(
    const iree_hal_amdgpu_aql_block_processor_profile_t* processor,
    uint64_t event_position,
    iree_hal_amdgpu_host_queue_command_buffer_packet_flags_t packet_flags,
    iree_hal_amdgpu_aql_block_processor_profile_state_t* state) {
  iree_hal_amdgpu_host_queue_emplace_profile_trace_start_packet(
      processor->queue, event_position, processor->packets.first_payload_id,
      state->packets.emitted,
      iree_hal_amdgpu_aql_block_processor_profile_packet_control(
          processor, state->packets.emitted, IREE_HSA_FENCE_SCOPE_NONE,
          packet_flags),
      processor->packets.headers, processor->packets.setups);
  ++state->packets.emitted;
  iree_hal_amdgpu_host_queue_emplace_profile_trace_code_object_packet(
      processor->queue, event_position, processor->packets.first_payload_id,
      state->packets.emitted,
      iree_hal_amdgpu_aql_packet_control_barrier(IREE_HSA_FENCE_SCOPE_AGENT,
                                                 IREE_HSA_FENCE_SCOPE_AGENT),
      processor->packets.headers, processor->packets.setups);
  ++state->packets.emitted;
}

static void iree_hal_amdgpu_aql_block_processor_profile_emit_trace_stop(
    const iree_hal_amdgpu_aql_block_processor_profile_t* processor,
    uint64_t event_position,
    iree_hal_amdgpu_aql_block_processor_profile_state_t* state) {
  iree_hal_amdgpu_host_queue_emplace_profile_trace_stop_packet(
      processor->queue, event_position, processor->packets.first_payload_id,
      state->packets.emitted,
      iree_hal_amdgpu_aql_packet_control_barrier(IREE_HSA_FENCE_SCOPE_AGENT,
                                                 IREE_HSA_FENCE_SCOPE_AGENT),
      processor->packets.headers, processor->packets.setups);
  ++state->packets.emitted;
}

static iree_status_t
iree_hal_amdgpu_aql_block_processor_profile_emit_direct_dispatch(
    const iree_hal_amdgpu_aql_block_processor_profile_t* processor,
    const iree_hal_amdgpu_command_buffer_dispatch_command_t* dispatch_command,
    iree_hal_amdgpu_aql_block_processor_profile_state_t* state) {
  const iree_hal_amdgpu_aql_block_processor_dispatch_profile_t profile =
      iree_hal_amdgpu_aql_block_processor_profile_dispatch_profile(
          processor, state, /*recorded_packet_count=*/1,
          /*dispatch_packet_ordinal=*/state->packets.recorded);
  const iree_hal_amdgpu_host_queue_command_buffer_packet_flags_t
      profile_packet_flags =
          iree_hal_amdgpu_aql_block_processor_profile_packet_flags(profile);
  const iree_hal_amdgpu_aql_block_processor_dispatch_profile_flags_t
      profile_flags = profile.flags;

  if (iree_any_bit_set(
          profile_flags,
          IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_DISPATCH_PROFILE_FLAG_COUNTER_PACKETS)) {
    iree_hal_amdgpu_aql_block_processor_profile_emit_counter_starts(
        processor, profile.event_position,
        iree_hal_amdgpu_aql_block_processor_profile_packet_control(
            processor, state->packets.emitted, IREE_HSA_FENCE_SCOPE_NONE,
            iree_hal_amdgpu_aql_block_processor_profile_agent_barrier_packet_flags()),
        state);
  }
  if (iree_any_bit_set(
          profile_flags,
          IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_DISPATCH_PROFILE_FLAG_TRACE_PACKETS)) {
    iree_hal_amdgpu_aql_block_processor_profile_emit_trace_start(
        processor, profile.event_position,
        iree_hal_amdgpu_aql_block_processor_profile_agent_barrier_packet_flags(),
        state);
  }

  const uint32_t dispatch_packet_index = state->packets.emitted;
  iree_hal_amdgpu_aql_packet_t* packet =
      iree_hal_amdgpu_aql_block_processor_profile_packet(processor,
                                                         dispatch_packet_index);
  const iree_hsa_signal_t completion_signal =
      iree_hal_amdgpu_aql_block_processor_profile_completion_signal(
          processor, state, profile);
  uint8_t* kernarg_data = NULL;
  if (iree_hal_amdgpu_aql_block_processor_profile_command_uses_queue_kernargs(
          &dispatch_command->header)) {
    kernarg_data = processor->kernargs.blocks[state->kernargs.block].data;
  }
  iree_status_t status =
      iree_hal_amdgpu_aql_block_processor_profile_replay_dispatch_packet_body(
          processor, dispatch_command, packet, kernarg_data, completion_signal,
          &processor->packets.setups[dispatch_packet_index]);
  if (iree_status_is_ok(status)) {
    const iree_hal_amdgpu_host_queue_command_buffer_packet_flags_t packet_flags =
        iree_hal_amdgpu_aql_block_processor_profile_packet_flags_merge(
            iree_hal_amdgpu_aql_block_processor_profile_command_packet_flags(
                &dispatch_command->header),
            profile_packet_flags);
    const iree_hsa_fence_scope_t payload_acquire_scope =
        iree_hal_amdgpu_aql_block_processor_profile_payload_acquire_scope(
            processor, state, dispatch_packet_index, &dispatch_command->header,
            packet_flags);
    iree_hal_amdgpu_aql_block_processor_profile_emit_source(processor, profile,
                                                            state);
    processor->packets.headers[dispatch_packet_index] =
        iree_hal_amdgpu_aql_make_header(
            IREE_HSA_PACKET_TYPE_KERNEL_DISPATCH,
            iree_hal_amdgpu_aql_block_processor_profile_packet_control(
                processor, dispatch_packet_index, payload_acquire_scope,
                packet_flags));
    ++state->packets.emitted;
    ++state->packets.recorded;
    state->kernargs.block +=
        iree_hal_amdgpu_aql_block_processor_profile_dispatch_kernarg_block_count(
            dispatch_command);
    if (iree_any_bit_set(
            profile_flags,
            IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_DISPATCH_PROFILE_FLAG_TRACE_PACKETS)) {
      iree_hal_amdgpu_aql_block_processor_profile_emit_trace_stop(
          processor, profile.event_position, state);
    }
    if (iree_any_bit_set(
            profile_flags,
            IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_DISPATCH_PROFILE_FLAG_COUNTER_PACKETS)) {
      iree_hal_amdgpu_aql_block_processor_profile_emit_counter_read_stops(
          processor, profile.event_position, state);
    }
  }
  return status;
}

static iree_status_t
iree_hal_amdgpu_aql_block_processor_profile_emit_indirect_dispatch(
    const iree_hal_amdgpu_aql_block_processor_profile_t* processor,
    const iree_hal_amdgpu_command_buffer_dispatch_command_t* dispatch_command,
    iree_hal_amdgpu_aql_block_processor_profile_state_t* state) {
  const iree_hal_amdgpu_aql_block_processor_dispatch_profile_t profile =
      iree_hal_amdgpu_aql_block_processor_profile_dispatch_profile(
          processor, state, /*recorded_packet_count=*/2,
          /*dispatch_packet_ordinal=*/state->packets.recorded + 1u);
  const iree_hal_amdgpu_host_queue_command_buffer_packet_flags_t
      profile_packet_flags =
          iree_hal_amdgpu_aql_block_processor_profile_packet_flags(profile);
  const iree_hal_amdgpu_aql_block_processor_profile_packet_flag_pair_t
      command_packet_flags =
          iree_hal_amdgpu_aql_block_processor_profile_split_command_packet_flags(
              &dispatch_command->header);
  const iree_hal_amdgpu_aql_block_processor_dispatch_profile_flags_t
      profile_flags = profile.flags;

  const uint32_t patch_packet_index = state->packets.emitted++;
  if (iree_any_bit_set(
          profile_flags,
          IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_DISPATCH_PROFILE_FLAG_COUNTER_PACKETS)) {
    iree_hal_amdgpu_aql_block_processor_profile_emit_counter_starts(
        processor, profile.event_position,
        iree_hal_amdgpu_aql_block_processor_profile_packet_control(
            processor, state->packets.emitted, IREE_HSA_FENCE_SCOPE_NONE,
            iree_hal_amdgpu_aql_block_processor_profile_agent_barrier_packet_flags()),
        state);
  }
  if (iree_any_bit_set(
          profile_flags,
          IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_DISPATCH_PROFILE_FLAG_TRACE_PACKETS)) {
    iree_hal_amdgpu_aql_block_processor_profile_emit_trace_start(
        processor, profile.event_position,
        iree_hal_amdgpu_aql_block_processor_profile_agent_barrier_packet_flags(),
        state);
  }
  const uint32_t dispatch_packet_index = state->packets.emitted;
  iree_hal_amdgpu_aql_packet_t* patch_packet =
      iree_hal_amdgpu_aql_block_processor_profile_packet(processor,
                                                         patch_packet_index);
  iree_hal_amdgpu_aql_packet_t* dispatch_packet =
      iree_hal_amdgpu_aql_block_processor_profile_packet(processor,
                                                         dispatch_packet_index);
  const iree_hsa_signal_t completion_signal =
      iree_hal_amdgpu_aql_block_processor_profile_completion_signal(
          processor, state, profile);
  const iree_hal_amdgpu_host_queue_command_buffer_packet_flags_t
      dispatch_packet_flags =
          iree_hal_amdgpu_aql_block_processor_profile_packet_flags_merge(
              command_packet_flags.final, profile_packet_flags);
  const uint16_t dispatch_header = iree_hal_amdgpu_aql_make_header(
      IREE_HSA_PACKET_TYPE_KERNEL_DISPATCH,
      iree_hal_amdgpu_aql_block_processor_profile_packet_control(
          processor, dispatch_packet_index, IREE_HSA_FENCE_SCOPE_NONE,
          dispatch_packet_flags));

  iree_status_t status =
      iree_hal_amdgpu_aql_block_processor_profile_replay_indirect_dispatch_packet_bodies(
          processor, dispatch_command, patch_packet, dispatch_packet,
          processor->kernargs.blocks[state->kernargs.block].data,
          processor->kernargs.blocks[state->kernargs.block + 1].data,
          completion_signal, dispatch_header,
          &processor->packets.setups[patch_packet_index],
          &processor->packets.setups[dispatch_packet_index]);
  if (iree_status_is_ok(status)) {
    const iree_hal_amdgpu_host_queue_command_buffer_packet_flags_t patch_flags =
        iree_hal_amdgpu_aql_block_processor_profile_packet_flags_merge(
            iree_hal_amdgpu_aql_block_processor_profile_execution_barrier_packet_flags(),
            command_packet_flags.first);
    const iree_hsa_fence_scope_t patch_acquire_scope =
        iree_hal_amdgpu_aql_block_processor_profile_payload_acquire_scope(
            processor, state, patch_packet_index, &dispatch_command->header,
            patch_flags);
    iree_hal_amdgpu_aql_block_processor_profile_emit_source(processor, profile,
                                                            state);
    // The patch dispatch publishes the following dispatch packet header, so it
    // must retire before the CP observes that slot.
    processor->packets.headers[patch_packet_index] =
        iree_hal_amdgpu_aql_make_header(
            IREE_HSA_PACKET_TYPE_KERNEL_DISPATCH,
            iree_hal_amdgpu_aql_block_processor_profile_packet_control(
                processor, patch_packet_index, patch_acquire_scope,
                patch_flags));
    // The patch dispatch publishes the target dispatch header after it has
    // updated dynamic workgroup counts on device.
    processor->packets.headers[dispatch_packet_index] =
        IREE_HSA_PACKET_TYPE_INVALID;
    state->packets.emitted = dispatch_packet_index + /*dispatch packet=*/1;
    state->packets.recorded += 2;
    state->kernargs.block +=
        iree_hal_amdgpu_aql_block_processor_profile_dispatch_kernarg_block_count(
            dispatch_command);
    if (iree_any_bit_set(
            profile_flags,
            IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_DISPATCH_PROFILE_FLAG_TRACE_PACKETS)) {
      iree_hal_amdgpu_aql_block_processor_profile_emit_trace_stop(
          processor, profile.event_position, state);
    }
    if (iree_any_bit_set(
            profile_flags,
            IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_DISPATCH_PROFILE_FLAG_COUNTER_PACKETS)) {
      iree_hal_amdgpu_aql_block_processor_profile_emit_counter_read_stops(
          processor, profile.event_position, state);
    }
  }
  return status;
}

static iree_status_t iree_hal_amdgpu_aql_block_processor_profile_emit_dispatch(
    const iree_hal_amdgpu_aql_block_processor_profile_t* processor,
    const iree_hal_amdgpu_command_buffer_dispatch_command_t* dispatch_command,
    iree_hal_amdgpu_aql_block_processor_profile_state_t* state) {
  if (iree_hal_amdgpu_aql_block_processor_profile_dispatch_uses_indirect(
          dispatch_command)) {
    return iree_hal_amdgpu_aql_block_processor_profile_emit_indirect_dispatch(
        processor, dispatch_command, state);
  }
  return iree_hal_amdgpu_aql_block_processor_profile_emit_direct_dispatch(
      processor, dispatch_command, state);
}

static iree_status_t iree_hal_amdgpu_aql_block_processor_profile_emit_transfer(
    const iree_hal_amdgpu_aql_block_processor_profile_t* processor,
    const iree_hal_amdgpu_command_buffer_command_header_t* command,
    iree_hal_amdgpu_aql_block_processor_profile_state_t* state) {
  const uint32_t packet_index = state->packets.emitted;
  iree_hal_amdgpu_host_queue_command_buffer_packet_flags_t packet_flags =
      iree_hal_amdgpu_aql_block_processor_profile_command_packet_flags(command);
  if (iree_hal_amdgpu_aql_block_processor_profile_is_block_final(
          processor, state, /*recorded_packet_count=*/1) &&
      !iree_any_bit_set(
          processor->flags,
          IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_PROFILE_FLAG_DISPATCH_PACKETS |
              IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_PROFILE_FLAG_QUEUE_DEVICE_EVENT)) {
    packet_flags |= IREE_HAL_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_PACKET_FLAG_FINAL;
  }
  iree_hal_amdgpu_aql_packet_t* packet =
      iree_hal_amdgpu_aql_block_processor_profile_packet(processor,
                                                         packet_index);
  const iree_hsa_signal_t completion_signal = iree_hsa_signal_null();

  iree_status_t status = iree_ok_status();
  if (command->opcode == IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_FILL) {
    status =
        iree_hal_amdgpu_aql_block_processor_profile_replay_fill_packet_body(
            processor,
            (const iree_hal_amdgpu_command_buffer_fill_command_t*)command,
            packet, &processor->kernargs.blocks[state->kernargs.block],
            completion_signal, &processor->packets.setups[packet_index]);
  } else if (command->opcode == IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_COPY) {
    status =
        iree_hal_amdgpu_aql_block_processor_profile_replay_copy_packet_body(
            processor,
            (const iree_hal_amdgpu_command_buffer_copy_command_t*)command,
            packet, &processor->kernargs.blocks[state->kernargs.block],
            completion_signal, &processor->packets.setups[packet_index]);
  } else {
    const iree_hal_amdgpu_command_buffer_update_command_t* update_command =
        (const iree_hal_amdgpu_command_buffer_update_command_t*)command;
    const iree_host_size_t kernarg_length =
        iree_hal_amdgpu_aql_block_processor_profile_update_kernarg_length(
            update_command->length);
    const iree_host_size_t kernarg_block_count = iree_host_size_ceil_div(
        kernarg_length, sizeof(iree_hal_amdgpu_kernarg_block_t));
    status =
        iree_hal_amdgpu_aql_block_processor_profile_replay_update_packet_body(
            processor, update_command, packet,
            processor->kernargs.blocks[state->kernargs.block].data,
            kernarg_block_count * sizeof(iree_hal_amdgpu_kernarg_block_t),
            completion_signal, &processor->packets.setups[packet_index]);
    if (iree_status_is_ok(status)) {
      state->kernargs.block += (uint32_t)kernarg_block_count - 1u;
    }
  }
  if (iree_status_is_ok(status)) {
    processor->packets.headers[packet_index] = iree_hal_amdgpu_aql_make_header(
        IREE_HSA_PACKET_TYPE_KERNEL_DISPATCH,
        iree_hal_amdgpu_aql_block_processor_profile_packet_control(
            processor, packet_index,
            iree_hal_amdgpu_aql_block_processor_profile_payload_acquire_scope(
                processor, state, packet_index, command, packet_flags),
            packet_flags));
    ++state->packets.emitted;
    ++state->packets.recorded;
    ++state->kernargs.block;
  }
  return status;
}

void iree_hal_amdgpu_aql_block_processor_profile_initialize(
    const iree_hal_amdgpu_aql_block_processor_profile_t* params,
    iree_hal_amdgpu_aql_block_processor_profile_t* out_processor) {
  *out_processor = *params;
}

void iree_hal_amdgpu_aql_block_processor_profile_deinitialize(
    iree_hal_amdgpu_aql_block_processor_profile_t* processor) {
  memset(processor, 0, sizeof(*processor));
}

iree_status_t iree_hal_amdgpu_aql_block_processor_profile_invoke(
    const iree_hal_amdgpu_aql_block_processor_profile_t* processor,
    iree_hal_amdgpu_aql_block_processor_profile_result_t* out_result) {
  memset(out_result, 0, sizeof(*out_result));
  if (IREE_UNLIKELY(processor->profile.dispatches.count !=
                    processor->profile.dispatch_events.event_count)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AQL command-buffer block %" PRIu32
                            " has %u profile dispatch sidecars "
                            "but reserved %u profile dispatch events",
                            processor->block->block_ordinal,
                            processor->profile.dispatches.count,
                            processor->profile.dispatch_events.event_count);
  }
  const iree_hal_amdgpu_command_buffer_command_header_t* command =
      iree_hal_amdgpu_command_buffer_block_commands_const(processor->block);
  iree_hal_amdgpu_aql_block_processor_profile_state_t state = {0};
  bool reached_terminator = false;
  iree_status_t status = iree_ok_status();
  for (uint16_t i = 0; i < processor->block->command_count &&
                       iree_status_is_ok(status) && !reached_terminator;
       ++i) {
    switch (command->opcode) {
      case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_BARRIER:
        break;
      case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_DISPATCH: {
        const iree_hal_amdgpu_command_buffer_dispatch_command_t*
            dispatch_command =
                (const iree_hal_amdgpu_command_buffer_dispatch_command_t*)
                    command;
        status = iree_hal_amdgpu_aql_block_processor_profile_emit_dispatch(
            processor, dispatch_command, &state);
        break;
      }
      case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_FILL:
      case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_COPY:
      case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_UPDATE:
        status = iree_hal_amdgpu_aql_block_processor_profile_emit_transfer(
            processor, command, &state);
        break;
      case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_RETURN:
        out_result->terminator =
            IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_PROFILE_TERMINATOR_RETURN;
        reached_terminator = true;
        break;
      case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_BRANCH: {
        const iree_hal_amdgpu_command_buffer_branch_command_t* branch_command =
            (const iree_hal_amdgpu_command_buffer_branch_command_t*)command;
        out_result->terminator =
            IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_PROFILE_TERMINATOR_BRANCH;
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
                              processor->block->block_ordinal);
  }
  if (iree_status_is_ok(status) &&
      state.packets.recorded != processor->block->aql_packet_count) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AQL command-buffer block %" PRIu32 " consumed %" PRIu32
        " packets but declares %" PRIu32,
        processor->block->block_ordinal, state.packets.recorded,
        processor->block->aql_packet_count);
  }
  if (iree_status_is_ok(status) &&
      state.packets.emitted != processor->packets.count) {
    status =
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "AQL command-buffer block %" PRIu32 " emitted %" PRIu32
                         " payload packets but reserved %u",
                         processor->block->block_ordinal, state.packets.emitted,
                         processor->packets.count);
  }
  if (iree_status_is_ok(status) &&
      state.kernargs.block != processor->kernargs.count) {
    status =
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "AQL command-buffer block %" PRIu32 " emitted %" PRIu32
                         " kernarg blocks but reserved %" PRIu32,
                         processor->block->block_ordinal, state.kernargs.block,
                         processor->kernargs.count);
  }
  if (iree_status_is_ok(status) &&
      state.profile.event != processor->profile.dispatch_events.event_count) {
    status =
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "AQL command-buffer block %" PRIu32 " emitted %" PRIu32
                         " profile dispatch events but reserved %u",
                         processor->block->block_ordinal, state.profile.event,
                         processor->profile.dispatch_events.event_count);
  }
  out_result->packets.recorded = state.packets.recorded;
  out_result->packets.emitted = state.packets.emitted;
  out_result->kernargs.consumed = state.kernargs.block;
  out_result->profile.events = state.profile.event;
  return status;
}
