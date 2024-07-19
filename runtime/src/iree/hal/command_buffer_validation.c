// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/command_buffer_validation.h"

#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "iree/hal/allocator.h"
#include "iree/hal/buffer.h"
#include "iree/hal/detail.h"
#include "iree/hal/event.h"
#include "iree/hal/executable.h"
#include "iree/hal/pipeline_layout.h"
#include "iree/hal/resource.h"

// Returns success iff the queue supports the given command categories.
static iree_status_t iree_hal_command_buffer_validate_categories(
    const iree_hal_command_buffer_t* command_buffer,
    const iree_hal_command_buffer_validation_state_t* validation_state,
    iree_hal_command_category_t required_categories) {
  if (IREE_UNLIKELY(!validation_state->is_recording)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "command buffer is not in a recording state");
  }
  if (!iree_all_bits_set(command_buffer->allowed_categories,
                         required_categories)) {
#if IREE_STATUS_MODE
    iree_bitfield_string_temp_t temp0, temp1;
    iree_string_view_t required_categories_str =
        iree_hal_command_category_format(required_categories, &temp0);
    iree_string_view_t allowed_categories_str =
        iree_hal_command_category_format(command_buffer->allowed_categories,
                                         &temp1);
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "operation requires categories %.*s but command buffer only supports "
        "%.*s",
        (int)required_categories_str.size, required_categories_str.data,
        (int)allowed_categories_str.size, allowed_categories_str.data);
#else
    return iree_status_from_code(IREE_STATUS_FAILED_PRECONDITION);
#endif  // IREE_STATUS_MODE
  }
  return iree_ok_status();
}

// Returns success iff the buffer is compatible with the device.
static iree_status_t iree_hal_command_buffer_validate_buffer_compatibility(
    const iree_hal_command_buffer_t* command_buffer,
    const iree_hal_command_buffer_validation_state_t* validation_state,
    iree_hal_buffer_t* buffer,
    iree_hal_buffer_compatibility_t required_compatibility,
    iree_hal_buffer_usage_t intended_usage) {
  iree_hal_buffer_compatibility_t allowed_compatibility =
      iree_hal_allocator_query_buffer_compatibility(
          validation_state->device_allocator,
          (iree_hal_buffer_params_t){
              .type = iree_hal_buffer_memory_type(buffer),
              .usage = iree_hal_buffer_allowed_usage(buffer) & intended_usage,
          },
          iree_hal_buffer_allocation_size(buffer), /*out_params=*/NULL,
          /*out_allocation_size=*/NULL);
  if (!iree_all_bits_set(allowed_compatibility, required_compatibility)) {
#if IREE_STATUS_MODE
    // Buffer cannot be used on the queue for the given usage.
    iree_bitfield_string_temp_t temp0, temp1;
    iree_string_view_t allowed_usage_str = iree_hal_buffer_usage_format(
        iree_hal_buffer_allowed_usage(buffer), &temp0);
    iree_string_view_t intended_usage_str =
        iree_hal_buffer_usage_format(intended_usage, &temp1);
    return iree_make_status(
        IREE_STATUS_PERMISSION_DENIED,
        "requested buffer usage is not supported for the buffer on this queue; "
        "buffer allows %.*s, operation requires %.*s (allocator compatibility "
        "mismatch)",
        (int)allowed_usage_str.size, allowed_usage_str.data,
        (int)intended_usage_str.size, intended_usage_str.data);
#else
    return iree_status_from_code(IREE_STATUS_PERMISSION_DENIED);
#endif  // IREE_STATUS_MODE
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_command_buffer_validate_binding_requirements(
    iree_hal_command_buffer_t* command_buffer,
    const iree_hal_command_buffer_validation_state_t* validation_state,
    iree_hal_buffer_binding_t binding,
    iree_hal_buffer_binding_requirements_t requirements) {
  // Check for binding presence.
  if (requirements.usage == IREE_HAL_BUFFER_USAGE_NONE) {
    // Binding slot is unused and its value in the table is ignored.
    return iree_ok_status();
  } else if (!binding.buffer) {
    // Binding is used and required.
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "binding table slot requires a buffer but none was provided");
  }

  // Ensure the buffer is compatible with the device.
  // NOTE: this check is very slow! We may want to disable this outside of debug
  // mode or try to fast path it if the buffer is known-good.
  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_validate_buffer_compatibility(
      command_buffer, validation_state, binding.buffer,
      requirements.required_compatibility, requirements.usage));

  // Verify buffer compatibility.
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_usage(
      iree_hal_buffer_allowed_usage(binding.buffer), requirements.usage));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_access(
      iree_hal_buffer_allowed_access(binding.buffer), requirements.access));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_memory_type(
      iree_hal_buffer_memory_type(binding.buffer), requirements.type));

  // Verify that the binding range is valid and that any commands that reference
  // it are in range.
  if (requirements.max_byte_offset > 0) {
    iree_device_size_t end = binding.offset + requirements.max_byte_offset;
    if (IREE_UNLIKELY(end > binding.offset + binding.length)) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "at least one command attempted to access an "
          "address outside of the valid bound buffer "
          "range (length=%" PRIdsz ", end(inc)=%" PRIdsz
          ", binding offset=%" PRIdsz ", binding length=%" PRIdsz
          ", binding end(inc)=%" PRIdsz ")",
          requirements.max_byte_offset, end - 1, binding.offset, binding.length,
          binding.offset + binding.length - 1);
    }
  }

  // Ensure the offset and length have an alignment matching the value length.
  if (requirements.min_byte_alignment &&
      (binding.offset % requirements.min_byte_alignment) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "binding offset does not match the required "
                            "alignment of one or more command (offset=%" PRIdsz
                            ", min_byte_alignment=%" PRIhsz ")",
                            binding.offset, requirements.min_byte_alignment);
  }

  return iree_ok_status();
}

static iree_status_t iree_hal_command_buffer_validate_buffer_requirements(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_command_buffer_validation_state_t* validation_state,
    iree_hal_buffer_ref_t buffer_ref,
    iree_hal_buffer_binding_requirements_t requirements) {
  // If the buffer is directly specified we can validate it inline.
  if (buffer_ref.buffer) {
    iree_hal_buffer_binding_t binding = {
        .buffer = buffer_ref.buffer,
        .offset = 0,
        .length = buffer_ref.offset + buffer_ref.length,
    };
    return iree_hal_command_buffer_validate_binding_requirements(
        command_buffer, validation_state, binding, requirements);
  }

  // Ensure the buffer binding table slot is within range. Note that the
  // binding table provided may have more bindings than required so we only
  // verify against the declared command buffer capacity.
  if (IREE_UNLIKELY(buffer_ref.buffer_slot >=
                    command_buffer->binding_capacity)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "indirect buffer reference slot %u is out range of the declared "
        "binding capacity of the command buffer %u",
        buffer_ref.buffer_slot, command_buffer->binding_capacity);
  }
  command_buffer->binding_count =
      iree_max(command_buffer->binding_count, buffer_ref.buffer_slot + 1);

  // Merge the binding requirements into the table.
  iree_hal_buffer_binding_requirements_t* table_requirements =
      &validation_state->binding_requirements[buffer_ref.buffer_slot];
  table_requirements->required_compatibility |=
      requirements.required_compatibility;
  table_requirements->usage |= requirements.usage;
  table_requirements->access |= requirements.access;
  table_requirements->type |= requirements.type;
  table_requirements->max_byte_offset = iree_max(
      table_requirements->max_byte_offset, requirements.max_byte_offset);
  if (requirements.min_byte_alignment) {
    table_requirements->min_byte_alignment =
        iree_device_size_lcm(table_requirements->min_byte_alignment,
                             requirements.min_byte_alignment);
  }

  return iree_ok_status();
}

// Returns success iff the currently bound descriptor sets are valid for the
// given executable entry point.
static iree_status_t iree_hal_command_buffer_validate_dispatch_bindings(
    iree_hal_command_buffer_t* command_buffer,
    const iree_hal_command_buffer_validation_state_t* validation_state,
    iree_hal_executable_t* executable, int32_t entry_point) {
  // TODO(benvanik): validate buffers referenced have compatible memory types
  // and access rights.
  // TODO(benvanik): validate no aliasing between inputs/outputs.
  return iree_ok_status();
}

void iree_hal_command_buffer_initialize_validation(
    iree_hal_allocator_t* device_allocator,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_command_buffer_validation_state_t* out_validation_state) {
  out_validation_state->device_allocator = device_allocator;
  out_validation_state->is_recording = false;
  out_validation_state->debug_group_depth = 0;
}

iree_status_t iree_hal_command_buffer_begin_validation(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_command_buffer_validation_state_t* validation_state) {
  if (validation_state->is_recording) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "command buffer is already in a recording state");
  }
  validation_state->is_recording = true;
  return iree_ok_status();
}

iree_status_t iree_hal_command_buffer_end_validation(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_command_buffer_validation_state_t* validation_state) {
  if (validation_state->debug_group_depth != 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "unbalanced debug group depth (expected 0, is %d)",
                            validation_state->debug_group_depth);
  } else if (!validation_state->is_recording) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "command buffer is not in a recording state");
  }
  validation_state->is_recording = false;
  return iree_ok_status();
}

void iree_hal_command_buffer_begin_debug_group_validation(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_command_buffer_validation_state_t* validation_state,
    iree_string_view_t label, iree_hal_label_color_t label_color,
    const iree_hal_label_location_t* location) {
  ++validation_state->debug_group_depth;
}

void iree_hal_command_buffer_end_debug_group_validation(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_command_buffer_validation_state_t* validation_state) {
  --validation_state->debug_group_depth;
}

iree_status_t iree_hal_command_buffer_execution_barrier_validation(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_command_buffer_validation_state_t* validation_state,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_execution_barrier_flags_t flags,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  // NOTE: all command buffer types can perform this so no need to check.

  // TODO(benvanik): additional synchronization validation.

  return iree_ok_status();
}

iree_status_t iree_hal_command_buffer_signal_event_validation(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_command_buffer_validation_state_t* validation_state,
    iree_hal_event_t* event, iree_hal_execution_stage_t source_stage_mask) {
  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_validate_categories(
      command_buffer, validation_state, IREE_HAL_COMMAND_CATEGORY_DISPATCH));

  // TODO(benvanik): additional synchronization validation.

  return iree_ok_status();
}

iree_status_t iree_hal_command_buffer_reset_event_validation(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_command_buffer_validation_state_t* validation_state,
    iree_hal_event_t* event, iree_hal_execution_stage_t source_stage_mask) {
  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_validate_categories(
      command_buffer, validation_state, IREE_HAL_COMMAND_CATEGORY_DISPATCH));

  // TODO(benvanik): additional synchronization validation.

  return iree_ok_status();
}

iree_status_t iree_hal_command_buffer_wait_events_validation(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_command_buffer_validation_state_t* validation_state,
    iree_host_size_t event_count, const iree_hal_event_t** events,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_validate_categories(
      command_buffer, validation_state, IREE_HAL_COMMAND_CATEGORY_DISPATCH));

  // TODO(benvanik): additional synchronization validation.

  return iree_ok_status();
}

iree_status_t iree_hal_command_buffer_discard_buffer_validation(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_command_buffer_validation_state_t* validation_state,
    iree_hal_buffer_ref_t buffer_ref) {
  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_validate_categories(
      command_buffer, validation_state, IREE_HAL_COMMAND_CATEGORY_TRANSFER));

  const iree_hal_buffer_binding_requirements_t buffer_reqs = {
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
      .max_byte_offset = buffer_ref.offset + buffer_ref.length,
  };
  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_validate_buffer_requirements(
      command_buffer, validation_state, buffer_ref, buffer_reqs));

  return iree_ok_status();
}

iree_status_t iree_hal_command_buffer_fill_buffer_validation(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_command_buffer_validation_state_t* validation_state,
    iree_hal_buffer_ref_t target_ref, const void* pattern,
    iree_host_size_t pattern_length) {
  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_validate_categories(
      command_buffer, validation_state, IREE_HAL_COMMAND_CATEGORY_TRANSFER));

  // Ensure the value length is supported.
  if (pattern_length != 1 && pattern_length != 2 && pattern_length != 4) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "fill value length is not one of the supported "
                            "values (pattern_length=%" PRIhsz ")",
                            pattern_length);
  }

  if ((target_ref.offset % pattern_length) != 0 ||
      (target_ref.length % pattern_length) != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "binding offset and/or length do not match the required alignment of "
        "one or more command (offset=%" PRIdsz ", length=%" PRIdsz
        ", pattern_length=%" PRIhsz ")",
        target_ref.offset, target_ref.length, pattern_length);
  }

  const iree_hal_buffer_binding_requirements_t target_reqs = {
      .required_compatibility = IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_TRANSFER,
      .usage = IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET,
      .access = IREE_HAL_MEMORY_ACCESS_WRITE,
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
      .max_byte_offset = target_ref.offset + target_ref.length,
      .min_byte_alignment = pattern_length,
  };
  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_validate_buffer_requirements(
      command_buffer, validation_state, target_ref, target_reqs));

  return iree_ok_status();
}

iree_status_t iree_hal_command_buffer_update_buffer_validation(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_command_buffer_validation_state_t* validation_state,
    const void* source_buffer, iree_host_size_t source_offset,
    iree_hal_buffer_ref_t target_ref) {
  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_validate_categories(
      command_buffer, validation_state, IREE_HAL_COMMAND_CATEGORY_TRANSFER));

  const iree_hal_buffer_binding_requirements_t target_reqs = {
      .required_compatibility = IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_TRANSFER,
      .usage = IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET,
      .access = IREE_HAL_MEMORY_ACCESS_WRITE,
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
      .max_byte_offset = target_ref.offset + target_ref.length,
  };
  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_validate_buffer_requirements(
      command_buffer, validation_state, target_ref, target_reqs));

  return iree_ok_status();
}

iree_status_t iree_hal_command_buffer_copy_buffer_validation(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_command_buffer_validation_state_t* validation_state,
    iree_hal_buffer_ref_t source_ref, iree_hal_buffer_ref_t target_ref) {
  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_validate_categories(
      command_buffer, validation_state, IREE_HAL_COMMAND_CATEGORY_TRANSFER));

  if (source_ref.length != target_ref.length) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "copy spans between source and target must match "
                            "(source_length=%" PRIdsz ", target_length=%" PRIdsz
                            ")",
                            source_ref.length, target_ref.length);
  }

  const iree_hal_buffer_binding_requirements_t source_reqs = {
      .required_compatibility = IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_TRANSFER,
      .usage = IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE,
      .access = IREE_HAL_MEMORY_ACCESS_READ,
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
      .max_byte_offset = source_ref.offset + source_ref.length,
  };
  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_validate_buffer_requirements(
      command_buffer, validation_state, source_ref, source_reqs));

  const iree_hal_buffer_binding_requirements_t target_reqs = {
      .required_compatibility = IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_TRANSFER,
      .usage = IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET,
      .access = IREE_HAL_MEMORY_ACCESS_WRITE,
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
      .max_byte_offset = target_ref.offset + target_ref.length,
  };
  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_validate_buffer_requirements(
      command_buffer, validation_state, target_ref, target_reqs));

  // Check for overlap - just like memcpy we don't handle that.
  // Note that it's only undefined behavior if violated so we are ok if tricky
  // situations (subspans of subspans of binding table subranges etc) make it
  // through. This is only possible if both buffers are directly referenced -
  // we _could_ try to catch this for indirect references by stashing the
  // overlap check metadata for validation when the binding table is available
  // but that's too costly to be worth it.
  if (source_ref.buffer && target_ref.buffer) {
    if (iree_hal_buffer_test_overlap(source_ref.buffer, source_ref.offset,
                                     source_ref.length, target_ref.buffer,
                                     target_ref.offset, target_ref.length) !=
        IREE_HAL_BUFFER_OVERLAP_DISJOINT) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "source and target ranges overlap within the same buffer");
    }
  }

  return iree_ok_status();
}

iree_status_t iree_hal_command_buffer_collective_validation(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_command_buffer_validation_state_t* validation_state,
    iree_hal_channel_t* channel, iree_hal_collective_op_t op, uint32_t param,
    iree_hal_buffer_ref_t send_ref, iree_hal_buffer_ref_t recv_ref,
    iree_device_size_t element_count) {
  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_validate_categories(
      command_buffer, validation_state, IREE_HAL_COMMAND_CATEGORY_DISPATCH));

  if (op.kind > IREE_HAL_COLLECTIVE_KIND_MAX_VALUE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown collective operation");
  } else if (op.reduction > IREE_HAL_COLLECTIVE_REDUCTION_MAX_VALUE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown collective reduction");
  } else if (op.element_type > IREE_HAL_COLLECTIVE_ELEMENT_TYPE_MAX_VALUE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown collective element type");
  }
  enum iree_hal_collective_info_bits_t {
    IREE_HAL_COLLECTIVE_IS_REDUCTION = 1u << 0,
    IREE_HAL_COLLECTIVE_REQUIRES_SEND_BINDING = 1u << 1,
    IREE_HAL_COLLECTIVE_REQUIRES_RECV_BINDING = 1u << 2,
  };
  static const uint32_t
      info_bits_table[IREE_HAL_COLLECTIVE_KIND_MAX_VALUE + 1] = {
          [IREE_HAL_COLLECTIVE_KIND_ALL_GATHER] =
              IREE_HAL_COLLECTIVE_REQUIRES_SEND_BINDING |
              IREE_HAL_COLLECTIVE_REQUIRES_RECV_BINDING,
          [IREE_HAL_COLLECTIVE_KIND_ALL_REDUCE] =
              IREE_HAL_COLLECTIVE_IS_REDUCTION |
              IREE_HAL_COLLECTIVE_REQUIRES_SEND_BINDING |
              IREE_HAL_COLLECTIVE_REQUIRES_RECV_BINDING,
          [IREE_HAL_COLLECTIVE_KIND_ALL_TO_ALL] =
              IREE_HAL_COLLECTIVE_REQUIRES_SEND_BINDING |
              IREE_HAL_COLLECTIVE_REQUIRES_RECV_BINDING,
          [IREE_HAL_COLLECTIVE_KIND_BROADCAST] =
              IREE_HAL_COLLECTIVE_REQUIRES_SEND_BINDING |
              IREE_HAL_COLLECTIVE_REQUIRES_RECV_BINDING,
          [IREE_HAL_COLLECTIVE_KIND_REDUCE] =
              IREE_HAL_COLLECTIVE_IS_REDUCTION |
              IREE_HAL_COLLECTIVE_REQUIRES_SEND_BINDING |
              IREE_HAL_COLLECTIVE_REQUIRES_RECV_BINDING,
          [IREE_HAL_COLLECTIVE_KIND_REDUCE_SCATTER] =
              IREE_HAL_COLLECTIVE_IS_REDUCTION |
              IREE_HAL_COLLECTIVE_REQUIRES_SEND_BINDING |
              IREE_HAL_COLLECTIVE_REQUIRES_RECV_BINDING,
          [IREE_HAL_COLLECTIVE_KIND_SEND] =
              IREE_HAL_COLLECTIVE_REQUIRES_SEND_BINDING,
          [IREE_HAL_COLLECTIVE_KIND_RECV] =
              IREE_HAL_COLLECTIVE_REQUIRES_RECV_BINDING,
          [IREE_HAL_COLLECTIVE_KIND_SEND_RECV] =
              IREE_HAL_COLLECTIVE_REQUIRES_SEND_BINDING |
              IREE_HAL_COLLECTIVE_REQUIRES_RECV_BINDING,
      };
  const uint32_t info_bits = info_bits_table[op.kind];
  if (!(info_bits & IREE_HAL_COLLECTIVE_IS_REDUCTION) && op.reduction != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "reduction operation cannot be specified on a non-reducing collective");
  }

  // TODO(benvanik): add queue cap/usage for COLLECTIVE source/dest?
  if (info_bits & IREE_HAL_COLLECTIVE_REQUIRES_SEND_BINDING) {
    const iree_hal_buffer_binding_requirements_t send_reqs = {
        .required_compatibility = IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_DISPATCH,
        .usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE_READ,
        .access = IREE_HAL_MEMORY_ACCESS_READ,
        .type = IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
        .max_byte_offset = send_ref.offset + send_ref.length,
    };
    IREE_RETURN_IF_ERROR(iree_hal_command_buffer_validate_buffer_requirements(
        command_buffer, validation_state, send_ref, send_reqs));
  } else {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "collective operation does not use a send buffer binding");
  }

  if (info_bits & IREE_HAL_COLLECTIVE_REQUIRES_RECV_BINDING) {
    const iree_hal_buffer_binding_requirements_t recv_reqs = {
        .required_compatibility = IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_DISPATCH,
        .usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE_WRITE,
        .access = IREE_HAL_MEMORY_ACCESS_WRITE,
        .type = IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
        .max_byte_offset = recv_ref.offset + recv_ref.length,
    };
    IREE_RETURN_IF_ERROR(iree_hal_command_buffer_validate_buffer_requirements(
        command_buffer, validation_state, recv_ref, recv_reqs));
  } else {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "collective operation does not use a recv buffer binding");
  }

  return iree_ok_status();
}

iree_status_t iree_hal_command_buffer_push_constants_validation(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_command_buffer_validation_state_t* validation_state,
    iree_hal_pipeline_layout_t* pipeline_layout, iree_host_size_t offset,
    const void* values, iree_host_size_t values_length) {
  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_validate_categories(
      command_buffer, validation_state, IREE_HAL_COMMAND_CATEGORY_DISPATCH));

  if (IREE_UNLIKELY((values_length % 4) != 0)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "invalid alignment %" PRIhsz ", must be 4-byte aligned", values_length);
  }

  // TODO(benvanik): validate offset and value count with layout.

  return iree_ok_status();
}

iree_status_t iree_hal_command_buffer_push_descriptor_set_validation(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_command_buffer_validation_state_t* validation_state,
    iree_hal_pipeline_layout_t* pipeline_layout, uint32_t set,
    iree_host_size_t binding_count, const iree_hal_buffer_ref_t* bindings) {
  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_validate_categories(
      command_buffer, validation_state, IREE_HAL_COMMAND_CATEGORY_DISPATCH));

  // TODO(benvanik): validate set index.

  // TODO(benvanik): use pipeline layout to derive usage and access bits.
  // For now we conservatively say _any_ access may be performed (read/write).
  iree_hal_buffer_binding_requirements_t requirements = {
      .required_compatibility = IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_DISPATCH,
      .usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE,
      .access = IREE_HAL_MEMORY_ACCESS_ANY,
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
  };
  for (iree_host_size_t i = 0; i < binding_count; ++i) {
    // TODO(benvanik): validate binding ordinal against pipeline layout.
    requirements.max_byte_offset = bindings[i].offset + bindings[i].length;
    IREE_RETURN_IF_ERROR(
        iree_hal_command_buffer_validate_buffer_requirements(
            command_buffer, validation_state, bindings[i], requirements),
        "set[%u] binding[%u] (arg[%" PRIhsz "])", set, bindings[i].ordinal, i);
  }

  return iree_ok_status();
}

iree_status_t iree_hal_command_buffer_dispatch_validation(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_command_buffer_validation_state_t* validation_state,
    iree_hal_executable_t* executable, int32_t entry_point,
    uint32_t workgroup_x, uint32_t workgroup_y, uint32_t workgroup_z) {
  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_validate_categories(
      command_buffer, validation_state, IREE_HAL_COMMAND_CATEGORY_DISPATCH));
  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_validate_dispatch_bindings(
      command_buffer, validation_state, executable, entry_point));
  return iree_ok_status();
}

iree_status_t iree_hal_command_buffer_dispatch_indirect_validation(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_command_buffer_validation_state_t* validation_state,
    iree_hal_executable_t* executable, int32_t entry_point,
    iree_hal_buffer_ref_t workgroups_ref) {
  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_validate_categories(
      command_buffer, validation_state, IREE_HAL_COMMAND_CATEGORY_DISPATCH));

  if ((workgroups_ref.offset % sizeof(uint32_t)) != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "workgroup count offset does not match the required natural alignment "
        "of uint32_t (offset=%" PRIdsz ", min_byte_alignment=%" PRIhsz ")",
        workgroups_ref.offset, sizeof(uint32_t));
  } else if (workgroups_ref.length < 3 * sizeof(uint32_t)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "workgroup count buffer does not have the capacity "
                            "to store the required 3 uint32_t values "
                            "(length=%" PRIdsz ", min_length=%" PRIhsz ")",
                            workgroups_ref.length, 3 * sizeof(uint32_t));
  }

  const iree_hal_buffer_binding_requirements_t workgroups_reqs = {
      .required_compatibility = IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_DISPATCH,
      .usage = IREE_HAL_BUFFER_USAGE_DISPATCH_INDIRECT_PARAMS,
      .access = IREE_HAL_MEMORY_ACCESS_READ,
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
      .max_byte_offset = workgroups_ref.offset + workgroups_ref.length,
      .min_byte_alignment = sizeof(uint32_t),
  };
  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_validate_buffer_requirements(
      command_buffer, validation_state, workgroups_ref, workgroups_reqs));

  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_validate_dispatch_bindings(
      command_buffer, validation_state, executable, entry_point));

  return iree_ok_status();
}

iree_status_t iree_hal_command_buffer_binding_table_validation(
    iree_hal_command_buffer_t* command_buffer,
    const iree_hal_command_buffer_validation_state_t* validation_state,
    iree_hal_buffer_binding_table_t binding_table) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, command_buffer->binding_count);

  // NOTE: we only validate from [0, binding_count) and don't care if there are
  // extra bindings present.
  for (uint32_t i = 0; i < command_buffer->binding_count; ++i) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0,
        iree_hal_command_buffer_validate_binding_requirements(
            command_buffer, validation_state, binding_table.bindings[i],
            validation_state->binding_requirements[i]),
        "binding table slot %u", i);
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}
