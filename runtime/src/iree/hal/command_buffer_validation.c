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

#include "iree/base/tracing.h"
#include "iree/hal/allocator.h"
#include "iree/hal/buffer.h"
#include "iree/hal/detail.h"
#include "iree/hal/device.h"
#include "iree/hal/event.h"
#include "iree/hal/executable.h"
#include "iree/hal/pipeline_layout.h"
#include "iree/hal/resource.h"

// Returns success iff the queue supports the given command categories.
static iree_status_t iree_hal_command_buffer_validate_categories(
    const iree_hal_command_buffer_t* command_buffer,
    iree_hal_command_buffer_validation_state_t* validation_state,
    iree_hal_command_category_t required_categories) {
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
    iree_hal_command_buffer_validation_state_t* validation_state,
    iree_hal_buffer_t* buffer,
    iree_hal_buffer_compatibility_t required_compatibility,
    iree_hal_buffer_usage_t intended_usage) {
  iree_hal_buffer_compatibility_t allowed_compatibility =
      iree_hal_allocator_query_buffer_compatibility(
          iree_hal_device_allocator(validation_state->device),
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

// Returns success iff the currently bound descriptor sets are valid for the
// given executable entry point.
static iree_status_t iree_hal_command_buffer_validate_dispatch_bindings(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_command_buffer_validation_state_t* validation_state,
    iree_hal_executable_t* executable, int32_t entry_point) {
  // TODO(benvanik): validate buffers referenced have compatible memory types
  // and access rights.
  // TODO(benvanik): validate no aliasing between inputs/outputs.
  return iree_ok_status();
}

void iree_hal_command_buffer_initialize_validation(
    iree_hal_device_t* device, iree_hal_command_buffer_t* command_buffer,
    iree_hal_command_buffer_validation_state_t* out_validation_state) {
  out_validation_state->device = device;
  out_validation_state->is_recording = false;
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
    iree_hal_buffer_t* buffer) {
  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_validate_categories(
      command_buffer, validation_state, IREE_HAL_COMMAND_CATEGORY_TRANSFER));

  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_memory_type(
      iree_hal_buffer_memory_type(buffer),
      IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE));

  return iree_ok_status();
}

iree_status_t iree_hal_command_buffer_fill_buffer_validation(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_command_buffer_validation_state_t* validation_state,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, const void* pattern,
    iree_host_size_t pattern_length) {
  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_validate_categories(
      command_buffer, validation_state, IREE_HAL_COMMAND_CATEGORY_TRANSFER));
  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_validate_buffer_compatibility(
      command_buffer, validation_state, target_buffer,
      IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_TRANSFER,
      IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET));

  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_memory_type(
      iree_hal_buffer_memory_type(target_buffer),
      IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_access(
      iree_hal_buffer_allowed_access(target_buffer),
      IREE_HAL_MEMORY_ACCESS_WRITE));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_usage(
      iree_hal_buffer_allowed_usage(target_buffer),
      IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET));
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_validate_range(target_buffer, target_offset, length));

  // Ensure the value length is supported.
  if (pattern_length != 1 && pattern_length != 2 && pattern_length != 4) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "fill value length is not one of the supported "
                            "values (pattern_length=%zu)",
                            pattern_length);
  }

  // Ensure the offset and length have an alignment matching the value length.
  if ((target_offset % pattern_length) != 0 || (length % pattern_length) != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "fill offset and/or length do not match the natural alignment of the "
        "fill value (target_offset=%" PRIdsz ", length=%" PRIdsz
        ", pattern_length=%zu)",
        target_offset, length, pattern_length);
  }

  return iree_ok_status();
}

iree_status_t iree_hal_command_buffer_update_buffer_validation(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_command_buffer_validation_state_t* validation_state,
    const void* source_buffer, iree_host_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length) {
  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_validate_categories(
      command_buffer, validation_state, IREE_HAL_COMMAND_CATEGORY_TRANSFER));
  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_validate_buffer_compatibility(
      command_buffer, validation_state, target_buffer,
      IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_TRANSFER,
      IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET));

  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_memory_type(
      iree_hal_buffer_memory_type(target_buffer),
      IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_access(
      iree_hal_buffer_allowed_access(target_buffer),
      IREE_HAL_MEMORY_ACCESS_WRITE));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_usage(
      iree_hal_buffer_allowed_usage(target_buffer),
      IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET));
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_validate_range(target_buffer, target_offset, length));

  return iree_ok_status();
}

iree_status_t iree_hal_command_buffer_copy_buffer_validation(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_command_buffer_validation_state_t* validation_state,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length) {
  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_validate_categories(
      command_buffer, validation_state, IREE_HAL_COMMAND_CATEGORY_TRANSFER));
  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_validate_buffer_compatibility(
      command_buffer, validation_state, source_buffer,
      IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_TRANSFER,
      IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE));
  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_validate_buffer_compatibility(
      command_buffer, validation_state, target_buffer,
      IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_TRANSFER,
      IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET));

  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_access(
      iree_hal_buffer_allowed_access(source_buffer),
      IREE_HAL_MEMORY_ACCESS_READ));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_usage(
      iree_hal_buffer_allowed_usage(source_buffer),
      IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE));
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_validate_range(source_buffer, source_offset, length));

  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_usage(
      iree_hal_buffer_allowed_usage(target_buffer),
      IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_access(
      iree_hal_buffer_allowed_access(target_buffer),
      IREE_HAL_MEMORY_ACCESS_WRITE));
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_validate_range(target_buffer, target_offset, length));

  // At least source or destination must be device-visible to enable
  // host->device, device->host, and device->device.
  // TODO(benvanik): host->host copies.
  if (!iree_any_bit_set(iree_hal_buffer_memory_type(source_buffer),
                        IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE) &&
      !iree_any_bit_set(iree_hal_buffer_memory_type(target_buffer),
                        IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE)) {
#if IREE_STATUS_MODE
    iree_bitfield_string_temp_t temp0, temp1;
    iree_string_view_t source_memory_type_str = iree_hal_memory_type_format(
        iree_hal_buffer_memory_type(source_buffer), &temp0);
    iree_string_view_t target_memory_type_str = iree_hal_memory_type_format(
        iree_hal_buffer_memory_type(target_buffer), &temp1);
    return iree_make_status(
        IREE_STATUS_PERMISSION_DENIED,
        "at least one buffer must be device-visible for a copy; "
        "source_buffer=%.*s, target_buffer=%.*s",
        (int)source_memory_type_str.size, source_memory_type_str.data,
        (int)target_memory_type_str.size, target_memory_type_str.data);
#else
    return iree_status_from_code(IREE_STATUS_PERMISSION_DENIED);
#endif  // IREE_STATUS_MODE
  }

  // Check for overlap - just like memcpy we don't handle that.
  if (iree_hal_buffer_test_overlap(source_buffer, source_offset, length,
                                   target_buffer, target_offset, length) !=
      IREE_HAL_BUFFER_OVERLAP_DISJOINT) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "source and target ranges overlap within the same buffer");
  }

  return iree_ok_status();
}

iree_status_t iree_hal_command_buffer_collective_validation(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_command_buffer_validation_state_t* validation_state,
    iree_hal_channel_t* channel, iree_hal_collective_op_t op, uint32_t param,
    iree_hal_buffer_binding_t send_binding,
    iree_hal_buffer_binding_t recv_binding, iree_device_size_t element_count) {
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
    if (!send_binding.buffer) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "collective operation requires a send buffer binding");
    } else {
      IREE_RETURN_IF_ERROR(
          iree_hal_command_buffer_validate_buffer_compatibility(
              command_buffer, validation_state, send_binding.buffer,
              IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_DISPATCH,
              IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE_READ));
    }
  } else {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "collective operation does not use a send buffer binding");
  }

  if (info_bits & IREE_HAL_COLLECTIVE_REQUIRES_RECV_BINDING) {
    if (!recv_binding.buffer) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "collective operation requires a recv buffer binding");
    } else {
      IREE_RETURN_IF_ERROR(
          iree_hal_command_buffer_validate_buffer_compatibility(
              command_buffer, validation_state, recv_binding.buffer,
              IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_DISPATCH,
              IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE_WRITE));
    }
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
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid alignment %zu, must be 4-byte aligned",
                            values_length);
  }

  // TODO(benvanik): validate offset and value count with layout.

  return iree_ok_status();
}

iree_status_t iree_hal_command_buffer_push_descriptor_set_validation(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_command_buffer_validation_state_t* validation_state,
    iree_hal_pipeline_layout_t* pipeline_layout, uint32_t set,
    iree_host_size_t binding_count,
    const iree_hal_descriptor_set_binding_t* bindings) {
  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_validate_categories(
      command_buffer, validation_state, IREE_HAL_COMMAND_CATEGORY_DISPATCH));

  // TODO(benvanik): validate set index.

  // TODO(benvanik): allow indirect bindings on primary command buffers?
  const bool has_binding_table =
      iree_all_bits_set(iree_hal_command_buffer_mode(command_buffer),
                        IREE_HAL_COMMAND_BUFFER_MODE_NESTED);
  for (iree_host_size_t i = 0; i < binding_count; ++i) {
    const iree_hal_descriptor_set_binding_t* binding = &bindings[i];
    // TODO(benvanik): validate binding index.
    // TODO(benvanik): validate binding buffer parameters/access.
    // TODO(benvanik): validate binding range (if possible).

    // Validate that indirect buffer references are supported and in bounds.
    if (!binding->buffer) {
      if (!has_binding_table) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "bindings[%" PRIhsz
                                "] is indirect but the command buffer does not "
                                "support binding tables",
                                i);
      } else if (binding->buffer_slot >= command_buffer->binding_capacity) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "bindings[%" PRIhsz
            "] references binding table slot %u but table capacity is %u",
            i, binding->buffer_slot, command_buffer->binding_capacity);
      }
    }
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
    iree_hal_buffer_t* workgroups_buffer,
    iree_device_size_t workgroups_offset) {
  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_validate_categories(
      command_buffer, validation_state, IREE_HAL_COMMAND_CATEGORY_DISPATCH));
  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_validate_buffer_compatibility(
      command_buffer, validation_state, workgroups_buffer,
      IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_DISPATCH,
      IREE_HAL_BUFFER_USAGE_DISPATCH_INDIRECT_PARAMS));

  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_memory_type(
      iree_hal_buffer_memory_type(workgroups_buffer),
      IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_access(
      iree_hal_buffer_allowed_access(workgroups_buffer),
      IREE_HAL_MEMORY_ACCESS_READ));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_usage(
      iree_hal_buffer_allowed_usage(workgroups_buffer),
      IREE_HAL_BUFFER_USAGE_DISPATCH_INDIRECT_PARAMS));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_range(
      workgroups_buffer, workgroups_offset, sizeof(uint32_t) * 3));

  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_validate_dispatch_bindings(
      command_buffer, validation_state, executable, entry_point));

  return iree_ok_status();
}

iree_status_t iree_hal_command_buffer_execute_commands_validation(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_command_buffer_validation_state_t* validation_state,
    iree_hal_command_buffer_t* commands,
    iree_hal_buffer_binding_table_t binding_table) {
  if (iree_all_bits_set(command_buffer->mode,
                        IREE_HAL_COMMAND_BUFFER_MODE_NESTED)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "command buffers can only be nested one level "
                            "(nested cannot execute nested)");
  }
  if (!iree_all_bits_set(commands->mode, IREE_HAL_COMMAND_BUFFER_MODE_NESTED)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "only nested command buffers can be executed as "
                            "part of a primary command buffer");
  }

  // TODO(benvanik): validate bindings as with push descriptor sets.

  return iree_ok_status();
}
