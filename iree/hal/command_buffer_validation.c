// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <inttypes.h>

#include "iree/base/tracing.h"
#include "iree/hal/allocator.h"
#include "iree/hal/command_buffer.h"
#include "iree/hal/device.h"

typedef struct {
  iree_hal_resource_t resource;
  iree_hal_device_t* device;
  iree_hal_command_buffer_t* target_command_buffer;
  iree_hal_command_category_t allowed_categories;

  bool is_recording;
  // TODO(benvanik): current executable layout/descriptor set layout info.
  // TODO(benvanik): valid push constant bit ranges.
} iree_hal_validating_command_buffer_t;

static const iree_hal_command_buffer_vtable_t
    iree_hal_validating_command_buffer_vtable;

// Returns success iff the queue supports the given command categories.
static iree_status_t iree_hal_command_buffer_validate_categories(
    const iree_hal_validating_command_buffer_t* command_buffer,
    iree_hal_command_category_t required_categories) {
  if (!iree_all_bits_set(command_buffer->allowed_categories,
                         required_categories)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "operation requires categories %s but command buffer only supports %s",
        iree_hal_command_category_string(required_categories),
        iree_hal_command_category_string(command_buffer->allowed_categories));
  }
  return iree_ok_status();
}

// Returns success iff the buffer is compatible with the device.
static iree_status_t iree_hal_command_buffer_validate_buffer_compatibility(
    const iree_hal_validating_command_buffer_t* command_buffer,
    iree_hal_buffer_t* buffer,
    iree_hal_buffer_compatibility_t required_compatibility,
    iree_hal_buffer_usage_t intended_usage) {
  iree_hal_buffer_compatibility_t allowed_compatibility =
      iree_hal_allocator_query_buffer_compatibility(
          iree_hal_device_allocator(command_buffer->device),
          iree_hal_buffer_memory_type(buffer),
          iree_hal_buffer_allowed_usage(buffer), intended_usage,
          iree_hal_buffer_allocation_size(buffer));
  if (!iree_all_bits_set(allowed_compatibility, required_compatibility)) {
    // Buffer cannot be used on the queue for the given usage.
    return iree_make_status(
        IREE_STATUS_PERMISSION_DENIED,
        "requested buffer usage is not supported for the buffer on this queue; "
        "buffer allows %s, operation requires %s",
        iree_hal_buffer_usage_string(iree_hal_buffer_allowed_usage(buffer)),
        iree_hal_buffer_usage_string(intended_usage));
  }
  return iree_ok_status();
}

// Returns success iff the currently bound descriptor sets are valid for the
// given executable entry point.
static iree_status_t iree_hal_command_buffer_validate_dispatch_bindings(
    iree_hal_validating_command_buffer_t* command_buffer,
    iree_hal_executable_t* executable, int32_t entry_point) {
  // TODO(benvanik): validate buffers referenced have compatible memory types,
  // access rights, and usage.
  // TODO(benvanik): validate no aliasing between inputs/outputs.
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_hal_command_buffer_wrap_validation(
    iree_hal_device_t* device, iree_hal_command_buffer_t* target_command_buffer,
    iree_hal_command_buffer_t** out_command_buffer) {
  IREE_ASSERT_ARGUMENT(target_command_buffer);
  IREE_ASSERT_ARGUMENT(out_command_buffer);
  *out_command_buffer = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_validating_command_buffer_t* command_buffer = NULL;
  iree_status_t status =
      iree_allocator_malloc(iree_hal_device_host_allocator(device),
                            sizeof(*command_buffer), (void**)&command_buffer);
  if (iree_status_is_ok(status)) {
    iree_hal_resource_initialize(&iree_hal_validating_command_buffer_vtable,
                                 &command_buffer->resource);
    command_buffer->device = device;
    iree_hal_device_retain(command_buffer->device);
    command_buffer->target_command_buffer = target_command_buffer;
    iree_hal_command_buffer_retain(command_buffer->target_command_buffer);
    command_buffer->allowed_categories =
        iree_hal_command_buffer_allowed_categories(
            command_buffer->target_command_buffer);

    command_buffer->is_recording = false;
  }

  *out_command_buffer = (iree_hal_command_buffer_t*)command_buffer;
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_validating_command_buffer_destroy(
    iree_hal_command_buffer_t* base_command_buffer) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_validating_command_buffer_t* command_buffer =
      (iree_hal_validating_command_buffer_t*)base_command_buffer;
  iree_allocator_t host_allocator =
      iree_hal_device_host_allocator(command_buffer->device);
  iree_hal_command_buffer_release(command_buffer->target_command_buffer);
  iree_hal_device_release(command_buffer->device);
  iree_allocator_free(host_allocator, command_buffer);
  IREE_TRACE_ZONE_END(z0);
}

static iree_hal_command_category_t
iree_hal_validating_command_buffer_allowed_categories(
    const iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_validating_command_buffer_t* command_buffer =
      (iree_hal_validating_command_buffer_t*)base_command_buffer;
  return command_buffer->allowed_categories;
}

static iree_status_t iree_hal_validating_command_buffer_begin(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_validating_command_buffer_t* command_buffer =
      (iree_hal_validating_command_buffer_t*)base_command_buffer;

  if (command_buffer->is_recording) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "command buffer is already in a recording state");
  }
  command_buffer->is_recording = true;

  return iree_hal_command_buffer_begin(command_buffer->target_command_buffer);
}

static iree_status_t iree_hal_validating_command_buffer_end(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_validating_command_buffer_t* command_buffer =
      (iree_hal_validating_command_buffer_t*)base_command_buffer;

  if (!command_buffer->is_recording) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "command buffer is not in a recording state");
  }
  command_buffer->is_recording = false;

  return iree_hal_command_buffer_end(command_buffer->target_command_buffer);
}

static iree_status_t iree_hal_validating_command_buffer_execution_barrier(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_execution_barrier_flags_t flags,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  iree_hal_validating_command_buffer_t* command_buffer =
      (iree_hal_validating_command_buffer_t*)base_command_buffer;

  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_validate_categories(
      command_buffer, IREE_HAL_COMMAND_CATEGORY_ANY));

  // TODO(benvanik): additional synchronization validation.

  return iree_hal_command_buffer_execution_barrier(
      command_buffer->target_command_buffer, source_stage_mask,
      target_stage_mask, flags, memory_barrier_count, memory_barriers,
      buffer_barrier_count, buffer_barriers);
}

static iree_status_t iree_hal_validating_command_buffer_signal_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  iree_hal_validating_command_buffer_t* command_buffer =
      (iree_hal_validating_command_buffer_t*)base_command_buffer;

  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_validate_categories(
      command_buffer, IREE_HAL_COMMAND_CATEGORY_DISPATCH));

  // TODO(benvanik): additional synchronization validation.

  return iree_hal_command_buffer_signal_event(
      command_buffer->target_command_buffer, event, source_stage_mask);
}

static iree_status_t iree_hal_validating_command_buffer_reset_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  iree_hal_validating_command_buffer_t* command_buffer =
      (iree_hal_validating_command_buffer_t*)base_command_buffer;

  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_validate_categories(
      command_buffer, IREE_HAL_COMMAND_CATEGORY_DISPATCH));

  // TODO(benvanik): additional synchronization validation.

  return iree_hal_command_buffer_reset_event(
      command_buffer->target_command_buffer, event, source_stage_mask);
}

static iree_status_t iree_hal_validating_command_buffer_wait_events(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_host_size_t event_count, const iree_hal_event_t** events,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  iree_hal_validating_command_buffer_t* command_buffer =
      (iree_hal_validating_command_buffer_t*)base_command_buffer;

  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_validate_categories(
      command_buffer, IREE_HAL_COMMAND_CATEGORY_DISPATCH));

  // TODO(benvanik): additional synchronization validation.

  return iree_hal_command_buffer_wait_events(
      command_buffer->target_command_buffer, event_count, events,
      source_stage_mask, target_stage_mask, memory_barrier_count,
      memory_barriers, buffer_barrier_count, buffer_barriers);
}

static iree_status_t iree_hal_validating_command_buffer_discard_buffer(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_buffer_t* buffer) {
  iree_hal_validating_command_buffer_t* command_buffer =
      (iree_hal_validating_command_buffer_t*)base_command_buffer;

  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_validate_categories(
      command_buffer, IREE_HAL_COMMAND_CATEGORY_TRANSFER));

  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_memory_type(
      iree_hal_buffer_memory_type(buffer),
      IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE));

  return iree_hal_command_buffer_discard_buffer(
      command_buffer->target_command_buffer, buffer);
}

static iree_status_t iree_hal_validating_command_buffer_fill_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, const void* pattern,
    iree_host_size_t pattern_length) {
  iree_hal_validating_command_buffer_t* command_buffer =
      (iree_hal_validating_command_buffer_t*)base_command_buffer;

  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_validate_categories(
      command_buffer, IREE_HAL_COMMAND_CATEGORY_TRANSFER));
  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_validate_buffer_compatibility(
      command_buffer, target_buffer,
      IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_TRANSFER,
      IREE_HAL_BUFFER_USAGE_TRANSFER));

  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_memory_type(
      iree_hal_buffer_memory_type(target_buffer),
      IREE_HAL_MEMORY_ACCESS_WRITE));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_access(
      iree_hal_buffer_allowed_access(target_buffer),
      IREE_HAL_MEMORY_ACCESS_WRITE));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_usage(
      iree_hal_buffer_allowed_usage(target_buffer),
      IREE_HAL_BUFFER_USAGE_TRANSFER));
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
        "fill value (target_offset=%" PRIu64 ", length=%" PRIu64
        ", pattern_length=%zu)",
        target_offset, length, pattern_length);
  }

  return iree_hal_command_buffer_fill_buffer(
      command_buffer->target_command_buffer, target_buffer, target_offset,
      length, pattern, pattern_length);
}

static iree_status_t iree_hal_validating_command_buffer_update_buffer(
    iree_hal_command_buffer_t* base_command_buffer, const void* source_buffer,
    iree_host_size_t source_offset, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_device_size_t length) {
  iree_hal_validating_command_buffer_t* command_buffer =
      (iree_hal_validating_command_buffer_t*)base_command_buffer;

  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_validate_categories(
      command_buffer, IREE_HAL_COMMAND_CATEGORY_TRANSFER));
  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_validate_buffer_compatibility(
      command_buffer, target_buffer,
      IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_TRANSFER,
      IREE_HAL_BUFFER_USAGE_TRANSFER));

  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_memory_type(
      iree_hal_buffer_memory_type(target_buffer),
      IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_access(
      iree_hal_buffer_allowed_access(target_buffer),
      IREE_HAL_MEMORY_ACCESS_WRITE));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_usage(
      iree_hal_buffer_allowed_usage(target_buffer),
      IREE_HAL_BUFFER_USAGE_TRANSFER));
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_validate_range(target_buffer, target_offset, length));

  return iree_hal_command_buffer_update_buffer(
      command_buffer->target_command_buffer, source_buffer, source_offset,
      target_buffer, target_offset, length);
}

static iree_status_t iree_hal_validating_command_buffer_copy_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length) {
  iree_hal_validating_command_buffer_t* command_buffer =
      (iree_hal_validating_command_buffer_t*)base_command_buffer;

  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_validate_categories(
      command_buffer, IREE_HAL_COMMAND_CATEGORY_TRANSFER));
  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_validate_buffer_compatibility(
      command_buffer, source_buffer,
      IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_TRANSFER,
      IREE_HAL_BUFFER_USAGE_TRANSFER));
  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_validate_buffer_compatibility(
      command_buffer, target_buffer,
      IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_TRANSFER,
      IREE_HAL_BUFFER_USAGE_TRANSFER));

  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_access(
      iree_hal_buffer_allowed_access(source_buffer),
      IREE_HAL_MEMORY_ACCESS_READ));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_usage(
      iree_hal_buffer_allowed_usage(source_buffer),
      IREE_HAL_BUFFER_USAGE_TRANSFER));
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_validate_range(source_buffer, source_offset, length));

  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_usage(
      iree_hal_buffer_allowed_usage(target_buffer),
      IREE_HAL_BUFFER_USAGE_TRANSFER));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_access(
      iree_hal_buffer_allowed_access(target_buffer),
      IREE_HAL_MEMORY_ACCESS_WRITE));
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_validate_range(target_buffer, target_offset, length));

  // At least source or destination must be device-visible to enable
  // host->device, device->host, and device->device.
  // TODO(b/117338171): host->host copies.
  if (!iree_any_bit_set(iree_hal_buffer_memory_type(source_buffer),
                        IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE) &&
      !iree_any_bit_set(iree_hal_buffer_memory_type(target_buffer),
                        IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE)) {
    return iree_make_status(
        IREE_STATUS_PERMISSION_DENIED,
        "at least one buffer must be device-visible for a copy; "
        "source_buffer=%s, target_buffer=%s",
        iree_hal_memory_type_string(iree_hal_buffer_memory_type(source_buffer)),
        iree_hal_memory_type_string(
            iree_hal_buffer_memory_type(target_buffer)));
  }

  // Check for overlap - just like memcpy we don't handle that.
  if (iree_hal_buffer_test_overlap(source_buffer, source_offset, length,
                                   target_buffer, target_offset, length) !=
      IREE_HAL_BUFFER_OVERLAP_DISJOINT) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "source and target ranges overlap within the same buffer");
  }

  return iree_hal_command_buffer_copy_buffer(
      command_buffer->target_command_buffer, source_buffer, source_offset,
      target_buffer, target_offset, length);
}

static iree_status_t iree_hal_validating_command_buffer_push_constants(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_layout_t* executable_layout, iree_host_size_t offset,
    const void* values, iree_host_size_t values_length) {
  iree_hal_validating_command_buffer_t* command_buffer =
      (iree_hal_validating_command_buffer_t*)base_command_buffer;

  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_validate_categories(
      command_buffer, IREE_HAL_COMMAND_CATEGORY_DISPATCH));

  if (IREE_UNLIKELY((values_length % 4) != 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid alignment %zu, must be 4-byte aligned",
                            values_length);
  }

  // TODO(benvanik): validate offset and value count with layout.

  return iree_hal_command_buffer_push_constants(
      command_buffer->target_command_buffer, executable_layout, offset, values,
      values_length);
}

static iree_status_t iree_hal_validating_command_buffer_push_descriptor_set(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_layout_t* executable_layout, uint32_t set,
    iree_host_size_t binding_count,
    const iree_hal_descriptor_set_binding_t* bindings) {
  iree_hal_validating_command_buffer_t* command_buffer =
      (iree_hal_validating_command_buffer_t*)base_command_buffer;

  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_validate_categories(
      command_buffer, IREE_HAL_COMMAND_CATEGORY_DISPATCH));

  // TODO(benvanik): validate set index.
  // TODO(benvanik): validate binding_offset.
  // TODO(benvanik): validate bindings.

  return iree_hal_command_buffer_push_descriptor_set(
      command_buffer->target_command_buffer, executable_layout, set,
      binding_count, bindings);
}

static iree_status_t iree_hal_validating_command_buffer_bind_descriptor_set(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_layout_t* executable_layout, uint32_t set,
    iree_hal_descriptor_set_t* descriptor_set,
    iree_host_size_t dynamic_offset_count,
    const iree_device_size_t* dynamic_offsets) {
  iree_hal_validating_command_buffer_t* command_buffer =
      (iree_hal_validating_command_buffer_t*)base_command_buffer;

  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_validate_categories(
      command_buffer, IREE_HAL_COMMAND_CATEGORY_DISPATCH));

  // TODO(benvanik): validate set index.
  // TODO(benvanik): validate dynamic offsets (both count and offsets).

  return iree_hal_command_buffer_bind_descriptor_set(
      command_buffer->target_command_buffer, executable_layout, set,
      descriptor_set, dynamic_offset_count, dynamic_offsets);
}

static iree_status_t iree_hal_validating_command_buffer_dispatch(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_t* executable, int32_t entry_point,
    uint32_t workgroup_x, uint32_t workgroup_y, uint32_t workgroup_z) {
  iree_hal_validating_command_buffer_t* command_buffer =
      (iree_hal_validating_command_buffer_t*)base_command_buffer;

  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_validate_categories(
      command_buffer, IREE_HAL_COMMAND_CATEGORY_DISPATCH));
  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_validate_dispatch_bindings(
      command_buffer, executable, entry_point));

  return iree_hal_command_buffer_dispatch(command_buffer->target_command_buffer,
                                          executable, entry_point, workgroup_x,
                                          workgroup_y, workgroup_z);
}

static iree_status_t iree_hal_validating_command_buffer_dispatch_indirect(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_t* executable, int32_t entry_point,
    iree_hal_buffer_t* workgroups_buffer,
    iree_device_size_t workgroups_offset) {
  iree_hal_validating_command_buffer_t* command_buffer =
      (iree_hal_validating_command_buffer_t*)base_command_buffer;

  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_validate_categories(
      command_buffer, IREE_HAL_COMMAND_CATEGORY_DISPATCH));
  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_validate_buffer_compatibility(
      command_buffer, workgroups_buffer,
      IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_DISPATCH,
      IREE_HAL_BUFFER_USAGE_DISPATCH));

  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_memory_type(
      iree_hal_buffer_memory_type(workgroups_buffer),
      IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_access(
      iree_hal_buffer_allowed_access(workgroups_buffer),
      IREE_HAL_MEMORY_ACCESS_READ));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_usage(
      iree_hal_buffer_allowed_usage(workgroups_buffer),
      IREE_HAL_BUFFER_USAGE_DISPATCH));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_range(
      workgroups_buffer, workgroups_offset, sizeof(uint32_t) * 3));

  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_validate_dispatch_bindings(
      command_buffer, executable, entry_point));

  return iree_hal_command_buffer_dispatch_indirect(
      command_buffer->target_command_buffer, executable, entry_point,
      workgroups_buffer, workgroups_offset);
}

static const iree_hal_command_buffer_vtable_t
    iree_hal_validating_command_buffer_vtable = {
        .destroy = iree_hal_validating_command_buffer_destroy,
        .allowed_categories =
            iree_hal_validating_command_buffer_allowed_categories,
        .begin = iree_hal_validating_command_buffer_begin,
        .end = iree_hal_validating_command_buffer_end,
        .execution_barrier =
            iree_hal_validating_command_buffer_execution_barrier,
        .signal_event = iree_hal_validating_command_buffer_signal_event,
        .reset_event = iree_hal_validating_command_buffer_reset_event,
        .wait_events = iree_hal_validating_command_buffer_wait_events,
        .discard_buffer = iree_hal_validating_command_buffer_discard_buffer,
        .fill_buffer = iree_hal_validating_command_buffer_fill_buffer,
        .update_buffer = iree_hal_validating_command_buffer_update_buffer,
        .copy_buffer = iree_hal_validating_command_buffer_copy_buffer,
        .push_constants = iree_hal_validating_command_buffer_push_constants,
        .push_descriptor_set =
            iree_hal_validating_command_buffer_push_descriptor_set,
        .bind_descriptor_set =
            iree_hal_validating_command_buffer_bind_descriptor_set,
        .dispatch = iree_hal_validating_command_buffer_dispatch,
        .dispatch_indirect =
            iree_hal_validating_command_buffer_dispatch_indirect,
};
