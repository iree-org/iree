// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/command_buffer.h"

#include <stddef.h>

#include "iree/base/api.h"
#include "iree/base/tracing.h"
#include "iree/hal/command_buffer_validation.h"
#include "iree/hal/detail.h"
#include "iree/hal/device.h"
#include "iree/hal/resource.h"

// Conditionally executes an expression based on whether command buffer
// validation was enabled in the build and the command buffer wants validation.
#if IREE_HAL_COMMAND_BUFFER_VALIDATION_ENABLE
#define IF_VALIDATING(command_buffer, expr)                                  \
  if (((command_buffer)->mode & IREE_HAL_COMMAND_BUFFER_MODE_UNVALIDATED) == \
      0) {                                                                   \
    expr;                                                                    \
  }
#define VALIDATION_STATE(command_buffer) (&(command_buffer)->validation)
#else
#define IF_VALIDATING(command_buffer, expr)
#define VALIDATION_STATE(command_buffer) \
  ((iree_hal_command_buffer_validation_state_t*)NULL)
#endif  // IREE_HAL_COMMAND_BUFFER_VALIDATION_ENABLE

#define _VTABLE_DISPATCH(command_buffer, method_name) \
  IREE_HAL_VTABLE_DISPATCH(command_buffer, iree_hal_command_buffer, method_name)

//===----------------------------------------------------------------------===//
// String utils
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_string_view_t iree_hal_collective_op_format(
    const iree_hal_collective_op_t* op, iree_bitfield_string_temp_t* out_temp) {
  static const iree_string_view_t
      kind_names[IREE_HAL_COLLECTIVE_KIND_MAX_VALUE + 1] = {
          [IREE_HAL_COLLECTIVE_KIND_ALL_GATHER] = IREE_SVL("all_gather"),
          [IREE_HAL_COLLECTIVE_KIND_ALL_REDUCE] = IREE_SVL("all_reduce"),
          [IREE_HAL_COLLECTIVE_KIND_ALL_TO_ALL] = IREE_SVL("all_to_all"),
          [IREE_HAL_COLLECTIVE_KIND_BROADCAST] = IREE_SVL("broadcast"),
          [IREE_HAL_COLLECTIVE_KIND_REDUCE] = IREE_SVL("reduce"),
          [IREE_HAL_COLLECTIVE_KIND_REDUCE_SCATTER] =
              IREE_SVL("reduce_scatter"),
          [IREE_HAL_COLLECTIVE_KIND_SEND] = IREE_SVL("send"),
          [IREE_HAL_COLLECTIVE_KIND_RECV] = IREE_SVL("recv"),
      };
  static const iree_string_view_t
      reduction_names[IREE_HAL_COLLECTIVE_REDUCTION_MAX_VALUE + 1] = {
          [IREE_HAL_COLLECTIVE_REDUCTION_SUM] = IREE_SVL("sum"),
          [IREE_HAL_COLLECTIVE_REDUCTION_PRODUCT] = IREE_SVL("product"),
          [IREE_HAL_COLLECTIVE_REDUCTION_MINIMUM] = IREE_SVL("minimum"),
          [IREE_HAL_COLLECTIVE_REDUCTION_MAXIMUM] = IREE_SVL("maximum"),
          [IREE_HAL_COLLECTIVE_REDUCTION_AVERAGE] = IREE_SVL("average"),
      };
  static const iree_string_view_t
      element_type_names[IREE_HAL_COLLECTIVE_ELEMENT_TYPE_MAX_VALUE + 1] = {
          [IREE_HAL_COLLECTIVE_ELEMENT_TYPE_SINT_8] = IREE_SVL("si8"),
          [IREE_HAL_COLLECTIVE_ELEMENT_TYPE_UINT_8] = IREE_SVL("ui8"),
          [IREE_HAL_COLLECTIVE_ELEMENT_TYPE_SINT_16] = IREE_SVL("si16"),
          [IREE_HAL_COLLECTIVE_ELEMENT_TYPE_UINT_16] = IREE_SVL("ui16"),
          [IREE_HAL_COLLECTIVE_ELEMENT_TYPE_SINT_32] = IREE_SVL("si32"),
          [IREE_HAL_COLLECTIVE_ELEMENT_TYPE_UINT_32] = IREE_SVL("ui32"),
          [IREE_HAL_COLLECTIVE_ELEMENT_TYPE_SINT_64] = IREE_SVL("si64"),
          [IREE_HAL_COLLECTIVE_ELEMENT_TYPE_UINT_64] = IREE_SVL("ui64"),
          [IREE_HAL_COLLECTIVE_ELEMENT_TYPE_FLOAT_16] = IREE_SVL("f16"),
          [IREE_HAL_COLLECTIVE_ELEMENT_TYPE_FLOAT_32] = IREE_SVL("f32"),
          [IREE_HAL_COLLECTIVE_ELEMENT_TYPE_FLOAT_64] = IREE_SVL("f64"),
          [IREE_HAL_COLLECTIVE_ELEMENT_TYPE_BFLOAT_16] = IREE_SVL("bf16"),
      };
  IREE_ASSERT_LE((int)op->kind, IREE_HAL_COLLECTIVE_KIND_MAX_VALUE);
  IREE_ASSERT_LE((int)op->reduction, IREE_HAL_COLLECTIVE_REDUCTION_MAX_VALUE);
  IREE_ASSERT_LE((int)op->element_type,
                 IREE_HAL_COLLECTIVE_ELEMENT_TYPE_MAX_VALUE);
  const iree_string_view_t kind_name = kind_names[(int)op->kind];
  const iree_string_view_t element_type_name =
      element_type_names[(int)op->element_type];
  int length = 0;
  switch (op->kind) {
    default:
      length = snprintf(out_temp->buffer, sizeof(out_temp->buffer),
                        "iree_hal_collective_%.*s_%.*s", (int)kind_name.size,
                        kind_name.data, (int)element_type_name.size,
                        element_type_name.data);
      break;
    case IREE_HAL_COLLECTIVE_KIND_ALL_REDUCE:
    case IREE_HAL_COLLECTIVE_KIND_REDUCE:
    case IREE_HAL_COLLECTIVE_KIND_REDUCE_SCATTER: {
      const iree_string_view_t reduction_name =
          reduction_names[(int)op->reduction];
      length = snprintf(out_temp->buffer, sizeof(out_temp->buffer),
                        "iree_hal_collective_%.*s_%.*s_%.*s",
                        (int)kind_name.size, kind_name.data,
                        (int)reduction_name.size, reduction_name.data,
                        (int)element_type_name.size, element_type_name.data);
      break;
    }
  }
  return length > 0 ? iree_make_string_view(out_temp->buffer, length)
                    : IREE_SV("iree_hal_collective_unknown");
}

IREE_API_EXPORT iree_string_view_t
iree_hal_command_buffer_mode_format(iree_hal_command_buffer_mode_t value,
                                    iree_bitfield_string_temp_t* out_temp) {
  static const iree_bitfield_string_mapping_t mappings[] = {
      {IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT, IREE_SVL("ONE_SHOT")},
      {IREE_HAL_COMMAND_BUFFER_MODE_ALLOW_INLINE_EXECUTION,
       IREE_SVL("ALLOW_INLINE_EXECUTION")},
      {IREE_HAL_COMMAND_BUFFER_MODE_UNVALIDATED, IREE_SVL("UNVALIDATED")},
  };
  return iree_bitfield_format_inline(value, IREE_ARRAYSIZE(mappings), mappings,
                                     out_temp);
}

IREE_API_EXPORT iree_string_view_t iree_hal_command_category_format(
    iree_hal_command_category_t value, iree_bitfield_string_temp_t* out_temp) {
  static const iree_bitfield_string_mapping_t mappings[] = {
      // Combined:
      {IREE_HAL_COMMAND_CATEGORY_ANY, IREE_SVL("ANY")},
      // Separate:
      {IREE_HAL_COMMAND_CATEGORY_TRANSFER, IREE_SVL("TRANSFER")},
      {IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_SVL("DISPATCH")},
  };
  return iree_bitfield_format_inline(value, IREE_ARRAYSIZE(mappings), mappings,
                                     out_temp);
}

//===----------------------------------------------------------------------===//
// iree_hal_collective_element_t
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_device_size_t iree_hal_collective_element_byte_count(
    iree_hal_collective_element_type_t element_type) {
  switch (element_type) {
    case IREE_HAL_COLLECTIVE_ELEMENT_TYPE_SINT_8:
    case IREE_HAL_COLLECTIVE_ELEMENT_TYPE_UINT_8:
      return 1;
    case IREE_HAL_COLLECTIVE_ELEMENT_TYPE_SINT_16:
    case IREE_HAL_COLLECTIVE_ELEMENT_TYPE_UINT_16:
    case IREE_HAL_COLLECTIVE_ELEMENT_TYPE_FLOAT_16:
    case IREE_HAL_COLLECTIVE_ELEMENT_TYPE_BFLOAT_16:
      return 2;
    case IREE_HAL_COLLECTIVE_ELEMENT_TYPE_SINT_32:
    case IREE_HAL_COLLECTIVE_ELEMENT_TYPE_UINT_32:
    case IREE_HAL_COLLECTIVE_ELEMENT_TYPE_FLOAT_32:
      return 4;
    case IREE_HAL_COLLECTIVE_ELEMENT_TYPE_SINT_64:
    case IREE_HAL_COLLECTIVE_ELEMENT_TYPE_UINT_64:
    case IREE_HAL_COLLECTIVE_ELEMENT_TYPE_FLOAT_64:
      return 8;
    default:
      IREE_ASSERT(false, "unhandled element type for collective op");
      return 0;
  }
}

//===----------------------------------------------------------------------===//
// iree_hal_command_buffer_t
//===----------------------------------------------------------------------===//

IREE_HAL_API_RETAIN_RELEASE(command_buffer);

IREE_API_EXPORT void iree_hal_command_buffer_initialize(
    iree_hal_device_t* device, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    const iree_hal_command_buffer_vtable_t* vtable,
    iree_hal_command_buffer_t* command_buffer) {
  iree_hal_resource_initialize(vtable, &command_buffer->resource);
  command_buffer->mode = mode;
  command_buffer->allowed_categories = command_categories;
  command_buffer->queue_affinity = queue_affinity;
  command_buffer->binding_capacity = binding_capacity;

  // Perform initialization validation after we allocate/initialize the concrete
  // implementation.
  IF_VALIDATING(command_buffer, {
    iree_hal_command_buffer_initialize_validation(
        device, command_buffer, VALIDATION_STATE(command_buffer));
  });
}

IREE_API_EXPORT iree_status_t iree_hal_command_buffer_create(
    iree_hal_device_t* device, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_hal_command_buffer_t** out_command_buffer) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_command_buffer);
  *out_command_buffer = NULL;

  if (iree_all_bits_set(mode,
                        IREE_HAL_COMMAND_BUFFER_MODE_ALLOW_INLINE_EXECUTION)) {
    if (!iree_all_bits_set(mode, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "inline command buffers must be one-shot");
    } else if (iree_all_bits_set(mode, IREE_HAL_COMMAND_BUFFER_MODE_NESTED)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "inline command buffers cannot be nested");
    }
  }
  if (binding_capacity > 0 &&
      !iree_all_bits_set(mode, IREE_HAL_COMMAND_BUFFER_MODE_NESTED)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "command buffer bindings are only supported for "
                            "nested command buffers (today)");
  }

  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status =
      IREE_HAL_VTABLE_DISPATCH(device, iree_hal_device, create_command_buffer)(
          device, mode, command_categories, queue_affinity, binding_capacity,
          out_command_buffer);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_hal_command_buffer_mode_t
iree_hal_command_buffer_mode(const iree_hal_command_buffer_t* command_buffer) {
  IREE_ASSERT_ARGUMENT(command_buffer);
  return command_buffer->mode;
}

IREE_API_EXPORT iree_hal_command_category_t
iree_hal_command_buffer_allowed_categories(
    const iree_hal_command_buffer_t* command_buffer) {
  IREE_ASSERT_ARGUMENT(command_buffer);
  return command_buffer->allowed_categories;
}

IREE_API_EXPORT iree_status_t
iree_hal_command_buffer_begin(iree_hal_command_buffer_t* command_buffer) {
  IREE_ASSERT_ARGUMENT(command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);
  IF_VALIDATING(command_buffer, {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_command_buffer_begin_validation(
                command_buffer, VALIDATION_STATE(command_buffer)));
  });
  iree_status_t status =
      _VTABLE_DISPATCH(command_buffer, begin)(command_buffer);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t
iree_hal_command_buffer_end(iree_hal_command_buffer_t* command_buffer) {
  IREE_ASSERT_ARGUMENT(command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);
  IF_VALIDATING(command_buffer, {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_command_buffer_end_validation(
                command_buffer, VALIDATION_STATE(command_buffer)));
  });
  iree_status_t status = _VTABLE_DISPATCH(command_buffer, end)(command_buffer);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT void iree_hal_command_buffer_begin_debug_group(
    iree_hal_command_buffer_t* command_buffer, iree_string_view_t label,
    iree_hal_label_color_t label_color,
    const iree_hal_label_location_t* location) {
  IREE_ASSERT_ARGUMENT(command_buffer);
  IF_VALIDATING(command_buffer,
                iree_hal_command_buffer_begin_debug_group_validation(
                    command_buffer, VALIDATION_STATE(command_buffer), label,
                    label_color, location));
  _VTABLE_DISPATCH(command_buffer, begin_debug_group)
  (command_buffer, label, label_color, location);
}

IREE_API_EXPORT void iree_hal_command_buffer_end_debug_group(
    iree_hal_command_buffer_t* command_buffer) {
  IREE_ASSERT_ARGUMENT(command_buffer);
  IF_VALIDATING(command_buffer,
                iree_hal_command_buffer_end_debug_group_validation(
                    command_buffer, VALIDATION_STATE(command_buffer)));
  _VTABLE_DISPATCH(command_buffer, end_debug_group)
  (command_buffer);
}

IREE_API_EXPORT iree_status_t iree_hal_command_buffer_execution_barrier(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_execution_barrier_flags_t flags,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  IREE_ASSERT_ARGUMENT(command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);
  IF_VALIDATING(command_buffer, {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0,
        iree_hal_command_buffer_execution_barrier_validation(
            command_buffer, VALIDATION_STATE(command_buffer), source_stage_mask,
            target_stage_mask, flags, memory_barrier_count, memory_barriers,
            buffer_barrier_count, buffer_barriers));
  });
  iree_status_t status = _VTABLE_DISPATCH(command_buffer, execution_barrier)(
      command_buffer, source_stage_mask, target_stage_mask, flags,
      memory_barrier_count, memory_barriers, buffer_barrier_count,
      buffer_barriers);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_command_buffer_signal_event(
    iree_hal_command_buffer_t* command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  IREE_ASSERT_ARGUMENT(command_buffer);
  IREE_ASSERT_ARGUMENT(event);
  IREE_TRACE_ZONE_BEGIN(z0);
  IF_VALIDATING(command_buffer, {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_command_buffer_signal_event_validation(
                command_buffer, VALIDATION_STATE(command_buffer), event,
                source_stage_mask));
  });
  iree_status_t status = _VTABLE_DISPATCH(command_buffer, signal_event)(
      command_buffer, event, source_stage_mask);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_command_buffer_reset_event(
    iree_hal_command_buffer_t* command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  IREE_ASSERT_ARGUMENT(command_buffer);
  IREE_ASSERT_ARGUMENT(event);
  IREE_TRACE_ZONE_BEGIN(z0);
  IF_VALIDATING(command_buffer, {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_command_buffer_reset_event_validation(
                command_buffer, VALIDATION_STATE(command_buffer), event,
                source_stage_mask));
  });
  iree_status_t status = _VTABLE_DISPATCH(command_buffer, reset_event)(
      command_buffer, event, source_stage_mask);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_command_buffer_wait_events(
    iree_hal_command_buffer_t* command_buffer, iree_host_size_t event_count,
    const iree_hal_event_t** events,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  IREE_ASSERT_ARGUMENT(command_buffer);
  IREE_ASSERT_ARGUMENT(!event_count || events);
  IREE_ASSERT_ARGUMENT(!memory_barrier_count || memory_barriers);
  IREE_ASSERT_ARGUMENT(!buffer_barrier_count || buffer_barriers);
  IREE_TRACE_ZONE_BEGIN(z0);
  IF_VALIDATING(command_buffer, {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0,
        iree_hal_command_buffer_wait_events_validation(
            command_buffer, VALIDATION_STATE(command_buffer), event_count,
            events, source_stage_mask, target_stage_mask, memory_barrier_count,
            memory_barriers, buffer_barrier_count, buffer_barriers));
  });
  iree_status_t status = _VTABLE_DISPATCH(command_buffer, wait_events)(
      command_buffer, event_count, events, source_stage_mask, target_stage_mask,
      memory_barrier_count, memory_barriers, buffer_barrier_count,
      buffer_barriers);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_command_buffer_discard_buffer(
    iree_hal_command_buffer_t* command_buffer, iree_hal_buffer_t* buffer) {
  IREE_ASSERT_ARGUMENT(command_buffer);
  IREE_ASSERT_ARGUMENT(buffer);
  IREE_TRACE_ZONE_BEGIN(z0);
  IF_VALIDATING(command_buffer, {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_command_buffer_discard_buffer_validation(
                command_buffer, VALIDATION_STATE(command_buffer), buffer));
  });
  iree_status_t status =
      _VTABLE_DISPATCH(command_buffer, discard_buffer)(command_buffer, buffer);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_command_buffer_fill_buffer(
    iree_hal_command_buffer_t* command_buffer, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_device_size_t length,
    const void* pattern, iree_host_size_t pattern_length) {
  IREE_ASSERT_ARGUMENT(command_buffer);
  IREE_ASSERT_ARGUMENT(target_buffer);
  if (length == 0) {
    // No-op fill. All other validation is skipped.
    return iree_ok_status();
  }
  IREE_TRACE_ZONE_BEGIN(z0);
  IF_VALIDATING(command_buffer, {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_command_buffer_fill_buffer_validation(
                command_buffer, VALIDATION_STATE(command_buffer), target_buffer,
                target_offset, length, pattern, pattern_length));
  });
  iree_status_t status = _VTABLE_DISPATCH(command_buffer, fill_buffer)(
      command_buffer, target_buffer, target_offset, length, pattern,
      pattern_length);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_command_buffer_update_buffer(
    iree_hal_command_buffer_t* command_buffer, const void* source_buffer,
    iree_host_size_t source_offset, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_device_size_t length) {
  IREE_ASSERT_ARGUMENT(command_buffer);
  IREE_ASSERT_ARGUMENT(source_buffer);
  IREE_ASSERT_ARGUMENT(target_buffer);
  if (length == 0) {
    // No-op update. All other validation is skipped.
    return iree_ok_status();
  }
  IREE_TRACE_ZONE_BEGIN(z0);
  IF_VALIDATING(command_buffer, {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_command_buffer_update_buffer_validation(
                command_buffer, VALIDATION_STATE(command_buffer), source_buffer,
                source_offset, target_buffer, target_offset, length));
  });
  iree_status_t status = _VTABLE_DISPATCH(command_buffer, update_buffer)(
      command_buffer, source_buffer, source_offset, target_buffer,
      target_offset, length);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_command_buffer_copy_buffer(
    iree_hal_command_buffer_t* command_buffer, iree_hal_buffer_t* source_buffer,
    iree_device_size_t source_offset, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_device_size_t length) {
  IREE_ASSERT_ARGUMENT(command_buffer);
  if (length == 0) {
    // No-op copy. All other validation is skipped.
    return iree_ok_status();
  }
  IREE_TRACE_ZONE_BEGIN(z0);
  IF_VALIDATING(command_buffer, {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_command_buffer_copy_buffer_validation(
                command_buffer, VALIDATION_STATE(command_buffer), source_buffer,
                source_offset, target_buffer, target_offset, length));
  });
  iree_status_t status = _VTABLE_DISPATCH(command_buffer, copy_buffer)(
      command_buffer, source_buffer, source_offset, target_buffer,
      target_offset, length);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_command_buffer_collective(
    iree_hal_command_buffer_t* command_buffer, iree_hal_channel_t* channel,
    iree_hal_collective_op_t op, uint32_t param,
    iree_hal_buffer_binding_t send_binding,
    iree_hal_buffer_binding_t recv_binding, iree_device_size_t element_count) {
  IREE_ASSERT_ARGUMENT(command_buffer);
  IREE_ASSERT_ARGUMENT(channel);
  IREE_TRACE_ZONE_BEGIN(z0);
  IF_VALIDATING(command_buffer, {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_command_buffer_collective_validation(
                command_buffer, VALIDATION_STATE(command_buffer), channel, op,
                param, send_binding, recv_binding, element_count));
  });
#if IREE_HAL_VERBOSE_TRACING_ENABLE
  IREE_TRACE({
    iree_bitfield_string_temp_t string_temp;
    iree_string_view_t collective_str =
        iree_hal_collective_op_format(&op, &string_temp);
    IREE_TRACE_ZONE_APPEND_TEXT(z0, collective_str.data, collective_str.size);
  });
#endif  // IREE_HAL_VERBOSE_TRACING_ENABLE
  iree_status_t status = _VTABLE_DISPATCH(command_buffer, collective)(
      command_buffer, channel, op, param, send_binding, recv_binding,
      element_count);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_command_buffer_push_constants(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_pipeline_layout_t* pipeline_layout, iree_host_size_t offset,
    const void* values, iree_host_size_t values_length) {
  IREE_ASSERT_ARGUMENT(command_buffer);
  IREE_ASSERT_ARGUMENT(pipeline_layout);
  IREE_ASSERT_ARGUMENT(values);
  if (IREE_UNLIKELY(values_length == 0)) {
    return iree_ok_status();
  }
  IREE_TRACE_ZONE_BEGIN(z0);
  IF_VALIDATING(command_buffer, {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_command_buffer_push_constants_validation(
                command_buffer, VALIDATION_STATE(command_buffer),
                pipeline_layout, offset, values, values_length));
  });
  iree_status_t status = _VTABLE_DISPATCH(command_buffer, push_constants)(
      command_buffer, pipeline_layout, offset, values, values_length);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_command_buffer_push_descriptor_set(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_pipeline_layout_t* pipeline_layout, uint32_t set,
    iree_host_size_t binding_count,
    const iree_hal_descriptor_set_binding_t* bindings) {
  IREE_ASSERT_ARGUMENT(command_buffer);
  IREE_ASSERT_ARGUMENT(pipeline_layout);
  IREE_ASSERT_ARGUMENT(!binding_count || bindings);
  IREE_TRACE_ZONE_BEGIN(z0);
  IF_VALIDATING(command_buffer, {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_command_buffer_push_descriptor_set_validation(
                command_buffer, VALIDATION_STATE(command_buffer),
                pipeline_layout, set, binding_count, bindings));
  });
  iree_status_t status = _VTABLE_DISPATCH(command_buffer, push_descriptor_set)(
      command_buffer, pipeline_layout, set, binding_count, bindings);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_command_buffer_dispatch(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_executable_t* executable, int32_t entry_point,
    uint32_t workgroup_x, uint32_t workgroup_y, uint32_t workgroup_z) {
  IREE_ASSERT_ARGUMENT(command_buffer);
  IREE_ASSERT_ARGUMENT(executable);
  if ((workgroup_x | workgroup_y | workgroup_z) == 0) {
    // No-op dispatch. All implementations are expected to do this but we ensure
    // it happens here to avoid the overhead of going all the way down into the
    // device layer for something we know should have no (intentional)
    // side-effects. Note that this does mean that validation is skipped and
    // the executable/etc could be bogus but that's fine.
    return iree_ok_status();
  }
  IREE_TRACE_ZONE_BEGIN(z0);
  IF_VALIDATING(command_buffer, {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_command_buffer_dispatch_validation(
                command_buffer, VALIDATION_STATE(command_buffer), executable,
                entry_point, workgroup_x, workgroup_y, workgroup_z));
  });
#if IREE_HAL_VERBOSE_TRACING_ENABLE
  // TODO(benvanik): add a tracing.h helper that does the snprintf directly
  // into a tracy_malloc buffer so that we can avoid the memcpy. Today this can
  // take 4-5us which adds too much overhead when trying to get accurate timings
  // with tracing enabled. Because benchmarks shouldn't be run with asserts
  // enabled we only enable these when assertions are enabled. Ideally we'd
  // slice off a much larger allocation and then suballocate from that ourselves
  // so that we could avoid the tracy_malloc overheads per-dispatch.
  IREE_TRACE({
    char xyz_string[32];
    int xyz_string_length =
        snprintf(xyz_string, IREE_ARRAYSIZE(xyz_string), "%ux%ux%u",
                 workgroup_x, workgroup_y, workgroup_z);
    IREE_TRACE_ZONE_APPEND_TEXT_STRING_VIEW(z0, xyz_string, xyz_string_length);
  });
#endif  // IREE_HAL_VERBOSE_TRACING_ENABLE
  iree_status_t status = _VTABLE_DISPATCH(command_buffer, dispatch)(
      command_buffer, executable, entry_point, workgroup_x, workgroup_y,
      workgroup_z);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_command_buffer_dispatch_indirect(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_executable_t* executable, int32_t entry_point,
    iree_hal_buffer_t* workgroups_buffer,
    iree_device_size_t workgroups_offset) {
  IREE_ASSERT_ARGUMENT(command_buffer);
  IREE_ASSERT_ARGUMENT(executable);
  IREE_ASSERT_ARGUMENT(workgroups_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);
  IF_VALIDATING(command_buffer, {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_command_buffer_dispatch_indirect_validation(
                command_buffer, VALIDATION_STATE(command_buffer), executable,
                entry_point, workgroups_buffer, workgroups_offset));
  });
  iree_status_t status = _VTABLE_DISPATCH(command_buffer, dispatch_indirect)(
      command_buffer, executable, entry_point, workgroups_buffer,
      workgroups_offset);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_command_buffer_execute_commands(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_command_buffer_t* commands,
    iree_hal_buffer_binding_table_t binding_table) {
  IREE_ASSERT_ARGUMENT(command_buffer);
  IREE_ASSERT_ARGUMENT(commands);
  IREE_ASSERT_ARGUMENT(!binding_table.count || binding_table.bindings);
  IREE_TRACE_ZONE_BEGIN(z0);
  IF_VALIDATING(command_buffer, {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_command_buffer_execute_commands_validation(
                command_buffer, VALIDATION_STATE(command_buffer), commands,
                binding_table));
  });
  iree_status_t status = _VTABLE_DISPATCH(command_buffer, execute_commands)(
      command_buffer, commands, binding_table);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Utilities for command buffer creation
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t iree_hal_create_transfer_command_buffer(
    iree_hal_device_t* device, iree_hal_command_buffer_mode_t mode,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t transfer_count,
    const iree_hal_transfer_command_t* transfer_commands,
    iree_hal_command_buffer_t** out_command_buffer) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_command_buffer_t* command_buffer = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_command_buffer_create(
              device, mode, IREE_HAL_COMMAND_CATEGORY_TRANSFER, queue_affinity,
              /*binding_capacity=*/0, &command_buffer));

  iree_status_t status = iree_hal_command_buffer_begin(command_buffer);
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < transfer_count; ++i) {
      const iree_hal_transfer_command_t* transfer_command =
          &transfer_commands[i];
      switch (transfer_command->type) {
        case IREE_HAL_TRANSFER_COMMAND_TYPE_FILL:
          status = iree_hal_command_buffer_fill_buffer(
              command_buffer, transfer_command->fill.target_buffer,
              transfer_command->fill.target_offset,
              transfer_command->fill.length, transfer_command->fill.pattern,
              transfer_command->fill.pattern_length);
          break;
        case IREE_HAL_TRANSFER_COMMAND_TYPE_COPY:
          status = iree_hal_command_buffer_copy_buffer(
              command_buffer, transfer_command->copy.source_buffer,
              transfer_command->copy.source_offset,
              transfer_command->copy.target_buffer,
              transfer_command->copy.target_offset,
              transfer_command->copy.length);
          break;
        case IREE_HAL_TRANSFER_COMMAND_TYPE_UPDATE:
          status = iree_hal_command_buffer_update_buffer(
              command_buffer, transfer_command->update.source_buffer,
              transfer_command->update.source_offset,
              transfer_command->update.target_buffer,
              transfer_command->update.target_offset,
              transfer_command->update.length);
          break;
        default:
          status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                    "unknown transfer_commands[%zu] type %d", i,
                                    (int)transfer_command->type);
          break;
      }
      if (!iree_status_is_ok(status)) break;
    }
  }
  status =
      iree_status_join(status, iree_hal_command_buffer_end(command_buffer));

  if (iree_status_is_ok(status)) {
    *out_command_buffer = command_buffer;
  } else {
    iree_hal_command_buffer_release(command_buffer);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}
