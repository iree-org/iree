// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tooling/profile/internal.h"

void iree_profile_dispatch_context_initialize(
    iree_allocator_t host_allocator,
    iree_profile_dispatch_context_t* out_context) {
  memset(out_context, 0, sizeof(*out_context));
  out_context->host_allocator = host_allocator;
}

void iree_profile_dispatch_context_deinitialize(
    iree_profile_dispatch_context_t* context) {
  iree_allocator_free(context->host_allocator, context->queue_device_events);
  iree_allocator_free(context->host_allocator, context->queue_events);
  iree_allocator_free(context->host_allocator, context->queue_aggregates);
  iree_allocator_free(context->host_allocator, context->command_aggregates);
  iree_allocator_free(context->host_allocator, context->aggregates);
  iree_allocator_free(context->host_allocator, context->devices);
  iree_allocator_free(context->host_allocator, context->queues);
  iree_allocator_free(context->host_allocator, context->command_operations);
  iree_allocator_free(context->host_allocator, context->command_buffers);
  iree_allocator_free(context->host_allocator, context->exports);
  iree_allocator_free(context->host_allocator, context->executables);
  memset(context, 0, sizeof(*context));
}

static iree_status_t iree_profile_dispatch_get_device(
    iree_profile_dispatch_context_t* context, uint32_t physical_device_ordinal,
    iree_profile_dispatch_device_t** out_device) {
  *out_device = NULL;

  for (iree_host_size_t i = 0; i < context->device_count; ++i) {
    if (context->devices[i].physical_device_ordinal ==
        physical_device_ordinal) {
      *out_device = &context->devices[i];
      return iree_ok_status();
    }
  }

  if (context->device_count + 1 > context->device_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        context->host_allocator,
        iree_max((iree_host_size_t)4, context->device_count + 1),
        sizeof(context->devices[0]), &context->device_capacity,
        (void**)&context->devices));
  }

  iree_profile_dispatch_device_t* device =
      &context->devices[context->device_count++];
  memset(device, 0, sizeof(*device));
  device->physical_device_ordinal = physical_device_ordinal;
  *out_device = device;
  return iree_ok_status();
}

static void iree_profile_dispatch_record_clock_sample(
    iree_profile_dispatch_device_t* device,
    const iree_hal_profile_clock_correlation_record_t* record) {
  if (device->clock_sample_count == 0) {
    device->first_clock_sample = *record;
  }
  device->last_clock_sample = *record;
  ++device->clock_sample_count;
}

static const iree_profile_dispatch_queue_t* iree_profile_dispatch_find_queue(
    const iree_profile_dispatch_context_t* context,
    uint32_t physical_device_ordinal, uint32_t queue_ordinal,
    uint64_t stream_id) {
  for (iree_host_size_t i = 0; i < context->queue_count; ++i) {
    const iree_profile_dispatch_queue_t* queue_info = &context->queues[i];
    if (queue_info->record.physical_device_ordinal == physical_device_ordinal &&
        queue_info->record.queue_ordinal == queue_ordinal &&
        queue_info->record.stream_id == stream_id) {
      return queue_info;
    }
  }
  return NULL;
}

static const iree_profile_dispatch_executable_t*
iree_profile_dispatch_find_executable(
    const iree_profile_dispatch_context_t* context, uint64_t executable_id) {
  for (iree_host_size_t i = 0; i < context->executable_count; ++i) {
    const iree_profile_dispatch_executable_t* executable_info =
        &context->executables[i];
    if (executable_info->record.executable_id == executable_id) {
      return executable_info;
    }
  }
  return NULL;
}

static const iree_profile_dispatch_command_buffer_t*
iree_profile_dispatch_find_command_buffer(
    const iree_profile_dispatch_context_t* context,
    uint64_t command_buffer_id) {
  for (iree_host_size_t i = 0; i < context->command_buffer_count; ++i) {
    const iree_profile_dispatch_command_buffer_t* command_buffer_info =
        &context->command_buffers[i];
    if (command_buffer_info->record.command_buffer_id == command_buffer_id) {
      return command_buffer_info;
    }
  }
  return NULL;
}

bool iree_profile_dispatch_device_try_fit_clock(
    const iree_profile_dispatch_device_t* device, double* out_ns_per_tick,
    double* out_tick_frequency_hz) {
  *out_ns_per_tick = 0.0;
  *out_tick_frequency_hz = 0.0;
  if (!device || device->clock_sample_count < 2) return false;

  const iree_hal_profile_clock_correlation_record_t* first =
      &device->first_clock_sample;
  const iree_hal_profile_clock_correlation_record_t* last =
      &device->last_clock_sample;
  if (!iree_all_bits_set(
          first->flags,
          IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_DEVICE_TICK |
              IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_HOST_CPU_TIMESTAMP) ||
      !iree_all_bits_set(
          last->flags,
          IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_DEVICE_TICK |
              IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_HOST_CPU_TIMESTAMP)) {
    return false;
  }
  if (last->device_tick <= first->device_tick ||
      last->host_cpu_timestamp_ns <= first->host_cpu_timestamp_ns) {
    return false;
  }

  const double device_delta_ticks =
      (double)(last->device_tick - first->device_tick);
  const double host_delta_ns =
      (double)(last->host_cpu_timestamp_ns - first->host_cpu_timestamp_ns);
  *out_ns_per_tick = host_delta_ns / device_delta_ticks;
  *out_tick_frequency_hz = 1000000000.0 / *out_ns_per_tick;
  return true;
}

static const iree_profile_dispatch_export_t* iree_profile_dispatch_find_export(
    const iree_profile_dispatch_context_t* context, uint64_t executable_id,
    uint32_t export_ordinal) {
  for (iree_host_size_t i = 0; i < context->export_count; ++i) {
    const iree_profile_dispatch_export_t* export_info = &context->exports[i];
    if (export_info->executable_id == executable_id &&
        export_info->export_ordinal == export_ordinal) {
      return export_info;
    }
  }
  return NULL;
}

static iree_string_view_t iree_profile_dispatch_format_numeric_key(
    uint32_t physical_device_ordinal, uint64_t executable_id,
    uint32_t export_ordinal, char* buffer, iree_host_size_t buffer_capacity) {
  if (buffer_capacity == 0) return iree_string_view_empty();
  int result = 0;
  if (physical_device_ordinal == UINT32_MAX) {
    result = snprintf(buffer, buffer_capacity, "executable%" PRIu64 "#%u",
                      executable_id, export_ordinal);
  } else {
    result =
        snprintf(buffer, buffer_capacity, "device%u/executable%" PRIu64 "#%u",
                 physical_device_ordinal, executable_id, export_ordinal);
  }
  if (result < 0) return iree_string_view_empty();
  iree_host_size_t length = (iree_host_size_t)result;
  if (length >= buffer_capacity) length = buffer_capacity - 1;
  return iree_make_string_view(buffer, length);
}

iree_string_view_t iree_profile_dispatch_format_export_key(
    const iree_profile_dispatch_export_t* export_info,
    uint32_t physical_device_ordinal, char* numeric_buffer,
    iree_host_size_t numeric_buffer_capacity) {
  if (!iree_string_view_is_empty(export_info->name)) {
    return export_info->name;
  }
  return iree_profile_dispatch_format_numeric_key(
      physical_device_ordinal, export_info->executable_id,
      export_info->export_ordinal, numeric_buffer, numeric_buffer_capacity);
}

iree_status_t iree_profile_dispatch_resolve_key(
    const iree_profile_dispatch_context_t* context,
    uint32_t physical_device_ordinal, uint64_t executable_id,
    uint32_t export_ordinal, char* numeric_buffer,
    iree_host_size_t numeric_buffer_capacity, iree_string_view_t* out_key) {
  *out_key = iree_string_view_empty();
  const iree_profile_dispatch_executable_t* executable_info =
      iree_profile_dispatch_find_executable(context, executable_id);
  if (!executable_info) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "dispatch event references missing executable metadata "
        "device=%u executable=%" PRIu64 " export=%u",
        physical_device_ordinal, executable_id, export_ordinal);
  }
  const iree_profile_dispatch_export_t* export_info =
      iree_profile_dispatch_find_export(context, executable_id, export_ordinal);
  if (!export_info) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "dispatch event references missing executable export metadata "
        "device=%u executable=%" PRIu64 " export=%u",
        physical_device_ordinal, executable_id, export_ordinal);
  }
  if (!iree_string_view_is_empty(export_info->name)) {
    *out_key = export_info->name;
  } else {
    *out_key = iree_profile_dispatch_format_export_key(
        export_info, physical_device_ordinal, numeric_buffer,
        numeric_buffer_capacity);
  }
  return iree_ok_status();
}

static iree_status_t iree_profile_dispatch_validate_event_metadata(
    const iree_profile_dispatch_context_t* context,
    const iree_hal_profile_file_record_t* record,
    const iree_hal_profile_dispatch_event_t* event,
    iree_profile_projection_mode_t projection_mode) {
  const iree_profile_dispatch_queue_t* queue_info =
      iree_profile_dispatch_find_queue(
          context, record->header.physical_device_ordinal,
          record->header.queue_ordinal, record->header.stream_id);
  if (!queue_info) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "dispatch event references missing queue metadata "
        "device=%u queue=%u stream=%" PRIu64 " submission=%" PRIu64,
        record->header.physical_device_ordinal, record->header.queue_ordinal,
        record->header.stream_id, event->submission_id);
  }
  if (projection_mode == IREE_PROFILE_PROJECTION_MODE_COMMAND &&
      event->command_buffer_id != 0 &&
      !iree_profile_dispatch_find_command_buffer(context,
                                                 event->command_buffer_id)) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "dispatch event references missing command-buffer metadata "
        "command_buffer=%" PRIu64 " submission=%" PRIu64,
        event->command_buffer_id, event->submission_id);
  }
  return iree_ok_status();
}

bool iree_profile_dispatch_key_matches(iree_string_view_t key,
                                       iree_string_view_t filter) {
  if (iree_string_view_is_empty(filter) ||
      iree_string_view_equal(filter, IREE_SV("*"))) {
    return true;
  }
  return iree_string_view_match_pattern(key, filter);
}

static bool iree_profile_dispatch_event_matches_id(
    const iree_hal_profile_dispatch_event_t* event,
    iree_profile_projection_mode_t mode, int64_t id_filter) {
  if (id_filter < 0) return true;
  const uint64_t id = (uint64_t)id_filter;
  switch (mode) {
    case IREE_PROFILE_PROJECTION_MODE_DISPATCH:
      return event->event_id == id;
    case IREE_PROFILE_PROJECTION_MODE_EXECUTABLE:
      return event->executable_id == id;
    case IREE_PROFILE_PROJECTION_MODE_COMMAND:
      return event->command_buffer_id == id;
    case IREE_PROFILE_PROJECTION_MODE_QUEUE:
      return event->submission_id == id;
    default:
      return false;
  }
}

static iree_status_t iree_profile_dispatch_append_export(
    iree_profile_dispatch_context_t* context,
    const iree_profile_dispatch_export_t* export_info) {
  if (context->export_count + 1 > context->export_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        context->host_allocator,
        iree_max((iree_host_size_t)16, context->export_count + 1),
        sizeof(context->exports[0]), &context->export_capacity,
        (void**)&context->exports));
  }
  context->exports[context->export_count++] = *export_info;
  return iree_ok_status();
}

static iree_status_t iree_profile_dispatch_append_executable(
    iree_profile_dispatch_context_t* context,
    const iree_hal_profile_executable_record_t* record) {
  if (context->executable_count + 1 > context->executable_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        context->host_allocator,
        iree_max((iree_host_size_t)8, context->executable_count + 1),
        sizeof(context->executables[0]), &context->executable_capacity,
        (void**)&context->executables));
  }
  iree_profile_dispatch_executable_t* executable_info =
      &context->executables[context->executable_count++];
  executable_info->record = *record;
  return iree_ok_status();
}

static iree_status_t iree_profile_dispatch_append_command_buffer(
    iree_profile_dispatch_context_t* context,
    const iree_hal_profile_command_buffer_record_t* record) {
  if (context->command_buffer_count + 1 > context->command_buffer_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        context->host_allocator,
        iree_max((iree_host_size_t)16, context->command_buffer_count + 1),
        sizeof(context->command_buffers[0]), &context->command_buffer_capacity,
        (void**)&context->command_buffers));
  }
  iree_profile_dispatch_command_buffer_t* command_buffer_info =
      &context->command_buffers[context->command_buffer_count++];
  command_buffer_info->record = *record;
  return iree_ok_status();
}

static iree_status_t iree_profile_dispatch_append_command_operation(
    iree_profile_dispatch_context_t* context,
    const iree_hal_profile_command_operation_record_t* record) {
  if (context->command_operation_count + 1 >
      context->command_operation_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        context->host_allocator,
        iree_max((iree_host_size_t)64, context->command_operation_count + 1),
        sizeof(context->command_operations[0]),
        &context->command_operation_capacity,
        (void**)&context->command_operations));
  }
  iree_profile_dispatch_command_operation_t* operation_info =
      &context->command_operations[context->command_operation_count++];
  operation_info->record = *record;
  return iree_ok_status();
}

static iree_status_t iree_profile_dispatch_append_queue(
    iree_profile_dispatch_context_t* context,
    const iree_hal_profile_queue_record_t* record) {
  if (context->queue_count + 1 > context->queue_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        context->host_allocator,
        iree_max((iree_host_size_t)4, context->queue_count + 1),
        sizeof(context->queues[0]), &context->queue_capacity,
        (void**)&context->queues));
  }
  iree_profile_dispatch_queue_t* queue_info =
      &context->queues[context->queue_count++];
  queue_info->record = *record;
  return iree_ok_status();
}

static iree_status_t iree_profile_dispatch_append_queue_event(
    iree_profile_dispatch_context_t* context,
    const iree_hal_profile_queue_event_t* record) {
  if (context->queue_event_count + 1 > context->queue_event_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        context->host_allocator,
        iree_max((iree_host_size_t)64, context->queue_event_count + 1),
        sizeof(context->queue_events[0]), &context->queue_event_capacity,
        (void**)&context->queue_events));
  }
  iree_profile_dispatch_queue_event_t* event_info =
      &context->queue_events[context->queue_event_count++];
  event_info->record = *record;
  return iree_ok_status();
}

static iree_status_t iree_profile_dispatch_append_queue_device_event(
    iree_profile_dispatch_context_t* context,
    const iree_hal_profile_queue_device_event_t* record) {
  if (context->queue_device_event_count + 1 >
      context->queue_device_event_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        context->host_allocator,
        iree_max((iree_host_size_t)64, context->queue_device_event_count + 1),
        sizeof(context->queue_device_events[0]),
        &context->queue_device_event_capacity,
        (void**)&context->queue_device_events));
  }
  iree_profile_dispatch_queue_device_event_t* event_info =
      &context->queue_device_events[context->queue_device_event_count++];
  event_info->record = *record;
  return iree_ok_status();
}

static iree_status_t iree_profile_dispatch_get_aggregate(
    iree_profile_dispatch_context_t* context, uint32_t physical_device_ordinal,
    uint64_t executable_id, uint32_t export_ordinal,
    iree_profile_dispatch_aggregate_t** out_aggregate) {
  *out_aggregate = NULL;

  for (iree_host_size_t i = 0; i < context->aggregate_count; ++i) {
    iree_profile_dispatch_aggregate_t* aggregate = &context->aggregates[i];
    if (aggregate->physical_device_ordinal == physical_device_ordinal &&
        aggregate->executable_id == executable_id &&
        aggregate->export_ordinal == export_ordinal) {
      *out_aggregate = aggregate;
      return iree_ok_status();
    }
  }

  if (context->aggregate_count + 1 > context->aggregate_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        context->host_allocator,
        iree_max((iree_host_size_t)16, context->aggregate_count + 1),
        sizeof(context->aggregates[0]), &context->aggregate_capacity,
        (void**)&context->aggregates));
  }

  iree_profile_dispatch_aggregate_t* aggregate =
      &context->aggregates[context->aggregate_count++];
  memset(aggregate, 0, sizeof(*aggregate));
  aggregate->physical_device_ordinal = physical_device_ordinal;
  aggregate->executable_id = executable_id;
  aggregate->export_ordinal = export_ordinal;
  aggregate->earliest_start_tick = UINT64_MAX;
  aggregate->minimum_ticks = UINT64_MAX;
  *out_aggregate = aggregate;
  return iree_ok_status();
}

static iree_status_t iree_profile_dispatch_get_command_aggregate(
    iree_profile_dispatch_context_t* context, uint64_t command_buffer_id,
    uint64_t submission_id, uint32_t physical_device_ordinal,
    uint32_t queue_ordinal, uint64_t stream_id,
    iree_profile_dispatch_command_aggregate_t** out_aggregate) {
  *out_aggregate = NULL;

  for (iree_host_size_t i = 0; i < context->command_aggregate_count; ++i) {
    iree_profile_dispatch_command_aggregate_t* aggregate =
        &context->command_aggregates[i];
    if (aggregate->command_buffer_id == command_buffer_id &&
        aggregate->submission_id == submission_id &&
        aggregate->physical_device_ordinal == physical_device_ordinal &&
        aggregate->queue_ordinal == queue_ordinal &&
        aggregate->stream_id == stream_id) {
      *out_aggregate = aggregate;
      return iree_ok_status();
    }
  }

  if (context->command_aggregate_count + 1 >
      context->command_aggregate_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        context->host_allocator,
        iree_max((iree_host_size_t)16, context->command_aggregate_count + 1),
        sizeof(context->command_aggregates[0]),
        &context->command_aggregate_capacity,
        (void**)&context->command_aggregates));
  }

  iree_profile_dispatch_command_aggregate_t* aggregate =
      &context->command_aggregates[context->command_aggregate_count++];
  memset(aggregate, 0, sizeof(*aggregate));
  aggregate->command_buffer_id = command_buffer_id;
  aggregate->submission_id = submission_id;
  aggregate->physical_device_ordinal = physical_device_ordinal;
  aggregate->queue_ordinal = queue_ordinal;
  aggregate->stream_id = stream_id;
  aggregate->earliest_start_tick = UINT64_MAX;
  *out_aggregate = aggregate;
  return iree_ok_status();
}

static iree_status_t iree_profile_dispatch_get_queue_aggregate(
    iree_profile_dispatch_context_t* context, uint32_t physical_device_ordinal,
    uint32_t queue_ordinal, uint64_t stream_id, uint64_t submission_id,
    iree_profile_dispatch_queue_aggregate_t** out_aggregate) {
  *out_aggregate = NULL;

  for (iree_host_size_t i = 0; i < context->queue_aggregate_count; ++i) {
    iree_profile_dispatch_queue_aggregate_t* aggregate =
        &context->queue_aggregates[i];
    if (aggregate->physical_device_ordinal == physical_device_ordinal &&
        aggregate->queue_ordinal == queue_ordinal &&
        aggregate->stream_id == stream_id &&
        aggregate->submission_id == submission_id) {
      *out_aggregate = aggregate;
      return iree_ok_status();
    }
  }

  if (context->queue_aggregate_count + 1 > context->queue_aggregate_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        context->host_allocator,
        iree_max((iree_host_size_t)16, context->queue_aggregate_count + 1),
        sizeof(context->queue_aggregates[0]),
        &context->queue_aggregate_capacity,
        (void**)&context->queue_aggregates));
  }

  iree_profile_dispatch_queue_aggregate_t* aggregate =
      &context->queue_aggregates[context->queue_aggregate_count++];
  memset(aggregate, 0, sizeof(*aggregate));
  aggregate->physical_device_ordinal = physical_device_ordinal;
  aggregate->queue_ordinal = queue_ordinal;
  aggregate->stream_id = stream_id;
  aggregate->submission_id = submission_id;
  aggregate->earliest_start_tick = UINT64_MAX;
  *out_aggregate = aggregate;
  return iree_ok_status();
}

static void iree_profile_dispatch_record_aggregate_event(
    iree_profile_dispatch_aggregate_t* aggregate,
    const iree_hal_profile_dispatch_event_t* event) {
  ++aggregate->dispatch_count;
  memcpy(aggregate->last_workgroup_count, event->workgroup_count,
         sizeof(aggregate->last_workgroup_count));
  memcpy(aggregate->last_workgroup_size, event->workgroup_size,
         sizeof(aggregate->last_workgroup_size));

  if (event->start_tick == 0 || event->end_tick == 0 ||
      event->end_tick < event->start_tick) {
    ++aggregate->invalid_count;
    return;
  }

  const uint64_t duration_ticks = event->end_tick - event->start_tick;
  ++aggregate->valid_count;
  aggregate->earliest_start_tick =
      iree_min(aggregate->earliest_start_tick, event->start_tick);
  aggregate->latest_end_tick =
      iree_max(aggregate->latest_end_tick, event->end_tick);
  aggregate->minimum_ticks = iree_min(aggregate->minimum_ticks, duration_ticks);
  aggregate->maximum_ticks = iree_max(aggregate->maximum_ticks, duration_ticks);
  aggregate->total_ticks += (double)duration_ticks;

  const double duration = (double)duration_ticks;
  const double delta = duration - aggregate->mean_ticks;
  aggregate->mean_ticks += delta / (double)aggregate->valid_count;
  const double delta2 = duration - aggregate->mean_ticks;
  aggregate->m2_ticks += delta * delta2;
}

static void iree_profile_dispatch_record_command_aggregate_event(
    iree_profile_dispatch_command_aggregate_t* aggregate,
    const iree_hal_profile_dispatch_event_t* event) {
  ++aggregate->dispatch_count;
  if (event->start_tick == 0 || event->end_tick == 0 ||
      event->end_tick < event->start_tick) {
    ++aggregate->invalid_count;
    return;
  }
  ++aggregate->valid_count;
  aggregate->earliest_start_tick =
      iree_min(aggregate->earliest_start_tick, event->start_tick);
  aggregate->latest_end_tick =
      iree_max(aggregate->latest_end_tick, event->end_tick);
  aggregate->total_ticks += (double)(event->end_tick - event->start_tick);
}

static void iree_profile_dispatch_record_queue_aggregate_event(
    iree_profile_dispatch_queue_aggregate_t* aggregate,
    const iree_hal_profile_dispatch_event_t* event) {
  ++aggregate->dispatch_count;
  if (event->start_tick == 0 || event->end_tick == 0 ||
      event->end_tick < event->start_tick) {
    ++aggregate->invalid_count;
    return;
  }
  ++aggregate->valid_count;
  aggregate->earliest_start_tick =
      iree_min(aggregate->earliest_start_tick, event->start_tick);
  aggregate->latest_end_tick =
      iree_max(aggregate->latest_end_tick, event->end_tick);
  aggregate->total_ticks += (double)(event->end_tick - event->start_tick);
}

static void iree_profile_dispatch_record_top_event(
    iree_profile_dispatch_context_t* context,
    const iree_hal_profile_file_record_t* file_record,
    const iree_hal_profile_dispatch_event_t* event) {
  if (event->start_tick == 0 || event->end_tick == 0 ||
      event->end_tick < event->start_tick) {
    return;
  }

  const uint64_t duration_ticks = event->end_tick - event->start_tick;
  iree_host_size_t target_index = context->top_dispatch_count;
  if (context->top_dispatch_count < IREE_PROFILE_EXPLAIN_TOP_DISPATCH_COUNT) {
    ++context->top_dispatch_count;
  } else {
    target_index = 0;
    for (iree_host_size_t i = 1; i < context->top_dispatch_count; ++i) {
      if (context->top_dispatches[i].duration_ticks <
          context->top_dispatches[target_index].duration_ticks) {
        target_index = i;
      }
    }
    if (duration_ticks <=
        context->top_dispatches[target_index].duration_ticks) {
      return;
    }
  }

  iree_profile_dispatch_top_event_t* top_event =
      &context->top_dispatches[target_index];
  top_event->physical_device_ordinal =
      file_record->header.physical_device_ordinal;
  top_event->queue_ordinal = file_record->header.queue_ordinal;
  top_event->stream_id = file_record->header.stream_id;
  top_event->duration_ticks = duration_ticks;
  top_event->event = *event;
}

static iree_status_t iree_profile_dispatch_process_queue_records(
    iree_profile_dispatch_context_t* context,
    const iree_hal_profile_file_record_t* record) {
  iree_profile_typed_record_iterator_t iterator;
  iree_profile_typed_record_iterator_initialize(
      record, sizeof(iree_hal_profile_queue_record_t), &iterator);
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status)) {
    iree_profile_typed_record_t typed_record;
    bool has_record = false;
    status = iree_profile_typed_record_iterator_next(&iterator, &typed_record,
                                                     &has_record);
    if (!iree_status_is_ok(status) || !has_record) break;

    iree_hal_profile_queue_record_t record_value;
    memcpy(&record_value, typed_record.contents.data, sizeof(record_value));
    status = iree_profile_dispatch_append_queue(context, &record_value);
  }
  return status;
}

static iree_status_t iree_profile_dispatch_process_executable_records(
    iree_profile_dispatch_context_t* context,
    const iree_hal_profile_file_record_t* record) {
  iree_profile_typed_record_iterator_t iterator;
  iree_profile_typed_record_iterator_initialize(
      record, sizeof(iree_hal_profile_executable_record_t), &iterator);
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status)) {
    iree_profile_typed_record_t typed_record;
    bool has_record = false;
    status = iree_profile_typed_record_iterator_next(&iterator, &typed_record,
                                                     &has_record);
    if (!iree_status_is_ok(status) || !has_record) break;

    iree_hal_profile_executable_record_t record_value;
    memcpy(&record_value, typed_record.contents.data, sizeof(record_value));
    status = iree_profile_dispatch_append_executable(context, &record_value);
  }
  return status;
}

static iree_status_t iree_profile_dispatch_process_export_records(
    iree_profile_dispatch_context_t* context,
    const iree_hal_profile_file_record_t* record) {
  iree_profile_typed_record_iterator_t iterator;
  iree_profile_typed_record_iterator_initialize(
      record, sizeof(iree_hal_profile_executable_export_record_t), &iterator);
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status)) {
    iree_profile_typed_record_t typed_record;
    bool has_record = false;
    status = iree_profile_typed_record_iterator_next(&iterator, &typed_record,
                                                     &has_record);
    if (!iree_status_is_ok(status) || !has_record) break;

    iree_hal_profile_executable_export_record_t record_value;
    memcpy(&record_value, typed_record.contents.data, sizeof(record_value));
    if ((iree_host_size_t)record_value.name_length !=
        typed_record.inline_payload.data_length) {
      status =
          iree_make_status(IREE_STATUS_DATA_LOSS,
                           "executable export name length is inconsistent");
    }
    if (iree_status_is_ok(status)) {
      iree_profile_dispatch_export_t export_info = {
          .executable_id = record_value.executable_id,
          .flags = record_value.flags,
          .export_ordinal = record_value.export_ordinal,
          .constant_count = record_value.constant_count,
          .binding_count = record_value.binding_count,
          .parameter_count = record_value.parameter_count,
          .workgroup_size = {record_value.workgroup_size[0],
                             record_value.workgroup_size[1],
                             record_value.workgroup_size[2]},
          .pipeline_hash = {record_value.pipeline_hash[0],
                            record_value.pipeline_hash[1]},
          .name = iree_make_string_view(
              (const char*)typed_record.inline_payload.data,
              typed_record.inline_payload.data_length),
      };
      status = iree_profile_dispatch_append_export(context, &export_info);
    }
  }
  return status;
}

static iree_status_t iree_profile_dispatch_process_command_buffer_records(
    iree_profile_dispatch_context_t* context,
    const iree_hal_profile_file_record_t* record) {
  iree_profile_typed_record_iterator_t iterator;
  iree_profile_typed_record_iterator_initialize(
      record, sizeof(iree_hal_profile_command_buffer_record_t), &iterator);
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status)) {
    iree_profile_typed_record_t typed_record;
    bool has_record = false;
    status = iree_profile_typed_record_iterator_next(&iterator, &typed_record,
                                                     &has_record);
    if (!iree_status_is_ok(status) || !has_record) break;

    iree_hal_profile_command_buffer_record_t record_value;
    memcpy(&record_value, typed_record.contents.data, sizeof(record_value));
    status =
        iree_profile_dispatch_append_command_buffer(context, &record_value);
  }
  return status;
}

static iree_status_t iree_profile_dispatch_process_command_operation_records(
    iree_profile_dispatch_context_t* context,
    const iree_hal_profile_file_record_t* record) {
  iree_profile_typed_record_iterator_t iterator;
  iree_profile_typed_record_iterator_initialize(
      record, sizeof(iree_hal_profile_command_operation_record_t), &iterator);
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status)) {
    iree_profile_typed_record_t typed_record;
    bool has_record = false;
    status = iree_profile_typed_record_iterator_next(&iterator, &typed_record,
                                                     &has_record);
    if (!iree_status_is_ok(status) || !has_record) break;

    iree_hal_profile_command_operation_record_t record_value;
    memcpy(&record_value, typed_record.contents.data, sizeof(record_value));
    if (!iree_profile_dispatch_find_command_buffer(
            context, record_value.command_buffer_id)) {
      status = iree_make_status(
          IREE_STATUS_DATA_LOSS,
          "command operation references missing command-buffer metadata "
          "command_buffer=%" PRIu64 " command_index=%u",
          record_value.command_buffer_id, record_value.command_index);
    }
    if (iree_status_is_ok(status)) {
      status = iree_profile_dispatch_append_command_operation(context,
                                                              &record_value);
    }
  }
  return status;
}

static iree_status_t iree_profile_dispatch_process_clock_records(
    iree_profile_dispatch_context_t* context,
    const iree_hal_profile_file_record_t* record) {
  iree_profile_typed_record_iterator_t iterator;
  iree_profile_typed_record_iterator_initialize(
      record, sizeof(iree_hal_profile_clock_correlation_record_t), &iterator);
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status)) {
    iree_profile_typed_record_t typed_record;
    bool has_record = false;
    status = iree_profile_typed_record_iterator_next(&iterator, &typed_record,
                                                     &has_record);
    if (!iree_status_is_ok(status) || !has_record) break;

    iree_hal_profile_clock_correlation_record_t clock_record;
    memcpy(&clock_record, typed_record.contents.data, sizeof(clock_record));

    iree_profile_dispatch_device_t* device = NULL;
    status = iree_profile_dispatch_get_device(
        context, clock_record.physical_device_ordinal, &device);
    if (iree_status_is_ok(status)) {
      iree_profile_dispatch_record_clock_sample(device, &clock_record);
    }
  }
  return status;
}
iree_status_t iree_profile_dispatch_process_metadata_record(
    iree_profile_dispatch_context_t* context,
    const iree_hal_profile_file_record_t* record) {
  if (record->header.record_type != IREE_HAL_PROFILE_FILE_RECORD_TYPE_CHUNK) {
    return iree_ok_status();
  }
  if (iree_string_view_equal(record->content_type,
                             IREE_HAL_PROFILE_CONTENT_TYPE_QUEUES)) {
    return iree_profile_dispatch_process_queue_records(context, record);
  } else if (iree_string_view_equal(
                 record->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLES)) {
    return iree_profile_dispatch_process_executable_records(context, record);
  } else if (iree_string_view_equal(
                 record->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLE_EXPORTS)) {
    return iree_profile_dispatch_process_export_records(context, record);
  } else if (iree_string_view_equal(
                 record->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_COMMAND_BUFFERS)) {
    return iree_profile_dispatch_process_command_buffer_records(context,
                                                                record);
  } else if (iree_string_view_equal(
                 record->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_COMMAND_OPERATIONS)) {
    return iree_profile_dispatch_process_command_operation_records(context,
                                                                   record);
  } else if (iree_string_view_equal(
                 record->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_CLOCK_CORRELATIONS)) {
    return iree_profile_dispatch_process_clock_records(context, record);
  }
  return iree_ok_status();
}

static void iree_profile_dispatch_print_event_jsonl(
    const iree_hal_profile_file_record_t* file_record,
    const iree_hal_profile_dispatch_event_t* event, iree_string_view_t key,
    double ns_per_tick, bool has_clock_fit, FILE* file) {
  const bool is_valid = event->start_tick != 0 && event->end_tick != 0 &&
                        event->end_tick >= event->start_tick;
  const uint64_t duration_ticks =
      is_valid ? event->end_tick - event->start_tick : 0;

  fprintf(file,
          "{\"type\":\"dispatch_event\",\"physical_device_ordinal\":%u"
          ",\"queue_ordinal\":%u,\"stream_id\":%" PRIu64
          ",\"event_id\":%" PRIu64 ",\"submission_id\":%" PRIu64,
          file_record->header.physical_device_ordinal,
          file_record->header.queue_ordinal, file_record->header.stream_id,
          event->event_id, event->submission_id);
  fprintf(file,
          ",\"command_buffer_id\":%" PRIu64
          ",\"command_index\":%u"
          ",\"executable_id\":%" PRIu64 ",\"export_ordinal\":%u",
          event->command_buffer_id, event->command_index, event->executable_id,
          event->export_ordinal);
  fprintf(file, ",\"key\":");
  iree_profile_fprint_json_string(file, key);
  fprintf(file,
          ",\"flags\":%u,\"workgroup_count\":[%u,%u,%u]"
          ",\"workgroup_size\":[%u,%u,%u]",
          event->flags, event->workgroup_count[0], event->workgroup_count[1],
          event->workgroup_count[2], event->workgroup_size[0],
          event->workgroup_size[1], event->workgroup_size[2]);
  fprintf(file,
          ",\"start_tick\":%" PRIu64 ",\"end_tick\":%" PRIu64
          ",\"duration_ticks\":%" PRIu64 ",\"valid\":%s",
          event->start_tick, event->end_tick, duration_ticks,
          is_valid ? "true" : "false");
  fprintf(file, ",\"clock_fit_available\":%s",
          has_clock_fit ? "true" : "false");
  fprintf(
      file,
      ",\"device_tick_domain\":\"device_tick\""
      ",\"duration_time_domain\":\"device_tick_duration_ns\""
      ",\"duration_ns\":%.3f",
      has_clock_fit && is_valid ? (double)duration_ticks * ns_per_tick : 0.0);
  fputs("}\n", file);
}

const char* iree_profile_command_operation_type_name(
    iree_hal_profile_command_operation_type_t type) {
  switch (type) {
    case IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_BARRIER:
      return "barrier";
    case IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_DISPATCH:
      return "dispatch";
    case IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_FILL:
      return "fill";
    case IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_COPY:
      return "copy";
    case IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_UPDATE:
      return "update";
    case IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_PROFILE_MARKER:
      return "profile_marker";
    case IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_BRANCH:
      return "branch";
    case IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_COND_BRANCH:
      return "cond_branch";
    case IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_RETURN:
      return "return";
    default:
      return "unknown";
  }
}

const char* iree_profile_queue_event_type_name(
    iree_hal_profile_queue_event_type_t type) {
  switch (type) {
    case IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_BARRIER:
      return "barrier";
    case IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_DISPATCH:
      return "dispatch";
    case IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_EXECUTE:
      return "execute";
    case IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_COPY:
      return "copy";
    case IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_FILL:
      return "fill";
    case IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_UPDATE:
      return "update";
    case IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_READ:
      return "read";
    case IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_WRITE:
      return "write";
    case IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_ALLOCA:
      return "alloca";
    case IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_DEALLOCA:
      return "dealloca";
    case IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_HOST_CALL:
      return "host_call";
    default:
      return "unknown";
  }
}

const char* iree_profile_queue_dependency_strategy_name(
    iree_hal_profile_queue_dependency_strategy_t strategy) {
  switch (strategy) {
    case IREE_HAL_PROFILE_QUEUE_DEPENDENCY_STRATEGY_NONE:
      return "none";
    case IREE_HAL_PROFILE_QUEUE_DEPENDENCY_STRATEGY_INLINE:
      return "inline";
    case IREE_HAL_PROFILE_QUEUE_DEPENDENCY_STRATEGY_DEVICE_BARRIER:
      return "device_barrier";
    case IREE_HAL_PROFILE_QUEUE_DEPENDENCY_STRATEGY_SOFTWARE_DEFER:
      return "software_defer";
    default:
      return "unknown";
  }
}

const char* iree_profile_event_relationship_type_name(
    iree_hal_profile_event_relationship_type_t type) {
  switch (type) {
    case IREE_HAL_PROFILE_EVENT_RELATIONSHIP_TYPE_QUEUE_SUBMISSION_DISPATCH:
      return "queue_submission_dispatch";
    case IREE_HAL_PROFILE_EVENT_RELATIONSHIP_TYPE_QUEUE_SUBMISSION_QUEUE_DEVICE_EVENT:
      return "queue_submission_queue_device_event";
    default:
      return "unknown";
  }
}

const char* iree_profile_event_endpoint_type_name(
    iree_hal_profile_event_endpoint_type_t type) {
  switch (type) {
    case IREE_HAL_PROFILE_EVENT_ENDPOINT_TYPE_QUEUE_SUBMISSION:
      return "queue_submission";
    case IREE_HAL_PROFILE_EVENT_ENDPOINT_TYPE_QUEUE_EVENT:
      return "queue_event";
    case IREE_HAL_PROFILE_EVENT_ENDPOINT_TYPE_DISPATCH_EVENT:
      return "dispatch_event";
    case IREE_HAL_PROFILE_EVENT_ENDPOINT_TYPE_COMMAND_OPERATION:
      return "command_operation";
    case IREE_HAL_PROFILE_EVENT_ENDPOINT_TYPE_MEMORY_EVENT:
      return "memory_event";
    case IREE_HAL_PROFILE_EVENT_ENDPOINT_TYPE_ARTIFACT:
      return "artifact";
    case IREE_HAL_PROFILE_EVENT_ENDPOINT_TYPE_QUEUE_DEVICE_EVENT:
      return "queue_device_event";
    default:
      return "unknown";
  }
}

static bool iree_profile_queue_event_matches(
    const iree_hal_profile_queue_event_t* event, int64_t id_filter,
    iree_string_view_t filter) {
  if (id_filter >= 0 && event->submission_id != (uint64_t)id_filter) {
    return false;
  }
  iree_string_view_t type_name =
      iree_make_cstring_view(iree_profile_queue_event_type_name(event->type));
  return iree_profile_dispatch_key_matches(type_name, filter);
}

static bool iree_profile_queue_device_event_matches(
    const iree_hal_profile_queue_device_event_t* event, int64_t id_filter,
    iree_string_view_t filter) {
  if (id_filter >= 0 && event->submission_id != (uint64_t)id_filter) {
    return false;
  }
  iree_string_view_t type_name =
      iree_make_cstring_view(iree_profile_queue_event_type_name(event->type));
  return iree_profile_dispatch_key_matches(type_name, filter);
}

static iree_status_t iree_profile_dispatch_process_event_records(
    iree_profile_dispatch_context_t* context,
    const iree_hal_profile_file_record_t* record, iree_string_view_t filter,
    iree_profile_projection_mode_t projection_mode, int64_t id_filter,
    bool emit_events, FILE* file) {
  iree_profile_dispatch_device_t* device = NULL;
  IREE_RETURN_IF_ERROR(iree_profile_dispatch_get_device(
      context, record->header.physical_device_ordinal, &device));
  double ns_per_tick = 0.0;
  double tick_frequency_hz = 0.0;
  const bool has_clock_fit = iree_profile_dispatch_device_try_fit_clock(
      device, &ns_per_tick, &tick_frequency_hz);
  (void)tick_frequency_hz;

  iree_profile_typed_record_iterator_t iterator;
  iree_profile_typed_record_iterator_initialize(
      record, sizeof(iree_hal_profile_dispatch_event_t), &iterator);
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status)) {
    iree_profile_typed_record_t typed_record;
    bool has_record = false;
    status = iree_profile_typed_record_iterator_next(&iterator, &typed_record,
                                                     &has_record);
    if (!iree_status_is_ok(status) || !has_record) break;

    iree_hal_profile_dispatch_event_t event;
    memcpy(&event, typed_record.contents.data, sizeof(event));
    ++context->total_dispatch_count;

    char numeric_buffer[128];
    iree_string_view_t key = iree_string_view_empty();
    const bool id_matches = iree_profile_dispatch_event_matches_id(
        &event, projection_mode, id_filter);
    if (id_matches) {
      status = iree_profile_dispatch_validate_event_metadata(
          context, record, &event, projection_mode);
    }
    if (iree_status_is_ok(status) && id_matches) {
      status = iree_profile_dispatch_resolve_key(
          context, record->header.physical_device_ordinal, event.executable_id,
          event.export_ordinal, numeric_buffer, sizeof(numeric_buffer), &key);
    }
    if (iree_status_is_ok(status) && !iree_string_view_is_empty(key) &&
        iree_profile_dispatch_key_matches(key, filter)) {
      ++context->matched_dispatch_count;
      const bool is_valid = event.start_tick != 0 && event.end_tick != 0 &&
                            event.end_tick >= event.start_tick;
      if (is_valid) {
        ++context->valid_dispatch_count;
        iree_profile_dispatch_record_top_event(context, record, &event);
      } else {
        ++context->invalid_dispatch_count;
      }
      if (emit_events) {
        iree_profile_dispatch_print_event_jsonl(
            record, &event, key, ns_per_tick, has_clock_fit, file);
      } else {
        iree_profile_dispatch_aggregate_t* aggregate = NULL;
        status = iree_profile_dispatch_get_aggregate(
            context, record->header.physical_device_ordinal,
            event.executable_id, event.export_ordinal, &aggregate);
        if (iree_status_is_ok(status)) {
          iree_profile_dispatch_record_aggregate_event(aggregate, &event);
        }
        if (iree_status_is_ok(status) && event.command_buffer_id != 0) {
          iree_profile_dispatch_command_aggregate_t* command_aggregate = NULL;
          status = iree_profile_dispatch_get_command_aggregate(
              context, event.command_buffer_id, event.submission_id,
              record->header.physical_device_ordinal,
              record->header.queue_ordinal, record->header.stream_id,
              &command_aggregate);
          if (iree_status_is_ok(status)) {
            iree_profile_dispatch_record_command_aggregate_event(
                command_aggregate, &event);
          }
        }
        if (iree_status_is_ok(status)) {
          iree_profile_dispatch_queue_aggregate_t* queue_aggregate = NULL;
          status = iree_profile_dispatch_get_queue_aggregate(
              context, record->header.physical_device_ordinal,
              record->header.queue_ordinal, record->header.stream_id,
              event.submission_id, &queue_aggregate);
          if (iree_status_is_ok(status)) {
            iree_profile_dispatch_record_queue_aggregate_event(queue_aggregate,
                                                               &event);
          }
        }
      }
    }
  }
  return status;
}

iree_status_t iree_profile_dispatch_process_queue_event_records(
    iree_profile_dispatch_context_t* context,
    const iree_hal_profile_file_record_t* record, iree_string_view_t filter,
    int64_t id_filter) {
  iree_profile_typed_record_iterator_t iterator;
  iree_profile_typed_record_iterator_initialize(
      record, sizeof(iree_hal_profile_queue_event_t), &iterator);
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status)) {
    iree_profile_typed_record_t typed_record;
    bool has_record = false;
    status = iree_profile_typed_record_iterator_next(&iterator, &typed_record,
                                                     &has_record);
    if (!iree_status_is_ok(status) || !has_record) break;

    iree_hal_profile_queue_event_t event;
    memcpy(&event, typed_record.contents.data, sizeof(event));
    ++context->total_queue_event_count;
    if (iree_profile_queue_event_matches(&event, id_filter, filter)) {
      ++context->matched_queue_event_count;
      const iree_profile_dispatch_queue_t* queue_info =
          iree_profile_dispatch_find_queue(
              context, event.physical_device_ordinal, event.queue_ordinal,
              event.stream_id);
      if (!queue_info) {
        status = iree_make_status(
            IREE_STATUS_DATA_LOSS,
            "queue event references missing queue metadata "
            "device=%u queue=%u stream=%" PRIu64 " submission=%" PRIu64,
            event.physical_device_ordinal, event.queue_ordinal, event.stream_id,
            event.submission_id);
      }
      if (iree_status_is_ok(status)) {
        status = iree_profile_dispatch_append_queue_event(context, &event);
      }
    }
  }
  return status;
}

static iree_status_t iree_profile_dispatch_process_queue_device_event_records(
    iree_profile_dispatch_context_t* context,
    const iree_hal_profile_file_record_t* record, iree_string_view_t filter,
    int64_t id_filter) {
  iree_profile_typed_record_iterator_t iterator;
  iree_profile_typed_record_iterator_initialize(
      record, sizeof(iree_hal_profile_queue_device_event_t), &iterator);
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status)) {
    iree_profile_typed_record_t typed_record;
    bool has_record = false;
    status = iree_profile_typed_record_iterator_next(&iterator, &typed_record,
                                                     &has_record);
    if (!iree_status_is_ok(status) || !has_record) break;

    iree_hal_profile_queue_device_event_t event;
    memcpy(&event, typed_record.contents.data, sizeof(event));
    ++context->total_queue_device_event_count;
    if (iree_profile_queue_device_event_matches(&event, id_filter, filter)) {
      ++context->matched_queue_device_event_count;
      const iree_profile_dispatch_queue_t* queue_info =
          iree_profile_dispatch_find_queue(
              context, event.physical_device_ordinal, event.queue_ordinal,
              event.stream_id);
      if (!queue_info) {
        status = iree_make_status(
            IREE_STATUS_DATA_LOSS,
            "queue device event references missing queue metadata "
            "device=%u queue=%u stream=%" PRIu64 " submission=%" PRIu64,
            event.physical_device_ordinal, event.queue_ordinal, event.stream_id,
            event.submission_id);
      }
      if (iree_status_is_ok(status)) {
        status =
            iree_profile_dispatch_append_queue_device_event(context, &event);
      }
    }
  }
  return status;
}

iree_status_t iree_profile_dispatch_process_events_record(
    iree_profile_dispatch_context_t* context,
    const iree_hal_profile_file_record_t* record, iree_string_view_t filter,
    iree_profile_projection_mode_t projection_mode, int64_t id_filter,
    bool emit_events, FILE* file) {
  if (record->header.record_type != IREE_HAL_PROFILE_FILE_RECORD_TYPE_CHUNK) {
    return iree_ok_status();
  }
  if (!iree_string_view_equal(record->content_type,
                              IREE_HAL_PROFILE_CONTENT_TYPE_DISPATCH_EVENTS)) {
    if (projection_mode == IREE_PROFILE_PROJECTION_MODE_QUEUE &&
        iree_string_view_equal(record->content_type,
                               IREE_HAL_PROFILE_CONTENT_TYPE_QUEUE_EVENTS)) {
      return iree_profile_dispatch_process_queue_event_records(
          context, record, filter, id_filter);
    }
    if (projection_mode == IREE_PROFILE_PROJECTION_MODE_QUEUE &&
        iree_string_view_equal(
            record->content_type,
            IREE_HAL_PROFILE_CONTENT_TYPE_QUEUE_DEVICE_EVENTS)) {
      return iree_profile_dispatch_process_queue_device_event_records(
          context, record, filter, id_filter);
    }
    return iree_ok_status();
  }
  return iree_profile_dispatch_process_event_records(
      context, record, filter, projection_mode, id_filter, emit_events, file);
}

const iree_profile_dispatch_device_t* iree_profile_dispatch_find_device(
    const iree_profile_dispatch_context_t* context,
    uint32_t physical_device_ordinal) {
  for (iree_host_size_t i = 0; i < context->device_count; ++i) {
    if (context->devices[i].physical_device_ordinal ==
        physical_device_ordinal) {
      return &context->devices[i];
    }
  }
  return NULL;
}

double iree_profile_dispatch_span_ticks(uint64_t earliest_start_tick,
                                        uint64_t latest_end_tick) {
  if (earliest_start_tick == UINT64_MAX || latest_end_tick == 0 ||
      latest_end_tick < earliest_start_tick) {
    return 0.0;
  }
  return (double)(latest_end_tick - earliest_start_tick);
}

iree_status_t iree_profile_command_operation_resolve_key(
    const iree_profile_dispatch_context_t* context,
    const iree_hal_profile_command_operation_record_t* operation,
    char* numeric_buffer, iree_host_size_t numeric_buffer_capacity,
    iree_string_view_t* out_key) {
  *out_key = iree_make_cstring_view(
      iree_profile_command_operation_type_name(operation->type));
  if (operation->type != IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_DISPATCH ||
      operation->executable_id == 0 ||
      operation->export_ordinal == UINT32_MAX) {
    return iree_ok_status();
  }

  const iree_profile_dispatch_command_buffer_t* command_buffer =
      iree_profile_dispatch_find_command_buffer(context,
                                                operation->command_buffer_id);
  if (!command_buffer) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "command operation references missing command-buffer metadata "
        "command_buffer=%" PRIu64 " command_index=%u",
        operation->command_buffer_id, operation->command_index);
  }
  return iree_profile_dispatch_resolve_key(
      context, command_buffer->record.physical_device_ordinal,
      operation->executable_id, operation->export_ordinal, numeric_buffer,
      numeric_buffer_capacity, out_key);
}
