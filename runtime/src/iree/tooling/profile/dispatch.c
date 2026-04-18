// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tooling/profile/dispatch.h"

#include <string.h>

#include "iree/tooling/profile/reader.h"

void iree_profile_dispatch_context_initialize(
    iree_allocator_t host_allocator,
    iree_profile_dispatch_context_t* out_context) {
  memset(out_context, 0, sizeof(*out_context));
  out_context->host_allocator = host_allocator;
  iree_profile_model_initialize(host_allocator, &out_context->model);
  iree_profile_queue_event_query_initialize(host_allocator,
                                            &out_context->queue_query);
}

void iree_profile_dispatch_context_deinitialize(
    iree_profile_dispatch_context_t* context) {
  iree_allocator_free(context->host_allocator, context->queue_aggregates);
  iree_allocator_free(context->host_allocator, context->command_aggregates);
  iree_allocator_free(context->host_allocator, context->aggregates);
  iree_profile_queue_event_query_deinitialize(&context->queue_query);
  iree_profile_model_deinitialize(&context->model);
  memset(context, 0, sizeof(*context));
}

static iree_status_t iree_profile_dispatch_validate_event_metadata(
    const iree_profile_dispatch_context_t* context,
    const iree_hal_profile_file_record_t* record,
    const iree_hal_profile_dispatch_event_t* event,
    iree_profile_projection_mode_t projection_mode) {
  const iree_profile_model_t* model = &context->model;
  const iree_profile_model_queue_t* queue_info = iree_profile_model_find_queue(
      model, record->header.physical_device_ordinal,
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
      !iree_profile_model_find_command_buffer(model,
                                              event->command_buffer_id)) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "dispatch event references missing command-buffer metadata "
        "command_buffer=%" PRIu64 " submission=%" PRIu64,
        event->command_buffer_id, event->submission_id);
  }
  return iree_ok_status();
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

static bool iree_profile_dispatch_accumulate_ticks(uint64_t* total_ticks,
                                                   uint64_t duration_ticks) {
  if (duration_ticks > UINT64_MAX - *total_ticks) return false;
  *total_ticks += duration_ticks;
  return true;
}

static bool iree_profile_dispatch_record_aggregate_event(
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
    return true;
  }

  const uint64_t duration_ticks = event->end_tick - event->start_tick;
  ++aggregate->valid_count;
  aggregate->earliest_start_tick =
      iree_min(aggregate->earliest_start_tick, event->start_tick);
  aggregate->latest_end_tick =
      iree_max(aggregate->latest_end_tick, event->end_tick);
  aggregate->minimum_ticks = iree_min(aggregate->minimum_ticks, duration_ticks);
  aggregate->maximum_ticks = iree_max(aggregate->maximum_ticks, duration_ticks);
  if (!iree_profile_dispatch_accumulate_ticks(&aggregate->total_ticks,
                                              duration_ticks)) {
    return false;
  }

  const double duration = (double)duration_ticks;
  const double delta = duration - aggregate->mean_ticks;
  aggregate->mean_ticks += delta / (double)aggregate->valid_count;
  const double delta2 = duration - aggregate->mean_ticks;
  aggregate->m2_ticks += delta * delta2;
  return true;
}

static bool iree_profile_dispatch_record_command_aggregate_event(
    iree_profile_dispatch_command_aggregate_t* aggregate,
    const iree_hal_profile_dispatch_event_t* event) {
  ++aggregate->dispatch_count;
  if (event->start_tick == 0 || event->end_tick == 0 ||
      event->end_tick < event->start_tick) {
    ++aggregate->invalid_count;
    return true;
  }
  ++aggregate->valid_count;
  aggregate->earliest_start_tick =
      iree_min(aggregate->earliest_start_tick, event->start_tick);
  aggregate->latest_end_tick =
      iree_max(aggregate->latest_end_tick, event->end_tick);
  return iree_profile_dispatch_accumulate_ticks(
      &aggregate->total_ticks, event->end_tick - event->start_tick);
}

static bool iree_profile_dispatch_record_queue_aggregate_event(
    iree_profile_dispatch_queue_aggregate_t* aggregate,
    const iree_hal_profile_dispatch_event_t* event) {
  ++aggregate->dispatch_count;
  if (event->start_tick == 0 || event->end_tick == 0 ||
      event->end_tick < event->start_tick) {
    ++aggregate->invalid_count;
    return true;
  }
  ++aggregate->valid_count;
  aggregate->earliest_start_tick =
      iree_min(aggregate->earliest_start_tick, event->start_tick);
  aggregate->latest_end_tick =
      iree_max(aggregate->latest_end_tick, event->end_tick);
  return iree_profile_dispatch_accumulate_ticks(
      &aggregate->total_ticks, event->end_tick - event->start_tick);
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
  if (context->top_dispatch_count < IREE_PROFILE_DISPATCH_TOP_EVENT_COUNT) {
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

static void iree_profile_dispatch_print_event_jsonl(
    const iree_hal_profile_file_record_t* file_record,
    const iree_hal_profile_dispatch_event_t* event, iree_string_view_t key,
    const iree_profile_model_clock_fit_t* clock_fit, bool has_clock_fit,
    FILE* file) {
  const bool is_valid = event->start_tick != 0 && event->end_tick != 0 &&
                        event->end_tick >= event->start_tick;
  const uint64_t duration_ticks =
      is_valid ? event->end_tick - event->start_tick : 0;
  int64_t duration_ns = 0;
  const bool has_duration_ns = has_clock_fit && is_valid &&
                               iree_profile_model_clock_fit_scale_ticks_to_ns(
                                   clock_fit, duration_ticks, &duration_ns);

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
  fprintf(file,
          ",\"device_tick_domain\":\"device_tick\""
          ",\"duration_time_domain\":\"device_tick_duration_ns\""
          ",\"duration_ns\":%" PRId64,
          has_duration_ns ? duration_ns : 0);
  fputs("}\n", file);
}

static iree_status_t iree_profile_dispatch_process_event_records(
    iree_profile_dispatch_context_t* context,
    const iree_hal_profile_file_record_t* record, iree_string_view_t filter,
    iree_profile_projection_mode_t projection_mode, int64_t id_filter,
    bool emit_events, FILE* file) {
  const iree_profile_model_t* model = &context->model;
  iree_profile_model_device_t* device = NULL;
  IREE_RETURN_IF_ERROR(iree_profile_model_ensure_device(
      &context->model, record->header.physical_device_ordinal, &device));
  iree_profile_model_clock_fit_t clock_fit;
  const bool has_clock_fit = iree_profile_model_device_try_fit_clock_exact(
      device, IREE_PROFILE_MODEL_CLOCK_TIME_DOMAIN_HOST_CPU_TIMESTAMP_NS,
      &clock_fit);

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
      status = iree_profile_model_resolve_dispatch_key(
          model, record->header.physical_device_ordinal, event.executable_id,
          event.export_ordinal, numeric_buffer, sizeof(numeric_buffer), &key);
    }
    if (iree_status_is_ok(status) && !iree_string_view_is_empty(key) &&
        iree_profile_key_matches(key, filter)) {
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
        iree_profile_dispatch_print_event_jsonl(record, &event, key, &clock_fit,
                                                has_clock_fit, file);
      } else {
        iree_profile_dispatch_aggregate_t* aggregate = NULL;
        status = iree_profile_dispatch_get_aggregate(
            context, record->header.physical_device_ordinal,
            event.executable_id, event.export_ordinal, &aggregate);
        if (iree_status_is_ok(status) &&
            !iree_profile_dispatch_record_aggregate_event(aggregate, &event)) {
          status =
              iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                               "dispatch aggregate tick total overflow "
                               "device=%u executable=%" PRIu64 " export=%u",
                               record->header.physical_device_ordinal,
                               event.executable_id, event.export_ordinal);
        }
        if (iree_status_is_ok(status) && event.command_buffer_id != 0) {
          iree_profile_dispatch_command_aggregate_t* command_aggregate = NULL;
          status = iree_profile_dispatch_get_command_aggregate(
              context, event.command_buffer_id, event.submission_id,
              record->header.physical_device_ordinal,
              record->header.queue_ordinal, record->header.stream_id,
              &command_aggregate);
          if (iree_status_is_ok(status) &&
              !iree_profile_dispatch_record_command_aggregate_event(
                  command_aggregate, &event)) {
            status = iree_make_status(
                IREE_STATUS_OUT_OF_RANGE,
                "command aggregate tick total overflow command_buffer=%" PRIu64
                " submission=%" PRIu64,
                event.command_buffer_id, event.submission_id);
          }
        }
        if (iree_status_is_ok(status)) {
          iree_profile_dispatch_queue_aggregate_t* queue_aggregate = NULL;
          status = iree_profile_dispatch_get_queue_aggregate(
              context, record->header.physical_device_ordinal,
              record->header.queue_ordinal, record->header.stream_id,
              event.submission_id, &queue_aggregate);
          if (iree_status_is_ok(status) &&
              !iree_profile_dispatch_record_queue_aggregate_event(
                  queue_aggregate, &event)) {
            status = iree_make_status(
                IREE_STATUS_OUT_OF_RANGE,
                "queue aggregate tick total overflow device=%u queue=%u"
                " stream=%" PRIu64 " submission=%" PRIu64,
                record->header.physical_device_ordinal,
                record->header.queue_ordinal, record->header.stream_id,
                event.submission_id);
          }
        }
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
    if (projection_mode == IREE_PROFILE_PROJECTION_MODE_QUEUE) {
      return iree_profile_queue_event_query_process_record(
          &context->queue_query, &context->model, record, filter, id_filter);
    }
    return iree_ok_status();
  }
  return iree_profile_dispatch_process_event_records(
      context, record, filter, projection_mode, id_filter, emit_events, file);
}
