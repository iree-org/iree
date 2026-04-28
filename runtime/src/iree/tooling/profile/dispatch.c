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
  iree_profile_index_deinitialize(&context->host_command_aggregate_index,
                                  context->host_allocator);
  iree_profile_index_deinitialize(&context->host_dispatch_aggregate_index,
                                  context->host_allocator);
  iree_profile_index_deinitialize(&context->queue_aggregate_index,
                                  context->host_allocator);
  iree_profile_index_deinitialize(&context->command_aggregate_index,
                                  context->host_allocator);
  iree_profile_index_deinitialize(&context->aggregate_index,
                                  context->host_allocator);
  iree_allocator_free(context->host_allocator,
                      context->host_command_aggregates);
  iree_allocator_free(context->host_allocator,
                      context->host_dispatch_aggregates);
  iree_allocator_free(context->host_allocator, context->queue_aggregates);
  iree_allocator_free(context->host_allocator, context->command_aggregates);
  iree_allocator_free(context->host_allocator, context->aggregates);
  iree_profile_queue_event_query_deinitialize(&context->queue_query);
  iree_profile_model_deinitialize(&context->model);
  memset(context, 0, sizeof(*context));
}

typedef struct iree_profile_dispatch_export_lookup_t {
  // Dispatch context owning candidate aggregate rows.
  const iree_profile_dispatch_context_t* context;
  // Session-local physical device ordinal.
  uint32_t physical_device_ordinal;
  // Producer-local executable identifier.
  uint64_t executable_id;
  // Export ordinal within |executable_id|.
  uint32_t export_ordinal;
} iree_profile_dispatch_export_lookup_t;

typedef struct iree_profile_dispatch_command_lookup_t {
  // Dispatch context owning candidate aggregate rows.
  const iree_profile_dispatch_context_t* context;
  // Producer-local command-buffer identifier.
  uint64_t command_buffer_id;
  // Queue submission epoch containing the command buffer execution.
  uint64_t submission_id;
  // Session-local physical device ordinal.
  uint32_t physical_device_ordinal;
  // Session-local queue ordinal.
  uint32_t queue_ordinal;
  // Producer-defined stream identifier.
  uint64_t stream_id;
} iree_profile_dispatch_command_lookup_t;

typedef struct iree_profile_dispatch_queue_lookup_t {
  // Dispatch context owning candidate aggregate rows.
  const iree_profile_dispatch_context_t* context;
  // Session-local physical device ordinal.
  uint32_t physical_device_ordinal;
  // Session-local queue ordinal.
  uint32_t queue_ordinal;
  // Producer-defined stream identifier.
  uint64_t stream_id;
  // Queue submission epoch.
  uint64_t submission_id;
} iree_profile_dispatch_queue_lookup_t;

static uint64_t iree_profile_dispatch_export_hash(
    uint32_t physical_device_ordinal, uint64_t executable_id,
    uint32_t export_ordinal) {
  uint64_t hash = iree_profile_index_mix_u64(physical_device_ordinal);
  hash = iree_profile_index_combine_u64(hash, executable_id);
  return iree_profile_index_combine_u64(hash, export_ordinal);
}

static uint64_t iree_profile_dispatch_command_hash(
    uint64_t command_buffer_id, uint64_t submission_id,
    uint32_t physical_device_ordinal, uint32_t queue_ordinal,
    uint64_t stream_id) {
  uint64_t hash = iree_profile_index_mix_u64(command_buffer_id);
  hash = iree_profile_index_combine_u64(hash, submission_id);
  hash = iree_profile_index_combine_u64(hash, physical_device_ordinal);
  hash = iree_profile_index_combine_u64(hash, queue_ordinal);
  return iree_profile_index_combine_u64(hash, stream_id);
}

static uint64_t iree_profile_dispatch_queue_hash(
    uint32_t physical_device_ordinal, uint32_t queue_ordinal,
    uint64_t stream_id, uint64_t submission_id) {
  uint64_t hash = iree_profile_index_mix_u64(physical_device_ordinal);
  hash = iree_profile_index_combine_u64(hash, queue_ordinal);
  hash = iree_profile_index_combine_u64(hash, stream_id);
  return iree_profile_index_combine_u64(hash, submission_id);
}

static bool iree_profile_dispatch_aggregate_matches(const void* user_data,
                                                    iree_host_size_t value) {
  const iree_profile_dispatch_export_lookup_t* lookup =
      (const iree_profile_dispatch_export_lookup_t*)user_data;
  const iree_profile_dispatch_aggregate_t* aggregate =
      &lookup->context->aggregates[value];
  return aggregate->physical_device_ordinal ==
             lookup->physical_device_ordinal &&
         aggregate->executable_id == lookup->executable_id &&
         aggregate->export_ordinal == lookup->export_ordinal;
}

static bool iree_profile_dispatch_command_aggregate_matches(
    const void* user_data, iree_host_size_t value) {
  const iree_profile_dispatch_command_lookup_t* lookup =
      (const iree_profile_dispatch_command_lookup_t*)user_data;
  const iree_profile_dispatch_command_aggregate_t* aggregate =
      &lookup->context->command_aggregates[value];
  return aggregate->command_buffer_id == lookup->command_buffer_id &&
         aggregate->submission_id == lookup->submission_id &&
         aggregate->physical_device_ordinal ==
             lookup->physical_device_ordinal &&
         aggregate->queue_ordinal == lookup->queue_ordinal &&
         aggregate->stream_id == lookup->stream_id;
}

static bool iree_profile_dispatch_queue_aggregate_matches(
    const void* user_data, iree_host_size_t value) {
  const iree_profile_dispatch_queue_lookup_t* lookup =
      (const iree_profile_dispatch_queue_lookup_t*)user_data;
  const iree_profile_dispatch_queue_aggregate_t* aggregate =
      &lookup->context->queue_aggregates[value];
  return aggregate->physical_device_ordinal ==
             lookup->physical_device_ordinal &&
         aggregate->queue_ordinal == lookup->queue_ordinal &&
         aggregate->stream_id == lookup->stream_id &&
         aggregate->submission_id == lookup->submission_id;
}

static bool iree_profile_dispatch_host_aggregate_matches(
    const void* user_data, iree_host_size_t value) {
  const iree_profile_dispatch_export_lookup_t* lookup =
      (const iree_profile_dispatch_export_lookup_t*)user_data;
  const iree_profile_host_dispatch_aggregate_t* aggregate =
      &lookup->context->host_dispatch_aggregates[value];
  return aggregate->physical_device_ordinal ==
             lookup->physical_device_ordinal &&
         aggregate->executable_id == lookup->executable_id &&
         aggregate->export_ordinal == lookup->export_ordinal;
}

static bool iree_profile_dispatch_host_command_aggregate_matches(
    const void* user_data, iree_host_size_t value) {
  const iree_profile_dispatch_command_lookup_t* lookup =
      (const iree_profile_dispatch_command_lookup_t*)user_data;
  const iree_profile_host_dispatch_command_aggregate_t* aggregate =
      &lookup->context->host_command_aggregates[value];
  return aggregate->command_buffer_id == lookup->command_buffer_id &&
         aggregate->submission_id == lookup->submission_id &&
         aggregate->physical_device_ordinal ==
             lookup->physical_device_ordinal &&
         aggregate->queue_ordinal == lookup->queue_ordinal &&
         aggregate->stream_id == lookup->stream_id;
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

static iree_status_t iree_profile_dispatch_validate_host_event_metadata(
    const iree_profile_dispatch_context_t* context,
    const iree_hal_profile_host_execution_event_t* event,
    iree_profile_projection_mode_t projection_mode) {
  const iree_profile_model_t* model = &context->model;
  const iree_profile_model_queue_t* queue_info =
      iree_profile_model_find_queue(model, event->physical_device_ordinal,
                                    event->queue_ordinal, event->stream_id);
  if (!queue_info) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "host dispatch event references missing queue metadata "
        "device=%u queue=%u stream=%" PRIu64 " submission=%" PRIu64,
        event->physical_device_ordinal, event->queue_ordinal, event->stream_id,
        event->submission_id);
  }
  if (projection_mode == IREE_PROFILE_PROJECTION_MODE_COMMAND &&
      event->command_buffer_id != 0 &&
      !iree_profile_model_find_command_buffer(model,
                                              event->command_buffer_id)) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "host dispatch event references missing command-buffer metadata "
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

static bool iree_profile_dispatch_host_event_matches_id(
    const iree_hal_profile_host_execution_event_t* event,
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

  const iree_profile_dispatch_export_lookup_t lookup = {
      .context = context,
      .physical_device_ordinal = physical_device_ordinal,
      .executable_id = executable_id,
      .export_ordinal = export_ordinal,
  };
  const uint64_t hash = iree_profile_dispatch_export_hash(
      physical_device_ordinal, executable_id, export_ordinal);
  iree_host_size_t existing_index = 0;
  if (iree_profile_index_find(&context->aggregate_index, hash,
                              iree_profile_dispatch_aggregate_matches, &lookup,
                              &existing_index)) {
    *out_aggregate = &context->aggregates[existing_index];
    return iree_ok_status();
  }

  if (context->aggregate_count + 1 > context->aggregate_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        context->host_allocator,
        iree_max((iree_host_size_t)16, context->aggregate_count + 1),
        sizeof(context->aggregates[0]), &context->aggregate_capacity,
        (void**)&context->aggregates));
  }

  const iree_host_size_t aggregate_index = context->aggregate_count;
  IREE_RETURN_IF_ERROR(iree_profile_index_insert(&context->aggregate_index,
                                                 context->host_allocator, hash,
                                                 aggregate_index));
  iree_profile_dispatch_aggregate_t* aggregate =
      &context->aggregates[aggregate_index];
  memset(aggregate, 0, sizeof(*aggregate));
  aggregate->physical_device_ordinal = physical_device_ordinal;
  aggregate->executable_id = executable_id;
  aggregate->export_ordinal = export_ordinal;
  aggregate->earliest_start_tick = UINT64_MAX;
  aggregate->minimum_ticks = UINT64_MAX;
  ++context->aggregate_count;
  *out_aggregate = aggregate;
  return iree_ok_status();
}

static iree_status_t iree_profile_dispatch_get_command_aggregate(
    iree_profile_dispatch_context_t* context, uint64_t command_buffer_id,
    uint64_t submission_id, uint32_t physical_device_ordinal,
    uint32_t queue_ordinal, uint64_t stream_id,
    iree_profile_dispatch_command_aggregate_t** out_aggregate) {
  *out_aggregate = NULL;

  const iree_profile_dispatch_command_lookup_t lookup = {
      .context = context,
      .command_buffer_id = command_buffer_id,
      .submission_id = submission_id,
      .physical_device_ordinal = physical_device_ordinal,
      .queue_ordinal = queue_ordinal,
      .stream_id = stream_id,
  };
  const uint64_t hash = iree_profile_dispatch_command_hash(
      command_buffer_id, submission_id, physical_device_ordinal, queue_ordinal,
      stream_id);
  iree_host_size_t existing_index = 0;
  if (iree_profile_index_find(&context->command_aggregate_index, hash,
                              iree_profile_dispatch_command_aggregate_matches,
                              &lookup, &existing_index)) {
    *out_aggregate = &context->command_aggregates[existing_index];
    return iree_ok_status();
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

  const iree_host_size_t aggregate_index = context->command_aggregate_count;
  IREE_RETURN_IF_ERROR(iree_profile_index_insert(
      &context->command_aggregate_index, context->host_allocator, hash,
      aggregate_index));
  iree_profile_dispatch_command_aggregate_t* aggregate =
      &context->command_aggregates[aggregate_index];
  memset(aggregate, 0, sizeof(*aggregate));
  aggregate->command_buffer_id = command_buffer_id;
  aggregate->submission_id = submission_id;
  aggregate->physical_device_ordinal = physical_device_ordinal;
  aggregate->queue_ordinal = queue_ordinal;
  aggregate->stream_id = stream_id;
  aggregate->earliest_start_tick = UINT64_MAX;
  ++context->command_aggregate_count;
  *out_aggregate = aggregate;
  return iree_ok_status();
}

static iree_status_t iree_profile_dispatch_get_queue_aggregate(
    iree_profile_dispatch_context_t* context, uint32_t physical_device_ordinal,
    uint32_t queue_ordinal, uint64_t stream_id, uint64_t submission_id,
    iree_profile_dispatch_queue_aggregate_t** out_aggregate) {
  *out_aggregate = NULL;

  const iree_profile_dispatch_queue_lookup_t lookup = {
      .context = context,
      .physical_device_ordinal = physical_device_ordinal,
      .queue_ordinal = queue_ordinal,
      .stream_id = stream_id,
      .submission_id = submission_id,
  };
  const uint64_t hash = iree_profile_dispatch_queue_hash(
      physical_device_ordinal, queue_ordinal, stream_id, submission_id);
  iree_host_size_t existing_index = 0;
  if (iree_profile_index_find(&context->queue_aggregate_index, hash,
                              iree_profile_dispatch_queue_aggregate_matches,
                              &lookup, &existing_index)) {
    *out_aggregate = &context->queue_aggregates[existing_index];
    return iree_ok_status();
  }

  if (context->queue_aggregate_count + 1 > context->queue_aggregate_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        context->host_allocator,
        iree_max((iree_host_size_t)16, context->queue_aggregate_count + 1),
        sizeof(context->queue_aggregates[0]),
        &context->queue_aggregate_capacity,
        (void**)&context->queue_aggregates));
  }

  const iree_host_size_t aggregate_index = context->queue_aggregate_count;
  IREE_RETURN_IF_ERROR(iree_profile_index_insert(
      &context->queue_aggregate_index, context->host_allocator, hash,
      aggregate_index));
  iree_profile_dispatch_queue_aggregate_t* aggregate =
      &context->queue_aggregates[aggregate_index];
  memset(aggregate, 0, sizeof(*aggregate));
  aggregate->physical_device_ordinal = physical_device_ordinal;
  aggregate->queue_ordinal = queue_ordinal;
  aggregate->stream_id = stream_id;
  aggregate->submission_id = submission_id;
  aggregate->earliest_start_tick = UINT64_MAX;
  ++context->queue_aggregate_count;
  *out_aggregate = aggregate;
  return iree_ok_status();
}

static iree_status_t iree_profile_dispatch_get_host_aggregate(
    iree_profile_dispatch_context_t* context, uint32_t physical_device_ordinal,
    uint64_t executable_id, uint32_t export_ordinal,
    iree_profile_host_dispatch_aggregate_t** out_aggregate) {
  *out_aggregate = NULL;

  const iree_profile_dispatch_export_lookup_t lookup = {
      .context = context,
      .physical_device_ordinal = physical_device_ordinal,
      .executable_id = executable_id,
      .export_ordinal = export_ordinal,
  };
  const uint64_t hash = iree_profile_dispatch_export_hash(
      physical_device_ordinal, executable_id, export_ordinal);
  iree_host_size_t existing_index = 0;
  if (iree_profile_index_find(&context->host_dispatch_aggregate_index, hash,
                              iree_profile_dispatch_host_aggregate_matches,
                              &lookup, &existing_index)) {
    *out_aggregate = &context->host_dispatch_aggregates[existing_index];
    return iree_ok_status();
  }

  if (context->host_dispatch_aggregate_count + 1 >
      context->host_dispatch_aggregate_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        context->host_allocator,
        iree_max((iree_host_size_t)16,
                 context->host_dispatch_aggregate_count + 1),
        sizeof(context->host_dispatch_aggregates[0]),
        &context->host_dispatch_aggregate_capacity,
        (void**)&context->host_dispatch_aggregates));
  }

  const iree_host_size_t aggregate_index =
      context->host_dispatch_aggregate_count;
  IREE_RETURN_IF_ERROR(iree_profile_index_insert(
      &context->host_dispatch_aggregate_index, context->host_allocator, hash,
      aggregate_index));
  iree_profile_host_dispatch_aggregate_t* aggregate =
      &context->host_dispatch_aggregates[aggregate_index];
  memset(aggregate, 0, sizeof(*aggregate));
  aggregate->physical_device_ordinal = physical_device_ordinal;
  aggregate->executable_id = executable_id;
  aggregate->export_ordinal = export_ordinal;
  aggregate->earliest_start_host_time_ns = INT64_MAX;
  aggregate->minimum_ns = INT64_MAX;
  ++context->host_dispatch_aggregate_count;
  *out_aggregate = aggregate;
  return iree_ok_status();
}

static iree_status_t iree_profile_dispatch_get_host_command_aggregate(
    iree_profile_dispatch_context_t* context, uint64_t command_buffer_id,
    uint64_t submission_id, uint32_t physical_device_ordinal,
    uint32_t queue_ordinal, uint64_t stream_id,
    iree_profile_host_dispatch_command_aggregate_t** out_aggregate) {
  *out_aggregate = NULL;

  const iree_profile_dispatch_command_lookup_t lookup = {
      .context = context,
      .command_buffer_id = command_buffer_id,
      .submission_id = submission_id,
      .physical_device_ordinal = physical_device_ordinal,
      .queue_ordinal = queue_ordinal,
      .stream_id = stream_id,
  };
  const uint64_t hash = iree_profile_dispatch_command_hash(
      command_buffer_id, submission_id, physical_device_ordinal, queue_ordinal,
      stream_id);
  iree_host_size_t existing_index = 0;
  if (iree_profile_index_find(
          &context->host_command_aggregate_index, hash,
          iree_profile_dispatch_host_command_aggregate_matches, &lookup,
          &existing_index)) {
    *out_aggregate = &context->host_command_aggregates[existing_index];
    return iree_ok_status();
  }

  if (context->host_command_aggregate_count + 1 >
      context->host_command_aggregate_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        context->host_allocator,
        iree_max((iree_host_size_t)16,
                 context->host_command_aggregate_count + 1),
        sizeof(context->host_command_aggregates[0]),
        &context->host_command_aggregate_capacity,
        (void**)&context->host_command_aggregates));
  }

  const iree_host_size_t aggregate_index =
      context->host_command_aggregate_count;
  IREE_RETURN_IF_ERROR(iree_profile_index_insert(
      &context->host_command_aggregate_index, context->host_allocator, hash,
      aggregate_index));
  iree_profile_host_dispatch_command_aggregate_t* aggregate =
      &context->host_command_aggregates[aggregate_index];
  memset(aggregate, 0, sizeof(*aggregate));
  aggregate->command_buffer_id = command_buffer_id;
  aggregate->submission_id = submission_id;
  aggregate->physical_device_ordinal = physical_device_ordinal;
  aggregate->queue_ordinal = queue_ordinal;
  aggregate->stream_id = stream_id;
  aggregate->earliest_start_host_time_ns = INT64_MAX;
  ++context->host_command_aggregate_count;
  *out_aggregate = aggregate;
  return iree_ok_status();
}

static bool iree_profile_dispatch_accumulate_ticks(uint64_t* total_ticks,
                                                   uint64_t duration_ticks) {
  if (duration_ticks > UINT64_MAX - *total_ticks) return false;
  *total_ticks += duration_ticks;
  return true;
}

static bool iree_profile_dispatch_accumulate_ns(int64_t* total_ns,
                                                int64_t duration_ns) {
  if (duration_ns < 0 || duration_ns > INT64_MAX - *total_ns) return false;
  *total_ns += duration_ns;
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

static bool iree_profile_dispatch_record_host_aggregate_event(
    iree_profile_host_dispatch_aggregate_t* aggregate,
    const iree_hal_profile_host_execution_event_t* event) {
  ++aggregate->dispatch_count;
  memcpy(aggregate->last_workgroup_count, event->workgroup_count,
         sizeof(aggregate->last_workgroup_count));
  memcpy(aggregate->last_workgroup_size, event->workgroup_size,
         sizeof(aggregate->last_workgroup_size));

  if (event->start_host_time_ns < 0 ||
      event->end_host_time_ns < event->start_host_time_ns) {
    ++aggregate->invalid_count;
    return true;
  }

  const int64_t duration_ns =
      event->end_host_time_ns - event->start_host_time_ns;
  ++aggregate->valid_count;
  aggregate->earliest_start_host_time_ns = iree_min(
      aggregate->earliest_start_host_time_ns, event->start_host_time_ns);
  aggregate->latest_end_host_time_ns =
      iree_max(aggregate->latest_end_host_time_ns, event->end_host_time_ns);
  aggregate->minimum_ns = iree_min(aggregate->minimum_ns, duration_ns);
  aggregate->maximum_ns = iree_max(aggregate->maximum_ns, duration_ns);
  if (!iree_profile_dispatch_accumulate_ns(&aggregate->total_ns, duration_ns) ||
      !iree_profile_dispatch_accumulate_ns(
          &aggregate->total_tile_duration_sum_ns,
          event->tile_duration_sum_ns)) {
    return false;
  }
  if (event->tile_count > UINT64_MAX - aggregate->total_tile_count) {
    return false;
  }
  aggregate->total_tile_count += event->tile_count;

  const double duration = (double)duration_ns;
  const double delta = duration - aggregate->mean_ns;
  aggregate->mean_ns += delta / (double)aggregate->valid_count;
  const double delta2 = duration - aggregate->mean_ns;
  aggregate->m2_ns += delta * delta2;
  return true;
}

static bool iree_profile_dispatch_record_host_command_aggregate_event(
    iree_profile_host_dispatch_command_aggregate_t* aggregate,
    const iree_hal_profile_host_execution_event_t* event) {
  ++aggregate->dispatch_count;
  if (event->start_host_time_ns < 0 ||
      event->end_host_time_ns < event->start_host_time_ns) {
    ++aggregate->invalid_count;
    return true;
  }
  const int64_t duration_ns =
      event->end_host_time_ns - event->start_host_time_ns;
  ++aggregate->valid_count;
  aggregate->earliest_start_host_time_ns = iree_min(
      aggregate->earliest_start_host_time_ns, event->start_host_time_ns);
  aggregate->latest_end_host_time_ns =
      iree_max(aggregate->latest_end_host_time_ns, event->end_host_time_ns);
  if (!iree_profile_dispatch_accumulate_ns(&aggregate->total_ns, duration_ns) ||
      !iree_profile_dispatch_accumulate_ns(
          &aggregate->total_tile_duration_sum_ns,
          event->tile_duration_sum_ns)) {
    return false;
  }
  if (event->tile_count > UINT64_MAX - aggregate->total_tile_count) {
    return false;
  }
  aggregate->total_tile_count += event->tile_count;
  return true;
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

static void iree_profile_dispatch_record_top_host_event(
    iree_profile_dispatch_context_t* context,
    const iree_hal_profile_host_execution_event_t* event) {
  if (event->start_host_time_ns < 0 ||
      event->end_host_time_ns < event->start_host_time_ns) {
    return;
  }

  const int64_t duration_ns =
      event->end_host_time_ns - event->start_host_time_ns;
  iree_host_size_t target_index = context->top_host_dispatch_count;
  if (context->top_host_dispatch_count <
      IREE_PROFILE_DISPATCH_TOP_EVENT_COUNT) {
    ++context->top_host_dispatch_count;
  } else {
    target_index = 0;
    for (iree_host_size_t i = 1; i < context->top_host_dispatch_count; ++i) {
      if (context->top_host_dispatches[i].duration_ns <
          context->top_host_dispatches[target_index].duration_ns) {
        target_index = i;
      }
    }
    if (duration_ns <= context->top_host_dispatches[target_index].duration_ns) {
      return;
    }
  }

  iree_profile_host_dispatch_top_event_t* top_event =
      &context->top_host_dispatches[target_index];
  top_event->physical_device_ordinal = event->physical_device_ordinal;
  top_event->queue_ordinal = event->queue_ordinal;
  top_event->stream_id = event->stream_id;
  top_event->duration_ns = duration_ns;
  top_event->event = *event;
}

static iree_status_t iree_profile_dispatch_process_event_records(
    iree_profile_dispatch_context_t* context,
    const iree_hal_profile_file_record_t* record, iree_string_view_t filter,
    iree_profile_projection_mode_t projection_mode, int64_t id_filter,
    bool aggregate_events,
    iree_profile_dispatch_event_callback_t event_callback) {
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
      if (event_callback.fn) {
        const iree_profile_dispatch_event_row_t event_row = {
            .file_record = record,
            .event = &event,
            .key = key,
            .clock_fit = &clock_fit,
            .has_clock_fit = has_clock_fit,
        };
        status = event_callback.fn(event_callback.user_data, &event_row);
      }
      if (iree_status_is_ok(status) && aggregate_events) {
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

static iree_status_t iree_profile_dispatch_process_host_execution_records(
    iree_profile_dispatch_context_t* context,
    const iree_hal_profile_file_record_t* record, iree_string_view_t filter,
    iree_profile_projection_mode_t projection_mode, int64_t id_filter,
    bool aggregate_events,
    iree_profile_dispatch_event_callback_t event_callback) {
  const iree_profile_model_t* model = &context->model;
  iree_profile_typed_record_iterator_t iterator;
  iree_profile_typed_record_iterator_initialize(
      record, sizeof(iree_hal_profile_host_execution_event_t), &iterator);
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status)) {
    iree_profile_typed_record_t typed_record;
    bool has_record = false;
    status = iree_profile_typed_record_iterator_next(&iterator, &typed_record,
                                                     &has_record);
    if (!iree_status_is_ok(status) || !has_record) break;

    iree_hal_profile_host_execution_event_t event;
    memcpy(&event, typed_record.contents.data, sizeof(event));
    if (event.type != IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_DISPATCH) {
      continue;
    }
    ++context->total_host_dispatch_count;

    char numeric_buffer[128];
    iree_string_view_t key = iree_string_view_empty();
    const bool id_matches = iree_profile_dispatch_host_event_matches_id(
        &event, projection_mode, id_filter);
    if (id_matches) {
      status = iree_profile_dispatch_validate_host_event_metadata(
          context, &event, projection_mode);
    }
    if (iree_status_is_ok(status) && id_matches) {
      status = iree_profile_model_resolve_dispatch_key(
          model, event.physical_device_ordinal, event.executable_id,
          event.export_ordinal, numeric_buffer, sizeof(numeric_buffer), &key);
    }
    if (iree_status_is_ok(status) && !iree_string_view_is_empty(key) &&
        iree_profile_key_matches(key, filter)) {
      ++context->matched_host_dispatch_count;
      const bool is_valid = event.start_host_time_ns >= 0 &&
                            event.end_host_time_ns >= event.start_host_time_ns;
      if (is_valid) {
        ++context->valid_host_dispatch_count;
        iree_profile_dispatch_record_top_host_event(context, &event);
      } else {
        ++context->invalid_host_dispatch_count;
      }
      if (event_callback.host_fn) {
        const iree_profile_host_dispatch_event_row_t event_row = {
            .event = &event,
            .key = key,
        };
        status = event_callback.host_fn(event_callback.user_data, &event_row);
      }
      if (iree_status_is_ok(status) && aggregate_events) {
        iree_profile_host_dispatch_aggregate_t* aggregate = NULL;
        status = iree_profile_dispatch_get_host_aggregate(
            context, event.physical_device_ordinal, event.executable_id,
            event.export_ordinal, &aggregate);
        if (iree_status_is_ok(status) &&
            !iree_profile_dispatch_record_host_aggregate_event(aggregate,
                                                               &event)) {
          status = iree_make_status(
              IREE_STATUS_OUT_OF_RANGE,
              "host dispatch aggregate total overflow device=%u "
              "executable=%" PRIu64 " export=%u",
              event.physical_device_ordinal, event.executable_id,
              event.export_ordinal);
        }
        if (iree_status_is_ok(status) && event.command_buffer_id != 0) {
          iree_profile_host_dispatch_command_aggregate_t* command_aggregate =
              NULL;
          status = iree_profile_dispatch_get_host_command_aggregate(
              context, event.command_buffer_id, event.submission_id,
              event.physical_device_ordinal, event.queue_ordinal,
              event.stream_id, &command_aggregate);
          if (iree_status_is_ok(status) &&
              !iree_profile_dispatch_record_host_command_aggregate_event(
                  command_aggregate, &event)) {
            status = iree_make_status(
                IREE_STATUS_OUT_OF_RANGE,
                "host command aggregate total overflow command_buffer=%" PRIu64
                " submission=%" PRIu64,
                event.command_buffer_id, event.submission_id);
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
    bool aggregate_events,
    iree_profile_dispatch_event_callback_t event_callback) {
  if (record->header.record_type != IREE_HAL_PROFILE_FILE_RECORD_TYPE_CHUNK) {
    return iree_ok_status();
  }
  if (!iree_string_view_equal(record->content_type,
                              IREE_HAL_PROFILE_CONTENT_TYPE_DISPATCH_EVENTS)) {
    if (iree_string_view_equal(
            record->content_type,
            IREE_HAL_PROFILE_CONTENT_TYPE_HOST_EXECUTION_EVENTS)) {
      if (projection_mode == IREE_PROFILE_PROJECTION_MODE_QUEUE) {
        IREE_RETURN_IF_ERROR(iree_profile_queue_event_query_process_record(
            &context->queue_query, &context->model, record, filter, id_filter));
      }
      return iree_profile_dispatch_process_host_execution_records(
          context, record, filter, projection_mode, id_filter, aggregate_events,
          event_callback);
    }
    if (projection_mode == IREE_PROFILE_PROJECTION_MODE_QUEUE) {
      return iree_profile_queue_event_query_process_record(
          &context->queue_query, &context->model, record, filter, id_filter);
    }
    return iree_ok_status();
  }
  return iree_profile_dispatch_process_event_records(
      context, record, filter, projection_mode, id_filter, aggregate_events,
      event_callback);
}
