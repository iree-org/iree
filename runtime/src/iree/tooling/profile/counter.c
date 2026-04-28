// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tooling/profile/counter.h"

#include <float.h>
#include <string.h>

#include "iree/tooling/profile/model.h"
#include "iree/tooling/profile/reader.h"

void iree_profile_counter_context_initialize(
    iree_allocator_t host_allocator,
    iree_profile_counter_context_t* out_context) {
  memset(out_context, 0, sizeof(*out_context));
  out_context->host_allocator = host_allocator;
}

void iree_profile_counter_context_deinitialize(
    iree_profile_counter_context_t* context) {
  iree_profile_index_deinitialize(&context->aggregate_index,
                                  context->host_allocator);
  iree_profile_index_deinitialize(&context->counter_index,
                                  context->host_allocator);
  iree_profile_index_deinitialize(&context->counter_set_index,
                                  context->host_allocator);
  iree_allocator_free(context->host_allocator, context->aggregates);
  iree_allocator_free(context->host_allocator, context->counters);
  iree_allocator_free(context->host_allocator, context->counter_sets);
  memset(context, 0, sizeof(*context));
}

typedef struct iree_profile_counter_set_lookup_t {
  // Counter context owning candidate counter-set rows.
  const iree_profile_counter_context_t* context;
  // Producer-local counter-set identifier.
  uint64_t counter_set_id;
} iree_profile_counter_set_lookup_t;

typedef struct iree_profile_counter_lookup_t {
  // Counter context owning candidate counter rows.
  const iree_profile_counter_context_t* context;
  // Producer-local counter-set identifier.
  uint64_t counter_set_id;
  // Counter ordinal within |counter_set_id|.
  uint32_t counter_ordinal;
} iree_profile_counter_lookup_t;

typedef struct iree_profile_counter_aggregate_lookup_t {
  // Counter context owning candidate aggregate rows.
  const iree_profile_counter_context_t* context;
  // Session-local physical device ordinal.
  uint32_t physical_device_ordinal;
  // Session-local queue ordinal.
  uint32_t queue_ordinal;
  // Scope measured by samples contributing to the aggregate.
  iree_hal_profile_counter_sample_scope_t scope;
  // Producer-defined stream identifier.
  uint64_t stream_id;
  // Producer-local counter-set identifier.
  uint64_t counter_set_id;
  // Counter ordinal within |counter_set_id|.
  uint32_t counter_ordinal;
  // Producer-local executable identifier.
  uint64_t executable_id;
  // Export ordinal within |executable_id|.
  uint32_t export_ordinal;
  // Process-local command-buffer identifier.
  uint64_t command_buffer_id;
} iree_profile_counter_aggregate_lookup_t;

static uint64_t iree_profile_counter_set_hash(uint64_t counter_set_id) {
  return iree_profile_index_mix_u64(counter_set_id);
}

static uint64_t iree_profile_counter_hash(uint64_t counter_set_id,
                                          uint32_t counter_ordinal) {
  uint64_t hash = iree_profile_counter_set_hash(counter_set_id);
  return iree_profile_index_combine_u64(hash, counter_ordinal);
}

static uint64_t iree_profile_counter_aggregate_hash(
    uint32_t physical_device_ordinal, uint32_t queue_ordinal,
    uint64_t stream_id, iree_hal_profile_counter_sample_scope_t scope,
    uint64_t counter_set_id, uint32_t counter_ordinal, uint64_t executable_id,
    uint32_t export_ordinal, uint64_t command_buffer_id) {
  uint64_t hash = iree_profile_index_mix_u64(physical_device_ordinal);
  hash = iree_profile_index_combine_u64(hash, queue_ordinal);
  hash = iree_profile_index_combine_u64(hash, stream_id);
  hash = iree_profile_index_combine_u64(hash, scope);
  hash = iree_profile_index_combine_u64(hash, counter_set_id);
  hash = iree_profile_index_combine_u64(hash, counter_ordinal);
  hash = iree_profile_index_combine_u64(hash, executable_id);
  hash = iree_profile_index_combine_u64(hash, export_ordinal);
  return iree_profile_index_combine_u64(hash, command_buffer_id);
}

static bool iree_profile_counter_set_matches(const void* user_data,
                                             iree_host_size_t value) {
  const iree_profile_counter_set_lookup_t* lookup =
      (const iree_profile_counter_set_lookup_t*)user_data;
  return lookup->context->counter_sets[value].record.counter_set_id ==
         lookup->counter_set_id;
}

static bool iree_profile_counter_matches(const void* user_data,
                                         iree_host_size_t value) {
  const iree_profile_counter_lookup_t* lookup =
      (const iree_profile_counter_lookup_t*)user_data;
  const iree_profile_counter_t* counter = &lookup->context->counters[value];
  return counter->record.counter_set_id == lookup->counter_set_id &&
         counter->record.counter_ordinal == lookup->counter_ordinal;
}

static bool iree_profile_counter_aggregate_matches(const void* user_data,
                                                   iree_host_size_t value) {
  const iree_profile_counter_aggregate_lookup_t* lookup =
      (const iree_profile_counter_aggregate_lookup_t*)user_data;
  const iree_profile_counter_aggregate_t* aggregate =
      &lookup->context->aggregates[value];
  return aggregate->physical_device_ordinal ==
             lookup->physical_device_ordinal &&
         aggregate->queue_ordinal == lookup->queue_ordinal &&
         aggregate->stream_id == lookup->stream_id &&
         aggregate->scope == lookup->scope &&
         aggregate->counter_set_id == lookup->counter_set_id &&
         aggregate->counter_ordinal == lookup->counter_ordinal &&
         aggregate->executable_id == lookup->executable_id &&
         aggregate->export_ordinal == lookup->export_ordinal &&
         aggregate->command_buffer_id == lookup->command_buffer_id;
}

static iree_status_t iree_profile_counter_append_counter_set(
    iree_profile_counter_context_t* context,
    const iree_profile_counter_set_t* counter_set) {
  const iree_profile_counter_set_lookup_t lookup = {
      .context = context,
      .counter_set_id = counter_set->record.counter_set_id,
  };
  const uint64_t hash =
      iree_profile_counter_set_hash(counter_set->record.counter_set_id);
  iree_host_size_t existing_index = 0;
  if (iree_profile_index_find(&context->counter_set_index, hash,
                              iree_profile_counter_set_matches, &lookup,
                              &existing_index)) {
    return iree_ok_status();
  }

  if (context->counter_set_count + 1 > context->counter_set_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        context->host_allocator,
        iree_max((iree_host_size_t)4, context->counter_set_count + 1),
        sizeof(context->counter_sets[0]), &context->counter_set_capacity,
        (void**)&context->counter_sets));
  }
  const iree_host_size_t counter_set_index = context->counter_set_count;
  IREE_RETURN_IF_ERROR(iree_profile_index_insert(&context->counter_set_index,
                                                 context->host_allocator, hash,
                                                 counter_set_index));
  context->counter_sets[counter_set_index] = *counter_set;
  context->counter_sets[counter_set_index].first_counter_index =
      IREE_HOST_SIZE_MAX;
  context->counter_sets[counter_set_index].last_counter_index =
      IREE_HOST_SIZE_MAX;
  context->counter_sets[counter_set_index].counter_count = 0;
  ++context->counter_set_count;
  return iree_ok_status();
}

static iree_status_t iree_profile_counter_append_counter(
    iree_profile_counter_context_t* context,
    const iree_profile_counter_t* counter) {
  const iree_profile_counter_lookup_t lookup = {
      .context = context,
      .counter_set_id = counter->record.counter_set_id,
      .counter_ordinal = counter->record.counter_ordinal,
  };
  const uint64_t hash = iree_profile_counter_hash(
      counter->record.counter_set_id, counter->record.counter_ordinal);
  iree_host_size_t existing_index = 0;
  if (iree_profile_index_find(&context->counter_index, hash,
                              iree_profile_counter_matches, &lookup,
                              &existing_index)) {
    return iree_ok_status();
  }

  iree_profile_counter_set_t* counter_set = NULL;
  iree_host_size_t counter_set_index = 0;
  const iree_profile_counter_set_lookup_t counter_set_lookup = {
      .context = context,
      .counter_set_id = counter->record.counter_set_id,
  };
  if (!iree_profile_index_find(
          &context->counter_set_index,
          iree_profile_counter_set_hash(counter->record.counter_set_id),
          iree_profile_counter_set_matches, &counter_set_lookup,
          &counter_set_index)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "counter references missing counter-set metadata "
                            "counter_set=%" PRIu64 " counter_ordinal=%u",
                            counter->record.counter_set_id,
                            counter->record.counter_ordinal);
  }
  counter_set = &context->counter_sets[counter_set_index];

  if (context->counter_count + 1 > context->counter_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        context->host_allocator,
        iree_max((iree_host_size_t)8, context->counter_count + 1),
        sizeof(context->counters[0]), &context->counter_capacity,
        (void**)&context->counters));
  }

  const iree_host_size_t counter_index = context->counter_count;
  IREE_RETURN_IF_ERROR(iree_profile_index_insert(
      &context->counter_index, context->host_allocator, hash, counter_index));
  context->counters[counter_index] = *counter;
  context->counters[counter_index].next_counter_index = IREE_HOST_SIZE_MAX;
  if (counter_set->first_counter_index == IREE_HOST_SIZE_MAX) {
    counter_set->first_counter_index = counter_index;
  } else {
    context->counters[counter_set->last_counter_index].next_counter_index =
        counter_index;
  }
  counter_set->last_counter_index = counter_index;
  ++counter_set->counter_count;
  ++context->counter_count;
  return iree_ok_status();
}

static const iree_profile_counter_set_t* iree_profile_counter_find_counter_set(
    const iree_profile_counter_context_t* context, uint64_t counter_set_id) {
  const iree_profile_counter_set_lookup_t lookup = {
      .context = context,
      .counter_set_id = counter_set_id,
  };
  iree_host_size_t index = 0;
  if (iree_profile_index_find(&context->counter_set_index,
                              iree_profile_counter_set_hash(counter_set_id),
                              iree_profile_counter_set_matches, &lookup,
                              &index)) {
    return &context->counter_sets[index];
  }
  return NULL;
}

static const iree_profile_counter_t* iree_profile_counter_find_counter(
    const iree_profile_counter_context_t* context, uint64_t counter_set_id,
    uint32_t counter_ordinal) {
  const iree_profile_counter_lookup_t lookup = {
      .context = context,
      .counter_set_id = counter_set_id,
      .counter_ordinal = counter_ordinal,
  };
  iree_host_size_t index = 0;
  if (iree_profile_index_find(
          &context->counter_index,
          iree_profile_counter_hash(counter_set_id, counter_ordinal),
          iree_profile_counter_matches, &lookup, &index)) {
    return &context->counters[index];
  }
  return NULL;
}

static iree_status_t iree_profile_counter_get_aggregate(
    iree_profile_counter_context_t* context, uint32_t physical_device_ordinal,
    uint32_t queue_ordinal, uint64_t stream_id,
    iree_hal_profile_counter_sample_scope_t scope, uint64_t counter_set_id,
    uint32_t counter_ordinal, uint64_t executable_id, uint32_t export_ordinal,
    uint64_t command_buffer_id,
    iree_profile_counter_aggregate_t** out_aggregate) {
  *out_aggregate = NULL;

  const iree_profile_counter_aggregate_lookup_t lookup = {
      .context = context,
      .physical_device_ordinal = physical_device_ordinal,
      .queue_ordinal = queue_ordinal,
      .stream_id = stream_id,
      .scope = scope,
      .counter_set_id = counter_set_id,
      .counter_ordinal = counter_ordinal,
      .executable_id = executable_id,
      .export_ordinal = export_ordinal,
      .command_buffer_id = command_buffer_id,
  };
  const uint64_t hash = iree_profile_counter_aggregate_hash(
      physical_device_ordinal, queue_ordinal, stream_id, scope, counter_set_id,
      counter_ordinal, executable_id, export_ordinal, command_buffer_id);
  iree_host_size_t existing_index = 0;
  if (iree_profile_index_find(&context->aggregate_index, hash,
                              iree_profile_counter_aggregate_matches, &lookup,
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
  iree_profile_counter_aggregate_t* aggregate =
      &context->aggregates[aggregate_index];
  memset(aggregate, 0, sizeof(*aggregate));
  aggregate->physical_device_ordinal = physical_device_ordinal;
  aggregate->queue_ordinal = queue_ordinal;
  aggregate->stream_id = stream_id;
  aggregate->scope = scope;
  aggregate->counter_set_id = counter_set_id;
  aggregate->counter_ordinal = counter_ordinal;
  aggregate->executable_id = executable_id;
  aggregate->export_ordinal = export_ordinal;
  aggregate->command_buffer_id = command_buffer_id;
  aggregate->minimum_value = DBL_MAX;
  ++context->aggregate_count;
  *out_aggregate = aggregate;
  return iree_ok_status();
}

static iree_status_t iree_profile_counter_decode_counter_set_record(
    const iree_profile_typed_record_t* typed_record,
    iree_profile_counter_set_t* out_counter_set) {
  iree_hal_profile_counter_set_record_t record_value;
  memcpy(&record_value, typed_record->contents.data, sizeof(record_value));
  if ((iree_host_size_t)record_value.name_length !=
      typed_record->inline_payload.data_length) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "counter set name length is inconsistent with record length");
  }

  *out_counter_set = (iree_profile_counter_set_t){
      .record = record_value,
      .name =
          iree_make_string_view((const char*)typed_record->inline_payload.data,
                                typed_record->inline_payload.data_length),
  };
  return iree_ok_status();
}

static iree_status_t iree_profile_counter_decode_counter_record(
    const iree_profile_counter_context_t* context,
    const iree_profile_typed_record_t* typed_record,
    iree_profile_counter_t* out_counter) {
  iree_hal_profile_counter_record_t record_value;
  memcpy(&record_value, typed_record->contents.data, sizeof(record_value));
  iree_host_size_t trailing_length = 0;
  if (!iree_host_size_checked_add(record_value.block_name_length,
                                  record_value.name_length, &trailing_length) ||
      !iree_host_size_checked_add(
          trailing_length, record_value.description_length, &trailing_length) ||
      trailing_length != typed_record->inline_payload.data_length) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "counter string lengths are inconsistent with record length");
  }
  if (!iree_profile_counter_find_counter_set(context,
                                             record_value.counter_set_id)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "counter references missing counter-set metadata "
                            "counter_set=%" PRIu64 " counter_ordinal=%u",
                            record_value.counter_set_id,
                            record_value.counter_ordinal);
  }

  const char* string_base = (const char*)typed_record->inline_payload.data;
  *out_counter = (iree_profile_counter_t){
      .record = record_value,
      .block_name =
          iree_make_string_view(string_base, record_value.block_name_length),
      .name =
          iree_make_string_view(string_base + record_value.block_name_length,
                                record_value.name_length),
      .description =
          iree_make_string_view(string_base + record_value.block_name_length +
                                    record_value.name_length,
                                record_value.description_length),
  };
  return iree_ok_status();
}

static iree_status_t iree_profile_counter_process_counter_set_records(
    iree_profile_counter_context_t* context,
    const iree_hal_profile_file_record_t* record) {
  iree_profile_typed_record_iterator_t iterator;
  iree_profile_typed_record_iterator_initialize(
      record, sizeof(iree_hal_profile_counter_set_record_t), &iterator);
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status)) {
    iree_profile_typed_record_t typed_record;
    bool has_record = false;
    status = iree_profile_typed_record_iterator_next(&iterator, &typed_record,
                                                     &has_record);
    if (!iree_status_is_ok(status) || !has_record) break;

    iree_profile_counter_set_t counter_set;
    status = iree_profile_counter_decode_counter_set_record(&typed_record,
                                                            &counter_set);
    if (iree_status_is_ok(status)) {
      status = iree_profile_counter_append_counter_set(context, &counter_set);
    }
  }
  return status;
}

static iree_status_t iree_profile_counter_process_counter_records(
    iree_profile_counter_context_t* context,
    const iree_hal_profile_file_record_t* record) {
  iree_profile_typed_record_iterator_t iterator;
  iree_profile_typed_record_iterator_initialize(
      record, sizeof(iree_hal_profile_counter_record_t), &iterator);
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status)) {
    iree_profile_typed_record_t typed_record;
    bool has_record = false;
    status = iree_profile_typed_record_iterator_next(&iterator, &typed_record,
                                                     &has_record);
    if (!iree_status_is_ok(status) || !has_record) break;

    iree_profile_counter_t counter;
    status = iree_profile_counter_decode_counter_record(context, &typed_record,
                                                        &counter);
    if (iree_status_is_ok(status)) {
      status = iree_profile_counter_append_counter(context, &counter);
    }
  }
  return status;
}

iree_status_t iree_profile_counter_process_metadata_record(
    iree_profile_counter_context_t* context,
    const iree_hal_profile_file_record_t* record) {
  if (record->header.record_type != IREE_HAL_PROFILE_FILE_RECORD_TYPE_CHUNK) {
    return iree_ok_status();
  }
  if (iree_string_view_equal(record->content_type,
                             IREE_HAL_PROFILE_CONTENT_TYPE_COUNTER_SETS)) {
    return iree_profile_counter_process_counter_set_records(context, record);
  } else if (iree_string_view_equal(record->content_type,
                                    IREE_HAL_PROFILE_CONTENT_TYPE_COUNTERS)) {
    return iree_profile_counter_process_counter_records(context, record);
  }
  return iree_ok_status();
}

static const char* iree_profile_counter_unit_name(
    iree_hal_profile_counter_unit_t unit) {
  switch (unit) {
    case IREE_HAL_PROFILE_COUNTER_UNIT_NONE:
      return "none";
    case IREE_HAL_PROFILE_COUNTER_UNIT_COUNT:
      return "count";
    case IREE_HAL_PROFILE_COUNTER_UNIT_CYCLES:
      return "cycles";
    case IREE_HAL_PROFILE_COUNTER_UNIT_BYTES:
      return "bytes";
    default:
      return "unknown";
  }
}

static const char* iree_profile_counter_sample_scope_name(
    iree_hal_profile_counter_sample_scope_t scope) {
  switch (scope) {
    case IREE_HAL_PROFILE_COUNTER_SAMPLE_SCOPE_NONE:
      return "none";
    case IREE_HAL_PROFILE_COUNTER_SAMPLE_SCOPE_DISPATCH:
      return "dispatch";
    case IREE_HAL_PROFILE_COUNTER_SAMPLE_SCOPE_COMMAND_OPERATION:
      return "command_operation";
    case IREE_HAL_PROFILE_COUNTER_SAMPLE_SCOPE_DEVICE_TIME_RANGE:
      return "device_time_range";
    default:
      return "unknown";
  }
}

static bool iree_profile_counter_sample_matches_id(
    const iree_hal_profile_counter_sample_record_t* sample, int64_t id_filter) {
  if (id_filter < 0) return true;
  const uint64_t id = (uint64_t)id_filter;
  return sample->sample_id == id || sample->dispatch_event_id == id ||
         sample->submission_id == id || sample->command_buffer_id == id;
}

static bool iree_profile_counter_filter_matches(
    const iree_profile_counter_set_t* counter_set,
    const iree_profile_counter_t* counter, iree_string_view_t key,
    iree_string_view_t filter) {
  return iree_profile_key_matches(key, filter) ||
         iree_profile_key_matches(counter_set->name, filter) ||
         iree_profile_key_matches(counter->block_name, filter) ||
         iree_profile_key_matches(counter->name, filter);
}

static iree_status_t iree_profile_counter_resolve_sample_key(
    const iree_profile_model_t* model,
    const iree_hal_profile_counter_sample_record_t* sample,
    char* numeric_buffer, iree_host_size_t numeric_buffer_capacity,
    iree_string_view_t* out_key) {
  *out_key = IREE_SV("unattributed");
  if (sample->executable_id == 0 || sample->export_ordinal == UINT32_MAX) {
    return iree_ok_status();
  }
  return iree_profile_model_resolve_dispatch_key(
      model, sample->physical_device_ordinal, sample->executable_id,
      sample->export_ordinal, numeric_buffer, numeric_buffer_capacity, out_key);
}

static iree_status_t iree_profile_counter_sum_value(
    const iree_profile_counter_t* counter,
    const iree_hal_profile_counter_sample_record_t* sample,
    const uint8_t* sample_values, double* out_value_sum) {
  *out_value_sum = 0.0;
  if (counter->record.sample_value_offset > sample->sample_value_count ||
      sample->sample_value_count - counter->record.sample_value_offset <
          counter->record.sample_value_count) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "counter sample value layout is inconsistent with counter metadata "
        "counter_set=%" PRIu64 " counter_ordinal=%u",
        counter->record.counter_set_id, counter->record.counter_ordinal);
  }

  double value_sum = 0.0;
  for (uint32_t i = 0; i < counter->record.sample_value_count; ++i) {
    uint64_t raw_value = 0;
    const iree_host_size_t value_offset =
        ((iree_host_size_t)counter->record.sample_value_offset + i) *
        sizeof(raw_value);
    memcpy(&raw_value, sample_values + value_offset, sizeof(raw_value));
    value_sum += (double)raw_value;
  }
  *out_value_sum = value_sum;
  return iree_ok_status();
}

static void iree_profile_counter_print_sample_values_jsonl(
    const iree_profile_counter_t* counter, const uint8_t* sample_values,
    FILE* file) {
  fputc('[', file);
  for (uint32_t i = 0; i < counter->record.sample_value_count; ++i) {
    uint64_t raw_value = 0;
    const iree_host_size_t value_offset =
        ((iree_host_size_t)counter->record.sample_value_offset + i) *
        sizeof(raw_value);
    memcpy(&raw_value, sample_values + value_offset, sizeof(raw_value));
    if (i != 0) fputc(',', file);
    fprintf(file, "%" PRIu64, raw_value);
  }
  fputc(']', file);
}

static void iree_profile_counter_record_aggregate_sample(
    iree_profile_counter_aggregate_t* aggregate,
    const iree_profile_counter_t* counter,
    const iree_hal_profile_counter_sample_record_t* sample, double value_sum) {
  ++aggregate->sample_count;
  aggregate->raw_value_count += counter->record.sample_value_count;
  if (aggregate->first_sample_id == 0) {
    aggregate->first_sample_id = sample->sample_id;
  }
  aggregate->last_sample_id = sample->sample_id;
  aggregate->minimum_value = iree_min(aggregate->minimum_value, value_sum);
  aggregate->maximum_value = iree_max(aggregate->maximum_value, value_sum);
  aggregate->total_value += value_sum;

  const double delta = value_sum - aggregate->mean_value;
  aggregate->mean_value += delta / (double)aggregate->sample_count;
  const double delta2 = value_sum - aggregate->mean_value;
  aggregate->m2_value += delta * delta2;
}

static void iree_profile_counter_print_sample_jsonl(
    const iree_hal_profile_counter_sample_record_t* sample,
    const iree_profile_counter_set_t* counter_set,
    const iree_profile_counter_t* counter, iree_string_view_t key,
    const uint8_t* sample_values, double value_sum, FILE* file) {
  const bool has_device_tick_range = iree_all_bits_set(
      sample->flags, IREE_HAL_PROFILE_COUNTER_SAMPLE_FLAG_DEVICE_TICK_RANGE);
  const bool valid_device_tick_range =
      has_device_tick_range && sample->start_tick != 0 &&
      sample->end_tick != 0 && sample->end_tick >= sample->start_tick;
  const uint64_t duration_ticks =
      valid_device_tick_range ? sample->end_tick - sample->start_tick : 0;
  fprintf(file,
          "{\"type\":\"counter_sample\",\"sample_id\":%" PRIu64
          ",\"counter_set_id\":%" PRIu64 ",\"counter_set\":",
          sample->sample_id, sample->counter_set_id);
  iree_profile_fprint_json_string(file, counter_set->name);
  fprintf(file, ",\"counter_ordinal\":%u,\"counter\":",
          counter->record.counter_ordinal);
  iree_profile_fprint_json_string(file, counter->name);
  fprintf(file, ",\"block\":");
  iree_profile_fprint_json_string(file, counter->block_name);
  fprintf(file, ",\"unit\":");
  iree_profile_fprint_json_string(
      file, iree_make_cstring_view(
                iree_profile_counter_unit_name(counter->record.unit)));
  fprintf(file, ",\"scope\":");
  iree_profile_fprint_json_string(
      file, iree_make_cstring_view(
                iree_profile_counter_sample_scope_name(sample->scope)));
  fprintf(file,
          ",\"scope_value\":%u"
          ",\"dispatch_event_id\":%" PRIu64 ",\"submission_id\":%" PRIu64
          ",\"command_buffer_id\":%" PRIu64
          ",\"command_index\":%u,\"executable_id\":%" PRIu64
          ",\"export_ordinal\":%u,\"key\":",
          sample->scope, sample->dispatch_event_id, sample->submission_id,
          sample->command_buffer_id, sample->command_index,
          sample->executable_id, sample->export_ordinal);
  iree_profile_fprint_json_string(file, key);
  fprintf(file,
          ",\"physical_device_ordinal\":%u,\"queue_ordinal\":%u"
          ",\"stream_id\":%" PRIu64
          ",\"flags\":%u"
          ",\"device_tick_range_present\":%s"
          ",\"device_tick_domain\":\"device_tick\""
          ",\"start_tick\":%" PRIu64 ",\"end_tick\":%" PRIu64
          ",\"duration_ticks\":%" PRIu64
          ",\"device_tick_range_valid\":%s"
          ",\"value\":%.3f"
          ",\"values\":",
          sample->physical_device_ordinal, sample->queue_ordinal,
          sample->stream_id, sample->flags,
          has_device_tick_range ? "true" : "false", sample->start_tick,
          sample->end_tick, duration_ticks,
          valid_device_tick_range ? "true" : "false", value_sum);
  iree_profile_counter_print_sample_values_jsonl(counter, sample_values, file);
  fputs("}\n", file);
}

static iree_status_t iree_profile_counter_emit_sample_jsonl(
    void* user_data, const iree_profile_counter_sample_row_t* row) {
  FILE* file = (FILE*)user_data;
  iree_profile_counter_print_sample_jsonl(
      row->sample, row->counter_set, row->counter, row->key,
      row->sample_values.data, row->value_sum, file);
  return iree_ok_status();
}

typedef struct iree_profile_counter_sample_state_t {
  // Decoded counter sample record.
  iree_hal_profile_counter_sample_record_t sample;
  // Counter set metadata referenced by |sample|.
  const iree_profile_counter_set_t* counter_set;
  // Resolved executable export key for filtering and reporting.
  iree_string_view_t key;
  // Raw uint64_t counter sample values trailing |sample|.
  iree_const_byte_span_t sample_values;
  // True when the source chunk was marked truncated.
  bool is_truncated;
} iree_profile_counter_sample_state_t;

static iree_status_t iree_profile_counter_sample_state_initialize(
    const iree_profile_typed_record_t* typed_record, bool is_truncated,
    iree_profile_counter_sample_state_t* out_state) {
  memset(out_state, 0, sizeof(*out_state));
  memcpy(&out_state->sample, typed_record->contents.data,
         sizeof(out_state->sample));
  out_state->sample_values =
      iree_make_const_byte_span(typed_record->inline_payload.data,
                                typed_record->inline_payload.data_length);
  out_state->is_truncated = is_truncated;

  iree_host_size_t values_length = 0;
  if (!iree_host_size_checked_mul(out_state->sample.sample_value_count,
                                  sizeof(uint64_t), &values_length) ||
      values_length != typed_record->inline_payload.data_length) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "counter sample value count is inconsistent with record length");
  }
  return iree_ok_status();
}

static iree_status_t iree_profile_counter_sample_state_attach_metadata(
    const iree_profile_counter_context_t* counter_context,
    iree_profile_counter_sample_state_t* state) {
  state->counter_set = iree_profile_counter_find_counter_set(
      counter_context, state->sample.counter_set_id);
  if (!state->counter_set) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "counter sample references missing counter-set metadata "
        "sample=%" PRIu64 " counter_set=%" PRIu64,
        state->sample.sample_id, state->sample.counter_set_id);
  }
  if (state->counter_set->record.sample_value_count !=
      state->sample.sample_value_count) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "counter sample value count does not match counter-set metadata "
        "sample=%" PRIu64 " counter_set=%" PRIu64,
        state->sample.sample_id, state->sample.counter_set_id);
  }
  return iree_ok_status();
}

static iree_status_t iree_profile_counter_sample_state_resolve_key(
    const iree_profile_model_t* model,
    iree_profile_counter_sample_state_t* state, char* numeric_buffer,
    iree_host_size_t numeric_buffer_capacity) {
  return iree_profile_counter_resolve_sample_key(
      model, &state->sample, numeric_buffer, numeric_buffer_capacity,
      &state->key);
}

static iree_status_t iree_profile_counter_process_sample_counter(
    iree_profile_counter_context_t* counter_context,
    const iree_profile_counter_sample_state_t* state,
    const iree_profile_counter_t* counter, iree_string_view_t filter,
    iree_profile_counter_sample_callback_t sample_callback,
    bool* out_matched_counter) {
  *out_matched_counter = false;
  if (!iree_profile_counter_filter_matches(state->counter_set, counter,
                                           state->key, filter)) {
    return iree_ok_status();
  }

  double value_sum = 0.0;
  IREE_RETURN_IF_ERROR(iree_profile_counter_sum_value(
      counter, &state->sample, state->sample_values.data, &value_sum));

  iree_profile_counter_aggregate_t* aggregate = NULL;
  IREE_RETURN_IF_ERROR(iree_profile_counter_get_aggregate(
      counter_context, state->sample.physical_device_ordinal,
      state->sample.queue_ordinal, state->sample.stream_id, state->sample.scope,
      state->sample.counter_set_id, counter->record.counter_ordinal,
      state->sample.executable_id, state->sample.export_ordinal,
      state->sample.command_buffer_id, &aggregate));
  iree_profile_counter_record_aggregate_sample(aggregate, counter,
                                               &state->sample, value_sum);

  if (sample_callback.fn) {
    const iree_profile_counter_sample_row_t sample_row = {
        .sample = &state->sample,
        .counter_set = state->counter_set,
        .counter = counter,
        .key = state->key,
        .sample_values = state->sample_values,
        .value_sum = value_sum,
        .is_truncated = state->is_truncated,
    };
    IREE_RETURN_IF_ERROR(
        sample_callback.fn(sample_callback.user_data, &sample_row));
  }
  *out_matched_counter = true;
  return iree_ok_status();
}

static iree_status_t iree_profile_counter_process_matching_sample(
    iree_profile_counter_context_t* counter_context,
    const iree_profile_counter_sample_state_t* state, iree_string_view_t filter,
    int64_t id_filter, iree_profile_counter_sample_callback_t sample_callback,
    bool* out_matched_sample) {
  *out_matched_sample = false;
  if (!iree_profile_counter_sample_matches_id(&state->sample, id_filter)) {
    return iree_ok_status();
  }

  bool found_counter = false;
  iree_status_t status = iree_ok_status();
  iree_host_size_t counter_index = state->counter_set->first_counter_index;
  while (counter_index != IREE_HOST_SIZE_MAX && iree_status_is_ok(status)) {
    const iree_profile_counter_t* counter =
        &counter_context->counters[counter_index];
    found_counter = true;

    bool matched_counter = false;
    status = iree_profile_counter_process_sample_counter(
        counter_context, state, counter, filter, sample_callback,
        &matched_counter);
    if (iree_status_is_ok(status) && matched_counter) {
      *out_matched_sample = true;
    }
    counter_index = counter->next_counter_index;
  }
  if (iree_status_is_ok(status) && !found_counter) {
    status = iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "counter sample references counter set with no counter metadata "
        "sample=%" PRIu64 " counter_set=%" PRIu64,
        state->sample.sample_id, state->sample.counter_set_id);
  }
  return status;
}

static iree_status_t iree_profile_counter_process_sample_record(
    iree_profile_counter_context_t* counter_context,
    const iree_profile_model_t* model,
    const iree_profile_typed_record_t* typed_record, bool is_truncated,
    iree_string_view_t filter, int64_t id_filter,
    iree_profile_counter_sample_callback_t sample_callback,
    bool* out_matched_sample) {
  *out_matched_sample = false;
  iree_profile_counter_sample_state_t sample_state;
  IREE_RETURN_IF_ERROR(iree_profile_counter_sample_state_initialize(
      typed_record, is_truncated, &sample_state));
  IREE_RETURN_IF_ERROR(iree_profile_counter_sample_state_attach_metadata(
      counter_context, &sample_state));
  char numeric_buffer[128];
  IREE_RETURN_IF_ERROR(iree_profile_counter_sample_state_resolve_key(
      model, &sample_state, numeric_buffer, sizeof(numeric_buffer)));
  return iree_profile_counter_process_matching_sample(
      counter_context, &sample_state, filter, id_filter, sample_callback,
      out_matched_sample);
}

iree_status_t iree_profile_counter_process_sample_records(
    iree_profile_counter_context_t* counter_context,
    const iree_profile_model_t* model,
    const iree_hal_profile_file_record_t* record, iree_string_view_t filter,
    int64_t id_filter, iree_profile_counter_sample_callback_t sample_callback) {
  if (record->header.record_type != IREE_HAL_PROFILE_FILE_RECORD_TYPE_CHUNK) {
    return iree_ok_status();
  }
  if (!iree_string_view_equal(record->content_type,
                              IREE_HAL_PROFILE_CONTENT_TYPE_COUNTER_SAMPLES)) {
    return iree_ok_status();
  }

  const bool is_truncated = iree_any_bit_set(
      record->header.chunk_flags, IREE_HAL_PROFILE_CHUNK_FLAG_TRUNCATED);
  iree_profile_typed_record_iterator_t iterator;
  iree_profile_typed_record_iterator_initialize(
      record, sizeof(iree_hal_profile_counter_sample_record_t), &iterator);
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status)) {
    iree_profile_typed_record_t typed_record;
    bool has_record = false;
    status = iree_profile_typed_record_iterator_next(&iterator, &typed_record,
                                                     &has_record);
    if (!iree_status_is_ok(status) || !has_record) break;

    ++counter_context->total_sample_count;

    bool matched_sample = false;
    status = iree_profile_counter_process_sample_record(
        counter_context, model, &typed_record, is_truncated, filter, id_filter,
        sample_callback, &matched_sample);
    if (iree_status_is_ok(status) && matched_sample) {
      ++counter_context->matched_sample_count;
      if (is_truncated) ++counter_context->truncated_sample_count;
    }
  }
  return status;
}

static iree_status_t iree_profile_counter_resolve_aggregate_key(
    const iree_profile_model_t* model,
    const iree_profile_counter_aggregate_t* aggregate, char* numeric_buffer,
    iree_host_size_t numeric_buffer_capacity, iree_string_view_t* out_key) {
  *out_key = IREE_SV("unattributed");
  if (aggregate->executable_id == 0 ||
      aggregate->export_ordinal == UINT32_MAX) {
    return iree_ok_status();
  }
  return iree_profile_model_resolve_dispatch_key(
      model, aggregate->physical_device_ordinal, aggregate->executable_id,
      aggregate->export_ordinal, numeric_buffer, numeric_buffer_capacity,
      out_key);
}

typedef struct iree_profile_counter_aggregate_row_t {
  // Counter aggregate being rendered.
  const iree_profile_counter_aggregate_t* aggregate;
  // Counter set metadata referenced by |aggregate|.
  const iree_profile_counter_set_t* counter_set;
  // Counter metadata referenced by |aggregate|.
  const iree_profile_counter_t* counter;
  // Resolved executable export key for the aggregate.
  iree_string_view_t key;
  // Sample standard deviation for the aggregate's summed counter value.
  double stddev;
} iree_profile_counter_aggregate_row_t;

static double iree_profile_counter_aggregate_stddev(
    const iree_profile_counter_aggregate_t* aggregate) {
  const double variance =
      aggregate->sample_count > 1
          ? aggregate->m2_value / (double)(aggregate->sample_count - 1)
          : 0.0;
  return iree_profile_sqrt_f64(variance);
}

static iree_status_t iree_profile_counter_resolve_aggregate_row(
    const iree_profile_counter_context_t* counter_context,
    const iree_profile_model_t* model,
    const iree_profile_counter_aggregate_t* aggregate, char* numeric_buffer,
    iree_host_size_t numeric_buffer_capacity,
    iree_profile_counter_aggregate_row_t* out_row) {
  memset(out_row, 0, sizeof(*out_row));
  out_row->aggregate = aggregate;
  out_row->counter_set = iree_profile_counter_find_counter_set(
      counter_context, aggregate->counter_set_id);
  out_row->counter = iree_profile_counter_find_counter(
      counter_context, aggregate->counter_set_id, aggregate->counter_ordinal);
  if (!out_row->counter_set || !out_row->counter) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "counter aggregate references missing metadata counter_set=%" PRIu64
        " counter_ordinal=%u",
        aggregate->counter_set_id, aggregate->counter_ordinal);
  }

  IREE_RETURN_IF_ERROR(iree_profile_counter_resolve_aggregate_key(
      model, aggregate, numeric_buffer, numeric_buffer_capacity,
      &out_row->key));
  out_row->stddev = iree_profile_counter_aggregate_stddev(aggregate);
  return iree_ok_status();
}

static iree_status_t iree_profile_counter_print_metadata_text(
    const iree_profile_counter_context_t* counter_context, FILE* file) {
  fprintf(file, "counter_sets:\n");
  for (iree_host_size_t i = 0; i < counter_context->counter_set_count; ++i) {
    const iree_profile_counter_set_t* counter_set =
        &counter_context->counter_sets[i];
    fprintf(file,
            "  counter_set %" PRIu64
            ": device=%u counters=%u sample_values=%u flags=%u name=%.*s\n",
            counter_set->record.counter_set_id,
            counter_set->record.physical_device_ordinal,
            counter_set->record.counter_count,
            counter_set->record.sample_value_count, counter_set->record.flags,
            (int)counter_set->name.size, counter_set->name.data);
  }

  fprintf(file, "counters:\n");
  for (iree_host_size_t i = 0; i < counter_context->counter_count; ++i) {
    const iree_profile_counter_t* counter = &counter_context->counters[i];
    if (!iree_profile_counter_find_counter_set(
            counter_context, counter->record.counter_set_id)) {
      return iree_make_status(IREE_STATUS_DATA_LOSS,
                              "counter references missing counter-set metadata "
                              "counter_set=%" PRIu64 " counter_ordinal=%u",
                              counter->record.counter_set_id,
                              counter->record.counter_ordinal);
    }
    fprintf(file,
            "  counter %" PRIu64
            "#%u: device=%u block=%.*s name=%.*s "
            "unit=%s values=[%u,%u) flags=%u description=%.*s\n",
            counter->record.counter_set_id, counter->record.counter_ordinal,
            counter->record.physical_device_ordinal,
            (int)counter->block_name.size, counter->block_name.data,
            (int)counter->name.size, counter->name.data,
            iree_profile_counter_unit_name(counter->record.unit),
            counter->record.sample_value_offset,
            counter->record.sample_value_offset +
                counter->record.sample_value_count,
            counter->record.flags, (int)counter->description.size,
            counter->description.data);
  }
  return iree_ok_status();
}

static void iree_profile_counter_print_metadata_jsonl(
    const iree_profile_counter_context_t* counter_context, FILE* file) {
  for (iree_host_size_t i = 0; i < counter_context->counter_set_count; ++i) {
    const iree_profile_counter_set_t* counter_set =
        &counter_context->counter_sets[i];
    fprintf(file,
            "{\"type\":\"counter_set\",\"counter_set_id\":%" PRIu64
            ",\"physical_device_ordinal\":%u,\"flags\":%u"
            ",\"counter_count\":%u,\"sample_value_count\":%u,\"name\":",
            counter_set->record.counter_set_id,
            counter_set->record.physical_device_ordinal,
            counter_set->record.flags, counter_set->record.counter_count,
            counter_set->record.sample_value_count);
    iree_profile_fprint_json_string(file, counter_set->name);
    fputs("}\n", file);
  }

  for (iree_host_size_t i = 0; i < counter_context->counter_count; ++i) {
    const iree_profile_counter_t* counter = &counter_context->counters[i];
    fprintf(file,
            "{\"type\":\"counter\",\"counter_set_id\":%" PRIu64
            ",\"counter_ordinal\":%u,\"physical_device_ordinal\":%u"
            ",\"flags\":%u,\"unit\":",
            counter->record.counter_set_id, counter->record.counter_ordinal,
            counter->record.physical_device_ordinal, counter->record.flags);
    iree_profile_fprint_json_string(
        file, iree_make_cstring_view(
                  iree_profile_counter_unit_name(counter->record.unit)));
    fprintf(file, ",\"unit_value\":%u,\"sample_value_offset\":%u",
            counter->record.unit, counter->record.sample_value_offset);
    fprintf(file, ",\"sample_value_count\":%u,\"block\":",
            counter->record.sample_value_count);
    iree_profile_fprint_json_string(file, counter->block_name);
    fprintf(file, ",\"name\":");
    iree_profile_fprint_json_string(file, counter->name);
    fprintf(file, ",\"description\":");
    iree_profile_fprint_json_string(file, counter->description);
    fputs("}\n", file);
  }
}

static void iree_profile_counter_print_text_summary(
    const iree_profile_counter_context_t* counter_context,
    iree_string_view_t filter, int64_t id_filter, FILE* file) {
  fprintf(file, "IREE HAL profile counter summary\n");
  fprintf(file, "filter: %.*s\n", (int)filter.size, filter.data);
  if (id_filter >= 0) {
    fprintf(file, "id_filter: %" PRId64 "\n", id_filter);
  }
  fprintf(file,
          "samples: total=%" PRIu64 " matched=%" PRIu64
          " truncated_matched=%" PRIu64 " counter_sets=%" PRIhsz
          " counters=%" PRIhsz " groups=%" PRIhsz "\n",
          counter_context->total_sample_count,
          counter_context->matched_sample_count,
          counter_context->truncated_sample_count,
          counter_context->counter_set_count, counter_context->counter_count,
          counter_context->aggregate_count);
}

static void iree_profile_counter_print_text_group(
    const iree_profile_counter_aggregate_row_t* row, FILE* file) {
  const iree_profile_counter_aggregate_t* aggregate = row->aggregate;
  const iree_profile_counter_set_t* counter_set = row->counter_set;
  const iree_profile_counter_t* counter = row->counter;
  fprintf(file,
          "  %.*s / %.*s.%.*s\n"
          "    scope=%s device=%u queue=%u stream=%" PRIu64
          " command_buffer=%" PRIu64 " executable=%" PRIu64
          " export=%u key=%.*s\n"
          "    samples=%" PRIu64 " raw_values=%" PRIu64
          " value[min/avg/stddev/max/total]=%.3f/%.3f/%.3f/%.3f/%.3f "
          "unit=%s first_sample=%" PRIu64 " last_sample=%" PRIu64 "\n",
          (int)counter_set->name.size, counter_set->name.data,
          (int)counter->block_name.size, counter->block_name.data,
          (int)counter->name.size, counter->name.data,
          iree_profile_counter_sample_scope_name(aggregate->scope),
          aggregate->physical_device_ordinal, aggregate->queue_ordinal,
          aggregate->stream_id, aggregate->command_buffer_id,
          aggregate->executable_id, aggregate->export_ordinal,
          (int)row->key.size, row->key.data, aggregate->sample_count,
          aggregate->raw_value_count,
          aggregate->sample_count ? aggregate->minimum_value : 0.0,
          aggregate->sample_count ? aggregate->mean_value : 0.0, row->stddev,
          aggregate->sample_count ? aggregate->maximum_value : 0.0,
          aggregate->total_value,
          iree_profile_counter_unit_name(counter->record.unit),
          aggregate->first_sample_id, aggregate->last_sample_id);
}

static iree_status_t iree_profile_counter_print_text_groups(
    const iree_profile_counter_context_t* counter_context,
    const iree_profile_model_t* model, FILE* file) {
  fprintf(file, "groups:\n");
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < counter_context->aggregate_count && iree_status_is_ok(status); ++i) {
    const iree_profile_counter_aggregate_t* aggregate =
        &counter_context->aggregates[i];
    char numeric_buffer[128];
    iree_profile_counter_aggregate_row_t row;
    status = iree_profile_counter_resolve_aggregate_row(
        counter_context, model, aggregate, numeric_buffer,
        sizeof(numeric_buffer), &row);
    if (iree_status_is_ok(status)) {
      iree_profile_counter_print_text_group(&row, file);
    }
  }
  return status;
}

static iree_status_t iree_profile_counter_print_text(
    const iree_profile_counter_context_t* counter_context,
    const iree_profile_model_t* model, iree_string_view_t filter,
    int64_t id_filter, FILE* file) {
  iree_profile_counter_print_text_summary(counter_context, filter, id_filter,
                                          file);
  IREE_RETURN_IF_ERROR(
      iree_profile_counter_print_metadata_text(counter_context, file));
  return iree_profile_counter_print_text_groups(counter_context, model, file);
}

static void iree_profile_counter_print_jsonl_summary(
    const iree_profile_counter_context_t* counter_context,
    iree_string_view_t filter, int64_t id_filter, bool emit_samples,
    FILE* file) {
  fprintf(file, "{\"type\":\"counter_summary\",\"filter\":");
  iree_profile_fprint_json_string(file, filter);
  fprintf(file,
          ",\"id_filter\":%" PRId64
          ",\"mode\":\"%s\",\"total_samples\":%" PRIu64
          ",\"matched_samples\":%" PRIu64
          ",\"truncated_matched_samples\":%" PRIu64 ",\"counter_sets\":%" PRIhsz
          ",\"counters\":%" PRIhsz ",\"aggregate_groups\":%" PRIhsz "}\n",
          id_filter, emit_samples ? "samples" : "aggregate",
          counter_context->total_sample_count,
          counter_context->matched_sample_count,
          counter_context->truncated_sample_count,
          counter_context->counter_set_count, counter_context->counter_count,
          counter_context->aggregate_count);
}

static void iree_profile_counter_print_jsonl_group(
    const iree_profile_counter_aggregate_row_t* row, FILE* file) {
  const iree_profile_counter_aggregate_t* aggregate = row->aggregate;
  const iree_profile_counter_set_t* counter_set = row->counter_set;
  const iree_profile_counter_t* counter = row->counter;
  fprintf(file,
          "{\"type\":\"counter_group\",\"physical_device_ordinal\":%u"
          ",\"queue_ordinal\":%u,\"stream_id\":%" PRIu64 ",\"scope\":",
          aggregate->physical_device_ordinal, aggregate->queue_ordinal,
          aggregate->stream_id);
  iree_profile_fprint_json_string(
      file, iree_make_cstring_view(
                iree_profile_counter_sample_scope_name(aggregate->scope)));
  fprintf(file,
          ",\"scope_value\":%u,\"counter_set_id\":%" PRIu64 ",\"counter_set\":",
          aggregate->scope, aggregate->counter_set_id);
  iree_profile_fprint_json_string(file, counter_set->name);
  fprintf(file,
          ",\"counter_ordinal\":%u,\"counter\":", aggregate->counter_ordinal);
  iree_profile_fprint_json_string(file, counter->name);
  fprintf(file, ",\"block\":");
  iree_profile_fprint_json_string(file, counter->block_name);
  fprintf(file, ",\"unit\":");
  iree_profile_fprint_json_string(
      file, iree_make_cstring_view(
                iree_profile_counter_unit_name(counter->record.unit)));
  fprintf(file,
          ",\"command_buffer_id\":%" PRIu64 ",\"executable_id\":%" PRIu64
          ",\"export_ordinal\":%u"
          ",\"key\":",
          aggregate->command_buffer_id, aggregate->executable_id,
          aggregate->export_ordinal);
  iree_profile_fprint_json_string(file, row->key);
  fprintf(file,
          ",\"samples\":%" PRIu64 ",\"raw_values\":%" PRIu64
          ",\"min\":%.3f,\"avg\":%.3f,\"stddev\":%.3f"
          ",\"max\":%.3f,\"sum\":%.3f"
          ",\"first_sample_id\":%" PRIu64 ",\"last_sample_id\":%" PRIu64 "}\n",
          aggregate->sample_count, aggregate->raw_value_count,
          aggregate->sample_count ? aggregate->minimum_value : 0.0,
          aggregate->sample_count ? aggregate->mean_value : 0.0, row->stddev,
          aggregate->sample_count ? aggregate->maximum_value : 0.0,
          aggregate->total_value, aggregate->first_sample_id,
          aggregate->last_sample_id);
}

static iree_status_t iree_profile_counter_print_jsonl_groups(
    const iree_profile_counter_context_t* counter_context,
    const iree_profile_model_t* model, FILE* file) {
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < counter_context->aggregate_count && iree_status_is_ok(status); ++i) {
    const iree_profile_counter_aggregate_t* aggregate =
        &counter_context->aggregates[i];
    char numeric_buffer[128];
    iree_profile_counter_aggregate_row_t row;
    status = iree_profile_counter_resolve_aggregate_row(
        counter_context, model, aggregate, numeric_buffer,
        sizeof(numeric_buffer), &row);
    if (iree_status_is_ok(status)) {
      iree_profile_counter_print_jsonl_group(&row, file);
    }
  }
  return status;
}

static iree_status_t iree_profile_counter_print_jsonl(
    const iree_profile_counter_context_t* counter_context,
    const iree_profile_model_t* model, iree_string_view_t filter,
    int64_t id_filter, bool emit_samples, FILE* file) {
  iree_profile_counter_print_jsonl_summary(counter_context, filter, id_filter,
                                           emit_samples, file);
  iree_profile_counter_print_metadata_jsonl(counter_context, file);
  return iree_profile_counter_print_jsonl_groups(counter_context, model, file);
}

typedef struct iree_profile_counter_parse_context_t {
  // Shared profile metadata used to resolve executable export keys.
  iree_profile_model_t* model;
  // Counter metadata and aggregate state.
  iree_profile_counter_context_t* counter_context;
  // Optional glob filter applied to projected keys.
  iree_string_view_t filter;
  // Optional entity identifier filter, or -1 when disabled.
  int64_t id_filter;
  // Optional callback receiving matched raw counter samples.
  iree_profile_counter_sample_callback_t sample_callback;
} iree_profile_counter_parse_context_t;

static iree_status_t iree_profile_counter_metadata_record(
    void* user_data, const iree_hal_profile_file_record_t* record,
    iree_host_size_t record_index) {
  (void)record_index;
  iree_profile_counter_parse_context_t* context =
      (iree_profile_counter_parse_context_t*)user_data;
  IREE_RETURN_IF_ERROR(
      iree_profile_model_process_metadata_record(context->model, record));
  return iree_profile_counter_process_metadata_record(context->counter_context,
                                                      record);
}

static iree_status_t iree_profile_counter_sample_record(
    void* user_data, const iree_hal_profile_file_record_t* record,
    iree_host_size_t record_index) {
  (void)record_index;
  iree_profile_counter_parse_context_t* context =
      (iree_profile_counter_parse_context_t*)user_data;
  return iree_profile_counter_process_sample_records(
      context->counter_context, context->model, record, context->filter,
      context->id_filter, context->sample_callback);
}

iree_status_t iree_profile_counter_file(iree_string_view_t path,
                                        iree_string_view_t format,
                                        iree_string_view_t filter,
                                        int64_t id_filter, bool emit_samples,
                                        FILE* file,
                                        iree_allocator_t host_allocator) {
  bool is_text = iree_string_view_equal(format, IREE_SV("text"));
  bool is_jsonl = iree_string_view_equal(format, IREE_SV("jsonl"));
  if (!is_text && !is_jsonl) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported profile output format '%.*s'",
                            (int)format.size, format.data);
  }
  if (emit_samples && !is_jsonl) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "--counter_samples requires --format=jsonl");
  }

  iree_profile_file_t profile_file;
  IREE_RETURN_IF_ERROR(
      iree_profile_file_open(path, host_allocator, &profile_file));

  iree_profile_model_t model;
  iree_profile_model_initialize(host_allocator, &model);
  iree_profile_counter_context_t counter_context;
  iree_profile_counter_context_initialize(host_allocator, &counter_context);
  const iree_profile_counter_sample_callback_t sample_callback = {
      .fn = emit_samples ? iree_profile_counter_emit_sample_jsonl : NULL,
      .user_data = file,
  };
  iree_profile_counter_parse_context_t parse_context = {
      .model = &model,
      .counter_context = &counter_context,
      .filter = filter,
      .id_filter = id_filter,
      .sample_callback = sample_callback,
  };
  iree_profile_file_record_callback_t record_callback = {
      .fn = iree_profile_counter_metadata_record,
      .user_data = &parse_context,
  };
  iree_status_t status =
      iree_profile_file_for_each_record(&profile_file, record_callback);
  if (iree_status_is_ok(status)) {
    record_callback.fn = iree_profile_counter_sample_record;
    status = iree_profile_file_for_each_record(&profile_file, record_callback);
  }

  if (iree_status_is_ok(status)) {
    if (is_text) {
      status = iree_profile_counter_print_text(&counter_context, &model, filter,
                                               id_filter, file);
    } else {
      status = iree_profile_counter_print_jsonl(
          &counter_context, &model, filter, id_filter, emit_samples, file);
    }
  }

  iree_profile_counter_context_deinitialize(&counter_context);
  iree_profile_model_deinitialize(&model);
  iree_profile_file_close(&profile_file);
  return status;
}
