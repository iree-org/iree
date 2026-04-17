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

static void iree_profile_counter_context_initialize(
    iree_allocator_t host_allocator,
    iree_profile_counter_context_t* out_context) {
  memset(out_context, 0, sizeof(*out_context));
  out_context->host_allocator = host_allocator;
}

static void iree_profile_counter_context_deinitialize(
    iree_profile_counter_context_t* context) {
  iree_allocator_free(context->host_allocator, context->aggregates);
  iree_allocator_free(context->host_allocator, context->counters);
  iree_allocator_free(context->host_allocator, context->counter_sets);
  memset(context, 0, sizeof(*context));
}
static iree_status_t iree_profile_counter_append_counter_set(
    iree_profile_counter_context_t* context,
    const iree_profile_counter_set_t* counter_set) {
  if (context->counter_set_count + 1 > context->counter_set_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        context->host_allocator,
        iree_max((iree_host_size_t)4, context->counter_set_count + 1),
        sizeof(context->counter_sets[0]), &context->counter_set_capacity,
        (void**)&context->counter_sets));
  }
  context->counter_sets[context->counter_set_count++] = *counter_set;
  return iree_ok_status();
}

static iree_status_t iree_profile_counter_append_counter(
    iree_profile_counter_context_t* context,
    const iree_profile_counter_t* counter) {
  if (context->counter_count + 1 > context->counter_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        context->host_allocator,
        iree_max((iree_host_size_t)8, context->counter_count + 1),
        sizeof(context->counters[0]), &context->counter_capacity,
        (void**)&context->counters));
  }
  context->counters[context->counter_count++] = *counter;
  return iree_ok_status();
}

static const iree_profile_counter_set_t* iree_profile_counter_find_counter_set(
    const iree_profile_counter_context_t* context, uint64_t counter_set_id) {
  for (iree_host_size_t i = 0; i < context->counter_set_count; ++i) {
    const iree_profile_counter_set_t* counter_set = &context->counter_sets[i];
    if (counter_set->record.counter_set_id == counter_set_id) {
      return counter_set;
    }
  }
  return NULL;
}

static const iree_profile_counter_t* iree_profile_counter_find_counter(
    const iree_profile_counter_context_t* context, uint64_t counter_set_id,
    uint32_t counter_ordinal) {
  for (iree_host_size_t i = 0; i < context->counter_count; ++i) {
    const iree_profile_counter_t* counter = &context->counters[i];
    if (counter->record.counter_set_id == counter_set_id &&
        counter->record.counter_ordinal == counter_ordinal) {
      return counter;
    }
  }
  return NULL;
}

static iree_status_t iree_profile_counter_get_aggregate(
    iree_profile_counter_context_t* context, uint32_t physical_device_ordinal,
    uint32_t queue_ordinal, uint64_t stream_id, uint64_t counter_set_id,
    uint32_t counter_ordinal, uint64_t executable_id, uint32_t export_ordinal,
    uint64_t command_buffer_id,
    iree_profile_counter_aggregate_t** out_aggregate) {
  *out_aggregate = NULL;

  for (iree_host_size_t i = 0; i < context->aggregate_count; ++i) {
    iree_profile_counter_aggregate_t* aggregate = &context->aggregates[i];
    if (aggregate->physical_device_ordinal == physical_device_ordinal &&
        aggregate->queue_ordinal == queue_ordinal &&
        aggregate->stream_id == stream_id &&
        aggregate->counter_set_id == counter_set_id &&
        aggregate->counter_ordinal == counter_ordinal &&
        aggregate->executable_id == executable_id &&
        aggregate->export_ordinal == export_ordinal &&
        aggregate->command_buffer_id == command_buffer_id) {
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

  iree_profile_counter_aggregate_t* aggregate =
      &context->aggregates[context->aggregate_count++];
  memset(aggregate, 0, sizeof(*aggregate));
  aggregate->physical_device_ordinal = physical_device_ordinal;
  aggregate->queue_ordinal = queue_ordinal;
  aggregate->stream_id = stream_id;
  aggregate->counter_set_id = counter_set_id;
  aggregate->counter_ordinal = counter_ordinal;
  aggregate->executable_id = executable_id;
  aggregate->export_ordinal = export_ordinal;
  aggregate->command_buffer_id = command_buffer_id;
  aggregate->minimum_value = DBL_MAX;
  *out_aggregate = aggregate;
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

    iree_hal_profile_counter_set_record_t record_value;
    memcpy(&record_value, typed_record.contents.data, sizeof(record_value));
    if ((iree_host_size_t)record_value.name_length !=
        typed_record.inline_payload.data_length) {
      status = iree_make_status(
          IREE_STATUS_DATA_LOSS,
          "counter set name length is inconsistent with record length");
    }
    if (iree_status_is_ok(status)) {
      iree_profile_counter_set_t counter_set = {
          .record = record_value,
          .name = iree_make_string_view(
              (const char*)typed_record.inline_payload.data,
              typed_record.inline_payload.data_length),
      };
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

    iree_hal_profile_counter_record_t record_value;
    memcpy(&record_value, typed_record.contents.data, sizeof(record_value));
    iree_host_size_t trailing_length = 0;
    if (!iree_host_size_checked_add(record_value.block_name_length,
                                    record_value.name_length,
                                    &trailing_length) ||
        !iree_host_size_checked_add(trailing_length,
                                    record_value.description_length,
                                    &trailing_length) ||
        trailing_length != typed_record.inline_payload.data_length) {
      status = iree_make_status(
          IREE_STATUS_DATA_LOSS,
          "counter string lengths are inconsistent with record length");
    }
    if (iree_status_is_ok(status) &&
        !iree_profile_counter_find_counter_set(context,
                                               record_value.counter_set_id)) {
      status = iree_make_status(
          IREE_STATUS_DATA_LOSS,
          "counter references missing counter-set metadata "
          "counter_set=%" PRIu64 " counter_ordinal=%u",
          record_value.counter_set_id, record_value.counter_ordinal);
    }
    if (iree_status_is_ok(status)) {
      const char* string_base = (const char*)typed_record.inline_payload.data;
      iree_profile_counter_t counter = {
          .record = record_value,
          .block_name = iree_make_string_view(string_base,
                                              record_value.block_name_length),
          .name = iree_make_string_view(
              string_base + record_value.block_name_length,
              record_value.name_length),
          .description = iree_make_string_view(
              string_base + record_value.block_name_length +
                  record_value.name_length,
              record_value.description_length),
      };
      status = iree_profile_counter_append_counter(context, &counter);
    }
  }
  return status;
}

static iree_status_t iree_profile_counter_process_metadata_record(
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
  fprintf(file,
          ",\"dispatch_event_id\":%" PRIu64 ",\"submission_id\":%" PRIu64
          ",\"command_buffer_id\":%" PRIu64
          ",\"command_index\":%u,\"executable_id\":%" PRIu64
          ",\"export_ordinal\":%u,\"key\":",
          sample->dispatch_event_id, sample->submission_id,
          sample->command_buffer_id, sample->command_index,
          sample->executable_id, sample->export_ordinal);
  iree_profile_fprint_json_string(file, key);
  fprintf(file,
          ",\"physical_device_ordinal\":%u,\"queue_ordinal\":%u"
          ",\"stream_id\":%" PRIu64
          ",\"flags\":%u,\"value\":%.3f"
          ",\"values\":",
          sample->physical_device_ordinal, sample->queue_ordinal,
          sample->stream_id, sample->flags, value_sum);
  iree_profile_counter_print_sample_values_jsonl(counter, sample_values, file);
  fputs("}\n", file);
}

static iree_status_t iree_profile_counter_process_sample_records(
    iree_profile_counter_context_t* counter_context,
    const iree_profile_model_t* model,
    const iree_hal_profile_file_record_t* record, iree_string_view_t filter,
    int64_t id_filter, bool emit_samples, FILE* file) {
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

    iree_hal_profile_counter_sample_record_t sample;
    memcpy(&sample, typed_record.contents.data, sizeof(sample));
    ++counter_context->total_sample_count;

    iree_host_size_t values_length = 0;
    if (!iree_host_size_checked_mul(sample.sample_value_count, sizeof(uint64_t),
                                    &values_length) ||
        values_length != typed_record.inline_payload.data_length) {
      status = iree_make_status(
          IREE_STATUS_DATA_LOSS,
          "counter sample value count is inconsistent with record length");
    }

    const iree_profile_counter_set_t* counter_set = NULL;
    if (iree_status_is_ok(status)) {
      counter_set = iree_profile_counter_find_counter_set(
          counter_context, sample.counter_set_id);
      if (!counter_set) {
        status = iree_make_status(
            IREE_STATUS_DATA_LOSS,
            "counter sample references missing counter-set metadata "
            "sample=%" PRIu64 " counter_set=%" PRIu64,
            sample.sample_id, sample.counter_set_id);
      }
    }
    if (iree_status_is_ok(status) &&
        counter_set->record.sample_value_count != sample.sample_value_count) {
      status = iree_make_status(
          IREE_STATUS_DATA_LOSS,
          "counter sample value count does not match counter-set metadata "
          "sample=%" PRIu64 " counter_set=%" PRIu64,
          sample.sample_id, sample.counter_set_id);
    }

    char numeric_buffer[128];
    iree_string_view_t key = iree_string_view_empty();
    if (iree_status_is_ok(status)) {
      status = iree_profile_counter_resolve_sample_key(
          model, &sample, numeric_buffer, sizeof(numeric_buffer), &key);
    }

    bool matched_sample = false;
    const uint8_t* sample_values = typed_record.inline_payload.data;
    if (iree_status_is_ok(status) &&
        iree_profile_counter_sample_matches_id(&sample, id_filter)) {
      bool found_counter = false;
      for (iree_host_size_t i = 0;
           i < counter_context->counter_count && iree_status_is_ok(status);
           ++i) {
        const iree_profile_counter_t* counter = &counter_context->counters[i];
        if (counter->record.counter_set_id != sample.counter_set_id) {
          continue;
        }
        found_counter = true;
        if (!iree_profile_counter_filter_matches(counter_set, counter, key,
                                                 filter)) {
          continue;
        }

        double value_sum = 0.0;
        status = iree_profile_counter_sum_value(counter, &sample, sample_values,
                                                &value_sum);
        if (iree_status_is_ok(status)) {
          matched_sample = true;
          iree_profile_counter_aggregate_t* aggregate = NULL;
          status = iree_profile_counter_get_aggregate(
              counter_context, sample.physical_device_ordinal,
              sample.queue_ordinal, sample.stream_id, sample.counter_set_id,
              counter->record.counter_ordinal, sample.executable_id,
              sample.export_ordinal, sample.command_buffer_id, &aggregate);
          if (iree_status_is_ok(status)) {
            iree_profile_counter_record_aggregate_sample(aggregate, counter,
                                                         &sample, value_sum);
          }
          if (iree_status_is_ok(status) && emit_samples) {
            iree_profile_counter_print_sample_jsonl(&sample, counter_set,
                                                    counter, key, sample_values,
                                                    value_sum, file);
          }
        }
      }
      if (iree_status_is_ok(status) && !found_counter) {
        status = iree_make_status(
            IREE_STATUS_DATA_LOSS,
            "counter sample references counter set with no counter metadata "
            "sample=%" PRIu64 " counter_set=%" PRIu64,
            sample.sample_id, sample.counter_set_id);
      }
    }
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

static iree_status_t iree_profile_counter_print_text(
    const iree_profile_counter_context_t* counter_context,
    const iree_profile_model_t* model, iree_string_view_t filter,
    int64_t id_filter, FILE* file) {
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

  IREE_RETURN_IF_ERROR(
      iree_profile_counter_print_metadata_text(counter_context, file));

  fprintf(file, "groups:\n");
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < counter_context->aggregate_count && iree_status_is_ok(status); ++i) {
    const iree_profile_counter_aggregate_t* aggregate =
        &counter_context->aggregates[i];
    const iree_profile_counter_set_t* counter_set =
        iree_profile_counter_find_counter_set(counter_context,
                                              aggregate->counter_set_id);
    const iree_profile_counter_t* counter = iree_profile_counter_find_counter(
        counter_context, aggregate->counter_set_id, aggregate->counter_ordinal);
    if (!counter_set || !counter) {
      status = iree_make_status(
          IREE_STATUS_DATA_LOSS,
          "counter aggregate references missing metadata counter_set=%" PRIu64
          " counter_ordinal=%u",
          aggregate->counter_set_id, aggregate->counter_ordinal);
      continue;
    }

    char numeric_buffer[128];
    iree_string_view_t key = iree_string_view_empty();
    status = iree_profile_counter_resolve_aggregate_key(
        model, aggregate, numeric_buffer, sizeof(numeric_buffer), &key);
    if (iree_status_is_ok(status)) {
      const double variance =
          aggregate->sample_count > 1
              ? aggregate->m2_value / (double)(aggregate->sample_count - 1)
              : 0.0;
      const double stddev = iree_profile_sqrt_f64(variance);
      fprintf(file,
              "  %.*s / %.*s.%.*s\n"
              "    device=%u queue=%u stream=%" PRIu64
              " command_buffer=%" PRIu64 " executable=%" PRIu64
              " export=%u key=%.*s\n"
              "    samples=%" PRIu64 " raw_values=%" PRIu64
              " value[min/avg/stddev/max/total]=%.3f/%.3f/%.3f/%.3f/%.3f "
              "unit=%s first_sample=%" PRIu64 " last_sample=%" PRIu64 "\n",
              (int)counter_set->name.size, counter_set->name.data,
              (int)counter->block_name.size, counter->block_name.data,
              (int)counter->name.size, counter->name.data,
              aggregate->physical_device_ordinal, aggregate->queue_ordinal,
              aggregate->stream_id, aggregate->command_buffer_id,
              aggregate->executable_id, aggregate->export_ordinal,
              (int)key.size, key.data, aggregate->sample_count,
              aggregate->raw_value_count,
              aggregate->sample_count ? aggregate->minimum_value : 0.0,
              aggregate->sample_count ? aggregate->mean_value : 0.0, stddev,
              aggregate->sample_count ? aggregate->maximum_value : 0.0,
              aggregate->total_value,
              iree_profile_counter_unit_name(counter->record.unit),
              aggregate->first_sample_id, aggregate->last_sample_id);
    }
  }
  return status;
}

static iree_status_t iree_profile_counter_print_jsonl(
    const iree_profile_counter_context_t* counter_context,
    const iree_profile_model_t* model, iree_string_view_t filter,
    int64_t id_filter, bool emit_samples, FILE* file) {
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

  iree_profile_counter_print_metadata_jsonl(counter_context, file);

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < counter_context->aggregate_count && iree_status_is_ok(status); ++i) {
    const iree_profile_counter_aggregate_t* aggregate =
        &counter_context->aggregates[i];
    const iree_profile_counter_set_t* counter_set =
        iree_profile_counter_find_counter_set(counter_context,
                                              aggregate->counter_set_id);
    const iree_profile_counter_t* counter = iree_profile_counter_find_counter(
        counter_context, aggregate->counter_set_id, aggregate->counter_ordinal);
    if (!counter_set || !counter) {
      status = iree_make_status(
          IREE_STATUS_DATA_LOSS,
          "counter aggregate references missing metadata counter_set=%" PRIu64
          " counter_ordinal=%u",
          aggregate->counter_set_id, aggregate->counter_ordinal);
      continue;
    }

    char numeric_buffer[128];
    iree_string_view_t key = iree_string_view_empty();
    status = iree_profile_counter_resolve_aggregate_key(
        model, aggregate, numeric_buffer, sizeof(numeric_buffer), &key);
    if (iree_status_is_ok(status)) {
      const double variance =
          aggregate->sample_count > 1
              ? aggregate->m2_value / (double)(aggregate->sample_count - 1)
              : 0.0;
      const double stddev = iree_profile_sqrt_f64(variance);
      fprintf(file,
              "{\"type\":\"counter_group\",\"physical_device_ordinal\":%u"
              ",\"queue_ordinal\":%u,\"stream_id\":%" PRIu64
              ",\"counter_set_id\":%" PRIu64 ",\"counter_set\":",
              aggregate->physical_device_ordinal, aggregate->queue_ordinal,
              aggregate->stream_id, aggregate->counter_set_id);
      iree_profile_fprint_json_string(file, counter_set->name);
      fprintf(file, ",\"counter_ordinal\":%u,\"counter\":",
              aggregate->counter_ordinal);
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
      iree_profile_fprint_json_string(file, key);
      fprintf(file,
              ",\"samples\":%" PRIu64 ",\"raw_values\":%" PRIu64
              ",\"min\":%.3f,\"avg\":%.3f,\"stddev\":%.3f"
              ",\"max\":%.3f,\"sum\":%.3f"
              ",\"first_sample_id\":%" PRIu64 ",\"last_sample_id\":%" PRIu64
              "}\n",
              aggregate->sample_count, aggregate->raw_value_count,
              aggregate->sample_count ? aggregate->minimum_value : 0.0,
              aggregate->sample_count ? aggregate->mean_value : 0.0, stddev,
              aggregate->sample_count ? aggregate->maximum_value : 0.0,
              aggregate->total_value, aggregate->first_sample_id,
              aggregate->last_sample_id);
    }
  }
  return status;
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
  // True when raw sample rows should be streamed while parsing.
  bool emit_samples;
  // Output stream receiving raw sample rows when enabled.
  FILE* file;
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
      context->id_filter, context->emit_samples, context->file);
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
  iree_profile_counter_parse_context_t parse_context = {
      .model = &model,
      .counter_context = &counter_context,
      .filter = filter,
      .id_filter = id_filter,
      .emit_samples = emit_samples,
      .file = file,
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
