// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/utils/statistics_sink.h"

#include <inttypes.h>
#include <stddef.h>
#include <string.h>

typedef struct iree_hal_profile_statistics_index_entry_t {
  // Mixed key hash used to select and compare occupied slots.
  uint64_t hash;
  // Stored value plus one; zero marks an empty slot.
  iree_host_size_t value_plus_one;
} iree_hal_profile_statistics_index_entry_t;

typedef struct iree_hal_profile_statistics_index_t {
  // Open-addressed entry table with a power-of-two capacity.
  iree_hal_profile_statistics_index_entry_t* entries;
  // Number of occupied entries in |entries|.
  iree_host_size_t count;
  // Allocated entry count for |entries|.
  iree_host_size_t capacity;
} iree_hal_profile_statistics_index_t;

typedef struct iree_hal_profile_statistics_export_t {
  // Session-local executable identifier owning this export.
  uint64_t executable_id;
  // Export ordinal within |executable_id|.
  uint32_t export_ordinal;
  // Reserved for natural alignment.
  uint32_t reserved0;
  // Export name storage owned by the sink.
  iree_string_view_t name;
} iree_hal_profile_statistics_export_t;

typedef struct iree_hal_profile_statistics_device_t {
  // Session-local physical device ordinal.
  uint32_t physical_device_ordinal;
  // Number of clock-correlation samples seen for this device.
  uint64_t clock_sample_count;
  // First clock-correlation sample seen for this device.
  iree_hal_profile_clock_correlation_record_t first_clock_sample;
  // Last clock-correlation sample seen for this device.
  iree_hal_profile_clock_correlation_record_t last_clock_sample;
} iree_hal_profile_statistics_device_t;

typedef struct iree_hal_profile_statistics_sink_t {
  // HAL resource header for the sink interface.
  iree_hal_resource_t resource;
  // Host allocator used for sink lifetime and aggregate storage.
  iree_allocator_t host_allocator;
  // Dynamic array of aggregate rows.
  iree_hal_profile_statistics_row_t* rows;
  // Number of valid entries in |rows|.
  iree_host_size_t row_count;
  // Capacity of |rows| in entries.
  iree_host_size_t row_capacity;
  // Lookup index from aggregate row keys to |rows| entry indexes.
  iree_hal_profile_statistics_index_t row_index;
  // Dynamic array of executable export metadata rows.
  iree_hal_profile_statistics_export_t* exports;
  // Number of valid entries in |exports|.
  iree_host_size_t export_count;
  // Capacity of |exports| in entries.
  iree_host_size_t export_capacity;
  // Lookup index from executable/export keys to |exports| entry indexes.
  iree_hal_profile_statistics_index_t export_index;
  // Dynamic array of per-device clock-correlation rows.
  iree_hal_profile_statistics_device_t* devices;
  // Number of valid entries in |devices|.
  iree_host_size_t device_count;
  // Capacity of |devices| in entries.
  iree_host_size_t device_capacity;
  // Source records reported as dropped by truncated producer chunks.
  uint64_t dropped_record_count;
} iree_hal_profile_statistics_sink_t;

static const iree_hal_profile_sink_vtable_t
    iree_hal_profile_statistics_sink_vtable;

static iree_hal_profile_statistics_sink_t*
iree_hal_profile_statistics_sink_cast(iree_hal_profile_sink_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_profile_statistics_sink_vtable);
  return (iree_hal_profile_statistics_sink_t*)base_value;
}

static uint64_t iree_hal_profile_statistics_mix_u64(uint64_t value) {
  value += UINT64_C(0x9e3779b97f4a7c15);
  value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
  return value ^ (value >> 31);
}

static uint64_t iree_hal_profile_statistics_combine_u64(uint64_t hash,
                                                        uint64_t value) {
  return hash ^ (iree_hal_profile_statistics_mix_u64(value) +
                 UINT64_C(0x9e3779b97f4a7c15) + (hash << 6) + (hash >> 2));
}

static iree_host_size_t iree_hal_profile_statistics_index_slot(
    const iree_hal_profile_statistics_index_t* index, uint64_t hash,
    iree_host_size_t probe) {
  return (iree_host_size_t)((hash + probe) & (index->capacity - 1));
}

typedef bool (*iree_hal_profile_statistics_index_match_fn_t)(
    const void* user_data, iree_host_size_t value);

static bool iree_hal_profile_statistics_index_find(
    const iree_hal_profile_statistics_index_t* index, uint64_t hash,
    iree_hal_profile_statistics_index_match_fn_t match, const void* user_data,
    iree_host_size_t* out_value) {
  if (index->capacity == 0) return false;
  for (iree_host_size_t i = 0; i < index->capacity; ++i) {
    const iree_hal_profile_statistics_index_entry_t* entry =
        &index->entries[iree_hal_profile_statistics_index_slot(index, hash, i)];
    if (entry->value_plus_one == 0) return false;
    if (entry->hash == hash && match(user_data, entry->value_plus_one - 1)) {
      *out_value = entry->value_plus_one - 1;
      return true;
    }
  }
  return false;
}

static void iree_hal_profile_statistics_index_insert_existing(
    iree_hal_profile_statistics_index_t* index, uint64_t hash,
    iree_host_size_t value) {
  for (iree_host_size_t i = 0; i < index->capacity; ++i) {
    iree_hal_profile_statistics_index_entry_t* entry =
        &index->entries[iree_hal_profile_statistics_index_slot(index, hash, i)];
    if (entry->value_plus_one == 0) {
      entry->hash = hash;
      entry->value_plus_one = value + 1;
      ++index->count;
      return;
    }
  }
  IREE_ASSERT_UNREACHABLE("statistics index must have reserved capacity");
}

static iree_status_t iree_hal_profile_statistics_index_reserve(
    iree_hal_profile_statistics_index_t* index, iree_allocator_t host_allocator,
    iree_host_size_t minimum_count) {
  if (minimum_count <= index->capacity / 2) return iree_ok_status();

  iree_host_size_t new_capacity = index->capacity ? index->capacity : 16;
  while (minimum_count > new_capacity / 2) {
    if (new_capacity > IREE_HOST_SIZE_MAX / 2) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "profile statistics index capacity overflow");
    }
    new_capacity *= 2;
  }
  if (new_capacity > IREE_HOST_SIZE_MAX / sizeof(index->entries[0])) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "profile statistics index storage overflow");
  }

  iree_hal_profile_statistics_index_entry_t* new_entries = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, new_capacity * sizeof(new_entries[0]),
      (void**)&new_entries));
  memset(new_entries, 0, new_capacity * sizeof(new_entries[0]));

  iree_hal_profile_statistics_index_t new_index = {
      .entries = new_entries,
      .count = 0,
      .capacity = new_capacity,
  };
  for (iree_host_size_t i = 0; i < index->capacity; ++i) {
    const iree_hal_profile_statistics_index_entry_t* entry = &index->entries[i];
    if (entry->value_plus_one != 0) {
      iree_hal_profile_statistics_index_insert_existing(
          &new_index, entry->hash, entry->value_plus_one - 1);
    }
  }

  iree_allocator_free(host_allocator, index->entries);
  *index = new_index;
  return iree_ok_status();
}

static iree_status_t iree_hal_profile_statistics_index_insert(
    iree_hal_profile_statistics_index_t* index, iree_allocator_t host_allocator,
    uint64_t hash, iree_host_size_t value) {
  IREE_RETURN_IF_ERROR(iree_hal_profile_statistics_index_reserve(
      index, host_allocator, index->count + 1));
  iree_hal_profile_statistics_index_insert_existing(index, hash, value);
  return iree_ok_status();
}

static void iree_hal_profile_statistics_index_deinitialize(
    iree_hal_profile_statistics_index_t* index,
    iree_allocator_t host_allocator) {
  iree_allocator_free(host_allocator, index->entries);
  memset(index, 0, sizeof(*index));
}

static uint64_t iree_hal_profile_statistics_row_hash(
    const iree_hal_profile_statistics_row_t* row) {
  uint64_t hash = iree_hal_profile_statistics_mix_u64(row->row_type);
  hash = iree_hal_profile_statistics_combine_u64(hash, row->time_domain);
  hash = iree_hal_profile_statistics_combine_u64(hash,
                                                 row->physical_device_ordinal);
  hash = iree_hal_profile_statistics_combine_u64(hash, row->queue_ordinal);
  hash = iree_hal_profile_statistics_combine_u64(hash, row->event_type);
  hash = iree_hal_profile_statistics_combine_u64(hash, row->executable_id);
  hash = iree_hal_profile_statistics_combine_u64(hash, row->command_buffer_id);
  hash = iree_hal_profile_statistics_combine_u64(hash, row->export_ordinal);
  return iree_hal_profile_statistics_combine_u64(hash, row->command_index);
}

static bool iree_hal_profile_statistics_row_matches(const void* user_data,
                                                    iree_host_size_t value) {
  const iree_hal_profile_statistics_sink_t* sink =
      (const iree_hal_profile_statistics_sink_t*)user_data;
  const iree_hal_profile_statistics_row_t* expected =
      &sink->rows[sink->row_count];
  const iree_hal_profile_statistics_row_t* candidate = &sink->rows[value];
  return candidate->row_type == expected->row_type &&
         candidate->time_domain == expected->time_domain &&
         candidate->physical_device_ordinal ==
             expected->physical_device_ordinal &&
         candidate->queue_ordinal == expected->queue_ordinal &&
         candidate->event_type == expected->event_type &&
         candidate->executable_id == expected->executable_id &&
         candidate->command_buffer_id == expected->command_buffer_id &&
         candidate->export_ordinal == expected->export_ordinal &&
         candidate->command_index == expected->command_index;
}

static void iree_hal_profile_statistics_row_initialize(
    iree_hal_profile_statistics_row_t* row,
    iree_hal_profile_statistics_row_type_t row_type,
    iree_hal_profile_statistics_time_domain_t time_domain,
    uint32_t physical_device_ordinal, uint32_t queue_ordinal,
    uint32_t event_type, uint64_t executable_id, uint64_t command_buffer_id,
    uint32_t export_ordinal, uint32_t command_index) {
  memset(row, 0, sizeof(*row));
  row->row_type = row_type;
  row->time_domain = time_domain;
  row->physical_device_ordinal = physical_device_ordinal;
  row->queue_ordinal = queue_ordinal;
  row->event_type = event_type;
  row->executable_id = executable_id;
  row->command_buffer_id = command_buffer_id;
  row->export_ordinal = export_ordinal;
  row->command_index = command_index;
  row->first_start_time = UINT64_MAX;
  row->minimum_duration = UINT64_MAX;
}

static iree_status_t iree_hal_profile_statistics_sink_ensure_row(
    iree_hal_profile_statistics_sink_t* sink,
    iree_hal_profile_statistics_row_type_t row_type,
    iree_hal_profile_statistics_time_domain_t time_domain,
    uint32_t physical_device_ordinal, uint32_t queue_ordinal,
    uint32_t event_type, uint64_t executable_id, uint64_t command_buffer_id,
    uint32_t export_ordinal, uint32_t command_index,
    iree_hal_profile_statistics_row_t** out_row) {
  *out_row = NULL;

  if (sink->row_count + 1 > sink->row_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        sink->host_allocator,
        iree_max((iree_host_size_t)16, sink->row_count + 1),
        sizeof(sink->rows[0]), &sink->row_capacity, (void**)&sink->rows));
  }

  iree_hal_profile_statistics_row_t* probe = &sink->rows[sink->row_count];
  iree_hal_profile_statistics_row_initialize(
      probe, row_type, time_domain, physical_device_ordinal, queue_ordinal,
      event_type, executable_id, command_buffer_id, export_ordinal,
      command_index);
  const uint64_t hash = iree_hal_profile_statistics_row_hash(probe);

  iree_host_size_t existing_index = 0;
  if (iree_hal_profile_statistics_index_find(
          &sink->row_index, hash, iree_hal_profile_statistics_row_matches, sink,
          &existing_index)) {
    *out_row = &sink->rows[existing_index];
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(iree_hal_profile_statistics_index_insert(
      &sink->row_index, sink->host_allocator, hash, sink->row_count));
  *out_row = probe;
  ++sink->row_count;
  return iree_ok_status();
}

static uint64_t iree_hal_profile_statistics_export_hash(
    uint64_t executable_id, uint32_t export_ordinal) {
  uint64_t hash = iree_hal_profile_statistics_mix_u64(executable_id);
  return iree_hal_profile_statistics_combine_u64(hash, export_ordinal);
}

typedef struct iree_hal_profile_statistics_export_lookup_t {
  // Statistics sink owning candidate exports.
  const iree_hal_profile_statistics_sink_t* sink;
  // Session-local executable identifier.
  uint64_t executable_id;
  // Export ordinal within |executable_id|.
  uint32_t export_ordinal;
} iree_hal_profile_statistics_export_lookup_t;

static bool iree_hal_profile_statistics_export_matches(const void* user_data,
                                                       iree_host_size_t value) {
  const iree_hal_profile_statistics_export_lookup_t* lookup =
      (const iree_hal_profile_statistics_export_lookup_t*)user_data;
  const iree_hal_profile_statistics_export_t* candidate =
      &lookup->sink->exports[value];
  return candidate->executable_id == lookup->executable_id &&
         candidate->export_ordinal == lookup->export_ordinal;
}

static iree_status_t iree_hal_profile_statistics_sink_insert_export(
    iree_hal_profile_statistics_sink_t* sink, uint64_t executable_id,
    uint32_t export_ordinal, iree_string_view_t name) {
  const uint64_t hash =
      iree_hal_profile_statistics_export_hash(executable_id, export_ordinal);
  const iree_hal_profile_statistics_export_lookup_t lookup = {
      .sink = sink,
      .executable_id = executable_id,
      .export_ordinal = export_ordinal,
  };
  iree_host_size_t existing_index = 0;
  if (iree_hal_profile_statistics_index_find(
          &sink->export_index, hash, iree_hal_profile_statistics_export_matches,
          &lookup, &existing_index)) {
    return iree_ok_status();
  }

  if (sink->export_count + 1 > sink->export_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        sink->host_allocator,
        iree_max((iree_host_size_t)16, sink->export_count + 1),
        sizeof(sink->exports[0]), &sink->export_capacity,
        (void**)&sink->exports));
  }
  IREE_RETURN_IF_ERROR(iree_hal_profile_statistics_index_reserve(
      &sink->export_index, sink->host_allocator, sink->export_index.count + 1));

  char* name_storage = NULL;
  if (!iree_string_view_is_empty(name)) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(sink->host_allocator, name.size,
                                               (void**)&name_storage));
    memcpy(name_storage, name.data, name.size);
  }

  const iree_host_size_t export_index = sink->export_count;
  iree_hal_profile_statistics_index_insert_existing(&sink->export_index, hash,
                                                    export_index);
  sink->exports[export_index] = (iree_hal_profile_statistics_export_t){
      .executable_id = executable_id,
      .export_ordinal = export_ordinal,
      .reserved0 = 0,
      .name = iree_make_string_view(name_storage, name.size),
  };
  ++sink->export_count;
  return iree_ok_status();
}

static const iree_hal_profile_statistics_device_t*
iree_hal_profile_statistics_sink_find_device(
    const iree_hal_profile_statistics_sink_t* sink,
    uint32_t physical_device_ordinal) {
  for (iree_host_size_t i = 0; i < sink->device_count; ++i) {
    if (sink->devices[i].physical_device_ordinal == physical_device_ordinal) {
      return &sink->devices[i];
    }
  }
  return NULL;
}

static iree_hal_profile_statistics_device_t*
iree_hal_profile_statistics_sink_find_device_mutable(
    iree_hal_profile_statistics_sink_t* sink,
    uint32_t physical_device_ordinal) {
  for (iree_host_size_t i = 0; i < sink->device_count; ++i) {
    if (sink->devices[i].physical_device_ordinal == physical_device_ordinal) {
      return &sink->devices[i];
    }
  }
  return NULL;
}

static iree_status_t iree_hal_profile_statistics_sink_ensure_device(
    iree_hal_profile_statistics_sink_t* sink, uint32_t physical_device_ordinal,
    iree_hal_profile_statistics_device_t** out_device) {
  *out_device = iree_hal_profile_statistics_sink_find_device_mutable(
      sink, physical_device_ordinal);
  if (*out_device) return iree_ok_status();

  if (sink->device_count + 1 > sink->device_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        sink->host_allocator,
        iree_max((iree_host_size_t)4, sink->device_count + 1),
        sizeof(sink->devices[0]), &sink->device_capacity,
        (void**)&sink->devices));
  }

  iree_hal_profile_statistics_device_t* device =
      &sink->devices[sink->device_count++];
  memset(device, 0, sizeof(*device));
  device->physical_device_ordinal = physical_device_ordinal;
  *out_device = device;
  return iree_ok_status();
}

static void iree_hal_profile_statistics_device_record_clock_sample(
    iree_hal_profile_statistics_device_t* device,
    const iree_hal_profile_clock_correlation_record_t* record) {
  if (device->clock_sample_count == 0) {
    device->first_clock_sample = *record;
  }
  device->last_clock_sample = *record;
  ++device->clock_sample_count;
}

static bool iree_hal_profile_statistics_clock_sample_host_midpoint(
    const iree_hal_profile_clock_correlation_record_t* sample,
    int64_t* out_time_ns) {
  if (!iree_all_bits_set(
          sample->flags,
          IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_HOST_TIME_BRACKET) ||
      sample->host_time_begin_ns < 0 ||
      sample->host_time_end_ns < sample->host_time_begin_ns) {
    return false;
  }
  *out_time_ns = sample->host_time_begin_ns +
                 (sample->host_time_end_ns - sample->host_time_begin_ns) / 2;
  return true;
}

static bool iree_hal_profile_statistics_clock_sample_host_cpu_timestamp(
    const iree_hal_profile_clock_correlation_record_t* sample,
    int64_t* out_time_ns) {
  if (!iree_all_bits_set(
          sample->flags,
          IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_HOST_CPU_TIMESTAMP) ||
      sample->host_cpu_timestamp_ns > INT64_MAX) {
    return false;
  }
  *out_time_ns = (int64_t)sample->host_cpu_timestamp_ns;
  return true;
}

static bool iree_hal_profile_statistics_round_mul_div_u64(
    uint64_t value, uint64_t numerator, uint64_t denominator,
    uint64_t* out_result) {
  *out_result = 0;
  if (denominator == 0) return false;
  if (value == 0 || numerator == 0) return true;

#if defined(__SIZEOF_INT128__)
  __uint128_t product = (__uint128_t)value * (__uint128_t)numerator;
  product += denominator / 2;
  __uint128_t quotient = product / denominator;
  if (quotient > UINT64_MAX) return false;
  *out_result = (uint64_t)quotient;
  return true;
#else
  const uint64_t whole = value / denominator;
  const uint64_t remainder = value % denominator;
  if (whole > UINT64_MAX / numerator) return false;
  uint64_t scaled = whole * numerator;
  if (remainder != 0) {
    if (remainder > UINT64_MAX / numerator) return false;
    uint64_t fractional_product = remainder * numerator;
    if (fractional_product > UINT64_MAX - denominator / 2) return false;
    uint64_t fractional = (fractional_product + denominator / 2) / denominator;
    if (scaled > UINT64_MAX - fractional) return false;
    scaled += fractional;
  }
  *out_result = scaled;
  return true;
#endif  // defined(__SIZEOF_INT128__)
}

static bool iree_hal_profile_statistics_scale_device_ticks_to_ns(
    const iree_hal_profile_statistics_device_t* device, uint64_t duration_ticks,
    uint64_t* out_duration_ns) {
  *out_duration_ns = 0;
  if (!device || device->clock_sample_count < 2) return false;

  const iree_hal_profile_clock_correlation_record_t* first =
      &device->first_clock_sample;
  const iree_hal_profile_clock_correlation_record_t* last =
      &device->last_clock_sample;
  if (!iree_all_bits_set(first->flags,
                         IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_DEVICE_TICK) ||
      !iree_all_bits_set(last->flags,
                         IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_DEVICE_TICK) ||
      last->device_tick <= first->device_tick) {
    return false;
  }

  int64_t first_time_ns = 0;
  int64_t last_time_ns = 0;
  if (!iree_hal_profile_statistics_clock_sample_host_cpu_timestamp(
          first, &first_time_ns) ||
      !iree_hal_profile_statistics_clock_sample_host_cpu_timestamp(
          last, &last_time_ns)) {
    if (!iree_hal_profile_statistics_clock_sample_host_midpoint(
            first, &first_time_ns) ||
        !iree_hal_profile_statistics_clock_sample_host_midpoint(
            last, &last_time_ns)) {
      return false;
    }
  }
  if (last_time_ns <= first_time_ns) return false;

  return iree_hal_profile_statistics_round_mul_div_u64(
      duration_ticks, (uint64_t)(last_time_ns - first_time_ns),
      last->device_tick - first->device_tick, out_duration_ns);
}

static bool iree_hal_profile_statistics_add_u64(uint64_t* value,
                                                uint64_t delta) {
  if (delta > UINT64_MAX - *value) return false;
  *value += delta;
  return true;
}

static iree_status_t iree_hal_profile_statistics_sink_add_dropped_records(
    iree_hal_profile_statistics_sink_t* sink,
    const iree_hal_profile_chunk_metadata_t* metadata) {
  if (metadata->dropped_record_count == 0) return iree_ok_status();
  if (!iree_hal_profile_statistics_add_u64(&sink->dropped_record_count,
                                           metadata->dropped_record_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "profile statistics dropped record count overflow");
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_profile_statistics_row_add_sample(
    iree_hal_profile_statistics_row_t* row, bool has_timing,
    uint64_t start_time, uint64_t end_time, uint64_t operation_count,
    uint64_t payload_bytes, uint64_t tile_count,
    uint64_t tile_duration_sum_ns) {
  if (!iree_hal_profile_statistics_add_u64(&row->sample_count, 1)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "profile statistics sample count overflow");
  }
  if (operation_count != 0) {
    if (!iree_hal_profile_statistics_add_u64(&row->operation_count,
                                             operation_count)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "profile statistics operation count overflow");
    }
    row->flags |= IREE_HAL_PROFILE_STATISTICS_ROW_FLAG_OPERATION_COUNT;
  }
  if (payload_bytes != 0) {
    if (!iree_hal_profile_statistics_add_u64(&row->payload_bytes,
                                             payload_bytes)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "profile statistics payload byte count overflow");
    }
    row->flags |= IREE_HAL_PROFILE_STATISTICS_ROW_FLAG_PAYLOAD_BYTES;
  }
  if (tile_count != 0) {
    if (!iree_hal_profile_statistics_add_u64(&row->tile_count, tile_count) ||
        !iree_hal_profile_statistics_add_u64(&row->tile_duration_sum_ns,
                                             tile_duration_sum_ns)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "profile statistics tile count overflow");
    }
    row->flags |= IREE_HAL_PROFILE_STATISTICS_ROW_FLAG_TILE_TOTALS;
  }

  if (!has_timing || end_time < start_time) {
    if (!iree_hal_profile_statistics_add_u64(&row->invalid_sample_count, 1)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "profile statistics invalid sample overflow");
    }
    return iree_ok_status();
  }

  const uint64_t duration = end_time - start_time;
  if (!iree_hal_profile_statistics_add_u64(&row->total_duration, duration)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "profile statistics duration overflow");
  }
  row->flags |= IREE_HAL_PROFILE_STATISTICS_ROW_FLAG_TIMING;
  row->first_start_time = iree_min(row->first_start_time, start_time);
  row->last_end_time = iree_max(row->last_end_time, end_time);
  row->minimum_duration = iree_min(row->minimum_duration, duration);
  row->maximum_duration = iree_max(row->maximum_duration, duration);
  return iree_ok_status();
}

static iree_status_t iree_hal_profile_statistics_sink_add_row_sample(
    iree_hal_profile_statistics_sink_t* sink,
    iree_hal_profile_statistics_row_type_t row_type,
    iree_hal_profile_statistics_time_domain_t time_domain,
    uint32_t physical_device_ordinal, uint32_t queue_ordinal,
    uint32_t event_type, uint64_t executable_id, uint64_t command_buffer_id,
    uint32_t export_ordinal, uint32_t command_index, bool has_timing,
    uint64_t start_time, uint64_t end_time, uint64_t operation_count,
    uint64_t payload_bytes, uint64_t tile_count,
    uint64_t tile_duration_sum_ns) {
  iree_hal_profile_statistics_row_t* row = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_profile_statistics_sink_ensure_row(
      sink, row_type, time_domain, physical_device_ordinal, queue_ordinal,
      event_type, executable_id, command_buffer_id, export_ordinal,
      command_index, &row));
  return iree_hal_profile_statistics_row_add_sample(
      row, has_timing, start_time, end_time, operation_count, payload_bytes,
      tile_count, tile_duration_sum_ns);
}

static bool iree_hal_profile_statistics_host_span_is_valid(
    int64_t start_time_ns, int64_t end_time_ns) {
  return start_time_ns >= 0 && end_time_ns >= start_time_ns;
}

typedef iree_status_t (*iree_hal_profile_statistics_fixed_record_fn_t)(
    iree_hal_profile_statistics_sink_t* sink,
    const iree_hal_profile_chunk_metadata_t* metadata, const void* record);

static iree_status_t iree_hal_profile_statistics_for_each_fixed_record(
    iree_hal_profile_statistics_sink_t* sink,
    const iree_hal_profile_chunk_metadata_t* metadata,
    iree_host_size_t iovec_count, const iree_const_byte_span_t* iovecs,
    iree_host_size_t record_size,
    iree_hal_profile_statistics_fixed_record_fn_t record_callback) {
  if (iovec_count > 0 && !iovecs) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "profile statistics chunk iovec list is required");
  }
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; iree_status_is_ok(status) && i < iovec_count;
       ++i) {
    if (iovecs[i].data_length % record_size != 0) {
      return iree_make_status(
          IREE_STATUS_DATA_LOSS,
          "profile statistics chunk '%.*s' has a partial record",
          (int)metadata->content_type.size, metadata->content_type.data);
    }
    for (iree_host_size_t offset = 0;
         iree_status_is_ok(status) && offset < iovecs[i].data_length;
         offset += record_size) {
      status = record_callback(sink, metadata, iovecs[i].data + offset);
    }
  }
  return status;
}

static iree_status_t iree_hal_profile_statistics_process_dispatch_event(
    iree_hal_profile_statistics_sink_t* sink,
    const iree_hal_profile_chunk_metadata_t* metadata, const void* record) {
  const iree_hal_profile_dispatch_event_t* event =
      (const iree_hal_profile_dispatch_event_t*)record;
  const bool has_timing =
      event->start_tick != 0 && event->end_tick >= event->start_tick;

  iree_status_t status = iree_ok_status();
  if (event->executable_id != 0 && event->export_ordinal != UINT32_MAX) {
    status = iree_hal_profile_statistics_sink_add_row_sample(
        sink, IREE_HAL_PROFILE_STATISTICS_ROW_TYPE_DISPATCH_EXPORT,
        IREE_HAL_PROFILE_STATISTICS_TIME_DOMAIN_DEVICE_TICK,
        metadata->physical_device_ordinal, metadata->queue_ordinal,
        /*event_type=*/0, event->executable_id, /*command_buffer_id=*/0,
        event->export_ordinal, UINT32_MAX, has_timing, event->start_tick,
        event->end_tick, /*operation_count=*/0, /*payload_bytes=*/0,
        /*tile_count=*/0, /*tile_duration_sum_ns=*/0);
  }
  if (iree_status_is_ok(status) && event->command_buffer_id != 0) {
    status = iree_hal_profile_statistics_sink_add_row_sample(
        sink, IREE_HAL_PROFILE_STATISTICS_ROW_TYPE_DISPATCH_COMMAND_BUFFER,
        IREE_HAL_PROFILE_STATISTICS_TIME_DOMAIN_DEVICE_TICK,
        metadata->physical_device_ordinal, metadata->queue_ordinal,
        /*event_type=*/0, /*executable_id=*/0, event->command_buffer_id,
        UINT32_MAX, UINT32_MAX, has_timing, event->start_tick, event->end_tick,
        /*operation_count=*/0, /*payload_bytes=*/0, /*tile_count=*/0,
        /*tile_duration_sum_ns=*/0);
  }
  if (iree_status_is_ok(status) && event->command_buffer_id != 0 &&
      event->command_index != UINT32_MAX) {
    status = iree_hal_profile_statistics_sink_add_row_sample(
        sink, IREE_HAL_PROFILE_STATISTICS_ROW_TYPE_DISPATCH_COMMAND_OPERATION,
        IREE_HAL_PROFILE_STATISTICS_TIME_DOMAIN_DEVICE_TICK,
        metadata->physical_device_ordinal, metadata->queue_ordinal,
        /*event_type=*/0, event->executable_id, event->command_buffer_id,
        event->export_ordinal, event->command_index, has_timing,
        event->start_tick, event->end_tick, /*operation_count=*/0,
        /*payload_bytes=*/0, /*tile_count=*/0, /*tile_duration_sum_ns=*/0);
  }
  return status;
}

static iree_status_t iree_hal_profile_statistics_process_clock_correlation(
    iree_hal_profile_statistics_sink_t* sink,
    const iree_hal_profile_chunk_metadata_t* metadata, const void* record) {
  (void)metadata;
  const iree_hal_profile_clock_correlation_record_t* clock_record =
      (const iree_hal_profile_clock_correlation_record_t*)record;
  iree_hal_profile_statistics_device_t* device = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_profile_statistics_sink_ensure_device(
      sink, clock_record->physical_device_ordinal, &device));
  iree_hal_profile_statistics_device_record_clock_sample(device, clock_record);
  return iree_ok_status();
}

static iree_status_t iree_hal_profile_statistics_process_queue_device_event(
    iree_hal_profile_statistics_sink_t* sink,
    const iree_hal_profile_chunk_metadata_t* metadata, const void* record) {
  (void)metadata;
  const iree_hal_profile_queue_device_event_t* event =
      (const iree_hal_profile_queue_device_event_t*)record;
  const bool has_timing =
      event->start_tick != 0 && event->end_tick >= event->start_tick;
  return iree_hal_profile_statistics_sink_add_row_sample(
      sink, IREE_HAL_PROFILE_STATISTICS_ROW_TYPE_QUEUE_DEVICE_OPERATION,
      IREE_HAL_PROFILE_STATISTICS_TIME_DOMAIN_DEVICE_TICK,
      event->physical_device_ordinal, event->queue_ordinal, event->type,
      /*executable_id=*/0, event->command_buffer_id, UINT32_MAX, UINT32_MAX,
      has_timing, event->start_tick, event->end_tick, event->operation_count,
      event->payload_length, /*tile_count=*/0, /*tile_duration_sum_ns=*/0);
}

static iree_status_t iree_hal_profile_statistics_process_queue_event(
    iree_hal_profile_statistics_sink_t* sink,
    const iree_hal_profile_chunk_metadata_t* metadata, const void* record) {
  (void)metadata;
  const iree_hal_profile_queue_event_t* event =
      (const iree_hal_profile_queue_event_t*)record;
  const bool has_timing = iree_hal_profile_statistics_host_span_is_valid(
      event->host_time_ns, event->ready_host_time_ns);
  return iree_hal_profile_statistics_sink_add_row_sample(
      sink, IREE_HAL_PROFILE_STATISTICS_ROW_TYPE_QUEUE_HOST_OPERATION,
      IREE_HAL_PROFILE_STATISTICS_TIME_DOMAIN_IREE_HOST_TIME_NS,
      event->physical_device_ordinal, event->queue_ordinal, event->type,
      /*executable_id=*/0, event->command_buffer_id, UINT32_MAX, UINT32_MAX,
      has_timing, has_timing ? (uint64_t)event->host_time_ns : 0,
      has_timing ? (uint64_t)event->ready_host_time_ns : 0,
      event->operation_count, event->payload_length, /*tile_count=*/0,
      /*tile_duration_sum_ns=*/0);
}

static iree_status_t iree_hal_profile_statistics_process_host_execution_event(
    iree_hal_profile_statistics_sink_t* sink,
    const iree_hal_profile_chunk_metadata_t* metadata, const void* record) {
  (void)metadata;
  const iree_hal_profile_host_execution_event_t* event =
      (const iree_hal_profile_host_execution_event_t*)record;
  const bool has_timing = iree_hal_profile_statistics_host_span_is_valid(
      event->start_host_time_ns, event->end_host_time_ns);
  const uint64_t start_time =
      has_timing ? (uint64_t)event->start_host_time_ns : 0;
  const uint64_t end_time = has_timing ? (uint64_t)event->end_host_time_ns : 0;
  const uint64_t tile_duration_sum_ns =
      event->tile_duration_sum_ns >= 0 ? (uint64_t)event->tile_duration_sum_ns
                                       : 0;

  iree_status_t status = iree_hal_profile_statistics_sink_add_row_sample(
      sink, IREE_HAL_PROFILE_STATISTICS_ROW_TYPE_HOST_EXECUTION_QUEUE_OPERATION,
      IREE_HAL_PROFILE_STATISTICS_TIME_DOMAIN_IREE_HOST_TIME_NS,
      event->physical_device_ordinal, event->queue_ordinal, event->type,
      /*executable_id=*/0, event->command_buffer_id, UINT32_MAX, UINT32_MAX,
      has_timing, start_time, end_time, event->operation_count,
      event->payload_length, event->tile_count, tile_duration_sum_ns);
  if (iree_status_is_ok(status) && event->executable_id != 0 &&
      event->export_ordinal != UINT32_MAX) {
    status = iree_hal_profile_statistics_sink_add_row_sample(
        sink, IREE_HAL_PROFILE_STATISTICS_ROW_TYPE_HOST_EXECUTION_EXPORT,
        IREE_HAL_PROFILE_STATISTICS_TIME_DOMAIN_IREE_HOST_TIME_NS,
        event->physical_device_ordinal, event->queue_ordinal,
        /*event_type=*/0, event->executable_id, /*command_buffer_id=*/0,
        event->export_ordinal, UINT32_MAX, has_timing, start_time, end_time,
        event->operation_count, event->payload_length, event->tile_count,
        tile_duration_sum_ns);
  }
  if (iree_status_is_ok(status) && event->command_buffer_id != 0) {
    status = iree_hal_profile_statistics_sink_add_row_sample(
        sink,
        IREE_HAL_PROFILE_STATISTICS_ROW_TYPE_HOST_EXECUTION_COMMAND_BUFFER,
        IREE_HAL_PROFILE_STATISTICS_TIME_DOMAIN_IREE_HOST_TIME_NS,
        event->physical_device_ordinal, event->queue_ordinal,
        /*event_type=*/0, /*executable_id=*/0, event->command_buffer_id,
        UINT32_MAX, UINT32_MAX, has_timing, start_time, end_time,
        event->operation_count, event->payload_length, event->tile_count,
        tile_duration_sum_ns);
  }
  if (iree_status_is_ok(status) && event->command_buffer_id != 0 &&
      event->command_index != UINT32_MAX) {
    status = iree_hal_profile_statistics_sink_add_row_sample(
        sink,
        IREE_HAL_PROFILE_STATISTICS_ROW_TYPE_HOST_EXECUTION_COMMAND_OPERATION,
        IREE_HAL_PROFILE_STATISTICS_TIME_DOMAIN_IREE_HOST_TIME_NS,
        event->physical_device_ordinal, event->queue_ordinal,
        /*event_type=*/0, event->executable_id, event->command_buffer_id,
        event->export_ordinal, event->command_index, has_timing, start_time,
        end_time, event->operation_count, event->payload_length,
        event->tile_count, tile_duration_sum_ns);
  }
  return status;
}

static iree_status_t iree_hal_profile_statistics_process_memory_event(
    iree_hal_profile_statistics_sink_t* sink,
    const iree_hal_profile_chunk_metadata_t* metadata, const void* record) {
  (void)metadata;
  const iree_hal_profile_memory_event_t* event =
      (const iree_hal_profile_memory_event_t*)record;
  return iree_hal_profile_statistics_sink_add_row_sample(
      sink, IREE_HAL_PROFILE_STATISTICS_ROW_TYPE_MEMORY_LIFECYCLE,
      IREE_HAL_PROFILE_STATISTICS_TIME_DOMAIN_NONE,
      event->physical_device_ordinal, event->queue_ordinal, event->type,
      /*executable_id=*/0, /*command_buffer_id=*/0, UINT32_MAX, UINT32_MAX,
      /*has_timing=*/false, /*start_time=*/0, /*end_time=*/0,
      /*operation_count=*/0, event->length, /*tile_count=*/0,
      /*tile_duration_sum_ns=*/0);
}

static iree_status_t iree_hal_profile_statistics_process_export_iovec(
    iree_hal_profile_statistics_sink_t* sink, iree_const_byte_span_t iovec) {
  iree_status_t status = iree_ok_status();
  iree_host_size_t offset = 0;
  while (iree_status_is_ok(status) && offset < iovec.data_length) {
    const iree_host_size_t remaining_length = iovec.data_length - offset;
    if (remaining_length <
        sizeof(iree_hal_profile_executable_export_record_t)) {
      return iree_make_status(
          IREE_STATUS_DATA_LOSS,
          "profile statistics executable-export chunk has a partial record");
    }

    iree_hal_profile_executable_export_record_t record;
    memcpy(&record, iovec.data + offset, sizeof(record));
    if (record.record_length <
            sizeof(iree_hal_profile_executable_export_record_t) ||
        record.record_length > remaining_length) {
      return iree_make_status(
          IREE_STATUS_DATA_LOSS,
          "profile statistics executable-export record length is invalid");
    }

    const iree_host_size_t trailing_length =
        record.record_length - sizeof(record);
    if ((iree_host_size_t)record.name_length > trailing_length) {
      return iree_make_status(
          IREE_STATUS_DATA_LOSS,
          "profile statistics executable-export name length is invalid");
    }

    const char* name_data = (const char*)iovec.data + offset + sizeof(record);
    const iree_string_view_t name =
        iree_make_string_view(name_data, record.name_length);
    status = iree_hal_profile_statistics_sink_insert_export(
        sink, record.executable_id, record.export_ordinal, name);
    offset += record.record_length;
  }
  return status;
}

static iree_status_t iree_hal_profile_statistics_process_export_records(
    iree_hal_profile_statistics_sink_t* sink, iree_host_size_t iovec_count,
    const iree_const_byte_span_t* iovecs) {
  if (iovec_count > 0 && !iovecs) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "profile statistics chunk iovec list is required");
  }
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; iree_status_is_ok(status) && i < iovec_count;
       ++i) {
    status = iree_hal_profile_statistics_process_export_iovec(sink, iovecs[i]);
  }
  return status;
}

static void iree_hal_profile_statistics_sink_reset(
    iree_hal_profile_statistics_sink_t* sink) {
  iree_allocator_t host_allocator = sink->host_allocator;
  for (iree_host_size_t i = 0; i < sink->export_count; ++i) {
    iree_allocator_free(host_allocator, (void*)sink->exports[i].name.data);
  }
  iree_hal_profile_statistics_index_deinitialize(&sink->export_index,
                                                 host_allocator);
  iree_hal_profile_statistics_index_deinitialize(&sink->row_index,
                                                 host_allocator);
  iree_allocator_free(host_allocator, sink->exports);
  iree_allocator_free(host_allocator, sink->rows);
  iree_allocator_free(host_allocator, sink->devices);
  sink->rows = NULL;
  sink->row_count = 0;
  sink->row_capacity = 0;
  sink->exports = NULL;
  sink->export_count = 0;
  sink->export_capacity = 0;
  sink->devices = NULL;
  sink->device_count = 0;
  sink->device_capacity = 0;
  sink->dropped_record_count = 0;
}

IREE_API_EXPORT iree_status_t iree_hal_profile_statistics_sink_create(
    iree_allocator_t host_allocator,
    iree_hal_profile_statistics_sink_t** out_sink) {
  IREE_ASSERT_ARGUMENT(out_sink);
  *out_sink = NULL;

  iree_hal_profile_statistics_sink_t* sink = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*sink), (void**)&sink));
  memset(sink, 0, sizeof(*sink));
  iree_hal_resource_initialize(&iree_hal_profile_statistics_sink_vtable,
                               &sink->resource);
  sink->host_allocator = host_allocator;
  *out_sink = sink;
  return iree_ok_status();
}

IREE_API_EXPORT iree_hal_profile_sink_t* iree_hal_profile_statistics_sink_base(
    iree_hal_profile_statistics_sink_t* sink) {
  IREE_ASSERT_ARGUMENT(sink);
  return (iree_hal_profile_sink_t*)sink;
}

IREE_API_EXPORT void iree_hal_profile_statistics_sink_retain(
    iree_hal_profile_statistics_sink_t* sink) {
  iree_hal_profile_sink_retain((iree_hal_profile_sink_t*)sink);
}

IREE_API_EXPORT void iree_hal_profile_statistics_sink_release(
    iree_hal_profile_statistics_sink_t* sink) {
  iree_hal_profile_sink_release((iree_hal_profile_sink_t*)sink);
}

IREE_API_EXPORT iree_host_size_t iree_hal_profile_statistics_sink_row_count(
    const iree_hal_profile_statistics_sink_t* sink) {
  IREE_ASSERT_ARGUMENT(sink);
  return sink->row_count;
}

IREE_API_EXPORT uint64_t iree_hal_profile_statistics_sink_dropped_record_count(
    const iree_hal_profile_statistics_sink_t* sink) {
  IREE_ASSERT_ARGUMENT(sink);
  return sink->dropped_record_count;
}

IREE_API_EXPORT iree_status_t iree_hal_profile_statistics_sink_for_each_row(
    const iree_hal_profile_statistics_sink_t* sink,
    iree_hal_profile_statistics_row_callback_t callback) {
  IREE_ASSERT_ARGUMENT(sink);
  IREE_ASSERT_ARGUMENT(callback.fn);
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; iree_status_is_ok(status) && i < sink->row_count;
       ++i) {
    status = callback.fn(callback.user_data, &sink->rows[i]);
  }
  return status;
}

IREE_API_EXPORT bool iree_hal_profile_statistics_sink_find_export_name(
    const iree_hal_profile_statistics_sink_t* sink, uint64_t executable_id,
    uint32_t export_ordinal, iree_string_view_t* out_name) {
  IREE_ASSERT_ARGUMENT(sink);
  IREE_ASSERT_ARGUMENT(out_name);
  *out_name = iree_string_view_empty();

  const uint64_t hash =
      iree_hal_profile_statistics_export_hash(executable_id, export_ordinal);
  const iree_hal_profile_statistics_export_lookup_t lookup = {
      .sink = sink,
      .executable_id = executable_id,
      .export_ordinal = export_ordinal,
  };
  iree_host_size_t export_index = 0;
  if (!iree_hal_profile_statistics_index_find(
          &sink->export_index, hash, iree_hal_profile_statistics_export_matches,
          &lookup, &export_index)) {
    return false;
  }
  *out_name = sink->exports[export_index].name;
  return !iree_string_view_is_empty(*out_name);
}

IREE_API_EXPORT bool iree_hal_profile_statistics_sink_scale_duration_to_ns(
    const iree_hal_profile_statistics_sink_t* sink,
    const iree_hal_profile_statistics_row_t* row, uint64_t duration,
    uint64_t* out_duration_ns) {
  IREE_ASSERT_ARGUMENT(sink);
  IREE_ASSERT_ARGUMENT(row);
  IREE_ASSERT_ARGUMENT(out_duration_ns);
  *out_duration_ns = 0;

  switch (row->time_domain) {
    case IREE_HAL_PROFILE_STATISTICS_TIME_DOMAIN_IREE_HOST_TIME_NS:
      *out_duration_ns = duration;
      return true;
    case IREE_HAL_PROFILE_STATISTICS_TIME_DOMAIN_DEVICE_TICK: {
      const iree_hal_profile_statistics_device_t* device =
          iree_hal_profile_statistics_sink_find_device(
              sink, row->physical_device_ordinal);
      return iree_hal_profile_statistics_scale_device_ticks_to_ns(
          device, duration, out_duration_ns);
    }
    default:
      return false;
  }
}

typedef struct iree_hal_profile_statistics_print_context_t {
  // Statistics sink being printed.
  const iree_hal_profile_statistics_sink_t* sink;
  // Output stream receiving human-readable statistics.
  FILE* file;
  // Total scaled dispatch duration across executable export rows.
  uint64_t dispatch_export_duration_ns;
  // Total scaled host execution duration across executable export rows.
  uint64_t host_execution_export_duration_ns;
  // Total scaled host execution duration across queue operation rows.
  uint64_t host_execution_queue_operation_duration_ns;
  // Total scaled device-queue operation duration.
  uint64_t queue_device_operation_duration_ns;
  // Total scaled host queue ready latency.
  uint64_t queue_host_operation_duration_ns;
  // Number of rows with timing values that could not be scaled to nanoseconds.
  uint64_t unscaled_timing_row_count;
} iree_hal_profile_statistics_print_context_t;

static const char* iree_hal_profile_statistics_queue_event_type_name(
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

static const char* iree_hal_profile_statistics_memory_event_type_name(
    iree_hal_profile_memory_event_type_t type) {
  switch (type) {
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_SLAB_ACQUIRE:
      return "slab_acquire";
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_SLAB_RELEASE:
      return "slab_release";
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_RESERVE:
      return "pool_reserve";
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_MATERIALIZE:
      return "pool_materialize";
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_RELEASE:
      return "pool_release";
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_WAIT:
      return "pool_wait";
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_ALLOCA:
      return "queue_alloca";
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_DEALLOCA:
      return "queue_dealloca";
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_BUFFER_ALLOCATE:
      return "buffer_allocate";
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_BUFFER_FREE:
      return "buffer_free";
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_BUFFER_IMPORT:
      return "buffer_import";
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_BUFFER_UNIMPORT:
      return "buffer_unimport";
    default:
      return "unknown";
  }
}

static bool iree_hal_profile_statistics_row_duration_ns(
    const iree_hal_profile_statistics_print_context_t* context,
    const iree_hal_profile_statistics_row_t* row, uint64_t duration,
    uint64_t* out_duration_ns) {
  if (!iree_all_bits_set(row->flags,
                         IREE_HAL_PROFILE_STATISTICS_ROW_FLAG_TIMING)) {
    *out_duration_ns = 0;
    return false;
  }
  return iree_hal_profile_statistics_sink_scale_duration_to_ns(
      context->sink, row, duration, out_duration_ns);
}

static void iree_hal_profile_statistics_fprint_scaled_duration(FILE* file,
                                                               uint64_t ns) {
  if (ns >= 1000000) {
    fprintf(file, "%" PRIu64 ".%03" PRIu64 " ms", ns / 1000000,
            (ns % 1000000) / 1000);
  } else if (ns >= 1000) {
    fprintf(file, "%" PRIu64 ".%03" PRIu64 " us", ns / 1000, ns % 1000);
  } else {
    fprintf(file, "%" PRIu64 " ns", ns);
  }
}

static void iree_hal_profile_statistics_fprint_duration(
    const iree_hal_profile_statistics_print_context_t* context,
    const iree_hal_profile_statistics_row_t* row, uint64_t duration) {
  if (!iree_all_bits_set(row->flags,
                         IREE_HAL_PROFILE_STATISTICS_ROW_FLAG_TIMING)) {
    fprintf(context->file, "-");
    return;
  }

  uint64_t duration_ns = 0;
  if (iree_hal_profile_statistics_row_duration_ns(context, row, duration,
                                                  &duration_ns)) {
    iree_hal_profile_statistics_fprint_scaled_duration(context->file,
                                                       duration_ns);
  } else if (row->time_domain ==
             IREE_HAL_PROFILE_STATISTICS_TIME_DOMAIN_DEVICE_TICK) {
    fprintf(context->file, "%" PRIu64 " ticks", duration);
  } else {
    fprintf(context->file, "-");
  }
}

static iree_status_t iree_hal_profile_statistics_accumulate_row(
    void* user_data, const iree_hal_profile_statistics_row_t* row) {
  iree_hal_profile_statistics_print_context_t* context =
      (iree_hal_profile_statistics_print_context_t*)user_data;

  uint64_t total_duration_ns = 0;
  if (iree_all_bits_set(row->flags,
                        IREE_HAL_PROFILE_STATISTICS_ROW_FLAG_TIMING) &&
      !iree_hal_profile_statistics_row_duration_ns(
          context, row, row->total_duration, &total_duration_ns)) {
    ++context->unscaled_timing_row_count;
  }

  switch (row->row_type) {
    case IREE_HAL_PROFILE_STATISTICS_ROW_TYPE_DISPATCH_EXPORT:
      if (!iree_hal_profile_statistics_add_u64(
              &context->dispatch_export_duration_ns, total_duration_ns)) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "profile statistics dispatch duration overflow");
      }
      break;
    case IREE_HAL_PROFILE_STATISTICS_ROW_TYPE_HOST_EXECUTION_EXPORT:
      if (!iree_hal_profile_statistics_add_u64(
              &context->host_execution_export_duration_ns, total_duration_ns)) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "profile statistics host execution duration overflow");
      }
      break;
    case IREE_HAL_PROFILE_STATISTICS_ROW_TYPE_HOST_EXECUTION_QUEUE_OPERATION:
      if (!iree_hal_profile_statistics_add_u64(
              &context->host_execution_queue_operation_duration_ns,
              total_duration_ns)) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "profile statistics host execution queue duration overflow");
      }
      break;
    case IREE_HAL_PROFILE_STATISTICS_ROW_TYPE_QUEUE_DEVICE_OPERATION:
      if (!iree_hal_profile_statistics_add_u64(
              &context->queue_device_operation_duration_ns,
              total_duration_ns)) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "profile statistics device queue duration overflow");
      }
      break;
    case IREE_HAL_PROFILE_STATISTICS_ROW_TYPE_QUEUE_HOST_OPERATION:
      if (!iree_hal_profile_statistics_add_u64(
              &context->queue_host_operation_duration_ns, total_duration_ns)) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "profile statistics host queue duration overflow");
      }
      break;
    default:
      break;
  }
  return iree_ok_status();
}

static void iree_hal_profile_statistics_fprint_row_timing(
    const iree_hal_profile_statistics_print_context_t* context,
    const iree_hal_profile_statistics_row_t* row) {
  fprintf(context->file, "  count=%" PRIu64 " total=", row->sample_count);
  iree_hal_profile_statistics_fprint_duration(context, row,
                                              row->total_duration);
  fprintf(context->file, " avg=");
  if (row->sample_count != 0) {
    iree_hal_profile_statistics_fprint_duration(
        context, row, row->total_duration / row->sample_count);
  } else {
    fprintf(context->file, "-");
  }
  if (row->invalid_sample_count != 0) {
    fprintf(context->file, " invalid=%" PRIu64, row->invalid_sample_count);
  }
}

static void iree_hal_profile_statistics_fprint_tile_totals(
    const iree_hal_profile_statistics_print_context_t* context,
    const iree_hal_profile_statistics_row_t* row) {
  if (row->tile_count == 0) return;
  fprintf(context->file, " tiles=%" PRIu64 " tile_sum=", row->tile_count);
  iree_hal_profile_statistics_fprint_scaled_duration(context->file,
                                                     row->tile_duration_sum_ns);
}

static void iree_hal_profile_statistics_fprint_operation_totals(
    const iree_hal_profile_statistics_print_context_t* context,
    const iree_hal_profile_statistics_row_t* row) {
  if (row->operation_count != 0) {
    fprintf(context->file, " operations=%" PRIu64, row->operation_count);
  }
  iree_hal_profile_statistics_fprint_tile_totals(context, row);
  if (row->payload_bytes != 0) {
    fprintf(context->file, " payload=%" PRIu64 "B", row->payload_bytes);
  }
}

static void iree_hal_profile_statistics_fprint_export_key(
    const iree_hal_profile_statistics_print_context_t* context,
    const iree_hal_profile_statistics_row_t* row) {
  iree_string_view_t export_name = iree_string_view_empty();
  if (iree_hal_profile_statistics_sink_find_export_name(
          context->sink, row->executable_id, row->export_ordinal,
          &export_name)) {
    fprintf(context->file, "%.*s", (int)export_name.size, export_name.data);
  } else {
    fprintf(context->file, "executable=%" PRIu64 " export=%" PRIu32,
            row->executable_id, row->export_ordinal);
  }
}

static iree_status_t iree_hal_profile_statistics_print_row(
    void* user_data, const iree_hal_profile_statistics_row_t* row) {
  const iree_hal_profile_statistics_print_context_t* context =
      (const iree_hal_profile_statistics_print_context_t*)user_data;
  FILE* file = context->file;
  switch (row->row_type) {
    case IREE_HAL_PROFILE_STATISTICS_ROW_TYPE_DISPATCH_EXPORT:
      fprintf(file, "  dispatch ");
      iree_hal_profile_statistics_fprint_export_key(context, row);
      iree_hal_profile_statistics_fprint_row_timing(context, row);
      fprintf(file, "\n");
      break;
    case IREE_HAL_PROFILE_STATISTICS_ROW_TYPE_HOST_EXECUTION_EXPORT:
      fprintf(file, "  host_execute ");
      iree_hal_profile_statistics_fprint_export_key(context, row);
      iree_hal_profile_statistics_fprint_row_timing(context, row);
      iree_hal_profile_statistics_fprint_tile_totals(context, row);
      fprintf(file, "\n");
      break;
    case IREE_HAL_PROFILE_STATISTICS_ROW_TYPE_HOST_EXECUTION_QUEUE_OPERATION:
      fprintf(file, "  host_execute_queue p=%" PRIu32 " q=%" PRIu32 " %s",
              row->physical_device_ordinal, row->queue_ordinal,
              iree_hal_profile_statistics_queue_event_type_name(
                  (iree_hal_profile_queue_event_type_t)row->event_type));
      iree_hal_profile_statistics_fprint_row_timing(context, row);
      iree_hal_profile_statistics_fprint_operation_totals(context, row);
      fprintf(file, "\n");
      break;
    case IREE_HAL_PROFILE_STATISTICS_ROW_TYPE_QUEUE_DEVICE_OPERATION:
      fprintf(file, "  device_queue p=%" PRIu32 " q=%" PRIu32 " %s",
              row->physical_device_ordinal, row->queue_ordinal,
              iree_hal_profile_statistics_queue_event_type_name(
                  (iree_hal_profile_queue_event_type_t)row->event_type));
      iree_hal_profile_statistics_fprint_row_timing(context, row);
      iree_hal_profile_statistics_fprint_operation_totals(context, row);
      fprintf(file, "\n");
      break;
    case IREE_HAL_PROFILE_STATISTICS_ROW_TYPE_QUEUE_HOST_OPERATION:
      fprintf(file, "  host_queue p=%" PRIu32 " q=%" PRIu32 " %s",
              row->physical_device_ordinal, row->queue_ordinal,
              iree_hal_profile_statistics_queue_event_type_name(
                  (iree_hal_profile_queue_event_type_t)row->event_type));
      iree_hal_profile_statistics_fprint_row_timing(context, row);
      iree_hal_profile_statistics_fprint_operation_totals(context, row);
      fprintf(file, "\n");
      break;
    case IREE_HAL_PROFILE_STATISTICS_ROW_TYPE_MEMORY_LIFECYCLE:
      fprintf(file,
              "  memory p=%" PRIu32 " q=%" PRIu32 " %s count=%" PRIu64
              " bytes=%" PRIu64 "\n",
              row->physical_device_ordinal, row->queue_ordinal,
              iree_hal_profile_statistics_memory_event_type_name(
                  (iree_hal_profile_memory_event_type_t)row->event_type),
              row->sample_count, row->payload_bytes);
      break;
    default:
      break;
  }
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_hal_profile_statistics_sink_fprint(
    FILE* file, const iree_hal_profile_statistics_sink_t* sink) {
  IREE_ASSERT_ARGUMENT(file);
  IREE_ASSERT_ARGUMENT(sink);

  iree_hal_profile_statistics_print_context_t context = {
      .sink = sink,
      .file = file,
  };
  iree_hal_profile_statistics_row_callback_t accumulate_callback = {
      .fn = iree_hal_profile_statistics_accumulate_row,
      .user_data = &context,
  };
  IREE_RETURN_IF_ERROR(
      iree_hal_profile_statistics_sink_for_each_row(sink, accumulate_callback));

  fprintf(file, "\nIREE HAL device statistics:\n");
  const iree_host_size_t row_count =
      iree_hal_profile_statistics_sink_row_count(sink);
  const uint64_t dropped_record_count =
      iree_hal_profile_statistics_sink_dropped_record_count(sink);
  if (row_count == 0 && dropped_record_count == 0) {
    fprintf(file, "  no profiling records captured\n");
    return iree_ok_status();
  }
  fprintf(file, "  aggregate_rows=%" PRIhsz "\n", row_count);
  if (dropped_record_count != 0) {
    fprintf(file, "  dropped_records=%" PRIu64 "\n", dropped_record_count);
  }
  if (row_count == 0) return iree_ok_status();
  fprintf(file, "  dispatch_export_total=");
  iree_hal_profile_statistics_fprint_scaled_duration(
      file, context.dispatch_export_duration_ns);
  fprintf(file, "\n  host_execution_export_total=");
  iree_hal_profile_statistics_fprint_scaled_duration(
      file, context.host_execution_export_duration_ns);
  fprintf(file, "\n  host_execution_queue_total=");
  iree_hal_profile_statistics_fprint_scaled_duration(
      file, context.host_execution_queue_operation_duration_ns);
  fprintf(file, "\n  device_queue_total=");
  iree_hal_profile_statistics_fprint_scaled_duration(
      file, context.queue_device_operation_duration_ns);
  fprintf(file, "\n  host_queue_ready_total=");
  iree_hal_profile_statistics_fprint_scaled_duration(
      file, context.queue_host_operation_duration_ns);
  if (context.unscaled_timing_row_count != 0) {
    fprintf(file, "\n  unscaled_timing_rows=%" PRIu64,
            context.unscaled_timing_row_count);
  }
  fprintf(file, "\n");

  iree_hal_profile_statistics_row_callback_t print_callback = {
      .fn = iree_hal_profile_statistics_print_row,
      .user_data = &context,
  };
  return iree_hal_profile_statistics_sink_for_each_row(sink, print_callback);
}

static void iree_hal_profile_statistics_sink_destroy(
    iree_hal_profile_sink_t* base_sink) {
  iree_hal_profile_statistics_sink_t* sink =
      iree_hal_profile_statistics_sink_cast(base_sink);
  iree_allocator_t host_allocator = sink->host_allocator;
  iree_hal_profile_statistics_sink_reset(sink);
  iree_allocator_free(host_allocator, sink);
}

static iree_status_t iree_hal_profile_statistics_sink_begin_session(
    iree_hal_profile_sink_t* base_sink,
    const iree_hal_profile_chunk_metadata_t* metadata) {
  (void)metadata;
  iree_hal_profile_statistics_sink_t* sink =
      iree_hal_profile_statistics_sink_cast(base_sink);
  iree_hal_profile_statistics_sink_reset(sink);
  return iree_ok_status();
}

static iree_status_t iree_hal_profile_statistics_sink_write(
    iree_hal_profile_sink_t* base_sink,
    const iree_hal_profile_chunk_metadata_t* metadata,
    iree_host_size_t iovec_count, const iree_const_byte_span_t* iovecs) {
  iree_hal_profile_statistics_sink_t* sink =
      iree_hal_profile_statistics_sink_cast(base_sink);
  IREE_RETURN_IF_ERROR(
      iree_hal_profile_statistics_sink_add_dropped_records(sink, metadata));
  if (iree_string_view_equal(metadata->content_type,
                             IREE_HAL_PROFILE_CONTENT_TYPE_DISPATCH_EVENTS)) {
    return iree_hal_profile_statistics_for_each_fixed_record(
        sink, metadata, iovec_count, iovecs,
        sizeof(iree_hal_profile_dispatch_event_t),
        iree_hal_profile_statistics_process_dispatch_event);
  } else if (iree_string_view_equal(
                 metadata->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_CLOCK_CORRELATIONS)) {
    return iree_hal_profile_statistics_for_each_fixed_record(
        sink, metadata, iovec_count, iovecs,
        sizeof(iree_hal_profile_clock_correlation_record_t),
        iree_hal_profile_statistics_process_clock_correlation);
  } else if (iree_string_view_equal(
                 metadata->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_QUEUE_DEVICE_EVENTS)) {
    return iree_hal_profile_statistics_for_each_fixed_record(
        sink, metadata, iovec_count, iovecs,
        sizeof(iree_hal_profile_queue_device_event_t),
        iree_hal_profile_statistics_process_queue_device_event);
  } else if (iree_string_view_equal(
                 metadata->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_QUEUE_EVENTS)) {
    return iree_hal_profile_statistics_for_each_fixed_record(
        sink, metadata, iovec_count, iovecs,
        sizeof(iree_hal_profile_queue_event_t),
        iree_hal_profile_statistics_process_queue_event);
  } else if (iree_string_view_equal(
                 metadata->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_HOST_EXECUTION_EVENTS)) {
    return iree_hal_profile_statistics_for_each_fixed_record(
        sink, metadata, iovec_count, iovecs,
        sizeof(iree_hal_profile_host_execution_event_t),
        iree_hal_profile_statistics_process_host_execution_event);
  } else if (iree_string_view_equal(
                 metadata->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_MEMORY_EVENTS)) {
    return iree_hal_profile_statistics_for_each_fixed_record(
        sink, metadata, iovec_count, iovecs,
        sizeof(iree_hal_profile_memory_event_t),
        iree_hal_profile_statistics_process_memory_event);
  } else if (iree_string_view_equal(
                 metadata->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLE_EXPORTS)) {
    return iree_hal_profile_statistics_process_export_records(sink, iovec_count,
                                                              iovecs);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_profile_statistics_sink_end_session(
    iree_hal_profile_sink_t* base_sink,
    const iree_hal_profile_chunk_metadata_t* metadata,
    iree_status_code_t session_status_code) {
  (void)base_sink;
  (void)metadata;
  (void)session_status_code;
  return iree_ok_status();
}

static const iree_hal_profile_sink_vtable_t
    iree_hal_profile_statistics_sink_vtable = {
        .destroy = iree_hal_profile_statistics_sink_destroy,
        .begin_session = iree_hal_profile_statistics_sink_begin_session,
        .write = iree_hal_profile_statistics_sink_write,
        .end_session = iree_hal_profile_statistics_sink_end_session,
};
