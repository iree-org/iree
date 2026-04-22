// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tooling/profile/model.h"

#include <string.h>

#include "iree/tooling/profile/reader.h"

void iree_profile_model_initialize(iree_allocator_t host_allocator,
                                   iree_profile_model_t* out_model) {
  memset(out_model, 0, sizeof(*out_model));
  out_model->host_allocator = host_allocator;
}

void iree_profile_model_deinitialize(iree_profile_model_t* model) {
  iree_profile_index_deinitialize(&model->device_index, model->host_allocator);
  iree_profile_index_deinitialize(&model->metric_descriptor_index,
                                  model->host_allocator);
  iree_profile_index_deinitialize(&model->metric_source_index,
                                  model->host_allocator);
  iree_profile_index_deinitialize(&model->queue_index, model->host_allocator);
  iree_profile_index_deinitialize(&model->command_buffer_index,
                                  model->host_allocator);
  iree_profile_index_deinitialize(&model->export_index, model->host_allocator);
  iree_profile_index_deinitialize(&model->executable_index,
                                  model->host_allocator);
  iree_allocator_free(model->host_allocator, model->devices);
  iree_allocator_free(model->host_allocator, model->metric_descriptors);
  iree_allocator_free(model->host_allocator, model->metric_sources);
  iree_allocator_free(model->host_allocator, model->queues);
  iree_allocator_free(model->host_allocator, model->command_operations);
  iree_allocator_free(model->host_allocator, model->command_buffers);
  iree_allocator_free(model->host_allocator, model->exports);
  iree_allocator_free(model->host_allocator, model->executables);
  memset(model, 0, sizeof(*model));
}

typedef struct iree_profile_model_device_lookup_t {
  // Model owning the candidate rows.
  const iree_profile_model_t* model;
  // Session-local physical device ordinal.
  uint32_t physical_device_ordinal;
} iree_profile_model_device_lookup_t;

typedef struct iree_profile_model_queue_lookup_t {
  // Model owning the candidate rows.
  const iree_profile_model_t* model;
  // Session-local physical device ordinal.
  uint32_t physical_device_ordinal;
  // Session-local queue ordinal.
  uint32_t queue_ordinal;
  // Producer-defined queue stream identifier.
  uint64_t stream_id;
} iree_profile_model_queue_lookup_t;

typedef struct iree_profile_model_executable_lookup_t {
  // Model owning the candidate rows.
  const iree_profile_model_t* model;
  // Producer-local executable identifier.
  uint64_t executable_id;
} iree_profile_model_executable_lookup_t;

typedef struct iree_profile_model_export_lookup_t {
  // Model owning the candidate rows.
  const iree_profile_model_t* model;
  // Producer-local executable identifier.
  uint64_t executable_id;
  // Export ordinal within |executable_id|.
  uint32_t export_ordinal;
} iree_profile_model_export_lookup_t;

typedef struct iree_profile_model_command_buffer_lookup_t {
  // Model owning the candidate rows.
  const iree_profile_model_t* model;
  // Producer-local command-buffer identifier.
  uint64_t command_buffer_id;
} iree_profile_model_command_buffer_lookup_t;

typedef struct iree_profile_model_metric_source_lookup_t {
  // Model owning the candidate rows.
  const iree_profile_model_t* model;
  // Producer-defined metric source identifier.
  uint64_t source_id;
} iree_profile_model_metric_source_lookup_t;

typedef struct iree_profile_model_metric_descriptor_lookup_t {
  // Model owning the candidate rows.
  const iree_profile_model_t* model;
  // Producer-defined metric source identifier.
  uint64_t source_id;
  // Source-scoped metric identifier.
  uint64_t metric_id;
} iree_profile_model_metric_descriptor_lookup_t;

static uint64_t iree_profile_model_device_hash(
    uint32_t physical_device_ordinal) {
  return iree_profile_index_mix_u64(physical_device_ordinal);
}

static uint64_t iree_profile_model_queue_hash(uint32_t physical_device_ordinal,
                                              uint32_t queue_ordinal,
                                              uint64_t stream_id) {
  uint64_t hash = iree_profile_model_device_hash(physical_device_ordinal);
  hash = iree_profile_index_combine_u64(hash, queue_ordinal);
  return iree_profile_index_combine_u64(hash, stream_id);
}

static uint64_t iree_profile_model_executable_hash(uint64_t executable_id) {
  return iree_profile_index_mix_u64(executable_id);
}

static uint64_t iree_profile_model_export_hash(uint64_t executable_id,
                                               uint32_t export_ordinal) {
  uint64_t hash = iree_profile_model_executable_hash(executable_id);
  return iree_profile_index_combine_u64(hash, export_ordinal);
}

static uint64_t iree_profile_model_command_buffer_hash(
    uint64_t command_buffer_id) {
  return iree_profile_index_mix_u64(command_buffer_id);
}

static uint64_t iree_profile_model_metric_source_hash(uint64_t source_id) {
  return iree_profile_index_mix_u64(source_id);
}

static uint64_t iree_profile_model_metric_descriptor_hash(uint64_t source_id,
                                                          uint64_t metric_id) {
  uint64_t hash = iree_profile_model_metric_source_hash(source_id);
  return iree_profile_index_combine_u64(hash, metric_id);
}

static bool iree_profile_model_device_matches(const void* user_data,
                                              iree_host_size_t value) {
  const iree_profile_model_device_lookup_t* lookup =
      (const iree_profile_model_device_lookup_t*)user_data;
  return lookup->model->devices[value].physical_device_ordinal ==
         lookup->physical_device_ordinal;
}

static bool iree_profile_model_queue_matches(const void* user_data,
                                             iree_host_size_t value) {
  const iree_profile_model_queue_lookup_t* lookup =
      (const iree_profile_model_queue_lookup_t*)user_data;
  const iree_hal_profile_queue_record_t* record =
      &lookup->model->queues[value].record;
  return record->physical_device_ordinal == lookup->physical_device_ordinal &&
         record->queue_ordinal == lookup->queue_ordinal &&
         record->stream_id == lookup->stream_id;
}

static bool iree_profile_model_executable_matches(const void* user_data,
                                                  iree_host_size_t value) {
  const iree_profile_model_executable_lookup_t* lookup =
      (const iree_profile_model_executable_lookup_t*)user_data;
  return lookup->model->executables[value].record.executable_id ==
         lookup->executable_id;
}

static bool iree_profile_model_export_matches(const void* user_data,
                                              iree_host_size_t value) {
  const iree_profile_model_export_lookup_t* lookup =
      (const iree_profile_model_export_lookup_t*)user_data;
  const iree_profile_model_export_t* export_info =
      &lookup->model->exports[value];
  return export_info->executable_id == lookup->executable_id &&
         export_info->export_ordinal == lookup->export_ordinal;
}

static bool iree_profile_model_command_buffer_matches(const void* user_data,
                                                      iree_host_size_t value) {
  const iree_profile_model_command_buffer_lookup_t* lookup =
      (const iree_profile_model_command_buffer_lookup_t*)user_data;
  return lookup->model->command_buffers[value].record.command_buffer_id ==
         lookup->command_buffer_id;
}

static bool iree_profile_model_metric_source_matches(const void* user_data,
                                                     iree_host_size_t value) {
  const iree_profile_model_metric_source_lookup_t* lookup =
      (const iree_profile_model_metric_source_lookup_t*)user_data;
  return lookup->model->metric_sources[value].record.source_id ==
         lookup->source_id;
}

static bool iree_profile_model_metric_descriptor_matches(
    const void* user_data, iree_host_size_t value) {
  const iree_profile_model_metric_descriptor_lookup_t* lookup =
      (const iree_profile_model_metric_descriptor_lookup_t*)user_data;
  const iree_hal_profile_device_metric_descriptor_record_t* record =
      &lookup->model->metric_descriptors[value].record;
  return record->source_id == lookup->source_id &&
         record->metric_id == lookup->metric_id;
}

iree_status_t iree_profile_model_ensure_device(
    iree_profile_model_t* model, uint32_t physical_device_ordinal,
    iree_profile_model_device_t** out_device) {
  *out_device = NULL;

  const iree_profile_model_device_lookup_t lookup = {
      .model = model,
      .physical_device_ordinal = physical_device_ordinal,
  };
  const uint64_t hash = iree_profile_model_device_hash(physical_device_ordinal);
  iree_host_size_t existing_index = 0;
  if (iree_profile_index_find(&model->device_index, hash,
                              iree_profile_model_device_matches, &lookup,
                              &existing_index)) {
    *out_device = &model->devices[existing_index];
    return iree_ok_status();
  }

  if (model->device_count + 1 > model->device_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        model->host_allocator,
        iree_max((iree_host_size_t)4, model->device_count + 1),
        sizeof(model->devices[0]), &model->device_capacity,
        (void**)&model->devices));
  }

  const iree_host_size_t device_index = model->device_count;
  IREE_RETURN_IF_ERROR(iree_profile_index_insert(
      &model->device_index, model->host_allocator, hash, device_index));
  ++model->device_count;

  iree_profile_model_device_t* device = &model->devices[device_index];
  memset(device, 0, sizeof(*device));
  device->physical_device_ordinal = physical_device_ordinal;
  *out_device = device;
  return iree_ok_status();
}

static void iree_profile_model_record_clock_sample(
    iree_profile_model_device_t* device,
    const iree_hal_profile_clock_correlation_record_t* record) {
  if (device->clock_sample_count == 0) {
    device->first_clock_sample = *record;
  }
  device->last_clock_sample = *record;
  ++device->clock_sample_count;
}

const iree_profile_model_device_t* iree_profile_model_find_device(
    const iree_profile_model_t* model, uint32_t physical_device_ordinal) {
  if (!model) return NULL;
  const iree_profile_model_device_lookup_t lookup = {
      .model = model,
      .physical_device_ordinal = physical_device_ordinal,
  };
  iree_host_size_t index = 0;
  if (iree_profile_index_find(
          &model->device_index,
          iree_profile_model_device_hash(physical_device_ordinal),
          iree_profile_model_device_matches, &lookup, &index)) {
    return &model->devices[index];
  }
  return NULL;
}

const iree_profile_model_queue_t* iree_profile_model_find_queue(
    const iree_profile_model_t* model, uint32_t physical_device_ordinal,
    uint32_t queue_ordinal, uint64_t stream_id) {
  if (!model) return NULL;
  const iree_profile_model_queue_lookup_t lookup = {
      .model = model,
      .physical_device_ordinal = physical_device_ordinal,
      .queue_ordinal = queue_ordinal,
      .stream_id = stream_id,
  };
  iree_host_size_t index = 0;
  if (iree_profile_index_find(
          &model->queue_index,
          iree_profile_model_queue_hash(physical_device_ordinal, queue_ordinal,
                                        stream_id),
          iree_profile_model_queue_matches, &lookup, &index)) {
    return &model->queues[index];
  }
  return NULL;
}

const iree_profile_model_executable_t* iree_profile_model_find_executable(
    const iree_profile_model_t* model, uint64_t executable_id) {
  if (!model) return NULL;
  const iree_profile_model_executable_lookup_t lookup = {
      .model = model,
      .executable_id = executable_id,
  };
  iree_host_size_t index = 0;
  if (iree_profile_index_find(&model->executable_index,
                              iree_profile_model_executable_hash(executable_id),
                              iree_profile_model_executable_matches, &lookup,
                              &index)) {
    return &model->executables[index];
  }
  return NULL;
}

static iree_profile_model_executable_t*
iree_profile_model_find_executable_mutable(iree_profile_model_t* model,
                                           uint64_t executable_id) {
  if (!model) return NULL;
  const iree_profile_model_executable_lookup_t lookup = {
      .model = model,
      .executable_id = executable_id,
  };
  iree_host_size_t index = 0;
  if (iree_profile_index_find(&model->executable_index,
                              iree_profile_model_executable_hash(executable_id),
                              iree_profile_model_executable_matches, &lookup,
                              &index)) {
    return &model->executables[index];
  }
  return NULL;
}

const iree_profile_model_export_t* iree_profile_model_find_export(
    const iree_profile_model_t* model, uint64_t executable_id,
    uint32_t export_ordinal) {
  if (!model) return NULL;
  const iree_profile_model_export_lookup_t lookup = {
      .model = model,
      .executable_id = executable_id,
      .export_ordinal = export_ordinal,
  };
  iree_host_size_t index = 0;
  if (iree_profile_index_find(
          &model->export_index,
          iree_profile_model_export_hash(executable_id, export_ordinal),
          iree_profile_model_export_matches, &lookup, &index)) {
    return &model->exports[index];
  }
  return NULL;
}

const iree_profile_model_command_buffer_t*
iree_profile_model_find_command_buffer(const iree_profile_model_t* model,
                                       uint64_t command_buffer_id) {
  if (!model) return NULL;
  const iree_profile_model_command_buffer_lookup_t lookup = {
      .model = model,
      .command_buffer_id = command_buffer_id,
  };
  iree_host_size_t index = 0;
  if (iree_profile_index_find(
          &model->command_buffer_index,
          iree_profile_model_command_buffer_hash(command_buffer_id),
          iree_profile_model_command_buffer_matches, &lookup, &index)) {
    return &model->command_buffers[index];
  }
  return NULL;
}

const iree_profile_model_metric_source_t* iree_profile_model_find_metric_source(
    const iree_profile_model_t* model, uint64_t source_id) {
  if (!model) return NULL;
  const iree_profile_model_metric_source_lookup_t lookup = {
      .model = model,
      .source_id = source_id,
  };
  iree_host_size_t index = 0;
  if (iree_profile_index_find(&model->metric_source_index,
                              iree_profile_model_metric_source_hash(source_id),
                              iree_profile_model_metric_source_matches, &lookup,
                              &index)) {
    return &model->metric_sources[index];
  }
  return NULL;
}

const iree_profile_model_metric_descriptor_t*
iree_profile_model_find_metric_descriptor(const iree_profile_model_t* model,
                                          uint64_t source_id,
                                          uint64_t metric_id) {
  if (!model) return NULL;
  const iree_profile_model_metric_descriptor_lookup_t lookup = {
      .model = model,
      .source_id = source_id,
      .metric_id = metric_id,
  };
  iree_host_size_t index = 0;
  if (iree_profile_index_find(
          &model->metric_descriptor_index,
          iree_profile_model_metric_descriptor_hash(source_id, metric_id),
          iree_profile_model_metric_descriptor_matches, &lookup, &index)) {
    return &model->metric_descriptors[index];
  }
  return NULL;
}

iree_status_t iree_profile_model_resolve_metric_descriptor(
    const iree_profile_model_t* model, uint64_t source_id, uint64_t metric_id,
    const iree_profile_model_metric_descriptor_t** out_descriptor) {
  *out_descriptor =
      iree_profile_model_find_metric_descriptor(model, source_id, metric_id);
  if (!*out_descriptor) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "device metric sample references missing descriptor source=%" PRIu64
        " metric=%" PRIu64,
        source_id, metric_id);
  }
  return iree_ok_status();
}

static iree_profile_model_command_buffer_t*
iree_profile_model_find_command_buffer_mutable(iree_profile_model_t* model,
                                               uint64_t command_buffer_id) {
  if (!model) return NULL;
  const iree_profile_model_command_buffer_lookup_t lookup = {
      .model = model,
      .command_buffer_id = command_buffer_id,
  };
  iree_host_size_t index = 0;
  if (iree_profile_index_find(
          &model->command_buffer_index,
          iree_profile_model_command_buffer_hash(command_buffer_id),
          iree_profile_model_command_buffer_matches, &lookup, &index)) {
    return &model->command_buffers[index];
  }
  return NULL;
}

static bool iree_profile_model_host_time_midpoint(
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

static bool iree_profile_model_sample_time_ns(
    const iree_hal_profile_clock_correlation_record_t* sample,
    iree_profile_model_clock_time_domain_t time_domain, int64_t* out_time_ns) {
  switch (time_domain) {
    case IREE_PROFILE_MODEL_CLOCK_TIME_DOMAIN_HOST_CPU_TIMESTAMP_NS:
      if (!iree_all_bits_set(
              sample->flags,
              IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_HOST_CPU_TIMESTAMP) ||
          sample->host_cpu_timestamp_ns > INT64_MAX) {
        return false;
      }
      *out_time_ns = (int64_t)sample->host_cpu_timestamp_ns;
      return true;
    case IREE_PROFILE_MODEL_CLOCK_TIME_DOMAIN_IREE_HOST_TIME_NS:
      return iree_profile_model_host_time_midpoint(sample, out_time_ns);
    default:
      return false;
  }
}

bool iree_profile_model_device_try_fit_clock_exact(
    const iree_profile_model_device_t* device,
    iree_profile_model_clock_time_domain_t time_domain,
    iree_profile_model_clock_fit_t* out_fit) {
  memset(out_fit, 0, sizeof(*out_fit));
  if (!device || device->clock_sample_count < 2) return false;

  const iree_hal_profile_clock_correlation_record_t* first =
      &device->first_clock_sample;
  const iree_hal_profile_clock_correlation_record_t* last =
      &device->last_clock_sample;
  if (!iree_all_bits_set(first->flags,
                         IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_DEVICE_TICK) ||
      !iree_all_bits_set(last->flags,
                         IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_DEVICE_TICK)) {
    return false;
  }
  int64_t first_time_ns = 0;
  int64_t last_time_ns = 0;
  if (!iree_profile_model_sample_time_ns(first, time_domain, &first_time_ns) ||
      !iree_profile_model_sample_time_ns(last, time_domain, &last_time_ns)) {
    return false;
  }
  if (last->device_tick <= first->device_tick ||
      last_time_ns <= first_time_ns) {
    return false;
  }

  out_fit->time_domain = time_domain;
  out_fit->first_sample_id = first->sample_id;
  out_fit->last_sample_id = last->sample_id;
  out_fit->first_device_tick = first->device_tick;
  out_fit->last_device_tick = last->device_tick;
  out_fit->first_time_ns = first_time_ns;
  out_fit->last_time_ns = last_time_ns;
  out_fit->device_tick_span = last->device_tick - first->device_tick;
  out_fit->time_span_ns = (uint64_t)(last_time_ns - first_time_ns);
  return true;
}

static bool iree_profile_model_round_mul_div_u64(uint64_t value,
                                                 uint64_t numerator,
                                                 uint64_t denominator,
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

bool iree_profile_model_clock_fit_scale_ticks_to_ns(
    const iree_profile_model_clock_fit_t* fit, uint64_t device_tick_count,
    int64_t* out_duration_ns) {
  *out_duration_ns = 0;
  if (!fit || fit->device_tick_span == 0) return false;
  uint64_t duration_ns = 0;
  if (!iree_profile_model_round_mul_div_u64(
          device_tick_count, fit->time_span_ns, fit->device_tick_span,
          &duration_ns) ||
      duration_ns > INT64_MAX) {
    return false;
  }
  *out_duration_ns = (int64_t)duration_ns;
  return true;
}

bool iree_profile_model_clock_fit_map_tick(
    const iree_profile_model_clock_fit_t* fit, uint64_t device_tick,
    int64_t* out_time_ns) {
  *out_time_ns = 0;
  if (!fit || fit->device_tick_span == 0) return false;

  const bool after_base = device_tick >= fit->first_device_tick;
  const uint64_t tick_delta = after_base ? device_tick - fit->first_device_tick
                                         : fit->first_device_tick - device_tick;
  int64_t time_delta_ns = 0;
  if (!iree_profile_model_clock_fit_scale_ticks_to_ns(fit, tick_delta,
                                                      &time_delta_ns)) {
    return false;
  }

  if (after_base) {
    if (fit->first_time_ns > INT64_MAX - time_delta_ns) return false;
    *out_time_ns = fit->first_time_ns + time_delta_ns;
  } else {
    if (fit->first_time_ns < INT64_MIN + time_delta_ns) return false;
    *out_time_ns = fit->first_time_ns - time_delta_ns;
  }
  return true;
}

double iree_profile_model_clock_fit_ns_per_tick(
    const iree_profile_model_clock_fit_t* fit) {
  if (!fit || fit->device_tick_span == 0) return 0.0;
  return (double)fit->time_span_ns / (double)fit->device_tick_span;
}

double iree_profile_model_clock_fit_tick_frequency_hz(
    const iree_profile_model_clock_fit_t* fit) {
  const double ns_per_tick = iree_profile_model_clock_fit_ns_per_tick(fit);
  return ns_per_tick > 0.0 ? 1000000000.0 / ns_per_tick : 0.0;
}

bool iree_profile_model_device_try_fit_clock(
    const iree_profile_model_device_t* device, double* out_ns_per_tick,
    double* out_tick_frequency_hz) {
  *out_ns_per_tick = 0.0;
  *out_tick_frequency_hz = 0.0;
  iree_profile_model_clock_fit_t fit;
  if (!iree_profile_model_device_try_fit_clock_exact(
          device, IREE_PROFILE_MODEL_CLOCK_TIME_DOMAIN_HOST_CPU_TIMESTAMP_NS,
          &fit)) {
    return false;
  }
  *out_ns_per_tick = iree_profile_model_clock_fit_ns_per_tick(&fit);
  *out_tick_frequency_hz = iree_profile_model_clock_fit_tick_frequency_hz(&fit);
  return true;
}

static iree_string_view_t iree_profile_model_format_numeric_key(
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

iree_string_view_t iree_profile_model_format_export_key(
    const iree_profile_model_export_t* export_info,
    uint32_t physical_device_ordinal, char* numeric_buffer,
    iree_host_size_t numeric_buffer_capacity) {
  if (!iree_string_view_is_empty(export_info->name)) {
    return export_info->name;
  }
  return iree_profile_model_format_numeric_key(
      physical_device_ordinal, export_info->executable_id,
      export_info->export_ordinal, numeric_buffer, numeric_buffer_capacity);
}

iree_status_t iree_profile_model_resolve_dispatch_key(
    const iree_profile_model_t* model, uint32_t physical_device_ordinal,
    uint64_t executable_id, uint32_t export_ordinal, char* numeric_buffer,
    iree_host_size_t numeric_buffer_capacity, iree_string_view_t* out_key) {
  *out_key = iree_string_view_empty();
  const iree_profile_model_executable_t* executable_info =
      iree_profile_model_find_executable(model, executable_id);
  if (!executable_info) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "dispatch event references missing executable metadata "
        "device=%u executable=%" PRIu64 " export=%u",
        physical_device_ordinal, executable_id, export_ordinal);
  }
  const iree_profile_model_export_t* export_info =
      iree_profile_model_find_export(model, executable_id, export_ordinal);
  if (!export_info) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "dispatch event references missing executable export metadata "
        "device=%u executable=%" PRIu64 " export=%u",
        physical_device_ordinal, executable_id, export_ordinal);
  }
  *out_key = iree_profile_model_format_export_key(
      export_info, physical_device_ordinal, numeric_buffer,
      numeric_buffer_capacity);
  return iree_ok_status();
}

static iree_status_t iree_profile_model_append_export(
    iree_profile_model_t* model,
    const iree_profile_model_export_t* export_info) {
  iree_profile_model_executable_t* executable =
      iree_profile_model_find_executable_mutable(model,
                                                 export_info->executable_id);
  if (!executable) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "executable export references missing executable metadata "
        "executable=%" PRIu64 " export=%u",
        export_info->executable_id, export_info->export_ordinal);
  }

  const iree_profile_model_export_lookup_t lookup = {
      .model = model,
      .executable_id = export_info->executable_id,
      .export_ordinal = export_info->export_ordinal,
  };
  const uint64_t hash = iree_profile_model_export_hash(
      export_info->executable_id, export_info->export_ordinal);
  iree_host_size_t existing_index = 0;
  if (iree_profile_index_find(&model->export_index, hash,
                              iree_profile_model_export_matches, &lookup,
                              &existing_index)) {
    return iree_ok_status();
  }

  if (model->export_count + 1 > model->export_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        model->host_allocator,
        iree_max((iree_host_size_t)16, model->export_count + 1),
        sizeof(model->exports[0]), &model->export_capacity,
        (void**)&model->exports));
  }

  const iree_host_size_t export_index = model->export_count;
  IREE_RETURN_IF_ERROR(iree_profile_index_insert(
      &model->export_index, model->host_allocator, hash, export_index));
  model->exports[export_index] = *export_info;
  model->exports[export_index].next_export_index = IREE_HOST_SIZE_MAX;
  if (executable->first_export_index == IREE_HOST_SIZE_MAX) {
    executable->first_export_index = export_index;
  } else {
    model->exports[executable->last_export_index].next_export_index =
        export_index;
  }
  executable->last_export_index = export_index;
  ++executable->export_row_count;
  ++model->export_count;
  return iree_ok_status();
}

static iree_status_t iree_profile_model_append_executable(
    iree_profile_model_t* model,
    const iree_hal_profile_executable_record_t* record) {
  const iree_profile_model_executable_lookup_t lookup = {
      .model = model,
      .executable_id = record->executable_id,
  };
  const uint64_t hash =
      iree_profile_model_executable_hash(record->executable_id);
  iree_host_size_t existing_index = 0;
  if (iree_profile_index_find(&model->executable_index, hash,
                              iree_profile_model_executable_matches, &lookup,
                              &existing_index)) {
    return iree_ok_status();
  }

  if (model->executable_count + 1 > model->executable_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        model->host_allocator,
        iree_max((iree_host_size_t)8, model->executable_count + 1),
        sizeof(model->executables[0]), &model->executable_capacity,
        (void**)&model->executables));
  }
  const iree_host_size_t executable_index = model->executable_count;
  IREE_RETURN_IF_ERROR(iree_profile_index_insert(
      &model->executable_index, model->host_allocator, hash, executable_index));
  iree_profile_model_executable_t* executable_info =
      &model->executables[executable_index];
  executable_info->record = *record;
  executable_info->first_export_index = IREE_HOST_SIZE_MAX;
  executable_info->last_export_index = IREE_HOST_SIZE_MAX;
  executable_info->export_row_count = 0;
  ++model->executable_count;
  return iree_ok_status();
}

static iree_status_t iree_profile_model_append_command_buffer(
    iree_profile_model_t* model,
    const iree_hal_profile_command_buffer_record_t* record) {
  const iree_profile_model_command_buffer_lookup_t lookup = {
      .model = model,
      .command_buffer_id = record->command_buffer_id,
  };
  const uint64_t hash =
      iree_profile_model_command_buffer_hash(record->command_buffer_id);
  iree_host_size_t existing_index = 0;
  if (iree_profile_index_find(&model->command_buffer_index, hash,
                              iree_profile_model_command_buffer_matches,
                              &lookup, &existing_index)) {
    return iree_ok_status();
  }

  if (model->command_buffer_count + 1 > model->command_buffer_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        model->host_allocator,
        iree_max((iree_host_size_t)16, model->command_buffer_count + 1),
        sizeof(model->command_buffers[0]), &model->command_buffer_capacity,
        (void**)&model->command_buffers));
  }
  const iree_host_size_t command_buffer_index = model->command_buffer_count;
  IREE_RETURN_IF_ERROR(iree_profile_index_insert(&model->command_buffer_index,
                                                 model->host_allocator, hash,
                                                 command_buffer_index));
  iree_profile_model_command_buffer_t* command_buffer_info =
      &model->command_buffers[command_buffer_index];
  command_buffer_info->record = *record;
  command_buffer_info->first_operation_index = IREE_HOST_SIZE_MAX;
  command_buffer_info->last_operation_index = IREE_HOST_SIZE_MAX;
  command_buffer_info->operation_count = 0;
  ++model->command_buffer_count;
  return iree_ok_status();
}

static iree_status_t iree_profile_model_append_command_operation(
    iree_profile_model_t* model,
    const iree_hal_profile_command_operation_record_t* record) {
  iree_profile_model_command_buffer_t* command_buffer =
      iree_profile_model_find_command_buffer_mutable(model,
                                                     record->command_buffer_id);
  if (!command_buffer) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "command operation references missing command-buffer metadata "
        "command_buffer=%" PRIu64 " command_index=%u",
        record->command_buffer_id, record->command_index);
  }

  if (model->command_operation_count + 1 > model->command_operation_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        model->host_allocator,
        iree_max((iree_host_size_t)64, model->command_operation_count + 1),
        sizeof(model->command_operations[0]),
        &model->command_operation_capacity,
        (void**)&model->command_operations));
  }

  const iree_host_size_t operation_index = model->command_operation_count;
  iree_profile_model_command_operation_t* operation_info =
      &model->command_operations[operation_index];
  operation_info->record = *record;
  operation_info->next_operation_index = IREE_HOST_SIZE_MAX;
  if (command_buffer->first_operation_index == IREE_HOST_SIZE_MAX) {
    command_buffer->first_operation_index = operation_index;
  } else {
    model->command_operations[command_buffer->last_operation_index]
        .next_operation_index = operation_index;
  }
  command_buffer->last_operation_index = operation_index;
  ++command_buffer->operation_count;
  ++model->command_operation_count;
  return iree_ok_status();
}

static iree_status_t iree_profile_model_validate_command_operation(
    const iree_hal_profile_command_operation_record_t* record) {
  const bool has_block_structure =
      iree_hal_profile_command_operation_has_block_structure(record);
  if (has_block_structure) {
    if (record->block_ordinal == UINT32_MAX ||
        record->block_command_ordinal == UINT32_MAX) {
      return iree_make_status(
          IREE_STATUS_DATA_LOSS,
          "command operation declares block structure without block "
          "coordinates command_buffer=%" PRIu64 " command_index=%u",
          record->command_buffer_id, record->command_index);
    }
  } else if (record->block_ordinal != UINT32_MAX ||
             record->block_command_ordinal != UINT32_MAX ||
             record->target_block_ordinal != UINT32_MAX ||
             record->alternate_block_ordinal != UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "command operation carries block coordinates without the block "
        "structure flag command_buffer=%" PRIu64 " command_index=%u",
        record->command_buffer_id, record->command_index);
  }
  return iree_ok_status();
}

static iree_status_t iree_profile_model_append_queue(
    iree_profile_model_t* model,
    const iree_hal_profile_queue_record_t* record) {
  const iree_profile_model_queue_lookup_t lookup = {
      .model = model,
      .physical_device_ordinal = record->physical_device_ordinal,
      .queue_ordinal = record->queue_ordinal,
      .stream_id = record->stream_id,
  };
  const uint64_t hash =
      iree_profile_model_queue_hash(record->physical_device_ordinal,
                                    record->queue_ordinal, record->stream_id);
  iree_host_size_t existing_index = 0;
  if (iree_profile_index_find(&model->queue_index, hash,
                              iree_profile_model_queue_matches, &lookup,
                              &existing_index)) {
    return iree_ok_status();
  }

  if (model->queue_count + 1 > model->queue_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        model->host_allocator,
        iree_max((iree_host_size_t)4, model->queue_count + 1),
        sizeof(model->queues[0]), &model->queue_capacity,
        (void**)&model->queues));
  }
  const iree_host_size_t queue_index = model->queue_count;
  IREE_RETURN_IF_ERROR(iree_profile_index_insert(
      &model->queue_index, model->host_allocator, hash, queue_index));
  iree_profile_model_queue_t* queue_info = &model->queues[queue_index];
  queue_info->record = *record;
  ++model->queue_count;
  return iree_ok_status();
}

static iree_status_t iree_profile_model_append_metric_source(
    iree_profile_model_t* model,
    const iree_profile_model_metric_source_t* source_info) {
  const iree_profile_model_metric_source_lookup_t lookup = {
      .model = model,
      .source_id = source_info->record.source_id,
  };
  const uint64_t hash =
      iree_profile_model_metric_source_hash(source_info->record.source_id);
  iree_host_size_t existing_index = 0;
  if (iree_profile_index_find(&model->metric_source_index, hash,
                              iree_profile_model_metric_source_matches, &lookup,
                              &existing_index)) {
    return iree_ok_status();
  }

  if (model->metric_source_count + 1 > model->metric_source_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        model->host_allocator,
        iree_max((iree_host_size_t)4, model->metric_source_count + 1),
        sizeof(model->metric_sources[0]), &model->metric_source_capacity,
        (void**)&model->metric_sources));
  }
  const iree_host_size_t source_index = model->metric_source_count;
  IREE_RETURN_IF_ERROR(iree_profile_index_insert(
      &model->metric_source_index, model->host_allocator, hash, source_index));
  model->metric_sources[source_index] = *source_info;
  ++model->metric_source_count;
  return iree_ok_status();
}

static iree_status_t iree_profile_model_append_metric_descriptor(
    iree_profile_model_t* model,
    const iree_profile_model_metric_descriptor_t* descriptor_info) {
  const iree_profile_model_metric_descriptor_lookup_t lookup = {
      .model = model,
      .source_id = descriptor_info->record.source_id,
      .metric_id = descriptor_info->record.metric_id,
  };
  const uint64_t hash = iree_profile_model_metric_descriptor_hash(
      descriptor_info->record.source_id, descriptor_info->record.metric_id);
  iree_host_size_t existing_index = 0;
  if (iree_profile_index_find(&model->metric_descriptor_index, hash,
                              iree_profile_model_metric_descriptor_matches,
                              &lookup, &existing_index)) {
    return iree_ok_status();
  }

  if (model->metric_descriptor_count + 1 > model->metric_descriptor_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        model->host_allocator,
        iree_max((iree_host_size_t)16, model->metric_descriptor_count + 1),
        sizeof(model->metric_descriptors[0]),
        &model->metric_descriptor_capacity,
        (void**)&model->metric_descriptors));
  }
  const iree_host_size_t descriptor_index = model->metric_descriptor_count;
  IREE_RETURN_IF_ERROR(
      iree_profile_index_insert(&model->metric_descriptor_index,
                                model->host_allocator, hash, descriptor_index));
  model->metric_descriptors[descriptor_index] = *descriptor_info;
  ++model->metric_descriptor_count;
  return iree_ok_status();
}

static iree_status_t iree_profile_model_process_queue_records(
    iree_profile_model_t* model, const iree_hal_profile_file_record_t* record) {
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
    status = iree_profile_model_append_queue(model, &record_value);
  }
  return status;
}

static iree_status_t iree_profile_model_process_metric_source_records(
    iree_profile_model_t* model, const iree_hal_profile_file_record_t* record) {
  iree_profile_typed_record_iterator_t iterator;
  iree_profile_typed_record_iterator_initialize(
      record, sizeof(iree_hal_profile_device_metric_source_record_t),
      &iterator);
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status)) {
    iree_profile_typed_record_t typed_record;
    bool has_record = false;
    status = iree_profile_typed_record_iterator_next(&iterator, &typed_record,
                                                     &has_record);
    if (!iree_status_is_ok(status) || !has_record) break;

    iree_hal_profile_device_metric_source_record_t record_value;
    memcpy(&record_value, typed_record.contents.data, sizeof(record_value));
    if ((iree_host_size_t)record_value.name_length !=
        typed_record.inline_payload.data_length) {
      status = iree_make_status(IREE_STATUS_DATA_LOSS,
                                "device metric source name length is "
                                "inconsistent with record length");
    }
    if (iree_status_is_ok(status)) {
      iree_profile_model_metric_source_t source_info = {
          .record = record_value,
          .name = iree_make_string_view(
              (const char*)typed_record.inline_payload.data,
              typed_record.inline_payload.data_length),
      };
      status = iree_profile_model_append_metric_source(model, &source_info);
    }
  }
  return status;
}

static iree_status_t iree_profile_model_process_metric_descriptor_records(
    iree_profile_model_t* model, const iree_hal_profile_file_record_t* record) {
  iree_profile_typed_record_iterator_t iterator;
  iree_profile_typed_record_iterator_initialize(
      record, sizeof(iree_hal_profile_device_metric_descriptor_record_t),
      &iterator);
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status)) {
    iree_profile_typed_record_t typed_record;
    bool has_record = false;
    status = iree_profile_typed_record_iterator_next(&iterator, &typed_record,
                                                     &has_record);
    if (!iree_status_is_ok(status) || !has_record) break;

    iree_hal_profile_device_metric_descriptor_record_t record_value;
    memcpy(&record_value, typed_record.contents.data, sizeof(record_value));
    iree_host_size_t trailing_length = 0;
    if (!iree_host_size_checked_add(record_value.name_length,
                                    record_value.description_length,
                                    &trailing_length) ||
        trailing_length != typed_record.inline_payload.data_length) {
      status = iree_make_status(
          IREE_STATUS_DATA_LOSS,
          "device metric descriptor string lengths are inconsistent with "
          "record length");
    }
    if (iree_status_is_ok(status)) {
      const char* string_base = (const char*)typed_record.inline_payload.data;
      iree_profile_model_metric_descriptor_t descriptor_info = {
          .record = record_value,
          .name = iree_make_string_view(string_base, record_value.name_length),
          .description =
              iree_make_string_view(string_base + record_value.name_length,
                                    record_value.description_length),
      };
      status =
          iree_profile_model_append_metric_descriptor(model, &descriptor_info);
    }
  }
  return status;
}

static iree_status_t iree_profile_model_process_executable_records(
    iree_profile_model_t* model, const iree_hal_profile_file_record_t* record) {
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
    status = iree_profile_model_append_executable(model, &record_value);
  }
  return status;
}

static iree_status_t iree_profile_model_process_export_records(
    iree_profile_model_t* model, const iree_hal_profile_file_record_t* record) {
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
      iree_profile_model_export_t export_info = {
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
      status = iree_profile_model_append_export(model, &export_info);
    }
  }
  return status;
}

static iree_status_t iree_profile_model_process_command_buffer_records(
    iree_profile_model_t* model, const iree_hal_profile_file_record_t* record) {
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
    status = iree_profile_model_append_command_buffer(model, &record_value);
  }
  return status;
}

static iree_status_t iree_profile_model_process_command_operation_records(
    iree_profile_model_t* model, const iree_hal_profile_file_record_t* record) {
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
    status = iree_profile_model_validate_command_operation(&record_value);
    if (iree_status_is_ok(status)) {
      status =
          iree_profile_model_append_command_operation(model, &record_value);
    }
  }
  return status;
}

static iree_status_t iree_profile_model_process_clock_records(
    iree_profile_model_t* model, const iree_hal_profile_file_record_t* record) {
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

    iree_profile_model_device_t* device = NULL;
    status = iree_profile_model_ensure_device(
        model, clock_record.physical_device_ordinal, &device);
    if (iree_status_is_ok(status)) {
      iree_profile_model_record_clock_sample(device, &clock_record);
    }
  }
  return status;
}

typedef iree_status_t (*iree_profile_model_chunk_processor_fn_t)(
    iree_profile_model_t* model, const iree_hal_profile_file_record_t* record);

typedef struct iree_profile_model_chunk_route_t {
  // Profile chunk content type handled by this route.
  iree_string_view_t content_type;
  // Processor used to add the chunk's metadata rows to the model.
  iree_profile_model_chunk_processor_fn_t process;
} iree_profile_model_chunk_route_t;

iree_status_t iree_profile_model_process_metadata_record(
    iree_profile_model_t* model, const iree_hal_profile_file_record_t* record) {
  if (record->header.record_type != IREE_HAL_PROFILE_FILE_RECORD_TYPE_CHUNK) {
    return iree_ok_status();
  }
  const iree_profile_model_chunk_route_t routes[] = {
      {IREE_HAL_PROFILE_CONTENT_TYPE_QUEUES,
       iree_profile_model_process_queue_records},
      {IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLES,
       iree_profile_model_process_executable_records},
      {IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLE_EXPORTS,
       iree_profile_model_process_export_records},
      {IREE_HAL_PROFILE_CONTENT_TYPE_COMMAND_BUFFERS,
       iree_profile_model_process_command_buffer_records},
      {IREE_HAL_PROFILE_CONTENT_TYPE_COMMAND_OPERATIONS,
       iree_profile_model_process_command_operation_records},
      {IREE_HAL_PROFILE_CONTENT_TYPE_CLOCK_CORRELATIONS,
       iree_profile_model_process_clock_records},
      {IREE_HAL_PROFILE_CONTENT_TYPE_DEVICE_METRIC_SOURCES,
       iree_profile_model_process_metric_source_records},
      {IREE_HAL_PROFILE_CONTENT_TYPE_DEVICE_METRIC_DESCRIPTORS,
       iree_profile_model_process_metric_descriptor_records},
  };
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(routes); ++i) {
    if (iree_string_view_equal(record->content_type, routes[i].content_type)) {
      return routes[i].process(model, record);
    }
  }
  return iree_ok_status();
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
    case IREE_HAL_PROFILE_EVENT_RELATIONSHIP_TYPE_QUEUE_EVENT_HOST_EXECUTION_EVENT:
      return "queue_event_host_execution_event";
    case IREE_HAL_PROFILE_EVENT_RELATIONSHIP_TYPE_QUEUE_SUBMISSION_HOST_EXECUTION_EVENT:
      return "queue_submission_host_execution_event";
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
    case IREE_HAL_PROFILE_EVENT_ENDPOINT_TYPE_HOST_EXECUTION_EVENT:
      return "host_execution_event";
    default:
      return "unknown";
  }
}

double iree_profile_model_span_ticks(uint64_t earliest_start_tick,
                                     uint64_t latest_end_tick) {
  if (earliest_start_tick == UINT64_MAX || latest_end_tick == 0 ||
      latest_end_tick < earliest_start_tick) {
    return 0.0;
  }
  return (double)(latest_end_tick - earliest_start_tick);
}

iree_status_t iree_profile_model_resolve_command_operation_key(
    const iree_profile_model_t* model,
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

  const iree_profile_model_command_buffer_t* command_buffer =
      iree_profile_model_find_command_buffer(model,
                                             operation->command_buffer_id);
  if (!command_buffer) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "command operation references missing command-buffer metadata "
        "command_buffer=%" PRIu64 " command_index=%u",
        operation->command_buffer_id, operation->command_index);
  }
  return iree_profile_model_resolve_dispatch_key(
      model, command_buffer->record.physical_device_ordinal,
      operation->executable_id, operation->export_ordinal, numeric_buffer,
      numeric_buffer_capacity, out_key);
}
