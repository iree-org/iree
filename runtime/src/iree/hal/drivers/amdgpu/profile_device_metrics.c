// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/profile_device_metrics.h"

#include <inttypes.h>
#include <string.h>

#include "iree/hal/api.h"
#include "iree/hal/drivers/amdgpu/logical_device.h"
#include "iree/hal/drivers/amdgpu/profile_device_metrics_source.h"

//===----------------------------------------------------------------------===//
// Device metric source session
//===----------------------------------------------------------------------===//

struct iree_hal_amdgpu_profile_device_metrics_session_t {
  // Host allocator used for session storage.
  iree_allocator_t host_allocator;
  // Number of initialized entries in |sources|.
  iree_host_size_t source_count;
  // Per-physical-device metric sources.
  iree_hal_amdgpu_profile_device_metric_source_t sources[];
};

//===----------------------------------------------------------------------===//
// Sample builder
//===----------------------------------------------------------------------===//

static bool iree_hal_amdgpu_profile_device_metrics_value_is_present(
    const iree_hal_amdgpu_profile_device_metric_sample_builder_t* builder,
    uint64_t metric_id) {
  for (uint32_t i = 0; i < builder->record.value_count; ++i) {
    if (builder->values[i].metric_id == metric_id) return true;
  }
  return false;
}

void iree_hal_amdgpu_profile_device_metric_sample_builder_append_u64(
    iree_hal_amdgpu_profile_device_metric_sample_builder_t* builder,
    uint64_t metric_id, uint64_t value) {
  if (iree_hal_amdgpu_profile_device_metrics_value_is_present(builder,
                                                              metric_id)) {
    return;
  }
  if (builder->record.value_count >=
      IREE_HAL_AMDGPU_PROFILE_DEVICE_METRICS_MAX_SAMPLE_VALUES) {
    builder->record.flags |= IREE_HAL_PROFILE_DEVICE_METRIC_SAMPLE_FLAG_PARTIAL;
    return;
  }
  builder->values[builder->record.value_count++] =
      (iree_hal_profile_device_metric_value_t){
          .metric_id = metric_id,
          .value_bits = value,
      };
}

void iree_hal_amdgpu_profile_device_metric_sample_builder_append_i64(
    iree_hal_amdgpu_profile_device_metric_sample_builder_t* builder,
    uint64_t metric_id, int64_t value) {
  iree_hal_amdgpu_profile_device_metric_sample_builder_append_u64(
      builder, metric_id, (uint64_t)value);
}

//===----------------------------------------------------------------------===//
// Profile record emission
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_amdgpu_profile_device_metrics_write_chunk(
    iree_hal_profile_sink_t* sink, uint64_t session_id,
    iree_string_view_t stream_name, iree_string_view_t content_type,
    const uint8_t* storage, iree_host_size_t storage_size) {
  iree_hal_profile_chunk_metadata_t metadata =
      iree_hal_profile_chunk_metadata_default();
  metadata.content_type = content_type;
  metadata.name = stream_name;
  metadata.session_id = session_id;
  iree_const_byte_span_t iovec =
      iree_make_const_byte_span(storage, storage_size);
  return iree_hal_profile_sink_write(sink, &metadata, 1, &iovec);
}

static iree_status_t iree_hal_amdgpu_profile_device_metrics_source_record_size(
    const iree_hal_amdgpu_profile_device_metric_source_t* source,
    iree_host_size_t* out_record_size) {
  const iree_host_size_t name_length = strlen(source->metadata.name);
  return IREE_STRUCT_LAYOUT(
      0, out_record_size,
      IREE_STRUCT_FIELD(1, iree_hal_profile_device_metric_source_record_t,
                        NULL),
      IREE_STRUCT_FIELD(name_length, char, NULL));
}

static iree_status_t
iree_hal_amdgpu_profile_device_metrics_descriptor_record_size(
    const iree_hal_profile_metric_descriptor_t* descriptor,
    iree_host_size_t* out_record_size) {
  return IREE_STRUCT_LAYOUT(
      0, out_record_size,
      IREE_STRUCT_FIELD(1, iree_hal_profile_device_metric_descriptor_record_t,
                        NULL),
      IREE_STRUCT_FIELD(descriptor->name.size + descriptor->description.size,
                        char, NULL));
}

static iree_status_t iree_hal_amdgpu_profile_device_metrics_pack_source_record(
    const iree_hal_amdgpu_profile_device_metric_source_t* source,
    uint8_t* storage, iree_host_size_t storage_capacity,
    iree_host_size_t* out_storage_size) {
  iree_host_size_t record_size = 0;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_profile_device_metrics_source_record_size(source,
                                                                &record_size));
  if (record_size > UINT32_MAX || record_size > storage_capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU metric source record exceeds storage");
  }

  const iree_host_size_t name_length = strlen(source->metadata.name);
  iree_hal_profile_device_metric_source_record_t record =
      iree_hal_profile_device_metric_source_record_default();
  record.record_length = (uint32_t)record_size;
  record.source_id = source->metadata.id;
  record.physical_device_ordinal = source->metadata.physical_device_ordinal;
  record.device_class = IREE_HAL_PROFILE_DEVICE_CLASS_GPU;
  record.source_kind = source->metadata.kind;
  record.source_revision = source->metadata.revision;
  record.metric_count = source->metrics.count;
  record.name_length = (uint32_t)name_length;

  memcpy(storage, &record, sizeof(record));
  memcpy(storage + sizeof(record), source->metadata.name, name_length);
  *out_storage_size = record_size;
  return iree_ok_status();
}

static iree_status_t
iree_hal_amdgpu_profile_device_metrics_pack_descriptor_record(
    const iree_hal_amdgpu_profile_device_metric_source_t* source,
    const iree_hal_profile_metric_descriptor_t* descriptor, uint8_t* storage,
    iree_host_size_t storage_capacity, iree_host_size_t* out_storage_size) {
  iree_host_size_t record_size = 0;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_profile_device_metrics_descriptor_record_size(
          descriptor, &record_size));
  if (record_size > UINT32_MAX || record_size > storage_capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU metric descriptor record exceeds storage");
  }

  iree_hal_profile_device_metric_descriptor_record_t record =
      iree_hal_profile_device_metric_descriptor_record_default();
  record.record_length = (uint32_t)record_size;
  record.source_id = source->metadata.id;
  record.metric_id = descriptor->metric_id;
  record.unit = descriptor->unit;
  record.value_kind = descriptor->value_kind;
  record.semantic = descriptor->semantic;
  record.plot_hint = descriptor->plot_hint;
  record.name_length = (uint32_t)descriptor->name.size;
  record.description_length = (uint32_t)descriptor->description.size;

  memcpy(storage, &record, sizeof(record));
  uint8_t* string_ptr = storage + sizeof(record);
  memcpy(string_ptr, descriptor->name.data, descriptor->name.size);
  string_ptr += descriptor->name.size;
  memcpy(string_ptr, descriptor->description.data,
         descriptor->description.size);
  *out_storage_size = record_size;
  return iree_ok_status();
}

static iree_status_t
iree_hal_amdgpu_profile_device_metrics_write_source_metadata(
    const iree_hal_amdgpu_profile_device_metric_source_t* source,
    iree_hal_profile_sink_t* sink, uint64_t session_id,
    iree_string_view_t stream_name) {
  uint8_t storage[128] = {0};
  iree_host_size_t storage_size = 0;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_profile_device_metrics_pack_source_record(
          source, storage, sizeof(storage), &storage_size));
  return iree_hal_amdgpu_profile_device_metrics_write_chunk(
      sink, session_id, stream_name,
      IREE_HAL_PROFILE_CONTENT_TYPE_DEVICE_METRIC_SOURCES, storage,
      storage_size);
}

static iree_status_t
iree_hal_amdgpu_profile_device_metrics_write_descriptor_metadata(
    const iree_hal_amdgpu_profile_device_metric_source_t* source,
    iree_hal_profile_sink_t* sink, uint64_t session_id,
    iree_string_view_t stream_name, uint64_t metric_id) {
  const iree_hal_profile_metric_descriptor_t* descriptor =
      iree_hal_profile_builtin_metric_descriptor_lookup(metric_id);
  if (!descriptor) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "AMDGPU metric id %" PRIu64 " has no built-in descriptor", metric_id);
  }

  uint8_t storage[256] = {0};
  iree_host_size_t storage_size = 0;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_profile_device_metrics_pack_descriptor_record(
          source, descriptor, storage, sizeof(storage), &storage_size));
  return iree_hal_amdgpu_profile_device_metrics_write_chunk(
      sink, session_id, stream_name,
      IREE_HAL_PROFILE_CONTENT_TYPE_DEVICE_METRIC_DESCRIPTORS, storage,
      storage_size);
}

static iree_status_t iree_hal_amdgpu_profile_device_metrics_write_sample(
    const iree_hal_amdgpu_profile_device_metric_sample_builder_t* builder,
    iree_hal_profile_sink_t* sink, uint64_t session_id,
    iree_string_view_t stream_name) {
  iree_host_size_t storage_size = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      0, &storage_size,
      IREE_STRUCT_FIELD(1, iree_hal_profile_device_metric_sample_record_t,
                        NULL),
      IREE_STRUCT_FIELD(builder->record.value_count,
                        iree_hal_profile_device_metric_value_t, NULL)));
  if (IREE_UNLIKELY(storage_size > UINT32_MAX)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU metric sample exceeds uint32_t");
  }

  iree_hal_profile_device_metric_sample_record_t record = builder->record;
  record.record_length = (uint32_t)storage_size;
  iree_const_byte_span_t iovecs[2] = {
      iree_make_const_byte_span((const uint8_t*)&record, sizeof(record)),
      iree_make_const_byte_span(
          (const uint8_t*)builder->values,
          builder->record.value_count * sizeof(builder->values[0])),
  };

  iree_hal_profile_chunk_metadata_t metadata =
      iree_hal_profile_chunk_metadata_default();
  metadata.content_type = IREE_HAL_PROFILE_CONTENT_TYPE_DEVICE_METRIC_SAMPLES;
  metadata.name = stream_name;
  metadata.session_id = session_id;
  metadata.stream_id = record.source_id;
  return iree_hal_profile_sink_write(
      sink, &metadata, builder->record.value_count ? 2 : 1, iovecs);
}

//===----------------------------------------------------------------------===//
// Session lifecycle
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_amdgpu_profile_device_metrics_session_allocate(
    iree_hal_amdgpu_logical_device_t* logical_device,
    const iree_hal_device_profiling_options_t* options,
    iree_allocator_t host_allocator,
    iree_hal_amdgpu_profile_device_metrics_session_t** out_session) {
  *out_session = NULL;
  if (!iree_hal_device_profiling_options_requests_device_metrics(options)) {
    return iree_ok_status();
  }
  if (IREE_UNLIKELY(!options->sink)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU device metrics profiling requires a profile sink");
  }
  if (IREE_UNLIKELY(logical_device->physical_device_count > UINT32_MAX)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU physical device count exceeds uint32_t");
  }

  iree_host_size_t session_size = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_amdgpu_profile_device_metrics_session_t), &session_size,
      IREE_STRUCT_FIELD(logical_device->physical_device_count,
                        iree_hal_amdgpu_profile_device_metric_source_t, NULL)));

  iree_hal_amdgpu_profile_device_metrics_session_t* session = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, session_size, (void**)&session));
  memset(session, 0, session_size);
  session->host_allocator = host_allocator;

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < logical_device->physical_device_count && iree_status_is_ok(status);
       ++i) {
    status = iree_hal_amdgpu_profile_device_metric_source_initialize(
        logical_device->physical_devices[i], host_allocator,
        &session->sources[i]);
    if (iree_status_is_ok(status)) {
      ++session->source_count;
    }
  }
  if (iree_status_is_ok(status)) {
    *out_session = session;
  } else {
    iree_hal_amdgpu_profile_device_metrics_session_free(session);
  }
  return status;
}

void iree_hal_amdgpu_profile_device_metrics_session_free(
    iree_hal_amdgpu_profile_device_metrics_session_t* session) {
  if (!session) return;
  for (iree_host_size_t i = 0; i < session->source_count; ++i) {
    iree_hal_amdgpu_profile_device_metric_source_t* source =
        &session->sources[i];
    iree_hal_amdgpu_profile_device_metric_source_deinitialize(source);
  }
  iree_allocator_free(session->host_allocator, session);
}

iree_status_t iree_hal_amdgpu_profile_device_metrics_session_write_metadata(
    const iree_hal_amdgpu_profile_device_metrics_session_t* session,
    iree_hal_profile_sink_t* sink, uint64_t session_id,
    iree_string_view_t stream_name) {
  if (!session) return iree_ok_status();
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < session->source_count && iree_status_is_ok(status); ++i) {
    const iree_hal_amdgpu_profile_device_metric_source_t* source =
        &session->sources[i];
    status = iree_hal_amdgpu_profile_device_metrics_write_source_metadata(
        source, sink, session_id, stream_name);
    for (iree_host_size_t j = 0;
         j < source->metrics.count && iree_status_is_ok(status); ++j) {
      status = iree_hal_amdgpu_profile_device_metrics_write_descriptor_metadata(
          source, sink, session_id, stream_name, source->metrics.ids[j]);
    }
  }
  return status;
}

iree_status_t iree_hal_amdgpu_profile_device_metrics_session_sample_and_write(
    iree_hal_amdgpu_profile_device_metrics_session_t* session,
    iree_hal_profile_sink_t* sink, uint64_t session_id,
    iree_string_view_t stream_name) {
  if (!session) return iree_ok_status();
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < session->source_count && iree_status_is_ok(status); ++i) {
    iree_hal_amdgpu_profile_device_metric_source_t* source =
        &session->sources[i];
    iree_hal_amdgpu_profile_device_metric_sample_builder_t builder;
    memset(&builder, 0, sizeof(builder));
    builder.record = iree_hal_profile_device_metric_sample_record_default();
    builder.record.sample_id = source->sampling.next_sample_id++;
    builder.record.source_id = source->metadata.id;
    builder.record.physical_device_ordinal =
        source->metadata.physical_device_ordinal;
    builder.record.host_time_begin_ns = iree_time_now();
    status =
        iree_hal_amdgpu_profile_device_metric_source_sample(source, &builder);
    builder.record.host_time_end_ns = iree_time_now();
    if (iree_status_is_ok(status)) {
      status = iree_hal_amdgpu_profile_device_metrics_write_sample(
          &builder, sink, session_id, stream_name);
    }
  }
  return status;
}
