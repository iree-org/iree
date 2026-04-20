// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>
#include <stdlib.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/utils/profile_file.h"
#include "iree/io/file_handle.h"

static const uint64_t kSmokeSessionId = 1;
static const uint32_t kSmokePhysicalDevice = 0;
static const uint32_t kSmokeQueueOrdinal = 0;
static const uint64_t kSmokeStreamId = 11;
static const uint64_t kSmokeCommandBufferId = 7;
static const uint32_t kSmokeCommandIndex = 2;
static const uint64_t kSmokeExecutableId = 13;
static const uint32_t kSmokeExportOrdinal = 1;
static const uint64_t kSmokeAllocationId = 31;
static const uint64_t kSmokeMetricSourceId = 41;
static const uint64_t kSmokeSourceSpecificMetricId =
    IREE_HAL_PROFILE_METRIC_ID_PRODUCER_BASE + 1;
static const uint64_t kSmokeSubmissionId = 77;

static iree_status_t write_profile_chunk_iovecs(
    iree_hal_profile_sink_t* sink, iree_string_view_t content_type,
    iree_string_view_t name, iree_host_size_t iovec_count,
    const iree_const_byte_span_t* iovecs) {
  iree_hal_profile_chunk_metadata_t metadata =
      iree_hal_profile_chunk_metadata_default();
  metadata.content_type = content_type;
  metadata.name = name;
  metadata.session_id = kSmokeSessionId;
  metadata.physical_device_ordinal = kSmokePhysicalDevice;
  metadata.queue_ordinal = kSmokeQueueOrdinal;
  metadata.stream_id = kSmokeStreamId;
  if (iovec_count == 0) {
    return iree_hal_profile_sink_write(sink, &metadata, 0, NULL);
  }
  return iree_hal_profile_sink_write(sink, &metadata, iovec_count, iovecs);
}

static iree_status_t write_profile_chunk(iree_hal_profile_sink_t* sink,
                                         iree_string_view_t content_type,
                                         iree_string_view_t name,
                                         iree_const_byte_span_t payload) {
  if (payload.data_length == 0) {
    return write_profile_chunk_iovecs(sink, content_type, name, 0, NULL);
  }
  return write_profile_chunk_iovecs(sink, content_type, name, 1, &payload);
}

static iree_status_t write_smoke_profile(iree_string_view_t path) {
  iree_io_file_handle_t* file_handle = NULL;
  IREE_RETURN_IF_ERROR(iree_io_file_handle_create(
      IREE_IO_FILE_MODE_READ | IREE_IO_FILE_MODE_WRITE, path,
      /*initial_size=*/0, iree_allocator_system(), &file_handle));

  iree_hal_profile_sink_t* sink = NULL;
  iree_status_t status = iree_hal_profile_file_sink_create(
      file_handle, iree_allocator_system(), &sink);
  iree_io_file_handle_release(file_handle);
  if (!iree_status_is_ok(status)) return status;

  iree_hal_profile_chunk_metadata_t session_metadata =
      iree_hal_profile_chunk_metadata_default();
  session_metadata.content_type = IREE_HAL_PROFILE_CONTENT_TYPE_SESSION;
  session_metadata.name = IREE_SV("cli-smoke");
  session_metadata.session_id = kSmokeSessionId;
  status = iree_hal_profile_sink_begin_session(sink, &session_metadata);

  if (iree_status_is_ok(status)) {
    iree_hal_profile_device_record_t device =
        iree_hal_profile_device_record_default();
    device.physical_device_ordinal = kSmokePhysicalDevice;
    device.queue_count = 1;
    status = write_profile_chunk(
        sink, IREE_HAL_PROFILE_CONTENT_TYPE_DEVICES, IREE_SV("devices"),
        iree_make_const_byte_span(&device, sizeof(device)));
  }

  if (iree_status_is_ok(status)) {
    iree_hal_profile_queue_record_t queue =
        iree_hal_profile_queue_record_default();
    queue.physical_device_ordinal = kSmokePhysicalDevice;
    queue.queue_ordinal = kSmokeQueueOrdinal;
    queue.stream_id = kSmokeStreamId;
    status = write_profile_chunk(
        sink, IREE_HAL_PROFILE_CONTENT_TYPE_QUEUES, IREE_SV("queues"),
        iree_make_const_byte_span(&queue, sizeof(queue)));
  }

  if (iree_status_is_ok(status)) {
    iree_hal_profile_executable_record_t executable =
        iree_hal_profile_executable_record_default();
    executable.executable_id = kSmokeExecutableId;
    executable.export_count = 1;
    status = write_profile_chunk(
        sink, IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLES, IREE_SV("executables"),
        iree_make_const_byte_span(&executable, sizeof(executable)));
  }

  if (iree_status_is_ok(status)) {
    const char export_name[] = "smoke_export";
    iree_hal_profile_executable_export_record_t export_record =
        iree_hal_profile_executable_export_record_default();
    export_record.executable_id = kSmokeExecutableId;
    export_record.export_ordinal = kSmokeExportOrdinal;
    export_record.binding_count = 3;
    export_record.workgroup_size[0] = 7;
    export_record.workgroup_size[1] = 8;
    export_record.workgroup_size[2] = 9;
    export_record.name_length = sizeof(export_name) - 1;
    export_record.record_length =
        sizeof(export_record) + export_record.name_length;
    iree_const_byte_span_t iovecs[2] = {
        iree_make_const_byte_span(&export_record, sizeof(export_record)),
        iree_make_const_byte_span(export_name, export_record.name_length),
    };
    status = write_profile_chunk_iovecs(
        sink, IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLE_EXPORTS,
        IREE_SV("executable-exports"), IREE_ARRAYSIZE(iovecs), iovecs);
  }

  if (iree_status_is_ok(status)) {
    iree_hal_profile_command_buffer_record_t command_buffer =
        iree_hal_profile_command_buffer_record_default();
    command_buffer.command_buffer_id = kSmokeCommandBufferId;
    command_buffer.queue_affinity = 1;
    command_buffer.physical_device_ordinal = kSmokePhysicalDevice;
    status = write_profile_chunk(
        sink, IREE_HAL_PROFILE_CONTENT_TYPE_COMMAND_BUFFERS,
        IREE_SV("command-buffers"),
        iree_make_const_byte_span(&command_buffer, sizeof(command_buffer)));
  }

  if (iree_status_is_ok(status)) {
    iree_hal_profile_command_operation_record_t command_operation =
        iree_hal_profile_command_operation_record_default();
    command_operation.type = IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_DISPATCH;
    command_operation.command_index = kSmokeCommandIndex;
    command_operation.command_buffer_id = kSmokeCommandBufferId;
    command_operation.executable_id = kSmokeExecutableId;
    command_operation.export_ordinal = kSmokeExportOrdinal;
    command_operation.binding_count = 3;
    command_operation.workgroup_count[0] = 4;
    command_operation.workgroup_count[1] = 5;
    command_operation.workgroup_count[2] = 6;
    command_operation.workgroup_size[0] = 7;
    command_operation.workgroup_size[1] = 8;
    command_operation.workgroup_size[2] = 9;
    status = write_profile_chunk(
        sink, IREE_HAL_PROFILE_CONTENT_TYPE_COMMAND_OPERATIONS,
        IREE_SV("command-operations"),
        iree_make_const_byte_span(&command_operation,
                                  sizeof(command_operation)));
  }

  if (iree_status_is_ok(status)) {
    iree_hal_profile_clock_correlation_record_t clock_samples[2];
    clock_samples[0] = iree_hal_profile_clock_correlation_record_default();
    clock_samples[0].flags =
        IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_DEVICE_TICK |
        IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_HOST_CPU_TIMESTAMP |
        IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_HOST_TIME_BRACKET;
    clock_samples[0].physical_device_ordinal = kSmokePhysicalDevice;
    clock_samples[0].sample_id = 1;
    clock_samples[0].device_tick = 100;
    clock_samples[0].host_cpu_timestamp_ns = 1000;
    clock_samples[0].host_time_begin_ns = 900;
    clock_samples[0].host_time_end_ns = 910;
    clock_samples[1] = clock_samples[0];
    clock_samples[1].sample_id = 2;
    clock_samples[1].device_tick = 200;
    clock_samples[1].host_cpu_timestamp_ns = 2000;
    clock_samples[1].host_time_begin_ns = 1900;
    clock_samples[1].host_time_end_ns = 1910;
    status = write_profile_chunk(
        sink, IREE_HAL_PROFILE_CONTENT_TYPE_CLOCK_CORRELATIONS,
        IREE_SV("clock-correlations"),
        iree_make_const_byte_span(clock_samples, sizeof(clock_samples)));
  }

  if (iree_status_is_ok(status)) {
    iree_hal_profile_dispatch_event_t dispatch =
        iree_hal_profile_dispatch_event_default();
    dispatch.flags = IREE_HAL_PROFILE_DISPATCH_EVENT_FLAG_COMMAND_BUFFER;
    dispatch.event_id = 200;
    dispatch.submission_id = kSmokeSubmissionId;
    dispatch.command_buffer_id = kSmokeCommandBufferId;
    dispatch.executable_id = kSmokeExecutableId;
    dispatch.command_index = kSmokeCommandIndex;
    dispatch.export_ordinal = kSmokeExportOrdinal;
    dispatch.workgroup_count[0] = 4;
    dispatch.workgroup_count[1] = 5;
    dispatch.workgroup_count[2] = 6;
    dispatch.workgroup_size[0] = 7;
    dispatch.workgroup_size[1] = 8;
    dispatch.workgroup_size[2] = 9;
    dispatch.start_tick = 120;
    dispatch.end_tick = 160;
    status = write_profile_chunk(
        sink, IREE_HAL_PROFILE_CONTENT_TYPE_DISPATCH_EVENTS,
        IREE_SV("dispatch-events"),
        iree_make_const_byte_span(&dispatch, sizeof(dispatch)));
  }

  if (iree_status_is_ok(status)) {
    iree_hal_profile_queue_event_t queue_event =
        iree_hal_profile_queue_event_default();
    queue_event.type = IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_EXECUTE;
    queue_event.dependency_strategy =
        IREE_HAL_PROFILE_QUEUE_DEPENDENCY_STRATEGY_DEVICE_BARRIER;
    queue_event.event_id = 100;
    queue_event.host_time_ns = 3000;
    queue_event.ready_host_time_ns = 3050;
    queue_event.submission_id = kSmokeSubmissionId;
    queue_event.command_buffer_id = kSmokeCommandBufferId;
    queue_event.stream_id = kSmokeStreamId;
    queue_event.physical_device_ordinal = kSmokePhysicalDevice;
    queue_event.queue_ordinal = kSmokeQueueOrdinal;
    queue_event.wait_count = 1;
    queue_event.signal_count = 1;
    queue_event.barrier_count = 2;
    queue_event.operation_count = 1;
    queue_event.payload_length = 4096;
    status = write_profile_chunk(
        sink, IREE_HAL_PROFILE_CONTENT_TYPE_QUEUE_EVENTS,
        IREE_SV("queue-events"),
        iree_make_const_byte_span(&queue_event, sizeof(queue_event)));
  }

  if (iree_status_is_ok(status)) {
    iree_hal_profile_queue_device_event_t device_event =
        iree_hal_profile_queue_device_event_default();
    device_event.type = IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_EXECUTE;
    device_event.event_id = 101;
    device_event.submission_id = kSmokeSubmissionId;
    device_event.command_buffer_id = kSmokeCommandBufferId;
    device_event.stream_id = kSmokeStreamId;
    device_event.payload_length = 4096;
    device_event.physical_device_ordinal = kSmokePhysicalDevice;
    device_event.queue_ordinal = kSmokeQueueOrdinal;
    device_event.operation_count = 1;
    device_event.start_tick = 120;
    device_event.end_tick = 160;
    status = write_profile_chunk(
        sink, IREE_HAL_PROFILE_CONTENT_TYPE_QUEUE_DEVICE_EVENTS,
        IREE_SV("queue-device-events"),
        iree_make_const_byte_span(&device_event, sizeof(device_event)));
  }

  if (iree_status_is_ok(status)) {
    iree_hal_profile_host_execution_event_t host_events[2];
    host_events[0] = iree_hal_profile_host_execution_event_default();
    host_events[0].type = IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_EXECUTE;
    host_events[0].status_code = IREE_STATUS_OK;
    host_events[0].event_id = 102;
    host_events[0].submission_id = kSmokeSubmissionId;
    host_events[0].command_buffer_id = kSmokeCommandBufferId;
    host_events[0].executable_id = kSmokeExecutableId;
    host_events[0].stream_id = kSmokeStreamId;
    host_events[0].physical_device_ordinal = kSmokePhysicalDevice;
    host_events[0].queue_ordinal = kSmokeQueueOrdinal;
    host_events[0].command_index = kSmokeCommandIndex;
    host_events[0].export_ordinal = kSmokeExportOrdinal;
    host_events[0].start_host_time_ns = 3100;
    host_events[0].end_host_time_ns = 3150;
    host_events[0].payload_length = 4096;
    host_events[0].operation_count = 1;
    host_events[1] = host_events[0];
    host_events[1].type = IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_DISPATCH;
    host_events[1].flags =
        IREE_HAL_PROFILE_HOST_EXECUTION_EVENT_FLAG_COMMAND_BUFFER;
    host_events[1].event_id = 103;
    host_events[1].start_host_time_ns = 3110;
    host_events[1].end_host_time_ns = 3140;
    host_events[1].workgroup_count[0] = 4;
    host_events[1].workgroup_count[1] = 5;
    host_events[1].workgroup_count[2] = 6;
    host_events[1].workgroup_size[0] = 7;
    host_events[1].workgroup_size[1] = 8;
    host_events[1].workgroup_size[2] = 9;
    host_events[1].tile_count = 4;
    host_events[1].tile_duration_sum_ns = 30;
    status = write_profile_chunk(
        sink, IREE_HAL_PROFILE_CONTENT_TYPE_HOST_EXECUTION_EVENTS,
        IREE_SV("host-execution-events"),
        iree_make_const_byte_span(host_events, sizeof(host_events)));
  }

  if (iree_status_is_ok(status)) {
    iree_hal_profile_memory_event_t memory_events[2];
    memory_events[0] = iree_hal_profile_memory_event_default();
    memory_events[0].type = IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_BUFFER_ALLOCATE;
    memory_events[0].result = IREE_STATUS_OK;
    memory_events[0].event_id = 300;
    memory_events[0].host_time_ns = 3200;
    memory_events[0].allocation_id = kSmokeAllocationId;
    memory_events[0].pool_id = 23;
    memory_events[0].backing_id = 29;
    memory_events[0].physical_device_ordinal = kSmokePhysicalDevice;
    memory_events[0].length = 2048;
    memory_events[0].alignment = 64;
    memory_events[1] = memory_events[0];
    memory_events[1].type = IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_BUFFER_FREE;
    memory_events[1].event_id = 301;
    memory_events[1].host_time_ns = 3300;
    status = write_profile_chunk(
        sink, IREE_HAL_PROFILE_CONTENT_TYPE_MEMORY_EVENTS,
        IREE_SV("memory-events"),
        iree_make_const_byte_span(memory_events, sizeof(memory_events)));
  }

  if (iree_status_is_ok(status)) {
    const char source_name[] = "smoke.metrics";
    iree_hal_profile_device_metric_source_record_t source_record =
        iree_hal_profile_device_metric_source_record_default();
    source_record.source_id = kSmokeMetricSourceId;
    source_record.physical_device_ordinal = kSmokePhysicalDevice;
    source_record.device_class = IREE_HAL_PROFILE_DEVICE_CLASS_GPU;
    source_record.source_kind = 1;
    source_record.source_revision = 1;
    source_record.metric_count = 2;
    source_record.name_length = sizeof(source_name) - 1;
    source_record.record_length =
        sizeof(source_record) + source_record.name_length;
    iree_const_byte_span_t iovecs[2] = {
        iree_make_const_byte_span(&source_record, sizeof(source_record)),
        iree_make_const_byte_span(source_name, source_record.name_length),
    };
    status = write_profile_chunk_iovecs(
        sink, IREE_HAL_PROFILE_CONTENT_TYPE_DEVICE_METRIC_SOURCES,
        IREE_SV("device-metric-sources"), IREE_ARRAYSIZE(iovecs), iovecs);
  }

  if (iree_status_is_ok(status)) {
    const iree_hal_profile_metric_descriptor_t* builtin_metric =
        iree_hal_profile_builtin_metric_descriptor_lookup(
            IREE_HAL_PROFILE_BUILTIN_METRIC_ID_ACTIVITY_COMPUTE);
    if (!builtin_metric) {
      status = iree_make_status(IREE_STATUS_INTERNAL,
                                "missing built-in metric descriptor");
    } else {
      const char source_specific_name[] = "smoke.source_specific";
      const char source_specific_description[] =
          "Smoke fixture source-specific metric.";
      iree_hal_profile_device_metric_descriptor_record_t builtin_record =
          iree_hal_profile_device_metric_descriptor_record_default();
      builtin_record.source_id = kSmokeMetricSourceId;
      builtin_record.metric_id = builtin_metric->metric_id;
      builtin_record.unit = builtin_metric->unit;
      builtin_record.value_kind = builtin_metric->value_kind;
      builtin_record.semantic = builtin_metric->semantic;
      builtin_record.plot_hint = builtin_metric->plot_hint;
      builtin_record.name_length = builtin_metric->name.size;
      builtin_record.description_length = builtin_metric->description.size;
      builtin_record.record_length = sizeof(builtin_record) +
                                     builtin_record.name_length +
                                     builtin_record.description_length;

      iree_hal_profile_device_metric_descriptor_record_t
          source_specific_record =
              iree_hal_profile_device_metric_descriptor_record_default();
      source_specific_record.source_id = kSmokeMetricSourceId;
      source_specific_record.metric_id = kSmokeSourceSpecificMetricId;
      source_specific_record.unit = IREE_HAL_PROFILE_METRIC_UNIT_COUNT;
      source_specific_record.value_kind =
          IREE_HAL_PROFILE_METRIC_VALUE_KIND_U64;
      source_specific_record.semantic =
          IREE_HAL_PROFILE_METRIC_SEMANTIC_INSTANT;
      source_specific_record.plot_hint =
          IREE_HAL_PROFILE_METRIC_PLOT_HINT_NUMBER;
      source_specific_record.name_length = sizeof(source_specific_name) - 1;
      source_specific_record.description_length =
          sizeof(source_specific_description) - 1;
      source_specific_record.record_length =
          sizeof(source_specific_record) + source_specific_record.name_length +
          source_specific_record.description_length;

      iree_const_byte_span_t iovecs[6] = {
          iree_make_const_byte_span(&builtin_record, sizeof(builtin_record)),
          iree_make_const_byte_span(builtin_metric->name.data,
                                    builtin_metric->name.size),
          iree_make_const_byte_span(builtin_metric->description.data,
                                    builtin_metric->description.size),
          iree_make_const_byte_span(&source_specific_record,
                                    sizeof(source_specific_record)),
          iree_make_const_byte_span(source_specific_name,
                                    source_specific_record.name_length),
          iree_make_const_byte_span(source_specific_description,
                                    source_specific_record.description_length),
      };
      status = write_profile_chunk_iovecs(
          sink, IREE_HAL_PROFILE_CONTENT_TYPE_DEVICE_METRIC_DESCRIPTORS,
          IREE_SV("device-metric-descriptors"), IREE_ARRAYSIZE(iovecs), iovecs);
    }
  }

  if (iree_status_is_ok(status)) {
    iree_hal_profile_device_metric_sample_record_t sample_record =
        iree_hal_profile_device_metric_sample_record_default();
    sample_record.flags =
        IREE_HAL_PROFILE_DEVICE_METRIC_SAMPLE_FLAG_SOURCE_TIMESTAMP;
    sample_record.sample_id = 1;
    sample_record.source_id = kSmokeMetricSourceId;
    sample_record.host_time_begin_ns = 3400;
    sample_record.host_time_end_ns = 3410;
    sample_record.source_timestamp = 500;
    sample_record.source_timestamp_frequency_hz = 1000000;
    sample_record.physical_device_ordinal = kSmokePhysicalDevice;
    sample_record.value_count = 2;
    sample_record.record_length =
        sizeof(sample_record) +
        sample_record.value_count *
            sizeof(iree_hal_profile_device_metric_value_t);
    iree_hal_profile_device_metric_value_t values[2] = {
        {
            .metric_id = IREE_HAL_PROFILE_BUILTIN_METRIC_ID_ACTIVITY_COMPUTE,
            .value_bits = 87500,
            .flags = IREE_HAL_PROFILE_DEVICE_METRIC_VALUE_FLAG_NONE,
        },
        {
            .metric_id = kSmokeSourceSpecificMetricId,
            .value_bits = 12345,
            .flags = IREE_HAL_PROFILE_DEVICE_METRIC_VALUE_FLAG_NONE,
        },
    };
    iree_const_byte_span_t iovecs[2] = {
        iree_make_const_byte_span(&sample_record, sizeof(sample_record)),
        iree_make_const_byte_span(values, sizeof(values)),
    };
    status = write_profile_chunk_iovecs(
        sink, IREE_HAL_PROFILE_CONTENT_TYPE_DEVICE_METRIC_SAMPLES,
        IREE_SV("device-metric-samples"), IREE_ARRAYSIZE(iovecs), iovecs);
  }

  if (iree_status_is_ok(status)) {
    iree_hal_profile_event_relationship_record_t relationship =
        iree_hal_profile_event_relationship_record_default();
    relationship.type =
        IREE_HAL_PROFILE_EVENT_RELATIONSHIP_TYPE_QUEUE_EVENT_HOST_EXECUTION_EVENT;
    relationship.relationship_id = 1;
    relationship.source_type = IREE_HAL_PROFILE_EVENT_ENDPOINT_TYPE_QUEUE_EVENT;
    relationship.target_type =
        IREE_HAL_PROFILE_EVENT_ENDPOINT_TYPE_HOST_EXECUTION_EVENT;
    relationship.physical_device_ordinal = kSmokePhysicalDevice;
    relationship.queue_ordinal = kSmokeQueueOrdinal;
    relationship.stream_id = kSmokeStreamId;
    relationship.source_id = 100;
    relationship.target_id = 102;
    status = write_profile_chunk(
        sink, IREE_HAL_PROFILE_CONTENT_TYPE_EVENT_RELATIONSHIPS,
        IREE_SV("event-relationships"),
        iree_make_const_byte_span(&relationship, sizeof(relationship)));
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_profile_sink_end_session(sink, &session_metadata,
                                               IREE_STATUS_OK);
  }
  iree_hal_profile_sink_release(sink);
  return status;
}

int main(int argc, char** argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: %s OUTPUT.ireeprof\n", argv[0]);
    return EXIT_FAILURE;
  }

  iree_status_t status = write_smoke_profile(iree_make_cstring_view(argv[1]));
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
