// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tooling/profile/summary.h"

#include <cstdint>
#include <cstring>
#include <vector>

#include "iree/hal/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

static iree_hal_profile_file_record_t MakeChunk(
    const std::vector<uint8_t>& payload) {
  iree_hal_profile_file_record_t chunk;
  memset(&chunk, 0, sizeof(chunk));
  chunk.header.record_type = IREE_HAL_PROFILE_FILE_RECORD_TYPE_CHUNK;
  chunk.content_type = IREE_SV("application/vnd.iree.test");
  chunk.payload = iree_make_const_byte_span(payload.data(), payload.size());
  return chunk;
}

static iree_hal_profile_file_record_t MakeDispatchEventsChunk(
    const std::vector<uint8_t>& payload) {
  iree_hal_profile_file_record_t chunk = MakeChunk(payload);
  chunk.content_type = IREE_HAL_PROFILE_CONTENT_TYPE_DISPATCH_EVENTS;
  return chunk;
}

static iree_hal_profile_file_record_t MakeHostExecutionEventsChunk(
    const std::vector<uint8_t>& payload) {
  iree_hal_profile_file_record_t chunk = MakeChunk(payload);
  chunk.content_type = IREE_HAL_PROFILE_CONTENT_TYPE_HOST_EXECUTION_EVENTS;
  return chunk;
}

static iree_hal_profile_file_record_t MakeDeviceMetricSourcesChunk(
    const std::vector<uint8_t>& payload) {
  iree_hal_profile_file_record_t chunk = MakeChunk(payload);
  chunk.content_type = IREE_HAL_PROFILE_CONTENT_TYPE_DEVICE_METRIC_SOURCES;
  return chunk;
}

static iree_hal_profile_file_record_t MakeDeviceMetricDescriptorsChunk(
    const std::vector<uint8_t>& payload) {
  iree_hal_profile_file_record_t chunk = MakeChunk(payload);
  chunk.content_type = IREE_HAL_PROFILE_CONTENT_TYPE_DEVICE_METRIC_DESCRIPTORS;
  return chunk;
}

static iree_hal_profile_file_record_t MakeDeviceMetricSamplesChunk(
    const std::vector<uint8_t>& payload) {
  iree_hal_profile_file_record_t chunk = MakeChunk(payload);
  chunk.content_type = IREE_HAL_PROFILE_CONTENT_TYPE_DEVICE_METRIC_SAMPLES;
  return chunk;
}

template <typename T>
static void AppendPlainRecord(std::vector<uint8_t>* payload, const T& record) {
  const iree_host_size_t offset = payload->size();
  payload->resize(offset + sizeof(record));
  memcpy(payload->data() + offset, &record, sizeof(record));
}

template <typename T>
static void AppendInlineRecord(std::vector<uint8_t>* payload, const T& record,
                               iree_const_byte_span_t inline_payload) {
  const iree_host_size_t offset = payload->size();
  payload->resize(offset + sizeof(record) + inline_payload.data_length);
  memcpy(payload->data() + offset, &record, sizeof(record));
  memcpy(payload->data() + offset + sizeof(record), inline_payload.data,
         inline_payload.data_length);
}

TEST(ProfileSummaryTest, AccumulatesDroppedRecordsFromTruncatedChunks) {
  std::vector<uint8_t> payload;
  iree_hal_profile_file_record_t chunk = MakeChunk(payload);
  chunk.header.record_type = IREE_HAL_PROFILE_FILE_RECORD_TYPE_CHUNK;
  chunk.header.chunk_flags = IREE_HAL_PROFILE_CHUNK_FLAG_TRUNCATED;
  chunk.header.dropped_record_count = 7;

  iree_profile_summary_t summary;
  iree_profile_summary_initialize(iree_allocator_system(), &summary);
  IREE_ASSERT_OK(iree_profile_summary_process_record(&summary, &chunk));
  EXPECT_EQ(1u, summary.truncated_chunk_count);
  EXPECT_EQ(7u, summary.dropped_record_count);
  iree_profile_summary_deinitialize(&summary);
}

TEST(ProfileSummaryTest, RejectsMalformedHostExecutionEventRecord) {
  std::vector<uint8_t> payload(sizeof(uint32_t), 0);
  iree_hal_profile_file_record_t chunk = MakeHostExecutionEventsChunk(payload);

  iree_profile_summary_t summary;
  iree_profile_summary_initialize(iree_allocator_system(), &summary);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS,
                        iree_profile_summary_process_record(&summary, &chunk));

  EXPECT_EQ(1u, summary.chunk_count);
  EXPECT_EQ(1u, summary.host_execution_event_chunk_count);
  EXPECT_EQ(0u, summary.host_execution_event_record_count);
  EXPECT_EQ(0u, summary.invalid_host_execution_event_record_count);
  EXPECT_EQ(0u, summary.total_host_execution_duration_ns);

  iree_profile_summary_deinitialize(&summary);
}

TEST(ProfileSummaryTest, RecordsFirstNonOkSessionEndStatus) {
  iree_hal_profile_file_record_t first_session_end;
  memset(&first_session_end, 0, sizeof(first_session_end));
  first_session_end.header.record_type =
      IREE_HAL_PROFILE_FILE_RECORD_TYPE_SESSION_END;
  first_session_end.header.session_id = 42;
  first_session_end.header.stream_id = 7;
  first_session_end.header.event_id = 8;
  first_session_end.header.session_status_code = IREE_STATUS_CANCELLED;

  iree_hal_profile_file_record_t second_session_end = first_session_end;
  second_session_end.header.session_id = 99;
  second_session_end.header.session_status_code =
      IREE_STATUS_RESOURCE_EXHAUSTED;

  iree_profile_summary_t summary;
  iree_profile_summary_initialize(iree_allocator_system(), &summary);
  IREE_ASSERT_OK(
      iree_profile_summary_process_record(&summary, &first_session_end));
  IREE_ASSERT_OK(
      iree_profile_summary_process_record(&summary, &second_session_end));

  EXPECT_EQ(2u, summary.session_end_count);
  EXPECT_EQ(2u, summary.non_ok_session_end_count);
  EXPECT_EQ(42u, summary.first_non_ok_session_id);
  EXPECT_EQ(7u, summary.first_non_ok_stream_id);
  EXPECT_EQ(8u, summary.first_non_ok_event_id);
  EXPECT_EQ(IREE_STATUS_CANCELLED, summary.first_non_ok_session_status_code);
  iree_profile_summary_deinitialize(&summary);
}

TEST(ProfileSummaryTest, RecordsDeviceMetricMetadataAndSamples) {
  const char source_name[] = "summary.metrics";
  iree_hal_profile_device_metric_source_record_t source_record =
      iree_hal_profile_device_metric_source_record_default();
  source_record.source_id = 1;
  source_record.physical_device_ordinal = 2;
  source_record.device_class = IREE_HAL_PROFILE_DEVICE_CLASS_GPU;
  source_record.metric_count = 1;
  source_record.name_length = sizeof(source_name) - 1;
  source_record.record_length =
      sizeof(source_record) + source_record.name_length;
  std::vector<uint8_t> source_payload;
  AppendInlineRecord(
      &source_payload, source_record,
      iree_make_const_byte_span(source_name, source_record.name_length));

  const char metric_name[] = "summary.metric";
  const char metric_description[] = "Summary metric.";
  iree_hal_profile_device_metric_descriptor_record_t descriptor_record =
      iree_hal_profile_device_metric_descriptor_record_default();
  descriptor_record.source_id = source_record.source_id;
  descriptor_record.metric_id = IREE_HAL_PROFILE_METRIC_ID_PRODUCER_BASE + 7;
  descriptor_record.unit = IREE_HAL_PROFILE_METRIC_UNIT_COUNT;
  descriptor_record.value_kind = IREE_HAL_PROFILE_METRIC_VALUE_KIND_U64;
  descriptor_record.semantic = IREE_HAL_PROFILE_METRIC_SEMANTIC_INSTANT;
  descriptor_record.plot_hint = IREE_HAL_PROFILE_METRIC_PLOT_HINT_NUMBER;
  descriptor_record.name_length = sizeof(metric_name) - 1;
  descriptor_record.description_length = sizeof(metric_description) - 1;
  descriptor_record.record_length = sizeof(descriptor_record) +
                                    descriptor_record.name_length +
                                    descriptor_record.description_length;
  std::vector<uint8_t> descriptor_payload;
  AppendInlineRecord(
      &descriptor_payload, descriptor_record,
      iree_make_const_byte_span(metric_name, descriptor_record.name_length));
  descriptor_payload.insert(
      descriptor_payload.end(), metric_description,
      metric_description + descriptor_record.description_length);

  iree_hal_profile_device_metric_sample_record_t sample_record =
      iree_hal_profile_device_metric_sample_record_default();
  sample_record.sample_id = 1;
  sample_record.source_id = source_record.source_id;
  sample_record.physical_device_ordinal = source_record.physical_device_ordinal;
  sample_record.host_time_begin_ns = 100;
  sample_record.host_time_end_ns = 110;
  sample_record.value_count = 1;
  sample_record.record_length =
      sizeof(sample_record) + sizeof(iree_hal_profile_device_metric_value_t);
  iree_hal_profile_device_metric_value_t sample_value = {
      .metric_id = descriptor_record.metric_id,
      .value_bits = 42,
      .flags = IREE_HAL_PROFILE_DEVICE_METRIC_VALUE_FLAG_NONE,
  };
  std::vector<uint8_t> sample_payload;
  AppendInlineRecord(
      &sample_payload, sample_record,
      iree_make_const_byte_span(&sample_value, sizeof(sample_value)));

  iree_profile_summary_t summary;
  iree_profile_summary_initialize(iree_allocator_system(), &summary);
  iree_hal_profile_file_record_t source_chunk =
      MakeDeviceMetricSourcesChunk(source_payload);
  iree_hal_profile_file_record_t descriptor_chunk =
      MakeDeviceMetricDescriptorsChunk(descriptor_payload);
  iree_hal_profile_file_record_t sample_chunk =
      MakeDeviceMetricSamplesChunk(sample_payload);
  IREE_ASSERT_OK(iree_profile_summary_process_record(&summary, &source_chunk));
  IREE_ASSERT_OK(
      iree_profile_summary_process_record(&summary, &descriptor_chunk));
  IREE_ASSERT_OK(iree_profile_summary_process_record(&summary, &sample_chunk));

  EXPECT_EQ(1u, summary.device_metric.source_record_count);
  EXPECT_EQ(1u, summary.device_metric.descriptor_record_count);
  EXPECT_EQ(1u, summary.device_metric.sample_record_count);
  EXPECT_EQ(1u, summary.device_metric.value_count);
  ASSERT_EQ(1u, summary.device_count);
  EXPECT_EQ(2u, summary.devices[0].physical_device_ordinal);
  EXPECT_EQ(1u, summary.devices[0].device_metric.sample_count);
  EXPECT_EQ(1u, summary.devices[0].device_metric.value_count);
  EXPECT_EQ(0u, summary.devices[0].device_metric.invalid_sample_count);

  iree_profile_summary_deinitialize(&summary);
}

TEST(ProfileSummaryTest, SeparatesHostDurationsFromDeviceDispatchTicks) {
  std::vector<uint8_t> dispatch_payload;
  iree_hal_profile_dispatch_event_t dispatch =
      iree_hal_profile_dispatch_event_default();
  dispatch.start_tick = 10;
  dispatch.end_tick = 30;
  AppendPlainRecord(&dispatch_payload, dispatch);
  dispatch.start_tick = 50;
  dispatch.end_tick = 40;
  AppendPlainRecord(&dispatch_payload, dispatch);
  iree_hal_profile_file_record_t dispatch_chunk =
      MakeDispatchEventsChunk(dispatch_payload);
  dispatch_chunk.header.physical_device_ordinal = 2;

  std::vector<uint8_t> host_execution_payload;
  iree_hal_profile_host_execution_event_t host_execution =
      iree_hal_profile_host_execution_event_default();
  host_execution.start_host_time_ns = 100;
  host_execution.end_host_time_ns = 175;
  AppendPlainRecord(&host_execution_payload, host_execution);
  host_execution.start_host_time_ns = 200;
  host_execution.end_host_time_ns = 150;
  AppendPlainRecord(&host_execution_payload, host_execution);
  iree_hal_profile_file_record_t host_execution_chunk =
      MakeHostExecutionEventsChunk(host_execution_payload);

  iree_profile_summary_t summary;
  iree_profile_summary_initialize(iree_allocator_system(), &summary);
  IREE_ASSERT_OK(
      iree_profile_summary_process_record(&summary, &dispatch_chunk));
  IREE_ASSERT_OK(
      iree_profile_summary_process_record(&summary, &host_execution_chunk));

  ASSERT_EQ(1u, summary.device_count);
  const iree_profile_device_summary_t* device = &summary.devices[0];
  EXPECT_EQ(2u, device->physical_device_ordinal);
  EXPECT_EQ(2u, device->dispatch_event_count);
  EXPECT_EQ(1u, device->invalid_dispatch_event_count);
  EXPECT_EQ(20u, device->total_dispatch_ticks);
  EXPECT_EQ(0u, summary.queue_device_event_record_count);
  EXPECT_EQ(2u, summary.host_execution_event_record_count);
  EXPECT_EQ(1u, summary.invalid_host_execution_event_record_count);
  EXPECT_EQ(75u, summary.total_host_execution_duration_ns);

  iree_profile_summary_deinitialize(&summary);
}

}  // namespace
