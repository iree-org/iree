// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/utils/statistics_sink.h"

#include <cstdint>
#include <cstring>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

struct CollectedRows {
  // Aggregate rows copied from the statistics sink.
  std::vector<iree_hal_profile_statistics_row_t> rows;
};

static iree_status_t CollectRow(void* user_data,
                                const iree_hal_profile_statistics_row_t* row) {
  reinterpret_cast<CollectedRows*>(user_data)->rows.push_back(*row);
  return iree_ok_status();
}

static CollectedRows CollectRows(
    iree_hal_profile_statistics_sink_t* statistics_sink) {
  CollectedRows rows;
  iree_hal_profile_statistics_row_callback_t callback = {
      .fn = CollectRow,
      .user_data = &rows,
  };
  IREE_EXPECT_OK(
      iree_hal_profile_statistics_sink_for_each_row(statistics_sink, callback));
  return rows;
}

static const iree_hal_profile_statistics_row_t* FindRow(
    const CollectedRows& rows, iree_hal_profile_statistics_row_type_t row_type,
    uint64_t executable_id, uint32_t export_ordinal, uint64_t command_buffer_id,
    uint32_t command_index, uint32_t event_type) {
  for (const auto& row : rows.rows) {
    if (row.row_type == row_type && row.executable_id == executable_id &&
        row.export_ordinal == export_ordinal &&
        row.command_buffer_id == command_buffer_id &&
        row.command_index == command_index && row.event_type == event_type) {
      return &row;
    }
  }
  return nullptr;
}

static void WriteChunk(iree_hal_profile_sink_t* sink,
                       iree_string_view_t content_type,
                       uint32_t physical_device_ordinal, uint32_t queue_ordinal,
                       const void* data, iree_host_size_t data_length) {
  iree_hal_profile_chunk_metadata_t metadata =
      iree_hal_profile_chunk_metadata_default();
  metadata.content_type = content_type;
  metadata.physical_device_ordinal = physical_device_ordinal;
  metadata.queue_ordinal = queue_ordinal;
  iree_const_byte_span_t iovec = iree_make_const_byte_span(data, data_length);
  IREE_ASSERT_OK(iree_hal_profile_sink_write(sink, &metadata, 1, &iovec));
}

static std::vector<uint8_t> MakeExportRecord(uint64_t executable_id,
                                             uint32_t export_ordinal,
                                             const char* name) {
  const iree_host_size_t name_length = strlen(name);
  std::vector<uint8_t> storage(
      sizeof(iree_hal_profile_executable_export_record_t) + name_length);
  iree_hal_profile_executable_export_record_t record =
      iree_hal_profile_executable_export_record_default();
  record.record_length = static_cast<uint32_t>(storage.size());
  record.executable_id = executable_id;
  record.export_ordinal = export_ordinal;
  record.name_length = static_cast<uint32_t>(name_length);
  memcpy(storage.data(), &record, sizeof(record));
  memcpy(storage.data() + sizeof(record), name, name_length);
  return storage;
}

TEST(StatisticsSinkTest, AggregatesDispatchEventsByExport) {
  iree_hal_profile_statistics_sink_t* statistics_sink = nullptr;
  IREE_ASSERT_OK(iree_hal_profile_statistics_sink_create(
      iree_allocator_system(), &statistics_sink));
  iree_hal_profile_sink_t* sink =
      iree_hal_profile_statistics_sink_base(statistics_sink);

  iree_hal_profile_chunk_metadata_t session_metadata =
      iree_hal_profile_chunk_metadata_default();
  session_metadata.content_type = IREE_HAL_PROFILE_CONTENT_TYPE_SESSION;
  IREE_ASSERT_OK(iree_hal_profile_sink_begin_session(sink, &session_metadata));

  std::vector<uint8_t> export_record = MakeExportRecord(
      /*executable_id=*/7, /*export_ordinal=*/3, "kernel_main");
  WriteChunk(sink, IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLE_EXPORTS,
             /*physical_device_ordinal=*/0, /*queue_ordinal=*/0,
             export_record.data(), export_record.size());

  iree_hal_profile_clock_correlation_record_t clock_samples[2];
  clock_samples[0] = iree_hal_profile_clock_correlation_record_default();
  clock_samples[0].flags =
      IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_DEVICE_TICK |
      IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_HOST_CPU_TIMESTAMP;
  clock_samples[0].physical_device_ordinal = 2;
  clock_samples[0].sample_id = 1;
  clock_samples[0].device_tick = 10;
  clock_samples[0].host_cpu_timestamp_ns = 1000;
  clock_samples[1] = iree_hal_profile_clock_correlation_record_default();
  clock_samples[1].flags =
      IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_DEVICE_TICK |
      IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_HOST_CPU_TIMESTAMP;
  clock_samples[1].physical_device_ordinal = 2;
  clock_samples[1].sample_id = 2;
  clock_samples[1].device_tick = 110;
  clock_samples[1].host_cpu_timestamp_ns = 1500;
  WriteChunk(sink, IREE_HAL_PROFILE_CONTENT_TYPE_CLOCK_CORRELATIONS,
             /*physical_device_ordinal=*/0, /*queue_ordinal=*/0, clock_samples,
             sizeof(clock_samples));

  iree_hal_profile_dispatch_event_t events[2];
  events[0] = iree_hal_profile_dispatch_event_default();
  events[0].executable_id = 7;
  events[0].export_ordinal = 3;
  events[0].start_tick = 10;
  events[0].end_tick = 30;
  events[1] = iree_hal_profile_dispatch_event_default();
  events[1].executable_id = 7;
  events[1].export_ordinal = 3;
  events[1].start_tick = 35;
  events[1].end_tick = 45;
  WriteChunk(sink, IREE_HAL_PROFILE_CONTENT_TYPE_DISPATCH_EVENTS,
             /*physical_device_ordinal=*/2, /*queue_ordinal=*/1, events,
             sizeof(events));

  CollectedRows rows = CollectRows(statistics_sink);
  ASSERT_EQ(rows.rows.size(), 1u);
  const iree_hal_profile_statistics_row_t* row = FindRow(
      rows, IREE_HAL_PROFILE_STATISTICS_ROW_TYPE_DISPATCH_EXPORT,
      /*executable_id=*/7, /*export_ordinal=*/3, /*command_buffer_id=*/0,
      /*command_index=*/UINT32_MAX, /*event_type=*/0);
  ASSERT_NE(row, nullptr);
  EXPECT_EQ(row->time_domain,
            IREE_HAL_PROFILE_STATISTICS_TIME_DOMAIN_DEVICE_TICK);
  EXPECT_EQ(row->physical_device_ordinal, 2u);
  EXPECT_EQ(row->queue_ordinal, 1u);
  EXPECT_EQ(row->sample_count, 2u);
  EXPECT_EQ(row->invalid_sample_count, 0u);
  EXPECT_EQ(row->total_duration, 30u);
  EXPECT_EQ(row->minimum_duration, 10u);
  EXPECT_EQ(row->maximum_duration, 20u);

  uint64_t total_duration_ns = 0;
  ASSERT_TRUE(iree_hal_profile_statistics_sink_scale_duration_to_ns(
      statistics_sink, row, row->total_duration, &total_duration_ns));
  EXPECT_EQ(total_duration_ns, 150u);

  iree_string_view_t name = iree_string_view_empty();
  ASSERT_TRUE(iree_hal_profile_statistics_sink_find_export_name(
      statistics_sink, /*executable_id=*/7, /*export_ordinal=*/3, &name));
  EXPECT_TRUE(iree_string_view_equal(name, IREE_SV("kernel_main")));

  iree_hal_profile_statistics_sink_release(statistics_sink);
}

TEST(StatisticsSinkTest, AggregatesHostAndQueueDeviceEvents) {
  iree_hal_profile_statistics_sink_t* statistics_sink = nullptr;
  IREE_ASSERT_OK(iree_hal_profile_statistics_sink_create(
      iree_allocator_system(), &statistics_sink));
  iree_hal_profile_sink_t* sink =
      iree_hal_profile_statistics_sink_base(statistics_sink);

  iree_hal_profile_chunk_metadata_t session_metadata =
      iree_hal_profile_chunk_metadata_default();
  session_metadata.content_type = IREE_HAL_PROFILE_CONTENT_TYPE_SESSION;
  IREE_ASSERT_OK(iree_hal_profile_sink_begin_session(sink, &session_metadata));

  iree_hal_profile_host_execution_event_t host_event =
      iree_hal_profile_host_execution_event_default();
  host_event.type = IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_DISPATCH;
  host_event.executable_id = 5;
  host_event.export_ordinal = 2;
  host_event.command_buffer_id = 11;
  host_event.command_index = 4;
  host_event.physical_device_ordinal = 0;
  host_event.queue_ordinal = 1;
  host_event.operation_count = 1;
  host_event.payload_length = 256;
  host_event.tile_count = 3;
  host_event.tile_duration_sum_ns = 90;
  host_event.start_host_time_ns = 100;
  host_event.end_host_time_ns = 160;
  WriteChunk(sink, IREE_HAL_PROFILE_CONTENT_TYPE_HOST_EXECUTION_EVENTS,
             /*physical_device_ordinal=*/0, /*queue_ordinal=*/1, &host_event,
             sizeof(host_event));

  iree_hal_profile_queue_device_event_t queue_event =
      iree_hal_profile_queue_device_event_default();
  queue_event.type = IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_EXECUTE;
  queue_event.physical_device_ordinal = 0;
  queue_event.queue_ordinal = 1;
  queue_event.command_buffer_id = 11;
  queue_event.operation_count = 7;
  queue_event.payload_length = 4096;
  queue_event.start_tick = 1000;
  queue_event.end_tick = 1100;
  WriteChunk(sink, IREE_HAL_PROFILE_CONTENT_TYPE_QUEUE_DEVICE_EVENTS,
             /*physical_device_ordinal=*/0, /*queue_ordinal=*/1, &queue_event,
             sizeof(queue_event));

  CollectedRows rows = CollectRows(statistics_sink);
  const iree_hal_profile_statistics_row_t* export_row = FindRow(
      rows, IREE_HAL_PROFILE_STATISTICS_ROW_TYPE_HOST_EXECUTION_EXPORT,
      /*executable_id=*/5, /*export_ordinal=*/2, /*command_buffer_id=*/0,
      /*command_index=*/UINT32_MAX, /*event_type=*/0);
  ASSERT_NE(export_row, nullptr);
  EXPECT_EQ(export_row->total_duration, 60u);
  EXPECT_EQ(export_row->tile_count, 3u);
  EXPECT_EQ(export_row->tile_duration_sum_ns, 90u);

  const iree_hal_profile_statistics_row_t* command_row = FindRow(
      rows,
      IREE_HAL_PROFILE_STATISTICS_ROW_TYPE_HOST_EXECUTION_COMMAND_OPERATION,
      /*executable_id=*/5, /*export_ordinal=*/2, /*command_buffer_id=*/11,
      /*command_index=*/4, /*event_type=*/0);
  ASSERT_NE(command_row, nullptr);
  EXPECT_EQ(command_row->sample_count, 1u);
  EXPECT_EQ(command_row->payload_bytes, 256u);

  const iree_hal_profile_statistics_row_t* queue_row =
      FindRow(rows, IREE_HAL_PROFILE_STATISTICS_ROW_TYPE_QUEUE_DEVICE_OPERATION,
              /*executable_id=*/0, /*export_ordinal=*/UINT32_MAX,
              /*command_buffer_id=*/11, /*command_index=*/UINT32_MAX,
              IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_EXECUTE);
  ASSERT_NE(queue_row, nullptr);
  EXPECT_EQ(queue_row->time_domain,
            IREE_HAL_PROFILE_STATISTICS_TIME_DOMAIN_DEVICE_TICK);
  EXPECT_EQ(queue_row->total_duration, 100u);
  EXPECT_EQ(queue_row->operation_count, 7u);
  EXPECT_EQ(queue_row->payload_bytes, 4096u);

  iree_hal_profile_statistics_sink_release(statistics_sink);
}

TEST(StatisticsSinkTest, CountsDroppedRecords) {
  iree_hal_profile_statistics_sink_t* statistics_sink = nullptr;
  IREE_ASSERT_OK(iree_hal_profile_statistics_sink_create(
      iree_allocator_system(), &statistics_sink));
  iree_hal_profile_sink_t* sink =
      iree_hal_profile_statistics_sink_base(statistics_sink);

  iree_hal_profile_chunk_metadata_t session_metadata =
      iree_hal_profile_chunk_metadata_default();
  session_metadata.content_type = IREE_HAL_PROFILE_CONTENT_TYPE_SESSION;
  IREE_ASSERT_OK(iree_hal_profile_sink_begin_session(sink, &session_metadata));

  iree_hal_profile_chunk_metadata_t metadata =
      iree_hal_profile_chunk_metadata_default();
  metadata.content_type = IREE_HAL_PROFILE_CONTENT_TYPE_QUEUE_EVENTS;
  metadata.flags = IREE_HAL_PROFILE_CHUNK_FLAG_TRUNCATED;
  metadata.dropped_record_count = 7;
  IREE_ASSERT_OK(iree_hal_profile_sink_write(sink, &metadata,
                                             /*iovec_count=*/0,
                                             /*iovecs=*/nullptr));

  EXPECT_EQ(iree_hal_profile_statistics_sink_row_count(statistics_sink), 0u);
  EXPECT_EQ(
      iree_hal_profile_statistics_sink_dropped_record_count(statistics_sink),
      7u);

  iree_hal_profile_statistics_sink_release(statistics_sink);
}

}  // namespace
