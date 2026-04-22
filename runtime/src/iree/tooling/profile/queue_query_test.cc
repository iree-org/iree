// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tooling/profile/queue_query.h"

#include <cstdint>
#include <cstring>
#include <vector>

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

static iree_hal_profile_file_record_t MakeQueueEventsChunk(
    const std::vector<uint8_t>& payload) {
  iree_hal_profile_file_record_t chunk = MakeChunk(payload);
  chunk.content_type = IREE_HAL_PROFILE_CONTENT_TYPE_QUEUE_EVENTS;
  return chunk;
}

static iree_hal_profile_file_record_t MakeQueueDeviceEventsChunk(
    const std::vector<uint8_t>& payload) {
  iree_hal_profile_file_record_t chunk = MakeChunk(payload);
  chunk.content_type = IREE_HAL_PROFILE_CONTENT_TYPE_QUEUE_DEVICE_EVENTS;
  return chunk;
}

static iree_hal_profile_file_record_t MakeHostExecutionEventsChunk(
    const std::vector<uint8_t>& payload) {
  iree_hal_profile_file_record_t chunk = MakeChunk(payload);
  chunk.content_type = IREE_HAL_PROFILE_CONTENT_TYPE_HOST_EXECUTION_EVENTS;
  return chunk;
}

static iree_hal_profile_file_record_t MakeQueuesChunk(
    const std::vector<uint8_t>& payload) {
  iree_hal_profile_file_record_t chunk = MakeChunk(payload);
  chunk.content_type = IREE_HAL_PROFILE_CONTENT_TYPE_QUEUES;
  return chunk;
}

template <typename T>
static void AppendPlainRecord(std::vector<uint8_t>* payload, const T& record) {
  const iree_host_size_t offset = payload->size();
  payload->resize(offset + sizeof(record));
  memcpy(payload->data() + offset, &record, sizeof(record));
}

TEST(ProfileQueueQueryTest, RecordsZeroPayloadTruncatedChunks) {
  std::vector<uint8_t> payload;
  iree_hal_profile_file_record_t chunk = MakeQueueEventsChunk(payload);
  chunk.header.chunk_flags = IREE_HAL_PROFILE_CHUNK_FLAG_TRUNCATED;
  chunk.header.dropped_record_count = 5;

  iree_profile_model_t model;
  iree_profile_model_initialize(iree_allocator_system(), &model);
  iree_profile_queue_event_query_t query;
  iree_profile_queue_event_query_initialize(iree_allocator_system(), &query);
  IREE_ASSERT_OK(iree_profile_queue_event_query_process_record(
      &query, &model, &chunk, IREE_SV("*"), -1));

  EXPECT_EQ(1u, query.truncated_chunk_count);
  EXPECT_EQ(5u, query.dropped_record_count);
  EXPECT_EQ(0u, query.total_queue_event_count);
  EXPECT_EQ(0u, query.matched_queue_event_count);
  EXPECT_EQ(0u, query.queue_event_count);

  iree_profile_queue_event_query_deinitialize(&query);
  iree_profile_model_deinitialize(&model);
}

TEST(ProfileQueueQueryTest, RejectsMalformedHostExecutionEventRecord) {
  std::vector<uint8_t> payload(sizeof(uint32_t), 0);
  iree_hal_profile_file_record_t chunk = MakeHostExecutionEventsChunk(payload);

  iree_profile_model_t model;
  iree_profile_model_initialize(iree_allocator_system(), &model);
  iree_profile_queue_event_query_t query;
  iree_profile_queue_event_query_initialize(iree_allocator_system(), &query);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS,
                        iree_profile_queue_event_query_process_record(
                            &query, &model, &chunk, IREE_SV("*"), -1));

  EXPECT_EQ(0u, query.total_host_execution_event_count);
  EXPECT_EQ(0u, query.matched_host_execution_event_count);
  EXPECT_EQ(0u, query.host_execution_event_count);

  iree_profile_queue_event_query_deinitialize(&query);
  iree_profile_model_deinitialize(&model);
}

TEST(ProfileQueueQueryTest, SeparatesHostAndDeviceTimedEventFamilies) {
  std::vector<uint8_t> queue_metadata_payload;
  iree_hal_profile_queue_record_t queue =
      iree_hal_profile_queue_record_default();
  queue.physical_device_ordinal = 2;
  queue.queue_ordinal = 3;
  queue.stream_id = 4;
  AppendPlainRecord(&queue_metadata_payload, queue);
  iree_hal_profile_file_record_t queue_metadata_chunk =
      MakeQueuesChunk(queue_metadata_payload);

  std::vector<uint8_t> queue_event_payload;
  iree_hal_profile_queue_event_t queue_event =
      iree_hal_profile_queue_event_default();
  queue_event.type = IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_DISPATCH;
  queue_event.event_id = 10;
  queue_event.submission_id = 20;
  queue_event.stream_id = queue.stream_id;
  queue_event.physical_device_ordinal = queue.physical_device_ordinal;
  queue_event.queue_ordinal = queue.queue_ordinal;
  queue_event.host_time_ns = 100;
  AppendPlainRecord(&queue_event_payload, queue_event);
  iree_hal_profile_file_record_t queue_event_chunk =
      MakeQueueEventsChunk(queue_event_payload);

  std::vector<uint8_t> queue_device_event_payload;
  iree_hal_profile_queue_device_event_t queue_device_event =
      iree_hal_profile_queue_device_event_default();
  queue_device_event.type = IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_DISPATCH;
  queue_device_event.event_id = 11;
  queue_device_event.submission_id = queue_event.submission_id;
  queue_device_event.stream_id = queue.stream_id;
  queue_device_event.physical_device_ordinal = queue.physical_device_ordinal;
  queue_device_event.queue_ordinal = queue.queue_ordinal;
  queue_device_event.start_tick = 1000;
  queue_device_event.end_tick = 1100;
  AppendPlainRecord(&queue_device_event_payload, queue_device_event);
  iree_hal_profile_file_record_t queue_device_event_chunk =
      MakeQueueDeviceEventsChunk(queue_device_event_payload);

  std::vector<uint8_t> host_execution_payload;
  iree_hal_profile_host_execution_event_t host_execution =
      iree_hal_profile_host_execution_event_default();
  host_execution.type = IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_DISPATCH;
  host_execution.event_id = 12;
  host_execution.submission_id = queue_event.submission_id;
  host_execution.stream_id = queue.stream_id;
  host_execution.physical_device_ordinal = queue.physical_device_ordinal;
  host_execution.queue_ordinal = queue.queue_ordinal;
  host_execution.start_host_time_ns = 200;
  host_execution.end_host_time_ns = 250;
  AppendPlainRecord(&host_execution_payload, host_execution);
  iree_hal_profile_file_record_t host_execution_chunk =
      MakeHostExecutionEventsChunk(host_execution_payload);

  iree_profile_model_t model;
  iree_profile_model_initialize(iree_allocator_system(), &model);
  IREE_ASSERT_OK(iree_profile_model_process_metadata_record(
      &model, &queue_metadata_chunk));

  iree_profile_queue_event_query_t query;
  iree_profile_queue_event_query_initialize(iree_allocator_system(), &query);
  IREE_ASSERT_OK(iree_profile_queue_event_query_process_record(
      &query, &model, &queue_event_chunk, IREE_SV("dispatch"), 20));
  IREE_ASSERT_OK(iree_profile_queue_event_query_process_record(
      &query, &model, &queue_device_event_chunk, IREE_SV("dispatch"), 20));
  IREE_ASSERT_OK(iree_profile_queue_event_query_process_record(
      &query, &model, &host_execution_chunk, IREE_SV("dispatch"), 20));

  EXPECT_EQ(1u, query.total_queue_event_count);
  EXPECT_EQ(1u, query.matched_queue_event_count);
  EXPECT_EQ(1u, query.queue_event_count);
  EXPECT_EQ(100, query.queue_events[0].record.host_time_ns);

  EXPECT_EQ(1u, query.total_queue_device_event_count);
  EXPECT_EQ(1u, query.matched_queue_device_event_count);
  EXPECT_EQ(1u, query.queue_device_event_count);
  EXPECT_EQ(1000u, query.queue_device_events[0].record.start_tick);
  EXPECT_EQ(1100u, query.queue_device_events[0].record.end_tick);

  EXPECT_EQ(1u, query.total_host_execution_event_count);
  EXPECT_EQ(1u, query.matched_host_execution_event_count);
  EXPECT_EQ(1u, query.host_execution_event_count);
  EXPECT_EQ(200, query.host_execution_events[0].record.start_host_time_ns);
  EXPECT_EQ(250, query.host_execution_events[0].record.end_host_time_ns);
  EXPECT_EQ(0u, model.device_count);

  iree_profile_queue_event_query_deinitialize(&query);
  iree_profile_model_deinitialize(&model);
}

TEST(ProfileQueueQueryTest, RejectsMatchedEventWithoutQueueMetadata) {
  std::vector<uint8_t> queue_event_payload;
  iree_hal_profile_queue_event_t queue_event =
      iree_hal_profile_queue_event_default();
  queue_event.type = IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_DISPATCH;
  queue_event.event_id = 10;
  queue_event.submission_id = 20;
  queue_event.physical_device_ordinal = 2;
  queue_event.queue_ordinal = 3;
  queue_event.stream_id = 4;
  AppendPlainRecord(&queue_event_payload, queue_event);
  iree_hal_profile_file_record_t queue_event_chunk =
      MakeQueueEventsChunk(queue_event_payload);

  iree_profile_model_t model;
  iree_profile_model_initialize(iree_allocator_system(), &model);
  iree_profile_queue_event_query_t query;
  iree_profile_queue_event_query_initialize(iree_allocator_system(), &query);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_DATA_LOSS,
      iree_profile_queue_event_query_process_record(
          &query, &model, &queue_event_chunk, IREE_SV("*"), -1));

  EXPECT_EQ(1u, query.total_queue_event_count);
  EXPECT_EQ(1u, query.matched_queue_event_count);
  EXPECT_EQ(0u, query.queue_event_count);

  iree_profile_queue_event_query_deinitialize(&query);
  iree_profile_model_deinitialize(&model);
}

}  // namespace
