// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/tooling/profile/counter.h"
#include "iree/tooling/profile/memory.h"
#include "iree/tooling/profile/model.h"
#include "iree/tooling/profile/queue_query.h"
#include "iree/tooling/profile/reader.h"
#include "iree/tooling/profile/summary.h"

namespace {

typedef struct test_profile_record_t {
  uint32_t record_length;
  uint32_t value;
} test_profile_record_t;

TEST(ProfileCommonTest, NamesProfileStatusCodes) {
  EXPECT_STREQ("CANCELLED",
               iree_profile_status_code_name(IREE_STATUS_CANCELLED));
  EXPECT_STREQ("UNKNOWN_STATUS", iree_profile_status_code_name(UINT32_MAX));
}

static iree_hal_profile_file_record_t MakeChunk(
    const std::vector<uint8_t>& payload) {
  iree_hal_profile_file_record_t chunk;
  memset(&chunk, 0, sizeof(chunk));
  chunk.header.record_type = IREE_HAL_PROFILE_FILE_RECORD_TYPE_CHUNK;
  chunk.content_type = IREE_SV("application/vnd.iree.test");
  chunk.payload = iree_make_const_byte_span(payload.data(), payload.size());
  return chunk;
}

static iree_hal_profile_file_record_t MakeMemoryChunk(
    const std::vector<uint8_t>& payload) {
  iree_hal_profile_file_record_t chunk = MakeChunk(payload);
  chunk.content_type = IREE_HAL_PROFILE_CONTENT_TYPE_MEMORY_EVENTS;
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

static iree_hal_profile_file_record_t MakeCounterSetsChunk(
    const std::vector<uint8_t>& payload) {
  iree_hal_profile_file_record_t chunk = MakeChunk(payload);
  chunk.content_type = IREE_HAL_PROFILE_CONTENT_TYPE_COUNTER_SETS;
  return chunk;
}

static iree_hal_profile_file_record_t MakeCountersChunk(
    const std::vector<uint8_t>& payload) {
  iree_hal_profile_file_record_t chunk = MakeChunk(payload);
  chunk.content_type = IREE_HAL_PROFILE_CONTENT_TYPE_COUNTERS;
  return chunk;
}

static iree_hal_profile_file_record_t MakeCounterSamplesChunk(
    const std::vector<uint8_t>& payload) {
  iree_hal_profile_file_record_t chunk = MakeChunk(payload);
  chunk.content_type = IREE_HAL_PROFILE_CONTENT_TYPE_COUNTER_SAMPLES;
  return chunk;
}

static iree_hal_profile_file_record_t MakeCommandBuffersChunk(
    const std::vector<uint8_t>& payload) {
  iree_hal_profile_file_record_t chunk = MakeChunk(payload);
  chunk.content_type = IREE_HAL_PROFILE_CONTENT_TYPE_COMMAND_BUFFERS;
  return chunk;
}

static iree_hal_profile_file_record_t MakeCommandOperationsChunk(
    const std::vector<uint8_t>& payload) {
  iree_hal_profile_file_record_t chunk = MakeChunk(payload);
  chunk.content_type = IREE_HAL_PROFILE_CONTENT_TYPE_COMMAND_OPERATIONS;
  return chunk;
}

static iree_hal_profile_file_record_t MakeDispatchEventsChunk(
    const std::vector<uint8_t>& payload) {
  iree_hal_profile_file_record_t chunk = MakeChunk(payload);
  chunk.content_type = IREE_HAL_PROFILE_CONTENT_TYPE_DISPATCH_EVENTS;
  return chunk;
}

static void AppendRecord(std::vector<uint8_t>* payload, uint32_t value,
                         std::initializer_list<uint8_t> inline_payload) {
  test_profile_record_t record;
  memset(&record, 0, sizeof(record));
  record.record_length = (uint32_t)(sizeof(record) + inline_payload.size());
  record.value = value;

  const iree_host_size_t offset = payload->size();
  payload->resize(offset + sizeof(record) + inline_payload.size());
  memcpy(payload->data() + offset, &record, sizeof(record));
  if (inline_payload.size() != 0) {
    memcpy(payload->data() + offset + sizeof(record), inline_payload.begin(),
           inline_payload.size());
  }
}

static void AppendMemoryEvent(std::vector<uint8_t>* payload,
                              iree_hal_profile_memory_event_type_t event_type,
                              uint64_t event_id, uint64_t allocation_id,
                              uint64_t pool_id, uint64_t length) {
  iree_hal_profile_memory_event_t event =
      iree_hal_profile_memory_event_default();
  event.type = event_type;
  event.event_id = event_id;
  event.allocation_id = allocation_id;
  event.pool_id = pool_id;
  event.physical_device_ordinal = 0;
  event.queue_ordinal = 0;
  event.memory_type = 1;
  event.buffer_usage = 1;
  event.length = length;
  event.alignment = 1;
  switch (event_type) {
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_RESERVE:
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_MATERIALIZE:
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_RELEASE:
      event.flags = IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_POOL_RESERVATION;
      break;
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_ALLOCA:
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_DEALLOCA:
      event.flags = IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_QUEUE_OPERATION;
      event.submission_id = event_id;
      break;
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_BUFFER_IMPORT:
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_BUFFER_UNIMPORT:
      event.flags = IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_EXTERNALLY_OWNED;
      break;
    default:
      break;
  }

  const iree_host_size_t offset = payload->size();
  payload->resize(offset + sizeof(event));
  memcpy(payload->data() + offset, &event, sizeof(event));
}

static void AppendCounterSet(std::vector<uint8_t>* payload,
                             uint64_t counter_set_id, uint32_t counter_count,
                             uint32_t sample_value_count,
                             iree_string_view_t name) {
  iree_hal_profile_counter_set_record_t record =
      iree_hal_profile_counter_set_record_default();
  record.record_length = (uint32_t)(sizeof(record) + name.size);
  record.counter_set_id = counter_set_id;
  record.physical_device_ordinal = 0;
  record.counter_count = counter_count;
  record.sample_value_count = sample_value_count;
  record.name_length = (uint32_t)name.size;

  const iree_host_size_t offset = payload->size();
  payload->resize(offset + record.record_length);
  memcpy(payload->data() + offset, &record, sizeof(record));
  memcpy(payload->data() + offset + sizeof(record), name.data, name.size);
}

static void AppendCounter(std::vector<uint8_t>* payload,
                          uint64_t counter_set_id, uint32_t counter_ordinal,
                          uint32_t sample_value_offset,
                          iree_string_view_t block_name,
                          iree_string_view_t name) {
  iree_hal_profile_counter_record_t record =
      iree_hal_profile_counter_record_default();
  record.record_length =
      (uint32_t)(sizeof(record) + block_name.size + name.size);
  record.flags = IREE_HAL_PROFILE_COUNTER_FLAG_RAW;
  record.unit = IREE_HAL_PROFILE_COUNTER_UNIT_COUNT;
  record.physical_device_ordinal = 0;
  record.counter_set_id = counter_set_id;
  record.counter_ordinal = counter_ordinal;
  record.sample_value_offset = sample_value_offset;
  record.sample_value_count = 1;
  record.block_name_length = (uint32_t)block_name.size;
  record.name_length = (uint32_t)name.size;

  const iree_host_size_t offset = payload->size();
  payload->resize(offset + record.record_length);
  uint8_t* target = payload->data() + offset;
  memcpy(target, &record, sizeof(record));
  target += sizeof(record);
  memcpy(target, block_name.data, block_name.size);
  target += block_name.size;
  memcpy(target, name.data, name.size);
}

static void AppendCounterSample(std::vector<uint8_t>* payload,
                                uint64_t sample_id, uint64_t counter_set_id,
                                std::initializer_list<uint64_t> values) {
  iree_hal_profile_counter_sample_record_t sample =
      iree_hal_profile_counter_sample_record_default();
  sample.record_length =
      (uint32_t)(sizeof(sample) + values.size() * sizeof(uint64_t));
  sample.sample_id = sample_id;
  sample.counter_set_id = counter_set_id;
  sample.sample_value_count = (uint32_t)values.size();
  sample.physical_device_ordinal = 0;
  sample.queue_ordinal = 0;

  const iree_host_size_t offset = payload->size();
  payload->resize(offset + sample.record_length);
  memcpy(payload->data() + offset, &sample, sizeof(sample));
  memcpy(payload->data() + offset + sizeof(sample), values.begin(),
         values.size() * sizeof(uint64_t));
}

template <typename T>
static void AppendPlainRecord(std::vector<uint8_t>* payload, const T& record) {
  const iree_host_size_t offset = payload->size();
  payload->resize(offset + sizeof(record));
  memcpy(payload->data() + offset, &record, sizeof(record));
}

static void AppendCommandBuffer(std::vector<uint8_t>* payload,
                                uint64_t command_buffer_id) {
  iree_hal_profile_command_buffer_record_t record =
      iree_hal_profile_command_buffer_record_default();
  record.command_buffer_id = command_buffer_id;
  record.physical_device_ordinal = 0;

  const iree_host_size_t offset = payload->size();
  payload->resize(offset + sizeof(record));
  memcpy(payload->data() + offset, &record, sizeof(record));
}

static void AppendCommandOperation(
    std::vector<uint8_t>* payload,
    const iree_hal_profile_command_operation_record_t& record) {
  AppendPlainRecord(payload, record);
}

static void AddPoolStatsToLastMemoryEvent(
    std::vector<uint8_t>* payload, uint64_t bytes_reserved, uint64_t bytes_free,
    uint64_t bytes_committed, uint64_t budget_limit, uint32_t reservation_count,
    uint32_t slab_count) {
  const iree_host_size_t offset =
      payload->size() - sizeof(iree_hal_profile_memory_event_t);
  iree_hal_profile_memory_event_t event;
  memcpy(&event, payload->data() + offset, sizeof(event));
  event.flags |= IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_POOL_STATS;
  event.pool_bytes_reserved = bytes_reserved;
  event.pool_bytes_free = bytes_free;
  event.pool_bytes_committed = bytes_committed;
  event.pool_budget_limit = budget_limit;
  event.pool_reservation_count = reservation_count;
  event.pool_slab_count = slab_count;
  memcpy(payload->data() + offset, &event, sizeof(event));
}

typedef struct memory_event_collector_t {
  uint64_t event_count;
  uint64_t first_event_id;
  bool saw_truncated_chunk;
} memory_event_collector_t;

static iree_status_t CollectMemoryEvent(
    void* user_data, const iree_profile_memory_event_row_t* row) {
  memory_event_collector_t* collector =
      static_cast<memory_event_collector_t*>(user_data);
  if (collector->event_count == 0) {
    collector->first_event_id = row->event->event_id;
  }
  ++collector->event_count;
  collector->saw_truncated_chunk |= row->is_truncated;
  return iree_ok_status();
}

typedef struct counter_sample_collector_t {
  uint64_t row_count;
  double value_sum;
  bool saw_truncated_chunk;
} counter_sample_collector_t;

static iree_status_t CollectCounterSample(
    void* user_data, const iree_profile_counter_sample_row_t* row) {
  counter_sample_collector_t* collector =
      static_cast<counter_sample_collector_t*>(user_data);
  ++collector->row_count;
  collector->value_sum += row->value_sum;
  collector->saw_truncated_chunk |= row->is_truncated;
  return iree_ok_status();
}

TEST(ProfileTypedRecordTest, IteratesTypedRecords) {
  std::vector<uint8_t> payload;
  AppendRecord(&payload, 1, {0xAA, 0xBB, 0xCC});
  AppendRecord(&payload, 2, {0xDD, 0xEE});
  iree_hal_profile_file_record_t chunk = MakeChunk(payload);

  iree_profile_typed_record_iterator_t iterator;
  iree_profile_typed_record_iterator_initialize(
      &chunk, sizeof(test_profile_record_t), &iterator);

  iree_profile_typed_record_t first_record;
  bool has_record = false;
  IREE_ASSERT_OK(iree_profile_typed_record_iterator_next(
      &iterator, &first_record, &has_record));
  EXPECT_TRUE(has_record);
  EXPECT_EQ(0u, first_record.record_index);
  EXPECT_EQ(0u, first_record.payload_offset);
  EXPECT_EQ(sizeof(test_profile_record_t) + 3, first_record.record_length);
  EXPECT_EQ(sizeof(test_profile_record_t) + 3,
            first_record.contents.data_length);
  EXPECT_EQ(3u, first_record.inline_payload.data_length);
  EXPECT_EQ(0xAA, first_record.inline_payload.data[0]);
  EXPECT_EQ(sizeof(test_profile_record_t) + 2,
            first_record.following_payload.data_length);

  test_profile_record_t first_header;
  memcpy(&first_header, first_record.contents.data, sizeof(first_header));
  EXPECT_EQ(1u, first_header.value);

  iree_profile_typed_record_t second_record;
  IREE_ASSERT_OK(iree_profile_typed_record_iterator_next(
      &iterator, &second_record, &has_record));
  EXPECT_TRUE(has_record);
  EXPECT_EQ(1u, second_record.record_index);
  EXPECT_EQ(sizeof(test_profile_record_t) + 3, second_record.payload_offset);
  EXPECT_EQ(sizeof(test_profile_record_t) + 2, second_record.record_length);
  EXPECT_EQ(2u, second_record.inline_payload.data_length);
  EXPECT_EQ(0u, second_record.following_payload.data_length);

  test_profile_record_t second_header;
  memcpy(&second_header, second_record.contents.data, sizeof(second_header));
  EXPECT_EQ(2u, second_header.value);

  iree_profile_typed_record_t end_record;
  IREE_ASSERT_OK(iree_profile_typed_record_iterator_next(&iterator, &end_record,
                                                         &has_record));
  EXPECT_FALSE(has_record);
}

TEST(ProfileTypedRecordTest, ParseExposesFollowingPayload) {
  std::vector<uint8_t> payload;
  AppendRecord(&payload, 1, {});
  payload.push_back(0xAA);
  payload.push_back(0xBB);
  iree_hal_profile_file_record_t chunk = MakeChunk(payload);

  iree_profile_typed_record_t record;
  IREE_ASSERT_OK(iree_profile_typed_record_parse(
      &chunk, 0, sizeof(test_profile_record_t), 0, &record));
  EXPECT_EQ(sizeof(test_profile_record_t), record.record_length);
  EXPECT_EQ(0u, record.inline_payload.data_length);
  EXPECT_EQ(2u, record.following_payload.data_length);
  EXPECT_EQ(0xAA, record.following_payload.data[0]);
  EXPECT_EQ(0xBB, record.following_payload.data[1]);
}

TEST(ProfileTypedRecordTest, RejectsTruncatedRecord) {
  std::vector<uint8_t> payload(sizeof(uint32_t), 0);
  iree_hal_profile_file_record_t chunk = MakeChunk(payload);

  iree_profile_typed_record_t record;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_DATA_LOSS,
      iree_profile_typed_record_parse(&chunk, 0, sizeof(test_profile_record_t),
                                      0, &record));
}

TEST(ProfileTypedRecordTest, RejectsShortRecordLength) {
  std::vector<uint8_t> payload(sizeof(test_profile_record_t), 0);
  uint32_t record_length = sizeof(uint32_t);
  memcpy(payload.data(), &record_length, sizeof(record_length));
  iree_hal_profile_file_record_t chunk = MakeChunk(payload);

  iree_profile_typed_record_t record;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_DATA_LOSS,
      iree_profile_typed_record_parse(&chunk, 0, sizeof(test_profile_record_t),
                                      0, &record));
}

TEST(ProfileTypedRecordTest, RejectsOversizedRecordLength) {
  std::vector<uint8_t> payload(sizeof(test_profile_record_t), 0);
  uint32_t record_length = sizeof(test_profile_record_t) + 1;
  memcpy(payload.data(), &record_length, sizeof(record_length));
  iree_hal_profile_file_record_t chunk = MakeChunk(payload);

  iree_profile_typed_record_t record;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_DATA_LOSS,
      iree_profile_typed_record_parse(&chunk, 0, sizeof(test_profile_record_t),
                                      0, &record));
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

static iree_hal_profile_clock_correlation_record_t MakeClockSample(
    uint64_t sample_id, uint64_t device_tick, uint64_t host_cpu_timestamp_ns) {
  iree_hal_profile_clock_correlation_record_t sample =
      iree_hal_profile_clock_correlation_record_default();
  sample.flags = IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_DEVICE_TICK |
                 IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_HOST_CPU_TIMESTAMP;
  sample.sample_id = sample_id;
  sample.device_tick = device_tick;
  sample.host_cpu_timestamp_ns = host_cpu_timestamp_ns;
  return sample;
}

TEST(ProfileClockFitTest, MapsTicksWithIntegerRounding) {
  iree_profile_model_device_t device;
  memset(&device, 0, sizeof(device));
  device.clock_sample_count = 2;
  device.first_clock_sample =
      MakeClockSample(10, (1ull << 60) + 3, 900000000000000000ull);
  device.last_clock_sample =
      MakeClockSample(11, (1ull << 60) + 10, 900000000000000010ull);

  iree_profile_model_clock_fit_t fit;
  ASSERT_TRUE(iree_profile_model_device_try_fit_clock_exact(
      &device, IREE_PROFILE_MODEL_CLOCK_TIME_DOMAIN_HOST_CPU_TIMESTAMP_NS,
      &fit));
  EXPECT_EQ(10u, fit.first_sample_id);
  EXPECT_EQ(11u, fit.last_sample_id);
  EXPECT_EQ(7u, fit.device_tick_span);
  EXPECT_EQ(10u, fit.time_span_ns);

  int64_t time_ns = 0;
  EXPECT_TRUE(
      iree_profile_model_clock_fit_map_tick(&fit, (1ull << 60) + 6, &time_ns));
  EXPECT_EQ(900000000000000004ll, time_ns);
  EXPECT_TRUE(
      iree_profile_model_clock_fit_map_tick(&fit, (1ull << 60), &time_ns));
  EXPECT_EQ(899999999999999996ll, time_ns);

  int64_t duration_ns = 0;
  EXPECT_TRUE(
      iree_profile_model_clock_fit_scale_ticks_to_ns(&fit, 7, &duration_ns));
  EXPECT_EQ(10, duration_ns);
  EXPECT_TRUE(
      iree_profile_model_clock_fit_scale_ticks_to_ns(&fit, 4, &duration_ns));
  EXPECT_EQ(6, duration_ns);
}

TEST(ProfileClockFitTest, FitsIreeHostTimeFromBracketMidpoints) {
  iree_profile_model_device_t device;
  memset(&device, 0, sizeof(device));
  device.clock_sample_count = 2;
  device.first_clock_sample = MakeClockSample(1, 1000, 5000);
  device.first_clock_sample.flags |=
      IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_HOST_TIME_BRACKET;
  device.first_clock_sample.host_time_begin_ns = 100;
  device.first_clock_sample.host_time_end_ns = 103;
  device.last_clock_sample = MakeClockSample(2, 1010, 6000);
  device.last_clock_sample.flags |=
      IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_HOST_TIME_BRACKET;
  device.last_clock_sample.host_time_begin_ns = 170;
  device.last_clock_sample.host_time_end_ns = 173;

  iree_profile_model_clock_fit_t fit;
  ASSERT_TRUE(iree_profile_model_device_try_fit_clock_exact(
      &device, IREE_PROFILE_MODEL_CLOCK_TIME_DOMAIN_IREE_HOST_TIME_NS, &fit));
  EXPECT_EQ(101, fit.first_time_ns);
  EXPECT_EQ(171, fit.last_time_ns);

  int64_t time_ns = 0;
  EXPECT_TRUE(iree_profile_model_clock_fit_map_tick(&fit, 1005, &time_ns));
  EXPECT_EQ(136, time_ns);
}

TEST(ProfileModelTest, AcceptsLinearCommandOperationsWithoutBlockStructure) {
  std::vector<uint8_t> command_buffer_payload;
  AppendCommandBuffer(&command_buffer_payload, 1);
  iree_hal_profile_file_record_t command_buffer_chunk =
      MakeCommandBuffersChunk(command_buffer_payload);

  iree_hal_profile_command_operation_record_t operation =
      iree_hal_profile_command_operation_record_default();
  operation.type = IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_DISPATCH;
  operation.command_buffer_id = 1;
  operation.command_index = 0;

  std::vector<uint8_t> command_operation_payload;
  AppendCommandOperation(&command_operation_payload, operation);
  iree_hal_profile_file_record_t command_operation_chunk =
      MakeCommandOperationsChunk(command_operation_payload);

  iree_profile_model_t model;
  iree_profile_model_initialize(iree_allocator_system(), &model);
  IREE_ASSERT_OK(iree_profile_model_process_metadata_record(
      &model, &command_buffer_chunk));
  IREE_ASSERT_OK(iree_profile_model_process_metadata_record(
      &model, &command_operation_chunk));
  ASSERT_EQ(1u, model.command_operation_count);
  EXPECT_FALSE(iree_hal_profile_command_operation_has_block_structure(
      &model.command_operations[0].record));
  iree_profile_model_deinitialize(&model);
}

TEST(ProfileModelTest, RejectsCommandOperationBlockFieldsWithoutFlag) {
  std::vector<uint8_t> command_buffer_payload;
  AppendCommandBuffer(&command_buffer_payload, 1);
  iree_hal_profile_file_record_t command_buffer_chunk =
      MakeCommandBuffersChunk(command_buffer_payload);

  iree_hal_profile_command_operation_record_t operation =
      iree_hal_profile_command_operation_record_default();
  operation.type = IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_DISPATCH;
  operation.command_buffer_id = 1;
  operation.command_index = 0;
  operation.block_ordinal = 2;
  operation.block_command_ordinal = 3;

  std::vector<uint8_t> command_operation_payload;
  AppendCommandOperation(&command_operation_payload, operation);
  iree_hal_profile_file_record_t command_operation_chunk =
      MakeCommandOperationsChunk(command_operation_payload);

  iree_profile_model_t model;
  iree_profile_model_initialize(iree_allocator_system(), &model);
  IREE_ASSERT_OK(iree_profile_model_process_metadata_record(
      &model, &command_buffer_chunk));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS,
                        iree_profile_model_process_metadata_record(
                            &model, &command_operation_chunk));
  iree_profile_model_deinitialize(&model);
}

TEST(ProfileModelTest, RejectsBlockStructureWithoutBlockCoordinates) {
  std::vector<uint8_t> command_buffer_payload;
  AppendCommandBuffer(&command_buffer_payload, 1);
  iree_hal_profile_file_record_t command_buffer_chunk =
      MakeCommandBuffersChunk(command_buffer_payload);

  iree_hal_profile_command_operation_record_t operation =
      iree_hal_profile_command_operation_record_default();
  operation.type = IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_DISPATCH;
  operation.command_buffer_id = 1;
  operation.command_index = 0;
  operation.flags = IREE_HAL_PROFILE_COMMAND_OPERATION_FLAG_BLOCK_STRUCTURE;

  std::vector<uint8_t> command_operation_payload;
  AppendCommandOperation(&command_operation_payload, operation);
  iree_hal_profile_file_record_t command_operation_chunk =
      MakeCommandOperationsChunk(command_operation_payload);

  iree_profile_model_t model;
  iree_profile_model_initialize(iree_allocator_system(), &model);
  IREE_ASSERT_OK(iree_profile_model_process_metadata_record(
      &model, &command_buffer_chunk));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS,
                        iree_profile_model_process_metadata_record(
                            &model, &command_operation_chunk));
  iree_profile_model_deinitialize(&model);
}

TEST(ProfileCounterTest, InvokesSampleCallbackForMatchedValues) {
  std::vector<uint8_t> counter_set_payload;
  AppendCounterSet(&counter_set_payload, 9, 2, 2, IREE_SV("test_set"));
  iree_hal_profile_file_record_t counter_set_chunk =
      MakeCounterSetsChunk(counter_set_payload);

  std::vector<uint8_t> counter_payload;
  AppendCounter(&counter_payload, 9, 0, 0, IREE_SV("SQ"), IREE_SV("WAVES"));
  AppendCounter(&counter_payload, 9, 1, 1, IREE_SV("SQ"), IREE_SV("BUSY"));
  iree_hal_profile_file_record_t counter_chunk =
      MakeCountersChunk(counter_payload);

  std::vector<uint8_t> sample_payload;
  AppendCounterSample(&sample_payload, 5, 9, {7, 11});
  iree_hal_profile_file_record_t sample_chunk =
      MakeCounterSamplesChunk(sample_payload);
  sample_chunk.header.chunk_flags = IREE_HAL_PROFILE_CHUNK_FLAG_TRUNCATED;

  iree_profile_model_t model;
  iree_profile_model_initialize(iree_allocator_system(), &model);
  iree_profile_counter_context_t context;
  iree_profile_counter_context_initialize(iree_allocator_system(), &context);
  IREE_ASSERT_OK(iree_profile_counter_process_metadata_record(
      &context, &counter_set_chunk));
  IREE_ASSERT_OK(
      iree_profile_counter_process_metadata_record(&context, &counter_chunk));

  counter_sample_collector_t collector = {0};
  const iree_profile_counter_sample_callback_t sample_callback = {
      CollectCounterSample,
      &collector,
  };
  IREE_ASSERT_OK(iree_profile_counter_process_sample_records(
      &context, &model, &sample_chunk, IREE_SV("*"), -1, sample_callback));

  EXPECT_EQ(1u, context.total_sample_count);
  EXPECT_EQ(1u, context.matched_sample_count);
  EXPECT_EQ(1u, context.truncated_sample_count);
  EXPECT_EQ(2u, context.aggregate_count);
  EXPECT_EQ(2u, collector.row_count);
  EXPECT_EQ(18.0, collector.value_sum);
  EXPECT_TRUE(collector.saw_truncated_chunk);

  iree_profile_counter_context_deinitialize(&context);
  iree_profile_model_deinitialize(&model);
}

TEST(ProfileMemoryTest, SeparatesReserveMaterializedAndInflightBytes) {
  std::vector<uint8_t> payload;
  AppendMemoryEvent(&payload, IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_RELEASE,
                    1, 99, 7, 64);
  AppendMemoryEvent(&payload, IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_RESERVE,
                    2, 1, 7, 256);
  AppendMemoryEvent(&payload,
                    IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_MATERIALIZE, 3, 1,
                    7, 256);
  AppendMemoryEvent(&payload, IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_ALLOCA,
                    4, 1, 7, 40);
  AppendMemoryEvent(&payload, IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_DEALLOCA,
                    5, 1, 7, 40);
  AppendMemoryEvent(&payload, IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_RELEASE,
                    6, 1, 7, 256);
  iree_hal_profile_file_record_t chunk = MakeMemoryChunk(payload);

  iree_profile_memory_context_t context;
  iree_profile_memory_context_initialize(iree_allocator_system(), &context);
  IREE_ASSERT_OK(iree_profile_memory_context_accumulate_record(
      &context, &chunk, IREE_SV("*"), -1,
      iree_profile_memory_event_callback_t{}));

  ASSERT_EQ(1u, context.device_count);
  const iree_profile_memory_device_t* device = &context.devices[0];
  EXPECT_EQ(0u, device->pool_reservation_balance.current_bytes);
  EXPECT_EQ(256u, device->pool_reservation_balance.high_water_bytes);
  EXPECT_EQ(64u, device->pool_reservation_balance.partial_close_bytes);
  EXPECT_EQ(0u, device->pool_materialization_balance.current_bytes);
  EXPECT_EQ(256u, device->pool_materialization_balance.high_water_bytes);
  EXPECT_EQ(0u, device->pool_materialization_balance.partial_close_count);
  EXPECT_EQ(0u, device->queue_inflight_balance.current_bytes);
  EXPECT_EQ(40u, device->queue_inflight_balance.high_water_bytes);

  iree_profile_memory_context_deinitialize(&context);
}

TEST(ProfileMemoryTest, InvokesEventCallbackForMatchedEvents) {
  std::vector<uint8_t> payload;
  AppendMemoryEvent(&payload, IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_RESERVE,
                    1, 1, 7, 256);
  AppendMemoryEvent(&payload, IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_ALLOCA,
                    2, 1, 7, 40);
  iree_hal_profile_file_record_t chunk = MakeMemoryChunk(payload);
  chunk.header.chunk_flags = IREE_HAL_PROFILE_CHUNK_FLAG_TRUNCATED;

  memory_event_collector_t collector = {0};
  const iree_profile_memory_event_callback_t event_callback = {
      CollectMemoryEvent,
      &collector,
  };
  iree_profile_memory_context_t context;
  iree_profile_memory_context_initialize(iree_allocator_system(), &context);
  IREE_ASSERT_OK(iree_profile_memory_context_accumulate_record(
      &context, &chunk, IREE_SV("pool_*"), -1, event_callback));

  EXPECT_EQ(2u, context.total_event_count);
  EXPECT_EQ(1u, context.matched_event_count);
  EXPECT_EQ(1u, context.truncated_event_count);
  EXPECT_EQ(1u, collector.event_count);
  EXPECT_EQ(1u, collector.first_event_id);
  EXPECT_TRUE(collector.saw_truncated_chunk);

  iree_profile_memory_context_deinitialize(&context);
}

TEST(ProfileMemoryTest, RecordsZeroPayloadTruncatedChunks) {
  std::vector<uint8_t> payload;
  iree_hal_profile_file_record_t chunk = MakeMemoryChunk(payload);
  chunk.header.chunk_flags = IREE_HAL_PROFILE_CHUNK_FLAG_TRUNCATED;
  chunk.header.dropped_record_count = 7;

  iree_profile_memory_context_t context;
  iree_profile_memory_context_initialize(iree_allocator_system(), &context);
  IREE_ASSERT_OK(iree_profile_memory_context_accumulate_record(
      &context, &chunk, IREE_SV("*"), -1,
      iree_profile_memory_event_callback_t{0}));

  EXPECT_EQ(1u, context.truncated_chunk_count);
  EXPECT_EQ(7u, context.dropped_record_count);
  EXPECT_EQ(0u, context.total_event_count);
  EXPECT_EQ(0u, context.matched_event_count);
  EXPECT_EQ(0u, context.truncated_event_count);

  iree_profile_memory_context_deinitialize(&context);
}

TEST(ProfileMemoryTest, RecordsPoolStatSnapshots) {
  std::vector<uint8_t> payload;
  AppendMemoryEvent(&payload, IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_RESERVE,
                    1, 1, 7, 256);
  AddPoolStatsToLastMemoryEvent(&payload, 256, 768, 1024, 2048, 1, 1);
  AppendMemoryEvent(&payload, IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_ALLOCA,
                    2, 1, 7, 128);
  AddPoolStatsToLastMemoryEvent(&payload, 256, 768, 1024, 2048, 1, 1);
  AppendMemoryEvent(&payload, IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_RELEASE,
                    3, 1, 7, 256);
  AddPoolStatsToLastMemoryEvent(&payload, 0, 1024, 1024, 2048, 0, 1);
  iree_hal_profile_file_record_t chunk = MakeMemoryChunk(payload);

  iree_profile_memory_context_t context;
  iree_profile_memory_context_initialize(iree_allocator_system(), &context);
  IREE_ASSERT_OK(iree_profile_memory_context_accumulate_record(
      &context, &chunk, IREE_SV("*"), -1,
      iree_profile_memory_event_callback_t{}));

  const iree_profile_memory_pool_t* pool = nullptr;
  for (iree_host_size_t i = 0; i < context.pool_count; ++i) {
    if (context.pools[i].kind ==
            IREE_PROFILE_MEMORY_LIFECYCLE_KIND_POOL_RESERVATION &&
        context.pools[i].pool_id == 7) {
      pool = &context.pools[i];
      break;
    }
  }
  ASSERT_NE(nullptr, pool);
  EXPECT_EQ(3u, pool->pool_stats_sample_count);
  EXPECT_EQ(0u, pool->pool_bytes_reserved);
  EXPECT_EQ(256u, pool->pool_bytes_reserved_high_water);
  EXPECT_EQ(1024u, pool->pool_bytes_free);
  EXPECT_EQ(768u, pool->pool_bytes_free_low_water);
  EXPECT_EQ(1024u, pool->pool_bytes_committed);
  EXPECT_EQ(1024u, pool->pool_bytes_committed_high_water);
  EXPECT_EQ(2048u, pool->pool_budget_limit);
  EXPECT_EQ(0u, pool->pool_reservation_count);
  EXPECT_EQ(1u, pool->pool_reservation_high_water_count);
  EXPECT_EQ(1u, pool->pool_slab_count);
  EXPECT_EQ(1u, pool->pool_slab_high_water_count);

  iree_profile_memory_context_deinitialize(&context);
}

TEST(ProfileMemoryTest, SeparatesImportedBufferBytes) {
  std::vector<uint8_t> payload;
  AppendMemoryEvent(&payload, IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_BUFFER_IMPORT,
                    1, 11, 0, 4096);
  AppendMemoryEvent(&payload,
                    IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_BUFFER_UNIMPORT, 2, 11,
                    0, 4096);
  iree_hal_profile_file_record_t chunk = MakeMemoryChunk(payload);

  iree_profile_memory_context_t context;
  iree_profile_memory_context_initialize(iree_allocator_system(), &context);
  IREE_ASSERT_OK(iree_profile_memory_context_accumulate_record(
      &context, &chunk, IREE_SV("*"), -1,
      iree_profile_memory_event_callback_t{}));

  ASSERT_EQ(1u, context.device_count);
  const iree_profile_memory_device_t* device = &context.devices[0];
  EXPECT_EQ(1u, device->buffer_import_count);
  EXPECT_EQ(1u, device->buffer_unimport_count);
  EXPECT_EQ(0u, device->buffer_allocation_balance.total_open_count);
  EXPECT_EQ(0u, device->buffer_import_balance.current_bytes);
  EXPECT_EQ(4096u, device->buffer_import_balance.high_water_bytes);

  ASSERT_EQ(1u, context.allocation_count);
  const iree_profile_memory_allocation_t* allocation = &context.allocations[0];
  EXPECT_EQ(IREE_PROFILE_MEMORY_LIFECYCLE_KIND_IMPORTED_BUFFER,
            allocation->kind);
  EXPECT_TRUE(iree_all_bits_set(
      allocation->flags, IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_EXTERNALLY_OWNED));

  iree_profile_memory_context_deinitialize(&context);
}

}  // namespace
