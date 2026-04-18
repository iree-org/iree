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
#include "iree/tooling/profile/memory.h"
#include "iree/tooling/profile/model.h"
#include "iree/tooling/profile/reader.h"

namespace {

using ::iree::Status;
using ::iree::StatusCode;
using ::iree::testing::status::StatusIs;

typedef struct test_profile_record_t {
  uint32_t record_length;
  uint32_t value;
} test_profile_record_t;

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
    default:
      break;
  }

  const iree_host_size_t offset = payload->size();
  payload->resize(offset + sizeof(event));
  memcpy(payload->data() + offset, &event, sizeof(event));
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
  EXPECT_THAT(Status(iree_profile_typed_record_parse(
                  &chunk, 0, sizeof(test_profile_record_t), 0, &record)),
              StatusIs(StatusCode::kDataLoss));
}

TEST(ProfileTypedRecordTest, RejectsShortRecordLength) {
  std::vector<uint8_t> payload(sizeof(test_profile_record_t), 0);
  uint32_t record_length = sizeof(uint32_t);
  memcpy(payload.data(), &record_length, sizeof(record_length));
  iree_hal_profile_file_record_t chunk = MakeChunk(payload);

  iree_profile_typed_record_t record;
  EXPECT_THAT(Status(iree_profile_typed_record_parse(
                  &chunk, 0, sizeof(test_profile_record_t), 0, &record)),
              StatusIs(StatusCode::kDataLoss));
}

TEST(ProfileTypedRecordTest, RejectsOversizedRecordLength) {
  std::vector<uint8_t> payload(sizeof(test_profile_record_t), 0);
  uint32_t record_length = sizeof(test_profile_record_t) + 1;
  memcpy(payload.data(), &record_length, sizeof(record_length));
  iree_hal_profile_file_record_t chunk = MakeChunk(payload);

  iree_profile_typed_record_t record;
  EXPECT_THAT(Status(iree_profile_typed_record_parse(
                  &chunk, 0, sizeof(test_profile_record_t), 0, &record)),
              StatusIs(StatusCode::kDataLoss));
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
      &context, &chunk, IREE_SV("*"), -1, false, nullptr));

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
      &context, &chunk, IREE_SV("*"), -1, false, nullptr));

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

}  // namespace
