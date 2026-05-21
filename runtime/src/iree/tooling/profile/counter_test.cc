// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tooling/profile/counter.h"

#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/tooling/profile/model.h"

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

static void AppendCounterSampleRecord(
    std::vector<uint8_t>* payload,
    const iree_hal_profile_counter_sample_record_t& base_sample,
    std::initializer_list<uint64_t> values) {
  iree_hal_profile_counter_sample_record_t sample = base_sample;
  sample.record_length =
      (uint32_t)(sizeof(sample) + values.size() * sizeof(uint64_t));
  sample.sample_value_count = (uint32_t)values.size();

  const iree_host_size_t offset = payload->size();
  payload->resize(offset + sample.record_length);
  memcpy(payload->data() + offset, &sample, sizeof(sample));
  memcpy(payload->data() + offset + sizeof(sample), values.begin(),
         values.size() * sizeof(uint64_t));
}

static void AppendDispatchCounterSample(
    std::vector<uint8_t>* payload, uint64_t sample_id, uint64_t counter_set_id,
    std::initializer_list<uint64_t> values) {
  iree_hal_profile_counter_sample_record_t sample =
      iree_hal_profile_counter_sample_record_default();
  sample.flags = IREE_HAL_PROFILE_COUNTER_SAMPLE_FLAG_DISPATCH_EVENT |
                 IREE_HAL_PROFILE_COUNTER_SAMPLE_FLAG_DEVICE_TICK_RANGE;
  sample.scope = IREE_HAL_PROFILE_COUNTER_SAMPLE_SCOPE_DISPATCH;
  sample.sample_id = sample_id;
  sample.counter_set_id = counter_set_id;
  sample.dispatch_event_id = 27;
  sample.start_tick = 100;
  sample.end_tick = 132;
  sample.physical_device_ordinal = 0;
  sample.queue_ordinal = 0;
  AppendCounterSampleRecord(payload, sample, values);
}

typedef struct counter_sample_collector_t {
  // Number of counter value rows observed by the callback.
  uint64_t row_count;
  // Number of callback rows from dispatch-scoped samples.
  uint64_t dispatch_row_count;
  // Number of callback rows from physical-device time-range samples.
  uint64_t device_time_range_row_count;
  // Number of callback rows whose samples had device tick ranges.
  uint64_t device_tick_range_row_count;
  // Last callback row device-tick range start.
  uint64_t last_start_tick;
  // Last callback row device-tick range end.
  uint64_t last_end_tick;
  // Sum of callback row counter values.
  double value_sum;
  // True when any callback row came from a truncated source chunk.
  bool saw_truncated_chunk;
} counter_sample_collector_t;

static iree_status_t CollectCounterSample(
    void* user_data, const iree_profile_counter_sample_row_t* row) {
  counter_sample_collector_t* collector =
      static_cast<counter_sample_collector_t*>(user_data);
  ++collector->row_count;
  if (row->sample->scope == IREE_HAL_PROFILE_COUNTER_SAMPLE_SCOPE_DISPATCH) {
    ++collector->dispatch_row_count;
  } else if (row->sample->scope ==
             IREE_HAL_PROFILE_COUNTER_SAMPLE_SCOPE_DEVICE_TIME_RANGE) {
    ++collector->device_time_range_row_count;
  }
  if (iree_all_bits_set(
          row->sample->flags,
          IREE_HAL_PROFILE_COUNTER_SAMPLE_FLAG_DEVICE_TICK_RANGE)) {
    ++collector->device_tick_range_row_count;
    collector->last_start_tick = row->sample->start_tick;
    collector->last_end_tick = row->sample->end_tick;
  }
  collector->value_sum += row->value_sum;
  collector->saw_truncated_chunk |= row->is_truncated;
  return iree_ok_status();
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
  AppendDispatchCounterSample(&sample_payload, 5, 9, {7, 11});
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
  EXPECT_EQ(2u, collector.dispatch_row_count);
  EXPECT_EQ(18.0, collector.value_sum);
  EXPECT_EQ(2u, collector.device_tick_range_row_count);
  EXPECT_EQ(100u, collector.last_start_tick);
  EXPECT_EQ(132u, collector.last_end_tick);
  EXPECT_TRUE(collector.saw_truncated_chunk);

  iree_profile_counter_context_deinitialize(&context);
  iree_profile_model_deinitialize(&model);
}

TEST(ProfileCounterTest, ProcessesDeviceTimeRangeSamplesWithoutDispatch) {
  std::vector<uint8_t> counter_set_payload;
  AppendCounterSet(&counter_set_payload, 10, 1, 1, IREE_SV("device_set"));
  iree_hal_profile_file_record_t counter_set_chunk =
      MakeCounterSetsChunk(counter_set_payload);

  std::vector<uint8_t> counter_payload;
  AppendCounter(&counter_payload, 10, 0, 0, IREE_SV("GRBM"),
                IREE_SV("GUI_ACTIVE"));
  iree_hal_profile_file_record_t counter_chunk =
      MakeCountersChunk(counter_payload);

  iree_hal_profile_counter_sample_record_t sample =
      iree_hal_profile_counter_sample_record_default();
  sample.flags = IREE_HAL_PROFILE_COUNTER_SAMPLE_FLAG_DEVICE_TICK_RANGE;
  sample.scope = IREE_HAL_PROFILE_COUNTER_SAMPLE_SCOPE_DEVICE_TIME_RANGE;
  sample.sample_id = 6;
  sample.counter_set_id = 10;
  sample.start_tick = 200;
  sample.end_tick = 260;
  sample.physical_device_ordinal = 0;

  std::vector<uint8_t> sample_payload;
  AppendCounterSampleRecord(&sample_payload, sample, {19});
  iree_hal_profile_file_record_t sample_chunk =
      MakeCounterSamplesChunk(sample_payload);

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
  EXPECT_EQ(1u, context.aggregate_count);
  EXPECT_EQ(1u, collector.row_count);
  EXPECT_EQ(0u, collector.dispatch_row_count);
  EXPECT_EQ(1u, collector.device_time_range_row_count);
  EXPECT_EQ(1u, collector.device_tick_range_row_count);
  EXPECT_EQ(200u, collector.last_start_tick);
  EXPECT_EQ(260u, collector.last_end_tick);
  EXPECT_EQ(19.0, collector.value_sum);

  iree_profile_counter_context_deinitialize(&context);
  iree_profile_model_deinitialize(&model);
}

}  // namespace
