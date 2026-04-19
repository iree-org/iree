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

}  // namespace
