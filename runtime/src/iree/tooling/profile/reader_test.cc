// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tooling/profile/reader.h"

#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

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

static std::vector<uint8_t> MakeExecutableTracePayload(
    uint32_t record_length, uint64_t data_length,
    std::initializer_list<uint8_t> trace_data) {
  iree_hal_profile_executable_trace_record_t record =
      iree_hal_profile_executable_trace_record_default();
  record.record_length = record_length;
  record.data_length = data_length;

  std::vector<uint8_t> payload(sizeof(record), 0);
  memcpy(payload.data(), &record, sizeof(record));
  if (record_length > payload.size()) {
    payload.resize(record_length, 0xCC);
  }
  const iree_host_size_t offset = payload.size();
  payload.resize(offset + trace_data.size());
  if (trace_data.size() != 0) {
    memcpy(payload.data() + offset, trace_data.begin(), trace_data.size());
  }
  return payload;
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

static void AppendProfileFileHeader(std::vector<uint8_t>* storage,
                                    uint64_t file_length) {
  iree_hal_profile_file_header_t header;
  memset(&header, 0, sizeof(header));
  header.magic = IREE_HAL_PROFILE_FILE_MAGIC;
  header.version_major = IREE_HAL_PROFILE_FILE_VERSION_MAJOR;
  header.version_minor = IREE_HAL_PROFILE_FILE_VERSION_MINOR;
  header.header_length = sizeof(header);
  header.file_length = file_length;

  const iree_host_size_t offset = storage->size();
  storage->resize(offset + sizeof(header));
  memcpy(storage->data() + offset, &header, sizeof(header));
}

static void AppendProfileFileRecord(
    std::vector<uint8_t>* storage,
    iree_hal_profile_file_record_type_t record_type) {
  iree_hal_profile_file_record_header_t header;
  memset(&header, 0, sizeof(header));
  header.record_length = sizeof(header);
  header.header_length = sizeof(header);
  header.record_type = record_type;

  const iree_host_size_t offset = storage->size();
  storage->resize(offset + sizeof(header));
  memcpy(storage->data() + offset, &header, sizeof(header));
}

typedef struct RecordCounter {
  // Number of records observed by the iterator callback.
  iree_host_size_t count;
} RecordCounter;

static iree_status_t CountProfileFileRecord(
    void* user_data, const iree_hal_profile_file_record_t* record,
    iree_host_size_t record_index) {
  (void)record;
  RecordCounter* counter = static_cast<RecordCounter*>(user_data);
  EXPECT_EQ(counter->count, record_index);
  ++counter->count;
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

TEST(ProfileExecutableTraceRecordTest, ParsesFollowingTracePayload) {
  std::vector<uint8_t> payload = MakeExecutableTracePayload(
      sizeof(iree_hal_profile_executable_trace_record_t), 3,
      {0xAA, 0xBB, 0xCC});
  iree_hal_profile_file_record_t chunk = MakeChunk(payload);

  iree_hal_profile_executable_trace_record_t record;
  iree_const_byte_span_t trace_data = iree_const_byte_span_empty();
  IREE_ASSERT_OK(
      iree_profile_executable_trace_record_parse(&chunk, &record, &trace_data));
  EXPECT_EQ(sizeof(record), record.record_length);
  EXPECT_EQ(3u, record.data_length);
  ASSERT_EQ(3u, trace_data.data_length);
  EXPECT_EQ(0xAA, trace_data.data[0]);
  EXPECT_EQ(0xBB, trace_data.data[1]);
  EXPECT_EQ(0xCC, trace_data.data[2]);
}

TEST(ProfileExecutableTraceRecordTest, RejectsInlineTracePayload) {
  std::vector<uint8_t> payload = MakeExecutableTracePayload(
      sizeof(iree_hal_profile_executable_trace_record_t) + 1, 3,
      {0xAA, 0xBB, 0xCC});
  iree_hal_profile_file_record_t chunk = MakeChunk(payload);

  iree_hal_profile_executable_trace_record_t record;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS,
                        iree_profile_executable_trace_record_parse(
                            &chunk, &record, /*out_trace_data=*/NULL));
}

TEST(ProfileExecutableTraceRecordTest, RejectsExtraTracePayload) {
  std::vector<uint8_t> payload = MakeExecutableTracePayload(
      sizeof(iree_hal_profile_executable_trace_record_t), 2,
      {0xAA, 0xBB, 0xCC});
  iree_hal_profile_file_record_t chunk = MakeChunk(payload);

  iree_hal_profile_executable_trace_record_t record;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS,
                        iree_profile_executable_trace_record_parse(
                            &chunk, &record, /*out_trace_data=*/NULL));
}

TEST(ProfileExecutableTraceRecordTest, RejectsTruncatedTracePayload) {
  std::vector<uint8_t> payload = MakeExecutableTracePayload(
      sizeof(iree_hal_profile_executable_trace_record_t), 4,
      {0xAA, 0xBB, 0xCC});
  iree_hal_profile_file_record_t chunk = MakeChunk(payload);

  iree_hal_profile_executable_trace_record_t record;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS,
                        iree_profile_executable_trace_record_parse(
                            &chunk, &record, /*out_trace_data=*/NULL));
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

TEST(ProfileFileReaderTest, IteratesOnlyLogicalFileLength) {
  std::vector<uint8_t> storage;
  AppendProfileFileHeader(&storage, /*file_length=*/0);
  AppendProfileFileRecord(&storage,
                          IREE_HAL_PROFILE_FILE_RECORD_TYPE_SESSION_BEGIN);
  AppendProfileFileRecord(&storage, IREE_HAL_PROFILE_FILE_RECORD_TYPE_CHUNK);
  AppendProfileFileRecord(&storage,
                          IREE_HAL_PROFILE_FILE_RECORD_TYPE_SESSION_END);
  const uint64_t file_length = storage.size();
  reinterpret_cast<iree_hal_profile_file_header_t*>(storage.data())
      ->file_length = file_length;

  AppendProfileFileRecord(&storage, IREE_HAL_PROFILE_FILE_RECORD_TYPE_CHUNK);

  iree_io_file_contents_t contents;
  memset(&contents, 0, sizeof(contents));
  contents.const_buffer =
      iree_make_const_byte_span(storage.data(), storage.size());

  iree_profile_file_t profile_file;
  memset(&profile_file, 0, sizeof(profile_file));
  profile_file.contents = &contents;
  IREE_ASSERT_OK(iree_hal_profile_file_parse_header(
      contents.const_buffer, &profile_file.header,
      &profile_file.first_record_offset));
  profile_file.file_length = (iree_host_size_t)profile_file.header.file_length;

  RecordCounter counter = {};
  iree_profile_file_record_callback_t callback = {};
  callback.fn = CountProfileFileRecord;
  callback.user_data = &counter;
  IREE_ASSERT_OK(iree_profile_file_for_each_record(&profile_file, callback));
  EXPECT_EQ(3u, counter.count);
}

}  // namespace
