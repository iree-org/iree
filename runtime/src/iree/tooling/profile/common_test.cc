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
#include "iree/tooling/profile/internal.h"

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

}  // namespace
