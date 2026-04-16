// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/utils/profile_file.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "iree/hal/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

bool StringViewEqual(iree_string_view_t lhs, const char* rhs) {
  return iree_string_view_equal(lhs, iree_make_cstring_view(rhs));
}

TEST(ProfileFileSinkTest, WritesProfileBundleRecords) {
  std::vector<uint8_t> storage(4096, 0);
  iree_io_file_handle_t* file_handle = nullptr;
  IREE_ASSERT_OK(iree_io_file_handle_wrap_host_allocation(
      IREE_IO_FILE_ACCESS_READ | IREE_IO_FILE_ACCESS_WRITE,
      iree_make_byte_span(storage.data(), storage.size()),
      iree_io_file_handle_release_callback_null(), iree_allocator_system(),
      &file_handle));

  iree_hal_profile_sink_t* sink = nullptr;
  IREE_ASSERT_OK(iree_hal_profile_file_sink_create(
      file_handle, iree_allocator_system(), &sink));
  iree_io_file_handle_release(file_handle);

  iree_hal_profile_chunk_metadata_t session_metadata =
      iree_hal_profile_chunk_metadata_default();
  session_metadata.content_type = IREE_HAL_PROFILE_CONTENT_TYPE_SESSION;
  session_metadata.name = IREE_SV("test-session");
  session_metadata.session_id = 42;
  IREE_ASSERT_OK(iree_hal_profile_sink_begin_session(sink, &session_metadata));

  const uint32_t first_payload = 0x12345678u;
  const uint16_t second_payload = 0xABCDu;
  iree_const_byte_span_t iovecs[2] = {
      iree_make_const_byte_span(&first_payload, sizeof(first_payload)),
      iree_make_const_byte_span(&second_payload, sizeof(second_payload)),
  };
  iree_hal_profile_chunk_metadata_t chunk_metadata =
      iree_hal_profile_chunk_metadata_default();
  chunk_metadata.content_type = IREE_SV("application/vnd.iree.test");
  chunk_metadata.name = IREE_SV("test-chunk");
  chunk_metadata.session_id = 42;
  chunk_metadata.stream_id = 7;
  chunk_metadata.event_id = 8;
  chunk_metadata.executable_id = 9;
  chunk_metadata.command_buffer_id = 10;
  chunk_metadata.physical_device_ordinal = 11;
  chunk_metadata.queue_ordinal = 12;
  chunk_metadata.flags = IREE_HAL_PROFILE_CHUNK_FLAG_TRUNCATED;
  IREE_ASSERT_OK(iree_hal_profile_sink_write(sink, &chunk_metadata, 2, iovecs));

  IREE_ASSERT_OK(iree_hal_profile_sink_end_session(sink, &session_metadata,
                                                   IREE_STATUS_CANCELLED));
  iree_hal_profile_sink_release(sink);

  iree_hal_profile_file_header_t file_header;
  iree_host_size_t offset = 0;
  IREE_ASSERT_OK(iree_hal_profile_file_parse_header(
      iree_make_const_byte_span(storage.data(), storage.size()), &file_header,
      &offset));
  EXPECT_EQ(IREE_HAL_PROFILE_FILE_MAGIC, file_header.magic);
  EXPECT_EQ(IREE_HAL_PROFILE_FILE_VERSION_MAJOR, file_header.version_major);
  EXPECT_EQ(IREE_HAL_PROFILE_FILE_VERSION_MINOR, file_header.version_minor);
  EXPECT_EQ(sizeof(file_header), file_header.header_length);
  EXPECT_EQ(0u, file_header.flags);

  iree_hal_profile_file_record_t begin_record;
  IREE_ASSERT_OK(iree_hal_profile_file_parse_record(
      iree_make_const_byte_span(storage.data(), storage.size()), offset,
      &begin_record, &offset));
  EXPECT_EQ(IREE_HAL_PROFILE_FILE_RECORD_TYPE_SESSION_BEGIN,
            begin_record.header.record_type);
  EXPECT_EQ(42u, begin_record.header.session_id);
  EXPECT_TRUE(StringViewEqual(begin_record.content_type,
                              "application/vnd.iree.hal.profile.session"));
  EXPECT_TRUE(StringViewEqual(begin_record.name, "test-session"));
  EXPECT_EQ(0u, begin_record.payload.data_length);

  iree_hal_profile_file_record_t chunk_record;
  IREE_ASSERT_OK(iree_hal_profile_file_parse_record(
      iree_make_const_byte_span(storage.data(), storage.size()), offset,
      &chunk_record, &offset));
  EXPECT_EQ(IREE_HAL_PROFILE_FILE_RECORD_TYPE_CHUNK,
            chunk_record.header.record_type);
  EXPECT_EQ(42u, chunk_record.header.session_id);
  EXPECT_EQ(7u, chunk_record.header.stream_id);
  EXPECT_EQ(8u, chunk_record.header.event_id);
  EXPECT_EQ(9u, chunk_record.header.executable_id);
  EXPECT_EQ(10u, chunk_record.header.command_buffer_id);
  EXPECT_EQ(11u, chunk_record.header.physical_device_ordinal);
  EXPECT_EQ(12u, chunk_record.header.queue_ordinal);
  EXPECT_EQ(IREE_HAL_PROFILE_CHUNK_FLAG_TRUNCATED,
            chunk_record.header.chunk_flags);
  EXPECT_TRUE(
      StringViewEqual(chunk_record.content_type, "application/vnd.iree.test"));
  EXPECT_TRUE(StringViewEqual(chunk_record.name, "test-chunk"));
  ASSERT_EQ(sizeof(first_payload) + sizeof(second_payload),
            chunk_record.payload.data_length);
  EXPECT_EQ(0, memcmp(chunk_record.payload.data, &first_payload,
                      sizeof(first_payload)));
  EXPECT_EQ(0, memcmp(chunk_record.payload.data + sizeof(first_payload),
                      &second_payload, sizeof(second_payload)));

  iree_hal_profile_file_record_t end_record;
  IREE_ASSERT_OK(iree_hal_profile_file_parse_record(
      iree_make_const_byte_span(storage.data(), storage.size()), offset,
      &end_record, &offset));
  EXPECT_EQ(IREE_HAL_PROFILE_FILE_RECORD_TYPE_SESSION_END,
            end_record.header.record_type);
  EXPECT_EQ(42u, end_record.header.session_id);
  EXPECT_EQ((uint32_t)IREE_STATUS_CANCELLED,
            end_record.header.session_status_code);
  EXPECT_TRUE(StringViewEqual(end_record.content_type,
                              "application/vnd.iree.hal.profile.session"));
  EXPECT_TRUE(StringViewEqual(end_record.name, "test-session"));
  EXPECT_EQ(0u, end_record.payload.data_length);
}

TEST(ProfileFileParseTest, RejectsBadMagic) {
  std::vector<uint8_t> storage(sizeof(iree_hal_profile_file_header_t), 0);
  iree_hal_profile_file_header_t file_header;
  iree_host_size_t offset = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_profile_file_parse_header(
          iree_make_const_byte_span(storage.data(), storage.size()),
          &file_header, &offset));
}

TEST(ProfileFileParseTest, RejectsTruncatedRecord) {
  std::vector<uint8_t> storage(
      sizeof(iree_hal_profile_file_header_t) +
          sizeof(iree_hal_profile_file_record_header_t),
      0);
  auto* file_header =
      reinterpret_cast<iree_hal_profile_file_header_t*>(storage.data());
  file_header->magic = IREE_HAL_PROFILE_FILE_MAGIC;
  file_header->version_major = IREE_HAL_PROFILE_FILE_VERSION_MAJOR;
  file_header->version_minor = IREE_HAL_PROFILE_FILE_VERSION_MINOR;
  file_header->header_length = sizeof(*file_header);

  auto* record_header =
      reinterpret_cast<iree_hal_profile_file_record_header_t*>(
          storage.data() + sizeof(*file_header));
  record_header->record_length = sizeof(*record_header) + 1;
  record_header->payload_length = 1;
  record_header->header_length = sizeof(*record_header);
  record_header->record_type = IREE_HAL_PROFILE_FILE_RECORD_TYPE_CHUNK;

  iree_hal_profile_file_header_t parsed_header;
  iree_host_size_t offset = 0;
  IREE_ASSERT_OK(iree_hal_profile_file_parse_header(
      iree_make_const_byte_span(storage.data(), storage.size()), &parsed_header,
      &offset));

  iree_hal_profile_file_record_t record;
  iree_host_size_t next_offset = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_DATA_LOSS,
      iree_hal_profile_file_parse_record(
          iree_make_const_byte_span(storage.data(), storage.size()), offset,
          &record, &next_offset));
}

}  // namespace
