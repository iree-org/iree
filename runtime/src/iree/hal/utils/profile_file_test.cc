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

struct ParsedRecord {
  // Parsed profile file record header.
  const iree_hal_profile_file_record_header_t* header = nullptr;
  // Content type string following |header|.
  std::string content_type;
  // Record name string following |content_type|.
  std::string name;
  // Payload bytes following |name|.
  std::string payload;
};

ParsedRecord ParseRecord(const std::vector<uint8_t>& storage,
                         iree_host_size_t* offset) {
  ParsedRecord record;
  const uint8_t* base = storage.data() + *offset;
  record.header =
      reinterpret_cast<const iree_hal_profile_file_record_header_t*>(base);
  EXPECT_EQ(sizeof(*record.header), record.header->header_length);
  EXPECT_GE(record.header->record_length, record.header->header_length);
  EXPECT_LE(*offset + record.header->record_length, storage.size());

  const char* content_type =
      reinterpret_cast<const char*>(base + record.header->header_length);
  record.content_type.assign(content_type, record.header->content_type_length);

  const char* name = content_type + record.header->content_type_length;
  record.name.assign(name, record.header->name_length);

  const char* payload = name + record.header->name_length;
  record.payload.assign(payload, record.header->payload_length);

  *offset += record.header->record_length;
  return record;
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

  const auto* file_header =
      reinterpret_cast<const iree_hal_profile_file_header_t*>(storage.data());
  EXPECT_EQ(IREE_HAL_PROFILE_FILE_MAGIC, file_header->magic);
  EXPECT_EQ(IREE_HAL_PROFILE_FILE_VERSION_MAJOR, file_header->version_major);
  EXPECT_EQ(IREE_HAL_PROFILE_FILE_VERSION_MINOR, file_header->version_minor);
  EXPECT_EQ(sizeof(*file_header), file_header->header_length);
  EXPECT_EQ(0u, file_header->flags);

  iree_host_size_t offset = sizeof(*file_header);
  ParsedRecord begin_record = ParseRecord(storage, &offset);
  EXPECT_EQ(IREE_HAL_PROFILE_FILE_RECORD_TYPE_SESSION_BEGIN,
            begin_record.header->record_type);
  EXPECT_EQ(42u, begin_record.header->session_id);
  EXPECT_EQ("application/vnd.iree.hal.profile.session",
            begin_record.content_type);
  EXPECT_EQ("test-session", begin_record.name);
  EXPECT_TRUE(begin_record.payload.empty());

  ParsedRecord chunk_record = ParseRecord(storage, &offset);
  EXPECT_EQ(IREE_HAL_PROFILE_FILE_RECORD_TYPE_CHUNK,
            chunk_record.header->record_type);
  EXPECT_EQ(42u, chunk_record.header->session_id);
  EXPECT_EQ(7u, chunk_record.header->stream_id);
  EXPECT_EQ(8u, chunk_record.header->event_id);
  EXPECT_EQ(9u, chunk_record.header->executable_id);
  EXPECT_EQ(10u, chunk_record.header->command_buffer_id);
  EXPECT_EQ(11u, chunk_record.header->physical_device_ordinal);
  EXPECT_EQ(12u, chunk_record.header->queue_ordinal);
  EXPECT_EQ(IREE_HAL_PROFILE_CHUNK_FLAG_TRUNCATED,
            chunk_record.header->chunk_flags);
  EXPECT_EQ("application/vnd.iree.test", chunk_record.content_type);
  EXPECT_EQ("test-chunk", chunk_record.name);
  ASSERT_EQ(sizeof(first_payload) + sizeof(second_payload),
            chunk_record.payload.size());
  EXPECT_EQ(0, memcmp(chunk_record.payload.data(), &first_payload,
                      sizeof(first_payload)));
  EXPECT_EQ(0, memcmp(chunk_record.payload.data() + sizeof(first_payload),
                      &second_payload, sizeof(second_payload)));

  ParsedRecord end_record = ParseRecord(storage, &offset);
  EXPECT_EQ(IREE_HAL_PROFILE_FILE_RECORD_TYPE_SESSION_END,
            end_record.header->record_type);
  EXPECT_EQ(42u, end_record.header->session_id);
  EXPECT_EQ((uint32_t)IREE_STATUS_CANCELLED,
            end_record.header->session_status_code);
  EXPECT_EQ("application/vnd.iree.hal.profile.session",
            end_record.content_type);
  EXPECT_EQ("test-session", end_record.name);
  EXPECT_TRUE(end_record.payload.empty());
}

}  // namespace
