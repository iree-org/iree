// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/replay/dump.h"

#include <cstdint>
#include <string>
#include <vector>

#include "iree/hal/replay/file_writer.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

using ::testing::HasSubstr;

static iree_status_t AppendToString(void* user_data, iree_string_view_t text) {
  auto* output = static_cast<std::string*>(user_data);
  output->append(text.data, text.size);
  return iree_ok_status();
}

static std::vector<uint8_t> MakeReplayFileStorage() {
  std::vector<uint8_t> storage(4096, 0);
  iree_io_file_handle_t* file_handle = nullptr;
  IREE_CHECK_OK(iree_io_file_handle_wrap_host_allocation(
      IREE_IO_FILE_ACCESS_READ | IREE_IO_FILE_ACCESS_WRITE,
      iree_make_byte_span(storage.data(), storage.size()),
      iree_io_file_handle_release_callback_null(), iree_allocator_system(),
      &file_handle));

  iree_hal_replay_file_writer_t* writer = nullptr;
  IREE_CHECK_OK(iree_hal_replay_file_writer_create(
      file_handle, iree_allocator_system(), &writer));
  iree_io_file_handle_release(file_handle);

  iree_hal_replay_file_record_metadata_t session_metadata = {};
  session_metadata.sequence_ordinal = 0;
  session_metadata.record_type = IREE_HAL_REPLAY_FILE_RECORD_TYPE_SESSION;
  IREE_CHECK_OK(iree_hal_replay_file_writer_append_record(
      writer, &session_metadata, 0, nullptr, nullptr));

  iree_hal_replay_buffer_object_payload_t buffer_payload = {};
  buffer_payload.allocation_size = 256;
  buffer_payload.byte_length = 64;
  buffer_payload.allowed_usage = 0x11;
  iree_const_byte_span_t buffer_payload_span =
      iree_make_const_byte_span(&buffer_payload, sizeof(buffer_payload));
  iree_hal_replay_file_record_metadata_t object_metadata = {};
  object_metadata.sequence_ordinal = 1;
  object_metadata.object_id = 7;
  object_metadata.record_type = IREE_HAL_REPLAY_FILE_RECORD_TYPE_OBJECT;
  object_metadata.payload_type = IREE_HAL_REPLAY_PAYLOAD_TYPE_BUFFER_OBJECT;
  object_metadata.object_type = IREE_HAL_REPLAY_OBJECT_TYPE_BUFFER;
  IREE_CHECK_OK(iree_hal_replay_file_writer_append_record(
      writer, &object_metadata, 1, &buffer_payload_span, nullptr));

  IREE_CHECK_OK(iree_hal_replay_file_writer_close(writer));
  iree_hal_replay_file_writer_free(writer);
  return storage;
}

static iree_const_byte_span_t MakeReplayFileContents(
    const std::vector<uint8_t>& storage) {
  auto* file_header =
      reinterpret_cast<const iree_hal_replay_file_header_t*>(storage.data());
  return iree_make_const_byte_span(
      storage.data(), static_cast<iree_host_size_t>(file_header->file_length));
}

TEST(ReplayDumpTest, EmitsTextSummary) {
  std::vector<uint8_t> storage = MakeReplayFileStorage();
  iree_hal_replay_dump_options_t options =
      iree_hal_replay_dump_options_default();

  std::string output;
  IREE_ASSERT_OK(iree_hal_replay_dump_file(MakeReplayFileContents(storage),
                                           &options, AppendToString, &output,
                                           iree_allocator_system()));

  EXPECT_THAT(output, HasSubstr("IREE HAL replay v1.0"));
  EXPECT_THAT(output, HasSubstr("#0 session"));
  EXPECT_THAT(output, HasSubstr("#1 object"));
  EXPECT_THAT(output, HasSubstr("object=buffer"));
  EXPECT_THAT(output, HasSubstr("payload=buffer_object"));
  EXPECT_THAT(output, HasSubstr("allocation_size=256"));
}

TEST(ReplayDumpTest, EmitsJsonlWithPayloadRanges) {
  std::vector<uint8_t> storage = MakeReplayFileStorage();
  iree_hal_replay_dump_options_t options =
      iree_hal_replay_dump_options_default();
  options.format = IREE_HAL_REPLAY_DUMP_FORMAT_JSONL;

  std::string output;
  IREE_ASSERT_OK(iree_hal_replay_dump_file(MakeReplayFileContents(storage),
                                           &options, AppendToString, &output,
                                           iree_allocator_system()));

  EXPECT_THAT(output, HasSubstr("\"kind\":\"file\""));
  EXPECT_THAT(output, HasSubstr("\"kind\":\"session\""));
  EXPECT_THAT(output, HasSubstr("\"kind\":\"object\""));
  EXPECT_THAT(output, HasSubstr("\"payload_type\":\"buffer_object\""));
  EXPECT_THAT(output, HasSubstr("\"payload_range\""));
  EXPECT_THAT(output, HasSubstr("\"allocation_size\":256"));
}

}  // namespace
