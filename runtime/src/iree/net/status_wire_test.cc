// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/status_wire.h"

#include <cstring>
#include <vector>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

// Helper to serialize a status and return the wire bytes.
static std::vector<uint8_t> Serialize(iree_status_t status) {
  iree_host_size_t size = 0;
  iree_net_status_wire_size(status, &size);
  std::vector<uint8_t> buffer(size);
  iree_status_t serialize_status = iree_net_status_wire_serialize(
      status, iree_make_byte_span(buffer.data(), buffer.size()));
  IREE_CHECK_OK(serialize_status);
  return buffer;
}

// Helper to deserialize wire bytes into a status.
static iree_status_t Deserialize(const std::vector<uint8_t>& buffer) {
  iree_status_t out_status = iree_ok_status();
  iree_status_t deserialize_status = iree_net_status_wire_deserialize(
      iree_make_const_byte_span(buffer.data(), buffer.size()), &out_status);
  IREE_CHECK_OK(deserialize_status);
  return out_status;
}

//===----------------------------------------------------------------------===//
// OK status
//===----------------------------------------------------------------------===//

TEST(StatusWire, OkStatusSize) {
  iree_host_size_t size = 0;
  iree_net_status_wire_size(iree_ok_status(), &size);
  EXPECT_EQ(size, sizeof(iree_net_status_wire_header_t));
}

TEST(StatusWire, OkStatusRoundTrip) {
  auto buffer = Serialize(iree_ok_status());
  EXPECT_EQ(buffer.size(), sizeof(iree_net_status_wire_header_t));

  // Verify header fields.
  iree_net_status_wire_header_t header;
  std::memcpy(&header, buffer.data(), sizeof(header));
  EXPECT_EQ(header.version, IREE_NET_STATUS_WIRE_VERSION);
  EXPECT_EQ(header.status_code, IREE_STATUS_OK);
  EXPECT_EQ(header.entry_count, 0);
  EXPECT_EQ(header.total_size, sizeof(iree_net_status_wire_header_t));

  iree_status_t result = Deserialize(buffer);
  EXPECT_TRUE(iree_status_is_ok(result));
}

//===----------------------------------------------------------------------===//
// Error code only (no storage)
//===----------------------------------------------------------------------===//

TEST(StatusWire, CodeOnlyRoundTrip) {
  iree_status_t status = iree_status_from_code(IREE_STATUS_CANCELLED);
  auto buffer = Serialize(status);

  // Header present, but code-only status has no entries when status mode is 0.
  iree_net_status_wire_header_t header;
  std::memcpy(&header, buffer.data(), sizeof(header));
  EXPECT_EQ(header.status_code, IREE_STATUS_CANCELLED);

  iree_status_t result = Deserialize(buffer);
  EXPECT_TRUE(iree_status_is_cancelled(result));
  iree_status_ignore(result);
}

//===----------------------------------------------------------------------===//
// Status with message
//===----------------------------------------------------------------------===//

TEST(StatusWire, StatusWithMessageRoundTrip) {
  iree_status_t status =
      iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "bad value: %d", 42);

  auto buffer = Serialize(status);

  iree_net_status_wire_header_t header;
  std::memcpy(&header, buffer.data(), sizeof(header));
  EXPECT_EQ(header.status_code, IREE_STATUS_INVALID_ARGUMENT);

  iree_status_t result = Deserialize(buffer);
  EXPECT_TRUE(iree_status_is_invalid_argument(result));

  // The deserialized status should contain the original message.
  iree_string_view_t message = iree_status_message(result);
  std::string message_str(message.data, message.size);
  EXPECT_NE(message_str.find("bad value: 42"), std::string::npos);

  iree_status_ignore(result);
  iree_status_ignore(status);
}

//===----------------------------------------------------------------------===//
// Status with annotations
//===----------------------------------------------------------------------===//

TEST(StatusWire, StatusWithAnnotationsRoundTrip) {
  iree_status_t status = iree_make_status(IREE_STATUS_INTERNAL, "root cause");
  status = iree_status_annotate(status,
                                iree_make_cstring_view("additional context"));

  auto buffer = Serialize(status);

  iree_status_t result = Deserialize(buffer);
  EXPECT_TRUE(iree_status_is_internal(result));

  // Format the full status to check both message and annotation are present.
  char format_buffer[512];
  iree_host_size_t format_length = 0;
  iree_status_format(result, sizeof(format_buffer), format_buffer,
                     &format_length);
  std::string formatted(format_buffer, format_length);
  EXPECT_NE(formatted.find("root cause"), std::string::npos);
  EXPECT_NE(formatted.find("additional context"), std::string::npos);

  iree_status_ignore(result);
  iree_status_ignore(status);
}

//===----------------------------------------------------------------------===//
// Source location preservation
//===----------------------------------------------------------------------===//

TEST(StatusWire, SourceLocationPreserved) {
  iree_status_t status =
      iree_make_status(IREE_STATUS_NOT_FOUND, "missing thing");

  // The source location should point to this test file.
  iree_status_source_location_t original_location =
      iree_status_source_location(status);

  auto buffer = Serialize(status);
  iree_status_t result = Deserialize(buffer);

  iree_status_source_location_t deserialized_location =
      iree_status_source_location(result);

  if (original_location.file) {
    // When source location is available, verify it round-trips.
    ASSERT_NE(deserialized_location.file, nullptr);
    // The filename might be the full path or just the basename — as long as it
    // matches what was serialized.
    EXPECT_STREQ(deserialized_location.file, original_location.file);
    EXPECT_EQ(deserialized_location.line, original_location.line);
  }

  iree_status_ignore(result);
  iree_status_ignore(status);
}

//===----------------------------------------------------------------------===//
// Wire format validation
//===----------------------------------------------------------------------===//

TEST(StatusWire, DeserializeTruncatedHeader) {
  uint8_t short_buffer[4] = {0};
  iree_status_t out_status = iree_ok_status();
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_status_wire_deserialize(
          iree_make_const_byte_span(short_buffer, sizeof(short_buffer)),
          &out_status));
}

TEST(StatusWire, DeserializeBadVersion) {
  iree_net_status_wire_header_t header;
  std::memset(&header, 0, sizeof(header));
  header.version = 255;  // Invalid version.
  header.status_code = IREE_STATUS_OK;
  header.entry_count = 0;
  header.total_size = sizeof(header);

  iree_status_t out_status = iree_ok_status();
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_status_wire_deserialize(
          iree_make_const_byte_span((const uint8_t*)&header, sizeof(header)),
          &out_status));
}

TEST(StatusWire, DeserializeTotalSizeExceedsData) {
  iree_net_status_wire_header_t header;
  std::memset(&header, 0, sizeof(header));
  header.version = IREE_NET_STATUS_WIRE_VERSION;
  header.status_code = IREE_STATUS_INTERNAL;
  header.entry_count = 1;
  header.total_size = 1024;  // Claims to be much larger than actual data.

  iree_status_t out_status = iree_ok_status();
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_status_wire_deserialize(
          iree_make_const_byte_span((const uint8_t*)&header, sizeof(header)),
          &out_status));
}

TEST(StatusWire, DeserializeTruncatedEntry) {
  // Header says 1 entry but no entry data follows.
  iree_net_status_wire_header_t header;
  std::memset(&header, 0, sizeof(header));
  header.version = IREE_NET_STATUS_WIRE_VERSION;
  header.status_code = IREE_STATUS_INTERNAL;
  header.entry_count = 1;
  header.total_size = sizeof(header);  // No room for entries.

  iree_status_t out_status = iree_ok_status();
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_status_wire_deserialize(
          iree_make_const_byte_span((const uint8_t*)&header, sizeof(header)),
          &out_status));
}

//===----------------------------------------------------------------------===//
// Serialization buffer too small
//===----------------------------------------------------------------------===//

TEST(StatusWire, SerializeBufferTooSmall) {
  iree_status_t status = iree_make_status(IREE_STATUS_INTERNAL, "some message");
  uint8_t tiny_buffer[4];
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      iree_net_status_wire_serialize(
          status, iree_make_byte_span(tiny_buffer, sizeof(tiny_buffer))));
  iree_status_ignore(status);
}

//===----------------------------------------------------------------------===//
// All status codes
//===----------------------------------------------------------------------===//

TEST(StatusWire, AllStatusCodesRoundTrip) {
  // Verify every status code survives serialization.
  for (int code = 0; code <= IREE_STATUS_CODE_MASK; ++code) {
    iree_status_t status = iree_status_from_code((iree_status_code_t)code);
    auto buffer = Serialize(status);
    iree_status_t result = Deserialize(buffer);
    EXPECT_EQ(iree_status_code(result), (iree_status_code_t)code)
        << "code=" << code;
    iree_status_ignore(result);
    // status_from_code doesn't allocate, no need to free.
  }
}

//===----------------------------------------------------------------------===//
// 8-byte alignment
//===----------------------------------------------------------------------===//

TEST(StatusWire, AlignmentPadding) {
  // Use a message with length not divisible by 8 to exercise padding.
  iree_status_t status = iree_make_status(IREE_STATUS_INTERNAL, "hi");

  auto buffer = Serialize(status);

  // Total size should be 8-byte aligned.
  EXPECT_EQ(buffer.size() % 8, 0);

  // Verify round-trip still works with padding.
  iree_status_t result = Deserialize(buffer);
  EXPECT_TRUE(iree_status_is_internal(result));
  iree_status_ignore(result);
  iree_status_ignore(status);
}

}  // namespace
