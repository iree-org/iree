// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/utils/executable_header.h"

#include <cstring>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal {
namespace {

static std::vector<uint8_t> BuildExecutable(uint32_t version) {
  iree_flatbuffer_file_header_t header = {};
  std::memcpy(&header.magic, "TEST", 4);
  header.version = version;
  header.content_size = 4;
  std::vector<uint8_t> data(sizeof(header) + header.content_size, 0x42);
  std::memcpy(data.data(), &header, sizeof(header));
  return data;
}

static iree_const_byte_span_t ByteSpan(const std::vector<uint8_t>& data) {
  return iree_make_const_byte_span(data.data(), data.size());
}

TEST(ExecutableHeaderTest, LegacyReaderAcceptsVersionZero) {
  auto data = BuildExecutable(0);
  iree_const_byte_span_t content = {};
  IREE_ASSERT_OK(iree_hal_read_executable_flatbuffer_header(
      ByteSpan(data), false, "TEST", &content));
  EXPECT_EQ(content.data, data.data() + sizeof(iree_flatbuffer_file_header_t));
  ASSERT_EQ(content.data_length, 4u);
  EXPECT_EQ(content.data[0], 0x42);
}

TEST(ExecutableHeaderTest, LegacyReaderRejectsNewPayloadContract) {
  auto data = BuildExecutable(1);
  iree_const_byte_span_t content = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INCOMPATIBLE,
                        iree_hal_read_executable_flatbuffer_header(
                            ByteSpan(data), false, "TEST", &content));
}

TEST(ExecutableHeaderTest, VersionedReaderAcceptsOldAndCurrentPayloads) {
  for (uint32_t version : {0u, 1u}) {
    auto data = BuildExecutable(version);
    iree_const_byte_span_t content = {};
    IREE_ASSERT_OK(iree_hal_read_executable_flatbuffer_header_versioned(
        ByteSpan(data), false, "TEST", 1, &content));
    EXPECT_EQ(content.data,
              data.data() + sizeof(iree_flatbuffer_file_header_t));
    EXPECT_EQ(content.data_length, 4u);
  }
}

TEST(ExecutableHeaderTest, VersionedReaderRejectsFuturePayload) {
  auto data = BuildExecutable(2);
  iree_const_byte_span_t content = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INCOMPATIBLE,
                        iree_hal_read_executable_flatbuffer_header_versioned(
                            ByteSpan(data), false, "TEST", 1, &content));
}

TEST(ExecutableHeaderTest, VersionedReaderChecksFileExtents) {
  auto data = BuildExecutable(1);
  data.pop_back();
  iree_const_byte_span_t content = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        iree_hal_read_executable_flatbuffer_header_versioned(
                            ByteSpan(data), false, "TEST", 1, &content));
}

TEST(ExecutableHeaderTest, VersionedReaderChecksMagic) {
  auto data = BuildExecutable(1);
  iree_const_byte_span_t content = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_read_executable_flatbuffer_header_versioned(
                            ByteSpan(data), false, "OTHER", 1, &content));
}

TEST(ExecutableHeaderTest, VersionedReaderChecksHeaderSize) {
  auto data = BuildExecutable(1);
  data.resize(sizeof(iree_flatbuffer_file_header_t) - 1);
  iree_const_byte_span_t content = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_read_executable_flatbuffer_header_versioned(
                            ByteSpan(data), false, "TEST", 1, &content));
}

TEST(ExecutableHeaderTest, UnknownSizeStillChecksVersion) {
  auto data = BuildExecutable(1);
  auto unsized = iree_make_const_byte_span(data.data(), 0);
  iree_const_byte_span_t content = {};
  IREE_ASSERT_OK(iree_hal_read_executable_flatbuffer_header_versioned(
      unsized, true, "TEST", 1, &content));
  EXPECT_EQ(content.data_length, 4u);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INCOMPATIBLE,
                        iree_hal_read_executable_flatbuffer_header(
                            unsized, true, "TEST", &content));
}

}  // namespace
}  // namespace iree::hal
