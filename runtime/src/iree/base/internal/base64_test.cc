// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/base64.h"

#include <cstring>
#include <string>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

// Helper: decode base64 string and return decoded bytes as a string.
std::string Decode(const char* input) {
  iree_string_view_t encoded = iree_make_cstring_view(input);
  iree_host_size_t max_size = iree_base64_decoded_size(encoded);
  std::vector<uint8_t> buffer(max_size);
  iree_host_size_t actual_length = 0;
  iree_status_t status =
      iree_base64_decode(encoded, buffer.size(), buffer.data(), &actual_length);
  IREE_EXPECT_OK(status);
  return std::string(reinterpret_cast<const char*>(buffer.data()),
                     actual_length);
}

// Helper: decode and expect failure with specific status code.
void ExpectDecodeError(iree_status_code_t expected_code, const char* input) {
  iree_string_view_t encoded = iree_make_cstring_view(input);
  iree_host_size_t max_size = iree_base64_decoded_size(encoded);
  std::vector<uint8_t> buffer(max_size + 1);  // Extra byte to avoid zero-size.
  iree_host_size_t actual_length = 0;
  IREE_EXPECT_STATUS_IS(
      expected_code, iree_base64_decode(encoded, buffer.size(), buffer.data(),
                                        &actual_length));
}

//===----------------------------------------------------------------------===//
// RFC 4648 Test Vectors
//===----------------------------------------------------------------------===//

TEST(Base64, EmptyInput) {
  iree_string_view_t encoded = iree_string_view_empty();
  EXPECT_EQ(iree_base64_decoded_size(encoded), 0u);
  iree_host_size_t actual_length = 0;
  IREE_EXPECT_OK(iree_base64_decode(encoded, 0, nullptr, &actual_length));
  EXPECT_EQ(actual_length, 0u);
}

TEST(Base64, RFC4648Vectors) {
  // From RFC 4648 section 10.
  EXPECT_EQ(Decode("Zg=="), "f");
  EXPECT_EQ(Decode("Zm8="), "fo");
  EXPECT_EQ(Decode("Zm9v"), "foo");
  EXPECT_EQ(Decode("Zm9vYg=="), "foob");
  EXPECT_EQ(Decode("Zm9vYmE="), "fooba");
  EXPECT_EQ(Decode("Zm9vYmFy"), "foobar");
}

TEST(Base64, RFC4648WithoutPadding) {
  // Same vectors without trailing = padding.
  EXPECT_EQ(Decode("Zg"), "f");
  EXPECT_EQ(Decode("Zm8"), "fo");
  EXPECT_EQ(Decode("Zm9v"), "foo");
  EXPECT_EQ(Decode("Zm9vYg"), "foob");
  EXPECT_EQ(Decode("Zm9vYmE"), "fooba");
  EXPECT_EQ(Decode("Zm9vYmFy"), "foobar");
}

//===----------------------------------------------------------------------===//
// Single Byte Values
//===----------------------------------------------------------------------===//

TEST(Base64, SingleByteNullByte) {
  // \0 encodes to "AA==" in base64.
  std::string result = Decode("AA==");
  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(static_cast<uint8_t>(result[0]), 0u);
}

TEST(Base64, SingleByteFF) {
  // 0xFF encodes to "/w==" in base64.
  std::string result = Decode("/w==");
  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(static_cast<uint8_t>(result[0]), 0xFF);
}

TEST(Base64, SingleByteSpace) {
  // 0x20 (space) encodes to "IA==" in base64.
  std::string result = Decode("IA==");
  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(static_cast<uint8_t>(result[0]), 0x20);
}

//===----------------------------------------------------------------------===//
// Multi-Byte Sequences (Tiktoken-Representative)
//===----------------------------------------------------------------------===//

TEST(Base64, TwoSpaces) {
  // "  " (two spaces, 0x20 0x20) encodes to "ICA=".
  std::string result = Decode("ICA=");
  ASSERT_EQ(result.size(), 2u);
  EXPECT_EQ(result, "  ");
}

TEST(Base64, TokenIn) {
  // "in" encodes to "aW4=".
  EXPECT_EQ(Decode("aW4="), "in");
}

TEST(Base64, LongerToken) {
  // " daycare" encodes to "IGRheWNhcmU=".
  EXPECT_EQ(Decode("IGRheWNhcmU="), " daycare");
}

//===----------------------------------------------------------------------===//
// Size Calculation
//===----------------------------------------------------------------------===//

TEST(Base64, DecodedSizeCalculation) {
  EXPECT_EQ(iree_base64_decoded_size(IREE_SV("")), 0u);
  EXPECT_EQ(iree_base64_decoded_size(IREE_SV("Zg==")), 3u);  // Max; actual=1.
  EXPECT_EQ(iree_base64_decoded_size(IREE_SV("Zg")), 1u);    // No padding.
  EXPECT_EQ(iree_base64_decoded_size(IREE_SV("Zm8=")), 3u);
  EXPECT_EQ(iree_base64_decoded_size(IREE_SV("Zm8")), 2u);
  EXPECT_EQ(iree_base64_decoded_size(IREE_SV("Zm9v")), 3u);
  EXPECT_EQ(iree_base64_decoded_size(IREE_SV("Zm9vYmFy")), 6u);
}

//===----------------------------------------------------------------------===//
// Error Cases
//===----------------------------------------------------------------------===//

TEST(Base64, InvalidCharacter) {
  ExpectDecodeError(IREE_STATUS_INVALID_ARGUMENT, "Z!==");
  ExpectDecodeError(IREE_STATUS_INVALID_ARGUMENT, "@@@@");
  ExpectDecodeError(IREE_STATUS_INVALID_ARGUMENT, "Zm9\x01");
}

TEST(Base64, InvalidLengthMod4Is1) {
  // After stripping padding, length mod 4 == 1 is invalid.
  ExpectDecodeError(IREE_STATUS_INVALID_ARGUMENT, "Z");
  ExpectDecodeError(IREE_STATUS_INVALID_ARGUMENT, "ZZZZZ");
}

TEST(Base64, ExcessivePadding) {
  // More than 2 padding characters is invalid.
  ExpectDecodeError(IREE_STATUS_INVALID_ARGUMENT, "Z===");
  ExpectDecodeError(IREE_STATUS_INVALID_ARGUMENT, "====");
}

TEST(Base64, BufferTooSmall) {
  iree_string_view_t encoded = IREE_SV("Zm9vYmFy");  // "foobar" = 6 bytes.
  uint8_t buffer[4];                                 // Too small.
  iree_host_size_t actual_length = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_base64_decode(encoded, sizeof(buffer), buffer, &actual_length));
}

//===----------------------------------------------------------------------===//
// All Byte Values Roundtrip
//===----------------------------------------------------------------------===//

TEST(Base64, AllByteValuesRoundtrip) {
  // Pre-encoded base64 for each single byte value (with padding stripped).
  // We test a representative subset to keep the test concise.
  struct TestCase {
    uint8_t byte_value;
    const char* base64;
  };
  TestCase cases[] = {
      {0x00, "AA"}, {0x01, "AQ"}, {0x0A, "Cg"}, {0x20, "IA"}, {0x21, "IQ"},
      {0x41, "QQ"}, {0x5A, "Wg"}, {0x61, "YQ"}, {0x7A, "eg"}, {0x7E, "fg"},
      {0x7F, "fw"}, {0x80, "gA"}, {0xA0, "oA"}, {0xAD, "rQ"}, {0xFF, "/w"},
  };
  for (const auto& tc : cases) {
    iree_string_view_t encoded = iree_make_cstring_view(tc.base64);
    iree_host_size_t max_size = iree_base64_decoded_size(encoded);
    std::vector<uint8_t> buffer(max_size);
    iree_host_size_t actual_length = 0;
    IREE_ASSERT_OK(iree_base64_decode(encoded, buffer.size(), buffer.data(),
                                      &actual_length))
        << "Failed to decode base64 for byte 0x" << std::hex
        << (int)tc.byte_value;
    ASSERT_EQ(actual_length, 1u) << "byte 0x" << std::hex << (int)tc.byte_value;
    EXPECT_EQ(buffer[0], tc.byte_value)
        << "byte 0x" << std::hex << (int)tc.byte_value;
  }
}

}  // namespace
