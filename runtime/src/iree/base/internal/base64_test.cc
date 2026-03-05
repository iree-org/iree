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

//===----------------------------------------------------------------------===//
// Test Helpers
//===----------------------------------------------------------------------===//

// Decode base64 string and return decoded bytes as a string.
std::string Decode(const char* input) {
  iree_string_view_t encoded = iree_make_cstring_view(input);
  iree_host_size_t max_size = iree_base64_decoded_size(encoded);
  std::vector<uint8_t> buffer(max_size);
  iree_host_size_t actual_length = 0;
  iree_status_t status = iree_base64_decode(
      encoded, iree_make_byte_span(buffer.data(), buffer.size()),
      &actual_length);
  IREE_EXPECT_OK(status);
  return std::string(reinterpret_cast<const char*>(buffer.data()),
                     actual_length);
}

// Decode and expect failure with specific status code.
void ExpectDecodeError(iree_status_code_t expected_code, const char* input) {
  iree_string_view_t encoded = iree_make_cstring_view(input);
  iree_host_size_t max_size = iree_base64_decoded_size(encoded);
  std::vector<uint8_t> buffer(max_size + 1);  // Extra byte to avoid zero-size.
  iree_host_size_t actual_length = 0;
  IREE_EXPECT_STATUS_IS(
      expected_code,
      iree_base64_decode(encoded,
                         iree_make_byte_span(buffer.data(), buffer.size()),
                         &actual_length));
}

// Encode binary data and return base64 string.
std::string Encode(const void* data, size_t length) {
  iree_const_byte_span_t input =
      iree_make_const_byte_span(data, (iree_host_size_t)length);
  iree_host_size_t encoded_size = iree_base64_encoded_size(input.data_length);
  std::vector<char> buffer(encoded_size);
  iree_host_size_t actual_length = 0;
  iree_status_t status = iree_base64_encode(
      input, iree_make_mutable_string_view(buffer.data(), buffer.size()),
      &actual_length);
  IREE_EXPECT_OK(status);
  return std::string(buffer.data(), actual_length);
}

// Convenience overload for string literals.
std::string Encode(const char* str) { return Encode(str, strlen(str)); }

//===----------------------------------------------------------------------===//
// Decoding: RFC 4648 Test Vectors
//===----------------------------------------------------------------------===//

TEST(Base64Decode, EmptyInput) {
  iree_string_view_t encoded = iree_string_view_empty();
  EXPECT_EQ(iree_base64_decoded_size(encoded), 0u);
  iree_host_size_t actual_length = 0;
  IREE_EXPECT_OK(iree_base64_decode(encoded, iree_make_byte_span(NULL, 0),
                                    &actual_length));
  EXPECT_EQ(actual_length, 0u);
}

TEST(Base64Decode, RFC4648Vectors) {
  // From RFC 4648 section 10.
  EXPECT_EQ(Decode("Zg=="), "f");
  EXPECT_EQ(Decode("Zm8="), "fo");
  EXPECT_EQ(Decode("Zm9v"), "foo");
  EXPECT_EQ(Decode("Zm9vYg=="), "foob");
  EXPECT_EQ(Decode("Zm9vYmE="), "fooba");
  EXPECT_EQ(Decode("Zm9vYmFy"), "foobar");
}

TEST(Base64Decode, RFC4648WithoutPadding) {
  // Same vectors without trailing = padding.
  EXPECT_EQ(Decode("Zg"), "f");
  EXPECT_EQ(Decode("Zm8"), "fo");
  EXPECT_EQ(Decode("Zm9v"), "foo");
  EXPECT_EQ(Decode("Zm9vYg"), "foob");
  EXPECT_EQ(Decode("Zm9vYmE"), "fooba");
  EXPECT_EQ(Decode("Zm9vYmFy"), "foobar");
}

//===----------------------------------------------------------------------===//
// Decoding: Single Byte Values
//===----------------------------------------------------------------------===//

TEST(Base64Decode, SingleByteNullByte) {
  // \0 encodes to "AA==" in base64.
  std::string result = Decode("AA==");
  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(static_cast<uint8_t>(result[0]), 0u);
}

TEST(Base64Decode, SingleByteFF) {
  // 0xFF encodes to "/w==" in base64.
  std::string result = Decode("/w==");
  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(static_cast<uint8_t>(result[0]), 0xFF);
}

TEST(Base64Decode, SingleByteSpace) {
  // 0x20 (space) encodes to "IA==" in base64.
  std::string result = Decode("IA==");
  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(static_cast<uint8_t>(result[0]), 0x20);
}

//===----------------------------------------------------------------------===//
// Decoding: Multi-Byte Sequences (Tiktoken-Representative)
//===----------------------------------------------------------------------===//

TEST(Base64Decode, TwoSpaces) {
  // "  " (two spaces, 0x20 0x20) encodes to "ICA=".
  std::string result = Decode("ICA=");
  ASSERT_EQ(result.size(), 2u);
  EXPECT_EQ(result, "  ");
}

TEST(Base64Decode, TokenIn) {
  // "in" encodes to "aW4=".
  EXPECT_EQ(Decode("aW4="), "in");
}

TEST(Base64Decode, LongerToken) {
  // " daycare" encodes to "IGRheWNhcmU=".
  EXPECT_EQ(Decode("IGRheWNhcmU="), " daycare");
}

//===----------------------------------------------------------------------===//
// Decoded Size Calculation
//===----------------------------------------------------------------------===//

TEST(Base64Decode, DecodedSizeCalculation) {
  // iree_base64_decoded_size strips padding and returns exact decoded size.
  EXPECT_EQ(iree_base64_decoded_size(IREE_SV("")), 0u);

  // With padding: "Zg==" encodes "f" (1 byte). After stripping ==, length=2,
  // 2/4*3 + (2%4 - 1) = 0 + 1 = 1.
  EXPECT_EQ(iree_base64_decoded_size(IREE_SV("Zg==")), 1u);
  EXPECT_EQ(iree_base64_decoded_size(IREE_SV("Zg")), 1u);

  // "Zm8=" encodes "fo" (2 bytes). After stripping =, length=3,
  // 3/4*3 + (3%4 - 1) = 0 + 2 = 2.
  EXPECT_EQ(iree_base64_decoded_size(IREE_SV("Zm8=")), 2u);
  EXPECT_EQ(iree_base64_decoded_size(IREE_SV("Zm8")), 2u);

  // Full groups with no padding.
  EXPECT_EQ(iree_base64_decoded_size(IREE_SV("Zm9v")), 3u);
  EXPECT_EQ(iree_base64_decoded_size(IREE_SV("Zm9vYmFy")), 6u);

  // Two full groups + 2-char remainder = 6 + 1 = 7.
  EXPECT_EQ(iree_base64_decoded_size(IREE_SV("Zm9vYmFyZg==")), 7u);
  EXPECT_EQ(iree_base64_decoded_size(IREE_SV("Zm9vYmFyZg")), 7u);
}

//===----------------------------------------------------------------------===//
// Decoding: Error Cases
//===----------------------------------------------------------------------===//

TEST(Base64Decode, InvalidCharacter) {
  ExpectDecodeError(IREE_STATUS_INVALID_ARGUMENT, "Z!==");
  ExpectDecodeError(IREE_STATUS_INVALID_ARGUMENT, "@@@@");
  ExpectDecodeError(IREE_STATUS_INVALID_ARGUMENT, "Zm9\x01");
}

TEST(Base64Decode, InvalidLengthMod4Is1) {
  // After stripping padding, length mod 4 == 1 is invalid.
  ExpectDecodeError(IREE_STATUS_INVALID_ARGUMENT, "Z");
  ExpectDecodeError(IREE_STATUS_INVALID_ARGUMENT, "ZZZZZ");
}

TEST(Base64Decode, ExcessivePadding) {
  // More than 2 padding characters is invalid.
  ExpectDecodeError(IREE_STATUS_INVALID_ARGUMENT, "Z===");
  ExpectDecodeError(IREE_STATUS_INVALID_ARGUMENT, "====");
}

TEST(Base64Decode, BufferTooSmall) {
  iree_string_view_t encoded = IREE_SV("Zm9vYmFy");  // "foobar" = 6 bytes.
  uint8_t buffer[4];                                 // Too small.
  iree_host_size_t actual_length = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_base64_decode(encoded, iree_make_byte_span(buffer, sizeof(buffer)),
                         &actual_length));
}

//===----------------------------------------------------------------------===//
// Encoding: RFC 4648 Test Vectors
//===----------------------------------------------------------------------===//

TEST(Base64Encode, EmptyInput) {
  iree_const_byte_span_t data = iree_const_byte_span_empty();
  EXPECT_EQ(iree_base64_encoded_size(0), 0u);
  iree_host_size_t actual_length = 0;
  IREE_EXPECT_OK(iree_base64_encode(
      data, iree_make_mutable_string_view(NULL, 0), &actual_length));
  EXPECT_EQ(actual_length, 0u);
}

TEST(Base64Encode, RFC4648Vectors) {
  // From RFC 4648 section 10.
  EXPECT_EQ(Encode("f"), "Zg==");
  EXPECT_EQ(Encode("fo"), "Zm8=");
  EXPECT_EQ(Encode("foo"), "Zm9v");
  EXPECT_EQ(Encode("foob"), "Zm9vYg==");
  EXPECT_EQ(Encode("fooba"), "Zm9vYmE=");
  EXPECT_EQ(Encode("foobar"), "Zm9vYmFy");
}

TEST(Base64Encode, EncodedSizeCalculation) {
  EXPECT_EQ(iree_base64_encoded_size(0), 0u);
  EXPECT_EQ(iree_base64_encoded_size(1), 4u);
  EXPECT_EQ(iree_base64_encoded_size(2), 4u);
  EXPECT_EQ(iree_base64_encoded_size(3), 4u);
  EXPECT_EQ(iree_base64_encoded_size(4), 8u);
  EXPECT_EQ(iree_base64_encoded_size(5), 8u);
  EXPECT_EQ(iree_base64_encoded_size(6), 8u);
  EXPECT_EQ(iree_base64_encoded_size(7), 12u);
}

TEST(Base64Encode, EncodedSizeOverflow) {
  // Pathological input sizes must return IREE_HOST_SIZE_MAX (the overflow
  // sentinel) rather than wrapping to a small value.
  EXPECT_EQ(iree_base64_encoded_size(IREE_HOST_SIZE_MAX), IREE_HOST_SIZE_MAX);
  EXPECT_EQ(iree_base64_encoded_size(IREE_HOST_SIZE_MAX - 1),
            IREE_HOST_SIZE_MAX);
}

TEST(Base64Encode, EncodeRejectsOverflow) {
  // Crafted span with pathological length. The encode function should reject
  // this with OUT_OF_RANGE rather than overflowing the size calculation.
  // Uses a valid pointer with an absurd length to avoid triggering
  // IREE_ASSERT_ARGUMENT(data.data) in debug builds.
  uint8_t one_byte = 0;
  iree_const_byte_span_t huge = {&one_byte, IREE_HOST_SIZE_MAX};
  char dummy;
  iree_host_size_t actual_length = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_base64_encode(huge, iree_make_mutable_string_view(&dummy, 1),
                         &actual_length));
}

TEST(Base64Encode, BufferTooSmall) {
  const char* input = "foobar";
  iree_const_byte_span_t data = iree_make_const_byte_span(input, strlen(input));
  char buffer[4];  // Needs 8.
  iree_host_size_t actual_length = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_base64_encode(data, iree_make_mutable_string_view(buffer, 4),
                         &actual_length));
}

TEST(Base64Encode, SingleByteValues) {
  // Spot-check single byte encoding.
  uint8_t byte;

  byte = 0x00;
  EXPECT_EQ(Encode(&byte, 1), "AA==");

  byte = 0xFF;
  EXPECT_EQ(Encode(&byte, 1), "/w==");

  byte = 0x20;
  EXPECT_EQ(Encode(&byte, 1), "IA==");
}

//===----------------------------------------------------------------------===//
// Roundtrip: Encode -> Decode
//===----------------------------------------------------------------------===//

TEST(Base64Roundtrip, AllSingleByteValues) {
  for (int i = 0; i <= 0xFF; ++i) {
    uint8_t byte = static_cast<uint8_t>(i);
    iree_const_byte_span_t input = iree_make_const_byte_span(&byte, 1);

    // Encode.
    char encode_buffer[4];
    iree_host_size_t encode_length = 0;
    IREE_ASSERT_OK(iree_base64_encode(
        input,
        iree_make_mutable_string_view(encode_buffer, sizeof(encode_buffer)),
        &encode_length))
        << "Failed to encode byte 0x" << std::hex << i;
    ASSERT_EQ(encode_length, 4u);

    // Decode.
    iree_string_view_t encoded =
        iree_make_string_view(encode_buffer, encode_length);
    uint8_t decode_buffer[1];
    iree_host_size_t decode_length = 0;
    IREE_ASSERT_OK(iree_base64_decode(
        encoded, iree_make_byte_span(decode_buffer, sizeof(decode_buffer)),
        &decode_length))
        << "Failed to decode base64 for byte 0x" << std::hex << i;
    ASSERT_EQ(decode_length, 1u) << "byte 0x" << std::hex << i;
    EXPECT_EQ(decode_buffer[0], byte) << "byte 0x" << std::hex << i;
  }
}

TEST(Base64Roundtrip, VariousLengths) {
  // Test roundtrip for data lengths 0 through 33 (covers all remainder cases
  // across multiple groups).
  for (size_t length = 0; length <= 33; ++length) {
    std::vector<uint8_t> original(length);
    for (size_t i = 0; i < length; ++i) {
      original[i] = static_cast<uint8_t>((i * 7 + 13) & 0xFF);
    }

    iree_const_byte_span_t input =
        iree_make_const_byte_span(original.data(), original.size());

    // Encode.
    iree_host_size_t encoded_size = iree_base64_encoded_size(input.data_length);
    std::vector<char> encode_buffer(encoded_size);
    iree_host_size_t encode_length = 0;
    IREE_ASSERT_OK(
        iree_base64_encode(input,
                           iree_make_mutable_string_view(encode_buffer.data(),
                                                         encode_buffer.size()),
                           &encode_length))
        << "Failed to encode data of length " << length;

    // Decode.
    iree_string_view_t encoded =
        iree_make_string_view(encode_buffer.data(), encode_length);
    iree_host_size_t decoded_size = iree_base64_decoded_size(encoded);
    EXPECT_EQ(decoded_size, length)
        << "Decoded size mismatch for length " << length;

    std::vector<uint8_t> decode_buffer(decoded_size);
    iree_host_size_t decode_length = 0;
    IREE_ASSERT_OK(iree_base64_decode(
        encoded,
        iree_make_byte_span(decode_buffer.data(), decode_buffer.size()),
        &decode_length))
        << "Failed to decode for original length " << length;
    ASSERT_EQ(decode_length, length)
        << "Decoded length mismatch for length " << length;
    EXPECT_EQ(
        std::string(reinterpret_cast<const char*>(original.data()), length),
        std::string(reinterpret_cast<const char*>(decode_buffer.data()),
                    decode_length))
        << "Data mismatch for length " << length;
  }
}

//===----------------------------------------------------------------------===//
// Decode: All Byte Values (Pre-Computed)
//===----------------------------------------------------------------------===//

TEST(Base64Decode, AllByteValuesPreComputed) {
  // Pre-encoded base64 for each single byte value (without padding).
  // A representative subset to verify the decode table is correct.
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
    IREE_ASSERT_OK(iree_base64_decode(
        encoded, iree_make_byte_span(buffer.data(), buffer.size()),
        &actual_length))
        << "Failed to decode base64 for byte 0x" << std::hex
        << (int)tc.byte_value;
    ASSERT_EQ(actual_length, 1u) << "byte 0x" << std::hex << (int)tc.byte_value;
    EXPECT_EQ(buffer[0], tc.byte_value)
        << "byte 0x" << std::hex << (int)tc.byte_value;
  }
}

}  // namespace
