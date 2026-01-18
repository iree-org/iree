// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/tokenizer/normalizer.h"
#include "iree/tokenizer/transforms/transform.h"

namespace {

using iree::Status;
using iree::StatusCode;
using iree::testing::status::StatusIs;

// Callback that concatenates all segments into a single string.
struct ConcatContext {
  std::string* output;
};

static iree_status_t ConcatenateSegments(void* user_data,
                                         iree_string_view_list_t segments) {
  auto* context = static_cast<ConcatContext*>(user_data);
  for (size_t i = 0; i < segments.count; ++i) {
    context->output->append(segments.values[i].data, segments.values[i].size);
  }
  return iree_ok_status();
}

// Callback that counts segments.
struct CountContext {
  size_t count;
};

static iree_status_t CountSegments(void* user_data,
                                   iree_string_view_list_t segments) {
  auto* context = static_cast<CountContext*>(user_data);
  context->count += segments.count;
  return iree_ok_status();
}

class ByteLevelTransformTest : public ::testing::Test {
 protected:
  std::string Encode(const char* text,
                     iree_tokenizer_byte_level_flags_t flags =
                         IREE_TOKENIZER_BYTE_LEVEL_FLAG_ADD_PREFIX_SPACE) {
    iree_tokenizer_text_transform_t transform;
    iree_tokenizer_text_transform_initialize_byte_level(flags, &transform);

    std::string output;
    ConcatContext context = {&output};
    iree_status_t status = iree_tokenizer_text_transform_encode(
        NULL, &transform, IREE_SV(text), ConcatenateSegments, &context);
    IREE_EXPECT_OK(status);

    iree_tokenizer_text_transform_deinitialize(&transform);
    return output;
  }

  std::string EncodeBytes(const std::string& input,
                          iree_tokenizer_byte_level_flags_t flags =
                              IREE_TOKENIZER_BYTE_LEVEL_FLAG_DEFAULT) {
    iree_tokenizer_text_transform_t transform;
    iree_tokenizer_text_transform_initialize_byte_level(flags, &transform);

    std::string output;
    ConcatContext context = {&output};
    iree_status_t status = iree_tokenizer_text_transform_encode(
        NULL, &transform, iree_make_string_view(input.data(), input.size()),
        ConcatenateSegments, &context);
    IREE_EXPECT_OK(status);

    iree_tokenizer_text_transform_deinitialize(&transform);
    return output;
  }

  std::string Decode(const std::string& text,
                     iree_tokenizer_byte_level_flags_t flags =
                         IREE_TOKENIZER_BYTE_LEVEL_FLAG_ADD_PREFIX_SPACE) {
    iree_tokenizer_text_transform_t transform;
    iree_tokenizer_text_transform_initialize_byte_level(flags, &transform);

    char decoded[1024];
    iree_host_size_t decoded_size = 0;
    iree_status_t status = iree_tokenizer_text_transform_decode(
        &transform, iree_make_string_view(text.data(), text.size()), decoded,
        sizeof(decoded), &decoded_size);
    IREE_EXPECT_OK(status);

    iree_tokenizer_text_transform_deinitialize(&transform);
    return std::string(decoded, decoded_size);
  }
};

TEST_F(ByteLevelTransformTest, EmptyInput) {
  iree_tokenizer_text_transform_t transform;
  iree_tokenizer_text_transform_initialize_byte_level(
      IREE_TOKENIZER_BYTE_LEVEL_FLAG_ADD_PREFIX_SPACE, &transform);

  CountContext context = {0};
  iree_status_t status = iree_tokenizer_text_transform_encode(
      NULL, &transform, IREE_SVL(""), CountSegments, &context);
  IREE_EXPECT_OK(status);
  EXPECT_EQ(context.count, 0u);

  iree_tokenizer_text_transform_deinitialize(&transform);
}

TEST_F(ByteLevelTransformTest, PrintableAsciiDirect) {
  auto output = Encode("A", IREE_TOKENIZER_BYTE_LEVEL_FLAG_DEFAULT);
  EXPECT_EQ(output, "A");
}

TEST_F(ByteLevelTransformTest, HelloWorld) {
  auto output = Encode("hello", IREE_TOKENIZER_BYTE_LEVEL_FLAG_DEFAULT);
  EXPECT_EQ(output, "hello");
}

TEST_F(ByteLevelTransformTest, AddPrefixSpace) {
  auto output =
      Encode("hello", IREE_TOKENIZER_BYTE_LEVEL_FLAG_ADD_PREFIX_SPACE);
  EXPECT_EQ(output, "\xC4\xA0hello");
}

TEST_F(ByteLevelTransformTest, NoPrefixSpaceWhenStartsWithSpace) {
  auto output =
      Encode(" hello", IREE_TOKENIZER_BYTE_LEVEL_FLAG_ADD_PREFIX_SPACE);
  EXPECT_EQ(output.substr(0, 2), "\xC4\xA0");
}

TEST_F(ByteLevelTransformTest, SpaceMapping) {
  auto output = Encode(" ", IREE_TOKENIZER_BYTE_LEVEL_FLAG_DEFAULT);
  EXPECT_EQ(output, "\xC4\xA0");  // 0x120 in UTF-8
}

TEST_F(ByteLevelTransformTest, MultipleSpaces) {
  auto output = Encode("   ", IREE_TOKENIZER_BYTE_LEVEL_FLAG_DEFAULT);
  EXPECT_EQ(output, "\xC4\xA0\xC4\xA0\xC4\xA0");
}

TEST_F(ByteLevelTransformTest, HelloWorldWithSpaces) {
  auto output = Encode("hello world", IREE_TOKENIZER_BYTE_LEVEL_FLAG_DEFAULT);
  EXPECT_EQ(output, "hello\xC4\xA0world");
}

TEST_F(ByteLevelTransformTest, ControlCharacters) {
  std::string input("\x00", 1);
  auto output = EncodeBytes(input);
  EXPECT_EQ(output, "\xC4\x80");  // 0x100 in UTF-8
}

TEST_F(ByteLevelTransformTest, HighBytes) {
  std::string input("\xFF", 1);
  auto output = EncodeBytes(input);
  EXPECT_EQ(output, "\xC3\xBF");  // 0xFF in UTF-8
}

TEST_F(ByteLevelTransformTest, TabCharacter) {
  auto output = Encode("\t", IREE_TOKENIZER_BYTE_LEVEL_FLAG_DEFAULT);
  EXPECT_EQ(output, "\xC4\x89");
}

TEST_F(ByteLevelTransformTest, NewlineCharacter) {
  auto output = Encode("\n", IREE_TOKENIZER_BYTE_LEVEL_FLAG_DEFAULT);
  EXPECT_EQ(output, "\xC4\x8A");
}

TEST_F(ByteLevelTransformTest, AllPrintableAscii) {
  std::string printable;
  for (int c = 0x21; c <= 0x7E; ++c) {
    printable += static_cast<char>(c);
  }
  auto output =
      Encode(printable.c_str(), IREE_TOKENIZER_BYTE_LEVEL_FLAG_DEFAULT);
  EXPECT_EQ(output, printable);
}

TEST_F(ByteLevelTransformTest, MixedContent) {
  auto output = Encode("a b\tc", IREE_TOKENIZER_BYTE_LEVEL_FLAG_DEFAULT);
  EXPECT_EQ(output,
            "a\xC4\xA0"
            "b\xC4\x89"
            "c");
}

//===----------------------------------------------------------------------===//
// Lookup Table Correctness Tests
//===----------------------------------------------------------------------===//

TEST_F(ByteLevelTransformTest, LookupTableCoverage) {
  iree_tokenizer_text_transform_t transform;
  iree_tokenizer_text_transform_initialize_byte_level(
      IREE_TOKENIZER_BYTE_LEVEL_FLAG_DEFAULT, &transform);

  for (int byte = 0; byte < 256; ++byte) {
    std::string input(1, static_cast<char>(byte));
    std::string output;
    ConcatContext context = {&output};
    iree_status_t status = iree_tokenizer_text_transform_encode(
        NULL, &transform, iree_make_string_view(input.data(), input.size()),
        ConcatenateSegments, &context);
    IREE_EXPECT_OK(status) << "Failed for byte " << byte;
    EXPECT_GE(output.size(), 1u) << "Empty output for byte " << byte;
    EXPECT_LE(output.size(), 2u) << "Output too long for byte " << byte;
  }
  iree_tokenizer_text_transform_deinitialize(&transform);
}

//===----------------------------------------------------------------------===//
// ByteLevel Decode Tests
//===----------------------------------------------------------------------===//

TEST_F(ByteLevelTransformTest, DecodeEmpty) {
  auto decoded = Decode("", IREE_TOKENIZER_BYTE_LEVEL_FLAG_DEFAULT);
  EXPECT_EQ(decoded, "");
}

TEST_F(ByteLevelTransformTest, DecodePrintableAscii) {
  auto decoded = Decode("hello", IREE_TOKENIZER_BYTE_LEVEL_FLAG_DEFAULT);
  EXPECT_EQ(decoded, "hello");
}

TEST_F(ByteLevelTransformTest, DecodeSpace) {
  // 0x120 (Ġ) -> space
  auto decoded = Decode("\xC4\xA0", IREE_TOKENIZER_BYTE_LEVEL_FLAG_DEFAULT);
  EXPECT_EQ(decoded, " ");
}

TEST_F(ByteLevelTransformTest, DecodeMixed) {
  // "hello world" encoded is "hello\xC4\xA0world"
  auto decoded =
      Decode("hello\xC4\xA0world", IREE_TOKENIZER_BYTE_LEVEL_FLAG_DEFAULT);
  EXPECT_EQ(decoded, "hello world");
}

TEST_F(ByteLevelTransformTest, DecodeWithPrefixSpaceStrip) {
  // With add_prefix_space=true, decode strips leading space.
  auto decoded =
      Decode("\xC4\xA0hello", IREE_TOKENIZER_BYTE_LEVEL_FLAG_ADD_PREFIX_SPACE);
  EXPECT_EQ(decoded, "hello");
}

TEST_F(ByteLevelTransformTest, DecodeControlChar) {
  // 0x100 (Ā) -> 0x00
  std::string encoded = "\xC4\x80";
  auto decoded = Decode(encoded, IREE_TOKENIZER_BYTE_LEVEL_FLAG_DEFAULT);
  ASSERT_EQ(decoded.size(), 1);
  EXPECT_EQ(decoded[0], '\x00');
}

//===----------------------------------------------------------------------===//
// Round-Trip Tests
//===----------------------------------------------------------------------===//

TEST_F(ByteLevelTransformTest, RoundTripBasic) {
  const char* original = "hello world";
  auto encoded = Encode(original, IREE_TOKENIZER_BYTE_LEVEL_FLAG_DEFAULT);
  auto decoded = Decode(encoded, IREE_TOKENIZER_BYTE_LEVEL_FLAG_DEFAULT);
  EXPECT_EQ(decoded, original);
}

TEST_F(ByteLevelTransformTest, RoundTripWithPrefixSpace) {
  const char* original = "hello world";
  auto encoded =
      Encode(original, IREE_TOKENIZER_BYTE_LEVEL_FLAG_ADD_PREFIX_SPACE);
  auto decoded =
      Decode(encoded, IREE_TOKENIZER_BYTE_LEVEL_FLAG_ADD_PREFIX_SPACE);
  EXPECT_EQ(decoded, original);
}

TEST_F(ByteLevelTransformTest, RoundTripAllBytes) {
  // Test round-trip for all 256 byte values.
  iree_tokenizer_text_transform_t transform;
  iree_tokenizer_text_transform_initialize_byte_level(
      IREE_TOKENIZER_BYTE_LEVEL_FLAG_DEFAULT, &transform);

  for (int byte = 0; byte < 256; ++byte) {
    std::string original(1, static_cast<char>(byte));

    // Encode
    std::string encoded;
    ConcatContext encode_context = {&encoded};
    iree_status_t status = iree_tokenizer_text_transform_encode(
        NULL, &transform,
        iree_make_string_view(original.data(), original.size()),
        ConcatenateSegments, &encode_context);
    IREE_EXPECT_OK(status) << "Encode failed for byte " << byte;

    // Decode
    char decoded[16];
    iree_host_size_t decoded_size = 0;
    status = iree_tokenizer_text_transform_decode(
        &transform, iree_make_string_view(encoded.data(), encoded.size()),
        decoded, sizeof(decoded), &decoded_size);
    IREE_EXPECT_OK(status) << "Decode failed for byte " << byte;
    std::string decoded_str(decoded, decoded_size);

    EXPECT_EQ(decoded_str, original) << "Round-trip failed for byte " << byte;
  }
  iree_tokenizer_text_transform_deinitialize(&transform);
}

TEST_F(ByteLevelTransformTest, RoundTripMixedContent) {
  // Mix of printable ASCII, spaces, tabs, and control chars.
  std::string original = "hello world\ttab\nnewline";
  original += '\x00';  // NUL
  original += '\x7F';  // DEL
  original += '\xFF';  // High byte

  iree_tokenizer_text_transform_t transform;
  iree_tokenizer_text_transform_initialize_byte_level(
      IREE_TOKENIZER_BYTE_LEVEL_FLAG_DEFAULT, &transform);

  // Encode
  std::string encoded;
  ConcatContext encode_context = {&encoded};
  iree_status_t status = iree_tokenizer_text_transform_encode(
      NULL, &transform, iree_make_string_view(original.data(), original.size()),
      ConcatenateSegments, &encode_context);
  IREE_EXPECT_OK(status);

  // Decode
  char decoded[1024];
  iree_host_size_t decoded_size = 0;
  status = iree_tokenizer_text_transform_decode(
      &transform, iree_make_string_view(encoded.data(), encoded.size()),
      decoded, sizeof(decoded), &decoded_size);
  IREE_EXPECT_OK(status);
  std::string decoded_str(decoded, decoded_size);

  iree_tokenizer_text_transform_deinitialize(&transform);
  EXPECT_EQ(decoded_str, original);
}

//===----------------------------------------------------------------------===//
// Edge Case and Stress Tests
//===----------------------------------------------------------------------===//

// Note: With ADD_PREFIX_SPACE, decode always strips the leading space.
// This means inputs that actually start with a space won't round-trip
// correctly. This test documents the known behavior.
TEST_F(ByteLevelTransformTest, LeadingSpaceWithPrefixSpaceKnownLimitation) {
  // Input starts with space, and ADD_PREFIX_SPACE is set.
  // Encode does NOT add a prefix (already has space), but decode DOES strip.
  auto encoded =
      Encode(" hello", IREE_TOKENIZER_BYTE_LEVEL_FLAG_ADD_PREFIX_SPACE);
  auto decoded =
      Decode(encoded, IREE_TOKENIZER_BYTE_LEVEL_FLAG_ADD_PREFIX_SPACE);

  // Known limitation: the leading space is stripped during decode.
  // This is documented behavior matching GPT-2's tokenizer.
  EXPECT_EQ(decoded, "hello");  // NOT " hello"
}

// Test round-trip without ADD_PREFIX_SPACE - should preserve leading space.
TEST_F(ByteLevelTransformTest, LeadingSpaceWithoutPrefixSpace) {
  auto encoded = Encode(" hello", IREE_TOKENIZER_BYTE_LEVEL_FLAG_DEFAULT);
  auto decoded = Decode(encoded, IREE_TOKENIZER_BYTE_LEVEL_FLAG_DEFAULT);
  EXPECT_EQ(decoded, " hello");  // Preserved!
}

// Test with large input that triggers multiple batch emissions.
TEST_F(ByteLevelTransformTest, LargeInputMultipleBatches) {
  // Create input larger than IREE_TOKENIZER_DATA_BATCH_CAPACITY (4KB).
  // Each space becomes 2 bytes (Ġ), so ~2500 spaces > 4KB.
  std::string input(2500, ' ');
  input += "hello";

  iree_tokenizer_text_transform_t transform;
  iree_tokenizer_text_transform_initialize_byte_level(
      IREE_TOKENIZER_BYTE_LEVEL_FLAG_DEFAULT, &transform);

  std::string encoded;
  ConcatContext context = {&encoded};
  iree_status_t status = iree_tokenizer_text_transform_encode(
      NULL, &transform, iree_make_string_view(input.data(), input.size()),
      ConcatenateSegments, &context);
  IREE_EXPECT_OK(status);

  // Should have all the encoded content.
  // Each space -> Ġ (2 bytes), "hello" -> "hello" (5 bytes).
  EXPECT_EQ(encoded.size(), 2500 * 2 + 5);

  // Round-trip should work.
  char decoded[16384];
  iree_host_size_t decoded_size = 0;
  status = iree_tokenizer_text_transform_decode(
      &transform, iree_make_string_view(encoded.data(), encoded.size()),
      decoded, sizeof(decoded), &decoded_size);
  IREE_EXPECT_OK(status);
  EXPECT_EQ(decoded_size, input.size());
  EXPECT_EQ(std::string(decoded, decoded_size), input);

  iree_tokenizer_text_transform_deinitialize(&transform);
}

TEST_F(ByteLevelTransformTest, DecodeBufferTooSmall) {
  iree_tokenizer_text_transform_t transform;
  iree_tokenizer_text_transform_initialize_byte_level(
      IREE_TOKENIZER_BYTE_LEVEL_FLAG_DEFAULT, &transform);

  // "hello" encoded is "hello" (same), requires 5 bytes.
  auto encoded = Encode("hello", IREE_TOKENIZER_BYTE_LEVEL_FLAG_DEFAULT);
  char decoded[3];  // Too small.
  iree_host_size_t decoded_size = 0;
  iree_status_t status = iree_tokenizer_text_transform_decode(
      &transform, iree_make_string_view(encoded.data(), encoded.size()),
      decoded, sizeof(decoded), &decoded_size);

  EXPECT_THAT(Status(std::move(status)),
              StatusIs(StatusCode::kResourceExhausted));
  iree_tokenizer_text_transform_deinitialize(&transform);
}

// Callback that returns an error to test error propagation.
static iree_status_t ByteLevelFailingCallback(
    void* user_data, iree_string_view_list_t segments) {
  (void)user_data;
  (void)segments;
  return iree_make_status(IREE_STATUS_CANCELLED, "intentional failure");
}

TEST_F(ByteLevelTransformTest, CallbackErrorPropagation) {
  iree_tokenizer_text_transform_t transform;
  iree_tokenizer_text_transform_initialize_byte_level(
      IREE_TOKENIZER_BYTE_LEVEL_FLAG_DEFAULT, &transform);

  iree_status_t status = iree_tokenizer_text_transform_encode(
      NULL, &transform, IREE_SVL("hello"), ByteLevelFailingCallback, nullptr);

  EXPECT_THAT(Status(std::move(status)), StatusIs(StatusCode::kCancelled));
  iree_tokenizer_text_transform_deinitialize(&transform);
}

//===----------------------------------------------------------------------===//
// NFC Normalizer Integration Tests
//===----------------------------------------------------------------------===//

TEST_F(ByteLevelTransformTest, NfcNormalizerComposesDecomposedInput) {
  // Test that NFC normalizer composes decomposed Unicode before byte-level
  // encoding. Decomposed "café" (e + combining acute) should produce the same
  // result as pre-composed "café".

  iree_tokenizer_text_transform_t transform;
  iree_tokenizer_text_transform_initialize_byte_level(
      IREE_TOKENIZER_BYTE_LEVEL_FLAG_DEFAULT, &transform);

  // Set up NFC normalizer.
  iree_tokenizer_normalizer_t nfc_normalizer;
  iree_tokenizer_normalizer_initialize_nfc(&nfc_normalizer);

  // Encode pre-composed "café" (c a f é).
  std::string composed_output;
  ConcatContext composed_ctx = {&composed_output};
  const char composed_input[] = "caf\xc3\xa9";  // café with precomposed é
  IREE_ASSERT_OK(iree_tokenizer_text_transform_encode(
      &nfc_normalizer, &transform,
      iree_make_string_view(composed_input, sizeof(composed_input) - 1),
      ConcatenateSegments, &composed_ctx));

  // Encode decomposed "café" (c a f e + combining acute).
  std::string decomposed_output;
  ConcatContext decomposed_ctx = {&decomposed_output};
  const char decomposed_input[] = "cafe\xcc\x81";  // cafe + combining acute
  IREE_ASSERT_OK(iree_tokenizer_text_transform_encode(
      &nfc_normalizer, &transform,
      iree_make_string_view(decomposed_input, sizeof(decomposed_input) - 1),
      ConcatenateSegments, &decomposed_ctx));

  // Both should produce identical byte-level output after NFC normalization.
  EXPECT_EQ(composed_output, decomposed_output)
      << "NFC normalizer should compose decomposed input to match pre-composed";

  iree_tokenizer_normalizer_deinitialize(&nfc_normalizer);
  iree_tokenizer_text_transform_deinitialize(&transform);
}

TEST_F(ByteLevelTransformTest, NfcNormalizerMultipleDecomposed) {
  // Test NFC with multiple decomposed characters: "résumé".

  iree_tokenizer_text_transform_t transform;
  iree_tokenizer_text_transform_initialize_byte_level(
      IREE_TOKENIZER_BYTE_LEVEL_FLAG_DEFAULT, &transform);

  iree_tokenizer_normalizer_t nfc_normalizer;
  iree_tokenizer_normalizer_initialize_nfc(&nfc_normalizer);

  // Pre-composed "résumé".
  std::string composed_output;
  ConcatContext composed_ctx = {&composed_output};
  const char composed_input[] = "r\xc3\xa9sum\xc3\xa9";
  IREE_ASSERT_OK(iree_tokenizer_text_transform_encode(
      &nfc_normalizer, &transform,
      iree_make_string_view(composed_input, sizeof(composed_input) - 1),
      ConcatenateSegments, &composed_ctx));

  // Decomposed "résumé" (e + combining acute for both é).
  std::string decomposed_output;
  ConcatContext decomposed_ctx = {&decomposed_output};
  const char decomposed_input[] = "re\xcc\x81sume\xcc\x81";
  IREE_ASSERT_OK(iree_tokenizer_text_transform_encode(
      &nfc_normalizer, &transform,
      iree_make_string_view(decomposed_input, sizeof(decomposed_input) - 1),
      ConcatenateSegments, &decomposed_ctx));

  EXPECT_EQ(composed_output, decomposed_output);

  iree_tokenizer_normalizer_deinitialize(&nfc_normalizer);
  iree_tokenizer_text_transform_deinitialize(&transform);
}

TEST_F(ByteLevelTransformTest, NfcNormalizerAsciiPassthrough) {
  // ASCII-only input should pass through NFC unchanged.

  iree_tokenizer_text_transform_t transform;
  iree_tokenizer_text_transform_initialize_byte_level(
      IREE_TOKENIZER_BYTE_LEVEL_FLAG_DEFAULT, &transform);

  iree_tokenizer_normalizer_t nfc_normalizer;
  iree_tokenizer_normalizer_initialize_nfc(&nfc_normalizer);

  // Encode with NFC normalizer.
  std::string nfc_output;
  ConcatContext nfc_ctx = {&nfc_output};
  IREE_ASSERT_OK(iree_tokenizer_text_transform_encode(
      &nfc_normalizer, &transform, IREE_SVL("Hello, World!"),
      ConcatenateSegments, &nfc_ctx));

  // Encode without normalizer.
  std::string raw_output;
  ConcatContext raw_ctx = {&raw_output};
  IREE_ASSERT_OK(iree_tokenizer_text_transform_encode(
      NULL, &transform, IREE_SVL("Hello, World!"), ConcatenateSegments,
      &raw_ctx));

  // Should be identical (ASCII is already NFC).
  EXPECT_EQ(nfc_output, raw_output);

  iree_tokenizer_normalizer_deinitialize(&nfc_normalizer);
  iree_tokenizer_text_transform_deinitialize(&transform);
}

}  // namespace
