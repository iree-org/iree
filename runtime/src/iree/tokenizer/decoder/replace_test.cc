// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/decoder/replace.h"

#include <cstring>
#include <string>
#include <vector>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/tokenizer/decoder/decoder_test_util.h"

namespace iree::tokenizer {
namespace {

using testing::ProcessAndFinalize;
using testing::ScopedDecoder;
using testing::TestWithAllBatchSizes;

//===----------------------------------------------------------------------===//
// Test fixtures
//===----------------------------------------------------------------------===//

class ReplaceDecoderTest : public ::testing::Test {
 protected:
  ScopedDecoder CreateReplace(iree_string_view_t pattern,
                              iree_string_view_t content) {
    iree_tokenizer_decoder_t* raw_decoder = nullptr;
    IREE_EXPECT_OK(iree_tokenizer_decoder_replace_allocate(
        pattern, content, iree_allocator_system(), &raw_decoder));
    return ScopedDecoder(raw_decoder);
  }
};

//===----------------------------------------------------------------------===//
// Lifecycle tests
//===----------------------------------------------------------------------===//

TEST_F(ReplaceDecoderTest, CreateAndDestroy) {
  auto decoder = CreateReplace(IREE_SV("a"), IREE_SV("b"));
  EXPECT_NE(decoder.get(), nullptr);
}

TEST_F(ReplaceDecoderTest, RejectsEmptyPattern) {
  iree_tokenizer_decoder_t* decoder = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_tokenizer_decoder_replace_allocate(
                            iree_string_view_empty(), IREE_SV("x"),
                            iree_allocator_system(), &decoder));
}

TEST_F(ReplaceDecoderTest, RejectsExpandingPattern) {
  // Pattern "a" (1 byte) -> content "abc" (3 bytes) would expand.
  iree_tokenizer_decoder_t* decoder = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_tokenizer_decoder_replace_allocate(
          IREE_SV("a"), IREE_SV("abc"), iree_allocator_system(), &decoder));
}

TEST_F(ReplaceDecoderTest, AllowsSameLengthPattern) {
  // Same length is fine.
  auto decoder = CreateReplace(IREE_SV("abc"), IREE_SV("xyz"));
  EXPECT_NE(decoder.get(), nullptr);
}

TEST_F(ReplaceDecoderTest, AllowsShrinkingPattern) {
  // Shrinking is the common case.
  auto decoder = CreateReplace(IREE_SV("abc"), IREE_SV("x"));
  EXPECT_NE(decoder.get(), nullptr);
}

//===----------------------------------------------------------------------===//
// Basic replacement tests
//===----------------------------------------------------------------------===//

TEST_F(ReplaceDecoderTest, NoMatchPassthrough) {
  auto decoder = CreateReplace(IREE_SV("X"), IREE_SV("Y"));
  ASSERT_NE(decoder.get(), nullptr);

  TestWithAllBatchSizes(decoder.get(), {"Hello", "World"}, "HelloWorld",
                        /*expect_pending_after_process=*/false);
}

TEST_F(ReplaceDecoderTest, SingleMatch) {
  auto decoder = CreateReplace(IREE_SV("X"), IREE_SV("Y"));
  ASSERT_NE(decoder.get(), nullptr);

  TestWithAllBatchSizes(decoder.get(), {"HeXlo"}, "HeYlo",
                        /*expect_pending_after_process=*/false);
}

TEST_F(ReplaceDecoderTest, MultipleMatchesSingleToken) {
  auto decoder = CreateReplace(IREE_SV("X"), IREE_SV("Y"));
  ASSERT_NE(decoder.get(), nullptr);

  TestWithAllBatchSizes(decoder.get(), {"XAXBXCX"}, "YAYBYCY",
                        /*expect_pending_after_process=*/false);
}

TEST_F(ReplaceDecoderTest, MatchAcrossTokens) {
  auto decoder = CreateReplace(IREE_SV("X"), IREE_SV("Y"));
  ASSERT_NE(decoder.get(), nullptr);

  TestWithAllBatchSizes(decoder.get(), {"aXb", "cXd"}, "aYbcYd",
                        /*expect_pending_after_process=*/false);
}

TEST_F(ReplaceDecoderTest, ShrinkingReplacement) {
  // 3 bytes -> 1 byte (like metaspace).
  auto decoder = CreateReplace(IREE_SV("abc"), IREE_SV("X"));
  ASSERT_NE(decoder.get(), nullptr);

  TestWithAllBatchSizes(decoder.get(), {"1abc2abc3"}, "1X2X3",
                        /*expect_pending_after_process=*/false);
}

TEST_F(ReplaceDecoderTest, ReplaceWithEmpty) {
  // Delete pattern entirely.
  auto decoder = CreateReplace(IREE_SV("X"), iree_string_view_empty());
  ASSERT_NE(decoder.get(), nullptr);

  TestWithAllBatchSizes(decoder.get(), {"aXbXc"}, "abc",
                        /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// HuggingFace pattern: metaspace to space
//===----------------------------------------------------------------------===//

TEST_F(ReplaceDecoderTest, MetaspaceToSpace) {
  // U+2581 (LOWER ONE EIGHTH BLOCK) in UTF-8: E2 96 81
  static const char kMetaspace[] = "\xE2\x96\x81";
  auto decoder =
      CreateReplace(iree_make_cstring_view(kMetaspace), IREE_SV(" "));
  ASSERT_NE(decoder.get(), nullptr);

  std::string input = std::string(kMetaspace) + "Hello" + kMetaspace + "World";
  TestWithAllBatchSizes(decoder.get(), {input}, " Hello World",
                        /*expect_pending_after_process=*/false);
}

TEST_F(ReplaceDecoderTest, MetaspaceToSpaceMultipleTokens) {
  static const char kMetaspace[] = "\xE2\x96\x81";
  auto decoder =
      CreateReplace(iree_make_cstring_view(kMetaspace), IREE_SV(" "));
  ASSERT_NE(decoder.get(), nullptr);

  std::string t1 = std::string(kMetaspace) + "Hello";
  std::string t2 = std::string(kMetaspace) + "World";
  TestWithAllBatchSizes(decoder.get(), {t1, t2}, " Hello World",
                        /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// Edge cases
//===----------------------------------------------------------------------===//

TEST_F(ReplaceDecoderTest, EmptyInput) {
  auto decoder = CreateReplace(IREE_SV("X"), IREE_SV("Y"));
  ASSERT_NE(decoder.get(), nullptr);

  TestWithAllBatchSizes(decoder.get(), {}, "",
                        /*expect_pending_after_process=*/false);
}

TEST_F(ReplaceDecoderTest, EmptyTokens) {
  auto decoder = CreateReplace(IREE_SV("X"), IREE_SV("Y"));
  ASSERT_NE(decoder.get(), nullptr);

  TestWithAllBatchSizes(decoder.get(), {"", "a", "", "b", ""}, "ab",
                        /*expect_pending_after_process=*/false);
}

TEST_F(ReplaceDecoderTest, PatternAtStart) {
  auto decoder = CreateReplace(IREE_SV("XX"), IREE_SV("Y"));
  ASSERT_NE(decoder.get(), nullptr);

  TestWithAllBatchSizes(decoder.get(), {"XXhello"}, "Yhello",
                        /*expect_pending_after_process=*/false);
}

TEST_F(ReplaceDecoderTest, PatternAtEnd) {
  auto decoder = CreateReplace(IREE_SV("XX"), IREE_SV("Y"));
  ASSERT_NE(decoder.get(), nullptr);

  TestWithAllBatchSizes(decoder.get(), {"helloXX"}, "helloY",
                        /*expect_pending_after_process=*/false);
}

TEST_F(ReplaceDecoderTest, ConsecutivePatterns) {
  auto decoder = CreateReplace(IREE_SV("XX"), IREE_SV("Y"));
  ASSERT_NE(decoder.get(), nullptr);

  TestWithAllBatchSizes(decoder.get(), {"XXXXXX"}, "YYY",
                        /*expect_pending_after_process=*/false);
}

TEST_F(ReplaceDecoderTest, OverlappingPatternsNoMatch) {
  // Pattern "aa" in "aaa" should only match once (first two chars).
  auto decoder = CreateReplace(IREE_SV("aa"), IREE_SV("X"));
  ASSERT_NE(decoder.get(), nullptr);

  TestWithAllBatchSizes(decoder.get(), {"aaa"}, "Xa",
                        /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// Buffer-full edge cases
//===----------------------------------------------------------------------===//

TEST_F(ReplaceDecoderTest, LimitedOutputCapacity) {
  auto decoder = CreateReplace(IREE_SV("XX"), IREE_SV("Y"));
  ASSERT_NE(decoder.get(), nullptr);

  testing::TestLimitedOutputCapacity(decoder.get(), {"aXXbXXc"}, "aYbYc");
}

TEST_F(ReplaceDecoderTest, BufferFullMidTokenWithResume) {
  // Test mid-token buffer overflow with resume: when buffer fills mid-token,
  // the decoder must track position and resume correctly on the next call.
  auto decoder = CreateReplace(IREE_SV("XX"), IREE_SV("Y"));
  ASSERT_NE(decoder.get(), nullptr);

  testing::ScopedDecoderState state(decoder.get());

  // Token with multiple patterns. Buffer size 5 means we can only fit part
  // of the output per call (output is "aYbYc" = 5 chars, forces resume).
  // Keep string alive - views point into this vector.
  std::vector<std::string> token_vec = {"aXXbXXc"};  // Output: "aYbYc"
  auto views = testing::ToStringViews(token_vec);

  // Use a buffer that forces multiple calls.
  char small_buffer[4];
  std::string result;
  bool token_consumed = false;

  // Process repeatedly until token consumed.
  while (!token_consumed) {
    iree_host_size_t strings_consumed = 0;
    iree_host_size_t bytes_written = 0;

    IREE_ASSERT_OK(iree_tokenizer_decoder_state_process(
        state.get(), iree_tokenizer_make_string_list(views.data(), 1),
        iree_make_mutable_string_view(small_buffer, sizeof(small_buffer)),
        &strings_consumed, &bytes_written));

    result.append(small_buffer, bytes_written);
    if (strings_consumed > 0) {
      token_consumed = true;
    }

    // Safety: break if no progress at all.
    if (strings_consumed == 0 && bytes_written == 0) {
      FAIL() << "No progress made - likely infinite loop";
    }
  }

  // Finalize.
  iree_host_size_t finalize_written = 0;
  IREE_ASSERT_OK(iree_tokenizer_decoder_state_finalize(
      state.get(),
      iree_make_mutable_string_view(small_buffer, sizeof(small_buffer)),
      &finalize_written));
  result.append(small_buffer, finalize_written);

  // Both XX should be replaced with Y, no duplication.
  EXPECT_EQ(result, "aYbYc");
}

TEST_F(ReplaceDecoderTest, BufferFullAfterReplacement) {
  // Test buffer filling right after writing a replacement.
  auto decoder = CreateReplace(IREE_SV("XXX"), IREE_SV("Y"));
  ASSERT_NE(decoder.get(), nullptr);

  // Input: "12XXX34567890" -> "12Y34567890"
  // With small buffer, replacement should work correctly.
  testing::TestLimitedOutputCapacity(decoder.get(), {"12XXX34567890"},
                                     "12Y34567890");
}

TEST_F(ReplaceDecoderTest, MultipleTokensWithLimitedCapacity) {
  auto decoder = CreateReplace(IREE_SV("XX"), IREE_SV("Y"));
  ASSERT_NE(decoder.get(), nullptr);

  testing::TestLimitedOutputCapacity(decoder.get(), {"aXX", "bXX", "cXX", "d"},
                                     "aYbYcYd");
}

}  // namespace
}  // namespace iree::tokenizer
