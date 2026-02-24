// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/decoder/byte_fallback.h"

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
using testing::ProcessBatchedAndFinalize;
using testing::ScopedDecoder;
using testing::ScopedDecoderState;
using testing::TestWithAllBatchSizes;
using testing::TestZeroCapacityOutput;
using testing::ToStringViews;

// U+FFFD replacement character in UTF-8.
const std::string kReplacementChar = "\xEF\xBF\xBD";

//===----------------------------------------------------------------------===//
// Test fixture for ByteFallback decoder tests.
//===----------------------------------------------------------------------===//

class ByteFallbackDecoderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_tokenizer_decoder_t* raw_decoder = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_decoder_byte_fallback_allocate(
        iree_allocator_system(), &raw_decoder));
    decoder_ = ScopedDecoder(raw_decoder);
  }

  iree_tokenizer_decoder_t* decoder() { return decoder_.get(); }

 private:
  ScopedDecoder decoder_;
};

//===----------------------------------------------------------------------===//
// Lifecycle
//===----------------------------------------------------------------------===//

TEST_F(ByteFallbackDecoderTest, CreateAndDestroy) {
  EXPECT_NE(decoder(), nullptr);
}

TEST_F(ByteFallbackDecoderTest, StateSizeIsReasonable) {
  iree_host_size_t state_size = iree_tokenizer_decoder_state_size(decoder());
  EXPECT_GT(state_size, 0u);
  // State should be small: base + pending bytes + counters.
  EXPECT_LE(state_size, 64u);
}

//===----------------------------------------------------------------------===//
// No-ops
//===----------------------------------------------------------------------===//

TEST_F(ByteFallbackDecoderTest, EmptyInput) {
  std::string result =
      ProcessAndFinalize(decoder(), {}, /*expect_pending_after_process=*/false);
  EXPECT_TRUE(result.empty());
}

TEST_F(ByteFallbackDecoderTest, ZeroCapacityOutput) {
  TestZeroCapacityOutput(decoder(), {"Hello", "<0xC3>", "<0xA9>"});
}

//===----------------------------------------------------------------------===//
// Edge Cases
//===----------------------------------------------------------------------===//

TEST_F(ByteFallbackDecoderTest, SingleNonByteToken) {
  TestWithAllBatchSizes(decoder(), {"Hello"}, "Hello",
                        /*expect_pending_after_process=*/false);
}

TEST_F(ByteFallbackDecoderTest, SingleAsciiByteToken) {
  // ASCII byte: <0x41> -> 'A' (single-byte UTF-8).
  TestWithAllBatchSizes(decoder(), {"<0x41>"}, "A",
                        /*expect_pending_after_process=*/false);
}

TEST_F(ByteFallbackDecoderTest, SingleAsciiByteTokenLowercase) {
  // Lowercase hex: <0x41> -> 'A'.
  TestWithAllBatchSizes(decoder(), {"<0x41>"}, "A",
                        /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// Basic Functionality
//===----------------------------------------------------------------------===//

TEST_F(ByteFallbackDecoderTest, PassthroughNonByteTokens) {
  // Non-byte tokens pass through unchanged.
  TestWithAllBatchSizes(decoder(), {"Hello", " ", "World"}, "Hello World",
                        /*expect_pending_after_process=*/false);
}

TEST_F(ByteFallbackDecoderTest, ValidTwoByteSequence) {
  // é (U+00E9) = C3 A9 in UTF-8.
  TestWithAllBatchSizes(decoder(), {"<0xC3>", "<0xA9>"}, "é",
                        /*expect_pending_after_process=*/false);
}

TEST_F(ByteFallbackDecoderTest, ValidThreeByteSequence) {
  // ✓ (U+2713) = E2 9C 93 in UTF-8.
  TestWithAllBatchSizes(decoder(), {"<0xE2>", "<0x9C>", "<0x93>"}, "✓",
                        /*expect_pending_after_process=*/false);
}

TEST_F(ByteFallbackDecoderTest, ValidFourByteSequence) {
  // 😀 (U+1F600) = F0 9F 98 80 in UTF-8.
  TestWithAllBatchSizes(decoder(), {"<0xF0>", "<0x9F>", "<0x98>", "<0x80>"},
                        "😀",
                        /*expect_pending_after_process=*/false);
}

TEST_F(ByteFallbackDecoderTest, MultipleValidSequences) {
  // Two é characters: C3 A9 C3 A9.
  TestWithAllBatchSizes(decoder(), {"<0xC3>", "<0xA9>", "<0xC3>", "<0xA9>"},
                        "éé",
                        /*expect_pending_after_process=*/false);
}

TEST_F(ByteFallbackDecoderTest, MixedAsciiAndMultibyte) {
  // "café" with byte fallback for 'é'.
  TestWithAllBatchSizes(decoder(), {"caf", "<0xC3>", "<0xA9>"}, "café",
                        /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// Invalid Sequence Handling
//===----------------------------------------------------------------------===//

TEST_F(ByteFallbackDecoderTest, InvalidLeadByte) {
  // Continuation byte (0x80-0xBF) used as lead byte -> replacement.
  TestWithAllBatchSizes(decoder(), {"<0x80>"}, kReplacementChar,
                        /*expect_pending_after_process=*/false);
}

TEST_F(ByteFallbackDecoderTest, InvalidLeadByte_BF) {
  // Another continuation byte as lead.
  TestWithAllBatchSizes(decoder(), {"<0xBF>"}, kReplacementChar,
                        /*expect_pending_after_process=*/false);
}

TEST_F(ByteFallbackDecoderTest, IncompleteSequenceAtFinalize) {
  // C3 alone is incomplete (expects continuation) -> replacement on finalize.
  ScopedDecoderState state(decoder());

  std::vector<std::string> tokens = {"<0xC3>"};
  auto views = ToStringViews(tokens);
  iree_tokenizer_string_list_t token_list = {views.size(), views.data()};

  char output[64];
  iree_host_size_t strings_consumed = 0;
  iree_host_size_t bytes_written = 0;
  IREE_ASSERT_OK(iree_tokenizer_decoder_state_process(
      state.get(), token_list,
      iree_make_mutable_string_view(output, sizeof(output)), &strings_consumed,
      &bytes_written));

  EXPECT_EQ(strings_consumed, 1u);
  EXPECT_EQ(bytes_written, 0u);  // Nothing emitted yet - waiting for more.
  EXPECT_TRUE(iree_tokenizer_decoder_state_has_pending(state.get()));

  // Finalize should emit replacement for the incomplete byte.
  iree_host_size_t finalize_written = 0;
  IREE_ASSERT_OK(iree_tokenizer_decoder_state_finalize(
      state.get(), iree_make_mutable_string_view(output, sizeof(output)),
      &finalize_written));

  EXPECT_EQ(finalize_written, 3u);  // U+FFFD is 3 bytes.
  EXPECT_EQ(std::string(output, finalize_written), kReplacementChar);
}

TEST_F(ByteFallbackDecoderTest, InvalidContinuationByte) {
  // C3 followed by non-continuation (0x41 = 'A') -> replacement + new sequence.
  // C3 expects continuation but gets ASCII, so C3 is invalid (replacement).
  // Then 0x41 is processed as a new valid ASCII byte.
  TestWithAllBatchSizes(decoder(), {"<0xC3>", "<0x41>"}, kReplacementChar + "A",
                        /*expect_pending_after_process=*/false);
}

TEST_F(ByteFallbackDecoderTest, TooManyBytes) {
  // 4-byte sequence but only 3 bytes provided, then different lead byte.
  // F0 expects 3 more bytes but gets F0 (another lead) after 2 continuations.
  // F0 9F 98 -> incomplete, then F0 starts new sequence.
  ScopedDecoderState state(decoder());

  std::vector<std::string> tokens = {"<0xF0>", "<0x9F>", "<0x98>", "<0xF0>"};
  auto views = ToStringViews(tokens);
  iree_tokenizer_string_list_t token_list = {views.size(), views.data()};

  char output[64];
  iree_host_size_t strings_consumed = 0;
  iree_host_size_t bytes_written = 0;
  IREE_ASSERT_OK(iree_tokenizer_decoder_state_process(
      state.get(), token_list,
      iree_make_mutable_string_view(output, sizeof(output)), &strings_consumed,
      &bytes_written));

  // F0 9F 98 is incomplete, then F0 starts new - so 3 replacements + pending.
  EXPECT_EQ(strings_consumed, 4u);
  // 3 replacement chars (9 bytes) for the invalid F0 9F 98.
  EXPECT_EQ(bytes_written, 9u);
  EXPECT_TRUE(iree_tokenizer_decoder_state_has_pending(state.get()));
}

TEST_F(ByteFallbackDecoderTest, OverlongTwoByte) {
  // Overlong encoding: 'A' (U+0041) encoded as 2 bytes (C1 81).
  // This is invalid - should produce 2 replacement characters.
  TestWithAllBatchSizes(decoder(), {"<0xC1>", "<0x81>"},
                        kReplacementChar + kReplacementChar,
                        /*expect_pending_after_process=*/false);
}

TEST_F(ByteFallbackDecoderTest, SurrogateCodepoint) {
  // ED A0 80 would encode U+D800 (surrogate), which is invalid in UTF-8.
  // Should produce 3 replacement characters.
  TestWithAllBatchSizes(decoder(), {"<0xED>", "<0xA0>", "<0x80>"},
                        kReplacementChar + kReplacementChar + kReplacementChar,
                        /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// Streaming Behavior
//===----------------------------------------------------------------------===//

TEST_F(ByteFallbackDecoderTest, BatchedProcessing) {
  // Multiple valid sequences processed in batches.
  std::vector<std::string> tokens = {"a",      "<0xC3>", "<0xA9>", "b",
                                     "<0xC3>", "<0xA9>", "c"};
  TestWithAllBatchSizes(decoder(), tokens, "aébéc",
                        /*expect_pending_after_process=*/false);
}

TEST_F(ByteFallbackDecoderTest, HasPendingAfterPartialSequence) {
  ScopedDecoderState state(decoder());

  // Process first byte of 2-byte sequence.
  std::vector<std::string> tokens = {"<0xC3>"};
  auto views = ToStringViews(tokens);
  iree_tokenizer_string_list_t token_list = {views.size(), views.data()};

  char output[64];
  iree_host_size_t strings_consumed = 0;
  iree_host_size_t bytes_written = 0;
  IREE_ASSERT_OK(iree_tokenizer_decoder_state_process(
      state.get(), token_list,
      iree_make_mutable_string_view(output, sizeof(output)), &strings_consumed,
      &bytes_written));

  EXPECT_TRUE(iree_tokenizer_decoder_state_has_pending(state.get()));

  // Process continuation byte - sequence complete.
  tokens = {"<0xA9>"};
  views = ToStringViews(tokens);
  token_list = {views.size(), views.data()};

  IREE_ASSERT_OK(iree_tokenizer_decoder_state_process(
      state.get(), token_list,
      iree_make_mutable_string_view(output, sizeof(output)), &strings_consumed,
      &bytes_written));

  EXPECT_FALSE(iree_tokenizer_decoder_state_has_pending(state.get()));
  EXPECT_EQ(bytes_written, 2u);  // é is 2 bytes.
  EXPECT_EQ(std::string(output, bytes_written), "é");
}

//===----------------------------------------------------------------------===//
// Buffer-Full Edge Cases
//===----------------------------------------------------------------------===//

TEST_F(ByteFallbackDecoderTest, BufferFullMidValidSequence) {
  // Buffer fills while writing valid UTF-8 sequence output.
  // With 4-byte buffer, first token is consumed (pending) but second can't emit
  // until we have enough room.
  ScopedDecoderState state(decoder());

  std::vector<std::string> tokens = {"<0xC3>", "<0xA9>"};
  auto views = ToStringViews(tokens);
  iree_tokenizer_string_list_t token_list = {views.size(), views.data()};

  // First call: accumulate first byte, second byte can't emit yet.
  char small[4];
  iree_host_size_t strings_consumed = 0;
  iree_host_size_t bytes_written = 0;
  IREE_ASSERT_OK(iree_tokenizer_decoder_state_process(
      state.get(), token_list,
      iree_make_mutable_string_view(small, sizeof(small)), &strings_consumed,
      &bytes_written));

  // Both tokens consumed, sequence emitted (2 bytes fit in 4-byte buffer).
  EXPECT_EQ(strings_consumed, 2u);
  EXPECT_EQ(bytes_written, 2u);
  EXPECT_EQ(std::string(small, bytes_written), "é");
  EXPECT_FALSE(iree_tokenizer_decoder_state_has_pending(state.get()));
}

TEST_F(ByteFallbackDecoderTest, BufferFullDuringReplacementOutput) {
  // Buffer fills while writing replacement characters for invalid sequence.
  // Each replacement is 3 bytes (U+FFFD = EF BF BD).
  ScopedDecoderState state(decoder());

  // Invalid 3-byte overlong sequence: C1 81 41 (C1 81 is overlong 'A').
  // Should produce 2 replacement chars (6 bytes) + 'A' (1 byte).
  std::vector<std::string> tokens = {"<0xC1>", "<0x81>", "<0x41>"};

  // Buffer only fits 1 replacement (3 bytes) but we need 2 (6 bytes).
  char small_buffer[4];
  iree_host_size_t total_consumed = 0;
  iree_host_size_t bytes_written = 0;
  std::string result;

  // Process tokens incrementally, tracking how many we've consumed.
  while (total_consumed < tokens.size()) {
    // Build views for remaining tokens - keep vector alive in this scope.
    std::vector<std::string> remaining_tokens(tokens.begin() + total_consumed,
                                              tokens.end());
    auto views = ToStringViews(remaining_tokens);
    iree_tokenizer_string_list_t token_list = {views.size(), views.data()};

    iree_host_size_t consumed = 0;
    IREE_ASSERT_OK(iree_tokenizer_decoder_state_process(
        state.get(), token_list,
        iree_make_mutable_string_view(small_buffer, sizeof(small_buffer)),
        &consumed, &bytes_written));
    result.append(small_buffer, bytes_written);
    total_consumed += consumed;
    ASSERT_TRUE(bytes_written > 0 || consumed > 0) << "No progress";
  }

  // Final result: 2 replacements + 'A'.
  EXPECT_EQ(result, kReplacementChar + kReplacementChar + "A");
}

TEST_F(ByteFallbackDecoderTest, FinalizeIncompleteSequence) {
  // Finalize with incomplete pending sequence emits replacement char.
  ScopedDecoderState state(decoder());

  // Start a 2-byte sequence but don't complete it.
  std::vector<std::string> tokens = {"<0xC3>"};
  auto views = ToStringViews(tokens);
  iree_tokenizer_string_list_t token_list = {views.size(), views.data()};

  char output[64];
  iree_host_size_t strings_consumed = 0;
  iree_host_size_t bytes_written = 0;
  IREE_ASSERT_OK(iree_tokenizer_decoder_state_process(
      state.get(), token_list,
      iree_make_mutable_string_view(output, sizeof(output)), &strings_consumed,
      &bytes_written));
  EXPECT_TRUE(iree_tokenizer_decoder_state_has_pending(state.get()));

  // Finalize emits replacement character for incomplete sequence.
  char finalize_buffer[4];
  iree_host_size_t finalize_written = 0;
  IREE_ASSERT_OK(iree_tokenizer_decoder_state_finalize(
      state.get(),
      iree_make_mutable_string_view(finalize_buffer, sizeof(finalize_buffer)),
      &finalize_written));
  EXPECT_EQ(finalize_written, 3u);
  EXPECT_EQ(std::string(finalize_buffer, 3), kReplacementChar);
  EXPECT_FALSE(iree_tokenizer_decoder_state_has_pending(state.get()));
}

TEST_F(ByteFallbackDecoderTest, SmallOutputBuffer) {
  // Stress test streaming with small buffer (4 bytes = max UTF-8 sequence).
  // UTF-8 characters must be emitted atomically, so buffer must be at least
  // as large as the maximum character size to guarantee progress.
  ScopedDecoderState state(decoder());

  // "café" with byte fallback for é.
  std::vector<std::string> all_tokens = {"caf", "<0xC3>", "<0xA9>"};
  std::string result;

  for (const auto& token : all_tokens) {
    std::vector<std::string> token_vec = {token};
    auto views = ToStringViews(token_vec);
    bool consumed = false;

    while (!consumed) {
      char small_buf[4];  // Min viable: max UTF-8 sequence length.
      iree_host_size_t strings_consumed = 0;
      iree_host_size_t bytes_written = 0;

      IREE_ASSERT_OK(iree_tokenizer_decoder_state_process(
          state.get(), iree_tokenizer_make_string_list(views.data(), 1),
          iree_make_mutable_string_view(small_buf, sizeof(small_buf)),
          &strings_consumed, &bytes_written));

      if (bytes_written > 0) {
        result.append(small_buf, bytes_written);
      }
      if (strings_consumed > 0) {
        consumed = true;
      }
      // Handle pending bytes that produce no output yet.
      if (bytes_written == 0 && strings_consumed == 0) {
        // Pending - try again with same token.
        continue;
      }
      ASSERT_TRUE(bytes_written > 0 || strings_consumed > 0) << "No progress";
    }
  }

  EXPECT_EQ(result, "café");
}

//===----------------------------------------------------------------------===//
// Mixed Tokens
//===----------------------------------------------------------------------===//

TEST_F(ByteFallbackDecoderTest, MixedByteAndNonByte) {
  // Mix of regular tokens and byte tokens.
  std::vector<std::string> tokens = {"Hello", "<0xC3>", "<0xA9>", " ", "World"};
  TestWithAllBatchSizes(decoder(), tokens, "Helloé World",
                        /*expect_pending_after_process=*/false);
}

TEST_F(ByteFallbackDecoderTest, NonByteTokenInterruptsSequence) {
  // Non-byte token between byte tokens interrupts the sequence.
  // C3 is incomplete when "x" arrives, so C3 -> replacement, then "x".
  std::vector<std::string> tokens = {"<0xC3>", "x", "<0xA9>"};
  // C3 -> replacement, "x" passthrough, A9 (continuation alone) -> replacement.
  TestWithAllBatchSizes(decoder(), tokens,
                        kReplacementChar + "x" + kReplacementChar,
                        /*expect_pending_after_process=*/false);
}

TEST_F(ByteFallbackDecoderTest, InvalidPatternNotMatched) {
  // Tokens that look like byte patterns but aren't valid.
  TestWithAllBatchSizes(decoder(), {"<0x>"}, "<0x>",  // Too short.
                        /*expect_pending_after_process=*/false);
  TestWithAllBatchSizes(decoder(), {"<0xGG>"}, "<0xGG>",  // Invalid hex.
                        /*expect_pending_after_process=*/false);
  TestWithAllBatchSizes(decoder(), {"<0x123>"}, "<0x123>",  // Too long.
                        /*expect_pending_after_process=*/false);
  TestWithAllBatchSizes(decoder(), {"0xC3"}, "0xC3",  // Missing brackets.
                        /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// HuggingFace Compatibility
//===----------------------------------------------------------------------===//

TEST_F(ByteFallbackDecoderTest, HF_CafeWithByteFallback) {
  // Real-world example: "café" encoded with byte fallback for 'é'.
  std::vector<std::string> tokens = {"caf", "<0xC3>", "<0xA9>"};
  TestWithAllBatchSizes(decoder(), tokens, "café",
                        /*expect_pending_after_process=*/false);
}

TEST_F(ByteFallbackDecoderTest, HF_EmojiWithByteFallback) {
  // Emoji: 🎉 (U+1F389) = F0 9F 8E 89.
  std::vector<std::string> tokens = {"Party", "<0xF0>", "<0x9F>", "<0x8E>",
                                     "<0x89>"};
  TestWithAllBatchSizes(decoder(), tokens, "Party🎉",
                        /*expect_pending_after_process=*/false);
}

TEST_F(ByteFallbackDecoderTest, HF_ChineseWithByteFallback) {
  // 中 (U+4E2D) = E4 B8 AD in UTF-8.
  std::vector<std::string> tokens = {"<0xE4>", "<0xB8>", "<0xAD>"};
  TestWithAllBatchSizes(decoder(), tokens, "中",
                        /*expect_pending_after_process=*/false);
}

}  // namespace
}  // namespace iree::tokenizer
