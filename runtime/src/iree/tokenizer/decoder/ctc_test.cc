// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/decoder/ctc.h"

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
using testing::TestLimitedOutputCapacity;
using testing::TestWithAllBatchSizes;
using testing::TestZeroCapacityOutput;
using testing::ToStringViews;

// Helper to allocate CTC decoder with default settings (for tests only).
ScopedDecoder AllocateDefaultCTC() {
  iree_tokenizer_decoder_t* raw_decoder = nullptr;
  IREE_CHECK_OK(iree_tokenizer_decoder_ctc_allocate(
      iree_make_cstring_view("<pad>"), iree_make_cstring_view("|"),
      /*cleanup=*/true, iree_allocator_system(), &raw_decoder));
  return ScopedDecoder(raw_decoder);
}

// Splits a space-separated string into tokens.
std::vector<std::string> SplitTokens(const char* space_separated) {
  std::vector<std::string> tokens;
  std::string current;
  for (const char* p = space_separated; *p; ++p) {
    if (*p == ' ') {
      if (!current.empty()) {
        tokens.push_back(current);
        current.clear();
      }
    } else {
      current += *p;
    }
  }
  if (!current.empty()) {
    tokens.push_back(current);
  }
  return tokens;
}

//===----------------------------------------------------------------------===//
// Test fixture for CTC decoder tests.
//===----------------------------------------------------------------------===//

class CTCDecoderTest : public ::testing::Test {
 protected:
  void SetUp() override { decoder_ = AllocateDefaultCTC(); }
  iree_tokenizer_decoder_t* decoder() { return decoder_.get(); }

 private:
  ScopedDecoder decoder_;
};

//===----------------------------------------------------------------------===//
// Lifecycle
//===----------------------------------------------------------------------===//

TEST_F(CTCDecoderTest, CreateAndDestroy) { EXPECT_NE(decoder(), nullptr); }

TEST_F(CTCDecoderTest, StateSizeIsReasonable) {
  iree_host_size_t state_size = iree_tokenizer_decoder_state_size(decoder());
  EXPECT_GT(state_size, 0u);
  // State: 5 * 64-byte buffers + lengths + flags ≈ 400 bytes.
  EXPECT_LE(state_size, 512u);
}

//===----------------------------------------------------------------------===//
// No-ops and Edge Cases
//===----------------------------------------------------------------------===//

TEST_F(CTCDecoderTest, EmptyInput) {
  std::string result =
      ProcessAndFinalize(decoder(), {}, /*expect_pending_after_process=*/false);
  EXPECT_TRUE(result.empty());
}

TEST_F(CTCDecoderTest, ZeroCapacityOutput) {
  // CTC-specific: With zero output capacity, CTC can still accept the first
  // token into prev_token (internal buffering). This differs from decoders
  // that immediately produce output.
  ScopedDecoderState state(decoder());
  std::vector<std::string> tokens = {"h", "e", "l", "l", "o"};
  auto views = ToStringViews(tokens);
  iree_tokenizer_string_list_t token_list = {
      .count = views.size(),
      .values = views.data(),
  };
  iree_host_size_t strings_consumed = 0;
  iree_host_size_t bytes_written = 0;
  IREE_ASSERT_OK(iree_tokenizer_decoder_state_process(
      state.get(), token_list, iree_make_mutable_string_view(nullptr, 0),
      &strings_consumed, &bytes_written));
  // CTC buffers first token in prev_token, so consumes 1.
  // Second token triggers emit, but buffer full → stops processing.
  EXPECT_GE(strings_consumed, 1u);
  EXPECT_EQ(bytes_written, 0u);
}

TEST_F(CTCDecoderTest, SingleToken) {
  // Single token requires finalize to emit (1-token delay).
  TestWithAllBatchSizes(decoder(), {"h"}, "h",
                        /*expect_pending_after_process=*/true);
}

TEST_F(CTCDecoderTest, OnlyPads) {
  TestWithAllBatchSizes(decoder(), {"<pad>", "<pad>", "<pad>"}, "",
                        /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// HuggingFace Parity
//===----------------------------------------------------------------------===//

TEST_F(CTCDecoderTest, HF_HandmadeSample) {
  // From: tokenizers/src/decoders/ctc.rs handmade_sample test.
  // Input: "<pad> <pad> h e e l l <pad> l o o o <pad>"
  // Expected output tokens after decode_chain: ["h", "e", "l", "l", "o"]
  // Joined: "hello"
  auto tokens = SplitTokens("<pad> <pad> h e e l l <pad> l o o o <pad>");
  TestWithAllBatchSizes(decoder(), tokens, "hello",
                        /*expect_pending_after_process=*/false);
}

TEST_F(CTCDecoderTest, HF_HandmadeWithDelimiterSample) {
  // From: tokenizers/src/decoders/ctc.rs handmade_with_delimiter_sample test.
  // Input includes word delimiter "|" which becomes space.
  auto tokens = SplitTokens(
      "<pad> <pad> h e e l l <pad> l o o o <pad> <pad> | <pad> w o o o r <pad> "
      "<pad> l l d <pad> <pad> <pad> <pad>");
  TestWithAllBatchSizes(decoder(), tokens, "hello world",
                        /*expect_pending_after_process=*/false);
}

TEST_F(CTCDecoderTest, HF_LibriSpeechSample) {
  // From: tokenizers/src/decoders/ctc.rs librispeech_sample test.
  // Real speech recognition output.
  auto tokens = SplitTokens(
      "<pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> "
      "<pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> "
      "<pad> <pad> <pad> <pad> <pad> <pad> A | | <pad> M <pad> <pad> <pad> "
      "<pad> A <pad> <pad> N <pad> <pad> <pad> | | | <pad> <pad> <pad> <pad> S "
      "<pad> <pad> <pad> A I <pad> D D | | T T <pad> O <pad> | | T H E E | | | "
      "<pad> U U <pad> N N <pad> I <pad> <pad> V <pad> <pad> <pad> E R R <pad> "
      "<pad> <pad> S E E | | <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> "
      "<pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> "
      "<pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> "
      "<pad> <pad> <pad> <pad> <pad> <pad> S S <pad> <pad> <pad> <pad> I <pad> "
      "R R <pad> <pad> | | | <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> "
      "<pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> I "
      "<pad> <pad> <pad> | <pad> <pad> <pad> E X <pad> <pad> <pad> <pad> <pad> "
      "<pad> <pad> <pad> <pad> <pad> I <pad> S <pad> <pad> T <pad> <pad> | | "
      "<pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> "
      "<pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> "
      "<pad> <pad> <pad>");
  TestWithAllBatchSizes(decoder(), tokens,
                        "A MAN SAID TO THE UNIVERSE SIR I EXIST ",
                        /*expect_pending_after_process=*/false);
}

TEST_F(CTCDecoderTest, HF_AnotherLibriSpeechSample) {
  // From: tokenizers/src/decoders/ctc.rs another_librispeech_sample test.
  auto tokens = SplitTokens(
      "<pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> "
      "<pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> H <pad> I <pad> S S | | "
      "<pad> <pad> <pad> I N <pad> <pad> S <pad> T T <pad> <pad> A N C C T "
      "<pad> | | | | | <pad> <pad> <pad> <pad> P <pad> <pad> <pad> <pad> A "
      "<pad> <pad> N N N <pad> <pad> I <pad> C <pad> <pad> | | <pad> W <pad> "
      "<pad> A S <pad> | | <pad> <pad> <pad> F <pad> <pad> O L <pad> <pad> L L "
      "O O W E E D | | <pad> B <pad> <pad> <pad> Y <pad> | | | A | | <pad> S S "
      "S <pad> M M <pad> <pad> <pad> A L L <pad> <pad> <pad> <pad> L <pad> | | "
      "| <pad> <pad> <pad> <pad> S H H <pad> <pad> <pad> <pad> A R R <pad> "
      "<pad> "
      "P <pad> <pad> | <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> B "
      "<pad> <pad> L L <pad> <pad> <pad> <pad> <pad> O W W <pad> <pad> | | | "
      "<pad> <pad> <pad> <pad> <pad> <pad> <pad> H <pad> <pad> <pad> <pad> "
      "<pad> "
      "<pad> <pad> I G H H | | <pad> <pad> O N <pad> | | H <pad> I S S | | "
      "<pad> <pad> C H H <pad> <pad> <pad> E <pad> S S <pad> T T <pad> <pad> | "
      "| | <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> "
      "<pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> "
      "<pad> <pad> <pad> <pad> <pad>");
  TestWithAllBatchSizes(decoder(), tokens,
                        "HIS INSTANCT PANIC WAS FOLLOWED BY A SMALL SHARP BLOW "
                        "HIGH ON HIS CHEST ",
                        /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// Dedup Behavior
//===----------------------------------------------------------------------===//

TEST_F(CTCDecoderTest, DedupConsecutive) {
  // Consecutive duplicates are removed.
  TestWithAllBatchSizes(decoder(), {"h", "h", "h", "e", "e"}, "he",
                        /*expect_pending_after_process=*/true);
}

TEST_F(CTCDecoderTest, DedupHelloBecomesHelo) {
  // HuggingFace parity: ["h","e","l","l","o"] → "helo" (consecutive l's
  // deduped)
  TestWithAllBatchSizes(decoder(), {"h", "e", "l", "l", "o"}, "helo",
                        /*expect_pending_after_process=*/true);
}

TEST_F(CTCDecoderTest, PadBreaksDedup) {
  // Pad between same letters prevents dedup - produces both.
  TestWithAllBatchSizes(decoder(), {"l", "<pad>", "l"}, "ll",
                        /*expect_pending_after_process=*/true);
}

TEST_F(CTCDecoderTest, ConsecutivePads) {
  // Multiple consecutive pads are removed (deduped then filtered).
  // Last token is pad which clears prev_token, so has_pending=false.
  TestWithAllBatchSizes(decoder(), {"<pad>", "<pad>", "h", "<pad>", "<pad>"},
                        "h",
                        /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// Streaming Dedup Across Batch Boundaries
//===----------------------------------------------------------------------===//

TEST_F(CTCDecoderTest, DedupAcrossBatches_DuplicateSpansBoundary) {
  // Batch 1: ["h", "h"] -> prev="h", output=""
  // Batch 2: ["h", "e"] -> dedup "h", output="h", prev="e"
  // Finalize: output="e"
  // Total: "he" (not "hhe")
  ScopedDecoderState state(decoder());
  std::string result;
  char output[64];

  // Batch 1.
  std::vector<std::string> batch1 = {"h", "h"};
  auto views1 = ToStringViews(batch1);
  iree_host_size_t consumed = 0, written = 0;
  IREE_ASSERT_OK(iree_tokenizer_decoder_state_process(
      state.get(),
      iree_tokenizer_make_string_list(views1.data(), views1.size()),
      iree_make_mutable_string_view(output, sizeof(output)), &consumed,
      &written));
  EXPECT_EQ(consumed, 2u);
  result.append(output, written);

  // Batch 2.
  std::vector<std::string> batch2 = {"h", "e"};
  auto views2 = ToStringViews(batch2);
  IREE_ASSERT_OK(iree_tokenizer_decoder_state_process(
      state.get(),
      iree_tokenizer_make_string_list(views2.data(), views2.size()),
      iree_make_mutable_string_view(output, sizeof(output)), &consumed,
      &written));
  EXPECT_EQ(consumed, 2u);
  result.append(output, written);

  // Finalize.
  IREE_ASSERT_OK(iree_tokenizer_decoder_state_finalize(
      state.get(), iree_make_mutable_string_view(output, sizeof(output)),
      &written));
  result.append(output, written);

  EXPECT_EQ(result, "he");
}

TEST_F(CTCDecoderTest, PadAtBatchBoundary) {
  // Batch 1: ["l", "<pad>"] -> output="l", prev cleared
  // Batch 2: ["l", "o"]     -> prev="l", output=""
  // Finalize: output="lo"
  // Total: "llo"
  ScopedDecoderState state(decoder());
  std::string result;
  char output[64];

  // Batch 1.
  std::vector<std::string> batch1 = {"l", "<pad>"};
  auto views1 = ToStringViews(batch1);
  iree_host_size_t consumed = 0, written = 0;
  IREE_ASSERT_OK(iree_tokenizer_decoder_state_process(
      state.get(),
      iree_tokenizer_make_string_list(views1.data(), views1.size()),
      iree_make_mutable_string_view(output, sizeof(output)), &consumed,
      &written));
  result.append(output, written);

  // Batch 2.
  std::vector<std::string> batch2 = {"l", "o"};
  auto views2 = ToStringViews(batch2);
  IREE_ASSERT_OK(iree_tokenizer_decoder_state_process(
      state.get(),
      iree_tokenizer_make_string_list(views2.data(), views2.size()),
      iree_make_mutable_string_view(output, sizeof(output)), &consumed,
      &written));
  result.append(output, written);

  // Finalize.
  IREE_ASSERT_OK(iree_tokenizer_decoder_state_finalize(
      state.get(), iree_make_mutable_string_view(output, sizeof(output)),
      &written));
  result.append(output, written);

  EXPECT_EQ(result, "llo");
}

//===----------------------------------------------------------------------===//
// Buffer Exhaustion
//===----------------------------------------------------------------------===//

TEST_F(CTCDecoderTest, LimitedOutputCapacity) {
  auto tokens = SplitTokens("<pad> <pad> h e e l l <pad> l o o o <pad>");
  TestLimitedOutputCapacity(decoder(), tokens, "hello");
}

TEST_F(CTCDecoderTest, MultipleFinalizeCallsSmallBuffer) {
  // Process CTC-style tokens, then finalize with tiny buffer requiring multiple
  // calls. Use pads between l's so both appear (CTC dedup removes consecutive
  // identical tokens - this matches HuggingFace exactly).
  ScopedDecoderState state(decoder());
  std::vector<std::string> tokens = {"h", "e", "l", "<pad>", "l", "o"};
  auto views = ToStringViews(tokens);
  char output[1];  // Tiny buffer.
  std::string result;

  // Process all tokens - they go into prev_token chain.
  iree_host_size_t consumed = 0, written = 0;
  size_t position = 0;
  while (position < tokens.size()) {
    iree_tokenizer_string_list_t token_list = {
        .count = tokens.size() - position,
        .values = views.data() + position,
    };
    IREE_ASSERT_OK(iree_tokenizer_decoder_state_process(
        state.get(), token_list,
        iree_make_mutable_string_view(output, sizeof(output)), &consumed,
        &written));
    result.append(output, written);
    position += consumed;
  }

  // Finalize with tiny buffer - may need multiple calls.
  while (iree_tokenizer_decoder_state_has_pending(state.get())) {
    IREE_ASSERT_OK(iree_tokenizer_decoder_state_finalize(
        state.get(), iree_make_mutable_string_view(output, sizeof(output)),
        &written));
    result.append(output, written);
  }

  EXPECT_EQ(result, "hello");
}

TEST_F(CTCDecoderTest, WordDelimiterAtBatchBoundary) {
  // Verify word delimiter at batch boundary produces correct space.
  // Use CTC-style input with pads to break dedup for repeated letters.
  ScopedDecoderState state(decoder());
  std::string result;
  char output[64];
  iree_host_size_t consumed = 0, written = 0;

  // Batch 1: "hello" (pad between l's for CTC dedup)
  std::vector<std::string> batch1 = {"h", "e", "l", "<pad>", "l", "o"};
  auto views1 = ToStringViews(batch1);
  IREE_ASSERT_OK(iree_tokenizer_decoder_state_process(
      state.get(),
      iree_tokenizer_make_string_list(views1.data(), views1.size()),
      iree_make_mutable_string_view(output, sizeof(output)), &consumed,
      &written));
  result.append(output, written);

  // Batch 2: "|world" - delimiter at start of batch
  std::vector<std::string> batch2 = {"|", "w", "o", "r", "l", "d"};
  auto views2 = ToStringViews(batch2);
  IREE_ASSERT_OK(iree_tokenizer_decoder_state_process(
      state.get(),
      iree_tokenizer_make_string_list(views2.data(), views2.size()),
      iree_make_mutable_string_view(output, sizeof(output)), &consumed,
      &written));
  result.append(output, written);

  // Finalize.
  IREE_ASSERT_OK(iree_tokenizer_decoder_state_finalize(
      state.get(), iree_make_mutable_string_view(output, sizeof(output)),
      &written));
  result.append(output, written);

  EXPECT_EQ(result, "hello world");
}

//===----------------------------------------------------------------------===//
// Configuration Variants
//===----------------------------------------------------------------------===//

TEST_F(CTCDecoderTest, CustomPadToken) {
  // Use "[PAD]" instead of default "<pad>".
  iree_tokenizer_decoder_t* raw_decoder = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_decoder_ctc_allocate(
      iree_make_cstring_view("[PAD]"), iree_make_cstring_view("|"),
      /*cleanup=*/true, iree_allocator_system(), &raw_decoder));
  ScopedDecoder decoder(raw_decoder);

  std::vector<std::string> tokens = {"[PAD]", "[PAD]", "h",     "e", "e",
                                     "l",     "l",     "[PAD]", "l", "o",
                                     "o",     "o",     "[PAD]"};
  TestWithAllBatchSizes(decoder.get(), tokens, "hello",
                        /*expect_pending_after_process=*/false);
}

TEST_F(CTCDecoderTest, CustomWordDelimiter) {
  // Use "<space>" instead of default "|".
  iree_tokenizer_decoder_t* raw_decoder = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_decoder_ctc_allocate(
      iree_make_cstring_view("<pad>"), iree_make_cstring_view("<space>"),
      /*cleanup=*/true, iree_allocator_system(), &raw_decoder));
  ScopedDecoder decoder(raw_decoder);

  std::vector<std::string> tokens = {"h", "i", "<space>", "t",
                                     "h", "e", "r",       "e"};
  TestWithAllBatchSizes(decoder.get(), tokens, "hi there",
                        /*expect_pending_after_process=*/true);
}

TEST_F(CTCDecoderTest, CleanupDisabled) {
  // cleanup=false -> word delimiter NOT replaced with space.
  iree_tokenizer_decoder_t* raw_decoder = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_decoder_ctc_allocate(
      iree_make_cstring_view("<pad>"), iree_make_cstring_view("|"),
      /*cleanup=*/false, iree_allocator_system(), &raw_decoder));
  ScopedDecoder decoder(raw_decoder);

  std::vector<std::string> tokens = {"h", "i", "|", "t", "h", "e", "r", "e"};
  // With cleanup=false, "|" stays as "|".
  TestWithAllBatchSizes(decoder.get(), tokens, "hi|there",
                        /*expect_pending_after_process=*/true);
}

//===----------------------------------------------------------------------===//
// Wordpiece Cleanup Rules
//===----------------------------------------------------------------------===//

// Note: These rules apply to multi-character tokens. Single-char CTC tokens
// won't match, but we test them for completeness with multi-char tokens.

TEST_F(CTCDecoderTest, CleanupRule_SpaceBeforePunctuation) {
  // Test cleanup rule: " ." -> "."
  // We need a multi-char token containing " ." for this to apply.
  std::vector<std::string> tokens = {"hello .", "world"};
  TestWithAllBatchSizes(decoder(), tokens, "hello.world",
                        /*expect_pending_after_process=*/true);
}

TEST_F(CTCDecoderTest, CleanupRule_SpaceBeforeQuestion) {
  std::vector<std::string> tokens = {"what ?", "yes"};
  TestWithAllBatchSizes(decoder(), tokens, "what?yes",
                        /*expect_pending_after_process=*/true);
}

TEST_F(CTCDecoderTest, CleanupRule_SpaceBeforeExclamation) {
  std::vector<std::string> tokens = {"wow !", "cool"};
  TestWithAllBatchSizes(decoder(), tokens, "wow!cool",
                        /*expect_pending_after_process=*/true);
}

TEST_F(CTCDecoderTest, CleanupRule_SpaceBeforeComma) {
  std::vector<std::string> tokens = {"hi ,", "there"};
  TestWithAllBatchSizes(decoder(), tokens, "hi,there",
                        /*expect_pending_after_process=*/true);
}

TEST_F(CTCDecoderTest, CleanupRule_SpaceAroundApostrophe) {
  // " ' " -> "'"
  std::vector<std::string> tokens = {"it ' s"};
  TestWithAllBatchSizes(decoder(), tokens, "it's",
                        /*expect_pending_after_process=*/true);
}

TEST_F(CTCDecoderTest, CleanupRule_Contraction_Nt) {
  // " n't" -> "n't"
  std::vector<std::string> tokens = {"do n't"};
  TestWithAllBatchSizes(decoder(), tokens, "don't",
                        /*expect_pending_after_process=*/true);
}

TEST_F(CTCDecoderTest, CleanupRule_Contraction_M) {
  // " 'm" -> "'m"
  std::vector<std::string> tokens = {"I 'm"};
  TestWithAllBatchSizes(decoder(), tokens, "I'm",
                        /*expect_pending_after_process=*/true);
}

TEST_F(CTCDecoderTest, CleanupRule_DoNot) {
  // " do not" -> " don't"
  std::vector<std::string> tokens = {"I do not"};
  TestWithAllBatchSizes(decoder(), tokens, "I don't",
                        /*expect_pending_after_process=*/true);
}

TEST_F(CTCDecoderTest, CleanupRule_Contraction_S) {
  // " 's" -> "'s"
  std::vector<std::string> tokens = {"it 's"};
  TestWithAllBatchSizes(decoder(), tokens, "it's",
                        /*expect_pending_after_process=*/true);
}

TEST_F(CTCDecoderTest, CleanupRule_Contraction_Ve) {
  // " 've" -> "'ve"
  std::vector<std::string> tokens = {"I 've"};
  TestWithAllBatchSizes(decoder(), tokens, "I've",
                        /*expect_pending_after_process=*/true);
}

TEST_F(CTCDecoderTest, CleanupRule_Contraction_Re) {
  // " 're" -> "'re"
  std::vector<std::string> tokens = {"you 're"};
  TestWithAllBatchSizes(decoder(), tokens, "you're",
                        /*expect_pending_after_process=*/true);
}

//===----------------------------------------------------------------------===//
// has_pending Behavior
//===----------------------------------------------------------------------===//

TEST_F(CTCDecoderTest, HasPending_AfterSingleToken) {
  ScopedDecoderState state(decoder());
  std::vector<std::string> tokens = {"h"};
  auto views = ToStringViews(tokens);
  char output[64];
  iree_host_size_t consumed = 0, written = 0;

  IREE_ASSERT_OK(iree_tokenizer_decoder_state_process(
      state.get(), iree_tokenizer_make_string_list(views.data(), views.size()),
      iree_make_mutable_string_view(output, sizeof(output)), &consumed,
      &written));

  // Token is buffered in prev_token, so has_pending should be true.
  EXPECT_TRUE(iree_tokenizer_decoder_state_has_pending(state.get()));
}

TEST_F(CTCDecoderTest, HasPending_FalseAfterFinalize) {
  ScopedDecoderState state(decoder());
  std::vector<std::string> tokens = {"h"};
  auto views = ToStringViews(tokens);
  char output[64];
  iree_host_size_t consumed = 0, written = 0;

  IREE_ASSERT_OK(iree_tokenizer_decoder_state_process(
      state.get(), iree_tokenizer_make_string_list(views.data(), views.size()),
      iree_make_mutable_string_view(output, sizeof(output)), &consumed,
      &written));

  IREE_ASSERT_OK(iree_tokenizer_decoder_state_finalize(
      state.get(), iree_make_mutable_string_view(output, sizeof(output)),
      &written));

  EXPECT_FALSE(iree_tokenizer_decoder_state_has_pending(state.get()));
}

//===----------------------------------------------------------------------===//
// Error Handling
//===----------------------------------------------------------------------===//

TEST(CTCDecoderAllocationTest, PadTokenTooLong) {
  // Create a pad_token that exceeds max size.
  std::string long_token(IREE_TOKENIZER_DECODER_CTC_MAX_TOKEN_SIZE + 1, 'x');
  iree_tokenizer_decoder_t* decoder = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_tokenizer_decoder_ctc_allocate(
          iree_make_string_view(long_token.data(), long_token.size()),
          iree_make_cstring_view("|"), /*cleanup=*/true,
          iree_allocator_system(), &decoder));
}

TEST(CTCDecoderAllocationTest, WordDelimiterTooLong) {
  std::string long_token(IREE_TOKENIZER_DECODER_CTC_MAX_TOKEN_SIZE + 1, 'x');
  iree_tokenizer_decoder_t* decoder = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_tokenizer_decoder_ctc_allocate(
          iree_make_cstring_view("<pad>"),
          iree_make_string_view(long_token.data(), long_token.size()),
          /*cleanup=*/true, iree_allocator_system(), &decoder));
}

TEST(CTCDecoderAllocationTest, EmptyPadTokenRejected) {
  iree_tokenizer_decoder_t* decoder = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_tokenizer_decoder_ctc_allocate(
          iree_make_string_view("", 0), iree_make_cstring_view("|"),
          /*cleanup=*/true, iree_allocator_system(), &decoder));
}

TEST(CTCDecoderAllocationTest, EmptyWordDelimiterRejected) {
  iree_tokenizer_decoder_t* decoder = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_tokenizer_decoder_ctc_allocate(
          iree_make_cstring_view("<pad>"), iree_make_string_view("", 0),
          /*cleanup=*/true, iree_allocator_system(), &decoder));
}

}  // namespace
}  // namespace iree::tokenizer
