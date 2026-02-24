// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/normalizer/prepend.h"

#include <string>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/tokenizer/normalizer/normalizer_test_util.h"

namespace iree::tokenizer {
namespace {

using testing::ProcessAndFinalize;
using testing::ScopedNormalizer;
using testing::ScopedNormalizerState;
using testing::TestLimitedOutputCapacity;
using testing::TestWithAllChunkSizes;

//===----------------------------------------------------------------------===//
// Test fixture for prepend normalizer tests.
//===----------------------------------------------------------------------===//

class PrependNormalizerTest : public ::testing::Test {
 protected:
  // Creates a prepend normalizer with the given prepend string.
  ScopedNormalizer CreatePrepend(iree_string_view_t prepend_string) {
    iree_tokenizer_normalizer_t* raw_normalizer = nullptr;
    IREE_EXPECT_OK(iree_tokenizer_normalizer_prepend_allocate(
        prepend_string, /*skip_if_prefix_matches=*/false,
        iree_allocator_system(), &raw_normalizer));
    return ScopedNormalizer(raw_normalizer);
  }
};

//===----------------------------------------------------------------------===//
// Lifecycle
//===----------------------------------------------------------------------===//

TEST_F(PrependNormalizerTest, CreateAndDestroy) {
  auto normalizer = CreatePrepend(IREE_SV("prefix"));
  EXPECT_NE(normalizer.get(), nullptr);
}

TEST_F(PrependNormalizerTest, CreateWithEmptyPrepend) {
  auto normalizer = CreatePrepend(iree_string_view_empty());
  EXPECT_NE(normalizer.get(), nullptr);
}

TEST_F(PrependNormalizerTest, StateSizeIsReasonable) {
  auto normalizer = CreatePrepend(IREE_SV("prefix"));
  iree_host_size_t state_size =
      iree_tokenizer_normalizer_state_size(normalizer.get());
  EXPECT_GT(state_size, 0u);
  // State includes hot fields, decision state, and 16-byte prefix buffer.
  EXPECT_LE(state_size, 128u);
}

//===----------------------------------------------------------------------===//
// No-ops
//===----------------------------------------------------------------------===//

TEST_F(PrependNormalizerTest, EmptyInputProducesEmptyOutput) {
  // Per HuggingFace behavior: empty input → empty output (no prepend).
  auto normalizer = CreatePrepend(IREE_SV("prefix"));
  std::string result = ProcessAndFinalize(
      normalizer.get(), "", /*expect_pending_after_process=*/false);
  EXPECT_TRUE(result.empty());
}

TEST_F(PrependNormalizerTest, EmptyPrependPassesThrough) {
  auto normalizer = CreatePrepend(iree_string_view_empty());
  TestWithAllChunkSizes(normalizer.get(), "hello", "hello",
                        /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// Basic Prepending
//===----------------------------------------------------------------------===//

TEST_F(PrependNormalizerTest, BasicPrepend) {
  auto normalizer = CreatePrepend(IREE_SV("prefix"));
  TestWithAllChunkSizes(normalizer.get(), "hello", "prefixhello",
                        /*expect_pending_after_process=*/false);
}

TEST_F(PrependNormalizerTest, SingleCharPrepend) {
  auto normalizer = CreatePrepend(IREE_SV("X"));
  TestWithAllChunkSizes(normalizer.get(), "hello", "Xhello",
                        /*expect_pending_after_process=*/false);
}

TEST_F(PrependNormalizerTest, SentencePieceWhitespace) {
  // The common SentencePiece use case: prepend "▁" (U+2581).
  // ▁ in UTF-8 is 0xE2 0x96 0x81.
  auto normalizer = CreatePrepend(IREE_SV("\xE2\x96\x81"));
  TestWithAllChunkSizes(normalizer.get(), "hello", "\xE2\x96\x81hello",
                        /*expect_pending_after_process=*/false);
}

TEST_F(PrependNormalizerTest, PrependToUtf8Input) {
  auto normalizer = CreatePrepend(IREE_SV("→ "));
  // Input: café (with é = U+00E9).
  TestWithAllChunkSizes(normalizer.get(), "caf\xC3\xA9",
                        "\xE2\x86\x92 caf\xC3\xA9",
                        /*expect_pending_after_process=*/false);
}

TEST_F(PrependNormalizerTest, MaxLengthPrepend) {
  // 16 bytes is the maximum supported prepend length.
  std::string max_prepend(16, 'X');
  auto normalizer = CreatePrepend(
      iree_make_string_view(max_prepend.data(), max_prepend.size()));
  std::string expected = max_prepend + "hello";
  TestWithAllChunkSizes(normalizer.get(), "hello", expected,
                        /*expect_pending_after_process=*/false);
}

TEST_F(PrependNormalizerTest, ExceedMaxLengthFails) {
  std::string too_long(17, 'X');
  iree_tokenizer_normalizer_t* raw = nullptr;
  iree_status_t status = iree_tokenizer_normalizer_prepend_allocate(
      iree_make_string_view(too_long.data(), too_long.size()),
      /*skip_if_prefix_matches=*/false, iree_allocator_system(), &raw);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
  EXPECT_EQ(raw, nullptr);
}

//===----------------------------------------------------------------------===//
// Streaming Behavior
//===----------------------------------------------------------------------===//

TEST_F(PrependNormalizerTest, PrependOnlyOnFirstInput) {
  // Verify that prepend happens once per state, not once per process() call.
  auto normalizer = CreatePrepend(IREE_SV("X"));
  ScopedNormalizerState state(normalizer.get());

  // First call with "a".
  char output[64];
  iree_host_size_t consumed = 0;
  iree_host_size_t written = 0;
  IREE_ASSERT_OK(iree_tokenizer_normalizer_state_process(
      state.get(), IREE_SV("a"),
      iree_make_mutable_string_view(output, sizeof(output)),
      IREE_TOKENIZER_NORMALIZER_FLAG_NONE, &consumed, &written));
  EXPECT_EQ(consumed, 1u);
  EXPECT_EQ(written, 2u);  // "X" + "a"
  EXPECT_EQ(std::string(output, written), "Xa");

  // Second call with "b" - should NOT prepend again.
  IREE_ASSERT_OK(iree_tokenizer_normalizer_state_process(
      state.get(), IREE_SV("b"),
      iree_make_mutable_string_view(output, sizeof(output)),
      IREE_TOKENIZER_NORMALIZER_FLAG_NONE, &consumed, &written));
  EXPECT_EQ(consumed, 1u);
  EXPECT_EQ(written, 1u);  // Just "b", no prepend.
  EXPECT_EQ(std::string(output, written), "b");
}

TEST_F(PrependNormalizerTest, HasPendingDuringPartialPrependWrite) {
  // When output buffer is smaller than prepend string, has_pending should be
  // true until we finish emitting the prepend.
  auto normalizer = CreatePrepend(IREE_SV("prefix"));
  ScopedNormalizerState state(normalizer.get());

  // Process with output capacity = 1.
  char output_char;
  std::string result;
  const char* input = "x";
  size_t input_position = 0;

  // Should take multiple calls to emit "prefix" + "x".
  while (input_position < 1 ||
         iree_tokenizer_normalizer_state_has_pending(state.get())) {
    iree_host_size_t consumed = 0;
    iree_host_size_t written = 0;
    iree_string_view_t remaining = {input + input_position, 1 - input_position};
    IREE_ASSERT_OK(iree_tokenizer_normalizer_state_process(
        state.get(), remaining, iree_make_mutable_string_view(&output_char, 1),
        IREE_TOKENIZER_NORMALIZER_FLAG_NONE, &consumed, &written));
    if (written > 0) {
      result += output_char;
    }
    ASSERT_TRUE(consumed > 0 || written > 0) << "No progress";
    input_position += consumed;
  }

  // Finalize.
  while (true) {
    iree_host_size_t finalize_written = 0;
    IREE_ASSERT_OK(iree_tokenizer_normalizer_state_finalize(
        state.get(), iree_make_mutable_string_view(&output_char, 1),
        &finalize_written));
    if (finalize_written > 0) {
      result += output_char;
    } else {
      break;
    }
  }

  EXPECT_EQ(result, "prefixx");
}

TEST_F(PrependNormalizerTest, LimitedOutputCapacity) {
  auto normalizer = CreatePrepend(IREE_SV(">>"));
  TestLimitedOutputCapacity(normalizer.get(), "hello", ">>hello");
}

//===----------------------------------------------------------------------===//
// State Reset Behavior
//===----------------------------------------------------------------------===//

TEST_F(PrependNormalizerTest, StateResetAllowsNewPrepend) {
  // After Reset(), the prepend should happen again.
  auto normalizer = CreatePrepend(IREE_SV("X"));
  ScopedNormalizerState state(normalizer.get());

  // First use.
  char output[64];
  iree_host_size_t consumed = 0;
  iree_host_size_t written = 0;
  IREE_ASSERT_OK(iree_tokenizer_normalizer_state_process(
      state.get(), IREE_SV("a"),
      iree_make_mutable_string_view(output, sizeof(output)),
      IREE_TOKENIZER_NORMALIZER_FLAG_NONE, &consumed, &written));
  EXPECT_EQ(std::string(output, written), "Xa");

  // Reset state.
  state.Reset();

  // Should prepend again.
  IREE_ASSERT_OK(iree_tokenizer_normalizer_state_process(
      state.get(), IREE_SV("b"),
      iree_make_mutable_string_view(output, sizeof(output)),
      IREE_TOKENIZER_NORMALIZER_FLAG_NONE, &consumed, &written));
  EXPECT_EQ(std::string(output, written), "Xb");
}

//===----------------------------------------------------------------------===//
// Various Input Patterns
//===----------------------------------------------------------------------===//

TEST_F(PrependNormalizerTest, SingleByteInput) {
  auto normalizer = CreatePrepend(IREE_SV("pre"));
  TestWithAllChunkSizes(normalizer.get(), "x", "prex",
                        /*expect_pending_after_process=*/false);
}

TEST_F(PrependNormalizerTest, LongInput) {
  auto normalizer = CreatePrepend(IREE_SV(">>"));
  std::string long_input(1000, 'a');
  std::string expected = ">>" + long_input;
  TestWithAllChunkSizes(normalizer.get(), long_input, expected,
                        /*expect_pending_after_process=*/false);
}

TEST_F(PrependNormalizerTest, BinaryDataInput) {
  // Prepend normalizer should handle binary data (including null bytes).
  auto normalizer = CreatePrepend(IREE_SV("X"));
  std::string input = std::string("a\x00b", 3);
  std::string expected = std::string("Xa\x00b", 4);
  TestWithAllChunkSizes(normalizer.get(), input, expected,
                        /*expect_pending_after_process=*/false);
}

TEST_F(PrependNormalizerTest, SpacePrepend) {
  // ASCII space as prepend string.
  auto normalizer = CreatePrepend(IREE_SV(" "));
  TestWithAllChunkSizes(normalizer.get(), "Hello", " Hello",
                        /*expect_pending_after_process=*/false);
}

TEST_F(PrependNormalizerTest, WhitespaceOnlyInput) {
  // Whitespace input is non-empty, so prepend is emitted.
  auto normalizer = CreatePrepend(IREE_SV("\xE2\x96\x81"));
  TestWithAllChunkSizes(normalizer.get(), "   ", "\xE2\x96\x81   ",
                        /*expect_pending_after_process=*/false);
  TestWithAllChunkSizes(normalizer.get(), "\t\n", "\xE2\x96\x81\t\n",
                        /*expect_pending_after_process=*/false);
}

TEST_F(PrependNormalizerTest, AlreadyPrefixed) {
  // Input already has the prefix - prepend still adds another.
  // This matches HuggingFace behavior: no deduplication.
  auto normalizer = CreatePrepend(IREE_SV("\xE2\x96\x81"));
  TestWithAllChunkSizes(normalizer.get(), "\xE2\x96\x81hello",
                        "\xE2\x96\x81\xE2\x96\x81hello",
                        /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// Unicode Prepend Strings
//===----------------------------------------------------------------------===//

TEST_F(PrependNormalizerTest, EmojiPrepend) {
  // 4-byte emoji as prepend string.
  auto normalizer = CreatePrepend(IREE_SV("\xF0\x9F\x98\x80"));  // 😀
  TestWithAllChunkSizes(normalizer.get(), "hello", "\xF0\x9F\x98\x80hello",
                        /*expect_pending_after_process=*/false);
}

TEST_F(PrependNormalizerTest, MultiByteUtf8Prepend) {
  // Two CJK characters as prepend (6 bytes UTF-8, within 16 byte limit).
  // 你好 = U+4F60 U+597D = \xE4\xBD\xA0 \xE5\xA5\xBD
  auto normalizer = CreatePrepend(IREE_SV("\xE4\xBD\xA0\xE5\xA5\xBD"));
  TestWithAllChunkSizes(normalizer.get(), "hi", "\xE4\xBD\xA0\xE5\xA5\xBDhi",
                        /*expect_pending_after_process=*/false);
}

TEST_F(PrependNormalizerTest, CombiningMarkPrepend) {
  // Combining acute accent as prepend (orphan combining mark).
  // U+0301 = 0xCC 0x81
  auto normalizer = CreatePrepend(IREE_SV("\xCC\x81"));
  TestWithAllChunkSizes(normalizer.get(), "e",
                        "\xCC\x81"
                        "e",
                        /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// Unicode Input
//===----------------------------------------------------------------------===//

TEST_F(PrependNormalizerTest, CjkInput) {
  // Chinese characters.
  auto normalizer = CreatePrepend(IREE_SV("\xE2\x96\x81"));
  // 你好 = U+4F60 U+597D
  TestWithAllChunkSizes(normalizer.get(), "\xE4\xBD\xA0\xE5\xA5\xBD",
                        "\xE2\x96\x81\xE4\xBD\xA0\xE5\xA5\xBD",
                        /*expect_pending_after_process=*/false);
}

TEST_F(PrependNormalizerTest, JapaneseInput) {
  // Japanese hiragana: こんにちは
  auto normalizer = CreatePrepend(IREE_SV("\xE2\x96\x81"));
  TestWithAllChunkSizes(normalizer.get(),
                        "\xE3\x81\x93\xE3\x82\x93\xE3\x81\xAB"
                        "\xE3\x81\xA1\xE3\x81\xAF",
                        "\xE2\x96\x81\xE3\x81\x93\xE3\x82\x93\xE3\x81\xAB"
                        "\xE3\x81\xA1\xE3\x81\xAF",
                        /*expect_pending_after_process=*/false);
}

TEST_F(PrependNormalizerTest, ArabicInput) {
  // Arabic text (right-to-left): مرحبا
  auto normalizer = CreatePrepend(IREE_SV("\xE2\x96\x81"));
  TestWithAllChunkSizes(normalizer.get(),
                        "\xD9\x85\xD8\xB1\xD8\xAD\xD8\xA8\xD8\xA7",
                        "\xE2\x96\x81\xD9\x85\xD8\xB1\xD8\xAD\xD8\xA8\xD8\xA7",
                        /*expect_pending_after_process=*/false);
}

TEST_F(PrependNormalizerTest, InputWithCombiningMark) {
  // Input with combining mark: e + combining acute (é in NFD form).
  auto normalizer = CreatePrepend(IREE_SV("\xE2\x96\x81"));
  TestWithAllChunkSizes(normalizer.get(), "e\xCC\x81",
                        "\xE2\x96\x81"
                        "e\xCC\x81",
                        /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// Empty and Null Handling
//===----------------------------------------------------------------------===//

TEST_F(PrependNormalizerTest, EmptyStringViewNull) {
  // iree_string_view_empty() returns {NULL, 0} - verify allocation handles
  // NULL data pointer correctly.
  auto normalizer = CreatePrepend(iree_string_view_empty());
  EXPECT_NE(normalizer.get(), nullptr);

  // Empty prepend with NULL data = passthrough behavior.
  TestWithAllChunkSizes(normalizer.get(), "Hello", "Hello",
                        /*expect_pending_after_process=*/false);
  TestWithAllChunkSizes(normalizer.get(), "", "",
                        /*expect_pending_after_process=*/false);
}

TEST_F(PrependNormalizerTest, EmptyBoth) {
  // Both empty prepend and empty input = empty output.
  auto normalizer = CreatePrepend(iree_string_view_empty());
  std::string result = ProcessAndFinalize(
      normalizer.get(), "", /*expect_pending_after_process=*/false);
  EXPECT_TRUE(result.empty());
}

//===----------------------------------------------------------------------===//
// Conditional Prepend (skip_if_prefix_matches Mode)
//===----------------------------------------------------------------------===//

class ConditionalPrependTest : public ::testing::Test {
 protected:
  // Creates a conditional prepend normalizer (skip_if_prefix_matches=true).
  ScopedNormalizer CreateConditionalPrepend(iree_string_view_t prepend_string) {
    iree_tokenizer_normalizer_t* raw_normalizer = nullptr;
    IREE_EXPECT_OK(iree_tokenizer_normalizer_prepend_allocate(
        prepend_string, /*skip_if_prefix_matches=*/true,
        iree_allocator_system(), &raw_normalizer));
    return ScopedNormalizer(raw_normalizer);
  }
};

TEST_F(ConditionalPrependTest, SkipsWhenInputStartsWithPrepend) {
  // Input already starts with ▁ → skip prepend.
  auto normalizer = CreateConditionalPrepend(IREE_SV("\xE2\x96\x81"));
  TestWithAllChunkSizes(normalizer.get(), "\xE2\x96\x81hello",
                        "\xE2\x96\x81hello",
                        /*expect_pending_after_process=*/false);
}

TEST_F(ConditionalPrependTest, PrependsWhenInputDoesNotMatch) {
  // Input starts with 'H' which doesn't match ▁ → prepend.
  auto normalizer = CreateConditionalPrepend(IREE_SV("\xE2\x96\x81"));
  TestWithAllChunkSizes(normalizer.get(), "Hello", "\xE2\x96\x81Hello",
                        /*expect_pending_after_process=*/false);
}

TEST_F(ConditionalPrependTest, SkipsWhenInputIsExactlyPrepend) {
  // Input is exactly the prepend string → skip prepend, output unchanged.
  auto normalizer = CreateConditionalPrepend(IREE_SV("\xE2\x96\x81"));
  TestWithAllChunkSizes(normalizer.get(), "\xE2\x96\x81", "\xE2\x96\x81",
                        /*expect_pending_after_process=*/false);
}

TEST_F(ConditionalPrependTest, PrependsWhenInputIsShorterThanPrepend) {
  // Input (1 byte 'H') is shorter than prepend (3 bytes ▁).
  // Can't fully match → prepend fires.
  auto normalizer = CreateConditionalPrepend(IREE_SV("\xE2\x96\x81"));
  TestWithAllChunkSizes(normalizer.get(), "H", "\xE2\x96\x81H",
                        /*expect_pending_after_process=*/false);
}

TEST_F(ConditionalPrependTest, PrependsOnPartialByteMatch) {
  // First byte matches (0xE2) but second doesn't → prepend.
  auto normalizer = CreateConditionalPrepend(IREE_SV("\xE2\x96\x81"));
  TestWithAllChunkSizes(normalizer.get(), "\xE2\x80\x8Dtext",
                        "\xE2\x96\x81\xE2\x80\x8Dtext",
                        /*expect_pending_after_process=*/false);
}

TEST_F(ConditionalPrependTest, SingleBytePrependSkips) {
  // Single-byte prepend "X" — skip when input starts with 'X'.
  auto normalizer = CreateConditionalPrepend(IREE_SV("X"));
  TestWithAllChunkSizes(normalizer.get(), "Xhello", "Xhello",
                        /*expect_pending_after_process=*/false);
}

TEST_F(ConditionalPrependTest, SingleBytePrependFires) {
  // Single-byte prepend "X" — fires when input starts with 'Y'.
  auto normalizer = CreateConditionalPrepend(IREE_SV("X"));
  TestWithAllChunkSizes(normalizer.get(), "Yhello", "XYhello",
                        /*expect_pending_after_process=*/false);
}

TEST_F(ConditionalPrependTest, MetaspaceScenarioSpaceInput) {
  // The actual Mistral scenario: input "▁" (after replace), skip prepend.
  auto normalizer = CreateConditionalPrepend(IREE_SV("\xE2\x96\x81"));
  TestWithAllChunkSizes(normalizer.get(), "\xE2\x96\x81", "\xE2\x96\x81",
                        /*expect_pending_after_process=*/false);
}

TEST_F(ConditionalPrependTest, MetaspaceScenarioWordInput) {
  // Input "Hello▁world" (after replace) — doesn't start with ▁, so prepend.
  auto normalizer = CreateConditionalPrepend(IREE_SV("\xE2\x96\x81"));
  TestWithAllChunkSizes(normalizer.get(), "Hello\xE2\x96\x81world",
                        "\xE2\x96\x81Hello\xE2\x96\x81world",
                        /*expect_pending_after_process=*/false);
}

TEST_F(ConditionalPrependTest, LimitedOutputCapacity) {
  // Verify streaming with small output buffer works for both skip and prepend.
  auto normalizer_skip = CreateConditionalPrepend(IREE_SV("\xE2\x96\x81"));
  TestLimitedOutputCapacity(normalizer_skip.get(), "\xE2\x96\x81hello",
                            "\xE2\x96\x81hello");

  auto normalizer_prepend = CreateConditionalPrepend(IREE_SV("\xE2\x96\x81"));
  TestLimitedOutputCapacity(normalizer_prepend.get(), "hello",
                            "\xE2\x96\x81hello");
}

TEST_F(ConditionalPrependTest, EmptyInputNoOutput) {
  // Empty input → no prepend, no output (regardless of mode).
  auto normalizer = CreateConditionalPrepend(IREE_SV("\xE2\x96\x81"));
  std::string result = ProcessAndFinalize(
      normalizer.get(), "", /*expect_pending_after_process=*/false);
  EXPECT_TRUE(result.empty());
}

TEST_F(ConditionalPrependTest, StateResetAllowsNewDecision) {
  // After reset, the decision is re-evaluated.
  auto normalizer = CreateConditionalPrepend(IREE_SV("X"));
  ScopedNormalizerState state(normalizer.get());

  // First use: input "Xhello" → skip prepend.
  char output[64];
  iree_host_size_t consumed = 0;
  iree_host_size_t written = 0;
  IREE_ASSERT_OK(iree_tokenizer_normalizer_state_process(
      state.get(), IREE_SV("Xhello"),
      iree_make_mutable_string_view(output, sizeof(output)),
      IREE_TOKENIZER_NORMALIZER_FLAG_NONE, &consumed, &written));
  EXPECT_EQ(std::string(output, written), "Xhello");

  // Reset state.
  state.Reset();

  // Second use: input "Yhello" → prepend fires.
  IREE_ASSERT_OK(iree_tokenizer_normalizer_state_process(
      state.get(), IREE_SV("Yhello"),
      iree_make_mutable_string_view(output, sizeof(output)),
      IREE_TOKENIZER_NORMALIZER_FLAG_NONE, &consumed, &written));
  EXPECT_EQ(std::string(output, written), "XYhello");
}

TEST_F(ConditionalPrependTest, FirstConsumedFlagSkipsPrepend) {
  // When FIRST_CONSUMED flag is set, skip prepend for prepend_scheme="first"
  // semantics. This handles the case where a special token like <s> consumed
  // position 0 of the original input, so subsequent text is not "first".
  auto normalizer = CreateConditionalPrepend(IREE_SV("\xE2\x96\x81"));
  ScopedNormalizerState state(normalizer.get());

  // Input "hello" normally gets ▁ prepended, but with FIRST_CONSUMED flag
  // the prepend is skipped because position 0 was consumed by a special token.
  char output[64];
  iree_host_size_t consumed = 0;
  iree_host_size_t written = 0;
  IREE_ASSERT_OK(iree_tokenizer_normalizer_state_process(
      state.get(), IREE_SV("hello"),
      iree_make_mutable_string_view(output, sizeof(output)),
      IREE_TOKENIZER_NORMALIZER_FLAG_FIRST_CONSUMED, &consumed, &written));
  EXPECT_EQ(consumed, 5u);
  EXPECT_EQ(written, 5u);
  EXPECT_EQ(std::string(output, written), "hello");  // No prepend!
}

TEST_F(ConditionalPrependTest, FirstConsumedFlagIgnoredInUnconditionalMode) {
  // In unconditional mode (skip_if_prefix_matches=false), the FIRST_CONSUMED
  // flag is ignored and prepend always fires.
  iree_tokenizer_normalizer_t* raw_normalizer = nullptr;
  IREE_EXPECT_OK(iree_tokenizer_normalizer_prepend_allocate(
      IREE_SV("\xE2\x96\x81"), /*skip_if_prefix_matches=*/false,
      iree_allocator_system(), &raw_normalizer));
  ScopedNormalizer normalizer(raw_normalizer);
  ScopedNormalizerState state(normalizer.get());

  // Even with FIRST_CONSUMED, unconditional mode still prepends.
  char output[64];
  iree_host_size_t consumed = 0;
  iree_host_size_t written = 0;
  IREE_ASSERT_OK(iree_tokenizer_normalizer_state_process(
      state.get(), IREE_SV("hello"),
      iree_make_mutable_string_view(output, sizeof(output)),
      IREE_TOKENIZER_NORMALIZER_FLAG_FIRST_CONSUMED, &consumed, &written));
  EXPECT_EQ(consumed, 5u);
  EXPECT_EQ(written, 8u);  // 3 bytes ▁ + 5 bytes "hello"
  EXPECT_EQ(std::string(output, written), "\xE2\x96\x81hello");
}

}  // namespace
}  // namespace iree::tokenizer
