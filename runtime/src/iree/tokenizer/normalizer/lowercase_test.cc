// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/normalizer/lowercase.h"

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
// Test fixture for lowercase normalizer tests.
//===----------------------------------------------------------------------===//

class LowercaseNormalizerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_tokenizer_normalizer_t* raw_normalizer = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_normalizer_lowercase_allocate(
        iree_allocator_system(), &raw_normalizer));
    normalizer_ = ScopedNormalizer(raw_normalizer);
  }

  iree_tokenizer_normalizer_t* normalizer() { return normalizer_.get(); }

 private:
  ScopedNormalizer normalizer_;
};

//===----------------------------------------------------------------------===//
// Lifecycle
//===----------------------------------------------------------------------===//

TEST_F(LowercaseNormalizerTest, CreateAndDestroy) {
  EXPECT_NE(normalizer(), nullptr);
}

TEST_F(LowercaseNormalizerTest, StateSizeIsReasonable) {
  iree_host_size_t state_size =
      iree_tokenizer_normalizer_state_size(normalizer());
  EXPECT_GT(state_size, 0u);
  // State should be small (pending UTF-8 + pending output).
  EXPECT_LE(state_size, 64u);
}

//===----------------------------------------------------------------------===//
// No-ops
//===----------------------------------------------------------------------===//

TEST_F(LowercaseNormalizerTest, EmptyInput) {
  std::string result = ProcessAndFinalize(
      normalizer(), "", /*expect_pending_after_process=*/false);
  EXPECT_TRUE(result.empty());
}

TEST_F(LowercaseNormalizerTest, AlreadyLowercase) {
  TestWithAllChunkSizes(normalizer(), "hello world", "hello world",
                        /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// ASCII Lowercasing (Fast Path)
//===----------------------------------------------------------------------===//

TEST_F(LowercaseNormalizerTest, AsciiUppercase) {
  TestWithAllChunkSizes(normalizer(), "HELLO", "hello",
                        /*expect_pending_after_process=*/false);
}

TEST_F(LowercaseNormalizerTest, AsciiMixedCase) {
  TestWithAllChunkSizes(normalizer(), "HeLLo WoRLD", "hello world",
                        /*expect_pending_after_process=*/false);
}

TEST_F(LowercaseNormalizerTest, AsciiWithNumbersAndSymbols) {
  TestWithAllChunkSizes(normalizer(), "ABC123!@#XYZ", "abc123!@#xyz",
                        /*expect_pending_after_process=*/false);
}

TEST_F(LowercaseNormalizerTest, LongAsciiRun) {
  // Test that ASCII fast path handles long runs efficiently.
  std::string input(1000, 'A');
  std::string expected(1000, 'a');
  TestWithAllChunkSizes(normalizer(), input, expected,
                        /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// UTF-8 Lowercasing (Non-ASCII)
//===----------------------------------------------------------------------===//

TEST_F(LowercaseNormalizerTest, Utf8LatinAccented) {
  // CAFÉ → café (É is U+00C9, é is U+00E9).
  TestWithAllChunkSizes(normalizer(), "CAF\xC3\x89", "caf\xC3\xA9",
                        /*expect_pending_after_process=*/false);
}

TEST_F(LowercaseNormalizerTest, Utf8Greek) {
  // ΩMEGA → ωmega (Ω is U+03A9, ω is U+03C9).
  TestWithAllChunkSizes(normalizer(), "\xCE\xA9MEGA", "\xCF\x89mega",
                        /*expect_pending_after_process=*/false);
}

TEST_F(LowercaseNormalizerTest, Utf8Cyrillic) {
  // ПРИВЕТ → привет.
  TestWithAllChunkSizes(
      normalizer(),
      "\xD0\x9F\xD0\xA0\xD0\x98\xD0\x92\xD0\x95\xD0\xA2",  // ПРИВЕТ
      "\xD0\xBF\xD1\x80\xD0\xB8\xD0\xB2\xD0\xB5\xD1\x82",  // привет
      /*expect_pending_after_process=*/false);
}

TEST_F(LowercaseNormalizerTest, MixedAsciiAndUtf8) {
  // "HELLO WÖRLD" → "hello wörld" (tests ASCII fast path → UTF-8 transition).
  TestWithAllChunkSizes(normalizer(), "HELLO W\xC3\x96RLD",
                        "hello w\xC3\xB6rld",
                        /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// İ Expansion (U+0130 → U+0069 + U+0307)
//===----------------------------------------------------------------------===//

TEST_F(LowercaseNormalizerTest, TurkishDottedI) {
  // İ (U+0130) → i + combining dot above (U+0069 + U+0307).
  // İ in UTF-8 is 0xC4 0xB0.
  // i is 0x69, combining dot above is 0xCC 0x87.
  TestWithAllChunkSizes(normalizer(), "\xC4\xB0", "i\xCC\x87",
                        /*expect_pending_after_process=*/false);
}

TEST_F(LowercaseNormalizerTest, TurkishDottedIInWord) {
  // İSTANBUL → i̇stanbul.
  TestWithAllChunkSizes(normalizer(), "\xC4\xB0STANBUL", "i\xCC\x87stanbul",
                        /*expect_pending_after_process=*/false);
}

TEST_F(LowercaseNormalizerTest, MultipleTurkishDottedI) {
  // İİİ → i̇i̇i̇ (3 expansions).
  TestWithAllChunkSizes(normalizer(), "\xC4\xB0\xC4\xB0\xC4\xB0",
                        "i\xCC\x87i\xCC\x87i\xCC\x87",
                        /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// Characters That Don't Change
//===----------------------------------------------------------------------===//

TEST_F(LowercaseNormalizerTest, NumbersUnchanged) {
  TestWithAllChunkSizes(normalizer(), "12345", "12345",
                        /*expect_pending_after_process=*/false);
}

TEST_F(LowercaseNormalizerTest, SymbolsUnchanged) {
  TestWithAllChunkSizes(normalizer(), "!@#$%^&*()", "!@#$%^&*()",
                        /*expect_pending_after_process=*/false);
}

TEST_F(LowercaseNormalizerTest, EmojiUnchanged) {
  // Emoji have no case. 🎉 is U+1F389 (4-byte UTF-8).
  TestWithAllChunkSizes(normalizer(), "\xF0\x9F\x8E\x89", "\xF0\x9F\x8E\x89",
                        /*expect_pending_after_process=*/false);
}

TEST_F(LowercaseNormalizerTest, CjkUnchanged) {
  // CJK has no case. 你好 (U+4F60 U+597D).
  TestWithAllChunkSizes(normalizer(), "\xE4\xBD\xA0\xE5\xA5\xBD",
                        "\xE4\xBD\xA0\xE5\xA5\xBD",
                        /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// Streaming Behavior
//===----------------------------------------------------------------------===//

// Tests for partial UTF-8 handling were removed because the normalizer
// interface contract guarantees callers provide input on codepoint boundaries.

TEST_F(LowercaseNormalizerTest, LimitedOutputCapacity) {
  TestLimitedOutputCapacity(normalizer(), "HELLO WORLD", "hello world");
}

TEST_F(LowercaseNormalizerTest, Capacity1WithExpansion) {
  // Test that İ expansion works correctly with output capacity=1.
  ScopedNormalizerState state(normalizer());

  // İ expands to 3 bytes output (i + combining dot = 0x69 + 0xCC 0x87).
  std::string result;
  char output_char;
  const char* input = "\xC4\xB0";  // İ
  size_t position = 0;

  while (position < 2 ||
         iree_tokenizer_normalizer_state_has_pending(state.get())) {
    iree_host_size_t consumed = 0;
    iree_host_size_t written = 0;

    iree_string_view_t remaining = {input + position, 2 - position};
    IREE_ASSERT_OK(iree_tokenizer_normalizer_state_process(
        state.get(), remaining, iree_make_mutable_string_view(&output_char, 1),
        IREE_TOKENIZER_NORMALIZER_FLAG_NONE, &consumed, &written));

    if (written > 0) {
      result += output_char;
    }

    ASSERT_TRUE(consumed > 0 || written > 0) << "No progress";
    position += consumed;
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

  EXPECT_EQ(result, "i\xCC\x87");
}

}  // namespace
}  // namespace iree::tokenizer
