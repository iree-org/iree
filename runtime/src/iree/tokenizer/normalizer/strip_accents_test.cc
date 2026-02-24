// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/normalizer/strip_accents.h"

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
// Test fixture for strip_accents normalizer tests.
//===----------------------------------------------------------------------===//

class StripAccentsNormalizerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_tokenizer_normalizer_t* raw_normalizer = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_normalizer_strip_accents_allocate(
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

TEST_F(StripAccentsNormalizerTest, CreateAndDestroy) {
  EXPECT_NE(normalizer(), nullptr);
}

TEST_F(StripAccentsNormalizerTest, StateSizeIsReasonable) {
  iree_host_size_t state_size =
      iree_tokenizer_normalizer_state_size(normalizer());
  EXPECT_GT(state_size, 0u);
  // State should be minimal (just base struct, no buffering needed).
  EXPECT_LE(state_size, 16u);
}

//===----------------------------------------------------------------------===//
// No-ops
//===----------------------------------------------------------------------===//

TEST_F(StripAccentsNormalizerTest, EmptyInput) {
  std::string result = ProcessAndFinalize(
      normalizer(), "", /*expect_pending_after_process=*/false);
  EXPECT_TRUE(result.empty());
}

TEST_F(StripAccentsNormalizerTest, AsciiUnchanged) {
  // Pure ASCII has no combining marks.
  TestWithAllChunkSizes(normalizer(), "hello world", "hello world",
                        /*expect_pending_after_process=*/false);
}

TEST_F(StripAccentsNormalizerTest, NumbersUnchanged) {
  TestWithAllChunkSizes(normalizer(), "12345", "12345",
                        /*expect_pending_after_process=*/false);
}

TEST_F(StripAccentsNormalizerTest, SymbolsUnchanged) {
  TestWithAllChunkSizes(normalizer(), "!@#$%^&*()", "!@#$%^&*()",
                        /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// Precomposed Characters Pass Through Unchanged
//===----------------------------------------------------------------------===//

TEST_F(StripAccentsNormalizerTest, PrecomposedEAccuteUnchanged) {
  // é (U+00E9) is a single codepoint, NOT a base + combining mark.
  // It should pass through unchanged.
  TestWithAllChunkSizes(normalizer(), "\xC3\xA9", "\xC3\xA9",
                        /*expect_pending_after_process=*/false);
}

TEST_F(StripAccentsNormalizerTest, PrecomposedCafeUnchanged) {
  // "café" with composed é should be unchanged.
  TestWithAllChunkSizes(normalizer(), "caf\xC3\xA9", "caf\xC3\xA9",
                        /*expect_pending_after_process=*/false);
}

TEST_F(StripAccentsNormalizerTest, PrecomposedGermanUnchanged) {
  // German umlauts as precomposed characters should be unchanged.
  // ä (U+00E4), ö (U+00F6), ü (U+00FC).
  TestWithAllChunkSizes(normalizer(), "\xC3\xA4\xC3\xB6\xC3\xBC",
                        "\xC3\xA4\xC3\xB6\xC3\xBC",
                        /*expect_pending_after_process=*/false);
}

TEST_F(StripAccentsNormalizerTest, PrecomposedFrenchUnchanged) {
  // "être" with precomposed ê should be unchanged.
  TestWithAllChunkSizes(normalizer(), "\xC3\xAAtre", "\xC3\xAAtre",
                        /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// Decomposed Characters (Marks Stripped)
//===----------------------------------------------------------------------===//

TEST_F(StripAccentsNormalizerTest, CombiningAcuteStripped) {
  // e + combining acute accent (U+0301) → e
  // Combining acute in UTF-8: 0xCC 0x81
  TestWithAllChunkSizes(normalizer(), "e\xCC\x81", "e",
                        /*expect_pending_after_process=*/false);
}

TEST_F(StripAccentsNormalizerTest, DecomposedCafe) {
  // "cafe" + combining acute (decomposed form) → "cafe"
  TestWithAllChunkSizes(normalizer(), "cafe\xCC\x81", "cafe",
                        /*expect_pending_after_process=*/false);
}

TEST_F(StripAccentsNormalizerTest, MultipleMarksStripped) {
  // e + combining macron (U+0304) x3 → e
  // Combining macron in UTF-8: 0xCC 0x84
  TestWithAllChunkSizes(normalizer(), "e\xCC\x84\xCC\x84\xCC\x84", "e",
                        /*expect_pending_after_process=*/false);
}

TEST_F(StripAccentsNormalizerTest, MarksOnMultipleBases) {
  // a + combining grave, b + combining acute, c + combining circumflex → abc
  // Combining grave (U+0300): 0xCC 0x80
  // Combining acute (U+0301): 0xCC 0x81
  // Combining circumflex (U+0302): 0xCC 0x82
  TestWithAllChunkSizes(normalizer(),
                        "a\xCC\x80"
                        "b\xCC\x81"
                        "c\xCC\x82",
                        "abc", /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// Mark Categories (Mn, Mc, Me)
//===----------------------------------------------------------------------===//

TEST_F(StripAccentsNormalizerTest, NonspacingMarkStripped) {
  // Mn (Nonspacing Mark): Combining acute accent U+0301.
  TestWithAllChunkSizes(normalizer(), "a\xCC\x81", "a",
                        /*expect_pending_after_process=*/false);
}

TEST_F(StripAccentsNormalizerTest, SpacingCombiningMarkStripped) {
  // Mc (Spacing Combining Mark): Devanagari sign visarga U+0903.
  // UTF-8: 0xE0 0xA4 0x83
  // This follows Devanagari letter A (U+0905): 0xE0 0xA4 0x85
  TestWithAllChunkSizes(normalizer(), "\xE0\xA4\x85\xE0\xA4\x83",
                        "\xE0\xA4\x85",
                        /*expect_pending_after_process=*/false);
}

TEST_F(StripAccentsNormalizerTest, EnclosingMarkStripped) {
  // Me (Enclosing Mark): Combining enclosing circle U+20DD.
  // UTF-8: 0xE2 0x83 0x9D
  TestWithAllChunkSizes(normalizer(), "A\xE2\x83\x9D", "A",
                        /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// Script Coverage
//===----------------------------------------------------------------------===//

TEST_F(StripAccentsNormalizerTest, ChineseUnchanged) {
  // Chinese has no combining marks.
  // 你好 (U+4F60 U+597D).
  TestWithAllChunkSizes(normalizer(), "\xE4\xBD\xA0\xE5\xA5\xBD",
                        "\xE4\xBD\xA0\xE5\xA5\xBD",
                        /*expect_pending_after_process=*/false);
}

TEST_F(StripAccentsNormalizerTest, JapaneseUnchanged) {
  // Hiragana/Katakana have no combining marks.
  // こんにちは (hiragana).
  TestWithAllChunkSizes(
      normalizer(),
      "\xE3\x81\x93\xE3\x82\x93\xE3\x81\xAB\xE3\x81\xA1\xE3\x81\xAF",
      "\xE3\x81\x93\xE3\x82\x93\xE3\x81\xAB\xE3\x81\xA1\xE3\x81\xAF",
      /*expect_pending_after_process=*/false);
}

TEST_F(StripAccentsNormalizerTest, KoreanUnchanged) {
  // Precomposed Hangul syllables have no combining marks.
  // 안녕 (U+C548 U+B155).
  TestWithAllChunkSizes(normalizer(), "\xEC\x95\x88\xEB\x85\x95",
                        "\xEC\x95\x88\xEB\x85\x95",
                        /*expect_pending_after_process=*/false);
}

TEST_F(StripAccentsNormalizerTest, GreekWithCombiningMark) {
  // Greek alpha (U+03B1) + combining acute (U+0301) → alpha
  TestWithAllChunkSizes(normalizer(), "\xCE\xB1\xCC\x81", "\xCE\xB1",
                        /*expect_pending_after_process=*/false);
}

TEST_F(StripAccentsNormalizerTest, CyrillicWithCombiningMark) {
  // Cyrillic small letter a (U+0430) + combining breve (U+0306) → a
  // Combining breve in UTF-8: 0xCC 0x86
  TestWithAllChunkSizes(normalizer(), "\xD0\xB0\xCC\x86", "\xD0\xB0",
                        /*expect_pending_after_process=*/false);
}

TEST_F(StripAccentsNormalizerTest, ThaiWithToneMark) {
  // Thai character with combining tone mark should have mark stripped.
  // Thai mai ek (U+0E48) is a combining mark.
  // Thai ko kai (U+0E01): 0xE0 0xB8 0x81
  // Thai mai ek (U+0E48): 0xE0 0xB9 0x88
  TestWithAllChunkSizes(normalizer(), "\xE0\xB8\x81\xE0\xB9\x88",
                        "\xE0\xB8\x81",
                        /*expect_pending_after_process=*/false);
}

TEST_F(StripAccentsNormalizerTest, EmojiUnchanged) {
  // Emoji have no combining marks (skin tone modifiers are not marks).
  // 🎉 (U+1F389).
  TestWithAllChunkSizes(normalizer(), "\xF0\x9F\x8E\x89", "\xF0\x9F\x8E\x89",
                        /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// Edge Cases
//===----------------------------------------------------------------------===//

TEST_F(StripAccentsNormalizerTest, OnlyMarks) {
  // Input consisting entirely of combining marks → empty.
  // Three combining acute accents.
  TestWithAllChunkSizes(normalizer(), "\xCC\x81\xCC\x81\xCC\x81", "",
                        /*expect_pending_after_process=*/false);
}

TEST_F(StripAccentsNormalizerTest, MarkAtStart) {
  // Mark at start (orphan mark) should be stripped.
  // Combining acute + abc → abc
  TestWithAllChunkSizes(normalizer(),
                        "\xCC\x81"
                        "abc",
                        "abc", /*expect_pending_after_process=*/false);
}

TEST_F(StripAccentsNormalizerTest, MarkAtEnd) {
  // Mark at end should be stripped.
  // abc + combining acute → abc
  TestWithAllChunkSizes(normalizer(), "abc\xCC\x81", "abc",
                        /*expect_pending_after_process=*/false);
}

TEST_F(StripAccentsNormalizerTest, AlternatingBaseAndMark) {
  // a + mark + b + mark + c + mark → abc
  TestWithAllChunkSizes(normalizer(),
                        "a\xCC\x81"
                        "b\xCC\x82"
                        "c\xCC\x83",
                        "abc", /*expect_pending_after_process=*/false);
}

TEST_F(StripAccentsNormalizerTest, LongRunOfMarks) {
  // Single base followed by many marks.
  // a + 10 combining acute accents → a
  std::string input = "a";
  for (int i = 0; i < 10; ++i) {
    input += "\xCC\x81";
  }
  TestWithAllChunkSizes(normalizer(), input, "a",
                        /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// HuggingFace Compatibility Quirks
//===----------------------------------------------------------------------===//

TEST_F(StripAccentsNormalizerTest, VariationSelectorsAreStripped) {
  // QUIRK: Variation selectors (U+FE0E text, U+FE0F emoji) ARE Mn marks!
  // They get stripped, which can affect emoji rendering.
  // ❤ (U+2764) + U+FE0F (emoji style) → ❤ alone
  // U+2764: 0xE2 0x9D 0xA4
  // U+FE0F: 0xEF 0xB8 0x8F
  TestWithAllChunkSizes(normalizer(), "\xE2\x9D\xA4\xEF\xB8\x8F",
                        "\xE2\x9D\xA4",
                        /*expect_pending_after_process=*/false);
  // Text style selector U+FE0E also stripped.
  // U+FE0E: 0xEF 0xB8 0x8E
  TestWithAllChunkSizes(normalizer(), "\xE2\x9D\xA4\xEF\xB8\x8E",
                        "\xE2\x9D\xA4",
                        /*expect_pending_after_process=*/false);
}

TEST_F(StripAccentsNormalizerTest, EmojiSkinTonesPreserved) {
  // Skin tone modifiers are Sk (Symbol, modifier), NOT marks.
  // They should be preserved.
  // 👍 (U+1F44D): 0xF0 0x9F 0x91 0x8D
  // Light skin tone (U+1F3FB): 0xF0 0x9F 0x8F 0xBB
  TestWithAllChunkSizes(normalizer(), "\xF0\x9F\x91\x8D\xF0\x9F\x8F\xBB",
                        "\xF0\x9F\x91\x8D\xF0\x9F\x8F\xBB",
                        /*expect_pending_after_process=*/false);
}

TEST_F(StripAccentsNormalizerTest, ZwjSequencesPreserved) {
  // ZWJ (U+200D) is Cf (Format), NOT a mark - preserved.
  // Family emoji: 👨 ZWJ 👩 ZWJ 👧
  // 👨 (U+1F468): 0xF0 0x9F 0x91 0xA8
  // ZWJ (U+200D): 0xE2 0x80 0x8D
  // 👩 (U+1F469): 0xF0 0x9F 0x91 0xA9
  // 👧 (U+1F467): 0xF0 0x9F 0x91 0xA7
  TestWithAllChunkSizes(normalizer(),
                        "\xF0\x9F\x91\xA8\xE2\x80\x8D"
                        "\xF0\x9F\x91\xA9\xE2\x80\x8D"
                        "\xF0\x9F\x91\xA7",
                        "\xF0\x9F\x91\xA8\xE2\x80\x8D"
                        "\xF0\x9F\x91\xA9\xE2\x80\x8D"
                        "\xF0\x9F\x91\xA7",
                        /*expect_pending_after_process=*/false);
}

TEST_F(StripAccentsNormalizerTest, HebrewVowelPointsStripped) {
  // Hebrew vowel points are Mn marks - they get stripped.
  // Alef (U+05D0): 0xD7 0x90
  // Patah (U+05B7): 0xD6 0xB7
  TestWithAllChunkSizes(normalizer(), "\xD7\x90\xD6\xB7", "\xD7\x90",
                        /*expect_pending_after_process=*/false);
}

TEST_F(StripAccentsNormalizerTest, ArabicDiacriticsStripped) {
  // Arabic diacritics (harakat) are Mn marks - they get stripped.
  // Ba (U+0628): 0xD8 0xA8
  // Fatha (U+064E): 0xD9 0x8E
  TestWithAllChunkSizes(normalizer(), "\xD8\xA8\xD9\x8E", "\xD8\xA8",
                        /*expect_pending_after_process=*/false);
}

TEST_F(StripAccentsNormalizerTest, CombiningGraphemeJoinerStripped) {
  // Combining Grapheme Joiner (U+034F) is Mn - gets stripped.
  // Used to prevent ligature formation but is a mark.
  // a + CGJ + b → ab
  // CGJ (U+034F): 0xCD 0x8F
  TestWithAllChunkSizes(normalizer(),
                        "a\xCD\x8F"
                        "b",
                        "ab", /*expect_pending_after_process=*/false);
}

TEST_F(StripAccentsNormalizerTest, MathCombiningOverlineStripped) {
  // Math combining overline (U+0305) is Mn - stripped.
  // x + overline → x
  // U+0305: 0xCC 0x85
  TestWithAllChunkSizes(normalizer(), "x\xCC\x85", "x",
                        /*expect_pending_after_process=*/false);
}

TEST_F(StripAccentsNormalizerTest, MixedPrecomposedAndDecomposed) {
  // Precomposed é followed by decomposed é (e + combining acute).
  // Precomposed stays, decomposed loses its mark.
  // é (U+00E9): 0xC3 0xA9
  // e + U+0301: 0x65 0xCC 0x81
  TestWithAllChunkSizes(normalizer(),
                        "\xC3\xA9"
                        "e\xCC\x81",
                        "\xC3\xA9"
                        "e",
                        /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// Streaming Behavior
//===----------------------------------------------------------------------===//

TEST_F(StripAccentsNormalizerTest, LimitedOutputCapacity) {
  // Test with output buffer size 1.
  TestLimitedOutputCapacity(normalizer(),
                            "a\xCC\x81"
                            "b\xCC\x82"
                            "c",
                            "abc");
}

TEST_F(StripAccentsNormalizerTest, LimitedOutputWithAscii) {
  TestLimitedOutputCapacity(normalizer(), "hello", "hello");
}

TEST_F(StripAccentsNormalizerTest, HasPendingAlwaysFalse) {
  // Strip accents never has pending data (only filters, never buffers).
  ScopedNormalizerState state(normalizer());

  // Process some input.
  char output_buffer[64];
  iree_host_size_t consumed = 0;
  iree_host_size_t written = 0;
  IREE_ASSERT_OK(iree_tokenizer_normalizer_state_process(
      state.get(), iree_make_cstring_view("a\xCC\x81"),
      iree_make_mutable_string_view(output_buffer, sizeof(output_buffer)),
      IREE_TOKENIZER_NORMALIZER_FLAG_NONE, &consumed, &written));

  // Should never have pending.
  EXPECT_FALSE(iree_tokenizer_normalizer_state_has_pending(state.get()));

  // After finalize, still no pending.
  iree_host_size_t finalize_written = 0;
  IREE_ASSERT_OK(iree_tokenizer_normalizer_state_finalize(
      state.get(),
      iree_make_mutable_string_view(output_buffer, sizeof(output_buffer)),
      &finalize_written));
  EXPECT_FALSE(iree_tokenizer_normalizer_state_has_pending(state.get()));
  EXPECT_EQ(finalize_written, 0u);
}

}  // namespace
}  // namespace iree::tokenizer
