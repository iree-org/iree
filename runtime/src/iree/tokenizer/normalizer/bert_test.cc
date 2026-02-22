// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/normalizer/bert.h"

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
// Test Helpers
//===----------------------------------------------------------------------===//

// Helper to create a BERT normalizer with specific flags.
ScopedNormalizer CreateBertNormalizer(
    iree_tokenizer_bert_normalizer_flags_t flags) {
  iree_tokenizer_normalizer_t* raw_normalizer = nullptr;
  IREE_CHECK_OK(iree_tokenizer_normalizer_bert_allocate(
      flags, iree_allocator_system(), &raw_normalizer));
  return ScopedNormalizer(raw_normalizer);
}

//===----------------------------------------------------------------------===//
// Lifecycle
//===----------------------------------------------------------------------===//

class BertNormalizerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    normalizer_ =
        CreateBertNormalizer(IREE_TOKENIZER_BERT_NORMALIZER_FLAG_DEFAULT);
  }

  iree_tokenizer_normalizer_t* normalizer() { return normalizer_.get(); }

 private:
  ScopedNormalizer normalizer_;
};

TEST_F(BertNormalizerTest, CreateAndDestroy) {
  EXPECT_NE(normalizer(), nullptr);
}

TEST_F(BertNormalizerTest, StateSizeIsReasonable) {
  iree_host_size_t state_size =
      iree_tokenizer_normalizer_state_size(normalizer());
  EXPECT_GT(state_size, 0u);
  // State includes pending buffer, flags, etc.
  EXPECT_LE(state_size, 128u);
}

//===----------------------------------------------------------------------===//
// No-ops and Passthrough
//===----------------------------------------------------------------------===//

TEST_F(BertNormalizerTest, EmptyInput) {
  std::string result = ProcessAndFinalize(
      normalizer(), "", /*expect_pending_after_process=*/false);
  EXPECT_TRUE(result.empty());
}

TEST(BertNoFlagsTest, PassthroughWhenNoFlags) {
  // With no flags, input should pass through unchanged.
  auto normalizer =
      CreateBertNormalizer(IREE_TOKENIZER_BERT_NORMALIZER_FLAG_NONE);
  TestWithAllChunkSizes(normalizer.get(), "Hello World!", "Hello World!",
                        /*expect_pending_after_process=*/false);
}

TEST(BertNoFlagsTest, PassthroughUnicodeWhenNoFlags) {
  auto normalizer =
      CreateBertNormalizer(IREE_TOKENIZER_BERT_NORMALIZER_FLAG_NONE);
  // Chinese + Japanese + Korean mixed.
  TestWithAllChunkSizes(normalizer.get(),
                        "\xE4\xBD\xA0\xE5\xA5\xBD"   // 你好 (Chinese)
                        "\xE3\x81\x93\xE3\x82\x93"   // こん (Japanese)
                        "\xEC\x95\x88\xEB\x85\x95",  // 안녕 (Korean)
                        "\xE4\xBD\xA0\xE5\xA5\xBD"
                        "\xE3\x81\x93\xE3\x82\x93"
                        "\xEC\x95\x88\xEB\x85\x95",
                        /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// Control Character Handling
//===----------------------------------------------------------------------===//

class BertCleanTextTest : public ::testing::Test {
 protected:
  void SetUp() override {
    normalizer_ =
        CreateBertNormalizer(IREE_TOKENIZER_BERT_NORMALIZER_FLAG_CLEAN_TEXT);
  }

  iree_tokenizer_normalizer_t* normalizer() { return normalizer_.get(); }

 private:
  ScopedNormalizer normalizer_;
};

TEST_F(BertCleanTextTest, RemovesNullCharacter) {
  // Null character should be removed.
  // Must use std::string to construct input with embedded null.
  std::string input = "a";
  input.push_back('\0');
  input.push_back('b');
  std::string result = ProcessAndFinalize(
      normalizer(), input, /*expect_pending_after_process=*/false);
  EXPECT_EQ(result, "ab");
}

TEST_F(BertCleanTextTest, RemovesReplacementChar) {
  // U+FFFD (replacement character) should be removed.
  // UTF-8: 0xEF 0xBF 0xBD
  TestWithAllChunkSizes(normalizer(),
                        "a\xEF\xBF\xBD"
                        "b",
                        "ab", /*expect_pending_after_process=*/false);
}

TEST_F(BertCleanTextTest, TabToSpace) {
  // Tab is whitespace, not control - mapped to space.
  TestWithAllChunkSizes(normalizer(), "a\tb", "a b",
                        /*expect_pending_after_process=*/false);
}

TEST_F(BertCleanTextTest, NewlineToSpace) {
  // Newline is whitespace, not control - mapped to space.
  TestWithAllChunkSizes(normalizer(), "a\nb", "a b",
                        /*expect_pending_after_process=*/false);
}

TEST_F(BertCleanTextTest, CarriageReturnToSpace) {
  // Carriage return is whitespace, not control - mapped to space.
  TestWithAllChunkSizes(normalizer(), "a\rb", "a b",
                        /*expect_pending_after_process=*/false);
}

TEST_F(BertCleanTextTest, ControlCharactersRemoved) {
  // Control characters (not \t, \n, \r) should be removed.
  // U+0001 (Start of Heading): 0x01
  TestWithAllChunkSizes(normalizer(),
                        "a\x01"
                        "b",
                        "ab", /*expect_pending_after_process=*/false);
}

TEST_F(BertCleanTextTest, DELRemoved) {
  // DEL (U+007F) is a control character.
  TestWithAllChunkSizes(normalizer(),
                        "a\x7F"
                        "b",
                        "ab", /*expect_pending_after_process=*/false);
}

TEST_F(BertCleanTextTest, FormatCharsRemoved) {
  // Format characters (Cf) should be removed.
  // Zero Width Space (U+200B): 0xE2 0x80 0x8B
  TestWithAllChunkSizes(normalizer(),
                        "a\xE2\x80\x8B"
                        "b",
                        "ab", /*expect_pending_after_process=*/false);
}

TEST_F(BertCleanTextTest, ZWJRemoved) {
  // Zero Width Joiner (U+200D): 0xE2 0x80 0x8D
  TestWithAllChunkSizes(normalizer(),
                        "a\xE2\x80\x8D"
                        "b",
                        "ab", /*expect_pending_after_process=*/false);
}

TEST_F(BertCleanTextTest, DirectionalMarksRemoved) {
  // Left-to-Right Mark (U+200E): 0xE2 0x80 0x8E
  TestWithAllChunkSizes(normalizer(),
                        "a\xE2\x80\x8E"
                        "b",
                        "ab", /*expect_pending_after_process=*/false);
}

TEST_F(BertCleanTextTest, BOMRemoved) {
  // Byte Order Mark (U+FEFF): 0xEF 0xBB 0xBF
  TestWithAllChunkSizes(normalizer(), "\xEF\xBB\xBFhello", "hello",
                        /*expect_pending_after_process=*/false);
}

TEST_F(BertCleanTextTest, WhitespaceNormalized) {
  // Various Unicode whitespace should become regular space.
  // Non-breaking space (U+00A0): 0xC2 0xA0
  TestWithAllChunkSizes(normalizer(),
                        "a\xC2\xA0"
                        "b",
                        "a b", /*expect_pending_after_process=*/false);
}

TEST_F(BertCleanTextTest, MultipleSpacesPreserved) {
  // Multiple spaces are preserved (only mapping, not collapsing).
  TestWithAllChunkSizes(normalizer(), "a   b", "a   b",
                        /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// CJK Spacing
//===----------------------------------------------------------------------===//

class BertChineseTest : public ::testing::Test {
 protected:
  void SetUp() override {
    normalizer_ = CreateBertNormalizer(
        IREE_TOKENIZER_BERT_NORMALIZER_FLAG_HANDLE_CHINESE_CHARS);
  }

  iree_tokenizer_normalizer_t* normalizer() { return normalizer_.get(); }

 private:
  ScopedNormalizer normalizer_;
};

TEST_F(BertChineseTest, SpacesAroundCJKChar) {
  // 中 (U+4E2D): 0xE4 0xB8 0xAD → " 中 "
  TestWithAllChunkSizes(normalizer(), "\xE4\xB8\xAD", " \xE4\xB8\xAD ",
                        /*expect_pending_after_process=*/false);
}

TEST_F(BertChineseTest, MultipleCJKChars) {
  // 你好 → " 你  好 "
  TestWithAllChunkSizes(normalizer(), "\xE4\xBD\xA0\xE5\xA5\xBD",
                        " \xE4\xBD\xA0  \xE5\xA5\xBD ",
                        /*expect_pending_after_process=*/false);
}

TEST_F(BertChineseTest, MixedASCIIAndCJK) {
  // "hi你好" → "hi 你  好 "
  TestWithAllChunkSizes(normalizer(), "hi\xE4\xBD\xA0\xE5\xA5\xBD",
                        "hi \xE4\xBD\xA0  \xE5\xA5\xBD ",
                        /*expect_pending_after_process=*/false);
}

TEST_F(BertChineseTest, JapaneseHiraganaUnchanged) {
  // Hiragana is NOT treated as Chinese - no spacing added.
  // こ (U+3053): 0xE3 0x81 0x93
  TestWithAllChunkSizes(normalizer(), "\xE3\x81\x93", "\xE3\x81\x93",
                        /*expect_pending_after_process=*/false);
}

TEST_F(BertChineseTest, JapaneseKatakanaUnchanged) {
  // Katakana is NOT treated as Chinese - no spacing added.
  // コ (U+30B3): 0xE3 0x82 0xB3
  TestWithAllChunkSizes(normalizer(), "\xE3\x82\xB3", "\xE3\x82\xB3",
                        /*expect_pending_after_process=*/false);
}

TEST_F(BertChineseTest, KoreanHangulUnchanged) {
  // Hangul is NOT treated as Chinese - no spacing added.
  // 안 (U+C548): 0xEC 0x95 0x88
  TestWithAllChunkSizes(normalizer(), "\xEC\x95\x88", "\xEC\x95\x88",
                        /*expect_pending_after_process=*/false);
}

TEST_F(BertChineseTest, CJKExtensionB) {
  // Test CJK Extension B (U+20000-U+2A6DF).
  // 𠀀 (U+20000): 0xF0 0xA0 0x80 0x80
  TestWithAllChunkSizes(normalizer(), "\xF0\xA0\x80\x80", " \xF0\xA0\x80\x80 ",
                        /*expect_pending_after_process=*/false);
}

TEST_F(BertChineseTest, CJKCompatibilityIdeographs) {
  // Test CJK Compatibility Ideographs (U+F900-U+FAFF).
  // 豈 (U+F900): 0xEF 0xA4 0x80
  TestWithAllChunkSizes(normalizer(), "\xEF\xA4\x80", " \xEF\xA4\x80 ",
                        /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// NFD + Mn Mark Removal
//===----------------------------------------------------------------------===//

class BertStripAccentsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    normalizer_ =
        CreateBertNormalizer(IREE_TOKENIZER_BERT_NORMALIZER_FLAG_STRIP_ACCENTS);
  }

  iree_tokenizer_normalizer_t* normalizer() { return normalizer_.get(); }

 private:
  ScopedNormalizer normalizer_;
};

TEST_F(BertStripAccentsTest, PrecomposedEAccuteToE) {
  // é (U+00E9) → NFD → e + combining acute → strip Mn → e
  // This is different from standalone StripAccents which doesn't decompose!
  TestWithAllChunkSizes(normalizer(), "\xC3\xA9", "e",
                        /*expect_pending_after_process=*/false);
}

TEST_F(BertStripAccentsTest, PrecomposedCafeToToCafe) {
  // café → cafe (strip all accents).
  TestWithAllChunkSizes(normalizer(), "caf\xC3\xA9", "cafe",
                        /*expect_pending_after_process=*/false);
}

TEST_F(BertStripAccentsTest, GermanUmlautsStripped) {
  // ä (U+00E4) → a, ö (U+00F6) → o, ü (U+00FC) → u
  TestWithAllChunkSizes(normalizer(), "\xC3\xA4\xC3\xB6\xC3\xBC", "aou",
                        /*expect_pending_after_process=*/false);
}

TEST_F(BertStripAccentsTest, DecomposedAlsoStripped) {
  // Already decomposed: e + combining acute → e
  TestWithAllChunkSizes(normalizer(), "e\xCC\x81", "e",
                        /*expect_pending_after_process=*/false);
}

TEST_F(BertStripAccentsTest, SpacingCombiningMarkPreserved) {
  // BERT's strip_accents only removes Mn (Nonspacing Mark), NOT Mc.
  // Devanagari visarga (U+0903) is Mc - should be preserved.
  // Devanagari A (U+0905): 0xE0 0xA4 0x85
  // Visarga (U+0903): 0xE0 0xA4 0x83
  TestWithAllChunkSizes(normalizer(), "\xE0\xA4\x85\xE0\xA4\x83",
                        "\xE0\xA4\x85\xE0\xA4\x83",
                        /*expect_pending_after_process=*/false);
}

TEST_F(BertStripAccentsTest, EnclosingMarkPreserved) {
  // BERT's strip_accents only removes Mn, NOT Me.
  // Enclosing circle (U+20DD) is Me - should be preserved.
  TestWithAllChunkSizes(normalizer(), "A\xE2\x83\x9D", "A\xE2\x83\x9D",
                        /*expect_pending_after_process=*/false);
}

TEST_F(BertStripAccentsTest, VietnameseAccentsStripped) {
  // Vietnamese with complex accents.
  // ử (U+1EED) → u
  TestWithAllChunkSizes(normalizer(), "\xE1\xBB\xAD", "u",
                        /*expect_pending_after_process=*/false);
}

TEST_F(BertStripAccentsTest, ASCIIUnchanged) {
  // Plain ASCII should pass through unchanged.
  TestWithAllChunkSizes(normalizer(), "hello", "hello",
                        /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// Case Folding
//===----------------------------------------------------------------------===//

class BertLowercaseTest : public ::testing::Test {
 protected:
  void SetUp() override {
    normalizer_ =
        CreateBertNormalizer(IREE_TOKENIZER_BERT_NORMALIZER_FLAG_LOWERCASE);
  }

  iree_tokenizer_normalizer_t* normalizer() { return normalizer_.get(); }

 private:
  ScopedNormalizer normalizer_;
};

TEST_F(BertLowercaseTest, ASCIIUpperToLower) {
  TestWithAllChunkSizes(normalizer(), "HELLO", "hello",
                        /*expect_pending_after_process=*/false);
}

TEST_F(BertLowercaseTest, MixedCase) {
  TestWithAllChunkSizes(normalizer(), "HeLLo WoRLd", "hello world",
                        /*expect_pending_after_process=*/false);
}

TEST_F(BertLowercaseTest, GermanEsszett) {
  // ß stays ß (already lowercase).
  TestWithAllChunkSizes(normalizer(), "\xC3\x9F", "\xC3\x9F",
                        /*expect_pending_after_process=*/false);
}

TEST_F(BertLowercaseTest, GreekUpperToLower) {
  // Σ (U+03A3) → σ (U+03C3)
  // UTF-8: 0xCE 0xA3 → 0xCF 0x83
  TestWithAllChunkSizes(normalizer(), "\xCE\xA3", "\xCF\x83",
                        /*expect_pending_after_process=*/false);
}

TEST_F(BertLowercaseTest, TurkishI) {
  // İ (U+0130) → i + combining dot above (expands to 2 codepoints)
  // This is the main case where lowercase can expand.
  // UTF-8: 0xC4 0xB0 → 0x69 + 0xCC 0x87
  std::string result =
      ProcessAndFinalize(normalizer(), "\xC4\xB0",
                         /*expect_pending_after_process=*/false);
  // The exact expansion depends on our unicode_to_lower implementation.
  // Common behavior is İ → i or İ → i + combining dot.
  EXPECT_FALSE(result.empty());
  EXPECT_EQ(result[0], 'i');  // First char should be 'i'.
}

TEST_F(BertLowercaseTest, NumbersUnchanged) {
  TestWithAllChunkSizes(normalizer(), "12345", "12345",
                        /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// Default BERT Behavior
//===----------------------------------------------------------------------===//

TEST_F(BertNormalizerTest, DefaultFlagsAllTransformations) {
  // Default: clean_text + handle_chinese_chars + strip_accents + lowercase
  // "Hello 中Café\tWorld" should become:
  // - "Hello" → "hello"
  // - " " → " "
  // - "中" → " 中 " (CJK spacing)
  // - "Café" → "cafe" (strip accents + lowercase)
  // - "\t" → " " (clean_text)
  // - "World" → "world"
  // Expected: "hello  中 cafe world"
  TestWithAllChunkSizes(normalizer(),
                        "Hello \xE4\xB8\xAD"
                        "Caf\xC3\xA9\tWorld",
                        "hello  \xE4\xB8\xAD cafe world",
                        /*expect_pending_after_process=*/false);
}

TEST_F(BertNormalizerTest, ControlAndWhitespaceRemoval) {
  // Mix of control chars and whitespace.
  TestWithAllChunkSizes(normalizer(), "a\x01\tb\nc\rd", "a b c d",
                        /*expect_pending_after_process=*/false);
}

TEST_F(BertNormalizerTest, ComplexMixedInput) {
  // Complex input with all features:
  // "BOM + Chinese + French + control"
  // BOM + "你" + "Étranger" + control char
  std::string input =
      "\xEF\xBB\xBF"     // BOM (removed)
      "\xE4\xBD\xA0"     // 你 (spaced)
      "\xC3\x89tranger"  // Étranger (É→e, lowercase)
      "\x01";            // Control (removed)
  TestWithAllChunkSizes(normalizer(), input, " \xE4\xBD\xA0 etranger",
                        /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// Cased Model (No Lowercase or Accent Stripping)
//===----------------------------------------------------------------------===//

TEST(BertCasedTest, PreservesCase) {
  // Cased model: only clean_text + handle_chinese_chars.
  auto normalizer = CreateBertNormalizer(
      IREE_TOKENIZER_BERT_NORMALIZER_FLAG_CLEAN_TEXT |
      IREE_TOKENIZER_BERT_NORMALIZER_FLAG_HANDLE_CHINESE_CHARS);
  TestWithAllChunkSizes(normalizer.get(), "Hello World", "Hello World",
                        /*expect_pending_after_process=*/false);
}

TEST(BertCasedTest, PreservesAccents) {
  auto normalizer = CreateBertNormalizer(
      IREE_TOKENIZER_BERT_NORMALIZER_FLAG_CLEAN_TEXT |
      IREE_TOKENIZER_BERT_NORMALIZER_FLAG_HANDLE_CHINESE_CHARS);
  TestWithAllChunkSizes(normalizer.get(), "Caf\xC3\xA9", "Caf\xC3\xA9",
                        /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// Edge Cases
//===----------------------------------------------------------------------===//

TEST_F(BertNormalizerTest, OnlyControlChars) {
  // Input consisting entirely of control chars → empty.
  TestWithAllChunkSizes(normalizer(), "\x01\x02\x03", "",
                        /*expect_pending_after_process=*/false);
}

TEST_F(BertNormalizerTest, OnlyCJK) {
  // Just CJK chars with default flags.
  TestWithAllChunkSizes(normalizer(), "\xE4\xBD\xA0\xE5\xA5\xBD",
                        " \xE4\xBD\xA0  \xE5\xA5\xBD ",
                        /*expect_pending_after_process=*/false);
}

TEST_F(BertNormalizerTest, EmojiPreserved) {
  // Emoji should be preserved (not filtered, not CJK).
  // 🎉 (U+1F389): 0xF0 0x9F 0x8E 0x89
  TestWithAllChunkSizes(normalizer(), "\xF0\x9F\x8E\x89", "\xF0\x9F\x8E\x89",
                        /*expect_pending_after_process=*/false);
}

TEST_F(BertNormalizerTest, LongInput) {
  // Test with longer input to exercise buffering.
  std::string input = "The quick brown fox jumps over the lazy dog.";
  std::string expected = "the quick brown fox jumps over the lazy dog.";
  TestWithAllChunkSizes(normalizer(), input, expected,
                        /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// Streaming Behavior
//===----------------------------------------------------------------------===//

TEST_F(BertNormalizerTest, LimitedOutputCapacity) {
  // Test with output buffer size 1.
  TestLimitedOutputCapacity(normalizer(), "Hello World", "hello world");
}

TEST_F(BertNormalizerTest, LimitedOutputWithCJK) {
  // CJK expansion with limited output.
  TestLimitedOutputCapacity(normalizer(), "\xE4\xB8\xAD", " \xE4\xB8\xAD ");
}

TEST_F(BertNormalizerTest, HasPendingWithExpansion) {
  // Process CJK with tiny output buffer - should have pending.
  ScopedNormalizerState state(normalizer());

  char output_buffer[1];  // Very small buffer.
  iree_host_size_t consumed = 0;
  iree_host_size_t written = 0;

  // CJK char 中 (3 bytes) expands to " 中 " (5 bytes).
  IREE_ASSERT_OK(iree_tokenizer_normalizer_state_process(
      state.get(), iree_make_cstring_view("\xE4\xB8\xAD"),
      iree_make_mutable_string_view(output_buffer, sizeof(output_buffer)),
      IREE_TOKENIZER_NORMALIZER_FLAG_NONE, &consumed, &written));

  // Should have pending data.
  EXPECT_TRUE(iree_tokenizer_normalizer_state_has_pending(state.get()));
}

//===----------------------------------------------------------------------===//
// HuggingFace Compatibility Validation
//===----------------------------------------------------------------------===//

TEST(BertHFCompatTest, CleanTextWhitespaceVsControl) {
  // HuggingFace specifically treats \t, \n, \r as whitespace, not control.
  auto normalizer =
      CreateBertNormalizer(IREE_TOKENIZER_BERT_NORMALIZER_FLAG_CLEAN_TEXT);

  // \t → space (whitespace, not removed).
  std::string result = ProcessAndFinalize(normalizer.get(), "a\tb", false);
  EXPECT_EQ(result, "a b");

  // \x01 → removed (control).
  result = ProcessAndFinalize(normalizer.get(),
                              "a\x01"
                              "b",
                              false);
  EXPECT_EQ(result, "ab");
}

TEST(BertHFCompatTest, StripAccentsUsesNFD) {
  // BERT's strip_accents performs NFD decomposition first.
  // This differs from standalone StripAccents which doesn't decompose.
  auto normalizer =
      CreateBertNormalizer(IREE_TOKENIZER_BERT_NORMALIZER_FLAG_STRIP_ACCENTS);

  // Precomposed é (U+00E9) → NFD → e + combining acute → e
  std::string result = ProcessAndFinalize(normalizer.get(), "\xC3\xA9", false);
  EXPECT_EQ(result, "e");
}

TEST(BertHFCompatTest, StripAccentsOnlyRemovesMn) {
  // BERT only removes Mn marks, not Mc or Me.
  auto normalizer =
      CreateBertNormalizer(IREE_TOKENIZER_BERT_NORMALIZER_FLAG_STRIP_ACCENTS);

  // Devanagari A + visarga (Mc) - visarga should be preserved.
  std::string result =
      ProcessAndFinalize(normalizer.get(), "\xE0\xA4\x85\xE0\xA4\x83", false);
  EXPECT_EQ(result, "\xE0\xA4\x85\xE0\xA4\x83");

  // A + enclosing circle (Me) - circle should be preserved.
  result = ProcessAndFinalize(normalizer.get(), "A\xE2\x83\x9D", false);
  EXPECT_EQ(result, "A\xE2\x83\x9D");
}

TEST(BertHFCompatTest, DefaultStripAccentsMatchesLowercase) {
  // In HuggingFace, strip_accents=None defaults to the same as lowercase.
  // So default (lowercase=true) means strip_accents is also enabled.
  // This is what IREE_TOKENIZER_BERT_NORMALIZER_FLAG_DEFAULT provides.
  auto normalizer =
      CreateBertNormalizer(IREE_TOKENIZER_BERT_NORMALIZER_FLAG_DEFAULT);
  std::string result =
      ProcessAndFinalize(normalizer.get(), "\xC3\x89", false);  // É
  EXPECT_EQ(result, "e");  // É → NFD → E + mark → E → e
}

}  // namespace
}  // namespace iree::tokenizer
