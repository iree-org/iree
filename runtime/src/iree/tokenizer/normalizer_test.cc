// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/normalizer.h"

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/tokenizer/regex/compile.h"

namespace {

class NormalizerTest : public ::testing::Test {
 protected:
  std::string Apply(const iree_tokenizer_normalizer_t* normalizer,
                    const char* input) {
    char buffer[1024];
    iree_host_size_t length = 0;
    iree_status_t status = iree_tokenizer_normalizer_apply(
        normalizer, IREE_SV(input), buffer, sizeof(buffer), &length);
    IREE_EXPECT_OK(status);
    return std::string(buffer, length);
  }

  // Helper to compile a regex pattern and initialize a replace_regex
  // normalizer. Returns false on failure.
  bool InitializeReplaceRegex(const char* pattern, const char* content,
                              iree_tokenizer_normalizer_t* out_normalizer) {
    iree_tokenizer_regex_compile_error_t compile_error = {0};
    uint8_t* dfa_data = nullptr;
    iree_host_size_t dfa_size = 0;
    iree_status_t status = iree_tokenizer_regex_compile(
        IREE_SV(pattern), IREE_TOKENIZER_REGEX_COMPILE_FLAG_NONE,
        iree_allocator_system(), &dfa_data, &dfa_size, &compile_error);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      return false;
    }
    status = iree_tokenizer_normalizer_initialize_replace_regex(
        dfa_data, dfa_size, IREE_SV(content), iree_allocator_system(),
        out_normalizer);
    if (!iree_status_is_ok(status)) {
      iree_allocator_free(iree_allocator_system(), dfa_data);
      iree_status_ignore(status);
      return false;
    }
    return true;
  }
};

//===----------------------------------------------------------------------===//
// None Normalizer
//===----------------------------------------------------------------------===//

TEST_F(NormalizerTest, NonePassthrough) {
  iree_tokenizer_normalizer_t norm;
  iree_tokenizer_normalizer_initialize_none(&norm);

  EXPECT_EQ(Apply(&norm, "Hello World"), "Hello World");
  EXPECT_EQ(Apply(&norm, ""), "");
  EXPECT_EQ(Apply(&norm, "123!@#"), "123!@#");

  iree_tokenizer_normalizer_deinitialize(&norm);
}

//===----------------------------------------------------------------------===//
// Lowercase Normalizer
//===----------------------------------------------------------------------===//

TEST_F(NormalizerTest, LowercaseBasic) {
  iree_tokenizer_normalizer_t norm;
  iree_tokenizer_normalizer_initialize_lowercase(&norm);

  EXPECT_EQ(Apply(&norm, "Hello World"), "hello world");
  EXPECT_EQ(Apply(&norm, "UPPERCASE"), "uppercase");
  EXPECT_EQ(Apply(&norm, "already lowercase"), "already lowercase");
  EXPECT_EQ(Apply(&norm, "MiXeD CaSe"), "mixed case");

  iree_tokenizer_normalizer_deinitialize(&norm);
}

TEST_F(NormalizerTest, LowercaseUnicode) {
  iree_tokenizer_normalizer_t norm;
  iree_tokenizer_normalizer_initialize_lowercase(&norm);

  // German sharp s and umlauts.
  EXPECT_EQ(Apply(&norm, "ÜBER"), "über");
  // Greek.
  EXPECT_EQ(Apply(&norm, "ΩMEGA"), "ωmega");

  iree_tokenizer_normalizer_deinitialize(&norm);
}

//===----------------------------------------------------------------------===//
// Strip Accents Normalizer
//===----------------------------------------------------------------------===//

TEST_F(NormalizerTest, StripAccentsBasic) {
  iree_tokenizer_normalizer_t norm;
  iree_tokenizer_normalizer_initialize_strip_accents(&norm);

  EXPECT_EQ(Apply(&norm, "café"), "cafe");
  EXPECT_EQ(Apply(&norm, "naïve"), "naive");
  EXPECT_EQ(Apply(&norm, "résumé"), "resume");

  iree_tokenizer_normalizer_deinitialize(&norm);
}

TEST_F(NormalizerTest, StripAccentsExtended) {
  iree_tokenizer_normalizer_t norm;
  iree_tokenizer_normalizer_initialize_strip_accents(&norm);

  // Spanish.
  EXPECT_EQ(Apply(&norm, "señor"), "senor");
  // Portuguese.
  EXPECT_EQ(Apply(&norm, "coração"), "coracao");

  iree_tokenizer_normalizer_deinitialize(&norm);
}

//===----------------------------------------------------------------------===//
// BERT Normalizer
//===----------------------------------------------------------------------===//

TEST_F(NormalizerTest, BertLowercaseOnly) {
  iree_tokenizer_normalizer_t norm;
  iree_tokenizer_normalizer_initialize_bert(
      IREE_TOKENIZER_BERT_NORMALIZER_FLAG_LOWERCASE, &norm);

  EXPECT_EQ(Apply(&norm, "Hello World"), "hello world");

  iree_tokenizer_normalizer_deinitialize(&norm);
}

TEST_F(NormalizerTest, BertLowercaseAndStripAccents) {
  iree_tokenizer_normalizer_t norm;
  iree_tokenizer_normalizer_initialize_bert(
      IREE_TOKENIZER_BERT_NORMALIZER_FLAG_LOWERCASE |
          IREE_TOKENIZER_BERT_NORMALIZER_FLAG_STRIP_ACCENTS,
      &norm);

  EXPECT_EQ(Apply(&norm, "Café"), "cafe");
  EXPECT_EQ(Apply(&norm, "RÉSUMÉ"), "resume");

  iree_tokenizer_normalizer_deinitialize(&norm);
}

TEST_F(NormalizerTest, BertCleanText) {
  iree_tokenizer_normalizer_t norm;
  iree_tokenizer_normalizer_initialize_bert(
      IREE_TOKENIZER_BERT_NORMALIZER_FLAG_CLEAN_TEXT, &norm);

  // Control characters should be removed.
  EXPECT_EQ(Apply(&norm, "Hello\x01World"), "HelloWorld");
  // But whitespace is preserved.
  EXPECT_EQ(Apply(&norm, "Hello\tWorld"), "Hello\tWorld");

  iree_tokenizer_normalizer_deinitialize(&norm);
}

TEST_F(NormalizerTest, BertHandleChineseChars) {
  iree_tokenizer_normalizer_t norm;
  iree_tokenizer_normalizer_initialize_bert(
      IREE_TOKENIZER_BERT_NORMALIZER_FLAG_HANDLE_CHINESE_CHARS, &norm);

  // Spaces should be added around CJK characters.
  EXPECT_EQ(Apply(&norm, "Hello中文World"), "Hello 中  文 World");

  iree_tokenizer_normalizer_deinitialize(&norm);
}

TEST_F(NormalizerTest, BertFullConfig) {
  // Standard BERT configuration: all flags enabled.
  iree_tokenizer_normalizer_t norm;
  iree_tokenizer_normalizer_initialize_bert(
      IREE_TOKENIZER_BERT_NORMALIZER_FLAG_CLEAN_TEXT |
          IREE_TOKENIZER_BERT_NORMALIZER_FLAG_HANDLE_CHINESE_CHARS |
          IREE_TOKENIZER_BERT_NORMALIZER_FLAG_STRIP_ACCENTS |
          IREE_TOKENIZER_BERT_NORMALIZER_FLAG_LOWERCASE,
      &norm);

  EXPECT_EQ(Apply(&norm, "Hello, World!"), "hello, world!");
  EXPECT_EQ(Apply(&norm, "Café"), "cafe");

  iree_tokenizer_normalizer_deinitialize(&norm);
}

TEST_F(NormalizerTest, BertCombiningMarks) {
  // Test NFD form: é as e + combining acute (U+0301).
  // In NFD: 'e' (0x65) + combining acute (0xCC 0x81 in UTF-8).
  iree_tokenizer_normalizer_t norm;
  iree_tokenizer_normalizer_initialize_bert(
      IREE_TOKENIZER_BERT_NORMALIZER_FLAG_STRIP_ACCENTS, &norm);

  // Precomposed form: é (U+00E9) -> e.
  EXPECT_EQ(Apply(&norm, "caf\xC3\xA9"), "cafe");

  // NFD decomposed form: e + combining acute -> e.
  EXPECT_EQ(Apply(&norm, "cafe\xCC\x81"), "cafe");

  iree_tokenizer_normalizer_deinitialize(&norm);
}

TEST_F(NormalizerTest, BertMultipleCombiningMarks) {
  // Test character with multiple combining marks.
  // ö́ = o + combining umlaut (U+0308) + combining acute (U+0301).
  // UTF-8: o (0x6F) + umlaut (0xCC 0x88) + acute (0xCC 0x81).
  iree_tokenizer_normalizer_t norm;
  iree_tokenizer_normalizer_initialize_bert(
      IREE_TOKENIZER_BERT_NORMALIZER_FLAG_STRIP_ACCENTS, &norm);

  // Both combining marks should be stripped, leaving just 'o'.
  EXPECT_EQ(Apply(&norm, "o\xCC\x88\xCC\x81"), "o");

  iree_tokenizer_normalizer_deinitialize(&norm);
}

TEST_F(NormalizerTest, BertControlCharacterRemoval) {
  // Test removal of various control characters (U+0001-U+001F except
  // whitespace). Note: U+0000 (NUL) cannot be tested via C strings due to null
  // termination.
  iree_tokenizer_normalizer_t norm;
  iree_tokenizer_normalizer_initialize_bert(
      IREE_TOKENIZER_BERT_NORMALIZER_FLAG_CLEAN_TEXT, &norm);

  // SOH (U+0001), STX (U+0002), ETX (U+0003).
  EXPECT_EQ(Apply(&norm, "Hello\x01\x02\x03World"), "HelloWorld");

  // Bell (U+0007), backspace (U+0008).
  EXPECT_EQ(Apply(&norm,
                  "A\x07\x08"
                  "B"),
            "AB");

  // DEL (U+007F).
  EXPECT_EQ(Apply(&norm, "Test\x7FText"), "TestText");

  iree_tokenizer_normalizer_deinitialize(&norm);
}

TEST_F(NormalizerTest, BertZeroWidthChars) {
  // Test handling of zero-width characters.
  iree_tokenizer_normalizer_t norm;
  iree_tokenizer_normalizer_initialize_bert(
      IREE_TOKENIZER_BERT_NORMALIZER_FLAG_CLEAN_TEXT, &norm);

  // ZWNJ (U+200C) - Zero Width Non-Joiner - UTF-8: E2 80 8C.
  // ZWJ (U+200D) - Zero Width Joiner - UTF-8: E2 80 8D.
  // These are format characters and may be preserved or removed.
  std::string result = Apply(&norm,
                             "ab\xE2\x80\x8C"
                             "cd");
  // Result depends on implementation - verify no crash and reasonable output.
  EXPECT_GE(result.size(), 4u);  // At least "abcd".

  iree_tokenizer_normalizer_deinitialize(&norm);
}

//===----------------------------------------------------------------------===//
// Unicode Edge Case Tests
//===----------------------------------------------------------------------===//

TEST_F(NormalizerTest, UnicodeRTLScript) {
  // Test Right-to-Left scripts (Arabic, Hebrew).
  iree_tokenizer_normalizer_t norm;
  iree_tokenizer_normalizer_initialize_lowercase(&norm);

  // Arabic text - no lowercase transformation expected.
  // مرحبا (Marhaba - Hello)
  EXPECT_EQ(Apply(&norm, "\xD9\x85\xD8\xB1\xD8\xAD\xD8\xA8\xD8\xA7"),
            "\xD9\x85\xD8\xB1\xD8\xAD\xD8\xA8\xD8\xA7");

  // Hebrew text.
  // שלום (Shalom - Peace)
  EXPECT_EQ(Apply(&norm, "\xD7\xA9\xD7\x9C\xD7\x95\xD7\x9D"),
            "\xD7\xA9\xD7\x9C\xD7\x95\xD7\x9D");

  iree_tokenizer_normalizer_deinitialize(&norm);
}

TEST_F(NormalizerTest, UnicodeCJKMixedWithLatin) {
  // Test CJK characters mixed with Latin.
  iree_tokenizer_normalizer_t norm;
  iree_tokenizer_normalizer_initialize_bert(
      IREE_TOKENIZER_BERT_NORMALIZER_FLAG_HANDLE_CHINESE_CHARS |
          IREE_TOKENIZER_BERT_NORMALIZER_FLAG_LOWERCASE,
      &norm);

  // Japanese hiragana + Latin.
  // "Hello日本語World" - spaces added around CJK chars.
  std::string result =
      Apply(&norm, "Hello\xE6\x97\xA5\xE6\x9C\xAC\xE8\xAA\x9EWorld");
  // Should have spaces around each CJK character and be lowercased.
  EXPECT_EQ(result[0], 'h');                       // Lowercased.
  EXPECT_NE(result.find(' '), std::string::npos);  // Has spaces.

  iree_tokenizer_normalizer_deinitialize(&norm);
}

TEST_F(NormalizerTest, UnicodeEmoji) {
  // Test emoji handling.
  iree_tokenizer_normalizer_t norm;
  iree_tokenizer_normalizer_initialize_none(&norm);

  // Basic emoji: 😀 (U+1F600).
  EXPECT_EQ(Apply(&norm, "Hello\xF0\x9F\x98\x80World"),
            "Hello\xF0\x9F\x98\x80World");

  // Emoji with ZWJ sequence: 👨‍👩‍👧 (family).
  // This is: man + ZWJ + woman + ZWJ + girl.
  std::string family =
      "\xF0\x9F\x91\xA8"   // 👨
      "\xE2\x80\x8D"       // ZWJ
      "\xF0\x9F\x91\xA9"   // 👩
      "\xE2\x80\x8D"       // ZWJ
      "\xF0\x9F\x91\xA7";  // 👧
  EXPECT_EQ(Apply(&norm, family.c_str()), family);

  iree_tokenizer_normalizer_deinitialize(&norm);
}

//===----------------------------------------------------------------------===//
// Prepend Normalizer
//===----------------------------------------------------------------------===//

TEST_F(NormalizerTest, PrependBasic) {
  iree_tokenizer_normalizer_t norm;
  IREE_ASSERT_OK(iree_tokenizer_normalizer_initialize_prepend(
      IREE_SV("\xE2\x96\x81"), &norm));  // ▁ (U+2581)

  EXPECT_EQ(norm.type, IREE_TOKENIZER_NORMALIZER_PREPEND);
  EXPECT_EQ(Apply(&norm, "Hello"), "\xE2\x96\x81Hello");
  EXPECT_EQ(Apply(&norm, "world"), "\xE2\x96\x81world");

  iree_tokenizer_normalizer_deinitialize(&norm);
}

TEST_F(NormalizerTest, PrependEmpty) {
  iree_tokenizer_normalizer_t norm;
  IREE_ASSERT_OK(
      iree_tokenizer_normalizer_initialize_prepend(IREE_SV(""), &norm));

  // Empty prepend is a passthrough.
  EXPECT_EQ(Apply(&norm, "Hello"), "Hello");
  EXPECT_EQ(Apply(&norm, ""), "");

  iree_tokenizer_normalizer_deinitialize(&norm);
}

TEST_F(NormalizerTest, PrependSpace) {
  iree_tokenizer_normalizer_t norm;
  IREE_ASSERT_OK(
      iree_tokenizer_normalizer_initialize_prepend(IREE_SV(" "), &norm));

  EXPECT_EQ(Apply(&norm, "Hello"), " Hello");

  iree_tokenizer_normalizer_deinitialize(&norm);
}

TEST_F(NormalizerTest, PrependMaxLength) {
  iree_tokenizer_normalizer_t norm;
  // Exactly 16 bytes should succeed (max is 16).
  IREE_ASSERT_OK(iree_tokenizer_normalizer_initialize_prepend(
      IREE_SV("0123456789abcdef"), &norm));
  EXPECT_EQ(Apply(&norm, "X"), "0123456789abcdefX");
  iree_tokenizer_normalizer_deinitialize(&norm);
}

TEST_F(NormalizerTest, PrependTooLong) {
  iree_tokenizer_normalizer_t norm;
  // 17 bytes exceeds max of 16.
  iree_status_t status = iree_tokenizer_normalizer_initialize_prepend(
      IREE_SV("0123456789abcdefg"), &norm);
  EXPECT_TRUE(iree_status_is_invalid_argument(status));
  iree_status_ignore(status);
}

//===----------------------------------------------------------------------===//
// Replace Normalizer
//===----------------------------------------------------------------------===//

TEST_F(NormalizerTest, ReplaceBasic) {
  iree_tokenizer_normalizer_t norm;
  IREE_ASSERT_OK(iree_tokenizer_normalizer_initialize_replace(
      IREE_SV(" "), IREE_SV("\xE2\x96\x81"), &norm));  // space -> ▁

  EXPECT_EQ(norm.type, IREE_TOKENIZER_NORMALIZER_REPLACE);
  EXPECT_EQ(Apply(&norm, "Hello World"), "Hello\xE2\x96\x81World");
  EXPECT_EQ(Apply(&norm, "a b c"),
            "a\xE2\x96\x81"
            "b\xE2\x96\x81"
            "c");

  iree_tokenizer_normalizer_deinitialize(&norm);
}

TEST_F(NormalizerTest, ReplaceNoMatch) {
  iree_tokenizer_normalizer_t norm;
  IREE_ASSERT_OK(iree_tokenizer_normalizer_initialize_replace(
      IREE_SV("XYZ"), IREE_SV("abc"), &norm));

  // No matches -> passthrough.
  EXPECT_EQ(Apply(&norm, "Hello World"), "Hello World");

  iree_tokenizer_normalizer_deinitialize(&norm);
}

TEST_F(NormalizerTest, ReplaceMultiple) {
  iree_tokenizer_normalizer_t norm;
  IREE_ASSERT_OK(iree_tokenizer_normalizer_initialize_replace(
      IREE_SV("aa"), IREE_SV("b"), &norm));

  // Multiple occurrences.
  EXPECT_EQ(Apply(&norm, "aaaa"), "bb");
  // "xaayaaz" has "aa" at positions 1 and 4 -> "xbybz"
  EXPECT_EQ(Apply(&norm, "xaayaaz"), "xbybz");

  iree_tokenizer_normalizer_deinitialize(&norm);
}

TEST_F(NormalizerTest, ReplaceEmptyPattern) {
  iree_tokenizer_normalizer_t norm;
  IREE_ASSERT_OK(iree_tokenizer_normalizer_initialize_replace(
      IREE_SV(""), IREE_SV("X"), &norm));

  // Empty pattern is passthrough.
  EXPECT_EQ(Apply(&norm, "Hello"), "Hello");

  iree_tokenizer_normalizer_deinitialize(&norm);
}

TEST_F(NormalizerTest, ReplaceEmptyContent) {
  iree_tokenizer_normalizer_t norm;
  IREE_ASSERT_OK(iree_tokenizer_normalizer_initialize_replace(
      IREE_SV(" "), IREE_SV(""), &norm));

  // Delete spaces.
  EXPECT_EQ(Apply(&norm, "Hello World"), "HelloWorld");

  iree_tokenizer_normalizer_deinitialize(&norm);
}

TEST_F(NormalizerTest, ReplaceUnicodePattern) {
  iree_tokenizer_normalizer_t norm;
  IREE_ASSERT_OK(iree_tokenizer_normalizer_initialize_replace(
      IREE_SV("\xE2\x96\x81"), IREE_SV(" "), &norm));  // ▁ -> space

  EXPECT_EQ(Apply(&norm, "\xE2\x96\x81Hello\xE2\x96\x81World"), " Hello World");

  iree_tokenizer_normalizer_deinitialize(&norm);
}

TEST_F(NormalizerTest, ReplaceMaxLength) {
  iree_tokenizer_normalizer_t norm;
  // Exactly 8 bytes for pattern and content should succeed (max is 8).
  IREE_ASSERT_OK(iree_tokenizer_normalizer_initialize_replace(
      IREE_SV("01234567"), IREE_SV("ABCDEFGH"), &norm));
  EXPECT_EQ(Apply(&norm, "X01234567Y"), "XABCDEFGHY");
  iree_tokenizer_normalizer_deinitialize(&norm);
}

TEST_F(NormalizerTest, ReplacePatternTooLong) {
  iree_tokenizer_normalizer_t norm;
  // 9 bytes exceeds max of 8.
  iree_status_t status = iree_tokenizer_normalizer_initialize_replace(
      IREE_SV("012345678"), IREE_SV("X"), &norm);
  EXPECT_TRUE(iree_status_is_invalid_argument(status));
  iree_status_ignore(status);
}

TEST_F(NormalizerTest, ReplaceContentTooLong) {
  iree_tokenizer_normalizer_t norm;
  // 9 bytes exceeds max of 8.
  iree_status_t status = iree_tokenizer_normalizer_initialize_replace(
      IREE_SV("X"), IREE_SV("012345678"), &norm);
  EXPECT_TRUE(iree_status_is_invalid_argument(status));
  iree_status_ignore(status);
}

//===----------------------------------------------------------------------===//
// Replace-Regex Normalizer
//===----------------------------------------------------------------------===//

TEST_F(NormalizerTest, ReplaceRegexWhitespaceCollapse) {
  // The canonical CLIP normalizer use case: collapse whitespace.
  iree_tokenizer_normalizer_t norm;
  ASSERT_TRUE(InitializeReplaceRegex("\\s+", " ", &norm));

  EXPECT_EQ(norm.type, IREE_TOKENIZER_NORMALIZER_REPLACE_REGEX);
  EXPECT_EQ(Apply(&norm, "Hello   World"), "Hello World");
  EXPECT_EQ(Apply(&norm, "a\t\t\tb"), "a b");
  EXPECT_EQ(Apply(&norm, "  leading"), " leading");
  EXPECT_EQ(Apply(&norm, "trailing  "), "trailing ");

  iree_tokenizer_normalizer_deinitialize(&norm);
}

TEST_F(NormalizerTest, ReplaceRegexNoMatch) {
  iree_tokenizer_normalizer_t norm;
  ASSERT_TRUE(InitializeReplaceRegex("[0-9]+", "X", &norm));

  // No digits -> passthrough.
  EXPECT_EQ(Apply(&norm, "Hello World"), "Hello World");
  EXPECT_EQ(Apply(&norm, ""), "");

  iree_tokenizer_normalizer_deinitialize(&norm);
}

TEST_F(NormalizerTest, ReplaceRegexMultipleMatches) {
  iree_tokenizer_normalizer_t norm;
  ASSERT_TRUE(InitializeReplaceRegex("[0-9]+", "#", &norm));

  // Replace all digit sequences.
  EXPECT_EQ(Apply(&norm, "abc123def456ghi"), "abc#def#ghi");
  EXPECT_EQ(Apply(&norm, "123"), "#");
  EXPECT_EQ(Apply(&norm, "a1b2c3"), "a#b#c#");

  iree_tokenizer_normalizer_deinitialize(&norm);
}

TEST_F(NormalizerTest, ReplaceRegexDeleteMatches) {
  // Empty replacement (delete matches).
  iree_tokenizer_normalizer_t norm;
  ASSERT_TRUE(InitializeReplaceRegex("\\s+", "", &norm));

  EXPECT_EQ(Apply(&norm, "Hello World"), "HelloWorld");
  EXPECT_EQ(Apply(&norm, "  a  b  c  "), "abc");

  iree_tokenizer_normalizer_deinitialize(&norm);
}

TEST_F(NormalizerTest, ReplaceRegexUnicodePattern) {
  // Regex matching Unicode whitespace.
  iree_tokenizer_normalizer_t norm;
  ASSERT_TRUE(InitializeReplaceRegex("\\s+", "_", &norm));

  // Tab and newline are whitespace.
  EXPECT_EQ(Apply(&norm, "a\tb\nc"), "a_b_c");

  iree_tokenizer_normalizer_deinitialize(&norm);
}

TEST_F(NormalizerTest, ReplaceRegexAlternation) {
  // Alternation pattern.
  iree_tokenizer_normalizer_t norm;
  ASSERT_TRUE(InitializeReplaceRegex("cat|dog", "pet", &norm));

  EXPECT_EQ(Apply(&norm, "I have a cat and a dog"), "I have a pet and a pet");
  EXPECT_EQ(Apply(&norm, "catdog"), "petpet");

  iree_tokenizer_normalizer_deinitialize(&norm);
}

TEST_F(NormalizerTest, ReplaceRegexCharacterClass) {
  // Character class pattern.
  iree_tokenizer_normalizer_t norm;
  ASSERT_TRUE(InitializeReplaceRegex("[aeiou]", "*", &norm));

  EXPECT_EQ(Apply(&norm, "hello world"), "h*ll* w*rld");

  iree_tokenizer_normalizer_deinitialize(&norm);
}

TEST_F(NormalizerTest, ReplaceRegexLongerReplacement) {
  // Replacement longer than match.
  iree_tokenizer_normalizer_t norm;
  ASSERT_TRUE(InitializeReplaceRegex("a", "AAA", &norm));

  EXPECT_EQ(Apply(&norm, "banana"), "bAAAnAAAnAAA");

  iree_tokenizer_normalizer_deinitialize(&norm);
}

TEST_F(NormalizerTest, ReplaceRegexContentTooLong) {
  // Content > 31 bytes should fail.
  iree_tokenizer_regex_compile_error_t compile_error = {0};
  uint8_t* dfa_data = nullptr;
  iree_host_size_t dfa_size = 0;
  IREE_ASSERT_OK(iree_tokenizer_regex_compile(
      IREE_SV("a"), IREE_TOKENIZER_REGEX_COMPILE_FLAG_NONE,
      iree_allocator_system(), &dfa_data, &dfa_size, &compile_error));

  iree_tokenizer_normalizer_t norm;
  iree_status_t status = iree_tokenizer_normalizer_initialize_replace_regex(
      dfa_data, dfa_size,
      IREE_SV("01234567890123456789012345678901"),  // 32 bytes
      iree_allocator_system(), &norm);
  EXPECT_TRUE(iree_status_is_invalid_argument(status));
  iree_status_ignore(status);
  iree_allocator_free(iree_allocator_system(), dfa_data);
}

//===----------------------------------------------------------------------===//
// NFC Normalizer
//===----------------------------------------------------------------------===//

TEST_F(NormalizerTest, NfcPassthrough) {
  // Already NFC-normalized text should pass through unchanged.
  iree_tokenizer_normalizer_t norm;
  iree_tokenizer_normalizer_initialize_nfc(&norm);

  EXPECT_EQ(Apply(&norm, "Hello World"), "Hello World");
  EXPECT_EQ(Apply(&norm, "café"), "café");  // Precomposed é (U+00E9).
  EXPECT_EQ(Apply(&norm, ""), "");

  iree_tokenizer_normalizer_deinitialize(&norm);
}

TEST_F(NormalizerTest, NfcComposesDecomposed) {
  // NFD (decomposed) -> NFC (composed).
  // e + combining acute (U+0301) -> é (U+00E9).
  iree_tokenizer_normalizer_t norm;
  iree_tokenizer_normalizer_initialize_nfc(&norm);

  // NFD: "cafe" + combining acute = café in decomposed form.
  // UTF-8: e (0x65) + combining acute (0xCC 0x81) -> é (0xC3 0xA9).
  EXPECT_EQ(Apply(&norm, "cafe\xCC\x81"), "caf\xC3\xA9");

  // Multiple decomposed accents.
  // a + combining umlaut (U+0308) -> ä.
  EXPECT_EQ(Apply(&norm, "a\xCC\x88"), "\xC3\xA4");

  iree_tokenizer_normalizer_deinitialize(&norm);
}

TEST_F(NormalizerTest, NfcKoreanHangul) {
  // Korean Hangul passthrough test.
  // Note: Hangul Jamo composition (ㄱ + ㅏ → 가) requires algorithmic handling
  // (LV = 0xAC00 + (L-0x1100)*588 + (V-0x1161)*28) which is not yet
  // implemented. Pre-composed Hangul syllables pass through unchanged.
  iree_tokenizer_normalizer_t norm;
  iree_tokenizer_normalizer_initialize_nfc(&norm);

  // Pre-composed 가 (U+AC00) passes through unchanged.
  EXPECT_EQ(Apply(&norm, "\xEA\xB0\x80"), "\xEA\xB0\x80");

  // Pre-composed 한글 passes through unchanged.
  EXPECT_EQ(Apply(&norm, "\xED\x95\x9C\xEA\xB8\x80"),
            "\xED\x95\x9C\xEA\xB8\x80");

  iree_tokenizer_normalizer_deinitialize(&norm);
}

TEST_F(NormalizerTest, NfcMixedContent) {
  // Mix of already-composed, decomposed, and ASCII.
  iree_tokenizer_normalizer_t norm;
  iree_tokenizer_normalizer_initialize_nfc(&norm);

  // "Hello café world" with decomposed é.
  std::string input = "Hello cafe\xCC\x81 world";
  std::string expected = "Hello caf\xC3\xA9 world";
  EXPECT_EQ(Apply(&norm, input.c_str()), expected);

  iree_tokenizer_normalizer_deinitialize(&norm);
}

TEST_F(NormalizerTest, NfcCjkUnaffected) {
  // CJK characters should pass through unchanged.
  iree_tokenizer_normalizer_t norm;
  iree_tokenizer_normalizer_initialize_nfc(&norm);

  // Chinese: 你好世界 (Hello World).
  EXPECT_EQ(Apply(&norm, "\xE4\xBD\xA0\xE5\xA5\xBD\xE4\xB8\x96\xE7\x95\x8C"),
            "\xE4\xBD\xA0\xE5\xA5\xBD\xE4\xB8\x96\xE7\x95\x8C");

  // Japanese hiragana: こんにちは (Hello).
  EXPECT_EQ(
      Apply(&norm,
            "\xE3\x81\x93\xE3\x82\x93\xE3\x81\xAB\xE3\x81\xA1\xE3\x81\xAF"),
      "\xE3\x81\x93\xE3\x82\x93\xE3\x81\xAB\xE3\x81\xA1\xE3\x81\xAF");

  iree_tokenizer_normalizer_deinitialize(&norm);
}

TEST_F(NormalizerTest, NfcEmoji) {
  // Emoji should pass through unchanged.
  iree_tokenizer_normalizer_t norm;
  iree_tokenizer_normalizer_initialize_nfc(&norm);

  // 😀 (U+1F600).
  EXPECT_EQ(Apply(&norm, "\xF0\x9F\x98\x80"), "\xF0\x9F\x98\x80");

  // Emoji with skin tone modifier (already composed).
  // 👍🏽 = 👍 (U+1F44D) + medium skin tone (U+1F3FD).
  EXPECT_EQ(Apply(&norm, "\xF0\x9F\x91\x8D\xF0\x9F\x8F\xBD"),
            "\xF0\x9F\x91\x8D\xF0\x9F\x8F\xBD");

  iree_tokenizer_normalizer_deinitialize(&norm);
}

//===----------------------------------------------------------------------===//
// Strip Normalizer
//===----------------------------------------------------------------------===//

TEST_F(NormalizerTest, StripBasic) {
  iree_tokenizer_normalizer_t norm;
  iree_tokenizer_normalizer_initialize_strip(true, true, &norm);

  EXPECT_EQ(norm.type, IREE_TOKENIZER_NORMALIZER_STRIP);
  EXPECT_EQ(Apply(&norm, "  hello world  "), "hello world");
  EXPECT_EQ(Apply(&norm, "\t\nhello\t\n"), "hello");
  EXPECT_EQ(Apply(&norm, "no whitespace"), "no whitespace");
  EXPECT_EQ(Apply(&norm, ""), "");

  iree_tokenizer_normalizer_deinitialize(&norm);
}

TEST_F(NormalizerTest, StripLeftOnly) {
  iree_tokenizer_normalizer_t norm;
  iree_tokenizer_normalizer_initialize_strip(true, false, &norm);

  EXPECT_EQ(Apply(&norm, "  hello world  "), "hello world  ");
  EXPECT_EQ(Apply(&norm, "\t\nhello"), "hello");
  EXPECT_EQ(Apply(&norm, "no leading"), "no leading");

  iree_tokenizer_normalizer_deinitialize(&norm);
}

TEST_F(NormalizerTest, StripRightOnly) {
  iree_tokenizer_normalizer_t norm;
  iree_tokenizer_normalizer_initialize_strip(false, true, &norm);

  EXPECT_EQ(Apply(&norm, "  hello world  "), "  hello world");
  EXPECT_EQ(Apply(&norm, "hello\t\n"), "hello");
  EXPECT_EQ(Apply(&norm, "no trailing"), "no trailing");

  iree_tokenizer_normalizer_deinitialize(&norm);
}

TEST_F(NormalizerTest, StripBothFalse) {
  iree_tokenizer_normalizer_t norm;
  iree_tokenizer_normalizer_initialize_strip(false, false, &norm);

  // Both false = passthrough.
  EXPECT_EQ(Apply(&norm, "  hello world  "), "  hello world  ");
  EXPECT_EQ(Apply(&norm, "\t\nhello\t\n"), "\t\nhello\t\n");

  iree_tokenizer_normalizer_deinitialize(&norm);
}

TEST_F(NormalizerTest, StripUnicodeWhitespace) {
  iree_tokenizer_normalizer_t norm;
  iree_tokenizer_normalizer_initialize_strip(true, true, &norm);

  // Non-breaking space (U+00A0) - UTF-8: C2 A0.
  EXPECT_EQ(Apply(&norm, "\xC2\xA0hello\xC2\xA0"), "hello");

  // Ideographic space (U+3000) - UTF-8: E3 80 80.
  EXPECT_EQ(Apply(&norm, "\xE3\x80\x80hello\xE3\x80\x80"), "hello");

  // Mixed ASCII and Unicode whitespace.
  EXPECT_EQ(Apply(&norm, " \xC2\xA0\t hello \t\xC2\xA0 "), "hello");

  iree_tokenizer_normalizer_deinitialize(&norm);
}

TEST_F(NormalizerTest, StripAllWhitespace) {
  iree_tokenizer_normalizer_t norm;
  iree_tokenizer_normalizer_initialize_strip(true, true, &norm);

  // Input is only whitespace -> empty result.
  EXPECT_EQ(Apply(&norm, "   "), "");
  EXPECT_EQ(Apply(&norm, "\t\n\r"), "");

  iree_tokenizer_normalizer_deinitialize(&norm);
}

//===----------------------------------------------------------------------===//
// Sequence Normalizer
//===----------------------------------------------------------------------===//

TEST_F(NormalizerTest, SequenceEmpty) {
  iree_tokenizer_normalizer_t norm;
  IREE_ASSERT_OK(iree_tokenizer_normalizer_initialize_sequence(
      NULL, 0, iree_allocator_system(), &norm));

  // Empty sequence is passthrough.
  EXPECT_EQ(norm.type, IREE_TOKENIZER_NORMALIZER_NONE);
  EXPECT_EQ(Apply(&norm, "Hello"), "Hello");

  iree_tokenizer_normalizer_deinitialize(&norm);
}

TEST_F(NormalizerTest, SequenceChain) {
  // Chain: strip accents -> lowercase.
  iree_tokenizer_normalizer_t children[2];
  iree_tokenizer_normalizer_initialize_strip_accents(&children[0]);
  iree_tokenizer_normalizer_initialize_lowercase(&children[1]);

  iree_tokenizer_normalizer_t norm;
  IREE_ASSERT_OK(iree_tokenizer_normalizer_initialize_sequence(
      children, 2, iree_allocator_system(), &norm));

  // Sources are zeroed after move (ownership transferred).
  EXPECT_EQ(children[0].type, IREE_TOKENIZER_NORMALIZER_NONE);
  EXPECT_EQ(children[1].type, IREE_TOKENIZER_NORMALIZER_NONE);

  EXPECT_EQ(Apply(&norm, "CAFÉ"), "cafe");

  iree_tokenizer_normalizer_deinitialize(&norm);
}

TEST_F(NormalizerTest, SequenceTinyLlamaStyle) {
  // TinyLlama-style: Prepend ▁, then Replace space with ▁.
  iree_tokenizer_normalizer_t children[2];
  IREE_ASSERT_OK(iree_tokenizer_normalizer_initialize_prepend(
      IREE_SV("\xE2\x96\x81"), &children[0]));
  IREE_ASSERT_OK(iree_tokenizer_normalizer_initialize_replace(
      IREE_SV(" "), IREE_SV("\xE2\x96\x81"), &children[1]));

  iree_tokenizer_normalizer_t norm;
  IREE_ASSERT_OK(iree_tokenizer_normalizer_initialize_sequence(
      children, 2, iree_allocator_system(), &norm));

  // "Hello World" -> "▁Hello World" -> "▁Hello▁World"
  EXPECT_EQ(Apply(&norm, "Hello World"), "\xE2\x96\x81Hello\xE2\x96\x81World");

  iree_tokenizer_normalizer_deinitialize(&norm);
}

}  // namespace
