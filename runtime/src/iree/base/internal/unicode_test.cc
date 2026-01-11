// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/unicode.h"

#include <string>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

using iree::Status;
using iree::StatusCode;
using iree::testing::status::StatusIs;

//===----------------------------------------------------------------------===//
// UTF-8 Codec Tests
//===----------------------------------------------------------------------===//

TEST(UnicodeUtf8Test, DecodeAscii) {
  iree_string_view_t text = iree_make_cstring_view("Hello");
  iree_host_size_t position = 0;

  EXPECT_EQ(iree_unicode_utf8_decode(text, &position), 'H');
  EXPECT_EQ(position, 1u);
  EXPECT_EQ(iree_unicode_utf8_decode(text, &position), 'e');
  EXPECT_EQ(position, 2u);
  EXPECT_EQ(iree_unicode_utf8_decode(text, &position), 'l');
  EXPECT_EQ(position, 3u);
  EXPECT_EQ(iree_unicode_utf8_decode(text, &position), 'l');
  EXPECT_EQ(position, 4u);
  EXPECT_EQ(iree_unicode_utf8_decode(text, &position), 'o');
  EXPECT_EQ(position, 5u);
}

TEST(UnicodeUtf8Test, DecodeTwoByteSequence) {
  // "é" is U+00E9, encoded as 0xC3 0xA9
  const char data[] = "\xC3\xA9";
  iree_string_view_t text = iree_make_string_view(data, 2);
  iree_host_size_t position = 0;

  EXPECT_EQ(iree_unicode_utf8_decode(text, &position), 0x00E9u);
  EXPECT_EQ(position, 2u);
}

TEST(UnicodeUtf8Test, DecodeThreeByteSequence) {
  // "中" is U+4E2D, encoded as 0xE4 0xB8 0xAD
  const char data[] = "\xE4\xB8\xAD";
  iree_string_view_t text = iree_make_string_view(data, 3);
  iree_host_size_t position = 0;

  EXPECT_EQ(iree_unicode_utf8_decode(text, &position), 0x4E2Du);
  EXPECT_EQ(position, 3u);
}

TEST(UnicodeUtf8Test, DecodeFourByteSequence) {
  // U+1F600 (grinning face emoji), encoded as 0xF0 0x9F 0x98 0x80
  const char data[] = "\xF0\x9F\x98\x80";
  iree_string_view_t text = iree_make_string_view(data, 4);
  iree_host_size_t position = 0;

  EXPECT_EQ(iree_unicode_utf8_decode(text, &position), 0x1F600u);
  EXPECT_EQ(position, 4u);
}

TEST(UnicodeUtf8Test, DecodeInvalidSequence) {
  // Invalid continuation byte.
  const char data[] = "\xC3\x00";
  iree_string_view_t text = iree_make_string_view(data, 2);
  iree_host_size_t position = 0;

  // Should return replacement character and advance by 1.
  EXPECT_EQ(iree_unicode_utf8_decode(text, &position),
            IREE_UNICODE_REPLACEMENT_CHAR);
  EXPECT_EQ(position, 1u);
}

TEST(UnicodeUtf8Test, DecodeOverlongSequence) {
  // Overlong encoding of ASCII 'A' (0x41) as 2 bytes: 0xC1 0x81
  const char data[] = "\xC1\x81";
  iree_string_view_t text = iree_make_string_view(data, 2);
  iree_host_size_t position = 0;

  EXPECT_EQ(iree_unicode_utf8_decode(text, &position),
            IREE_UNICODE_REPLACEMENT_CHAR);
}

TEST(UnicodeUtf8Test, DecodeSurrogateHalf) {
  // U+D800 (surrogate) encoded as 0xED 0xA0 0x80 - invalid in UTF-8.
  const char data[] = "\xED\xA0\x80";
  iree_string_view_t text = iree_make_string_view(data, 3);
  iree_host_size_t position = 0;

  EXPECT_EQ(iree_unicode_utf8_decode(text, &position),
            IREE_UNICODE_REPLACEMENT_CHAR);
}

TEST(UnicodeUtf8Test, EncodeAscii) {
  char buffer[4];
  EXPECT_EQ(iree_unicode_utf8_encode('A', buffer), 1);
  EXPECT_EQ(buffer[0], 'A');
}

TEST(UnicodeUtf8Test, EncodeTwoByteSequence) {
  char buffer[4];
  EXPECT_EQ(iree_unicode_utf8_encode(0x00E9, buffer), 2);
  EXPECT_EQ((uint8_t)buffer[0], 0xC3);
  EXPECT_EQ((uint8_t)buffer[1], 0xA9);
}

TEST(UnicodeUtf8Test, EncodeThreeByteSequence) {
  char buffer[4];
  EXPECT_EQ(iree_unicode_utf8_encode(0x4E2D, buffer), 3);
  EXPECT_EQ((uint8_t)buffer[0], 0xE4);
  EXPECT_EQ((uint8_t)buffer[1], 0xB8);
  EXPECT_EQ((uint8_t)buffer[2], 0xAD);
}

TEST(UnicodeUtf8Test, EncodeFourByteSequence) {
  char buffer[4];
  EXPECT_EQ(iree_unicode_utf8_encode(0x1F600, buffer), 4);
  EXPECT_EQ((uint8_t)buffer[0], 0xF0);
  EXPECT_EQ((uint8_t)buffer[1], 0x9F);
  EXPECT_EQ((uint8_t)buffer[2], 0x98);
  EXPECT_EQ((uint8_t)buffer[3], 0x80);
}

TEST(UnicodeUtf8Test, EncodeInvalidCodepoint) {
  char buffer[4];
  // Beyond max codepoint.
  EXPECT_EQ(iree_unicode_utf8_encode(0x110000, buffer), 0);
  // Surrogate half.
  EXPECT_EQ(iree_unicode_utf8_encode(0xD800, buffer), 0);
}

TEST(UnicodeUtf8Test, RoundTrip) {
  std::vector<uint32_t> codepoints = {'A', 0x00E9, 0x4E2D, 0x1F600};
  for (uint32_t codepoint : codepoints) {
    char buffer[4];
    int length = iree_unicode_utf8_encode(codepoint, buffer);
    ASSERT_GT(length, 0);

    iree_string_view_t text = iree_make_string_view(buffer, length);
    iree_host_size_t position = 0;
    EXPECT_EQ(iree_unicode_utf8_decode(text, &position), codepoint);
    EXPECT_EQ(position, (iree_host_size_t)length);
  }
}

TEST(UnicodeUtf8Test, CodepointCount) {
  // ASCII only.
  EXPECT_EQ(iree_unicode_utf8_codepoint_count(iree_make_cstring_view("Hello")),
            5u);

  // Mixed: "Héllo中" - H, é (2 bytes), l, l, o, 中 (3 bytes)
  const char mixed[] = "H\xC3\xA9llo\xE4\xB8\xAD";
  EXPECT_EQ(iree_unicode_utf8_codepoint_count(
                iree_make_string_view(mixed, sizeof(mixed) - 1)),
            6u);
}

TEST(UnicodeUtf8Test, Validate) {
  EXPECT_TRUE(iree_unicode_utf8_validate(iree_make_cstring_view("Hello")));
  EXPECT_TRUE(iree_unicode_utf8_validate(iree_make_cstring_view("")));

  // Valid UTF-8 with multi-byte sequences.
  const char valid[] = "\xC3\xA9\xE4\xB8\xAD\xF0\x9F\x98\x80";
  EXPECT_TRUE(iree_unicode_utf8_validate(
      iree_make_string_view(valid, sizeof(valid) - 1)));

  // Invalid: truncated sequence.
  const char invalid[] = "\xC3";
  EXPECT_FALSE(iree_unicode_utf8_validate(
      iree_make_string_view(invalid, sizeof(invalid) - 1)));
}

TEST(UnicodeUtf8Test, EncodedLength) {
  EXPECT_EQ(iree_unicode_utf8_encoded_length('A'), 1);
  EXPECT_EQ(iree_unicode_utf8_encoded_length(0x00E9), 2);
  EXPECT_EQ(iree_unicode_utf8_encoded_length(0x4E2D), 3);
  EXPECT_EQ(iree_unicode_utf8_encoded_length(0x1F600), 4);
  EXPECT_EQ(iree_unicode_utf8_encoded_length(0x110000), 0);  // Invalid
}

TEST(UnicodeUtf8Test, IncompleteTailLengthEmpty) {
  // Empty buffer has no incomplete tail.
  EXPECT_EQ(iree_unicode_utf8_incomplete_tail_length("", 0), 0u);
}

TEST(UnicodeUtf8Test, IncompleteTailLengthCompleteAscii) {
  // Complete ASCII strings have no incomplete tail.
  EXPECT_EQ(iree_unicode_utf8_incomplete_tail_length("Hello", 5), 0u);
  EXPECT_EQ(iree_unicode_utf8_incomplete_tail_length("x", 1), 0u);
}

TEST(UnicodeUtf8Test, IncompleteTailLengthCompleteTwoByte) {
  // Complete 2-byte UTF-8 sequence "é" (U+00E9 = 0xC3 0xA9).
  const char complete[] = "\xC3\xA9";
  EXPECT_EQ(iree_unicode_utf8_incomplete_tail_length(complete, 2), 0u);
}

TEST(UnicodeUtf8Test, IncompleteTailLengthCompleteThreeByte) {
  // Complete 3-byte UTF-8 sequence "中" (U+4E2D = 0xE4 0xB8 0xAD).
  const char complete[] = "\xE4\xB8\xAD";
  EXPECT_EQ(iree_unicode_utf8_incomplete_tail_length(complete, 3), 0u);
}

TEST(UnicodeUtf8Test, IncompleteTailLengthCompleteFourByte) {
  // Complete 4-byte UTF-8 sequence U+1F600 (grinning face emoji).
  const char complete[] = "\xF0\x9F\x98\x80";
  EXPECT_EQ(iree_unicode_utf8_incomplete_tail_length(complete, 4), 0u);
}

TEST(UnicodeUtf8Test, IncompleteTailLengthIncompleteTwoByte) {
  // Incomplete 2-byte sequence: only lead byte present (0xC3).
  const char incomplete[] = "abc\xC3";
  EXPECT_EQ(iree_unicode_utf8_incomplete_tail_length(incomplete, 4), 1u);
}

TEST(UnicodeUtf8Test, IncompleteTailLengthIncompleteThreeByte) {
  // Incomplete 3-byte sequence: lead + 1 continuation (0xE4 0xB8).
  const char incomplete1[] = "abc\xE4\xB8";
  EXPECT_EQ(iree_unicode_utf8_incomplete_tail_length(incomplete1, 5), 2u);

  // Incomplete 3-byte sequence: only lead byte (0xE4).
  const char incomplete2[] = "abc\xE4";
  EXPECT_EQ(iree_unicode_utf8_incomplete_tail_length(incomplete2, 4), 1u);
}

TEST(UnicodeUtf8Test, IncompleteTailLengthIncompleteFourByte) {
  // Incomplete 4-byte sequence: lead + 2 continuations (0xF0 0x9F 0x98).
  const char incomplete1[] = "abc\xF0\x9F\x98";
  EXPECT_EQ(iree_unicode_utf8_incomplete_tail_length(incomplete1, 6), 3u);

  // Incomplete 4-byte sequence: lead + 1 continuation (0xF0 0x9F).
  const char incomplete2[] = "abc\xF0\x9F";
  EXPECT_EQ(iree_unicode_utf8_incomplete_tail_length(incomplete2, 5), 2u);

  // Incomplete 4-byte sequence: only lead byte (0xF0).
  const char incomplete3[] = "abc\xF0";
  EXPECT_EQ(iree_unicode_utf8_incomplete_tail_length(incomplete3, 4), 1u);
}

TEST(UnicodeUtf8Test, IncompleteTailLengthOrphanContinuations) {
  // Four continuation bytes with no lead byte - malformed, treated as complete.
  const char orphans[] = "\x80\x80\x80\x80";
  EXPECT_EQ(iree_unicode_utf8_incomplete_tail_length(orphans, 4), 0u);
}

TEST(UnicodeUtf8Test, IncompleteTailLengthInvalidLeadByte) {
  // Invalid lead byte 0xFF followed by nothing - treated as complete.
  const char invalid[] = "abc\xFF";
  EXPECT_EQ(iree_unicode_utf8_incomplete_tail_length(invalid, 4), 0u);
}

//===----------------------------------------------------------------------===//
// Category Tests
//===----------------------------------------------------------------------===//

TEST(UnicodeCategoryTest, AsciiLetters) {
  for (uint32_t codepoint = 'A'; codepoint <= 'Z'; ++codepoint) {
    EXPECT_TRUE(iree_unicode_is_letter(codepoint))
        << "U+" << std::hex << codepoint;
  }
  for (uint32_t codepoint = 'a'; codepoint <= 'z'; ++codepoint) {
    EXPECT_TRUE(iree_unicode_is_letter(codepoint))
        << "U+" << std::hex << codepoint;
  }
}

TEST(UnicodeCategoryTest, AsciiNumbers) {
  for (uint32_t codepoint = '0'; codepoint <= '9'; ++codepoint) {
    EXPECT_TRUE(iree_unicode_is_number(codepoint))
        << "U+" << std::hex << codepoint;
  }
}

TEST(UnicodeCategoryTest, AsciiPunctuation) {
  EXPECT_TRUE(iree_unicode_is_punctuation('.'));
  EXPECT_TRUE(iree_unicode_is_punctuation(','));
  EXPECT_TRUE(iree_unicode_is_punctuation('!'));
  EXPECT_TRUE(iree_unicode_is_punctuation('?'));
  EXPECT_TRUE(iree_unicode_is_punctuation(';'));
  EXPECT_TRUE(iree_unicode_is_punctuation(':'));
  EXPECT_TRUE(iree_unicode_is_punctuation('\''));
  EXPECT_TRUE(iree_unicode_is_punctuation('"'));
  EXPECT_TRUE(iree_unicode_is_punctuation('('));
  EXPECT_TRUE(iree_unicode_is_punctuation(')'));
  EXPECT_TRUE(iree_unicode_is_punctuation('['));
  EXPECT_TRUE(iree_unicode_is_punctuation(']'));
  EXPECT_TRUE(iree_unicode_is_punctuation('{'));
  EXPECT_TRUE(iree_unicode_is_punctuation('}'));
}

TEST(UnicodeCategoryTest, AsciiSymbols) {
  EXPECT_TRUE(iree_unicode_is_symbol('$'));
  EXPECT_TRUE(iree_unicode_is_symbol('+'));
  EXPECT_TRUE(iree_unicode_is_symbol('<'));
  EXPECT_TRUE(iree_unicode_is_symbol('='));
  EXPECT_TRUE(iree_unicode_is_symbol('>'));
  EXPECT_TRUE(iree_unicode_is_symbol('|'));
  EXPECT_TRUE(iree_unicode_is_symbol('~'));
}

TEST(UnicodeCategoryTest, AsciiControl) {
  EXPECT_TRUE(iree_unicode_is_control(0x00));  // NUL
  EXPECT_TRUE(iree_unicode_is_control(0x1F));  // US
  EXPECT_TRUE(iree_unicode_is_control(0x7F));  // DEL
  EXPECT_FALSE(iree_unicode_is_control(' '));
  EXPECT_FALSE(iree_unicode_is_control('A'));
}

TEST(UnicodeCategoryTest, AsciiWhitespace) {
  EXPECT_TRUE(iree_unicode_is_whitespace(' '));
  EXPECT_TRUE(iree_unicode_is_whitespace('\t'));
  EXPECT_TRUE(iree_unicode_is_whitespace('\n'));
  EXPECT_TRUE(iree_unicode_is_whitespace('\r'));
  EXPECT_TRUE(iree_unicode_is_whitespace('\f'));
  EXPECT_TRUE(iree_unicode_is_whitespace('\v'));
  EXPECT_FALSE(iree_unicode_is_whitespace('A'));
  EXPECT_FALSE(iree_unicode_is_whitespace('0'));
}

TEST(UnicodeCategoryTest, NonAsciiWhitespace) {
  EXPECT_TRUE(iree_unicode_is_whitespace(0x00A0));  // No-Break Space
  EXPECT_TRUE(iree_unicode_is_whitespace(0x2000));  // En Quad
  EXPECT_TRUE(iree_unicode_is_whitespace(0x2003));  // Em Space
  EXPECT_TRUE(iree_unicode_is_whitespace(0x3000));  // Ideographic Space
}

TEST(UnicodeCategoryTest, CjkLetters) {
  EXPECT_TRUE(iree_unicode_is_letter(0x4E00));  // 一 (first CJK)
  EXPECT_TRUE(iree_unicode_is_letter(0x4E2D));  // 中
  EXPECT_TRUE(iree_unicode_is_letter(0x9FFF));  // Last CJK in basic block
}

TEST(UnicodeCategoryTest, GreekLetters) {
  EXPECT_TRUE(iree_unicode_is_letter(0x0391));  // Α (Alpha)
  EXPECT_TRUE(iree_unicode_is_letter(0x03B1));  // α (alpha)
  EXPECT_TRUE(iree_unicode_is_letter(0x03A9));  // Ω (Omega)
  EXPECT_TRUE(iree_unicode_is_letter(0x03C9));  // ω (omega)
}

TEST(UnicodeCategoryTest, CombiningMarks) {
  EXPECT_TRUE(iree_unicode_is_mark(0x0300));  // Combining Grave Accent
  EXPECT_TRUE(iree_unicode_is_mark(0x0301));  // Combining Acute Accent
  EXPECT_TRUE(iree_unicode_is_mark(0x0302));  // Combining Circumflex Accent
}

TEST(UnicodeCategoryTest, C1ControlCharacters) {
  // C1 controls (0x80-0x9F) should be detected as control characters.
  EXPECT_TRUE(iree_unicode_is_control(0x80));
  EXPECT_TRUE(iree_unicode_is_control(0x85));  // NEL (Next Line)
  EXPECT_TRUE(iree_unicode_is_control(0x9F));
  EXPECT_FALSE(iree_unicode_is_control(0xA0));  // NBSP is not a control
}

TEST(UnicodeCategoryTest, CJKCharacters) {
  // CJK Unified Ideographs (U+4E00-U+9FFF).
  EXPECT_TRUE(iree_unicode_is_cjk(0x4E00));  // First CJK Unified Ideograph
  EXPECT_TRUE(iree_unicode_is_cjk(0x4E2D));  // 中 (middle)
  EXPECT_TRUE(iree_unicode_is_cjk(0x6587));  // 文 (text)
  EXPECT_TRUE(iree_unicode_is_cjk(0x9FFF));  // Last CJK Unified Ideograph

  // CJK Extension A (U+3400-U+4DBF).
  EXPECT_TRUE(iree_unicode_is_cjk(0x3400));
  EXPECT_TRUE(iree_unicode_is_cjk(0x4DBF));

  // CJK Extension B (U+20000-U+2A6DF).
  EXPECT_TRUE(iree_unicode_is_cjk(0x20000));
  EXPECT_TRUE(iree_unicode_is_cjk(0x2A6DF));

  // CJK Compatibility Ideographs (U+F900-U+FAFF).
  EXPECT_TRUE(iree_unicode_is_cjk(0xF900));
  EXPECT_TRUE(iree_unicode_is_cjk(0xFAFF));

  // Non-CJK characters.
  EXPECT_FALSE(iree_unicode_is_cjk('A'));     // ASCII
  EXPECT_FALSE(iree_unicode_is_cjk(0x3041));  // Hiragana あ
  EXPECT_FALSE(iree_unicode_is_cjk(0x30A1));  // Katakana ア
  EXPECT_FALSE(iree_unicode_is_cjk(0xAC00));  // Hangul 가
}

TEST(UnicodeCategoryTest, NonAsciiPunctuation) {
  EXPECT_TRUE(iree_unicode_is_punctuation(0x00A1));  // ¡ Inverted Exclamation
  EXPECT_TRUE(iree_unicode_is_punctuation(0x00BF));  // ¿ Inverted Question
  EXPECT_TRUE(iree_unicode_is_punctuation(0x2010));  // ‐ Hyphen
  EXPECT_TRUE(iree_unicode_is_punctuation(0x2014));  // — Em Dash
  EXPECT_TRUE(iree_unicode_is_punctuation(0x2018));  // ' Left Single Quote
  EXPECT_TRUE(iree_unicode_is_punctuation(0x201C));  // " Left Double Quote
}

TEST(UnicodeCategoryTest, NonAsciiSymbols) {
  EXPECT_TRUE(iree_unicode_is_symbol(0x00A2));  // ¢ Cent Sign
  EXPECT_TRUE(iree_unicode_is_symbol(0x00A3));  // £ Pound Sign
  EXPECT_TRUE(iree_unicode_is_symbol(0x00A5));  // ¥ Yen Sign
  EXPECT_TRUE(iree_unicode_is_symbol(0x00A9));  // © Copyright
  EXPECT_TRUE(iree_unicode_is_symbol(0x00AE));  // ® Registered
  EXPECT_TRUE(iree_unicode_is_symbol(0x2122));  // ™ Trade Mark
}

TEST(UnicodeCategoryTest, Separators) {
  EXPECT_TRUE(iree_unicode_is_separator(0x00A0));  // No-Break Space
  EXPECT_TRUE(iree_unicode_is_separator(0x2028));  // Line Separator
  EXPECT_TRUE(iree_unicode_is_separator(0x2029));  // Paragraph Separator
  EXPECT_TRUE(iree_unicode_is_separator(0x202F));  // Narrow No-Break Space
  EXPECT_TRUE(iree_unicode_is_separator(0x3000));  // Ideographic Space
}

TEST(UnicodeCategoryTest, OtherCategory) {
  // Format characters (Cf) are in the Other category.
  EXPECT_TRUE(iree_unicode_is_other(0x200B));  // Zero Width Space
  EXPECT_TRUE(iree_unicode_is_other(0x200C));  // Zero Width Non-Joiner
  EXPECT_TRUE(iree_unicode_is_other(0x200D));  // Zero Width Joiner
  EXPECT_TRUE(iree_unicode_is_other(0xFEFF));  // BOM / ZWNBSP
}

TEST(UnicodeCategoryTest, TableBoundaries) {
  // Test first and last entries in various Unicode blocks to ensure
  // binary search handles boundaries correctly.

  // Latin Extended-A block (0x0100-0x017F) - all letters.
  EXPECT_TRUE(iree_unicode_is_letter(0x0100));  // First
  EXPECT_TRUE(iree_unicode_is_letter(0x017F));  // Last

  // Latin Extended-B block (0x0180-0x024F) - all letters.
  EXPECT_TRUE(iree_unicode_is_letter(0x0180));  // First
  EXPECT_TRUE(iree_unicode_is_letter(0x024F));  // Last

  // Combining Diacritical Marks (0x0300-0x036F) - all marks.
  EXPECT_TRUE(iree_unicode_is_mark(0x0300));  // First
  EXPECT_TRUE(iree_unicode_is_mark(0x036F));  // Last

  // General Punctuation (0x2000-0x206F) - mixed categories.
  EXPECT_TRUE(iree_unicode_is_separator(0x2000));    // En Quad (space)
  EXPECT_TRUE(iree_unicode_is_punctuation(0x2010));  // Hyphen
}

TEST(UnicodeCategoryTest, HighCodepoints) {
  // Emoji are symbols (So category) in the Supplementary Multilingual Plane.
  EXPECT_TRUE(iree_unicode_is_symbol(0x1F600));  // 😀 Grinning Face
  EXPECT_TRUE(iree_unicode_is_symbol(0x1F4A9));  // 💩 Pile of Poo
  EXPECT_TRUE(iree_unicode_is_symbol(0x2764));   // ❤ Heavy Black Heart

  // Mathematical Alphanumeric Symbols (1D400-1D7FF) are letters.
  EXPECT_TRUE(
      iree_unicode_is_letter(0x1D400));  // 𝐀 Mathematical Bold Capital A
  EXPECT_TRUE(iree_unicode_is_letter(0x1D41A));  // 𝐚 Mathematical Bold Small A

  // Musical Symbols (1D100-1D1FF) are symbols.
  EXPECT_TRUE(iree_unicode_is_symbol(0x1D11E));  // 𝄞 Musical Symbol G Clef
}

TEST(UnicodeCategoryTest, UnassignedCodepoints) {
  // Unassigned codepoints (Cn) should return OTHER category.
  // These are gaps in the Unicode assignment that may be filled in future
  // versions.
  EXPECT_TRUE(iree_unicode_is_other(0x0378));    // Unassigned in Greek block
  EXPECT_TRUE(iree_unicode_is_other(0x0530));    // Unassigned in Armenian block
  EXPECT_TRUE(iree_unicode_is_other(0xFFFF));    // Noncharacter (still OTHER)
  EXPECT_TRUE(iree_unicode_is_other(0x10FFFF));  // Max codepoint (unassigned)
}

//===----------------------------------------------------------------------===//
// Case Folding Tests
//===----------------------------------------------------------------------===//

TEST(UnicodeCaseTest, AsciiLowercase) {
  EXPECT_EQ(iree_unicode_to_lower('A'), 'a');
  EXPECT_EQ(iree_unicode_to_lower('Z'), 'z');
  EXPECT_EQ(iree_unicode_to_lower('a'), 'a');  // Already lowercase
  EXPECT_EQ(iree_unicode_to_lower('0'), '0');  // Not a letter
}

TEST(UnicodeCaseTest, AsciiUppercase) {
  EXPECT_EQ(iree_unicode_to_upper('a'), 'A');
  EXPECT_EQ(iree_unicode_to_upper('z'), 'Z');
  EXPECT_EQ(iree_unicode_to_upper('A'), 'A');  // Already uppercase
  EXPECT_EQ(iree_unicode_to_upper('0'), '0');  // Not a letter
}

TEST(UnicodeCaseTest, Latin1Lowercase) {
  EXPECT_EQ(iree_unicode_to_lower(0x00C0), 0x00E0u);  // À → à
  EXPECT_EQ(iree_unicode_to_lower(0x00C9), 0x00E9u);  // É → é
  EXPECT_EQ(iree_unicode_to_lower(0x00D6), 0x00F6u);  // Ö → ö
}

TEST(UnicodeCaseTest, Latin1Uppercase) {
  EXPECT_EQ(iree_unicode_to_upper(0x00E0), 0x00C0u);  // à → À
  EXPECT_EQ(iree_unicode_to_upper(0x00E9), 0x00C9u);  // é → É
  EXPECT_EQ(iree_unicode_to_upper(0x00F6), 0x00D6u);  // ö → Ö
}

TEST(UnicodeCaseTest, Greek) {
  EXPECT_EQ(iree_unicode_to_lower(0x0391), 0x03B1u);  // Α → α
  EXPECT_EQ(iree_unicode_to_lower(0x03A9), 0x03C9u);  // Ω → ω
  EXPECT_EQ(iree_unicode_to_upper(0x03B1), 0x0391u);  // α → Α
  EXPECT_EQ(iree_unicode_to_upper(0x03C9), 0x03A9u);  // ω → Ω
}

TEST(UnicodeCaseTest, Cyrillic) {
  EXPECT_EQ(iree_unicode_to_lower(0x0410), 0x0430u);  // А → а
  EXPECT_EQ(iree_unicode_to_lower(0x042F), 0x044Fu);  // Я → я
  EXPECT_EQ(iree_unicode_to_upper(0x0430), 0x0410u);  // а → А
  EXPECT_EQ(iree_unicode_to_upper(0x044F), 0x042Fu);  // я → Я
}

TEST(UnicodeCaseTest, NoMapping) {
  // Characters without case mappings should return unchanged.
  EXPECT_EQ(iree_unicode_to_lower(0x4E2D), 0x4E2Du);  // 中 (CJK)
  EXPECT_EQ(iree_unicode_to_upper(0x4E2D), 0x4E2Du);
  EXPECT_EQ(iree_unicode_to_lower(0x3042), 0x3042u);  // あ (Hiragana)
  EXPECT_EQ(iree_unicode_to_upper(0x3042), 0x3042u);
}

//===----------------------------------------------------------------------===//
// NFD Base Character Tests
//===----------------------------------------------------------------------===//

TEST(UnicodeNfdTest, AsciiUnchanged) {
  EXPECT_EQ(iree_unicode_nfd_base('A'), 'A');
  EXPECT_EQ(iree_unicode_nfd_base('a'), 'a');
  EXPECT_EQ(iree_unicode_nfd_base('0'), '0');
}

TEST(UnicodeNfdTest, Latin1AccentStripping) {
  EXPECT_EQ(iree_unicode_nfd_base(0x00C0), 0x0041u);  // À → A
  EXPECT_EQ(iree_unicode_nfd_base(0x00C1), 0x0041u);  // Á → A
  EXPECT_EQ(iree_unicode_nfd_base(0x00C2), 0x0041u);  // Â → A
  EXPECT_EQ(iree_unicode_nfd_base(0x00C4), 0x0041u);  // Ä → A
  EXPECT_EQ(iree_unicode_nfd_base(0x00E0), 0x0061u);  // à → a
  EXPECT_EQ(iree_unicode_nfd_base(0x00E9), 0x0065u);  // é → e
  EXPECT_EQ(iree_unicode_nfd_base(0x00F1), 0x006Eu);  // ñ → n
  EXPECT_EQ(iree_unicode_nfd_base(0x00FC), 0x0075u);  // ü → u
}

TEST(UnicodeNfdTest, LatinExtendedA) {
  EXPECT_EQ(iree_unicode_nfd_base(0x0100), 0x0041u);  // Ā → A
  EXPECT_EQ(iree_unicode_nfd_base(0x0101), 0x0061u);  // ā → a
  EXPECT_EQ(iree_unicode_nfd_base(0x010C), 0x0043u);  // Č → C
  EXPECT_EQ(iree_unicode_nfd_base(0x010D), 0x0063u);  // č → c
  EXPECT_EQ(iree_unicode_nfd_base(0x0160), 0x0053u);  // Š → S
  EXPECT_EQ(iree_unicode_nfd_base(0x0161), 0x0073u);  // š → s
}

TEST(UnicodeNfdTest, NonDecomposable) {
  // Characters without NFD decomposition should return unchanged.
  EXPECT_EQ(iree_unicode_nfd_base(0x00C6), 0x00C6u);  // Æ (ligature, no decomp)
  EXPECT_EQ(iree_unicode_nfd_base(0x00D0), 0x00D0u);  // Ð (Eth)
  EXPECT_EQ(iree_unicode_nfd_base(0x00D8), 0x00D8u);  // Ø (O with stroke)
  EXPECT_EQ(iree_unicode_nfd_base(0x00DE), 0x00DEu);  // Þ (Thorn)
}

//===----------------------------------------------------------------------===//
// Canonical Combining Class (CCC) Tests
//===----------------------------------------------------------------------===//

TEST(UnicodeCccTest, BaseCharactersHaveZeroCcc) {
  // Base characters (starters) have CCC=0.
  EXPECT_EQ(iree_unicode_ccc('A'), 0);
  EXPECT_EQ(iree_unicode_ccc('a'), 0);
  EXPECT_EQ(iree_unicode_ccc('0'), 0);
  EXPECT_EQ(iree_unicode_ccc(0x00E9), 0);  // é (precomposed, is a starter)
  EXPECT_EQ(iree_unicode_ccc(0x4E2D), 0);  // 中 (CJK)
}

TEST(UnicodeCccTest, CombiningDiacriticalMarks) {
  // Combining marks have non-zero CCC values.
  EXPECT_EQ(iree_unicode_ccc(0x0300), 230);  // Combining Grave Accent
  EXPECT_EQ(iree_unicode_ccc(0x0301), 230);  // Combining Acute Accent
  EXPECT_EQ(iree_unicode_ccc(0x0302), 230);  // Combining Circumflex Accent
  EXPECT_EQ(iree_unicode_ccc(0x0303), 230);  // Combining Tilde
  EXPECT_EQ(iree_unicode_ccc(0x0304), 230);  // Combining Macron
  EXPECT_EQ(iree_unicode_ccc(0x0308), 230);  // Combining Diaeresis
  EXPECT_EQ(iree_unicode_ccc(0x030A), 230);  // Combining Ring Above
  EXPECT_EQ(iree_unicode_ccc(0x030C), 230);  // Combining Caron
}

TEST(UnicodeCccTest, BelowMarks) {
  // Marks that attach below have lower CCC values.
  EXPECT_EQ(iree_unicode_ccc(0x0327), 202);  // Combining Cedilla
  EXPECT_EQ(iree_unicode_ccc(0x0328), 202);  // Combining Ogonek
}

TEST(UnicodeCccTest, DoubleMarks) {
  // Double diacritics (spanning two base characters).
  EXPECT_EQ(iree_unicode_ccc(0x035C), 233);  // Combining Double Breve Below
  EXPECT_EQ(iree_unicode_ccc(0x035D), 234);  // Combining Double Breve
  EXPECT_EQ(iree_unicode_ccc(0x035E), 234);  // Combining Double Macron
}

TEST(UnicodeCccTest, HebrewCombining) {
  // Hebrew combining marks have specific CCC values.
  EXPECT_EQ(iree_unicode_ccc(0x05B0), 10);  // Hebrew Point Sheva
  EXPECT_EQ(iree_unicode_ccc(0x05B1), 11);  // Hebrew Point Hataf Segol
  EXPECT_EQ(iree_unicode_ccc(0x05BC), 21);  // Hebrew Point Dagesh
  EXPECT_EQ(iree_unicode_ccc(0x05C1), 24);  // Hebrew Point Shin Dot
}

//===----------------------------------------------------------------------===//
// NFC Composition Pair Tests
//===----------------------------------------------------------------------===//

TEST(UnicodeNfcComposeTest, LatinCompositions) {
  // Common Latin letter + combining mark compositions.
  EXPECT_EQ(iree_unicode_nfc_compose('A', 0x0300), 0x00C0u);  // A + grave → À
  EXPECT_EQ(iree_unicode_nfc_compose('A', 0x0301), 0x00C1u);  // A + acute → Á
  EXPECT_EQ(iree_unicode_nfc_compose('A', 0x0302),
            0x00C2u);  // A + circumflex → Â
  EXPECT_EQ(iree_unicode_nfc_compose('A', 0x0303), 0x00C3u);  // A + tilde → Ã
  EXPECT_EQ(iree_unicode_nfc_compose('A', 0x0308),
            0x00C4u);  // A + diaeresis → Ä
  EXPECT_EQ(iree_unicode_nfc_compose('A', 0x030A),
            0x00C5u);  // A + ring above → Å
  EXPECT_EQ(iree_unicode_nfc_compose('C', 0x0327), 0x00C7u);  // C + cedilla → Ç
  EXPECT_EQ(iree_unicode_nfc_compose('E', 0x0301), 0x00C9u);  // E + acute → É
  EXPECT_EQ(iree_unicode_nfc_compose('N', 0x0303), 0x00D1u);  // N + tilde → Ñ
  EXPECT_EQ(iree_unicode_nfc_compose('O', 0x0308),
            0x00D6u);  // O + diaeresis → Ö
  EXPECT_EQ(iree_unicode_nfc_compose('U', 0x0308),
            0x00DCu);  // U + diaeresis → Ü
}

TEST(UnicodeNfcComposeTest, LowercaseLatinCompositions) {
  EXPECT_EQ(iree_unicode_nfc_compose('a', 0x0300), 0x00E0u);  // a + grave → à
  EXPECT_EQ(iree_unicode_nfc_compose('a', 0x0301), 0x00E1u);  // a + acute → á
  EXPECT_EQ(iree_unicode_nfc_compose('e', 0x0301), 0x00E9u);  // e + acute → é
  EXPECT_EQ(iree_unicode_nfc_compose('n', 0x0303), 0x00F1u);  // n + tilde → ñ
  EXPECT_EQ(iree_unicode_nfc_compose('o', 0x0308),
            0x00F6u);  // o + diaeresis → ö
  EXPECT_EQ(iree_unicode_nfc_compose('u', 0x0308),
            0x00FCu);  // u + diaeresis → ü
}

TEST(UnicodeNfcComposeTest, GreekCompositions) {
  // Greek letter + combining mark compositions.
  EXPECT_EQ(iree_unicode_nfc_compose(0x0391, 0x0301),
            0x0386u);  // Α + acute → Ά
  EXPECT_EQ(iree_unicode_nfc_compose(0x03B1, 0x0301),
            0x03ACu);  // α + acute → ά
  EXPECT_EQ(iree_unicode_nfc_compose(0x03B5, 0x0301),
            0x03ADu);  // ε + acute → έ
}

TEST(UnicodeNfcComposeTest, NoComposition) {
  // Pairs that don't compose should return 0.
  EXPECT_EQ(iree_unicode_nfc_compose('A', 'B'), 0u);  // Not a valid pair
  EXPECT_EQ(iree_unicode_nfc_compose('X', 0x0301),
            0u);  // X doesn't compose with acute
  EXPECT_EQ(iree_unicode_nfc_compose(0x4E2D, 0x0301),
            0u);  // CJK doesn't compose
}

TEST(UnicodeNfcComposeTest, LatinExtendedCompositions) {
  // Compositions in Latin Extended blocks.
  EXPECT_EQ(iree_unicode_nfc_compose('A', 0x0304), 0x0100u);  // A + macron → Ā
  EXPECT_EQ(iree_unicode_nfc_compose('a', 0x0304), 0x0101u);  // a + macron → ā
  EXPECT_EQ(iree_unicode_nfc_compose('C', 0x030C), 0x010Cu);  // C + caron → Č
  EXPECT_EQ(iree_unicode_nfc_compose('c', 0x030C), 0x010Du);  // c + caron → č
  EXPECT_EQ(iree_unicode_nfc_compose('S', 0x030C), 0x0160u);  // S + caron → Š
  EXPECT_EQ(iree_unicode_nfc_compose('s', 0x030C), 0x0161u);  // s + caron → š
}

//===----------------------------------------------------------------------===//
// Full NFC Normalization Tests
//===----------------------------------------------------------------------===//

TEST(UnicodeNfcTest, AsciiPassthrough) {
  char buffer[64];
  iree_host_size_t length = 0;
  IREE_ASSERT_OK(iree_unicode_nfc(iree_make_cstring_view("Hello World"), buffer,
                                  sizeof(buffer), &length));
  EXPECT_EQ(std::string(buffer, length), "Hello World");
}

TEST(UnicodeNfcTest, EmptyString) {
  char buffer[64];
  iree_host_size_t length = 0;
  IREE_ASSERT_OK(iree_unicode_nfc(iree_make_cstring_view(""), buffer,
                                  sizeof(buffer), &length));
  EXPECT_EQ(length, 0u);
}

TEST(UnicodeNfcTest, AlreadyComposed) {
  // Already-composed characters should pass through unchanged.
  char buffer[64];
  iree_host_size_t length = 0;

  // "café" with precomposed é (U+00E9).
  const char input[] = "caf\xC3\xA9";
  IREE_ASSERT_OK(
      iree_unicode_nfc(iree_make_string_view(input, sizeof(input) - 1), buffer,
                       sizeof(buffer), &length));
  EXPECT_EQ(std::string(buffer, length), std::string(input, sizeof(input) - 1));
}

TEST(UnicodeNfcTest, DecomposedToComposed) {
  char buffer[64];
  iree_host_size_t length = 0;

  // "café" with decomposed é: 'e' (U+0065) + combining acute (U+0301).
  const char input[] = "cafe\xCC\x81";  // e + combining acute
  IREE_ASSERT_OK(
      iree_unicode_nfc(iree_make_string_view(input, sizeof(input) - 1), buffer,
                       sizeof(buffer), &length));

  // Should compose to "café" with precomposed é.
  const char expected[] = "caf\xC3\xA9";
  EXPECT_EQ(std::string(buffer, length),
            std::string(expected, sizeof(expected) - 1));
}

TEST(UnicodeNfcTest, MultipleDecomposedCharacters) {
  char buffer[64];
  iree_host_size_t length = 0;

  // "résumé" with all decomposed accents.
  // r + e + acute + s + u + m + e + acute
  const char input[] = "re\xCC\x81sume\xCC\x81";
  IREE_ASSERT_OK(
      iree_unicode_nfc(iree_make_string_view(input, sizeof(input) - 1), buffer,
                       sizeof(buffer), &length));

  // Should compose to "résumé" with precomposed é.
  const char expected[] = "r\xC3\xA9sum\xC3\xA9";
  EXPECT_EQ(std::string(buffer, length),
            std::string(expected, sizeof(expected) - 1));
}

TEST(UnicodeNfcTest, MultipleMarksOnSameBase) {
  char buffer[64];
  iree_host_size_t length = 0;

  // o + combining acute (U+0301) + combining diaeresis (U+0308).
  // NFC can only compose one mark, so result should be ó + diaeresis.
  const char input[] = "o\xCC\x81\xCC\x88";
  IREE_ASSERT_OK(
      iree_unicode_nfc(iree_make_string_view(input, sizeof(input) - 1), buffer,
                       sizeof(buffer), &length));

  // o + acute composes to ó (U+00F3), diaeresis remains.
  // Expected: ó (C3 B3) + combining diaeresis (CC 88).
  const char expected[] = "\xC3\xB3\xCC\x88";
  EXPECT_EQ(std::string(buffer, length),
            std::string(expected, sizeof(expected) - 1));
}

TEST(UnicodeNfcTest, CanonicalOrdering) {
  char buffer[64];
  iree_host_size_t length = 0;

  // Test that combining marks are canonically ordered.
  // o + cedilla (CCC=202) + acute (CCC=230).
  // Cedilla is below (202), acute is above (230).
  // Since 202 < 230, order is already canonical.
  const char input1[] = "o\xCC\xA7\xCC\x81";  // cedilla + acute
  IREE_ASSERT_OK(
      iree_unicode_nfc(iree_make_string_view(input1, sizeof(input1) - 1),
                       buffer, sizeof(buffer), &length));

  // o + acute (CCC=230) + cedilla (CCC=202) is NOT canonical order.
  // NFC should reorder to cedilla + acute, then compose.
  char buffer2[64];
  iree_host_size_t length2 = 0;
  const char input2[] = "o\xCC\x81\xCC\xA7";  // acute + cedilla (wrong order)
  IREE_ASSERT_OK(
      iree_unicode_nfc(iree_make_string_view(input2, sizeof(input2) - 1),
                       buffer2, sizeof(buffer2), &length2));

  // Both should produce the same canonical result.
  EXPECT_EQ(std::string(buffer, length), std::string(buffer2, length2));
}

TEST(UnicodeNfcTest, NonComposingMarks) {
  char buffer[64];
  iree_host_size_t length = 0;

  // 'x' doesn't compose with acute accent, so mark should remain.
  const char input[] = "x\xCC\x81";  // x + combining acute
  IREE_ASSERT_OK(
      iree_unicode_nfc(iree_make_string_view(input, sizeof(input) - 1), buffer,
                       sizeof(buffer), &length));

  // Should remain unchanged (x doesn't compose with acute).
  EXPECT_EQ(std::string(buffer, length), std::string(input, sizeof(input) - 1));
}

TEST(UnicodeNfcTest, CjkUnchanged) {
  char buffer[64];
  iree_host_size_t length = 0;

  // CJK characters pass through unchanged.
  const char input[] = "\xE4\xB8\xAD\xE6\x96\x87";  // 中文
  IREE_ASSERT_OK(
      iree_unicode_nfc(iree_make_string_view(input, sizeof(input) - 1), buffer,
                       sizeof(buffer), &length));
  EXPECT_EQ(std::string(buffer, length), std::string(input, sizeof(input) - 1));
}

TEST(UnicodeNfcTest, MixedContent) {
  char buffer[128];
  iree_host_size_t length = 0;

  // Mixed ASCII, decomposed accents, and CJK.
  // "Hello café 中文"
  // With decomposed é: e + combining acute.
  const char input[] = "Hello cafe\xCC\x81 \xE4\xB8\xAD\xE6\x96\x87";
  IREE_ASSERT_OK(
      iree_unicode_nfc(iree_make_string_view(input, sizeof(input) - 1), buffer,
                       sizeof(buffer), &length));

  // Expected: "Hello café 中文" with composed é.
  const char expected[] = "Hello caf\xC3\xA9 \xE4\xB8\xAD\xE6\x96\x87";
  EXPECT_EQ(std::string(buffer, length),
            std::string(expected, sizeof(expected) - 1));
}

TEST(UnicodeNfcTest, BufferTooSmall) {
  char buffer[4];
  iree_host_size_t length = 0;

  // Buffer too small for output.
  iree_status_t status = iree_unicode_nfc(iree_make_cstring_view("Hello World"),
                                          buffer, sizeof(buffer), &length);
  EXPECT_THAT(Status(std::move(status)),
              StatusIs(StatusCode::kResourceExhausted));
}

TEST(UnicodeNfcTest, ScratchSpaceRequirement) {
  // NFC uses the output buffer as scratch space for uint32_t codepoints.
  // Buffer must be at least (codepoint_count * 4) bytes for non-ASCII input.

  // "café" with decomposed é: 5 codepoints × 4 bytes = 20 bytes needed.
  const char input[] = "cafe\xCC\x81";    // 5 bytes, 5 codepoints
  const char expected[] = "caf\xC3\xA9";  // 5 bytes output

  // Buffer too small for scratch space (only 8 bytes, need 20).
  char small_buffer[8];
  iree_host_size_t length = 0;
  iree_status_t status =
      iree_unicode_nfc(iree_make_string_view(input, sizeof(input) - 1),
                       small_buffer, sizeof(small_buffer), &length);
  EXPECT_THAT(Status(std::move(status)),
              StatusIs(StatusCode::kResourceExhausted));

  // Adequate buffer for scratch space.
  char buffer[20];
  length = 0;
  IREE_ASSERT_OK(
      iree_unicode_nfc(iree_make_string_view(input, sizeof(input) - 1), buffer,
                       sizeof(buffer), &length));
  EXPECT_EQ(length, 5u);
  EXPECT_EQ(std::string(buffer, length),
            std::string(expected, sizeof(expected) - 1));
}

TEST(UnicodeNfcTest, GreekWithAccents) {
  char buffer[64];
  iree_host_size_t length = 0;

  // Greek alpha + combining acute.
  const char input[] = "\xCE\xB1\xCC\x81";  // α + acute
  IREE_ASSERT_OK(
      iree_unicode_nfc(iree_make_string_view(input, sizeof(input) - 1), buffer,
                       sizeof(buffer), &length));

  // Should compose to ά (U+03AC).
  const char expected[] = "\xCE\xAC";
  EXPECT_EQ(std::string(buffer, length),
            std::string(expected, sizeof(expected) - 1));
}

TEST(UnicodeNfcTest, Emoji) {
  char buffer[64];
  iree_host_size_t length = 0;

  // Emoji pass through unchanged.
  const char input[] = "\xF0\x9F\x98\x80";  // 😀
  IREE_ASSERT_OK(
      iree_unicode_nfc(iree_make_string_view(input, sizeof(input) - 1), buffer,
                       sizeof(buffer), &length));
  EXPECT_EQ(std::string(buffer, length), std::string(input, sizeof(input) - 1));
}

TEST(UnicodeNfcTest, HangulUnchanged) {
  char buffer[64];
  iree_host_size_t length = 0;

  // Pre-composed Hangul syllables pass through unchanged.
  // We don't do Hangul algorithmic composition.
  const char input[] = "\xEA\xB0\x80";  // 가 (U+AC00)
  IREE_ASSERT_OK(
      iree_unicode_nfc(iree_make_string_view(input, sizeof(input) - 1), buffer,
                       sizeof(buffer), &length));
  EXPECT_EQ(std::string(buffer, length), std::string(input, sizeof(input) - 1));
}

}  // namespace
