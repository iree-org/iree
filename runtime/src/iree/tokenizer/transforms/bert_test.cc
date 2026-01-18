// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/tokenizer/transforms/transform.h"

namespace {

// Callback that accumulates segments into a vector.
struct EncodeContext {
  std::vector<std::string>* segments;
};

static iree_status_t AccumulateSegments(void* user_data,
                                        iree_string_view_list_t segments) {
  auto* context = static_cast<EncodeContext*>(user_data);
  for (size_t i = 0; i < segments.count; ++i) {
    context->segments->push_back(
        std::string(segments.values[i].data, segments.values[i].size));
  }
  return iree_ok_status();
}

class BertTransformTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_tokenizer_text_transform_initialize_bert(&transform_);
  }

  void TearDown() override {
    iree_tokenizer_text_transform_deinitialize(&transform_);
  }

  std::vector<std::string> Encode(const char* text) {
    std::vector<std::string> segments;
    EncodeContext context = {&segments};
    iree_status_t status = iree_tokenizer_text_transform_encode(
        NULL, &transform_, IREE_SV(text), AccumulateSegments, &context);
    IREE_EXPECT_OK(status);
    return segments;
  }

  std::string Decode(const char* text) {
    char decoded[1024];
    iree_host_size_t decoded_size = 0;
    iree_status_t status = iree_tokenizer_text_transform_decode(
        &transform_, IREE_SV(text), decoded, sizeof(decoded), &decoded_size);
    IREE_EXPECT_OK(status);
    return std::string(decoded, decoded_size);
  }

  iree_tokenizer_text_transform_t transform_;
};

TEST_F(BertTransformTest, EmptyInput) {
  auto segments = Encode("");
  EXPECT_TRUE(segments.empty());
}

TEST_F(BertTransformTest, SingleWord) {
  auto segments = Encode("hello");
  ASSERT_EQ(segments.size(), 1);
  EXPECT_EQ(segments[0], "hello");
}

TEST_F(BertTransformTest, TwoWords) {
  auto segments = Encode("hello world");
  ASSERT_EQ(segments.size(), 2);
  EXPECT_EQ(segments[0], "hello");
  EXPECT_EQ(segments[1], "world");
}

TEST_F(BertTransformTest, MultipleSpaces) {
  auto segments = Encode("hello   world");
  ASSERT_EQ(segments.size(), 2);
  EXPECT_EQ(segments[0], "hello");
  EXPECT_EQ(segments[1], "world");
}

TEST_F(BertTransformTest, LeadingTrailingSpaces) {
  auto segments = Encode("  hello world  ");
  ASSERT_EQ(segments.size(), 2);
  EXPECT_EQ(segments[0], "hello");
  EXPECT_EQ(segments[1], "world");
}

TEST_F(BertTransformTest, TabsAndNewlines) {
  auto segments = Encode("hello\tworld\nfoo");
  ASSERT_EQ(segments.size(), 3);
  EXPECT_EQ(segments[0], "hello");
  EXPECT_EQ(segments[1], "world");
  EXPECT_EQ(segments[2], "foo");
}

TEST_F(BertTransformTest, PunctuationIsolated) {
  auto segments = Encode("hello, world!");
  ASSERT_EQ(segments.size(), 4);
  EXPECT_EQ(segments[0], "hello");
  EXPECT_EQ(segments[1], ",");
  EXPECT_EQ(segments[2], "world");
  EXPECT_EQ(segments[3], "!");
}

TEST_F(BertTransformTest, PunctuationAtStart) {
  auto segments = Encode("...hello");
  ASSERT_EQ(segments.size(), 4);
  EXPECT_EQ(segments[0], ".");
  EXPECT_EQ(segments[1], ".");
  EXPECT_EQ(segments[2], ".");
  EXPECT_EQ(segments[3], "hello");
}

TEST_F(BertTransformTest, PunctuationAtEnd) {
  auto segments = Encode("hello...");
  ASSERT_EQ(segments.size(), 4);
  EXPECT_EQ(segments[0], "hello");
  EXPECT_EQ(segments[1], ".");
  EXPECT_EQ(segments[2], ".");
  EXPECT_EQ(segments[3], ".");
}

TEST_F(BertTransformTest, MixedPunctuation) {
  auto segments = Encode("hello,world;foo:bar");
  ASSERT_EQ(segments.size(), 7);
  EXPECT_EQ(segments[0], "hello");
  EXPECT_EQ(segments[1], ",");
  EXPECT_EQ(segments[2], "world");
  EXPECT_EQ(segments[3], ";");
  EXPECT_EQ(segments[4], "foo");
  EXPECT_EQ(segments[5], ":");
  EXPECT_EQ(segments[6], "bar");
}

TEST_F(BertTransformTest, OnlyPunctuation) {
  auto segments = Encode("...");
  ASSERT_EQ(segments.size(), 3);
  EXPECT_EQ(segments[0], ".");
  EXPECT_EQ(segments[1], ".");
  EXPECT_EQ(segments[2], ".");
}

TEST_F(BertTransformTest, OnlyWhitespace) {
  auto segments = Encode("   \t\n  ");
  EXPECT_TRUE(segments.empty());
}

TEST_F(BertTransformTest, Apostrophe) {
  auto segments = Encode("don't");
  ASSERT_EQ(segments.size(), 3);
  EXPECT_EQ(segments[0], "don");
  EXPECT_EQ(segments[1], "'");
  EXPECT_EQ(segments[2], "t");
}

TEST_F(BertTransformTest, Hyphen) {
  auto segments = Encode("well-known");
  ASSERT_EQ(segments.size(), 3);
  EXPECT_EQ(segments[0], "well");
  EXPECT_EQ(segments[1], "-");
  EXPECT_EQ(segments[2], "known");
}

TEST_F(BertTransformTest, Quotes) {
  auto segments = Encode("\"hello\"");
  ASSERT_EQ(segments.size(), 3);
  EXPECT_EQ(segments[0], "\"");
  EXPECT_EQ(segments[1], "hello");
  EXPECT_EQ(segments[2], "\"");
}

TEST_F(BertTransformTest, UnicodeLetters) {
  // Non-CJK unicode letters should stay together.
  auto segments = Encode("hello мир world");  // Russian "мир" = peace/world
  ASSERT_EQ(segments.size(), 3);
  EXPECT_EQ(segments[0], "hello");
  EXPECT_EQ(segments[1], "мир");
  EXPECT_EQ(segments[2], "world");
}

TEST_F(BertTransformTest, CjkCharactersSplitIndividually) {
  // CJK ideographs should each become their own segment (like punctuation).
  // This matches HuggingFace BertPreTokenizer behavior.
  auto segments = Encode("日本語");  // Japanese: nihongo (3 CJK chars)
  ASSERT_EQ(segments.size(), 3);
  EXPECT_EQ(segments[0], "日");
  EXPECT_EQ(segments[1], "本");
  EXPECT_EQ(segments[2], "語");
}

TEST_F(BertTransformTest, CjkMixedWithAscii) {
  // CJK chars split individually, ASCII words stay together.
  auto segments = Encode("hello日本world");
  ASSERT_EQ(segments.size(), 4);
  EXPECT_EQ(segments[0], "hello");
  EXPECT_EQ(segments[1], "日");
  EXPECT_EQ(segments[2], "本");
  EXPECT_EQ(segments[3], "world");
}

TEST_F(BertTransformTest, CjkWithSpaces) {
  // Spaces between CJK chars shouldn't matter.
  auto segments = Encode("hello 日 本 world");
  ASSERT_EQ(segments.size(), 4);
  EXPECT_EQ(segments[0], "hello");
  EXPECT_EQ(segments[1], "日");
  EXPECT_EQ(segments[2], "本");
  EXPECT_EQ(segments[3], "world");
}

TEST_F(BertTransformTest, CjkKatakanaHiragana) {
  // Japanese Katakana (テスト = "test") and Hiragana are NOT CJK ideographs.
  // They should be treated as regular letters (stay together).
  auto segments = Encode("テスト");
  ASSERT_EQ(segments.size(), 1);
  EXPECT_EQ(segments[0], "テスト");
}

TEST_F(BertTransformTest, CjkMixedWithKatakana) {
  // Mix of CJK ideographs (split) and Katakana (stay together).
  // 日本語テスト = nihongo-test (3 CJK + 3 Katakana)
  auto segments = Encode("日本語テスト");
  ASSERT_EQ(segments.size(), 4);
  EXPECT_EQ(segments[0], "日");
  EXPECT_EQ(segments[1], "本");
  EXPECT_EQ(segments[2], "語");
  EXPECT_EQ(segments[3], "テスト");  // Katakana stays together
}

TEST_F(BertTransformTest, UnicodeWhitespace) {
  auto segments = Encode("hello\xC2\xA0world");  // hello<NBSP>world
  ASSERT_EQ(segments.size(), 2);
  EXPECT_EQ(segments[0], "hello");
  EXPECT_EQ(segments[1], "world");
}

TEST_F(BertTransformTest, UnicodePunctuation) {
  auto segments = Encode("hello\xEF\xBC\x8Cworld");  // hello，world
  ASSERT_EQ(segments.size(), 3);
  EXPECT_EQ(segments[0], "hello");
  EXPECT_EQ(segments[1], "\xEF\xBC\x8C");  // ，
  EXPECT_EQ(segments[2], "world");
}

TEST_F(BertTransformTest, SingleCharacter) {
  auto segments = Encode("a");
  ASSERT_EQ(segments.size(), 1);
  EXPECT_EQ(segments[0], "a");
}

TEST_F(BertTransformTest, SinglePunctuation) {
  auto segments = Encode(".");
  ASSERT_EQ(segments.size(), 1);
  EXPECT_EQ(segments[0], ".");
}

//===----------------------------------------------------------------------===//
// BERT-Specific Punctuation Tests
//===----------------------------------------------------------------------===//
// BERT uses a custom punctuation definition that differs from Unicode.
// Google BERT's _is_punctuation() treats ALL non-letter/non-number ASCII as
// punctuation, including characters like =, $, ^, ` that Unicode classifies
// as Symbol (S*) rather than Punctuation (P*).
//
// ASCII ranges treated as punctuation:
//   33-47:   ! " # $ % & ' ( ) * + , - . /
//   58-64:   : ; < = > ? @
//   91-96:   [ \ ] ^ _ `
//   123-126: { | } ~

TEST_F(BertTransformTest, EqualsSignIsolated) {
  // = is in range 58-64 (Unicode: Sm = Math Symbol, not punctuation)
  auto segments = Encode("x=1");
  ASSERT_EQ(segments.size(), 3);
  EXPECT_EQ(segments[0], "x");
  EXPECT_EQ(segments[1], "=");
  EXPECT_EQ(segments[2], "1");
}

TEST_F(BertTransformTest, EqualsWithSpaces) {
  auto segments = Encode("x = 1");
  ASSERT_EQ(segments.size(), 3);
  EXPECT_EQ(segments[0], "x");
  EXPECT_EQ(segments[1], "=");
  EXPECT_EQ(segments[2], "1");
}

TEST_F(BertTransformTest, DollarSignIsolated) {
  // $ is in range 33-47 (Unicode: Sc = Currency Symbol, not punctuation)
  auto segments = Encode("$100");
  ASSERT_EQ(segments.size(), 2);
  EXPECT_EQ(segments[0], "$");
  EXPECT_EQ(segments[1], "100");
}

TEST_F(BertTransformTest, CaretIsolated) {
  // ^ is in range 91-96 (Unicode: Sk = Modifier Symbol, not punctuation)
  auto segments = Encode("a^b");
  ASSERT_EQ(segments.size(), 3);
  EXPECT_EQ(segments[0], "a");
  EXPECT_EQ(segments[1], "^");
  EXPECT_EQ(segments[2], "b");
}

TEST_F(BertTransformTest, BacktickIsolated) {
  // ` is in range 91-96 (Unicode: Sk = Modifier Symbol, not punctuation)
  auto segments = Encode("x`y");
  ASSERT_EQ(segments.size(), 3);
  EXPECT_EQ(segments[0], "x");
  EXPECT_EQ(segments[1], "`");
  EXPECT_EQ(segments[2], "y");
}

TEST_F(BertTransformTest, PipeIsolated) {
  // | is in range 123-126 (Unicode: Sm = Math Symbol, not punctuation)
  auto segments = Encode("a|b");
  ASSERT_EQ(segments.size(), 3);
  EXPECT_EQ(segments[0], "a");
  EXPECT_EQ(segments[1], "|");
  EXPECT_EQ(segments[2], "b");
}

TEST_F(BertTransformTest, TildeIsolated) {
  // ~ is in range 123-126 (Unicode: Sm = Math Symbol, not punctuation)
  auto segments = Encode("a~b");
  ASSERT_EQ(segments.size(), 3);
  EXPECT_EQ(segments[0], "a");
  EXPECT_EQ(segments[1], "~");
  EXPECT_EQ(segments[2], "b");
}

TEST_F(BertTransformTest, PlusAndAsterisk) {
  // + and * are in range 33-47 (Unicode: Sm = Math Symbol, not punctuation)
  auto segments = Encode("a+b*c");
  ASSERT_EQ(segments.size(), 5);
  EXPECT_EQ(segments[0], "a");
  EXPECT_EQ(segments[1], "+");
  EXPECT_EQ(segments[2], "b");
  EXPECT_EQ(segments[3], "*");
  EXPECT_EQ(segments[4], "c");
}

TEST_F(BertTransformTest, LessThanGreaterThan) {
  // < and > are in range 58-64 (Unicode: Sm = Math Symbol, not punctuation)
  auto segments = Encode("a<b>c");
  ASSERT_EQ(segments.size(), 5);
  EXPECT_EQ(segments[0], "a");
  EXPECT_EQ(segments[1], "<");
  EXPECT_EQ(segments[2], "b");
  EXPECT_EQ(segments[3], ">");
  EXPECT_EQ(segments[4], "c");
}

TEST_F(BertTransformTest, CodeWithBackticks) {
  // Common pattern that was failing: Code: `x = 1`
  auto segments = Encode("Code: `x = 1`");
  ASSERT_EQ(segments.size(), 7);
  EXPECT_EQ(segments[0], "Code");
  EXPECT_EQ(segments[1], ":");
  EXPECT_EQ(segments[2], "`");
  EXPECT_EQ(segments[3], "x");
  EXPECT_EQ(segments[4], "=");
  EXPECT_EQ(segments[5], "1");
  EXPECT_EQ(segments[6], "`");
}

TEST_F(BertTransformTest, PriceWithDollar) {
  // Common pattern: $99.99
  auto segments = Encode("$99.99");
  ASSERT_EQ(segments.size(), 4);
  EXPECT_EQ(segments[0], "$");
  EXPECT_EQ(segments[1], "99");
  EXPECT_EQ(segments[2], ".");
  EXPECT_EQ(segments[3], "99");
}

TEST_F(BertTransformTest, ScientificNotation) {
  // E=mc² pattern (the superscript 2 is non-ASCII, stays with m)
  auto segments = Encode("E=mc");
  ASSERT_EQ(segments.size(), 3);
  EXPECT_EQ(segments[0], "E");
  EXPECT_EQ(segments[1], "=");
  EXPECT_EQ(segments[2], "mc");
}

//===----------------------------------------------------------------------===//
// Real-World Pattern Tests
//===----------------------------------------------------------------------===//
// Real-world pattern tests harvested from testing with actual BERT tokenizers.

TEST_F(BertTransformTest, EmailAddress) {
  auto segments = Encode("test@example.com");
  ASSERT_EQ(segments.size(), 5);
  EXPECT_EQ(segments[0], "test");
  EXPECT_EQ(segments[1], "@");
  EXPECT_EQ(segments[2], "example");
  EXPECT_EQ(segments[3], ".");
  EXPECT_EQ(segments[4], "com");
}

TEST_F(BertTransformTest, UrlWithProtocol) {
  auto segments = Encode("https://example.org/path");
  ASSERT_EQ(segments.size(), 9);
  EXPECT_EQ(segments[0], "https");
  EXPECT_EQ(segments[1], ":");
  EXPECT_EQ(segments[2], "/");
  EXPECT_EQ(segments[3], "/");
  EXPECT_EQ(segments[4], "example");
  EXPECT_EQ(segments[5], ".");
  EXPECT_EQ(segments[6], "org");
  EXPECT_EQ(segments[7], "/");
  EXPECT_EQ(segments[8], "path");
}

TEST_F(BertTransformTest, DecimalNumber) {
  auto segments = Encode("3.14159");
  ASSERT_EQ(segments.size(), 3);
  EXPECT_EQ(segments[0], "3");
  EXPECT_EQ(segments[1], ".");
  EXPECT_EQ(segments[2], "14159");
}

TEST_F(BertTransformTest, NumberWithCommas) {
  auto segments = Encode("1,000,000");
  ASSERT_EQ(segments.size(), 5);
  EXPECT_EQ(segments[0], "1");
  EXPECT_EQ(segments[1], ",");
  EXPECT_EQ(segments[2], "000");
  EXPECT_EQ(segments[3], ",");
  EXPECT_EQ(segments[4], "000");
}

TEST_F(BertTransformTest, MultipleContractions) {
  auto segments = Encode("can't won't shouldn't");
  ASSERT_EQ(segments.size(), 9);
  EXPECT_EQ(segments[0], "can");
  EXPECT_EQ(segments[1], "'");
  EXPECT_EQ(segments[2], "t");
  EXPECT_EQ(segments[3], "won");
  EXPECT_EQ(segments[4], "'");
  EXPECT_EQ(segments[5], "t");
  EXPECT_EQ(segments[6], "shouldn");
  EXPECT_EQ(segments[7], "'");
  EXPECT_EQ(segments[8], "t");
}

TEST_F(BertTransformTest, SentenceWithPunctuation) {
  auto segments = Encode("Hello, world! How are you?");
  ASSERT_EQ(segments.size(), 8);
  EXPECT_EQ(segments[0], "Hello");
  EXPECT_EQ(segments[1], ",");
  EXPECT_EQ(segments[2], "world");
  EXPECT_EQ(segments[3], "!");
  EXPECT_EQ(segments[4], "How");
  EXPECT_EQ(segments[5], "are");
  EXPECT_EQ(segments[6], "you");
  EXPECT_EQ(segments[7], "?");
}

TEST_F(BertTransformTest, Pangram) {
  auto segments = Encode("The quick brown fox jumps over the lazy dog.");
  ASSERT_EQ(segments.size(), 10);
  EXPECT_EQ(segments[0], "The");
  EXPECT_EQ(segments[1], "quick");
  EXPECT_EQ(segments[2], "brown");
  EXPECT_EQ(segments[3], "fox");
  EXPECT_EQ(segments[4], "jumps");
  EXPECT_EQ(segments[5], "over");
  EXPECT_EQ(segments[6], "the");
  EXPECT_EQ(segments[7], "lazy");
  EXPECT_EQ(segments[8], "dog");
  EXPECT_EQ(segments[9], ".");
}

//===----------------------------------------------------------------------===//
// BERT Decode Tests
//===----------------------------------------------------------------------===//

TEST_F(BertTransformTest, DecodePassthrough) {
  auto decoded = Decode("hello world");
  EXPECT_EQ(decoded, "hello world");
}

TEST_F(BertTransformTest, DecodeEmpty) {
  auto decoded = Decode("");
  EXPECT_EQ(decoded, "");
}

TEST_F(BertTransformTest, DecodeUnicode) {
  auto decoded = Decode("hello 你好");
  EXPECT_EQ(decoded, "hello 你好");
}

//===----------------------------------------------------------------------===//
// Whitespace Transform Tests
//===----------------------------------------------------------------------===//

class WhitespaceTransformTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_tokenizer_text_transform_initialize_whitespace(&transform_);
  }

  void TearDown() override {
    iree_tokenizer_text_transform_deinitialize(&transform_);
  }

  std::vector<std::string> Encode(const char* text) {
    std::vector<std::string> segments;
    EncodeContext context = {&segments};
    iree_status_t status = iree_tokenizer_text_transform_encode(
        NULL, &transform_, IREE_SV(text), AccumulateSegments, &context);
    IREE_EXPECT_OK(status);
    return segments;
  }

  iree_tokenizer_text_transform_t transform_;
};

TEST_F(WhitespaceTransformTest, PunctuationNotIsolated) {
  auto segments = Encode("hello, world!");
  ASSERT_EQ(segments.size(), 2);
  EXPECT_EQ(segments[0], "hello,");
  EXPECT_EQ(segments[1], "world!");
}

TEST_F(WhitespaceTransformTest, BasicSplit) {
  auto segments = Encode("one two three");
  ASSERT_EQ(segments.size(), 3);
  EXPECT_EQ(segments[0], "one");
  EXPECT_EQ(segments[1], "two");
  EXPECT_EQ(segments[2], "three");
}

}  // namespace
