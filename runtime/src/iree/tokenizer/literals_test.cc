// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/literals.h"

#include <string>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

//===----------------------------------------------------------------------===//
// Test Helpers
//===----------------------------------------------------------------------===//

// Collects text segments and token IDs from intercept callbacks.
struct InterceptCollector {
  std::vector<std::string> text_segments;
  std::vector<int32_t> token_ids;

  static iree_status_t TextCallback(void* user_data,
                                    iree_string_view_list_t strings) {
    auto* self = static_cast<InterceptCollector*>(user_data);
    for (iree_host_size_t i = 0; i < strings.count; ++i) {
      self->text_segments.emplace_back(strings.values[i].data,
                                       strings.values[i].size);
    }
    return iree_ok_status();
  }

  static iree_status_t TokenCallback(void* user_data,
                                     iree_tokenizer_id_list_t ids) {
    auto* self = static_cast<InterceptCollector*>(user_data);
    for (iree_host_size_t i = 0; i < ids.count; ++i) {
      self->token_ids.push_back(ids.values[i]);
    }
    return iree_ok_status();
  }
};

//===----------------------------------------------------------------------===//
// Lifecycle Tests
//===----------------------------------------------------------------------===//

TEST(LiteralsTest, InitializeEmpty) {
  iree_tokenizer_literals_t literals;
  iree_tokenizer_literals_initialize(iree_allocator_system(), &literals);
  EXPECT_EQ(literals.count, 0);
  EXPECT_FALSE(literals.needs_interception);
  iree_tokenizer_literals_deinitialize(&literals);
}

TEST(LiteralsTest, AddAndFinalize) {
  iree_tokenizer_literals_t literals;
  iree_tokenizer_literals_initialize(iree_allocator_system(), &literals);

  IREE_ASSERT_OK(iree_tokenizer_literals_add(
      &literals, 100, IREE_SV("<mask>"), IREE_TOKENIZER_LITERAL_FLAG_LSTRIP,
      (iree_tokenizer_special_token_t)-1));
  IREE_ASSERT_OK(iree_tokenizer_literals_add(&literals, 101, IREE_SV("<pad>"),
                                             IREE_TOKENIZER_LITERAL_FLAG_NONE,
                                             IREE_TOKENIZER_SPECIAL_TOKEN_PAD));

  EXPECT_EQ(literals.count, 2);
  EXPECT_TRUE(literals.needs_interception);

  IREE_ASSERT_OK(iree_tokenizer_literals_finalize(&literals));
  EXPECT_NE(literals.match_order, nullptr);

  iree_tokenizer_literals_deinitialize(&literals);
}

TEST(LiteralsTest, SkipDuplicateIds) {
  iree_tokenizer_literals_t literals;
  iree_tokenizer_literals_initialize(iree_allocator_system(), &literals);

  IREE_ASSERT_OK(iree_tokenizer_literals_add(
      &literals, 100, IREE_SV("first"), IREE_TOKENIZER_LITERAL_FLAG_NONE,
      (iree_tokenizer_special_token_t)-1));
  IREE_ASSERT_OK(iree_tokenizer_literals_add(
      &literals, 100, IREE_SV("duplicate"), IREE_TOKENIZER_LITERAL_FLAG_NONE,
      (iree_tokenizer_special_token_t)-1));

  EXPECT_EQ(literals.count, 1);
  EXPECT_TRUE(
      iree_string_view_equal(literals.entries[0].content, IREE_SV("first")));

  iree_tokenizer_literals_deinitialize(&literals);
}

TEST(LiteralsTest, SkipEmptyContent) {
  iree_tokenizer_literals_t literals;
  iree_tokenizer_literals_initialize(iree_allocator_system(), &literals);

  IREE_ASSERT_OK(iree_tokenizer_literals_add(
      &literals, 100, IREE_SV(""), IREE_TOKENIZER_LITERAL_FLAG_NONE,
      (iree_tokenizer_special_token_t)-1));

  EXPECT_EQ(literals.count, 0);

  iree_tokenizer_literals_deinitialize(&literals);
}

//===----------------------------------------------------------------------===//
// Match Order Tests
//===----------------------------------------------------------------------===//

TEST(LiteralsTest, MatchOrderLongestFirst) {
  iree_tokenizer_literals_t literals;
  iree_tokenizer_literals_initialize(iree_allocator_system(), &literals);

  // Add in non-length order.
  IREE_ASSERT_OK(iree_tokenizer_literals_add(
      &literals, 1, IREE_SV("ab"), IREE_TOKENIZER_LITERAL_FLAG_LSTRIP,
      (iree_tokenizer_special_token_t)-1));
  IREE_ASSERT_OK(iree_tokenizer_literals_add(
      &literals, 2, IREE_SV("abcd"), IREE_TOKENIZER_LITERAL_FLAG_LSTRIP,
      (iree_tokenizer_special_token_t)-1));
  IREE_ASSERT_OK(iree_tokenizer_literals_add(
      &literals, 3, IREE_SV("abc"), IREE_TOKENIZER_LITERAL_FLAG_LSTRIP,
      (iree_tokenizer_special_token_t)-1));

  IREE_ASSERT_OK(iree_tokenizer_literals_finalize(&literals));

  // Match order should be: abcd (4), abc (3), ab (2).
  EXPECT_EQ(literals.entries[literals.match_order[0]].content.size, 4);
  EXPECT_EQ(literals.entries[literals.match_order[1]].content.size, 3);
  EXPECT_EQ(literals.entries[literals.match_order[2]].content.size, 2);

  iree_tokenizer_literals_deinitialize(&literals);
}

//===----------------------------------------------------------------------===//
// Intercept Tests - No Flags
//===----------------------------------------------------------------------===//

TEST(LiteralsTest, InterceptNoFlagsPassThrough) {
  iree_tokenizer_literals_t literals;
  iree_tokenizer_literals_initialize(iree_allocator_system(), &literals);

  // Literal without interception flags - should NOT be intercepted.
  IREE_ASSERT_OK(iree_tokenizer_literals_add(
      &literals, 100, IREE_SV("<mask>"), IREE_TOKENIZER_LITERAL_FLAG_NONE,
      (iree_tokenizer_special_token_t)-1));
  IREE_ASSERT_OK(iree_tokenizer_literals_finalize(&literals));

  EXPECT_FALSE(literals.needs_interception);

  InterceptCollector collector;
  IREE_ASSERT_OK(iree_tokenizer_literals_intercept(
      &literals, IREE_SV("hello <mask> world"),
      InterceptCollector::TextCallback, InterceptCollector::TokenCallback,
      &collector));

  // All text passed through.
  EXPECT_EQ(collector.text_segments.size(), 1);
  EXPECT_EQ(collector.text_segments[0], "hello <mask> world");
  EXPECT_EQ(collector.token_ids.size(), 0);

  iree_tokenizer_literals_deinitialize(&literals);
}

//===----------------------------------------------------------------------===//
// Intercept Tests - Lstrip
//===----------------------------------------------------------------------===//

TEST(LiteralsTest, InterceptLstripMiddle) {
  iree_tokenizer_literals_t literals;
  iree_tokenizer_literals_initialize(iree_allocator_system(), &literals);

  IREE_ASSERT_OK(iree_tokenizer_literals_add(
      &literals, 100, IREE_SV("<mask>"), IREE_TOKENIZER_LITERAL_FLAG_LSTRIP,
      (iree_tokenizer_special_token_t)-1));
  IREE_ASSERT_OK(iree_tokenizer_literals_finalize(&literals));

  InterceptCollector collector;
  IREE_ASSERT_OK(iree_tokenizer_literals_intercept(
      &literals, IREE_SV("hello <mask> world"),
      InterceptCollector::TextCallback, InterceptCollector::TokenCallback,
      &collector));

  // "hello" emitted as text, <mask> matched (consuming leading space), "
  // world".
  EXPECT_EQ(collector.text_segments.size(), 2);
  EXPECT_EQ(collector.text_segments[0], "hello");
  EXPECT_EQ(collector.text_segments[1], " world");
  EXPECT_EQ(collector.token_ids.size(), 1);
  EXPECT_EQ(collector.token_ids[0], 100);

  iree_tokenizer_literals_deinitialize(&literals);
}

TEST(LiteralsTest, InterceptLstripAtStart) {
  iree_tokenizer_literals_t literals;
  iree_tokenizer_literals_initialize(iree_allocator_system(), &literals);

  IREE_ASSERT_OK(iree_tokenizer_literals_add(
      &literals, 100, IREE_SV("<mask>"), IREE_TOKENIZER_LITERAL_FLAG_LSTRIP,
      (iree_tokenizer_special_token_t)-1));
  IREE_ASSERT_OK(iree_tokenizer_literals_finalize(&literals));

  InterceptCollector collector;
  IREE_ASSERT_OK(iree_tokenizer_literals_intercept(
      &literals, IREE_SV("<mask> world"), InterceptCollector::TextCallback,
      InterceptCollector::TokenCallback, &collector));

  EXPECT_EQ(collector.text_segments.size(), 1);
  EXPECT_EQ(collector.text_segments[0], " world");
  EXPECT_EQ(collector.token_ids.size(), 1);
  EXPECT_EQ(collector.token_ids[0], 100);

  iree_tokenizer_literals_deinitialize(&literals);
}

TEST(LiteralsTest, InterceptLstripAtEnd) {
  iree_tokenizer_literals_t literals;
  iree_tokenizer_literals_initialize(iree_allocator_system(), &literals);

  IREE_ASSERT_OK(iree_tokenizer_literals_add(
      &literals, 100, IREE_SV("<mask>"), IREE_TOKENIZER_LITERAL_FLAG_LSTRIP,
      (iree_tokenizer_special_token_t)-1));
  IREE_ASSERT_OK(iree_tokenizer_literals_finalize(&literals));

  InterceptCollector collector;
  IREE_ASSERT_OK(iree_tokenizer_literals_intercept(
      &literals, IREE_SV("hello <mask>"), InterceptCollector::TextCallback,
      InterceptCollector::TokenCallback, &collector));

  EXPECT_EQ(collector.text_segments.size(), 1);
  EXPECT_EQ(collector.text_segments[0], "hello");
  EXPECT_EQ(collector.token_ids.size(), 1);
  EXPECT_EQ(collector.token_ids[0], 100);

  iree_tokenizer_literals_deinitialize(&literals);
}

//===----------------------------------------------------------------------===//
// Intercept Tests - Rstrip
//===----------------------------------------------------------------------===//

TEST(LiteralsTest, InterceptRstrip) {
  iree_tokenizer_literals_t literals;
  iree_tokenizer_literals_initialize(iree_allocator_system(), &literals);

  IREE_ASSERT_OK(iree_tokenizer_literals_add(
      &literals, 100, IREE_SV("<mask>"), IREE_TOKENIZER_LITERAL_FLAG_RSTRIP,
      (iree_tokenizer_special_token_t)-1));
  IREE_ASSERT_OK(iree_tokenizer_literals_finalize(&literals));

  InterceptCollector collector;
  IREE_ASSERT_OK(iree_tokenizer_literals_intercept(
      &literals, IREE_SV("<mask>  world"), InterceptCollector::TextCallback,
      InterceptCollector::TokenCallback, &collector));

  // <mask> matched with trailing spaces consumed.
  EXPECT_EQ(collector.text_segments.size(), 1);
  EXPECT_EQ(collector.text_segments[0], "world");
  EXPECT_EQ(collector.token_ids.size(), 1);
  EXPECT_EQ(collector.token_ids[0], 100);

  iree_tokenizer_literals_deinitialize(&literals);
}

//===----------------------------------------------------------------------===//
// Intercept Tests - Single Word
//===----------------------------------------------------------------------===//

TEST(LiteralsTest, InterceptSingleWordMatch) {
  iree_tokenizer_literals_t literals;
  iree_tokenizer_literals_initialize(iree_allocator_system(), &literals);

  IREE_ASSERT_OK(iree_tokenizer_literals_add(
      &literals, 100, IREE_SV("mask"), IREE_TOKENIZER_LITERAL_FLAG_SINGLE_WORD,
      (iree_tokenizer_special_token_t)-1));
  IREE_ASSERT_OK(iree_tokenizer_literals_finalize(&literals));

  InterceptCollector collector;
  IREE_ASSERT_OK(iree_tokenizer_literals_intercept(
      &literals, IREE_SV("the mask is"), InterceptCollector::TextCallback,
      InterceptCollector::TokenCallback, &collector));

  EXPECT_EQ(collector.text_segments.size(), 2);
  EXPECT_EQ(collector.text_segments[0], "the ");
  EXPECT_EQ(collector.text_segments[1], " is");
  EXPECT_EQ(collector.token_ids.size(), 1);
  EXPECT_EQ(collector.token_ids[0], 100);

  iree_tokenizer_literals_deinitialize(&literals);
}

TEST(LiteralsTest, InterceptSingleWordNoMatchInWord) {
  iree_tokenizer_literals_t literals;
  iree_tokenizer_literals_initialize(iree_allocator_system(), &literals);

  IREE_ASSERT_OK(iree_tokenizer_literals_add(
      &literals, 100, IREE_SV("mask"), IREE_TOKENIZER_LITERAL_FLAG_SINGLE_WORD,
      (iree_tokenizer_special_token_t)-1));
  IREE_ASSERT_OK(iree_tokenizer_literals_finalize(&literals));

  InterceptCollector collector;
  IREE_ASSERT_OK(iree_tokenizer_literals_intercept(
      &literals, IREE_SV("unmaskable"), InterceptCollector::TextCallback,
      InterceptCollector::TokenCallback, &collector));

  // "mask" appears but not at word boundaries - no match.
  EXPECT_EQ(collector.text_segments.size(), 1);
  EXPECT_EQ(collector.text_segments[0], "unmaskable");
  EXPECT_EQ(collector.token_ids.size(), 0);

  iree_tokenizer_literals_deinitialize(&literals);
}

//===----------------------------------------------------------------------===//
// Intercept Tests - Normalized (Case Insensitive)
//===----------------------------------------------------------------------===//

TEST(LiteralsTest, InterceptNormalized) {
  iree_tokenizer_literals_t literals;
  iree_tokenizer_literals_initialize(iree_allocator_system(), &literals);

  IREE_ASSERT_OK(iree_tokenizer_literals_add(
      &literals, 100, IREE_SV("<MASK>"), IREE_TOKENIZER_LITERAL_FLAG_NORMALIZED,
      (iree_tokenizer_special_token_t)-1));
  IREE_ASSERT_OK(iree_tokenizer_literals_finalize(&literals));

  InterceptCollector collector;
  IREE_ASSERT_OK(iree_tokenizer_literals_intercept(
      &literals, IREE_SV("hello <mask> world"),
      InterceptCollector::TextCallback, InterceptCollector::TokenCallback,
      &collector));

  EXPECT_EQ(collector.text_segments.size(), 2);
  EXPECT_EQ(collector.text_segments[0], "hello ");
  EXPECT_EQ(collector.text_segments[1], " world");
  EXPECT_EQ(collector.token_ids.size(), 1);
  EXPECT_EQ(collector.token_ids[0], 100);

  iree_tokenizer_literals_deinitialize(&literals);
}

//===----------------------------------------------------------------------===//
// Intercept Tests - Multiple Literals
//===----------------------------------------------------------------------===//

TEST(LiteralsTest, InterceptMultipleLiterals) {
  iree_tokenizer_literals_t literals;
  iree_tokenizer_literals_initialize(iree_allocator_system(), &literals);

  IREE_ASSERT_OK(iree_tokenizer_literals_add(
      &literals, 100, IREE_SV("<mask>"), IREE_TOKENIZER_LITERAL_FLAG_LSTRIP,
      (iree_tokenizer_special_token_t)-1));
  IREE_ASSERT_OK(iree_tokenizer_literals_add(
      &literals, 101, IREE_SV("<cls>"), IREE_TOKENIZER_LITERAL_FLAG_LSTRIP,
      (iree_tokenizer_special_token_t)-1));
  IREE_ASSERT_OK(iree_tokenizer_literals_finalize(&literals));

  InterceptCollector collector;
  IREE_ASSERT_OK(iree_tokenizer_literals_intercept(
      &literals, IREE_SV("<cls> hello <mask>"),
      InterceptCollector::TextCallback, InterceptCollector::TokenCallback,
      &collector));

  EXPECT_EQ(collector.text_segments.size(), 1);
  EXPECT_EQ(collector.text_segments[0], " hello");
  EXPECT_EQ(collector.token_ids.size(), 2);
  EXPECT_EQ(collector.token_ids[0], 101);  // <cls>
  EXPECT_EQ(collector.token_ids[1], 100);  // <mask>

  iree_tokenizer_literals_deinitialize(&literals);
}

TEST(LiteralsTest, InterceptLongestMatchWins) {
  iree_tokenizer_literals_t literals;
  iree_tokenizer_literals_initialize(iree_allocator_system(), &literals);

  IREE_ASSERT_OK(iree_tokenizer_literals_add(
      &literals, 100, IREE_SV("<mask>"), IREE_TOKENIZER_LITERAL_FLAG_LSTRIP,
      (iree_tokenizer_special_token_t)-1));
  IREE_ASSERT_OK(iree_tokenizer_literals_add(
      &literals, 101, IREE_SV("<mask_token>"),
      IREE_TOKENIZER_LITERAL_FLAG_LSTRIP, (iree_tokenizer_special_token_t)-1));
  IREE_ASSERT_OK(iree_tokenizer_literals_finalize(&literals));

  InterceptCollector collector;
  IREE_ASSERT_OK(iree_tokenizer_literals_intercept(
      &literals, IREE_SV("hello <mask_token> world"),
      InterceptCollector::TextCallback, InterceptCollector::TokenCallback,
      &collector));

  EXPECT_EQ(collector.token_ids.size(), 1);
  EXPECT_EQ(collector.token_ids[0], 101);  // <mask_token> wins over <mask>

  iree_tokenizer_literals_deinitialize(&literals);
}

//===----------------------------------------------------------------------===//
// Lookup Tests
//===----------------------------------------------------------------------===//

TEST(LiteralsTest, FindById) {
  iree_tokenizer_literals_t literals;
  iree_tokenizer_literals_initialize(iree_allocator_system(), &literals);

  IREE_ASSERT_OK(iree_tokenizer_literals_add(
      &literals, 100, IREE_SV("<mask>"), IREE_TOKENIZER_LITERAL_FLAG_SPECIAL,
      IREE_TOKENIZER_SPECIAL_TOKEN_MASK));
  IREE_ASSERT_OK(iree_tokenizer_literals_finalize(&literals));

  const iree_tokenizer_literal_t* lit =
      iree_tokenizer_literals_find_by_id(&literals, 100);
  ASSERT_NE(lit, nullptr);
  EXPECT_EQ(lit->id, 100);
  EXPECT_TRUE(iree_string_view_equal(lit->content, IREE_SV("<mask>")));

  lit = iree_tokenizer_literals_find_by_id(&literals, 999);
  EXPECT_EQ(lit, nullptr);

  iree_tokenizer_literals_deinitialize(&literals);
}

TEST(LiteralsTest, IsSpecial) {
  iree_tokenizer_literals_t literals;
  iree_tokenizer_literals_initialize(iree_allocator_system(), &literals);

  IREE_ASSERT_OK(iree_tokenizer_literals_add(
      &literals, 100, IREE_SV("<mask>"), IREE_TOKENIZER_LITERAL_FLAG_SPECIAL,
      IREE_TOKENIZER_SPECIAL_TOKEN_MASK));
  IREE_ASSERT_OK(iree_tokenizer_literals_add(
      &literals, 101, IREE_SV("hello"), IREE_TOKENIZER_LITERAL_FLAG_NONE,
      (iree_tokenizer_special_token_t)-1));
  IREE_ASSERT_OK(iree_tokenizer_literals_finalize(&literals));

  EXPECT_TRUE(iree_tokenizer_literals_is_special(&literals, 100));
  EXPECT_FALSE(iree_tokenizer_literals_is_special(&literals, 101));
  EXPECT_FALSE(iree_tokenizer_literals_is_special(&literals, 999));

  iree_tokenizer_literals_deinitialize(&literals);
}

//===----------------------------------------------------------------------===//
// Special Token Matching Tests
//===----------------------------------------------------------------------===//

TEST(LiteralsTest, MatchSpecialTokenBERT) {
  EXPECT_EQ(iree_tokenizer_match_special_token(IREE_SV("[UNK]")),
            IREE_TOKENIZER_SPECIAL_TOKEN_UNK);
  EXPECT_EQ(iree_tokenizer_match_special_token(IREE_SV("[CLS]")),
            IREE_TOKENIZER_SPECIAL_TOKEN_CLS);
  EXPECT_EQ(iree_tokenizer_match_special_token(IREE_SV("[SEP]")),
            IREE_TOKENIZER_SPECIAL_TOKEN_SEP);
  EXPECT_EQ(iree_tokenizer_match_special_token(IREE_SV("[PAD]")),
            IREE_TOKENIZER_SPECIAL_TOKEN_PAD);
  EXPECT_EQ(iree_tokenizer_match_special_token(IREE_SV("[MASK]")),
            IREE_TOKENIZER_SPECIAL_TOKEN_MASK);
}

TEST(LiteralsTest, MatchSpecialTokenGPT) {
  EXPECT_EQ(iree_tokenizer_match_special_token(IREE_SV("<|endoftext|>")),
            IREE_TOKENIZER_SPECIAL_TOKEN_EOS);
  EXPECT_EQ(iree_tokenizer_match_special_token(IREE_SV("<|pad|>")),
            IREE_TOKENIZER_SPECIAL_TOKEN_PAD);
}

TEST(LiteralsTest, MatchSpecialTokenSentencePiece) {
  EXPECT_EQ(iree_tokenizer_match_special_token(IREE_SV("<s>")),
            IREE_TOKENIZER_SPECIAL_TOKEN_BOS);
  EXPECT_EQ(iree_tokenizer_match_special_token(IREE_SV("</s>")),
            IREE_TOKENIZER_SPECIAL_TOKEN_EOS);
  EXPECT_EQ(iree_tokenizer_match_special_token(IREE_SV("<unk>")),
            IREE_TOKENIZER_SPECIAL_TOKEN_UNK);
}

TEST(LiteralsTest, MatchSpecialTokenUnknown) {
  EXPECT_EQ((int)iree_tokenizer_match_special_token(IREE_SV("hello")), -1);
  EXPECT_EQ((int)iree_tokenizer_match_special_token(IREE_SV("<custom>")), -1);
}

}  // namespace
