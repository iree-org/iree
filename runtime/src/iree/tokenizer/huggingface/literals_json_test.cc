// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/huggingface/literals_json.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

//===----------------------------------------------------------------------===//
// Helper
//===----------------------------------------------------------------------===//

class LiteralsJsonTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_tokenizer_literals_initialize(iree_allocator_system(), &literals_);
  }

  void TearDown() override { iree_tokenizer_literals_deinitialize(&literals_); }

  iree_tokenizer_literals_t literals_;
};

//===----------------------------------------------------------------------===//
// Parse Tests
//===----------------------------------------------------------------------===//

TEST_F(LiteralsJsonTest, ParseEmpty) {
  const char* json = R"({})";
  IREE_ASSERT_OK(iree_tokenizer_literals_parse_json(IREE_SV(json), &literals_));
  EXPECT_EQ(literals_.count, 0);
}

TEST_F(LiteralsJsonTest, ParseEmptyArray) {
  const char* json = R"({"added_tokens": []})";
  IREE_ASSERT_OK(iree_tokenizer_literals_parse_json(IREE_SV(json), &literals_));
  EXPECT_EQ(literals_.count, 0);
}

TEST_F(LiteralsJsonTest, ParseBasic) {
  const char* json = R"({
    "added_tokens": [
      {"id": 0, "content": "<s>", "special": true},
      {"id": 1, "content": "</s>", "special": true},
      {"id": 2, "content": "<unk>", "special": true}
    ]
  })";
  IREE_ASSERT_OK(iree_tokenizer_literals_parse_json(IREE_SV(json), &literals_));
  EXPECT_EQ(literals_.count, 3);

  EXPECT_EQ(literals_.entries[0].id, 0);
  EXPECT_TRUE(
      iree_string_view_equal(literals_.entries[0].content, IREE_SV("<s>")));
  EXPECT_TRUE(literals_.entries[0].flags & IREE_TOKENIZER_LITERAL_FLAG_SPECIAL);
  EXPECT_EQ(literals_.entries[0].special_type,
            IREE_TOKENIZER_SPECIAL_TOKEN_BOS);

  EXPECT_EQ(literals_.entries[1].id, 1);
  EXPECT_TRUE(
      iree_string_view_equal(literals_.entries[1].content, IREE_SV("</s>")));
  EXPECT_EQ(literals_.entries[1].special_type,
            IREE_TOKENIZER_SPECIAL_TOKEN_EOS);

  EXPECT_EQ(literals_.entries[2].id, 2);
  EXPECT_TRUE(
      iree_string_view_equal(literals_.entries[2].content, IREE_SV("<unk>")));
  EXPECT_EQ(literals_.entries[2].special_type,
            IREE_TOKENIZER_SPECIAL_TOKEN_UNK);
}

TEST_F(LiteralsJsonTest, ParseWithLstrip) {
  const char* json = R"({
    "added_tokens": [
      {"id": 50264, "content": "<mask>", "special": true, "lstrip": true}
    ]
  })";
  IREE_ASSERT_OK(iree_tokenizer_literals_parse_json(IREE_SV(json), &literals_));
  EXPECT_EQ(literals_.count, 1);

  EXPECT_EQ(literals_.entries[0].id, 50264);
  EXPECT_TRUE(literals_.entries[0].flags & IREE_TOKENIZER_LITERAL_FLAG_LSTRIP);
  EXPECT_TRUE(literals_.entries[0].flags & IREE_TOKENIZER_LITERAL_FLAG_SPECIAL);
  EXPECT_TRUE(literals_.needs_interception);
}

TEST_F(LiteralsJsonTest, ParseWithRstrip) {
  const char* json = R"({
    "added_tokens": [
      {"id": 100, "content": "<tok>", "special": true, "rstrip": true}
    ]
  })";
  IREE_ASSERT_OK(iree_tokenizer_literals_parse_json(IREE_SV(json), &literals_));
  EXPECT_EQ(literals_.count, 1);

  EXPECT_TRUE(literals_.entries[0].flags & IREE_TOKENIZER_LITERAL_FLAG_RSTRIP);
  EXPECT_TRUE(literals_.needs_interception);
}

TEST_F(LiteralsJsonTest, ParseWithSingleWord) {
  const char* json = R"({
    "added_tokens": [
      {"id": 100, "content": "hello", "special": false, "single_word": true}
    ]
  })";
  IREE_ASSERT_OK(iree_tokenizer_literals_parse_json(IREE_SV(json), &literals_));
  EXPECT_EQ(literals_.count, 1);

  EXPECT_TRUE(literals_.entries[0].flags &
              IREE_TOKENIZER_LITERAL_FLAG_SINGLE_WORD);
  EXPECT_FALSE(literals_.entries[0].flags &
               IREE_TOKENIZER_LITERAL_FLAG_SPECIAL);
  EXPECT_TRUE(literals_.needs_interception);
}

TEST_F(LiteralsJsonTest, ParseWithNormalized) {
  const char* json = R"({
    "added_tokens": [
      {"id": 100, "content": "HELLO", "special": false, "normalized": true}
    ]
  })";
  IREE_ASSERT_OK(iree_tokenizer_literals_parse_json(IREE_SV(json), &literals_));
  EXPECT_EQ(literals_.count, 1);

  EXPECT_TRUE(literals_.entries[0].flags &
              IREE_TOKENIZER_LITERAL_FLAG_NORMALIZED);
  EXPECT_TRUE(literals_.needs_interception);
}

TEST_F(LiteralsJsonTest, ParseMixedFlags) {
  const char* json = R"({
    "added_tokens": [
      {
        "id": 100,
        "content": "<mask>",
        "special": true,
        "lstrip": true,
        "rstrip": true,
        "single_word": false,
        "normalized": false
      }
    ]
  })";
  IREE_ASSERT_OK(iree_tokenizer_literals_parse_json(IREE_SV(json), &literals_));
  EXPECT_EQ(literals_.count, 1);

  EXPECT_TRUE(literals_.entries[0].flags & IREE_TOKENIZER_LITERAL_FLAG_LSTRIP);
  EXPECT_TRUE(literals_.entries[0].flags & IREE_TOKENIZER_LITERAL_FLAG_RSTRIP);
  EXPECT_FALSE(literals_.entries[0].flags &
               IREE_TOKENIZER_LITERAL_FLAG_SINGLE_WORD);
  EXPECT_FALSE(literals_.entries[0].flags &
               IREE_TOKENIZER_LITERAL_FLAG_NORMALIZED);
  EXPECT_TRUE(literals_.entries[0].flags & IREE_TOKENIZER_LITERAL_FLAG_SPECIAL);
}

TEST_F(LiteralsJsonTest, ParseSkipDuplicateIds) {
  const char* json = R"({
    "added_tokens": [
      {"id": 0, "content": "first", "special": true},
      {"id": 0, "content": "duplicate", "special": true}
    ]
  })";
  IREE_ASSERT_OK(iree_tokenizer_literals_parse_json(IREE_SV(json), &literals_));
  EXPECT_EQ(literals_.count, 1);
  EXPECT_TRUE(
      iree_string_view_equal(literals_.entries[0].content, IREE_SV("first")));
}

TEST_F(LiteralsJsonTest, ParseSkipEmptyContent) {
  const char* json = R"({
    "added_tokens": [
      {"id": 0, "content": "", "special": true}
    ]
  })";
  IREE_ASSERT_OK(iree_tokenizer_literals_parse_json(IREE_SV(json), &literals_));
  EXPECT_EQ(literals_.count, 0);
}

TEST_F(LiteralsJsonTest, ParseEscapedContent) {
  const char* json = R"({
    "added_tokens": [
      {"id": 0, "content": "<|endoftext|>", "special": true}
    ]
  })";
  IREE_ASSERT_OK(iree_tokenizer_literals_parse_json(IREE_SV(json), &literals_));
  EXPECT_EQ(literals_.count, 1);
  EXPECT_TRUE(iree_string_view_equal(literals_.entries[0].content,
                                     IREE_SV("<|endoftext|>")));
}

TEST_F(LiteralsJsonTest, ParseSpecialTokenDetection) {
  const char* json = R"({
    "added_tokens": [
      {"id": 0, "content": "[CLS]", "special": true},
      {"id": 1, "content": "[SEP]", "special": true},
      {"id": 2, "content": "[PAD]", "special": true},
      {"id": 3, "content": "[MASK]", "special": true},
      {"id": 4, "content": "[UNK]", "special": true}
    ]
  })";
  IREE_ASSERT_OK(iree_tokenizer_literals_parse_json(IREE_SV(json), &literals_));
  EXPECT_EQ(literals_.count, 5);

  EXPECT_EQ(literals_.entries[0].special_type,
            IREE_TOKENIZER_SPECIAL_TOKEN_CLS);
  EXPECT_EQ(literals_.entries[1].special_type,
            IREE_TOKENIZER_SPECIAL_TOKEN_SEP);
  EXPECT_EQ(literals_.entries[2].special_type,
            IREE_TOKENIZER_SPECIAL_TOKEN_PAD);
  EXPECT_EQ(literals_.entries[3].special_type,
            IREE_TOKENIZER_SPECIAL_TOKEN_MASK);
  EXPECT_EQ(literals_.entries[4].special_type,
            IREE_TOKENIZER_SPECIAL_TOKEN_UNK);
}

TEST_F(LiteralsJsonTest, ParseNonSpecialToken) {
  const char* json = R"({
    "added_tokens": [
      {"id": 100, "content": "hello", "special": false}
    ]
  })";
  IREE_ASSERT_OK(iree_tokenizer_literals_parse_json(IREE_SV(json), &literals_));
  EXPECT_EQ(literals_.count, 1);
  EXPECT_EQ((int)literals_.entries[0].special_type, -1);
  EXPECT_FALSE(literals_.entries[0].flags &
               IREE_TOKENIZER_LITERAL_FLAG_SPECIAL);
}

TEST_F(LiteralsJsonTest, ParseNoInterceptionNeeded) {
  const char* json = R"({
    "added_tokens": [
      {"id": 0, "content": "<s>", "special": true},
      {"id": 1, "content": "</s>", "special": true}
    ]
  })";
  IREE_ASSERT_OK(iree_tokenizer_literals_parse_json(IREE_SV(json), &literals_));

  // No lstrip/rstrip/single_word/normalized flags - no interception needed.
  EXPECT_FALSE(literals_.needs_interception);
}

TEST_F(LiteralsJsonTest, ParseRoBERTaStyle) {
  // RoBERTa-style added_tokens with lstrip=true on <mask>.
  const char* json = R"({
    "added_tokens": [
      {
        "id": 0,
        "content": "<s>",
        "single_word": false,
        "lstrip": false,
        "rstrip": false,
        "normalized": true,
        "special": true
      },
      {
        "id": 1,
        "content": "<pad>",
        "single_word": false,
        "lstrip": false,
        "rstrip": false,
        "normalized": true,
        "special": true
      },
      {
        "id": 2,
        "content": "</s>",
        "single_word": false,
        "lstrip": false,
        "rstrip": false,
        "normalized": true,
        "special": true
      },
      {
        "id": 3,
        "content": "<unk>",
        "single_word": false,
        "lstrip": false,
        "rstrip": false,
        "normalized": true,
        "special": true
      },
      {
        "id": 50264,
        "content": "<mask>",
        "single_word": false,
        "lstrip": true,
        "rstrip": false,
        "normalized": false,
        "special": true
      }
    ]
  })";
  IREE_ASSERT_OK(iree_tokenizer_literals_parse_json(IREE_SV(json), &literals_));
  EXPECT_EQ(literals_.count, 5);

  // <mask> has lstrip=true, so interception is needed.
  EXPECT_TRUE(literals_.needs_interception);

  // Find <mask> entry.
  const iree_tokenizer_literal_t* mask = nullptr;
  for (iree_host_size_t i = 0; i < literals_.count; ++i) {
    if (literals_.entries[i].id == 50264) {
      mask = &literals_.entries[i];
      break;
    }
  }
  ASSERT_NE(mask, nullptr);
  EXPECT_TRUE(iree_string_view_equal(mask->content, IREE_SV("<mask>")));
  EXPECT_TRUE(mask->flags & IREE_TOKENIZER_LITERAL_FLAG_LSTRIP);
  EXPECT_FALSE(mask->flags & IREE_TOKENIZER_LITERAL_FLAG_RSTRIP);
  EXPECT_FALSE(mask->flags & IREE_TOKENIZER_LITERAL_FLAG_NORMALIZED);
  EXPECT_TRUE(mask->flags & IREE_TOKENIZER_LITERAL_FLAG_SPECIAL);
  EXPECT_EQ(mask->special_type, IREE_TOKENIZER_SPECIAL_TOKEN_MASK);
}

}  // namespace
