// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/huggingface/bpe_json.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/tokenizer/huggingface/testdata/bpe_testdata.h"

namespace {

static iree_string_view_t GetTestFile(const char* name) {
  const struct iree_file_toc_t* file_toc = iree_tokenizer_bpe_testdata_create();
  for (size_t i = 0; i < iree_tokenizer_bpe_testdata_size(); ++i) {
    if (strcmp(file_toc[i].name, name) == 0) {
      return iree_make_string_view((const char*)file_toc[i].data,
                                   file_toc[i].size);
    }
  }
  return iree_string_view_empty();
}

TEST(BpeJsonTest, LoadMinimalVocab) {
  iree_string_view_t json = GetTestFile("bpe_minimal.json");
  ASSERT_GT(json.size, 0u);

  iree_tokenizer_vocab_t* vocab = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_import_bpe_json(
      json, iree_allocator_system(), &vocab));
  ASSERT_NE(vocab, nullptr);

  EXPECT_EQ(iree_tokenizer_vocab_capacity(vocab), 21u);

  iree_tokenizer_vocab_free(vocab);
}

TEST(BpeJsonTest, VocabLookupHit) {
  iree_string_view_t json = GetTestFile("bpe_minimal.json");
  iree_tokenizer_vocab_t* vocab = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_import_bpe_json(
      json, iree_allocator_system(), &vocab));

  EXPECT_EQ(iree_tokenizer_vocab_lookup(vocab, IREE_SVL("<|endoftext|>")), 0);
  EXPECT_EQ(iree_tokenizer_vocab_lookup(vocab, IREE_SVL("<|pad|>")), 1);
  EXPECT_EQ(iree_tokenizer_vocab_lookup(vocab, IREE_SVL("h")), 3);
  EXPECT_EQ(iree_tokenizer_vocab_lookup(vocab, IREE_SVL("hello")), 20);
  EXPECT_EQ(iree_tokenizer_vocab_lookup(vocab, IREE_SVL("world")), 19);

  iree_tokenizer_vocab_free(vocab);
}

TEST(BpeJsonTest, VocabLookupMiss) {
  iree_string_view_t json = GetTestFile("bpe_minimal.json");
  iree_tokenizer_vocab_t* vocab = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_import_bpe_json(
      json, iree_allocator_system(), &vocab));

  EXPECT_EQ(iree_tokenizer_vocab_lookup(vocab, IREE_SVL("notfound")), -1);
  EXPECT_EQ(iree_tokenizer_vocab_lookup(vocab, IREE_SVL("HELLO")),
            -1);  // Case-sensitive.

  iree_tokenizer_vocab_free(vocab);
}

TEST(BpeJsonTest, SpecialTokensDetected) {
  iree_string_view_t json = GetTestFile("bpe_minimal.json");
  iree_tokenizer_vocab_t* vocab = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_import_bpe_json(
      json, iree_allocator_system(), &vocab));

  iree_tokenizer_special_ids_t ids = iree_tokenizer_vocab_special_ids(vocab);
  EXPECT_EQ(ids.eos, 0);  // <|endoftext|> maps to EOS.
  EXPECT_EQ(ids.pad, 1);  // <|pad|> maps to PAD.
  EXPECT_EQ(ids.unk, -1);
  EXPECT_EQ(ids.bos, -1);
  EXPECT_EQ(ids.cls, -1);
  EXPECT_EQ(ids.sep, -1);
  EXPECT_EQ(ids.mask, -1);

  iree_tokenizer_vocab_free(vocab);
}

TEST(BpeJsonTest, MergesLoaded) {
  iree_string_view_t json = GetTestFile("bpe_minimal.json");
  iree_tokenizer_vocab_t* vocab = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_import_bpe_json(
      json, iree_allocator_system(), &vocab));

  // Should have 11 merges: h e, l l, l o, w o, o r, l d, he l, ll o, wo r,
  // wor ld, hel lo.
  EXPECT_EQ(iree_tokenizer_vocab_merge_count(vocab), 11u);

  // Check first merge: h(3) + e(4) -> he(10).
  iree_tokenizer_merge_t merge0 = iree_tokenizer_vocab_merge(vocab, 0);
  EXPECT_EQ(merge0.left_id, 3u);
  EXPECT_EQ(merge0.right_id, 4u);

  // Check second merge: l(5) + l(5) -> ll(11).
  iree_tokenizer_merge_t merge1 = iree_tokenizer_vocab_merge(vocab, 1);
  EXPECT_EQ(merge1.left_id, 5u);
  EXPECT_EQ(merge1.right_id, 5u);

  // Check last merge: hel(16) + lo(12) -> hello(20).
  iree_tokenizer_merge_t merge10 = iree_tokenizer_vocab_merge(vocab, 10);
  EXPECT_EQ(merge10.left_id, 16u);
  EXPECT_EQ(merge10.right_id, 12u);

  iree_tokenizer_vocab_free(vocab);
}

TEST(BpeJsonTest, MergeOutOfBounds) {
  iree_string_view_t json = GetTestFile("bpe_minimal.json");
  iree_tokenizer_vocab_t* vocab = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_import_bpe_json(
      json, iree_allocator_system(), &vocab));

  // Access out of bounds should return {0, 0}.
  iree_tokenizer_merge_t bad = iree_tokenizer_vocab_merge(vocab, 999);
  EXPECT_EQ(bad.left_id, 0u);
  EXPECT_EQ(bad.right_id, 0u);

  iree_tokenizer_vocab_free(vocab);
}

TEST(BpeJsonTest, TokenTextRetrieval) {
  iree_string_view_t json = GetTestFile("bpe_minimal.json");
  iree_tokenizer_vocab_t* vocab = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_import_bpe_json(
      json, iree_allocator_system(), &vocab));

  iree_string_view_t text3 = iree_tokenizer_vocab_token_text(vocab, 3);
  iree_string_view_t text20 = iree_tokenizer_vocab_token_text(vocab, 20);
  EXPECT_EQ(iree_string_view_compare(text3, IREE_SVL("h")), 0);
  EXPECT_EQ(iree_string_view_compare(text20, IREE_SVL("hello")), 0);

  iree_tokenizer_vocab_free(vocab);
}

TEST(BpeJsonTest, WrongModelTypeError) {
  const char* wp_json = R"({
    "model": {
      "type": "WordPiece",
      "vocab": {"hello": 0}
    }
  })";
  iree_string_view_t json = iree_make_string_view(wp_json, strlen(wp_json));

  iree_tokenizer_vocab_t* vocab = nullptr;
  iree_status_t status = iree_tokenizer_vocab_import_bpe_json(
      json, iree_allocator_system(), &vocab);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
  EXPECT_EQ(vocab, nullptr);
  iree_status_free(status);
}

TEST(BpeJsonTest, MissingModelError) {
  const char* no_model = R"({"added_tokens": []})";
  iree_string_view_t json = iree_make_string_view(no_model, strlen(no_model));

  iree_tokenizer_vocab_t* vocab = nullptr;
  iree_status_t status = iree_tokenizer_vocab_import_bpe_json(
      json, iree_allocator_system(), &vocab);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND, status);
  EXPECT_EQ(vocab, nullptr);
  iree_status_free(status);
}

TEST(BpeJsonTest, MissingVocabError) {
  const char* no_vocab = R"({
    "model": {
      "type": "BPE"
    }
  })";
  iree_string_view_t json = iree_make_string_view(no_vocab, strlen(no_vocab));

  iree_tokenizer_vocab_t* vocab = nullptr;
  iree_status_t status = iree_tokenizer_vocab_import_bpe_json(
      json, iree_allocator_system(), &vocab);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND, status);
  EXPECT_EQ(vocab, nullptr);
  iree_status_free(status);
}

TEST(BpeJsonTest, EmptyVocab) {
  const char* empty_vocab = R"({
    "model": {
      "type": "BPE",
      "vocab": {}
    }
  })";
  iree_string_view_t json =
      iree_make_string_view(empty_vocab, strlen(empty_vocab));

  iree_tokenizer_vocab_t* vocab = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_import_bpe_json(
      json, iree_allocator_system(), &vocab));
  EXPECT_EQ(iree_tokenizer_vocab_capacity(vocab), 0u);
  EXPECT_EQ(iree_tokenizer_vocab_merge_count(vocab), 0u);
  iree_tokenizer_vocab_free(vocab);
}

TEST(BpeJsonTest, VocabWithNoMerges) {
  const char* no_merges = R"({
    "model": {
      "type": "BPE",
      "vocab": {"a": 0, "b": 1, "c": 2}
    }
  })";
  iree_string_view_t json = iree_make_string_view(no_merges, strlen(no_merges));

  iree_tokenizer_vocab_t* vocab = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_import_bpe_json(
      json, iree_allocator_system(), &vocab));
  EXPECT_EQ(iree_tokenizer_vocab_capacity(vocab), 3u);
  EXPECT_EQ(iree_tokenizer_vocab_merge_count(vocab), 0u);
  iree_tokenizer_vocab_free(vocab);
}

TEST(BpeJsonTest, InvalidMergeFormatFails) {
  // Merges with invalid format (no space) should fail at load time.
  const char* bad_merge = R"({
    "model": {
      "type": "BPE",
      "vocab": {"a": 0, "b": 1, "ab": 2},
      "merges": ["ab", "a b"]
    }
  })";
  iree_string_view_t json = iree_make_string_view(bad_merge, strlen(bad_merge));

  iree_tokenizer_vocab_t* vocab = nullptr;
  iree_status_t status = iree_tokenizer_vocab_import_bpe_json(
      json, iree_allocator_system(), &vocab);
  // The "ab" merge is invalid (no space separator) - should fail.
  EXPECT_TRUE(iree_status_is_invalid_argument(status));
  iree_status_ignore(status);
}

TEST(BpeJsonTest, MergeWithSpaceInToken) {
  // Tokens containing spaces should still be merged correctly.
  // The parser tries each space position to find valid token pairs.
  const char* space_token = R"({
    "model": {
      "type": "BPE",
      "vocab": {"a b": 0, "c": 1, "a b c": 2, "a": 3, "b c": 4}
    }
  })";
  iree_string_view_t json =
      iree_make_string_view(space_token, strlen(space_token));

  iree_tokenizer_vocab_t* vocab = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_import_bpe_json(
      json, iree_allocator_system(), &vocab));
  // Token "a b" contains a space and should be found.
  EXPECT_EQ(iree_tokenizer_vocab_lookup(vocab, IREE_SVL("a b")), 0);
  EXPECT_EQ(iree_tokenizer_vocab_lookup(vocab, IREE_SVL("b c")), 4);
  iree_tokenizer_vocab_free(vocab);
}

TEST(BpeJsonTest, MergeWithUnknownTokensFails) {
  // Merges referencing unknown tokens should fail at load time.
  const char* unknown_merge = R"({
    "model": {
      "type": "BPE",
      "vocab": {"a": 0, "b": 1, "ab": 2},
      "merges": ["a b", "x y"]
    }
  })";
  iree_string_view_t json =
      iree_make_string_view(unknown_merge, strlen(unknown_merge));

  iree_tokenizer_vocab_t* vocab = nullptr;
  iree_status_t status = iree_tokenizer_vocab_import_bpe_json(
      json, iree_allocator_system(), &vocab);
  EXPECT_TRUE(iree_status_is_invalid_argument(status));
  iree_status_ignore(status);
}

TEST(BpeJsonTest, OutOfRangeSpecialTokenIdSkipped) {
  // Special token with ID > INT32_MAX should be skipped.
  const char* huge_id = R"({
    "model": {
      "type": "BPE",
      "vocab": {"<|endoftext|>": 0}
    },
    "added_tokens": [
      {"id": 9999999999, "content": "<|huge|>", "special": true},
      {"id": 0, "content": "<|endoftext|>", "special": true}
    ]
  })";
  iree_string_view_t json = iree_make_string_view(huge_id, strlen(huge_id));

  iree_tokenizer_vocab_t* vocab = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_import_bpe_json(
      json, iree_allocator_system(), &vocab));
  iree_tokenizer_special_ids_t ids = iree_tokenizer_vocab_special_ids(vocab);
  EXPECT_EQ(ids.eos, 0);  // <|endoftext|> detected.
  iree_tokenizer_vocab_free(vocab);
}

TEST(BpeJsonTest, NegativeSpecialTokenIdSkipped) {
  // Special token with negative ID should be skipped.
  const char* neg_id = R"({
    "model": {
      "type": "BPE",
      "vocab": {"<|endoftext|>": 0}
    },
    "added_tokens": [
      {"id": -1, "content": "<|negative|>", "special": true},
      {"id": 0, "content": "<|endoftext|>", "special": true}
    ]
  })";
  iree_string_view_t json = iree_make_string_view(neg_id, strlen(neg_id));

  iree_tokenizer_vocab_t* vocab = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_import_bpe_json(
      json, iree_allocator_system(), &vocab));
  iree_tokenizer_special_ids_t ids = iree_tokenizer_vocab_special_ids(vocab);
  EXPECT_EQ(ids.eos, 0);  // <|endoftext|> detected.
  iree_tokenizer_vocab_free(vocab);
}

TEST(BpeJsonTest, OutOfRangeVocabTokenIdError) {
  // Token ID > INT32_MAX should cause an error.
  const char* huge_token_id = R"({
    "model": {
      "type": "BPE",
      "vocab": {"hello": 9999999999}
    }
  })";
  iree_string_view_t json =
      iree_make_string_view(huge_token_id, strlen(huge_token_id));

  iree_tokenizer_vocab_t* vocab = nullptr;
  iree_status_t status = iree_tokenizer_vocab_import_bpe_json(
      json, iree_allocator_system(), &vocab);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE, status);
  EXPECT_EQ(vocab, nullptr);
  iree_status_free(status);
}

TEST(BpeJsonTest, NegativeVocabTokenIdError) {
  // Negative token ID should cause an error.
  const char* neg_token_id = R"({
    "model": {
      "type": "BPE",
      "vocab": {"hello": -1}
    }
  })";
  iree_string_view_t json =
      iree_make_string_view(neg_token_id, strlen(neg_token_id));

  iree_tokenizer_vocab_t* vocab = nullptr;
  iree_status_t status = iree_tokenizer_vocab_import_bpe_json(
      json, iree_allocator_system(), &vocab);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE, status);
  EXPECT_EQ(vocab, nullptr);
  iree_status_free(status);
}

TEST(BpeJsonTest, MalformedJsonError) {
  // Completely malformed JSON should fail gracefully.
  const char* garbage = "not json at all {{{";
  iree_string_view_t json = iree_make_string_view(garbage, strlen(garbage));

  iree_tokenizer_vocab_t* vocab = nullptr;
  iree_status_t status = iree_tokenizer_vocab_import_bpe_json(
      json, iree_allocator_system(), &vocab);
  EXPECT_THAT(status, ::testing::Not(::iree::testing::status::IsOk()));
  EXPECT_EQ(vocab, nullptr);
  iree_status_free(status);
}

TEST(BpeJsonTest, EmptyInput) {
  // Empty input should fail.
  iree_string_view_t json = iree_string_view_empty();

  iree_tokenizer_vocab_t* vocab = nullptr;
  iree_status_t status = iree_tokenizer_vocab_import_bpe_json(
      json, iree_allocator_system(), &vocab);
  EXPECT_THAT(status, ::testing::Not(::iree::testing::status::IsOk()));
  EXPECT_EQ(vocab, nullptr);
  iree_status_free(status);
}

TEST(BpeJsonTest, UnicodeTokensPreserved) {
  // Unicode characters in tokens should be preserved correctly.
  const char* unicode_vocab = R"({
    "model": {
      "type": "BPE",
      "vocab": {"\u4e2d\u6587": 0, "\u00e9": 1, "\ud83d\ude00": 2}
    }
  })";
  iree_string_view_t json =
      iree_make_string_view(unicode_vocab, strlen(unicode_vocab));

  iree_tokenizer_vocab_t* vocab = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_import_bpe_json(
      json, iree_allocator_system(), &vocab));
  EXPECT_EQ(iree_tokenizer_vocab_capacity(vocab), 3u);

  // Look up Chinese characters.
  EXPECT_EQ(
      iree_tokenizer_vocab_lookup(vocab, IREE_SVL("\xe4\xb8\xad\xe6\x96\x87")),
      0);
  // Look up e with acute.
  EXPECT_EQ(iree_tokenizer_vocab_lookup(vocab, IREE_SVL("\xc3\xa9")), 1);
  // Look up emoji (grinning face).
  EXPECT_EQ(iree_tokenizer_vocab_lookup(vocab, IREE_SVL("\xf0\x9f\x98\x80")),
            2);

  iree_tokenizer_vocab_free(vocab);
}

TEST(BpeJsonTest, SpecialTokensHaveAttrSpecial) {
  iree_string_view_t json = GetTestFile("bpe_minimal.json");
  iree_tokenizer_vocab_t* vocab = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_import_bpe_json(
      json, iree_allocator_system(), &vocab));

  // Special tokens (from added_tokens with special: true) should have
  // ATTR_SPECIAL.
  EXPECT_TRUE(iree_tokenizer_vocab_token_attrs(vocab, 0) &
              IREE_TOKENIZER_TOKEN_ATTR_SPECIAL);  // <|endoftext|>
  EXPECT_TRUE(iree_tokenizer_vocab_token_attrs(vocab, 1) &
              IREE_TOKENIZER_TOKEN_ATTR_SPECIAL);  // <|pad|>

  // Regular tokens should NOT have ATTR_SPECIAL.
  EXPECT_FALSE(iree_tokenizer_vocab_token_attrs(vocab, 2) &
               IREE_TOKENIZER_TOKEN_ATTR_SPECIAL);  // Ġ
  EXPECT_FALSE(iree_tokenizer_vocab_token_attrs(vocab, 3) &
               IREE_TOKENIZER_TOKEN_ATTR_SPECIAL);  // h
  EXPECT_FALSE(iree_tokenizer_vocab_token_attrs(vocab, 20) &
               IREE_TOKENIZER_TOKEN_ATTR_SPECIAL);  // hello

  iree_tokenizer_vocab_free(vocab);
}

TEST(BpeJsonTest, AddedTokensNotInVocab) {
  // Test the edge case where added_tokens contains tokens NOT in model.vocab.
  // The token at ID 5 is only in added_tokens, not in vocab.
  const char* json_with_extra = R"({
    "model": {
      "type": "BPE",
      "vocab": {
        "<|endoftext|>": 0, "<|pad|>": 1, "a": 2, "b": 3, "ab": 4
      },
      "merges": ["a b"]
    },
    "added_tokens": [
      {"id": 0, "content": "<|endoftext|>", "special": true},
      {"id": 1, "content": "<|pad|>", "special": true},
      {"id": 5, "content": "<|extra|>", "special": true}
    ]
  })";
  iree_string_view_t json =
      iree_make_string_view(json_with_extra, strlen(json_with_extra));

  iree_tokenizer_vocab_t* vocab = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_import_bpe_json(
      json, iree_allocator_system(), &vocab));

  // Should have 6 tokens: 5 from vocab + 1 extra from added_tokens.
  EXPECT_EQ(iree_tokenizer_vocab_capacity(vocab), 6u);

  // Verify the extra token was added.
  EXPECT_EQ(iree_tokenizer_vocab_lookup(vocab, IREE_SVL("<|extra|>")), 5);

  // Verify it has ATTR_SPECIAL.
  EXPECT_TRUE(iree_tokenizer_vocab_token_attrs(vocab, 5) &
              IREE_TOKENIZER_TOKEN_ATTR_SPECIAL);

  // Verify merges still work correctly.
  EXPECT_EQ(iree_tokenizer_vocab_merge_count(vocab), 1u);
  iree_tokenizer_merge_t merge0 = iree_tokenizer_vocab_merge(vocab, 0);
  EXPECT_EQ(merge0.left_id, 2u);   // a
  EXPECT_EQ(merge0.right_id, 3u);  // b

  iree_tokenizer_vocab_free(vocab);
}

TEST(BpeJsonTest, AddedTokensContentMismatchError) {
  // Test that mismatched content between added_tokens and vocab fails.
  const char* mismatched = R"({
    "model": {
      "type": "BPE",
      "vocab": {
        "<|endoftext|>": 0, "<|pad|>": 1, "hello": 2
      }
    },
    "added_tokens": [
      {"id": 0, "content": "<|endoftext|>", "special": true},
      {"id": 2, "content": "WRONG", "special": false}
    ]
  })";
  iree_string_view_t json =
      iree_make_string_view(mismatched, strlen(mismatched));

  iree_tokenizer_vocab_t* vocab = nullptr;
  iree_status_t status = iree_tokenizer_vocab_import_bpe_json(
      json, iree_allocator_system(), &vocab);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
  EXPECT_EQ(vocab, nullptr);
  iree_status_free(status);
}

TEST(BpeJsonTest, DuplicateAddedTokensDeduped) {
  // Test that duplicate IDs in added_tokens are handled (first entry wins).
  const char* duplicates = R"({
    "model": {
      "type": "BPE",
      "vocab": {
        "<|endoftext|>": 0, "<|pad|>": 1
      }
    },
    "added_tokens": [
      {"id": 0, "content": "<|endoftext|>", "special": true},
      {"id": 0, "content": "<|endoftext|>", "special": false},
      {"id": 1, "content": "<|pad|>", "special": true}
    ]
  })";
  iree_string_view_t json =
      iree_make_string_view(duplicates, strlen(duplicates));

  iree_tokenizer_vocab_t* vocab = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_import_bpe_json(
      json, iree_allocator_system(), &vocab));

  // First entry had special=true, so <|endoftext|> should have ATTR_SPECIAL.
  EXPECT_TRUE(iree_tokenizer_vocab_token_attrs(vocab, 0) &
              IREE_TOKENIZER_TOKEN_ATTR_SPECIAL);

  iree_tokenizer_vocab_free(vocab);
}

TEST(BpeJsonTest, FallbackSpecialDetectionNoAddedTokens) {
  // Test that specials are detected from vocab when added_tokens is absent.
  const char* no_added = R"({
    "model": {
      "type": "BPE",
      "vocab": {
        "<|endoftext|>": 0, "<|pad|>": 1, "<unk>": 2, "hello": 3
      },
      "unk_token": "<unk>"
    }
  })";
  iree_string_view_t json = iree_make_string_view(no_added, strlen(no_added));

  iree_tokenizer_vocab_t* vocab = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_import_bpe_json(
      json, iree_allocator_system(), &vocab));

  // Special IDs should be detected from vocab patterns.
  iree_tokenizer_special_ids_t ids = iree_tokenizer_vocab_special_ids(vocab);
  EXPECT_EQ(ids.eos, 0);  // <|endoftext|>
  EXPECT_EQ(ids.pad, 1);  // <|pad|>
  EXPECT_EQ(ids.unk, 2);  // <unk>

  iree_tokenizer_vocab_free(vocab);
}

TEST(BpeJsonTest, FallbackUnkTokenFromModel) {
  // Test that model.unk_token is used to identify UNK even if pattern doesn't
  // match.
  const char* custom_unk = R"({
    "model": {
      "type": "BPE",
      "vocab": {
        "<CUSTOM_UNK>": 0, "hello": 1
      },
      "unk_token": "<CUSTOM_UNK>"
    }
  })";
  iree_string_view_t json =
      iree_make_string_view(custom_unk, strlen(custom_unk));

  iree_tokenizer_vocab_t* vocab = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_import_bpe_json(
      json, iree_allocator_system(), &vocab));

  // UNK should be detected from model.unk_token.
  iree_tokenizer_special_ids_t ids = iree_tokenizer_vocab_special_ids(vocab);
  EXPECT_EQ(ids.unk, 0);

  iree_tokenizer_vocab_free(vocab);
}

//===----------------------------------------------------------------------===//
// Array-Format Merge Tests
//
// Modern HuggingFace tokenizers use array-format merge rules [["a", "b"]]
// instead of legacy string format "a b". These tests verify support for both.
//===----------------------------------------------------------------------===//

TEST(BpeJsonTest, ArrayFormatMergeRules) {
  // Modern HuggingFace format uses arrays: [["a", "b"]] instead of ["a b"].
  // This is used by GPT-2, RoBERTa, CLIP, Mistral, Qwen, etc.
  const char* array_merges = R"({
    "model": {
      "type": "BPE",
      "vocab": {"a": 0, "b": 1, "ab": 2},
      "merges": [["a", "b"]]
    }
  })";
  iree_string_view_t json =
      iree_make_string_view(array_merges, strlen(array_merges));

  iree_tokenizer_vocab_t* vocab = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_import_bpe_json(
      json, iree_allocator_system(), &vocab));

  EXPECT_EQ(iree_tokenizer_vocab_merge_count(vocab), 1u);
  iree_tokenizer_merge_t merge0 = iree_tokenizer_vocab_merge(vocab, 0);
  EXPECT_EQ(merge0.left_id, 0u);   // a
  EXPECT_EQ(merge0.right_id, 1u);  // b

  iree_tokenizer_vocab_free(vocab);
}

TEST(BpeJsonTest, ArrayFormatMultipleMerges) {
  // Multiple array-format merges in sequence.
  const char* multi_merges = R"({
    "model": {
      "type": "BPE",
      "vocab": {"h": 0, "e": 1, "l": 2, "o": 3, "he": 4, "ll": 5, "hello": 6},
      "merges": [["h", "e"], ["l", "l"], ["he", "ll"]]
    }
  })";
  iree_string_view_t json =
      iree_make_string_view(multi_merges, strlen(multi_merges));

  iree_tokenizer_vocab_t* vocab = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_import_bpe_json(
      json, iree_allocator_system(), &vocab));

  EXPECT_EQ(iree_tokenizer_vocab_merge_count(vocab), 3u);

  // h + e -> he
  iree_tokenizer_merge_t merge0 = iree_tokenizer_vocab_merge(vocab, 0);
  EXPECT_EQ(merge0.left_id, 0u);
  EXPECT_EQ(merge0.right_id, 1u);

  // l + l -> ll
  iree_tokenizer_merge_t merge1 = iree_tokenizer_vocab_merge(vocab, 1);
  EXPECT_EQ(merge1.left_id, 2u);
  EXPECT_EQ(merge1.right_id, 2u);

  // he + ll -> hell
  iree_tokenizer_merge_t merge2 = iree_tokenizer_vocab_merge(vocab, 2);
  EXPECT_EQ(merge2.left_id, 4u);
  EXPECT_EQ(merge2.right_id, 5u);

  iree_tokenizer_vocab_free(vocab);
}

TEST(BpeJsonTest, ArrayFormatWithUnicodeEscapes) {
  // GPT-2/RoBERTa use \u0120 (Ġ) for space prefix. Test Unicode in arrays.
  const char* unicode_merges = R"({
    "model": {
      "type": "BPE",
      "vocab": {"\u0120": 0, "t": 1, "\u0120t": 2, "\u00e9": 3, "s": 4, "\u00e9s": 5},
      "merges": [["\u0120", "t"], ["\u00e9", "s"]]
    }
  })";
  iree_string_view_t json =
      iree_make_string_view(unicode_merges, strlen(unicode_merges));

  iree_tokenizer_vocab_t* vocab = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_import_bpe_json(
      json, iree_allocator_system(), &vocab));

  EXPECT_EQ(iree_tokenizer_vocab_merge_count(vocab), 2u);

  // Ġ + t -> Ġt
  iree_tokenizer_merge_t merge0 = iree_tokenizer_vocab_merge(vocab, 0);
  EXPECT_EQ(merge0.left_id, 0u);
  EXPECT_EQ(merge0.right_id, 1u);

  // é + s -> és
  iree_tokenizer_merge_t merge1 = iree_tokenizer_vocab_merge(vocab, 1);
  EXPECT_EQ(merge1.left_id, 3u);
  EXPECT_EQ(merge1.right_id, 4u);

  iree_tokenizer_vocab_free(vocab);
}

TEST(BpeJsonTest, ArrayFormatWithSpaceInToken) {
  // Tokens containing literal spaces in array format.
  const char* space_token = R"({
    "model": {
      "type": "BPE",
      "vocab": {"a b": 0, "c": 1, "a b c": 2},
      "merges": [["a b", "c"]]
    }
  })";
  iree_string_view_t json =
      iree_make_string_view(space_token, strlen(space_token));

  iree_tokenizer_vocab_t* vocab = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_import_bpe_json(
      json, iree_allocator_system(), &vocab));

  EXPECT_EQ(iree_tokenizer_vocab_merge_count(vocab), 1u);
  iree_tokenizer_merge_t merge0 = iree_tokenizer_vocab_merge(vocab, 0);
  EXPECT_EQ(merge0.left_id, 0u);   // "a b"
  EXPECT_EQ(merge0.right_id, 1u);  // "c"

  iree_tokenizer_vocab_free(vocab);
}

TEST(BpeJsonTest, MixedMergeFormats) {
  // Some tokenizers might theoretically mix formats - handle both gracefully.
  const char* mixed = R"({
    "model": {
      "type": "BPE",
      "vocab": {"a": 0, "b": 1, "c": 2, "ab": 3, "bc": 4},
      "merges": ["a b", ["b", "c"]]
    }
  })";
  iree_string_view_t json = iree_make_string_view(mixed, strlen(mixed));

  iree_tokenizer_vocab_t* vocab = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_import_bpe_json(
      json, iree_allocator_system(), &vocab));

  EXPECT_EQ(iree_tokenizer_vocab_merge_count(vocab), 2u);

  // String format: a + b -> ab
  iree_tokenizer_merge_t merge0 = iree_tokenizer_vocab_merge(vocab, 0);
  EXPECT_EQ(merge0.left_id, 0u);
  EXPECT_EQ(merge0.right_id, 1u);

  // Array format: b + c -> bc
  iree_tokenizer_merge_t merge1 = iree_tokenizer_vocab_merge(vocab, 1);
  EXPECT_EQ(merge1.left_id, 1u);
  EXPECT_EQ(merge1.right_id, 2u);

  iree_tokenizer_vocab_free(vocab);
}

TEST(BpeJsonTest, ArrayFormatEmptyFirstToken) {
  // Edge case: empty string as first token in merge.
  const char* empty_first = R"({
    "model": {
      "type": "BPE",
      "vocab": {"": 0, "a": 1, "a": 2},
      "merges": [["", "a"]]
    }
  })";
  iree_string_view_t json =
      iree_make_string_view(empty_first, strlen(empty_first));

  iree_tokenizer_vocab_t* vocab = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_import_bpe_json(
      json, iree_allocator_system(), &vocab));

  EXPECT_EQ(iree_tokenizer_vocab_merge_count(vocab), 1u);
  iree_tokenizer_merge_t merge0 = iree_tokenizer_vocab_merge(vocab, 0);
  EXPECT_EQ(merge0.left_id, 0u);   // ""
  EXPECT_EQ(merge0.right_id, 1u);  // "a"

  iree_tokenizer_vocab_free(vocab);
}

TEST(BpeJsonTest, ArrayFormatEmptySecondToken) {
  // Edge case: empty string as second token in merge.
  const char* empty_second = R"({
    "model": {
      "type": "BPE",
      "vocab": {"a": 0, "": 1, "a": 2},
      "merges": [["a", ""]]
    }
  })";
  iree_string_view_t json =
      iree_make_string_view(empty_second, strlen(empty_second));

  iree_tokenizer_vocab_t* vocab = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_import_bpe_json(
      json, iree_allocator_system(), &vocab));

  EXPECT_EQ(iree_tokenizer_vocab_merge_count(vocab), 1u);
  iree_tokenizer_merge_t merge0 = iree_tokenizer_vocab_merge(vocab, 0);
  EXPECT_EQ(merge0.left_id, 0u);   // "a"
  EXPECT_EQ(merge0.right_id, 1u);  // ""

  iree_tokenizer_vocab_free(vocab);
}

TEST(BpeJsonTest, ArrayFormatSingleElementError) {
  // Array with only one element should fail - merges need exactly two tokens.
  const char* single_element = R"({
    "model": {
      "type": "BPE",
      "vocab": {"a": 0, "b": 1},
      "merges": [["a"]]
    }
  })";
  iree_string_view_t json =
      iree_make_string_view(single_element, strlen(single_element));

  iree_tokenizer_vocab_t* vocab = nullptr;
  iree_status_t status = iree_tokenizer_vocab_import_bpe_json(
      json, iree_allocator_system(), &vocab);
  EXPECT_TRUE(iree_status_is_invalid_argument(status));
  iree_status_ignore(status);
}

TEST(BpeJsonTest, ArrayFormatThreeElementsError) {
  // Array with three elements should fail - merges need exactly two tokens.
  const char* three_elements = R"({
    "model": {
      "type": "BPE",
      "vocab": {"a": 0, "b": 1, "c": 2},
      "merges": [["a", "b", "c"]]
    }
  })";
  iree_string_view_t json =
      iree_make_string_view(three_elements, strlen(three_elements));

  iree_tokenizer_vocab_t* vocab = nullptr;
  iree_status_t status = iree_tokenizer_vocab_import_bpe_json(
      json, iree_allocator_system(), &vocab);
  EXPECT_TRUE(iree_status_is_invalid_argument(status));
  iree_status_ignore(status);
}

TEST(BpeJsonTest, ArrayFormatUnknownTokensFails) {
  // Array merge referencing unknown tokens should fail.
  const char* unknown_merge = R"({
    "model": {
      "type": "BPE",
      "vocab": {"a": 0, "b": 1, "ab": 2},
      "merges": [["a", "b"], ["x", "y"]]
    }
  })";
  iree_string_view_t json =
      iree_make_string_view(unknown_merge, strlen(unknown_merge));

  iree_tokenizer_vocab_t* vocab = nullptr;
  iree_status_t status = iree_tokenizer_vocab_import_bpe_json(
      json, iree_allocator_system(), &vocab);
  EXPECT_TRUE(iree_status_is_invalid_argument(status));
  iree_status_ignore(status);
}

TEST(BpeJsonTest, ArrayFormatEmptyArrayError) {
  // Empty array [] as a merge should fail.
  const char* empty_array = R"({
    "model": {
      "type": "BPE",
      "vocab": {"a": 0, "b": 1},
      "merges": [[]]
    }
  })";
  iree_string_view_t json =
      iree_make_string_view(empty_array, strlen(empty_array));

  iree_tokenizer_vocab_t* vocab = nullptr;
  iree_status_t status = iree_tokenizer_vocab_import_bpe_json(
      json, iree_allocator_system(), &vocab);
  EXPECT_TRUE(iree_status_is_invalid_argument(status));
  iree_status_ignore(status);
}

TEST(BpeJsonTest, StringFormatMergeStartingWithBracket) {
  // String-format merges where the token itself is "[" (common in code
  // tokenizers like Qwen2). The parser must distinguish between:
  //   - Array format: ["a", "b"] → starts with `[` (JSON array)
  //   - String format: "[ i" → starts with `"` (JSON string)
  // Using JSON lexical type detection: check first char of raw JSON element.
  // Note: Using R"json(...)json" delimiter because JSON contains ")".
  const char* bracket_merges = R"json({
    "model": {
      "type": "BPE",
      "vocab": {"[": 0, "i": 1, "[ i": 2, "]": 3, "] x": 4, "x": 5},
      "merges": ["[ i", "] x"]
    }
  })json";
  iree_string_view_t json =
      iree_make_string_view(bracket_merges, strlen(bracket_merges));

  iree_tokenizer_vocab_t* vocab = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_import_bpe_json(
      json, iree_allocator_system(), &vocab));

  EXPECT_EQ(iree_tokenizer_vocab_merge_count(vocab), 2u);

  // "[" + "i" -> "[ i" (string-format merge starting with "[")
  iree_tokenizer_merge_t merge0 = iree_tokenizer_vocab_merge(vocab, 0);
  EXPECT_EQ(merge0.left_id, 0u);   // "["
  EXPECT_EQ(merge0.right_id, 1u);  // "i"

  // "]" + "x" -> "] x"
  iree_tokenizer_merge_t merge1 = iree_tokenizer_vocab_merge(vocab, 1);
  EXPECT_EQ(merge1.left_id, 3u);   // "]"
  EXPECT_EQ(merge1.right_id, 5u);  // "x"

  iree_tokenizer_vocab_free(vocab);
}

TEST(BpeJsonTest, ArrayFormatWithPrettyPrintedWhitespace) {
  // Pretty-printed JSON with whitespace after "[" (like Qwen3).
  // The parser must handle:
  //   [ "a", "b" ] and [\n  "a",\n  "b"\n] as array format
  const char* pretty_merges = R"({
    "model": {
      "type": "BPE",
      "vocab": {"a": 0, "b": 1, "ab": 2, "c": 3, "d": 4, "cd": 5},
      "merges": [
        [
          "a",
          "b"
        ],
        [ "c", "d" ]
      ]
    }
  })";
  iree_string_view_t json =
      iree_make_string_view(pretty_merges, strlen(pretty_merges));

  iree_tokenizer_vocab_t* vocab = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_import_bpe_json(
      json, iree_allocator_system(), &vocab));

  EXPECT_EQ(iree_tokenizer_vocab_merge_count(vocab), 2u);

  // a + b -> ab
  iree_tokenizer_merge_t merge0 = iree_tokenizer_vocab_merge(vocab, 0);
  EXPECT_EQ(merge0.left_id, 0u);
  EXPECT_EQ(merge0.right_id, 1u);

  // c + d -> cd
  iree_tokenizer_merge_t merge1 = iree_tokenizer_vocab_merge(vocab, 1);
  EXPECT_EQ(merge1.left_id, 3u);
  EXPECT_EQ(merge1.right_id, 4u);

  iree_tokenizer_vocab_free(vocab);
}

//===----------------------------------------------------------------------===//
// Factory Tests (iree_tokenizer_from_bpe_json)
//===----------------------------------------------------------------------===//

TEST(BpeJsonFactoryTest, CreateTokenizerFromMinimalJson) {
  iree_string_view_t json = GetTestFile("bpe_minimal.json");
  ASSERT_GT(json.size, 0u);

  iree_tokenizer_t* tokenizer = nullptr;
  IREE_ASSERT_OK(
      iree_tokenizer_from_bpe_json(json, iree_allocator_system(), &tokenizer));
  ASSERT_NE(tokenizer, nullptr);

  // Verify vocab is accessible.
  const iree_tokenizer_vocab_t* vocab = iree_tokenizer_vocab(tokenizer);
  ASSERT_NE(vocab, nullptr);
  EXPECT_EQ(iree_tokenizer_vocab_capacity(vocab), 21u);

  iree_tokenizer_free(tokenizer);
}

TEST(BpeJsonFactoryTest, CreateTokenizerWithPreTokenizer) {
  // Test with a pre_tokenizer configuration.
  const char* with_pretokenizer = R"({
    "model": {
      "type": "BPE",
      "vocab": {"hello": 0, "world": 1}
    },
    "pre_tokenizer": {
      "type": "Whitespace"
    }
  })";
  iree_string_view_t json =
      iree_make_string_view(with_pretokenizer, strlen(with_pretokenizer));

  iree_tokenizer_t* tokenizer = nullptr;
  IREE_ASSERT_OK(
      iree_tokenizer_from_bpe_json(json, iree_allocator_system(), &tokenizer));
  ASSERT_NE(tokenizer, nullptr);

  iree_tokenizer_free(tokenizer);
}

TEST(BpeJsonFactoryTest, CreateTokenizerWithDecoder) {
  // Test with a decoder configuration.
  const char* with_decoder = R"({
    "model": {
      "type": "BPE",
      "vocab": {"\u0120hello": 0, "\u0120world": 1}
    },
    "decoder": {
      "type": "ByteLevel"
    }
  })";
  iree_string_view_t json =
      iree_make_string_view(with_decoder, strlen(with_decoder));

  iree_tokenizer_t* tokenizer = nullptr;
  IREE_ASSERT_OK(
      iree_tokenizer_from_bpe_json(json, iree_allocator_system(), &tokenizer));
  ASSERT_NE(tokenizer, nullptr);

  iree_tokenizer_free(tokenizer);
}

TEST(BpeJsonFactoryTest, CreateTokenizerWithPostProcessor) {
  // Test with a post_processor configuration.
  const char* with_postprocessor = R"({
    "model": {
      "type": "BPE",
      "vocab": {"<s>": 0, "</s>": 1, "hello": 2}
    },
    "added_tokens": [
      {"id": 0, "content": "<s>", "special": true},
      {"id": 1, "content": "</s>", "special": true}
    ],
    "post_processor": {
      "type": "TemplateProcessing",
      "single": [{"SpecialToken": {"id": "<s>", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}}, {"SpecialToken": {"id": "</s>", "type_id": 0}}],
      "pair": null,
      "special_tokens": {"<s>": {"id": "<s>", "ids": [0], "tokens": ["<s>"]}, "</s>": {"id": "</s>", "ids": [1], "tokens": ["</s>"]}}
    }
  })";
  iree_string_view_t json =
      iree_make_string_view(with_postprocessor, strlen(with_postprocessor));

  iree_tokenizer_t* tokenizer = nullptr;
  IREE_ASSERT_OK(
      iree_tokenizer_from_bpe_json(json, iree_allocator_system(), &tokenizer));
  ASSERT_NE(tokenizer, nullptr);

  iree_tokenizer_free(tokenizer);
}

TEST(BpeJsonFactoryTest, InvalidModelTypeFails) {
  // Factory should fail with wrong model type.
  const char* wordpiece = R"({
    "model": {
      "type": "WordPiece",
      "vocab": {"hello": 0}
    }
  })";
  iree_string_view_t json = iree_make_string_view(wordpiece, strlen(wordpiece));

  iree_tokenizer_t* tokenizer = nullptr;
  iree_status_t status =
      iree_tokenizer_from_bpe_json(json, iree_allocator_system(), &tokenizer);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
  EXPECT_EQ(tokenizer, nullptr);
  iree_status_free(status);
}

TEST(BpeJsonFactoryTest, MalformedJsonFails) {
  const char* garbage = "not json {{{";
  iree_string_view_t json = iree_make_string_view(garbage, strlen(garbage));

  iree_tokenizer_t* tokenizer = nullptr;
  iree_status_t status =
      iree_tokenizer_from_bpe_json(json, iree_allocator_system(), &tokenizer);
  EXPECT_THAT(status, ::testing::Not(::iree::testing::status::IsOk()));
  EXPECT_EQ(tokenizer, nullptr);
  iree_status_free(status);
}

TEST(BpeJsonFactoryTest, EncodeDecodeRoundTrip) {
  iree_string_view_t json = GetTestFile("bpe_minimal.json");
  ASSERT_GT(json.size, 0u);

  iree_tokenizer_t* tokenizer = nullptr;
  IREE_ASSERT_OK(
      iree_tokenizer_from_bpe_json(json, iree_allocator_system(), &tokenizer));

  // Encode "hello" - a single word the vocab supports.
  // Note: The minimal vocab doesn't have space so we test single words.
  // BPE may produce subword tokens depending on merges.
  int32_t ids[16];
  iree_host_size_t count = 0;
  iree_tokenizer_encode_options_t options = {0};
  IREE_ASSERT_OK(iree_tokenizer_encode(tokenizer, IREE_SV("hello"), options,
                                       ids, IREE_ARRAYSIZE(ids), &count));
  ASSERT_GT(count, 0u);

  // Decode back.
  char text[256];
  iree_host_size_t length = 0;
  IREE_ASSERT_OK(iree_tokenizer_decode(tokenizer, ids, count,
                                       IREE_TOKENIZER_DECODE_FLAG_DEFAULT, text,
                                       sizeof(text), &length));

  // Decoded text should match original input.
  EXPECT_EQ(length, 5u);
  EXPECT_EQ(std::string(text, length), "hello");

  iree_tokenizer_free(tokenizer);
}

TEST(BpeJsonFactoryTest, VocabOwnershipOnSuccess) {
  // Verify vocab is owned by tokenizer and freed correctly.
  const char* minimal = R"({
    "model": {
      "type": "BPE",
      "vocab": {"a": 0, "b": 1}
    }
  })";
  iree_string_view_t json = iree_make_string_view(minimal, strlen(minimal));

  iree_tokenizer_t* tokenizer = nullptr;
  IREE_ASSERT_OK(
      iree_tokenizer_from_bpe_json(json, iree_allocator_system(), &tokenizer));

  // Access vocab through tokenizer.
  const iree_tokenizer_vocab_t* vocab = iree_tokenizer_vocab(tokenizer);
  EXPECT_NE(vocab, nullptr);
  EXPECT_EQ(iree_tokenizer_vocab_capacity(vocab), 2u);

  // Free tokenizer - this should also free the vocab.
  // (No double-free or leak should occur.)
  iree_tokenizer_free(tokenizer);
}

//===----------------------------------------------------------------------===//
// Sparse ID Tests
//
// Some tokenizers have non-sequential/sparse vocabulary IDs. The vocab
// rebuild process must preserve original IDs so merge rules remain valid.
//===----------------------------------------------------------------------===//

TEST(BpeJsonTest, SparseVocabIDsPreserved) {
  // Test vocabulary with non-sequential IDs (sparse vocab).
  // This is common with added_tokens that have IDs beyond vocab.size().
  // The loader must use add_token_with_id to preserve original IDs.
  const char* sparse_vocab = R"({
    "model": {
      "type": "BPE",
      "vocab": {
        "a": 0,
        "b": 5,
        "c": 10,
        "ab": 15
      },
      "merges": ["a b"]
    }
  })";
  iree_string_view_t json =
      iree_make_string_view(sparse_vocab, strlen(sparse_vocab));

  iree_tokenizer_vocab_t* vocab = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_import_bpe_json(
      json, iree_allocator_system(), &vocab));

  // Vocabulary should span IDs 0-15 (max ID + 1).
  EXPECT_EQ(iree_tokenizer_vocab_capacity(vocab), 16u);

  // Verify original IDs are preserved via lookup.
  EXPECT_EQ(iree_tokenizer_vocab_lookup(vocab, IREE_SVL("a")), 0);
  EXPECT_EQ(iree_tokenizer_vocab_lookup(vocab, IREE_SVL("b")), 5);
  EXPECT_EQ(iree_tokenizer_vocab_lookup(vocab, IREE_SVL("c")), 10);
  EXPECT_EQ(iree_tokenizer_vocab_lookup(vocab, IREE_SVL("ab")), 15);

  // Verify token text at original IDs.
  iree_string_view_t text_a = iree_tokenizer_vocab_token_text(vocab, 0);
  iree_string_view_t text_b = iree_tokenizer_vocab_token_text(vocab, 5);
  iree_string_view_t text_c = iree_tokenizer_vocab_token_text(vocab, 10);
  iree_string_view_t text_ab = iree_tokenizer_vocab_token_text(vocab, 15);
  EXPECT_EQ(iree_string_view_compare(text_a, IREE_SVL("a")), 0);
  EXPECT_EQ(iree_string_view_compare(text_b, IREE_SVL("b")), 0);
  EXPECT_EQ(iree_string_view_compare(text_c, IREE_SVL("c")), 0);
  EXPECT_EQ(iree_string_view_compare(text_ab, IREE_SVL("ab")), 0);

  // CRITICAL: Merge rules must reference original IDs.
  // Using add_token (sequential IDs) instead of add_token_with_id would cause
  // merge "a b" to incorrectly reference IDs 0,1 instead of 0,5.
  EXPECT_EQ(iree_tokenizer_vocab_merge_count(vocab), 1u);
  iree_tokenizer_merge_t merge0 = iree_tokenizer_vocab_merge(vocab, 0);
  EXPECT_EQ(merge0.left_id, 0u);   // "a" at ID 0
  EXPECT_EQ(merge0.right_id, 5u);  // "b" at ID 5 (NOT ID 1!)

  iree_tokenizer_vocab_free(vocab);
}

TEST(BpeJsonTest, SparseVocabWithAddedTokens) {
  // Test sparse vocab where added_tokens extend beyond model.vocab range.
  const char* sparse_with_added = R"({
    "model": {
      "type": "BPE",
      "vocab": {
        "a": 0,
        "b": 1,
        "ab": 2
      },
      "merges": ["a b"]
    },
    "added_tokens": [
      {"id": 0, "content": "a", "special": false},
      {"id": 1, "content": "b", "special": false},
      {"id": 100, "content": "<|special|>", "special": true}
    ]
  })";
  iree_string_view_t json =
      iree_make_string_view(sparse_with_added, strlen(sparse_with_added));

  iree_tokenizer_vocab_t* vocab = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_import_bpe_json(
      json, iree_allocator_system(), &vocab));

  // Vocabulary size should be 101 (0 to 100 inclusive).
  EXPECT_EQ(iree_tokenizer_vocab_capacity(vocab), 101u);

  // Verify the added token at ID 100.
  EXPECT_EQ(iree_tokenizer_vocab_lookup(vocab, IREE_SVL("<|special|>")), 100);
  iree_string_view_t text_100 = iree_tokenizer_vocab_token_text(vocab, 100);
  EXPECT_EQ(iree_string_view_compare(text_100, IREE_SVL("<|special|>")), 0);

  // Verify ATTR_SPECIAL is set.
  EXPECT_TRUE(iree_tokenizer_vocab_token_attrs(vocab, 100) &
              IREE_TOKENIZER_TOKEN_ATTR_SPECIAL);

  // Verify merge rules still reference correct IDs.
  EXPECT_EQ(iree_tokenizer_vocab_merge_count(vocab), 1u);
  iree_tokenizer_merge_t merge0 = iree_tokenizer_vocab_merge(vocab, 0);
  EXPECT_EQ(merge0.left_id, 0u);   // "a"
  EXPECT_EQ(merge0.right_id, 1u);  // "b"

  iree_tokenizer_vocab_free(vocab);
}

TEST(BpeJsonTest, SparseVocabMergesWithGaps) {
  // Test merge rules in a vocab with gaps.
  // The merged token ("ab") is at a non-sequential ID.
  const char* gaps_vocab = R"({
    "model": {
      "type": "BPE",
      "vocab": {
        "a": 0,
        "b": 1,
        "c": 2,
        "ab": 10,
        "abc": 20
      },
      "merges": ["a b", "ab c"]
    }
  })";
  iree_string_view_t json =
      iree_make_string_view(gaps_vocab, strlen(gaps_vocab));

  iree_tokenizer_vocab_t* vocab = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_import_bpe_json(
      json, iree_allocator_system(), &vocab));

  // Check merge rules reference correct IDs (including the gap at ID 10).
  EXPECT_EQ(iree_tokenizer_vocab_merge_count(vocab), 2u);

  // First merge: a(0) + b(1) -> ab(10)
  iree_tokenizer_merge_t merge0 = iree_tokenizer_vocab_merge(vocab, 0);
  EXPECT_EQ(merge0.left_id, 0u);
  EXPECT_EQ(merge0.right_id, 1u);

  // Second merge: ab(10) + c(2) -> abc(20)
  // This is the critical test - the merge must use ID 10, not a sequential ID.
  iree_tokenizer_merge_t merge1 = iree_tokenizer_vocab_merge(vocab, 1);
  EXPECT_EQ(merge1.left_id, 10u);  // "ab" at ID 10 (NOT ID 3!)
  EXPECT_EQ(merge1.right_id, 2u);  // "c"

  iree_tokenizer_vocab_free(vocab);
}

TEST(BpeJsonFactoryTest, SparseVocabEncodeDecodeRoundTrip) {
  // End-to-end test: sparse vocab with BPE encoding/decoding.
  const char* sparse_json = R"({
    "model": {
      "type": "BPE",
      "vocab": {
        "h": 0,
        "e": 5,
        "l": 10,
        "o": 15,
        "he": 20,
        "ll": 25,
        "lo": 30,
        "hel": 35,
        "llo": 40,
        "hello": 45
      },
      "merges": [
        "h e",
        "l l",
        "l o",
        "he l",
        "ll o",
        "hel lo"
      ]
    }
  })";
  iree_string_view_t json =
      iree_make_string_view(sparse_json, strlen(sparse_json));

  iree_tokenizer_t* tokenizer = nullptr;
  IREE_ASSERT_OK(
      iree_tokenizer_from_bpe_json(json, iree_allocator_system(), &tokenizer));

  // Verify vocab size accounts for gaps.
  const iree_tokenizer_vocab_t* vocab = iree_tokenizer_vocab(tokenizer);
  EXPECT_EQ(iree_tokenizer_vocab_capacity(vocab), 46u);  // 0-45 inclusive

  // Encode "hello" - should produce single token 45 after all merges.
  int32_t ids[16];
  iree_host_size_t count = 0;
  iree_tokenizer_encode_options_t options = {0};
  IREE_ASSERT_OK(iree_tokenizer_encode(tokenizer, IREE_SV("hello"), options,
                                       ids, IREE_ARRAYSIZE(ids), &count));
  EXPECT_EQ(count, 1u);
  EXPECT_EQ(ids[0], 45);  // "hello" token at sparse ID 45

  // Decode back.
  char text[256];
  iree_host_size_t length = 0;
  IREE_ASSERT_OK(iree_tokenizer_decode(tokenizer, ids, count,
                                       IREE_TOKENIZER_DECODE_FLAG_DEFAULT, text,
                                       sizeof(text), &length));
  EXPECT_EQ(length, 5u);
  EXPECT_EQ(std::string(text, length), "hello");

  iree_tokenizer_free(tokenizer);
}

}  // namespace
