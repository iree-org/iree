// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/huggingface/wordpiece_json.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/tokenizer/huggingface/testdata/wordpiece_testdata.h"

namespace {

static iree_string_view_t GetTestFile(const char* name) {
  const struct iree_file_toc_t* file_toc =
      iree_tokenizer_wordpiece_testdata_create();
  for (size_t i = 0; i < iree_tokenizer_wordpiece_testdata_size(); ++i) {
    if (strcmp(file_toc[i].name, name) == 0) {
      return iree_make_string_view((const char*)file_toc[i].data,
                                   file_toc[i].size);
    }
  }
  return iree_string_view_empty();
}

TEST(WordPieceJsonTest, LoadMinimalVocab) {
  iree_string_view_t json = GetTestFile("wordpiece_minimal.json");
  ASSERT_GT(json.size, 0u);

  iree_tokenizer_vocab_t* vocab = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_import_wordpiece_json(
      json, iree_allocator_system(), &vocab));
  ASSERT_NE(vocab, nullptr);

  EXPECT_EQ(iree_tokenizer_vocab_capacity(vocab), 10u);

  iree_tokenizer_vocab_free(vocab);
}

TEST(WordPieceJsonTest, VocabLookupHit) {
  iree_string_view_t json = GetTestFile("wordpiece_minimal.json");
  iree_tokenizer_vocab_t* vocab = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_import_wordpiece_json(
      json, iree_allocator_system(), &vocab));

  EXPECT_EQ(iree_tokenizer_vocab_lookup(vocab, IREE_SVL("[PAD]")), 0);
  EXPECT_EQ(iree_tokenizer_vocab_lookup(vocab, IREE_SVL("[UNK]")), 1);
  EXPECT_EQ(iree_tokenizer_vocab_lookup(vocab, IREE_SVL("hello")), 5);
  EXPECT_EQ(iree_tokenizer_vocab_lookup(vocab, IREE_SVL("world")), 6);
  EXPECT_EQ(iree_tokenizer_vocab_lookup(vocab, IREE_SVL("##ing")), 7);

  iree_tokenizer_vocab_free(vocab);
}

TEST(WordPieceJsonTest, VocabLookupMiss) {
  iree_string_view_t json = GetTestFile("wordpiece_minimal.json");
  iree_tokenizer_vocab_t* vocab = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_import_wordpiece_json(
      json, iree_allocator_system(), &vocab));

  EXPECT_EQ(iree_tokenizer_vocab_lookup(vocab, IREE_SVL("notfound")), -1);
  EXPECT_EQ(iree_tokenizer_vocab_lookup(vocab, IREE_SVL("HELLO")),
            -1);  // Case-sensitive.

  iree_tokenizer_vocab_free(vocab);
}

TEST(WordPieceJsonTest, SpecialTokensDetected) {
  iree_string_view_t json = GetTestFile("wordpiece_minimal.json");
  iree_tokenizer_vocab_t* vocab = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_import_wordpiece_json(
      json, iree_allocator_system(), &vocab));

  iree_tokenizer_special_ids_t ids = iree_tokenizer_vocab_special_ids(vocab);
  EXPECT_EQ(ids.pad, 0);
  EXPECT_EQ(ids.unk, 1);
  EXPECT_EQ(ids.cls, 2);
  EXPECT_EQ(ids.sep, 3);
  EXPECT_EQ(ids.mask, 4);
  EXPECT_EQ(ids.bos, -1);  // Not in test file.
  EXPECT_EQ(ids.eos, -1);  // Not in test file.

  iree_tokenizer_vocab_free(vocab);
}

TEST(WordPieceJsonTest, TokenTextRetrieval) {
  iree_string_view_t json = GetTestFile("wordpiece_minimal.json");
  iree_tokenizer_vocab_t* vocab = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_import_wordpiece_json(
      json, iree_allocator_system(), &vocab));

  iree_string_view_t text5 = iree_tokenizer_vocab_token_text(vocab, 5);
  iree_string_view_t text7 = iree_tokenizer_vocab_token_text(vocab, 7);
  EXPECT_EQ(iree_string_view_compare(text5, IREE_SVL("hello")), 0);
  EXPECT_EQ(iree_string_view_compare(text7, IREE_SVL("##ing")), 0);

  iree_tokenizer_vocab_free(vocab);
}

TEST(WordPieceJsonTest, WrongModelTypeError) {
  // BPE model type instead of WordPiece.
  const char* bpe_json = R"({
    "model": {
      "type": "BPE",
      "vocab": {"hello": 0}
    }
  })";
  iree_string_view_t json = iree_make_string_view(bpe_json, strlen(bpe_json));

  iree_tokenizer_vocab_t* vocab = nullptr;
  iree_status_t status = iree_tokenizer_vocab_import_wordpiece_json(
      json, iree_allocator_system(), &vocab);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
  EXPECT_EQ(vocab, nullptr);
  iree_status_free(status);
}

TEST(WordPieceJsonTest, MissingModelError) {
  const char* no_model = R"({"added_tokens": []})";
  iree_string_view_t json = iree_make_string_view(no_model, strlen(no_model));

  iree_tokenizer_vocab_t* vocab = nullptr;
  iree_status_t status = iree_tokenizer_vocab_import_wordpiece_json(
      json, iree_allocator_system(), &vocab);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND, status);
  EXPECT_EQ(vocab, nullptr);
  iree_status_free(status);
}

TEST(WordPieceJsonTest, MissingVocabError) {
  const char* no_vocab = R"({
    "model": {
      "type": "WordPiece"
    }
  })";
  iree_string_view_t json = iree_make_string_view(no_vocab, strlen(no_vocab));

  iree_tokenizer_vocab_t* vocab = nullptr;
  iree_status_t status = iree_tokenizer_vocab_import_wordpiece_json(
      json, iree_allocator_system(), &vocab);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND, status);
  EXPECT_EQ(vocab, nullptr);
  iree_status_free(status);
}

TEST(WordPieceJsonTest, EmptyVocab) {
  const char* empty_vocab = R"({
    "model": {
      "type": "WordPiece",
      "vocab": {}
    }
  })";
  iree_string_view_t json =
      iree_make_string_view(empty_vocab, strlen(empty_vocab));

  iree_tokenizer_vocab_t* vocab = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_import_wordpiece_json(
      json, iree_allocator_system(), &vocab));
  EXPECT_EQ(iree_tokenizer_vocab_capacity(vocab), 0u);
  iree_tokenizer_vocab_free(vocab);
}

TEST(WordPieceJsonTest, SpecialTokensHaveAttrSpecial) {
  iree_string_view_t json = GetTestFile("wordpiece_minimal.json");
  iree_tokenizer_vocab_t* vocab = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_import_wordpiece_json(
      json, iree_allocator_system(), &vocab));

  // Special tokens (from added_tokens with special: true) should have
  // ATTR_SPECIAL.
  EXPECT_TRUE(iree_tokenizer_vocab_token_attrs(vocab, 0) &
              IREE_TOKENIZER_TOKEN_ATTR_SPECIAL);  // [PAD]
  EXPECT_TRUE(iree_tokenizer_vocab_token_attrs(vocab, 1) &
              IREE_TOKENIZER_TOKEN_ATTR_SPECIAL);  // [UNK]
  EXPECT_TRUE(iree_tokenizer_vocab_token_attrs(vocab, 2) &
              IREE_TOKENIZER_TOKEN_ATTR_SPECIAL);  // [CLS]
  EXPECT_TRUE(iree_tokenizer_vocab_token_attrs(vocab, 3) &
              IREE_TOKENIZER_TOKEN_ATTR_SPECIAL);  // [SEP]
  EXPECT_TRUE(iree_tokenizer_vocab_token_attrs(vocab, 4) &
              IREE_TOKENIZER_TOKEN_ATTR_SPECIAL);  // [MASK]

  // Regular tokens should NOT have ATTR_SPECIAL.
  EXPECT_FALSE(iree_tokenizer_vocab_token_attrs(vocab, 5) &
               IREE_TOKENIZER_TOKEN_ATTR_SPECIAL);  // hello
  EXPECT_FALSE(iree_tokenizer_vocab_token_attrs(vocab, 6) &
               IREE_TOKENIZER_TOKEN_ATTR_SPECIAL);  // world
  EXPECT_FALSE(iree_tokenizer_vocab_token_attrs(vocab, 7) &
               IREE_TOKENIZER_TOKEN_ATTR_SPECIAL);  // ##ing

  iree_tokenizer_vocab_free(vocab);
}

TEST(WordPieceJsonTest, AddedTokensNotInVocab) {
  // Test the edge case where added_tokens contains tokens NOT in model.vocab.
  // The token at ID 10 is only in added_tokens, not in vocab.
  const char* json_with_extra = R"({
    "model": {
      "type": "WordPiece",
      "vocab": {
        "[PAD]": 0, "[UNK]": 1, "hello": 2
      }
    },
    "added_tokens": [
      {"id": 0, "content": "[PAD]", "special": true},
      {"id": 1, "content": "[UNK]", "special": true},
      {"id": 3, "content": "[EXTRA]", "special": true}
    ]
  })";
  iree_string_view_t json =
      iree_make_string_view(json_with_extra, strlen(json_with_extra));

  iree_tokenizer_vocab_t* vocab = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_import_wordpiece_json(
      json, iree_allocator_system(), &vocab));

  // Should have 4 tokens: 3 from vocab + 1 extra from added_tokens.
  EXPECT_EQ(iree_tokenizer_vocab_capacity(vocab), 4u);

  // Verify the extra token was added.
  EXPECT_EQ(iree_tokenizer_vocab_lookup(vocab, IREE_SVL("[EXTRA]")), 3);

  // Verify it has ATTR_SPECIAL.
  EXPECT_TRUE(iree_tokenizer_vocab_token_attrs(vocab, 3) &
              IREE_TOKENIZER_TOKEN_ATTR_SPECIAL);

  iree_tokenizer_vocab_free(vocab);
}

TEST(WordPieceJsonTest, AddedTokensContentMismatchError) {
  // Test that mismatched content between added_tokens and vocab fails.
  const char* mismatched = R"({
    "model": {
      "type": "WordPiece",
      "vocab": {
        "[PAD]": 0, "[UNK]": 1, "hello": 2
      }
    },
    "added_tokens": [
      {"id": 0, "content": "[PAD]", "special": true},
      {"id": 2, "content": "WRONG", "special": false}
    ]
  })";
  iree_string_view_t json =
      iree_make_string_view(mismatched, strlen(mismatched));

  iree_tokenizer_vocab_t* vocab = nullptr;
  iree_status_t status = iree_tokenizer_vocab_import_wordpiece_json(
      json, iree_allocator_system(), &vocab);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
  EXPECT_EQ(vocab, nullptr);
  iree_status_free(status);
}

TEST(WordPieceJsonTest, DuplicateAddedTokensDeduped) {
  // Test that duplicate IDs in added_tokens are handled (first entry wins).
  const char* duplicates = R"({
    "model": {
      "type": "WordPiece",
      "vocab": {
        "[PAD]": 0, "[UNK]": 1
      }
    },
    "added_tokens": [
      {"id": 0, "content": "[PAD]", "special": true},
      {"id": 0, "content": "[PAD]", "special": false},
      {"id": 1, "content": "[UNK]", "special": true}
    ]
  })";
  iree_string_view_t json =
      iree_make_string_view(duplicates, strlen(duplicates));

  iree_tokenizer_vocab_t* vocab = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_import_wordpiece_json(
      json, iree_allocator_system(), &vocab));

  // First entry had special=true, so [PAD] should have ATTR_SPECIAL.
  EXPECT_TRUE(iree_tokenizer_vocab_token_attrs(vocab, 0) &
              IREE_TOKENIZER_TOKEN_ATTR_SPECIAL);

  iree_tokenizer_vocab_free(vocab);
}

TEST(WordPieceJsonTest, FallbackSpecialDetectionNoAddedTokens) {
  // Test that specials are detected from vocab when added_tokens is absent.
  const char* no_added = R"({
    "model": {
      "type": "WordPiece",
      "vocab": {
        "[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4, "hello": 5
      },
      "unk_token": "[UNK]"
    }
  })";
  iree_string_view_t json = iree_make_string_view(no_added, strlen(no_added));

  iree_tokenizer_vocab_t* vocab = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_import_wordpiece_json(
      json, iree_allocator_system(), &vocab));

  // Special IDs should be detected from vocab patterns.
  iree_tokenizer_special_ids_t ids = iree_tokenizer_vocab_special_ids(vocab);
  EXPECT_EQ(ids.pad, 0);
  EXPECT_EQ(ids.unk, 1);
  EXPECT_EQ(ids.cls, 2);
  EXPECT_EQ(ids.sep, 3);
  EXPECT_EQ(ids.mask, 4);

  // Fallback-detected specials should also have ATTR_SPECIAL set.
  EXPECT_TRUE(iree_tokenizer_vocab_token_attrs(vocab, 0) &
              IREE_TOKENIZER_TOKEN_ATTR_SPECIAL);  // [PAD]
  EXPECT_TRUE(iree_tokenizer_vocab_token_attrs(vocab, 1) &
              IREE_TOKENIZER_TOKEN_ATTR_SPECIAL);  // [UNK]
  EXPECT_TRUE(iree_tokenizer_vocab_token_attrs(vocab, 2) &
              IREE_TOKENIZER_TOKEN_ATTR_SPECIAL);  // [CLS]
  EXPECT_TRUE(iree_tokenizer_vocab_token_attrs(vocab, 3) &
              IREE_TOKENIZER_TOKEN_ATTR_SPECIAL);  // [SEP]
  EXPECT_TRUE(iree_tokenizer_vocab_token_attrs(vocab, 4) &
              IREE_TOKENIZER_TOKEN_ATTR_SPECIAL);  // [MASK]

  // Regular token should NOT have ATTR_SPECIAL.
  EXPECT_FALSE(iree_tokenizer_vocab_token_attrs(vocab, 5) &
               IREE_TOKENIZER_TOKEN_ATTR_SPECIAL);  // hello

  iree_tokenizer_vocab_free(vocab);
}

TEST(WordPieceJsonTest, FallbackUnkTokenFromModel) {
  // Test that model.unk_token is used to identify UNK even if pattern doesn't
  // match.
  const char* custom_unk = R"({
    "model": {
      "type": "WordPiece",
      "vocab": {
        "<CUSTOM_UNK>": 0, "hello": 1
      },
      "unk_token": "<CUSTOM_UNK>"
    }
  })";
  iree_string_view_t json =
      iree_make_string_view(custom_unk, strlen(custom_unk));

  iree_tokenizer_vocab_t* vocab = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_import_wordpiece_json(
      json, iree_allocator_system(), &vocab));

  // UNK should be detected from model.unk_token.
  iree_tokenizer_special_ids_t ids = iree_tokenizer_vocab_special_ids(vocab);
  EXPECT_EQ(ids.unk, 0);

  iree_tokenizer_vocab_free(vocab);
}

//===----------------------------------------------------------------------===//
// Factory Tests (iree_tokenizer_from_wordpiece_json)
//===----------------------------------------------------------------------===//

TEST(WordPieceJsonFactoryTest, CreateTokenizerFromMinimalJson) {
  iree_string_view_t json = GetTestFile("wordpiece_minimal.json");
  ASSERT_GT(json.size, 0u);

  iree_tokenizer_t* tokenizer = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_from_wordpiece_json(
      json, iree_allocator_system(), &tokenizer));
  ASSERT_NE(tokenizer, nullptr);

  // Verify vocab is accessible.
  const iree_tokenizer_vocab_t* vocab = iree_tokenizer_vocab(tokenizer);
  ASSERT_NE(vocab, nullptr);
  EXPECT_EQ(iree_tokenizer_vocab_capacity(vocab), 10u);

  iree_tokenizer_free(tokenizer);
}

TEST(WordPieceJsonFactoryTest, CreateTokenizerWithPreTokenizer) {
  // Test with a pre_tokenizer configuration.
  const char* with_pretokenizer = R"({
    "model": {
      "type": "WordPiece",
      "vocab": {"[UNK]": 0, "hello": 1, "world": 2}
    },
    "added_tokens": [
      {"id": 0, "content": "[UNK]", "special": true}
    ],
    "pre_tokenizer": {
      "type": "BertPreTokenizer"
    }
  })";
  iree_string_view_t json =
      iree_make_string_view(with_pretokenizer, strlen(with_pretokenizer));

  iree_tokenizer_t* tokenizer = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_from_wordpiece_json(
      json, iree_allocator_system(), &tokenizer));
  ASSERT_NE(tokenizer, nullptr);

  iree_tokenizer_free(tokenizer);
}

TEST(WordPieceJsonFactoryTest, CreateTokenizerWithDecoder) {
  // Test with a decoder configuration.
  const char* with_decoder = R"({
    "model": {
      "type": "WordPiece",
      "vocab": {"[UNK]": 0, "hello": 1, "##ing": 2}
    },
    "added_tokens": [
      {"id": 0, "content": "[UNK]", "special": true}
    ],
    "decoder": {
      "type": "WordPiece",
      "prefix": "##",
      "cleanup": true
    }
  })";
  iree_string_view_t json =
      iree_make_string_view(with_decoder, strlen(with_decoder));

  iree_tokenizer_t* tokenizer = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_from_wordpiece_json(
      json, iree_allocator_system(), &tokenizer));
  ASSERT_NE(tokenizer, nullptr);

  iree_tokenizer_free(tokenizer);
}

TEST(WordPieceJsonFactoryTest, CreateTokenizerWithPostProcessor) {
  // Test with a post_processor configuration for BERT style.
  const char* with_postprocessor = R"({
    "model": {
      "type": "WordPiece",
      "vocab": {"[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "hello": 3}
    },
    "added_tokens": [
      {"id": 0, "content": "[UNK]", "special": true},
      {"id": 1, "content": "[CLS]", "special": true},
      {"id": 2, "content": "[SEP]", "special": true}
    ],
    "post_processor": {
      "type": "TemplateProcessing",
      "single": [{"SpecialToken": {"id": "[CLS]", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}}, {"SpecialToken": {"id": "[SEP]", "type_id": 0}}],
      "pair": null,
      "special_tokens": {"[CLS]": {"id": "[CLS]", "ids": [1], "tokens": ["[CLS]"]}, "[SEP]": {"id": "[SEP]", "ids": [2], "tokens": ["[SEP]"]}}
    }
  })";
  iree_string_view_t json =
      iree_make_string_view(with_postprocessor, strlen(with_postprocessor));

  iree_tokenizer_t* tokenizer = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_from_wordpiece_json(
      json, iree_allocator_system(), &tokenizer));
  ASSERT_NE(tokenizer, nullptr);

  iree_tokenizer_free(tokenizer);
}

TEST(WordPieceJsonFactoryTest, InvalidModelTypeFails) {
  // Factory should fail with wrong model type.
  const char* bpe = R"({
    "model": {
      "type": "BPE",
      "vocab": {"hello": 0}
    }
  })";
  iree_string_view_t json = iree_make_string_view(bpe, strlen(bpe));

  iree_tokenizer_t* tokenizer = nullptr;
  iree_status_t status = iree_tokenizer_from_wordpiece_json(
      json, iree_allocator_system(), &tokenizer);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
  EXPECT_EQ(tokenizer, nullptr);
  iree_status_free(status);
}

TEST(WordPieceJsonFactoryTest, MissingUnkTokenFails) {
  // WordPiece requires [UNK] token - should fail without one.
  const char* no_unk = R"({
    "model": {
      "type": "WordPiece",
      "vocab": {"hello": 0, "world": 1}
    }
  })";
  iree_string_view_t json = iree_make_string_view(no_unk, strlen(no_unk));

  iree_tokenizer_t* tokenizer = nullptr;
  iree_status_t status = iree_tokenizer_from_wordpiece_json(
      json, iree_allocator_system(), &tokenizer);
  EXPECT_THAT(status, ::testing::Not(::iree::testing::status::IsOk()));
  EXPECT_EQ(tokenizer, nullptr);
  iree_status_free(status);
}

TEST(WordPieceJsonFactoryTest, MalformedJsonFails) {
  const char* garbage = "not json {{{";
  iree_string_view_t json = iree_make_string_view(garbage, strlen(garbage));

  iree_tokenizer_t* tokenizer = nullptr;
  iree_status_t status = iree_tokenizer_from_wordpiece_json(
      json, iree_allocator_system(), &tokenizer);
  EXPECT_THAT(status, ::testing::Not(::iree::testing::status::IsOk()));
  EXPECT_EQ(tokenizer, nullptr);
  iree_status_free(status);
}

TEST(WordPieceJsonFactoryTest, EncodeDecodeRoundTrip) {
  iree_string_view_t json = GetTestFile("wordpiece_minimal.json");
  ASSERT_GT(json.size, 0u);

  iree_tokenizer_t* tokenizer = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_from_wordpiece_json(
      json, iree_allocator_system(), &tokenizer));

  // Encode "hello" - a single word the vocab supports.
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

TEST(WordPieceJsonFactoryTest, VocabOwnershipOnSuccess) {
  // Verify vocab is owned by tokenizer and freed correctly.
  const char* minimal = R"({
    "model": {
      "type": "WordPiece",
      "vocab": {"[UNK]": 0, "a": 1, "b": 2}
    },
    "added_tokens": [
      {"id": 0, "content": "[UNK]", "special": true}
    ]
  })";
  iree_string_view_t json = iree_make_string_view(minimal, strlen(minimal));

  iree_tokenizer_t* tokenizer = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_from_wordpiece_json(
      json, iree_allocator_system(), &tokenizer));

  // Access vocab through tokenizer.
  const iree_tokenizer_vocab_t* vocab = iree_tokenizer_vocab(tokenizer);
  EXPECT_NE(vocab, nullptr);
  EXPECT_EQ(iree_tokenizer_vocab_capacity(vocab), 3u);

  // Free tokenizer - this should also free the vocab.
  // (No double-free or leak should occur.)
  iree_tokenizer_free(tokenizer);
}

TEST(WordPieceJsonFactoryTest, ConfigParsedFromModel) {
  // Test that WordPiece config (max_input_chars_per_word,
  // continuing_subword_prefix) is parsed from the model section.
  const char* with_config = R"({
    "model": {
      "type": "WordPiece",
      "vocab": {"[UNK]": 0, "hello": 1, "##ing": 2},
      "max_input_chars_per_word": 100,
      "continuing_subword_prefix": "##"
    },
    "added_tokens": [
      {"id": 0, "content": "[UNK]", "special": true}
    ]
  })";
  iree_string_view_t json =
      iree_make_string_view(with_config, strlen(with_config));

  iree_tokenizer_t* tokenizer = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_from_wordpiece_json(
      json, iree_allocator_system(), &tokenizer));
  ASSERT_NE(tokenizer, nullptr);

  // The config should be applied - verify by checking tokenizer works.
  int32_t ids[16];
  iree_host_size_t count = 0;
  iree_tokenizer_encode_options_t options = {0};
  IREE_ASSERT_OK(iree_tokenizer_encode(tokenizer, IREE_SV("hello"), options,
                                       ids, IREE_ARRAYSIZE(ids), &count));
  EXPECT_GT(count, 0u);

  iree_tokenizer_free(tokenizer);
}

}  // namespace
