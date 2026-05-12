// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/format/huggingface/tokenizer_json.h"

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class TokenizerJsonTest : public ::testing::Test {
 protected:
  iree_allocator_t allocator_ = iree_allocator_system();
};

//===----------------------------------------------------------------------===//
// Top-Level Validation Tests
//===----------------------------------------------------------------------===//

TEST_F(TokenizerJsonTest, EmptyInputError) {
  iree_string_view_t json = iree_string_view_empty();
  iree_tokenizer_builder_t builder;
  iree_tokenizer_builder_initialize(allocator_, &builder);

  iree_status_t status = iree_tokenizer_parse_huggingface_json(json, &builder);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);

  iree_tokenizer_builder_deinitialize(&builder);
}

TEST_F(TokenizerJsonTest, MissingModelError) {
  // No model field.
  iree_string_view_t json = IREE_SV(R"({"version": "1.0"})");
  iree_tokenizer_builder_t builder;
  iree_tokenizer_builder_initialize(allocator_, &builder);

  iree_status_t status = iree_tokenizer_parse_huggingface_json(json, &builder);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND, status);

  iree_tokenizer_builder_deinitialize(&builder);
}

TEST_F(TokenizerJsonTest, UnknownTopLevelKeyError) {
  // Unknown key at top level should fail.
  iree_string_view_t json = IREE_SV(R"({
    "model": {"type": "BPE", "vocab": {}, "merges": []},
    "unknown_field": true
  })");
  iree_tokenizer_builder_t builder;
  iree_tokenizer_builder_initialize(allocator_, &builder);

  iree_status_t status = iree_tokenizer_parse_huggingface_json(json, &builder);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);

  iree_tokenizer_builder_deinitialize(&builder);
}

TEST_F(TokenizerJsonTest, InvalidVersionError) {
  // Unsupported version should fail.
  iree_string_view_t json = IREE_SV(R"({
    "version": "2.0",
    "model": {"type": "BPE", "vocab": {}, "merges": []}
  })");
  iree_tokenizer_builder_t builder;
  iree_tokenizer_builder_initialize(allocator_, &builder);

  iree_status_t status = iree_tokenizer_parse_huggingface_json(json, &builder);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);

  iree_tokenizer_builder_deinitialize(&builder);
}

TEST_F(TokenizerJsonTest, UnsupportedModelTypeError) {
  // Unsupported model type should fail.
  iree_string_view_t json = IREE_SV(R"({
    "model": {"type": "SomeNewType", "vocab": {}}
  })");
  iree_tokenizer_builder_t builder;
  iree_tokenizer_builder_initialize(allocator_, &builder);

  iree_status_t status = iree_tokenizer_parse_huggingface_json(json, &builder);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED, status);

  iree_tokenizer_builder_deinitialize(&builder);
}

//===----------------------------------------------------------------------===//
// Model Type Detection Tests
//===----------------------------------------------------------------------===//

TEST_F(TokenizerJsonTest, ExplicitBPEType) {
  // Explicit BPE type with minimal valid model.
  iree_string_view_t json = IREE_SV(R"({
    "version": "1.0",
    "model": {"type": "BPE", "vocab": {"a": 0}, "merges": []}
  })");
  iree_tokenizer_builder_t builder;
  iree_tokenizer_builder_initialize(allocator_, &builder);

  iree_status_t status = iree_tokenizer_parse_huggingface_json(json, &builder);
  IREE_EXPECT_OK(status);

  // Verify model and vocab were set.
  EXPECT_NE(builder.model, nullptr);
  EXPECT_NE(builder.vocab, nullptr);

  iree_tokenizer_builder_deinitialize(&builder);
}

TEST_F(TokenizerJsonTest, ExplicitWordPieceType) {
  // Explicit WordPiece type with valid vocab.
  iree_string_view_t json = IREE_SV(R"({
    "version": "1.0",
    "model": {"type": "WordPiece", "vocab": {"[UNK]": 0, "a": 1},
              "unk_token": "[UNK]"}
  })");
  iree_tokenizer_builder_t builder;
  iree_tokenizer_builder_initialize(allocator_, &builder);

  IREE_ASSERT_OK(iree_tokenizer_parse_huggingface_json(json, &builder));
  EXPECT_NE(builder.model, nullptr);
  EXPECT_NE(builder.vocab, nullptr);

  iree_tokenizer_builder_deinitialize(&builder);
}

TEST_F(TokenizerJsonTest, ExplicitUnigramType) {
  // Explicit Unigram type with valid vocab.
  iree_string_view_t json = IREE_SV(R"({
    "version": "1.0",
    "model": {"type": "Unigram",
              "vocab": [["<unk>", -10.0], ["a", -1.0], ["b", -1.5]],
              "unk_id": 0}
  })");
  iree_tokenizer_builder_t builder;
  iree_tokenizer_builder_initialize(allocator_, &builder);

  IREE_ASSERT_OK(iree_tokenizer_parse_huggingface_json(json, &builder));
  EXPECT_NE(builder.model, nullptr);
  EXPECT_NE(builder.vocab, nullptr);

  iree_tokenizer_builder_deinitialize(&builder);
}

TEST_F(TokenizerJsonTest, InferBPEFromMerges) {
  // No type field, but has merges -> infer BPE. Includes valid vocab/merges.
  iree_string_view_t json = IREE_SV(R"({
    "model": {"vocab": {"a": 0, "b": 1, "ab": 2}, "merges": ["a b"]}
  })");
  iree_tokenizer_builder_t builder;
  iree_tokenizer_builder_initialize(allocator_, &builder);

  iree_status_t status = iree_tokenizer_parse_huggingface_json(json, &builder);
  IREE_EXPECT_OK(status);

  // Verify model and vocab were set.
  EXPECT_NE(builder.model, nullptr);
  EXPECT_NE(builder.vocab, nullptr);

  iree_tokenizer_builder_deinitialize(&builder);
}

TEST_F(TokenizerJsonTest, InferWordPieceFromPrefix) {
  // No type field, but has continuing_subword_prefix -> infer WordPiece.
  iree_string_view_t json = IREE_SV(R"({
    "model": {"vocab": {"[UNK]": 0, "a": 1, "##b": 2},
              "continuing_subword_prefix": "##"}
  })");
  iree_tokenizer_builder_t builder;
  iree_tokenizer_builder_initialize(allocator_, &builder);

  IREE_ASSERT_OK(iree_tokenizer_parse_huggingface_json(json, &builder));
  EXPECT_NE(builder.model, nullptr);
  EXPECT_NE(builder.vocab, nullptr);

  iree_tokenizer_builder_deinitialize(&builder);
}

TEST_F(TokenizerJsonTest, InferUnigramFromUnkId) {
  // No type field, but has unk_id -> infer Unigram.
  iree_string_view_t json = IREE_SV(R"({
    "model": {"vocab": [["<unk>", -10.0], ["hello", -2.0]], "unk_id": 0}
  })");
  iree_tokenizer_builder_t builder;
  iree_tokenizer_builder_initialize(allocator_, &builder);

  IREE_ASSERT_OK(iree_tokenizer_parse_huggingface_json(json, &builder));
  EXPECT_NE(builder.model, nullptr);
  EXPECT_NE(builder.vocab, nullptr);

  iree_tokenizer_builder_deinitialize(&builder);
}

TEST_F(TokenizerJsonTest, CannotInferTypeError) {
  // No type field and cannot infer from structure.
  iree_string_view_t json = IREE_SV(R"({
    "model": {"vocab": {}}
  })");
  iree_tokenizer_builder_t builder;
  iree_tokenizer_builder_initialize(allocator_, &builder);

  iree_status_t status = iree_tokenizer_parse_huggingface_json(json, &builder);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);

  iree_tokenizer_builder_deinitialize(&builder);
}

//===----------------------------------------------------------------------===//
// Version Handling Tests
//===----------------------------------------------------------------------===//

TEST_F(TokenizerJsonTest, Version10Accepted) {
  // Version 1.0 should be accepted.
  iree_string_view_t json = IREE_SV(R"({
    "version": "1.0",
    "model": {"type": "BPE", "vocab": {"a": 0}, "merges": []}
  })");
  iree_tokenizer_builder_t builder;
  iree_tokenizer_builder_initialize(allocator_, &builder);

  iree_status_t status = iree_tokenizer_parse_huggingface_json(json, &builder);
  IREE_EXPECT_OK(status);

  iree_tokenizer_builder_deinitialize(&builder);
}

TEST_F(TokenizerJsonTest, NoVersionAccepted) {
  // Missing version field should be accepted (optional).
  iree_string_view_t json = IREE_SV(R"({
    "model": {"type": "BPE", "vocab": {"a": 0}, "merges": []}
  })");
  iree_tokenizer_builder_t builder;
  iree_tokenizer_builder_initialize(allocator_, &builder);

  iree_status_t status = iree_tokenizer_parse_huggingface_json(json, &builder);
  IREE_EXPECT_OK(status);

  iree_tokenizer_builder_deinitialize(&builder);
}

//===----------------------------------------------------------------------===//
// Section Integration Tests
//
// Verify that the orchestrator correctly handles each optional section:
// finding the key, handling null/missing, and passing values to component
// parsers. Exhaustive per-component tests are in the respective *_json_test
// files.
//===----------------------------------------------------------------------===//

TEST_F(TokenizerJsonTest, NullNormalizer) {
  iree_string_view_t json = IREE_SV(R"({
    "model": {"type": "BPE", "vocab": {"a": 0}, "merges": []},
    "normalizer": null
  })");
  iree_tokenizer_builder_t builder;
  iree_tokenizer_builder_initialize(allocator_, &builder);

  IREE_ASSERT_OK(iree_tokenizer_parse_huggingface_json(json, &builder));
  EXPECT_EQ(builder.normalizer, nullptr);

  iree_tokenizer_builder_deinitialize(&builder);
}

TEST_F(TokenizerJsonTest, NormalizerSuccess) {
  iree_string_view_t json = IREE_SV(R"({
    "model": {"type": "BPE", "vocab": {"a": 0}, "merges": []},
    "normalizer": {"type": "Lowercase"}
  })");
  iree_tokenizer_builder_t builder;
  iree_tokenizer_builder_initialize(allocator_, &builder);

  IREE_ASSERT_OK(iree_tokenizer_parse_huggingface_json(json, &builder));
  EXPECT_NE(builder.normalizer, nullptr);

  iree_tokenizer_builder_deinitialize(&builder);
}

TEST_F(TokenizerJsonTest, NormalizerErrorPropagation) {
  iree_string_view_t json = IREE_SV(R"({
    "model": {"type": "BPE", "vocab": {"a": 0}, "merges": []},
    "normalizer": {"type": "SomeUnknownNormalizer"}
  })");
  iree_tokenizer_builder_t builder;
  iree_tokenizer_builder_initialize(allocator_, &builder);

  iree_status_t status = iree_tokenizer_parse_huggingface_json(json, &builder);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED, status);

  iree_tokenizer_builder_deinitialize(&builder);
}

TEST_F(TokenizerJsonTest, NullPreTokenizer) {
  iree_string_view_t json = IREE_SV(R"({
    "model": {"type": "BPE", "vocab": {"a": 0}, "merges": []},
    "pre_tokenizer": null
  })");
  iree_tokenizer_builder_t builder;
  iree_tokenizer_builder_initialize(allocator_, &builder);

  IREE_ASSERT_OK(iree_tokenizer_parse_huggingface_json(json, &builder));
  EXPECT_EQ(builder.segmenter, nullptr);

  iree_tokenizer_builder_deinitialize(&builder);
}

TEST_F(TokenizerJsonTest, PreTokenizerSuccess) {
  // Uses Split (not Metaspace) since Metaspace requires the space→replacement
  // normalizer which is gated as UNIMPLEMENTED.
  iree_string_view_t json = IREE_SV(R"({
    "model": {"type": "BPE", "vocab": {"a": 0}, "merges": []},
    "pre_tokenizer": {"type": "Split", "pattern": {"Regex": "\\s+"}, "behavior": "Removed", "invert": false}
  })");
  iree_tokenizer_builder_t builder;
  iree_tokenizer_builder_initialize(allocator_, &builder);

  IREE_ASSERT_OK(iree_tokenizer_parse_huggingface_json(json, &builder));
  EXPECT_NE(builder.segmenter, nullptr);

  iree_tokenizer_builder_deinitialize(&builder);
}

TEST_F(TokenizerJsonTest, MetaspaceSynthesizesNormalizer) {
  // A Metaspace pre_tokenizer synthesizes both replace (space→▁) and prepend
  // (▁ prefix) normalizers. Verify parsing succeeds and the normalizer is set.
  iree_string_view_t json = IREE_SV(R"({
    "model": {"type": "BPE", "vocab": {"a": 0}, "merges": []},
    "pre_tokenizer": {"type": "Metaspace", "replacement": "\u2581"}
  })");
  iree_tokenizer_builder_t builder;
  iree_tokenizer_builder_initialize(allocator_, &builder);

  IREE_ASSERT_OK(iree_tokenizer_parse_huggingface_json(json, &builder));
  EXPECT_NE(builder.segmenter, nullptr);
  EXPECT_NE(builder.normalizer, nullptr);

  iree_tokenizer_builder_deinitialize(&builder);
}

TEST_F(TokenizerJsonTest, PreTokenizerErrorPropagation) {
  iree_string_view_t json = IREE_SV(R"({
    "model": {"type": "BPE", "vocab": {"a": 0}, "merges": []},
    "pre_tokenizer": {"type": "SomeUnimplementedType"}
  })");
  iree_tokenizer_builder_t builder;
  iree_tokenizer_builder_initialize(allocator_, &builder);

  iree_status_t status = iree_tokenizer_parse_huggingface_json(json, &builder);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED, status);

  iree_tokenizer_builder_deinitialize(&builder);
}

TEST_F(TokenizerJsonTest, NullDecoder) {
  iree_string_view_t json = IREE_SV(R"({
    "model": {"type": "BPE", "vocab": {"a": 0}, "merges": []},
    "decoder": null
  })");
  iree_tokenizer_builder_t builder;
  iree_tokenizer_builder_initialize(allocator_, &builder);

  IREE_ASSERT_OK(iree_tokenizer_parse_huggingface_json(json, &builder));
  EXPECT_EQ(builder.decoder, nullptr);

  iree_tokenizer_builder_deinitialize(&builder);
}

TEST_F(TokenizerJsonTest, DecoderSuccess) {
  iree_string_view_t json = IREE_SV(R"({
    "model": {"type": "BPE", "vocab": {"a": 0}, "merges": []},
    "decoder": {"type": "ByteFallback"}
  })");
  iree_tokenizer_builder_t builder;
  iree_tokenizer_builder_initialize(allocator_, &builder);

  IREE_ASSERT_OK(iree_tokenizer_parse_huggingface_json(json, &builder));
  EXPECT_NE(builder.decoder, nullptr);

  iree_tokenizer_builder_deinitialize(&builder);
}

TEST_F(TokenizerJsonTest, DecoderErrorPropagation) {
  iree_string_view_t json = IREE_SV(R"({
    "model": {"type": "BPE", "vocab": {"a": 0}, "merges": []},
    "decoder": {"type": "UnknownDecoder"}
  })");
  iree_tokenizer_builder_t builder;
  iree_tokenizer_builder_initialize(allocator_, &builder);

  iree_status_t status = iree_tokenizer_parse_huggingface_json(json, &builder);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED, status);

  iree_tokenizer_builder_deinitialize(&builder);
}

//===----------------------------------------------------------------------===//
// Added Token Flag Tests
//===----------------------------------------------------------------------===//

// Tests that added_token flags (lstrip, rstrip, single_word) load successfully.
// These flags control matching behavior for special tokens.

TEST_F(TokenizerJsonTest, AddedTokenLstripFlag) {
  // Minimal tokenizer JSON with an added token that has lstrip: true.
  // The lstrip flag means the token only matches when preceded by whitespace
  // or at the start of input.
  iree_string_view_t json = IREE_SV(R"({
    "model": {
      "type": "BPE",
      "vocab": {"<|test|>": 0},
      "merges": []
    },
    "added_tokens": [{
      "id": 0,
      "content": "<|test|>",
      "single_word": false,
      "lstrip": true,
      "rstrip": false,
      "normalized": false,
      "special": true
    }],
    "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": true}
  })");

  iree_tokenizer_t* tokenizer = nullptr;
  IREE_ASSERT_OK(
      iree_tokenizer_from_huggingface_json(json, allocator_, &tokenizer));
  ASSERT_NE(tokenizer, nullptr);
  iree_tokenizer_free(tokenizer);
}

TEST_F(TokenizerJsonTest, AddedTokenRstripFlag) {
  // The rstrip flag indicates trailing whitespace should be stripped after
  // matching (post-processing). It does not affect match eligibility.
  iree_string_view_t json = IREE_SV(R"({
    "model": {
      "type": "BPE",
      "vocab": {"<|test|>": 0},
      "merges": []
    },
    "added_tokens": [{
      "id": 0,
      "content": "<|test|>",
      "single_word": false,
      "lstrip": false,
      "rstrip": true,
      "normalized": false,
      "special": true
    }],
    "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": true}
  })");

  iree_tokenizer_t* tokenizer = nullptr;
  IREE_ASSERT_OK(
      iree_tokenizer_from_huggingface_json(json, allocator_, &tokenizer));
  ASSERT_NE(tokenizer, nullptr);
  iree_tokenizer_free(tokenizer);
}

// Verifies that back-to-back special tokens are both emitted. Previously,
// when the pipeline had buffered content, the second match would overwrite
// pending_special_token before the first was emitted, dropping it.
TEST_F(TokenizerJsonTest, BackToBackSpecialTokens) {
  iree_string_view_t json = IREE_SV(R"({
    "model": {
      "type": "BPE",
      "vocab": {"hello": 1, "world": 2},
      "merges": []
    },
    "added_tokens": [{
      "id": 100,
      "content": "<|user|>",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": false,
      "special": true
    }, {
      "id": 101,
      "content": "<|end|>",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": false,
      "special": true
    }, {
      "id": 102,
      "content": "<|assistant|>",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": false,
      "special": true
    }],
    "pre_tokenizer": null
  })");

  iree_tokenizer_t* tokenizer = nullptr;
  IREE_ASSERT_OK(
      iree_tokenizer_from_huggingface_json(json, allocator_, &tokenizer));
  ASSERT_NE(tokenizer, nullptr);

  auto encode = [&](const char* text) -> std::vector<int32_t> {
    std::vector<int32_t> tokens(64);
    iree_host_size_t token_count = 0;
    IREE_CHECK_OK(
        iree_tokenizer_encode(tokenizer, iree_make_cstring_view(text),
                              IREE_TOKENIZER_ENCODE_FLAG_NONE,
                              iree_tokenizer_make_token_output(
                                  tokens.data(), NULL, NULL, tokens.size()),
                              allocator_, &token_count));
    tokens.resize(token_count);
    return tokens;
  };

  // Two special tokens with no text between them.
  {
    auto tokens = encode("<|end|><|assistant|>");
    ASSERT_EQ(tokens.size(), 2u);
    EXPECT_EQ(tokens[0], 101);  // <|end|>
    EXPECT_EQ(tokens[1], 102);  // <|assistant|>
  }

  // Text, then two back-to-back special tokens, then more text.
  {
    auto tokens = encode("<|user|>hello<|end|><|assistant|>world<|end|>");
    ASSERT_EQ(tokens.size(), 6u);
    EXPECT_EQ(tokens[0], 100);  // <|user|>
    EXPECT_EQ(tokens[1], 1);    // hello
    EXPECT_EQ(tokens[2], 101);  // <|end|>
    EXPECT_EQ(tokens[3], 102);  // <|assistant|>
    EXPECT_EQ(tokens[4], 2);    // world
    EXPECT_EQ(tokens[5], 101);  // <|end|>
  }

  // Three back-to-back special tokens.
  {
    auto tokens = encode("<|end|><|user|><|assistant|>");
    ASSERT_EQ(tokens.size(), 3u);
    EXPECT_EQ(tokens[0], 101);  // <|end|>
    EXPECT_EQ(tokens[1], 100);  // <|user|>
    EXPECT_EQ(tokens[2], 102);  // <|assistant|>
  }

  iree_tokenizer_free(tokenizer);
}

TEST_F(TokenizerJsonTest, MaxSpecialTokenCountNoPostProcessor) {
  iree_string_view_t json = IREE_SV(R"({
    "model": {"type": "BPE", "vocab": {"a": 0}, "merges": []}
  })");
  iree_tokenizer_t* tokenizer = nullptr;
  IREE_ASSERT_OK(
      iree_tokenizer_from_huggingface_json(json, allocator_, &tokenizer));
  EXPECT_EQ(iree_tokenizer_max_special_token_count(tokenizer), 0u);
  iree_tokenizer_free(tokenizer);
}

TEST_F(TokenizerJsonTest, MaxSpecialTokenCountWithPostProcessor) {
  iree_string_view_t json = IREE_SV(R"({
    "model": {"type": "BPE", "vocab": {"a": 0}, "merges": []},
    "added_tokens": [
      {"id": 1, "content": "<|bos|>", "single_word": false, "lstrip": false,
       "rstrip": false, "normalized": false, "special": true},
      {"id": 2, "content": "<|eos|>", "single_word": false, "lstrip": false,
       "rstrip": false, "normalized": false, "special": true}
    ],
    "post_processor": {
      "type": "TemplateProcessing",
      "single": [
        {"SpecialToken": {"id": "<|bos|>", "type_id": 0}},
        {"Sequence": {"id": "A", "type_id": 0}},
        {"SpecialToken": {"id": "<|eos|>", "type_id": 0}}
      ],
      "pair": [],
      "special_tokens": {
        "<|bos|>": {"id": "<|bos|>", "ids": [1], "tokens": ["<|bos|>"]},
        "<|eos|>": {"id": "<|eos|>", "ids": [2], "tokens": ["<|eos|>"]}
      }
    }
  })");
  iree_tokenizer_t* tokenizer = nullptr;
  IREE_ASSERT_OK(
      iree_tokenizer_from_huggingface_json(json, allocator_, &tokenizer));
  EXPECT_EQ(iree_tokenizer_max_special_token_count(tokenizer), 2u);
  iree_tokenizer_free(tokenizer);
}

// Verifies end-to-end rstrip behavior: the token matches regardless of what
// follows, and trailing whitespace after the token is consumed (not passed to
// BPE). See:
// https://github.com/huggingface/tokenizers/blob/main/tokenizers/src/tokenizer/added_vocabulary.rs
TEST_F(TokenizerJsonTest, RstripTokenMatchAndWhitespaceConsumption) {
  iree_string_view_t json = IREE_SV(R"({
    "model": {
      "type": "BPE",
      "vocab": {"hello": 1, "world": 2},
      "merges": []
    },
    "added_tokens": [{
      "id": 100,
      "content": "<|user|>",
      "single_word": false,
      "lstrip": false,
      "rstrip": true,
      "normalized": false,
      "special": true
    }],
    "pre_tokenizer": null
  })");

  iree_tokenizer_t* tokenizer = nullptr;
  IREE_ASSERT_OK(
      iree_tokenizer_from_huggingface_json(json, allocator_, &tokenizer));
  ASSERT_NE(tokenizer, nullptr);

  auto encode = [&](const char* text) -> std::vector<int32_t> {
    std::vector<int32_t> tokens(64);
    iree_host_size_t token_count = 0;
    IREE_CHECK_OK(
        iree_tokenizer_encode(tokenizer, iree_make_cstring_view(text),
                              IREE_TOKENIZER_ENCODE_FLAG_NONE,
                              iree_tokenizer_make_token_output(
                                  tokens.data(), NULL, NULL, tokens.size()),
                              allocator_, &token_count));
    tokens.resize(token_count);
    return tokens;
  };

  // rstrip token followed by text (no whitespace): token must match.
  {
    auto tokens = encode("<|user|>hello");
    ASSERT_EQ(tokens.size(), 2u);
    EXPECT_EQ(tokens[0], 100);  // <|user|>
    EXPECT_EQ(tokens[1], 1);    // hello
  }

  // rstrip token followed by space + text: whitespace must be consumed.
  {
    auto tokens = encode("<|user|> hello");
    ASSERT_EQ(tokens.size(), 2u);
    EXPECT_EQ(tokens[0], 100);  // <|user|>
    EXPECT_EQ(tokens[1], 1);    // hello (space consumed by rstrip)
  }

  // rstrip token followed by multiple spaces + text.
  {
    auto tokens = encode("<|user|>   hello");
    ASSERT_EQ(tokens.size(), 2u);
    EXPECT_EQ(tokens[0], 100);  // <|user|>
    EXPECT_EQ(tokens[1], 1);    // hello (spaces consumed by rstrip)
  }

  // rstrip token followed by tab + text.
  {
    auto tokens = encode("<|user|>\thello");
    ASSERT_EQ(tokens.size(), 2u);
    EXPECT_EQ(tokens[0], 100);  // <|user|>
    EXPECT_EQ(tokens[1], 1);    // hello (tab consumed by rstrip)
  }

  // rstrip token followed by newline + text.
  {
    auto tokens = encode("<|user|>\nhello");
    ASSERT_EQ(tokens.size(), 2u);
    EXPECT_EQ(tokens[0], 100);  // <|user|>
    EXPECT_EQ(tokens[1], 1);    // hello (newline consumed by rstrip)
  }

  // rstrip token followed by only whitespace (no text after).
  {
    auto tokens = encode("<|user|>   ");
    ASSERT_EQ(tokens.size(), 1u);
    EXPECT_EQ(tokens[0], 100);  // <|user|> (whitespace consumed, nothing left)
  }

  iree_tokenizer_free(tokenizer);
}

TEST_F(TokenizerJsonTest, AddedTokenSingleWordFlag) {
  // The single_word flag means the token only matches when surrounded by
  // word boundaries (whitespace or punctuation).
  iree_string_view_t json = IREE_SV(R"({
    "model": {
      "type": "BPE",
      "vocab": {"<|test|>": 0},
      "merges": []
    },
    "added_tokens": [{
      "id": 0,
      "content": "<|test|>",
      "single_word": true,
      "lstrip": false,
      "rstrip": false,
      "normalized": false,
      "special": true
    }],
    "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": true}
  })");

  iree_tokenizer_t* tokenizer = nullptr;
  IREE_ASSERT_OK(
      iree_tokenizer_from_huggingface_json(json, allocator_, &tokenizer));
  ASSERT_NE(tokenizer, nullptr);
  iree_tokenizer_free(tokenizer);
}

// Special tokens with normalized=true are matched AFTER the normalizer runs.
// This is useful for tokens that should match normalized text (e.g., a token
// "yesterday" with a lowercasing normalizer would match input "Yesterday").
TEST_F(TokenizerJsonTest, AddedTokenNormalizedFlag) {
  iree_string_view_t json = IREE_SV(R"({
    "model": {
      "type": "BPE",
      "vocab": {"<|test|>": 0},
      "merges": []
    },
    "added_tokens": [{
      "id": 0,
      "content": "<|test|>",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": true,
      "special": true
    }],
    "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": true}
  })");

  iree_tokenizer_t* tokenizer = nullptr;
  IREE_ASSERT_OK(
      iree_tokenizer_from_huggingface_json(json, allocator_, &tokenizer));
  ASSERT_NE(tokenizer, nullptr);
  iree_tokenizer_free(tokenizer);
}

// Tests that the default case (all flags false except special) loads fine.
TEST_F(TokenizerJsonTest, AddedTokenDefaultFlags) {
  iree_string_view_t json = IREE_SV(R"({
    "model": {
      "type": "BPE",
      "vocab": {"<|test|>": 0},
      "merges": []
    },
    "added_tokens": [{
      "id": 0,
      "content": "<|test|>",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": false,
      "special": true
    }],
    "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": true}
  })");

  iree_tokenizer_t* tokenizer = nullptr;
  IREE_ASSERT_OK(
      iree_tokenizer_from_huggingface_json(json, allocator_, &tokenizer));
  ASSERT_NE(tokenizer, nullptr);
  iree_tokenizer_free(tokenizer);
}

//===----------------------------------------------------------------------===//
// High-Level API and End-to-End Tests
//===----------------------------------------------------------------------===//

TEST_F(TokenizerJsonTest, HighLevelApiEmptyError) {
  // High-level API should fail on empty input.
  iree_string_view_t json = iree_string_view_empty();
  iree_tokenizer_t* tokenizer = nullptr;

  iree_status_t status =
      iree_tokenizer_from_huggingface_json(json, allocator_, &tokenizer);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
  EXPECT_EQ(tokenizer, nullptr);
}

// Tests that consecutive newlines are tokenized separately when followed by
// text, matching tiktoken behavior. The GPT-2 regex `\s+(?!\S)|\s+` should
// produce separate matches for each newline when followed by non-whitespace.
TEST_F(TokenizerJsonTest, ConsecutiveNewlinesBeforeText) {
  // Minimal GPT-2 style config with ByteLevel pre_tokenizer.
  // Ċ = U+010A = ByteLevel encoding of newline (0x0A).
  // The vocab has both Ċ (single newline) and ĊĊ (double newline).
  // Merge "Ċ Ċ" exists, but it should NOT apply when newlines are in
  // separate segments (regex matches them separately before text).
  iree_string_view_t json = IREE_SV(R"({
    "model": {
      "type": "BPE",
      "vocab": {
        "Ċ": 0,
        "ĊĊ": 1,
        "A": 2
      },
      "merges": ["Ċ Ċ"]
    },
    "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": true}
  })");

  iree_tokenizer_t* tokenizer = nullptr;
  IREE_ASSERT_OK(
      iree_tokenizer_from_huggingface_json(json, allocator_, &tokenizer));
  ASSERT_NE(tokenizer, nullptr);

  // Encode "\n\nA" - two newlines followed by 'A'.
  // Expected: [Ċ, Ċ, A] = [0, 0, 2] (separate newlines)
  // Bug:      [ĊĊ, A] = [1, 2] (merged newlines)
  const char* text = "\n\nA";
  std::vector<int32_t> tokens(16);
  iree_host_size_t token_count = 0;

  IREE_ASSERT_OK(iree_tokenizer_encode(
      tokenizer, iree_make_cstring_view(text), IREE_TOKENIZER_ENCODE_FLAG_NONE,
      iree_tokenizer_make_token_output(tokens.data(), NULL, NULL,
                                       tokens.size()),
      allocator_, &token_count));

  tokens.resize(token_count);

  // Should produce 3 tokens: [0, 0, 2] = [Ċ, Ċ, A]
  // NOT 2 tokens: [1, 2] = [ĊĊ, A]
  ASSERT_EQ(token_count, 3u)
      << "Expected 3 tokens [Ċ, Ċ, A], got " << token_count;
  EXPECT_EQ(tokens[0], 0);  // First Ċ (newline)
  EXPECT_EQ(tokens[1], 0);  // Second Ċ (newline)
  EXPECT_EQ(tokens[2], 2);  // A

  iree_tokenizer_free(tokenizer);
}

// Minimal GPT-2 ByteLevel BPE config shared by ring buffer edge case tests.
// Vocab: Ċ=0, ĊĊ=1, A=2, B=3. Merge: "Ċ Ċ".
const char kMinimalGPT2Config[] = R"({
  "model": {
    "type": "BPE",
    "vocab": {
      "\u010a": 0,
      "\u010a\u010a": 1,
      "A": 2,
      "B": 3
    },
    "merges": ["\u010a \u010a"]
  },
  "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": true}
})";

// Helper: encodes text with the minimal GPT-2 config and returns token IDs.
std::vector<int32_t> EncodeWithMinimalGPT2(const char* text) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_tokenizer_t* tokenizer = nullptr;
  iree_status_t status = iree_tokenizer_from_huggingface_json(
      iree_make_cstring_view(kMinimalGPT2Config), allocator, &tokenizer);
  IREE_CHECK_OK(status);

  std::vector<int32_t> tokens(64);
  iree_host_size_t token_count = 0;
  IREE_CHECK_OK(iree_tokenizer_encode(
      tokenizer, iree_make_cstring_view(text), IREE_TOKENIZER_ENCODE_FLAG_NONE,
      iree_tokenizer_make_token_output(tokens.data(), NULL, NULL,
                                       tokens.size()),
      allocator, &token_count));

  tokens.resize(token_count);
  iree_tokenizer_free(tokenizer);
  return tokens;
}

// Tests single newline before text. The GPT-2 regex emits one whitespace
// segment and holds the trailing letter pending. Model consumes the segment,
// ring empties, finalize must still produce the letter.
TEST_F(TokenizerJsonTest, SingleNewlineBeforeText) {
  // Segments: ["\n"] emitted, ["A"] pending.
  // Ring empties after model consumes 1 segment → tests ring reset guard.
  auto tokens = EncodeWithMinimalGPT2("\nA");
  ASSERT_EQ(tokens.size(), 2u) << "Expected [Ċ, A]";
  EXPECT_EQ(tokens[0], 0);  // Ċ
  EXPECT_EQ(tokens[1], 2);  // A
}

// Tests triple newlines before text. The greedy \s+(?!\S) match consumes
// the first two newlines as one segment (lookahead passes at second \n),
// then the third newline falls through to \s+ (lookahead fails at 'A').
TEST_F(TokenizerJsonTest, TripleNewlinesBeforeText) {
  // Segments: ["\n\n", "\n"] emitted, ["A"] pending.
  // The "\n\n" segment gets BPE-merged: Ċ+Ċ → ĊĊ = token 1.
  auto tokens = EncodeWithMinimalGPT2("\n\n\nA");
  ASSERT_EQ(tokens.size(), 3u) << "Expected [ĊĊ, Ċ, A]";
  EXPECT_EQ(tokens[0], 1);  // ĊĊ (merged pair)
  EXPECT_EQ(tokens[1], 0);  // Ċ
  EXPECT_EQ(tokens[2], 2);  // A
}

// Tests text-newline-text pattern. The regex matches "A" as a ?\p{L}+ segment,
// "\n" as \s+ (fallback), then holds trailing "A" pending. Model consumes
// both completed segments, ring empties.
TEST_F(TokenizerJsonTest, TextNewlineText) {
  // Segments: ["A", "\n"] emitted, ["A"] pending.
  auto tokens = EncodeWithMinimalGPT2("A\nA");
  ASSERT_EQ(tokens.size(), 3u) << "Expected [A, Ċ, A]";
  EXPECT_EQ(tokens[0], 2);  // A
  EXPECT_EQ(tokens[1], 0);  // Ċ
  EXPECT_EQ(tokens[2], 2);  // A
}

// Tests text-double-newline-text. Both newlines become separate segments
// (first passes lookahead, second uses fallback). Model consumes 3 segments
// before ring empties with pending trailing text.
TEST_F(TokenizerJsonTest, TextDoubleNewlineText) {
  // Segments: ["A", "\n", "\n"] emitted, ["A"] pending.
  auto tokens = EncodeWithMinimalGPT2("A\n\nA");
  ASSERT_EQ(tokens.size(), 4u) << "Expected [A, Ċ, Ċ, A]";
  EXPECT_EQ(tokens[0], 2);  // A
  EXPECT_EQ(tokens[1], 0);  // Ċ
  EXPECT_EQ(tokens[2], 0);  // Ċ
  EXPECT_EQ(tokens[3], 2);  // A
}

// Tests multiple words separated by newlines. Each word and each newline
// becomes a separate segment; the trailing word stays pending until finalize.
TEST_F(TokenizerJsonTest, MultipleWordsWithNewlines) {
  // Segments: ["A", "\n", "B", "\n"] emitted, ["A"] pending.
  auto tokens = EncodeWithMinimalGPT2("A\nB\nA");
  ASSERT_EQ(tokens.size(), 5u) << "Expected [A, Ċ, B, Ċ, A]";
  EXPECT_EQ(tokens[0], 2);  // A
  EXPECT_EQ(tokens[1], 0);  // Ċ
  EXPECT_EQ(tokens[2], 3);  // B
  EXPECT_EQ(tokens[3], 0);  // Ċ
  EXPECT_EQ(tokens[4], 2);  // A
}

// Tests input that produces 0 segments during process (entire input is one
// pending match). The ?\p{L}+ pattern starts matching but can't terminate
// without seeing a non-letter character. All tokens come from finalize.
TEST_F(TokenizerJsonTest, SingleLetterAllPending) {
  // Segments: [] emitted, ["A"] pending.
  // Finalize produces the only segment. Tests the chunk_base underflow path
  // where match.start(0) < chunk_base(1), wrapping correctly via adjustment.
  auto tokens = EncodeWithMinimalGPT2("A");
  ASSERT_EQ(tokens.size(), 1u) << "Expected [A]";
  EXPECT_EQ(tokens[0], 2);  // A
}

// Tests multi-letter input with 0 segments emitted (one pending match).
// The regex ?\p{L}+ greedily matches all letters without terminating.
TEST_F(TokenizerJsonTest, MultiLetterAllPending) {
  // Segments: [] emitted, ["AB"] pending.
  // Finalize produces one segment "AB". BPE encodes as [A, B] (no merge).
  auto tokens = EncodeWithMinimalGPT2("AB");
  ASSERT_EQ(tokens.size(), 2u) << "Expected [A, B]";
  EXPECT_EQ(tokens[0], 2);  // A
  EXPECT_EQ(tokens[1], 3);  // B
}

// Tests the streaming API with ring-draining input patterns. Verifies that
// the ring reset guard works correctly when tokens are collected incrementally
// via the state-based API (matching what iree_tokenizer_encode does
// internally).
TEST_F(TokenizerJsonTest, StreamingNewlinesBeforeText) {
  iree_tokenizer_t* tokenizer = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_from_huggingface_json(
      iree_make_cstring_view(kMinimalGPT2Config), allocator_, &tokenizer));
  ASSERT_NE(tokenizer, nullptr);

  // Allocate encode state with recommended transform buffer size.
  iree_host_size_t state_size = 0;
  IREE_ASSERT_OK(
      iree_tokenizer_encode_state_calculate_size(tokenizer, &state_size));
  std::vector<uint8_t> state_storage(state_size);
  iree_host_size_t transform_size =
      iree_tokenizer_transform_buffer_recommended_size(3);
  std::vector<uint8_t> transform_buffer(transform_size);
  iree_tokenizer_encode_state_t* state = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_encode_state_initialize(
      tokenizer,
      iree_make_byte_span(state_storage.data(), state_storage.size()),
      iree_make_byte_span(transform_buffer.data(), transform_buffer.size()),
      iree_tokenizer_offset_run_list_empty(),
      IREE_TOKENIZER_ENCODE_FLAG_AT_INPUT_START, &state));

  // Feed the full input, collecting tokens.
  iree_string_view_t input = IREE_SVL("\n\nA");
  iree_host_size_t total_tokens = 0;
  int32_t token_buffer[16];

  while (input.size > 0) {
    iree_host_size_t bytes_consumed = 0;
    iree_host_size_t token_count = 0;
    IREE_ASSERT_OK(iree_tokenizer_encode_state_feed(
        state, input,
        iree_tokenizer_make_token_output(token_buffer + total_tokens, NULL,
                                         NULL, 16 - total_tokens),
        &bytes_consumed, &token_count));
    total_tokens += token_count;
    input.data += bytes_consumed;
    input.size -= bytes_consumed;
    if (bytes_consumed == 0) break;
  }

  // Finalize to flush pending segments (single call like one-shot API).
  iree_host_size_t final_tokens = 0;
  IREE_ASSERT_OK(iree_tokenizer_encode_state_finalize(
      state,
      iree_tokenizer_make_token_output(token_buffer + total_tokens, NULL, NULL,
                                       16 - total_tokens),
      &final_tokens));
  total_tokens += final_tokens;

  // Should produce [Ċ, Ċ, A] = [0, 0, 2].
  ASSERT_EQ(total_tokens, 3u) << "Expected [Ċ, Ċ, A]";
  EXPECT_EQ(token_buffer[0], 0);  // Ċ
  EXPECT_EQ(token_buffer[1], 0);  // Ċ
  EXPECT_EQ(token_buffer[2], 2);  // A

  iree_tokenizer_encode_state_deinitialize(state);
  iree_tokenizer_free(tokenizer);
}

//===----------------------------------------------------------------------===//
// Pending Token Bound Tests
//===----------------------------------------------------------------------===//

// Helper: creates a streaming encode state for a tokenizer.
// Returns allocated state; caller must deinitialize + free storage vectors.
struct StreamingEncodeContext {
  iree_tokenizer_encode_state_t* state;
  std::vector<uint8_t> state_storage;
  std::vector<uint8_t> transform_buffer;
};

StreamingEncodeContext CreateStreamingState(
    const iree_tokenizer_t* tokenizer, iree_host_size_t text_hint,
    iree_tokenizer_encode_flags_t flags) {
  StreamingEncodeContext ctx = {};
  iree_host_size_t state_size = 0;
  IREE_CHECK_OK(
      iree_tokenizer_encode_state_calculate_size(tokenizer, &state_size));
  ctx.state_storage.resize(state_size);
  iree_host_size_t transform_size =
      iree_tokenizer_transform_buffer_recommended_size(text_hint);
  ctx.transform_buffer.resize(transform_size);
  IREE_CHECK_OK(iree_tokenizer_encode_state_initialize(
      tokenizer,
      iree_make_byte_span(ctx.state_storage.data(), ctx.state_storage.size()),
      iree_make_byte_span(ctx.transform_buffer.data(),
                          ctx.transform_buffer.size()),
      iree_tokenizer_offset_run_list_empty(), flags, &ctx.state));
  return ctx;
}

// Feeds all bytes of |text| into the state, collecting emitted tokens.
// Returns the total tokens emitted during feed (not finalize).
iree_host_size_t FeedAll(iree_tokenizer_encode_state_t* state,
                         iree_string_view_t text,
                         iree_tokenizer_token_id_t* token_buffer,
                         iree_host_size_t token_capacity) {
  iree_host_size_t total_tokens = 0;
  while (text.size > 0) {
    iree_host_size_t bytes_consumed = 0;
    iree_host_size_t token_count = 0;
    IREE_CHECK_OK(iree_tokenizer_encode_state_feed(
        state, text,
        iree_tokenizer_make_token_output(token_buffer + total_tokens, NULL,
                                         NULL, token_capacity - total_tokens),
        &bytes_consumed, &token_count));
    total_tokens += token_count;
    text.data += bytes_consumed;
    text.size -= bytes_consumed;
    if (bytes_consumed == 0 && token_count == 0) break;
  }
  return total_tokens;
}

// Test 1: pending_token_bound >= actual finalize count (key invariant).
TEST_F(TokenizerJsonTest, PendingTokenBoundAfterFeedMatchesFinalize) {
  iree_tokenizer_t* tokenizer = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_from_huggingface_json(
      iree_make_cstring_view(kMinimalGPT2Config), allocator_, &tokenizer));
  ASSERT_NE(tokenizer, nullptr);

  const char* texts[] = {"\n\nA", "\nA", "A\nB\nA", "A", "AB", "A\n\n\nA"};
  for (const char* text : texts) {
    auto ctx = CreateStreamingState(tokenizer, strlen(text),
                                    IREE_TOKENIZER_ENCODE_FLAG_AT_INPUT_START);

    int32_t token_buffer[256];
    iree_host_size_t feed_tokens =
        FeedAll(ctx.state, iree_make_cstring_view(text), token_buffer, 256);

    // Query the bound BEFORE finalize.
    iree_host_size_t bound =
        iree_tokenizer_encode_state_pending_token_bound(ctx.state);

    // Finalize and count actual tokens.
    iree_host_size_t finalize_count = 0;
    IREE_ASSERT_OK(iree_tokenizer_encode_state_finalize(
        ctx.state,
        iree_tokenizer_make_token_output(token_buffer + feed_tokens, NULL, NULL,
                                         256 - feed_tokens),
        &finalize_count));

    // Key invariant: bound must be >= actual finalize count.
    EXPECT_GE(bound, finalize_count)
        << "pending_token_bound (" << bound << ") < actual finalize count ("
        << finalize_count << ") for text: \"" << text << "\"";

    iree_tokenizer_encode_state_deinitialize(ctx.state);
  }

  iree_tokenizer_free(tokenizer);
}

// Test 2: pending_token_bound decreases as more tokens are emitted via feed.
TEST_F(TokenizerJsonTest, PendingTokenBoundDecreasesAfterFeed) {
  iree_tokenizer_t* tokenizer = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_from_huggingface_json(
      iree_make_cstring_view(kMinimalGPT2Config), allocator_, &tokenizer));
  ASSERT_NE(tokenizer, nullptr);

  // Use a multi-segment text that will emit tokens during feed.
  // "A\nB\nA" produces segments [A, \n, B, \n] emitted + [A] pending.
  const char* text = "A\nB\nA";
  auto ctx = CreateStreamingState(tokenizer, strlen(text),
                                  IREE_TOKENIZER_ENCODE_FLAG_AT_INPUT_START);

  int32_t token_buffer[256];

  // Feed first chunk.
  iree_string_view_t chunk1 = iree_make_cstring_view("A\n");
  iree_host_size_t bytes_consumed1 = 0;
  iree_host_size_t token_count1 = 0;
  IREE_ASSERT_OK(iree_tokenizer_encode_state_feed(
      ctx.state, chunk1,
      iree_tokenizer_make_token_output(token_buffer, NULL, NULL, 256),
      &bytes_consumed1, &token_count1));

  iree_host_size_t bound_after_first =
      iree_tokenizer_encode_state_pending_token_bound(ctx.state);

  // Feed second chunk — should emit more tokens, reducing pending.
  iree_string_view_t chunk2 = iree_make_cstring_view("B\nA");
  iree_host_size_t total_feed_tokens = token_count1;
  iree_host_size_t bytes_consumed2 = 0;
  iree_host_size_t token_count2 = 0;
  while (chunk2.size > 0) {
    IREE_ASSERT_OK(iree_tokenizer_encode_state_feed(
        ctx.state, chunk2,
        iree_tokenizer_make_token_output(token_buffer + total_feed_tokens, NULL,
                                         NULL, 256 - total_feed_tokens),
        &bytes_consumed2, &token_count2));
    total_feed_tokens += token_count2;
    chunk2.data += bytes_consumed2;
    chunk2.size -= bytes_consumed2;
    if (bytes_consumed2 == 0 && token_count2 == 0) break;
  }

  iree_host_size_t bound_after_second =
      iree_tokenizer_encode_state_pending_token_bound(ctx.state);

  // After feeding more text and emitting tokens, the bound should not be
  // greater than after the first chunk (we've consumed more of the pipeline).
  // Note: the bound may not strictly decrease because feeding also adds new
  // data, but after all text is fed and tokens emitted, the final bound should
  // be less than or equal to the initial bound.
  EXPECT_LE(bound_after_second, bound_after_first)
      << "Bound after feeding all text (" << bound_after_second
      << ") should not exceed bound after first chunk (" << bound_after_first
      << ")";

  iree_tokenizer_encode_state_deinitialize(ctx.state);
  iree_tokenizer_free(tokenizer);
}

// Test 3: pending_token_bound is 0 (or very small) when feed consumed all
// tokens and nothing remains in the pipeline.
TEST_F(TokenizerJsonTest, PendingTokenBoundZeroAfterFullConsumption) {
  iree_tokenizer_t* tokenizer = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_from_huggingface_json(
      iree_make_cstring_view(kMinimalGPT2Config), allocator_, &tokenizer));
  ASSERT_NE(tokenizer, nullptr);

  // With ByteLevel and the GPT-2 regex, segments are emitted on boundaries.
  // Feed the text, then check that the bound is small.
  // The trailing segment may be pending. The key test is that the bound is
  // at least as large as what finalize actually produces.
  const char* text = "A\nB";
  auto ctx = CreateStreamingState(tokenizer, strlen(text),
                                  IREE_TOKENIZER_ENCODE_FLAG_AT_INPUT_START);

  int32_t token_buffer[256];
  iree_host_size_t feed_tokens =
      FeedAll(ctx.state, iree_make_cstring_view(text), token_buffer, 256);

  iree_host_size_t bound =
      iree_tokenizer_encode_state_pending_token_bound(ctx.state);

  // Finalize to get actual count.
  iree_host_size_t finalize_count = 0;
  IREE_ASSERT_OK(iree_tokenizer_encode_state_finalize(
      ctx.state,
      iree_tokenizer_make_token_output(token_buffer + feed_tokens, NULL, NULL,
                                       256 - feed_tokens),
      &finalize_count));

  // The bound should be >= finalize_count (invariant) and reasonably small.
  EXPECT_GE(bound, finalize_count);

  // If finalize produced 0 tokens, the bound should also be 0.
  if (finalize_count == 0) {
    EXPECT_EQ(bound, 0u) << "Expected bound=0 when finalize produced 0 tokens";
  }

  iree_tokenizer_encode_state_deinitialize(ctx.state);
  iree_tokenizer_free(tokenizer);
}

// Test 4: pending_token_bound includes special tokens from postprocessor.
TEST_F(TokenizerJsonTest, PendingTokenBoundWithSpecialTokens) {
  // Build a tokenizer with BOS/EOS postprocessor. Uses the minimal GPT-2 vocab
  // (Ċ=0, ĊĊ=1, A=2, B=3) plus BOS (4) and EOS (5) special tokens.
  const char* json = R"({
    "model": {
      "type": "BPE",
      "vocab": {
        "\u010a": 0,
        "\u010a\u010a": 1,
        "A": 2,
        "B": 3,
        "<|bos|>": 4,
        "<|eos|>": 5
      },
      "merges": ["\u010a \u010a"]
    },
    "added_tokens": [
      {"id": 4, "content": "<|bos|>", "single_word": false, "lstrip": false,
       "rstrip": false, "normalized": false, "special": true},
      {"id": 5, "content": "<|eos|>", "single_word": false, "lstrip": false,
       "rstrip": false, "normalized": false, "special": true}
    ],
    "post_processor": {
      "type": "TemplateProcessing",
      "single": [
        {"SpecialToken": {"id": "<|bos|>", "type_id": 0}},
        {"Sequence": {"id": "A", "type_id": 0}},
        {"SpecialToken": {"id": "<|eos|>", "type_id": 0}}
      ],
      "pair": [],
      "special_tokens": {
        "<|bos|>": {"id": "<|bos|>", "ids": [4], "tokens": ["<|bos|>"]},
        "<|eos|>": {"id": "<|eos|>", "ids": [5], "tokens": ["<|eos|>"]}
      }
    },
    "pre_tokenizer": null
  })";

  iree_tokenizer_t* tokenizer = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_from_huggingface_json(
      iree_make_cstring_view(json), allocator_, &tokenizer));
  ASSERT_NE(tokenizer, nullptr);

  // Encode with add_special_tokens to activate the postprocessor.
  auto ctx =
      CreateStreamingState(tokenizer, 3,
                           IREE_TOKENIZER_ENCODE_FLAG_AT_INPUT_START |
                               IREE_TOKENIZER_ENCODE_FLAG_ADD_SPECIAL_TOKENS);

  int32_t token_buffer[256];
  iree_host_size_t feed_tokens =
      FeedAll(ctx.state, iree_make_cstring_view("AB"), token_buffer, 256);

  iree_host_size_t bound =
      iree_tokenizer_encode_state_pending_token_bound(ctx.state);

  // Finalize.
  iree_host_size_t finalize_count = 0;
  IREE_ASSERT_OK(iree_tokenizer_encode_state_finalize(
      ctx.state,
      iree_tokenizer_make_token_output(token_buffer + feed_tokens, NULL, NULL,
                                       256 - feed_tokens),
      &finalize_count));

  // Key invariant holds.
  EXPECT_GE(bound, finalize_count)
      << "pending_token_bound (" << bound << ") < actual finalize count ("
      << finalize_count << ") with special tokens";

  // The bound should account for the postprocessor special tokens (BOS, EOS).
  // With add_special_tokens, max_special_token_count is 2 (BOS + EOS).
  iree_host_size_t max_special =
      iree_tokenizer_max_special_token_count(tokenizer);
  EXPECT_EQ(max_special, 2u);

  // The total tokens (feed + finalize) should include BOS and EOS.
  iree_host_size_t total = feed_tokens + finalize_count;
  bool has_bos = false, has_eos = false;
  for (iree_host_size_t i = 0; i < total; ++i) {
    if (token_buffer[i] == 4) has_bos = true;
    if (token_buffer[i] == 5) has_eos = true;
  }
  EXPECT_TRUE(has_bos) << "Expected BOS token (4) in output";
  EXPECT_TRUE(has_eos) << "Expected EOS token (5) in output";

  iree_tokenizer_encode_state_deinitialize(ctx.state);
  iree_tokenizer_free(tokenizer);
}

}  // namespace
