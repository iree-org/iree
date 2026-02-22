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
  // The rstrip flag means the token only matches when followed by whitespace
  // or at the end of input.
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

}  // namespace
