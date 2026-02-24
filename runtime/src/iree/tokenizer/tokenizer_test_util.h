// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOKENIZER_TOKENIZER_TEST_UTIL_H_
#define IREE_TOKENIZER_TOKENIZER_TEST_UTIL_H_

#include <memory>
#include <string>
#include <vector>

#include "iree/base/api.h"
#include "iree/tokenizer/model/bpe.h"
#include "iree/tokenizer/model/wordpiece.h"
#include "iree/tokenizer/normalizer/bert.h"
#include "iree/tokenizer/segmenter/bert.h"
#include "iree/tokenizer/segmenter/punctuation.h"
#include "iree/tokenizer/segmenter/whitespace.h"
#include "iree/tokenizer/testing/scoped_resource.h"
#include "iree/tokenizer/tokenizer.h"
#include "iree/tokenizer/vocab/vocab.h"
#include "iree/tokenizer/vocab/vocab_builder.h"

namespace iree::tokenizer::testing {

//===----------------------------------------------------------------------===//
// RAII Helpers
//===----------------------------------------------------------------------===//

struct TokenizerDeleter {
  void operator()(iree_tokenizer_t* t) const { iree_tokenizer_free(t); }
};
using TokenizerPtr = std::unique_ptr<iree_tokenizer_t, TokenizerDeleter>;

static inline iree::StatusOr<TokenizerPtr> BuildTokenizerOr(
    iree_tokenizer_builder_t* builder) {
  iree_tokenizer_t* tokenizer = nullptr;
  iree_status_t status = iree_tokenizer_builder_build(builder, &tokenizer);
  iree_tokenizer_builder_deinitialize(builder);
  if (!iree_status_is_ok(status)) {
    return iree::Status(std::move(status));
  }
  return TokenizerPtr(tokenizer);
}

using ScopedVocab =
    ScopedResource<iree_tokenizer_vocab_t, iree_tokenizer_vocab_free>;

// RAII wrapper for vocab builder.
class ScopedVocabBuilder {
 public:
  explicit ScopedVocabBuilder(iree_host_size_t capacity = 64)
      : vocab_builder_(nullptr), built_(false) {
    IREE_CHECK_OK(iree_tokenizer_vocab_builder_allocate(
        capacity, iree_allocator_system(), &vocab_builder_));
  }

  ~ScopedVocabBuilder() {
    if (vocab_builder_ && !built_) {
      iree_tokenizer_vocab_builder_free(vocab_builder_);
    }
  }

  void AddToken(
      iree_tokenizer_token_id_t id, const char* text,
      iree_tokenizer_token_attr_t attrs = IREE_TOKENIZER_TOKEN_ATTR_NONE) {
    IREE_CHECK_OK(iree_tokenizer_vocab_builder_add_token_with_id(
        vocab_builder_, id, iree_make_cstring_view(text), 0.0f, attrs));
  }

  // Adds a token with explicit score (for Unigram models).
  void AddTokenWithScore(
      iree_tokenizer_token_id_t id, const char* text, float score,
      iree_tokenizer_token_attr_t attrs = IREE_TOKENIZER_TOKEN_ATTR_NONE) {
    IREE_CHECK_OK(iree_tokenizer_vocab_builder_add_token_with_id(
        vocab_builder_, id, iree_make_cstring_view(text), score, attrs));
  }

  void AddMerge(iree_tokenizer_token_id_t left,
                iree_tokenizer_token_id_t right) {
    IREE_CHECK_OK(
        iree_tokenizer_vocab_builder_add_merge(vocab_builder_, left, right));
  }

  // Sets a special token (UNK, BOS, EOS, etc.).
  void SetSpecialToken(iree_tokenizer_special_token_t type, int32_t id) {
    IREE_CHECK_OK(iree_tokenizer_vocab_builder_set_special_token(vocab_builder_,
                                                                 type, id));
  }

  // Builds and returns vocab wrapped in RAII. Builder becomes invalid after
  // this call. Use .get() to access, .release() to transfer ownership.
  ScopedVocab Build() {
    iree_tokenizer_vocab_t* vocab = nullptr;
    IREE_CHECK_OK(iree_tokenizer_vocab_builder_build(vocab_builder_, &vocab));
    built_ = true;
    return ScopedVocab(vocab);
  }

 private:
  iree_tokenizer_vocab_builder_t* vocab_builder_;
  bool built_;
};

// RAII wrapper for builder lifecycle.
class ScopedBuilder {
 public:
  explicit ScopedBuilder() {
    iree_tokenizer_builder_initialize(iree_allocator_system(), &builder_);
  }
  ~ScopedBuilder() { iree_tokenizer_builder_deinitialize(&builder_); }

  ScopedBuilder(const ScopedBuilder&) = delete;
  ScopedBuilder& operator=(const ScopedBuilder&) = delete;

  iree_tokenizer_builder_t* get() { return &builder_; }

 private:
  iree_tokenizer_builder_t builder_;
};

//===----------------------------------------------------------------------===//
// Factory Helpers
//===----------------------------------------------------------------------===//

// Creates a BPE model referencing the given vocab.
// The BPE model does not own the vocab — caller must ensure vocab outlives
// model, or transfer vocab ownership to the tokenizer builder.
static inline iree_tokenizer_model_t* CreateBPEModel(
    const iree_tokenizer_vocab_t* vocab,
    iree_tokenizer_bpe_flags_t flags = IREE_TOKENIZER_BPE_FLAG_NONE) {
  iree_tokenizer_model_t* model = nullptr;
  IREE_CHECK_OK(iree_tokenizer_bpe_model_allocate(
      vocab, flags, iree_allocator_system(), &model));
  return model;
}

// Creates a BPE model with IGNORE_MERGES flag.
// Use this for whole-word vocabs that don't have character-level tokens.
// This matches HuggingFace's ignore_merges=true behavior.
static inline iree_tokenizer_model_t* CreateBPEModelIgnoreMerges(
    const iree_tokenizer_vocab_t* vocab) {
  return CreateBPEModel(vocab, IREE_TOKENIZER_BPE_FLAG_IGNORE_MERGES);
}

// Creates a whitespace segmenter.
static inline iree_tokenizer_segmenter_t* CreateWhitespaceSegmenter() {
  iree_tokenizer_segmenter_t* segmenter = nullptr;
  IREE_CHECK_OK(iree_tokenizer_segmenter_whitespace_allocate(
      iree_allocator_system(), &segmenter));
  return segmenter;
}

// Creates a WordPiece model referencing the given vocab.
// |continuing_subword_prefix| defaults to "##" (standard BERT prefix).
// |max_input_chars_per_word| defaults to 200 (standard BERT limit).
// The WordPiece model does not own the vocab — caller must ensure vocab
// outlives model, or transfer vocab ownership to the tokenizer builder.
static inline iree_tokenizer_model_t* CreateWordPieceModel(
    const iree_tokenizer_vocab_t* vocab,
    iree_string_view_t continuing_subword_prefix = IREE_SV("##"),
    iree_host_size_t max_input_chars_per_word = 200,
    iree_tokenizer_wordpiece_flags_t flags =
        IREE_TOKENIZER_WORDPIECE_FLAG_NONE) {
  iree_tokenizer_model_t* model = nullptr;
  IREE_CHECK_OK(iree_tokenizer_wordpiece_model_allocate(
      vocab, continuing_subword_prefix, max_input_chars_per_word, flags,
      iree_allocator_system(), &model));
  return model;
}

// Creates a BERT normalizer.
// Default flags enable all BERT normalization: clean_text,
// handle_chinese_chars, strip_accents, and lowercase. Pass specific flags for
// cased models.
static inline iree_tokenizer_normalizer_t* CreateBertNormalizer(
    iree_tokenizer_bert_normalizer_flags_t flags =
        IREE_TOKENIZER_BERT_NORMALIZER_FLAG_DEFAULT) {
  iree_tokenizer_normalizer_t* normalizer = nullptr;
  IREE_CHECK_OK(iree_tokenizer_normalizer_bert_allocate(
      flags, iree_allocator_system(), &normalizer));
  return normalizer;
}

// Creates a BERT segmenter (whitespace + punctuation splitting).
static inline iree_tokenizer_segmenter_t* CreateBertSegmenter() {
  iree_tokenizer_segmenter_t* segmenter = nullptr;
  IREE_CHECK_OK(iree_tokenizer_segmenter_bert_allocate(iree_allocator_system(),
                                                       &segmenter));
  return segmenter;
}

// Creates a punctuation segmenter with the given delimiter behavior.
static inline iree_tokenizer_segmenter_t* CreatePunctuationSegmenter(
    iree_tokenizer_regex_split_behavior_t behavior =
        IREE_TOKENIZER_UTIL_REGEX_SPLIT_ISOLATED) {
  iree_tokenizer_segmenter_t* segmenter = nullptr;
  IREE_CHECK_OK(iree_tokenizer_segmenter_punctuation_allocate(
      behavior, iree_allocator_system(), &segmenter));
  return segmenter;
}

// Builds a tokenizer from a builder. Returns nullptr on failure.
static inline iree_tokenizer_t* BuildTokenizer(
    iree_tokenizer_builder_t* builder) {
  iree_tokenizer_t* tokenizer = nullptr;
  iree_status_t status = iree_tokenizer_builder_build(builder, &tokenizer);
  if (!iree_status_is_ok(status)) {
    iree_status_free(status);
    return nullptr;
  }
  return tokenizer;
}

// Encodes text and returns token IDs.
static inline std::vector<iree_tokenizer_token_id_t> Encode(
    iree_tokenizer_t* tokenizer, const char* text) {
  std::vector<iree_tokenizer_token_id_t> token_ids(256);
  iree_host_size_t token_count = 0;

  iree_status_t status = iree_tokenizer_encode(
      tokenizer, iree_make_cstring_view(text), IREE_TOKENIZER_ENCODE_FLAG_NONE,
      iree_tokenizer_make_token_output(token_ids.data(), NULL, NULL,
                                       token_ids.size()),
      iree_allocator_system(), &token_count);

  if (!iree_status_is_ok(status)) {
    iree_status_free(status);
    return {};
  }

  token_ids.resize(token_count);
  return token_ids;
}

//===----------------------------------------------------------------------===//
// State Storage Helpers
//===----------------------------------------------------------------------===//

class EncodeStateStorage {
 public:
  ~EncodeStateStorage() {
    if (state_) {
      iree_tokenizer_encode_state_deinitialize(state_);
    }
  }

  EncodeStateStorage(EncodeStateStorage&& other) noexcept
      : storage_(std::move(other.storage_)),
        transform_buffer_(std::move(other.transform_buffer_)),
        offset_runs_(std::move(other.offset_runs_)),
        state_(other.state_) {
    other.state_ = nullptr;
  }

  iree_tokenizer_encode_state_t* state() const { return state_; }

  static iree::StatusOr<EncodeStateStorage> Allocate(
      const iree_tokenizer_t* tokenizer,
      iree_host_size_t transform_buffer_capacity,
      iree_host_size_t offset_run_capacity,
      iree_tokenizer_encode_flags_t flags) {
    EncodeStateStorage storage;
    storage.transform_buffer_.resize(transform_buffer_capacity);
    storage.offset_runs_.resize(offset_run_capacity);

    iree_host_size_t state_size = 0;
    IREE_RETURN_IF_ERROR(
        iree_tokenizer_encode_state_calculate_size(tokenizer, &state_size));
    storage.storage_.resize(state_size);

    iree_tokenizer_offset_run_list_t offset_runs = {
        /*.capacity=*/storage.offset_runs_.size(),
        /*.values=*/storage.offset_runs_.empty() ? nullptr
                                                 : storage.offset_runs_.data(),
    };
    IREE_RETURN_IF_ERROR(iree_tokenizer_encode_state_initialize(
        tokenizer,
        iree_make_byte_span(storage.storage_.data(), storage.storage_.size()),
        iree_make_byte_span(storage.transform_buffer_.data(),
                            storage.transform_buffer_.size()),
        offset_runs, flags, &storage.state_));

    return storage;
  }

  EncodeStateStorage(const EncodeStateStorage&) = delete;
  EncodeStateStorage& operator=(const EncodeStateStorage&) = delete;

 private:
  EncodeStateStorage() = default;

  std::vector<uint8_t> storage_;
  std::vector<uint8_t> transform_buffer_;
  std::vector<iree_tokenizer_offset_run_t> offset_runs_;
  iree_tokenizer_encode_state_t* state_ = nullptr;
};

class DecodeStateStorage {
 public:
  ~DecodeStateStorage() {
    if (state_) {
      iree_tokenizer_decode_state_deinitialize(state_);
    }
  }

  DecodeStateStorage(DecodeStateStorage&& other) noexcept
      : storage_(std::move(other.storage_)), state_(other.state_) {
    other.state_ = nullptr;
  }

  iree_tokenizer_decode_state_t* state() const { return state_; }

  static iree::StatusOr<DecodeStateStorage> Allocate(
      const iree_tokenizer_t* tokenizer,
      iree_tokenizer_decode_flags_t flags = IREE_TOKENIZER_DECODE_FLAG_NONE) {
    DecodeStateStorage storage;

    iree_host_size_t state_size = 0;
    IREE_RETURN_IF_ERROR(
        iree_tokenizer_decode_state_calculate_size(tokenizer, &state_size));
    storage.storage_.resize(state_size);

    IREE_RETURN_IF_ERROR(iree_tokenizer_decode_state_initialize(
        tokenizer, flags,
        iree_make_byte_span(storage.storage_.data(), storage.storage_.size()),
        &storage.state_));

    return storage;
  }

  DecodeStateStorage(const DecodeStateStorage&) = delete;
  DecodeStateStorage& operator=(const DecodeStateStorage&) = delete;

 private:
  DecodeStateStorage() = default;

  std::vector<uint8_t> storage_;
  iree_tokenizer_decode_state_t* state_ = nullptr;
};

// Decodes a list of token IDs to a string.
// Returns StatusOr so the caller can assert with proper test diagnostics.
static inline iree::StatusOr<std::string> Decode(
    iree_tokenizer_t* tokenizer, const std::vector<int32_t>& token_ids,
    iree_tokenizer_decode_flags_t flags = IREE_TOKENIZER_DECODE_FLAG_NONE) {
  IREE_ASSIGN_OR_RETURN(auto state,
                        DecodeStateStorage::Allocate(tokenizer, flags));

  std::vector<char> output(token_ids.size() * 32 + 256);
  iree_host_size_t tokens_consumed = 0;
  iree_host_size_t text_length = 0;

  iree_tokenizer_token_id_list_t id_list = {
      /*.count=*/token_ids.size(),
      /*.values=*/token_ids.data(),
  };
  IREE_RETURN_IF_ERROR(iree_tokenizer_decode_state_feed(
      state.state(), id_list,
      iree_make_mutable_string_view(output.data(), output.size()),
      &tokens_consumed, &text_length));

  if (tokens_consumed != token_ids.size()) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "only consumed %" PRIhsz " of %" PRIhsz " tokens",
                            tokens_consumed,
                            (iree_host_size_t)token_ids.size());
  }

  // Finalize to flush any pending decoder state.
  iree_host_size_t finalize_length = 0;
  IREE_RETURN_IF_ERROR(iree_tokenizer_decode_state_finalize(
      state.state(),
      iree_make_mutable_string_view(output.data() + text_length,
                                    output.size() - text_length),
      &finalize_length));

  return std::string(output.data(), text_length + finalize_length);
}

}  // namespace iree::tokenizer::testing

#endif  // IREE_TOKENIZER_TOKENIZER_TEST_UTIL_H_
