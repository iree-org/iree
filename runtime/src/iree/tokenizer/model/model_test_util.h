// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Test utilities for model implementations.
//
// Provides:
// - RAII wrappers for automatic resource cleanup
// - Test helpers that verify has_pending() and finalize() behavior

#ifndef IREE_TOKENIZER_MODEL_MODEL_TEST_UTIL_H_
#define IREE_TOKENIZER_MODEL_MODEL_TEST_UTIL_H_

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/tokenizer/model.h"
#include "iree/tokenizer/testing/scoped_resource.h"
#include "iree/tokenizer/vocab/vocab.h"
#include "iree/tokenizer/vocab/vocab_builder.h"

namespace iree::tokenizer::testing {

//===----------------------------------------------------------------------===//
// RAII Wrappers
//===----------------------------------------------------------------------===//

// RAII wrapper for iree_tokenizer_vocab_builder_t*.
// Move-only; calls iree_tokenizer_vocab_builder_free() on destruction if not
// consumed by Build().
class ScopedVocabBuilder {
 public:
  explicit ScopedVocabBuilder(
      iree_allocator_t allocator = iree_allocator_system(),
      iree_host_size_t capacity_hint = 0) {
    IREE_CHECK_OK(iree_tokenizer_vocab_builder_allocate(capacity_hint,
                                                        allocator, &builder_));
  }

  ~ScopedVocabBuilder() {
    if (builder_) {
      iree_tokenizer_vocab_builder_free(builder_);
    }
  }

  ScopedVocabBuilder(ScopedVocabBuilder&& other) noexcept
      : builder_(std::exchange(other.builder_, nullptr)) {}
  ScopedVocabBuilder& operator=(ScopedVocabBuilder&& other) noexcept {
    if (this != &other) {
      if (builder_) {
        iree_tokenizer_vocab_builder_free(builder_);
      }
      builder_ = std::exchange(other.builder_, nullptr);
    }
    return *this;
  }

  ScopedVocabBuilder(const ScopedVocabBuilder&) = delete;
  ScopedVocabBuilder& operator=(const ScopedVocabBuilder&) = delete;

  // Adds a token with sequential ID.
  void AddToken(const char* text, iree_tokenizer_token_attr_t attrs = 0) {
    IREE_CHECK_OK(iree_tokenizer_vocab_builder_add_token(
        builder_, iree_make_cstring_view(text), 0.0f, attrs));
  }

  // Adds a token with explicit ID.
  void AddToken(int32_t id, const char* text,
                iree_tokenizer_token_attr_t attrs = 0) {
    IREE_CHECK_OK(iree_tokenizer_vocab_builder_add_token_with_id(
        builder_, id, iree_make_cstring_view(text), 0.0f, attrs));
  }

  // Adds a token with explicit ID and score (for Unigram models).
  void AddTokenWithScore(int32_t id, const char* text, float score,
                         iree_tokenizer_token_attr_t attrs = 0) {
    IREE_CHECK_OK(iree_tokenizer_vocab_builder_add_token_with_id(
        builder_, id, iree_make_cstring_view(text), score, attrs));
  }

  // Adds a BPE merge rule.
  void AddMerge(uint32_t left_id, uint32_t right_id) {
    IREE_CHECK_OK(
        iree_tokenizer_vocab_builder_add_merge(builder_, left_id, right_id));
  }

  // Sets a special token.
  void SetSpecialToken(iree_tokenizer_special_token_t type, int32_t id) {
    IREE_CHECK_OK(
        iree_tokenizer_vocab_builder_set_special_token(builder_, type, id));
  }

  iree_tokenizer_vocab_builder_t* get() { return builder_; }

  // Builds and returns the vocab, consuming this builder.
  // After calling Build(), the builder is invalid and must not be used.
  iree_tokenizer_vocab_t* Build() {
    iree_tokenizer_vocab_t* vocab = nullptr;
    IREE_CHECK_OK(iree_tokenizer_vocab_builder_build(builder_, &vocab));
    builder_ = nullptr;  // Consumed by build.
    return vocab;
  }

 private:
  iree_tokenizer_vocab_builder_t* builder_ = nullptr;
};

using ScopedVocab =
    ScopedResource<iree_tokenizer_vocab_t, iree_tokenizer_vocab_free>;

using ScopedModel =
    ScopedResource<iree_tokenizer_model_t, iree_tokenizer_model_free>;

using ScopedModelState =
    ScopedState<iree_tokenizer_model_t, iree_tokenizer_model_state_t,
                iree_tokenizer_model_state_size,
                iree_tokenizer_model_state_initialize,
                iree_tokenizer_model_state_deinitialize>;

//===----------------------------------------------------------------------===//
// Test Helpers
//===----------------------------------------------------------------------===//

// Result of encoding with offset tracking.
struct EncodeResult {
  std::vector<iree_tokenizer_token_id_t> tokens;
  std::vector<iree_tokenizer_offset_t> offsets;
};

// Encodes text through model with offset tracking enabled, returning both
// token IDs and byte offsets. Verifies lifecycle invariants.
EncodeResult EncodeWithOffsetsAndFinalize(
    iree_tokenizer_model_t* model, const std::string& text,
    const std::vector<iree_tokenizer_segment_t>& segments,
    bool expect_pending_after_encode);

// Convenience overload that creates a single segment covering the entire text.
EncodeResult EncodeWithOffsetsAndFinalize(iree_tokenizer_model_t* model,
                                          const std::string& text,
                                          bool expect_pending_after_encode);

// Encodes text through model using provided segments, returning token IDs.
// Verifies has_pending() matches |expect_pending_after_encode| after encode()
// and returns false after finalize().
std::vector<iree_tokenizer_token_id_t> EncodeAndFinalize(
    iree_tokenizer_model_t* model, const std::string& text,
    const std::vector<iree_tokenizer_segment_t>& segments,
    bool expect_pending_after_encode);

// Convenience overload that creates a single segment covering the entire text.
std::vector<iree_tokenizer_token_id_t> EncodeAndFinalize(
    iree_tokenizer_model_t* model, const std::string& text,
    bool expect_pending_after_encode);

// Tests that encoding produces expected tokens.
void TestEncode(iree_tokenizer_model_t* model, const std::string& text,
                const std::vector<iree_tokenizer_segment_t>& segments,
                const std::vector<iree_tokenizer_token_id_t>& expected_tokens,
                bool expect_pending_after_encode);

// Convenience overload with single segment covering entire text.
void TestEncode(iree_tokenizer_model_t* model, const std::string& text,
                const std::vector<iree_tokenizer_token_id_t>& expected_tokens,
                bool expect_pending_after_encode);

// Tests multiple independent encode calls with state reuse.
void TestMultipleEncodeCalls(
    iree_tokenizer_model_t* model, const std::string& text1,
    const std::vector<iree_tokenizer_segment_t>& segments1,
    const std::vector<iree_tokenizer_token_id_t>& expected1,
    const std::string& text2,
    const std::vector<iree_tokenizer_segment_t>& segments2,
    const std::vector<iree_tokenizer_token_id_t>& expected2);

// Tests output buffer overflow handling with capacity=1.
void TestLimitedOutputCapacity(
    iree_tokenizer_model_t* model, const std::string& text,
    const std::vector<iree_tokenizer_segment_t>& segments,
    const std::vector<iree_tokenizer_token_id_t>& expected_tokens);

}  // namespace iree::tokenizer::testing

#endif  // IREE_TOKENIZER_MODEL_MODEL_TEST_UTIL_H_
