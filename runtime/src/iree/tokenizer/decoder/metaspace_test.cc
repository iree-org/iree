// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/decoder/metaspace.h"

#include <cstring>
#include <string>
#include <vector>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/tokenizer/decoder/decoder_test_util.h"

namespace iree::tokenizer {
namespace {

using testing::ProcessAndFinalize;
using testing::ScopedDecoder;
using testing::ScopedDecoderState;
using testing::TestWithAllBatchSizes;
using testing::TestZeroCapacityOutput;
using testing::ToStringViews;

// U+2581 (LOWER ONE EIGHTH BLOCK) in UTF-8.
const std::string kMetaspace = "\xE2\x96\x81";

//===----------------------------------------------------------------------===//
// Test fixtures
//===----------------------------------------------------------------------===//

class MetaspaceDecoderAlwaysTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_tokenizer_decoder_t* raw_decoder = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_decoder_metaspace_allocate(
        0, IREE_TOKENIZER_DECODER_METASPACE_PREPEND_ALWAYS,
        iree_allocator_system(), &raw_decoder));
    decoder_ = ScopedDecoder(raw_decoder);
  }
  iree_tokenizer_decoder_t* decoder() { return decoder_.get(); }

 private:
  ScopedDecoder decoder_;
};

class MetaspaceDecoderNeverTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_tokenizer_decoder_t* raw_decoder = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_decoder_metaspace_allocate(
        0, IREE_TOKENIZER_DECODER_METASPACE_PREPEND_NEVER,
        iree_allocator_system(), &raw_decoder));
    decoder_ = ScopedDecoder(raw_decoder);
  }
  iree_tokenizer_decoder_t* decoder() { return decoder_.get(); }

 private:
  ScopedDecoder decoder_;
};

class MetaspaceDecoderFirstTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_tokenizer_decoder_t* raw_decoder = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_decoder_metaspace_allocate(
        0, IREE_TOKENIZER_DECODER_METASPACE_PREPEND_FIRST,
        iree_allocator_system(), &raw_decoder));
    decoder_ = ScopedDecoder(raw_decoder);
  }
  iree_tokenizer_decoder_t* decoder() { return decoder_.get(); }

 private:
  ScopedDecoder decoder_;
};

//===----------------------------------------------------------------------===//
// Lifecycle
//===----------------------------------------------------------------------===//

TEST_F(MetaspaceDecoderAlwaysTest, CreateAndDestroy) {
  EXPECT_NE(decoder(), nullptr);
}

TEST_F(MetaspaceDecoderAlwaysTest, StateSizeIsReasonable) {
  iree_host_size_t state_size = iree_tokenizer_decoder_state_size(decoder());
  EXPECT_GT(state_size, 0u);
  EXPECT_LE(state_size, 64u);
}

//===----------------------------------------------------------------------===//
// No-ops
//===----------------------------------------------------------------------===//

TEST_F(MetaspaceDecoderAlwaysTest, EmptyInput) {
  std::string result =
      ProcessAndFinalize(decoder(), {}, /*expect_pending_after_process=*/false);
  EXPECT_TRUE(result.empty());
}

TEST_F(MetaspaceDecoderAlwaysTest, ZeroCapacityOutput) {
  TestZeroCapacityOutput(decoder(), {kMetaspace + "Hello"});
}

//===----------------------------------------------------------------------===//
// Edge Cases
//===----------------------------------------------------------------------===//

TEST_F(MetaspaceDecoderAlwaysTest, SingleTokenNoMetaspace) {
  TestWithAllBatchSizes(decoder(), {"Hello"}, "Hello",
                        /*expect_pending_after_process=*/false);
}

TEST_F(MetaspaceDecoderAlwaysTest, SingleMetaspaceOnly) {
  // Leading metaspace stripped with ALWAYS.
  TestWithAllBatchSizes(decoder(), {kMetaspace}, "",
                        /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// Basic Functionality
//===----------------------------------------------------------------------===//

TEST_F(MetaspaceDecoderAlwaysTest, ReplacesMetaspaceWithSpace) {
  // "▁Hello▁World" -> " Hello World" but leading space stripped.
  TestWithAllBatchSizes(decoder(),
                        {kMetaspace + "Hello" + kMetaspace + "World"},
                        "Hello World", /*expect_pending_after_process=*/false);
}

TEST_F(MetaspaceDecoderAlwaysTest, MultipleTokens) {
  // Tokens: ["▁Hello", "▁World"] -> "Hello World" (first ▁ stripped).
  TestWithAllBatchSizes(decoder(), {kMetaspace + "Hello", kMetaspace + "World"},
                        "Hello World", /*expect_pending_after_process=*/false);
}

TEST_F(MetaspaceDecoderAlwaysTest, MetaspaceInMiddle) {
  TestWithAllBatchSizes(decoder(), {"Hello" + kMetaspace + "World"},
                        "Hello World", /*expect_pending_after_process=*/false);
}

TEST_F(MetaspaceDecoderAlwaysTest, ConsecutiveMetaspaces) {
  // Multiple metaspaces become multiple spaces (except first is stripped).
  TestWithAllBatchSizes(decoder(), {kMetaspace + kMetaspace + "Hello"},
                        " Hello",
                        /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// Prepend Scheme Variations
//===----------------------------------------------------------------------===//

TEST_F(MetaspaceDecoderNeverTest, NeverStripsLeading) {
  // NEVER: leading metaspace becomes space.
  TestWithAllBatchSizes(decoder(), {kMetaspace + "Hello"}, " Hello",
                        /*expect_pending_after_process=*/false);
}

TEST_F(MetaspaceDecoderNeverTest, MultipleTokensNeverStrip) {
  TestWithAllBatchSizes(decoder(), {kMetaspace + "Hello", kMetaspace + "World"},
                        " Hello World", /*expect_pending_after_process=*/false);
}

TEST_F(MetaspaceDecoderFirstTest, FirstStripsOnlyFirst) {
  // FIRST: only the very first metaspace is stripped.
  TestWithAllBatchSizes(decoder(), {kMetaspace + "Hello"}, "Hello",
                        /*expect_pending_after_process=*/false);
}

TEST_F(MetaspaceDecoderFirstTest, SecondTokenNotStripped) {
  // First token's leading ▁ stripped, second token's ▁ becomes space.
  TestWithAllBatchSizes(decoder(), {kMetaspace + "Hello", kMetaspace + "World"},
                        "Hello World", /*expect_pending_after_process=*/false);
}

TEST_F(MetaspaceDecoderFirstTest, NoLeadingMetaspace) {
  // If first token doesn't start with metaspace, nothing special.
  TestWithAllBatchSizes(decoder(), {"Hello", kMetaspace + "World"},
                        "Hello World", /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// Streaming Behavior
//===----------------------------------------------------------------------===//

TEST_F(MetaspaceDecoderAlwaysTest, BatchedProcessing) {
  std::vector<std::string> tokens = {kMetaspace + "The", kMetaspace + "quick",
                                     kMetaspace + "brown", kMetaspace + "fox"};
  TestWithAllBatchSizes(decoder(), tokens, "The quick brown fox",
                        /*expect_pending_after_process=*/false);
}

TEST_F(MetaspaceDecoderAlwaysTest, HasPendingAlwaysFalse) {
  ScopedDecoderState state(decoder());
  EXPECT_FALSE(iree_tokenizer_decoder_state_has_pending(state.get()));

  std::vector<std::string> tokens = {kMetaspace + "Hello"};
  auto views = ToStringViews(tokens);
  iree_tokenizer_string_list_t token_list = {views.size(), views.data()};

  char output[64];
  iree_host_size_t strings_consumed = 0;
  iree_host_size_t bytes_written = 0;
  IREE_ASSERT_OK(iree_tokenizer_decoder_state_process(
      state.get(), token_list,
      iree_make_mutable_string_view(output, sizeof(output)), &strings_consumed,
      &bytes_written));

  EXPECT_FALSE(iree_tokenizer_decoder_state_has_pending(state.get()));
}

//===----------------------------------------------------------------------===//
// Mixed Content
//===----------------------------------------------------------------------===//

TEST_F(MetaspaceDecoderAlwaysTest, MixedUtf8Content) {
  // UTF-8 content with metaspaces.
  TestWithAllBatchSizes(decoder(),
                        {kMetaspace + "café" + kMetaspace + "日本語"},
                        "café 日本語", /*expect_pending_after_process=*/false);
}

TEST_F(MetaspaceDecoderAlwaysTest, OnlyMetaspaces) {
  // Multiple metaspaces: first stripped, rest become spaces.
  TestWithAllBatchSizes(decoder(), {kMetaspace, kMetaspace, kMetaspace}, "  ",
                        /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// HuggingFace Compatibility
//===----------------------------------------------------------------------===//

TEST_F(MetaspaceDecoderAlwaysTest, HF_SentencePieceStyle) {
  // Standard SentencePiece output: "▁Hello▁world▁!" -> "Hello world !"
  std::vector<std::string> tokens = {kMetaspace + "Hello", kMetaspace + "world",
                                     kMetaspace + "!"};
  TestWithAllBatchSizes(decoder(), tokens, "Hello world !",
                        /*expect_pending_after_process=*/false);
}

TEST_F(MetaspaceDecoderAlwaysTest, HF_LlamaStyle) {
  // LLaMA tokenizer output.
  std::vector<std::string> tokens = {kMetaspace + "The", kMetaspace + "quick",
                                     kMetaspace + "brown", kMetaspace + "fox"};
  TestWithAllBatchSizes(decoder(), tokens, "The quick brown fox",
                        /*expect_pending_after_process=*/false);
}

}  // namespace
}  // namespace iree::tokenizer
