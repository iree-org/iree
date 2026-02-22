// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/decoder/strip.h"

#include <cstring>
#include <string>
#include <vector>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/tokenizer/decoder/decoder_test_util.h"

namespace iree::tokenizer {
namespace {

using testing::MakeStringList;
using testing::ProcessAndFinalize;
using testing::ScopedDecoder;
using testing::ScopedDecoderState;
using testing::TestLimitedOutputCapacity;
using testing::TestWithAllBatchSizes;
using testing::TestZeroCapacityOutput;
using testing::ToStringViews;

//===----------------------------------------------------------------------===//
// Test fixtures
//===----------------------------------------------------------------------===//

class StripDecoderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Default config: strip up to 1 leading space (matches all real configs).
    iree_tokenizer_decoder_t* raw_decoder = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_decoder_strip_allocate(
        IREE_SV(" "), /*start_count=*/1, /*stop_count=*/0,
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

TEST_F(StripDecoderTest, CreateAndDestroy) { EXPECT_NE(decoder(), nullptr); }

TEST_F(StripDecoderTest, StateSizeIsReasonable) {
  iree_host_size_t state_size = iree_tokenizer_decoder_state_size(decoder());
  EXPECT_GT(state_size, 0u);
  EXPECT_LE(state_size, 64u);
}

//===----------------------------------------------------------------------===//
// No-ops
//===----------------------------------------------------------------------===//

TEST_F(StripDecoderTest, EmptyInput) {
  std::string result =
      ProcessAndFinalize(decoder(), {}, /*expect_pending_after_process=*/false);
  EXPECT_TRUE(result.empty());
}

TEST_F(StripDecoderTest, ZeroCapacityOutput) {
  TestZeroCapacityOutput(decoder(), {" Hello"});
}

//===----------------------------------------------------------------------===//
// Edge Cases
//===----------------------------------------------------------------------===//

TEST_F(StripDecoderTest, NoLeadingSpace) {
  // No leading space to strip.
  TestWithAllBatchSizes(decoder(), {"Hello"}, "Hello",
                        /*expect_pending_after_process=*/false);
}

TEST_F(StripDecoderTest, OnlySpace) {
  // Single space stripped entirely.
  TestWithAllBatchSizes(decoder(), {" "}, "",
                        /*expect_pending_after_process=*/false);
}

TEST_F(StripDecoderTest, TwoSpaces) {
  // Only first space stripped.
  TestWithAllBatchSizes(decoder(), {"  "}, " ",
                        /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// Basic Functionality
//===----------------------------------------------------------------------===//

TEST_F(StripDecoderTest, StripsOneLeadingSpace) {
  TestWithAllBatchSizes(decoder(), {" Hello"}, "Hello",
                        /*expect_pending_after_process=*/false);
}

TEST_F(StripDecoderTest, StripsOnlyFirstSpace) {
  // Multiple leading spaces: only first is stripped.
  TestWithAllBatchSizes(decoder(), {"  Hello"}, " Hello",
                        /*expect_pending_after_process=*/false);
}

TEST_F(StripDecoderTest, MiddleSpaceNotStripped) {
  // Spaces in middle are preserved.
  TestWithAllBatchSizes(decoder(), {"Hello World"}, "Hello World",
                        /*expect_pending_after_process=*/false);
}

TEST_F(StripDecoderTest, TrailingSpaceNotStripped) {
  // Trailing spaces preserved (stop_count=0).
  TestWithAllBatchSizes(decoder(), {" Hello "}, "Hello ",
                        /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// Per-Token Stripping (HuggingFace Semantics)
//===----------------------------------------------------------------------===//

TEST_F(StripDecoderTest, StripsFromEachToken) {
  // Strip removes leading content from EACH token independently.
  // Per HuggingFace semantics, start_count applies per-token.
  TestWithAllBatchSizes(decoder(), {" Hello", " World"}, "HelloWorld",
                        /*expect_pending_after_process=*/false);
}

TEST_F(StripDecoderTest, TokenWithoutLeadingContent) {
  // Token without leading space is unchanged.
  TestWithAllBatchSizes(decoder(), {"Hello", " World"}, "HelloWorld",
                        /*expect_pending_after_process=*/false);
}

TEST_F(StripDecoderTest, SpaceOnlyToken) {
  // Token that is just the content to strip becomes empty.
  TestWithAllBatchSizes(decoder(), {" ", "Hello"}, "Hello",
                        /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// Multi-Character Content
//===----------------------------------------------------------------------===//

TEST(StripDecoderMultiCharTest, StripsMultiByteContent) {
  // Strip leading tab character.
  iree_tokenizer_decoder_t* raw_decoder = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_decoder_strip_allocate(
      IREE_SV("\t"), /*start_count=*/2, /*stop_count=*/0,
      iree_allocator_system(), &raw_decoder));
  ScopedDecoder decoder(raw_decoder);

  TestWithAllBatchSizes(decoder.get(), {"\t\tHello"}, "Hello",
                        /*expect_pending_after_process=*/false);
}

TEST(StripDecoderMultiCharTest, StripsMultiCharContent) {
  // Strip "##" prefix (common in WordPiece).
  iree_tokenizer_decoder_t* raw_decoder = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_decoder_strip_allocate(
      IREE_SV("##"), /*start_count=*/1, /*stop_count=*/0,
      iree_allocator_system(), &raw_decoder));
  ScopedDecoder decoder(raw_decoder);

  TestWithAllBatchSizes(decoder.get(), {"##Hello"}, "Hello",
                        /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// Validation
//===----------------------------------------------------------------------===//

TEST(StripDecoderValidationTest, RejectsEmptyContent) {
  iree_tokenizer_decoder_t* decoder = nullptr;
  iree_status_t status = iree_tokenizer_decoder_strip_allocate(
      IREE_SV(""), /*start_count=*/1, /*stop_count=*/0, iree_allocator_system(),
      &decoder);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
}

TEST(StripDecoderValidationTest, RejectsStopCountGreaterThanZero) {
  iree_tokenizer_decoder_t* decoder = nullptr;
  iree_status_t status = iree_tokenizer_decoder_strip_allocate(
      IREE_SV(" "), /*start_count=*/1, /*stop_count=*/1,
      iree_allocator_system(), &decoder);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED, status);
}

//===----------------------------------------------------------------------===//
// HuggingFace Compatibility
//===----------------------------------------------------------------------===//

TEST_F(StripDecoderTest, HF_MistralStyle) {
  // Mistral uses: content=" ", start=1, stop=0
  // This is what the default fixture is configured as.
  // Per HuggingFace semantics: strip leading space from EACH token.
  // Result is all tokens concatenated without their leading spaces.
  TestWithAllBatchSizes(decoder(), {" The", " quick", " brown", " fox"},
                        "Thequickbrownfox",
                        /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// Buffer-Full Edge Cases
//===----------------------------------------------------------------------===//

TEST_F(StripDecoderTest, LimitedOutputCapacity) {
  TestLimitedOutputCapacity(decoder(), {" Hello", " World"}, "HelloWorld");
}

TEST_F(StripDecoderTest, BufferFullMidTokenWithResume) {
  // Test mid-token buffer overflow: Strip doesn't need resume state since
  // stripping only affects the very beginning of the stream. Once we output
  // anything, no more stripping happens, so the token must either fit or not.
  // This test verifies the limited capacity utility handles Strip correctly.
  TestLimitedOutputCapacity(decoder(), {" abcdef"}, "abcdef");
}

TEST_F(StripDecoderTest, MultipleTokensWithLimitedCapacity) {
  TestLimitedOutputCapacity(decoder(), {" Hello", " World", "!"},
                            "HelloWorld!");
}

}  // namespace
}  // namespace iree::tokenizer
