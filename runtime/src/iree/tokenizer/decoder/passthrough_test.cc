// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/decoder/passthrough.h"

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
using testing::ProcessBatchedAndFinalize;
using testing::ScopedDecoder;
using testing::ScopedDecoderState;
using testing::TestLimitedOutputCapacity;
using testing::TestWithAllBatchSizes;
using testing::TestZeroCapacityOutput;
using testing::ToStringViews;

//===----------------------------------------------------------------------===//
// Test fixture for Passthrough decoder tests.
//===----------------------------------------------------------------------===//

class PassthroughDecoderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_tokenizer_decoder_t* raw_decoder = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_decoder_passthrough_allocate(
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

TEST_F(PassthroughDecoderTest, CreateAndDestroy) {
  EXPECT_NE(decoder(), nullptr);
}

TEST_F(PassthroughDecoderTest, StateSizeIsReasonable) {
  iree_host_size_t state_size = iree_tokenizer_decoder_state_size(decoder());
  EXPECT_GT(state_size, 0u);
  // Passthrough state should be tiny (just base struct).
  EXPECT_LE(state_size, 32u);
}

//===----------------------------------------------------------------------===//
// No-ops
//===----------------------------------------------------------------------===//

TEST_F(PassthroughDecoderTest, EmptyInput) {
  std::string result =
      ProcessAndFinalize(decoder(), {}, /*expect_pending_after_process=*/false);
  EXPECT_TRUE(result.empty());
}

TEST_F(PassthroughDecoderTest, ZeroCapacityOutput) {
  TestZeroCapacityOutput(decoder(), {"Hello", "World"});
}

//===----------------------------------------------------------------------===//
// Edge Cases
//===----------------------------------------------------------------------===//

TEST_F(PassthroughDecoderTest, SingleToken) {
  TestWithAllBatchSizes(decoder(), {"Hello"}, "Hello",
                        /*expect_pending_after_process=*/false);
}

TEST_F(PassthroughDecoderTest, EmptyToken) {
  // Empty tokens should be handled gracefully.
  std::string result = ProcessAndFinalize(
      decoder(), {""}, /*expect_pending_after_process=*/false);
  EXPECT_TRUE(result.empty());
}

TEST_F(PassthroughDecoderTest, MixedEmptyTokens) {
  // Mix of empty and non-empty tokens.
  TestWithAllBatchSizes(decoder(), {"", "a", "", "b", ""}, "ab",
                        /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// Basic Functionality
//===----------------------------------------------------------------------===//

TEST_F(PassthroughDecoderTest, ConcatenatesTokens) {
  TestWithAllBatchSizes(decoder(), {"Hello", " ", "World"}, "Hello World",
                        /*expect_pending_after_process=*/false);
}

TEST_F(PassthroughDecoderTest, PreservesContent) {
  // Test various content types.
  TestWithAllBatchSizes(decoder(), {"abc", "123", "!@#", "日本語"},
                        "abc123!@#日本語",
                        /*expect_pending_after_process=*/false);
}

TEST_F(PassthroughDecoderTest, PreservesWhitespace) {
  TestWithAllBatchSizes(decoder(), {"a", " ", "\t", "\n", "b"}, "a \t\nb",
                        /*expect_pending_after_process=*/false);
}

TEST_F(PassthroughDecoderTest, LongTokens) {
  // Test with longer tokens to ensure no length issues.
  std::string long_token(256, 'x');
  TestWithAllBatchSizes(decoder(), {long_token, long_token},
                        long_token + long_token,
                        /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// Streaming Behavior
//===----------------------------------------------------------------------===//

TEST_F(PassthroughDecoderTest, BatchedProcessing) {
  std::vector<std::string> tokens = {"The",   " ", "quick", " ",
                                     "brown", " ", "fox"};
  TestWithAllBatchSizes(decoder(), tokens, "The quick brown fox",
                        /*expect_pending_after_process=*/false);
}

TEST_F(PassthroughDecoderTest, HasPendingAlwaysFalse) {
  ScopedDecoderState state(decoder());
  EXPECT_FALSE(iree_tokenizer_decoder_state_has_pending(state.get()));

  // Process some data.
  std::vector<std::string> tokens = {"Hello"};
  auto views = ToStringViews(tokens);
  iree_tokenizer_string_list_t token_list = {views.size(), views.data()};

  char output[64];
  iree_host_size_t strings_consumed = 0;
  iree_host_size_t bytes_written = 0;
  IREE_ASSERT_OK(iree_tokenizer_decoder_state_process(
      state.get(), token_list,
      iree_make_mutable_string_view(output, sizeof(output)), &strings_consumed,
      &bytes_written));

  // Passthrough never has pending data.
  EXPECT_FALSE(iree_tokenizer_decoder_state_has_pending(state.get()));
}

TEST_F(PassthroughDecoderTest, FinalizeProducesNothing) {
  ScopedDecoderState state(decoder());

  // Process some data.
  std::vector<std::string> tokens = {"Hello"};
  auto views = ToStringViews(tokens);
  iree_tokenizer_string_list_t token_list = {views.size(), views.data()};

  char output[64];
  iree_host_size_t strings_consumed = 0;
  iree_host_size_t bytes_written = 0;
  IREE_ASSERT_OK(iree_tokenizer_decoder_state_process(
      state.get(), token_list,
      iree_make_mutable_string_view(output, sizeof(output)), &strings_consumed,
      &bytes_written));

  // Finalize should produce nothing.
  iree_host_size_t finalize_written = 0;
  IREE_ASSERT_OK(iree_tokenizer_decoder_state_finalize(
      state.get(), iree_make_mutable_string_view(output, sizeof(output)),
      &finalize_written));
  EXPECT_EQ(finalize_written, 0u);
}

//===----------------------------------------------------------------------===//
// Output Buffer Handling
//===----------------------------------------------------------------------===//

TEST_F(PassthroughDecoderTest, LimitedOutputCapacity) {
  TestLimitedOutputCapacity(decoder(), {"Hello", "World"}, "HelloWorld");
}

TEST_F(PassthroughDecoderTest, PartialTokenConsumption) {
  // When output buffer can't fit a whole token, it should stop.
  ScopedDecoderState state(decoder());

  std::vector<std::string> tokens = {"Hello", "World"};
  auto views = ToStringViews(tokens);
  iree_tokenizer_string_list_t token_list = {views.size(), views.data()};

  // Output buffer can only fit "Hello" (5 bytes), not "World" (5 more).
  char output[5];
  iree_host_size_t strings_consumed = 0;
  iree_host_size_t bytes_written = 0;

  IREE_ASSERT_OK(iree_tokenizer_decoder_state_process(
      state.get(), token_list,
      iree_make_mutable_string_view(output, sizeof(output)), &strings_consumed,
      &bytes_written));

  // Should consume first token and stop.
  EXPECT_EQ(strings_consumed, 1u);
  EXPECT_EQ(bytes_written, 5u);
  EXPECT_EQ(std::string(output, bytes_written), "Hello");
}

TEST_F(PassthroughDecoderTest, ConsumptionStopsAtBufferLimit) {
  // Test that we stop consuming when the next token won't fit.
  ScopedDecoderState state(decoder());

  std::vector<std::string> tokens = {"ab", "cd", "ef"};
  auto views = ToStringViews(tokens);
  iree_tokenizer_string_list_t token_list = {views.size(), views.data()};

  // Output buffer can fit "ab" and "cd" (4 bytes) but not "ef" too (6 total).
  char output[5];  // Can fit 2.5 tokens worth.
  iree_host_size_t strings_consumed = 0;
  iree_host_size_t bytes_written = 0;

  IREE_ASSERT_OK(iree_tokenizer_decoder_state_process(
      state.get(), token_list,
      iree_make_mutable_string_view(output, sizeof(output)), &strings_consumed,
      &bytes_written));

  // Should consume first two tokens (4 bytes), stop before third.
  EXPECT_EQ(strings_consumed, 2u);
  EXPECT_EQ(bytes_written, 4u);
  EXPECT_EQ(std::string(output, bytes_written), "abcd");
}

}  // namespace
}  // namespace iree::tokenizer
