// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/normalizer/passthrough.h"

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/tokenizer/normalizer/normalizer_test_util.h"

namespace iree::tokenizer {
namespace {

using testing::ProcessAndFinalize;
using testing::ScopedNormalizer;
using testing::ScopedNormalizerState;
using testing::TestLimitedOutputCapacity;
using testing::TestWithAllChunkSizes;
using testing::TestZeroCapacityOutput;

//===----------------------------------------------------------------------===//
// Test fixture for passthrough normalizer tests.
//===----------------------------------------------------------------------===//

class PassthroughNormalizerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_tokenizer_normalizer_t* raw_normalizer = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_normalizer_passthrough_allocate(
        iree_allocator_system(), &raw_normalizer));
    normalizer_ = ScopedNormalizer(raw_normalizer);
  }

  iree_tokenizer_normalizer_t* normalizer() { return normalizer_.get(); }

 private:
  ScopedNormalizer normalizer_;
};

//===----------------------------------------------------------------------===//
// Lifecycle
//===----------------------------------------------------------------------===//

TEST_F(PassthroughNormalizerTest, CreateAndDestroy) {
  // SetUp/TearDown handles lifecycle. Verify normalizer was created.
  EXPECT_NE(normalizer(), nullptr);
}

TEST_F(PassthroughNormalizerTest, StateSizeIsReasonable) {
  iree_host_size_t state_size =
      iree_tokenizer_normalizer_state_size(normalizer());
  EXPECT_GT(state_size, 0u);
  // Passthrough state should be minimal (just the base struct).
  EXPECT_LE(state_size, 64u);
}

//===----------------------------------------------------------------------===//
// No-ops
//===----------------------------------------------------------------------===//

TEST_F(PassthroughNormalizerTest, EmptyInput) {
  std::string result = ProcessAndFinalize(
      normalizer(), "", /*expect_pending_after_process=*/false);
  EXPECT_TRUE(result.empty());
}

TEST_F(PassthroughNormalizerTest, ZeroCapacityOutput) {
  TestZeroCapacityOutput(normalizer(), "Hello, World!");
}

//===----------------------------------------------------------------------===//
// Edge Cases
//===----------------------------------------------------------------------===//

TEST_F(PassthroughNormalizerTest, SingleChar) {
  TestWithAllChunkSizes(normalizer(), "a", "a",
                        /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// Basic Functionality
//===----------------------------------------------------------------------===//

TEST_F(PassthroughNormalizerTest, ProcessCopiesInputToOutput) {
  TestWithAllChunkSizes(normalizer(), "Hello, World!", "Hello, World!",
                        /*expect_pending_after_process=*/false);
}

TEST_F(PassthroughNormalizerTest, UnicodeContentPassedThrough) {
  // Passthrough should handle UTF-8 content without modification.
  TestWithAllChunkSizes(normalizer(), "こんにちは世界", "こんにちは世界",
                        /*expect_pending_after_process=*/false);
}

TEST_F(PassthroughNormalizerTest, MixedContent) {
  // Mixed ASCII, UTF-8, and special characters.
  TestWithAllChunkSizes(normalizer(), "Hello 世界! 🌍", "Hello 世界! 🌍",
                        /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// Streaming Behavior
//===----------------------------------------------------------------------===//

TEST_F(PassthroughNormalizerTest, HasPendingAlwaysFalse) {
  ScopedNormalizerState state(normalizer());
  EXPECT_FALSE(iree_tokenizer_normalizer_state_has_pending(state.get()));

  // Even after processing.
  char output[64];
  iree_host_size_t consumed = 0;
  iree_host_size_t written = 0;
  IREE_ASSERT_OK(iree_tokenizer_normalizer_state_process(
      state.get(), iree_make_cstring_view("Hello"),
      iree_make_mutable_string_view(output, sizeof(output)),
      IREE_TOKENIZER_NORMALIZER_FLAG_NONE, &consumed, &written));

  EXPECT_FALSE(iree_tokenizer_normalizer_state_has_pending(state.get()));
}

TEST_F(PassthroughNormalizerTest, FinalizeProducesNothing) {
  ScopedNormalizerState state(normalizer());

  char output[64];
  iree_host_size_t written = 0;
  IREE_ASSERT_OK(iree_tokenizer_normalizer_state_finalize(
      state.get(), iree_make_mutable_string_view(output, sizeof(output)),
      &written));

  // Passthrough never buffers, so finalize produces nothing.
  EXPECT_EQ(written, 0u);
}

//===----------------------------------------------------------------------===//
// Output Buffer Handling
//===----------------------------------------------------------------------===//

TEST_F(PassthroughNormalizerTest, LimitedOutputCapacity) {
  TestLimitedOutputCapacity(normalizer(), "Hello, World!", "Hello, World!");
}

TEST_F(PassthroughNormalizerTest, ProcessRespectsOutputCapacity) {
  ScopedNormalizerState state(normalizer());

  const char* input = "Hello, World!";
  char output[5];  // Smaller than input.
  iree_host_size_t consumed = 0;
  iree_host_size_t written = 0;

  IREE_ASSERT_OK(iree_tokenizer_normalizer_state_process(
      state.get(), iree_make_cstring_view(input),
      iree_make_mutable_string_view(output, 5),
      IREE_TOKENIZER_NORMALIZER_FLAG_NONE, &consumed, &written));

  // Should only consume/write what fits.
  EXPECT_EQ(consumed, 5u);
  EXPECT_EQ(written, 5u);
  EXPECT_EQ(std::string(output, 5), "Hello");
}

}  // namespace
}  // namespace iree::tokenizer
