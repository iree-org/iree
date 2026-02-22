// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/normalizer/nfd.h"

#include <string>

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

//===----------------------------------------------------------------------===//
// Test fixture for NFD normalizer tests.
//===----------------------------------------------------------------------===//

class NFDNormalizerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_tokenizer_normalizer_t* raw_normalizer = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_normalizer_nfd_allocate(
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

TEST_F(NFDNormalizerTest, CreateAndDestroy) {
  EXPECT_NE(normalizer(), nullptr);
}

TEST_F(NFDNormalizerTest, StateSizeIsReasonable) {
  iree_host_size_t state_size =
      iree_tokenizer_normalizer_state_size(normalizer());
  EXPECT_GT(state_size, 0u);
  // State should be bounded (combining sequence buffer + emit tracking).
  EXPECT_LE(state_size, 256u);
}

//===----------------------------------------------------------------------===//
// No-ops
//===----------------------------------------------------------------------===//

TEST_F(NFDNormalizerTest, EmptyInput) {
  std::string result = ProcessAndFinalize(
      normalizer(), "", /*expect_pending_after_process=*/false);
  EXPECT_TRUE(result.empty());
}

TEST_F(NFDNormalizerTest, AsciiOnly) {
  // Pure ASCII passes through unchanged.
  // The last ASCII byte is buffered (potential combining base), so
  // has_pending is true after process.
  TestWithAllChunkSizes(normalizer(), "HELLO WORLD 123!", "HELLO WORLD 123!",
                        /*expect_pending_after_process=*/true);
}

TEST_F(NFDNormalizerTest, LongAsciiRun) {
  std::string input(500, 'A');
  TestWithAllChunkSizes(normalizer(), input, input,
                        /*expect_pending_after_process=*/true);
}

//===----------------------------------------------------------------------===//
// Precomposed Characters Are Decomposed
//===----------------------------------------------------------------------===//

TEST_F(NFDNormalizerTest, PrecomposedDecomposed) {
  // é (U+00E9) -> e (U+0065) + combining acute (U+0301)
  // UTF-8: C3 A9 -> 65 CC 81
  TestWithAllChunkSizes(normalizer(), "\xC3\xA9", "e\xCC\x81",
                        /*expect_pending_after_process=*/true);
}

TEST_F(NFDNormalizerTest, CafeDecomposed) {
  // "café" with precomposed é -> "cafe" + combining acute after e
  TestWithAllChunkSizes(normalizer(), "caf\xC3\xA9", "cafe\xCC\x81",
                        /*expect_pending_after_process=*/true);
}

TEST_F(NFDNormalizerTest, MultiplePrecomposed) {
  // "naïve" with ï (U+00EF) -> i + combining diaeresis (U+0308)
  // ï = C3 AF -> 69 CC 88
  TestWithAllChunkSizes(normalizer(), "na\xC3\xAFve", "nai\xCC\x88ve",
                        /*expect_pending_after_process=*/true);
}

TEST_F(NFDNormalizerTest, AccentedUppercase) {
  // É (U+00C9) -> E (U+0045) + combining acute (U+0301)
  TestWithAllChunkSizes(normalizer(), "\xC3\x89", "E\xCC\x81",
                        /*expect_pending_after_process=*/true);
}

//===----------------------------------------------------------------------===//
// Already-decomposed Text Passes Through With Ordering
//===----------------------------------------------------------------------===//

TEST_F(NFDNormalizerTest, AlreadyDecomposedPassthrough) {
  // e + combining acute is already NFD.
  TestWithAllChunkSizes(normalizer(), "e\xCC\x81", "e\xCC\x81",
                        /*expect_pending_after_process=*/true);
}

TEST_F(NFDNormalizerTest, MultipleCombiningMarks) {
  // a + combining acute (U+0301) + combining tilde (U+0303)
  // Should be reordered by CCC if needed. Both have CCC > 0.
  // Acute (CCC=230), tilde (CCC=230) - same CCC, order preserved.
  TestWithAllChunkSizes(normalizer(), "a\xCC\x81\xCC\x83", "a\xCC\x81\xCC\x83",
                        /*expect_pending_after_process=*/true);
}

//===----------------------------------------------------------------------===//
// Hangul Decomposition
//===----------------------------------------------------------------------===//

TEST_F(NFDNormalizerTest, HangulSyllableDecomposed) {
  // 뮨 (U+BBA8) -> ᄆ (U+1106) + ᅲ (U+1172) + ᆫ (U+11AB)
  // UTF-8: EB AE A8 -> E1 84 86 E1 85 B2 E1 86 AB
  TestWithAllChunkSizes(normalizer(), "\xEB\xAE\xA8",
                        "\xE1\x84\x86\xE1\x85\xB2\xE1\x86\xAB",
                        /*expect_pending_after_process=*/true);
}

TEST_F(NFDNormalizerTest, HangulLVSyllable) {
  // 가 (U+AC00) -> ᄀ (U+1100) + ᅡ (U+1161)
  // UTF-8: EA B0 80 -> E1 84 80 E1 85 A1
  TestWithAllChunkSizes(normalizer(), "\xEA\xB0\x80",
                        "\xE1\x84\x80\xE1\x85\xA1",
                        /*expect_pending_after_process=*/true);
}

//===----------------------------------------------------------------------===//
// Streaming With Chunk Boundaries
//===----------------------------------------------------------------------===//

TEST_F(NFDNormalizerTest, ChunkBoundaryInPrecomposed) {
  // "café" split at various points should produce same result.
  std::string input = "caf\xC3\xA9";
  std::string expected = "cafe\xCC\x81";
  TestWithAllChunkSizes(normalizer(), input, expected,
                        /*expect_pending_after_process=*/true);
}

TEST_F(NFDNormalizerTest, ChunkBoundaryInCombiningSequence) {
  // a + multiple combining marks, chunked.
  std::string input = "a\xCC\x81\xCC\x83";  // a + acute + tilde
  std::string expected = "a\xCC\x81\xCC\x83";
  TestWithAllChunkSizes(normalizer(), input, expected,
                        /*expect_pending_after_process=*/true);
}

//===----------------------------------------------------------------------===//
// Limited Output Capacity
//===----------------------------------------------------------------------===//

TEST_F(NFDNormalizerTest, LimitedOutputCapacity) {
  // Verify correct behavior when output buffer is smaller than input.
  TestLimitedOutputCapacity(normalizer(), "caf\xC3\xA9", "cafe\xCC\x81");
}

TEST_F(NFDNormalizerTest, LimitedOutputWithHangul) {
  // Hangul expands from 3 bytes to 9 bytes.
  TestLimitedOutputCapacity(normalizer(), "\xEB\xAE\xA8",
                            "\xE1\x84\x86\xE1\x85\xB2\xE1\x86\xAB");
}

//===----------------------------------------------------------------------===//
// Defective Combining Sequences
//===----------------------------------------------------------------------===//

TEST_F(NFDNormalizerTest, DefectiveCombiningSequence) {
  // Combining mark without preceding base character.
  // Should pass through (defective but valid).
  TestWithAllChunkSizes(normalizer(), "\xCC\x81", "\xCC\x81",
                        /*expect_pending_after_process=*/true);
}

//===----------------------------------------------------------------------===//
// NFD vs NFC
//===----------------------------------------------------------------------===//

TEST_F(NFDNormalizerTest, DifferentFromNFC) {
  // Key difference: NFC would keep é as U+00E9 (precomposed).
  // NFD decomposes it to e + combining acute.
  //
  // Input: é (U+00E9 = C3 A9)
  // NFD output: e + U+0301 (65 CC 81)
  // NFC output: é (C3 A9) - unchanged
  std::string result = ProcessAndFinalize(
      normalizer(), "\xC3\xA9", /*expect_pending_after_process=*/true);
  EXPECT_EQ(result, "e\xCC\x81");
  // Verify it's different from the input (which is what NFC would produce).
  EXPECT_NE(result, "\xC3\xA9");
}

}  // namespace
}  // namespace iree::tokenizer
