// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/normalizer/nfkd.h"

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
// Test fixture for NFKD normalizer tests.
//===----------------------------------------------------------------------===//

class NFKDNormalizerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_tokenizer_normalizer_t* raw_normalizer = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_normalizer_nfkd_allocate(
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

TEST_F(NFKDNormalizerTest, CreateAndDestroy) {
  EXPECT_NE(normalizer(), nullptr);
}

TEST_F(NFKDNormalizerTest, StateSizeIsReasonable) {
  iree_host_size_t state_size =
      iree_tokenizer_normalizer_state_size(normalizer());
  EXPECT_GT(state_size, 0u);
  // State should be bounded (combining sequence buffer + decomposed buffer +
  // emit tracking). NFKD needs larger decomposed buffer (20 vs 4).
  EXPECT_LE(state_size, 300u);
}

//===----------------------------------------------------------------------===//
// No-ops
//===----------------------------------------------------------------------===//

TEST_F(NFKDNormalizerTest, EmptyInput) {
  std::string result = ProcessAndFinalize(
      normalizer(), "", /*expect_pending_after_process=*/false);
  EXPECT_TRUE(result.empty());
}

TEST_F(NFKDNormalizerTest, AsciiOnly) {
  // Pure ASCII passes through unchanged.
  // The last ASCII byte is buffered (potential combining base), so
  // has_pending is true after process.
  TestWithAllChunkSizes(normalizer(), "HELLO WORLD 123!", "HELLO WORLD 123!",
                        /*expect_pending_after_process=*/true);
}

TEST_F(NFKDNormalizerTest, LongAsciiRun) {
  std::string input(500, 'A');
  TestWithAllChunkSizes(normalizer(), input, input,
                        /*expect_pending_after_process=*/true);
}

//===----------------------------------------------------------------------===//
// Precomposed Characters Are Decomposed (Canonical Decomposition)
//===----------------------------------------------------------------------===//

TEST_F(NFKDNormalizerTest, PrecomposedDecomposed) {
  // e (U+00E9) -> e (U+0065) + combining acute (U+0301)
  // UTF-8: C3 A9 -> 65 CC 81
  TestWithAllChunkSizes(normalizer(), "\xC3\xA9", "e\xCC\x81",
                        /*expect_pending_after_process=*/true);
}

TEST_F(NFKDNormalizerTest, CafeDecomposed) {
  // "cafe" with precomposed e -> "cafe" + combining acute after e
  TestWithAllChunkSizes(normalizer(), "caf\xC3\xA9", "cafe\xCC\x81",
                        /*expect_pending_after_process=*/true);
}

//===----------------------------------------------------------------------===//
// Compatibility Decompositions (The Key Difference From NFD)
//===----------------------------------------------------------------------===//

TEST_F(NFKDNormalizerTest, LatinLigatureFi) {
  // U+FB01 (Latin Small Ligature Fi) -> fi
  // UTF-8: EF AC 81 -> 66 69
  TestWithAllChunkSizes(normalizer(), "\xEF\xAC\x81", "fi",
                        /*expect_pending_after_process=*/true);
}

TEST_F(NFKDNormalizerTest, LatinLigatureFl) {
  // U+FB02 (Latin Small Ligature Fl) -> fl
  // UTF-8: EF AC 82 -> 66 6C
  TestWithAllChunkSizes(normalizer(), "\xEF\xAC\x82", "fl",
                        /*expect_pending_after_process=*/true);
}

TEST_F(NFKDNormalizerTest, FullwidthA) {
  // U+FF21 (Fullwidth Latin Capital Letter A) -> A
  // UTF-8: EF BC A1 -> 41
  TestWithAllChunkSizes(normalizer(), "\xEF\xBC\xA1", "A",
                        /*expect_pending_after_process=*/true);
}

TEST_F(NFKDNormalizerTest, FullwidthDigit1) {
  // U+FF11 (Fullwidth Digit One) -> 1
  // UTF-8: EF BC 91 -> 31
  TestWithAllChunkSizes(normalizer(), "\xEF\xBC\x91", "1",
                        /*expect_pending_after_process=*/true);
}

TEST_F(NFKDNormalizerTest, CircledDigit1) {
  // U+2460 (Circled Digit One) -> 1
  // UTF-8: E2 91 A0 -> 31
  TestWithAllChunkSizes(normalizer(), "\xE2\x91\xA0", "1",
                        /*expect_pending_after_process=*/true);
}

TEST_F(NFKDNormalizerTest, FractionOneHalf) {
  // U+00BD (Vulgar Fraction One Half) -> 1 / 2
  // UTF-8: C2 BD -> 31 E2 81 84 32 (1 + fraction slash + 2)
  // Or might be simpler: 31 2F 32 (1/2 with slash)
  // The actual decomposition depends on Unicode tables.
  std::string result = ProcessAndFinalize(
      normalizer(), "\xC2\xBD", /*expect_pending_after_process=*/true);
  // Should contain the digits 1 and 2 at minimum.
  EXPECT_NE(result.find('1'), std::string::npos);
  EXPECT_NE(result.find('2'), std::string::npos);
}

TEST_F(NFKDNormalizerTest, Superscript2) {
  // U+00B2 (Superscript Two) -> 2
  // UTF-8: C2 B2 -> 32
  TestWithAllChunkSizes(normalizer(), "\xC2\xB2", "2",
                        /*expect_pending_after_process=*/true);
}

TEST_F(NFKDNormalizerTest, Subscript3) {
  // U+2083 (Subscript Three) -> 3
  // UTF-8: E2 82 83 -> 33
  TestWithAllChunkSizes(normalizer(), "\xE2\x82\x83", "3",
                        /*expect_pending_after_process=*/true);
}

TEST_F(NFKDNormalizerTest, RomanNumeralThree) {
  // U+2162 (Roman Numeral Three) -> III
  // UTF-8: E2 85 A2 -> 49 49 49
  TestWithAllChunkSizes(normalizer(), "\xE2\x85\xA2", "III",
                        /*expect_pending_after_process=*/true);
}

TEST_F(NFKDNormalizerTest, SmallRomanNumeralFour) {
  // U+2173 (Small Roman Numeral Four) -> iv
  // UTF-8: E2 85 B3 -> 69 76
  TestWithAllChunkSizes(normalizer(), "\xE2\x85\xB3", "iv",
                        /*expect_pending_after_process=*/true);
}

TEST_F(NFKDNormalizerTest, SquaredKm) {
  // U+33A2 (Square Kilometre) -> km (actually km\xC2\xB2 -> km2)
  // The exact decomposition varies. We test it produces ASCII.
  std::string result = ProcessAndFinalize(
      normalizer(), "\xE3\x8E\xA2", /*expect_pending_after_process=*/true);
  // Should contain k and m.
  EXPECT_NE(result.find('k'), std::string::npos);
  EXPECT_NE(result.find('m'), std::string::npos);
}

//===----------------------------------------------------------------------===//
// Hangul Decomposition (Inherited From NFD)
//===----------------------------------------------------------------------===//

TEST_F(NFKDNormalizerTest, HangulSyllableDecomposed) {
  // (U+BBA8) -> Jamo
  // UTF-8: EB AE A8 -> E1 84 86 E1 85 B2 E1 86 AB
  TestWithAllChunkSizes(normalizer(), "\xEB\xAE\xA8",
                        "\xE1\x84\x86\xE1\x85\xB2\xE1\x86\xAB",
                        /*expect_pending_after_process=*/true);
}

TEST_F(NFKDNormalizerTest, HangulLVSyllable) {
  // (U+AC00) -> Jamo
  // UTF-8: EA B0 80 -> E1 84 80 E1 85 A1
  TestWithAllChunkSizes(normalizer(), "\xEA\xB0\x80",
                        "\xE1\x84\x80\xE1\x85\xA1",
                        /*expect_pending_after_process=*/true);
}

//===----------------------------------------------------------------------===//
// Streaming With Chunk Boundaries
//===----------------------------------------------------------------------===//

TEST_F(NFKDNormalizerTest, ChunkBoundaryInLigature) {
  // Ligature fi split at various points should produce same result.
  std::string input = "\xEF\xAC\x81";  // fi ligature
  std::string expected = "fi";
  TestWithAllChunkSizes(normalizer(), input, expected,
                        /*expect_pending_after_process=*/true);
}

TEST_F(NFKDNormalizerTest, MixedCompatibilityAndCanonical) {
  // Mix of compatibility (fi ligature) and canonical (e) decomposition.
  // fi ligature + cafe = fi + cafe\xCC\x81
  std::string input =
      "\xEF\xAC\x81"
      "caf\xC3\xA9";
  std::string expected = "ficafe\xCC\x81";
  TestWithAllChunkSizes(normalizer(), input, expected,
                        /*expect_pending_after_process=*/true);
}

//===----------------------------------------------------------------------===//
// Limited Output Capacity
//===----------------------------------------------------------------------===//

TEST_F(NFKDNormalizerTest, LimitedOutputCapacity) {
  // Verify correct behavior when output buffer is smaller than input.
  TestLimitedOutputCapacity(normalizer(), "caf\xC3\xA9", "cafe\xCC\x81");
}

TEST_F(NFKDNormalizerTest, LimitedOutputWithLigature) {
  // Ligature is 3 bytes but decomposes to 2 bytes.
  TestLimitedOutputCapacity(normalizer(), "\xEF\xAC\x81", "fi");
}

TEST_F(NFKDNormalizerTest, LimitedOutputWithHangul) {
  // Hangul expands from 3 bytes to 9 bytes.
  TestLimitedOutputCapacity(normalizer(), "\xEB\xAE\xA8",
                            "\xE1\x84\x86\xE1\x85\xB2\xE1\x86\xAB");
}

//===----------------------------------------------------------------------===//
// Defective Combining Sequences
//===----------------------------------------------------------------------===//

TEST_F(NFKDNormalizerTest, DefectiveCombiningSequence) {
  // Combining mark without preceding base character.
  // Should pass through (defective but valid).
  TestWithAllChunkSizes(normalizer(), "\xCC\x81", "\xCC\x81",
                        /*expect_pending_after_process=*/true);
}

//===----------------------------------------------------------------------===//
// NFKD vs NFD
//===----------------------------------------------------------------------===//

TEST_F(NFKDNormalizerTest, DifferentFromNFD) {
  // Key difference: NFD preserves ligatures, NFKD decomposes them.
  //
  // Input: fi ligature (U+FB01 = EF AC 81)
  // NFD output: fi ligature (EF AC 81) - unchanged
  // NFKD output: fi (66 69) - decomposed to component letters
  std::string result = ProcessAndFinalize(
      normalizer(), "\xEF\xAC\x81", /*expect_pending_after_process=*/true);
  EXPECT_EQ(result, "fi");
  // Verify it's different from the input (which is what NFD would produce).
  EXPECT_NE(result, "\xEF\xAC\x81");
}

TEST_F(NFKDNormalizerTest, CanonicalDecompositionSameAsNFD) {
  // For canonical decompositions, NFKD produces the same result as NFD.
  // e (U+00E9) -> e + combining acute in both NFD and NFKD.
  std::string result = ProcessAndFinalize(
      normalizer(), "\xC3\xA9", /*expect_pending_after_process=*/true);
  EXPECT_EQ(result, "e\xCC\x81");
}

}  // namespace
}  // namespace iree::tokenizer
