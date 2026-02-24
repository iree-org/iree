// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/segmenter/digits.h"

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/tokenizer/segmenter/segmenter_test_util.h"

namespace iree::tokenizer {
namespace {

using testing::ScopedSegmenter;
using testing::ScopedSegmenterState;

//===----------------------------------------------------------------------===//
// Test fixture for individual_digits=true
//===----------------------------------------------------------------------===//

class DigitsIndividualTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_tokenizer_segmenter_t* raw_segmenter = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_segmenter_digits_allocate(
        /*individual_digits=*/true, iree_allocator_system(), &raw_segmenter));
    segmenter_ = ScopedSegmenter(raw_segmenter);
  }

  iree_tokenizer_segmenter_t* segmenter() { return segmenter_.get(); }

 private:
  ScopedSegmenter segmenter_;
};

//===----------------------------------------------------------------------===//
// Test fixture for individual_digits=false
//===----------------------------------------------------------------------===//

class DigitsGroupedTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_tokenizer_segmenter_t* raw_segmenter = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_segmenter_digits_allocate(
        /*individual_digits=*/false, iree_allocator_system(), &raw_segmenter));
    segmenter_ = ScopedSegmenter(raw_segmenter);
  }

  iree_tokenizer_segmenter_t* segmenter() { return segmenter_.get(); }

 private:
  ScopedSegmenter segmenter_;
};

//===----------------------------------------------------------------------===//
// Lifecycle
//===----------------------------------------------------------------------===//

TEST_F(DigitsIndividualTest, CreateAndDestroy) {
  EXPECT_NE(segmenter(), nullptr);
}

TEST_F(DigitsGroupedTest, CreateAndDestroy) { EXPECT_NE(segmenter(), nullptr); }

TEST_F(DigitsIndividualTest, StateSizeIsReasonable) {
  iree_host_size_t state_size =
      iree_tokenizer_segmenter_state_size(segmenter());
  EXPECT_GT(state_size, 0u);
  EXPECT_LE(state_size, 128u);
}

//===----------------------------------------------------------------------===//
// No-ops and Edge Cases
//===----------------------------------------------------------------------===//

TEST_F(DigitsIndividualTest, EmptyInput) {
  testing::TestWithAllChunkSizes(segmenter(), iree_make_cstring_view(""),
                                 /*expected_segments=*/{},
                                 /*expect_pending_after_process=*/false);
}

TEST_F(DigitsIndividualTest, ZeroCapacityOutput) {
  testing::TestZeroCapacityOutput(segmenter(),
                                  iree_make_cstring_view("abc123"));
}

TEST_F(DigitsIndividualTest, SingleNonDigit) {
  testing::TestWithAllChunkSizes(segmenter(), iree_make_cstring_view("a"),
                                 /*expected_segments=*/{"a"},
                                 /*expect_pending_after_process=*/true);
}

TEST_F(DigitsIndividualTest, SingleDigit) {
  testing::TestWithAllChunkSizes(segmenter(), iree_make_cstring_view("5"),
                                 /*expected_segments=*/{"5"},
                                 /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// Individual digits mode: each digit is separate
//===----------------------------------------------------------------------===//

TEST_F(DigitsIndividualTest, OnlyDigits) {
  testing::TestWithAllChunkSizes(segmenter(), iree_make_cstring_view("123"),
                                 /*expected_segments=*/{"1", "2", "3"},
                                 /*expect_pending_after_process=*/false);
}

TEST_F(DigitsIndividualTest, OnlyNonDigits) {
  testing::TestWithAllChunkSizes(segmenter(), iree_make_cstring_view("abc"),
                                 /*expected_segments=*/{"abc"},
                                 /*expect_pending_after_process=*/true);
}

TEST_F(DigitsIndividualTest, MixedDigitsAndText) {
  // "abc123def" -> ["abc", "1", "2", "3", "def"]
  testing::TestWithAllChunkSizes(
      segmenter(), iree_make_cstring_view("abc123def"),
      /*expected_segments=*/{"abc", "1", "2", "3", "def"},
      /*expect_pending_after_process=*/true);
}

TEST_F(DigitsIndividualTest, DigitsAtStart) {
  // "123abc" -> ["1", "2", "3", "abc"]
  testing::TestWithAllChunkSizes(segmenter(), iree_make_cstring_view("123abc"),
                                 /*expected_segments=*/{"1", "2", "3", "abc"},
                                 /*expect_pending_after_process=*/true);
}

TEST_F(DigitsIndividualTest, DigitsAtEnd) {
  // "abc123" -> ["abc", "1", "2", "3"]
  testing::TestWithAllChunkSizes(segmenter(), iree_make_cstring_view("abc123"),
                                 /*expected_segments=*/{"abc", "1", "2", "3"},
                                 /*expect_pending_after_process=*/false);
}

TEST_F(DigitsIndividualTest, AlternatingDigitsAndText) {
  // "a1b2c3" -> ["a", "1", "b", "2", "c", "3"]
  testing::TestWithAllChunkSizes(
      segmenter(), iree_make_cstring_view("a1b2c3"),
      /*expected_segments=*/{"a", "1", "b", "2", "c", "3"},
      /*expect_pending_after_process=*/false);
}

TEST_F(DigitsIndividualTest, MultipleDigitGroups) {
  // "abc12def34ghi" -> ["abc", "1", "2", "def", "3", "4", "ghi"]
  testing::TestWithAllChunkSizes(
      segmenter(), iree_make_cstring_view("abc12def34ghi"),
      /*expected_segments=*/{"abc", "1", "2", "def", "3", "4", "ghi"},
      /*expect_pending_after_process=*/true);
}

//===----------------------------------------------------------------------===//
// Grouped digits mode: contiguous digits stay together
//===----------------------------------------------------------------------===//

TEST_F(DigitsGroupedTest, OnlyDigits) {
  testing::TestWithAllChunkSizes(segmenter(), iree_make_cstring_view("123"),
                                 /*expected_segments=*/{"123"},
                                 /*expect_pending_after_process=*/true);
}

TEST_F(DigitsGroupedTest, OnlyNonDigits) {
  testing::TestWithAllChunkSizes(segmenter(), iree_make_cstring_view("abc"),
                                 /*expected_segments=*/{"abc"},
                                 /*expect_pending_after_process=*/true);
}

TEST_F(DigitsGroupedTest, MixedDigitsAndText) {
  // "abc123def" -> ["abc", "123", "def"]
  testing::TestWithAllChunkSizes(segmenter(),
                                 iree_make_cstring_view("abc123def"),
                                 /*expected_segments=*/{"abc", "123", "def"},
                                 /*expect_pending_after_process=*/true);
}

TEST_F(DigitsGroupedTest, DigitsAtStart) {
  // "123abc" -> ["123", "abc"]
  testing::TestWithAllChunkSizes(segmenter(), iree_make_cstring_view("123abc"),
                                 /*expected_segments=*/{"123", "abc"},
                                 /*expect_pending_after_process=*/true);
}

TEST_F(DigitsGroupedTest, DigitsAtEnd) {
  // "abc123" -> ["abc", "123"]
  testing::TestWithAllChunkSizes(segmenter(), iree_make_cstring_view("abc123"),
                                 /*expected_segments=*/{"abc", "123"},
                                 /*expect_pending_after_process=*/true);
}

TEST_F(DigitsGroupedTest, AlternatingDigitsAndText) {
  // "a1b2c3" -> ["a", "1", "b", "2", "c", "3"]
  testing::TestWithAllChunkSizes(
      segmenter(), iree_make_cstring_view("a1b2c3"),
      /*expected_segments=*/{"a", "1", "b", "2", "c", "3"},
      /*expect_pending_after_process=*/true);
}

TEST_F(DigitsGroupedTest, MultipleDigitGroups) {
  // "abc12def34ghi" -> ["abc", "12", "def", "34", "ghi"]
  testing::TestWithAllChunkSizes(
      segmenter(), iree_make_cstring_view("abc12def34ghi"),
      /*expected_segments=*/{"abc", "12", "def", "34", "ghi"},
      /*expect_pending_after_process=*/true);
}

TEST_F(DigitsGroupedTest, LongDigitSequence) {
  // "test1234567890end" -> ["test", "1234567890", "end"]
  testing::TestWithAllChunkSizes(
      segmenter(), iree_make_cstring_view("test1234567890end"),
      /*expected_segments=*/{"test", "1234567890", "end"},
      /*expect_pending_after_process=*/true);
}

//===----------------------------------------------------------------------===//
// Special characters (not digits, not letters)
//===----------------------------------------------------------------------===//

TEST_F(DigitsIndividualTest, WithPunctuation) {
  // "abc!123" -> ["abc!", "1", "2", "3"] (punctuation grouped with non-digits)
  testing::TestWithAllChunkSizes(segmenter(), iree_make_cstring_view("abc!123"),
                                 /*expected_segments=*/{"abc!", "1", "2", "3"},
                                 /*expect_pending_after_process=*/false);
}

TEST_F(DigitsGroupedTest, WithPunctuation) {
  // "abc!123" -> ["abc!", "123"]
  testing::TestWithAllChunkSizes(segmenter(), iree_make_cstring_view("abc!123"),
                                 /*expected_segments=*/{"abc!", "123"},
                                 /*expect_pending_after_process=*/true);
}

TEST_F(DigitsIndividualTest, WithSpaces) {
  // "abc 123" -> ["abc ", "1", "2", "3"] (space grouped with non-digits)
  testing::TestWithAllChunkSizes(segmenter(), iree_make_cstring_view("abc 123"),
                                 /*expected_segments=*/{"abc ", "1", "2", "3"},
                                 /*expect_pending_after_process=*/false);
}

TEST_F(DigitsGroupedTest, WithSpaces) {
  // "abc 123" -> ["abc ", "123"]
  testing::TestWithAllChunkSizes(segmenter(), iree_make_cstring_view("abc 123"),
                                 /*expected_segments=*/{"abc ", "123"},
                                 /*expect_pending_after_process=*/true);
}

//===----------------------------------------------------------------------===//
// Unicode (digits segmenter only handles ASCII 0-9)
//===----------------------------------------------------------------------===//

TEST_F(DigitsIndividualTest, UnicodeText) {
  // Unicode text without ASCII digits passes through as single segment.
  testing::TestWithAllChunkSizes(segmenter(),
                                 iree_make_cstring_view("café résumé"),
                                 /*expected_segments=*/{"café résumé"},
                                 /*expect_pending_after_process=*/true);
}

TEST_F(DigitsIndividualTest, UnicodeWithDigits) {
  // "café123" -> ["café", "1", "2", "3"]
  testing::TestWithAllChunkSizes(segmenter(), iree_make_cstring_view("café123"),
                                 /*expected_segments=*/{"café", "1", "2", "3"},
                                 /*expect_pending_after_process=*/false);
}

}  // namespace
}  // namespace iree::tokenizer
