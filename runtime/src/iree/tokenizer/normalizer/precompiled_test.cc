// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/normalizer/precompiled.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/tokenizer/normalizer/normalizer_test_util.h"

namespace iree::tokenizer {
namespace {

using testing::ProcessAndFinalize;
using testing::ScopedNormalizer;
using testing::TestLimitedOutputCapacity;
using testing::TestWithAllChunkSizes;

//===----------------------------------------------------------------------===//
// Test Data Helpers
//===----------------------------------------------------------------------===//

// Builds a minimal precompiled charsmap with an empty trie (passthrough).
std::vector<uint8_t> MakeEmptyTrieCharsmap() {
  // Format: [4 bytes: trie_size=0]
  return {0, 0, 0, 0};
}

// Double-array trie builder for testing.
// Implements XOR-based addressing: position ^= offset, position ^= byte_value.
//
// Lookup algorithm (from precompiled.c):
//   pos = 0
//   pos ^= trie[0].offset              // Get root's base offset
//   for each byte c in key:
//     pos ^= c                          // XOR with byte value
//     if pos >= trie_count: no match
//     unit = trie[pos]
//     if unit.label != c: no match
//     pos ^= unit.offset                // XOR for next iteration
//     if unit.has_leaf:
//       leaf = trie[pos]                // Leaf is at current position
//       if leaf.is_leaf: record match
//
// This implementation uses sparse allocation - only positions that are
// actually used get meaningful values. All other positions remain 0, which
// means label=0. The XOR-based addressing ensures we only land on positions
// we deliberately set, because child_pos = base ^ byte_value, and we choose
// base values to avoid collisions.
class TrieBuilder {
 public:
  void AddMapping(const std::string& key, const std::string& replacement) {
    mappings_[key] = replacement;
  }

  std::vector<uint8_t> Build() const {
    if (mappings_.empty()) {
      // Empty trie: just return trie_size=0.
      return {0, 0, 0, 0};
    }

    // Phase 1: Build pool with NUL-terminated strings.
    std::vector<uint8_t> pool;
    std::map<std::string, uint32_t> pool_offsets;
    for (const auto& [key, replacement] : mappings_) {
      pool_offsets[key] = static_cast<uint32_t>(pool.size());
      for (char c : replacement) {
        pool.push_back(static_cast<uint8_t>(c));
      }
      pool.push_back(0);  // NUL terminator.
    }

    // Phase 2: Build trie using sparse allocation.
    // We'll use a map to track allocated positions, then convert to vector.
    std::map<uint32_t, uint32_t> trie_entries;

    // Root at index 0. We need to find a base offset such that all first
    // bytes of keys land at non-colliding positions.
    // Simple strategy: use a high base offset (256) that puts first bytes
    // in range [256, 511], leaves in a separate range [512+).
    const uint32_t ROOT_BASE = 256;
    trie_entries[0] = (ROOT_BASE << 10);  // offset=ROOT_BASE, label=0

    uint32_t next_free = 512;  // Next free slot for leaves and intermediate

    for (const auto& [key, replacement] : mappings_) {
      if (key.empty()) continue;

      uint32_t pos =
          ROOT_BASE;  // Current base position (after XOR with offset)

      for (size_t i = 0; i < key.size(); ++i) {
        uint8_t c = static_cast<uint8_t>(key[i]);
        uint32_t child_pos = pos ^ c;
        bool is_last = (i == key.size() - 1);

        if (is_last) {
          // Last byte: create child node with has_leaf flag and leaf pointer.
          uint32_t leaf_slot = next_free++;
          uint32_t offset_to_leaf = child_pos ^ leaf_slot;

          // Child unit at child_pos: label=c, has_leaf=1, offset=offset_to_leaf
          trie_entries[child_pos] = c | (1u << 8) | (offset_to_leaf << 10);

          // Leaf at leaf_slot: is_leaf=1, value=pool_offset
          trie_entries[leaf_slot] = 0x80000000u | pool_offsets.at(key);
        } else {
          // Intermediate byte: create child node pointing to next level.
          // Allocate a new base for the next level.
          uint32_t next_base = next_free;
          next_free += 256;  // Reserve space for next level's children.

          uint32_t offset_to_next = child_pos ^ next_base;

          // Child unit at child_pos: label=c, has_leaf=0, offset=offset_to_next
          trie_entries[child_pos] = c | (offset_to_next << 10);

          pos = next_base;  // Next iteration starts from new base.
        }
      }
    }

    // Convert sparse map to dense vector.
    uint32_t max_index = 0;
    for (const auto& [idx, _] : trie_entries) {
      if (idx > max_index) max_index = idx;
    }
    std::vector<uint32_t> trie(max_index + 1, 0);
    for (const auto& [idx, value] : trie_entries) {
      trie[idx] = value;
    }

    // Phase 3: Build final charsmap binary.
    std::vector<uint8_t> result;
    uint32_t trie_size = static_cast<uint32_t>(trie.size() * sizeof(uint32_t));
    result.push_back(trie_size & 0xFF);
    result.push_back((trie_size >> 8) & 0xFF);
    result.push_back((trie_size >> 16) & 0xFF);
    result.push_back((trie_size >> 24) & 0xFF);

    for (uint32_t unit : trie) {
      result.push_back(unit & 0xFF);
      result.push_back((unit >> 8) & 0xFF);
      result.push_back((unit >> 16) & 0xFF);
      result.push_back((unit >> 24) & 0xFF);
    }

    for (uint8_t b : pool) {
      result.push_back(b);
    }

    return result;
  }

 private:
  std::map<std::string, std::string> mappings_;
};

// Builds a precompiled charsmap with a single mapping: key -> replacement.
std::vector<uint8_t> MakeSingleMappingCharsmap(const std::string& key,
                                               const std::string& replacement) {
  TrieBuilder builder;
  builder.AddMapping(key, replacement);
  return builder.Build();
}

iree_const_byte_span_t ToSpan(const std::vector<uint8_t>& data) {
  return iree_make_const_byte_span(data.data(), data.size());
}

//===----------------------------------------------------------------------===//
// Test fixture for precompiled normalizer tests.
//===----------------------------------------------------------------------===//

class PrecompiledNormalizerTest : public ::testing::Test {
 protected:
  void CreateNormalizer(const std::vector<uint8_t>& charsmap) {
    iree_tokenizer_normalizer_t* raw_normalizer = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_precompiled_normalizer_allocate(
        ToSpan(charsmap), iree_allocator_system(), &raw_normalizer));
    normalizer_ = ScopedNormalizer(raw_normalizer);
  }

  iree_tokenizer_normalizer_t* normalizer() { return normalizer_.get(); }

 private:
  ScopedNormalizer normalizer_;
};

//===----------------------------------------------------------------------===//
// Lifecycle
//===----------------------------------------------------------------------===//

TEST_F(PrecompiledNormalizerTest, CreateWithEmptyTrie) {
  auto charsmap = MakeEmptyTrieCharsmap();
  CreateNormalizer(charsmap);
  EXPECT_NE(normalizer(), nullptr);
}

TEST_F(PrecompiledNormalizerTest, StateSizeIsReasonable) {
  auto charsmap = MakeEmptyTrieCharsmap();
  CreateNormalizer(charsmap);
  iree_host_size_t state_size =
      iree_tokenizer_normalizer_state_size(normalizer());
  EXPECT_GT(state_size, 0u);
  // State should include overlap buffer (32) + pending buffer (64) + overhead.
  EXPECT_LE(state_size, 256u);
}

//===----------------------------------------------------------------------===//
// Format Validation
//===----------------------------------------------------------------------===//

TEST(PrecompiledNormalizerValidationTest, RejectsTooShort) {
  std::vector<uint8_t> data = {0, 0, 0};  // Only 3 bytes, need at least 4.
  iree_tokenizer_normalizer_t* normalizer = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_tokenizer_precompiled_normalizer_allocate(
          ToSpan(data), iree_allocator_system(), &normalizer));
}

TEST(PrecompiledNormalizerValidationTest, RejectsUnalignedTrieSize) {
  // Trie size = 5 (not multiple of 4)
  std::vector<uint8_t> data = {5, 0, 0, 0, 0, 0, 0, 0, 0};
  iree_tokenizer_normalizer_t* normalizer = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_tokenizer_precompiled_normalizer_allocate(
          ToSpan(data), iree_allocator_system(), &normalizer));
}

TEST(PrecompiledNormalizerValidationTest, RejectsTrieSizeExceedingData) {
  // Trie size = 1000 but only have 8 bytes total
  std::vector<uint8_t> data = {0xE8, 0x03, 0, 0, 0, 0, 0, 0};
  iree_tokenizer_normalizer_t* normalizer = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_tokenizer_precompiled_normalizer_allocate(
          ToSpan(data), iree_allocator_system(), &normalizer));
}

//===----------------------------------------------------------------------===//
// Empty Trie Passthrough
//===----------------------------------------------------------------------===//

TEST_F(PrecompiledNormalizerTest, EmptyTriePassthroughAscii) {
  auto charsmap = MakeEmptyTrieCharsmap();
  CreateNormalizer(charsmap);
  // Empty trie passes through directly without using overlap buffer.
  TestWithAllChunkSizes(normalizer(), "Hello World!", "Hello World!",
                        /*expect_pending_after_process=*/false);
}

TEST_F(PrecompiledNormalizerTest, EmptyTriePassthroughUnicode) {
  auto charsmap = MakeEmptyTrieCharsmap();
  CreateNormalizer(charsmap);
  // Japanese text should pass through unchanged (empty trie = no buffering).
  TestWithAllChunkSizes(normalizer(), "こんにちは", "こんにちは",
                        /*expect_pending_after_process=*/false);
}

TEST_F(PrecompiledNormalizerTest, EmptyTriePassthroughEmpty) {
  auto charsmap = MakeEmptyTrieCharsmap();
  CreateNormalizer(charsmap);
  std::string result = ProcessAndFinalize(
      normalizer(), "", /*expect_pending_after_process=*/false);
  EXPECT_TRUE(result.empty());
}

//===----------------------------------------------------------------------===//
// Streaming Tests
//===----------------------------------------------------------------------===//

TEST_F(PrecompiledNormalizerTest, StreamingChunkBoundary) {
  auto charsmap = MakeEmptyTrieCharsmap();
  CreateNormalizer(charsmap);
  // Test that chunk boundaries don't affect output.
  // With empty trie, input passes through directly without buffering.
  std::string input = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
  TestWithAllChunkSizes(normalizer(), input, input,
                        /*expect_pending_after_process=*/false);
}

TEST_F(PrecompiledNormalizerTest, StreamingLimitedOutput) {
  auto charsmap = MakeEmptyTrieCharsmap();
  CreateNormalizer(charsmap);
  // Test with very small output buffer (capacity=1).
  TestLimitedOutputCapacity(normalizer(), "Hello World!", "Hello World!");
}

//===----------------------------------------------------------------------===//
// Long Input Test
//===----------------------------------------------------------------------===//

TEST_F(PrecompiledNormalizerTest, LongInputPassthrough) {
  auto charsmap = MakeEmptyTrieCharsmap();
  CreateNormalizer(charsmap);
  // With empty trie, input passes through directly without buffering.
  std::string input(1000, 'X');
  TestWithAllChunkSizes(normalizer(), input, input,
                        /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// Single Mapping Tests
//===----------------------------------------------------------------------===//

TEST_F(PrecompiledNormalizerTest, SingleAsciiMapping) {
  // Map "X" -> "Y".
  auto charsmap = MakeSingleMappingCharsmap("X", "Y");
  CreateNormalizer(charsmap);
  TestWithAllChunkSizes(normalizer(), "X", "Y",
                        /*expect_pending_after_process=*/true);
}

TEST_F(PrecompiledNormalizerTest, SingleMappingWithSurroundingText) {
  // Map "X" -> "Y".
  auto charsmap = MakeSingleMappingCharsmap("X", "Y");
  CreateNormalizer(charsmap);
  TestWithAllChunkSizes(normalizer(), "aXb", "aYb",
                        /*expect_pending_after_process=*/true);
}

TEST_F(PrecompiledNormalizerTest, SingleMappingMultipleOccurrences) {
  // Map "X" -> "Y".
  auto charsmap = MakeSingleMappingCharsmap("X", "Y");
  CreateNormalizer(charsmap);
  TestWithAllChunkSizes(normalizer(), "XXXXX", "YYYYY",
                        /*expect_pending_after_process=*/true);
}

TEST_F(PrecompiledNormalizerTest, SingleMappingNoMatch) {
  // Map "X" -> "Y" but input has no X.
  auto charsmap = MakeSingleMappingCharsmap("X", "Y");
  CreateNormalizer(charsmap);
  TestWithAllChunkSizes(normalizer(), "ABCDEF", "ABCDEF",
                        /*expect_pending_after_process=*/true);
}

TEST_F(PrecompiledNormalizerTest, MappingExpansion) {
  // Map single byte to multiple bytes.
  auto charsmap = MakeSingleMappingCharsmap("X", "XXX");
  CreateNormalizer(charsmap);
  TestWithAllChunkSizes(normalizer(), "aXb", "aXXXb",
                        /*expect_pending_after_process=*/true);
}

TEST_F(PrecompiledNormalizerTest, MappingContraction) {
  // Map multiple bytes to single byte.
  auto charsmap = MakeSingleMappingCharsmap("XX", "X");
  CreateNormalizer(charsmap);
  TestWithAllChunkSizes(normalizer(), "aXXb", "aXb",
                        /*expect_pending_after_process=*/true);
}

TEST_F(PrecompiledNormalizerTest, MappingDeletion) {
  // Map to empty string.
  auto charsmap = MakeSingleMappingCharsmap("X", "");
  CreateNormalizer(charsmap);
  TestWithAllChunkSizes(normalizer(), "aXXXb", "ab",
                        /*expect_pending_after_process=*/true);
}

//===----------------------------------------------------------------------===//
// Multiple Mapping Tests
//===----------------------------------------------------------------------===//

TEST_F(PrecompiledNormalizerTest, MultipleMappings) {
  TrieBuilder builder;
  builder.AddMapping("A", "1");
  builder.AddMapping("B", "2");
  builder.AddMapping("C", "3");
  auto charsmap = builder.Build();
  CreateNormalizer(charsmap);
  TestWithAllChunkSizes(normalizer(), "ABC", "123",
                        /*expect_pending_after_process=*/true);
}

TEST_F(PrecompiledNormalizerTest, MultipleMappingsMixed) {
  TrieBuilder builder;
  builder.AddMapping("A", "1");
  builder.AddMapping("B", "2");
  auto charsmap = builder.Build();
  CreateNormalizer(charsmap);
  // Mix of mapped and unmapped characters.
  TestWithAllChunkSizes(normalizer(), "xAyBz", "x1y2z",
                        /*expect_pending_after_process=*/true);
}

//===----------------------------------------------------------------------===//
// Unicode Tests
//===----------------------------------------------------------------------===//

TEST_F(PrecompiledNormalizerTest, UnicodeMapping) {
  // Map ligature "ﬁ" (U+FB01) to "fi".
  // U+FB01 = 0xEF 0xAC 0x81 in UTF-8.
  auto charsmap = MakeSingleMappingCharsmap("\xEF\xAC\x81", "fi");
  CreateNormalizer(charsmap);
  TestWithAllChunkSizes(normalizer(),
                        "a\xEF\xAC\x81"
                        "b",
                        "afib",
                        /*expect_pending_after_process=*/true);
}

TEST_F(PrecompiledNormalizerTest, UnicodePassthrough) {
  // Map X -> Y but leave Unicode unchanged.
  auto charsmap = MakeSingleMappingCharsmap("X", "Y");
  CreateNormalizer(charsmap);
  TestWithAllChunkSizes(normalizer(), "X日本語X", "Y日本語Y",
                        /*expect_pending_after_process=*/true);
}

TEST_F(PrecompiledNormalizerTest, MultibyteKeyMultibyteReplacement) {
  // Map Japanese "あ" to "ア" (hiragana to katakana).
  auto charsmap = MakeSingleMappingCharsmap("\xE3\x81\x82", "\xE3\x82\xA2");
  CreateNormalizer(charsmap);
  TestWithAllChunkSizes(normalizer(), "\xE3\x81\x82", "\xE3\x82\xA2",
                        /*expect_pending_after_process=*/true);
}

//===----------------------------------------------------------------------===//
// Grapheme-First Matching Tests
//===----------------------------------------------------------------------===//

TEST_F(PrecompiledNormalizerTest, ShortGraphemeMatchedAsWhole) {
  // Short grapheme (< 6 bytes) should be tried as whole first.
  // Map "ab" -> "X".
  auto charsmap = MakeSingleMappingCharsmap("ab", "X");
  CreateNormalizer(charsmap);
  // "ab" is 2 bytes < 6, so grapheme-first matching should find it.
  TestWithAllChunkSizes(normalizer(), "ab", "X",
                        /*expect_pending_after_process=*/true);
}

TEST_F(PrecompiledNormalizerTest, GraphemeThresholdBehavior) {
  // Map "abc" (3 bytes < 6 threshold).
  auto charsmap = MakeSingleMappingCharsmap("abc", "X");
  CreateNormalizer(charsmap);
  TestWithAllChunkSizes(normalizer(), "abcdef", "Xdef",
                        /*expect_pending_after_process=*/true);
}

//===----------------------------------------------------------------------===//
// Streaming Chunk Boundary Tests
//===----------------------------------------------------------------------===//

TEST_F(PrecompiledNormalizerTest, ChunkBoundarySplitsMapping) {
  // Two-byte key that might be split across chunks.
  auto charsmap = MakeSingleMappingCharsmap("XY", "Z");
  CreateNormalizer(charsmap);
  // TestWithAllChunkSizes will test chunk size 1 which splits "XY".
  TestWithAllChunkSizes(normalizer(), "aXYb", "aZb",
                        /*expect_pending_after_process=*/true);
}

TEST_F(PrecompiledNormalizerTest, RepeatedPatternAcrossChunks) {
  auto charsmap = MakeSingleMappingCharsmap("AB", "X");
  CreateNormalizer(charsmap);
  TestWithAllChunkSizes(normalizer(), "ABABABABAB", "XXXXX",
                        /*expect_pending_after_process=*/true);
}

TEST_F(PrecompiledNormalizerTest, PartialMatchAtChunkEnd) {
  // Pattern is "ABC" but we only have "AB" at chunk end.
  auto charsmap = MakeSingleMappingCharsmap("ABC", "X");
  CreateNormalizer(charsmap);
  // "ABCABC" should become "XX".
  TestWithAllChunkSizes(normalizer(), "ABCABC", "XX",
                        /*expect_pending_after_process=*/true);
}

//===----------------------------------------------------------------------===//
// Pending Buffer Tests
//===----------------------------------------------------------------------===//

TEST_F(PrecompiledNormalizerTest, LargeReplacementWithSmallOutput) {
  // Map single byte to long replacement.
  auto charsmap = MakeSingleMappingCharsmap("X", "REPLACEMENTSTRING");
  CreateNormalizer(charsmap);
  TestLimitedOutputCapacity(normalizer(), "X", "REPLACEMENTSTRING");
}

TEST_F(PrecompiledNormalizerTest, MultipleExpansionsWithSmallOutput) {
  // Map single bytes to longer replacements.
  TrieBuilder builder;
  builder.AddMapping("A", "AAA");
  builder.AddMapping("B", "BBB");
  auto charsmap = builder.Build();
  CreateNormalizer(charsmap);
  TestLimitedOutputCapacity(normalizer(), "AB", "AAABBB");
}

//===----------------------------------------------------------------------===//
// Edge Cases
//===----------------------------------------------------------------------===//

TEST_F(PrecompiledNormalizerTest, SingleByteInput) {
  auto charsmap = MakeSingleMappingCharsmap("A", "B");
  CreateNormalizer(charsmap);
  std::string result = ProcessAndFinalize(
      normalizer(), "A", /*expect_pending_after_process=*/true);
  EXPECT_EQ(result, "B");
}

TEST_F(PrecompiledNormalizerTest, SingleByteNoMatch) {
  auto charsmap = MakeSingleMappingCharsmap("A", "B");
  CreateNormalizer(charsmap);
  std::string result = ProcessAndFinalize(
      normalizer(), "X", /*expect_pending_after_process=*/true);
  EXPECT_EQ(result, "X");
}

TEST_F(PrecompiledNormalizerTest, OnlyMappedChars) {
  auto charsmap = MakeSingleMappingCharsmap("X", "Y");
  CreateNormalizer(charsmap);
  TestWithAllChunkSizes(normalizer(), "XXXXXXXXXX", "YYYYYYYYYY",
                        /*expect_pending_after_process=*/true);
}

TEST_F(PrecompiledNormalizerTest, NoMappedChars) {
  auto charsmap = MakeSingleMappingCharsmap("Z", "Q");
  CreateNormalizer(charsmap);
  TestWithAllChunkSizes(normalizer(), "ABCDEFGHIJ", "ABCDEFGHIJ",
                        /*expect_pending_after_process=*/true);
}

TEST_F(PrecompiledNormalizerTest, EmptyInputWithNonEmptyTrie) {
  auto charsmap = MakeSingleMappingCharsmap("X", "Y");
  CreateNormalizer(charsmap);
  std::string result = ProcessAndFinalize(
      normalizer(), "", /*expect_pending_after_process=*/false);
  EXPECT_TRUE(result.empty());
}

//===----------------------------------------------------------------------===//
// Prefix/Longest Match Tests
//===----------------------------------------------------------------------===//

TEST_F(PrecompiledNormalizerTest, LongestMatchWins) {
  // Test that longer matches are preferred.
  TrieBuilder builder;
  builder.AddMapping("A", "1");
  builder.AddMapping("AB", "2");
  auto charsmap = builder.Build();
  CreateNormalizer(charsmap);
  // "AB" should match as "2", not "1" + "B".
  TestWithAllChunkSizes(normalizer(), "AB", "2",
                        /*expect_pending_after_process=*/true);
}

TEST_F(PrecompiledNormalizerTest, PrefixNoMatch) {
  // Key is "ABC" but input is "AB" - no match, passthrough.
  auto charsmap = MakeSingleMappingCharsmap("ABC", "X");
  CreateNormalizer(charsmap);
  TestWithAllChunkSizes(normalizer(), "AB", "AB",
                        /*expect_pending_after_process=*/true);
}

//===----------------------------------------------------------------------===//
// Format Validation (Additional)
//===----------------------------------------------------------------------===//

TEST(PrecompiledNormalizerValidationTest, RejectsEmptyData) {
  std::vector<uint8_t> data;
  iree_tokenizer_normalizer_t* normalizer = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_tokenizer_precompiled_normalizer_allocate(
          ToSpan(data), iree_allocator_system(), &normalizer));
}

TEST(PrecompiledNormalizerValidationTest, AcceptsMinimalValidData) {
  // Just trie_size=0 is valid (empty trie, passthrough mode).
  std::vector<uint8_t> data = {0, 0, 0, 0};
  iree_tokenizer_normalizer_t* normalizer = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_precompiled_normalizer_allocate(
      ToSpan(data), iree_allocator_system(), &normalizer));
  EXPECT_NE(normalizer, nullptr);
  iree_tokenizer_normalizer_free(normalizer);
}

TEST(PrecompiledNormalizerValidationTest,
     RejectsReplacementExceedingPendingBuffer) {
  // Replacement strings must fit in the 64-byte pending buffer.
  // A 65-byte replacement string must be rejected at allocation time.
  // This prevents heap-buffer-overflow when emit() copies to pending_buffer.
  TrieBuilder builder;
  builder.AddMapping("A",
                     std::string(65, 'X'));  // 65 bytes exceeds 64-byte limit.
  auto charsmap = builder.Build();

  iree_tokenizer_normalizer_t* normalizer = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_tokenizer_precompiled_normalizer_allocate(
          ToSpan(charsmap), iree_allocator_system(), &normalizer));
}

TEST(PrecompiledNormalizerValidationTest, Accepts64ByteReplacement) {
  // Exactly 64 bytes is the maximum allowed (fits in pending buffer).
  TrieBuilder builder;
  builder.AddMapping("A", std::string(64, 'X'));  // Exactly 64 bytes.
  auto charsmap = builder.Build();

  iree_tokenizer_normalizer_t* normalizer = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_precompiled_normalizer_allocate(
      ToSpan(charsmap), iree_allocator_system(), &normalizer));
  EXPECT_NE(normalizer, nullptr);
  iree_tokenizer_normalizer_free(normalizer);
}

}  // namespace
}  // namespace iree::tokenizer
